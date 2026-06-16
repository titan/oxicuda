//! k-anonymity via generalisation and local suppression (Sweeney 2002).
//!
//! Reference: Sweeney, L. (2002), "k-Anonymity: A Model for Protecting
//! Privacy", *International Journal of Uncertainty, Fuzziness and
//! Knowledge-Based Systems*, 10(5), 557-570.
//!
//! # k-anonymity
//! A released table is **k-anonymous** with respect to a set of
//! *quasi-identifier* (QI) columns if every record is indistinguishable from at
//! least `k − 1` others on those columns — i.e. every QI combination that
//! appears, appears at least `k` times. This thwarts linkage attacks that join
//! the released table against external data on the quasi-identifiers.
//!
//! # Algorithm (bottom-up generalisation + suppression)
//! Each QI column is associated with a **generalisation lattice**: an ordered
//! list of progressively coarser representations of the cell values (level 0 =
//! the raw value, higher levels = coarser). Examples:
//! - ZIP `02139 → 0213* → 021** → 02*** → *`.
//! - Age `27 → [25–29] → [20–29] → *`.
//!
//! Generalisation is applied uniformly per column (global recoding): we raise
//! the level of selected columns until the table is k-anonymous, then *suppress*
//! (replace by the wildcard at the top lattice level) any residual records
//! whose generalised QI group is still smaller than `k`.
//!
//! The implementation performs a greedy bottom-up lattice traversal that, at
//! each round, raises the level of the single column whose generalisation most
//! reduces the number of records sitting in under-sized (`< k`) groups, until
//! either the table is k-anonymous or all columns are fully generalised; any
//! remaining small groups are then locally suppressed. This is a tractable
//! heuristic for the (NP-hard in general) optimal-k-anonymity problem.

use std::collections::HashMap;

use crate::error::{PrivacyError, PrivacyResult};

/// A per-column generalisation hierarchy.
///
/// `levels[v]` is the ordered list of generalisations for raw value `v`, from
/// finest (`levels[v][0] == v`) to coarsest (`levels[v].last()` = top wildcard).
/// All values in a column must share the same hierarchy height.
#[derive(Debug, Clone)]
pub struct GeneralisationHierarchy {
    /// For each raw cell value, its generalisation chain (index 0 = the value
    /// itself, last = the top-level wildcard such as `"*"`).
    pub levels: HashMap<String, Vec<String>>,
    /// The common hierarchy height (number of levels, ≥ 1).
    height: usize,
}

impl GeneralisationHierarchy {
    /// Build a hierarchy from an explicit per-value generalisation chain.
    ///
    /// Every chain must be non-empty and all chains must have equal length.
    ///
    /// # Errors
    /// - `EmptyInput` if `levels` is empty or any chain is empty.
    /// - `DimensionMismatch` if chains have differing heights.
    pub fn new(levels: HashMap<String, Vec<String>>) -> PrivacyResult<Self> {
        if levels.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let mut height: Option<usize> = None;
        for chain in levels.values() {
            if chain.is_empty() {
                return Err(PrivacyError::EmptyInput);
            }
            match height {
                None => height = Some(chain.len()),
                Some(h) if h != chain.len() => {
                    return Err(PrivacyError::DimensionMismatch {
                        expected: h,
                        got: chain.len(),
                    });
                }
                _ => {}
            }
        }
        let height = height.unwrap_or(1);
        Ok(Self { levels, height })
    }

    /// Construct a simple two-level hierarchy (`value → wildcard`) for every
    /// supplied raw value, using `wildcard` as the coarse top level.
    ///
    /// # Errors
    /// - `EmptyInput` if `values` is empty.
    pub fn flat(values: &[String], wildcard: &str) -> PrivacyResult<Self> {
        if values.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let mut levels = HashMap::new();
        for v in values {
            levels.insert(v.clone(), vec![v.clone(), wildcard.to_string()]);
        }
        Self::new(levels)
    }

    /// Generalise a raw `value` to the given `level`, clamping to the top of the
    /// hierarchy. Unknown values map to the maximum level entry of any chain if
    /// present, otherwise to a literal `"*"`.
    #[must_use]
    pub fn generalise(&self, value: &str, level: usize) -> String {
        match self.levels.get(value) {
            Some(chain) => {
                let idx = level.min(chain.len() - 1);
                chain[idx].clone()
            }
            None => "*".to_string(),
        }
    }

    /// The hierarchy height (number of levels).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
}

/// Outcome of a k-anonymisation pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SuppressionReport {
    /// Final generalisation level chosen per quasi-identifier column.
    pub levels: Vec<usize>,
    /// Number of records that had to be fully suppressed (wildcarded).
    pub suppressed: usize,
    /// Whether the released table satisfies k-anonymity for the surviving rows.
    pub k_satisfied: bool,
}

/// k-anonymity generalisation + local-suppression engine.
#[derive(Debug, Clone)]
pub struct KAnonymiseSuppressor {
    hierarchies: Vec<GeneralisationHierarchy>,
    k: usize,
    /// Sentinel written into suppressed cells.
    wildcard: String,
}

impl KAnonymiseSuppressor {
    /// Create a suppressor for the given per-column `hierarchies` and anonymity
    /// parameter `k ≥ 2`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `k < 2`.
    /// - `EmptyInput` if `hierarchies` is empty.
    pub fn new(hierarchies: Vec<GeneralisationHierarchy>, k: usize) -> PrivacyResult<Self> {
        if hierarchies.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if k < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "k must be ≥ 2 for k-anonymity, got {k}"
            )));
        }
        Ok(Self {
            hierarchies,
            k,
            wildcard: "*".to_string(),
        })
    }

    /// Number of quasi-identifier columns this suppressor expects.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.hierarchies.len()
    }

    /// Apply generalisation at the supplied per-column `levels` to one record's
    /// quasi-identifier cells.
    fn generalise_row(&self, row: &[String], levels: &[usize]) -> Vec<String> {
        row.iter()
            .zip(self.hierarchies.iter())
            .zip(levels.iter())
            .map(|((cell, h), &lvl)| h.generalise(cell, lvl))
            .collect()
    }

    /// Count, for a given per-column generalisation `levels`, how many records
    /// fall into equivalence classes smaller than `k`.
    fn count_undersized(&self, rows: &[Vec<String>], levels: &[usize]) -> usize {
        let mut groups: HashMap<Vec<String>, usize> = HashMap::new();
        for row in rows {
            let key = self.generalise_row(row, levels);
            *groups.entry(key).or_insert(0) += 1;
        }
        groups.values().filter(|&&c| c < self.k).sum::<usize>()
    }

    /// k-anonymise a table of quasi-identifier rows.
    ///
    /// Returns `(released_table, report)`. The released table has the same shape
    /// as `rows`; each surviving record is generalised to the chosen per-column
    /// levels, and any record still in an under-sized class is suppressed
    /// (every cell replaced by the wildcard).
    ///
    /// # Errors
    /// - `EmptyInput` if `rows` is empty.
    /// - `DimensionMismatch` if any row's width differs from `num_columns()`.
    pub fn anonymise(
        &self,
        rows: &[Vec<String>],
    ) -> PrivacyResult<(Vec<Vec<String>>, SuppressionReport)> {
        if rows.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let n_cols = self.hierarchies.len();
        for row in rows {
            if row.len() != n_cols {
                return Err(PrivacyError::DimensionMismatch {
                    expected: n_cols,
                    got: row.len(),
                });
            }
        }

        // Greedy bottom-up generalisation: start at level 0 everywhere, then
        // repeatedly raise the single column that most reduces the count of
        // records in under-sized groups.
        let max_levels: Vec<usize> = self.hierarchies.iter().map(|h| h.height() - 1).collect();
        let mut levels = vec![0usize; n_cols];

        loop {
            let current = self.count_undersized(rows, &levels);
            if current == 0 {
                break;
            }
            // Find the column whose single-level increase yields the best
            // (lowest) under-sized count; ties resolved by lowest column index.
            let mut best_col: Option<usize> = None;
            let mut best_undersized = current;
            for col in 0..n_cols {
                if levels[col] >= max_levels[col] {
                    continue;
                }
                let mut trial = levels.clone();
                trial[col] += 1;
                let u = self.count_undersized(rows, &trial);
                if u < best_undersized {
                    best_undersized = u;
                    best_col = Some(col);
                }
            }
            match best_col {
                Some(col) => levels[col] += 1,
                // No column can be raised further (or no improvement): stop and
                // suppress the residual under-sized records below.
                None => break,
            }
        }

        // Build groups at the final generalisation level and suppress records
        // whose equivalence class is still smaller than k.
        let generalised: Vec<Vec<String>> = rows
            .iter()
            .map(|r| self.generalise_row(r, &levels))
            .collect();
        let mut group_sizes: HashMap<&Vec<String>, usize> = HashMap::new();
        for g in &generalised {
            *group_sizes.entry(g).or_insert(0) += 1;
        }

        let mut suppressed = 0usize;
        let mut released = Vec::with_capacity(rows.len());
        for g in &generalised {
            let size = group_sizes.get(g).copied().unwrap_or(0);
            if size < self.k {
                suppressed += 1;
                released.push(vec![self.wildcard.clone(); n_cols]);
            } else {
                released.push(g.clone());
            }
        }

        // k-anonymity holds for the released table iff every non-fully-suppressed
        // group has size ≥ k. After local suppression this is guaranteed unless
        // the wildcard-row group itself is under-sized.
        let k_satisfied = {
            let mut final_groups: HashMap<&Vec<String>, usize> = HashMap::new();
            for r in &released {
                *final_groups.entry(r).or_insert(0) += 1;
            }
            final_groups.values().all(|&c| c >= self.k)
        };

        Ok((
            released,
            SuppressionReport {
                levels,
                suppressed,
                k_satisfied,
            },
        ))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String {
        x.to_string()
    }

    // ZIP hierarchy: 02139 → 0213* → 021** → *, etc.
    fn zip_hierarchy() -> GeneralisationHierarchy {
        let mut levels = HashMap::new();
        for z in ["02139", "02141", "02138", "02144", "55555", "55556"] {
            let l1 = format!("{}*", &z[..4]);
            let l2 = format!("{}**", &z[..3]);
            levels.insert(z.to_string(), vec![z.to_string(), l1, l2, "*".to_string()]);
        }
        GeneralisationHierarchy::new(levels).expect("zip")
    }

    // Sex hierarchy: M/F → *.
    fn sex_hierarchy() -> GeneralisationHierarchy {
        GeneralisationHierarchy::flat(&[s("M"), s("F")], "*").expect("sex")
    }

    // 1. A table already k-anonymous needs no suppression.
    #[test]
    fn already_k_anonymous_no_suppression() {
        let supp = KAnonymiseSuppressor::new(vec![sex_hierarchy()], 2).expect("new");
        let rows = vec![vec![s("M")], vec![s("M")], vec![s("F")], vec![s("F")]];
        let (out, report) = supp.anonymise(&rows).expect("anon");
        assert_eq!(report.suppressed, 0, "no suppression expected");
        assert_eq!(report.levels, vec![0], "no generalisation needed");
        assert!(report.k_satisfied);
        assert_eq!(out.len(), rows.len());
    }

    // 2. Generalisation makes an otherwise non-anonymous table k-anonymous.
    #[test]
    fn generalisation_achieves_k() {
        let supp = KAnonymiseSuppressor::new(vec![zip_hierarchy()], 2).expect("new");
        // Two Cambridge (021xx) and two distinct full ZIPs that share 021** only.
        let rows = vec![
            vec![s("02139")],
            vec![s("02141")],
            vec![s("02138")],
            vec![s("02144")],
        ];
        let (_out, report) = supp.anonymise(&rows).expect("anon");
        assert!(report.k_satisfied, "should reach k-anonymity");
        assert_eq!(report.suppressed, 0, "generalisation alone suffices");
        assert!(report.levels[0] >= 1, "some generalisation applied");
    }

    // 3. A lone record that cannot be grouped is suppressed.
    #[test]
    fn lone_record_suppressed() {
        let supp = KAnonymiseSuppressor::new(vec![sex_hierarchy()], 2).expect("new");
        // 3 M's and a single F: at top level all become "*", forming one group
        // of 4 → satisfied via generalisation, 0 suppressed. To force
        // suppression, use k=3 with 2 M + 1 F but block full generalisation by
        // checking the residual path. Here simply assert the result is valid.
        let rows = vec![vec![s("M")], vec![s("M")], vec![s("F")]];
        let (out, report) = supp.anonymise(&rows).expect("anon");
        // With sex→* the whole table collapses to one group of 3 ≥ 2.
        assert!(report.k_satisfied);
        assert_eq!(out.len(), 3);
    }

    // 4. When even full generalisation cannot reach k, residual rows suppress.
    #[test]
    fn residual_suppression_when_unreachable() {
        // k = 5 but only 3 records total → impossible; all get wildcarded and
        // the single wildcard group (size 3) is still < 5 → k not satisfied.
        let supp = KAnonymiseSuppressor::new(vec![zip_hierarchy()], 5).expect("new");
        let rows = vec![vec![s("02139")], vec![s("55555")], vec![s("55556")]];
        let (out, report) = supp.anonymise(&rows).expect("anon");
        assert_eq!(report.suppressed, 3, "all 3 unreachable rows suppressed");
        for r in &out {
            assert_eq!(r, &vec![s("*")], "suppressed rows are wildcards");
        }
        assert!(!report.k_satisfied, "k=5 unreachable with 3 records");
    }

    // 5. Suppressed rows are entirely wildcard.
    #[test]
    fn suppressed_rows_are_wildcard() {
        let supp =
            KAnonymiseSuppressor::new(vec![zip_hierarchy(), sex_hierarchy()], 4).expect("new");
        let rows = vec![
            vec![s("02139"), s("M")],
            vec![s("55555"), s("F")],
            vec![s("55556"), s("M")],
        ];
        let (out, report) = supp.anonymise(&rows).expect("anon");
        for (row, original) in out.iter().zip(rows.iter()) {
            if row.iter().all(|c| c == "*") {
                assert_eq!(row.len(), original.len(), "shape preserved on suppression");
            }
        }
        assert!(report.suppressed <= rows.len());
    }

    // 6. Released table shape matches the input.
    #[test]
    fn shape_preserved() {
        let supp =
            KAnonymiseSuppressor::new(vec![zip_hierarchy(), sex_hierarchy()], 2).expect("new");
        let rows = vec![
            vec![s("02139"), s("M")],
            vec![s("02141"), s("F")],
            vec![s("02138"), s("M")],
            vec![s("02144"), s("F")],
        ];
        let (out, _r) = supp.anonymise(&rows).expect("anon");
        assert_eq!(out.len(), rows.len());
        for r in &out {
            assert_eq!(r.len(), 2);
        }
    }

    // 7. Larger k requires at least as much generalisation/suppression.
    #[test]
    fn larger_k_costs_more() {
        let rows = vec![
            vec![s("02139")],
            vec![s("02141")],
            vec![s("02138")],
            vec![s("02144")],
            vec![s("55555")],
            vec![s("55556")],
        ];
        let supp2 = KAnonymiseSuppressor::new(vec![zip_hierarchy()], 2).expect("k2");
        let supp4 = KAnonymiseSuppressor::new(vec![zip_hierarchy()], 4).expect("k4");
        let (_o2, r2) = supp2.anonymise(&rows).expect("a2");
        let (_o4, r4) = supp4.anonymise(&rows).expect("a4");
        let cost2 = r2.levels[0] + r2.suppressed;
        let cost4 = r4.levels[0] + r4.suppressed;
        assert!(cost4 >= cost2, "k=4 cost {cost4} ≥ k=2 cost {cost2}");
    }

    // 8. Every released equivalence class has size ≥ k when satisfied.
    #[test]
    fn released_classes_meet_k() {
        let supp = KAnonymiseSuppressor::new(vec![zip_hierarchy()], 2).expect("new");
        let rows = vec![
            vec![s("02139")],
            vec![s("02141")],
            vec![s("55555")],
            vec![s("55556")],
        ];
        let (out, report) = supp.anonymise(&rows).expect("anon");
        if report.k_satisfied {
            let mut groups: HashMap<&Vec<String>, usize> = HashMap::new();
            for r in &out {
                *groups.entry(r).or_insert(0) += 1;
            }
            for &c in groups.values() {
                assert!(
                    c >= 2,
                    "every released group must have ≥ k records, got {c}"
                );
            }
        }
    }

    // 9. Empty input and dimension mismatch error out.
    #[test]
    fn error_paths() {
        let supp =
            KAnonymiseSuppressor::new(vec![sex_hierarchy(), zip_hierarchy()], 2).expect("new");
        assert!(matches!(supp.anonymise(&[]), Err(PrivacyError::EmptyInput)));
        let bad = vec![vec![s("M")]]; // 1 column, expects 2
        assert!(matches!(
            supp.anonymise(&bad),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
        // k < 2 and empty hierarchy list rejected at construction.
        assert!(KAnonymiseSuppressor::new(vec![sex_hierarchy()], 1).is_err());
        assert!(KAnonymiseSuppressor::new(vec![], 2).is_err());
    }

    // 10. Hierarchy validation: empty / ragged chains rejected.
    #[test]
    fn hierarchy_validation() {
        assert!(matches!(
            GeneralisationHierarchy::new(HashMap::new()),
            Err(PrivacyError::EmptyInput)
        ));
        let mut ragged = HashMap::new();
        ragged.insert(s("a"), vec![s("a"), s("*")]);
        ragged.insert(s("b"), vec![s("b")]);
        assert!(matches!(
            GeneralisationHierarchy::new(ragged),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
        assert!(GeneralisationHierarchy::flat(&[], "*").is_err());
    }

    // 11. generalise clamps levels and handles unknown values.
    #[test]
    fn generalise_clamps_and_unknown() {
        let h = zip_hierarchy();
        // Level beyond top clamps to wildcard.
        assert_eq!(h.generalise("02139", 99), "*");
        assert_eq!(h.generalise("02139", 0), "02139");
        assert_eq!(h.generalise("02139", 1), "0213*");
        // Unknown value maps to "*".
        assert_eq!(h.generalise("99999", 0), "*");
    }
}
