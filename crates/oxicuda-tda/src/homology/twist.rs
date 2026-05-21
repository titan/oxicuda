//! Twist reduction for persistent homology (Chen-Kerber 2011).
//!
//! Reference: Chao Chen & Michael Kerber, *Persistent Homology Computation with a
//! Twist*, Proc. 27th European Workshop on Computational Geometry (EuroCG 2011);
//! see also *Discrete Comput. Geom.* 50:266 (2013).
//!
//! # Background
//!
//! The standard persistence reduction algorithm of Edelsbrunner-Letscher-Zomorodian
//! (2002) reduces the boundary matrix `∂` over Z₂ by left-to-right column additions
//! until every non-zero column has a unique pivot row.  The twist algorithm produces
//! the **same** persistence pairs but processes columns by **descending homological
//! dimension** and exploits a *clearing* rule.
//!
//! Concretely, when a column `j` of dimension `d` is reduced and ends up with a
//! pivot at row `r` (where `dim(r) = d − 1`), the pair `(r, j)` is locked in: `r`
//! is a positive (birth) simplex and `j` is a negative (death) simplex.  In the
//! standard algorithm the column at index `r` is guaranteed to reduce to zero
//! (since `r` is a cycle that has been killed by `j`).  Therefore in the twist
//! algorithm we can **clear** column `r` (zero it out) before ever reducing it.
//!
//! Processing the dimensions in descending order — `d = max_dim, max_dim − 1, …, 1`
//! — ensures that all clearings of dimension `d − 1` are recorded *before* the
//! `d − 1` pass starts; the `d − 1` pass then simply skips every cleared column.
//!
//! The persistence pairs are emitted as integer simplex indices `(birth_idx,
//! death_idx)` stored in the `f64` fields of [`PersistencePair`].  The dimension
//! of each pair equals the dimension of the *birth* simplex.

use crate::error::{TdaError, TdaResult};
use crate::homology::persistent::PersistencePair;
use std::collections::HashMap;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the [`TwistReduction`] algorithm.
#[derive(Debug, Clone, Default)]
pub struct TwistConfig {
    /// Maximum simplex dimension present in the filtration.
    ///
    /// All entries of the `col_dims` slice passed to [`TwistReduction::reduce`]
    /// must satisfy `col_dim ≤ max_dim`.
    pub max_dim: usize,
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result of a [`TwistReduction::reduce`] call.
#[derive(Debug, Clone, Default)]
pub struct TwistResult {
    /// Persistence pairs produced by the reduction.  The `birth` and `death`
    /// fields hold the **simplex indices** (as `f64`) into the input arrays;
    /// `dim` is the dimension of the birth simplex.
    pub pairs: Vec<PersistencePair>,
    /// Total number of column-add (symmetric-difference) operations performed.
    pub n_reductions: usize,
    /// Number of columns that were eliminated by the clearing rule (their
    /// pairs are still emitted, but those columns were not themselves reduced).
    pub n_cleared: usize,
}

// ─── Algorithm ────────────────────────────────────────────────────────────────

/// Twist reduction algorithm (Chen-Kerber 2011).
pub struct TwistReduction;

impl TwistReduction {
    /// Reduce a boundary matrix and emit the persistent homology pairs.
    ///
    /// # Inputs
    /// - `boundary`: sparse mod-2 boundary matrix.  `boundary[col_idx]` is the
    ///   sorted, strictly ascending list of row indices that are non-zero in
    ///   column `col_idx`.  The matrix must be strictly lower triangular
    ///   (`row < col_idx` for every row in `boundary[col_idx]`).
    /// - `col_dims`: parallel slice giving the homological dimension of each
    ///   column (`col_dims.len() == boundary.len()`).
    /// - `cfg`: algorithm configuration; `cfg.max_dim` must be `≥` every entry
    ///   of `col_dims`.
    ///
    /// # Output
    /// A [`TwistResult`] containing all finite pairs together with operation
    /// counters.  Essential (unpaired) cycles are *not* enumerated by the twist
    /// algorithm proper — the caller can recover them by enumerating column
    /// indices that are neither a birth nor a death in `result.pairs`.
    ///
    /// # Errors
    /// - [`TdaError::DimensionMismatch`] if `boundary.len() ≠ col_dims.len()`.
    /// - [`TdaError::DimensionTooLarge`] if any `col_dims[i] > cfg.max_dim`.
    /// - [`TdaError::InvalidSimplex`] if any row index in a column is not
    ///   strictly less than the column index (non-triangular input), or if a
    ///   column's row indices are not strictly ascending.
    pub fn reduce(
        boundary: &[Vec<usize>],
        col_dims: &[usize],
        cfg: &TwistConfig,
    ) -> TdaResult<TwistResult> {
        let n = boundary.len();

        // ── Validation ────────────────────────────────────────────────────────
        if col_dims.len() != n {
            return Err(TdaError::DimensionMismatch {
                expected: n,
                got: col_dims.len(),
            });
        }
        for (col_idx, col) in boundary.iter().enumerate() {
            let d = match col_dims.get(col_idx) {
                Some(value) => *value,
                None => {
                    return Err(TdaError::DimensionMismatch {
                        expected: n,
                        got: col_dims.len(),
                    });
                }
            };
            if d > cfg.max_dim {
                return Err(TdaError::DimensionTooLarge(d));
            }
            for &row in col {
                if row >= col_idx {
                    return Err(TdaError::InvalidSimplex(format!(
                        "row {row} >= col {col_idx} (boundary matrix must be lower-triangular)"
                    )));
                }
            }
            // Ensure each column is strictly ascending.
            for w in col.windows(2) {
                if w[0] >= w[1] {
                    return Err(TdaError::InvalidSimplex(format!(
                        "column {col_idx} row indices must be strictly ascending"
                    )));
                }
            }
        }

        // ── Working state ─────────────────────────────────────────────────────
        // Mutable copy of the boundary matrix we are reducing in place.
        let mut columns: Vec<Vec<usize>> = boundary.to_vec();
        // `pivot_to_col[r] = j` means the reduced column `j` has pivot row `r`.
        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
        // Columns marked as "cleared" by the twist optimisation: skipped when
        // their dimension pass arrives.
        let mut cleared = vec![false; n];

        let mut n_reductions: usize = 0usize;
        let mut n_cleared: usize = 0usize;

        // Group column indices by dimension once (preserving input order within
        // each group) so we can iterate by descending d.
        let mut by_dim: Vec<Vec<usize>> = vec![Vec::new(); cfg.max_dim + 1];
        for (col_idx, &d) in col_dims.iter().enumerate() {
            by_dim[d].push(col_idx);
        }

        // ── Descending-dimension reduction with clearing ────────────────────
        //
        // Iterate d from cfg.max_dim down to 0.  For each column `j` of
        // dimension `d`:
        //
        //   * If `cleared[j]` is true (set by a higher-dim reduction), `j` is a
        //     positive simplex paired with some `k > j`, and its column would
        //     reduce to zero in the standard algorithm.  Skip it entirely.
        //   * Otherwise reduce `columns[j]` against previously recorded pivots
        //     via repeated symmetric difference.  When a new pivot row `r`
        //     emerges, record `pivot_to_col[r] = j` and mark column `r` as
        //     cleared.
        let mut d_iter = cfg.max_dim;
        loop {
            let cols_at_d = match by_dim.get(d_iter) {
                Some(slice) => slice.clone(),
                None => Vec::new(),
            };
            for col_idx in cols_at_d {
                if cleared[col_idx] {
                    // Cleared: its pair was already recorded when a higher-dim
                    // column established this column as a positive simplex.
                    continue;
                }
                // Reduce column col_idx against existing pivot columns.
                while let Some(pivot_row) = columns[col_idx].last().copied() {
                    match pivot_to_col.get(&pivot_row).copied() {
                        Some(other) => {
                            let other_col = columns[other].clone();
                            columns[col_idx] = Self::sym_diff(&columns[col_idx], &other_col);
                            n_reductions += 1;
                        }
                        None => {
                            // New pivot found; record it and clear the
                            // corresponding lower-dim positive column.
                            pivot_to_col.insert(pivot_row, col_idx);
                            if !cleared[pivot_row] {
                                cleared[pivot_row] = true;
                                n_cleared += 1;
                                // Zero out the cleared column to make the
                                // invariant explicit.  This avoids redundant
                                // reduction work when the d − 1 pass starts.
                                columns[pivot_row].clear();
                            }
                            break;
                        }
                    }
                }
            }
            if d_iter == 0 {
                break;
            }
            d_iter -= 1;
        }

        // ── Pair extraction ──────────────────────────────────────────────────
        //
        // Each (pivot_row, col) in `pivot_to_col` describes a pair: simplex
        // `pivot_row` is born (positive) and simplex `col` is its death
        // (negative).  The dimension of the pair is `dim(pivot_row)`.
        let mut pairs: Vec<PersistencePair> = Vec::with_capacity(pivot_to_col.len());
        for (&pivot_row, &col_idx) in &pivot_to_col {
            let birth_dim = match col_dims.get(pivot_row) {
                Some(value) => *value,
                None => continue,
            };
            pairs.push(PersistencePair {
                dim: birth_dim,
                birth: pivot_row as f64,
                death: Some(col_idx as f64),
            });
        }
        // Deterministic ordering: sort by (birth, death).
        pairs.sort_by(|a, b| {
            let ord_birth = a
                .birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal);
            if ord_birth != std::cmp::Ordering::Equal {
                return ord_birth;
            }
            let ad = a.death.unwrap_or(f64::INFINITY);
            let bd = b.death.unwrap_or(f64::INFINITY);
            ad.partial_cmp(&bd).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(TwistResult {
            pairs,
            n_reductions,
            n_cleared,
        })
    }

    /// Symmetric difference of two strictly-ascending sorted slices of row
    /// indices.
    ///
    /// This is the mod-2 column-addition primitive: rows appearing in both
    /// inputs cancel, rows appearing in exactly one input survive.  The output
    /// is also strictly ascending.
    pub fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(a.len() + b.len());
        let mut ai = 0usize;
        let mut bi = 0usize;
        while ai < a.len() && bi < b.len() {
            match a[ai].cmp(&b[bi]) {
                std::cmp::Ordering::Less => {
                    out.push(a[ai]);
                    ai += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b[bi]);
                    bi += 1;
                }
                std::cmp::Ordering::Equal => {
                    ai += 1;
                    bi += 1;
                }
            }
        }
        while ai < a.len() {
            out.push(a[ai]);
            ai += 1;
        }
        while bi < b.len() {
            out.push(b[bi]);
            bi += 1;
        }
        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build the (boundary, dims, max_dim) of a filled triangle.
    // Indices: 0..3=vertices, 3..6=edges (01,02,12), 6=triangle 012.
    fn triangle_setup() -> (Vec<Vec<usize>>, Vec<usize>, usize) {
        let boundary: Vec<Vec<usize>> = vec![
            vec![],        // v0
            vec![],        // v1
            vec![],        // v2
            vec![0, 1],    // e01
            vec![0, 2],    // e02
            vec![1, 2],    // e12
            vec![3, 4, 5], // t012
        ];
        let dims = vec![0, 0, 0, 1, 1, 1, 2];
        (boundary, dims, 2)
    }

    // ── sym_diff tests ────────────────────────────────────────────────────────
    #[test]
    fn sym_diff_empty_empty() {
        let r = TwistReduction::sym_diff(&[], &[]);
        assert!(r.is_empty(), "empty ⊕ empty = empty");
    }

    #[test]
    fn sym_diff_identical_cancels() {
        let a = vec![1usize, 3, 5];
        let r = TwistReduction::sym_diff(&a, &a);
        assert!(r.is_empty(), "x ⊕ x = 0 over Z₂");
    }

    #[test]
    fn sym_diff_disjoint_is_union() {
        let r = TwistReduction::sym_diff(&[1, 3], &[2, 4]);
        assert_eq!(r, vec![1, 2, 3, 4]);
    }

    #[test]
    fn sym_diff_partial_overlap() {
        // {0,1,2,5} ⊕ {1,3,5,7} = {0,2,3,7}
        let r = TwistReduction::sym_diff(&[0, 1, 2, 5], &[1, 3, 5, 7]);
        assert_eq!(r, vec![0, 2, 3, 7]);
    }

    #[test]
    fn sym_diff_one_empty() {
        let r = TwistReduction::sym_diff(&[], &[2, 4, 6]);
        assert_eq!(r, vec![2, 4, 6]);
        let r2 = TwistReduction::sym_diff(&[2, 4, 6], &[]);
        assert_eq!(r2, vec![2, 4, 6]);
    }

    // ── reduce structural tests ───────────────────────────────────────────────
    #[test]
    fn reduce_filled_triangle_known_pairs() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = TwistConfig { max_dim };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        // For the filled triangle we expect exactly 3 finite pairs from the
        // twist algorithm:
        //   * two H0 pairs (two vertices killed by the first two edges)
        //   * one H1 pair (the cycle born by the last edge, killed by t012)
        let finite_count = res.pairs.iter().filter(|p| p.death.is_some()).count();
        assert_eq!(
            finite_count, 3,
            "expected 3 finite pairs, got {finite_count}"
        );
        // The H1 pair must be (5, 6) — last edge (idx 5) paired with t012 (idx 6).
        let has_h1 = res
            .pairs
            .iter()
            .any(|p| p.dim == 1 && (p.birth - 5.0).abs() < f64::EPSILON && p.death == Some(6.0));
        assert!(has_h1, "expected H1 pair (5, 6), got {:?}", res.pairs);
    }

    #[test]
    fn reduce_filled_triangle_has_clearings() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = TwistConfig { max_dim };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        // Clearing must fire on the filled triangle: the 2-simplex t012
        // clears edge e12, then the surviving edges e01,e02 clear vertices
        // v1 and v2.  Twist saves work precisely by performing zero sym-diffs
        // on this "already reduced" input.
        assert!(res.n_cleared > 0, "clearing must fire on triangle");
    }

    #[test]
    fn reduce_tetrahedron_boundary_has_reductions() {
        // The tetrahedron-boundary input forces actual column additions on
        // the t123 = e12⊕e13⊕e23 column: its pivot row 9 collides with t023,
        // which forces a sym_diff cascade through t013 and t012 down to 0.
        let boundary: Vec<Vec<usize>> = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
            vec![4, 5, 7],
            vec![4, 6, 8],
            vec![5, 6, 9],
            vec![7, 8, 9],
        ];
        let dims = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
        let cfg = TwistConfig { max_dim: 2 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert!(
            res.n_reductions > 0,
            "tetrahedron-boundary should perform sym-diff reductions"
        );
        assert!(
            res.n_cleared > 0,
            "tetrahedron-boundary should clear columns"
        );
    }

    #[test]
    fn reduce_all_empty_columns_no_pairs() {
        // 4 isolated vertices: no boundary entries → no pairs.
        let boundary: Vec<Vec<usize>> = vec![vec![]; 4];
        let dims = vec![0, 0, 0, 0];
        let cfg = TwistConfig { max_dim: 0 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty(), "isolated vertices yield no pairs");
        assert_eq!(res.n_reductions, 0);
        assert_eq!(res.n_cleared, 0);
    }

    #[test]
    fn reduce_deterministic() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = TwistConfig { max_dim };
        let a = TwistReduction::reduce(&boundary, &dims, &cfg).expect("a");
        let b = TwistReduction::reduce(&boundary, &dims, &cfg).expect("b");
        // Pairs are emitted in deterministic (sorted) order; equality must hold.
        let av: Vec<(usize, f64, Option<f64>)> =
            a.pairs.iter().map(|p| (p.dim, p.birth, p.death)).collect();
        let bv: Vec<(usize, f64, Option<f64>)> =
            b.pairs.iter().map(|p| (p.dim, p.birth, p.death)).collect();
        assert_eq!(av, bv, "reduction must be deterministic");
        assert_eq!(a.n_reductions, b.n_reductions);
        assert_eq!(a.n_cleared, b.n_cleared);
    }

    #[test]
    fn reduce_single_simplex_no_pairs() {
        let boundary: Vec<Vec<usize>> = vec![vec![]];
        let dims = vec![0usize];
        let cfg = TwistConfig { max_dim: 0 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
    }

    #[test]
    fn reduce_two_vertices_one_edge_yields_one_pair() {
        // v0, v1, e01: a single edge connects two vertices.
        // After reduction: edge has pivot 1, so the pair is (1, 2).  v0 stays essential.
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![], vec![0, 1]];
        let dims = vec![0, 0, 1];
        let cfg = TwistConfig { max_dim: 1 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert_eq!(res.pairs.len(), 1, "expected exactly one finite pair");
        let p = &res.pairs[0];
        assert_eq!(p.dim, 0);
        assert!((p.birth - 1.0).abs() < f64::EPSILON);
        assert_eq!(p.death, Some(2.0));
    }

    // ── error / edge-case tests ───────────────────────────────────────────────
    #[test]
    fn reduce_err_length_mismatch() {
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![]];
        let dims = vec![0usize]; // shorter than boundary
        let cfg = TwistConfig { max_dim: 0 };
        let err = TwistReduction::reduce(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::DimensionMismatch { .. })));
    }

    #[test]
    fn reduce_err_dim_exceeds_max_dim() {
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![0]];
        let dims = vec![0, 2]; // 2 > max_dim 1
        let cfg = TwistConfig { max_dim: 1 };
        let err = TwistReduction::reduce(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::DimensionTooLarge(2))));
    }

    #[test]
    fn reduce_err_row_index_not_lower() {
        // Column 1 has row index 1 == col_idx → not strictly lower-triangular.
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![1]];
        let dims = vec![0, 1];
        let cfg = TwistConfig { max_dim: 1 };
        let err = TwistReduction::reduce(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::InvalidSimplex(_))));
    }

    #[test]
    fn reduce_err_row_index_not_ascending() {
        // Column 2 rows [1,0] are not strictly ascending.
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![], vec![1, 0]];
        let dims = vec![0, 0, 1];
        let cfg = TwistConfig { max_dim: 1 };
        let err = TwistReduction::reduce(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::InvalidSimplex(_))));
    }

    #[test]
    fn reduce_max_dim_zero_only_vertices() {
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![], vec![]];
        let dims = vec![0, 0, 0];
        let cfg = TwistConfig { max_dim: 0 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
        assert_eq!(res.n_reductions, 0);
        assert_eq!(res.n_cleared, 0);
    }

    #[test]
    fn reduce_path_graph_three_vertices() {
        // Path v0 — v1 — v2: 3 vertices + 2 edges → 2 finite H0 pairs.
        // Columns: v0=[], v1=[], v2=[], e01=[0,1], e12=[1,2].
        let boundary: Vec<Vec<usize>> = vec![vec![], vec![], vec![], vec![0, 1], vec![1, 2]];
        let dims = vec![0, 0, 0, 1, 1];
        let cfg = TwistConfig { max_dim: 1 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        let finite_count = res.pairs.iter().filter(|p| p.death.is_some()).count();
        assert_eq!(
            finite_count, 2,
            "expected 2 finite H0 pairs in a path graph"
        );
    }

    #[test]
    fn reduce_triangle_cycle_two_finite_pairs() {
        // 3 vertices + 3 edges (triangle cycle, no 2-simplex): yields 2 finite
        // H0 pairs.  The H1 cycle is essential (no death) and therefore NOT
        // emitted by reduce() (which only returns finite pairs).
        let boundary: Vec<Vec<usize>> =
            vec![vec![], vec![], vec![], vec![0, 1], vec![0, 2], vec![1, 2]];
        let dims = vec![0, 0, 0, 1, 1, 1];
        let cfg = TwistConfig { max_dim: 1 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        let finite_count = res.pairs.iter().filter(|p| p.death.is_some()).count();
        assert_eq!(finite_count, 2, "expected 2 finite H0 pairs");
    }

    #[test]
    fn reduce_twist_pairs_equal_standard_reduction_on_triangle() {
        // Compare the twist output against a hand-derived standard-reduction
        // result on the filled triangle.  Standard reduction yields:
        //   pivots: row 1 → col 3, row 2 → col 4, row 5 → col 6.
        // Pairs: (1,3) H0, (2,4) H0, (5,6) H1.
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = TwistConfig { max_dim };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");

        let mut got: Vec<(usize, u64, u64)> = res
            .pairs
            .iter()
            .filter_map(|p| p.death.map(|d| (p.dim, p.birth as u64, d as u64)))
            .collect();
        got.sort();

        let mut expected: Vec<(usize, u64, u64)> = vec![(0, 1, 3), (0, 2, 4), (1, 5, 6)];
        expected.sort();

        assert_eq!(got, expected, "twist pairs must match standard reduction");
    }

    #[test]
    fn reduce_lower_triangular_check_with_higher_dim_input() {
        // Filtration with three 2-simplices forming the boundary of a
        // tetrahedron (4 vertices + 6 edges + 4 triangles, no 3-cell).
        let boundary: Vec<Vec<usize>> = vec![
            vec![],        // v0
            vec![],        // v1
            vec![],        // v2
            vec![],        // v3
            vec![0, 1],    // e01
            vec![0, 2],    // e02
            vec![0, 3],    // e03
            vec![1, 2],    // e12
            vec![1, 3],    // e13
            vec![2, 3],    // e23
            vec![4, 5, 7], // t012 = e01⊕e02⊕e12
            vec![4, 6, 8], // t013 = e01⊕e03⊕e13
            vec![5, 6, 9], // t023 = e02⊕e03⊕e23
            vec![7, 8, 9], // t123 = e12⊕e13⊕e23
        ];
        let dims = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
        let cfg = TwistConfig { max_dim: 2 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        // Tetrahedron boundary: 1 essential H0, 0 finite H1 in standard order,
        // and 1 essential H2.  Among the finite pairs we expect 3 H0 pairs and
        // 3 H1 pairs (1-cycles killed by 2-simplices).
        let h0 = res
            .pairs
            .iter()
            .filter(|p| p.dim == 0 && p.death.is_some())
            .count();
        assert_eq!(h0, 3, "tetrahedron-boundary has 3 finite H0 pairs");
        let h1 = res
            .pairs
            .iter()
            .filter(|p| p.dim == 1 && p.death.is_some())
            .count();
        assert_eq!(h1, 3, "tetrahedron-boundary has 3 finite H1 pairs");
    }

    #[test]
    fn reduce_default_config_max_dim_zero() {
        let cfg = TwistConfig::default();
        assert_eq!(cfg.max_dim, 0, "default max_dim must be 0");
    }

    #[test]
    fn reduce_handles_empty_boundary() {
        // Zero columns → zero pairs.  Acceptable (no error).
        let boundary: Vec<Vec<usize>> = Vec::new();
        let dims: Vec<usize> = Vec::new();
        let cfg = TwistConfig { max_dim: 0 };
        let res = TwistReduction::reduce(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
        assert_eq!(res.n_reductions, 0);
        assert_eq!(res.n_cleared, 0);
    }

    #[test]
    fn reduce_result_default_is_empty() {
        let r = TwistResult::default();
        assert!(r.pairs.is_empty());
        assert_eq!(r.n_reductions, 0);
        assert_eq!(r.n_cleared, 0);
    }
}
