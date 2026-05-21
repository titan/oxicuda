//! Persistent cohomology via coboundary matrix reduction.
//!
//! Implements the de Silva-Morozov-Vejdemo-Johansson (2011) cohomology algorithm
//! (Foundations Comp. Math. 11:465) together with the Chen-Kerber (2011) clearing
//! optimisation (Discrete Comput. Geom. 50:266).
//!
//! # Background
//!
//! Standard persistent homology reduces the boundary matrix ∂.  Persistent
//! cohomology instead reduces the coboundary matrix δ = ∂^T.  Both produce the same
//! persistence diagram, but the cohomological path is often faster because the
//! coboundary columns are sparser for "thin" filtrations.
//!
//! # Clearing optimisation (Chen-Kerber 2011)
//!
//! Before reducing δ, run a standard homology reduction on ∂ to identify "paired"
//! simplices (those that appear as a pivot in the reduced ∂).  The clearing lemma
//! states that any column k of δ corresponding to a paired simplex can be zeroed out
//! without changing the outcome.

use crate::error::{TdaError, TdaResult};
use crate::homology::boundary::BoundaryMatrix;
use crate::homology::reduction::reduce_boundary_matrix;
use std::collections::HashMap;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for cohomological persistence computation.
#[derive(Debug, Clone)]
pub struct CohomologyConfig {
    /// Use the clearing optimisation (Chen-Kerber 2011).  Default: `true`.
    pub use_clearing: bool,
    /// Maximum homological dimension to include in the output.  Default: `usize::MAX`.
    pub max_dim: usize,
}

impl Default for CohomologyConfig {
    fn default() -> Self {
        Self {
            use_clearing: true,
            max_dim: usize::MAX,
        }
    }
}

// ─── Result types ─────────────────────────────────────────────────────────────

/// A single persistence pair produced by cohomology reduction.
#[derive(Debug, Clone)]
pub struct CohomologyPair {
    /// Filtration value at birth.
    pub birth: f64,
    /// Filtration value at death (`None` = essential cycle).
    pub death: Option<f64>,
    /// Homological dimension of the pair.
    pub dim: usize,
    /// Simplex index (in the filtration ordering) at which the class is born.
    pub birth_idx: usize,
    /// Simplex index at which the class dies (if finite).
    pub death_idx: Option<usize>,
}

/// Result of a persistent cohomology computation.
#[derive(Debug, Clone)]
pub struct CohomologyResult {
    /// All persistence pairs extracted from the cohomology reduction.
    pub pairs: Vec<CohomologyPair>,
    /// Betti numbers β_d = number of unpaired simplices with `dim = d`.
    /// Length is `max_dim + 1` (or the highest dimension actually present).
    pub betti_numbers: Vec<usize>,
}

// ─── Coboundary matrix ────────────────────────────────────────────────────────

/// Transpose `boundary` to obtain the coboundary matrix δ = ∂^T.
///
/// If `∂[r, c] = 1` then `δ[c, r] = 1`.  The returned matrix has
/// `n_rows = boundary.n_cols` and `n_cols = boundary.n_rows`.
pub fn coboundary_matrix(boundary: &BoundaryMatrix) -> BoundaryMatrix {
    let n = boundary.n_cols;
    let m = boundary.n_rows;

    // δ has n_rows = m, n_cols = n (swap)
    // Column i of δ = set of row indices j such that ∂[i, j] = 1
    //               = set of rows i in ∂ where column j contains i
    // Equivalently: for each column j of ∂ and each row r in that column,
    // add row j to column r of δ.

    let mut cob_columns: Vec<Vec<usize>> = vec![Vec::new(); m];

    for col_idx in 0..n {
        for &row_idx in &boundary.columns[col_idx] {
            // ∂[row_idx, col_idx] = 1  =>  δ[col_idx, row_idx] = 1
            cob_columns[row_idx].push(col_idx);
        }
    }

    // Each column must be sorted (ascending) for the low() / add_cols() operations.
    for col in &mut cob_columns {
        col.sort_unstable();
    }

    BoundaryMatrix {
        n_rows: n,
        n_cols: m,
        columns: cob_columns,
    }
}

// ─── Coboundary reduction ─────────────────────────────────────────────────────

/// Reduce the coboundary matrix using the standard left-to-right column-reduction
/// algorithm over Z₂ (identical to [`reduce_boundary_matrix`] but applied to δ).
///
/// Returns `pivot_col` where `pivot_col[row] = Some(col)` iff column `col` has
/// `low(col) = row` after reduction.
pub fn reduce_coboundary_matrix(coboundary: &mut BoundaryMatrix) -> Vec<Option<usize>> {
    let n = coboundary.n_cols;
    let mut low_to_col: HashMap<usize, usize> = HashMap::new();

    for j in 0..n {
        loop {
            if coboundary.is_zero(j) {
                break;
            }
            let low_j = coboundary.low(j).expect("column is non-zero");
            match low_to_col.get(&low_j).copied() {
                Some(j_prime) => {
                    coboundary.add_cols(j, j_prime);
                }
                None => {
                    low_to_col.insert(low_j, j);
                    break;
                }
            }
        }
    }

    let mut result = vec![None; n];
    for (&row, &col) in &low_to_col {
        if row < n {
            result[row] = Some(col);
        }
    }
    result
}

// ─── Main algorithm ───────────────────────────────────────────────────────────

/// Compute persistent cohomology of a filtered simplicial complex.
///
/// # Arguments
///
/// * `boundary`          — boundary matrix ∂ built from the filtration.
/// * `filtration_values` — filtration value for each simplex (len = n_simplices).
/// * `simplex_dims`      — homological dimension for each simplex (len = n_simplices).
/// * `cfg`               — algorithm configuration.
///
/// # Algorithm (de Silva et al. 2011 + Chen-Kerber 2011)
///
/// 1. Build coboundary matrix δ = ∂^T.
/// 2. If `use_clearing`: run standard homology reduction on a *clone* of ∂ to
///    identify "positive" simplices (those that appear as a pivot row in some column
///    of the reduced ∂).  Zero out the corresponding columns of δ.
/// 3. Reduce δ via standard left-to-right column reduction over Z₂.
/// 4. Extract pairs: column `j` of reduced δ with pivot row `r` → pair
///    (birth = `filtration_values[r]`, death = `filtration_values[j]`, dim = `simplex_dims[r]`).
/// 5. Unpaired simplices (zero columns + not a pivot row in δ) are essential cycles.
///
/// # Errors
///
/// Returns [`TdaError::DimensionMismatch`] if `filtration_values` or `simplex_dims`
/// has a different length than `boundary.n_cols`, or
/// [`TdaError::ParameterOutOfRange`] if any filtration value is NaN.
pub fn persistent_cohomology(
    boundary: &BoundaryMatrix,
    filtration_values: &[f64],
    simplex_dims: &[usize],
    cfg: &CohomologyConfig,
) -> TdaResult<CohomologyResult> {
    let n = boundary.n_cols;

    if filtration_values.len() != n {
        return Err(TdaError::DimensionMismatch {
            expected: n,
            got: filtration_values.len(),
        });
    }
    if simplex_dims.len() != n {
        return Err(TdaError::DimensionMismatch {
            expected: n,
            got: simplex_dims.len(),
        });
    }
    for &v in filtration_values {
        if v.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }

    // Step 1: build coboundary matrix.
    let mut cob = coboundary_matrix(boundary);

    // ── Clearing pre-pass (Chen-Kerber 2011) ───────────────────────────────────
    //
    // Strategy: run standard homology reduction first, collect every (birth, death)
    // pair from ∂.  These pairs are **also** valid in cohomology.  Mark the "death"
    // simplices (columns with pivots in reduced ∂) as "known negative" — their
    // coboundary columns in δ will yield the same pair, so we keep them and let
    // normal reduction handle them.  Mark the "birth" simplices (pivot *rows* in
    // reduced ∂) as "known positive" — their coboundary columns in δ are guaranteed
    // to reduce to zero (they are already "used up" by the pairing), so we may zero
    // them out before reduction.  Zeroing a positive column does NOT lose any pair:
    // the pair is already encoded via the homology pre-pass.
    //
    // After reduction we emit pairs from:
    //   (a) the coboundary columns that survived (reduce to non-zero), AND
    //   (b) the cleared columns — each cleared column r was paired with its death
    //       simplex j as found in the homology pre-pass.

    // `hom_pairs`: birth_idx → death_idx (from homology pre-pass).
    let mut hom_pairs: HashMap<usize, usize> = HashMap::new();
    // `cleared_cols`: columns zeroed before cohomology reduction.
    let mut cleared_cols: std::collections::HashSet<usize> = std::collections::HashSet::new();

    if cfg.use_clearing {
        let mut bm_clone = boundary.clone();
        reduce_boundary_matrix(&mut bm_clone);

        // Collect pairs from homology reduction.
        for col in 0..n {
            if let Some(r) = bm_clone.low(col) {
                // Simplex r is positive (birth), simplex col is negative (death).
                hom_pairs.insert(r, col);
                // Zero out column r of δ (the positive/birth column).
                if r < cob.n_cols {
                    cob.columns[r].clear();
                    cleared_cols.insert(r);
                }
            }
        }
    }

    // ── Step 3: reduce the coboundary matrix ──────────────────────────────────
    reduce_coboundary_matrix(&mut cob);

    // ── Step 4: extract pairs from non-cleared non-zero columns ──────────────
    //
    // Cohomology pairing rule (de Silva et al. 2011):
    //   Column j of reduced δ has pivot row r  →  pair (birth = filt[j], death = filt[r],
    //   dim = dim(simplex j)).
    //
    // Note: r > j because the coboundary maps from lower- to higher-dimensional
    // simplices.  Thus filt[r] ≥ filt[j] always, so birth ≤ death.

    let mut pairs: Vec<CohomologyPair> = Vec::new();

    // Collect the pivot rows used by non-cleared columns (needed for essential detection).
    let mut pivot_rows_from_cob: std::collections::HashSet<usize> =
        std::collections::HashSet::new();

    for j in 0..n {
        // Skip cleared columns — they are handled from hom_pairs below.
        if cleared_cols.contains(&j) {
            continue;
        }
        if let Some(r) = cob.low(j) {
            // Column j of reduced δ has pivot at row r.
            // In cohomology: birth simplex is j, death simplex is r.
            pivot_rows_from_cob.insert(r);
            let birth = filtration_values[j];
            let death = filtration_values[r];
            let dim = simplex_dims[j];

            if dim > cfg.max_dim {
                continue;
            }

            // Skip zero-persistence pairs.
            let tol = f64::EPSILON * birth.abs().max(death.abs()).max(1.0);
            if (death - birth).abs() < tol {
                continue;
            }

            pairs.push(CohomologyPair {
                birth,
                death: Some(death),
                dim,
                birth_idx: j,
                death_idx: Some(r),
            });
        }
    }

    // ── Step 4b: pairs from cleared columns ───────────────────────────────────
    //
    // Each cleared column r was a positive (birth) simplex.  Its pairing with
    // the corresponding death simplex j_death was determined in the homology
    // pre-pass.  The pair is (birth = filt[r], death = filt[j_death], dim = dim(r)).
    for (&birth_idx, &death_idx) in &hom_pairs {
        let birth = filtration_values[birth_idx];
        let death = filtration_values[death_idx];
        let dim = simplex_dims[birth_idx];

        if dim > cfg.max_dim {
            continue;
        }

        let tol = f64::EPSILON * birth.abs().max(death.abs()).max(1.0);
        if (death - birth).abs() < tol {
            continue;
        }

        pairs.push(CohomologyPair {
            birth,
            death: Some(death),
            dim,
            birth_idx,
            death_idx: Some(death_idx),
        });
    }

    // ── Step 5: essential cycles ───────────────────────────────────────────────
    //
    // In cohomology, simplex i is an **essential** (infinite-persistence) cocycle if
    // it belongs to none of the paired sets and is not killed by any cohomology
    // column.
    //
    // With clearing enabled: we determine essentiality from the combined information
    // of the homology pre-pass (which told us which simplices are positive/negative)
    // and the coboundary reduction.
    //
    // A simplex i is essential if and only if:
    //   (a) It is NOT a birth simplex from the homology pre-pass (not cleared), AND
    //   (b) It is NOT a death simplex from the homology pre-pass, AND
    //   (c) It is NOT a pivot row in any non-cleared reduced column of δ (not killed
    //       by a coboundary), AND
    //   (d) [Without clearing] additionally, col i of reduced δ must be zero.
    //
    // When clearing is disabled, the same rule reduces to the simpler:
    //   col i zero AND i not a pivot row.

    // Collect death simplex indices from the homology pre-pass.
    let death_simplices: std::collections::HashSet<usize> = hom_pairs.values().copied().collect();

    for i in 0..n {
        let is_cleared = cleared_cols.contains(&i);
        let is_death = death_simplices.contains(&i);
        let is_pivot_row = pivot_rows_from_cob.contains(&i);

        let essential = if cfg.use_clearing {
            // With clearing: not in any pair (neither birth nor death) and not
            // killed by coboundary.
            !is_cleared && !is_death && !is_pivot_row
        } else {
            // Without clearing: col i is zero (cocycle) and not killed.
            cob.is_zero(i) && !is_pivot_row
        };

        if essential {
            let dim = simplex_dims[i];
            if dim <= cfg.max_dim {
                pairs.push(CohomologyPair {
                    birth: filtration_values[i],
                    death: None,
                    dim,
                    birth_idx: i,
                    death_idx: None,
                });
            }
        }
    }

    // ── Compute Betti numbers ─────────────────────────────────────────────────
    let max_dim_present = pairs
        .iter()
        .map(|p| p.dim)
        .max()
        .unwrap_or(0)
        .min(cfg.max_dim);
    let betti_len = max_dim_present + 1;
    let mut betti_numbers = vec![0usize; betti_len];
    for p in &pairs {
        if p.death.is_none() && p.dim < betti_len {
            betti_numbers[p.dim] += 1;
        }
    }

    Ok(CohomologyResult {
        pairs,
        betti_numbers,
    })
}

// ─── Euler characteristic ─────────────────────────────────────────────────────

/// Compute the Euler characteristic χ = Σ_k (-1)^k β_k from a slice of Betti numbers.
pub fn euler_characteristic(betti_numbers: &[usize]) -> i64 {
    betti_numbers
        .iter()
        .enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum()
}

// ─── Verification helper ──────────────────────────────────────────────────────

/// Verify that cohomology and homology yield the same Betti numbers.
///
/// Returns `Ok(())` if both slices agree element-wise (padding shorter slice with zeros),
/// or `Err(TdaError::ReductionFailed)` if they disagree.
pub fn verify_cohomology_homology_agreement(
    cohomology_bettis: &[usize],
    homology_bettis: &[usize],
) -> TdaResult<()> {
    let len = cohomology_bettis.len().max(homology_bettis.len());
    for i in 0..len {
        let coh = cohomology_bettis.get(i).copied().unwrap_or(0);
        let hom = homology_bettis.get(i).copied().unwrap_or(0);
        if coh != hom {
            return Err(TdaError::ReductionFailed);
        }
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::boundary::BoundaryMatrix;

    /// Build a path-graph boundary matrix for vertices 0,1,2 and edges (0,1),(1,2).
    /// Filtration order: v0,v1,v2,e01,e12.
    fn path_graph_bm() -> (BoundaryMatrix, Vec<f64>, Vec<usize>) {
        // 5 simplices: v0,v1,v2,e01,e12
        // columns: v0=[], v1=[], v2=[], e01=[0,1], e12=[1,2]
        let columns = vec![vec![], vec![], vec![], vec![0usize, 1], vec![1usize, 2]];
        let bm = BoundaryMatrix {
            n_rows: 5,
            n_cols: 5,
            columns,
        };
        let fv = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dims = vec![0, 0, 0, 1, 1];
        (bm, fv, dims)
    }

    /// Build a 1-cycle (triangle without face): v0,v1,v2,e01,e02,e12.
    fn cycle_bm() -> (BoundaryMatrix, Vec<f64>, Vec<usize>) {
        // 6 simplices: v0,v1,v2,e01,e02,e12
        let columns = vec![
            vec![],
            vec![],
            vec![],
            vec![0usize, 1], // e01
            vec![0usize, 2], // e02
            vec![1usize, 2], // e12
        ];
        let bm = BoundaryMatrix {
            n_rows: 6,
            n_cols: 6,
            columns,
        };
        let fv = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let dims = vec![0, 0, 0, 1, 1, 1];
        (bm, fv, dims)
    }

    /// Build a filled triangle: v0,v1,v2,e01,e02,e12,f012.
    fn filled_triangle_bm() -> (BoundaryMatrix, Vec<f64>, Vec<usize>) {
        // 7 simplices: v0,v1,v2,e01,e02,e12,f012
        // f012 has boundary e01⊕e02⊕e12 = rows 3,4,5
        let columns = vec![
            vec![],
            vec![],
            vec![],
            vec![0usize, 1],    // e01
            vec![0usize, 2],    // e02
            vec![1usize, 2],    // e12
            vec![3usize, 4, 5], // f012
        ];
        let bm = BoundaryMatrix {
            n_rows: 7,
            n_cols: 7,
            columns,
        };
        let fv = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let dims = vec![0, 0, 0, 1, 1, 1, 2];
        (bm, fv, dims)
    }

    // ── test 1 ────────────────────────────────────────────────────────────────
    #[test]
    fn coboundary_is_transpose_of_boundary() {
        // 3×3 boundary matrix: col 2 = {0,1}
        let bm = BoundaryMatrix {
            n_rows: 3,
            n_cols: 3,
            columns: vec![vec![0usize, 2], vec![1usize, 2], vec![]],
        };
        let cob = coboundary_matrix(&bm);
        // Row 0 of ∂ appears in col 0: so δ col 0 should contain row 0 → δ col 0 = [0]
        // Row 2 of ∂ appears in cols 0 and 1: so δ col 2 = [0, 1]
        // δ is n_rows=3, n_cols=3
        assert_eq!(cob.n_rows, 3);
        assert_eq!(cob.n_cols, 3);
        // column 0 of δ: rows j where ∂[0,j]=1 → j=0 (since ∂ col 0 has row 0)
        assert!(cob.columns[0].contains(&0));
        // column 2 of δ: rows j where ∂[2,j]=1 → j=0 (∂ col 0 has row 2), j=1 (∂ col 1 has row 2)
        assert!(cob.columns[2].contains(&0));
        assert!(cob.columns[2].contains(&1));
    }

    // ── test 2 ────────────────────────────────────────────────────────────────
    #[test]
    fn coboundary_empty_matrix() {
        let bm = BoundaryMatrix {
            n_rows: 0,
            n_cols: 0,
            columns: vec![],
        };
        let cob = coboundary_matrix(&bm);
        assert_eq!(cob.n_rows, 0);
        assert_eq!(cob.n_cols, 0);
        assert!(cob.columns.is_empty());
    }

    // ── test 3 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_betti_0_path_graph() {
        let (bm, fv, dims) = path_graph_bm();
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        // Path graph: β₀=1, β₁=0
        let b0 = res.betti_numbers.first().copied().unwrap_or(0);
        assert_eq!(b0, 1, "β₀ should be 1 for path graph");
        let b1 = res.betti_numbers.get(1).copied().unwrap_or(0);
        assert_eq!(b1, 0, "β₁ should be 0 for path graph");
    }

    // ── test 4 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_betti_1_cycle() {
        let (bm, fv, dims) = cycle_bm();
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let b0 = res.betti_numbers.first().copied().unwrap_or(0);
        assert_eq!(b0, 1, "β₀ should be 1 for triangle cycle");
        let b1 = res.betti_numbers.get(1).copied().unwrap_or(0);
        assert_eq!(b1, 1, "β₁ should be 1 for triangle cycle");
    }

    // ── test 5 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_betti_filled_triangle() {
        let (bm, fv, dims) = filled_triangle_bm();
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let b0 = res.betti_numbers.first().copied().unwrap_or(0);
        assert_eq!(b0, 1, "β₀ should be 1 for filled triangle");
        let b1 = res.betti_numbers.get(1).copied().unwrap_or(0);
        assert_eq!(b1, 0, "β₁ should be 0 for filled triangle");
    }

    // ── Helpers: homology Betti numbers via standard reduction ────────────────

    fn homology_betti(bm: &BoundaryMatrix, dims: &[usize]) -> Vec<usize> {
        let mut m = bm.clone();
        reduce_boundary_matrix(&mut m);
        let n = m.n_cols;
        let mut pivot_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for j in 0..n {
            if let Some(r) = m.low(j) {
                pivot_rows.insert(r);
            }
        }
        let max_dim = dims.iter().copied().max().unwrap_or(0);
        let mut betti = vec![0usize; max_dim + 1];
        for (i, &d) in dims.iter().enumerate().take(n) {
            if !pivot_rows.contains(&i) && m.is_zero(i) && d <= max_dim {
                betti[d] += 1;
            }
        }
        betti
    }

    // ── test 6 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_agrees_with_homology_path() {
        let (bm, fv, dims) = path_graph_bm();
        let cfg = CohomologyConfig::default();
        let coh = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let hom = homology_betti(&bm, &dims);
        verify_cohomology_homology_agreement(&coh.betti_numbers, &hom).expect("should agree");
    }

    // ── test 7 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_agrees_with_homology_cycle() {
        let (bm, fv, dims) = cycle_bm();
        let cfg = CohomologyConfig::default();
        let coh = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let hom = homology_betti(&bm, &dims);
        verify_cohomology_homology_agreement(&coh.betti_numbers, &hom).expect("should agree");
    }

    // ── test 8 ────────────────────────────────────────────────────────────────
    #[test]
    fn cohomology_agrees_with_homology_filled_triangle() {
        let (bm, fv, dims) = filled_triangle_bm();
        let cfg = CohomologyConfig::default();
        let coh = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let hom = homology_betti(&bm, &dims);
        verify_cohomology_homology_agreement(&coh.betti_numbers, &hom).expect("should agree");
    }

    // ── test 9 ────────────────────────────────────────────────────────────────
    #[test]
    fn clearing_gives_same_result() {
        let (bm, fv, dims) = cycle_bm();
        let cfg_clear = CohomologyConfig {
            use_clearing: true,
            ..Default::default()
        };
        let cfg_no_clear = CohomologyConfig {
            use_clearing: false,
            ..Default::default()
        };
        let res_clear = persistent_cohomology(&bm, &fv, &dims, &cfg_clear).expect("ok");
        let res_no_clear = persistent_cohomology(&bm, &fv, &dims, &cfg_no_clear).expect("ok");
        assert_eq!(
            res_clear.betti_numbers, res_no_clear.betti_numbers,
            "clearing must not change Betti numbers"
        );
    }

    // ── test 10 ───────────────────────────────────────────────────────────────
    #[test]
    fn euler_characteristic_path_graph() {
        // Path graph: β₀=1, β₁=0 → χ = 1
        let betti = vec![1usize, 0];
        assert_eq!(euler_characteristic(&betti), 1);
    }

    // ── test 11 ───────────────────────────────────────────────────────────────
    #[test]
    fn euler_characteristic_cycle() {
        // Single loop: β₀=1, β₁=1 → χ = 0
        let betti = vec![1usize, 1];
        assert_eq!(euler_characteristic(&betti), 0);
    }

    // ── test 12 ───────────────────────────────────────────────────────────────
    #[test]
    fn euler_characteristic_sphere_approx() {
        // Filled triangle (disc): β₀=1, β₁=0, β₂=0 → χ = 1
        let betti = vec![1usize, 0, 0];
        assert_eq!(euler_characteristic(&betti), 1);
    }

    // ── test 13 ───────────────────────────────────────────────────────────────
    #[test]
    fn persistence_pair_birth_less_than_or_equal_death() {
        let (bm, fv, dims) = filled_triangle_bm();
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        for pair in &res.pairs {
            if let Some(d) = pair.death {
                assert!(pair.birth <= d, "birth must be ≤ death");
            }
        }
    }

    // ── test 14 ───────────────────────────────────────────────────────────────
    #[test]
    fn zero_persistence_pairs_with_no_topology() {
        // Single vertex: no finite pairs, one essential β₀.
        let bm = BoundaryMatrix {
            n_rows: 1,
            n_cols: 1,
            columns: vec![vec![]],
        };
        let fv = vec![0.0];
        let dims = vec![0usize];
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        let finite: Vec<_> = res.pairs.iter().filter(|p| p.death.is_some()).collect();
        assert!(finite.is_empty(), "single vertex has no finite pairs");
    }

    // ── test 15 ───────────────────────────────────────────────────────────────
    #[test]
    fn betti_numbers_correct_length_for_max_dim() {
        let (bm, fv, dims) = cycle_bm();
        let cfg = CohomologyConfig::default();
        let res = persistent_cohomology(&bm, &fv, &dims, &cfg).expect("ok");
        // Must have at least 2 entries (β₀ and β₁).
        assert!(
            res.betti_numbers.len() >= 2,
            "need at least β₀ and β₁ entries"
        );
    }

    // ── test 16 ───────────────────────────────────────────────────────────────
    #[test]
    fn verify_agreement_passes() {
        let same = vec![1usize, 1, 0];
        verify_cohomology_homology_agreement(&same, &same).expect("identical arrays should pass");
    }

    // ── test 17 ───────────────────────────────────────────────────────────────
    #[test]
    fn verify_agreement_fails() {
        let a = vec![1usize, 1];
        let b = vec![1usize, 2];
        assert!(
            verify_cohomology_homology_agreement(&a, &b).is_err(),
            "different arrays should fail"
        );
    }

    // ── test 18 ───────────────────────────────────────────────────────────────
    #[test]
    fn err_filtration_length_mismatch() {
        let (bm, _fv, dims) = path_graph_bm();
        let wrong_fv = vec![0.0, 1.0]; // too short
        let cfg = CohomologyConfig::default();
        assert!(
            persistent_cohomology(&bm, &wrong_fv, &dims, &cfg).is_err(),
            "mismatched filtration_values length should error"
        );
    }
}
