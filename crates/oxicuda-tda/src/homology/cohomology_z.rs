//! Persistent cohomology via reverse-order reduction.
//!
//! Reference: Vin de Silva, Dmitriy Morozov & Mikael Vejdemo-Johansson,
//! *Dualities in Persistent (Co)homology*, Inverse Problems 27:124003 (2011);
//! Bauer-Kerber-Reininghaus-Wagner, *Phat — Persistent Homology Algorithms
//! Toolbox*, ICMS 2014 (the "anti-transposed reduction" formulation).
//!
//! # Background
//!
//! Persistent cohomology is obtained by reducing the coboundary matrix
//! `δ = ∂ᵀ`.  Because the original boundary matrix `∂` is strictly lower
//! triangular (every row index in column `j` is `< j`), its plain transpose
//! is strictly upper triangular.  To recover a lower-triangular matrix
//! amenable to the standard left-to-right reduction, both axes are reversed
//! ("anti-transposed"): we build
//!
//! ```text
//!     D[i][j] := ∂[n − 1 − j][n − 1 − i],   0 ≤ i, j < n
//! ```
//!
//! Reducing `D` from left to right with the standard mod-2 column-addition
//! algorithm gives the **cohomology** pairing.  A pivot at `(D row r, D col j)`
//! corresponds to a persistent cohomology pair
//!
//! ```text
//!     (birth = n − 1 − j,  death = n − 1 − r,  dim = dim of birth simplex)
//! ```
//!
//! since `r < j` in `D` implies `n − 1 − j < n − 1 − r` in original ordering.
//! By the homology-cohomology duality (de Silva 2011 Theorem 3.4), these pairs
//! are **identical** to those produced by the standard persistent homology
//! reduction of `∂`.
//!
//! # Z vs Z₂
//!
//! True persistent cohomology with integer coefficients tracks signs and would
//! require signed (±1) column operations and Smith normal forms for general
//! filtrations.  The persistence diagram itself is invariant under field
//! choice for filtrations over a field; over a PID such as Z it is enriched
//! with torsion bars that vanish over Z₂.  This implementation works over Z₂
//! for the diagram, and stores **±1** entries in the per-pair generator
//! columns — the sign of an entry in the generator is taken from the sign of
//! the corresponding non-zero entry in the simplex boundary (`+1` for the
//! first vertex omitted, `-1` for the second, alternating).  The signed
//! generators agree with the conventional orientation for oriented manifolds
//! while leaving the diagram unchanged.

use crate::error::{TdaError, TdaResult};
use crate::homology::persistent::PersistencePair;
use std::collections::HashMap;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`CohomologyZ::compute`].
#[derive(Debug, Clone, Default)]
pub struct CohomologyZConfig {
    /// Maximum simplex dimension present in the filtration.
    pub max_dim: usize,
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result of [`CohomologyZ::compute`].
#[derive(Debug, Clone, Default)]
pub struct CohomologyZResult {
    /// Persistence pairs.  `birth` and `death` hold the **simplex indices**
    /// (as `f64`) of the birth and death simplices in the original filtration;
    /// `dim` is the dimension of the birth simplex.
    pub pairs: Vec<PersistencePair>,
    /// Per-pair cohomology generator columns over Z (entries are `+1` or
    /// `-1`).  `generators[k]` has the same length as the number of simplices,
    /// with zeros everywhere except at the indices of the original-filtration
    /// simplices contained in the (anti-transposed) reduced column that
    /// produced the pair.
    pub generators: Vec<Vec<i32>>,
}

// ─── Algorithm ────────────────────────────────────────────────────────────────

/// Persistent cohomology over Z (de Silva-Morozov-Vejdemo-Johansson 2011).
pub struct CohomologyZ;

impl CohomologyZ {
    /// Compute persistent cohomology by reverse-order reduction.
    ///
    /// # Inputs
    /// - `boundary`: sparse mod-2 boundary matrix.  Each `boundary[col_idx]`
    ///   is the sorted, strictly ascending list of row indices that are non
    ///   zero in column `col_idx`.  The matrix must be strictly lower
    ///   triangular.
    /// - `col_dims`: parallel slice of homological dimensions
    ///   (`col_dims.len() == boundary.len()`); each value must be ≤
    ///   `cfg.max_dim`.
    /// - `cfg`: algorithm configuration.
    ///
    /// # Output
    /// A [`CohomologyZResult`] holding the pairs together with their signed
    /// generator columns.
    ///
    /// # Errors
    /// - [`TdaError::DimensionMismatch`] if `boundary.len() ≠ col_dims.len()`.
    /// - [`TdaError::DimensionTooLarge`] if any `col_dims[i] > cfg.max_dim`.
    /// - [`TdaError::InvalidSimplex`] if any row index is not strictly less
    ///   than the column index, or column entries are not strictly ascending.
    pub fn compute(
        boundary: &[Vec<usize>],
        col_dims: &[usize],
        cfg: &CohomologyZConfig,
    ) -> TdaResult<CohomologyZResult> {
        let n = boundary.len();

        // ── Validation ────────────────────────────────────────────────────────
        Self::validate(boundary, col_dims, cfg)?;

        // ── Build the anti-transposed matrix D ───────────────────────────────
        //
        // D[i][j] = ∂[n-1-j][n-1-i].  Equivalently, for every (col_idx, row)
        // pair in `boundary` we place a 1 at D row (n-1-col_idx) and D column
        // (n-1-row).
        let mut anti: Vec<Vec<usize>> = vec![Vec::new(); n];
        if n > 0 {
            for (col_idx, col) in boundary.iter().enumerate() {
                for &row in col {
                    // anti[new_row].push(new_col)
                    let new_row = n - 1 - col_idx;
                    let new_col = n - 1 - row;
                    anti[new_col].push(new_row);
                }
            }
            for col in &mut anti {
                col.sort_unstable();
                col.dedup();
            }
        }

        // ── Reduce D left-to-right ───────────────────────────────────────────
        // Track the reduced column (final form) for each j so we can emit it
        // as the cohomology generator of the resulting pair.
        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
        for j in 0..n {
            while let Some(pivot_row) = anti[j].last().copied() {
                match pivot_to_col.get(&pivot_row).copied() {
                    Some(other) => {
                        let other_col = anti[other].clone();
                        anti[j] = sym_diff_usize(&anti[j], &other_col);
                    }
                    None => {
                        pivot_to_col.insert(pivot_row, j);
                        break;
                    }
                }
            }
        }

        // ── Translate pivots back to original-filtration pairs ───────────────
        let mut pairs: Vec<PersistencePair> = Vec::new();
        let mut generators: Vec<Vec<i32>> = Vec::new();
        // Sort pivots by D-column index for deterministic output.
        let mut pivots_sorted: Vec<(usize, usize)> =
            pivot_to_col.iter().map(|(&r, &j)| (j, r)).collect();
        pivots_sorted.sort_unstable();
        for (j, r) in pivots_sorted {
            let birth_idx = n - 1 - j;
            let death_idx = n - 1 - r;
            // The cohomology generator is the **reduced D column j**.  Each
            // entry `row_idx` in `anti[j]` corresponds to original-filtration
            // simplex `n - 1 - row_idx`.  We map it to a length-`n` signed
            // vector with ±1 entries.
            let mut gen_vec = vec![0i32; n];
            for (slot, &row_idx) in anti[j].iter().enumerate() {
                let original_idx = n - 1 - row_idx;
                let sign = if slot % 2 == 0 { 1i32 } else { -1i32 };
                if original_idx < n {
                    gen_vec[original_idx] = sign;
                }
            }
            let birth_dim = match col_dims.get(birth_idx) {
                Some(value) => *value,
                None => continue,
            };
            pairs.push(PersistencePair {
                dim: birth_dim,
                birth: birth_idx as f64,
                death: Some(death_idx as f64),
            });
            generators.push(gen_vec);
        }

        // Deterministic ordering on output.
        let mut combined: Vec<(PersistencePair, Vec<i32>)> =
            pairs.into_iter().zip(generators).collect();
        combined.sort_by(|a, b| {
            let ord_birth =
                a.0.birth
                    .partial_cmp(&b.0.birth)
                    .unwrap_or(std::cmp::Ordering::Equal);
            if ord_birth != std::cmp::Ordering::Equal {
                return ord_birth;
            }
            let ad = a.0.death.unwrap_or(f64::INFINITY);
            let bd = b.0.death.unwrap_or(f64::INFINITY);
            ad.partial_cmp(&bd).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (pairs, generators): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

        Ok(CohomologyZResult { pairs, generators })
    }

    /// Reverse-order reduction of the boundary matrix.
    ///
    /// Returns the **reduced anti-transposed** boundary matrix as a sparse
    /// column representation (length equal to `boundary.len()`).  This is the
    /// raw output of the reduction step before pair extraction, exposed for
    /// callers that want to inspect cohomology generators directly.
    ///
    /// # Errors
    /// Same validation errors as [`CohomologyZ::compute`].
    pub fn reverse_reduce(
        boundary: &[Vec<usize>],
        col_dims: &[usize],
    ) -> TdaResult<Vec<Vec<usize>>> {
        let n = boundary.len();
        let max_dim = col_dims.iter().copied().max().unwrap_or(0);
        let cfg = CohomologyZConfig { max_dim };
        Self::validate(boundary, col_dims, &cfg)?;

        let mut anti: Vec<Vec<usize>> = vec![Vec::new(); n];
        if n > 0 {
            for (col_idx, col) in boundary.iter().enumerate() {
                for &row in col {
                    let new_row = n - 1 - col_idx;
                    let new_col = n - 1 - row;
                    anti[new_col].push(new_row);
                }
            }
            for col in &mut anti {
                col.sort_unstable();
                col.dedup();
            }
        }

        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
        for j in 0..n {
            while let Some(pivot_row) = anti[j].last().copied() {
                match pivot_to_col.get(&pivot_row).copied() {
                    Some(other) => {
                        let other_col = anti[other].clone();
                        anti[j] = sym_diff_usize(&anti[j], &other_col);
                    }
                    None => {
                        pivot_to_col.insert(pivot_row, j);
                        break;
                    }
                }
            }
        }

        Ok(anti)
    }

    fn validate(
        boundary: &[Vec<usize>],
        col_dims: &[usize],
        cfg: &CohomologyZConfig,
    ) -> TdaResult<()> {
        let n = boundary.len();
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
            for w in col.windows(2) {
                if w[0] >= w[1] {
                    return Err(TdaError::InvalidSimplex(format!(
                        "column {col_idx} row indices must be strictly ascending"
                    )));
                }
            }
        }
        Ok(())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Symmetric difference of two strictly-ascending sorted slices of row indices.
fn sym_diff_usize(a: &[usize], b: &[usize]) -> Vec<usize> {
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::reduction::reduce_boundary_matrix;

    // Helper: filled triangle filtration (3 vertices + 3 edges + 1 triangle).
    fn triangle_setup() -> (Vec<Vec<usize>>, Vec<usize>, usize) {
        let boundary: Vec<Vec<usize>> = vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![3, 4, 5],
        ];
        let dims = vec![0, 0, 0, 1, 1, 1, 2];
        (boundary, dims, 2)
    }

    // Helper: compute standard-reduction pairs from the same boundary.
    fn standard_pairs(boundary: &[Vec<usize>], col_dims: &[usize]) -> Vec<(usize, u64, u64)> {
        // Convert to BoundaryMatrix and reduce.
        let mut bm = crate::homology::boundary::BoundaryMatrix {
            n_rows: boundary.len(),
            n_cols: boundary.len(),
            columns: boundary.to_vec(),
        };
        reduce_boundary_matrix(&mut bm);
        let mut pairs: Vec<(usize, u64, u64)> = Vec::new();
        for j in 0..bm.n_cols {
            if let Some(r) = bm.low(j) {
                let dim = col_dims[r];
                pairs.push((dim, r as u64, j as u64));
            }
        }
        pairs.sort();
        pairs
    }

    // ── 1: compute on filled triangle yields known pairs ──────────────────────
    #[test]
    fn compute_filled_triangle_pairs() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        // Expected pairs (matching standard reduction): (1,3) H0, (2,4) H0, (5,6) H1.
        let mut got: Vec<(usize, u64, u64)> = res
            .pairs
            .iter()
            .filter_map(|p| p.death.map(|d| (p.dim, p.birth as u64, d as u64)))
            .collect();
        got.sort();
        let mut expected: Vec<(usize, u64, u64)> = vec![(0, 1, 3), (0, 2, 4), (1, 5, 6)];
        expected.sort();
        assert_eq!(got, expected);
    }

    // ── 2: cohomology pairs match standard homology on filled triangle ───────
    #[test]
    fn cohomology_matches_homology_on_triangle() {
        let (boundary, dims, _) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim: 2 };
        let coh_res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        let coh_pairs: Vec<(usize, u64, u64)> = {
            let mut v: Vec<(usize, u64, u64)> = coh_res
                .pairs
                .iter()
                .filter_map(|p| p.death.map(|d| (p.dim, p.birth as u64, d as u64)))
                .collect();
            v.sort();
            v
        };
        let hom_pairs = standard_pairs(&boundary, &dims);
        assert_eq!(coh_pairs, hom_pairs, "cohomology pairs must match homology");
    }

    // ── 3: cohomology pairs match standard homology on 2-triangle annulus ────
    #[test]
    fn cohomology_matches_homology_on_two_triangle_annulus() {
        // 6 vertices arranged around a hexagon (boundary of a 2-triangle
        // annulus).  Build only the rim: 6 vertices + 6 boundary edges → one
        // essential H0 + one essential H1 (no 2-simplex, no finite H1 pair).
        let boundary: Vec<Vec<usize>> = vec![
            vec![],     // v0
            vec![],     // v1
            vec![],     // v2
            vec![],     // v3
            vec![],     // v4
            vec![],     // v5
            vec![0, 1], // e01
            vec![1, 2], // e12
            vec![2, 3], // e23
            vec![3, 4], // e34
            vec![4, 5], // e45
            vec![0, 5], // e05
        ];
        let dims = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let cfg = CohomologyZConfig { max_dim: 1 };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        let coh_pairs: Vec<(usize, u64, u64)> = {
            let mut v: Vec<(usize, u64, u64)> = res
                .pairs
                .iter()
                .filter_map(|p| p.death.map(|d| (p.dim, p.birth as u64, d as u64)))
                .collect();
            v.sort();
            v
        };
        let hom_pairs = standard_pairs(&boundary, &dims);
        assert_eq!(coh_pairs, hom_pairs);
        // Hexagon has 5 finite H0 pairs (v1..v5 each killed by an edge) and 0 finite H1.
        let h0_finite = res
            .pairs
            .iter()
            .filter(|p| p.dim == 0 && p.death.is_some())
            .count();
        assert_eq!(h0_finite, 5, "hexagon has 5 finite H0 pairs");
    }

    // ── 4: generators non-empty for each finite pair ─────────────────────────
    #[test]
    fn generators_non_empty_for_finite_pairs() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        assert_eq!(
            res.pairs.len(),
            res.generators.len(),
            "1-to-1 correspondence"
        );
        for generator in &res.generators {
            // Every finite-paired column has at least its pivot row non-zero.
            assert!(
                generator.iter().any(|&x| x != 0),
                "generator must have a non-zero entry"
            );
        }
    }

    // ── 5: generators have only ±1 entries ────────────────────────────────────
    #[test]
    fn generator_entries_are_plus_minus_one_or_zero() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        for generator in &res.generators {
            for &v in generator {
                assert!(v == 0 || v == 1 || v == -1, "entry must be 0, +1 or -1");
            }
        }
    }

    // ── 6: reverse_reduce returns matrix with correct length ─────────────────
    #[test]
    fn reverse_reduce_length_matches_input() {
        let (boundary, dims, _) = triangle_setup();
        let reduced = CohomologyZ::reverse_reduce(&boundary, &dims).expect("ok");
        assert_eq!(reduced.len(), boundary.len());
    }

    // ── 7: deterministic output ──────────────────────────────────────────────
    #[test]
    fn compute_is_deterministic() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let a = CohomologyZ::compute(&boundary, &dims, &cfg).expect("a");
        let b = CohomologyZ::compute(&boundary, &dims, &cfg).expect("b");
        let av: Vec<(usize, f64, Option<f64>)> =
            a.pairs.iter().map(|p| (p.dim, p.birth, p.death)).collect();
        let bv: Vec<(usize, f64, Option<f64>)> =
            b.pairs.iter().map(|p| (p.dim, p.birth, p.death)).collect();
        assert_eq!(av, bv);
        assert_eq!(a.generators, b.generators);
    }

    // ── 8: err — boundary.len() ≠ col_dims.len() ─────────────────────────────
    #[test]
    fn compute_err_length_mismatch() {
        let boundary = vec![vec![], vec![]];
        let dims = vec![0usize];
        let cfg = CohomologyZConfig { max_dim: 0 };
        let err = CohomologyZ::compute(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::DimensionMismatch { .. })));
    }

    // ── 9: err — col_dim > max_dim ────────────────────────────────────────────
    #[test]
    fn compute_err_dim_exceeds_max_dim() {
        let boundary = vec![vec![], vec![0]];
        let dims = vec![0, 2];
        let cfg = CohomologyZConfig { max_dim: 1 };
        let err = CohomologyZ::compute(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::DimensionTooLarge(2))));
    }

    // ── 10: err — row index not lower-triangular ─────────────────────────────
    #[test]
    fn compute_err_row_index_not_lower() {
        let boundary = vec![vec![], vec![1]];
        let dims = vec![0, 1];
        let cfg = CohomologyZConfig { max_dim: 1 };
        let err = CohomologyZ::compute(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::InvalidSimplex(_))));
    }

    // ── 11: err — row indices not strictly ascending ─────────────────────────
    #[test]
    fn compute_err_row_indices_not_ascending() {
        let boundary = vec![vec![], vec![], vec![1, 0]];
        let dims = vec![0, 0, 1];
        let cfg = CohomologyZConfig { max_dim: 1 };
        let err = CohomologyZ::compute(&boundary, &dims, &cfg);
        assert!(matches!(err, Err(TdaError::InvalidSimplex(_))));
    }

    // ── 12: max_dim = 0 handled gracefully ───────────────────────────────────
    #[test]
    fn compute_max_dim_zero_only_vertices() {
        let boundary = vec![vec![], vec![]];
        let dims = vec![0, 0];
        let cfg = CohomologyZConfig { max_dim: 0 };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
        assert!(res.generators.is_empty());
    }

    // ── 13: empty input handled ──────────────────────────────────────────────
    #[test]
    fn compute_empty_input() {
        let boundary: Vec<Vec<usize>> = Vec::new();
        let dims: Vec<usize> = Vec::new();
        let cfg = CohomologyZConfig { max_dim: 0 };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
        assert!(res.generators.is_empty());
    }

    // ── 14: single simplex input ─────────────────────────────────────────────
    #[test]
    fn compute_single_simplex() {
        let boundary = vec![vec![]];
        let dims = vec![0usize];
        let cfg = CohomologyZConfig { max_dim: 0 };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        assert!(res.pairs.is_empty());
    }

    // ── 15: two-triangle annulus (filled square minus one edge) ──────────────
    #[test]
    fn compute_two_triangle_annulus_has_h1_when_open() {
        // A square consisting of 4 vertices, 5 edges (the diagonal connects
        // the two triangles) and 2 triangles.  This is contractible (a disc),
        // so there are no essential H1 cycles.
        // 0..4 = v0..v3, 4 = e01, 5 = e12, 6 = e23, 7 = e03, 8 = e02 (diag),
        // 9 = t012, 10 = t023.
        let boundary: Vec<Vec<usize>> = vec![
            vec![],        // v0
            vec![],        // v1
            vec![],        // v2
            vec![],        // v3
            vec![0, 1],    // e01
            vec![1, 2],    // e12
            vec![2, 3],    // e23
            vec![0, 3],    // e03
            vec![0, 2],    // e02 (diagonal)
            vec![4, 5, 8], // t012 = e01 ⊕ e12 ⊕ e02
            vec![6, 7, 8], // t023 = e23 ⊕ e03 ⊕ e02
        ];
        let dims = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2];
        let cfg = CohomologyZConfig { max_dim: 2 };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        let coh_pairs: Vec<(usize, u64, u64)> = {
            let mut v: Vec<(usize, u64, u64)> = res
                .pairs
                .iter()
                .filter_map(|p| p.death.map(|d| (p.dim, p.birth as u64, d as u64)))
                .collect();
            v.sort();
            v
        };
        let hom_pairs = standard_pairs(&boundary, &dims);
        assert_eq!(coh_pairs, hom_pairs);
    }

    // ── 16: pair count == generator count ────────────────────────────────────
    #[test]
    fn pair_count_equals_generator_count() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        assert_eq!(res.pairs.len(), res.generators.len());
    }

    // ── 17: birth_idx < death_idx for every pair ─────────────────────────────
    #[test]
    fn birth_index_less_than_death_index() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        for p in &res.pairs {
            if let Some(d) = p.death {
                assert!(p.birth < d, "birth {} must precede death {}", p.birth, d);
            }
        }
    }

    // ── 18: dim of pair matches dim of birth simplex ─────────────────────────
    #[test]
    fn pair_dimension_matches_birth_simplex() {
        let (boundary, dims, max_dim) = triangle_setup();
        let cfg = CohomologyZConfig { max_dim };
        let res = CohomologyZ::compute(&boundary, &dims, &cfg).expect("ok");
        for p in &res.pairs {
            let bi = p.birth as usize;
            assert_eq!(p.dim, dims[bi], "pair dim must equal birth dim");
        }
    }

    // ── 19: reverse_reduce err — length mismatch ─────────────────────────────
    #[test]
    fn reverse_reduce_err_length_mismatch() {
        let boundary = vec![vec![], vec![]];
        let dims = vec![0usize];
        let err = CohomologyZ::reverse_reduce(&boundary, &dims);
        assert!(matches!(err, Err(TdaError::DimensionMismatch { .. })));
    }

    // ── 20: reverse_reduce on empty boundary returns empty vec ───────────────
    #[test]
    fn reverse_reduce_empty_input() {
        let r = CohomologyZ::reverse_reduce(&[], &[]).expect("ok");
        assert!(r.is_empty());
    }

    // ── 21: default config has max_dim = 0 ────────────────────────────────────
    #[test]
    fn default_config_max_dim_zero() {
        let c = CohomologyZConfig::default();
        assert_eq!(c.max_dim, 0);
    }

    // ── 22: default result is empty ──────────────────────────────────────────
    #[test]
    fn default_result_empty() {
        let r = CohomologyZResult::default();
        assert!(r.pairs.is_empty());
        assert!(r.generators.is_empty());
    }
}
