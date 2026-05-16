//! Extraction of persistence pairs from a reduced boundary matrix.

use crate::complex::filtration::Filtration;
use crate::error::{TdaError, TdaResult};
use crate::homology::boundary::BoundaryMatrix;

/// A persistence pair: the birth and (optional) death filtration values of a homological cycle.
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Homological dimension of the cycle (0 = connected components, 1 = loops, 2 = voids, …).
    pub dim: usize,
    /// Filtration value at which the cycle is born.
    pub birth: f64,
    /// Filtration value at which the cycle dies (`None` = essential / infinite persistence).
    pub death: Option<f64>,
}

impl PersistencePair {
    /// Persistence lifetime: `death - birth` (or `default_death - birth` for essential classes).
    pub fn persistence(&self, default_death: f64) -> f64 {
        self.death.unwrap_or(default_death) - self.birth
    }

    /// Whether the cycle is essential (never dies within the filtration).
    pub fn is_essential(&self) -> bool {
        self.death.is_none()
    }
}

/// Extract persistence pairs from a reduced boundary matrix together with the filtration that
/// was used to build it.
///
/// **Prerequisite**: `matrix` must already be in reduced form (output of
/// `reduce_boundary_matrix`).  Pass the `pivot_col` vector returned by that function.
///
/// Algorithm:
/// - For each column j with nonzero pivot row i:
///   simplex i (dim d) is born at `filtration[i].value`, killed by simplex j (dim d+1)
///   at `filtration[j].value`.  Pair dim = d.
/// - Simplex i is an **essential** d-cycle if no column j has low(j) = i AND column i
///   itself is zero (i.e. it is not a boundary).
/// - Zero-persistence pairs (birth == death) are filtered out.
pub fn extract_persistence_pairs(
    matrix: &BoundaryMatrix,
    filtration: &Filtration,
) -> TdaResult<Vec<PersistencePair>> {
    let n = filtration.simplices.len();
    if matrix.n_cols != n {
        return Err(TdaError::DimensionMismatch {
            expected: n,
            got: matrix.n_cols,
        });
    }

    // pivot_rows: set of row indices that are claimed as pivots by some column j
    let mut pivot_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // paired: (birth_idx, death_idx) pairs
    let mut pairs: Vec<PersistencePair> = Vec::new();

    for j in 0..n {
        if let Some(i) = matrix.low(j) {
            pivot_rows.insert(i);
            let birth = filtration.simplices[i].value;
            let death = filtration.simplices[j].value;
            let dim = filtration.simplices[i].simplex.dim();
            // Filter out zero-persistence pairs
            if (death - birth).abs() < f64::EPSILON * birth.abs().max(death.abs()).max(1.0) {
                continue;
            }
            pairs.push(PersistencePair {
                dim,
                birth,
                death: Some(death),
            });
        }
    }

    // Essential classes: simplex i is essential if:
    // 1. Its column is zero (it is a cycle, not a boundary itself), AND
    // 2. No other column j has low(j) = i (it was never killed).
    for i in 0..n {
        if !pivot_rows.contains(&i) && matrix.is_zero(i) {
            let dim = filtration.simplices[i].simplex.dim();
            let birth = filtration.simplices[i].value;
            pairs.push(PersistencePair {
                dim,
                birth,
                death: None,
            });
        }
    }

    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::filtration::{FilteredSimplex, Filtration};
    use crate::complex::simplex::Simplex;
    use crate::homology::boundary::BoundaryMatrix;
    use crate::homology::reduction::reduce_boundary_matrix;

    #[test]
    fn single_edge_gives_one_component() {
        // 2 vertices + 1 edge: H0 = 1 (one connected component after edge merges them)
        let simplices = vec![
            FilteredSimplex {
                simplex: Simplex { vertices: vec![0] },
                value: 0.0,
            },
            FilteredSimplex {
                simplex: Simplex { vertices: vec![1] },
                value: 0.0,
            },
            FilteredSimplex {
                simplex: Simplex {
                    vertices: vec![0, 1],
                },
                value: 1.0,
            },
        ];
        let filt = Filtration::new(simplices).expect("ok");
        let mut bm = BoundaryMatrix::from_filtration(&filt).expect("ok");
        reduce_boundary_matrix(&mut bm);
        let pairs = extract_persistence_pairs(&bm, &filt).expect("ok");

        // Should have: 1 finite H0 pair (vertex 1 born at 0, killed at 1),
        //              1 essential H0 class (vertex 0 — connected component persists).
        let finite: Vec<_> = pairs.iter().filter(|p| p.death.is_some()).collect();
        let essential: Vec<_> = pairs.iter().filter(|p| p.death.is_none()).collect();
        assert_eq!(finite.len(), 1);
        assert_eq!(essential.len(), 1);
        assert_eq!(essential[0].dim, 0);
    }
}
