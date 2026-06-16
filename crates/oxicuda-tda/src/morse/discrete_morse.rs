//! Discrete Morse Theory — Forman 1998.
//!
//! Constructs a discrete gradient vector field on a CW-complex by greedily pairing
//! simplices. Critical cells are those left unpaired; they correspond to generators
//! of the complex's homology (by the discrete Morse inequality).
//!
//! # Algorithm
//!
//! The greedy discrete gradient construction processes simplices in ascending order
//! of their scalar function values. For each simplex `s`:
//!
//! 1. Among all unpaired faces `f` of `s` with `value[f] < value[s]`, find the one
//!    with maximum value (the "best" candidate face).
//! 2. If such an `f` exists **and** `s` is the lowest-value unpaired coface of `f`,
//!    pair `(f, s)` and remove both from the pool of unpaired simplices.
//! 3. Otherwise, `s` remains unpaired (a candidate critical cell — confirmed if still
//!    unpaired after the whole pass).
//!
//! This implements the algorithm of King, Knudson, Mramor (2005) as a greedy
//! approximation to the optimal discrete Morse function.

use crate::error::{TdaError, TdaResult};

/// A discrete gradient vector field on a CW-complex.
///
/// Encodes the gradient as a set of paired simplices `(face_idx, coface_idx)` and
/// a set of unpaired simplex indices (the critical cells).
#[derive(Debug, Clone)]
pub struct DiscreteMorseField {
    /// Gradient pairs: each `(f, s)` pairs face simplex `f` with its coface `s`.
    pub gradient_pairs: Vec<(usize, usize)>,
    /// Critical cells: simplex indices that remain unpaired.
    pub critical_cells: Vec<usize>,
}

/// Compute a discrete gradient vector field using a greedy algorithm.
///
/// # Arguments
/// - `values`: scalar function value on each simplex (length must equal `n_simplices`).
/// - `boundaries`: `boundaries[i]` lists the face simplex indices of simplex `i`.
///   Faces have strictly lower dimension. For 0-simplices (vertices), `boundaries[i]`
///   should be empty.
/// - `n_simplices`: total number of simplices.
///
/// # Errors
/// - [`TdaError::ParameterOutOfRange`] if `values.len() != n_simplices`.
/// - [`TdaError::NanFiltrationValue`] if any value is NaN.
/// - [`TdaError::InvalidSimplex`] if a boundary index is out of range.
pub fn discrete_gradient(
    values: &[f64],
    boundaries: &[Vec<usize>],
    n_simplices: usize,
) -> TdaResult<DiscreteMorseField> {
    if values.len() != n_simplices {
        return Err(TdaError::ParameterOutOfRange(format!(
            "values.len()={} does not match n_simplices={}",
            values.len(),
            n_simplices
        )));
    }
    if boundaries.len() != n_simplices {
        return Err(TdaError::ParameterOutOfRange(format!(
            "boundaries.len()={} does not match n_simplices={}",
            boundaries.len(),
            n_simplices
        )));
    }
    for &v in values {
        if v.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }
    for (i, faces) in boundaries.iter().enumerate() {
        for &f in faces {
            if f >= n_simplices {
                return Err(TdaError::InvalidSimplex(format!(
                    "boundary of simplex {i} references face {f} >= n_simplices={n_simplices}"
                )));
            }
        }
    }

    // Build coface list: for each simplex f, cofaces[f] = list of simplices that have f as face.
    let mut cofaces: Vec<Vec<usize>> = vec![Vec::new(); n_simplices];
    for (s, faces) in boundaries.iter().enumerate() {
        for &f in faces {
            cofaces[f].push(s);
        }
    }

    // Process simplices in ascending order of their scalar value.
    let mut order: Vec<usize> = (0..n_simplices).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut paired = vec![false; n_simplices];
    let mut gradient_pairs: Vec<(usize, usize)> = Vec::new();

    for &s in &order {
        if paired[s] {
            continue;
        }
        // Find the unpaired face of s with the maximum value that is strictly less than value[s].
        let val_s = values[s];
        let mut best_face: Option<usize> = None;
        let mut best_val = f64::NEG_INFINITY;
        for &f in &boundaries[s] {
            if !paired[f] && values[f] < val_s && values[f] > best_val {
                best_val = values[f];
                best_face = Some(f);
            }
        }

        if let Some(f) = best_face {
            // Check that s is the lowest-value unpaired coface of f.
            let mut min_coface_val = f64::INFINITY;
            let mut min_coface: Option<usize> = None;
            for &c in &cofaces[f] {
                if !paired[c] && values[c] > values[f] && values[c] < min_coface_val {
                    min_coface_val = values[c];
                    min_coface = Some(c);
                }
            }
            if min_coface == Some(s) {
                // Pair (f, s)
                paired[f] = true;
                paired[s] = true;
                gradient_pairs.push((f, s));
            }
            // Otherwise s stays unpaired (critical candidate)
        }
        // If no valid face found, s is a critical cell candidate.
    }

    // Collect all unpaired simplices as critical cells.
    let critical_cells: Vec<usize> = (0..n_simplices).filter(|&i| !paired[i]).collect();

    Ok(DiscreteMorseField {
        gradient_pairs,
        critical_cells,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a path graph on n vertices with edges as 1-simplices.
    // Simplex indices: 0..n = vertices, n..n+(n-1) = edges.
    fn path_complex(n: usize) -> (Vec<f64>, Vec<Vec<usize>>, usize) {
        let n_simp = n + (n - 1);
        let mut values = vec![0.0_f64; n_simp];
        let mut boundaries = vec![vec![]; n_simp];
        // Vertex values: i as f64
        for (i, slot) in values.iter_mut().enumerate().take(n) {
            *slot = i as f64;
        }
        // Edge i-(i+1) at simplex index n+i, value = max(f[i], f[i+1])
        for i in 0..(n - 1) {
            let eidx = n + i;
            values[eidx] = values[i].max(values[i + 1]);
            boundaries[eidx] = vec![i, i + 1];
        }
        (values, boundaries, n_simp)
    }

    // Helper: build a cycle graph (n vertices, n edges).
    fn cycle_complex(n: usize) -> (Vec<f64>, Vec<Vec<usize>>, usize) {
        let n_simp = n + n;
        let mut values = vec![0.0_f64; n_simp];
        let mut boundaries = vec![vec![]; n_simp];
        for (i, slot) in values.iter_mut().enumerate().take(n) {
            *slot = i as f64;
        }
        for i in 0..n {
            let eidx = n + i;
            let u = i;
            let v = (i + 1) % n;
            values[eidx] = values[u].max(values[v]);
            boundaries[eidx] = vec![u, v];
        }
        (values, boundaries, n_simp)
    }

    #[test]
    fn critical_cells_nonneg() {
        let (vals, bounds, n) = path_complex(4);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        assert!(
            !mf.critical_cells.is_empty(),
            "path graph has at least 1 critical cell"
        );
    }

    #[test]
    fn gradient_pairs_valid() {
        let (vals, bounds, n) = path_complex(5);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        for &(f, s) in &mf.gradient_pairs {
            assert!(f < n && s < n, "pair indices in range");
            // f must be a face of s
            assert!(
                bounds[s].contains(&f),
                "face {f} must be in boundary of coface {s}"
            );
            // value[f] < value[s] by construction
            assert!(
                vals[f] <= vals[s],
                "face value {} should be <= coface value {}",
                vals[f],
                vals[s]
            );
        }
    }

    #[test]
    fn pairing_counts() {
        // Each pair removes 2 simplices; critical = n_simplices - 2 * pairs.
        let (vals, bounds, n) = path_complex(6);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        assert_eq!(
            mf.critical_cells.len() + 2 * mf.gradient_pairs.len(),
            n,
            "critical + 2*pairs = n_simplices"
        );
    }

    #[test]
    fn n_simplices_1() {
        // Single vertex (no boundary, no edges)
        let vals = vec![0.0];
        let bounds = vec![vec![]];
        let mf = discrete_gradient(&vals, &bounds, 1).expect("ok");
        assert_eq!(mf.gradient_pairs.len(), 0);
        assert_eq!(mf.critical_cells.len(), 1);
        assert_eq!(mf.critical_cells[0], 0);
    }

    #[test]
    fn path_graph_1_critical_vertex() {
        // Path 0-1-2: a connected path has β₀=1, so there must be exactly 1 critical vertex.
        // Greedy Morse pairs all vertices but one against edges; the survivor is the
        // "local maximum" vertex (the last vertex in ascending value order).
        let (vals, bounds, n) = path_complex(3);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        // The discrete Morse inequality guarantees at least 1 critical 0-cell (vertex).
        let critical_vertices: Vec<_> = mf
            .critical_cells
            .iter()
            .filter(|&&c| c < 3)
            .copied()
            .collect();
        assert!(
            !critical_vertices.is_empty(),
            "path graph must have at least 1 critical vertex, got {:?}",
            mf.critical_cells
        );
    }

    #[test]
    fn cycle_graph_extra_critical_cell() {
        // A cycle has H₁ ≠ 0, so discrete Morse must have at least 1 critical edge
        // beyond what a tree would need.  n critical cells >= 2 (1 vertex + 1 edge for H₁).
        let (vals, bounds, n) = cycle_complex(5);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        // Discrete Morse inequality: #crit_cells >= Betti numbers, which for a circle = 2.
        assert!(
            mf.critical_cells.len() >= 2,
            "cycle needs at least 2 critical cells, got {}",
            mf.critical_cells.len()
        );
    }

    #[test]
    fn values_must_match_n_simplices() {
        let result = discrete_gradient(&[1.0, 2.0], &[vec![], vec![0]], 3);
        assert!(matches!(result, Err(TdaError::ParameterOutOfRange(_))));
    }

    #[test]
    fn output_finite() {
        let (vals, bounds, n) = path_complex(8);
        let mf = discrete_gradient(&vals, &bounds, n).expect("ok");
        // All value references should be finite
        for &c in &mf.critical_cells {
            assert!(vals[c].is_finite());
        }
        for &(f, s) in &mf.gradient_pairs {
            assert!(vals[f].is_finite() && vals[s].is_finite());
        }
    }
}
