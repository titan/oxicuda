//! Lazy Witness Complex construction.
//!
//! The lazy witness complex is a simplicial complex built from a set of landmark points
//! and a (larger) set of witness points.  The main advantages over Vietoris-Rips are:
//!  - much smaller complex (O(L^d) instead of O(N^d) simplices)
//!  - naturally adapted to the geometry of the data

use crate::complex::filtration::{FilteredSimplex, Filtration};
use crate::complex::simplex::Simplex;
use crate::error::{TdaError, TdaResult};
use crate::handle::LcgRng;

/// Configuration for the witness complex.
#[derive(Debug, Clone)]
pub struct WitnessConfig {
    /// Number of landmark points to select.
    pub n_landmarks: usize,
    /// Maximum radius for inclusion of a simplex.
    pub max_radius: f64,
    /// Maximum simplex dimension to build.
    pub max_dim: usize,
}

/// Greedy maxmin (farthest-point) landmark selection.
///
/// Starting from a random point, iteratively picks the point farthest from the current
/// landmark set until `n_landmarks` landmarks have been selected.
///
/// Returns the indices (into the `n_pts`-point set) of the selected landmarks.
pub fn maxmin_landmarks(
    dist: &[f64],
    n_pts: usize,
    n_landmarks: usize,
    rng: &mut LcgRng,
) -> TdaResult<Vec<usize>> {
    if n_pts == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if n_landmarks == 0 {
        return Err(TdaError::LandmarkSelectionFailed(
            "n_landmarks must be > 0".to_owned(),
        ));
    }
    if n_landmarks > n_pts {
        return Err(TdaError::LandmarkSelectionFailed(format!(
            "n_landmarks ({n_landmarks}) > n_pts ({n_pts})"
        )));
    }
    if dist.len() != n_pts * n_pts {
        return Err(TdaError::DimensionMismatch {
            expected: n_pts * n_pts,
            got: dist.len(),
        });
    }

    let mut landmarks: Vec<usize> = Vec::with_capacity(n_landmarks);
    // Start from a random point
    let first = rng.next_usize(n_pts);
    landmarks.push(first);

    // min_dist[i] = min distance from point i to any currently selected landmark
    let mut min_dist: Vec<f64> = (0..n_pts).map(|i| dist[first * n_pts + i]).collect();

    for _ in 1..n_landmarks {
        // Pick the point with maximum min-distance to landmarks
        let next = min_dist
            .iter()
            .enumerate()
            .filter(|&(i, _)| !landmarks.contains(&i))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                TdaError::LandmarkSelectionFailed("no more points to select".to_owned())
            })?;
        landmarks.push(next);
        // Update min_dist
        for i in 0..n_pts {
            let d = dist[next * n_pts + i];
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    Ok(landmarks)
}

/// Build the lazy witness complex.
///
/// A k-simplex `[l₀, …, l_k]` (over the landmark indices) is included in `LW(R)` if there
/// exists a witness point `w` such that:
///
///   max_{i=0..k} D[l_i, w] ≤ R + m_w
///
/// where `D[l, w] = dist(landmark_l, witness_w)` and `m_w = min_l D[l, w]` (distance from
/// `w` to its nearest landmark, i.e. the "0-th neighbourhood" contribution).
///
/// The filtration value of a simplex is the smallest `R` for which such a witness exists.
///
/// `dist_landmark_to_witness` is a flat array of size `n_landmarks × n_witnesses`,
/// row-major (row = landmark, column = witness).
pub fn lazy_witness_complex(
    dist_landmark_to_witness: &[f64],
    n_landmarks: usize,
    n_witnesses: usize,
    max_radius: f64,
    max_dim: usize,
) -> TdaResult<Filtration> {
    if n_landmarks == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if n_witnesses == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if dist_landmark_to_witness.len() != n_landmarks * n_witnesses {
        return Err(TdaError::DimensionMismatch {
            expected: n_landmarks * n_witnesses,
            got: dist_landmark_to_witness.len(),
        });
    }
    if max_dim > 6 {
        return Err(TdaError::DimensionTooLarge(max_dim));
    }

    // Precompute m_w = min_l D[l, w] for each witness w
    let mut m_w: Vec<f64> = vec![f64::INFINITY; n_witnesses];
    for w in 0..n_witnesses {
        for l in 0..n_landmarks {
            let d = dist_landmark_to_witness[l * n_witnesses + w];
            if d < m_w[w] {
                m_w[w] = d;
            }
        }
    }

    let mut simplices: Vec<FilteredSimplex> = Vec::new();

    // 0-simplices: one per landmark
    for l in 0..n_landmarks {
        simplices.push(FilteredSimplex {
            simplex: Simplex { vertices: vec![l] },
            value: 0.0,
        });
    }

    // Higher-dimensional simplices
    for d in 1..=max_dim {
        let size = d + 1;
        if size > n_landmarks {
            break;
        }
        let mut indices: Vec<usize> = (0..size).collect();
        loop {
            // For this (d+1)-subset of landmarks, find the minimum R such that
            // some witness w satisfies max_i D[indices[i], w] ≤ R + m_w
            let mut best_r = f64::INFINITY;
            for w in 0..n_witnesses {
                let max_d = indices
                    .iter()
                    .map(|&l| dist_landmark_to_witness[l * n_witnesses + w])
                    .fold(f64::NEG_INFINITY, f64::max);
                let r = max_d - m_w[w];
                if r < best_r {
                    best_r = r;
                }
            }
            if best_r <= max_radius {
                simplices.push(FilteredSimplex {
                    simplex: Simplex {
                        vertices: indices.clone(),
                    },
                    value: best_r.max(0.0),
                });
            }
            if !next_combination(&mut indices, n_landmarks) {
                break;
            }
        }
    }

    Filtration::new(simplices)
}

/// Advance `indices` to the next k-combination of {0, …, n-1} in lex order.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if indices[i] < n - (k - i) {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn maxmin_selects_n_landmarks() {
        // 5-point grid; select 3
        let pts: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let n = 5;
        let mut dist = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                dist[i * n + j] = (pts[i] - pts[j]).abs();
            }
        }
        let mut rng = LcgRng::new(42);
        let landmarks = maxmin_landmarks(&dist, n, 3, &mut rng).expect("ok");
        assert_eq!(landmarks.len(), 3);
        // No duplicate landmarks
        let mut l = landmarks.clone();
        l.dedup();
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn lazy_witness_has_vertices() {
        // 3 landmarks, 5 witnesses
        let n_l = 3;
        let n_w = 5;
        let d: Vec<f64> = (0..n_l * n_w).map(|x| (x as f64) * 0.1).collect();
        let filt = lazy_witness_complex(&d, n_l, n_w, 10.0, 1).expect("ok");
        let verts: Vec<_> = filt
            .simplices
            .iter()
            .filter(|fs| fs.simplex.dim() == 0)
            .collect();
        assert_eq!(verts.len(), n_l);
    }
}
