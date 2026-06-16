//! Landmark MDS (de Silva & Tenenbaum 2004).
//!
//! Classical MDS double-centres an `n × n` distance matrix and eigendecomposes
//! it, costing `O(n²)` memory and `O(n³)` time — prohibitive for large `n`.
//! **Landmark MDS** sidesteps this by embedding only a small set of `ℓ`
//! *landmark* points with classical MDS, then placing every remaining point by
//! *distance-based triangulation* against the landmarks.
//!
//! Algorithm:
//! 1. Choose `ℓ` landmarks (here: deterministic maxmin / farthest-point
//!    sampling from point `0`).
//! 2. Classical MDS on the `ℓ × ℓ` landmark distance matrix:
//!    `B = −½ H Δ_L H`, eigendecompose `B = V Λ Vᵀ`, landmark coordinates
//!    `L = V_+ √Λ_+`.
//! 3. Build the **pseudo-inverse embedding transform**
//!    `Lᵖ = Λ_+^{−1/2} V_+ᵀ` (shape `[k × ℓ]`).
//! 4. For every point `x`, with squared landmark distances `δ_x` and landmark
//!    mean `μ = (1/ℓ) Σ_j δ_{L_j}`, the embedding is
//!    `y = −½ Lᵖ (δ_x − μ)`.
//!    Landmarks themselves reproduce their step-2 coordinates exactly.
//!
//! This is `O(ℓ · n · dim)` distance work plus `O(ℓ³)` for the small
//! eigenproblem — linear in `n` for fixed `ℓ`.
//!
//! Reference: V. de Silva & J. B. Tenenbaum, "Sparse multidimensional scaling
//! using landmark points", Tech. report, Stanford, 2004.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Hyper-parameters for [`landmark_mds`].
#[derive(Debug, Clone)]
pub struct LandmarkMdsConfig {
    /// Target embedding dimensionality `k`.
    pub n_components: usize,
    /// Number of landmark points `ℓ`. Must satisfy `n_components < ℓ ≤ n`.
    pub n_landmarks: usize,
}

impl Default for LandmarkMdsConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_landmarks: 16,
        }
    }
}

/// Result of a Landmark-MDS fit.
#[derive(Debug, Clone)]
pub struct LandmarkMdsResult {
    /// Embedded coordinates, row-major `[n × n_components]`.
    pub embedding: Vec<f64>,
    /// Indices (into the original points) of the chosen landmarks.
    pub landmarks: Vec<usize>,
    /// Top-`n_components` eigenvalues of the landmark Gram matrix.
    pub eigenvalues: Vec<f64>,
}

const EPS: f64 = 1e-12;

/// Embed `n` points of dimensionality `dim` into `cfg.n_components` dimensions
/// with Landmark MDS.
///
/// `data` is a row-major `[n × dim]` slice.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] when `n == 0` or `dim == 0`.
/// - [`ManifoldError::ShapeMismatch`] when `data.len() != n * dim`.
/// - [`ManifoldError::InvalidParameter`] when `n_components == 0`,
///   `n_landmarks > n`, or `n_landmarks <= n_components`.
pub fn landmark_mds(
    data: &[f64],
    n: usize,
    dim: usize,
    cfg: &LandmarkMdsConfig,
) -> ManifoldResult<LandmarkMdsResult> {
    if n == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![data.len()],
        });
    }
    let k = cfg.n_components;
    let l = cfg.n_landmarks.min(n);
    if k == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: "must be >= 1".into(),
        });
    }
    if cfg.n_landmarks > n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_landmarks".into(),
            reason: format!("must be <= n={n}, got {}", cfg.n_landmarks),
        });
    }
    if l <= k {
        return Err(ManifoldError::InvalidParameter {
            name: "n_landmarks".into(),
            reason: format!("must exceed n_components={k}, got {l}"),
        });
    }

    let landmarks = maxmin_landmarks(data, n, dim, l);

    // Squared landmark-to-landmark distances Δ_L (ℓ × ℓ).
    let mut delta_l = vec![0.0; l * l];
    for a in 0..l {
        for b in (a + 1)..l {
            let d2 = sq_dist(data, dim, landmarks[a], landmarks[b]);
            delta_l[a * l + b] = d2;
            delta_l[b * l + a] = d2;
        }
    }

    // Double-centre: B = -1/2 H Δ_L H.
    let mut row_mean = vec![0.0; l];
    let mut total = 0.0;
    for a in 0..l {
        for b in 0..l {
            row_mean[a] += delta_l[a * l + b];
            total += delta_l[a * l + b];
        }
    }
    for v in &mut row_mean {
        *v /= l as f64;
    }
    total /= (l * l) as f64;
    let mut b_mat = vec![0.0; l * l];
    for a in 0..l {
        for b in 0..l {
            b_mat[a * l + b] = -0.5 * (delta_l[a * l + b] - row_mean[a] - row_mean[b] + total);
        }
    }

    let (mut w, mut v) = jacobi_eigh(&b_mat, l)?;
    sort_eigen_descending(&mut w, &mut v, l);

    // Landmark embedding L = V_+ √Λ_+ and pseudo-inverse transform Lᵖ.
    let mut eigenvalues = vec![0.0; k];
    let mut sqrt_lambda = vec![0.0; k];
    for c in 0..k {
        let lam = w[c].max(0.0);
        eigenvalues[c] = lam;
        sqrt_lambda[c] = lam.sqrt();
    }

    let mut embedding = vec![0.0; n * k];

    // Place every point via distance-based triangulation against landmarks.
    // y_c = -1/2 * (1/√λ_c) * Σ_a V[a,c] * (δ_x[a] - row_mean[a]).
    for i in 0..n {
        // Squared distances from point i to each landmark.
        let mut delta_x = vec![0.0; l];
        for (a, &lm) in landmarks.iter().enumerate() {
            delta_x[a] = sq_dist(data, dim, i, lm);
        }
        for c in 0..k {
            let s = sqrt_lambda[c];
            if s <= EPS {
                embedding[i * k + c] = 0.0;
                continue;
            }
            let mut acc = 0.0;
            for a in 0..l {
                acc += v[a * l + c] * (delta_x[a] - row_mean[a]);
            }
            embedding[i * k + c] = -0.5 * acc / s;
        }
    }

    Ok(LandmarkMdsResult {
        embedding,
        landmarks,
        eigenvalues,
    })
}

/// Squared Euclidean distance between rows `i` and `j` of `data`.
fn sq_dist(data: &[f64], dim: usize, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for k in 0..dim {
        let diff = data[i * dim + k] - data[j * dim + k];
        s += diff * diff;
    }
    s
}

/// Deterministic farthest-point (maxmin) landmark sampling, seeded at point 0.
fn maxmin_landmarks(data: &[f64], n: usize, dim: usize, l: usize) -> Vec<usize> {
    let mut landmarks = Vec::with_capacity(l);
    landmarks.push(0);
    // Nearest-landmark distance for every point.
    let mut min_dist = vec![f64::INFINITY; n];
    for (i, slot) in min_dist.iter_mut().enumerate() {
        *slot = sq_dist(data, dim, i, 0);
    }
    while landmarks.len() < l {
        // Pick the point farthest from the current landmark set.
        let mut best = 0;
        let mut best_d = -1.0;
        for (i, &d) in min_dist.iter().enumerate() {
            if d > best_d {
                best_d = d;
                best = i;
            }
        }
        landmarks.push(best);
        for (i, slot) in min_dist.iter_mut().enumerate() {
            let d = sq_dist(data, dim, i, best);
            if d < *slot {
                *slot = d;
            }
        }
    }
    landmarks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg() -> LandmarkMdsConfig {
        LandmarkMdsConfig::default()
    }

    /// `n` points sampled on a line in 3-D (true intrinsic dim 1).
    fn line_data(n: usize) -> Vec<f64> {
        let mut data = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = i as f64;
            data.extend_from_slice(&[t, 2.0 * t, -t]);
        }
        data
    }

    /// Three clusters in 4-D.
    fn three_clusters() -> (Vec<f64>, usize, usize) {
        let centres = [
            [0.0, 0.0, 0.0, 0.0],
            [12.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 0.0, 0.0],
        ];
        let mut rng = LcgRng::new(11);
        let mut data = Vec::new();
        for c in &centres {
            for _ in 0..10 {
                for &val in c.iter() {
                    data.push(val + (rng.next_f64() - 0.5) * 0.5);
                }
            }
        }
        (data, 30, 4)
    }

    // 1. Output shape is [n × n_components].
    #[test]
    fn output_shape() {
        let (data, n, dim) = three_clusters();
        let r = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        assert_eq!(r.embedding.len(), n * cfg().n_components);
    }

    // 2. All embedded coordinates are finite.
    #[test]
    fn finite() {
        let (data, n, dim) = three_clusters();
        let r = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        for &v in &r.embedding {
            assert!(v.is_finite(), "non-finite value {v}");
        }
    }

    // 3. Empty input errors.
    #[test]
    fn empty_input_error() {
        let err = landmark_mds(&[], 0, 0, &cfg());
        assert!(matches!(err, Err(ManifoldError::EmptyInput)), "got {err:?}");
    }

    // 4. Shape mismatch errors.
    #[test]
    fn shape_mismatch_error() {
        let err = landmark_mds(&[1.0, 2.0, 3.0], 2, 3, &cfg());
        assert!(
            matches!(err, Err(ManifoldError::ShapeMismatch { .. })),
            "got {err:?}"
        );
    }

    // 5. n_components == 0 errors.
    #[test]
    fn n_components_0_error() {
        let (data, n, dim) = three_clusters();
        let c = LandmarkMdsConfig {
            n_components: 0,
            ..cfg()
        };
        let err = landmark_mds(&data, n, dim, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 6. n_landmarks > n errors.
    #[test]
    fn n_landmarks_too_large_error() {
        let data = line_data(5);
        let c = LandmarkMdsConfig {
            n_components: 1,
            n_landmarks: 9,
        };
        let err = landmark_mds(&data, 5, 3, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 7. n_landmarks <= n_components errors.
    #[test]
    fn n_landmarks_le_components_error() {
        let data = line_data(10);
        let c = LandmarkMdsConfig {
            n_components: 3,
            n_landmarks: 3,
        };
        let err = landmark_mds(&data, 10, 3, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 8. Deterministic for the same input.
    #[test]
    fn deterministic() {
        let (data, n, dim) = three_clusters();
        let a = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        let b = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        assert_eq!(a.embedding, b.embedding);
        assert_eq!(a.landmarks, b.landmarks);
    }

    // 9. Landmarks are distinct indices within range.
    #[test]
    fn landmarks_distinct_in_range() {
        let (data, n, dim) = three_clusters();
        let r = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        assert_eq!(r.landmarks.len(), cfg().n_landmarks);
        let mut seen = std::collections::HashSet::new();
        for &lm in &r.landmarks {
            assert!(lm < n, "landmark {lm} out of range");
            assert!(seen.insert(lm), "duplicate landmark {lm}");
        }
    }

    // 10. Collinear points recover a 1-D ordering (embedding is monotone in the
    //     point index up to sign).
    #[test]
    fn line_recovers_ordering() {
        let n = 20;
        let data = line_data(n);
        let c = LandmarkMdsConfig {
            n_components: 1,
            n_landmarks: 6,
        };
        let r = landmark_mds(&data, n, 3, &c).expect("landmark_mds should succeed");
        let coords: Vec<f64> = (0..n).map(|i| r.embedding[i]).collect();
        // Orient so the sequence increases.
        let sign = if coords[n - 1] >= coords[0] {
            1.0
        } else {
            -1.0
        };
        let mut monotone = true;
        for i in 1..n {
            if sign * coords[i] < sign * coords[i - 1] - 1e-6 {
                monotone = false;
                break;
            }
        }
        assert!(monotone, "1-D embedding not monotone: {coords:?}");
    }

    // 11. Separated clusters stay separated in the embedding.
    #[test]
    fn clusters_separated() {
        let (data, n, dim) = three_clusters();
        let r = landmark_mds(&data, n, dim, &cfg()).expect("value should be present");
        let m = cfg().n_components;
        let label = |i: usize| i / 10;
        let dist = |i: usize, j: usize| -> f64 {
            let mut s = 0.0;
            for k in 0..m {
                let diff = r.embedding[i * m + k] - r.embedding[j * m + k];
                s += diff * diff;
            }
            s.sqrt()
        };
        let mut intra = (0.0, 0usize);
        let mut inter = (0.0, 0usize);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist(i, j);
                if label(i) == label(j) {
                    intra.0 += d;
                    intra.1 += 1;
                } else {
                    inter.0 += d;
                    inter.1 += 1;
                }
            }
        }
        let intra_mean = intra.0 / intra.1 as f64;
        let inter_mean = inter.0 / inter.1 as f64;
        assert!(
            inter_mean > intra_mean,
            "inter {inter_mean} <= intra {intra_mean}"
        );
    }
}
