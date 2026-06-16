//! Sammon mapping (Sammon 1969).
//!
//! Sammon's nonlinear mapping projects high-dimensional data into a
//! low-dimensional space while preserving the *pairwise distance structure*,
//! with a deliberate emphasis on small distances. It minimises the **Sammon
//! stress**
//! ```text
//!            1                    (d*_{ij} - d_{ij})^2
//! E  =  ───────────────  ·  Σ    ──────────────────────
//!        Σ_{i<j} d*_{ij}        i<j        d*_{ij}
//! ```
//! where `d*_{ij}` is the distance between points `i` and `j` in the original
//! space and `d_{ij}` is the distance between their images in the embedding.
//! Dividing each squared residual by `d*_{ij}` weights *short* original
//! distances more heavily than long ones, so local neighbourhoods are preserved
//! more faithfully than under plain metric MDS.
//!
//! The stress is minimised by a diagonal (pseudo-Newton) gradient step, exactly
//! as in Sammon's original paper:
//! ```text
//! y_{p,k}  ←  y_{p,k}  -  α · (∂E/∂y_{p,k}) / |∂²E/∂y_{p,k}²|
//! ```
//! with a "magic factor" learning rate `α` (Sammon recommended `α ≈ 0.3`). The
//! configuration is initialised either from a deterministic principal-axis
//! projection of the data (the leading coordinates) or, when that is degenerate,
//! from a reproducible [`LcgRng`] draw.
//!
//! Reference: J. W. Sammon, "A nonlinear mapping for data structure analysis",
//! IEEE Transactions on Computers, 1969.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

/// Hyper-parameters for [`sammon`].
#[derive(Debug, Clone)]
pub struct SammonConfig {
    /// Target embedding dimensionality.
    pub n_components: usize,
    /// Maximum number of gradient iterations.
    pub max_iter: usize,
    /// Learning rate ("magic factor") for the diagonal Newton step.
    pub learning_rate: f64,
    /// Stop early when the relative stress improvement drops below this.
    pub tol: f64,
    /// Seed for the random fallback initialisation.
    pub seed: u64,
}

impl Default for SammonConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 300,
            learning_rate: 0.3,
            tol: 1e-9,
            seed: 0x5A_3D_17_2B,
        }
    }
}

/// Result of a Sammon mapping.
#[derive(Debug, Clone)]
pub struct SammonResult {
    /// Embedded coordinates, row-major `[n × n_components]`.
    pub embedding: Vec<f64>,
    /// Final Sammon stress `E`.
    pub stress: f64,
    /// Number of iterations actually performed.
    pub n_iter: usize,
}

/// Tiny floor that keeps distance divisions finite.
const EPS: f64 = 1e-9;

/// Compute the Sammon mapping of `n` points living in `dim` dimensions.
///
/// `data` is a row-major `[n × dim]` slice. The result embeds the points into
/// `cfg.n_components` dimensions.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] when `n == 0` or `dim == 0`.
/// - [`ManifoldError::ShapeMismatch`] when `data.len() != n * dim`.
/// - [`ManifoldError::InvalidParameter`] when `n_components` is `0` or not less
///   than `n`.
pub fn sammon(
    data: &[f64],
    n: usize,
    dim: usize,
    cfg: &SammonConfig,
) -> ManifoldResult<SammonResult> {
    if n == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![data.len()],
        });
    }
    let m = cfg.n_components;
    if m == 0 || m >= n {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n}, got {m}"),
        });
    }

    // Original pairwise distances (dense, symmetric, zero diagonal).
    let dstar = original_distances(data, n, dim);
    let mut denom = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            denom += dstar[i * n + j];
        }
    }
    if denom <= EPS {
        // All points coincide: any embedding has zero stress; return zeros.
        return Ok(SammonResult {
            embedding: vec![0.0; n * m],
            stress: 0.0,
            n_iter: 0,
        });
    }

    let mut y = initial_embedding(data, n, dim, m, cfg.seed);
    let mut stress = sammon_stress(&y, &dstar, n, m, denom);

    let mut n_iter = 0;
    for _ in 0..cfg.max_iter {
        n_iter += 1;
        gradient_step(&mut y, &dstar, n, m, cfg.learning_rate);
        let new_stress = sammon_stress(&y, &dstar, n, m, denom);
        let improvement = (stress - new_stress).abs() / stress.abs().max(EPS);
        stress = new_stress;
        if improvement < cfg.tol {
            break;
        }
    }

    Ok(SammonResult {
        embedding: y,
        stress,
        n_iter,
    })
}

/// Dense Euclidean distance matrix of the input points.
fn original_distances(data: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..dim {
                let diff = data[i * dim + k] - data[j * dim + k];
                s += diff * diff;
            }
            let dij = s.sqrt();
            d[i * n + j] = dij;
            d[j * n + i] = dij;
        }
    }
    d
}

/// Deterministic initialisation: take the first `m` mean-centred input
/// coordinates; if the data has fewer than `m` columns, pad with small
/// reproducible random jitter so the coordinates are not collinear.
fn initial_embedding(data: &[f64], n: usize, dim: usize, m: usize, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    let mut y = vec![0.0; n * m];
    // Column means for the available input dimensions.
    let mut mean = vec![0.0; dim.min(m)];
    for i in 0..n {
        for (k, mk) in mean.iter_mut().enumerate() {
            *mk += data[i * dim + k];
        }
    }
    for mk in &mut mean {
        *mk /= n as f64;
    }
    for i in 0..n {
        for k in 0..m {
            if k < dim {
                y[i * m + k] = data[i * dim + k] - mean[k];
            } else {
                y[i * m + k] = (rng.next_f64() - 0.5) * 1e-2;
            }
        }
    }
    // If the chosen coordinates are degenerate (all equal), perturb them.
    let mut spread = 0.0;
    for k in 0..m {
        let c0 = y[k];
        for i in 1..n {
            spread += (y[i * m + k] - c0).abs();
        }
    }
    if spread < EPS {
        for v in &mut y {
            *v = (rng.next_f64() - 0.5) * 1.0;
        }
    }
    y
}

/// Sammon stress of the current embedding.
fn sammon_stress(y: &[f64], dstar: &[f64], n: usize, m: usize, denom: f64) -> f64 {
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let ds = dstar[i * n + j];
            if ds <= EPS {
                continue;
            }
            let d = embed_dist(y, i, j, m);
            let diff = ds - d;
            e += diff * diff / ds;
        }
    }
    e / denom
}

/// Euclidean distance between embedded points `i` and `j`.
fn embed_dist(y: &[f64], i: usize, j: usize, m: usize) -> f64 {
    let mut s = 0.0;
    for k in 0..m {
        let diff = y[i * m + k] - y[j * m + k];
        s += diff * diff;
    }
    s.sqrt()
}

/// One diagonal pseudo-Newton update over every embedded coordinate.
fn gradient_step(y: &mut [f64], dstar: &[f64], n: usize, m: usize, alpha: f64) {
    let mut step = vec![0.0; n * m];
    for p in 0..n {
        for k in 0..m {
            let mut grad1 = 0.0; // first derivative ∂E/∂y_{p,k}
            let mut grad2 = 0.0; // second derivative ∂²E/∂y_{p,k}²
            for j in 0..n {
                if j == p {
                    continue;
                }
                let ds = dstar[p * n + j];
                if ds <= EPS {
                    continue;
                }
                let d = embed_dist(y, p, j, m).max(EPS);
                let delta = y[p * m + k] - y[j * m + k];
                let inv = 1.0 / (ds * d);
                // Sammon (1969) eqs. (12)–(13), dropping the common 2/Σd* factor
                // (absorbed into the learning rate).
                grad1 += inv * (ds - d) * delta;
                grad2 += inv * ((ds - d) - (delta * delta / d) * (1.0 + (ds - d) / d));
            }
            let denom = grad2.abs().max(EPS);
            step[p * m + k] = alpha * grad1 / denom;
        }
    }
    for idx in 0..n * m {
        // Gradient points toward increasing stress for the residual sign used
        // above; adding the step descends E.
        y[idx] += step[idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SammonConfig {
        SammonConfig::default()
    }

    /// Three well-separated clusters in 3-D.
    fn three_clusters() -> (Vec<f64>, usize, usize) {
        let mut data = Vec::new();
        let centres = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]];
        let mut rng = LcgRng::new(7);
        for c in &centres {
            for _ in 0..6 {
                for &val in c.iter() {
                    data.push(val + (rng.next_f64() - 0.5) * 0.5);
                }
            }
        }
        (data, 18, 3)
    }

    // 1. Output shape is [n × n_components].
    #[test]
    fn output_shape() {
        let (data, n, dim) = three_clusters();
        let r = sammon(&data, n, dim, &cfg()).expect("value should be present");
        assert_eq!(r.embedding.len(), n * cfg().n_components);
    }

    // 2. All embedded coordinates are finite.
    #[test]
    fn finite() {
        let (data, n, dim) = three_clusters();
        let r = sammon(&data, n, dim, &cfg()).expect("value should be present");
        for &v in &r.embedding {
            assert!(v.is_finite(), "non-finite coordinate {v}");
        }
    }

    // 3. Empty input errors.
    #[test]
    fn empty_input_error() {
        let err = sammon(&[], 0, 0, &cfg());
        assert!(matches!(err, Err(ManifoldError::EmptyInput)), "got {err:?}");
    }

    // 4. Shape mismatch errors.
    #[test]
    fn shape_mismatch_error() {
        let err = sammon(&[1.0, 2.0, 3.0], 2, 3, &cfg());
        assert!(
            matches!(err, Err(ManifoldError::ShapeMismatch { .. })),
            "got {err:?}"
        );
    }

    // 5. n_components == 0 errors.
    #[test]
    fn n_components_0_error() {
        let (data, n, dim) = three_clusters();
        let c = SammonConfig {
            n_components: 0,
            ..cfg()
        };
        let err = sammon(&data, n, dim, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 6. n_components >= n errors.
    #[test]
    fn n_components_ge_n_error() {
        let data = vec![0.0, 1.0, 2.0, 3.0]; // n=2, dim=2
        let c = SammonConfig {
            n_components: 2,
            ..cfg()
        };
        let err = sammon(&data, 2, 2, &c);
        assert!(
            matches!(err, Err(ManifoldError::InvalidParameter { .. })),
            "got {err:?}"
        );
    }

    // 7. Final stress is non-negative.
    #[test]
    fn stress_nonneg() {
        let (data, n, dim) = three_clusters();
        let r = sammon(&data, n, dim, &cfg()).expect("value should be present");
        assert!(r.stress >= 0.0, "negative stress {}", r.stress);
    }

    // 8. Gradient descent does not increase stress relative to the initial map.
    #[test]
    fn stress_decreases() {
        let (data, n, dim) = three_clusters();
        let init_only = SammonConfig {
            max_iter: 0,
            ..cfg()
        };
        let r0 = sammon(&data, n, dim, &init_only).expect("sammon should succeed");
        let r = sammon(&data, n, dim, &cfg()).expect("value should be present");
        assert!(
            r.stress <= r0.stress + 1e-9,
            "final stress {} exceeded initial {}",
            r.stress,
            r0.stress
        );
    }

    // 9. Deterministic for a fixed seed.
    #[test]
    fn deterministic() {
        let (data, n, dim) = three_clusters();
        let a = sammon(&data, n, dim, &cfg()).expect("value should be present");
        let b = sammon(&data, n, dim, &cfg()).expect("value should be present");
        assert_eq!(a.embedding, b.embedding);
    }

    // 10. Separated clusters stay separated: the mean inter-cluster embedded
    //     distance exceeds the mean intra-cluster distance.
    #[test]
    fn clusters_separated() {
        let (data, n, dim) = three_clusters();
        let r = sammon(&data, n, dim, &cfg()).expect("value should be present");
        let m = cfg().n_components;
        let label = |i: usize| i / 6;
        let mut intra = (0.0, 0usize);
        let mut inter = (0.0, 0usize);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = embed_dist(&r.embedding, i, j, m);
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
            "inter {inter_mean} should exceed intra {intra_mean}"
        );
    }

    // 11. Coincident points yield a zero-stress all-zero embedding.
    #[test]
    fn coincident_points_zero() {
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 3 identical 2-D points
        let r = sammon(&data, 3, 2, &cfg()).expect("value should be present");
        assert!(r.stress.abs() < 1e-12, "stress {}", r.stress);
    }
}
