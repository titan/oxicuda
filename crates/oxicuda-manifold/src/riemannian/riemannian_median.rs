//! Riemannian Median on SPD(d) via Weiszfeld/IRLS on the manifold.
//!
//! Reference: Fletcher et al. 2009 CVPR
//! "The Geometric Median on Riemannian Manifolds with Application to Robust Atlas Estimation".
//!
//! The geometric (Fréchet) median minimises the sum of geodesic distances
//! `Σᵢ d(m, Xᵢ)` rather than squared distances (as the Fréchet mean does).
//! The Weiszfeld/IRLS iteration on the manifold uses a weighted log-average
//! where the weights are inversely proportional to the current distances.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::riemannian::spd::{spd_distance, spd_exp, spd_log};
use crate::riemannian::spd_kmeans::{FrechetMeanConfig, spd_frechet_mean};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Riemannian median algorithm on SPD(d).
#[derive(Debug, Clone)]
pub struct RiemannianMedianConfig {
    /// Dimension `d` of the SPD matrices (each matrix is `d×d`).
    pub matrix_dim: usize,
    /// Maximum number of Weiszfeld iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when `||V||_F < tol`.
    pub tol: f64,
    /// Regularisation added to distances to avoid division by zero: `wᵢ = 1/max(dᵢ, eps)`.
    pub eps: f64,
}

impl Default for RiemannianMedianConfig {
    fn default() -> Self {
        Self {
            matrix_dim: 2,
            max_iter: 500,
            tol: 1e-7,
            eps: 1e-10,
        }
    }
}

/// Result returned by [`riemannian_median`].
#[derive(Debug, Clone)]
pub struct RiemannianMedianResult {
    /// Riemannian median matrix, stored row-major as `[d × d]`.
    pub median: Vec<f64>,
    /// Dimension of each SPD matrix.
    pub matrix_dim: usize,
    /// Whether the algorithm converged within `max_iter` iterations.
    pub converged: bool,
    /// Number of Weiszfeld iterations performed.
    pub iterations: usize,
    /// Frobenius norm of the final tangent vector (proxy for the gradient step size).
    pub final_step: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a d×d matrix is SPD by checking all eigenvalues are strictly positive.
fn check_spd(m: &[f64], d: usize) -> ManifoldResult<()> {
    match jacobi_eigh(m, d) {
        Ok((w, _)) => {
            if w.iter().all(|&ev| ev > 1e-10) {
                Ok(())
            } else {
                Err(ManifoldError::ManifoldConstraint(
                    "matrix is not positive definite (eigenvalue ≤ 1e-10)".into(),
                ))
            }
        }
        Err(e) => Err(ManifoldError::ManifoldConstraint(format!(
            "eigendecomposition failed during SPD check: {e}"
        ))),
    }
}

#[inline]
fn frobenius_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Riemannian (geometric) median of `n` SPD matrices of size `d×d`.
///
/// Uses the Weiszfeld/IRLS iteration on the SPD manifold:
/// 1. Initialise `m` as the Fréchet mean of `X`.
/// 2. Compute distances `dᵢ = d(m, Xᵢ)`, weights `wᵢ = 1/max(dᵢ, eps)`.
/// 3. Normalise: `w̃ᵢ = wᵢ / Σwⱼ`.
/// 4. Tangent vector: `V = Σᵢ w̃ᵢ log_m(Xᵢ)`.
/// 5. Retract: `m ← exp_m(V)`.
/// 6. Repeat until `||V||_F < tol`.
///
/// # Arguments
/// * `x`      — flat `[n × d × d]` row-major SPD matrices.
/// * `n`      — number of matrices.
/// * `config` — algorithm parameters (uses `config.matrix_dim` as `d`).
pub fn riemannian_median(
    x: &[f64],
    n: usize,
    config: &RiemannianMedianConfig,
) -> ManifoldResult<RiemannianMedianResult> {
    let d = config.matrix_dim;
    let dd = d * d;

    // ── Input validation ───────────────────────────────────────────────────
    if n == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "number of matrices must be >= 1".into(),
        });
    }
    if x.len() != n * dd {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d, d],
            got: vec![x.len()],
        });
    }
    // Validate all inputs are SPD.
    for i in 0..n {
        let mi = &x[i * dd..(i + 1) * dd];
        check_spd(mi, d)?;
    }

    // ── Single-matrix shortcut ─────────────────────────────────────────────
    if n == 1 {
        return Ok(RiemannianMedianResult {
            median: x[..dd].to_vec(),
            matrix_dim: d,
            converged: true,
            iterations: 0,
            final_step: 0.0,
        });
    }

    // ── Initialise as Fréchet mean ─────────────────────────────────────────
    let frechet_config = FrechetMeanConfig {
        max_iter: 200,
        tol: 1e-8,
        step_size: 1.0,
    };
    let fm = spd_frechet_mean(x, n, d, &frechet_config)?;
    let mut m = fm.mean;

    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_step = 0.0_f64;

    // ── Weiszfeld / IRLS iterations ────────────────────────────────────────
    for _iter in 0..config.max_iter {
        iterations += 1;

        // Compute un-normalised weights wᵢ = 1/max(dᵢ, eps).
        let mut weights = vec![0.0f64; n];
        let mut w_sum = 0.0f64;
        for i in 0..n {
            let xi = &x[i * dd..(i + 1) * dd];
            let di = spd_distance(&m, xi, d).unwrap_or(0.0);
            let wi = 1.0 / di.max(config.eps);
            weights[i] = wi;
            w_sum += wi;
        }

        if w_sum <= 0.0 {
            break;
        }
        let inv_w_sum = 1.0 / w_sum;
        for w in weights.iter_mut() {
            *w *= inv_w_sum;
        }

        // Weighted sum of log maps: V = Σᵢ w̃ᵢ log_m(Xᵢ).
        let mut v_tan = vec![0.0f64; dd];
        for i in 0..n {
            let xi = &x[i * dd..(i + 1) * dd];
            if let Ok(log_i) = spd_log(&m, xi, d) {
                let wi = weights[i];
                for (acc, val) in v_tan.iter_mut().zip(log_i.iter()) {
                    *acc += wi * val;
                }
            }
        }

        let step = frobenius_norm(&v_tan);
        final_step = step;

        if step < config.tol {
            converged = true;
            break;
        }

        // Retract back to the manifold: m ← exp_m(V).
        if let Ok(m_new) = spd_exp(&m, &v_tan, d) {
            m = m_new;
        } else {
            break;
        }
    }

    Ok(RiemannianMedianResult {
        median: m,
        matrix_dim: d,
        converged,
        iterations,
        final_step,
    })
}

/// Compute the Riemannian median objective: `Σᵢ d(median, Xᵢ)`.
///
/// Returns the sum of affine-invariant SPD distances from `median` to each input matrix.
/// A lower value indicates a better approximation of the geometric median.
pub fn riemannian_median_objective(
    median: &[f64],
    x: &[f64],
    n: usize,
    d: usize,
) -> ManifoldResult<f64> {
    let dd = d * d;
    if median.len() != dd {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![d, d],
            got: vec![median.len()],
        });
    }
    if x.len() != n * dd {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d, d],
            got: vec![x.len()],
        });
    }
    let mut total = 0.0f64;
    for i in 0..n {
        let xi = &x[i * dd..(i + 1) * dd];
        let di = spd_distance(median, xi, d)?;
        total += di;
    }
    Ok(total)
}

/// Compute a trimmed Fréchet mean on SPD(d): remove the `alpha` fraction of
/// points farthest from the initial Fréchet mean, then recompute.
///
/// # Arguments
/// * `x`     — flat `[n × d × d]` row-major SPD matrices.
/// * `n`     — number of matrices.
/// * `d`     — matrix dimension.
/// * `alpha` — fraction in `[0, 0.5)` of points to trim from the farthest end.
pub fn riemannian_trimmed_mean(
    x: &[f64],
    n: usize,
    d: usize,
    alpha: f64,
) -> ManifoldResult<Vec<f64>> {
    if alpha < 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha".into(),
            reason: "trimming fraction must be >= 0".into(),
        });
    }
    if alpha >= 0.5 {
        return Err(ManifoldError::InvalidParameter {
            name: "alpha".into(),
            reason: "trimming fraction must be < 0.5 to keep majority".into(),
        });
    }
    if n == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "number of matrices must be >= 1".into(),
        });
    }
    let dd = d * d;
    if x.len() != n * dd {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d, d],
            got: vec![x.len()],
        });
    }

    // Compute Fréchet mean as reference point.
    let frechet_config = FrechetMeanConfig {
        max_iter: 200,
        tol: 1e-8,
        step_size: 1.0,
    };
    let fm = spd_frechet_mean(x, n, d, &frechet_config)?;
    let reference = fm.mean;

    // Distance from each point to the reference Fréchet mean.
    let mut indexed_dists: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let xi = &x[i * dd..(i + 1) * dd];
            let di = spd_distance(&reference, xi, d).unwrap_or(f64::INFINITY);
            (i, di)
        })
        .collect();

    // Sort ascending by distance (closest first).
    indexed_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Trim the farthest floor(alpha * n) points.
    let n_trim = (alpha * n as f64).floor() as usize;
    let n_keep = n.saturating_sub(n_trim).max(1);

    let trimmed_data: Vec<f64> = indexed_dists[..n_keep]
        .iter()
        .flat_map(|&(i, _)| x[i * dd..(i + 1) * dd].iter().copied())
        .collect();

    let result = spd_frechet_mean(&trimmed_data, n_keep, d, &frechet_config)?;
    Ok(result.mean)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::jacobi_eig::jacobi_eigh;
    use crate::riemannian::spd_kmeans::{FrechetMeanConfig, spd_frechet_mean};

    fn diag2(a: f64, b: f64) -> Vec<f64> {
        vec![a, 0.0, 0.0, b]
    }

    fn diag3(a: f64, b: f64, c: f64) -> Vec<f64> {
        vec![a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]
    }

    fn is_spd_mat(m: &[f64], n: usize) -> bool {
        match jacobi_eigh(m, n) {
            Ok((w, _)) => w.iter().all(|&ev| ev > 1e-10),
            Err(_) => false,
        }
    }

    fn frob_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    // Test 1: single matrix → median = that matrix
    #[test]
    fn single_matrix_median_equals_input() {
        let p = diag2(2.0, 3.0);
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&p, 1, &config);
        assert!(res.is_ok(), "should succeed: {:?}", res.err());
        let res = res.expect("res should be present");
        let diff = frob_diff(&res.median, &p);
        assert!(diff < 1e-5, "median differs from single input by {diff}");
    }

    // Test 2: two identical matrices → median = that matrix
    #[test]
    fn two_identical_matrices_median() {
        let p = diag2(4.0, 5.0);
        let x: Vec<f64> = p.iter().chain(p.iter()).copied().collect();
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&x, 2, &config);
        assert!(res.is_ok());
        let res = res.expect("res should be present");
        let diff = frob_diff(&res.median, &p);
        assert!(diff < 1e-5, "median differs from repeated matrix by {diff}");
    }

    // Test 3: outlier resistance — 9 near Identity + 1 far outlier
    #[test]
    fn outlier_resistance() {
        let identity = diag2(1.0, 1.0);
        let outlier = diag2(10.0, 10.0);
        let mut x: Vec<f64> = Vec::new();
        for _ in 0..9 {
            x.extend_from_slice(&identity);
        }
        x.extend_from_slice(&outlier);

        let config = RiemannianMedianConfig {
            matrix_dim: 2,
            max_iter: 500,
            tol: 1e-7,
            eps: 1e-10,
        };
        let res = riemannian_median(&x, 10, &config);
        assert!(res.is_ok());
        let median = res.expect("res should be present").median;
        let obj_median = riemannian_median_objective(&median, &x, 10, 2)
            .expect("riemannian_median_objective should succeed");

        let fm = spd_frechet_mean(
            &x,
            10,
            2,
            &FrechetMeanConfig {
                max_iter: 200,
                tol: 1e-8,
                step_size: 1.0,
            },
        )
        .expect("value should be present");
        let obj_mean = riemannian_median_objective(&fm.mean, &x, 10, 2)
            .expect("riemannian_median_objective should succeed");

        // Median minimises sum of distances, so obj(median) ≤ obj(Fréchet mean).
        assert!(
            obj_median <= obj_mean + 1e-5,
            "median obj {obj_median} > mean obj {obj_mean}"
        );
    }

    // Test 4: riemannian_median_objective is non-negative
    #[test]
    fn objective_nonnegative() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(3.0, 4.0), diag2(2.0, 3.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&x, 3, &config).expect("riemannian_median should succeed");
        let obj = riemannian_median_objective(&res.median, &x, 3, 2)
            .expect("riemannian_median_objective should succeed");
        assert!(obj >= 0.0, "objective must be non-negative, got {obj}");
    }

    // Test 5: result is SPD
    #[test]
    fn result_is_spd() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(2.0, 3.0), diag2(3.0, 5.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&x, 3, &config).expect("riemannian_median should succeed");
        assert!(
            is_spd_mat(&res.median, 2),
            "median is not SPD: {:?}",
            res.median
        );
    }

    // Test 6: converged=true on easy 2×2 problem
    #[test]
    fn converged_on_easy_problem() {
        let x: Vec<f64> = [diag2(1.0, 1.0), diag2(2.0, 2.0), diag2(1.5, 1.5)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-6,
            ..Default::default()
        };
        let res = riemannian_median(&x, 3, &config).expect("riemannian_median should succeed");
        assert!(res.converged, "should converge on easy diagonal problem");
    }

    // Test 7: 3×3 SPD works without numerical failure
    #[test]
    fn three_by_three_spd() {
        let x: Vec<f64> = [
            diag3(1.0, 2.0, 3.0),
            diag3(2.0, 3.0, 4.0),
            diag3(3.0, 1.0, 2.0),
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();
        let config = RiemannianMedianConfig {
            matrix_dim: 3,
            max_iter: 500,
            tol: 1e-6,
            eps: 1e-10,
        };
        let res = riemannian_median(&x, 3, &config);
        assert!(res.is_ok(), "3×3 SPD should work: {:?}", res.err());
        let res = res.expect("res should be present");
        assert!(is_spd_mat(&res.median, 3), "3×3 median must be SPD");
    }

    // Test 8: on symmetric data, median ≈ Fréchet mean (within 0.15)
    #[test]
    fn symmetric_data_median_near_frechet_mean() {
        let p1 = diag2(1.0, 2.0);
        let p2 = diag2(2.0, 1.0);
        let p3 = diag2(1.5, 1.5);
        let x: Vec<f64> = [&p1, &p2, &p3, &p1, &p2]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-8,
            ..Default::default()
        };
        let median_res =
            riemannian_median(&x, 5, &config).expect("riemannian_median should succeed");
        let fm = spd_frechet_mean(
            &x,
            5,
            2,
            &FrechetMeanConfig {
                max_iter: 200,
                tol: 1e-10,
                step_size: 1.0,
            },
        )
        .expect("value should be present");
        let diff = frob_diff(&median_res.median, &fm.mean);
        assert!(
            diff < 0.15,
            "median and Fréchet mean differ by {diff} > 0.15"
        );
    }

    // Test 9: median objective at returned point ≤ at any individual input matrix
    #[test]
    fn median_objective_better_than_inputs() {
        let mats = [
            diag2(1.0, 2.0),
            diag2(3.0, 5.0),
            diag2(2.0, 4.0),
            diag2(4.0, 3.0),
        ];
        let x: Vec<f64> = mats.iter().flat_map(|m| m.iter().copied()).collect();
        let config = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-7,
            ..Default::default()
        };
        let res = riemannian_median(&x, 4, &config).expect("riemannian_median should succeed");
        let obj_median = riemannian_median_objective(&res.median, &x, 4, 2)
            .expect("riemannian_median_objective should succeed");
        for (i, mat) in mats.iter().enumerate() {
            let obj_xi = riemannian_median_objective(mat, &x, 4, 2)
                .expect("riemannian_median_objective should succeed");
            assert!(
                obj_median <= obj_xi + 1e-4,
                "median obj {obj_median} > obj at input[{i}] = {obj_xi}"
            );
        }
    }

    // Test 10: n=1 works
    #[test]
    fn n_equals_1() {
        let p = diag2(3.0, 7.0);
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&p, 1, &config);
        assert!(res.is_ok());
    }

    // Test 11: n=2 works
    #[test]
    fn n_equals_2() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(3.0, 4.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&x, 2, &config);
        assert!(res.is_ok());
    }

    // Test 12: max_iter=1 with tight tol → converged=false
    #[test]
    fn max_iter_one_returns_partial() {
        let x: Vec<f64> = [
            diag2(1.0, 2.0),
            diag2(10.0, 20.0),
            diag2(5.0, 8.0),
            diag2(3.0, 6.0),
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();
        let config = RiemannianMedianConfig {
            max_iter: 1,
            tol: 1e-15,
            ..Default::default()
        };
        let res = riemannian_median(&x, 4, &config).expect("riemannian_median should succeed");
        assert!(
            !res.converged,
            "should not converge with max_iter=1 and tol=1e-15"
        );
    }

    // Test 13: loose tol → fewer iterations than tight tol
    #[test]
    fn loose_tol_fewer_iterations() {
        let x: Vec<f64> = [
            diag2(1.0, 2.0),
            diag2(4.0, 6.0),
            diag2(2.0, 5.0),
            diag2(3.0, 4.0),
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();
        let config_loose = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-2,
            ..Default::default()
        };
        let config_tight = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-7,
            ..Default::default()
        };
        let res_loose =
            riemannian_median(&x, 4, &config_loose).expect("riemannian_median should succeed");
        let res_tight =
            riemannian_median(&x, 4, &config_tight).expect("riemannian_median should succeed");
        assert!(
            res_loose.iterations <= res_tight.iterations,
            "loose tol iterations ({}) should be <= tight tol iterations ({})",
            res_loose.iterations,
            res_tight.iterations
        );
    }

    // Test 14: n=0 → InvalidParameter
    #[test]
    fn n_zero_error() {
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&[], 0, &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "n"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    // Test 15: x.len() != n*d² → ShapeMismatch
    #[test]
    fn shape_mismatch_error() {
        let x = vec![1.0; 6]; // n=2, d=2 needs 8 elements
        let config = RiemannianMedianConfig {
            matrix_dim: 2,
            ..Default::default()
        };
        let res = riemannian_median(&x, 2, &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::ShapeMismatch { .. } => {}
            e => panic!("expected ShapeMismatch, got {e:?}"),
        }
    }

    // Test 16: non-SPD input → ManifoldConstraint
    #[test]
    fn non_spd_input_error() {
        // [[-1, 0],[0, 1]] has eigenvalue -1 → not SPD
        let p_bad = [-1.0, 0.0, 0.0, 1.0];
        let p_good = diag2(2.0, 3.0);
        let x: Vec<f64> = p_good.iter().chain(p_bad.iter()).copied().collect();
        let config = RiemannianMedianConfig::default();
        let res = riemannian_median(&x, 2, &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::ManifoldConstraint(_) => {}
            e => panic!("expected ManifoldConstraint, got {e:?}"),
        }
    }

    // Test 17: riemannian_trimmed_mean alpha=0.0 → close to Fréchet mean
    #[test]
    fn trimmed_mean_alpha_zero_near_frechet() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(3.0, 4.0), diag2(2.0, 3.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let trimmed =
            riemannian_trimmed_mean(&x, 3, 2, 0.0).expect("riemannian_trimmed_mean should succeed");
        let fm = spd_frechet_mean(
            &x,
            3,
            2,
            &FrechetMeanConfig {
                max_iter: 200,
                tol: 1e-10,
                step_size: 1.0,
            },
        )
        .expect("value should be present");
        let diff = frob_diff(&trimmed, &fm.mean);
        assert!(
            diff < 1e-5,
            "trimmed mean (alpha=0) differs from Fréchet mean by {diff}"
        );
    }

    // Test 18: trimmed_mean alpha=0.2 removes outlier → closer to clean centroid
    #[test]
    fn trimmed_mean_alpha_removes_outlier() {
        let identity = diag2(1.0, 1.0);
        let outlier = diag2(1000.0, 1000.0);
        let mut x: Vec<f64> = Vec::new();
        for _ in 0..5 {
            x.extend_from_slice(&identity);
        }
        x.extend_from_slice(&outlier);

        let trimmed =
            riemannian_trimmed_mean(&x, 6, 2, 0.2).expect("riemannian_trimmed_mean should succeed");
        let untrimmed =
            riemannian_trimmed_mean(&x, 6, 2, 0.0).expect("riemannian_trimmed_mean should succeed");

        let dist_trimmed = frob_diff(&trimmed, &identity);
        let dist_untrimmed = frob_diff(&untrimmed, &identity);
        assert!(
            dist_trimmed < dist_untrimmed,
            "trimmed ({dist_trimmed}) should be closer to clean centroid than untrimmed ({dist_untrimmed})"
        );
    }

    // Test 19: trimmed_mean result is SPD
    #[test]
    fn trimmed_mean_result_is_spd() {
        let x: Vec<f64> = [
            diag2(1.0, 2.0),
            diag2(2.0, 3.0),
            diag2(3.0, 4.0),
            diag2(4.0, 5.0),
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();
        let res =
            riemannian_trimmed_mean(&x, 4, 2, 0.1).expect("riemannian_trimmed_mean should succeed");
        assert!(is_spd_mat(&res, 2), "trimmed mean must be SPD");
    }

    // Test 20: alpha < 0 → InvalidParameter
    #[test]
    fn trimmed_mean_alpha_negative_error() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(2.0, 3.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let res = riemannian_trimmed_mean(&x, 2, 2, -0.1);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    // Test 21: alpha >= 0.5 → InvalidParameter
    #[test]
    fn trimmed_mean_alpha_too_large_error() {
        let x: Vec<f64> = [diag2(1.0, 2.0), diag2(2.0, 3.0)]
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        let res = riemannian_trimmed_mean(&x, 2, 2, 0.5);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    // Test 22: median objective ≤ Fréchet mean objective on asymmetric data
    #[test]
    fn median_objective_not_worse_than_frechet_mean() {
        let x: Vec<f64> = [
            diag2(1.0, 1.0),
            diag2(1.1, 1.1),
            diag2(0.9, 0.9),
            diag2(1.05, 0.95),
            diag2(50.0, 50.0), // outlier
        ]
        .iter()
        .flat_map(|m| m.iter().copied())
        .collect();

        let config = RiemannianMedianConfig {
            max_iter: 500,
            tol: 1e-8,
            ..Default::default()
        };
        let median_res =
            riemannian_median(&x, 5, &config).expect("riemannian_median should succeed");
        let fm = spd_frechet_mean(
            &x,
            5,
            2,
            &FrechetMeanConfig {
                max_iter: 200,
                tol: 1e-8,
                step_size: 1.0,
            },
        )
        .expect("value should be present");

        let obj_median = riemannian_median_objective(&median_res.median, &x, 5, 2)
            .expect("riemannian_median_objective should succeed");
        let obj_mean = riemannian_median_objective(&fm.mean, &x, 5, 2)
            .expect("riemannian_median_objective should succeed");

        assert!(
            obj_median <= obj_mean + 1e-4,
            "median obj {obj_median} > Fréchet mean obj {obj_mean}"
        );
    }
}
