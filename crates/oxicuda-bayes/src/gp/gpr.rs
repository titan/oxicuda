//! Exact Gaussian Process Regression (GPR).
//!
//! Implements Rasmussen & Williams 2006 "Gaussian Processes for Machine
//! Learning" Algorithm 2.1: Cholesky-based GP prediction.
//!
//! Four kernel types are supported:
//! - RBF (squared exponential)
//! - Matérn 3/2
//! - Matérn 5/2
//! - Periodic
//!
//! Inference is performed via lower-triangular Cholesky decomposition.
//! If the covariance matrix is not positive-definite, automatic jitter
//! is added up to 6 times before returning `BayesError::SingularMatrix`.

use crate::error::{BayesError, BayesResult};

// ─── Kernel ─────────────────────────────────────────────────────────────────

/// Kernel function for GP regression.
#[derive(Debug, Clone, PartialEq)]
pub enum GprKernel {
    /// Squared exponential (RBF) kernel: σ²·exp(-r²/(2l²)).
    Rbf {
        length_scale: f64,
        signal_variance: f64,
    },
    /// Matérn 3/2 kernel: σ²(1+√3·r/l)·exp(-√3·r/l).
    Matern32 {
        length_scale: f64,
        signal_variance: f64,
    },
    /// Matérn 5/2 kernel: σ²(1+√5·r/l+5r²/(3l²))·exp(-√5·r/l).
    Matern52 {
        length_scale: f64,
        signal_variance: f64,
    },
    /// Periodic kernel: σ²·exp(-2·sin²(π·r/p)/l²).
    Periodic {
        length_scale: f64,
        period: f64,
        signal_variance: f64,
    },
}

impl GprKernel {
    /// Evaluate the kernel between two input vectors of dimension `d`.
    ///
    /// `xi` and `xj` must both have length `d`.
    pub fn eval(&self, xi: &[f64], xj: &[f64]) -> f64 {
        let r = euclidean_distance(xi, xj);
        match self {
            GprKernel::Rbf {
                length_scale,
                signal_variance,
            } => {
                let l = *length_scale;
                let sv = *signal_variance;
                sv * (-(r * r) / (2.0 * l * l)).exp()
            }
            GprKernel::Matern32 {
                length_scale,
                signal_variance,
            } => {
                let l = *length_scale;
                let sv = *signal_variance;
                let sqrt3_r_over_l = 3.0_f64.sqrt() * r / l;
                sv * (1.0 + sqrt3_r_over_l) * (-sqrt3_r_over_l).exp()
            }
            GprKernel::Matern52 {
                length_scale,
                signal_variance,
            } => {
                let l = *length_scale;
                let sv = *signal_variance;
                let sqrt5_r_over_l = 5.0_f64.sqrt() * r / l;
                let term = 1.0 + sqrt5_r_over_l + (5.0 * r * r) / (3.0 * l * l);
                sv * term * (-sqrt5_r_over_l).exp()
            }
            GprKernel::Periodic {
                length_scale,
                period,
                signal_variance,
            } => {
                let l = *length_scale;
                let p = *period;
                let sv = *signal_variance;
                let sin_val = (std::f64::consts::PI * r / p).sin();
                sv * (-2.0 * sin_val * sin_val / (l * l)).exp()
            }
        }
    }

    /// Evaluate the kernel matrix K(X_a, X_b) of shape n_a × n_b.
    ///
    /// `x_a` is row-major with shape (n_a, d), `x_b` with shape (n_b, d).
    /// Returns a row-major vector of length n_a * n_b.
    pub fn eval_matrix(
        &self,
        x_a: &[f64],
        n_a: usize,
        x_b: &[f64],
        n_b: usize,
        d: usize,
    ) -> Vec<f64> {
        let mut k = vec![0.0_f64; n_a * n_b];
        for i in 0..n_a {
            let xi = &x_a[i * d..(i + 1) * d];
            for j in 0..n_b {
                let xj = &x_b[j * d..(j + 1) * d];
                k[i * n_b + j] = self.eval(xi, xj);
            }
        }
        k
    }

    /// Validate that kernel hyperparameters are strictly positive.
    fn validate(&self) -> BayesResult<()> {
        match self {
            GprKernel::Rbf {
                length_scale,
                signal_variance,
            } => {
                if *length_scale <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "RBF length_scale must be > 0".into(),
                    ));
                }
                if *signal_variance <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "RBF signal_variance must be > 0".into(),
                    ));
                }
                Ok(())
            }
            GprKernel::Matern32 {
                length_scale,
                signal_variance,
            } => {
                if *length_scale <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Matern32 length_scale must be > 0".into(),
                    ));
                }
                if *signal_variance <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Matern32 signal_variance must be > 0".into(),
                    ));
                }
                Ok(())
            }
            GprKernel::Matern52 {
                length_scale,
                signal_variance,
            } => {
                if *length_scale <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Matern52 length_scale must be > 0".into(),
                    ));
                }
                if *signal_variance <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Matern52 signal_variance must be > 0".into(),
                    ));
                }
                Ok(())
            }
            GprKernel::Periodic {
                length_scale,
                period,
                signal_variance,
            } => {
                if *length_scale <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Periodic length_scale must be > 0".into(),
                    ));
                }
                if *period <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Periodic period must be > 0".into(),
                    ));
                }
                if *signal_variance <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Periodic signal_variance must be > 0".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Euclidean distance between two equal-length slices.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let d = ai - bi;
        sum += d * d;
    }
    sum.sqrt()
}

/// Lower-triangular Cholesky decomposition (Banachiewicz algorithm).
///
/// `a` is a row-major symmetric positive-definite matrix of size n×n.
/// Returns `Some(L)` (row-major lower triangular) if successful,
/// `None` if any diagonal pivot is non-positive.
fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - s;
                if diag <= 0.0 {
                    return None;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let lj = l[j * n + j];
                if lj == 0.0 {
                    return None;
                }
                l[i * n + j] = (a[i * n + j] - s) / lj;
            }
        }
    }
    Some(l)
}

/// Forward substitution: solve L·x = b, L lower triangular, returns x.
fn fwd_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() < 1e-300 { 0.0 } else { s / lii };
    }
    x
}

/// Backward substitution: solve Lᵀ·x = b, L lower triangular, returns x.
fn bwd_sub(lt: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            // Lᵀ[i,j] = L[j,i]
            s -= lt[j * n + i] * x[j];
        }
        let lii = lt[i * n + i];
        x[i] = if lii.abs() < 1e-300 { 0.0 } else { s / lii };
    }
    x
}

/// Build K + noise_variance·I + jitter·I (in-place on `k_matrix`).
fn add_noise_jitter(k: &mut [f64], n: usize, noise_variance: f64, jitter: f64) {
    for i in 0..n {
        k[i * n + i] += noise_variance + jitter;
    }
}

/// Attempt Cholesky with progressive jitter retries.
/// Returns (L, actual_jitter) or Err if all retries fail.
fn cholesky_with_jitter(
    k_base: &[f64],
    n: usize,
    noise_variance: f64,
    initial_jitter: f64,
) -> BayesResult<(Vec<f64>, f64)> {
    let mut jitter = initial_jitter;
    for _ in 0..7 {
        let mut k_aug = k_base.to_vec();
        add_noise_jitter(&mut k_aug, n, noise_variance, jitter);
        if let Some(l) = cholesky_lower(&k_aug, n) {
            return Ok((l, jitter));
        }
        jitter *= 10.0;
    }
    Err(BayesError::SingularMatrix(
        "GP covariance matrix is not positive-definite after jitter retries".into(),
    ))
}

// ─── Configuration & Fit structs ─────────────────────────────────────────────

/// Configuration for exact Gaussian Process regression.
#[derive(Debug, Clone)]
pub struct GprConfig {
    /// Kernel function.
    pub kernel: GprKernel,
    /// Observation noise variance ε² (added to diagonal). Default: 1e-4.
    pub noise_variance: f64,
    /// If true, normalise y to zero mean, unit variance before fitting.
    pub normalize_y: bool,
    /// Initial jitter for numerical stability. Default: 1e-6.
    pub jitter: f64,
}

impl Default for GprConfig {
    fn default() -> Self {
        Self {
            kernel: GprKernel::Rbf {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-4,
            normalize_y: false,
            jitter: 1e-6,
        }
    }
}

/// Fitted state of an exact GP regression model.
#[derive(Debug, Clone)]
pub struct GprFit {
    /// R&W Algorithm 2.1 dual variable: alpha = K_noisy⁻¹ y_normalized.
    pub alpha: Vec<f64>,
    /// Lower-triangular Cholesky factor L of K_noisy = L·Lᵀ.
    pub chol_l: Vec<f64>,
    /// Training inputs, row-major (n_train × d).
    pub x_train: Vec<f64>,
    /// Number of training points.
    pub n_train: usize,
    /// Input dimensionality.
    pub d: usize,
    /// Mean of training targets (0 when normalize_y=false).
    pub y_mean: f64,
    /// Std of training targets (1 when normalize_y=false or std≈0).
    pub y_std: f64,
    /// Log marginal likelihood log p(y|X) = −½ yᵀ alpha − Σlog `L[i,i]` − n/2 log(2π).
    pub log_marginal_likelihood: f64,
    /// Configuration used to fit this model.
    pub config: GprConfig,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Fit an exact GP regression model using Algorithm 2.1 (R&W 2006).
///
/// # Errors
/// - `InvalidConfig` if n==0, noise_variance<0, or kernel params invalid.
/// - `DimensionMismatch` if y.len() != n.
/// - `SingularMatrix` if covariance matrix not positive-definite after retries.
pub fn gpr_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    config: &GprConfig,
) -> BayesResult<GprFit> {
    // ── Validation ──────────────────────────────────────────────────────────
    if n == 0 {
        return Err(BayesError::InvalidConfig(
            "GP regression requires at least 1 training point".into(),
        ));
    }
    if y.len() != n {
        return Err(BayesError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }
    if config.noise_variance < 0.0 {
        return Err(BayesError::InvalidConfig(
            "noise_variance must be non-negative".into(),
        ));
    }
    config.kernel.validate()?;

    // ── Normalise y ─────────────────────────────────────────────────────────
    let (y_mean, y_std, y_norm) = if config.normalize_y {
        let mean = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|&yi| (yi - mean) * (yi - mean)).sum::<f64>() / n as f64;
        let std = if var < 1e-12 { 1.0 } else { var.sqrt() };
        let yn: Vec<f64> = y.iter().map(|&yi| (yi - mean) / std).collect();
        (mean, std, yn)
    } else {
        (0.0, 1.0, y.to_vec())
    };

    // ── Build K(X, X) ───────────────────────────────────────────────────────
    let k_base = config.kernel.eval_matrix(x, n, x, n, d);

    // ── Cholesky with progressive jitter ────────────────────────────────────
    let (chol_l, _actual_jitter) =
        cholesky_with_jitter(&k_base, n, config.noise_variance, config.jitter)?;

    // ── alpha = L⁻ᵀ (L⁻¹ y) ────────────────────────────────────────────────
    let v = fwd_sub(&chol_l, &y_norm, n);
    let alpha = bwd_sub(&chol_l, &v, n);

    // ── Log marginal likelihood ──────────────────────────────────────────────
    // log p(y|X) = -½ yᵀ alpha - Σᵢ log L[i,i] - n/2 log(2π)
    let y_dot_alpha: f64 = y_norm
        .iter()
        .zip(alpha.iter())
        .map(|(yi, ai)| yi * ai)
        .sum();
    let log_det_term: f64 = (0..n).map(|i| chol_l[i * n + i].ln()).sum();
    let log_marg =
        -0.5 * y_dot_alpha - log_det_term - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();

    Ok(GprFit {
        alpha,
        chol_l,
        x_train: x.to_vec(),
        n_train: n,
        d,
        y_mean,
        y_std,
        log_marginal_likelihood: log_marg,
        config: config.clone(),
    })
}

/// Predict GP posterior mean (and optionally std) at new input points.
///
/// # Errors
/// - `InvalidConfig` if n_new == 0.
/// - `DimensionMismatch` if x_new does not have n_new * d elements.
pub fn gpr_predict(
    fit: &GprFit,
    x_new: &[f64],
    n_new: usize,
    return_std: bool,
) -> BayesResult<(Vec<f64>, Option<Vec<f64>>)> {
    if n_new == 0 {
        return Err(BayesError::InvalidConfig(
            "prediction requires at least 1 test point".into(),
        ));
    }
    if x_new.len() != n_new * fit.d {
        return Err(BayesError::DimensionMismatch {
            expected: n_new * fit.d,
            got: x_new.len(),
        });
    }

    let n = fit.n_train;
    let d = fit.d;
    let mut means = Vec::with_capacity(n_new);
    let mut stds = if return_std {
        Some(Vec::with_capacity(n_new))
    } else {
        None
    };

    for idx in 0..n_new {
        let x_star = &x_new[idx * d..(idx + 1) * d];

        // k_star: kernel vector [k(x*, x_i)] for i = 0..n
        let k_star: Vec<f64> = (0..n)
            .map(|i| {
                fit.config
                    .kernel
                    .eval(x_star, &fit.x_train[i * d..(i + 1) * d])
            })
            .collect();

        // mean = k_star^T · alpha
        let mean_norm: f64 = k_star
            .iter()
            .zip(fit.alpha.iter())
            .map(|(k, a)| k * a)
            .sum();
        // un-normalise
        let mean = mean_norm * fit.y_std + fit.y_mean;
        means.push(mean);

        if let Some(ref mut s) = stds {
            // v = L⁻¹ k_star  (forward substitution)
            let v = fwd_sub(&fit.chol_l, &k_star, n);
            // k_star_star = k(x*, x*)
            let k_ss = fit.config.kernel.eval(x_star, x_star);
            // posterior variance = k** - v^T v
            let v_sq: f64 = v.iter().map(|vi| vi * vi).sum();
            let var_norm = (k_ss - v_sq).max(0.0);
            let std_val = var_norm.sqrt() * fit.y_std;
            s.push(std_val);
        }
    }

    Ok((means, stds))
}

/// Return the log marginal likelihood from a fitted GP.
#[must_use]
pub fn gpr_log_marginal_likelihood(fit: &GprFit) -> f64 {
    fit.log_marginal_likelihood
}

/// Compute kernel matrix K(X_a, X_b).
pub fn gpr_kernel_matrix(
    kernel: &GprKernel,
    x_a: &[f64],
    n_a: usize,
    x_b: &[f64],
    n_b: usize,
    d: usize,
) -> Vec<f64> {
    kernel.eval_matrix(x_a, n_a, x_b, n_b, d)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rbf_config() -> GprConfig {
        GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-4,
            normalize_y: false,
            jitter: 1e-6,
        }
    }

    /// Generate deterministic 1-D sin data: x_i = i*2π/n, y_i = sin(x_i).
    fn sin_data_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..n)
            .map(|i| i as f64 * 2.0 * std::f64::consts::PI / n as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        (xs, ys)
    }

    #[test]
    fn gpr_fit_rbf_sin_close_to_truth() {
        let (xs, ys) = sin_data_1d(20);
        let config = default_rbf_config();
        let fit = gpr_fit(&xs, &ys, 20, 1, &config).expect("gpr_fit should succeed");

        // Test at 10 new points
        let x_new: Vec<f64> = (0..10)
            .map(|i| i as f64 * 2.0 * std::f64::consts::PI / 10.0)
            .collect();
        let (means, _) = gpr_predict(&fit, &x_new, 10, false).expect("gpr_predict should succeed");

        for (i, (&xi, &mi)) in x_new.iter().zip(means.iter()).enumerate() {
            let truth = xi.sin();
            assert!(
                (mi - truth).abs() < 0.3,
                "point {i}: pred={mi:.4}, truth={truth:.4}, diff={:.4}",
                (mi - truth).abs()
            );
        }
    }

    #[test]
    fn gpr_mean_at_training_points_close_to_y_train() {
        let (xs, ys) = sin_data_1d(15);
        let config = GprConfig {
            noise_variance: 1e-4,
            ..default_rbf_config()
        };
        let fit = gpr_fit(&xs, &ys, 15, 1, &config).expect("gpr_fit should succeed");
        let (means, _) = gpr_predict(&fit, &xs, 15, false).expect("gpr_predict should succeed");

        for (i, (&yi, &mi)) in ys.iter().zip(means.iter()).enumerate() {
            assert!(
                (mi - yi).abs() < 1e-4 + 0.05,
                "pt {i}: pred={mi:.6}, y_train={yi:.6}"
            );
        }
    }

    #[test]
    fn gpr_std_grows_outside_training_range() {
        let (xs, ys) = sin_data_1d(10);
        let config = default_rbf_config();
        let fit = gpr_fit(&xs, &ys, 10, 1, &config).expect("gpr_fit should succeed");

        // Points inside training range
        let x_inside = vec![0.5, 1.5, 2.5];
        let (_, stds_inside) =
            gpr_predict(&fit, &x_inside, 3, true).expect("gpr_predict should succeed");

        // Points far outside training range
        let x_outside = vec![20.0, 30.0, 40.0];
        let (_, stds_outside) =
            gpr_predict(&fit, &x_outside, 3, true).expect("gpr_predict should succeed");

        let inside = stds_inside.expect("stds_inside should be present");
        let outside = stds_outside.expect("stds_outside should be present");

        let max_inside = inside.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_outside = outside.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_outside > max_inside,
            "std outside ({min_outside:.4}) should exceed max inside ({max_inside:.4})"
        );
    }

    #[test]
    fn gpr_kalpha_equals_y_residual() {
        // Verify ||K·alpha - y||_inf < 1e-4
        let (xs, ys) = sin_data_1d(12);
        let config = default_rbf_config();
        let fit = gpr_fit(&xs, &ys, 12, 1, &config).expect("gpr_fit should succeed");

        // Reconstruct K (without noise)
        let k_pure = fit.config.kernel.eval_matrix(&xs, 12, &xs, 12, 1);

        // K·alpha
        let mut k_alpha = [0.0_f64; 12];
        for i in 0..12 {
            let mut s = 0.0_f64;
            for j in 0..12 {
                s += k_pure[i * 12 + j] * fit.alpha[j];
            }
            k_alpha[i] = s;
        }

        // y_noisy = K_noisy · alpha, but we want K_noisy·alpha ≈ y
        // K_noisy = K + noise·I, so K_noisy·alpha ≈ y
        let k_noisy: Vec<f64> = k_pure
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                if idx / 12 == idx % 12 {
                    v + config.noise_variance
                } else {
                    v
                }
            })
            .collect();

        let mut k_noisy_alpha = [0.0_f64; 12];
        for i in 0..12 {
            let mut s = 0.0_f64;
            for j in 0..12 {
                s += k_noisy[i * 12 + j] * fit.alpha[j];
            }
            k_noisy_alpha[i] = s;
        }

        let max_err = k_noisy_alpha
            .iter()
            .zip(ys.iter())
            .map(|(ka, yi)| (ka - yi).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-4, "||K_noisy·alpha - y||_inf = {max_err:.2e}");
    }

    #[test]
    fn gpr_log_marginal_likelihood_is_finite() {
        let (xs, ys) = sin_data_1d(10);
        let fit = gpr_fit(&xs, &ys, 10, 1, &default_rbf_config()).expect("value should be present");
        let lml = gpr_log_marginal_likelihood(&fit);
        assert!(lml.is_finite(), "LML should be finite, got {lml}");
    }

    #[test]
    fn gpr_normalize_y_roundtrip() {
        let (xs, ys) = sin_data_1d(15);
        // Scale y to have non-trivial mean and std
        let ys_scaled: Vec<f64> = ys.iter().map(|&y| 100.0 * y + 50.0).collect();

        let config_norm = GprConfig {
            normalize_y: true,
            ..default_rbf_config()
        };
        let config_raw = GprConfig {
            normalize_y: false,
            ..default_rbf_config()
        };

        let fit_norm =
            gpr_fit(&xs, &ys_scaled, 15, 1, &config_norm).expect("gpr_fit should succeed");
        let fit_raw = gpr_fit(&xs, &ys_scaled, 15, 1, &config_raw).expect("gpr_fit should succeed");

        let x_test = vec![1.0, 2.0, 3.0];
        let (means_norm, _) =
            gpr_predict(&fit_norm, &x_test, 3, false).expect("gpr_predict should succeed");
        let (means_raw, _) =
            gpr_predict(&fit_raw, &x_test, 3, false).expect("gpr_predict should succeed");

        for (mn, mr) in means_norm.iter().zip(means_raw.iter()) {
            assert!(
                (mn - mr).abs() < 5.0,
                "normalized vs raw mean differ too much: {mn:.2} vs {mr:.2}"
            );
        }
    }

    #[test]
    fn gpr_kernel_matrix_shape_correct() {
        let x_a: Vec<f64> = vec![0.0, 1.0, 2.0];
        let x_b: Vec<f64> = vec![0.5, 1.5];
        let k = GprKernel::Rbf {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let km = gpr_kernel_matrix(&k, &x_a, 3, &x_b, 2, 1);
        assert_eq!(km.len(), 6);
    }

    #[test]
    fn gpr_kernel_matrix_symmetric() {
        let xs: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
        let k = GprKernel::Rbf {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let km = gpr_kernel_matrix(&k, &xs, 5, &xs, 5, 1);
        for i in 0..5 {
            for j in 0..5 {
                let diff = (km[i * 5 + j] - km[j * 5 + i]).abs();
                assert!(
                    diff < 1e-14,
                    "K[{i},{j}]={} != K[{j},{i}]={}",
                    km[i * 5 + j],
                    km[j * 5 + i]
                );
            }
        }
    }

    #[test]
    fn gpr_matern32_fit_and_predict() {
        let (xs, ys) = sin_data_1d(15);
        let config = GprConfig {
            kernel: GprKernel::Matern32 {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-4,
            normalize_y: false,
            jitter: 1e-6,
        };
        let fit = gpr_fit(&xs, &ys, 15, 1, &config).expect("gpr_fit should succeed");
        let x_test: Vec<f64> = vec![1.0, 2.0, 3.0];
        let (means, stds) =
            gpr_predict(&fit, &x_test, 3, true).expect("gpr_predict should succeed");
        assert_eq!(means.len(), 3);
        let stds_vals = stds.expect("stds should be present");
        for s in &stds_vals {
            assert!(*s >= 0.0, "std must be non-negative");
        }
    }

    #[test]
    fn gpr_matern52_fit_and_predict() {
        let (xs, ys) = sin_data_1d(15);
        let config = GprConfig {
            kernel: GprKernel::Matern52 {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-4,
            normalize_y: false,
            jitter: 1e-6,
        };
        let fit = gpr_fit(&xs, &ys, 15, 1, &config).expect("gpr_fit should succeed");
        let x_test: Vec<f64> = vec![1.0, 2.0, 3.0];
        let (means, _) = gpr_predict(&fit, &x_test, 3, false).expect("gpr_predict should succeed");
        assert_eq!(means.len(), 3);
    }

    #[test]
    fn gpr_periodic_kernel_fit_and_predict() {
        let (xs, ys) = sin_data_1d(20);
        let config = GprConfig {
            kernel: GprKernel::Periodic {
                length_scale: 1.0,
                period: std::f64::consts::PI,
                signal_variance: 1.0,
            },
            noise_variance: 1e-3,
            normalize_y: false,
            jitter: 1e-6,
        };
        let fit = gpr_fit(&xs, &ys, 20, 1, &config).expect("gpr_fit should succeed");
        let x_test: Vec<f64> = vec![0.5, 1.5, 2.5];
        let (means, _) = gpr_predict(&fit, &x_test, 3, false).expect("gpr_predict should succeed");
        assert_eq!(means.len(), 3);
        for m in &means {
            assert!(m.is_finite(), "predicted mean must be finite");
        }
    }

    #[test]
    fn gpr_std_is_nonnegative() {
        let (xs, ys) = sin_data_1d(10);
        let fit = gpr_fit(&xs, &ys, 10, 1, &default_rbf_config()).expect("value should be present");
        let x_test: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let (_, stds) = gpr_predict(&fit, &x_test, 20, true).expect("gpr_predict should succeed");
        for s in stds.expect("stds should be present") {
            assert!(s >= 0.0, "std = {s}");
        }
    }

    #[test]
    fn gpr_multidim_input() {
        // 2D inputs: (x1, x2), y = x1 + x2
        let n = 20;
        let d = 2;
        let xs: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = i as f64 * 0.1;
                let x2 = (i as f64 * 0.1 + 0.5) % 1.0;
                vec![x1, x2]
            })
            .collect();
        let ys: Vec<f64> = (0..n).map(|i| xs[i * d] + xs[i * d + 1]).collect();
        let config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.5,
                signal_variance: 1.0,
            },
            noise_variance: 1e-4,
            normalize_y: false,
            jitter: 1e-6,
        };
        let fit = gpr_fit(&xs, &ys, n, d, &config).expect("gpr_fit should succeed");
        let x_new = vec![0.5, 0.3];
        let (means, _) = gpr_predict(&fit, &x_new, 1, false).expect("gpr_predict should succeed");
        let truth = 0.5 + 0.3;
        assert!(
            (means[0] - truth).abs() < 0.5,
            "2D prediction {:.4} vs truth {:.4}",
            means[0],
            truth
        );
    }

    #[test]
    fn gpr_error_on_zero_n() {
        let config = default_rbf_config();
        let result = gpr_fit(&[], &[], 0, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn gpr_error_dimension_mismatch() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0]; // wrong length
        let result = gpr_fit(&xs, &ys, 3, 1, &default_rbf_config());
        assert!(matches!(result, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn gpr_error_negative_noise() {
        let config = GprConfig {
            noise_variance: -1.0,
            ..default_rbf_config()
        };
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let result = gpr_fit(&xs, &ys, 2, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn gpr_error_invalid_length_scale() {
        let config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.0,
                signal_variance: 1.0,
            },
            ..default_rbf_config()
        };
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let result = gpr_fit(&xs, &ys, 2, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn gpr_error_invalid_signal_variance() {
        let config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 1.0,
                signal_variance: -0.5,
            },
            ..default_rbf_config()
        };
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let result = gpr_fit(&xs, &ys, 2, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn gpr_predict_zero_n_new_error() {
        let (xs, ys) = sin_data_1d(5);
        let fit = gpr_fit(&xs, &ys, 5, 1, &default_rbf_config()).expect("value should be present");
        let result = gpr_predict(&fit, &[], 0, false);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn gpr_rbf_kernel_at_same_point() {
        let k = GprKernel::Rbf {
            length_scale: 1.0,
            signal_variance: 2.0,
        };
        let x = [0.5_f64];
        // k(x, x) = σ² * exp(0) = σ²
        let val = k.eval(&x, &x);
        assert!((val - 2.0).abs() < 1e-12);
    }

    #[test]
    fn gpr_matern32_kernel_at_same_point() {
        let k = GprKernel::Matern32 {
            length_scale: 1.0,
            signal_variance: 3.0,
        };
        let x = [1.0_f64];
        // k(x, x) = σ²*(1+0)*exp(0) = σ²
        let val = k.eval(&x, &x);
        assert!((val - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gpr_matern52_kernel_at_same_point() {
        let k = GprKernel::Matern52 {
            length_scale: 1.0,
            signal_variance: 1.5,
        };
        let x = [0.0_f64];
        let val = k.eval(&x, &x);
        assert!((val - 1.5).abs() < 1e-12);
    }

    #[test]
    fn gpr_periodic_kernel_symmetry() {
        let k = GprKernel::Periodic {
            length_scale: 1.0,
            period: 2.0,
            signal_variance: 1.0,
        };
        let xi = [0.3_f64];
        let xj = [0.8_f64];
        let kij = k.eval(&xi, &xj);
        let kji = k.eval(&xj, &xi);
        assert!((kij - kji).abs() < 1e-14, "periodic kernel not symmetric");
    }

    #[test]
    fn gpr_large_dataset_sin() {
        // n=50, ensure it computes without error
        let (xs, ys) = sin_data_1d(50);
        let config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-3,
            normalize_y: true,
            jitter: 1e-6,
        };
        let fit = gpr_fit(&xs, &ys, 50, 1, &config).expect("gpr_fit should succeed");
        let lml = gpr_log_marginal_likelihood(&fit);
        assert!(lml.is_finite());
        let x_test: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
        let (means, stds) =
            gpr_predict(&fit, &x_test, 5, true).expect("gpr_predict should succeed");
        assert_eq!(means.len(), 5);
        for (m, s) in means
            .iter()
            .zip(stds.expect("stds should be present").iter())
        {
            assert!(m.is_finite());
            assert!(*s >= 0.0);
        }
    }
}
