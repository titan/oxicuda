//! Sparse Gaussian Process Regression via FITC approximation.
//!
//! Implements Snelson & Ghahramani 2006 NeurIPS "Sparse Gaussian Processes
//! using Pseudo-inputs" (FITC: Fully Independent Training Conditional).
//!
//! Given n training points and m inducing points (m << n), FITC reduces
//! the O(n³) exact GP cost to O(nm²) by approximating the full covariance
//! via low-rank + diagonal structure.
//!
//! The ELBO (evidence lower bound) follows Titsias 2009 for the variational
//! free energy, providing a principled bound on the log marginal likelihood.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

use super::gpr::GprKernel;

// ─── Inducing point initialisation ──────────────────────────────────────────

/// Strategy for initialising inducing point locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InducingInit {
    /// Sub-sample m random indices from the training set.
    Random,
    /// Use the first m training points as inducing points.
    FirstN,
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for sparse GP regression (FITC).
#[derive(Debug, Clone)]
pub struct SparseGpConfig {
    /// Kernel function shared with the exact GP.
    pub kernel: GprKernel,
    /// Observation noise variance ε² (added to diagonal). Default: 1e-4.
    pub noise_variance: f64,
    /// Number of inducing points m. Clamped to min(n_inducing, n).
    pub n_inducing: usize,
    /// Numerical stability jitter. Default: 1e-6.
    pub jitter: f64,
    /// Normalise y to zero mean, unit variance before fitting.
    pub normalize_y: bool,
    /// Strategy for initialising inducing point locations.
    pub inducing_init: InducingInit,
}

// ─── Fit struct ─────────────────────────────────────────────────────────────

/// Fitted state of a sparse GP model.
#[derive(Debug, Clone)]
pub struct SparseGpFit {
    /// Inducing point locations, row-major (m × d).
    pub inducing_z: Vec<f64>,
    /// Number of inducing points m.
    pub n_inducing: usize,
    /// Number of training points n.
    pub n_train: usize,
    /// Input dimensionality.
    pub d: usize,
    /// Cholesky factor L_mm of K_mm + jitter·I (m×m lower triangular).
    pub chol_l_mm: Vec<f64>,
    /// Cholesky factor L_B of B = I_m + V·diag(1/Λ)·V^T (m×m lower triangular).
    pub chol_l_b: Vec<f64>,
    /// γ = L_B⁻¹ · h, where h = V · (y/Λ).
    pub gamma: Vec<f64>,
    /// Mean of training targets (0 when normalize_y=false).
    pub y_mean: f64,
    /// Std of training targets (1 when normalize_y=false or std≈0).
    pub y_std: f64,
    /// FITC ELBO value.
    pub elbo: f64,
    /// Configuration used to fit.
    pub config: SparseGpConfig,
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Lower-triangular Cholesky decomposition (Banachiewicz).
/// Returns `Some(L)` on success or `None` if not positive-definite.
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

/// Forward substitution: solve L·x = b, returns x.
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

/// Cholesky with progressive jitter, up to 7 attempts.
fn cholesky_jitter(a_base: &[f64], n: usize, initial_jitter: f64) -> BayesResult<Vec<f64>> {
    let mut jitter = initial_jitter;
    for _ in 0..7 {
        let mut a = a_base.to_vec();
        for i in 0..n {
            a[i * n + i] += jitter;
        }
        if let Some(l) = cholesky_lower(&a, n) {
            return Ok(l);
        }
        jitter *= 10.0;
    }
    Err(BayesError::SingularMatrix(
        "sparse GP matrix not positive-definite after jitter retries".into(),
    ))
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Fit a sparse GP using the FITC approximation.
///
/// # Errors
/// - `InvalidConfig` if n==0 or n_inducing==0 or noise_variance<0.
/// - `DimensionMismatch` if y.len() != n.
/// - `SingularMatrix` if any Cholesky fails after retries.
pub fn sparse_gp_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    config: &SparseGpConfig,
) -> BayesResult<SparseGpFit> {
    // ── Validation ──────────────────────────────────────────────────────────
    if n == 0 {
        return Err(BayesError::InvalidConfig(
            "sparse GP requires at least 1 training point".into(),
        ));
    }
    if config.n_inducing == 0 {
        return Err(BayesError::InvalidConfig("n_inducing must be >= 1".into()));
    }
    if config.noise_variance < 0.0 {
        return Err(BayesError::InvalidConfig(
            "noise_variance must be non-negative".into(),
        ));
    }
    if y.len() != n {
        return Err(BayesError::DimensionMismatch {
            expected: n,
            got: y.len(),
        });
    }

    // ── Clamp n_inducing ────────────────────────────────────────────────────
    let m = config.n_inducing.min(n);

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

    // ── Select inducing points Z ─────────────────────────────────────────────
    let inducing_z = select_inducing_points(x, n, d, m, config.inducing_init);

    // ── K_mm = K(Z, Z) ──────────────────────────────────────────────────────
    let k_mm = config.kernel.eval_matrix(&inducing_z, m, &inducing_z, m, d);

    // ── L_mm: Cholesky of K_mm + jitter·I ───────────────────────────────────
    let chol_l_mm = cholesky_jitter(&k_mm, m, config.jitter)?;

    // ── K_nm = K(X, Z) [n×m], K_mn = K_nm^T [m×n] ──────────────────────────
    let k_nm = config.kernel.eval_matrix(x, n, &inducing_z, m, d);

    // ── V = L_mm⁻¹ · K_mn [m×n]: solve L_mm·V[:,i] = K_mn[:,i] ─────────────
    // We store V as [m×n] row-major.
    // K_mn[k,i] = K_nm[i*m + k] (column k of K_nm is row k of K_mn)
    let mut v = vec![0.0_f64; m * n];
    for i in 0..n {
        // Extract K_mn[:, i] = K_nm[i, :] = k_nm[i*m .. (i+1)*m]
        let k_mn_col_i: Vec<f64> = (0..m).map(|k| k_nm[i * m + k]).collect();
        let v_col = fwd_sub(&chol_l_mm, &k_mn_col_i, m);
        // Store as column i of V (row-major, row k → V[k*n+i])
        for k in 0..m {
            v[k * n + i] = v_col[k];
        }
    }

    // ── diag_Knn = [k(x_i, x_i) for i=0..n] ───────────────────────────────
    let diag_knn: Vec<f64> = (0..n)
        .map(|i| {
            config
                .kernel
                .eval(&x[i * d..(i + 1) * d], &x[i * d..(i + 1) * d])
        })
        .collect();

    // ── Q_nn_diag = [||V[:,i]||² for i=0..n] ────────────────────────────────
    let q_nn_diag: Vec<f64> = (0..n)
        .map(|i| {
            (0..m)
                .map(|k| {
                    let vi = v[k * n + i];
                    vi * vi
                })
                .sum::<f64>()
        })
        .collect();

    // ── Λ = diag_Knn - Q_nn_diag + noise_variance (clipped to ≥ jitter) ─────
    let lambda: Vec<f64> = (0..n)
        .map(|i| {
            let raw = diag_knn[i] - q_nn_diag[i] + config.noise_variance;
            raw.max(config.jitter)
        })
        .collect();

    // ── B = I_m + V·diag(1/Λ)·V^T [m×m] ────────────────────────────────────
    // B[k,l] = δ_{kl} + Σᵢ V[k,i] * V[l,i] / Λ[i]
    let mut b_mat = vec![0.0_f64; m * m];
    for k in 0..m {
        for l in 0..m {
            let s: f64 = (0..n)
                .map(|i| v[k * n + i] * v[l * n + i] / lambda[i])
                .sum();
            b_mat[k * m + l] = s + if k == l { 1.0 } else { 0.0 };
        }
    }

    // ── L_B: Cholesky of B ──────────────────────────────────────────────────
    let chol_l_b = cholesky_jitter(&b_mat, m, config.jitter)?;

    // ── h = V·(y_norm/Λ) [m] ────────────────────────────────────────────────
    // h[k] = Σᵢ V[k,i] * y_norm[i] / λ[i]
    let h: Vec<f64> = (0..m)
        .map(|k| {
            (0..n)
                .map(|i| v[k * n + i] * y_norm[i] / lambda[i])
                .sum::<f64>()
        })
        .collect();

    // ── γ = L_B⁻¹ · h ───────────────────────────────────────────────────────
    let gamma = fwd_sub(&chol_l_b, &h, m);

    // ── ELBO ─────────────────────────────────────────────────────────────────
    // log_det = Σᵢ log(Λᵢ) + 2·Σⱼ log(L_B[j,j])
    let log_det_lambda: f64 = lambda.iter().map(|&li| li.ln()).sum();
    let log_det_lb: f64 = (0..m).map(|j| chol_l_b[j * m + j].ln()).sum();
    let log_det = log_det_lambda + 2.0 * log_det_lb;

    // quad = Σᵢ (y[i]² / Λ[i]) - γᵀ·γ
    let quad_y: f64 = (0..n).map(|i| y_norm[i] * y_norm[i] / lambda[i]).sum();
    let quad_gamma: f64 = gamma.iter().map(|&g| g * g).sum();
    let quad = quad_y - quad_gamma;

    // trace_term = (1/noise_variance) * Σᵢ max(0, diag_Knn[i] - Q_nn_diag[i])
    let trace_term = if config.noise_variance > 0.0 {
        let sum: f64 = (0..n).map(|i| (diag_knn[i] - q_nn_diag[i]).max(0.0)).sum();
        sum / config.noise_variance
    } else {
        0.0
    };

    let elbo =
        -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad) - 0.5 * trace_term;

    Ok(SparseGpFit {
        inducing_z,
        n_inducing: m,
        n_train: n,
        d,
        chol_l_mm,
        chol_l_b,
        gamma,
        y_mean,
        y_std,
        elbo,
        config: config.clone(),
    })
}

/// Predict FITC posterior mean (and optionally std) at new input points.
///
/// # Errors
/// - `InvalidConfig` if n_new == 0.
/// - `DimensionMismatch` if x_new.len() != n_new * d.
pub fn sparse_gp_predict(
    fit: &SparseGpFit,
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

    let m = fit.n_inducing;
    let d = fit.d;

    let mut means = Vec::with_capacity(n_new);
    let mut stds = if return_std {
        Some(Vec::with_capacity(n_new))
    } else {
        None
    };

    for idx in 0..n_new {
        let x_star = &x_new[idx * d..(idx + 1) * d];

        // k_*m = [K(x*, z_j) for j=0..m]
        let k_star_m: Vec<f64> = (0..m)
            .map(|j| {
                fit.config
                    .kernel
                    .eval(x_star, &fit.inducing_z[j * d..(j + 1) * d])
            })
            .collect();

        // e = L_mm⁻¹ · k_*m
        let e = fwd_sub(&fit.chol_l_mm, &k_star_m, m);

        // f = L_B⁻¹ · e
        let f = fwd_sub(&fit.chol_l_b, &e, m);

        // μ* = f^T · γ
        let mean_norm: f64 = f.iter().zip(fit.gamma.iter()).map(|(fi, gi)| fi * gi).sum();
        let mean = mean_norm * fit.y_std + fit.y_mean;
        means.push(mean);

        if let Some(ref mut s) = stds {
            // k(x*, x*)
            let k_ss = fit.config.kernel.eval(x_star, x_star);
            // σ*² = k** - e^T·e + f^T·f + noise_variance
            let e_sq: f64 = e.iter().map(|ei| ei * ei).sum();
            let f_sq: f64 = f.iter().map(|fi| fi * fi).sum();
            let var_norm = (k_ss - e_sq + f_sq + fit.config.noise_variance).max(0.0);
            let std_val = var_norm.sqrt() * fit.y_std;
            s.push(std_val);
        }
    }

    Ok((means, stds))
}

/// Return the ELBO from a fitted sparse GP.
#[must_use]
pub fn sparse_gp_elbo(fit: &SparseGpFit) -> f64 {
    fit.elbo
}

// ─── Helper: inducing point selection ────────────────────────────────────────

fn select_inducing_points(x: &[f64], n: usize, d: usize, m: usize, init: InducingInit) -> Vec<f64> {
    match init {
        InducingInit::FirstN => {
            // Take first m training points
            x[..m * d].to_vec()
        }
        InducingInit::Random => {
            let mut rng = LcgRng::new(0xdead_beef_cafe_u64);
            let mut indices: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut indices);
            let mut z = vec![0.0_f64; m * d];
            for (k, &idx) in indices[..m].iter().enumerate() {
                z[k * d..(k + 1) * d].copy_from_slice(&x[idx * d..(idx + 1) * d]);
            }
            z
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::gpr::{GprConfig, gpr_fit, gpr_predict};
    use super::*;

    fn default_sparse_config(n_inducing: usize) -> SparseGpConfig {
        SparseGpConfig {
            kernel: GprKernel::Rbf {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            n_inducing,
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        }
    }

    /// Generate deterministic 1D sin data: x_i = i/(n-1), y_i = sin(2π x_i).
    fn sin_data_unit(n: usize) -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| (2.0 * std::f64::consts::PI * x).sin())
            .collect();
        (xs, ys)
    }

    #[test]
    fn sparse_gp_fitc_close_to_exact_gp() {
        // n=80, m=20: FITC mean should be close to exact GP mean.
        // We evenly sub-sample every 4th training point as inducing points so
        // they span the full [0, 1] domain rather than piling up at the start.
        let n = 80;
        let (xs, _ys) = sin_data_unit(n);

        // Build evenly-spaced inducing Z: indices 0, 4, 8, … → m=20 points
        let m = 20;
        let step = n / m;
        let inducing_z: Vec<f64> = (0..m).map(|k| xs[k * step]).collect();

        // Build SparseGpConfig with FirstN (we override inducing_z via a
        // helper below). Instead, we use n_inducing=20 with the full xs but
        // pick every-step-th point. Achieve this by reordering xs so the
        // first 20 entries are the evenly-spaced ones.
        let mut xs_reordered = Vec::with_capacity(n);
        // First m: evenly spaced
        for k in 0..m {
            xs_reordered.push(xs[k * step]);
        }
        // Then the rest
        for (i, &xi) in xs.iter().enumerate().take(n) {
            if i % step != 0 {
                xs_reordered.push(xi);
            }
        }
        let ys_reordered: Vec<f64> = xs_reordered
            .iter()
            .map(|&x| (2.0 * std::f64::consts::PI * x).sin())
            .collect();

        let sparse_config = SparseGpConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.15,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            n_inducing: m,
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        };
        let sparse_fit = sparse_gp_fit(&xs_reordered, &ys_reordered, n, 1, &sparse_config)
            .expect("sparse_gp_fit should succeed");
        // Verify inducing points match the evenly-spaced ones
        for (k, &iz) in sparse_fit.inducing_z.iter().enumerate() {
            assert!((iz - inducing_z[k]).abs() < 1e-12, "inducing[{k}] mismatch");
        }

        let exact_config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.15,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            normalize_y: false,
            jitter: 1e-6,
        };
        let exact_fit = gpr_fit(&xs_reordered, &ys_reordered, n, 1, &exact_config)
            .expect("gpr_fit should succeed");

        // 20 test points in [0, 1]
        let x_test: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();

        let (sparse_means, _) = sparse_gp_predict(&sparse_fit, &x_test, 20, false)
            .expect("sparse_gp_predict should succeed");
        let (exact_means, _) =
            gpr_predict(&exact_fit, &x_test, 20, false).expect("gpr_predict should succeed");

        let max_diff = sparse_means
            .iter()
            .zip(exact_means.iter())
            .map(|(s, e)| (s - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.5,
            "FITC vs exact max |μ_FITC - μ_exact| = {max_diff:.4} (should be < 0.5)"
        );
    }

    #[test]
    fn sparse_gp_m_eq_n_first_n_close_to_exact() {
        // m=n: FITC with all training as inducing should approximate exact GP
        let n = 30;
        let (xs, ys) = sin_data_unit(n);
        let sparse_config = SparseGpConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.2,
                signal_variance: 1.0,
            },
            noise_variance: 1e-3,
            n_inducing: n,
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        };
        let sparse_fit =
            sparse_gp_fit(&xs, &ys, n, 1, &sparse_config).expect("sparse_gp_fit should succeed");
        assert_eq!(sparse_fit.n_inducing, n);

        let exact_config = GprConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.2,
                signal_variance: 1.0,
            },
            noise_variance: 1e-3,
            normalize_y: false,
            jitter: 1e-6,
        };
        let exact_fit = gpr_fit(&xs, &ys, n, 1, &exact_config).expect("gpr_fit should succeed");

        let x_test: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let (sparse_means, _) = sparse_gp_predict(&sparse_fit, &x_test, 10, false)
            .expect("sparse_gp_predict should succeed");
        let (exact_means, _) =
            gpr_predict(&exact_fit, &x_test, 10, false).expect("gpr_predict should succeed");

        let max_diff = sparse_means
            .iter()
            .zip(exact_means.iter())
            .map(|(s, e)| (s - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.1,
            "m=n FITC vs exact max diff = {max_diff:.4} (should be < 0.1)"
        );
    }

    #[test]
    fn sparse_gp_elbo_finite_not_nan() {
        let (xs, ys) = sin_data_unit(40);
        let config = default_sparse_config(10);
        let fit = sparse_gp_fit(&xs, &ys, 40, 1, &config).expect("sparse_gp_fit should succeed");
        let elbo = sparse_gp_elbo(&fit);
        assert!(elbo.is_finite(), "ELBO should be finite, got {elbo}");
        assert!(!elbo.is_nan(), "ELBO should not be NaN");
    }

    #[test]
    fn sparse_gp_std_positive_everywhere() {
        let (xs, ys) = sin_data_unit(40);
        let config = default_sparse_config(10);
        let fit = sparse_gp_fit(&xs, &ys, 40, 1, &config).expect("sparse_gp_fit should succeed");
        let x_test: Vec<f64> = (0..15).map(|i| i as f64 / 14.0).collect();
        let (_, stds) =
            sparse_gp_predict(&fit, &x_test, 15, true).expect("sparse_gp_predict should succeed");
        for s in stds.expect("stds should be present") {
            assert!(s >= 0.0, "std = {s} should be non-negative");
        }
    }

    #[test]
    fn sparse_gp_random_init_runs() {
        let (xs, ys) = sin_data_unit(50);
        let config = SparseGpConfig {
            inducing_init: InducingInit::Random,
            ..default_sparse_config(15)
        };
        let fit = sparse_gp_fit(&xs, &ys, 50, 1, &config).expect("sparse_gp_fit should succeed");
        let x_test: Vec<f64> = vec![0.2, 0.5, 0.8];
        let (means, _) =
            sparse_gp_predict(&fit, &x_test, 3, false).expect("sparse_gp_predict should succeed");
        for m in means {
            assert!(m.is_finite());
        }
    }

    #[test]
    fn sparse_gp_n_inducing_clamp_to_n() {
        let n = 10;
        let (xs, ys) = sin_data_unit(n);
        // n_inducing > n should be clamped
        let config = SparseGpConfig {
            n_inducing: 100,
            ..default_sparse_config(100)
        };
        let fit = sparse_gp_fit(&xs, &ys, n, 1, &config).expect("sparse_gp_fit should succeed");
        assert_eq!(
            fit.n_inducing, n,
            "n_inducing should be clamped to n={n}, got {}",
            fit.n_inducing
        );
    }

    #[test]
    fn sparse_gp_error_on_zero_n() {
        let config = default_sparse_config(5);
        let result = sparse_gp_fit(&[], &[], 0, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn sparse_gp_error_on_zero_n_inducing() {
        let (xs, ys) = sin_data_unit(10);
        let config = SparseGpConfig {
            n_inducing: 0,
            ..default_sparse_config(0)
        };
        let result = sparse_gp_fit(&xs, &ys, 10, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn sparse_gp_error_negative_noise() {
        let (xs, ys) = sin_data_unit(10);
        let config = SparseGpConfig {
            noise_variance: -0.1,
            ..default_sparse_config(5)
        };
        let result = sparse_gp_fit(&xs, &ys, 10, 1, &config);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn sparse_gp_error_dimension_mismatch() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0]; // wrong length
        let config = default_sparse_config(2);
        let result = sparse_gp_fit(&xs, &ys, 3, 1, &config);
        assert!(matches!(result, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn sparse_gp_predict_zero_n_new_error() {
        let (xs, ys) = sin_data_unit(10);
        let config = default_sparse_config(5);
        let fit = sparse_gp_fit(&xs, &ys, 10, 1, &config).expect("sparse_gp_fit should succeed");
        let result = sparse_gp_predict(&fit, &[], 0, false);
        assert!(matches!(result, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn sparse_gp_matern32_works() {
        let (xs, ys) = sin_data_unit(30);
        let config = SparseGpConfig {
            kernel: GprKernel::Matern32 {
                length_scale: 0.3,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            n_inducing: 8,
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        };
        let fit = sparse_gp_fit(&xs, &ys, 30, 1, &config).expect("sparse_gp_fit should succeed");
        let x_test = vec![0.3, 0.6, 0.9];
        let (means, stds) =
            sparse_gp_predict(&fit, &x_test, 3, true).expect("sparse_gp_predict should succeed");
        assert_eq!(means.len(), 3);
        for s in stds.expect("stds should be present") {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn sparse_gp_normalize_y_gives_finite_predictions() {
        let (xs, ys) = sin_data_unit(30);
        let ys_scaled: Vec<f64> = ys.iter().map(|&y| 100.0 * y + 50.0).collect();
        let config = SparseGpConfig {
            normalize_y: true,
            ..default_sparse_config(10)
        };
        let fit =
            sparse_gp_fit(&xs, &ys_scaled, 30, 1, &config).expect("sparse_gp_fit should succeed");
        let x_test: Vec<f64> = vec![0.1, 0.5, 0.9];
        let (means, _) =
            sparse_gp_predict(&fit, &x_test, 3, false).expect("sparse_gp_predict should succeed");
        for m in means {
            assert!(m.is_finite(), "mean must be finite: {m}");
            // should be roughly in range of ys_scaled ≈ [-50, 150]
            assert!(m > -200.0 && m < 300.0);
        }
    }

    #[test]
    fn sparse_gp_elbo_decreases_with_fewer_inducing() {
        // More inducing points → tighter ELBO (closer to exact GP)
        let n = 60;
        let (xs, ys) = sin_data_unit(n);
        let base_config = SparseGpConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.15,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            n_inducing: 0, // overridden below
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        };

        let config_few = SparseGpConfig {
            n_inducing: 5,
            ..base_config.clone()
        };
        let config_many = SparseGpConfig {
            n_inducing: 20,
            ..base_config
        };

        let fit_few =
            sparse_gp_fit(&xs, &ys, n, 1, &config_few).expect("sparse_gp_fit should succeed");
        let fit_many =
            sparse_gp_fit(&xs, &ys, n, 1, &config_many).expect("sparse_gp_fit should succeed");

        let elbo_few = sparse_gp_elbo(&fit_few);
        let elbo_many = sparse_gp_elbo(&fit_many);

        // More inducing points should give higher (less negative) ELBO
        assert!(
            elbo_many >= elbo_few - 1e-6,
            "elbo_many={elbo_many:.4} should be >= elbo_few={elbo_few:.4}"
        );
    }

    #[test]
    fn sparse_gp_means_finite_across_kernels() {
        let (xs, ys) = sin_data_unit(20);
        let x_test: Vec<f64> = vec![0.25, 0.5, 0.75];

        let kernels = vec![
            GprKernel::Rbf {
                length_scale: 0.3,
                signal_variance: 1.0,
            },
            GprKernel::Matern32 {
                length_scale: 0.3,
                signal_variance: 1.0,
            },
            GprKernel::Matern52 {
                length_scale: 0.3,
                signal_variance: 1.0,
            },
        ];

        for kernel in kernels {
            let config = SparseGpConfig {
                kernel,
                noise_variance: 1e-2,
                n_inducing: 6,
                jitter: 1e-6,
                normalize_y: false,
                inducing_init: InducingInit::FirstN,
            };
            let fit =
                sparse_gp_fit(&xs, &ys, 20, 1, &config).expect("sparse_gp_fit should succeed");
            let (means, _) = sparse_gp_predict(&fit, &x_test, 3, false)
                .expect("sparse_gp_predict should succeed");
            for m in &means {
                assert!(m.is_finite(), "mean must be finite: {m}");
            }
        }
    }

    #[test]
    fn sparse_gp_2d_input_works() {
        let n = 25;
        let d = 2;
        let xs: Vec<f64> = (0..n)
            .flat_map(|i| {
                let x1 = i as f64 / n as f64;
                let x2 = (i as f64 * 0.37) % 1.0;
                vec![x1, x2]
            })
            .collect();
        let ys: Vec<f64> = (0..n).map(|i| xs[i * d] + xs[i * d + 1]).collect();

        let config = SparseGpConfig {
            kernel: GprKernel::Rbf {
                length_scale: 0.5,
                signal_variance: 1.0,
            },
            noise_variance: 1e-2,
            n_inducing: 8,
            jitter: 1e-6,
            normalize_y: false,
            inducing_init: InducingInit::FirstN,
        };
        let fit = sparse_gp_fit(&xs, &ys, n, d, &config).expect("sparse_gp_fit should succeed");
        let x_test = vec![0.5, 0.3, 0.8, 0.2];
        let (means, stds) =
            sparse_gp_predict(&fit, &x_test, 2, true).expect("sparse_gp_predict should succeed");
        for m in &means {
            assert!(m.is_finite());
        }
        for s in stds.expect("stds should be present") {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn sparse_gp_inducing_z_has_correct_shape() {
        let n = 20;
        let d = 1;
        let (xs, ys) = sin_data_unit(n);
        let config = default_sparse_config(7);
        let fit = sparse_gp_fit(&xs, &ys, n, d, &config).expect("sparse_gp_fit should succeed");
        assert_eq!(fit.inducing_z.len(), fit.n_inducing * d);
    }
}
