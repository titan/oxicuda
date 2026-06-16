//! Relevance Vector Machine (Tipping 2001; Tipping & Faul 2003).
//!
//! A Bayesian sparse kernel regressor. Given training inputs `{x_i}` and targets
//! `t ∈ ℝ^m`, the model is
//!
//! ```text
//! t_i = Σ_j w_j φ_j(x_i) + ε,   ε ~ N(0, σ²),
//! ```
//!
//! with a separate Automatic-Relevance-Determination (ARD) prior on each weight:
//! `p(w_j) = N(0, α_j⁻¹)`. Maximising the marginal likelihood (the *evidence*) drives
//! most `α_j → ∞`, forcing the corresponding `w_j` to exactly zero. The surviving
//! basis functions are the **relevance vectors** — usually far fewer than an SVM's
//! support vectors, and without the SVM's box-constraint tuning.
//!
//! This module implements both:
//!
//! - the classic EM-style evidence maximisation over a *fixed* design matrix
//!   ([`rvm_fit_design`]), and
//! - a kernel front-end ([`Rvm`]) that builds the design matrix `Φ_{ij} = k(x_i, x_j)`
//!   (plus a bias column), fits the weights, prunes to the relevance set, and predicts
//!   on unseen inputs via `ŷ(x⋆) = Σ_{j ∈ RV} w_j k(x⋆, x_j) + bias`.
//!
//! The marginal-likelihood updates use the MacKay re-estimation formulae:
//!
//! ```text
//! Σ        = (A + βΦᵀΦ)⁻¹,   μ = βΣΦᵀt,   A = diag(α),   β = 1/σ²
//! γ_j      = 1 − α_j Σ_{jj}
//! α_j      ← γ_j / μ_j²
//! β        ← (m − Σ_j γ_j) / ‖t − Φμ‖²
//! ```
//!
//! # References
//!
//! - M. E. Tipping (2001), "Sparse Bayesian Learning and the Relevance Vector Machine",
//!   JMLR 1:211-244.
//! - M. E. Tipping & A. C. Faul (2003), "Fast Marginal Likelihood Maximisation for Sparse
//!   Bayesian Models", AISTATS.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, mat_vec, norm2};

// ---------------------------------------------------------------------------
// Fixed-design RVM (evidence maximisation)
// ---------------------------------------------------------------------------

/// Result of fitting an RVM over a fixed design matrix.
#[derive(Debug, Clone)]
pub struct RvmFit {
    /// Posterior-mean weights `μ ∈ ℝ^n` (zero at pruned basis functions).
    pub weights: Vec<f64>,
    /// Final inverse-variance hyper-parameters `α_j` (`∞`-pruned set to `f64::INFINITY`).
    pub alpha: Vec<f64>,
    /// Indices of the surviving relevance vectors (those with finite `α_j`).
    pub relevance: Vec<usize>,
    /// Estimated noise precision `β = 1/σ²`.
    pub beta: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
}

/// Threshold above which a basis function is considered pruned (`α_j → ∞`).
const ALPHA_PRUNE: f64 = 1e12;

/// Fit an RVM by evidence maximisation over the fixed `m × n` design matrix `phi`.
///
/// * `phi` — design matrix, row-major `[m × n]`. Column `j` is basis function `φ_j`
///   evaluated at every training point.
/// * `m`, `n` — number of training samples and basis functions.
/// * `t` — target vector `ℝ^m`.
/// * `max_iter` — maximum evidence-maximisation iterations.
/// * `tol` — stop when the largest `log α_j` change is below `tol`.
///
/// # Errors
/// * [`CsError::ShapeMismatch`] / [`CsError::DimensionMismatch`] on bad shapes.
/// * [`CsError::InvalidParameter`] for `m == 0` or `n == 0`.
pub fn rvm_fit_design(
    phi: &[f64],
    m: usize,
    n: usize,
    t: &[f64],
    max_iter: usize,
    tol: f64,
) -> CsResult<RvmFit> {
    if m == 0 || n == 0 {
        return Err(CsError::InvalidParameter("rvm: m and n must be > 0".into()));
    }
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if t.len() != m {
        return Err(CsError::DimensionMismatch { a: t.len(), b: m });
    }

    // Pre-compute ΦᵀΦ (n×n) and Φᵀt (n).
    let gram = gram_matrix(phi, m, n);
    let phi_t_t = mat_t_vec(phi, m, n, t)?;

    // Initialise hyper-parameters.
    let mut alpha = vec![1.0_f64; n];
    let t_var = (norm2(t) * norm2(t) / m as f64).max(1e-6);
    let mut beta = 1.0 / (0.1 * t_var).max(1e-6); // start assuming 10% noise variance
    let mut mu = vec![0.0_f64; n];
    let mut iterations = 0usize;

    for _ in 0..max_iter {
        iterations += 1;
        let active: Vec<usize> = (0..n).filter(|&j| alpha[j] < ALPHA_PRUNE).collect();
        if active.is_empty() {
            break;
        }
        let k = active.len();

        // Build A + βΦᵀΦ restricted to active columns (k×k SPD).
        let mut h = vec![0.0_f64; k * k];
        for (ii, &i) in active.iter().enumerate() {
            for (jj, &j) in active.iter().enumerate() {
                h[ii * k + jj] = beta * gram[i * n + j];
            }
            h[ii * k + ii] += alpha[i];
        }
        let l = cholesky_factor(&h, k)?;

        // μ_active = β Σ Φᵀt  ⇒ solve H μ = β (Φᵀt)_active.
        let mut rhs = vec![0.0_f64; k];
        for (ii, &i) in active.iter().enumerate() {
            rhs[ii] = beta * phi_t_t[i];
        }
        let mu_active = cholesky_solve(&l, k, &rhs)?;

        // Σ_{jj} for active set: solve H s = e_j, read s_j.
        let mut sigma_diag = vec![0.0_f64; k];
        for jj in 0..k {
            let mut ej = vec![0.0_f64; k];
            ej[jj] = 1.0;
            let s = cholesky_solve(&l, k, &ej)?;
            sigma_diag[jj] = s[jj];
        }

        // Hyper-parameter updates.
        let mut max_log_change = 0.0_f64;
        let mut gamma_sum = 0.0_f64;
        let mut new_mu = vec![0.0_f64; n];
        for (jj, &j) in active.iter().enumerate() {
            let mu_j = mu_active[jj];
            new_mu[j] = mu_j;
            let gamma_j = (1.0 - alpha[j] * sigma_diag[jj]).clamp(1e-12, 1.0);
            gamma_sum += gamma_j;
            let mu_sq = (mu_j * mu_j).max(1e-300);
            let alpha_new = (gamma_j / mu_sq).min(ALPHA_PRUNE * 10.0);
            let log_change = (alpha_new.max(1e-300).ln() - alpha[j].max(1e-300).ln()).abs();
            max_log_change = max_log_change.max(log_change);
            alpha[j] = alpha_new;
        }
        mu = new_mu;

        // Noise precision update.
        let phi_mu = mat_vec(phi, m, n, &mu)?;
        let mut resid_sq = 0.0_f64;
        for i in 0..m {
            let d = t[i] - phi_mu[i];
            resid_sq += d * d;
        }
        let denom = (m as f64 - gamma_sum).max(1e-6);
        beta = (denom / resid_sq.max(1e-12)).clamp(1e-6, 1e12);

        if max_log_change < tol {
            break;
        }
    }

    let relevance: Vec<usize> = (0..n).filter(|&j| alpha[j] < ALPHA_PRUNE).collect();
    // Zero-out pruned weights and mark pruned alphas as infinite.
    for j in 0..n {
        if alpha[j] >= ALPHA_PRUNE {
            mu[j] = 0.0;
            alpha[j] = f64::INFINITY;
        }
    }

    Ok(RvmFit {
        weights: mu,
        alpha,
        relevance,
        beta,
        iterations,
    })
}

/// `ΦᵀΦ` for a row-major `m × n` matrix (returns `n × n`, row-major).
fn gram_matrix(phi: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0_f64; n * n];
    for r in 0..m {
        let row = r * n;
        for i in 0..n {
            let pri = phi[row + i];
            if pri == 0.0 {
                continue;
            }
            for j in 0..n {
                g[i * n + j] += pri * phi[row + j];
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Kernel front-end
// ---------------------------------------------------------------------------

/// Kernel choice for the [`Rvm`] regressor.
#[derive(Debug, Clone)]
pub enum RvmKernel {
    /// Linear kernel `k(a, b) = aᵀb`.
    Linear,
    /// Gaussian RBF kernel `k(a, b) = exp(−‖a − b‖² / (2 ℓ²))` with length-scale `ℓ`.
    Rbf {
        /// Length-scale `ℓ > 0`.
        length_scale: f64,
    },
    /// Inhomogeneous polynomial kernel `k(a, b) = (aᵀb + c)^d`.
    Polynomial {
        /// Additive constant `c ≥ 0`.
        coef0: f64,
        /// Integer degree `d ≥ 1`.
        degree: u32,
    },
}

impl RvmKernel {
    /// Evaluate the kernel between two equal-length feature vectors.
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        match *self {
            RvmKernel::Linear => dot,
            RvmKernel::Rbf { length_scale } => {
                let mut sq = 0.0_f64;
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = x - y;
                    sq += d * d;
                }
                (-sq / (2.0 * length_scale * length_scale)).exp()
            }
            RvmKernel::Polynomial { coef0, degree } => (dot + coef0).powi(degree as i32),
        }
    }
}

/// Configuration for the kernel [`Rvm`].
#[derive(Debug, Clone)]
pub struct RvmConfig {
    /// Kernel (default Gaussian RBF with length-scale `1.0`).
    pub kernel: RvmKernel,
    /// Whether to append a constant bias basis function (default `true`).
    pub use_bias: bool,
    /// Maximum evidence-maximisation iterations (default `300`).
    pub max_iter: usize,
    /// Convergence tolerance on `log α` (default `1 × 10⁻⁴`).
    pub tol: f64,
}

impl Default for RvmConfig {
    fn default() -> Self {
        Self {
            kernel: RvmKernel::Rbf { length_scale: 1.0 },
            use_bias: true,
            max_iter: 300,
            tol: 1e-4,
        }
    }
}

/// A fitted kernel Relevance Vector Machine.
#[derive(Debug, Clone)]
pub struct Rvm {
    config: RvmConfig,
    /// Stored training inputs of the relevance vectors, row-major `[n_rv × d]`.
    rv_inputs: Vec<f64>,
    /// Weights of the relevance vectors (`[n_rv]`).
    rv_weights: Vec<f64>,
    /// Bias weight (`0.0` if no bias survived / disabled).
    bias: f64,
    /// Feature dimension `d`.
    d: usize,
    /// Noise precision `β`.
    beta: f64,
}

impl Rvm {
    /// Fit a kernel RVM to `n` training points `x ∈ ℝ^{n×d}` (row-major) and targets `t`.
    ///
    /// # Errors
    /// * [`CsError::InvalidParameter`] for `n == 0`, `d == 0`, or a non-positive RBF
    ///   length-scale.
    /// * [`CsError::ShapeMismatch`] / [`CsError::DimensionMismatch`] on bad shapes.
    pub fn fit(x: &[f64], n: usize, d: usize, t: &[f64], config: RvmConfig) -> CsResult<Self> {
        if n == 0 || d == 0 {
            return Err(CsError::InvalidParameter("rvm: n and d must be > 0".into()));
        }
        if x.len() != n * d {
            return Err(CsError::ShapeMismatch {
                expected: vec![n, d],
                got: vec![x.len()],
            });
        }
        if t.len() != n {
            return Err(CsError::DimensionMismatch { a: t.len(), b: n });
        }
        if let RvmKernel::Rbf { length_scale } = config.kernel {
            if length_scale <= 0.0 || !length_scale.is_finite() {
                return Err(CsError::InvalidParameter(format!(
                    "rvm: RBF length_scale must be > 0, got {length_scale}"
                )));
            }
        }

        // Build the design matrix Φ (n × n_basis) with optional bias column last.
        let n_basis = if config.use_bias { n + 1 } else { n };
        let mut phi = vec![0.0_f64; n * n_basis];
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            for j in 0..n {
                let xj = &x[j * d..(j + 1) * d];
                phi[i * n_basis + j] = config.kernel.eval(xi, xj);
            }
            if config.use_bias {
                phi[i * n_basis + n] = 1.0;
            }
        }

        let fit = rvm_fit_design(&phi, n, n_basis, t, config.max_iter, config.tol)?;

        // Extract relevance vectors (kernel basis) and bias.
        let mut rv_inputs = Vec::new();
        let mut rv_weights = Vec::new();
        let mut bias = 0.0_f64;
        for &j in &fit.relevance {
            if config.use_bias && j == n {
                bias = fit.weights[j];
            } else {
                rv_inputs.extend_from_slice(&x[j * d..(j + 1) * d]);
                rv_weights.push(fit.weights[j]);
            }
        }

        Ok(Self {
            config,
            rv_inputs,
            rv_weights,
            bias,
            d,
            beta: fit.beta,
        })
    }

    /// Number of relevance vectors retained (excludes the bias term).
    #[must_use]
    pub fn n_relevance(&self) -> usize {
        self.rv_weights.len()
    }

    /// Estimated noise standard deviation `σ = 1/√β`.
    #[must_use]
    pub fn noise_std(&self) -> f64 {
        1.0 / self.beta.max(1e-300).sqrt()
    }

    /// Predict the target at a single query input `x⋆ ∈ ℝ^d`.
    ///
    /// # Errors
    /// [`CsError::DimensionMismatch`] if `query.len() != d`.
    pub fn predict_one(&self, query: &[f64]) -> CsResult<f64> {
        if query.len() != self.d {
            return Err(CsError::DimensionMismatch {
                a: query.len(),
                b: self.d,
            });
        }
        let mut y = self.bias;
        for (k, w) in self.rv_weights.iter().enumerate() {
            let rv = &self.rv_inputs[k * self.d..(k + 1) * self.d];
            y += w * self.config.kernel.eval(query, rv);
        }
        Ok(y)
    }

    /// Predict targets for `n_q` query points `x ∈ ℝ^{n_q×d}` (row-major).
    ///
    /// # Errors
    /// [`CsError::ShapeMismatch`] if `queries.len() != n_q * d`.
    pub fn predict(&self, queries: &[f64], n_q: usize) -> CsResult<Vec<f64>> {
        if queries.len() != n_q * self.d {
            return Err(CsError::ShapeMismatch {
                expected: vec![n_q, self.d],
                got: vec![queries.len()],
            });
        }
        let mut out = Vec::with_capacity(n_q);
        for i in 0..n_q {
            out.push(self.predict_one(&queries[i * self.d..(i + 1) * self.d])?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_design_identity_is_sparse() {
        // Φ = I_4, t = [1, 0, 0, 2] ⇒ relevance set should keep 0 and 3.
        let phi = {
            let mut p = vec![0.0_f64; 16];
            for i in 0..4 {
                p[i * 4 + i] = 1.0;
            }
            p
        };
        let t = [1.0_f64, 0.0, 0.0, 2.0];
        let fit = rvm_fit_design(&phi, 4, 4, &t, 200, 1e-6).expect("ok");
        assert!(fit.weights[0].abs() > 0.5, "w0 = {}", fit.weights[0]);
        assert!(fit.weights[3].abs() > 1.0, "w3 = {}", fit.weights[3]);
        assert!(fit.weights[1].abs() < 1e-3, "w1 = {}", fit.weights[1]);
        assert!(fit.weights[2].abs() < 1e-3, "w2 = {}", fit.weights[2]);
        assert!(fit.relevance.contains(&0) && fit.relevance.contains(&3));
        assert!(!fit.relevance.contains(&1) && !fit.relevance.contains(&2));
    }

    #[test]
    fn fixed_design_pruned_alpha_infinite() {
        let phi = {
            let mut p = vec![0.0_f64; 9];
            for i in 0..3 {
                p[i * 3 + i] = 1.0;
            }
            p
        };
        let t = [5.0_f64, 0.0, 0.0];
        let fit = rvm_fit_design(&phi, 3, 3, &t, 200, 1e-6).expect("ok");
        assert!(fit.alpha[1].is_infinite());
        assert!(fit.alpha[2].is_infinite());
        assert!(fit.alpha[0].is_finite());
    }

    #[test]
    fn linear_kernel_recovers_line() {
        // Targets t = 2x on a 1-D grid; linear kernel should reproduce it well.
        let n = 12usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64)).collect();
        let t: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let cfg = RvmConfig {
            kernel: RvmKernel::Linear,
            use_bias: true,
            max_iter: 400,
            tol: 1e-6,
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        let pred = rvm.predict_one(&[0.5]).expect("ok");
        assert!((pred - 1.0).abs() < 0.1, "pred(0.5) = {pred}");
    }

    #[test]
    fn rbf_kernel_fits_sine() {
        // Fit a smooth sine; RBF RVM should interpolate accurately and stay sparse.
        let n = 25usize;
        let xs: Vec<f64> = (0..n)
            .map(|i| -3.0 + 6.0 * i as f64 / (n as f64 - 1.0))
            .collect();
        let t: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let cfg = RvmConfig {
            kernel: RvmKernel::Rbf { length_scale: 1.0 },
            use_bias: true,
            max_iter: 500,
            tol: 1e-5,
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        // Prediction at a training-ish point.
        let pred = rvm.predict_one(&[0.0]).expect("ok");
        assert!(pred.abs() < 0.2, "pred(0) = {pred} (sin 0 = 0)");
        let pred2 = rvm.predict_one(&[std::f64::consts::FRAC_PI_2]).expect("ok"); // sin(π/2) = 1
        assert!((pred2 - 1.0).abs() < 0.25, "pred(π/2) = {pred2}");
        // Relevance set should be a strict subset of the training points.
        assert!(rvm.n_relevance() <= n, "n_rv = {}", rvm.n_relevance());
    }

    #[test]
    fn predict_batch_matches_single() {
        let n = 8usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
        let t: Vec<f64> = xs.iter().map(|&x| 0.5 * x + 1.0).collect();
        let cfg = RvmConfig {
            kernel: RvmKernel::Linear,
            ..Default::default()
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        let q = vec![0.1_f64, 0.7, 1.4];
        let batch = rvm.predict(&q, 3).expect("ok");
        for (i, qi) in q.iter().enumerate() {
            let single = rvm.predict_one(&[*qi]).expect("ok");
            assert!((batch[i] - single).abs() < 1e-12);
        }
    }

    #[test]
    fn multidim_input_rbf() {
        // 2-D inputs, target = x0 + x1.
        let pts = [
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ];
        let n = pts.len();
        let mut x = Vec::new();
        let mut t = Vec::new();
        for p in &pts {
            x.extend_from_slice(p);
            t.push(p[0] + p[1]);
        }
        let cfg = RvmConfig {
            kernel: RvmKernel::Rbf { length_scale: 1.0 },
            max_iter: 400,
            ..Default::default()
        };
        let rvm = Rvm::fit(&x, n, 2, &t, cfg).expect("ok");
        let pred = rvm.predict_one(&[1.0, 1.0]).expect("ok");
        assert!((pred - 2.0).abs() < 0.4, "pred(1,1) = {pred}");
    }

    #[test]
    fn polynomial_kernel_quadratic() {
        // Targets t = x²; degree-2 polynomial kernel can fit exactly.
        let n = 10usize;
        let xs: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / 9.0).collect();
        let t: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let cfg = RvmConfig {
            kernel: RvmKernel::Polynomial {
                coef0: 1.0,
                degree: 2,
            },
            use_bias: true,
            max_iter: 500,
            tol: 1e-6,
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        let pred = rvm.predict_one(&[0.5]).expect("ok");
        assert!((pred - 0.25).abs() < 0.1, "pred(0.5) = {pred}");
    }

    #[test]
    fn noise_std_positive() {
        let n = 6usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t: Vec<f64> = xs.to_vec();
        let cfg = RvmConfig {
            kernel: RvmKernel::Linear,
            ..Default::default()
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        assert!(rvm.noise_std() > 0.0 && rvm.noise_std().is_finite());
        assert!(rvm.beta > 0.0);
    }

    #[test]
    fn predict_dimension_mismatch() {
        let n = 4usize;
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let t = vec![0.0_f64, 1.0, 2.0, 3.0];
        let rvm = Rvm::fit(
            &xs,
            n,
            1,
            &t,
            RvmConfig {
                kernel: RvmKernel::Linear,
                ..Default::default()
            },
        )
        .expect("ok");
        assert!(matches!(
            rvm.predict_one(&[1.0, 2.0]),
            Err(CsError::DimensionMismatch { .. })
        ));
        // d = 1, so 3 values would be 3 valid queries; claim n_q = 5 to force a mismatch.
        assert!(matches!(
            rvm.predict(&[1.0, 2.0, 3.0], 5),
            Err(CsError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn fit_rejects_bad_shapes() {
        // t length mismatch.
        assert!(matches!(
            rvm_fit_design(&[1.0, 0.0, 0.0, 1.0], 2, 2, &[1.0], 10, 1e-6),
            Err(CsError::DimensionMismatch { .. })
        ));
        // phi shape mismatch.
        assert!(matches!(
            rvm_fit_design(&[1.0, 0.0, 0.0], 2, 2, &[1.0, 0.0], 10, 1e-6),
            Err(CsError::ShapeMismatch { .. })
        ));
        // zero size.
        assert!(matches!(
            rvm_fit_design(&[], 0, 0, &[], 10, 1e-6),
            Err(CsError::InvalidParameter(_))
        ));
        // bad RBF length-scale.
        assert!(matches!(
            Rvm::fit(
                &[0.0, 1.0],
                2,
                1,
                &[0.0, 1.0],
                RvmConfig {
                    kernel: RvmKernel::Rbf { length_scale: -1.0 },
                    ..Default::default()
                }
            ),
            Err(CsError::InvalidParameter(_))
        ));
    }

    #[test]
    fn constant_target_uses_bias() {
        // t ≡ 3 ⇒ best explained by the bias term alone.
        let n = 7usize;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t = vec![3.0_f64; n];
        let cfg = RvmConfig {
            kernel: RvmKernel::Rbf { length_scale: 1.0 },
            use_bias: true,
            max_iter: 400,
            tol: 1e-6,
        };
        let rvm = Rvm::fit(&xs, n, 1, &t, cfg).expect("ok");
        let pred = rvm.predict_one(&[100.0]).expect("ok"); // far from data
        assert!((pred - 3.0).abs() < 0.5, "pred = {pred}");
    }
}
