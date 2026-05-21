//! TMLE — Targeted Maximum Likelihood Estimation of the Average Treatment Effect.
//!
//! Reference: van der Laan, M. J. & Rubin, D. (2006). "Targeted Maximum
//! Likelihood Learning." *International Journal of Biostatistics*, 2(1):11.
//! See also Gruber, S. & van der Laan, M. J. (2010). "A targeted maximum
//! likelihood estimator of a causal effect on a bounded continuous outcome."
//! *International Journal of Biostatistics*, 6(1):26.
//!
//! # Algorithm
//!
//! TMLE is a semiparametric, doubly robust plug-in estimator that combines an
//! initial outcome model `Q̂⁰(t, x) = E[Y | T = t, X = x]` and a propensity
//! score `ĝ(x) = P(T = 1 | X = x)` via a one-step *targeting* update that
//! solves the efficient-influence-function (EIF) estimating equation exactly.
//! The result is an asymptotically efficient estimator that retains valid
//! standard errors derived from the influence curve.
//!
//! ## Continuous-outcome TMLE (OLS targeting)
//!
//! 1. **Initial outcome model** — K-fold cross-fit ridge regression of `Y` on
//!    the full-interaction design `Z = [1, T, X, T·X]`, producing
//!    `Q̂⁰(T_i, X_i)` for each held-out sample.
//! 2. **Propensity** — K-fold cross-fit logistic regression of `T` on
//!    `[1, X]`, fitted by gradient descent with intercept, producing `ĝ(x_i)`.
//!    Propensities are clipped to `[clip_eps, 1 − clip_eps]` to keep the
//!    clever covariate finite.
//! 3. **Clever covariate** — `H(T, X) = T/ĝ(X) − (1 − T)/(1 − ĝ(X))`.
//! 4. **Targeting** — solve the univariate OLS
//!    `Y − Q̂⁰(T, X) = ε · H(T, X) + residual` and update
//!    `Q̂¹(t, x) = Q̂⁰(t, x) + ε · H(t, x)`.  Iterated until convergence;
//!    a single step suffices for OLS-targeting in theory but we retain the
//!    loop to absorb tiny numerical perturbations.
//! 5. **ATE plug-in** — `ψ̂ = (1/n) Σ_i [Q̂¹(1, X_i) − Q̂¹(0, X_i)]`.
//! 6. **Influence-curve SE** —
//!    ```text
//!      IC_i = H(T_i, X_i) · (Y_i − Q̂¹(T_i, X_i))
//!             + Q̂¹(1, X_i) − Q̂¹(0, X_i) − ψ̂
//!      Var(ψ̂) ≈ Var(IC)/n
//!    ```
//!
//! Cross-fitting (Chernozhukov et al. 2018) removes the over-fitting bias of
//! the nuisance functions; the deterministic fold assignment
//! `fold_id[i] = i % n_folds` keeps the estimator reproducible without any
//! random number generation.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`Tmle::estimate`].
#[derive(Debug, Clone)]
pub struct TmleConfig {
    /// Number of cross-fitting folds. Must satisfy `n_folds ≥ 2`.
    pub n_folds: usize,
    /// Ridge penalty for the outcome and propensity normal equations.
    /// Must be `≥ 0`.
    pub ridge_lambda: f64,
    /// Propensity clipping epsilon — propensities are constrained to
    /// `[clip_eps, 1 − clip_eps]`.  Must satisfy `0 < clip_eps < 0.5`.
    pub clip_eps: f64,
    /// Maximum number of outer targeting iterations.  Must be `≥ 1`.
    pub max_outer_iters: usize,
    /// Convergence tolerance on the targeting update `|ε|`.  Must be `> 0`.
    pub tol: f64,
}

impl Default for TmleConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            ridge_lambda: 1e-3,
            clip_eps: 0.025,
            max_outer_iters: 10,
            tol: 1e-8,
        }
    }
}

/// Result of [`Tmle::estimate`].
#[derive(Debug, Clone)]
pub struct TmleResult {
    /// Targeted ATE plug-in estimate `ψ̂`.
    pub ate: f64,
    /// Influence-curve standard error of `ψ̂`: `sqrt(Var(IC)/n)`.
    pub se: f64,
    /// Sample variance of the influence-curve values `IC_i`.
    pub ic_var: f64,
    /// Number of samples used.
    pub n: usize,
}

/// Zero-sized handle exposing the [`Tmle::estimate`] entry point.
pub struct Tmle;

impl Tmle {
    /// Estimate the ATE by cross-fit TMLE.
    ///
    /// # Parameters
    /// - `y`: continuous outcomes, length `n`.
    /// - `t`: binary treatment indicators (each in `{0, 1}`), length `n`.
    /// - `x`: covariates as a slice of length-`d` row vectors
    ///   (`x[i].len() == d` for every `i`).
    /// - `cfg`: see [`TmleConfig`].
    ///
    /// # Errors
    /// - [`CausalError::InvalidNumFolds`] if `cfg.n_folds < 2`.
    /// - [`CausalError::IncompatibleData`] for `ridge_lambda < 0`,
    ///   `clip_eps ∉ (0, 0.5)`, `tol ≤ 0`, `max_outer_iters == 0`, or any
    ///   `t[i]` outside `{0, 1}`.
    /// - [`CausalError::DimensionMismatch`] for empty data or length /
    ///   row-width mismatches between `y`, `t`, and `x`.
    /// - [`CausalError::MatrixSingular`] if any normal-equation solve fails.
    pub fn estimate(
        y: &[f64],
        t: &[u32],
        x: &[Vec<f64>],
        cfg: &TmleConfig,
    ) -> CausalResult<TmleResult> {
        // ---- config validation -----------------------------------------
        if cfg.n_folds < 2 {
            return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
        }
        if cfg.ridge_lambda < 0.0
            || !(cfg.clip_eps > 0.0 && cfg.clip_eps < 0.5)
            || cfg.tol <= 0.0
            || cfg.max_outer_iters == 0
        {
            return Err(CausalError::IncompatibleData);
        }

        // ---- shape validation ------------------------------------------
        let n = y.len();
        if n == 0 || t.is_empty() || x.is_empty() {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: 0,
            });
        }
        if t.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: t.len(),
            });
        }
        if x.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: x.len(),
            });
        }
        let d = x[0].len();
        if d == 0 {
            return Err(CausalError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        for row in x.iter() {
            if row.len() != d {
                return Err(CausalError::DimensionMismatch {
                    expected: d,
                    got: row.len(),
                });
            }
        }
        for &ti in t {
            if !(ti == 0 || ti == 1) {
                return Err(CausalError::IncompatibleData);
            }
        }
        if n < cfg.n_folds * (2 * d + 3) {
            return Err(CausalError::InvalidNumFolds { k: cfg.n_folds });
        }

        let k = cfg.n_folds;
        let t_f: Vec<f64> = t.iter().map(|&v| v as f64).collect();

        // ---- cross-fit outcome regression Q̂⁰(T, X) -------------------
        // p_q = 1 (intercept) + 1 (T) + d (X) + d (T·X) = 2d + 2
        let p_q = 2 * d + 2;
        let mut q0_obs = vec![0.0_f64; n]; // Q̂⁰(T_i, X_i)
        let mut q0_t1 = vec![0.0_f64; n]; // Q̂⁰(1, X_i)
        let mut q0_t0 = vec![0.0_f64; n]; // Q̂⁰(0, X_i)

        // p_g = 1 (intercept) + d (X)
        let p_g = d + 1;
        let mut g_hat = vec![0.0_f64; n];

        for fold in 0..k {
            // Train/test split by index mod k.
            let mut n_train = 0;
            for i in 0..n {
                if i % k != fold {
                    n_train += 1;
                }
            }
            if n_train < p_q || n_train < p_g {
                return Err(CausalError::InvalidNumFolds { k });
            }

            // --- outcome design Z = [1, T, X, T·X] over training rows ---
            let mut z_train = vec![0.0_f64; n_train * p_q];
            let mut y_train = vec![0.0_f64; n_train];
            let mut t_train = vec![0.0_f64; n_train];
            let mut x_train = vec![0.0_f64; n_train * p_g];
            let mut row = 0;
            for i in 0..n {
                if i % k == fold {
                    continue;
                }
                z_train[row * p_q] = 1.0;
                z_train[row * p_q + 1] = t_f[i];
                for j in 0..d {
                    z_train[row * p_q + 2 + j] = x[i][j];
                    z_train[row * p_q + 2 + d + j] = t_f[i] * x[i][j];
                }
                y_train[row] = y[i];
                t_train[row] = t_f[i];
                x_train[row * p_g] = 1.0;
                for j in 0..d {
                    x_train[row * p_g + 1 + j] = x[i][j];
                }
                row += 1;
            }

            let beta_q = ridge_solve(&z_train, &y_train, n_train, p_q, cfg.ridge_lambda)?;
            let beta_g =
                logistic_fit(&x_train, &t_train, n_train, p_g, cfg.ridge_lambda, 200, 0.5)?;

            // Predict on test indices of this fold.
            for i in 0..n {
                if i % k != fold {
                    continue;
                }
                // Q̂⁰ at (T_i, X_i), (1, X_i), (0, X_i)
                let mut q_obs = beta_q[0] + beta_q[1] * t_f[i];
                let mut q_t1 = beta_q[0] + beta_q[1];
                let mut q_t0 = beta_q[0];
                for j in 0..d {
                    let xij = x[i][j];
                    q_obs += beta_q[2 + j] * xij + beta_q[2 + d + j] * t_f[i] * xij;
                    q_t1 += beta_q[2 + j] * xij + beta_q[2 + d + j] * xij;
                    q_t0 += beta_q[2 + j] * xij;
                }
                q0_obs[i] = q_obs;
                q0_t1[i] = q_t1;
                q0_t0[i] = q_t0;

                // ĝ(x_i)
                let mut z = beta_g[0];
                for j in 0..d {
                    z += beta_g[1 + j] * x[i][j];
                }
                g_hat[i] = sigmoid(z).clamp(cfg.clip_eps, 1.0 - cfg.clip_eps);
            }
        }

        // ---- clever covariate H(T_i, X_i) ------------------------------
        let mut h_obs = vec![0.0_f64; n];
        let mut h_t1 = vec![0.0_f64; n];
        let mut h_t0 = vec![0.0_f64; n];
        for i in 0..n {
            let g = g_hat[i];
            let one_minus_g = 1.0 - g;
            h_t1[i] = 1.0 / g;
            h_t0[i] = -1.0 / one_minus_g;
            h_obs[i] = if t_f[i] == 1.0 { h_t1[i] } else { h_t0[i] };
        }

        // ---- targeting: iterate ε until |ε| < tol ----------------------
        let mut q1_obs = q0_obs.clone();
        let mut q1_t1 = q0_t1.clone();
        let mut q1_t0 = q0_t0.clone();
        let mut total_eps = 0.0_f64;

        for _ in 0..cfg.max_outer_iters {
            // ε = Σ H_i · (Y_i − Q̂_i) / Σ H_i²
            let mut num = 0.0_f64;
            let mut denom = 0.0_f64;
            for i in 0..n {
                let resid = y[i] - q1_obs[i];
                num += h_obs[i] * resid;
                denom += h_obs[i] * h_obs[i];
            }
            if denom <= 0.0 {
                return Err(CausalError::MatrixSingular);
            }
            let eps = num / denom;
            total_eps += eps;
            for i in 0..n {
                q1_obs[i] += eps * h_obs[i];
                q1_t1[i] += eps * h_t1[i];
                q1_t0[i] += eps * h_t0[i];
            }
            if eps.abs() < cfg.tol {
                break;
            }
        }
        // total_eps is kept for diagnostic computation – ensure it stays finite.
        if !total_eps.is_finite() {
            return Err(CausalError::MatrixSingular);
        }

        // ---- ATE plug-in -----------------------------------------------
        let psi = (0..n).map(|i| q1_t1[i] - q1_t0[i]).sum::<f64>() / n as f64;

        // ---- influence-curve SE ----------------------------------------
        let mut ic = vec![0.0_f64; n];
        for i in 0..n {
            ic[i] = h_obs[i] * (y[i] - q1_obs[i]) + q1_t1[i] - q1_t0[i] - psi;
        }
        let mean_ic = ic.iter().sum::<f64>() / n as f64;
        let var_ic = ic
            .iter()
            .map(|v| (v - mean_ic) * (v - mean_ic))
            .sum::<f64>()
            / n as f64;
        let se = (var_ic / n as f64).sqrt();

        Ok(TmleResult {
            ate: psi,
            se,
            ic_var: var_ic,
            n,
        })
    }
}

// =====================================================================
// helpers
// =====================================================================

#[inline]
pub(super) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Solve `(X^T X + λ I) β = X^T y` via Gauss-Jordan with partial pivoting.
fn ridge_solve(
    x_mat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
) -> CausalResult<Vec<f64>> {
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];
    for row in 0..n {
        for i in 0..p {
            let xri = x_mat[row * p + i];
            for j in 0..p {
                xtx[i * p + j] += xri * x_mat[row * p + j];
            }
            xty[i] += xri * y[row];
        }
    }
    for i in 0..p {
        xtx[i * p + i] += lambda;
    }
    gauss_jordan_solve(&xtx, &xty, p)
}

/// Solve a `p × p` linear system `A β = b` by Gauss-Jordan with partial pivoting.
fn gauss_jordan_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = p + 1;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p] = b[i];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        x[i] = m[i * cols + p];
    }
    Ok(x)
}

/// Fit a binary logistic regression `P(T = 1 | X̃) = σ(β·X̃)` on the
/// design `x_mat` (row-major `n × p`, intercept column assumed at offset 0)
/// with L2-regularised gradient descent.
fn logistic_fit(
    x_mat: &[f64],
    t: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
    n_epochs: usize,
    lr: f64,
) -> CausalResult<Vec<f64>> {
    if x_mat.len() != n * p || t.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n * p,
            got: x_mat.len(),
        });
    }
    let mut beta = vec![0.0_f64; p];
    let inv_n = 1.0 / n as f64;
    for _ in 0..n_epochs {
        let mut grad = vec![0.0_f64; p];
        for i in 0..n {
            let mut z = 0.0_f64;
            for j in 0..p {
                z += beta[j] * x_mat[i * p + j];
            }
            let pred = sigmoid(z);
            let err = pred - t[i];
            for j in 0..p {
                grad[j] += err * x_mat[i * p + j];
            }
        }
        // L2 penalty on non-intercept terms only (offset 0 is intercept by convention).
        for (j, b) in beta.iter_mut().enumerate() {
            let pen = if j == 0 { 0.0 } else { lambda * *b };
            *b -= lr * (grad[j] * inv_n + pen);
        }
    }
    Ok(beta)
}

// tests live in `tmle_tests.rs` (registered from `effect/mod.rs`).
