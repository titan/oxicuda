//! Joint longitudinal-survival model (Wulfsohn & Tsiatis 1997; Rizopoulos 2010).
//!
//! # Model specification
//!
//! ## Longitudinal sub-model (linear mixed effects)
//!
//! For subject i with longitudinal observations at times `{s_{ij}}`:
//! ```text
//! y_{ij} = α_0 + α_1 s_{ij} + b_{i0} + b_{i1} s_{ij} + ε_{ij}
//! ```
//! - Fixed effects: `α = [α_0, α_1]`
//! - Random effects: `b_i = [b_{i0}, b_{i1}]` ~ N(0, D), D 2×2 covariance
//! - Measurement error: `ε_{ij}` ~ N(0, σ²)
//!
//! True trajectory for subject i at time t: `μ_i(t) = α_0 + α_1 t + b_{i0} + b_{i1} t`
//!
//! ## Survival sub-model (Weibull baseline hazard)
//!
//! ```text
//! h_i(t) = h_0(t) * exp(γ * μ_i(t))
//! h_0(t) = (k/λ)(t/λ)^{k-1},  H_0(t) = (t/λ)^k
//! ```
//!
//! ## Estimation: EM with Laplace E-step
//!
//! Each EM iteration:
//! 1. **E-step**: for each subject find posterior mode `b̂_i` via 2D Newton-Raphson.
//! 2. **M-step**: update `(α, D, σ², γ, k, λ)` using `{b̂_i}`.

use crate::error::{SurvivalError, SurvivalResult};

// ──────────────────────────────────────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────────────────────────────────────

/// A single subject's longitudinal observations and survival outcome.
#[derive(Debug, Clone)]
pub struct JointObs {
    /// Longitudinal measurement times (sorted ascending).
    pub meas_times: Vec<f64>,
    /// Longitudinal measurements `y_{ij}` (same length as `meas_times`).
    pub measurements: Vec<f64>,
    /// Event time (or censoring time).
    pub event_time: f64,
    /// Event indicator: `true` = event occurred, `false` = right-censored.
    pub event: bool,
}

/// Configuration for the EM-Laplace joint model fitter.
#[derive(Debug, Clone)]
pub struct JointModelConfig {
    /// Maximum EM outer iterations (default 100).
    pub max_iter: usize,
    /// EM convergence tolerance on absolute log-likelihood change (default 1e-4).
    pub em_tol: f64,
    /// Maximum Newton-Raphson iterations for Laplace E-step (default 20).
    pub newton_max_iter: usize,
    /// Newton-Raphson convergence tolerance on |∇f|_∞ (default 1e-8).
    pub newton_tol: f64,
    /// L2 regularisation (ridge) added to estimated D to ensure positive definiteness
    /// (default 1e-6).
    pub re_ridge: f64,
}

impl Default for JointModelConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            em_tol: 1.0e-4,
            newton_max_iter: 20,
            newton_tol: 1.0e-8,
            re_ridge: 1.0e-6,
        }
    }
}

/// Fitted joint longitudinal-survival model.
#[derive(Debug, Clone)]
pub struct JointModelFit {
    /// Fixed effects `[α_0, α_1]` (intercept, slope).
    pub alpha: [f64; 2],
    /// Association parameter γ (how strongly the longitudinal trajectory affects hazard).
    pub gamma: f64,
    /// Measurement error variance σ².
    pub sigma_sq: f64,
    /// Random effects covariance D (row-major 2×2): `[D[0,0], D[0,1], D[1,0], D[1,1]]`.
    pub d_mat: [f64; 4],
    /// Weibull shape parameter k.
    pub weibull_shape: f64,
    /// Weibull scale parameter λ.
    pub weibull_scale: f64,
    /// Observed-data log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations consumed.
    pub n_iter: usize,
    /// Whether the EM algorithm met the convergence criterion.
    pub converged: bool,
    /// Posterior mode random effects, row-major `[n_subjects × 2]`.
    pub re_modes: Vec<f64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Small linear algebra helpers (self-contained, no external deps)
// ──────────────────────────────────────────────────────────────────────────────

/// Invert a 2×2 matrix `[[a,b],[c,d]]`.  Returns `None` if the determinant is
/// too small to be numerically reliable.
#[inline]
fn inv2x2(a: f64, b: f64, c: f64, d: f64) -> Option<[f64; 4]> {
    let det = a * d - b * c;
    if det.abs() < 1.0e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([d * inv_det, -b * inv_det, -c * inv_det, a * inv_det])
}

/// Solve a 3×3 linear system A x = b via Gaussian elimination with partial
/// pivoting.  The argument `aug` is the 3×4 augmented matrix `[A | b]`.
/// Returns `Some(x)` on success, `None` if the matrix is singular.
///
/// Index-based loops are intentional: forward elimination reads `aug[col][k]`
/// while writing `aug[row][k]`, which cannot be expressed with a single iterator.
#[allow(clippy::needless_range_loop)]
fn solve3x3(aug: &mut [[f64; 4]; 3]) -> Option<[f64; 3]> {
    const N: usize = 3;
    for col in 0..N {
        // Partial pivot: find the row with the largest absolute value in column `col`.
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..N {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        if aug[col][col].abs() < 1.0e-300 {
            return None;
        }
        let pivot = aug[col][col];
        for row in (col + 1)..N {
            let factor = aug[row][col] / pivot;
            // Must iterate with an index: need `aug[col][k]` and `aug[row][k]` simultaneously.
            for k in col..=N {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }
    // Back substitution.
    let mut x = [0.0_f64; 3];
    for row in (0..N).rev() {
        let mut s = aug[row][N]; // RHS
        for k in (row + 1)..N {
            s -= aug[row][k] * x[k];
        }
        x[row] = s / aug[row][row];
    }
    Some(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Weibull baseline hazard helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Weibull baseline log-hazard at `t`: log h_0(t) = log(k/λ) + (k-1)*log(t/λ).
#[inline]
fn weibull_log_hazard(t: f64, shape: f64, scale: f64) -> f64 {
    (shape / scale).ln() + (shape - 1.0) * (t / scale).ln()
}

/// Weibull cumulative baseline hazard: H_0(t) = (t/λ)^k.
#[inline]
fn weibull_cumhaz(t: f64, shape: f64, scale: f64) -> f64 {
    (t / scale).powf(shape)
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-posterior for a subject's random effects (used in E-step)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute log p(b | y_i, T_i, δ_i; θ) (up to a constant) for a single subject.
///
/// Returns the scalar log-posterior value.
fn log_posterior_re(
    b: [f64; 2],
    obs: &JointObs,
    alpha: [f64; 2],
    gamma: f64,
    sigma_sq: f64,
    d_inv: &[f64; 4], // row-major inverse of D
    shape: f64,
    scale: f64,
) -> f64 {
    // Prior: -0.5 b^T D^{-1} b
    let prior = -0.5
        * (d_inv[0] * b[0] * b[0] + (d_inv[1] + d_inv[2]) * b[0] * b[1] + d_inv[3] * b[1] * b[1]);

    // Longitudinal log-likelihood: -0.5 σ^{-2} Σ_j (y_ij - α_0 - α_1 s_ij - b_0 - b_1 s_ij)²
    let mut long_ll = 0.0_f64;
    let inv_sig2 = 1.0 / sigma_sq.max(1.0e-300);
    for j in 0..obs.meas_times.len() {
        let s = obs.meas_times[j];
        let resid = obs.measurements[j] - alpha[0] - alpha[1] * s - b[0] - b[1] * s;
        long_ll -= 0.5 * inv_sig2 * resid * resid;
    }

    // Survival log-likelihood:
    //   δ * [log h_0(T) + γ * μ_i(T)] - H_0(T) * exp(γ * μ_i(T))
    let t_ev = obs.event_time.max(1.0e-300);
    let mu_t = alpha[0] + alpha[1] * t_ev + b[0] + b[1] * t_ev;
    let cum_haz = weibull_cumhaz(t_ev, shape, scale);
    let surv_ll = if obs.event {
        let log_haz = weibull_log_hazard(t_ev, shape, scale);
        log_haz + gamma * mu_t - cum_haz * (gamma * mu_t).exp()
    } else {
        -cum_haz * (gamma * mu_t).exp()
    };

    prior + long_ll + surv_ll
}

/// Compute gradient ∇_b log p(b | ...) and Hessian H_b (both in 2D).
///
/// Returns `(grad, hess)` where `hess` is the 2×2 Hessian stored as `[h00, h01, h10, h11]`.
fn grad_hess_re(
    b: [f64; 2],
    obs: &JointObs,
    alpha: [f64; 2],
    gamma: f64,
    sigma_sq: f64,
    d_inv: &[f64; 4],
    shape: f64,
    scale: f64,
) -> ([f64; 2], [f64; 4]) {
    let inv_sig2 = 1.0 / sigma_sq.max(1.0e-300);
    let t_ev = obs.event_time.max(1.0e-300);
    let mu_t = alpha[0] + alpha[1] * t_ev + b[0] + b[1] * t_ev;
    let cum_haz = weibull_cumhaz(t_ev, shape, scale);
    let exp_term = (gamma * mu_t).exp();
    let h0_exp = cum_haz * exp_term; // H_0(T) * exp(γ μ_i(T))

    // ── gradient from the prior: -D^{-1} b ──
    let mut grad = [0.0_f64; 2];
    grad[0] -= d_inv[0] * b[0] + d_inv[1] * b[1];
    grad[1] -= d_inv[2] * b[0] + d_inv[3] * b[1];

    // ── gradient from longitudinal: σ^{-2} Z_i^T (y_i - X_i α - Z_i b) ──
    // Z_ij = [1, s_ij]
    let mut zt_r = [0.0_f64; 2];
    for j in 0..obs.meas_times.len() {
        let s = obs.meas_times[j];
        let resid = obs.measurements[j] - alpha[0] - alpha[1] * s - b[0] - b[1] * s;
        zt_r[0] += resid;
        zt_r[1] += s * resid;
    }
    grad[0] += inv_sig2 * zt_r[0];
    grad[1] += inv_sig2 * zt_r[1];

    // ── gradient from survival ──
    // δ γ [1, T_i] - H_0(T) γ exp(γ μ_i(T)) [1, T_i]
    let delta = if obs.event { 1.0 } else { 0.0 };
    grad[0] += delta * gamma - h0_exp * gamma;
    grad[1] += delta * gamma * t_ev - h0_exp * gamma * t_ev;

    // ── Hessian from the prior: -D^{-1} ──
    let mut hess = [-d_inv[0], -d_inv[1], -d_inv[2], -d_inv[3]];

    // ── Hessian from longitudinal: -σ^{-2} Z_i^T Z_i ──
    let mut ztz = [0.0_f64; 4];
    for j in 0..obs.meas_times.len() {
        let s = obs.meas_times[j];
        ztz[0] += 1.0;
        ztz[1] += s;
        ztz[2] += s;
        ztz[3] += s * s;
    }
    hess[0] -= inv_sig2 * ztz[0];
    hess[1] -= inv_sig2 * ztz[1];
    hess[2] -= inv_sig2 * ztz[2];
    hess[3] -= inv_sig2 * ztz[3];

    // ── Hessian from survival: -H_0(T) γ² exp(γ μ_i(T)) [[1, T]; [T, T²]] ──
    let coeff = -h0_exp * gamma * gamma;
    hess[0] += coeff;
    hess[1] += coeff * t_ev;
    hess[2] += coeff * t_ev;
    hess[3] += coeff * t_ev * t_ev;

    (grad, hess)
}

// ──────────────────────────────────────────────────────────────────────────────
// Laplace E-step: find posterior mode for one subject via Newton-Raphson
// ──────────────────────────────────────────────────────────────────────────────

/// Find the posterior mode `b̂_i` for subject `i` using Newton-Raphson in 2D.
///
/// Starting from `b_init`, iterates until convergence or `max_iter` steps.
/// Returns `(b_mode, converged)`.
fn laplace_estep_subject(
    obs: &JointObs,
    b_init: [f64; 2],
    alpha: [f64; 2],
    gamma: f64,
    sigma_sq: f64,
    d_inv: &[f64; 4],
    shape: f64,
    scale: f64,
    max_iter: usize,
    tol: f64,
) -> ([f64; 2], bool) {
    let mut b = b_init;
    for _ in 0..max_iter {
        let (grad, hess) = grad_hess_re(b, obs, alpha, gamma, sigma_sq, d_inv, shape, scale);
        // Newton step: b_new = b - H^{-1} grad  (H is negative definite, so -H^{-1} is PD)
        let h_inv = match inv2x2(hess[0], hess[1], hess[2], hess[3]) {
            Some(m) => m,
            None => return (b, false),
        };
        // delta = H^{-1} grad
        let delta = [
            h_inv[0] * grad[0] + h_inv[1] * grad[1],
            h_inv[2] * grad[0] + h_inv[3] * grad[1],
        ];
        b[0] -= delta[0];
        b[1] -= delta[1];
        if delta[0].abs().max(delta[1].abs()) < tol {
            return (b, true);
        }
    }
    (b, false)
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-likelihood computation (observed-data approximation via Laplace)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the Laplace-approximated observed log-likelihood:
/// `ℓ_obs ≈ Σ_i log p(b̂_i | data_i; θ) + 0.5 log|−H(b̂_i)|`
/// (constant terms dropped; used to monitor EM convergence).
fn laplace_log_likelihood(
    data: &[JointObs],
    re_modes: &[[f64; 2]],
    alpha: [f64; 2],
    gamma: f64,
    sigma_sq: f64,
    d_inv: &[f64; 4],
    shape: f64,
    scale: f64,
) -> f64 {
    let mut ll = 0.0_f64;
    for (i, obs) in data.iter().enumerate() {
        let b = re_modes[i];
        let lp = log_posterior_re(b, obs, alpha, gamma, sigma_sq, d_inv, shape, scale);
        // Laplace correction: +0.5 * log det(-H)
        let (_, hess) = grad_hess_re(b, obs, alpha, gamma, sigma_sq, d_inv, shape, scale);
        // H is negative semidefinite; -H is PSD.
        let neg_det = hess[0] * hess[3] - hess[1] * hess[2]; // det(-H) = det(H) for 2×2 NSD
        // det(H) = h00*h33 - h01*h10 (already negative for NSD, so -det(H) > 0)
        let log_det_neg_h = (-neg_det).abs().max(1.0e-300).ln();
        ll += lp + 0.5 * log_det_neg_h;
    }
    ll
}

// ──────────────────────────────────────────────────────────────────────────────
// M-step helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Update fixed effects α via OLS normal equations.
///
/// Minimise Σ_i Σ_j (y_{ij} - α_0 - α_1 s_{ij} - b̂_{i0} - b̂_{i1} s_{ij})² w.r.t. α.
fn update_alpha(data: &[JointObs], re_modes: &[[f64; 2]]) -> [f64; 2] {
    // 2×2 normal equations: [Σ 1, Σ s; Σ s, Σ s²] α = [Σ r, Σ r s]
    let mut xtx = [0.0_f64; 4]; // [XtX00, XtX01, XtX10, XtX11]
    let mut xtr = [0.0_f64; 2]; // X^T (y - Z b)
    for (i, obs) in data.iter().enumerate() {
        let b = re_modes[i];
        for j in 0..obs.meas_times.len() {
            let s = obs.meas_times[j];
            let r = obs.measurements[j] - b[0] - b[1] * s;
            xtx[0] += 1.0;
            xtx[1] += s;
            xtx[2] += s;
            xtx[3] += s * s;
            xtr[0] += r;
            xtr[1] += r * s;
        }
    }
    match inv2x2(xtx[0], xtx[1], xtx[2], xtx[3]) {
        Some(inv) => [
            inv[0] * xtr[0] + inv[1] * xtr[1],
            inv[2] * xtr[0] + inv[3] * xtr[1],
        ],
        None => [0.0, 0.0],
    }
}

/// Update D (random effects covariance) as sample covariance of `{b̂_i}` plus ridge.
fn update_d(re_modes: &[[f64; 2]], ridge: f64) -> [f64; 4] {
    let n = re_modes.len() as f64;
    if n < 1.0 {
        return [1.0 + ridge, 0.0, 0.0, 1.0 + ridge];
    }
    let mut d = [0.0_f64; 4];
    for b in re_modes {
        d[0] += b[0] * b[0];
        d[1] += b[0] * b[1];
        d[2] += b[1] * b[0];
        d[3] += b[1] * b[1];
    }
    let inv_n = 1.0 / n;
    d[0] = d[0] * inv_n + ridge;
    d[1] *= inv_n;
    d[2] *= inv_n;
    d[3] = d[3] * inv_n + ridge;
    d
}

/// Update σ² as mean squared residual.
fn update_sigma_sq(data: &[JointObs], re_modes: &[[f64; 2]], alpha: [f64; 2]) -> f64 {
    let mut sse = 0.0_f64;
    let mut count = 0usize;
    for (i, obs) in data.iter().enumerate() {
        let b = re_modes[i];
        for j in 0..obs.meas_times.len() {
            let s = obs.meas_times[j];
            let resid = obs.measurements[j] - alpha[0] - alpha[1] * s - b[0] - b[1] * s;
            sse += resid * resid;
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    (sse / count as f64).max(1.0e-10)
}

/// Survival log-likelihood summed over all subjects at given `(gamma, shape, scale)`.
fn survival_loglik(
    data: &[JointObs],
    re_modes: &[[f64; 2]],
    alpha: [f64; 2],
    gamma: f64,
    shape: f64,
    scale: f64,
) -> f64 {
    let mut ll = 0.0_f64;
    for (i, obs) in data.iter().enumerate() {
        let b = re_modes[i];
        let t_ev = obs.event_time.max(1.0e-300);
        let mu_t = alpha[0] + alpha[1] * t_ev + b[0] + b[1] * t_ev;
        let cum_haz = weibull_cumhaz(t_ev, shape, scale);
        let survival_contrib = if obs.event {
            let log_haz = weibull_log_hazard(t_ev, shape, scale);
            log_haz + gamma * mu_t - cum_haz * (gamma * mu_t).exp()
        } else {
            -cum_haz * (gamma * mu_t).exp()
        };
        ll += survival_contrib;
    }
    ll
}

/// Numerical gradient (central differences, step h=1e-7) and Hessian of the
/// survival log-likelihood w.r.t. `θ = (gamma, log_shape, log_scale)`.
///
/// We optimise on log(shape) and log(scale) to enforce positivity.
fn survival_grad_hess(
    data: &[JointObs],
    re_modes: &[[f64; 2]],
    alpha: [f64; 2],
    gamma: f64,
    log_shape: f64,
    log_scale: f64,
) -> ([f64; 3], [[f64; 3]; 3]) {
    let h = 1.0e-7_f64;
    let shape = log_shape.exp();
    let scale = log_scale.exp();
    let theta = [gamma, log_shape, log_scale];
    let f0 = survival_loglik(data, re_modes, alpha, gamma, shape, scale);
    let mut grad = [0.0_f64; 3];
    let mut fd_plus = [0.0_f64; 3];
    let mut fd_minus = [0.0_f64; 3];

    for k in 0..3 {
        let mut tp = theta;
        let mut tm = theta;
        tp[k] += h;
        tm[k] -= h;
        let sp = tp[1].exp();
        let slp = tp[2].exp();
        let sm = tm[1].exp();
        let slm = tm[2].exp();
        fd_plus[k] = survival_loglik(data, re_modes, alpha, tp[0], sp, slp);
        fd_minus[k] = survival_loglik(data, re_modes, alpha, tm[0], sm, slm);
        grad[k] = (fd_plus[k] - fd_minus[k]) / (2.0 * h);
    }

    // Hessian via central second differences.
    let mut hess = [[0.0_f64; 3]; 3];
    for k in 0..3 {
        // Diagonal: (f(+h) - 2f0 + f(-h)) / h²
        hess[k][k] = (fd_plus[k] - 2.0 * f0 + fd_minus[k]) / (h * h);
    }
    for k in 0..3 {
        for l in (k + 1)..3 {
            // Cross terms: (f(+h_k,+h_l) - f(+h_k,-h_l) - f(-h_k,+h_l) + f(-h_k,-h_l)) / (4h²)
            let mut tpp = theta;
            let mut tpm = theta;
            let mut tmp = theta;
            let mut tmm = theta;
            tpp[k] += h;
            tpp[l] += h;
            tpm[k] += h;
            tpm[l] -= h;
            tmp[k] -= h;
            tmp[l] += h;
            tmm[k] -= h;
            tmm[l] -= h;
            let fpp = survival_loglik(data, re_modes, alpha, tpp[0], tpp[1].exp(), tpp[2].exp());
            let fpm = survival_loglik(data, re_modes, alpha, tpm[0], tpm[1].exp(), tpm[2].exp());
            let fmp = survival_loglik(data, re_modes, alpha, tmp[0], tmp[1].exp(), tmp[2].exp());
            let fmm = survival_loglik(data, re_modes, alpha, tmm[0], tmm[1].exp(), tmm[2].exp());
            let cross = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hess[k][l] = cross;
            hess[l][k] = cross;
        }
    }
    (grad, hess)
}

/// One Newton step on `(γ, log k, log λ)` to maximise the survival log-likelihood.
/// Returns the updated `(gamma, shape, scale)`, or the unchanged values if the
/// 3×3 linear system is singular.
fn update_survival_params(
    data: &[JointObs],
    re_modes: &[[f64; 2]],
    alpha: [f64; 2],
    gamma: f64,
    shape: f64,
    scale: f64,
    step_size: f64,
) -> (f64, f64, f64) {
    let log_shape = shape.ln();
    let log_scale = scale.ln();
    let (grad, hess) = survival_grad_hess(data, re_modes, alpha, gamma, log_shape, log_scale);

    // Solve H Δθ = grad  (maximisation: follow the gradient).
    // Build augmented matrix [H | grad].
    let mut aug = [
        [hess[0][0], hess[0][1], hess[0][2], grad[0]],
        [hess[1][0], hess[1][1], hess[1][2], grad[1]],
        [hess[2][0], hess[2][1], hess[2][2], grad[2]],
    ];
    let delta = match solve3x3(&mut aug) {
        Some(d) => d,
        None => return (gamma, shape, scale),
    };

    let new_gamma = gamma + step_size * delta[0];
    let new_shape = (log_shape + step_size * delta[1]).exp().max(1.0e-6);
    let new_scale = (log_scale + step_size * delta[2]).exp().max(1.0e-6);

    // Accept only if the likelihood improves (safeguard against Newton overshoot).
    let ll_old = survival_loglik(data, re_modes, alpha, gamma, shape, scale);
    let ll_new = survival_loglik(data, re_modes, alpha, new_gamma, new_shape, new_scale);
    if ll_new > ll_old || (ll_new - ll_old).abs() < 1.0e-12 {
        (new_gamma, new_shape, new_scale)
    } else {
        // Halved step.
        let half = step_size * 0.5;
        let g2 = gamma + half * delta[0];
        let s2 = (log_shape + half * delta[1]).exp().max(1.0e-6);
        let l2 = (log_scale + half * delta[2]).exp().max(1.0e-6);
        let ll2 = survival_loglik(data, re_modes, alpha, g2, s2, l2);
        if ll2 > ll_old {
            (g2, s2, l2)
        } else {
            (gamma, shape, scale) // no improvement — skip the step
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Initialisation helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Initialise parameters from data via simple moment estimators.
fn initialise_params(data: &[JointObs]) -> (f64, f64, f64, f64, f64, f64, f64) {
    // α_0, α_1 via OLS on pooled longitudinal data.
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut n_long = 0usize;
    for obs in data {
        for j in 0..obs.meas_times.len() {
            let s = obs.meas_times[j];
            let y = obs.measurements[j];
            sx += s;
            sy += y;
            sxx += s * s;
            sxy += s * y;
            n_long += 1;
        }
    }
    let nl = n_long as f64;
    let alpha_0;
    let alpha_1;
    if n_long > 1 {
        let denom = nl * sxx - sx * sx;
        if denom.abs() > 1.0e-12 {
            alpha_1 = (nl * sxy - sx * sy) / denom;
            alpha_0 = (sy - alpha_1 * sx) / nl;
        } else {
            alpha_0 = sy / nl;
            alpha_1 = 0.0;
        }
    } else {
        alpha_0 = if n_long == 1 {
            data[0].measurements[0]
        } else {
            0.0
        };
        alpha_1 = 0.0;
    }

    // σ² from residual variance.
    let mut sse = 0.0_f64;
    for obs in data {
        for j in 0..obs.meas_times.len() {
            let s = obs.meas_times[j];
            let resid = obs.measurements[j] - alpha_0 - alpha_1 * s;
            sse += resid * resid;
        }
    }
    let sigma_sq = if n_long > 0 {
        (sse / nl).max(1.0e-4)
    } else {
        1.0
    };

    // Weibull parameters from marginal event times via MLE on shape=2, scale=mean.
    let n_subjects = data.len() as f64;
    let mean_time = data.iter().map(|o| o.event_time).sum::<f64>() / n_subjects.max(1.0);
    let shape = 2.0_f64;
    let scale = mean_time.max(1.0e-4);

    // γ = 0 initially (no association).
    let gamma = 0.0_f64;

    (alpha_0, alpha_1, gamma, sigma_sq, shape, scale, 1.0)
    // The last 1.0 is a placeholder for the D ridge — not returned; D is initialised separately.
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API: joint_model_fit
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a joint longitudinal-survival model by EM with Laplace E-step.
///
/// # Arguments
/// * `data` — one [`JointObs`] per subject
/// * `cfg`  — algorithm configuration
///
/// # Errors
/// Returns [`SurvivalError::EmptyDataset`] if `data` is empty,
/// [`SurvivalError::InvalidParameter`] for malformed observations,
/// or [`SurvivalError::NumericalInstability`] if D becomes non-invertible.
pub fn joint_model_fit(data: &[JointObs], cfg: &JointModelConfig) -> SurvivalResult<JointModelFit> {
    // ── validation ────────────────────────────────────────────────────────────
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    for (i, obs) in data.iter().enumerate() {
        if obs.meas_times.len() != obs.measurements.len() {
            return Err(SurvivalError::InvalidParameter(format!(
                "subject {i}: meas_times.len()={} != measurements.len()={}",
                obs.meas_times.len(),
                obs.measurements.len()
            )));
        }
        if obs.event_time <= 0.0 {
            return Err(SurvivalError::NegativeTime(obs.event_time));
        }
    }
    let n_subjects = data.len();

    // ── initialise parameters ─────────────────────────────────────────────────
    let (mut alpha_0, mut alpha_1, mut gamma, mut sigma_sq, mut shape, mut scale, _) =
        initialise_params(data);
    let mut alpha = [alpha_0, alpha_1];

    // D initialised as scaled identity.
    let mut d_mat = [1.0_f64, 0.0, 0.0, 1.0_f64];

    // Posterior modes: initialise to zero.
    let mut re_modes: Vec<[f64; 2]> = vec![[0.0, 0.0]; n_subjects];

    let mut ll_prev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    for em_iter in 0..cfg.max_iter {
        n_iter = em_iter + 1;

        // ── compute D inverse ──────────────────────────────────────────────────
        let d_inv = match inv2x2(d_mat[0], d_mat[1], d_mat[2], d_mat[3]) {
            Some(inv) => inv,
            None => {
                return Err(SurvivalError::NumericalInstability(
                    "D matrix is singular — cannot invert".to_string(),
                ));
            }
        };

        // ── E-step: update posterior modes ────────────────────────────────────
        for i in 0..n_subjects {
            let (b_hat, _) = laplace_estep_subject(
                &data[i],
                re_modes[i],
                alpha,
                gamma,
                sigma_sq,
                &d_inv,
                shape,
                scale,
                cfg.newton_max_iter,
                cfg.newton_tol,
            );
            re_modes[i] = b_hat;
        }

        // ── compute Laplace log-likelihood ────────────────────────────────────
        let ll_now = laplace_log_likelihood(
            data, &re_modes, alpha, gamma, sigma_sq, &d_inv, shape, scale,
        );

        // ── check convergence ─────────────────────────────────────────────────
        if (ll_now - ll_prev).abs() < cfg.em_tol && em_iter > 0 {
            converged = true;
            ll_prev = ll_now;
            break;
        }
        ll_prev = ll_now;

        // ── M-step ────────────────────────────────────────────────────────────
        // Update α
        alpha = update_alpha(data, &re_modes);
        alpha_0 = alpha[0];
        alpha_1 = alpha[1];

        // Update D
        d_mat = update_d(&re_modes, cfg.re_ridge);

        // Update σ²
        sigma_sq = update_sigma_sq(data, &re_modes, [alpha_0, alpha_1]);

        // Update (γ, k, λ) via Newton step on survival log-likelihood.
        // Use a conservative step size that decays slightly over iterations.
        let step = (1.0 / (1.0 + em_iter as f64 * 0.1)).max(0.1);
        let (new_gamma, new_shape, new_scale) = update_survival_params(
            data,
            &re_modes,
            [alpha_0, alpha_1],
            gamma,
            shape,
            scale,
            step,
        );
        gamma = new_gamma;
        shape = new_shape.max(1.0e-6);
        scale = new_scale.max(1.0e-6);
    }

    // Recompute final D inverse for ll computation.
    let d_inv_final =
        inv2x2(d_mat[0], d_mat[1], d_mat[2], d_mat[3]).unwrap_or([1.0, 0.0, 0.0, 1.0]);
    let final_ll = laplace_log_likelihood(
        data,
        &re_modes,
        [alpha_0, alpha_1],
        gamma,
        sigma_sq,
        &d_inv_final,
        shape,
        scale,
    );

    // Flatten re_modes to Vec<f64>.
    let re_modes_flat: Vec<f64> = re_modes.iter().flat_map(|b| b.iter().copied()).collect();

    Ok(JointModelFit {
        alpha: [alpha_0, alpha_1],
        gamma,
        sigma_sq,
        d_mat,
        weibull_shape: shape,
        weibull_scale: scale,
        log_likelihood: if ll_prev.is_finite() {
            ll_prev
        } else {
            final_ll
        },
        n_iter,
        converged,
        re_modes: re_modes_flat,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API: predictions
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the posterior mode random effects for a new subject given the fitted model.
///
/// Used internally by prediction functions, but also useful for diagnostics.
fn posterior_re_new_subject(
    fit: &JointModelFit,
    obs: &JointObs,
    cfg: &JointModelConfig,
) -> SurvivalResult<[f64; 2]> {
    let d_inv = match inv2x2(fit.d_mat[0], fit.d_mat[1], fit.d_mat[2], fit.d_mat[3]) {
        Some(inv) => inv,
        None => {
            return Err(SurvivalError::NumericalInstability(
                "D matrix is singular in predict".to_string(),
            ));
        }
    };
    let (b_hat, _) = laplace_estep_subject(
        obs,
        [0.0, 0.0],
        fit.alpha,
        fit.gamma,
        fit.sigma_sq,
        &d_inv,
        fit.weibull_shape,
        fit.weibull_scale,
        cfg.newton_max_iter,
        cfg.newton_tol,
    );
    Ok(b_hat)
}

/// Predict the individual survival probability at `horizon` for a new subject.
///
/// Computes S_i(horizon | T_i > s_last) where `s_last` is the last observed
/// longitudinal time.  The prediction integrates out the random effects using
/// the Laplace posterior mode `b̂_i` derived from `new_obs`.
///
/// # Returns
/// Survival probability in `(0, 1]`.
pub fn joint_model_predict_survival(
    fit: &JointModelFit,
    new_obs: &JointObs,
    horizon: f64,
    cfg: &JointModelConfig,
) -> SurvivalResult<f64> {
    if horizon <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "horizon must be positive".to_string(),
        ));
    }
    let b_hat = posterior_re_new_subject(fit, new_obs, cfg)?;
    // Conditional survival: S_i(t | b̂_i) = exp(-H_0(t) exp(γ μ_i(t)))
    let mu_t = fit.alpha[0] + fit.alpha[1] * horizon + b_hat[0] + b_hat[1] * horizon;
    let cum_haz = weibull_cumhaz(horizon, fit.weibull_shape, fit.weibull_scale);
    let surv = (-cum_haz * (fit.gamma * mu_t).exp()).exp();
    Ok(surv.clamp(0.0, 1.0))
}

/// Predict the longitudinal trajectory `μ_i(t) = α_0 + α_1 t + b̂_{i0} + b̂_{i1} t`
/// at a vector of times.
///
/// # Returns
/// A vector of length `times.len()`, one predicted value per time.
pub fn joint_model_predict_trajectory(
    fit: &JointModelFit,
    new_obs: &JointObs,
    times: &[f64],
    cfg: &JointModelConfig,
) -> SurvivalResult<Vec<f64>> {
    let b_hat = posterior_re_new_subject(fit, new_obs, cfg)?;
    let traj: Vec<f64> = times
        .iter()
        .map(|&t| fit.alpha[0] + fit.alpha[1] * t + b_hat[0] + b_hat[1] * t)
        .collect();
    Ok(traj)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate a deterministic synthetic dataset of `n` subjects with:
    /// - Longitudinal trajectory: μ_i(t) = 1.0 + 0.5*t + b_{i0} + b_{i1}*t
    /// - Event times drawn from Weibull(k=2, λ=3) with some censoring at 4.0.
    /// - Measurements at times [0.5, 1.0, 1.5, 2.0] with σ²=0.04.
    fn make_synthetic_data(n: usize, seed: u64) -> Vec<JointObs> {
        let mut rng = LcgRng::new(seed);
        let alpha_0 = 1.0_f64;
        let alpha_1 = 0.5_f64;
        let shape = 2.0_f64;
        let scale = 3.0_f64;
        let sigma = 0.2_f64;
        let meas_times = vec![0.5, 1.0, 1.5, 2.0];
        let censor_time = 4.0_f64;

        (0..n)
            .map(|_| {
                // Random effects from N(0, 0.25 I).
                let b0 = rng.next_normal() * 0.5;
                let b1 = rng.next_normal() * 0.2;

                // Measurements.
                let measurements: Vec<f64> = meas_times
                    .iter()
                    .map(|&s| alpha_0 + alpha_1 * s + b0 + b1 * s + rng.next_normal() * sigma)
                    .collect();

                // Event time from Weibull(shape, scale).
                // Inverse CDF: T = λ * (-ln U)^{1/k}.
                let u = rng.next_f64().max(1.0e-300);
                let event_time_raw = scale * (-u.ln()).powf(1.0 / shape);
                let event_time = event_time_raw.min(censor_time);
                let event = event_time_raw < censor_time;

                JointObs {
                    meas_times: meas_times.clone(),
                    measurements,
                    event_time,
                    event,
                }
            })
            .collect()
    }

    // ── Test 1: JointObs construction ─────────────────────────────────────────
    #[test]
    fn joint_obs_construction() {
        let obs = JointObs {
            meas_times: vec![0.5, 1.0, 1.5],
            measurements: vec![1.1, 1.4, 1.7],
            event_time: 2.5,
            event: true,
        };
        assert_eq!(obs.meas_times.len(), 3);
        assert_eq!(obs.measurements.len(), 3);
        assert!(obs.event);
        assert!((obs.event_time - 2.5).abs() < 1.0e-12);
    }

    // ── Test 2: JointModelConfig default values ───────────────────────────────
    #[test]
    fn joint_model_config_default() {
        let cfg = JointModelConfig::default();
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.em_tol - 1.0e-4).abs() < 1.0e-15);
        assert_eq!(cfg.newton_max_iter, 20);
        assert!((cfg.newton_tol - 1.0e-8).abs() < 1.0e-18);
        assert!((cfg.re_ridge - 1.0e-6).abs() < 1.0e-15);
    }

    // ── Test 3: empty data returns EmptyDataset ───────────────────────────────
    #[test]
    fn joint_fit_empty_data_error() {
        let cfg = JointModelConfig::default();
        let result = joint_model_fit(&[], &cfg);
        assert!(matches!(result, Err(SurvivalError::EmptyDataset)));
    }

    // ── Test 4: single subject runs without panic ─────────────────────────────
    #[test]
    fn joint_fit_single_subject() {
        let obs = JointObs {
            meas_times: vec![0.5, 1.0, 1.5],
            measurements: vec![1.25, 1.5, 1.75],
            event_time: 2.0,
            event: true,
        };
        let cfg = JointModelConfig {
            max_iter: 10,
            ..Default::default()
        };
        // May succeed or return an error, but must not panic.
        let _ = joint_model_fit(&[obs], &cfg);
    }

    // ── Test 5: two subjects run gracefully ───────────────────────────────────
    #[test]
    fn joint_fit_two_subjects() {
        let data = make_synthetic_data(2, 1234);
        let cfg = JointModelConfig {
            max_iter: 20,
            ..Default::default()
        };
        let result = joint_model_fit(&data, &cfg);
        // Must not panic and should either succeed or produce a known error.
        match result {
            Ok(_) => {}
            Err(SurvivalError::NumericalInstability(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── Test 6: convergence flag is set ──────────────────────────────────────
    #[test]
    fn joint_fit_convergence_flag() {
        let data = make_synthetic_data(8, 42);
        let cfg = JointModelConfig {
            max_iter: 200,
            em_tol: 1.0e-3,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        // With enough iterations and relaxed tolerance, should converge.
        assert!(
            fit.converged,
            "expected converged=true, got n_iter={}",
            fit.n_iter
        );
    }

    // ── Test 7: log-likelihood is finite ──────────────────────────────────────
    #[test]
    fn joint_fit_log_likelihood_finite() {
        let data = make_synthetic_data(6, 77);
        let cfg = JointModelConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        assert!(
            fit.log_likelihood.is_finite(),
            "log_likelihood = {}",
            fit.log_likelihood
        );
    }

    // ── Test 8: re_modes has correct shape ────────────────────────────────────
    #[test]
    fn joint_fit_re_modes_shape() {
        let n = 7usize;
        let data = make_synthetic_data(n, 99);
        let cfg = JointModelConfig {
            max_iter: 30,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        assert_eq!(fit.re_modes.len(), 2 * n, "re_modes length mismatch");
    }

    // ── Test 9: intercept near truth on simple synthetic data ─────────────────
    #[test]
    fn joint_fit_alpha_intercept_recovery() {
        // Use more subjects for better identifiability.
        let data = make_synthetic_data(10, 555);
        let cfg = JointModelConfig {
            max_iter: 100,
            em_tol: 1.0e-4,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        // True intercept is 1.0; accept within 2 units for small n.
        assert!(
            (fit.alpha[0] - 1.0).abs() < 2.0,
            "intercept={:.3} too far from truth=1.0",
            fit.alpha[0]
        );
    }

    // ── Test 10: gamma sign on positive-association data ──────────────────────
    #[test]
    fn joint_fit_gamma_sign() {
        // Construct subjects where higher trajectory predicts shorter survival.
        // γ > 0 means higher μ_i(T) increases hazard → shorter survival.
        let mut data: Vec<JointObs> = Vec::new();
        // High trajectory + short survival.
        for _ in 0..5 {
            data.push(JointObs {
                meas_times: vec![0.5, 1.0, 1.5, 2.0],
                measurements: vec![3.0, 3.5, 4.0, 4.5],
                event_time: 0.8,
                event: true,
            });
        }
        // Low trajectory + long survival (censored).
        for _ in 0..5 {
            data.push(JointObs {
                meas_times: vec![0.5, 1.0, 1.5, 2.0],
                measurements: vec![0.5, 0.6, 0.7, 0.8],
                event_time: 4.0,
                event: false,
            });
        }
        let cfg = JointModelConfig {
            max_iter: 80,
            em_tol: 1.0e-3,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        // Expect positive association.
        assert!(fit.gamma >= 0.0, "expected gamma >= 0, got {}", fit.gamma);
    }

    // ── Test 11: survival prediction in (0, 1] ────────────────────────────────
    #[test]
    fn joint_predict_survival_range() {
        let data = make_synthetic_data(6, 11);
        let cfg = JointModelConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        let new_obs = JointObs {
            meas_times: vec![0.5, 1.0],
            measurements: vec![1.25, 1.75],
            event_time: 2.0,
            event: false,
        };
        for &t in &[0.5, 1.0, 2.0, 3.0, 5.0] {
            let s = joint_model_predict_survival(&fit, &new_obs, t, &cfg)
                .expect("predict should succeed");
            assert!((0.0..=1.0).contains(&s), "S({t}) = {s} not in (0, 1]");
        }
    }

    // ── Test 12: survival is monotone non-increasing ───────────────────────────
    #[test]
    fn joint_predict_survival_monotone() {
        let data = make_synthetic_data(6, 22);
        let cfg = JointModelConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        let new_obs = JointObs {
            meas_times: vec![0.5, 1.0],
            measurements: vec![1.2, 1.6],
            event_time: 2.0,
            event: false,
        };
        let times = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
        let survs: Vec<f64> = times
            .iter()
            .map(|&t| {
                joint_model_predict_survival(&fit, &new_obs, t, &cfg)
                    .expect("predict should succeed")
            })
            .collect();
        for i in 1..survs.len() {
            assert!(
                survs[i] <= survs[i - 1] + 1.0e-10,
                "S({}) = {} > S({}) = {} — not monotone",
                times[i],
                survs[i],
                times[i - 1],
                survs[i - 1]
            );
        }
    }

    // ── Test 13: trajectory prediction returns correct length ─────────────────
    #[test]
    fn joint_predict_trajectory_shape() {
        let data = make_synthetic_data(5, 33);
        let cfg = JointModelConfig {
            max_iter: 30,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        let new_obs = JointObs {
            meas_times: vec![0.5, 1.0, 1.5],
            measurements: vec![1.1, 1.4, 1.7],
            event_time: 2.5,
            event: false,
        };
        let times: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let traj = joint_model_predict_trajectory(&fit, &new_obs, &times, &cfg)
            .expect("predict should succeed");
        assert_eq!(traj.len(), times.len());
    }

    // ── Test 14: trajectory is linear with zero random effects ────────────────
    #[test]
    fn joint_predict_trajectory_linear() {
        // Build a fit with zero random effects (b=0) and known alpha.
        let fit = JointModelFit {
            alpha: [2.0, 0.3],
            gamma: 0.5,
            sigma_sq: 0.04,
            d_mat: [1.0, 0.0, 0.0, 1.0],
            weibull_shape: 2.0,
            weibull_scale: 3.0,
            log_likelihood: -10.0,
            n_iter: 1,
            converged: true,
            re_modes: vec![],
        };
        // A new observation with no measurements: random effects will be pulled to 0.
        // Give a few noisy observations exactly on the line so b̂ ≈ 0.
        let new_obs = JointObs {
            meas_times: vec![1.0, 2.0, 3.0, 4.0],
            measurements: vec![2.3, 2.6, 2.9, 3.2], // exactly 2.0 + 0.3*t
            event_time: 5.0,
            event: false,
        };
        let cfg = JointModelConfig::default();
        let times = vec![0.0, 1.0, 2.0, 5.0];
        let traj = joint_model_predict_trajectory(&fit, &new_obs, &times, &cfg)
            .expect("predict should succeed");
        assert_eq!(traj.len(), 4);
        // With a strong D prior pulling b to 0, trajectory ≈ α_0 + α_1*t.
        for (k, &t) in times.iter().enumerate() {
            let expected = 2.0 + 0.3 * t;
            assert!(
                (traj[k] - expected).abs() < 0.5,
                "trajectory[{t}]={:.3} expected≈{:.3}",
                traj[k],
                expected
            );
        }
    }

    // ── Test 15: Weibull shape is positive after fit ──────────────────────────
    #[test]
    fn joint_fit_weibull_shape_positive() {
        let data = make_synthetic_data(8, 66);
        let cfg = JointModelConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        assert!(
            fit.weibull_shape > 0.0,
            "weibull_shape = {} not positive",
            fit.weibull_shape
        );
    }

    // ── Test 16: sigma_sq is positive after fit ───────────────────────────────
    #[test]
    fn joint_fit_sigma_sq_positive() {
        let data = make_synthetic_data(8, 88);
        let cfg = JointModelConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = joint_model_fit(&data, &cfg).expect("fit should succeed");
        assert!(
            fit.sigma_sq > 0.0,
            "sigma_sq = {} not positive",
            fit.sigma_sq
        );
    }
}
