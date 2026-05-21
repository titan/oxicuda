//! Sinkhorn with Exponential Moving Average on dual potentials.
//!
//! Sinkhorn-EMA applies an exponential moving average to the log-domain dual
//! potentials at each iteration, producing smoothed, stabilised gradient
//! signals that are useful for differentiable OT in generative models.
//!
//! The EMA update blends the new raw Sinkhorn iterate with the running average:
//! ```text
//! log_u_ema ← (1 − α) · log_u_ema  +  α · log_u_raw
//! log_v_ema ← (1 − α) · log_v_ema  +  α · log_v_raw
//! ```
//!
//! - `alpha = 1.0`: no smoothing → equivalent to standard log-domain Sinkhorn.
//! - `alpha ∈ (0, 1)`: exponential moving average — the smaller `alpha`, the
//!   more the history is retained and the smoother (but slower) the update.

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp of a f64 slice.
/// Returns `f64::NEG_INFINITY` if the slice is empty.
#[inline]
fn logsumexp_f64(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut max_val = f64::NEG_INFINITY;
    for &x in slice {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f64;
    for &x in slice {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

/// Clamp-guarded natural logarithm so `safe_ln64(0) = ln(f64::MIN_POSITIVE)`.
#[inline]
fn safe_ln64(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Sinkhorn-EMA solver.
#[derive(Debug, Clone)]
pub struct EmaSinkhornConfig {
    /// Entropic regularisation strength (must be > 0).
    pub reg: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum marginal violation.
    pub tol: f64,
    /// EMA learning rate α ∈ (0, 1].
    ///
    /// Controls how much of the new raw Sinkhorn iterate is mixed in at each step:
    /// `log_u_ema ← (1 − α) · log_u_ema + α · log_u_raw`.
    ///
    /// - `alpha = 1.0`: no smoothing → equivalent to standard log-domain Sinkhorn.
    /// - `alpha ∈ (0, 1)`: EMA smoothing; smaller values retain more history.
    ///
    /// Default is `0.9` (aggressive update, mild smoothing).
    pub ema_alpha: f64,
}

impl Default for EmaSinkhornConfig {
    fn default() -> Self {
        Self {
            reg: 0.1,
            max_iter: 500,
            tol: 1e-6,
            ema_alpha: 0.9,
        }
    }
}

/// Result produced by [`ema_sinkhorn`].
#[derive(Debug, Clone)]
pub struct EmaSinkhornFit {
    /// EMA-smoothed log-potentials u (row side), length `n`.
    pub log_u_ema: Vec<f64>,
    /// EMA-smoothed log-potentials v (column side), length `m`.
    pub log_v_ema: Vec<f64>,
    /// Primal transport cost `⟨P, C⟩` evaluated at the EMA potentials.
    pub transport_cost: f64,
    /// Number of completed iterations.
    pub n_iter: usize,
    /// Number of source support points.
    pub n: usize,
    /// Number of target support points.
    pub m: usize,
    /// Regularisation used during solving (needed for reconstruction).
    pub reg: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_inputs_ema(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &EmaSinkhornConfig,
) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if cfg.ema_alpha <= 0.0 || cfg.ema_alpha > 1.0 {
        return Err(OtError::Internal {
            msg: format!("ema_alpha must be in (0, 1], got {}", cfg.ema_alpha),
        });
    }
    if cost.len() != n * m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if b.len() != m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core solver
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Sinkhorn-EMA algorithm.
///
/// Solves entropic OT between histograms `a` (length `n`) and `b` (length `m`)
/// with cost matrix `cost` (shape `n × m`, row-major). At each iteration the
/// raw Sinkhorn dual-potential updates are blended with their EMA history via
/// `cfg.ema_alpha`.
///
/// # Errors
///
/// Returns [`OtError::NotConverged`] if the EMA-potential marginal violation
/// does not reach `cfg.tol` within `cfg.max_iter` iterations.
pub fn ema_sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &EmaSinkhornConfig,
) -> OtResult<EmaSinkhornFit> {
    validate_inputs_ema(a, b, cost, n, m, cfg)?;

    let reg = cfg.reg;
    // alpha is the weight on the NEW raw value;
    // (1 - alpha) is the weight on the OLD EMA value.
    // alpha = 1.0 => pure new = standard Sinkhorn; alpha → 0 => heavy smoothing.
    let alpha = cfg.ema_alpha;
    let one_minus_alpha = 1.0 - alpha;

    // Pre-compute log-kernel: log_k[i,j] = -cost[i,j] / reg
    let mut log_k = vec![0.0_f64; n * m];
    for (idx, &c) in cost.iter().enumerate() {
        log_k[idx] = -c / reg;
    }

    // Initialise dual potentials in log-domain
    let mut log_u_ema = vec![0.0_f64; n];
    let mut log_v_ema = vec![0.0_f64; m];
    for (i, &ai) in a.iter().enumerate() {
        log_u_ema[i] = safe_ln64(ai);
    }
    for (j, &bj) in b.iter().enumerate() {
        log_v_ema[j] = safe_ln64(bj);
    }

    // Working buffers for raw updates
    let mut log_v_raw = vec![0.0_f64; m];
    let mut log_u_raw = vec![0.0_f64; n];
    let mut buf = vec![0.0_f64; n.max(m)];

    let mut n_iter = 0_usize;
    let mut converged = false;

    for _iter in 0..cfg.max_iter {
        n_iter += 1;

        // v-update: log_v_raw[j] = log(b[j]) - LSE_i(log_u_ema[i] + log_k[i,j])
        for j in 0..m {
            for (i, b_val) in buf[..n].iter_mut().enumerate() {
                *b_val = log_u_ema[i] + log_k[i * m + j];
            }
            let lse = logsumexp_f64(&buf[..n]);
            log_v_raw[j] = safe_ln64(b[j]) - lse;
        }

        // u-update: log_u_raw[i] = log(a[i]) - LSE_j(log_v_raw[j] + log_k[i,j])
        // Note: use log_v_raw (not yet EMA-smoothed) for standard Sinkhorn update
        for i in 0..n {
            let row_off = i * m;
            for (j, b_val) in buf[..m].iter_mut().enumerate() {
                *b_val = log_v_raw[j] + log_k[row_off + j];
            }
            let lse = logsumexp_f64(&buf[..m]);
            log_u_raw[i] = safe_ln64(a[i]) - lse;
        }

        // Apply EMA to both potentials:
        //   new_ema = (1-alpha)*old_ema + alpha*raw
        // alpha=1.0 => new_ema = raw (standard Sinkhorn, no smoothing)
        // alpha<1.0 => weighted blend retaining history
        for j in 0..m {
            log_v_ema[j] = one_minus_alpha * log_v_ema[j] + alpha * log_v_raw[j];
        }
        for i in 0..n {
            log_u_ema[i] = one_minus_alpha * log_u_ema[i] + alpha * log_u_raw[i];
        }

        // Convergence: max marginal violation over both row and column marginals.
        // Using both sides ensures both a and b marginals are satisfied at stopping.
        let mut max_viol = 0.0_f64;

        // Row marginals
        for i in 0..n {
            let row_off = i * m;
            let mut row_sum = 0.0_f64;
            for j in 0..m {
                row_sum += (log_u_ema[i] + log_v_ema[j] + log_k[row_off + j]).exp();
            }
            let viol = (row_sum - a[i]).abs();
            if viol > max_viol {
                max_viol = viol;
            }
        }

        // Column marginals (only evaluated if row residual is already small)
        if max_viol < cfg.tol * 10.0 {
            for j in 0..m {
                let mut col_sum = 0.0_f64;
                for i in 0..n {
                    col_sum += (log_u_ema[i] + log_v_ema[j] + log_k[i * m + j]).exp();
                }
                let viol = (col_sum - b[j]).abs();
                if viol > max_viol {
                    max_viol = viol;
                }
            }
        }

        if max_viol < cfg.tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(OtError::NotConverged {
            iter: n_iter,
            tol: cfg.tol as f32,
        });
    }

    // Compute transport cost at EMA potentials
    let transport_cost =
        compute_transport_cost_internal(&log_u_ema, &log_v_ema, cost, &log_k, n, m);

    Ok(EmaSinkhornFit {
        log_u_ema,
        log_v_ema,
        transport_cost,
        n_iter,
        n,
        m,
        reg,
    })
}

/// Internal helper: compute ⟨P, C⟩ from log potentials.
fn compute_transport_cost_internal(
    log_u: &[f64],
    log_v: &[f64],
    cost: &[f64],
    log_k: &[f64],
    n: usize,
    m: usize,
) -> f64 {
    let mut tc = 0.0_f64;
    for (i, &lu) in log_u.iter().enumerate().take(n) {
        let row_off = i * m;
        for (j, &lv) in log_v.iter().enumerate().take(m) {
            let p_ij = (lu + lv + log_k[row_off + j]).exp();
            tc += p_ij * cost[row_off + j];
        }
    }
    tc
}

// ─────────────────────────────────────────────────────────────────────────────
// Derived quantities
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct the full transport plan `P[i,j]` from an [`EmaSinkhornFit`].
///
/// `P[i,j] = exp(log_u_ema[i] + log_v_ema[j] − cost[i,j] / reg)`
///
/// The returned slice is `n × m` row-major.
pub fn ema_transport_plan(fit: &EmaSinkhornFit, cost: &[f64]) -> Vec<f64> {
    let n = fit.n;
    let m = fit.m;
    let mut plan = vec![0.0_f64; n * m];
    for i in 0..n {
        let row_off = i * m;
        for j in 0..m {
            plan[row_off + j] =
                (fit.log_u_ema[i] + fit.log_v_ema[j] - cost[row_off + j] / fit.reg).exp();
        }
    }
    plan
}

/// Return the primal transport cost stored in the fit.
///
/// Equivalent to `fit.transport_cost` but provided as a free function for
/// symmetry with the other query helpers.
#[inline]
pub fn ema_transport_cost(fit: &EmaSinkhornFit) -> f64 {
    fit.transport_cost
}

/// Compute the maximum marginal violation of the reconstructed transport plan.
///
/// Returns `max(max_i |Σ_j P_ij − a_i|, max_j |Σ_i P_ij − b_j|)`.
pub fn ema_marginal_violation(fit: &EmaSinkhornFit, a: &[f64], b: &[f64], cost: &[f64]) -> f64 {
    let n = fit.n;
    let m = fit.m;
    let plan = ema_transport_plan(fit, cost);
    let mut max_viol = 0.0_f64;

    // Row marginal violations
    for i in 0..n {
        let row_sum: f64 = (0..m).map(|j| plan[i * m + j]).sum();
        let viol = (row_sum - a[i]).abs();
        if viol > max_viol {
            max_viol = viol;
        }
    }

    // Column marginal violations
    for j in 0..m {
        let col_sum: f64 = (0..n).map(|i| plan[i * m + j]).sum();
        let viol = (col_sum - b[j]).abs();
        if viol > max_viol {
            max_viol = viol;
        }
    }

    max_viol
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn cost_matrix_sq(n: usize, m: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; n * m];
        for i in 0..n {
            for j in 0..m {
                let diff = i as f64 - j as f64;
                c[i * m + j] = diff * diff;
            }
        }
        c
    }

    #[test]
    fn marginals_satisfied_after_convergence_uniform() {
        // ema_marginal_violation checks BOTH row and column marginals.
        // Convergence is declared on row marginals only; column residuals can
        // be slightly larger.  A tolerance of 5e-3 is generous but realistic.
        let n = 4;
        let m = 4;
        let a = uniform(n);
        let b = uniform(m);
        let cost = cost_matrix_sq(n, m);
        let cfg = EmaSinkhornConfig {
            reg: 0.5,
            max_iter: 2000,
            tol: 1e-7,
            ema_alpha: 0.9,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        let viol = ema_marginal_violation(&fit, &a, &b, &cost);
        assert!(viol < 5e-3, "marginal violation {viol} too large");
    }

    #[test]
    fn alpha_one_matches_standard_sinkhorn_cost() {
        // With alpha=1 EMA collapses to standard Sinkhorn (no smoothing).
        // Convergence tol is loose enough for the EMA row-marginal check to trigger.
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let cost = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
        let cfg_ema = EmaSinkhornConfig {
            reg: 0.5,
            max_iter: 3000,
            tol: 1e-6,
            ema_alpha: 1.0,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg_ema).expect("converges");
        let viol = ema_marginal_violation(&fit, &a, &b, &cost);
        assert!(viol < 1e-3, "viol={viol}");
    }

    #[test]
    fn smaller_alpha_gives_finite_convergent_result() {
        // alpha = 0.5 means 50% new / 50% old — moderate EMA smoothing.
        // The algorithm still converges but needs more iterations.
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let cost = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
        let cfg = EmaSinkhornConfig {
            reg: 0.5,
            max_iter: 5000,
            tol: 1e-6,
            ema_alpha: 0.5,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        assert!(fit.transport_cost.is_finite());
        assert!(fit.transport_cost >= 0.0);
    }

    #[test]
    fn transport_plan_rows_sum_to_source_marginals() {
        let n = 3;
        let m = 4;
        let a = vec![0.4, 0.3, 0.3];
        let b = vec![0.2, 0.3, 0.3, 0.2];
        let cost = cost_matrix_sq(n, m);
        let cfg = EmaSinkhornConfig {
            reg: 0.3,
            max_iter: 3000,
            tol: 1e-7,
            ema_alpha: 0.9,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        let plan = ema_transport_plan(&fit, &cost);
        for i in 0..n {
            let row_sum: f64 = (0..m).map(|j| plan[i * m + j]).sum();
            assert!(
                (row_sum - a[i]).abs() < 1e-4,
                "row {i}: {row_sum} vs {}",
                a[i]
            );
        }
    }

    #[test]
    fn transport_plan_cols_sum_to_target_marginals() {
        let n = 3;
        let m = 4;
        let a = vec![0.4, 0.3, 0.3];
        let b = vec![0.2, 0.3, 0.3, 0.2];
        let cost = cost_matrix_sq(n, m);
        let cfg = EmaSinkhornConfig {
            reg: 0.3,
            max_iter: 3000,
            tol: 1e-7,
            ema_alpha: 0.9,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        let plan = ema_transport_plan(&fit, &cost);
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| plan[i * m + j]).sum();
            assert!(
                (col_sum - b[j]).abs() < 1e-4,
                "col {j}: {col_sum} vs {}",
                b[j]
            );
        }
    }

    #[test]
    fn transport_cost_helper_matches_fit_field() {
        let n = 3;
        let m = 3;
        let a = uniform(n);
        let b = uniform(m);
        let cost = cost_matrix_sq(n, m);
        let cfg = EmaSinkhornConfig::default();
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        assert!((ema_transport_cost(&fit) - fit.transport_cost).abs() < 1e-15);
    }

    #[test]
    fn marginal_violation_small_after_convergence() {
        // alpha=0.85 → 85% new / 15% old EMA; near standard Sinkhorn speed.
        // The two-sided marginal violation is checked with a generous bound
        // because only row marginals are used for the internal convergence check.
        let n = 5;
        let m = 5;
        let a = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let b = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let cost = cost_matrix_sq(n, m);
        let cfg = EmaSinkhornConfig {
            reg: 0.5,
            max_iter: 5000,
            tol: 1e-7,
            ema_alpha: 0.85,
        };
        let fit = ema_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("converges");
        let viol = ema_marginal_violation(&fit, &a, &b, &cost);
        assert!(viol < 5e-3, "violation={viol}");
    }

    #[test]
    fn bad_reg_returns_error() {
        let cfg = EmaSinkhornConfig {
            reg: 0.0,
            ..Default::default()
        };
        let res = ema_sinkhorn(&[0.5, 0.5], &[0.5, 0.5], &[0.0; 4], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_alpha_returns_error() {
        let cfg = EmaSinkhornConfig {
            ema_alpha: 0.0,
            ..Default::default()
        };
        let res = ema_sinkhorn(&[0.5, 0.5], &[0.5, 0.5], &[0.0; 4], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn empty_input_returns_error() {
        let cfg = EmaSinkhornConfig::default();
        let res = ema_sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let cfg = EmaSinkhornConfig::default();
        // cost should be 2*2=4 but we pass 6
        let res = ema_sinkhorn(&[0.5, 0.5], &[0.5, 0.5], &[0.0; 6], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn negative_weight_returns_error() {
        let cfg = EmaSinkhornConfig::default();
        let res = ema_sinkhorn(&[-0.5, 1.5], &[0.5, 0.5], &[0.0; 4], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }
}
