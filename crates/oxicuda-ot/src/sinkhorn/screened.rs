//! Screened Sinkhorn — sparsity-inducing optimal transport (Alaya et al. 2019).
//!
//! Runs the standard log-domain Sinkhorn algorithm but prunes transport
//! edges below a screening threshold to achieve a sparse approximate plan.
//! Subsequent iterations operate only on the active (un-pruned) pairs,
//! reducing per-iteration cost from O(nm) to O(s) where s is the active-set
//! size.
//!
//! # Reference
//! Alaya, Berar, Gasso, Rakotomamonjy (2019). "Screening Sinkhorn Algorithm
//! for Regularized Optimal Transport." NeurIPS 2019.

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration / Result structs
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the screened Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct ScreenedConfig {
    /// Entropic regularisation strength (must be > 0).
    pub reg: f64,
    /// Minimum transport value below which pairs are pruned (≥ 0).
    pub screen_threshold: f64,
    /// Maximum number of Sinkhorn iterations.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance.
    pub tol: f64,
}

impl Default for ScreenedConfig {
    fn default() -> Self {
        Self {
            reg: 0.1,
            screen_threshold: 1e-8,
            max_iter: 500,
            tol: 1e-5,
        }
    }
}

/// Output of the screened Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct ScreenedFit {
    /// Log row-dual potentials, length `n`.
    pub log_u: Vec<f64>,
    /// Log column-dual potentials, length `m`.
    pub log_v: Vec<f64>,
    /// Active (i, j) pairs after screening.
    pub active_pairs: Vec<(usize, usize)>,
    /// Cost matrix slice at active pairs (same order as `active_pairs`).
    pub cost: Vec<f64>,
    /// Number of source points.
    pub n: usize,
    /// Number of target points.
    pub m: usize,
    /// Regularisation parameter used during the solve (stored for derived queries).
    pub reg: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn safe_ln64(x: f64) -> f64 {
    if x <= f64::MIN_POSITIVE {
        f64::MIN_POSITIVE.ln()
    } else {
        x.ln()
    }
}

/// Stable log-sum-exp of a slice (f64).
fn logsumexp64(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    max_val + slice.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    cfg: &ScreenedConfig,
) -> OtResult<()> {
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
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
    if a.len() != n || b.len() != m {
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

/// Run the screened Sinkhorn algorithm.
///
/// Starts with full log-domain Sinkhorn, then progressively screens (prunes)
/// pairs whose estimated transport value is below `cfg.screen_threshold`.
/// Subsequent iterations work only on the active set, achieving sparsity.
pub fn screened_sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[f64],
    n: usize,
    m: usize,
    cfg: &ScreenedConfig,
) -> OtResult<ScreenedFit> {
    validate_inputs(cost, a, b, n, m, cfg)?;

    let reg = cfg.reg;
    let threshold = cfg.screen_threshold.max(0.0);

    // ── Initialise log-dual potentials ────────────────────────────────────────
    let mut log_u: Vec<f64> = a.iter().map(|&ai| safe_ln64(ai)).collect();
    let mut log_v: Vec<f64> = b.iter().map(|&bj| safe_ln64(bj)).collect();

    // ── Initial active set: all n×m pairs ─────────────────────────────────────
    // Represent via per-row and per-column active-j / active-i lists.
    // For efficiency we store row-indexed: active_cols_for_row[i] = sorted vec of j
    let mut active_cols: Vec<Vec<usize>> = (0..n).map(|_| (0..m).collect()).collect();
    let mut active_rows: Vec<Vec<usize>> = (0..m).map(|_| (0..n).collect()).collect();

    // ── Screening phase: first warm-up with full Sinkhorn ─────────────────────
    // Run a few full iterations before screening to get reasonable potentials.
    let warmup = cfg.max_iter.min(20);
    let mut buf_row = vec![0.0_f64; m];
    let mut buf_col = vec![0.0_f64; n];

    for _it in 0..warmup {
        // Row update (full)
        for i in 0..n {
            for j in 0..m {
                buf_row[j] = log_v[j] - cost[i * m + j] / reg;
            }
            let lse = logsumexp64(&buf_row[..m]);
            log_u[i] = safe_ln64(a[i]) - lse;
        }
        // Col update (full)
        for j in 0..m {
            for i in 0..n {
                buf_col[i] = log_u[i] - cost[i * m + j] / reg;
            }
            let lse = logsumexp64(&buf_col[..n]);
            log_v[j] = safe_ln64(b[j]) - lse;
        }
    }

    // ── Screen: remove pairs with transport P[i,j] < threshold ───────────────
    if threshold > 0.0 {
        let log_thresh = safe_ln64(threshold);
        for i in 0..n {
            active_cols[i].retain(|&j| {
                let log_pij = log_u[i] + log_v[j] - cost[i * m + j] / reg;
                log_pij >= log_thresh
            });
        }
        for j in 0..m {
            active_rows[j].retain(|&i| {
                let log_pij = log_u[i] + log_v[j] - cost[i * m + j] / reg;
                log_pij >= log_thresh
            });
        }
    }

    // ── Main screened Sinkhorn iterations ─────────────────────────────────────
    for it in warmup..cfg.max_iter {
        let old_log_u = log_u.clone();
        let old_log_v = log_v.clone();

        // Row update (active set only)
        for i in 0..n {
            let cols = &active_cols[i];
            if cols.is_empty() {
                continue;
            }
            let buf: Vec<f64> = cols
                .iter()
                .map(|&j| log_v[j] - cost[i * m + j] / reg)
                .collect();
            let lse = logsumexp64(&buf);
            log_u[i] = safe_ln64(a[i]) - lse;
        }

        // Col update (active set only)
        for j in 0..m {
            let rows = &active_rows[j];
            if rows.is_empty() {
                continue;
            }
            let buf: Vec<f64> = rows
                .iter()
                .map(|&i| log_u[i] - cost[i * m + j] / reg)
                .collect();
            let lse = logsumexp64(&buf);
            log_v[j] = safe_ln64(b[j]) - lse;
        }

        // ── Periodic re-screening ─────────────────────────────────────────────
        // Re-screen every 10 iterations after warmup.
        if threshold > 0.0 && it % 10 == 0 {
            let log_thresh = safe_ln64(threshold);
            for i in 0..n {
                active_cols[i].retain(|&j| {
                    let log_pij = log_u[i] + log_v[j] - cost[i * m + j] / reg;
                    log_pij >= log_thresh
                });
            }
            for j in 0..m {
                active_rows[j].retain(|&i| {
                    let log_pij = log_u[i] + log_v[j] - cost[i * m + j] / reg;
                    log_pij >= log_thresh
                });
            }
        }

        // ── Convergence check ─────────────────────────────────────────────────
        let max_du = log_u
            .iter()
            .zip(old_log_u.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_dv = log_v
            .iter()
            .zip(old_log_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_du.max(max_dv) < cfg.tol {
            break;
        }
    }

    // ── Collect final active pairs ─────────────────────────────────────────────
    let mut active_pairs: Vec<(usize, usize)> = Vec::new();
    let mut cost_at_active: Vec<f64> = Vec::new();

    for i in 0..n {
        for &j in &active_cols[i] {
            active_pairs.push((i, j));
            cost_at_active.push(cost[i * m + j]);
        }
    }

    Ok(ScreenedFit {
        log_u,
        log_v,
        active_pairs,
        cost: cost_at_active,
        n,
        m,
        reg,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Derived-quantity functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the transport cost `<P, C>` summed over active pairs, using the
/// regularisation parameter stored in the fit.
///
/// `P[i,j] = exp(log_u[i] + log_v[j] − cost[i,j] / reg)`
pub fn screened_transport_cost(fit: &ScreenedFit) -> f64 {
    screened_transport_cost_with_reg(fit, fit.reg)
}

/// Compute the transport cost with explicit regularisation parameter.
pub fn screened_transport_cost_with_reg(fit: &ScreenedFit, reg: f64) -> f64 {
    let mut total = 0.0_f64;
    for (&(i, j), &cij) in fit.active_pairs.iter().zip(fit.cost.iter()) {
        let pij = (fit.log_u[i] + fit.log_v[j] - cij / reg).exp();
        total += pij * cij;
    }
    total
}

/// Compute max marginal violation over both source and target.
///
/// Returns `max(‖P 1_m − a‖_∞, ‖Pᵀ 1_n − b‖_∞)`.
/// Uses the regularisation parameter stored in `fit.reg`.
pub fn screened_marginal_violation(fit: &ScreenedFit, a: &[f64], b: &[f64]) -> f64 {
    let n = fit.n;
    let m = fit.m;
    let reg = fit.reg;
    let mut row_sums = vec![0.0_f64; n];
    let mut col_sums = vec![0.0_f64; m];
    for (&(i, j), &cij) in fit.active_pairs.iter().zip(fit.cost.iter()) {
        let pij = (fit.log_u[i] + fit.log_v[j] - cij / reg).exp();
        row_sums[i] += pij;
        col_sums[j] += pij;
    }
    let max_row = row_sums
        .iter()
        .zip(a.iter())
        .map(|(&rs, &ai)| (rs - ai).abs())
        .fold(0.0_f64, f64::max);
    let max_col = col_sums
        .iter()
        .zip(b.iter())
        .map(|(&cs, &bj)| (cs - bj).abs())
        .fold(0.0_f64, f64::max);
    max_row.max(max_col)
}

/// Fraction of active pairs relative to the full n×m grid.
pub fn screened_sparsity(fit: &ScreenedFit) -> f64 {
    let total = fit.n * fit.m;
    if total == 0 {
        return 0.0;
    }
    fit.active_pairs.len() as f64 / total as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_marginals(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn diagonal_cost(n: usize) -> Vec<f64> {
        // Low cost on diagonal, high off-diagonal
        (0..n * n)
            .map(|k| if k / n == k % n { 0.0 } else { 10.0 })
            .collect()
    }

    fn small_cost() -> Vec<f64> {
        // 3×3 symmetric cost
        vec![0.0_f64, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]
    }

    /// Test 1: Transport cost is finite.
    #[test]
    fn transport_cost_is_finite() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = small_cost();
        let cfg = ScreenedConfig {
            reg: 0.2,
            screen_threshold: 0.0,
            max_iter: 200,
            tol: 1e-5,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let tc = screened_transport_cost_with_reg(&fit, 0.2);
        assert!(tc.is_finite(), "transport cost must be finite: {tc}");
    }

    /// Test 2: Screened result has fewer active pairs with positive threshold.
    #[test]
    fn positive_threshold_reduces_active_pairs() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = diagonal_cost(n);
        let full_cfg = ScreenedConfig {
            reg: 0.1,
            screen_threshold: 0.0,
            max_iter: 100,
            tol: 1e-5,
        };
        let screened_cfg = ScreenedConfig {
            reg: 0.1,
            screen_threshold: 1e-4,
            max_iter: 100,
            tol: 1e-5,
        };
        let full_fit = screened_sinkhorn(&a, &b, &cost, n, m, &full_cfg).expect("ok");
        let screened_fit = screened_sinkhorn(&a, &b, &cost, n, m, &screened_cfg).expect("ok");
        assert!(
            screened_fit.active_pairs.len() <= full_fit.active_pairs.len(),
            "screened should have ≤ active pairs: {} vs {}",
            screened_fit.active_pairs.len(),
            full_fit.active_pairs.len()
        );
    }

    /// Test 3: sparsity() is in [0, 1].
    #[test]
    fn sparsity_in_unit_interval() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = small_cost();
        let cfg = ScreenedConfig {
            reg: 0.2,
            screen_threshold: 1e-6,
            max_iter: 100,
            tol: 1e-5,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let s = screened_sparsity(&fit);
        assert!((0.0..=1.0).contains(&s), "sparsity {s} out of [0,1]");
    }

    /// Test 4: Marginal violation < 0.05 for small example.
    #[test]
    fn marginal_violation_small_for_simple_example() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = small_cost();
        let cfg = ScreenedConfig {
            reg: 0.3,
            screen_threshold: 0.0,
            max_iter: 500,
            tol: 1e-7,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let mv = screened_marginal_violation(&fit, &a, &b);
        assert!(mv < 0.05, "marginal violation {mv} should be < 0.05");
    }

    /// Test 5: screen_threshold=0 keeps all n*m pairs active.
    #[test]
    fn zero_threshold_keeps_all_pairs() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = small_cost();
        let cfg = ScreenedConfig {
            reg: 0.2,
            screen_threshold: 0.0,
            max_iter: 100,
            tol: 1e-5,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        assert_eq!(
            fit.active_pairs.len(),
            n * m,
            "zero threshold should keep all {} pairs",
            n * m
        );
    }

    /// Test 6: Empty input rejected.
    #[test]
    fn empty_input_rejected() {
        let cfg = ScreenedConfig::default();
        let res = screened_sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    /// Test 7: Negative regularisation rejected.
    #[test]
    fn negative_reg_rejected() {
        let n = 2;
        let a = uniform_marginals(n);
        let b = uniform_marginals(n);
        let cost = vec![0.0_f64; 4];
        let cfg = ScreenedConfig {
            reg: -0.1,
            screen_threshold: 0.0,
            max_iter: 10,
            tol: 1e-5,
        };
        let res = screened_sinkhorn(&a, &b, &cost, n, n, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    /// Test 8: Negative weight rejected.
    #[test]
    fn negative_weight_rejected() {
        let cfg = ScreenedConfig::default();
        let cost = vec![0.0_f64; 4];
        let a = vec![-0.5_f64, 1.5];
        let b = vec![0.5_f64, 0.5];
        let res = screened_sinkhorn(&a, &b, &cost, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    /// Test 9: ScreenedFit fields have consistent dimensions.
    #[test]
    fn fit_dimensions_consistent() {
        let n = 3;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost: Vec<f64> = (0..(n * m)).map(|k| k as f64 * 0.1).collect();
        let cfg = ScreenedConfig {
            reg: 0.2,
            screen_threshold: 0.0,
            max_iter: 100,
            tol: 1e-5,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        assert_eq!(fit.log_u.len(), n);
        assert_eq!(fit.log_v.len(), m);
        assert_eq!(fit.n, n);
        assert_eq!(fit.m, m);
        assert_eq!(fit.active_pairs.len(), fit.cost.len());
    }

    /// Test 10: Sparsity decreases as threshold increases.
    #[test]
    fn sparsity_decreases_with_higher_threshold() {
        let n = 4;
        let m = 4;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = diagonal_cost(n);
        let cfg_lo = ScreenedConfig {
            reg: 0.1,
            screen_threshold: 1e-8,
            max_iter: 100,
            tol: 1e-5,
        };
        let cfg_hi = ScreenedConfig {
            reg: 0.1,
            screen_threshold: 1e-3,
            max_iter: 100,
            tol: 1e-5,
        };
        let fit_lo = screened_sinkhorn(&a, &b, &cost, n, m, &cfg_lo).expect("ok");
        let fit_hi = screened_sinkhorn(&a, &b, &cost, n, m, &cfg_hi).expect("ok");
        let s_lo = screened_sparsity(&fit_lo);
        let s_hi = screened_sparsity(&fit_hi);
        assert!(
            s_hi <= s_lo + 0.01,
            "higher threshold {s_hi} should give ≤ sparsity than lower {s_lo}"
        );
    }

    /// Test 11: Transport cost with reg parameter works.
    #[test]
    fn transport_cost_with_reg_is_positive() {
        let n = 3;
        let m = 3;
        let a = uniform_marginals(n);
        let b = uniform_marginals(m);
        let cost = small_cost();
        let reg = 0.3_f64;
        let cfg = ScreenedConfig {
            reg,
            screen_threshold: 0.0,
            max_iter: 100,
            tol: 1e-5,
        };
        let fit = screened_sinkhorn(&a, &b, &cost, n, m, &cfg).expect("ok");
        let tc = screened_transport_cost_with_reg(&fit, reg);
        assert!(tc >= 0.0, "transport cost must be non-negative: {tc}");
    }
}
