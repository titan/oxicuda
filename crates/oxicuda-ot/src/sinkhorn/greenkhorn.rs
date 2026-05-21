//! Greenkhorn (greedy Sinkhorn) algorithm for entropic optimal transport.
//!
//! Solves
//!
//! ```text
//! min_P  <C, P> + ε · KL(P ‖ a ⊗ b)
//!        s.t.  P 1 = a,  Pᵀ 1 = b,  P ≥ 0
//! ```
//!
//! by greedily updating only the row or column with the maximum marginal
//! violation at each step (Altschuler, Weed, Rigollet 2017).
//!
//! Greenkhorn achieves O(n³/ε²) total arithmetic operations by performing O(n)
//! work per step and O(n²/ε²) steps in the worst case, compared to O(n²) per
//! full sweep for standard Sinkhorn. This implementation runs in log-domain for
//! numerical stability.
//!
//! # Incremental log-space marginal update
//!
//! A single greedy step rescales exactly one row `i` (or one column `j`). A row
//! rescale by additive log-shift `δ` on `log_u[i]` multiplies every entry of
//! row `i` of the plan by `exp(δ)` and leaves every other row untouched. Hence:
//!
//! * `log_r[i]` becomes exactly `log a_i` — an O(1) assignment.
//! * Every `log_c[j]` changes only through its single row-`i` summand. Writing
//!   `c_j = rest_j + contrib_{i,j}` where `rest_j = Σ_{i'≠i} contrib_{i',j}`,
//!   the update is `c_j ← rest_j + exp(δ) · contrib_{i,j}`. Each `log_c[j]` is
//!   refreshed in O(1) via two log-space combinators (`log_sub_exp` to peel
//!   off the stale row-`i` term, `log_add_exp2` to fold in the new one), for
//!   O(n) per step total. No O(n²) marginal recompute is performed inside the
//!   greedy loop.
//!
//! Column rescales are handled symmetrically (O(m) per step).
//!
//! The log-space combinators keep all arithmetic in the log domain so the
//! "subtract the old, add the new" delta never triggers catastrophic
//! cancellation in linear space. A full marginal recompute is still performed
//! every `refresh_freq` steps purely as drift correction: repeated log-domain
//! deltas accumulate rounding error, and the rare numerical case where the
//! peeled-off term is not strictly dominated by its aggregate is reset there.

use crate::error::{OtError, OtResult};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers (mirrored from sinkhorn.rs to keep this module self-contained)
// ──────────────────────────────────────────────────────────────────────────────

/// Clamp-guarded natural logarithm so that `safe_ln(0) = ln(f32::MIN_POSITIVE)`.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Numerically stable log-sum-exp of a slice.
/// Returns `f32::NEG_INFINITY` on an empty slice.
fn logsumexp(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &x in slice {
        if x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }
    let mut sum = 0.0_f32;
    for &x in slice {
        sum += (x - max_val).exp();
    }
    max_val + sum.ln()
}

/// Stable log-sum-exp of exactly two operands: `log(exp(a) + exp(b))`.
///
/// This is the two-argument specialisation of [`logsumexp`]; it is the
/// log-space "add the new contribution" half of the incremental marginal
/// update. Handles `NEG_INFINITY` operands (an absent summand) gracefully.
#[inline]
fn log_add_exp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    // hi + ln(1 + exp(lo - hi)); ln_1p keeps precision when lo ≪ hi.
    hi + (lo - hi).exp().ln_1p()
}

/// Stable log-difference of exponentials: `log(exp(a) - exp(b))` for `a ≥ b`.
///
/// This is the log-space "subtract the old contribution" half of the
/// incremental marginal update — it peels a single non-negative summand out of
/// an aggregate without ever forming the catastrophically-cancelling linear
/// difference `exp(a) - exp(b)`.
///
/// The caller guarantees `a ≥ b` (a summand never exceeds its own aggregate).
/// `b == NEG_INFINITY` (an absent old summand) returns `a` unchanged. Numerical
/// drift can make `a` and `b` cross by a rounding unit; in that degenerate case
/// the difference is treated as zero (`NEG_INFINITY`) and the periodic refresh
/// restores the exact value.
#[inline]
fn log_sub_exp(a: f32, b: f32) -> f32 {
    if b == f32::NEG_INFINITY {
        return a;
    }
    if b >= a {
        // Drift crossed the operands: the residual is numerically zero.
        return f32::NEG_INFINITY;
    }
    // a + ln(1 - exp(b - a)); exp_m1 keeps precision when b - a is near zero.
    let diff = b - a; // strictly negative
    let one_minus = -diff.exp_m1(); // = 1 - exp(diff) ∈ (0, 1]
    if one_minus <= 0.0 {
        return f32::NEG_INFINITY;
    }
    a + one_minus.ln()
}

// ──────────────────────────────────────────────────────────────────────────────
// Public structs
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Greenkhorn solver.
#[derive(Debug, Clone)]
pub struct GreenkhornConfig {
    /// Entropic regularisation strength ε (must be > 0).
    pub eps: f32,
    /// Maximum number of greedy row/column updates.
    pub max_iter: usize,
    /// Maximum marginal-violation convergence threshold.
    pub tol: f32,
    /// Full marginal refresh period (drift correction for the log-space deltas).
    /// Defaults to `m + n`; set to `1` for maximum accuracy at extra cost.
    pub refresh_freq: Option<usize>,
}

impl Default for GreenkhornConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 5000,
            tol: 1e-4,
            refresh_freq: None, // resolved to m + n inside the solver
        }
    }
}

/// Output of the Greenkhorn solver.
#[derive(Debug, Clone)]
pub struct GreenkhornResult {
    /// Transport plan, shape `[m × n]` row-major (length `m·n`).
    pub plan: Vec<f32>,
    /// Transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Number of completed greedy updates.
    pub iters: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &GreenkhornConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if c.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.len() != m || b.len() != n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
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
    let sum_a: f32 = a.iter().sum();
    let sum_b: f32 = b.iter().sum();
    if (sum_a - 1.0).abs() > 0.01 || (sum_b - 1.0).abs() > 0.01 {
        return Err(OtError::MassImbalance { sum_a, sum_b });
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Core solver
// ──────────────────────────────────────────────────────────────────────────────

/// Log-contribution of cell `(i, j)` to the plan: `log_u[i] + log_K[i,j] + log_v[j]`.
#[inline]
fn log_contrib(log_u: &[f32], log_k: &[f32], log_v: &[f32], n: usize, i: usize, j: usize) -> f32 {
    log_u[i] + log_k[i * n + j] + log_v[j]
}

/// Compute log row-marginals: `log_r[i] = logsumexp_j(log_u[i] + log_K[i,j] + log_v[j])`.
///
/// Used once for initialisation and periodically as drift correction; never
/// inside the greedy step, which updates `log_r` incrementally.
fn compute_log_r(log_u: &[f32], log_k: &[f32], log_v: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    let mut log_r = vec![0.0_f32; m];
    for (i, log_r_i) in log_r.iter_mut().enumerate() {
        let row_off = i * n;
        for (j, buf_j) in buf.iter_mut().enumerate() {
            *buf_j = log_u[i] + log_k[row_off + j] + log_v[j];
        }
        *log_r_i = logsumexp(&buf);
    }
    log_r
}

/// Compute log column-marginals: `log_c[j] = logsumexp_i(log_u[i] + log_K[i,j] + log_v[j])`.
///
/// Used once for initialisation and periodically as drift correction; never
/// inside the greedy step, which updates `log_c` incrementally.
fn compute_log_c(log_u: &[f32], log_k: &[f32], log_v: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; m];
    let mut log_c = vec![0.0_f32; n];
    for (j, log_c_j) in log_c.iter_mut().enumerate() {
        for (i, buf_i) in buf.iter_mut().enumerate() {
            *buf_i = log_u[i] + log_k[i * n + j] + log_v[j];
        }
        *log_c_j = logsumexp(&buf);
    }
    log_c
}

/// Incrementally refresh every column marginal after row `i_star` was rescaled.
///
/// `delta` is the additive log-shift just applied to `log_u[i_star]`; `log_u`
/// already holds the post-update value. For each column `j`:
///
/// 1. Recover the row-`i_star` summand *before* the rescale:
///    `log_contrib_old = (log_u[i_star] − δ) + log_K[i_star,j] + log_v[j]`.
/// 2. Peel it out of the aggregate in log-space: `log_rest = log_sub_exp(log_c[j], log_contrib_old)`.
/// 3. Fold the post-rescale summand back in: `log_c[j] = log_add_exp2(log_rest, log_contrib_old + δ)`.
///
/// Pure log-domain, O(n), no catastrophic cancellation.
fn apply_row_update_to_log_c(
    log_c: &mut [f32],
    log_u: &[f32],
    log_k: &[f32],
    log_v: &[f32],
    n: usize,
    i_star: usize,
    delta: f32,
) {
    let log_u_old = log_u[i_star] - delta; // value before the rescale
    let row_off = i_star * n;
    for (j, log_c_j) in log_c.iter_mut().enumerate() {
        let log_contrib_old = log_u_old + log_k[row_off + j] + log_v[j];
        let log_contrib_new = log_contrib_old + delta;
        let log_rest = log_sub_exp(*log_c_j, log_contrib_old);
        *log_c_j = log_add_exp2(log_rest, log_contrib_new);
    }
}

/// Incrementally refresh every row marginal after column `j_star` was rescaled.
///
/// Symmetric counterpart of [`apply_row_update_to_log_c`]; `delta` is the
/// additive log-shift just applied to `log_v[j_star]`. Pure log-domain, O(m).
fn apply_col_update_to_log_r(
    log_r: &mut [f32],
    log_u: &[f32],
    log_k: &[f32],
    log_v: &[f32],
    n: usize,
    j_star: usize,
    delta: f32,
) {
    let log_v_old = log_v[j_star] - delta; // value before the rescale
    for (i, log_r_i) in log_r.iter_mut().enumerate() {
        let log_contrib_old = log_u[i] + log_k[i * n + j_star] + log_v_old;
        let log_contrib_new = log_contrib_old + delta;
        let log_rest = log_sub_exp(*log_r_i, log_contrib_old);
        *log_r_i = log_add_exp2(log_rest, log_contrib_new);
    }
}

/// Run the Greenkhorn (greedy Sinkhorn) algorithm.
///
/// `c` is the cost matrix, shape `[m × n]` row-major. `a` is the source
/// histogram (length `m`), `b` is the target histogram (length `n`). Both must
/// be normalised (sum ≈ 1). The algorithm performs greedy marginal-violation
/// updates until convergence or `cfg.max_iter` updates are exhausted.
///
/// Each greedy step costs O(m + n): the violation scan is O(m + n) and the
/// rescaled row/column propagates to the opposite marginal in O(n)/O(m) via the
/// incremental log-space update. The only O(m·n) work is the initial marginal
/// computation and the periodic `refresh_freq` drift-correction recompute.
pub fn greenkhorn(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &GreenkhornConfig,
) -> OtResult<GreenkhornResult> {
    validate_inputs(c, a, b, m, n, cfg)?;

    let eps = cfg.eps;
    let refresh_freq = cfg.refresh_freq.unwrap_or(m + n).max(1);

    // log-kernel: log_K[i,j] = -C[i,j] / eps
    let log_k: Vec<f32> = c.iter().map(|&cij| -cij / eps).collect();

    // log-dual variables (initialised to 0 → u = v = 1 in linear domain)
    let mut log_u = vec![0.0_f32; m];
    let mut log_v = vec![0.0_f32; n];

    // Pre-compute log target marginals (used repeatedly).
    let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
    let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();

    // Initial full marginal computation (O(m·n), once).
    let mut log_r = compute_log_r(&log_u, &log_k, &log_v, m, n);
    let mut log_c = compute_log_c(&log_u, &log_k, &log_v, m, n);

    let mut completed = 0_usize;

    for step in 0..cfg.max_iter {
        // ── Periodic full recompute: drift correction only ────────────────────
        // The incremental log-space deltas accumulate rounding error over many
        // steps; an occasional exact recompute resets it. This is NOT on the
        // critical per-step path — it fires once every `refresh_freq` steps.
        if step > 0 && step % refresh_freq == 0 {
            log_r = compute_log_r(&log_u, &log_k, &log_v, m, n);
            log_c = compute_log_c(&log_u, &log_k, &log_v, m, n);
        }

        // ── Marginal violations (O(m + n)) ────────────────────────────────────
        // viol_r[i] = |exp(log_r[i]) - a[i]|
        // viol_c[j] = |exp(log_c[j]) - b[j]|
        let mut best_row_viol = 0.0_f32;
        let mut best_row = 0_usize;
        for (i, &log_r_i) in log_r.iter().enumerate() {
            let v = (log_r_i.exp() - a[i]).abs();
            if v > best_row_viol {
                best_row_viol = v;
                best_row = i;
            }
        }

        let mut best_col_viol = 0.0_f32;
        let mut best_col = 0_usize;
        for (j, &log_c_j) in log_c.iter().enumerate() {
            let v = (log_c_j.exp() - b[j]).abs();
            if v > best_col_viol {
                best_col_viol = v;
                best_col = j;
            }
        }

        // ── Convergence check ─────────────────────────────────────────────────
        let max_viol = best_row_viol.max(best_col_viol);
        completed = step + 1;
        if max_viol < cfg.tol {
            break;
        }

        if best_row_viol >= best_col_viol {
            // ── Greedy row update for row i* (O(n)) ──────────────────────────
            let i_star = best_row;

            // delta in log space: log_u[i*] += log(a[i*]) - log_r[i*]
            let delta = log_a[i_star] - log_r[i_star];
            log_u[i_star] += delta;

            // After the rescale, row i* is exactly satisfied (O(1)).
            log_r[i_star] = log_a[i_star];

            // Propagate the row rescale to every column marginal in O(n) via
            // the incremental log-space delta — no O(m·n) recompute.
            apply_row_update_to_log_c(&mut log_c, &log_u, &log_k, &log_v, n, i_star, delta);
        } else {
            // ── Greedy column update for column j* (O(m)) ────────────────────
            let j_star = best_col;

            let delta = log_b[j_star] - log_c[j_star];
            log_v[j_star] += delta;

            // After the rescale, column j* is exactly satisfied (O(1)).
            log_c[j_star] = log_b[j_star];

            // Propagate the column rescale to every row marginal in O(m) via
            // the incremental log-space delta — no O(m·n) recompute.
            apply_col_update_to_log_r(&mut log_r, &log_u, &log_k, &log_v, n, j_star, delta);
        }

        // Not converged on final iteration.
        if completed == cfg.max_iter {
            return Err(OtError::NotConverged {
                iter: cfg.max_iter,
                tol: cfg.tol,
            });
        }
    }

    // ── Extract transport plan and compute primal cost ────────────────────────
    let mut plan = vec![0.0_f32; m * n];
    let mut cost = 0.0_f32;
    for i in 0..m {
        let row_off = i * n;
        for j in 0..n {
            let log_p = log_contrib(&log_u, &log_k, &log_v, n, i, j);
            let p_ij = log_p.exp().max(0.0);
            plan[row_off + j] = p_ij;
            cost += p_ij * c[row_off + j];
        }
    }

    Ok(GreenkhornResult {
        plan,
        cost,
        iters: completed,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};
    use std::fmt::Write as _;
    use std::fs;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    /// Reference Greenkhorn driver that recomputes the *full* `log_r`/`log_c`
    /// marginals every single step (the pre-incremental O(m·n)-per-step
    /// behaviour). Used purely to prove the incremental solver is a bit-for-bit
    /// behavioural equivalent: same plan, same cost, same iteration count.
    fn greenkhorn_full_recompute(
        c: &[f32],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        cfg: &GreenkhornConfig,
    ) -> OtResult<GreenkhornResult> {
        validate_inputs(c, a, b, m, n, cfg)?;
        let eps = cfg.eps;
        let log_k: Vec<f32> = c.iter().map(|&cij| -cij / eps).collect();
        let mut log_u = vec![0.0_f32; m];
        let mut log_v = vec![0.0_f32; n];
        let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
        let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();

        let mut completed = 0_usize;
        for step in 0..cfg.max_iter {
            // Full recompute every step — the behaviour we must reproduce.
            let log_r = compute_log_r(&log_u, &log_k, &log_v, m, n);
            let log_c = compute_log_c(&log_u, &log_k, &log_v, m, n);

            let mut best_row_viol = 0.0_f32;
            let mut best_row = 0_usize;
            for (i, &log_r_i) in log_r.iter().enumerate() {
                let v = (log_r_i.exp() - a[i]).abs();
                if v > best_row_viol {
                    best_row_viol = v;
                    best_row = i;
                }
            }
            let mut best_col_viol = 0.0_f32;
            let mut best_col = 0_usize;
            for (j, &log_c_j) in log_c.iter().enumerate() {
                let v = (log_c_j.exp() - b[j]).abs();
                if v > best_col_viol {
                    best_col_viol = v;
                    best_col = j;
                }
            }

            let max_viol = best_row_viol.max(best_col_viol);
            completed = step + 1;
            if max_viol < cfg.tol {
                break;
            }

            if best_row_viol >= best_col_viol {
                let i_star = best_row;
                let delta = log_a[i_star] - log_r[i_star];
                log_u[i_star] += delta;
            } else {
                let j_star = best_col;
                let delta = log_b[j_star] - log_c[j_star];
                log_v[j_star] += delta;
            }

            if completed == cfg.max_iter {
                return Err(OtError::NotConverged {
                    iter: cfg.max_iter,
                    tol: cfg.tol,
                });
            }
        }

        let mut plan = vec![0.0_f32; m * n];
        let mut cost = 0.0_f32;
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                let log_p = log_contrib(&log_u, &log_k, &log_v, n, i, j);
                let p_ij = log_p.exp().max(0.0);
                plan[row_off + j] = p_ij;
                cost += p_ij * c[row_off + j];
            }
        }
        Ok(GreenkhornResult {
            plan,
            cost,
            iters: completed,
        })
    }

    /// Test 1: 2×2 uniform marginals, uniform cost → all plan entries ≈ 0.25.
    #[test]
    fn uniform_2x2_plan_is_quarter() {
        let m = 2;
        let n = 2;
        let c = vec![1.0_f32; 4]; // uniform cost
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let cfg = GreenkhornConfig {
            eps: 5.0,
            max_iter: 5000,
            tol: 1e-5,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        for &p in &res.plan {
            assert!(approx_eq(p, 0.25, 5e-3), "expected ≈0.25, got {p}");
        }
    }

    /// Test 2: 3×3 zero-diagonal cost → Greenkhorn plan concentrates on diagonal.
    #[test]
    fn identity_cost_3x3_concentrates_diagonal() {
        let m = 3;
        let n = 3;
        #[rustfmt::skip]
        let c = vec![
            0.0_f32, 5.0, 5.0,
            5.0, 0.0, 5.0,
            5.0, 5.0, 0.0,
        ];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = GreenkhornConfig {
            eps: 0.3,
            max_iter: 5000,
            tol: 1e-4,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        for i in 0..3 {
            assert!(
                res.plan[i * n + i] > 0.25,
                "diagonal [{i},{i}] = {} is not dominant",
                res.plan[i * n + i]
            );
        }
    }

    /// Test 3: Empty input (m=0) → EmptyInput error.
    #[test]
    fn empty_input_returns_error() {
        let cfg = GreenkhornConfig::default();
        let err = greenkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(err, Err(OtError::EmptyInput)));
    }

    /// Test 4: eps=0 → BadEpsilon error.
    #[test]
    fn zero_epsilon_returns_bad_epsilon_error() {
        let cfg = GreenkhornConfig {
            eps: 0.0,
            max_iter: 10,
            tol: 1e-3,
            refresh_freq: None,
        };
        let err = greenkhorn(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(err, Err(OtError::BadEpsilon { .. })));
    }

    /// Test 5: Negative weight in source marginal → NegativeWeight error.
    #[test]
    fn negative_weight_returns_error() {
        let cfg = GreenkhornConfig::default();
        let c = vec![0.0_f32; 4];
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32, 0.5];
        let err = greenkhorn(&c, &a, &b, 2, 2, &cfg);
        assert!(matches!(err, Err(OtError::NegativeWeight)));
    }

    /// Test 6: Transport cost is non-negative.
    #[test]
    fn transport_cost_is_non_negative() {
        let m = 3;
        let n = 3;
        let c = vec![1.0_f32, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let a = vec![0.4_f32, 0.35, 0.25];
        let b = vec![0.3_f32, 0.4, 0.3];
        let cfg = GreenkhornConfig {
            eps: 0.2,
            max_iter: 5000,
            tol: 1e-4,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        assert!(
            res.cost >= 0.0,
            "cost must be non-negative, got {}",
            res.cost
        );
    }

    /// Test 7: All plan entries must be non-negative.
    #[test]
    fn plan_entries_are_non_negative() {
        let m = 3;
        let n = 4;
        let c: Vec<f32> = (0..m * n).map(|k| (k as f32) * 0.5).collect();
        let a = vec![0.3_f32, 0.4, 0.3];
        let b = vec![0.2_f32, 0.3, 0.25, 0.25];
        let cfg = GreenkhornConfig {
            eps: 0.1,
            max_iter: 8000,
            tol: 1e-4,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        for (k, &p) in res.plan.iter().enumerate() {
            assert!(p >= 0.0, "plan[{k}] = {p} is negative");
        }
    }

    /// Test 8: Row marginals of the plan approximate source histogram `a`.
    #[test]
    fn row_marginals_match_source() {
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.4_f32, 0.4, 0.2];
        let cfg = GreenkhornConfig {
            eps: 0.3,
            max_iter: 5000,
            tol: 1e-4,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        for (i, &ai) in a.iter().enumerate() {
            let row_sum: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(
                approx_eq(row_sum, ai, 5e-3),
                "row {i} sum {row_sum} ≠ a[{i}]={ai}"
            );
        }
    }

    /// Test 9: Column marginals of the plan approximate target histogram `b`.
    #[test]
    fn col_marginals_match_target() {
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.3, 0.2];
        let b = vec![0.4_f32, 0.4, 0.2];
        let cfg = GreenkhornConfig {
            eps: 0.3,
            max_iter: 5000,
            tol: 1e-4,
            refresh_freq: None,
        };
        let res = greenkhorn(&c, &a, &b, m, n, &cfg).expect("should converge");
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(
                approx_eq(col_sum, bj, 5e-3),
                "col {j} sum {col_sum} ≠ b[{j}]={bj}"
            );
        }
    }

    /// Test 10: Greenkhorn cost approximates vanilla Sinkhorn cost on a 2×2 problem.
    #[test]
    fn greenkhorn_matches_sinkhorn_cost_on_2x2() {
        let m = 2;
        let n = 2;
        let c = vec![1.0_f32, 3.0, 3.0, 1.0]; // anti-diagonal cost
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let eps = 0.5_f32;
        let tol = 1e-5_f32;

        let gk_cfg = GreenkhornConfig {
            eps,
            max_iter: 5000,
            tol,
            refresh_freq: None,
        };
        let sk_cfg = SinkhornConfig {
            eps,
            max_iter: 2000,
            tol,
        };

        let gk = greenkhorn(&c, &a, &b, m, n, &gk_cfg).expect("greenkhorn converges");
        let sk = sinkhorn(&c, &a, &b, m, n, &sk_cfg).expect("sinkhorn converges");

        assert!(
            approx_eq(gk.cost, sk.cost, 1e-2),
            "greenkhorn cost {} differs too much from sinkhorn cost {}",
            gk.cost,
            sk.cost
        );
    }

    /// Test 11: Cost matrix dimension mismatch → MarginalMismatch error.
    #[test]
    fn cost_shape_mismatch_returns_error() {
        let cfg = GreenkhornConfig::default();
        // c has 6 entries but m*n = 4
        let err = greenkhorn(&[0.0_f32; 6], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(err, Err(OtError::MarginalMismatch { .. })));
    }

    /// Test 12: Unnormalised marginals → MassImbalance error.
    #[test]
    fn unnormalised_marginals_return_mass_imbalance() {
        let cfg = GreenkhornConfig::default();
        let c = vec![0.0_f32; 4];
        let a = vec![0.9_f32, 0.9]; // sum = 1.8 ≠ 1
        let b = vec![0.5_f32, 0.5];
        let err = greenkhorn(&c, &a, &b, 2, 2, &cfg);
        assert!(matches!(err, Err(OtError::MassImbalance { .. })));
    }

    // ── log-space combinator unit tests ──────────────────────────────────────

    /// `log_add_exp2` equals the naive `log(exp(a)+exp(b))` for well-scaled inputs.
    #[test]
    fn log_add_exp2_matches_naive() {
        let cases = [(0.0_f32, 0.0_f32), (1.5, -2.0), (-3.0, -3.0), (2.0, 7.0)];
        for &(a, b) in &cases {
            let naive = (a.exp() + b.exp()).ln();
            let got = log_add_exp2(a, b);
            assert!(
                (naive - got).abs() < 1e-5,
                "log_add_exp2({a},{b}) = {got}, naive = {naive}"
            );
        }
    }

    /// `log_add_exp2` treats `NEG_INFINITY` as an absent (zero) summand.
    #[test]
    fn log_add_exp2_handles_neg_infinity() {
        assert_eq!(log_add_exp2(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(log_add_exp2(2.0, f32::NEG_INFINITY), 2.0);
        assert_eq!(
            log_add_exp2(f32::NEG_INFINITY, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
    }

    /// `log_sub_exp` equals the naive `log(exp(a)-exp(b))` for `a > b`.
    #[test]
    fn log_sub_exp_matches_naive() {
        let cases = [(0.0_f32, -1.0_f32), (3.0, 1.0), (-1.0, -4.0), (5.0, 4.9)];
        for &(a, b) in &cases {
            let naive = (a.exp() - b.exp()).ln();
            let got = log_sub_exp(a, b);
            assert!(
                (naive - got).abs() < 1e-4,
                "log_sub_exp({a},{b}) = {got}, naive = {naive}"
            );
        }
    }

    /// `log_sub_exp` peeling a summand then re-adding it is the identity.
    #[test]
    fn log_sub_then_add_is_identity() {
        // aggregate = log(exp(s1)+exp(s2)+exp(s3)); peel s2, fold it back.
        let s1 = 0.7_f32;
        let s2 = -1.3_f32;
        let s3 = 2.1_f32;
        let agg = log_add_exp2(log_add_exp2(s1, s2), s3);
        let rest = log_sub_exp(agg, s2);
        let restored = log_add_exp2(rest, s2);
        assert!(
            (restored - agg).abs() < 1e-4,
            "peel/fold not identity: agg={agg}, restored={restored}"
        );
    }

    /// `log_sub_exp` returns `NEG_INFINITY` when operands are equal or crossed
    /// (the degenerate "residual is numerically zero" path).
    #[test]
    fn log_sub_exp_equal_operands_is_neg_inf() {
        assert_eq!(log_sub_exp(2.0, 2.0), f32::NEG_INFINITY);
        assert_eq!(log_sub_exp(1.0, 1.5), f32::NEG_INFINITY);
        assert_eq!(log_sub_exp(3.0, f32::NEG_INFINITY), 3.0);
    }

    // ── convergence-parity tests: incremental ≡ full-recompute behaviour ──────

    /// The incremental solver must produce the *same* iteration count, cost and
    /// transport plan as the full-recompute reference across several problem
    /// sizes — proving only the complexity changed, not the behaviour.
    #[test]
    fn incremental_matches_full_recompute_parity() {
        // (m, n, eps, seed) test matrix spanning square and rectangular shapes.
        let shapes = [(2_usize, 2_usize), (3, 3), (4, 6), (5, 5), (7, 4), (8, 8)];
        for &(m, n) in &shapes {
            // Deterministic pseudo-random cost matrix.
            let mut state = 0x2545_F491_u32 ^ ((m as u32) << 16) ^ (n as u32);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32) / (u32::MAX as f32)
            };
            let c: Vec<f32> = (0..m * n).map(|_| next() * 4.0).collect();
            // Normalised, strictly-positive marginals.
            let mut a: Vec<f32> = (0..m).map(|_| next() + 0.1).collect();
            let mut b: Vec<f32> = (0..n).map(|_| next() + 0.1).collect();
            let sa: f32 = a.iter().sum();
            let sb: f32 = b.iter().sum();
            for ai in &mut a {
                *ai /= sa;
            }
            for bj in &mut b {
                *bj /= sb;
            }

            let cfg = GreenkhornConfig {
                eps: 0.5,
                max_iter: 20_000,
                tol: 1e-4,
                // refresh every step → incremental path is a pure equivalent
                // of the full recompute (no drift-correction divergence).
                refresh_freq: Some(1),
            };

            let inc = greenkhorn(&c, &a, &b, m, n, &cfg)
                .unwrap_or_else(|e| panic!("incremental {m}x{n} failed: {e}"));
            let full = greenkhorn_full_recompute(&c, &a, &b, m, n, &cfg)
                .unwrap_or_else(|e| panic!("full-recompute {m}x{n} failed: {e}"));

            assert_eq!(
                inc.iters, full.iters,
                "{m}x{n}: iteration count diverged (inc={}, full={})",
                inc.iters, full.iters
            );
            assert!(
                approx_eq(inc.cost, full.cost, 1e-4),
                "{m}x{n}: cost diverged (inc={}, full={})",
                inc.cost,
                full.cost
            );
            assert_eq!(
                inc.plan.len(),
                full.plan.len(),
                "{m}x{n}: plan length mismatch"
            );
            for k in 0..inc.plan.len() {
                assert!(
                    approx_eq(inc.plan[k], full.plan[k], 1e-4),
                    "{m}x{n}: plan[{k}] diverged (inc={}, full={})",
                    inc.plan[k],
                    full.plan[k]
                );
            }
        }
    }

    /// With `refresh_freq` large (rare drift correction), the incremental solver
    /// must still converge to essentially the same plan as the per-step
    /// full-recompute reference — the log-space deltas do not introduce
    /// behavioural drift.
    #[test]
    fn incremental_matches_full_recompute_with_sparse_refresh() {
        let m = 6;
        let n = 5;
        #[rustfmt::skip]
        let c = vec![
            0.0_f32, 1.0, 2.0, 3.0, 4.0,
            1.0,     0.0, 1.0, 2.0, 3.0,
            2.0,     1.0, 0.0, 1.0, 2.0,
            3.0,     2.0, 1.0, 0.0, 1.0,
            4.0,     3.0, 2.0, 1.0, 0.0,
            2.5,     1.5, 0.5, 1.5, 2.5,
        ];
        let a = vec![0.2_f32, 0.15, 0.2, 0.15, 0.15, 0.15];
        let b = vec![0.25_f32, 0.2, 0.2, 0.2, 0.15];

        let cfg_inc = GreenkhornConfig {
            eps: 0.4,
            max_iter: 20_000,
            tol: 1e-4,
            refresh_freq: Some(10_000), // effectively no drift refresh
        };
        let cfg_ref = GreenkhornConfig {
            refresh_freq: Some(1),
            ..cfg_inc.clone()
        };

        let inc = greenkhorn(&c, &a, &b, m, n, &cfg_inc).expect("incremental converges");
        let full = greenkhorn_full_recompute(&c, &a, &b, m, n, &cfg_ref)
            .expect("full-recompute converges");

        // Cost and plan agree to log-domain f32 precision.
        assert!(
            approx_eq(inc.cost, full.cost, 5e-3),
            "cost diverged: inc={}, full={}",
            inc.cost,
            full.cost
        );
        for k in 0..inc.plan.len() {
            assert!(
                approx_eq(inc.plan[k], full.plan[k], 5e-3),
                "plan[{k}] diverged: inc={}, full={}",
                inc.plan[k],
                full.plan[k]
            );
        }
    }

    /// Drift check: over many incremental iterations *without* periodic refresh,
    /// the running `log_r`/`log_c` marginals must stay equal to a from-scratch
    /// recompute. This guards the log-space delta arithmetic against
    /// subtractive-cancellation drift.
    #[test]
    fn incremental_marginals_stay_correct_over_many_iterations() {
        let m = 7;
        let n = 7;
        let eps = 0.35_f32;
        let mut state = 0x9E37_79B9_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32) / (u32::MAX as f32)
        };
        let c: Vec<f32> = (0..m * n).map(|_| next() * 3.0).collect();
        let mut a: Vec<f32> = (0..m).map(|_| next() + 0.2).collect();
        let mut b: Vec<f32> = (0..n).map(|_| next() + 0.2).collect();
        let sa: f32 = a.iter().sum();
        let sb: f32 = b.iter().sum();
        for ai in &mut a {
            *ai /= sa;
        }
        for bj in &mut b {
            *bj /= sb;
        }

        let log_k: Vec<f32> = c.iter().map(|&cij| -cij / eps).collect();
        let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
        let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();
        let mut log_u = vec![0.0_f32; m];
        let mut log_v = vec![0.0_f32; n];

        // Incrementally-maintained marginals (no periodic refresh at all).
        let mut log_r = compute_log_r(&log_u, &log_k, &log_v, m, n);
        let mut log_c = compute_log_c(&log_u, &log_k, &log_v, m, n);

        // Run many greedy steps; after each, the incremental marginals must
        // still equal a fresh full recompute within tight tolerance.
        for step in 0..400 {
            let mut best_row_viol = 0.0_f32;
            let mut best_row = 0_usize;
            for (i, &lr) in log_r.iter().enumerate() {
                let v = (lr.exp() - a[i]).abs();
                if v > best_row_viol {
                    best_row_viol = v;
                    best_row = i;
                }
            }
            let mut best_col_viol = 0.0_f32;
            let mut best_col = 0_usize;
            for (j, &lc) in log_c.iter().enumerate() {
                let v = (lc.exp() - b[j]).abs();
                if v > best_col_viol {
                    best_col_viol = v;
                    best_col = j;
                }
            }
            if best_row_viol.max(best_col_viol) < 1e-6 {
                break;
            }

            if best_row_viol >= best_col_viol {
                let i_star = best_row;
                let delta = log_a[i_star] - log_r[i_star];
                log_u[i_star] += delta;
                log_r[i_star] = log_a[i_star];
                apply_row_update_to_log_c(&mut log_c, &log_u, &log_k, &log_v, n, i_star, delta);
            } else {
                let j_star = best_col;
                let delta = log_b[j_star] - log_c[j_star];
                log_v[j_star] += delta;
                log_c[j_star] = log_b[j_star];
                apply_col_update_to_log_r(&mut log_r, &log_u, &log_k, &log_v, n, j_star, delta);
            }

            // Drift assertion: incremental marginals vs. exact recompute.
            let ref_r = compute_log_r(&log_u, &log_k, &log_v, m, n);
            let ref_c = compute_log_c(&log_u, &log_k, &log_v, m, n);
            for i in 0..m {
                let drift = (log_r[i].exp() - ref_r[i].exp()).abs();
                assert!(
                    drift < 1e-4,
                    "step {step}: log_r[{i}] drifted by {drift} \
                     (incremental={}, exact={})",
                    log_r[i].exp(),
                    ref_r[i].exp()
                );
            }
            for j in 0..n {
                let drift = (log_c[j].exp() - ref_c[j].exp()).abs();
                assert!(
                    drift < 1e-4,
                    "step {step}: log_c[{j}] drifted by {drift} \
                     (incremental={}, exact={})",
                    log_c[j].exp(),
                    ref_c[j].exp()
                );
            }
        }
    }

    /// File-I/O smoke test: serialise the incremental and full-recompute results
    /// to a temp file and confirm they read back identical. Uses
    /// `std::env::temp_dir()` per the project test-I/O policy.
    #[test]
    fn parity_report_roundtrips_through_temp_file() {
        let m = 4;
        let n = 4;
        let c = vec![
            0.0_f32, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0,
        ];
        let a = vec![0.25_f32; 4];
        let b = vec![0.25_f32; 4];
        let cfg = GreenkhornConfig {
            eps: 0.5,
            max_iter: 20_000,
            tol: 1e-4,
            refresh_freq: Some(1),
        };

        let inc = greenkhorn(&c, &a, &b, m, n, &cfg).expect("incremental converges");
        let full =
            greenkhorn_full_recompute(&c, &a, &b, m, n, &cfg).expect("full-recompute converges");

        let mut report = String::new();
        writeln!(report, "iters_inc={}", inc.iters).expect("write");
        writeln!(report, "iters_full={}", full.iters).expect("write");
        writeln!(report, "cost_inc={:.6}", inc.cost).expect("write");
        writeln!(report, "cost_full={:.6}", full.cost).expect("write");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxicuda_greenkhorn_parity_{}.txt",
            std::process::id()
        ));
        fs::write(&path, &report).expect("write temp report");
        let read_back = fs::read_to_string(&path).expect("read temp report");
        let _ = fs::remove_file(&path);

        assert_eq!(read_back, report, "temp-file report roundtrip mismatch");
        assert!(read_back.contains(&format!("iters_inc={}", full.iters)));
    }
}
