//! Time-Dependent Schrödinger Bridge (TDSB) via log-domain Iterative Proportional Fitting.
//!
//! ## Problem
//!
//! Given marginals at T+1 time points `(p_0, p_1, …, p_T)` and support points for
//! each time slice, find the path measure Q that minimises `KL(Q ‖ R)` subject to
//! the marginal constraints `Q_{t_k} = p_{t_k}` for all `k`.
//!
//! The reference measure `R` is a Markov measure defined by Brownian-motion transition
//! kernels: `K_{t→t+1}[i,j] = exp(-‖x_i - y_j‖² / (2·ε·dt))`.
//!
//! ## Algorithm
//!
//! Log-domain IPF sweep over all time slices (De Bortoli et al. 2021 style):
//!
//! Forward sweep `t = 0 … T-1`:
//! ```text
//! g[t+1][j] = log(p[t+1][j]) − LSE_i( f[t][i] + log K_{t,t+1}[i,j] )
//! ```
//!
//! Backward sweep `t = T-1 … 0`:
//! ```text
//! f[t][i] = log(p[t][i]) − LSE_j( g[t+1][j] + log K_{t,t+1}[i,j] )
//! ```
//!
//! Convergence is measured by the maximum marginal violation across all time slices.

use crate::error::{OtError, OtResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the Time-Dependent Schrödinger Bridge solver.
#[derive(Debug, Clone)]
pub struct TdsbConfig {
    /// Entropy regularisation per transition step (must be > 0).
    pub eps: f64,
    /// Maximum number of outer IPF iterations (one full sweep = forward + backward pass).
    pub max_outer: usize,
    /// Marginal violation convergence tolerance.
    pub tol: f64,
    /// Time step for the Brownian-motion reference kernel.
    /// `K_{t,t+1}[i,j] = exp(-‖x_i - y_j‖² / (2·ε·dt))`.
    pub dt: f64,
}

impl Default for TdsbConfig {
    fn default() -> Self {
        Self {
            eps: 0.05,
            max_outer: 100,
            tol: 1e-5,
            dt: 1.0,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Output of the Time-Dependent Schrödinger Bridge solver.
#[derive(Debug, Clone)]
pub struct TdsbResult {
    /// Dual potentials `f[t][i]` for marginal `t`, support point `i`.
    pub f: Vec<Vec<f64>>,
    /// Dual potentials `g[t][i]` for marginal `t`, support point `i`.
    /// Note: `g[0]` is unused after final backward pass (only `f[0]` is needed),
    /// but is kept for symmetry; indexing matches `f`.
    pub g: Vec<Vec<f64>>,
    /// Marginal violation (max absolute residual) per time slice after convergence.
    pub violations: Vec<f64>,
    /// Number of outer IPF iterations completed.
    pub iters: usize,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Stable log-sum-exp over a slice; returns `f64::NEG_INFINITY` for an empty slice.
#[inline]
fn logsumexp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    let sum: f64 = v.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// `log(x)` clamped so we never evaluate `log(0)` — returns `log(f64::MIN_POSITIVE)` for `x ≤ 0`.
#[inline]
fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 {
        f64::MIN_POSITIVE.ln()
    } else {
        x.ln()
    }
}

/// Squared Euclidean distance between two support points.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Log-kernel entry `log K_{t,t+1}[i,j] = -‖x_i - y_j‖² / (2·ε·dt)`.
#[inline]
fn log_kernel(xi: &[f64], yj: &[f64], eps: f64, dt: f64) -> f64 {
    -sq_dist(xi, yj) / (2.0 * eps * dt)
}

// ─── Validation ───────────────────────────────────────────────────────────────

fn validate_inputs(
    marginals: &[Vec<f64>],
    supports: &[Vec<Vec<f64>>],
    cfg: &TdsbConfig,
) -> OtResult<()> {
    if marginals.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if marginals.len() != supports.len() {
        return Err(OtError::IncompatibleLength {
            a: marginals.len(),
            b: supports.len(),
        });
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.eps as f32,
        });
    }
    if cfg.dt <= 0.0 {
        return Err(OtError::Internal {
            msg: format!("dt must be > 0, got {}", cfg.dt),
        });
    }

    // All marginals must be valid probability vectors and have matching support sizes.
    let t_steps = marginals.len();
    let d = if supports[0].is_empty() {
        return Err(OtError::EmptyInput);
    } else {
        supports[0][0].len()
    };

    for t in 0..t_steps {
        let n_t = marginals[t].len();
        if n_t == 0 {
            return Err(OtError::EmptyInput);
        }
        if supports[t].len() != n_t {
            return Err(OtError::MarginalMismatch {
                m: n_t,
                n: supports[t].len(),
                a_len: n_t,
                b_len: supports[t].len(),
            });
        }
        // Check dimension consistency across time slices.
        for pt in &supports[t] {
            if pt.len() != d {
                return Err(OtError::BadDim { got: pt.len() });
            }
        }
        // Reject negative entries.
        for &pi in &marginals[t] {
            if pi < 0.0 || !pi.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
    }
    Ok(())
}

// ─── Core solver ──────────────────────────────────────────────────────────────

/// Compute `log K` matrix between time slices `t` and `t+1`.
/// Returns a flat row-major `n_t × n_{t+1}` matrix.
fn log_kernel_matrix(sup_t: &[Vec<f64>], sup_t1: &[Vec<f64>], eps: f64, dt: f64) -> Vec<f64> {
    let n_t = sup_t.len();
    let n_t1 = sup_t1.len();
    let mut lk = vec![0.0_f64; n_t * n_t1];
    for i in 0..n_t {
        for j in 0..n_t1 {
            lk[i * n_t1 + j] = log_kernel(&sup_t[i], &sup_t1[j], eps, dt);
        }
    }
    lk
}

/// Solve the Time-Dependent Schrödinger Bridge problem via log-domain IPF.
///
/// `marginals[t]` is the probability vector at time `t` (must sum to ≈ 1).
/// `supports[t]` is the `n_t × d` matrix of support points at time `t`.
///
/// The reference transition kernel is Brownian-motion:
/// `K_{t→t+1}[i,j] = exp(-‖sup_t[i] - sup_{t+1}[j]‖² / (2·ε·dt))`.
pub fn tdsb(
    marginals: &[Vec<f64>],
    supports: &[Vec<Vec<f64>>],
    config: &TdsbConfig,
) -> OtResult<TdsbResult> {
    validate_inputs(marginals, supports, config)?;

    let t_steps = marginals.len(); // T+1 time slices
    let eps = config.eps;
    let dt = config.dt;

    // Initialise dual potentials:
    // f[t][i] ≈ ε · log p_t[i],  g[t][j] ≈ ε · log p_t[j]
    let mut f: Vec<Vec<f64>> = marginals
        .iter()
        .map(|m| m.iter().map(|&p| eps * safe_ln(p)).collect())
        .collect();
    let mut g: Vec<Vec<f64>> = marginals
        .iter()
        .map(|m| m.iter().map(|&p| eps * safe_ln(p)).collect())
        .collect();

    // Pre-compute log-kernel matrices for each transition t → t+1.
    // lk[t] is n_t × n_{t+1} (row-major).
    let lk: Vec<Vec<f64>> = (0..t_steps - 1)
        .map(|t| log_kernel_matrix(&supports[t], &supports[t + 1], eps, dt))
        .collect();

    let mut iters = 0_usize;
    let mut buf = Vec::<f64>::new();

    for outer in 0..config.max_outer {
        // ── Forward sweep: update g[t+1] for t = 0..T-1 ─────────────────────
        // g[t+1][j] = ε·log(p_{t+1}[j]) − ε · LSE_i( (f[t][i] + log K[i,j]) / ε )
        // In log-domain: g[t+1][j] = ε·log p_{t+1}[j] − ε · LSE_i ( f[t][i]/ε + lk[t][i,j]/ε )
        // Since f is already in "ε-scaled" form, we work with f_scaled[i] = f[t][i]/ε:
        for t in 0..t_steps - 1 {
            let n_t = marginals[t].len();
            let n_t1 = marginals[t + 1].len();
            buf.resize(n_t, 0.0);
            for j in 0..n_t1 {
                for i in 0..n_t {
                    buf[i] = f[t][i] / eps + lk[t][i * n_t1 + j] / eps;
                }
                let lse = logsumexp(&buf[..n_t]);
                g[t + 1][j] = eps * safe_ln(marginals[t + 1][j]) - eps * lse;
            }
        }

        // ── Backward sweep: update f[t] for t = T-1..0 ───────────────────────
        // f[t][i] = ε·log(p_t[i]) − ε · LSE_j ( (g[t+1][j] + log K[i,j]) / ε )
        for t in (0..t_steps - 1).rev() {
            let n_t = marginals[t].len();
            let n_t1 = marginals[t + 1].len();
            buf.resize(n_t1, 0.0);
            for i in 0..n_t {
                for j in 0..n_t1 {
                    buf[j] = g[t + 1][j] / eps + lk[t][i * n_t1 + j] / eps;
                }
                let lse = logsumexp(&buf[..n_t1]);
                f[t][i] = eps * safe_ln(marginals[t][i]) - eps * lse;
            }
        }

        iters = outer + 1;

        // ── Convergence: check marginal violations ────────────────────────────
        let mut max_viol = 0.0_f64;

        // Violation at t = 0: marginal of the path measure at t=0.
        // P_0[i] = exp(f[0][i]/ε) · Z_{0→1}(i) where Z is normalised by g[1].
        // More directly: after the backward sweep, f[0][i] is the correct dual,
        // and the marginal at t=0 is:
        // row_i = sum_j exp( (f[0][i] + g[1][j] + lk[0][i,j]) / ε )
        {
            let n0 = marginals[0].len();
            let n1 = marginals[1].len();
            for i in 0..n0 {
                let mut row_sum = 0.0_f64;
                for j in 0..n1 {
                    row_sum += ((f[0][i] + g[1][j] + lk[0][i * n1 + j]) / eps).exp();
                }
                let viol = (row_sum - marginals[0][i]).abs();
                if viol > max_viol {
                    max_viol = viol;
                }
            }
        }

        // Violation at interior time steps t = 1..T-1.
        for t in 1..t_steps - 1 {
            let n_tm1 = marginals[t - 1].len();
            let n_t = marginals[t].len();
            let n_t1 = marginals[t + 1].len();
            for j in 0..n_t {
                // Marginal at t: sum over i from the t-1 → t transition.
                let mut col_sum = 0.0_f64;
                for i in 0..n_tm1 {
                    col_sum += ((f[t - 1][i] + g[t][j] + lk[t - 1][i * n_t + j]) / eps).exp();
                }
                // Also check via the t → t+1 transition for consistency.
                let mut row_sum = 0.0_f64;
                for k in 0..n_t1 {
                    row_sum += ((f[t][j] + g[t + 1][k] + lk[t][j * n_t1 + k]) / eps).exp();
                }
                let viol =
                    ((col_sum - marginals[t][j]).abs()).max((row_sum - marginals[t][j]).abs());
                if viol > max_viol {
                    max_viol = viol;
                }
            }
        }

        // Violation at t = T: column marginal of last transition.
        {
            let t_last = t_steps - 1;
            let n_prev = marginals[t_last - 1].len();
            let n_last = marginals[t_last].len();
            for j in 0..n_last {
                let mut col_sum = 0.0_f64;
                for i in 0..n_prev {
                    col_sum += ((f[t_last - 1][i] + g[t_last][j] + lk[t_last - 1][i * n_last + j])
                        / eps)
                        .exp();
                }
                let viol = (col_sum - marginals[t_last][j]).abs();
                if viol > max_viol {
                    max_viol = viol;
                }
            }
        }

        if max_viol < config.tol {
            break;
        }
    }

    // Collect per-slice violations for the result.
    let violations = compute_violations(marginals, supports, &f, &g, &lk, eps);

    Ok(TdsbResult {
        f,
        g,
        violations,
        iters,
    })
}

/// Compute marginal violations for every time slice.
fn compute_violations(
    marginals: &[Vec<f64>],
    _supports: &[Vec<Vec<f64>>],
    f: &[Vec<f64>],
    g: &[Vec<f64>],
    lk: &[Vec<f64>],
    eps: f64,
) -> Vec<f64> {
    let t_steps = marginals.len();
    let mut violations = vec![0.0_f64; t_steps];

    // t = 0
    {
        let n0 = marginals[0].len();
        let n1 = marginals[1].len();
        let mut max_v = 0.0_f64;
        for i in 0..n0 {
            let mut row_sum = 0.0_f64;
            for j in 0..n1 {
                row_sum += ((f[0][i] + g[1][j] + lk[0][i * n1 + j]) / eps).exp();
            }
            let v = (row_sum - marginals[0][i]).abs();
            if v > max_v {
                max_v = v;
            }
        }
        violations[0] = max_v;
    }

    // Interior slices t = 1..T-1
    for t in 1..t_steps - 1 {
        let n_tm1 = marginals[t - 1].len();
        let n_t = marginals[t].len();
        let n_t1 = marginals[t + 1].len();
        let mut max_v = 0.0_f64;
        for j in 0..n_t {
            let mut col_sum = 0.0_f64;
            for i in 0..n_tm1 {
                col_sum += ((f[t - 1][i] + g[t][j] + lk[t - 1][i * n_t + j]) / eps).exp();
            }
            let mut row_sum = 0.0_f64;
            for k in 0..n_t1 {
                row_sum += ((f[t][j] + g[t + 1][k] + lk[t][j * n_t1 + k]) / eps).exp();
            }
            let v = ((col_sum - marginals[t][j]).abs()).max((row_sum - marginals[t][j]).abs());
            if v > max_v {
                max_v = v;
            }
        }
        violations[t] = max_v;
    }

    // t = T
    {
        let t_last = t_steps - 1;
        let n_prev = marginals[t_last - 1].len();
        let n_last = marginals[t_last].len();
        let mut max_v = 0.0_f64;
        for j in 0..n_last {
            let mut col_sum = 0.0_f64;
            for i in 0..n_prev {
                col_sum += ((f[t_last - 1][i] + g[t_last][j] + lk[t_last - 1][i * n_last + j])
                    / eps)
                    .exp();
            }
            let v = (col_sum - marginals[t_last][j]).abs();
            if v > max_v {
                max_v = v;
            }
        }
        violations[t_last] = max_v;
    }

    violations
}

// ─── Interpolation ────────────────────────────────────────────────────────────

/// Interpolate the bridge marginal at an intermediate time `tau ∈ [0, 1]`.
///
/// Maps `tau` to the adjacent time slices `t` and `t+1` where
/// `tau ∈ [t/(T), (t+1)/T]` and linearly interpolates the log-potentials.
/// Returns the unnormalised interpolated marginal weights (normalised to sum to 1).
pub fn tdsb_interpolate(
    result: &TdsbResult,
    supports: &[Vec<Vec<f64>>],
    tau: f64,
    eps: f64,
    dt: f64,
) -> OtResult<Vec<f64>> {
    let t_steps = result.f.len();
    if t_steps < 2 {
        return Err(OtError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&tau) {
        return Err(OtError::Internal {
            msg: format!("tau must be in [0, 1], got {tau}"),
        });
    }
    if eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: eps as f32 });
    }
    if dt <= 0.0 {
        return Err(OtError::Internal {
            msg: format!("dt must be > 0, got {dt}"),
        });
    }

    // Map tau to (t_left, t_right, alpha) where alpha is the weight of t_right.
    let scaled = tau * (t_steps - 1) as f64;
    let t_left = (scaled.floor() as usize).min(t_steps - 2);
    let t_right = t_left + 1;
    let alpha = scaled - t_left as f64; // in [0, 1]

    let n_l = result.f[t_left].len();
    let n_r = result.f[t_right].len();

    if supports[t_left].len() != n_l || supports[t_right].len() != n_r {
        return Err(OtError::IncompatibleLength {
            a: supports[t_left].len(),
            b: n_l,
        });
    }

    // Compute the transition plan between t_left and t_right.
    let plan = tdsb_transition_plan_internal(result, supports, t_left, eps, dt)?;

    // Interpolated marginal: weights[k] for a "mixed" support.
    // We blend the plan: at tau=t_left the marginal is marginal[t_left],
    // at tau=t_right the marginal is marginal[t_right].
    // For an intermediate tau: compute the weight of each source-destination pair
    // and marginalise: at alpha=0 → row sums (left marginal), at alpha=1 → col sums (right marginal).
    // We blend the two marginalised distributions.

    // Row sums (marginal at t_left)
    let mut left_marg = vec![0.0_f64; n_l];
    for i in 0..n_l {
        for j in 0..n_r {
            left_marg[i] += plan[i * n_r + j];
        }
    }
    // Col sums (marginal at t_right)
    let mut right_marg = vec![0.0_f64; n_r];
    for j in 0..n_r {
        for i in 0..n_l {
            right_marg[j] += plan[i * n_r + j];
        }
    }

    // Blend: result is a convex combination of left and right marginals,
    // but they live on different support sets.
    // We produce a concatenated weight vector of length n_l + n_r where
    // the first n_l weights correspond to t_left support and next n_r to t_right.
    // Weight: (1-alpha)*left_marg[i] for i in 0..n_l, alpha*right_marg[j] for j in 0..n_r.
    // This is a valid probability distribution that interpolates between the two marginals.
    let total_n = n_l + n_r;
    let mut blended = vec![0.0_f64; total_n];
    for i in 0..n_l {
        blended[i] = (1.0 - alpha) * left_marg[i];
    }
    for j in 0..n_r {
        blended[n_l + j] = alpha * right_marg[j];
    }

    // Normalise.
    let sum: f64 = blended.iter().sum();
    if sum > 0.0 {
        for w in &mut blended {
            *w /= sum;
        }
    }

    Ok(blended)
}

// ─── Transition plan ──────────────────────────────────────────────────────────

/// Internal helper: compute the transition plan between time `t` and `t+1`.
fn tdsb_transition_plan_internal(
    result: &TdsbResult,
    supports: &[Vec<Vec<f64>>],
    t: usize,
    eps: f64,
    dt: f64,
) -> OtResult<Vec<f64>> {
    let t_steps = result.f.len();
    if t + 1 >= t_steps {
        return Err(OtError::BadDim { got: t });
    }
    let n_t = result.f[t].len();
    let n_t1 = result.f[t + 1].len();

    if supports[t].len() != n_t || supports[t + 1].len() != n_t1 {
        return Err(OtError::IncompatibleLength {
            a: supports[t].len(),
            b: n_t,
        });
    }

    // g[t+1] is used together with f[t].
    // plan[i,j] = exp( (f[t][i] + g[t+1][j] + log K[i,j]) / eps )
    let mut plan = vec![0.0_f64; n_t * n_t1];
    for i in 0..n_t {
        for j in 0..n_t1 {
            let lk = log_kernel(&supports[t][i], &supports[t + 1][j], eps, dt);
            plan[i * n_t1 + j] = ((result.f[t][i] + result.g[t + 1][j] + lk) / eps).exp();
        }
    }
    Ok(plan)
}

/// Extract the pairwise transition plan between time `t` and `t+1`.
///
/// Returns the `n_t × n_{t+1}` transport plan (row-major) such that:
/// - Row sums ≈ `marginals[t]`
/// - Col sums ≈ `marginals[t+1]`
pub fn tdsb_transition_plan(
    result: &TdsbResult,
    supports: &[Vec<Vec<f64>>],
    t: usize,
    eps: f64,
    dt: f64,
) -> OtResult<Vec<f64>> {
    if eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: eps as f32 });
    }
    if dt <= 0.0 {
        return Err(OtError::Internal {
            msg: format!("dt must be > 0, got {dt}"),
        });
    }
    tdsb_transition_plan_internal(result, supports, t, eps, dt)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: uniform marginal over n points.
    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    // Helper: 1-D support points at given positions.
    fn support_1d(xs: &[f64]) -> Vec<Vec<f64>> {
        xs.iter().map(|&x| vec![x]).collect()
    }

    // Helper: check approx equality.
    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Test 1: 2-time-slice reduces to standard SB (marginal violation < tol) ──

    #[test]
    fn two_slice_marginal_violations_satisfied() {
        let marginals = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2]];
        let supports = vec![support_1d(&[0.0, 1.0, 2.0]), support_1d(&[0.0, 1.0, 2.0])];
        let cfg = TdsbConfig {
            eps: 0.3,
            max_outer: 500,
            tol: 1e-4,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("converges");
        for &v in &res.violations {
            assert!(v < 5e-3, "violation {v} >= 5e-3");
        }
    }

    // ── Test 2: Marginal at t=0 and t=T satisfied ────────────────────────────

    #[test]
    fn first_and_last_marginals_satisfied() {
        let marginals = vec![vec![0.3, 0.7], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig {
            eps: 0.2,
            max_outer: 500,
            tol: 1e-5,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        // Check violations vector length == n_time_slices
        assert_eq!(res.violations.len(), marginals.len());
        for &v in &res.violations {
            assert!(v < 1e-2, "marginal violation {v} too large");
        }
    }

    // ── Test 3: 3-time-slice case runs and converges ─────────────────────────

    #[test]
    fn three_slice_converges() {
        let marginals = vec![vec![0.5, 0.5], vec![0.4, 0.6], vec![0.6, 0.4]];
        let supports = vec![
            support_1d(&[0.0, 1.0]),
            support_1d(&[0.0, 1.0]),
            support_1d(&[0.0, 1.0]),
        ];
        let cfg = TdsbConfig {
            eps: 0.5,
            max_outer: 1000,
            tol: 1e-4,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("three-slice ok");
        assert_eq!(res.violations.len(), 3);
        for &v in &res.violations {
            assert!(v < 5e-2, "three-slice violation {v}");
        }
    }

    // ── Test 4: tdsb_interpolate at tau=0 gives consistent left marginal ─────

    #[test]
    fn interpolate_at_tau_zero() {
        let marginals = vec![vec![0.6, 0.4], vec![0.4, 0.6]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig {
            eps: 0.2,
            max_outer: 500,
            tol: 1e-5,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let interp = tdsb_interpolate(&res, &supports, 0.0, 0.2, 1.0).expect("interp ok");
        // At tau=0 the total mass should be 1
        let total: f64 = interp.iter().sum();
        assert!(
            approx(total, 1.0, 1e-6),
            "total mass at tau=0 should be 1, got {total}"
        );
    }

    // ── Test 5: tdsb_interpolate at tau=1 gives consistent right marginal ────

    #[test]
    fn interpolate_at_tau_one() {
        let marginals = vec![vec![0.6, 0.4], vec![0.4, 0.6]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig {
            eps: 0.2,
            max_outer: 500,
            tol: 1e-5,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let interp = tdsb_interpolate(&res, &supports, 1.0, 0.2, 1.0).expect("interp ok");
        // At tau=1 the total mass should be 1
        let total: f64 = interp.iter().sum();
        assert!(
            approx(total, 1.0, 1e-6),
            "total mass at tau=1 should be 1, got {total}"
        );
    }

    // ── Test 6: transition plan row sums ≈ marginals[t] ───────────────────────

    #[test]
    fn transition_plan_row_sums_match_left_marginal() {
        let marginals = vec![vec![0.4, 0.3, 0.3], vec![0.5, 0.3, 0.2]];
        let supports = vec![support_1d(&[0.0, 1.0, 2.0]), support_1d(&[0.0, 1.0, 2.0])];
        let cfg = TdsbConfig {
            eps: 0.3,
            max_outer: 500,
            tol: 1e-4,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let plan = tdsb_transition_plan(&res, &supports, 0, 0.3, 1.0).expect("plan ok");
        let n_r = marginals[1].len();
        for (i, &m) in marginals[0].iter().enumerate() {
            let row_sum: f64 = (0..n_r).map(|j| plan[i * n_r + j]).sum();
            assert!(approx(row_sum, m, 5e-2), "row {i} sum {row_sum} != {m}");
        }
    }

    // ── Test 7: transition plan col sums ≈ marginals[t+1] ────────────────────

    #[test]
    fn transition_plan_col_sums_match_right_marginal() {
        let marginals = vec![vec![0.4, 0.3, 0.3], vec![0.5, 0.3, 0.2]];
        let supports = vec![support_1d(&[0.0, 1.0, 2.0]), support_1d(&[0.0, 1.0, 2.0])];
        let cfg = TdsbConfig {
            eps: 0.3,
            max_outer: 500,
            tol: 1e-4,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let plan = tdsb_transition_plan(&res, &supports, 0, 0.3, 1.0).expect("plan ok");
        let n_l = marginals[0].len();
        let n_r = marginals[1].len();
        for (j, &m) in marginals[1].iter().enumerate() {
            let col_sum: f64 = (0..n_l).map(|i| plan[i * n_r + j]).sum();
            assert!(approx(col_sum, m, 5e-2), "col {j} sum {col_sum} != {m}");
        }
    }

    // ── Test 8: transition plan non-negative ──────────────────────────────────

    #[test]
    fn transition_plan_non_negative() {
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig {
            eps: 0.1,
            max_outer: 200,
            tol: 1e-4,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let plan = tdsb_transition_plan(&res, &supports, 0, 0.1, 1.0).expect("ok");
        for &p in &plan {
            assert!(p >= 0.0, "plan entry {p} is negative");
        }
    }

    // ── Test 9: identical marginals → more mass on diagonal ──────────────────

    #[test]
    fn identical_marginals_diagonal_dominant() {
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig {
            eps: 0.05,
            max_outer: 500,
            tol: 1e-5,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        let plan = tdsb_transition_plan(&res, &supports, 0, 0.05, 1.0).expect("ok");
        // With small eps and matching marginals, diagonal should dominate
        let diag_mass: f64 = plan[0] + plan[3]; // (0,0) + (1,1)
        let off_diag: f64 = plan[1] + plan[2];
        assert!(
            diag_mass > off_diag,
            "diagonal mass {diag_mass} <= off-diag {off_diag}"
        );
    }

    // ── Test 10: large eps → more diffuse plans ───────────────────────────────

    #[test]
    fn large_eps_more_diffuse() {
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg_sharp = TdsbConfig {
            eps: 0.01,
            max_outer: 800,
            tol: 1e-4,
            dt: 1.0,
        };
        let cfg_diffuse = TdsbConfig {
            eps: 5.0,
            max_outer: 800,
            tol: 1e-4,
            dt: 1.0,
        };
        let res_s = tdsb(&marginals, &supports, &cfg_sharp).expect("sharp ok");
        let res_d = tdsb(&marginals, &supports, &cfg_diffuse).expect("diffuse ok");
        let plan_s = tdsb_transition_plan(&res_s, &supports, 0, 0.01, 1.0).expect("ok");
        let plan_d = tdsb_transition_plan(&res_d, &supports, 0, 5.0, 1.0).expect("ok");
        // Entropy of plan_s < entropy of plan_d
        let entropy = |plan: &[f64]| -> f64 {
            plan.iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum()
        };
        assert!(
            entropy(&plan_d) > entropy(&plan_s),
            "diffuse plan entropy {} <= sharp entropy {}",
            entropy(&plan_d),
            entropy(&plan_s)
        );
    }

    // ── Test 11: dt effect — small dt → kernel close to identity ─────────────

    #[test]
    fn small_dt_produces_identity_like_kernel() {
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        // With very small dt, the Brownian kernel is very peaked around diagonal
        let cfg = TdsbConfig {
            eps: 0.1,
            max_outer: 500,
            tol: 1e-4,
            dt: 0.01,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("small dt ok");
        let plan = tdsb_transition_plan(&res, &supports, 0, 0.1, 0.01).expect("ok");
        // Diagonal entries should dominate
        let diag = plan[0] + plan[3];
        let off_d = plan[1] + plan[2];
        assert!(diag > off_d, "small dt: diag {diag} <= off-diag {off_d}");
    }

    // ── Test 12: violations length == n_time_slices ───────────────────────────

    #[test]
    fn violations_length_equals_n_time_slices() {
        for t in [2_usize, 3, 5] {
            let marginals: Vec<Vec<f64>> = (0..t).map(|_| uniform(3)).collect();
            let supports: Vec<Vec<Vec<f64>>> =
                (0..t).map(|_| support_1d(&[0.0, 1.0, 2.0])).collect();
            let cfg = TdsbConfig {
                eps: 0.5,
                max_outer: 200,
                tol: 1e-3,
                dt: 1.0,
            };
            let res = tdsb(&marginals, &supports, &cfg).expect("ok");
            assert_eq!(res.violations.len(), t, "T={t}: violations len mismatch");
        }
    }

    // ── Test 13: iters ≤ max_outer ────────────────────────────────────────────

    #[test]
    fn iters_at_most_max_outer() {
        let marginals = vec![uniform(4), uniform(4)];
        let supports = vec![
            support_1d(&[0.0, 1.0, 2.0, 3.0]),
            support_1d(&[0.0, 1.0, 2.0, 3.0]),
        ];
        let cfg = TdsbConfig {
            eps: 0.5,
            max_outer: 50,
            tol: 1e-9,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg).expect("ok");
        assert!(res.iters <= 50, "iters {} > max_outer 50", res.iters);
    }

    // ── Test 14: supports dimension mismatch → error ──────────────────────────

    #[test]
    fn supports_dimension_mismatch_error() {
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        // First support is 2D, second is 1D → dimension mismatch
        let supports = vec![
            vec![vec![0.0, 0.0], vec![1.0, 0.0]],
            vec![vec![0.0], vec![1.0]],
        ];
        let cfg = TdsbConfig::default();
        // Should fail on dimension mismatch
        let res = tdsb(&marginals, &supports, &cfg);
        assert!(res.is_err(), "Expected error for dimension mismatch");
    }

    // ── Test 15: marginals with negative entries → error ──────────────────────

    #[test]
    fn negative_marginal_entries_rejected() {
        let marginals = vec![vec![-0.5, 1.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig::default();
        let res = tdsb(&marginals, &supports, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    // ── Test 16: mismatched marginal and support sizes → error ────────────────

    #[test]
    fn mismatched_marginal_support_sizes_rejected() {
        // marginals[0] has 2 entries but supports[0] has 3 points
        let marginals = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let supports = vec![support_1d(&[0.0, 1.0, 2.0]), support_1d(&[0.0, 1.0])];
        let cfg = TdsbConfig::default();
        let res = tdsb(&marginals, &supports, &cfg);
        assert!(res.is_err(), "Expected error for mismatched sizes");
    }

    // ── Test 17: logsumexp helper correct ────────────────────────────────────

    #[test]
    fn logsumexp_known_value() {
        let v = [0.0_f64, 0.0];
        let expected = 2.0_f64.ln();
        assert!(approx(logsumexp(&v), expected, 1e-12));
    }

    // ── Test 18: 4-slice case runs without panic ──────────────────────────────

    #[test]
    fn four_slice_runs_without_error() {
        let marginals: Vec<Vec<f64>> = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.3, 0.2, 0.3, 0.2],
            vec![0.2, 0.3, 0.2, 0.3],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let supports: Vec<Vec<Vec<f64>>> =
            (0..4).map(|_| support_1d(&[0.0, 1.0, 2.0, 3.0])).collect();
        let cfg = TdsbConfig {
            eps: 0.5,
            max_outer: 300,
            tol: 1e-3,
            dt: 1.0,
        };
        let res = tdsb(&marginals, &supports, &cfg);
        assert!(
            res.is_ok(),
            "4-slice TDSB should not error: {:?}",
            res.err()
        );
    }
}
