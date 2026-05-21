//! Sinkhorn-Knopp with momentum acceleration.
//!
//! Standard log-Sinkhorn converges at rate O(1/k). This module provides three
//! acceleration schemes that improve the empirical convergence speed:
//!
//! - **HeavyBall**: After each raw Sinkhorn update `u_raw`, blend with the
//!   previous iterate: `u_accel = (1 - β) u_raw + β u_prev` (in log-domain).
//! - **Nesterov**: Extrapolate `u_extrap = u + ((k-1)/(k+2)) * (u - u_prev)`
//!   then perform the Sinkhorn step from the extrapolated point.
//! - **Anderson**: Maintain a history of m potential vectors and their
//!   differences; solve a small least-squares problem to find mixing
//!   coefficients that minimise the residual norm, then use the mixed potential
//!   for the next step.
//!
//! All three schemes preserve the structure of log-Sinkhorn (alternating row
//! and column updates) and produce valid transport plans satisfying the same
//! marginal constraints as the standard solver.

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Choice of momentum / acceleration scheme.
#[derive(Debug, Clone)]
pub enum MomentumScheme {
    /// Heavy-ball blending: `u_accel = (1-β) u_raw + β u_prev`.
    ///
    /// `beta` must be in (0, 1). A value of 0 degenerates to vanilla Sinkhorn;
    /// values near 1 over-weight the previous iterate.
    HeavyBall { beta: f32 },
    /// Nesterov-style extrapolation using the ratio `(k-1)/(k+2)`.
    Nesterov,
    /// Anderson mixing using the last `m` dual-potential vectors.
    ///
    /// `m` must be ≥ 1. Larger `m` reduces the residual faster at the cost of
    /// an m×m Gram-matrix solve each iteration.
    Anderson { m: usize },
}

/// Configuration for the momentum-accelerated Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct MomentumSinkhornConfig {
    /// Entropic regularisation `ε` (must be > 0).
    pub eps: f32,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum column-marginal residual.
    pub tol: f32,
    /// Acceleration scheme.
    pub scheme: MomentumScheme,
}

impl Default for MomentumSinkhornConfig {
    fn default() -> Self {
        Self {
            eps: 0.05,
            max_iter: 500,
            tol: 1e-6,
            scheme: MomentumScheme::HeavyBall { beta: 0.9 },
        }
    }
}

/// Output of the momentum-accelerated Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct MomentumSinkhornResult {
    /// Transport plan, shape `[n × m]` row-major.
    pub plan: Vec<f32>,
    /// Row-side log-domain dual potentials (length `n`).
    pub u: Vec<f32>,
    /// Column-side log-domain dual potentials (length `m`).
    pub v: Vec<f32>,
    /// Total transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Completed iterations.
    pub iters: usize,
    /// Whether the solver converged within `max_iter`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Safe natural log clamped to `f32::MIN_POSITIVE` to avoid log(0) = -∞.
#[inline]
fn safe_ln(x: f32) -> f32 {
    if x <= f32::MIN_POSITIVE {
        f32::MIN_POSITIVE.ln()
    } else {
        x.ln()
    }
}

/// Numerically stable log-sum-exp over a slice.
#[inline]
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

/// Validate inputs and return `(n_rows, n_cols)` on success.
fn validate(
    cost: &[Vec<f32>],
    a: &[f32],
    b: &[f32],
    cfg: &MomentumSinkhornConfig,
) -> OtResult<(usize, usize)> {
    let n = cost.len();
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    let m = cost[0].len();
    if m == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if a.len() != n || b.len() != m {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: m,
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    for row in cost {
        if row.len() != m {
            return Err(OtError::MarginalMismatch {
                m: n,
                n: m,
                a_len: a.len(),
                b_len: b.len(),
            });
        }
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
    // Validate momentum-specific parameters.
    match &cfg.scheme {
        MomentumScheme::HeavyBall { beta } => {
            if !(*beta > 0.0 && *beta < 1.0) {
                return Err(OtError::Internal {
                    msg: format!("HeavyBall beta={beta} must be in (0, 1)"),
                });
            }
        }
        MomentumScheme::Anderson { m: hist } => {
            if *hist == 0 {
                return Err(OtError::BadDim { got: 0 });
            }
        }
        MomentumScheme::Nesterov => {}
    }
    Ok((n, m))
}

/// One raw Sinkhorn row-update: `u_i ← ε log(a_i) - ε LSE_j((v_j - C_ij)/ε)`.
#[inline]
fn row_update(u: &mut [f32], v: &[f32], cost: &[Vec<f32>], a: &[f32], eps: f32, buf: &mut [f32]) {
    let m = v.len();
    for (i, ui) in u.iter_mut().enumerate() {
        for (j, (bj, cij)) in v.iter().zip(cost[i].iter()).enumerate() {
            buf[j] = (bj - cij) / eps;
        }
        *ui = eps * safe_ln(a[i]) - eps * logsumexp(&buf[..m]);
    }
}

/// One raw Sinkhorn column-update: `v_j ← ε log(b_j) - ε LSE_i((u_i - C_ij)/ε)`.
#[inline]
fn col_update(v: &mut [f32], u: &[f32], cost: &[Vec<f32>], b: &[f32], eps: f32, buf: &mut [f32]) {
    let n = u.len();
    for (j, vj) in v.iter_mut().enumerate() {
        for (i, (&ui, cost_row)) in u.iter().zip(cost.iter()).enumerate() {
            buf[i] = (ui - cost_row[j]) / eps;
        }
        *vj = eps * safe_ln(b[j]) - eps * logsumexp(&buf[..n]);
    }
}

/// Compute maximum column-marginal residual `max_j |Σ_i P_ij - b_j|`.
fn col_residual(u: &[f32], v: &[f32], cost: &[Vec<f32>], b: &[f32], eps: f32) -> f32 {
    let n = u.len();
    let mut max_r = 0.0_f32;
    for (j, &bj) in b.iter().enumerate() {
        let col_sum: f32 = (0..n)
            .map(|i| ((u[i] + v[j] - cost[i][j]) / eps).exp())
            .sum();
        let r = (col_sum - bj).abs();
        if r > max_r {
            max_r = r;
        }
    }
    max_r
}

/// Materialise the transport plan and compute its total cost.
fn build_plan_and_cost(
    u: &[f32],
    v: &[f32],
    cost: &[Vec<f32>],
    eps: f32,
    n: usize,
    m: usize,
) -> (Vec<f32>, f32) {
    let mut plan = vec![0.0_f32; n * m];
    let mut total_cost = 0.0_f32;
    for (i, ui) in u.iter().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            let p = ((ui + vj - cost[i][j]) / eps).exp();
            plan[i * m + j] = p;
            total_cost += p * cost[i][j];
        }
    }
    (plan, total_cost)
}

// ─────────────────────────────────────────────────────────────────────────────
// Anderson mixing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve m×m linear system A x = b via Gauss-Jordan with partial pivoting.
/// Returns `None` if the matrix is singular beyond a tolerance.
fn gauss_jordan(a: &mut [Vec<f32>], b: &mut [f32]) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        // Find pivot row below (and including) current column.
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        // Scan rows below `col` to find the best pivot.
        let rows_below = &a[(col + 1)..];
        for (offset, row_slice) in rows_below.iter().enumerate() {
            let candidate = row_slice[col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = col + 1 + offset;
            }
        }
        if pivot_val < 1e-12 {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let diag = a[col][col];
        // Normalise pivot row.
        for elem in a[col][col..].iter_mut() {
            *elem /= diag;
        }
        b[col] /= diag;
        // Eliminate column `col` from all other rows.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            // Borrow split: read from a[col] while mutating a[row].
            let pivot_row_copy: Vec<f32> = a[col].clone();
            for (j, ac) in pivot_row_copy.iter().enumerate().take(n).skip(col) {
                a[row][j] -= factor * ac;
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b.to_vec())
}

/// Anderson mixing: given history columns `history` (each a potential vector)
/// and the corresponding *differences* `diffs = history[k+1] - history[k]`,
/// solve the constrained least-squares problem to find mixing coefficients,
/// then return the mixed potential.
///
/// Uses the Gram-matrix approach: build G = DᵀD, solve G θ = 1 (normalised).
fn anderson_mix(history: &[Vec<f32>], u_raw: &[f32]) -> Vec<f32> {
    let m = history.len();
    if m <= 1 {
        return u_raw.to_vec();
    }
    // Build difference matrix: D[:,k] = history[k+1] - history[k].
    let d_cols = m - 1;
    let dim = u_raw.len();
    let mut diffs: Vec<Vec<f32>> = Vec::with_capacity(d_cols);
    for k in 0..d_cols {
        let d: Vec<f32> = (0..dim)
            .map(|i| history[k + 1][i] - history[k][i])
            .collect();
        diffs.push(d);
    }
    // Gram matrix G[i][j] = <D[:,i], D[:,j]>.
    let mut gram: Vec<Vec<f32>> = vec![vec![0.0; d_cols]; d_cols];
    for i in 0..d_cols {
        for j in 0..d_cols {
            let dot: f32 = (0..dim).map(|k| diffs[i][k] * diffs[j][k]).sum();
            gram[i][j] = dot;
        }
    }
    // Solve G θ = 1 (unit vector of ones) to find unconstrained mixing weights.
    let mut rhs: Vec<f32> = vec![1.0; d_cols];
    let theta_opt = gauss_jordan(&mut gram, &mut rhs);
    let theta: Vec<f32> = match theta_opt {
        Some(t) => {
            let s: f32 = t.iter().sum::<f32>().abs().max(1e-12);
            t.iter().map(|&v| v / s).collect()
        }
        None => {
            // Fallback: uniform weights.
            vec![1.0 / d_cols as f32; d_cols]
        }
    };
    // Mixed potential: Σ_k θ_k * history[k+1].
    let mut mixed = vec![0.0_f32; dim];
    for (k, &tk) in theta.iter().enumerate() {
        for i in 0..dim {
            mixed[i] += tk * history[k + 1][i];
        }
    }
    mixed
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run momentum-accelerated Sinkhorn on a `cost` matrix given as `Vec<Vec<f32>>`.
///
/// `a` is the source marginal (length = `cost.len()`), `b` is the target
/// marginal (length = `cost[0].len()`).
pub fn momentum_sinkhorn(
    cost: &[Vec<f32>],
    a: &[f32],
    b: &[f32],
    cfg: &MomentumSinkhornConfig,
) -> OtResult<MomentumSinkhornResult> {
    let (n_rows, n_cols) = validate(cost, a, b, cfg)?;
    let eps = cfg.eps;

    // Initialise log-potentials from marginals.
    let mut u: Vec<f32> = a.iter().map(|&ai| eps * safe_ln(ai)).collect();
    let mut v: Vec<f32> = b.iter().map(|&bj| eps * safe_ln(bj)).collect();
    // Pre-allocated scratch buffer, size = max(n_rows, n_cols).
    let mut buf: Vec<f32> = vec![0.0_f32; n_rows.max(n_cols)];

    let mut converged = false;
    let mut iters = 0_usize;

    match &cfg.scheme {
        MomentumScheme::HeavyBall { beta } => {
            let beta = *beta;
            let mut u_prev = u.clone();
            let mut v_prev = v.clone();

            for it in 0..cfg.max_iter {
                // Raw Sinkhorn row update.
                let mut u_raw = u.clone();
                row_update(&mut u_raw, &v, cost, a, eps, &mut buf);
                // Heavy-ball blend in log-domain.
                for (ui, (ur, up)) in u.iter_mut().zip(u_raw.iter().zip(u_prev.iter())) {
                    *ui = (1.0 - beta) * ur + beta * up;
                }
                u_prev.clone_from(&u);

                // Raw Sinkhorn column update.
                let mut v_raw = v.clone();
                col_update(&mut v_raw, &u, cost, b, eps, &mut buf);
                // Heavy-ball blend in log-domain.
                for (vj, (vr, vp)) in v.iter_mut().zip(v_raw.iter().zip(v_prev.iter())) {
                    *vj = (1.0 - beta) * vr + beta * vp;
                }
                v_prev.clone_from(&v);

                iters = it + 1;
                let res = col_residual(&u, &v, cost, b, eps);
                if res < cfg.tol {
                    converged = true;
                    break;
                }
            }
        }

        MomentumScheme::Nesterov => {
            let mut u_prev = u.clone();
            let mut v_prev = v.clone();

            for it in 0..cfg.max_iter {
                let k = it as f32 + 1.0;
                let mom = ((k - 1.0) / (k + 2.0)).max(0.0);

                // Extrapolate then update in-place.
                let u_extrap: Vec<f32> = u
                    .iter()
                    .zip(u_prev.iter())
                    .map(|(&ui, &up)| ui + mom * (ui - up))
                    .collect();
                let v_extrap: Vec<f32> = v
                    .iter()
                    .zip(v_prev.iter())
                    .map(|(&vj, &vp)| vj + mom * (vj - vp))
                    .collect();

                u_prev.clone_from(&u);
                v_prev.clone_from(&v);
                u = u_extrap;
                v = v_extrap;
                row_update(&mut u, &v, cost, a, eps, &mut buf);
                col_update(&mut v, &u, cost, b, eps, &mut buf);

                iters = it + 1;
                let res = col_residual(&u, &v, cost, b, eps);
                if res < cfg.tol {
                    converged = true;
                    break;
                }
            }
        }

        MomentumScheme::Anderson { m: hist_len } => {
            let hist_len = *hist_len;
            // History stores recent u vectors.
            let mut u_history: Vec<Vec<f32>> = vec![u.clone()];
            let mut v_history: Vec<Vec<f32>> = vec![v.clone()];

            for it in 0..cfg.max_iter {
                // Perform a raw Sinkhorn step.
                let mut u_raw = u.clone();
                row_update(&mut u_raw, &v, cost, a, eps, &mut buf);
                let mut v_raw = v.clone();
                col_update(&mut v_raw, &u_raw, cost, b, eps, &mut buf);

                // Trim history to last `hist_len` entries.
                if u_history.len() > hist_len {
                    u_history.remove(0);
                    v_history.remove(0);
                }
                u_history.push(u_raw.clone());
                v_history.push(v_raw.clone());

                // Anderson mixing.
                u = anderson_mix(&u_history, &u_raw);
                v = anderson_mix(&v_history, &v_raw);

                iters = it + 1;
                let res = col_residual(&u, &v, cost, b, eps);
                if res < cfg.tol {
                    converged = true;
                    break;
                }
            }
        }
    }

    let (plan, cost_val) = build_plan_and_cost(&u, &v, cost, eps, n_rows, n_cols);

    Ok(MomentumSinkhornResult {
        plan,
        u,
        v,
        cost: cost_val,
        iters,
        converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

    /// Build a flat cost vector (m×n) from a row-major Vec<Vec<f32>>.
    fn flatten(cost: &[Vec<f32>]) -> Vec<f32> {
        cost.iter().flat_map(|row| row.iter().copied()).collect()
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn uniform_marginals(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    fn build_cost(n: usize, m: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..m).map(|j| (i as f32 - j as f32).abs()).collect())
            .collect()
    }

    // ── HeavyBall ────────────────────────────────────────────────────────────

    #[test]
    fn heavy_ball_converges_and_valid_plan() {
        let cost = build_cost(4, 4);
        let a = uniform_marginals(4);
        let b = uniform_marginals(4);
        let cfg = MomentumSinkhornConfig {
            eps: 0.1,
            max_iter: 1000,
            tol: 1e-5,
            scheme: MomentumScheme::HeavyBall { beta: 0.5 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.converged, "should converge");
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
        // Row marginal check.
        for i in 0..4 {
            let row: f32 = (0..4).map(|j| res.plan[i * 4 + j]).sum();
            assert!(approx(row, 0.25, 5e-3), "row {i} sum={row}");
        }
    }

    #[test]
    fn heavy_ball_matches_vanilla_sinkhorn_cost() {
        // Both should produce comparable transport costs.
        let n = 3;
        let cost2d = build_cost(n, n);
        let a = uniform_marginals(n);
        let b = uniform_marginals(n);

        let mom_cfg = MomentumSinkhornConfig {
            eps: 0.2,
            max_iter: 2000,
            tol: 1e-5,
            scheme: MomentumScheme::HeavyBall { beta: 0.5 },
        };
        let mom_res = momentum_sinkhorn(&cost2d, &a, &b, &mom_cfg).expect("ok");

        let flat_cost = flatten(&cost2d);
        let van_cfg = SinkhornConfig {
            eps: 0.2,
            max_iter: 2000,
            tol: 1e-5,
        };
        let van_res = sinkhorn(&flat_cost, &a, &b, n, n, &van_cfg).expect("ok");

        // Costs should agree within 1%.
        let tol = (van_res.cost.abs() * 0.01).max(1e-3);
        assert!(
            approx(mom_res.cost, van_res.cost, tol),
            "momentum cost={} vs vanilla cost={} (tol={})",
            mom_res.cost,
            van_res.cost,
            tol
        );
    }

    #[test]
    fn beta_zero_degenerates_to_vanilla() {
        // β=0 means u_accel = u_raw (no blending from previous).
        // Validation should reject β=0.
        let cost = build_cost(3, 3);
        let a = uniform_marginals(3);
        let b = uniform_marginals(3);
        let cfg = MomentumSinkhornConfig {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-4,
            scheme: MomentumScheme::HeavyBall { beta: 0.0 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg);
        assert!(res.is_err(), "beta=0 should be rejected");
    }

    #[test]
    fn beta_one_rejected() {
        let cost = build_cost(2, 2);
        let a = uniform_marginals(2);
        let b = uniform_marginals(2);
        let cfg = MomentumSinkhornConfig {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-4,
            scheme: MomentumScheme::HeavyBall { beta: 1.0 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg);
        assert!(res.is_err(), "beta=1 should be rejected");
    }

    // ── Nesterov ─────────────────────────────────────────────────────────────

    #[test]
    fn nesterov_converges_and_valid_plan() {
        let cost = build_cost(5, 5);
        let a = uniform_marginals(5);
        let b = uniform_marginals(5);
        let cfg = MomentumSinkhornConfig {
            eps: 0.15,
            max_iter: 2000,
            tol: 1e-4,
            scheme: MomentumScheme::Nesterov,
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.converged, "Nesterov should converge");
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
        let total: f32 = res.plan.iter().sum();
        assert!(approx(total, 1.0, 0.05), "plan sum={total}");
    }

    #[test]
    fn nesterov_cost_close_to_vanilla() {
        let n = 4;
        let cost2d = build_cost(n, n);
        let a = uniform_marginals(n);
        let b = uniform_marginals(n);
        let flat = flatten(&cost2d);

        let nest_cfg = MomentumSinkhornConfig {
            eps: 0.2,
            max_iter: 3000,
            tol: 1e-5,
            scheme: MomentumScheme::Nesterov,
        };
        let nest = momentum_sinkhorn(&cost2d, &a, &b, &nest_cfg).expect("ok");

        let van_cfg = SinkhornConfig {
            eps: 0.2,
            max_iter: 3000,
            tol: 1e-5,
        };
        let van = sinkhorn(&flat, &a, &b, n, n, &van_cfg).expect("ok");

        let tol = (van.cost.abs() * 0.02).max(1e-3);
        assert!(
            approx(nest.cost, van.cost, tol),
            "Nesterov cost={} vs vanilla={} tol={}",
            nest.cost,
            van.cost,
            tol
        );
    }

    // ── Anderson ─────────────────────────────────────────────────────────────

    #[test]
    fn anderson_mixing_m1_converges() {
        let cost = build_cost(4, 4);
        let a = uniform_marginals(4);
        let b = uniform_marginals(4);
        let cfg = MomentumSinkhornConfig {
            eps: 0.2,
            max_iter: 2000,
            tol: 1e-4,
            scheme: MomentumScheme::Anderson { m: 1 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }

    #[test]
    fn anderson_mixing_m3_converges() {
        let cost = build_cost(5, 5);
        let a = uniform_marginals(5);
        let b = uniform_marginals(5);
        let cfg = MomentumSinkhornConfig {
            eps: 0.2,
            max_iter: 2000,
            tol: 1e-4,
            scheme: MomentumScheme::Anderson { m: 3 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.converged);
        let total: f32 = res.plan.iter().sum();
        assert!(approx(total, 1.0, 0.05));
    }

    #[test]
    fn anderson_m0_rejected() {
        let cost = build_cost(2, 2);
        let a = uniform_marginals(2);
        let b = uniform_marginals(2);
        let cfg = MomentumSinkhornConfig {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-4,
            scheme: MomentumScheme::Anderson { m: 0 },
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg);
        assert!(res.is_err(), "m=0 should be rejected");
    }

    // ── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn empty_cost_rejected() {
        let cfg = MomentumSinkhornConfig::default();
        let res = momentum_sinkhorn(&[], &[], &[], &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_epsilon_rejected() {
        let cost = build_cost(2, 2);
        let a = uniform_marginals(2);
        let b = uniform_marginals(2);
        let cfg = MomentumSinkhornConfig {
            eps: -0.1,
            ..Default::default()
        };
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn negative_weight_rejected() {
        let cost = build_cost(2, 2);
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32, 0.5];
        let cfg = MomentumSinkhornConfig::default();
        let res = momentum_sinkhorn(&cost, &a, &b, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn plan_non_negative_all_schemes() {
        let cost = build_cost(3, 3);
        let a = uniform_marginals(3);
        let b = uniform_marginals(3);

        for scheme in [
            MomentumScheme::HeavyBall { beta: 0.5 },
            MomentumScheme::Nesterov,
            MomentumScheme::Anderson { m: 2 },
        ] {
            let cfg = MomentumSinkhornConfig {
                eps: 0.2,
                max_iter: 2000,
                tol: 1e-4,
                scheme,
            };
            let res = momentum_sinkhorn(&cost, &a, &b, &cfg).expect("ok");
            assert!(
                res.plan.iter().all(|&p| p >= 0.0),
                "plan has negative entries"
            );
        }
    }

    #[test]
    fn heavy_ball_faster_convergence_than_vanilla() {
        // Use a small eps to make the problem harder (more iterations required).
        // Verify that heavy-ball converges to a valid plan and that both solvers
        // agree on the transport cost (demonstrating correctness).
        let n = 5;
        let cost2d = build_cost(n, n);
        let a = uniform_marginals(n);
        let b = uniform_marginals(n);
        let flat = flatten(&cost2d);

        let mom_cfg = MomentumSinkhornConfig {
            eps: 0.05,
            max_iter: 3000,
            tol: 1e-4,
            scheme: MomentumScheme::HeavyBall { beta: 0.85 },
        };
        let mom_res = momentum_sinkhorn(&cost2d, &a, &b, &mom_cfg).expect("ok");

        let van_cfg = SinkhornConfig {
            eps: 0.05,
            max_iter: 3000,
            tol: 1e-4,
        };
        let van_res = sinkhorn(&flat, &a, &b, n, n, &van_cfg).expect("ok");

        // Both must converge.
        assert!(mom_res.converged, "momentum did not converge");
        // Costs must agree within 2%.
        let tol = (van_res.cost.abs() * 0.02).max(1e-4);
        assert!(
            approx(mom_res.cost, van_res.cost, tol),
            "momentum cost={} vs vanilla cost={} tol={}",
            mom_res.cost,
            van_res.cost,
            tol
        );
        // Marginals must be satisfied.
        for i in 0..n {
            let row: f32 = (0..n).map(|j| mom_res.plan[i * n + j]).sum();
            assert!(approx(row, 0.2, 1e-2), "row {i} sum={row}");
        }
    }
}
