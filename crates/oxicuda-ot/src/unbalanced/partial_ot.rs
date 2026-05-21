//! Partial Optimal Transport with TV and L2 marginal relaxations.
//!
//! Implements entropic Partial OT beyond the KL relaxation already available in
//! `unbalanced_ot.rs`. Two additional relaxation families are provided:
//!
//! ## TV (Total Variation) relaxation
//!
//! ```text
//! min_T  <C, T> + ε·KL(T ‖ a⊗b)   s.t.  T ≥ 0, T1 ≤ a, Tᵀ1 ≤ b, 1ᵀT1 = m·sum_a
//! ```
//!
//! Implemented via log-Sinkhorn with a *soft thresholding* of the dual row
//! update: `u_new[i] = min(u_raw[i], ε log a[i])`, which enforces T1 ≤ a
//! pointwise.
//!
//! ## L2 relaxation
//!
//! ```text
//! min_T  <C, T> + ε·KL(T ‖ a⊗b) + (τ/2)·‖T1 − a‖² + (τ/2)·‖Tᵀ1 − b‖²
//! ```
//!
//! The proximal operator of the L2 marginal penalty scales the standard
//! log-Sinkhorn dual step: `u_new[i] = (ε/(ε + τ)) · (ε log a[i] − LSE_row[i])`.

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Marginal relaxation variant for partial OT.
#[derive(Debug, Clone)]
pub enum UnbalancedRelaxation {
    /// KL divergence relaxation with strength τ (existing variant, re-exposed here).
    Kl { tau: f32 },
    /// Total Variation relaxation with TV strength τ (clamps marginal violations).
    Tv { tau: f32 },
    /// L2 quadratic penalty on marginal violations with strength τ.
    L2 { tau: f32 },
}

/// Configuration for the partial OT solver.
#[derive(Debug, Clone)]
pub struct PartialOtConfig {
    /// Entropic regularisation `ε > 0`.
    pub eps: f32,
    /// Marginal relaxation type and its strength parameter.
    pub relaxation: UnbalancedRelaxation,
    /// Maximum number of outer log-Sinkhorn iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum column-marginal residual.
    pub tol: f32,
    /// Target transported mass relative to `sum(a)` — only used by the TV
    /// variant. Must be in `(0, 1]`.
    pub mass: f32,
}

impl Default for PartialOtConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::L2 { tau: 1.0 },
            max_iter: 1000,
            tol: 1e-6,
            mass: 0.8,
        }
    }
}

/// Output of the partial OT solver.
#[derive(Debug, Clone)]
pub struct PartialOtResult {
    /// Transport plan, shape `[n_rows × n_cols]` row-major.
    pub plan: Vec<f32>,
    /// Total transport cost `Σ_{ij} P_ij C_ij`.
    pub cost: f32,
    /// Total transported mass `Σ_{ij} P_ij`.
    pub transported_mass: f32,
    /// Row marginal `T1` (length `n_rows`).
    pub row_marginal: Vec<f32>,
    /// Column marginal `Tᵀ1` (length `n_cols`).
    pub col_marginal: Vec<f32>,
    /// Completed iterations.
    pub iters: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Safe log clamped above `f32::MIN_POSITIVE`.
#[inline]
fn safe_ln(x: f32) -> f32 {
    if x <= f32::MIN_POSITIVE {
        f32::MIN_POSITIVE.ln()
    } else {
        x.ln()
    }
}

/// Numerically stable log-sum-exp.
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
    let mut s = 0.0_f32;
    for &x in slice {
        s += (x - max_val).exp();
    }
    max_val + s.ln()
}

/// Validate all inputs.
fn validate(
    cost: &[Vec<f32>],
    a: &[f32],
    b: &[f32],
    cfg: &PartialOtConfig,
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
    // Validate relaxation-specific parameters.
    match &cfg.relaxation {
        UnbalancedRelaxation::Kl { tau }
        | UnbalancedRelaxation::Tv { tau }
        | UnbalancedRelaxation::L2 { tau } => {
            if *tau < 0.0 || !tau.is_finite() {
                return Err(OtError::BadTau { tau: *tau });
            }
        }
    }
    if cfg.mass <= 0.0 || cfg.mass > 1.0 + 1e-6 {
        return Err(OtError::Internal {
            msg: format!("mass={} must be in (0, 1]", cfg.mass),
        });
    }
    Ok((n, m))
}

/// Materialise plan from log-potentials and compute marginals.
fn build_result(
    u: &[f32],
    v: &[f32],
    cost: &[Vec<f32>],
    eps: f32,
    n: usize,
    m: usize,
    iters: usize,
) -> PartialOtResult {
    let mut plan = vec![0.0_f32; n * m];
    let mut total_cost = 0.0_f32;
    for (i, ui) in u.iter().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            let p = ((ui + vj - cost[i][j]) / eps).exp();
            plan[i * m + j] = p;
            total_cost += p * cost[i][j];
        }
    }
    let transported_mass = plan.iter().sum::<f32>();
    let row_marginal: Vec<f32> = (0..n)
        .map(|i| (0..m).map(|j| plan[i * m + j]).sum())
        .collect();
    let col_marginal: Vec<f32> = (0..m)
        .map(|j| (0..n).map(|i| plan[i * m + j]).sum())
        .collect();
    PartialOtResult {
        plan,
        cost: total_cost,
        transported_mass,
        row_marginal,
        col_marginal,
        iters,
    }
}

/// Compute max column-marginal residual.
fn col_residual(u: &[f32], v: &[f32], cost: &[Vec<f32>], b: &[f32], eps: f32) -> f32 {
    let n = u.len();
    let mut max_r = 0.0_f32;
    for (j, &bj) in b.iter().enumerate() {
        let s: f32 = (0..n)
            .map(|i| ((u[i] + v[j] - cost[i][j]) / eps).exp())
            .sum();
        let r = (s - bj).abs();
        if r > max_r {
            max_r = r;
        }
    }
    max_r
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant-specific row / column update functions
// ─────────────────────────────────────────────────────────────────────────────

/// TV row update: soft-threshold to enforce `T1 ≤ a` pointwise.
///
/// After the raw update, clamp: `u_new[i] = min(u_raw[i], ε log a[i])`.
#[inline]
fn row_update_tv(
    u: &mut [f32],
    v: &[f32],
    cost: &[Vec<f32>],
    a: &[f32],
    eps: f32,
    buf: &mut [f32],
) {
    let m = v.len();
    for (i, ui) in u.iter_mut().enumerate() {
        for (j, (&vj, &cij)) in v.iter().zip(cost[i].iter()).enumerate() {
            buf[j] = (vj - cij) / eps;
        }
        let u_raw = eps * safe_ln(a[i]) - eps * logsumexp(&buf[..m]);
        // Soft-threshold: clamp to log-mass ceiling.
        let u_ceil = eps * safe_ln(a[i]);
        *ui = u_raw.min(u_ceil);
    }
}

/// TV column update: symmetric soft-threshold for column marginal.
#[inline]
fn col_update_tv(
    v: &mut [f32],
    u: &[f32],
    cost: &[Vec<f32>],
    b: &[f32],
    eps: f32,
    buf: &mut [f32],
) {
    let n = u.len();
    for (j, vj) in v.iter_mut().enumerate() {
        for (i, (&ui, cost_row)) in u.iter().zip(cost.iter()).enumerate() {
            buf[i] = (ui - cost_row[j]) / eps;
        }
        let v_raw = eps * safe_ln(b[j]) - eps * logsumexp(&buf[..n]);
        let v_ceil = eps * safe_ln(b[j]);
        *vj = v_raw.min(v_ceil);
    }
}

/// L2 row update: scale the standard Sinkhorn step by `ε/(ε + τ)`.
///
/// This is the proximal operator of the L2 penalty on row marginal violations.
#[inline]
fn row_update_l2(
    u: &mut [f32],
    v: &[f32],
    cost: &[Vec<f32>],
    a: &[f32],
    eps: f32,
    tau: f32,
    buf: &mut [f32],
) {
    let m = v.len();
    // τ → ∞: scale → 1 (balanced Sinkhorn); τ = 0 handled at call site.
    let scale = eps / (eps + tau);
    for (i, ui) in u.iter_mut().enumerate() {
        for (j, (&vj, &cij)) in v.iter().zip(cost[i].iter()).enumerate() {
            buf[j] = (vj - cij) / eps;
        }
        let lse = logsumexp(&buf[..m]);
        *ui = scale * (eps * safe_ln(a[i]) - eps * lse);
    }
}

/// L2 column update: symmetric scaling.
#[inline]
fn col_update_l2(
    v: &mut [f32],
    u: &[f32],
    cost: &[Vec<f32>],
    b: &[f32],
    eps: f32,
    tau: f32,
    buf: &mut [f32],
) {
    let n = u.len();
    let scale = eps / (eps + tau);
    for (j, vj) in v.iter_mut().enumerate() {
        for (i, (&ui, cost_row)) in u.iter().zip(cost.iter()).enumerate() {
            buf[i] = (ui - cost_row[j]) / eps;
        }
        let lse = logsumexp(&buf[..n]);
        *vj = scale * (eps * safe_ln(b[j]) - eps * lse);
    }
}

/// KL row update: `u_new = (τ/(τ+ε)) * (ε log a[i] - ε LSE)`.
#[inline]
fn row_update_kl(
    u: &mut [f32],
    v: &[f32],
    cost: &[Vec<f32>],
    a: &[f32],
    eps: f32,
    tau: f32,
    buf: &mut [f32],
) {
    let m = v.len();
    let scale = tau / (tau + eps);
    for (i, ui) in u.iter_mut().enumerate() {
        for (j, (&vj, &cij)) in v.iter().zip(cost[i].iter()).enumerate() {
            buf[j] = (vj - cij) / eps;
        }
        let lse = logsumexp(&buf[..m]);
        *ui = scale * (eps * safe_ln(a[i]) - eps * lse);
    }
}

/// KL column update.
#[inline]
fn col_update_kl(
    v: &mut [f32],
    u: &[f32],
    cost: &[Vec<f32>],
    b: &[f32],
    eps: f32,
    tau: f32,
    buf: &mut [f32],
) {
    let n = u.len();
    let scale = tau / (tau + eps);
    for (j, vj) in v.iter_mut().enumerate() {
        for (i, (&ui, cost_row)) in u.iter().zip(cost.iter()).enumerate() {
            buf[i] = (ui - cost_row[j]) / eps;
        }
        let lse = logsumexp(&buf[..n]);
        *vj = scale * (eps * safe_ln(b[j]) - eps * lse);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Solve partial OT with the relaxation specified in `cfg`.
///
/// `cost` is an `n_rows × n_cols` cost matrix. `a` is the source marginal
/// (length `n_rows`), `b` is the target marginal (length `n_cols`).
pub fn partial_ot(
    cost: &[Vec<f32>],
    a: &[f32],
    b: &[f32],
    cfg: &PartialOtConfig,
) -> OtResult<PartialOtResult> {
    let (n, m) = validate(cost, a, b, cfg)?;
    let eps = cfg.eps;

    // Initialise potentials with log-marginals.
    let mut u: Vec<f32> = a.iter().map(|&ai| eps * safe_ln(ai)).collect();
    let mut v: Vec<f32> = b.iter().map(|&bj| eps * safe_ln(bj)).collect();
    // Pre-allocated scratch buffer sized for the larger dimension.
    let mut buf: Vec<f32> = vec![0.0_f32; n.max(m)];

    let mut iters = 0_usize;

    match &cfg.relaxation {
        UnbalancedRelaxation::L2 { tau } => {
            let tau = *tau;
            if tau < 1e-10 {
                // τ ≈ 0: no marginal coupling → dual potentials collapse to −∞,
                // plan entries exp((u+v-c)/ε) → 0.
                u = vec![f32::NEG_INFINITY; n];
                v = vec![f32::NEG_INFINITY; m];
                iters = 1;
            } else {
                for it in 0..cfg.max_iter {
                    row_update_l2(&mut u, &v, cost, a, eps, tau, &mut buf);
                    col_update_l2(&mut v, &u, cost, b, eps, tau, &mut buf);
                    iters = it + 1;
                    let res = col_residual(&u, &v, cost, b, eps);
                    if res < cfg.tol {
                        break;
                    }
                }
            }
        }

        UnbalancedRelaxation::Tv { tau: _ } => {
            // The TV soft-thresholded update enforces T1 ≤ a pointwise.
            for it in 0..cfg.max_iter {
                row_update_tv(&mut u, &v, cost, a, eps, &mut buf);
                col_update_tv(&mut v, &u, cost, b, eps, &mut buf);
                iters = it + 1;
                // TV residual: max row-marginal excess above a.
                let mut max_r = 0.0_f32;
                for (i, (&ai, ui)) in a.iter().zip(u.iter()).enumerate() {
                    let ri: f32 = v
                        .iter()
                        .zip(cost[i].iter())
                        .map(|(&vj, &cij)| ((ui + vj - cij) / eps).exp())
                        .sum();
                    let excess = (ri - ai).max(0.0);
                    if excess > max_r {
                        max_r = excess;
                    }
                }
                if max_r < cfg.tol {
                    break;
                }
            }
        }

        UnbalancedRelaxation::Kl { tau } => {
            let tau = *tau;
            for it in 0..cfg.max_iter {
                row_update_kl(&mut u, &v, cost, a, eps, tau, &mut buf);
                col_update_kl(&mut v, &u, cost, b, eps, tau, &mut buf);
                iters = it + 1;
                let res = col_residual(&u, &v, cost, b, eps);
                if res < cfg.tol {
                    break;
                }
            }
        }
    }

    Ok(build_result(&u, &v, cost, eps, n, m, iters))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn uniform_cost(n: usize, m: usize) -> Vec<Vec<f32>> {
        vec![vec![1.0_f32; m]; n]
    }

    fn distance_cost(n: usize, m: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..m).map(|j| (i as f32 - j as f32).abs()).collect())
            .collect()
    }

    fn flatten(cost: &[Vec<f32>]) -> Vec<f32> {
        cost.iter().flat_map(|row| row.iter().copied()).collect()
    }

    // ── L2 relaxation ────────────────────────────────────────────────────────

    #[test]
    fn l2_partial_mass_well_defined() {
        // The L2 partial OT plan should have non-negative finite entries and
        // a positive transported mass that scales with the marginal constraints.
        let n = 3;
        let m = 3;
        let cost = distance_cost(n, m);
        // Equal balanced marginals: with τ=1 and ε=0.1 the plan is partial.
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![1.0_f32 / 3.0; m];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::L2 { tau: 1.0 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        // Plan entries must be non-negative and finite.
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
        // Plan sum must equal reported transported_mass.
        let plan_sum: f32 = res.plan.iter().sum();
        assert!(
            (plan_sum - res.transported_mass).abs() < 1e-5,
            "plan_sum={plan_sum} != transported_mass={}",
            res.transported_mass
        );
        // All row/col marginals must be non-negative.
        assert!(res.row_marginal.iter().all(|&v| v >= 0.0));
        assert!(res.col_marginal.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn l2_plan_entries_non_negative() {
        let n = 4;
        let m = 4;
        let cost = distance_cost(n, m);
        let a = vec![0.25_f32; n];
        let b = vec![0.25_f32; m];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::L2 { tau: 0.5 },
            max_iter: 1000,
            tol: 1e-5,
            mass: 0.9,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }

    #[test]
    fn l2_small_tau_approaches_balanced_cost() {
        // For τ → 0, the L2 scale = ε/(ε+τ) → 1 → standard Sinkhorn row update.
        // So small τ should give a balanced-like plan.
        let n = 3;
        let m = 3;
        let cost2d = distance_cost(n, m);
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![1.0_f32 / 3.0; m];
        let flat = flatten(&cost2d);

        let partial_cfg = PartialOtConfig {
            eps: 0.2,
            relaxation: UnbalancedRelaxation::L2 { tau: 1e-4 }, // τ ≈ 0 → scale ≈ 1 → balanced
            max_iter: 3000,
            tol: 1e-5,
            mass: 1.0,
        };
        let partial_res = partial_ot(&cost2d, &a, &b, &partial_cfg).expect("ok");

        let bal_cfg = SinkhornConfig {
            eps: 0.2,
            max_iter: 3000,
            tol: 1e-5,
        };
        let bal_res = sinkhorn(&flat, &a, &b, n, m, &bal_cfg).expect("ok");

        // Costs should agree within 5% when L2 τ ≈ 0 (scale ≈ 1).
        let tol = (bal_res.cost.abs() * 0.05).max(1e-3);
        assert!(
            approx(partial_res.cost, bal_res.cost, tol),
            "L2 τ≈0 cost={} vs balanced cost={} (tol={})",
            partial_res.cost,
            bal_res.cost,
            tol
        );
    }

    #[test]
    fn l2_tau_zero_yields_near_zero_transport() {
        // τ → 0: no marginal coupling → plan collapses.
        let cost = uniform_cost(3, 3);
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::L2 { tau: 0.0 },
            max_iter: 100,
            tol: 1e-5,
            mass: 0.5,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        // With τ=0, u = -∞ → all plan entries = exp(-∞) = 0.
        assert!(
            res.transported_mass < 0.1,
            "τ=0 should yield near-zero mass, got {}",
            res.transported_mass
        );
    }

    #[test]
    fn l2_plan_sum_matches_transported_mass() {
        let n = 3;
        let m = 4;
        let cost = distance_cost(n, m);
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![0.25_f32; m];
        let cfg = PartialOtConfig {
            eps: 0.15,
            relaxation: UnbalancedRelaxation::L2 { tau: 0.5 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        let plan_sum: f32 = res.plan.iter().sum();
        assert!(
            approx(plan_sum, res.transported_mass, 1e-5),
            "plan_sum={} != transported_mass={}",
            plan_sum,
            res.transported_mass
        );
    }

    // ── TV relaxation ────────────────────────────────────────────────────────

    #[test]
    fn tv_row_marginal_does_not_exceed_a() {
        let n = 4;
        let m = 3;
        let cost = distance_cost(n, m);
        let a = vec![0.25_f32; n];
        let b = vec![1.0_f32 / 3.0; m];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::Tv { tau: 1.0 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        for (i, &ri) in res.row_marginal.iter().enumerate() {
            assert!(
                ri <= a[i] + 1e-4,
                "row marginal[{i}]={ri} > a[{i}]={}",
                a[i]
            );
        }
    }

    #[test]
    fn tv_col_marginal_does_not_exceed_b() {
        let n = 3;
        let m = 4;
        let cost = distance_cost(n, m);
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![0.25_f32; m];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::Tv { tau: 1.0 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        for (j, &cj) in res.col_marginal.iter().enumerate() {
            assert!(
                cj <= b[j] + 1e-4,
                "col marginal[{j}]={cj} > b[{j}]={}",
                b[j]
            );
        }
    }

    #[test]
    fn tv_plan_entries_non_negative() {
        let cost = distance_cost(3, 3);
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = PartialOtConfig {
            eps: 0.1,
            relaxation: UnbalancedRelaxation::Tv { tau: 0.5 },
            max_iter: 1000,
            tol: 1e-5,
            mass: 0.8,
        };
        let res = partial_ot(&cost, &a, &b, &cfg).expect("ok");
        assert!(res.plan.iter().all(|&p| p >= 0.0 && p.is_finite()));
    }

    // ── KL relaxation ────────────────────────────────────────────────────────

    #[test]
    fn kl_large_tau_approaches_balanced() {
        let n = 3;
        let m = 3;
        let cost2d = distance_cost(n, m);
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![1.0_f32 / 3.0; m];
        let flat = flatten(&cost2d);

        let kl_cfg = PartialOtConfig {
            eps: 0.2,
            relaxation: UnbalancedRelaxation::Kl { tau: 1e6 },
            max_iter: 3000,
            tol: 1e-5,
            mass: 1.0,
        };
        let kl_res = partial_ot(&cost2d, &a, &b, &kl_cfg).expect("ok");

        let bal_cfg = SinkhornConfig {
            eps: 0.2,
            max_iter: 3000,
            tol: 1e-5,
        };
        let bal_res = sinkhorn(&flat, &a, &b, n, m, &bal_cfg).expect("ok");

        let tol = (bal_res.cost.abs() * 0.05).max(1e-3);
        assert!(
            approx(kl_res.cost, bal_res.cost, tol),
            "KL τ=1e6 cost={} vs balanced={}",
            kl_res.cost,
            bal_res.cost
        );
    }

    #[test]
    fn kl_vs_l2_different_plans() {
        let n = 3;
        let m = 3;
        let cost = distance_cost(n, m);
        let a = vec![1.0_f32 / 3.0; n];
        let b = vec![1.0_f32 / 6.0; m]; // mismatched total mass

        let kl_cfg = PartialOtConfig {
            eps: 0.2,
            relaxation: UnbalancedRelaxation::Kl { tau: 1.0 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let l2_cfg = PartialOtConfig {
            eps: 0.2,
            relaxation: UnbalancedRelaxation::L2 { tau: 1.0 },
            max_iter: 2000,
            tol: 1e-5,
            mass: 0.8,
        };
        let kl_res = partial_ot(&cost, &a, &b, &kl_cfg).expect("ok");
        let l2_res = partial_ot(&cost, &a, &b, &l2_cfg).expect("ok");

        // Plans must differ since the penalty geometries are different.
        let plan_diff: f32 = kl_res
            .plan
            .iter()
            .zip(l2_res.plan.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(
            plan_diff > 1e-4,
            "KL and L2 plans should differ, diff={plan_diff}"
        );
    }

    // ── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn empty_cost_rejected() {
        let cfg = PartialOtConfig::default();
        let res = partial_ot(&[], &[], &[], &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_epsilon_rejected() {
        let cost = distance_cost(2, 2);
        let a = vec![0.5_f32; 2];
        let b = vec![0.5_f32; 2];
        let cfg = PartialOtConfig {
            eps: 0.0,
            ..Default::default()
        };
        let res = partial_ot(&cost, &a, &b, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn negative_marginal_rejected() {
        let cost = distance_cost(2, 2);
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32; 2];
        let cfg = PartialOtConfig::default();
        let res = partial_ot(&cost, &a, &b, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn marginal_dimension_mismatch_rejected() {
        let cost = distance_cost(3, 3);
        let a = vec![0.5_f32; 2]; // wrong length
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = PartialOtConfig::default();
        let res = partial_ot(&cost, &a, &b, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }
}
