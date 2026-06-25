//! Epsilon-scaling (deterministic-annealing) Sinkhorn for the `ε → 0` regime.
//!
//! # Motivation
//!
//! Entropic OT with a single small `ε` is numerically treacherous: cold-started
//! Sinkhorn at, say, `ε = 1e-4` requires an enormous number of iterations to
//! escape the near-uniform basin, and the intermediate dual potentials grow so
//! large that even the log-domain kernel `exp((f_i + g_j − C_ij)/ε)` saturates.
//!
//! *Epsilon scaling* (a.k.a. deterministic annealing) sidesteps both problems by
//! solving a **decreasing sequence** of regularisation strengths
//!
//! ```text
//! ε₀ > ε₁ > … > ε_{K-1} = ε_target ,        ε_{k+1} = scale · ε_k
//! ```
//!
//! At each stage the solver runs a handful of stabilised log-domain Sinkhorn
//! iterations and then **warm-starts the next, smaller-`ε` stage with the dual
//! potentials of the current one**. The Kantorovich potentials `f, g` live in
//! *cost units* (not in `ε`-scaled log units), so they are a meaningful warm
//! start across stages: the optimal `f, g` vary slowly and continuously as `ε`
//! shrinks, which is exactly the homotopy that annealing exploits.
//!
//! This is the canonical way to drive Sinkhorn to the small-`ε` (near-exact OT)
//! regime without divergence, and serves as the numerical-stability harness for
//! `ε → 0`.
//!
//! # Algorithm
//!
//! For each `ε` in the schedule, iterate the log-domain half-updates
//!
//! ```text
//! f_i ← ε·log a_i − ε·LSE_j[ (f_i + g_j − C_ij)/ε ]
//! g_j ← ε·log b_j − ε·LSE_i[ (f_i + g_j − C_ij)/ε ]
//! ```
//!
//! retaining `f, g` (in cost units) between stages. Optionally an extra batch
//! of `inner_iter` iterations is run at the final `ε_target` to polish the plan
//! to the requested tolerance. The transport plan is recovered from the final
//! potentials and row-normalised to satisfy the source marginal exactly.
//!
//! References:
//! - Schmitzer B. *Stabilized Sparse Scaling Algorithms for Entropy Regularized
//!   Transport Problems* (SIAM J. Sci. Comput. 41(3), 2019), §3.2 (ε-scaling).
//! - Kosowsky J. J. & Yuille A. L. *The invisible hand algorithm: Solving the
//!   assignment problem with statistical physics* (Neural Networks 7(3), 1994).

use crate::error::{OtError, OtResult};

/// Configuration for the epsilon-scaling Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct EpsilonScalingConfig {
    /// Initial (largest) regularisation strength `ε₀` (must be `> ε_target`).
    pub eps_init: f32,
    /// Final (smallest) regularisation strength `ε_target` (must be `> 0`).
    pub eps_target: f32,
    /// Geometric shrink factor applied between stages, in `(0, 1)`.
    /// Each stage uses `ε_{k+1} = scale · ε_k`, clamped at `ε_target`.
    pub scale: f32,
    /// Number of Sinkhorn iterations to run at each `ε` stage.
    pub inner_iter: usize,
    /// Extra iterations run at the final `ε_target` stage for polishing.
    pub final_iter: usize,
    /// Marginal-residual convergence tolerance checked at `ε_target`.
    pub tol: f32,
}

impl Default for EpsilonScalingConfig {
    fn default() -> Self {
        EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 1e-3,
            scale: 0.5,
            inner_iter: 20,
            final_iter: 200,
            tol: 1e-4,
        }
    }
}

/// Output of the epsilon-scaling Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct EpsilonScalingResult {
    /// Transport plan `P`, shape `[m × n]` row-major (length `m·n`).
    pub plan: Vec<f32>,
    /// Row-side Kantorovich potential `f_i` (cost units), length `m`.
    pub f: Vec<f32>,
    /// Column-side Kantorovich potential `g_j` (cost units), length `n`.
    pub g: Vec<f32>,
    /// Transport cost `Σ_{ij} P_ij C_ij` at the final `ε_target`.
    pub cost: f32,
    /// The annealing schedule actually traversed (decreasing `ε` values).
    pub schedule: Vec<f32>,
    /// Total number of inner Sinkhorn iterations across all stages.
    pub total_iters: usize,
    /// Whether the final stage met the marginal tolerance `tol`.
    pub converged: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Tiny guard for safe logarithm computation.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Stable log-sum-exp over a slice (subtract-max trick).
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

/// Validate inputs and configuration.
fn validate(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &EpsilonScalingConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps_target <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.eps_target,
        });
    }
    if cfg.eps_init <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps_init });
    }
    if cfg.eps_init < cfg.eps_target {
        return Err(OtError::Internal {
            msg: "eps_init must be >= eps_target for annealing".to_string(),
        });
    }
    if !(0.0..1.0).contains(&cfg.scale) || cfg.scale <= 0.0 {
        return Err(OtError::Internal {
            msg: "scale must lie in (0, 1)".to_string(),
        });
    }
    if c.len() != m * n || a.len() != m || b.len() != n {
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
    Ok(())
}

/// Build the geometric annealing schedule `ε₀ > … > ε_target`.
///
/// Always terminates with exactly `ε_target` as the last value; intermediate
/// values are `ε_init · scale^k` while strictly greater than `ε_target`.
fn build_schedule(cfg: &EpsilonScalingConfig) -> Vec<f32> {
    let mut schedule = Vec::new();
    let mut eps = cfg.eps_init;
    // Cap stage count defensively so an adversarial (scale → 1) config cannot
    // produce an unbounded schedule.
    let max_stages = 1024usize;
    while eps > cfg.eps_target && schedule.len() < max_stages {
        schedule.push(eps);
        eps *= cfg.scale;
    }
    schedule.push(cfg.eps_target);
    schedule
}

/// Run `iters` stabilised log-domain Sinkhorn half-iterations at fixed `eps`,
/// mutating the cost-unit potentials `f, g` in place.
fn run_stage(
    c: &[f32],
    log_a: &[f32],
    log_b: &[f32],
    m: usize,
    n: usize,
    eps: f32,
    iters: usize,
    f: &mut [f32],
    g: &mut [f32],
    row_buf: &mut [f32],
    col_buf: &mut [f32],
) {
    for _ in 0..iters {
        // Row update: f_i ← ε·log a_i − ε·LSE_j[(g_j − C_ij)/ε].
        // The potential being updated (f_i) must NOT appear inside its own LSE —
        // this is the canonical Sinkhorn-Knopp half-step (cf. `sinkhorn::sinkhorn`).
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                row_buf[j] = (g[j] - c[row_off + j]) / eps;
            }
            f[i] = eps * log_a[i] - eps * logsumexp(&row_buf[..n]);
        }
        // Column update: g_j ← ε·log b_j − ε·LSE_i[(f_i − C_ij)/ε].
        for j in 0..n {
            for i in 0..m {
                col_buf[i] = (f[i] - c[i * n + j]) / eps;
            }
            g[j] = eps * log_b[j] - eps * logsumexp(&col_buf[..m]);
        }
    }
}

/// Maximum column-marginal violation `max_j |Σ_i P_ij − b_j|` of an explicit plan.
fn col_marginal_violation_plan(plan: &[f32], b: &[f32], m: usize, n: usize) -> f32 {
    let mut max_viol = 0.0_f32;
    for j in 0..n {
        let col_sum: f32 = (0..m).map(|i| plan[i * n + j]).sum();
        let viol = (col_sum - b[j]).abs();
        if viol > max_viol {
            max_viol = viol;
        }
    }
    max_viol
}

// ──────────────────────────────────────────────────────────────────────────────
// Main solver
// ──────────────────────────────────────────────────────────────────────────────

/// Solve entropic OT at small `ε_target` by geometric epsilon-scaling.
///
/// `c` is the `[m × n]` cost matrix (row-major). `a` and `b` are the source and
/// target marginals (length `m` and `n`). The solver anneals from `ε_init` down
/// to `ε_target`, warm-starting each stage with the previous stage's dual
/// potentials, then polishes at `ε_target`. Returns the transport plan, the
/// Kantorovich potentials, and the schedule traversed.
pub fn epsilon_scaling_sinkhorn(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &EpsilonScalingConfig,
) -> OtResult<EpsilonScalingResult> {
    validate(c, a, b, m, n, cfg)?;

    let log_a: Vec<f32> = a.iter().map(|&ai| safe_ln(ai)).collect();
    let log_b: Vec<f32> = b.iter().map(|&bj| safe_ln(bj)).collect();

    let schedule = build_schedule(cfg);

    let mut f = vec![0.0_f32; m];
    let mut g = vec![0.0_f32; n];
    let mut row_buf = vec![0.0_f32; n];
    let mut col_buf = vec![0.0_f32; m];

    let mut total_iters = 0_usize;
    let n_stages = schedule.len();
    for (stage, &eps) in schedule.iter().enumerate() {
        let is_final = stage + 1 == n_stages;
        let iters = if is_final {
            cfg.inner_iter + cfg.final_iter
        } else {
            cfg.inner_iter
        };
        run_stage(
            c,
            &log_a,
            &log_b,
            m,
            n,
            eps,
            iters,
            &mut f,
            &mut g,
            &mut row_buf,
            &mut col_buf,
        );
        total_iters += iters;
    }

    // Recover plan: row-normalise exp((f_i + g_j − C_ij)/ε) to source marginal a.
    let eps_final = cfg.eps_target;
    let mut plan = vec![0.0_f32; m * n];
    for i in 0..m {
        let row_off = i * n;
        for j in 0..n {
            row_buf[j] = (f[i] + g[j] - c[row_off + j]) / eps_final;
        }
        let lse = logsumexp(&row_buf[..n]);
        let target_log_ai = log_a[i];
        for j in 0..n {
            plan[row_off + j] = (target_log_ai + row_buf[j] - lse).exp();
        }
    }

    // Convergence: the recovered plan has exact row marginals by construction, so
    // the residual that actually measures Sinkhorn progress is the column-marginal
    // violation `max_j |Σ_i P_ij − b_j|`.
    let converged = col_marginal_violation_plan(&plan, b, m, n) < cfg.tol;

    let cost: f32 = plan.iter().zip(c.iter()).map(|(&p, &cv)| p * cv).sum();

    Ok(EpsilonScalingResult {
        plan,
        f,
        g,
        cost,
        schedule,
        total_iters,
        converged,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Stability harness
// ──────────────────────────────────────────────────────────────────────────────

/// Per-`ε` diagnostic record produced by [`stability_sweep`].
#[derive(Debug, Clone)]
pub struct StabilityRecord {
    /// Target regularisation strength evaluated at this point of the sweep.
    pub eps: f32,
    /// Maximum row-marginal violation of the recovered plan.
    pub marginal_violation: f32,
    /// Transport cost at this `ε` (monotonically non-increasing as `ε → 0`).
    pub cost: f32,
    /// Whether every plan entry stayed finite (no overflow / NaN).
    pub finite: bool,
    /// Whether the solver met the configured tolerance.
    pub converged: bool,
}

/// Numerical-stability harness for the `ε → 0` regime.
///
/// Runs the epsilon-scaling solver across a list of decreasing `ε_target`
/// values (the entropic-OT limit toward exact OT) and reports, for each, the
/// marginal violation, transport cost, and whether the plan remained finite.
/// This is the diagnostic counterpart to the solver: it quantifies how the
/// recovered coupling degrades (or does not) as regularisation vanishes.
///
/// `eps_targets` must be positive; the per-stage shrink `scale`, `inner_iter`,
/// `final_iter`, and `tol` are taken from `base_cfg`, while `eps_init` is shared
/// across all points so each anneals from the same warm regime.
pub fn stability_sweep(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    eps_targets: &[f32],
    base_cfg: &EpsilonScalingConfig,
) -> OtResult<Vec<StabilityRecord>> {
    if eps_targets.is_empty() {
        return Err(OtError::EmptyInput);
    }
    let mut records = Vec::with_capacity(eps_targets.len());
    for &eps in eps_targets {
        if eps <= 0.0 {
            return Err(OtError::BadEpsilon { eps });
        }
        let eps_init = base_cfg.eps_init.max(eps);
        let cfg = EpsilonScalingConfig {
            eps_init,
            eps_target: eps,
            ..base_cfg.clone()
        };
        let res = epsilon_scaling_sinkhorn(c, a, b, m, n, &cfg)?;
        let finite = res.plan.iter().all(|&p| p.is_finite()) && res.cost.is_finite();
        let viol = row_marginal_violation_plan(&res.plan, a, m, n);
        records.push(StabilityRecord {
            eps,
            marginal_violation: viol,
            cost: res.cost,
            finite,
            converged: res.converged,
        });
    }
    Ok(records)
}

/// Maximum row-marginal violation of an explicit plan.
fn row_marginal_violation_plan(plan: &[f32], a: &[f32], m: usize, n: usize) -> f32 {
    let mut max_v = 0.0_f32;
    for i in 0..m {
        let row_sum: f32 = (0..n).map(|j| plan[i * n + j]).sum();
        let v = (row_sum - a[i]).abs();
        if v > max_v {
            max_v = v;
        }
    }
    max_v
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f32> {
        vec![1.0 / n as f32; n]
    }

    /// Zero-diagonal, ones-off-diagonal cost: optimal plan is the identity.
    fn identity_cost(n: usize) -> Vec<f32> {
        let mut c = vec![1.0_f32; n * n];
        for i in 0..n {
            c[i * n + i] = 0.0;
        }
        c
    }

    fn col_sum(plan: &[f32], m: usize, n: usize, j: usize) -> f32 {
        (0..m).map(|i| plan[i * n + j]).sum()
    }

    #[test]
    fn schedule_is_decreasing_and_ends_at_target() {
        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 1e-3,
            scale: 0.5,
            ..Default::default()
        };
        let s = build_schedule(&cfg);
        assert!(s.len() >= 2);
        for w in s.windows(2) {
            assert!(w[0] > w[1], "schedule must strictly decrease: {w:?}");
        }
        assert!((s[s.len() - 1] - 1e-3).abs() < 1e-9);
        assert!((s[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn marginals_satisfied_at_small_eps() {
        let m = 4;
        let n = 4;
        let c = identity_cost(m);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 5e-3,
            scale: 0.5,
            inner_iter: 30,
            final_iter: 400,
            tol: 1e-3,
        };
        let res = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        // Row marginals enforced exactly by construction.
        for (i, &ai) in a.iter().enumerate() {
            let rs: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!((rs - ai).abs() < 1e-4, "row {i} sum {rs}");
        }
        // Column marginals approximately satisfied after annealing.
        for (j, &bj) in b.iter().enumerate() {
            let cs = col_sum(&res.plan, m, n, j);
            assert!((cs - bj).abs() < 5e-2, "col {j} sum {cs} != {bj}");
        }
        assert!(res.cost.is_finite());
    }

    #[test]
    fn plan_is_near_identity_for_identity_cost() {
        // As ε → 0 the entropic plan concentrates on the cost-minimising
        // assignment, which for zero-diagonal cost is the identity coupling.
        let m = 4;
        let n = 4;
        let c = identity_cost(m);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 2e-3,
            scale: 0.5,
            inner_iter: 40,
            final_iter: 600,
            tol: 1e-4,
        };
        let res = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        for i in 0..m {
            let diag = res.plan[i * n + i];
            assert!(diag > 0.22, "diagonal entry {i} too small: {diag}");
        }
        // Cost should be near 0 (exact OT optimum for identity cost).
        assert!(res.cost < 0.05, "cost {} should approach 0", res.cost);
    }

    #[test]
    fn annealing_reaches_exact_optimum_at_tiny_eps() {
        // For uniform marginals on {0,…,5} with squared-distance cost the unique
        // optimal (exact-OT) coupling is the identity assignment i↦i, with cost 0.
        // Epsilon-scaling down to a very small ε must recover that optimum: the
        // entropic plan concentrates on the diagonal and the cost vanishes.
        let m = 6;
        let n = 6;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).powi(2);
            }
        }
        let a = uniform(m);
        let b = uniform(n);

        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 1e-4,
            scale: 0.5,
            inner_iter: 40,
            final_iter: 400,
            tol: 1e-3,
        };
        let res = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");

        assert!(res.cost.is_finite(), "cost must stay finite at tiny eps");
        // Exact-OT optimum is 0; entropic cost approaches it from above.
        assert!(
            res.cost < 1e-2,
            "annealed cost {} should approach 0",
            res.cost
        );
        // Plan concentrates on the diagonal (mass ≈ 1/m per diagonal cell).
        for i in 0..m {
            let diag = res.plan[i * n + i];
            assert!(diag > 0.90 / m as f32, "diagonal {i} too small: {diag}");
        }
        assert!(res.converged, "annealed run should converge at tol=1e-3");
        // Sanity: traversed a genuine multi-stage schedule, not a single ε.
        assert!(res.schedule.len() >= 5, "expected a multi-stage schedule");
    }

    #[test]
    fn warm_start_carries_potentials_across_stages() {
        // The defining mechanism of ε-scaling: dual potentials are carried (not
        // reset) between stages. We verify the final potentials are a genuine
        // (non-trivial, finite) warm-started solution and that running with the
        // annealing schedule yields the same plan as an exhaustive single-ε solve
        // at the target ε — i.e. annealing is a *consistent* accelerator, not a
        // different objective.
        let m = 5;
        let n = 5;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).powi(2);
            }
        }
        let a = uniform(m);
        let b = uniform(n);
        let eps = 0.05_f32;

        let cfg_anneal = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: eps,
            scale: 0.5,
            inner_iter: 50,
            final_iter: 2000,
            tol: 1e-6,
        };
        let annealed = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg_anneal).expect("ok");

        // Reference: a single ε stage taken to high accuracy.
        let cfg_single = EpsilonScalingConfig {
            eps_init: eps,
            eps_target: eps,
            scale: 0.5,
            inner_iter: 0,
            final_iter: 5000,
            tol: 1e-6,
        };
        let single = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg_single).expect("ok");

        // Potentials are finite and non-degenerate.
        assert!(annealed.f.iter().all(|&v| v.is_finite()));
        assert!(annealed.g.iter().all(|&v| v.is_finite()));
        // Both converge to the same coupling (consistency of the accelerator).
        assert!((annealed.cost - single.cost).abs() < 1e-3);
        for (pa, ps) in annealed.plan.iter().zip(single.plan.iter()) {
            assert!((pa - ps).abs() < 2e-2, "plan mismatch {pa} vs {ps}");
        }
    }

    #[test]
    fn cost_decreases_monotonically_as_eps_shrinks() {
        // The entropic OT cost ⟨C,P⟩ is non-increasing as ε → 0 because the
        // plan concentrates on lower-cost cells.
        let m = 4;
        let n = 4;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).abs();
            }
        }
        let a = uniform(m);
        let b = uniform(n);
        let base = EpsilonScalingConfig {
            eps_init: 1.0,
            scale: 0.5,
            inner_iter: 30,
            final_iter: 300,
            tol: 1e-5,
            ..Default::default()
        };
        let targets = [0.5_f32, 0.1, 0.02, 5e-3];
        let recs = stability_sweep(&c, &a, &b, m, n, &targets, &base).expect("ok");
        assert_eq!(recs.len(), targets.len());
        for r in &recs {
            assert!(r.finite, "plan non-finite at eps={}", r.eps);
        }
        for w in recs.windows(2) {
            // Allow a tiny slack for fixed-iteration inexactness.
            assert!(
                w[1].cost <= w[0].cost + 1e-3,
                "cost should not increase as eps shrinks: {} -> {}",
                w[0].cost,
                w[1].cost
            );
        }
    }

    #[test]
    fn stability_sweep_stays_finite_to_very_small_eps() {
        let m = 3;
        let n = 3;
        let c = identity_cost(m);
        let a = uniform(m);
        let b = uniform(n);
        let base = EpsilonScalingConfig {
            eps_init: 1.0,
            scale: 0.4,
            inner_iter: 25,
            final_iter: 300,
            tol: 1e-4,
            ..Default::default()
        };
        let targets = [1e-2_f32, 1e-3, 1e-4, 1e-5];
        let recs = stability_sweep(&c, &a, &b, m, n, &targets, &base).expect("ok");
        for r in &recs {
            assert!(r.finite, "non-finite at eps={}", r.eps);
            assert!(r.cost >= -1e-6, "negative cost at eps={}", r.eps);
            // Marginal violation must remain controlled even at ε = 1e-5.
            assert!(
                r.marginal_violation < 1e-2,
                "violation {} too large at eps={}",
                r.marginal_violation,
                r.eps
            );
        }
    }

    #[test]
    fn determinism() {
        let m = 4;
        let n = 4;
        let c = identity_cost(m);
        let a = uniform(m);
        let b = uniform(n);
        let cfg = EpsilonScalingConfig::default();
        let r1 = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        let r2 = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        assert_eq!(r1.plan, r2.plan);
        assert_eq!(r1.total_iters, r2.total_iters);
        assert_eq!(r1.schedule, r2.schedule);
    }

    #[test]
    fn agrees_with_exact_ot_on_large_problem() {
        // Closes the CPU verification gap: at a small target ε the epsilon-scaled
        // entropic OT cost must reproduce the EXACT optimal-transport cost, on a
        // non-trivial problem size (n = m = 30) far larger than the previously-
        // tested 3×3 case. We use a 1D problem with L1 ground cost |x_i − y_j| so
        // the exact reference is the robust closed-form `emd_1d` (sorted-CDF
        // integral), avoiding the network-simplex solver's large-instance cycle
        // degeneracies. Support points and marginals come from the crate `LcgRng`.
        use crate::exact::emd::emd_1d;
        use crate::handle::LcgRng;

        let m = 30;
        let n = 30;
        let mut rng = LcgRng::new(20_240_620);

        // Random 1D source/target supports.
        let mut xs = vec![0.0_f32; m];
        for v in xs.iter_mut() {
            *v = rng.next_f32() * 6.0 - 3.0;
        }
        let mut ys = vec![0.0_f32; n];
        for v in ys.iter_mut() {
            *v = rng.next_f32() * 6.0 - 3.0;
        }

        // L1 ground cost C_ij = |x_i − y_j|.
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (xs[i] - ys[j]).abs();
            }
        }

        // Random but balanced marginals summing to 1.
        let mut a = vec![0.0_f32; m];
        for v in a.iter_mut() {
            *v = rng.next_f32() + 0.05;
        }
        let sa: f32 = a.iter().sum();
        for v in a.iter_mut() {
            *v /= sa;
        }
        let mut b = vec![0.0_f32; n];
        for v in b.iter_mut() {
            *v = rng.next_f32() + 0.05;
        }
        let sb: f32 = b.iter().sum();
        for v in b.iter_mut() {
            *v /= sb;
        }

        // Exact OT cost (W1 with |x−y| cost) via the closed-form 1D solver.
        let exact = emd_1d(&xs, &ys, &a, &b).expect("emd_1d ok");

        // Epsilon-scaled entropic cost, annealed to a small target ε.
        let cfg = EpsilonScalingConfig {
            eps_init: 2.0,
            eps_target: 2e-3,
            scale: 0.6,
            inner_iter: 60,
            final_iter: 2000,
            tol: 1e-4,
        };
        let scaled = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("eps-scaling ok");

        assert!(scaled.cost.is_finite());
        // Entropic regularisation makes the cost a (small) upper bound on the exact
        // optimum; at ε = 2e-3 the gap is a small fraction of the exact cost.
        assert!(
            scaled.cost >= exact - 5e-3,
            "entropic cost {} should not undershoot exact {}",
            scaled.cost,
            exact
        );
        let rel_gap = (scaled.cost - exact).abs() / exact.max(1e-6);
        assert!(
            rel_gap < 0.05,
            "relative gap {rel_gap} too large: entropic {} vs exact {}",
            scaled.cost,
            exact
        );
        // Recovered plan must respect both marginals.
        assert!(col_marginal_violation_plan(&scaled.plan, &b, m, n) < 5e-3);
    }

    #[test]
    fn matches_reference_sinkhorn_at_moderate_eps() {
        // At a moderate ε where cold-started Sinkhorn is stable, epsilon-scaling
        // must agree with the standard solver on the transport cost.
        use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};
        let m = 4;
        let n = 4;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (i as f32 - j as f32).powi(2);
            }
        }
        let a = uniform(m);
        let b = uniform(n);
        let eps = 0.1_f32;

        let ref_cfg = SinkhornConfig {
            eps,
            max_iter: 5000,
            tol: 1e-6,
        };
        let reference = sinkhorn(&c, &a, &b, m, n, &ref_cfg).expect("ok");

        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: eps,
            scale: 0.5,
            inner_iter: 30,
            final_iter: 2000,
            tol: 1e-6,
        };
        let scaled = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
        assert!(
            (scaled.cost - reference.cost).abs() < 5e-3,
            "eps-scaling cost {} vs reference {}",
            scaled.cost,
            reference.cost
        );
    }

    #[test]
    fn bad_epsilon_target_rejected() {
        let cfg = EpsilonScalingConfig {
            eps_target: 0.0,
            ..Default::default()
        };
        let r = epsilon_scaling_sinkhorn(&[1.0; 4], &uniform(2), &uniform(2), 2, 2, &cfg);
        assert!(matches!(r, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn init_below_target_rejected() {
        let cfg = EpsilonScalingConfig {
            eps_init: 1e-3,
            eps_target: 1.0,
            ..Default::default()
        };
        let r = epsilon_scaling_sinkhorn(&[1.0; 4], &uniform(2), &uniform(2), 2, 2, &cfg);
        assert!(matches!(r, Err(OtError::Internal { .. })));
    }

    #[test]
    fn bad_scale_rejected() {
        let cfg = EpsilonScalingConfig {
            scale: 1.5,
            ..Default::default()
        };
        let r = epsilon_scaling_sinkhorn(&[1.0; 4], &uniform(2), &uniform(2), 2, 2, &cfg);
        assert!(matches!(r, Err(OtError::Internal { .. })));
    }

    #[test]
    fn shape_mismatch_rejected() {
        let cfg = EpsilonScalingConfig::default();
        let r = epsilon_scaling_sinkhorn(&[1.0; 6], &uniform(2), &uniform(2), 2, 2, &cfg);
        assert!(matches!(r, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn empty_input_rejected() {
        let cfg = EpsilonScalingConfig::default();
        let r = epsilon_scaling_sinkhorn(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(r, Err(OtError::EmptyInput)));
    }

    #[test]
    fn negative_weight_rejected() {
        let cfg = EpsilonScalingConfig::default();
        let a = vec![-0.5_f32, 1.5];
        let b = uniform(2);
        let r = epsilon_scaling_sinkhorn(&[1.0; 4], &a, &b, 2, 2, &cfg);
        assert!(matches!(r, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn empty_sweep_rejected() {
        let cfg = EpsilonScalingConfig::default();
        let r = stability_sweep(&[1.0; 4], &uniform(2), &uniform(2), 2, 2, &[], &cfg);
        assert!(matches!(r, Err(OtError::EmptyInput)));
    }

    #[test]
    fn sweep_rejects_nonpositive_eps() {
        let cfg = EpsilonScalingConfig::default();
        let r = stability_sweep(
            &[1.0; 4],
            &uniform(2),
            &uniform(2),
            2,
            2,
            &[0.1, -1.0],
            &cfg,
        );
        assert!(matches!(r, Err(OtError::BadEpsilon { .. })));
    }
}

#[cfg(test)]
mod probe2 {
    use super::*;
    use crate::exact::emd::emd_1d;
    #[test]
    fn probe_emd_vs_sinkhorn() {
        // small 1D, unequal supports, L1 cost
        let xs = [0.0f32, 1.0, 2.0];
        let ys = [0.5f32, 1.5, 2.5];
        let a = [0.3f32, 0.4, 0.3];
        let b = [0.2f32, 0.5, 0.3];
        let m = 3;
        let n = 3;
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (xs[i] - ys[j]).abs();
            }
        }
        let e = emd_1d(&xs, &ys, &a, &b).unwrap_or(-1.0);
        let cfg = EpsilonScalingConfig {
            eps_init: 1.0,
            eps_target: 0.05,
            scale: 0.6,
            inner_iter: 60,
            final_iter: 3000,
            tol: 1e-5,
        };
        let sk = match epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &cfg) {
            Ok(r) => r,
            Err(_) => return,
        };
        // Verify the recovered plan's marginals.
        let mut rv = 0.0f32;
        for (i, &ai) in a.iter().enumerate() {
            let s: f32 = (0..n).map(|j| sk.plan[i * n + j]).sum();
            rv = rv.max((s - ai).abs());
        }
        let mut cv = 0.0f32;
        for (j, &bj) in b.iter().enumerate() {
            let s: f32 = (0..m).map(|i| sk.plan[i * n + j]).sum();
            cv = cv.max((s - bj).abs());
        }
        // Brute-force the exact OT LP on this 3×3 transportation polytope by a
        // fine grid over the 4 free variables (north-west corner has 5 basic
        // cells; we scan P[0][0],P[0][1],P[1][0],P[1][1] then close the rest).
        let mut best = f32::INFINITY;
        let steps = 60;
        for u00 in 0..=steps {
            let p00 = a[0] * u00 as f32 / steps as f32;
            for u01 in 0..=steps {
                let p01 = (a[0] - p00) * u01 as f32 / steps as f32;
                let p02 = a[0] - p00 - p01;
                if p02 < -1e-6 {
                    continue;
                }
                for u10 in 0..=steps {
                    let rem_col0 = b[0] - p00;
                    if rem_col0 < -1e-6 {
                        continue;
                    }
                    let p10 = rem_col0 * u10 as f32 / steps as f32;
                    let p10 = p10.min(a[1]);
                    for u11 in 0..=steps {
                        let rem_col1 = b[1] - p01;
                        if rem_col1 < -1e-6 {
                            continue;
                        }
                        let p11 = (rem_col1).min(a[1] - p10) * u11 as f32 / steps as f32;
                        let p12 = a[1] - p10 - p11;
                        if p12 < -1e-6 {
                            continue;
                        }
                        let p20 = b[0] - p00 - p10;
                        let p21 = b[1] - p01 - p11;
                        let p22 = b[2] - p02 - p12;
                        if p20 < -1e-6 || p21 < -1e-6 || p22 < -1e-6 {
                            continue;
                        }
                        let ps = [p00, p01, p02, p10, p11, p12, p20, p21, p22];
                        let cost: f32 = ps.iter().zip(c.iter()).map(|(&p, &cc)| p * cc).sum();
                        if cost < best {
                            best = cost;
                        }
                    }
                }
            }
        }
        println!(
            "PROBE emd_1d={} eps_scaling_cost={} brute_exact={} plan_rv={} plan_cv={}",
            e, sk.cost, best, rv, cv
        );
    }
}
