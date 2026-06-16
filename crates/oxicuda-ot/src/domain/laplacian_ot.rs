//! Laplacian-regularised optimal transport (f64 API) — Courty et al. (2014).
//!
//! Adds a graph-Laplacian smoothness penalty to entropic OT so that source
//! samples that are *similar* (large affinity `S_s`) are mapped to *nearby*
//! targets, which dramatically improves OT-based domain adaptation by
//! preserving the geometry of the source manifold.  The problem solved is
//!
//! ```text
//! min_{T ∈ U(a,b)}  ⟨C, T⟩ + ε · Σ_ij T_ij (log T_ij − 1) + η · Ω_s(T),
//! ```
//!
//! with the (symmetric) Laplacian regulariser acting on the rows of the plan
//!
//! ```text
//! Ω_s(T) = Σ_{i,k} S_s[i,k] · ‖ T_i· − T_k· ‖²  =  tr( Tᵀ L_s T ),
//! L_s = D_s − S_s   (graph Laplacian of the source affinity S_s).
//! ```
//!
//! ## Generalised conditional gradient (GCG)
//! `Ω_s` is smooth (quadratic) while the entropic term and the marginal
//! polytope `U(a,b)` are handled by Sinkhorn.  Courty et al. therefore use the
//! generalised conditional-gradient scheme (Bredies 2009): linearise the smooth
//! Laplacian term around the current plan, fold its gradient
//!
//! ```text
//! ∇Ω_s(T) = 2 L_s T
//! ```
//!
//! into the transport cost, solve the resulting *entropic* OT subproblem for a
//! search plan `G`, then take a damped step `T ← (1 − γ) T + γ G` with `γ` from
//! a simple Armijo line search on the total objective.
//!
//! Reference: Courty, N., Flamary, R., & Tuia, D. (2014). *Domain adaptation
//! with regularized optimal transport.* ECML PKDD.

use crate::error::{OtError, OtResult};

/// Configuration for [`laplacian_ot`].
#[derive(Debug, Clone)]
pub struct LaplacianOtConfig {
    /// Entropic regularisation `ε > 0` for the inner Sinkhorn subproblems.
    pub reg: f64,
    /// Laplacian regularisation strength `η ≥ 0`.
    pub eta: f64,
    /// Maximum number of outer generalised-conditional-gradient iterations.
    pub max_outer: usize,
    /// Maximum number of inner Sinkhorn iterations.
    pub max_sinkhorn: usize,
    /// Convergence tolerance on the relative change of the objective.
    pub tol: f64,
}

impl Default for LaplacianOtConfig {
    fn default() -> Self {
        Self {
            reg: 0.1,
            eta: 0.1,
            max_outer: 50,
            max_sinkhorn: 200,
            tol: 1e-7,
        }
    }
}

/// Result of a Laplacian-regularised OT solve.
#[derive(Debug, Clone)]
pub struct LaplacianOtResult {
    /// Transport plan, row-major `[m × n]`.
    pub plan: Vec<f64>,
    /// Total objective `⟨C, T⟩ + η Ω_s(T)` (entropy excluded) at termination.
    pub objective: f64,
    /// Number of outer iterations performed.
    pub iters: usize,
}

/// Validate shapes and parameters.
fn validate(
    c: &[f64],
    a: &[f64],
    b: &[f64],
    s_s: &[f64],
    m: usize,
    n: usize,
    cfg: &LaplacianOtConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
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
    if s_s.len() != m * m {
        return Err(OtError::IncompatibleLength {
            a: s_s.len(),
            b: m * m,
        });
    }
    if !(cfg.reg > 0.0 && cfg.reg.is_finite()) {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if !(cfg.eta >= 0.0 && cfg.eta.is_finite()) {
        return Err(OtError::Internal {
            msg: format!("laplacian_ot: eta must be ≥ 0, got {}", cfg.eta),
        });
    }
    if a.iter().chain(b).any(|&v| v < 0.0) {
        return Err(OtError::NegativeWeight);
    }
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if (sum_a - sum_b).abs() > 1e-5 {
        return Err(OtError::MassImbalance {
            sum_a: sum_a as f32,
            sum_b: sum_b as f32,
        });
    }
    Ok(())
}

/// Symmetric graph Laplacian `L = D − S` (row-major `m × m`) from an affinity
/// matrix `S` (its symmetric part is used).
fn laplacian(s_s: &[f64], m: usize) -> Vec<f64> {
    let mut l = vec![0.0_f64; m * m];
    for i in 0..m {
        let mut deg = 0.0_f64;
        for k in 0..m {
            if k == i {
                continue;
            }
            // Symmetrised affinity between i and k.
            let sik = 0.5 * (s_s[i * m + k] + s_s[k * m + i]);
            deg += sik;
            l[i * m + k] = -sik;
        }
        // Diagonal degree term L_ii = Σ_{k≠i} S_ik.
        l[i * m + i] = deg;
    }
    l
}

/// `L · T` for row-major `L (m×m)` and `T (m×n)`, returning an `m×n` matrix.
fn lap_apply(l: &[f64], t: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..m {
                acc += l[i * m + k] * t[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Entropic OT (Sinkhorn) for cost `cost` (row-major `m×n`), returning the plan.
fn sinkhorn_plan(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    reg: f64,
    max_iter: usize,
) -> Vec<f64> {
    // Gibbs kernel K = exp(-cost/reg), stabilised by subtracting the per-call min.
    let cmin = cost.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut k = vec![0.0_f64; m * n];
    for idx in 0..m * n {
        k[idx] = (-(cost[idx] - cmin) / reg).exp();
    }
    let mut u = vec![1.0_f64; m];
    let mut v = vec![1.0_f64; n];
    for _ in 0..max_iter {
        // u = a ./ (K v)
        for i in 0..m {
            let mut kv = 0.0_f64;
            for j in 0..n {
                kv += k[i * n + j] * v[j];
            }
            u[i] = a[i] / kv.max(f64::MIN_POSITIVE);
        }
        // v = b ./ (Kᵀ u)
        for j in 0..n {
            let mut ktu = 0.0_f64;
            for i in 0..m {
                ktu += k[i * n + j] * u[i];
            }
            v[j] = b[j] / ktu.max(f64::MIN_POSITIVE);
        }
    }
    let mut t = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            t[i * n + j] = u[i] * k[i * n + j] * v[j];
        }
    }
    t
}

/// Evaluate the (entropy-excluded) objective `⟨C, T⟩ + η · tr(Tᵀ L T)`.
fn objective(c: &[f64], l: &[f64], t: &[f64], eta: f64, m: usize, n: usize) -> f64 {
    let mut linear = 0.0_f64;
    for idx in 0..m * n {
        linear += c[idx] * t[idx];
    }
    if eta == 0.0 {
        return linear;
    }
    let lt = lap_apply(l, t, m, n);
    let mut quad = 0.0_f64;
    for idx in 0..m * n {
        quad += t[idx] * lt[idx];
    }
    linear + eta * quad
}

/// Solve Laplacian-regularised OT by generalised conditional gradient.
///
/// # Arguments
/// - `c`: ground cost, row-major `[m × n]`.
/// - `a`: source marginal, length `m`.
/// - `b`: target marginal, length `n`.
/// - `s_s`: source affinity matrix, row-major `[m × m]` (its symmetric part is used).
/// - `m`, `n`: source / target sizes.
/// - `cfg`: solver configuration.
///
/// # Errors
/// - [`OtError::EmptyInput`] if `m == 0` or `n == 0`.
/// - [`OtError::MarginalMismatch`] / [`OtError::IncompatibleLength`] on shape mismatch.
/// - [`OtError::BadEpsilon`] if `reg ≤ 0`.
/// - [`OtError::MassImbalance`] if the marginals carry different total mass.
/// - [`OtError::NegativeWeight`] if a marginal entry is negative.
#[allow(clippy::too_many_arguments)]
pub fn laplacian_ot(
    c: &[f64],
    a: &[f64],
    b: &[f64],
    s_s: &[f64],
    m: usize,
    n: usize,
    cfg: &LaplacianOtConfig,
) -> OtResult<LaplacianOtResult> {
    validate(c, a, b, s_s, m, n, cfg)?;
    let l = laplacian(s_s, m);

    // Initialise with the plain entropic OT plan.
    let mut t = sinkhorn_plan(c, a, b, m, n, cfg.reg, cfg.max_sinkhorn);
    let mut obj = objective(c, &l, &t, cfg.eta, m, n);
    let mut iters = 0_usize;

    for it in 0..cfg.max_outer {
        iters = it + 1;
        // Modified cost = C + 2η L T  (gradient of the linearised objective).
        let cost = if cfg.eta == 0.0 {
            c.to_vec()
        } else {
            let lt = lap_apply(&l, &t, m, n);
            let mut cost = vec![0.0_f64; m * n];
            for idx in 0..m * n {
                cost[idx] = c[idx] + 2.0 * cfg.eta * lt[idx];
            }
            cost
        };
        // Solve the entropic OT subproblem for the search direction G.
        let g = sinkhorn_plan(&cost, a, b, m, n, cfg.reg, cfg.max_sinkhorn);

        // Armijo line search on the true objective along T → G.
        let mut gamma = 1.0_f64;
        let mut best_t = t.clone();
        let mut best_obj = obj;
        let mut improved = false;
        for _ in 0..20 {
            let mut trial = vec![0.0_f64; m * n];
            for idx in 0..m * n {
                trial[idx] = (1.0 - gamma) * t[idx] + gamma * g[idx];
            }
            let trial_obj = objective(c, &l, &trial, cfg.eta, m, n);
            if trial_obj < best_obj {
                best_obj = trial_obj;
                best_t = trial;
                improved = true;
                break;
            }
            gamma *= 0.5;
        }

        let rel = (obj - best_obj).abs() / obj.abs().max(1e-12);
        t = best_t;
        obj = best_obj;
        if !improved || rel < cfg.tol {
            break;
        }
    }

    Ok(LaplacianOtResult {
        plan: t,
        objective: obj,
        iters,
    })
}

/// Row-normalise a transport plan into a barycentric map, returning the mapped
/// source positions `T(x_i) = Σ_j (P_ij / Σ_k P_ik) y_j` (row-major `[m × dim]`).
///
/// # Errors
/// - [`OtError::IncompatibleLength`] if the plan or target shapes are wrong.
pub fn laplacian_barycentric_map(
    plan: &[f64],
    targets: &[f64],
    m: usize,
    n: usize,
    dim: usize,
) -> OtResult<Vec<f64>> {
    if plan.len() != m * n {
        return Err(OtError::IncompatibleLength {
            a: plan.len(),
            b: m * n,
        });
    }
    if targets.len() != n * dim {
        return Err(OtError::IncompatibleLength {
            a: targets.len(),
            b: n * dim,
        });
    }
    let mut mapped = vec![0.0_f64; m * dim];
    for i in 0..m {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            row_sum += plan[i * n + j];
        }
        let inv = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
        for j in 0..n {
            let w = plan[i * n + j] * inv;
            for d in 0..dim {
                mapped[i * dim + d] += w * targets[j * dim + d];
            }
        }
    }
    Ok(mapped)
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LaplacianOtConfig {
        LaplacianOtConfig::default()
    }

    fn marginal_row_sums(plan: &[f64], m: usize, n: usize) -> Vec<f64> {
        (0..m)
            .map(|i| (0..n).map(|j| plan[i * n + j]).sum())
            .collect()
    }

    fn marginal_col_sums(plan: &[f64], m: usize, n: usize) -> Vec<f64> {
        (0..n)
            .map(|j| (0..m).map(|i| plan[i * n + j]).sum())
            .collect()
    }

    #[test]
    fn plan_respects_marginals() {
        let m = 3;
        let n = 3;
        let c = vec![
            0.0, 1.0, 2.0, //
            1.0, 0.0, 1.0, //
            2.0, 1.0, 0.0,
        ];
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let s = vec![0.0_f64; 9];
        let r = laplacian_ot(&c, &a, &b, &s, m, n, &cfg()).expect("ok");
        for (rs, ai) in marginal_row_sums(&r.plan, m, n).iter().zip(&a) {
            assert!((rs - ai).abs() < 1e-3, "row sum {rs} vs {ai}");
        }
        for (cs, bj) in marginal_col_sums(&r.plan, m, n).iter().zip(&b) {
            assert!((cs - bj).abs() < 1e-3, "col sum {cs} vs {bj}");
        }
    }

    #[test]
    fn zero_eta_reduces_to_sinkhorn() {
        // With η = 0 the result is just the entropic OT plan.
        let m = 2;
        let n = 2;
        let c = vec![0.0, 4.0, 4.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let s = vec![0.0_f64; 4];
        let cfg0 = LaplacianOtConfig { eta: 0.0, ..cfg() };
        let r = laplacian_ot(&c, &a, &b, &s, m, n, &cfg0).expect("ok");
        let direct = sinkhorn_plan(&c, &a, &b, m, n, cfg0.reg, cfg0.max_sinkhorn);
        for (p, d) in r.plan.iter().zip(&direct) {
            assert!((p - d).abs() < 1e-6, "plan {p} vs sinkhorn {d}");
        }
    }

    #[test]
    fn laplacian_reduces_row_dispersion() {
        // Two near-identical source points (high affinity) should be transported
        // more similarly when the Laplacian penalty is on. Measure the squared
        // row-difference of their transport rows.
        let m = 2;
        let n = 3;
        // Sources symmetric, targets distinct so a single mass column is favoured.
        let c = vec![
            0.0, 1.0, 4.0, //
            0.1, 1.1, 3.9,
        ];
        let a = vec![0.5_f64, 0.5];
        let b = vec![1.0_f64 / 3.0; 3];
        // Strong source affinity between the two points.
        let s = vec![0.0, 1.0, 1.0, 0.0];

        let row_diff =
            |plan: &[f64]| -> f64 { (0..n).map(|j| (plan[j] - plan[n + j]).powi(2)).sum::<f64>() };
        let no_lap = LaplacianOtConfig { eta: 0.0, ..cfg() };
        let with_lap = LaplacianOtConfig { eta: 1.0, ..cfg() };
        let r0 = laplacian_ot(&c, &a, &b, &s, m, n, &no_lap).expect("ok");
        let r1 = laplacian_ot(&c, &a, &b, &s, m, n, &with_lap).expect("ok");
        assert!(
            row_diff(&r1.plan) <= row_diff(&r0.plan) + 1e-9,
            "lap row diff {} should be ≤ no-lap {}",
            row_diff(&r1.plan),
            row_diff(&r0.plan)
        );
    }

    #[test]
    fn plan_nonnegative() {
        let m = 3;
        let n = 2;
        let c = vec![0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![0.5_f64, 0.5];
        let s = vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0];
        let r = laplacian_ot(&c, &a, &b, &s, m, n, &cfg()).expect("ok");
        for &p in &r.plan {
            assert!(p >= -1e-12, "negative plan entry {p}");
        }
    }

    #[test]
    fn objective_finite_and_plan_sized() {
        let m = 2;
        let n = 2;
        let c = vec![0.0, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let s = vec![0.0, 1.0, 1.0, 0.0];
        let r = laplacian_ot(&c, &a, &b, &s, m, n, &cfg()).expect("ok");
        assert_eq!(r.plan.len(), m * n);
        assert!(r.objective.is_finite());
    }

    #[test]
    fn barycentric_map_works() {
        // Plan that maps source 0 → target 0, source 1 → target 1 exactly.
        let plan = vec![0.5, 0.0, 0.0, 0.5];
        let targets = vec![10.0, -1.0, 20.0, -2.0]; // 2 targets in dim 2
        let mapped = laplacian_barycentric_map(&plan, &targets, 2, 2, 2).expect("ok");
        assert!((mapped[0] - 10.0).abs() < 1e-12);
        assert!((mapped[1] + 1.0).abs() < 1e-12);
        assert!((mapped[2] - 20.0).abs() < 1e-12);
        assert!((mapped[3] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn objective_does_not_increase() {
        // GCG must be a descent method: the final objective ≤ the initial
        // (pure-Sinkhorn) objective.
        let m = 3;
        let n = 3;
        let c = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![1.0_f64 / 3.0; 3];
        let b = vec![1.0_f64 / 3.0; 3];
        let s = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let l = laplacian(&s, m);
        let init = sinkhorn_plan(&c, &a, &b, m, n, cfg().reg, cfg().max_sinkhorn);
        let init_obj = objective(&c, &l, &init, cfg().eta, m, n);
        let r = laplacian_ot(&c, &a, &b, &s, m, n, &cfg()).expect("ok");
        assert!(
            r.objective <= init_obj + 1e-9,
            "obj {} > init {}",
            r.objective,
            init_obj
        );
    }

    #[test]
    fn rejects_mass_imbalance() {
        let m = 2;
        let n = 2;
        let c = vec![0.0, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![1.0_f64, 0.5];
        let s = vec![0.0_f64; 4];
        assert!(matches!(
            laplacian_ot(&c, &a, &b, &s, m, n, &cfg()),
            Err(OtError::MassImbalance { .. })
        ));
    }

    #[test]
    fn rejects_bad_eps() {
        let m = 2;
        let n = 2;
        let c = vec![0.0, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let s = vec![0.0_f64; 4];
        let bad = LaplacianOtConfig { reg: -1.0, ..cfg() };
        assert!(matches!(
            laplacian_ot(&c, &a, &b, &s, m, n, &bad),
            Err(OtError::BadEpsilon { .. })
        ));
    }

    #[test]
    fn rejects_wrong_affinity_shape() {
        let m = 2;
        let n = 2;
        let c = vec![0.0, 1.0, 1.0, 0.0];
        let a = vec![0.5_f64, 0.5];
        let b = vec![0.5_f64, 0.5];
        let s = vec![0.0_f64; 3]; // should be m*m = 4
        assert!(matches!(
            laplacian_ot(&c, &a, &b, &s, m, n, &cfg()),
            Err(OtError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn empty_rejected() {
        let cfg = cfg();
        assert!(matches!(
            laplacian_ot(&[], &[], &[], &[], 0, 0, &cfg),
            Err(OtError::EmptyInput)
        ));
    }
}
