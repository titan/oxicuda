//! KL-relaxed unbalanced optimal transport via the generalised log-domain
//! Sinkhorn iteration.
//!
//! We solve
//!
//! ```text
//! min_T  <C, T> + ε · KL(T ‖ a ⊗ b) + τ_a · KL(T 1 ‖ a) + τ_b · KL(Tᵀ 1 ‖ b)
//! ```
//!
//! Following Chizat, Peyré, Schmitzer & Vialard (2018), the dual potentials
//! `f, g` satisfy the alternating fixed-point updates
//!
//! ```text
//! f_i ← (τ_a / (τ_a + ε)) · ( ε log a_i − ε · LSE_j ((g_j − C_ij)/ε) )
//! g_j ← (τ_b / (τ_b + ε)) · ( ε log b_j − ε · LSE_i ((f_i − C_ij)/ε) )
//! ```
//!
//! and the primal plan is
//!
//! ```text
//! P_ij = exp((f_i + g_j − C_ij) / ε).
//! ```
//!
//! The plan is unnormalised: marginals `P 1` and `Pᵀ 1` are pulled toward
//! `a` and `b`, but not forced to equal them.

use crate::error::{OtError, OtResult};

/// Configuration for the unbalanced Sinkhorn solver.
#[derive(Debug, Clone)]
pub struct UnbalancedConfig {
    /// Entropic regularisation strength `ε > 0`.
    pub eps: f32,
    /// KL relaxation strength on the row marginal (`> 0`).
    pub tau_a: f32,
    /// KL relaxation strength on the column marginal (`> 0`).
    pub tau_b: f32,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance: `‖P_{t+1} − P_t‖_∞ < tol`.
    pub tol: f32,
}

impl Default for UnbalancedConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            tau_a: 1.0,
            tau_b: 1.0,
            max_iter: 200,
            tol: 1e-4,
        }
    }
}

/// Output of the unbalanced solver.
#[derive(Debug, Clone)]
pub struct UnbalancedResult {
    /// Transport plan, shape `[m × n]` row-major.
    pub plan: Vec<f32>,
    /// Number of completed outer iterations.
    pub iters: usize,
}

/// Numerical floor for `τ` to avoid `0/0` in the contraction factor.
const TAU_FLOOR: f32 = 1e-6;

/// Stable log-sum-exp over a slice; returns `NEG_INFINITY` when empty.
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

/// Stable log of a non-negative `f32`. Maps `0` to `log(MIN_POSITIVE)`.
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Validate inputs and return a normalised `(τ_a, τ_b)` after clamping.
fn validate(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &UnbalancedConfig,
) -> OtResult<(f32, f32)> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if cfg.tau_a <= 0.0 || !cfg.tau_a.is_finite() {
        return Err(OtError::BadTau { tau: cfg.tau_a });
    }
    if cfg.tau_b <= 0.0 || !cfg.tau_b.is_finite() {
        return Err(OtError::BadTau { tau: cfg.tau_b });
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
    Ok((cfg.tau_a.max(TAU_FLOOR), cfg.tau_b.max(TAU_FLOOR)))
}

/// Materialise the plan from current potentials.
fn build_plan(c: &[f32], f: &[f32], g: &[f32], eps: f32, m: usize, n: usize) -> Vec<f32> {
    let mut plan = vec![0.0_f32; m * n];
    for (i, &fi) in f.iter().enumerate() {
        let row_off = i * n;
        for (j, &gj) in g.iter().enumerate() {
            plan[row_off + j] = ((fi + gj - c[row_off + j]) / eps).exp();
        }
    }
    plan
}

/// Run the KL-relaxed unbalanced Sinkhorn.
pub fn unbalanced_ot(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &UnbalancedConfig,
) -> OtResult<UnbalancedResult> {
    let (tau_a, tau_b) = validate(c, a, b, m, n, cfg)?;
    let eps = cfg.eps;

    let log_a: Vec<f32> = a.iter().map(|&v| safe_ln(v)).collect();
    let log_b: Vec<f32> = b.iter().map(|&v| safe_ln(v)).collect();

    let factor_a = tau_a / (tau_a + eps);
    let factor_b = tau_b / (tau_b + eps);

    let mut f = vec![0.0_f32; m];
    let mut g = vec![0.0_f32; n];
    let mut prev_plan = vec![0.0_f32; m * n];
    let mut buf = vec![0.0_f32; m.max(n)];

    let mut completed = 0_usize;
    for it in 0..cfg.max_iter {
        // Row update on f.
        for (i, fi_slot) in f.iter_mut().enumerate() {
            let row_off = i * n;
            for (j, slot) in buf[..n].iter_mut().enumerate() {
                *slot = (g[j] - c[row_off + j]) / eps;
            }
            let lse = logsumexp(&buf[..n]);
            *fi_slot = factor_a * (eps * log_a[i] - eps * lse);
        }
        // Column update on g.
        for (j, gj_slot) in g.iter_mut().enumerate() {
            for (i, slot) in buf[..m].iter_mut().enumerate() {
                *slot = (f[i] - c[i * n + j]) / eps;
            }
            let lse = logsumexp(&buf[..m]);
            *gj_slot = factor_b * (eps * log_b[j] - eps * lse);
        }

        let plan = build_plan(c, &f, &g, eps, m, n);

        // Convergence on the maximum entrywise change of the plan.
        let mut max_change = 0.0_f32;
        if it > 0 {
            for (p_new, p_old) in plan.iter().zip(prev_plan.iter()) {
                let d = (p_new - p_old).abs();
                if d > max_change {
                    max_change = d;
                }
            }
        } else {
            max_change = f32::INFINITY;
        }
        prev_plan = plan;
        completed = it + 1;
        if max_change < cfg.tol {
            break;
        }
    }

    Ok(UnbalancedResult {
        plan: prev_plan,
        iters: completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn shape_validation() {
        let cfg = UnbalancedConfig::default();
        let res = unbalanced_ot(&[0.0_f32; 3], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn bad_epsilon_rejected() {
        let cfg = UnbalancedConfig {
            eps: 0.0,
            ..Default::default()
        };
        let res = unbalanced_ot(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_tau_rejected() {
        let cfg = UnbalancedConfig {
            tau_a: 0.0,
            ..Default::default()
        };
        let res = unbalanced_ot(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadTau { .. })));
        let cfg = UnbalancedConfig {
            tau_b: -1.0,
            ..Default::default()
        };
        let res = unbalanced_ot(&[0.0_f32; 4], &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadTau { .. })));
    }

    #[test]
    fn empty_inputs_rejected() {
        let cfg = UnbalancedConfig::default();
        let res = unbalanced_ot(&[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn negative_weights_rejected() {
        let cfg = UnbalancedConfig::default();
        let res = unbalanced_ot(&[0.0_f32; 4], &[-0.5_f32, 1.5], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn shape_of_plan() {
        let cfg = UnbalancedConfig::default();
        let m = 3;
        let n = 4;
        let c = vec![0.5_f32; m * n];
        let a = vec![0.3_f32; m];
        let b = vec![0.25_f32; n];
        let res = unbalanced_ot(&c, &a, &b, m, n, &cfg).expect("converges");
        assert_eq!(res.plan.len(), m * n);
        assert!(res.iters >= 1);
        for &p in &res.plan {
            assert!(p >= 0.0 && p.is_finite());
        }
    }

    #[test]
    fn large_tau_approaches_balanced_sinkhorn() {
        // For very large tau, the unbalanced solution should respect the
        // marginals up to entropic blur, mirroring the balanced Sinkhorn
        // behaviour.
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = UnbalancedConfig {
            eps: 0.1,
            tau_a: 1e3,
            tau_b: 1e3,
            max_iter: 1000,
            tol: 1e-5,
        };
        let res = unbalanced_ot(&c, &a, &b, m, n, &cfg).expect("converges");
        for i in 0..m {
            let row: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row, 1.0 / 3.0, 5e-2), "row {i} sum {row}");
        }
        for j in 0..n {
            let col: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col, 1.0 / 3.0, 5e-2));
        }
    }

    #[test]
    fn small_tau_loses_mass() {
        // For very small tau, the KL penalty is weak and total transported
        // mass should be much smaller than total input mass; in particular
        // total(P) ≤ total(a) (or total(b)) up to numerical noise.
        let m = 2;
        let n = 2;
        let c = vec![10.0_f32, 10.0, 10.0, 10.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let cfg = UnbalancedConfig {
            eps: 0.1,
            tau_a: 0.01,
            tau_b: 0.01,
            max_iter: 200,
            tol: 1e-5,
        };
        let res = unbalanced_ot(&c, &a, &b, m, n, &cfg).expect("converges");
        let total: f32 = res.plan.iter().sum();
        // Mass should be at most the total source mass (1.0).
        assert!(total <= 1.0 + 1e-3, "total {total} exceeds source mass");
        // With very large cost and weak KL coupling, transported mass should
        // be markedly below 1.
        assert!(total < 0.9, "expected attenuated mass, got {total}");
    }
}
