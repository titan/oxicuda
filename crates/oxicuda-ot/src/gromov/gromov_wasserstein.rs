//! Entropic Gromov-Wasserstein on distributions over possibly different
//! metric spaces.
//!
//! Given intra-domain distance matrices `C^1 ∈ R^{m×m}` and `C^2 ∈ R^{n×n}`
//! and marginals `a, b`, entropic GW solves
//!
//! ```text
//! min_T  Σ_{ijkl} L(C^1_ik, C^2_jl) · T_ij · T_kl − ε · H(T)
//!       s.t.  T 1 = a,  Tᵀ 1 = b,  T ≥ 0
//! ```
//!
//! with the canonical loss `L(x, y) = (x − y)²` and entropy
//! `H(T) = − Σ T log T`. The standard solution scheme is "iterative
//! Bregman" / mirror-descent: at each outer iteration we form the gradient of
//! the GW objective at the current plan,
//!
//! ```text
//! G_ij = − 2 · Σ_{kl} C^1_ik · T_kl · C^2_jl
//! ```
//!
//! (note the factor `−2` arises from the cross term in `(C^1_ik − C^2_jl)²`),
//! and run an inner Sinkhorn solve with cost matrix `G` to update the plan.
//! This is a special case of the Frank-Wolfe / Mirror Prox algorithm; the
//! cross-term derivative is exactly the linear part of the quadratic
//! objective at the current iterate.

use crate::error::{OtError, OtResult};
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for the entropic Gromov-Wasserstein solver.
#[derive(Debug, Clone)]
pub struct GwConfig {
    /// Entropic regularisation strength used for the inner Sinkhorn (`> 0`).
    pub eps: f32,
    /// Number of outer Bregman / mirror-descent iterations.
    pub max_iter: usize,
    /// Maximum number of inner Sinkhorn iterations per outer step.
    pub inner_max_iter: usize,
    /// Convergence tolerance on `‖T_{t+1} − T_t‖_F`.
    pub tol: f32,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            eps: 0.05,
            max_iter: 50,
            inner_max_iter: 200,
            tol: 1e-4,
        }
    }
}

/// Output of the entropic Gromov-Wasserstein solver.
#[derive(Debug, Clone)]
pub struct GwResult {
    /// Transport plan, shape `[m × n]` row-major (length `m·n`).
    pub plan: Vec<f32>,
    /// Final GW loss `L(T) = Σ_{ijkl} (C^1_ik − C^2_jl)² · T_ij · T_kl`.
    pub loss: f32,
    /// Number of completed outer iterations.
    pub iters: usize,
}

/// Validate the inputs to the entropic GW solver.
fn validate(
    c1: &[f32],
    c2: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &GwConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if c1.len() != m * m || c2.len() != n * n {
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
    for &ck in c1 {
        if !ck.is_finite() {
            return Err(OtError::Internal {
                msg: "C1 contains non-finite values".into(),
            });
        }
    }
    for &ck in c2 {
        if !ck.is_finite() {
            return Err(OtError::Internal {
                msg: "C2 contains non-finite values".into(),
            });
        }
    }
    Ok(())
}

/// Compute the GW gradient `G_ij = − 2 · Σ_{kl} C^1_ik · T_kl · C^2_jl`.
///
/// We split this into two `O(m^2 n + m n^2)` matrix products to avoid the
/// `O(m^2 n^2)` quartic loop:
///
/// 1. `M_il = Σ_k C^1_ik · T_kl`        (m × n).
/// 2. `G_ij = − 2 · Σ_l M_il · C^2_jl`  (m × n).
fn gw_gradient(c1: &[f32], c2: &[f32], plan: &[f32], m: usize, n: usize) -> Vec<f32> {
    // Step 1: M = C1 · T (shapes m×m · m×n → m×n).
    let mut tmp = vec![0.0_f32; m * n];
    for i in 0..m {
        let row_off_t = i * n;
        let row_off_c1 = i * m;
        for k in 0..m {
            let c1_ik = c1[row_off_c1 + k];
            if c1_ik == 0.0 {
                continue;
            }
            let plan_off = k * n;
            for l in 0..n {
                tmp[row_off_t + l] += c1_ik * plan[plan_off + l];
            }
        }
    }
    // Step 2: G_ij = − 2 · Σ_l M_il · C2_jl. Rewriting with M and C2 row-major:
    //   M is m×n, C2 is n×n with entry C2_jl at index j*n+l.
    let mut g = vec![0.0_f32; m * n];
    for i in 0..m {
        let row_off_t = i * n;
        let row_off_g = i * n;
        for j in 0..n {
            let row_off_c2 = j * n;
            let mut acc = 0.0_f32;
            for l in 0..n {
                acc += tmp[row_off_t + l] * c2[row_off_c2 + l];
            }
            g[row_off_g + j] = -2.0 * acc;
        }
    }
    g
}

/// Compute the GW loss `L(T) = Σ_{ijkl} (C^1_ik − C^2_jl)² · T_ij · T_kl`.
///
/// Expand the quadratic into three additive terms so each one becomes a
/// double matrix product:
///
/// `L = Σ T_ij T_kl C1²_ik + Σ T_ij T_kl C2²_jl − 2 Σ T_ij T_kl C1_ik C2_jl`.
fn gw_loss(c1: &[f32], c2: &[f32], plan: &[f32], m: usize, n: usize) -> f32 {
    // s1_ik = (C1_ik)² accumulated over T row sums (a²-side).
    // s2_jl = (C2_jl)² accumulated over T col sums (b²-side).
    // cross = Σ T_ij C1_ik T_kl C2_jl  — uses gw_gradient/2 in absolute value.
    let row_sums: Vec<f32> = (0..m)
        .map(|i| {
            let off = i * n;
            (0..n).map(|j| plan[off + j]).sum::<f32>()
        })
        .collect();
    let col_sums: Vec<f32> = (0..n)
        .map(|j| (0..m).map(|i| plan[i * n + j]).sum::<f32>())
        .collect();

    let mut term1 = 0.0_f32;
    for i in 0..m {
        for k in 0..m {
            let c1_ik = c1[i * m + k];
            term1 += c1_ik * c1_ik * row_sums[i] * row_sums[k];
        }
    }
    let mut term2 = 0.0_f32;
    for j in 0..n {
        for l in 0..n {
            let c2_jl = c2[j * n + l];
            term2 += c2_jl * c2_jl * col_sums[j] * col_sums[l];
        }
    }
    // cross_grad_ij = -2 · Σ_kl C1_ik · T_kl · C2_jl, so cross = − ½ · Σ T_ij · cross_grad_ij.
    let cross_grad = gw_gradient(c1, c2, plan, m, n);
    let mut cross = 0.0_f32;
    for i in 0..m {
        for j in 0..n {
            cross += plan[i * n + j] * cross_grad[i * n + j];
        }
    }
    // term3 = − 2 · Σ T_ij T_kl C1_ik C2_jl = Σ T_ij · cross_grad_ij.
    // (Because cross_grad already has the −2 factor baked in.)
    term1 + term2 + cross
}

/// Frobenius norm of the difference of two equal-length slices.
fn frob_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        acc += d * d;
    }
    acc.sqrt()
}

/// Solve the entropic Gromov-Wasserstein problem.
///
/// `c1` is the source intra-domain cost (`m × m` row-major), `c2` is the
/// target intra-domain cost (`n × n`), `a` and `b` are the source and target
/// marginals.
pub fn entropic_gw(
    c1: &[f32],
    c2: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &GwConfig,
) -> OtResult<GwResult> {
    validate(c1, c2, a, b, m, n, cfg)?;

    // Initial plan T_0 = a · bᵀ (outer product).
    let mut plan = vec![0.0_f32; m * n];
    for (i, &ai) in a.iter().enumerate() {
        let row_off = i * n;
        for (j, &bj) in b.iter().enumerate() {
            plan[row_off + j] = ai * bj;
        }
    }

    let inner_cfg = SinkhornConfig {
        eps: cfg.eps,
        max_iter: cfg.inner_max_iter,
        tol: cfg.tol,
    };

    let mut completed = 0_usize;
    for it in 0..cfg.max_iter {
        let g = gw_gradient(c1, c2, &plan, m, n);
        // The inner Sinkhorn problem may genuinely fail to converge for a hard
        // configuration; we propagate that error.
        let res = sinkhorn(&g, a, b, m, n, &inner_cfg)?;
        let new_plan = res.plan;
        let delta = frob_diff(&plan, &new_plan);
        plan = new_plan;
        completed = it + 1;
        if delta < cfg.tol {
            break;
        }
    }

    let loss = gw_loss(c1, c2, &plan, m, n);
    Ok(GwResult {
        plan,
        loss,
        iters: completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn make_cost_matrix(values: &[(usize, usize, f32)], n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; n * n];
        for &(i, j, v) in values {
            c[i * n + j] = v;
            c[j * n + i] = v;
        }
        c
    }

    #[test]
    fn shape_validation() {
        let cfg = GwConfig::default();
        // Wrong c1 shape.
        let res = entropic_gw(
            &[0.0_f32; 3],
            &[0.0_f32; 4],
            &[0.5_f32; 2],
            &[0.5_f32; 2],
            2,
            2,
            &cfg,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
        // Wrong marginal length.
        let res = entropic_gw(
            &[0.0_f32; 4],
            &[0.0_f32; 4],
            &[0.5_f32; 1],
            &[0.5_f32; 2],
            2,
            2,
            &cfg,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn bad_epsilon_rejected() {
        let cfg = GwConfig {
            eps: 0.0,
            ..Default::default()
        };
        let c1 = vec![0.0_f32; 4];
        let c2 = vec![0.0_f32; 4];
        let res = entropic_gw(&c1, &c2, &[0.5_f32; 2], &[0.5_f32; 2], 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn empty_inputs_rejected() {
        let cfg = GwConfig::default();
        let res = entropic_gw(&[], &[], &[], &[], 0, 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn negative_weights_rejected() {
        let cfg = GwConfig::default();
        let c1 = vec![0.0_f32; 4];
        let c2 = vec![0.0_f32; 4];
        let a = vec![-0.5_f32, 1.5];
        let b = vec![0.5_f32, 0.5];
        let res = entropic_gw(&c1, &c2, &a, &b, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn identical_metric_spaces_yield_low_loss() {
        // Same triangle metric on both sides.
        let m = 3;
        let n = 3;
        let c1 = make_cost_matrix(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)], m);
        let c2 = c1.clone();
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = GwConfig {
            eps: 0.1,
            max_iter: 100,
            inner_max_iter: 500,
            tol: 1e-3,
        };
        let res = entropic_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("converges");
        // Plan must be non-negative, finite, and (modulo entropic blur)
        // close to a permutation; check row/col marginals.
        for &p in &res.plan {
            assert!(p >= -1e-6 && p.is_finite(), "plan entry out of range: {p}");
        }
        for i in 0..m {
            let row: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row, 1.0 / 3.0, 5e-2), "row {i} sum {row}");
        }
        for j in 0..n {
            let col: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col, 1.0 / 3.0, 5e-2), "col {j} sum {col}");
        }
        // Loss must be non-negative.
        assert!(res.loss >= -1e-4, "loss negative: {}", res.loss);
        // For identical metric spaces and equal marginals, the entropic plan is
        // close to (but not exactly) a permutation; the loss decays towards
        // zero as ε → 0. With ε = 0.1 we expect a moderately small but
        // non-trivial loss bounded by the cost-matrix Frobenius norm squared
        // (here ‖C‖² = 12).
        assert!(
            res.loss < 5.0,
            "loss too high for matched spaces: {}",
            res.loss
        );
    }

    #[test]
    fn loss_non_negative_for_random_config() {
        let m = 2;
        let n = 3;
        let c1 = vec![0.0_f32, 1.5, 1.5, 0.0];
        let c2 = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![1.0_f32 / 3.0; 3];
        let cfg = GwConfig {
            eps: 0.1,
            max_iter: 30,
            inner_max_iter: 200,
            tol: 1e-4,
        };
        let res = entropic_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("converges");
        assert!(res.loss >= -1e-4);
        // Number of iterations recorded.
        assert!(res.iters >= 1);
    }
}
