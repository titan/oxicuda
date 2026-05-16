//! Fused Gromov-Wasserstein.
//!
//! For domains with both intra-domain structure (`C^1`, `C^2`) and inter-domain
//! features (cost `C^{xy}` of mapping a point in `X` onto one in `Y`), Fused-GW
//! combines an entropic Wasserstein term and a Gromov-Wasserstein term:
//!
//! ```text
//! min_T (1 − α) · <C^{xy}, T> + α · Σ_{ijkl} (C^1_ik − C^2_jl)² T_ij T_kl − ε · H(T)
//! ```
//!
//! Following Vayer et al., the iteration is identical to entropic GW, except
//! the inner Sinkhorn cost matrix at each outer step is
//!
//! ```text
//! M = (1 − α) · C^{xy} + α · ∇_GW(T)
//! ```
//!
//! where `∇_GW(T)_ij = − 2 · Σ_{kl} C^1_ik · T_kl · C^2_jl` is the GW gradient
//! at the current plan. With `α = 0` the algorithm reduces to the standard
//! entropic Sinkhorn; with `α = 1` it recovers `entropic_gw`.

use crate::error::{OtError, OtResult};
use crate::gromov::gromov_wasserstein::{GwConfig, GwResult};
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for Fused-GW.
#[derive(Debug, Clone)]
pub struct FgwConfig {
    /// Mixing weight `α ∈ [0, 1]` between Wasserstein (`α = 0`) and GW (`α = 1`).
    pub alpha: f32,
    /// Underlying entropic GW configuration.
    pub gw: GwConfig,
}

impl Default for FgwConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            gw: GwConfig::default(),
        }
    }
}

/// Validate Fused-GW inputs.
fn validate(
    c1: &[f32],
    c2: &[f32],
    cxy: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &FgwConfig,
) -> OtResult<()> {
    if m == 0 || n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.gw.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.gw.eps });
    }
    if !cfg.alpha.is_finite() || !(0.0..=1.0).contains(&cfg.alpha) {
        return Err(OtError::Internal {
            msg: format!("alpha must be in [0, 1], got {}", cfg.alpha),
        });
    }
    if c1.len() != m * m || c2.len() != n * n || cxy.len() != m * n {
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
    for slice in [c1, c2, cxy].iter() {
        for &c in slice.iter() {
            if !c.is_finite() {
                return Err(OtError::Internal {
                    msg: "non-finite cost matrix entry".into(),
                });
            }
        }
    }
    Ok(())
}

/// Compute the GW gradient at the current plan,
/// `G_ij = − 2 · Σ_{kl} C^1_ik · T_kl · C^2_jl`, in `O(m^2 n + m n^2)`.
fn gw_gradient(c1: &[f32], c2: &[f32], plan: &[f32], m: usize, n: usize) -> Vec<f32> {
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

/// Compute the Fused-GW loss
/// `L(T) = (1 − α) · <C^{xy}, T> + α · Σ_{ijkl} (C^1_ik − C^2_jl)² T_ij T_kl`.
fn fgw_loss(
    c1: &[f32],
    c2: &[f32],
    cxy: &[f32],
    plan: &[f32],
    m: usize,
    n: usize,
    alpha: f32,
) -> f32 {
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
    let cross_grad = gw_gradient(c1, c2, plan, m, n);
    let mut cross = 0.0_f32;
    for i in 0..m {
        for j in 0..n {
            cross += plan[i * n + j] * cross_grad[i * n + j];
        }
    }
    let gw_part = term1 + term2 + cross;

    let mut wasserstein_part = 0.0_f32;
    for (idx, &p) in plan.iter().enumerate() {
        wasserstein_part += p * cxy[idx];
    }

    (1.0 - alpha) * wasserstein_part + alpha * gw_part
}

/// Frobenius norm of `a − b`.
fn frob_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        acc += d * d;
    }
    acc.sqrt()
}

/// Solve Fused-GW.
///
/// `cxy` is the inter-domain cost (`m × n`); the rest of the inputs match
/// `entropic_gw`.
pub fn fused_gw(
    c1: &[f32],
    c2: &[f32],
    cxy: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    cfg: &FgwConfig,
) -> OtResult<GwResult> {
    validate(c1, c2, cxy, a, b, m, n, cfg)?;

    let mut plan = vec![0.0_f32; m * n];
    for (i, &ai) in a.iter().enumerate() {
        let row_off = i * n;
        for (j, &bj) in b.iter().enumerate() {
            plan[row_off + j] = ai * bj;
        }
    }

    let inner_cfg = SinkhornConfig {
        eps: cfg.gw.eps,
        max_iter: cfg.gw.inner_max_iter,
        tol: cfg.gw.tol,
    };

    let mut completed = 0_usize;
    let mut cost_buf = vec![0.0_f32; m * n];
    for it in 0..cfg.gw.max_iter {
        let g = gw_gradient(c1, c2, &plan, m, n);
        for (idx, slot) in cost_buf.iter_mut().enumerate() {
            *slot = (1.0 - cfg.alpha) * cxy[idx] + cfg.alpha * g[idx];
        }
        let res = sinkhorn(&cost_buf, a, b, m, n, &inner_cfg)?;
        let new_plan = res.plan;
        let delta = frob_diff(&plan, &new_plan);
        plan = new_plan;
        completed = it + 1;
        if delta < cfg.gw.tol {
            break;
        }
    }

    let loss = fgw_loss(c1, c2, cxy, &plan, m, n, cfg.alpha);
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

    #[test]
    fn alpha_validation_rejects_out_of_range() {
        let cfg = FgwConfig {
            alpha: 1.5,
            gw: GwConfig::default(),
        };
        let c1 = vec![0.0_f32; 4];
        let c2 = vec![0.0_f32; 4];
        let cxy = vec![0.0_f32; 4];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let res = fused_gw(&c1, &c2, &cxy, &a, &b, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn alpha_zero_matches_sinkhorn_marginals() {
        // With α = 0, FGW reduces to entropic Sinkhorn on cxy. Check that the
        // resulting plan satisfies row and column marginals.
        let m = 2;
        let n = 2;
        let cxy = vec![0.0_f32, 1.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let c1 = vec![0.0_f32, 1.0, 1.0, 0.0];
        let c2 = vec![0.0_f32, 1.0, 1.0, 0.0];
        let cfg = FgwConfig {
            alpha: 0.0,
            gw: GwConfig {
                eps: 0.05,
                max_iter: 30,
                inner_max_iter: 500,
                tol: 1e-5,
            },
        };
        let res = fused_gw(&c1, &c2, &cxy, &a, &b, m, n, &cfg).expect("converges");
        for i in 0..m {
            let row: f32 = (0..n).map(|j| res.plan[i * n + j]).sum();
            assert!(approx(row, 0.5, 1e-2), "row {i} sum {row}");
        }
        for j in 0..n {
            let col: f32 = (0..m).map(|i| res.plan[i * n + j]).sum();
            assert!(approx(col, 0.5, 1e-2));
        }
    }

    #[test]
    fn alpha_one_matches_entropic_gw() {
        // With α = 1, FGW must agree with entropic_gw to numerical precision.
        let m = 3;
        let n = 3;
        let c1 = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let c2 = c1.clone();
        let cxy = vec![10.0_f32; m * n]; // wildly off-scale, must be ignored.
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let gw_cfg = GwConfig {
            eps: 0.1,
            max_iter: 30,
            inner_max_iter: 500,
            tol: 1e-4,
        };
        let cfg = FgwConfig {
            alpha: 1.0,
            gw: gw_cfg.clone(),
        };
        let fgw = fused_gw(&c1, &c2, &cxy, &a, &b, m, n, &cfg).expect("converges");
        let gw = crate::gromov::gromov_wasserstein::entropic_gw(&c1, &c2, &a, &b, m, n, &gw_cfg)
            .expect("converges");
        // Plans should match closely.
        for (p_fgw, p_gw) in fgw.plan.iter().zip(gw.plan.iter()) {
            assert!(
                approx(*p_fgw, *p_gw, 1e-2),
                "mismatch {} vs {}",
                p_fgw,
                p_gw
            );
        }
    }

    #[test]
    fn alpha_half_yields_valid_plan() {
        let m = 2;
        let n = 2;
        let c1 = vec![0.0_f32, 1.0, 1.0, 0.0];
        let c2 = vec![0.0_f32, 1.0, 1.0, 0.0];
        let cxy = vec![0.0_f32, 1.0, 1.0, 0.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let cfg = FgwConfig {
            alpha: 0.5,
            gw: GwConfig {
                eps: 0.1,
                max_iter: 30,
                inner_max_iter: 500,
                tol: 1e-4,
            },
        };
        let res = fused_gw(&c1, &c2, &cxy, &a, &b, m, n, &cfg).expect("converges");
        let total: f32 = res.plan.iter().sum();
        assert!(approx(total, 1.0, 5e-2), "total mass {total}");
        for &p in &res.plan {
            assert!(p >= -1e-6 && p.is_finite());
        }
    }

    #[test]
    fn shape_validation_rejects_bad_cxy() {
        let cfg = FgwConfig::default();
        let c1 = vec![0.0_f32; 4];
        let c2 = vec![0.0_f32; 4];
        let cxy = vec![0.0_f32; 5]; // wrong size.
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let res = fused_gw(&c1, &c2, &cxy, &a, &b, 2, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }
}
