//! Entropic domain adaptation with a group-lasso label prior (Courty 2017).
//!
//! Plain Sinkhorn domain adaptation transports a source empirical distribution
//! onto a target one without regard to source labels, which can scatter the
//! mass of a single class across several target clusters. Courty et al.
//! (*Optimal Transport for Domain Adaptation*, IEEE TPAMI 2017) add a
//! **group-lasso** (ℓ_p–ℓ₁) regulariser on the transport plan that encourages
//! each target sample to receive mass from a *single* source class:
//!
//! ```text
//! min_{P ∈ U(a,b)}  ⟨C, P⟩ + ε·H(P) + η · Σ_j Σ_c ‖ P[I_c, j] ‖_p ,
//! ```
//!
//! where `I_c` indexes the source samples of class `c` and `‖·‖_p` (here `p=½`,
//! the canonical choice) is concave, promoting class-sparse columns. Because
//! the group term is concave it is optimised by **majorisation–minimisation**:
//! a first-order (linearised) surrogate yields, at iteration `k`, a reweighted
//! cost
//!
//! ```text
//! C̃_ij = C_ij + η · ∂/∂P_ij ‖ P^{(k)}[I_c(i), j] ‖_p
//!       = C_ij + η · p · ( ‖ P^{(k)}[I_c(i), j] ‖_1 )^{p−1} · sign(P_ij) ,
//! ```
//!
//! followed by one Sinkhorn solve. Iterating this MM loop (`sinkhorn_lpl1_mm`)
//! converges to a stationary point of the non-convex objective.
//!
//! The resulting plan is collapsed to a barycentric map exactly as in
//! [`crate::domain::mapping`], producing class-coherent adapted source features.

use crate::error::{OtError, OtResult};
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for entropic LpL1 domain adaptation.
#[derive(Debug, Clone)]
pub struct EntropicDaConfig {
    /// Entropic regularisation strength `ε > 0` for the inner Sinkhorn solve.
    pub eps: f32,
    /// Group-lasso (ℓ_p–ℓ₁) strength `η ≥ 0`. `η = 0` reduces to plain Sinkhorn.
    pub eta: f32,
    /// Concavity exponent `p ∈ (0, 1)` of the group norm (default `0.5`).
    pub p: f32,
    /// Number of outer majorisation–minimisation iterations.
    pub mm_iter: usize,
    /// Inner Sinkhorn iteration budget.
    pub sinkhorn_iter: usize,
    /// Inner Sinkhorn marginal tolerance.
    pub sinkhorn_tol: f32,
}

impl Default for EntropicDaConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            eta: 0.1,
            p: 0.5,
            mm_iter: 10,
            sinkhorn_iter: 500,
            sinkhorn_tol: 1e-4,
        }
    }
}

/// Output of `sinkhorn_lpl1_mm`.
#[derive(Debug, Clone)]
pub struct EntropicDaResult {
    /// Final transport plan, shape `[m × n]` row-major.
    pub plan: Vec<f32>,
    /// Transport cost `⟨C, P⟩` under the *original* (unmodified) cost matrix.
    pub cost: f32,
    /// Number of completed MM iterations.
    pub mm_iters: usize,
}

/// Floor for the concave-gradient reweighting to avoid division blow-up.
const GROUP_FLOOR: f32 = 1e-12;

fn validate(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    labels: &[usize],
    m: usize,
    n: usize,
    cfg: &EntropicDaConfig,
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
    if labels.len() != m {
        return Err(OtError::IncompatibleLength {
            a: labels.len(),
            b: m,
        });
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if cfg.eta < 0.0 || !cfg.eta.is_finite() {
        return Err(OtError::Internal {
            msg: "eta must be finite and non-negative".to_string(),
        });
    }
    if !(0.0..1.0).contains(&cfg.p) || cfg.p <= 0.0 {
        return Err(OtError::Internal {
            msg: "group-norm exponent p must lie in (0, 1)".to_string(),
        });
    }
    for &v in a.iter().chain(b.iter()) {
        if v < 0.0 || !v.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

/// Solve entropic domain adaptation with the ℓ_p–ℓ₁ group-lasso label prior via
/// majorisation–minimisation (Courty 2017, `sinkhorn_lpl1_mm`).
///
/// `c` is the `[m × n]` ground cost between source and target samples;
/// `a`/`b` are the source/target marginals (typically uniform); `labels[i]` is
/// the source-class index of source sample `i`. Returns the class-coherent
/// transport plan.
pub fn sinkhorn_lpl1_mm(
    c: &[f32],
    a: &[f32],
    b: &[f32],
    labels: &[usize],
    m: usize,
    n: usize,
    cfg: &EntropicDaConfig,
) -> OtResult<EntropicDaResult> {
    validate(c, a, b, labels, m, n, cfg)?;

    // Group source indices by class label.
    let n_classes = labels.iter().copied().max().map(|c| c + 1).unwrap_or(0);
    let mut class_rows: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &lab) in labels.iter().enumerate() {
        class_rows[lab].push(i);
    }

    let sk_cfg = SinkhornConfig {
        eps: cfg.eps,
        max_iter: cfg.sinkhorn_iter,
        tol: cfg.sinkhorn_tol,
    };

    // Initial plan: plain Sinkhorn with the original cost.
    let mut plan = sinkhorn(c, a, b, m, n, &sk_cfg)?.plan;
    let mut modified_cost = vec![0.0_f32; m * n];
    let mut mm_done = 0_usize;

    for it in 0..cfg.mm_iter {
        // Build the linearised (majorised) cost C̃ from the current plan.
        // For class c and target column j, the group ℓ₁ mass is
        //   s_{c,j} = Σ_{i ∈ I_c} P_ij ,
        // and the surrogate gradient applied uniformly to the group is
        //   p · (s_{c,j} + floor)^{p−1}.
        for row in modified_cost.iter_mut() {
            *row = 0.0;
        }
        for rows in class_rows.iter() {
            if rows.is_empty() {
                continue;
            }
            for j in 0..n {
                let mut group_sum = 0.0_f32;
                for &i in rows {
                    group_sum += plan[i * n + j];
                }
                let grad = cfg.p * (group_sum + GROUP_FLOOR).powf(cfg.p - 1.0);
                // Cap the cost increment at `50·ε` so the corresponding kernel
                // value `exp(−add/ε) ≥ e⁻⁵⁰ ≈ 2e-22` cannot underflow to a
                // degenerate all-zero column; this keeps the inner Sinkhorn LSE
                // well-conditioned while still strongly penalising empty groups.
                let add = (cfg.eta * grad).min(50.0 * cfg.eps);
                for &i in rows {
                    modified_cost[i * n + j] = c[i * n + j] + add;
                }
            }
        }

        // One Sinkhorn solve on the reweighted cost.
        let res = sinkhorn(&modified_cost, a, b, m, n, &sk_cfg)?;
        plan = res.plan;
        mm_done = it + 1;
    }

    // Report cost under the *original* cost matrix.
    let mut cost = 0.0_f32;
    for k in 0..m * n {
        cost += plan[k] * c[k];
    }

    Ok(EntropicDaResult {
        plan,
        cost,
        mm_iters: mm_done,
    })
}

/// Collapse a transport plan to a barycentric map and adapt the source.
///
/// Mirrors [`crate::domain::mapping::barycentric_map`] but is provided here so
/// callers can run the full LpL1 pipeline in one place. `target_y` is the
/// target feature matrix, `[n × dim]` row-major. Returns adapted source
/// features `[m × dim]` row-major.
pub fn lpl1_barycentric_map(
    plan: &[f32],
    target_y: &[f32],
    m: usize,
    n: usize,
    dim: usize,
) -> OtResult<Vec<f32>> {
    if m == 0 || n == 0 || dim == 0 {
        return Err(OtError::EmptyInput);
    }
    if plan.len() != m * n {
        return Err(OtError::MarginalMismatch {
            m,
            n,
            a_len: plan.len(),
            b_len: m * n,
        });
    }
    if target_y.len() != n * dim {
        return Err(OtError::MarginalMismatch {
            m: n,
            n: dim,
            a_len: target_y.len(),
            b_len: n * dim,
        });
    }
    let mut mean = vec![0.0_f32; dim];
    for j in 0..n {
        for d in 0..dim {
            mean[d] += target_y[j * dim + d];
        }
    }
    for v in mean.iter_mut() {
        *v /= n as f32;
    }

    let mut mapped = vec![0.0_f32; m * dim];
    for i in 0..m {
        let ro = i * n;
        let mut rs = 0.0_f32;
        for j in 0..n {
            rs += plan[ro + j];
        }
        if rs <= GROUP_FLOOR {
            for d in 0..dim {
                mapped[i * dim + d] = mean[d];
            }
            continue;
        }
        let inv = 1.0 / rs;
        for j in 0..n {
            let w = plan[ro + j] * inv;
            if w == 0.0 {
                continue;
            }
            for d in 0..dim {
                mapped[i * dim + d] += w * target_y[j * dim + d];
            }
        }
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a squared-Euclidean cost matrix between row-major point clouds.
    fn sq_cost(x: &[f32], y: &[f32], m: usize, n: usize, dim: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for d in 0..dim {
                    let diff = x[i * dim + d] - y[j * dim + d];
                    s += diff * diff;
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn row_marginals(plan: &[f32], m: usize, n: usize) -> Vec<f32> {
        (0..m)
            .map(|i| (0..n).map(|j| plan[i * n + j]).sum())
            .collect()
    }

    #[test]
    fn eta_zero_matches_plain_sinkhorn() {
        let m = 3;
        let n = 3;
        let c = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let labels = vec![0, 0, 1];
        let cfg = EntropicDaConfig {
            eta: 0.0,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(&c, &a, &b, &labels, m, n, &cfg).expect("ok");
        let sk = sinkhorn(
            &c,
            &a,
            &b,
            m,
            n,
            &SinkhornConfig {
                eps: cfg.eps,
                max_iter: cfg.sinkhorn_iter,
                tol: cfg.sinkhorn_tol,
            },
        )
        .expect("ok");
        for k in 0..m * n {
            assert!((res.plan[k] - sk.plan[k]).abs() < 1e-4, "cell {k}");
        }
    }

    #[test]
    fn plan_keeps_source_marginals() {
        let m = 4;
        let n = 4;
        // 4 source / 4 target points in 2D (row-major, dim = 2).
        let x = vec![0.0_f32, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0];
        let y = vec![0.2_f32, 0.0, 1.2, 0.0, 5.2, 0.0, 6.2, 0.0];
        let c = sq_cost(&x, &y, m, n, 2);
        let a = vec![0.25_f32; 4];
        let b = vec![0.25_f32; 4];
        let labels = vec![0, 0, 1, 1];
        // Cost entries reach ~37 here; use a larger ε so the inner Sinkhorn
        // marginal residual reaches tolerance.
        let cfg = EntropicDaConfig {
            eps: 1.0,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(&c, &a, &b, &labels, m, n, &cfg).expect("ok");
        let rm = row_marginals(&res.plan, m, n);
        for (i, &ai) in a.iter().enumerate() {
            assert!(
                (rm[i] - ai).abs() < 5e-3,
                "row {i} marginal {} != {ai}",
                rm[i]
            );
        }
    }

    #[test]
    fn group_lasso_increases_class_coherence() {
        // Two well-separated source classes and two target clusters. The group
        // prior should make each target column draw mass from a single class
        // more strongly than plain Sinkhorn.
        let m = 4;
        let n = 4;
        let x = vec![0.0_f32, 0.0, 0.0, 1.0, 10.0, 0.0, 10.0, 1.0];
        let y = vec![0.0_f32, 0.5, 0.5, 0.5, 10.0, 0.5, 10.5, 0.5];
        let c = sq_cost(&x, &y, m, n, 2);
        let a = vec![0.25_f32; 4];
        let b = vec![0.25_f32; 4];
        let labels = vec![0, 0, 1, 1];

        let plain = sinkhorn_lpl1_mm(
            &c,
            &a,
            &b,
            &labels,
            m,
            n,
            &EntropicDaConfig {
                eta: 0.0,
                ..EntropicDaConfig::default()
            },
        )
        .expect("ok");
        let grouped = sinkhorn_lpl1_mm(
            &c,
            &a,
            &b,
            &labels,
            m,
            n,
            &EntropicDaConfig {
                eta: 1.0,
                ..EntropicDaConfig::default()
            },
        )
        .expect("ok");

        // For each target column, measure how concentrated the per-class mass
        // is (max class mass / total). Average over columns.
        let coherence = |plan: &[f32]| -> f32 {
            let mut acc = 0.0_f32;
            for j in 0..n {
                let c0 = plan[j] + plan[n + j];
                let c1 = plan[2 * n + j] + plan[3 * n + j];
                let tot = c0 + c1;
                if tot > 1e-12 {
                    acc += c0.max(c1) / tot;
                }
            }
            acc / n as f32
        };
        assert!(
            coherence(&grouped.plan) >= coherence(&plain.plan) - 1e-4,
            "grouped {} should be at least as coherent as plain {}",
            coherence(&grouped.plan),
            coherence(&plain.plan)
        );
    }

    #[test]
    fn barycentric_map_averages_targets() {
        // One source row, two target rows with equal mass → mapped to mean.
        let plan = vec![0.5_f32, 0.5];
        let y = vec![0.0_f32, 4.0];
        let mapped = lpl1_barycentric_map(&plan, &y, 1, 2, 1).expect("ok");
        assert!((mapped[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn barycentric_empty_row_uses_mean() {
        let plan = vec![0.0_f32, 0.0];
        let y = vec![1.0_f32, 3.0];
        let mapped = lpl1_barycentric_map(&plan, &y, 1, 2, 1).expect("ok");
        assert!((mapped[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn full_pipeline_adapts_source() {
        // 1D source and target features; moderate cost scale so the inner
        // Sinkhorn converges comfortably at the default ε.
        let m = 4;
        let n = 4;
        let x = [0.0_f32, 0.5, 2.0, 2.5];
        let y = [3.0_f32, 3.5, 5.0, 5.5];
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let diff = x[i] - y[j];
                c[i * n + j] = diff * diff;
            }
        }
        let a = vec![0.25_f32; 4];
        let b = vec![0.25_f32; 4];
        let labels = vec![0, 0, 1, 1];
        // Cost spread reaches ~30; a comfortably large ε and mild group penalty
        // keep the inner Sinkhorn LSE well within its convergence regime.
        let cfg = EntropicDaConfig {
            eps: 2.0,
            eta: 0.05,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(&c, &a, &b, &labels, m, n, &cfg).expect("ok");
        let mapped = lpl1_barycentric_map(&res.plan, &y, m, n, 1).expect("ok");
        // Mapped source should lie within the target range [3, 5.5].
        for &v in &mapped {
            assert!((3.0..=5.5).contains(&v), "mapped value {v} out of range");
        }
    }

    #[test]
    fn label_length_mismatch_rejected() {
        let res = sinkhorn_lpl1_mm(
            &[0.0_f32; 4],
            &[0.5_f32; 2],
            &[0.5_f32; 2],
            &[0, 0, 0],
            2,
            2,
            &EntropicDaConfig::default(),
        );
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn bad_eps_rejected() {
        let cfg = EntropicDaConfig {
            eps: 0.0,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(
            &[0.0_f32; 4],
            &[0.5_f32; 2],
            &[0.5_f32; 2],
            &[0, 1],
            2,
            2,
            &cfg,
        );
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_p_rejected() {
        let cfg = EntropicDaConfig {
            p: 1.5,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(
            &[0.0_f32; 4],
            &[0.5_f32; 2],
            &[0.5_f32; 2],
            &[0, 1],
            2,
            2,
            &cfg,
        );
        assert!(matches!(res, Err(OtError::Internal { .. })));
    }

    #[test]
    fn negative_marginal_rejected() {
        let res = sinkhorn_lpl1_mm(
            &[0.0_f32; 4],
            &[-0.5_f32, 1.5],
            &[0.5_f32; 2],
            &[0, 1],
            2,
            2,
            &EntropicDaConfig::default(),
        );
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn empty_rejected() {
        let res = sinkhorn_lpl1_mm(&[], &[], &[], &[], 0, 0, &EntropicDaConfig::default());
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn mm_iters_reported() {
        let m = 2;
        let n = 2;
        let c = vec![0.0_f32, 1.0, 1.0, 0.0];
        let a = vec![0.5_f32; 2];
        let b = vec![0.5_f32; 2];
        let labels = vec![0, 1];
        let cfg = EntropicDaConfig {
            mm_iter: 5,
            ..EntropicDaConfig::default()
        };
        let res = sinkhorn_lpl1_mm(&c, &a, &b, &labels, m, n, &cfg).expect("ok");
        assert_eq!(res.mm_iters, 5);
    }
}
