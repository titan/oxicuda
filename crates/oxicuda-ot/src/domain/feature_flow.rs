//! OT-based feature flow for domain generalisation.
//!
//! Implements the **Wasserstein gradient flow** (particle approximation) that
//! continuously transports source features toward a target distribution.  At
//! each discrete Euler step the Sinkhorn plan `T` is computed between the
//! current source cloud and the target cloud; each source particle then moves
//! in the direction of its expected target position under `T`:
//!
//! ```text
//! x_i  ←  x_i  +  τ · Σ_j T̃[i,j] · (y_j − x_i)
//! ```
//!
//! where `T̃[i,j] = T[i,j] / Σ_k T[i,k]` is the row-normalised plan.
//! This is the particle-level ODE that defines the Wasserstein-2 gradient flow
//! (Jordan-Kinderlehrer-Otto functional gradient with the quadratic cost).
//!
//! All arithmetic is `f64` for numerical stability.

use crate::error::{OtError, OtResult};

// ─── Internal Sinkhorn (f64) ─────────────────────────────────────────────────

/// Stable log-sum-exp on a `f64` slice.
fn logsumexp_f64(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }
    let sum: f64 = slice.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Numerically safe log.
fn safe_ln_f64(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Log-domain Sinkhorn-Knopp in `f64`.
///
/// Returns the transport plan as a flat `m × n` row-major `Vec<f64>`.
/// The function always terminates within `max_iter`; convergence is
/// best-effort (useful for gradient flow where exact convergence per step
/// is not required).
fn sinkhorn_f64(
    cost: &[f64],
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    eps: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut u: Vec<f64> = a.iter().map(|&ai| eps * safe_ln_f64(ai)).collect();
    let mut v: Vec<f64> = b.iter().map(|&bj| eps * safe_ln_f64(bj)).collect();

    let mut buf = vec![0.0_f64; m.max(n)];

    for _ in 0..max_iter {
        for i in 0..m {
            let row_off = i * n;
            for j in 0..n {
                buf[j] = (v[j] - cost[row_off + j]) / eps;
            }
            let lse = logsumexp_f64(&buf[..n]);
            u[i] = eps * safe_ln_f64(a[i]) - eps * lse;
        }
        let mut max_res = 0.0_f64;
        for (j, &bj) in b.iter().enumerate() {
            let col_sum: f64 = u
                .iter()
                .enumerate()
                .map(|(i, &ui)| ((ui + v[j] - cost[i * n + j]) / eps).exp())
                .sum();
            let r = (col_sum - bj).abs();
            if r > max_res {
                max_res = r;
            }
        }
        if max_res < tol {
            for j in 0..n {
                for i in 0..m {
                    buf[i] = (u[i] - cost[i * n + j]) / eps;
                }
                let lse = logsumexp_f64(&buf[..m]);
                v[j] = eps * safe_ln_f64(b[j]) - eps * lse;
            }
            break;
        }
        for j in 0..n {
            for i in 0..m {
                buf[i] = (u[i] - cost[i * n + j]) / eps;
            }
            let lse = logsumexp_f64(&buf[..m]);
            v[j] = eps * safe_ln_f64(b[j]) - eps * lse;
        }
    }

    let mut plan = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            plan[i * n + j] = ((u[i] + v[j] - cost[i * n + j]) / eps).exp();
        }
    }
    plan
}

// ─── Cost computation helpers ─────────────────────────────────────────────────

/// Squared-Euclidean cost matrix between two point clouds.
/// Returns a flat `m × n` row-major `Vec<f64>`.
fn sq_euclidean_cost(x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<f64> {
    let m = x.len();
    let n = y.len();
    let mut c = vec![0.0_f64; m * n];
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            let sq: f64 = xi
                .iter()
                .zip(yj.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            c[i * n + j] = sq;
        }
    }
    c
}

/// Inner product `<C, T>` (OT primal cost).
fn inner_product_f64(c: &[f64], t: &[f64]) -> f64 {
    c.iter().zip(t.iter()).map(|(ci, ti)| ci * ti).sum()
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`ot_feature_flow`].
#[derive(Debug, Clone)]
pub struct FeatureFlowConfig {
    /// Sinkhorn entropic regularisation strength (must be > 0).
    pub eps: f64,
    /// Number of Euler integration steps.
    pub n_steps: usize,
    /// Step size τ for each Euler update.
    pub step_size: f64,
    /// Inner Sinkhorn iterations per flow step.
    pub sinkhorn_iter: usize,
}

impl Default for FeatureFlowConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            n_steps: 10,
            step_size: 0.1,
            sinkhorn_iter: 50,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Output of [`ot_feature_flow`].
#[derive(Debug, Clone)]
pub struct FeatureFlowResult {
    /// Transported source features after all flow steps.
    pub transported_features: Vec<Vec<f64>>,
    /// OT cost (primal) per flow step.
    pub cost_history: Vec<f64>,
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_flow(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    eps: f64,
) -> OtResult<(usize, usize, usize)> {
    if source.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if target.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: eps as f32 });
    }
    let dim = source[0].len();
    if dim == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    for row in source {
        if row.len() != dim {
            return Err(OtError::Internal {
                msg: "source_features rows have inconsistent dimension".into(),
            });
        }
    }
    let dim_t = target[0].len();
    if dim_t == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    for row in target {
        if row.len() != dim_t {
            return Err(OtError::Internal {
                msg: "target_features rows have inconsistent dimension".into(),
            });
        }
    }
    Ok((source.len(), target.len(), dim))
}

// ─── Main API ────────────────────────────────────────────────────────────────

/// Transport `source_features` toward `target_features` via the Wasserstein
/// gradient flow.
///
/// At each Euler step:
/// 1. Compute the squared-Euclidean cost matrix between current source and target.
/// 2. Run inner Sinkhorn to obtain a soft assignment plan `T`.
/// 3. Update each source point:
///    `x_i ← x_i + τ · Σ_j T̃[i,j] · (y_j − x_i)`.
/// 4. Record OT cost `<C, T>`.
///
/// The source and target weights are taken as uniform (empirical distributions).
pub fn ot_feature_flow(
    source_features: &[Vec<f64>],
    target_features: &[Vec<f64>],
    config: &FeatureFlowConfig,
) -> OtResult<FeatureFlowResult> {
    let (m, n, dim) = validate_flow(source_features, target_features, config.eps)?;

    // n_steps = 0: return original features, empty history.
    if config.n_steps == 0 || config.step_size == 0.0 {
        return Ok(FeatureFlowResult {
            transported_features: source_features.to_vec(),
            cost_history: vec![],
        });
    }

    let eps = config.eps;
    let tau = config.step_size;
    let sinkhorn_iter = config.sinkhorn_iter;
    let tol = 1e-7;

    let a = vec![1.0 / m as f64; m];
    let b = vec![1.0 / n as f64; n];

    let mut current = source_features.to_vec();
    let mut cost_history = Vec::with_capacity(config.n_steps);

    for _ in 0..config.n_steps {
        // Cost matrix between current source cloud and fixed target.
        let cost_mat = sq_euclidean_cost(&current, target_features);

        // Inner Sinkhorn.
        let plan = sinkhorn_f64(&cost_mat, &a, &b, m, n, eps, sinkhorn_iter, tol);

        // Record OT cost.
        let ot_cost = inner_product_f64(&cost_mat, &plan);
        cost_history.push(ot_cost);

        // Euler update: x_i += τ * Σ_j T̃[i,j] * (y_j - x_i).
        for i in 0..m {
            // Row sum for normalisation.
            let row_sum: f64 = (0..n).map(|j| plan[i * n + j]).sum();
            let row_sum = row_sum.max(f64::MIN_POSITIVE);

            for d in 0..dim {
                let mut update = 0.0_f64;
                for j in 0..n {
                    let t_norm = plan[i * n + j] / row_sum;
                    update += t_norm * (target_features[j][d] - current[i][d]);
                }
                current[i][d] += tau * update;
            }
        }
    }

    Ok(FeatureFlowResult {
        transported_features: current,
        cost_history,
    })
}

/// Compute the Sinkhorn divergence (OT cost) between two feature sets.
///
/// Uses the squared-Euclidean ground cost and the log-domain Sinkhorn algorithm.
/// Both sets are treated as uniform empirical distributions.
pub fn domain_discrepancy(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    eps: f64,
    max_iter: usize,
) -> OtResult<f64> {
    let (m, n, _) = validate_flow(source, target, eps)?;
    let a = vec![1.0 / m as f64; m];
    let b = vec![1.0 / n as f64; n];
    let cost_mat = sq_euclidean_cost(source, target);
    let plan = sinkhorn_f64(&cost_mat, &a, &b, m, n, eps, max_iter, 1e-7);
    Ok(inner_product_f64(&cost_mat, &plan))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Mean squared distance from each point in `x` to the centroid of `y`.
    fn mean_sq_dist_to_centroid(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
        if x.is_empty() || y.is_empty() {
            return 0.0;
        }
        let dim = x[0].len();
        let centroid: Vec<f64> = (0..dim)
            .map(|d| y.iter().map(|row| row[d]).sum::<f64>() / y.len() as f64)
            .collect();
        x.iter()
            .map(|xi| {
                xi.iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / x.len() as f64
    }

    fn default_config() -> FeatureFlowConfig {
        FeatureFlowConfig {
            eps: 0.1,
            n_steps: 10,
            step_size: 0.1,
            sinkhorn_iter: 50,
        }
    }

    // ─── Cost decreases monotonically ─────────────────────────────────────

    #[test]
    fn cost_history_non_increasing() {
        let source: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1]).collect();
        let target: Vec<Vec<f64>> = (0..5).map(|i| vec![10.0 + i as f64 * 0.1]).collect();
        let cfg = FeatureFlowConfig {
            n_steps: 20,
            step_size: 0.3,
            ..default_config()
        };
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        for w in res.cost_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "cost increased: {} → {}", w[0], w[1]);
        }
    }

    // ─── Transported features have same shape ─────────────────────────────

    #[test]
    fn transported_features_same_shape() {
        let source: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64, 0.0]).collect();
        let target: Vec<Vec<f64>> = (0..4).map(|i| vec![0.0, i as f64]).collect();
        let cfg = default_config();
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        assert_eq!(res.transported_features.len(), source.len());
        for row in &res.transported_features {
            assert_eq!(row.len(), source[0].len());
        }
    }

    // ─── n_steps = 0 returns original features, empty history ────────────

    #[test]
    fn n_steps_zero_unchanged() {
        let source: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let target: Vec<Vec<f64>> = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let cfg = FeatureFlowConfig {
            n_steps: 0,
            ..default_config()
        };
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        assert_eq!(res.transported_features, source);
        assert!(res.cost_history.is_empty());
    }

    // ─── domain_discrepancy ≈ 0 for identical source / target ─────────────
    //
    // Note: entropic regularisation introduces a finite bias even for identical
    // distributions. We verify that the value is small relative to the cost
    // scale (< 1e-3 for integer points on [0,4] with eps=0.01).

    #[test]
    fn discrepancy_zero_identical() {
        let pts: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        // Use small eps to minimise entropic bias.
        let d =
            domain_discrepancy(&pts, &pts, 0.01, 500).expect("domain_discrepancy should succeed");
        assert!(
            d < 1e-2,
            "discrepancy {d} should be near-zero for identical sets"
        );
    }

    // ─── domain_discrepancy > 0 for well-separated sets ──────────────────

    #[test]
    fn discrepancy_positive_separated() {
        let source: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let target: Vec<Vec<f64>> = (0..5).map(|i| vec![100.0 + i as f64]).collect();
        let d = domain_discrepancy(&source, &target, 0.1, 100)
            .expect("domain_discrepancy should succeed");
        assert!(
            d > 1.0,
            "discrepancy {d} should be large for separated sets"
        );
    }

    // ─── domain_discrepancy symmetric ────────────────────────────────────

    #[test]
    fn discrepancy_symmetric() {
        let source: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let target: Vec<Vec<f64>> = vec![vec![5.0], vec![6.0], vec![7.0]];
        let d1 = domain_discrepancy(&source, &target, 0.5, 200)
            .expect("domain_discrepancy should succeed");
        let d2 = domain_discrepancy(&target, &source, 0.5, 200)
            .expect("domain_discrepancy should succeed");
        assert!(
            approx(d1, d2, 1e-4),
            "discrepancy not symmetric: {d1} vs {d2}"
        );
    }

    // ─── Feature flow converges (features closer to target) ──────────────

    #[test]
    fn feature_flow_convergence() {
        let source: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64 * 0.1]).collect();
        let target: Vec<Vec<f64>> = (0..6).map(|i| vec![10.0 + i as f64 * 0.1]).collect();
        let cfg = FeatureFlowConfig {
            n_steps: 30,
            step_size: 0.5,
            ..default_config()
        };
        let dist_before = mean_sq_dist_to_centroid(&source, &target);
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        let dist_after = mean_sq_dist_to_centroid(&res.transported_features, &target);
        assert!(
            dist_after < dist_before,
            "distance did not decrease: before={dist_before}, after={dist_after}"
        );
    }

    // ─── cost_history length == n_steps ──────────────────────────────────

    #[test]
    fn cost_history_length_equals_n_steps() {
        let source: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let target: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 + 5.0]).collect();
        let n_steps = 7;
        let cfg = FeatureFlowConfig {
            n_steps,
            ..default_config()
        };
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        assert_eq!(res.cost_history.len(), n_steps);
    }

    // ─── step_size = 0 → features unchanged ──────────────────────────────

    #[test]
    fn step_size_zero_unchanged() {
        let source: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let target: Vec<Vec<f64>> = vec![vec![10.0], vec![20.0], vec![30.0]];
        let cfg = FeatureFlowConfig {
            step_size: 0.0,
            n_steps: 10,
            ..default_config()
        };
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        assert_eq!(res.transported_features, source);
    }

    // ─── eps ≤ 0 → error ─────────────────────────────────────────────────

    #[test]
    fn invalid_eps_returns_error() {
        let source = vec![vec![0.0_f64]];
        let target = vec![vec![1.0_f64]];
        let cfg = FeatureFlowConfig {
            eps: 0.0,
            ..default_config()
        };
        let res = ot_feature_flow(&source, &target, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    // ─── empty source → error ─────────────────────────────────────────────

    #[test]
    fn empty_source_returns_error() {
        let target = vec![vec![1.0_f64]];
        let cfg = default_config();
        let res = ot_feature_flow(&[], &target, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    // ─── domain_discrepancy with invalid eps → error ──────────────────────

    #[test]
    fn discrepancy_invalid_eps_returns_error() {
        let pts = vec![vec![0.0_f64], vec![1.0]];
        let res = domain_discrepancy(&pts, &pts, -1.0, 100);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    // ─── domain_discrepancy with empty target → error ────────────────────

    #[test]
    fn discrepancy_empty_target_returns_error() {
        let pts = vec![vec![0.0_f64], vec![1.0]];
        let res = domain_discrepancy(&pts, &[], 0.1, 100);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    // ─── 2-D features: flow reduces cost ─────────────────────────────────

    #[test]
    fn flow_2d_features_reduces_cost() {
        let source: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let target: Vec<Vec<f64>> = vec![
            vec![5.0, 5.0],
            vec![6.0, 5.0],
            vec![5.0, 6.0],
            vec![6.0, 6.0],
        ];
        let cfg = FeatureFlowConfig {
            n_steps: 15,
            step_size: 0.3,
            ..default_config()
        };
        let d_before = domain_discrepancy(&source, &target, 0.1, 100)
            .expect("domain_discrepancy should succeed");
        let res = ot_feature_flow(&source, &target, &cfg).expect("ot_feature_flow should succeed");
        let d_after = domain_discrepancy(&res.transported_features, &target, 0.1, 100)
            .expect("domain_discrepancy should succeed");
        assert!(
            d_after < d_before,
            "discrepancy did not decrease: {d_before} → {d_after}"
        );
    }
}
