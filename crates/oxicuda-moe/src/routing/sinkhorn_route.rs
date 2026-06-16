//! Sinkhorn optimal-transport balanced token→expert assignment.
//!
//! Implements entropic-regularised optimal transport (Cuturi 2013) applied to
//! balanced expert routing, as used by BASE-style layers
//! (Lewis et al. "BASE Layers", ICML 2021) and Sinkhorn routers
//! (Clark et al. "Unified Scaling Laws for Routed Language Models", 2022).
//!
//! Given a router-logit (negative-cost) matrix `L ∈ R^{T×E}`, we seek a
//! transport plan `P ∈ R_{+}^{T×E}` that maximises `⟨P, L⟩ - (1/λ)·H(P)`
//! subject to the marginals
//!
//! ```text
//! Σ_e P[t,e] = r[t]   (each token sends 1/T of unit mass)
//! Σ_t P[t,e] = c[e]   (each expert receives 1/E of unit mass) → perfect balance
//! ```
//!
//! The unique solution has the form `P = diag(u) · K · diag(v)` with Gibbs
//! kernel `K = exp(λ·L)` and positive scaling vectors `u ∈ R^T`, `v ∈ R^E`
//! found by alternating projections (the Sinkhorn–Knopp iteration):
//!
//! ```text
//! u ← r ⊘ (K v)     v ← c ⊘ (Kᵀ u)
//! ```
//!
//! Unlike [`crate::routing::base`] — which performs iterative proportional
//! fitting directly on a row-softmax — this module operates on dual potentials
//! in the log domain for numerical stability and exposes the transport cost and
//! per-marginal convergence, matching the canonical entropic-OT formulation.

use crate::error::{MoeError, MoeResult};

/// Configuration for Sinkhorn optimal-transport routing.
#[derive(Debug, Clone)]
pub struct SinkhornRouteConfig {
    /// Number of experts `E`.
    pub n_experts: usize,
    /// Entropic-regularisation strength `λ` (inverse temperature).
    ///
    /// Larger `λ` → sharper (closer to the unregularised assignment problem),
    /// smaller `λ` → smoother, faster-converging plan. Must be finite and `> 0`.
    pub lambda: f32,
    /// Number of Sinkhorn (alternating-projection) iterations.
    pub n_iter: usize,
    /// Numerical-stability epsilon added to denominators.
    pub eps: f32,
}

impl Default for SinkhornRouteConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            lambda: 1.0,
            n_iter: 20,
            eps: 1e-9,
        }
    }
}

/// Result of Sinkhorn optimal-transport routing.
#[derive(Debug, Clone)]
pub struct SinkhornRouteResult {
    /// Transport plan `P ∈ [0,1]^{T×E}` (row-major). Sums to `1` over all entries.
    pub plan: Vec<f32>,
    /// Hard assignment: `expert_assignments[t]` = argmax-expert for token `t`.
    pub expert_assignments: Vec<usize>,
    /// Routing weights = row-normalised plan (`Σ_e weights[t,e] = 1` per token).
    pub weights: Vec<f32>,
    /// Transport cost `⟨P, L⟩` of the recovered plan (higher = better matched).
    pub transport_cost: f32,
}

/// Run Sinkhorn optimal-transport balanced routing on router logits.
///
/// # Arguments
/// * `logits` — router logits (negative cost) `L ∈ R^{T×E}`, row-major.
/// * `n_tokens` — `T`, number of tokens.
/// * `cfg` — Sinkhorn configuration.
///
/// # Errors
/// Returns [`MoeError`] for an empty input, invalid expert count, a non-positive
/// or non-finite `lambda`, or a logit-length / `T·E` mismatch.
pub fn sinkhorn_route(
    logits: &[f32],
    n_tokens: usize,
    cfg: &SinkhornRouteConfig,
) -> MoeResult<SinkhornRouteResult> {
    if cfg.n_experts == 0 {
        return Err(MoeError::InvalidExpertCount {
            n_experts: cfg.n_experts,
        });
    }
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if !cfg.lambda.is_finite() || cfg.lambda <= 0.0 {
        return Err(MoeError::Internal {
            msg: format!("invalid lambda {}: must be finite and > 0", cfg.lambda),
        });
    }
    let n_e = cfg.n_experts;
    let expected = n_tokens * n_e;
    if logits.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: logits.len(),
        });
    }

    // Target marginals: uniform mass per token (row) and per expert (column).
    let r = 1.0_f32 / n_tokens as f32;
    let c = 1.0_f32 / n_e as f32;
    let eps = cfg.eps.max(1e-30);

    // Log-domain Gibbs kernel: log K[t,e] = λ · L[t,e], max-shifted per row for
    // numerical stability (the shift cancels exactly in the scaling recursion).
    let mut log_k = vec![0.0_f32; expected];
    for t in 0..n_tokens {
        let row = &logits[t * n_e..(t + 1) * n_e];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shift = if max_v.is_finite() { max_v } else { 0.0 };
        for e in 0..n_e {
            log_k[t * n_e + e] = cfg.lambda * (row[e] - shift);
        }
    }

    // Dual potentials in the log domain: log_u ∈ R^T, log_v ∈ R^E.
    let mut log_u = vec![0.0_f32; n_tokens];
    let mut log_v = vec![0.0_f32; n_e];
    let log_r = r.ln();
    let log_c = c.ln();

    for _ in 0..cfg.n_iter {
        // log_u[t] = log r − logsumexp_e( log_k[t,e] + log_v[e] )
        for (t, u_slot) in log_u.iter_mut().enumerate() {
            let base = t * n_e;
            let mut max_term = f32::NEG_INFINITY;
            for e in 0..n_e {
                let term = log_k[base + e] + log_v[e];
                if term > max_term {
                    max_term = term;
                }
            }
            let mut sum = 0.0_f32;
            for e in 0..n_e {
                sum += (log_k[base + e] + log_v[e] - max_term).exp();
            }
            let lse = max_term + (sum + eps).ln();
            *u_slot = log_r - lse;
        }

        // log_v[e] = log c − logsumexp_t( log_k[t,e] + log_u[t] )
        for (e, v_slot) in log_v.iter_mut().enumerate() {
            let mut max_term = f32::NEG_INFINITY;
            for t in 0..n_tokens {
                let term = log_k[t * n_e + e] + log_u[t];
                if term > max_term {
                    max_term = term;
                }
            }
            let mut sum = 0.0_f32;
            for t in 0..n_tokens {
                sum += (log_k[t * n_e + e] + log_u[t] - max_term).exp();
            }
            let lse = max_term + (sum + eps).ln();
            *v_slot = log_c - lse;
        }
    }

    // Recover the transport plan P[t,e] = exp(log_u[t] + log_k[t,e] + log_v[e]).
    let mut plan = vec![0.0_f32; expected];
    for t in 0..n_tokens {
        for e in 0..n_e {
            plan[t * n_e + e] = (log_u[t] + log_k[t * n_e + e] + log_v[e]).exp();
        }
    }

    // Transport cost ⟨P, L⟩.
    let transport_cost: f32 = plan.iter().zip(logits.iter()).map(|(&p, &l)| p * l).sum();

    // Row-normalised routing weights and hard argmax assignment.
    let mut weights = vec![0.0_f32; expected];
    let mut expert_assignments = vec![0_usize; n_tokens];
    for (t, assign) in expert_assignments.iter_mut().enumerate() {
        let base = t * n_e;
        let row_sum: f32 = plan[base..base + n_e].iter().sum();
        let denom = row_sum + eps;
        let mut best_e = 0_usize;
        let mut best_v = f32::NEG_INFINITY;
        for e in 0..n_e {
            weights[base + e] = plan[base + e] / denom;
            if plan[base + e] > best_v {
                best_v = plan[base + e];
                best_e = e;
            }
        }
        *assign = best_e;
    }

    if !transport_cost.is_finite() || weights.iter().any(|v| !v.is_finite()) {
        return Err(MoeError::NanEncountered {
            context: "sinkhorn_route".to_string(),
        });
    }

    Ok(SinkhornRouteResult {
        plan,
        expert_assignments,
        weights,
        transport_cost,
    })
}

/// Maximum deviation of the plan's marginals from the balanced targets.
///
/// Returns `max_t |Σ_e P[t,e] − 1/T| + max_e |Σ_t P[t,e] − 1/E|`.
/// A perfectly balanced doubly-stochastic plan yields `0.0`.
#[must_use]
pub fn marginal_deviation(plan: &[f32], n_tokens: usize, n_experts: usize) -> f32 {
    if n_tokens == 0 || n_experts == 0 || plan.len() != n_tokens * n_experts {
        return 0.0;
    }
    let target_r = 1.0_f32 / n_tokens as f32;
    let target_c = 1.0_f32 / n_experts as f32;

    let mut max_row = 0.0_f32;
    for t in 0..n_tokens {
        let row_sum: f32 = plan[t * n_experts..(t + 1) * n_experts].iter().sum();
        let dev = (row_sum - target_r).abs();
        if dev > max_row {
            max_row = dev;
        }
    }

    let mut col_sums = vec![0.0_f32; n_experts];
    for t in 0..n_tokens {
        for e in 0..n_experts {
            col_sums[e] += plan[t * n_experts + e];
        }
    }
    let mut max_col = 0.0_f32;
    for &cs in &col_sums {
        let dev = (cs - target_c).abs();
        if dev > max_col {
            max_col = dev;
        }
    }

    max_row + max_col
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logits(n_tokens: usize, n_experts: usize, scale: f32) -> Vec<f32> {
        (0..n_tokens * n_experts)
            .map(|i| ((i as f32) * 0.137).sin() * scale)
            .collect()
    }

    #[test]
    fn route_zero_experts_errors() {
        let cfg = SinkhornRouteConfig {
            n_experts: 0,
            ..SinkhornRouteConfig::default()
        };
        let logits = vec![0.0_f32; 4];
        assert!(matches!(
            sinkhorn_route(&logits, 4, &cfg),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn route_zero_tokens_errors() {
        let cfg = SinkhornRouteConfig {
            n_experts: 4,
            ..SinkhornRouteConfig::default()
        };
        let logits: Vec<f32> = vec![];
        assert!(matches!(
            sinkhorn_route(&logits, 0, &cfg),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn route_invalid_lambda_errors() {
        let cfg = SinkhornRouteConfig {
            n_experts: 4,
            lambda: 0.0,
            ..SinkhornRouteConfig::default()
        };
        let logits = vec![0.0_f32; 4 * 4];
        assert!(sinkhorn_route(&logits, 4, &cfg).is_err());

        let cfg_nan = SinkhornRouteConfig {
            n_experts: 4,
            lambda: f32::NAN,
            ..SinkhornRouteConfig::default()
        };
        assert!(sinkhorn_route(&logits, 4, &cfg_nan).is_err());
    }

    #[test]
    fn route_logit_length_mismatch_errors() {
        let cfg = SinkhornRouteConfig {
            n_experts: 4,
            ..SinkhornRouteConfig::default()
        };
        let logits = vec![0.0_f32; 10]; // should be 3*4 = 12
        assert!(matches!(
            sinkhorn_route(&logits, 3, &cfg),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn plan_total_mass_is_one() {
        let n_tokens = 8;
        let n_experts = 4;
        let cfg = SinkhornRouteConfig {
            n_experts,
            lambda: 1.0,
            n_iter: 50,
            ..SinkhornRouteConfig::default()
        };
        let logits = make_logits(n_tokens, n_experts, 2.0);
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        let total: f32 = res.plan.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "total mass = {total}");
    }

    #[test]
    fn plan_all_non_negative() {
        let n_tokens = 6;
        let n_experts = 3;
        let cfg = SinkhornRouteConfig {
            n_experts,
            ..SinkhornRouteConfig::default()
        };
        let logits = make_logits(n_tokens, n_experts, 3.0);
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        assert!(res.plan.iter().all(|&v| v >= 0.0 && v.is_finite()));
    }

    #[test]
    fn expert_marginals_balanced_after_convergence() {
        let n_tokens = 12;
        let n_experts = 4;
        let cfg = SinkhornRouteConfig {
            n_experts,
            lambda: 1.0,
            n_iter: 100,
            ..SinkhornRouteConfig::default()
        };
        // Deliberately skewed logits (all prefer expert 0).
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for t in 0..n_tokens {
            logits[t * n_experts] = 4.0;
        }
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        // Each expert column should carry ≈ 1/E of the mass despite the skew.
        let target_c = 1.0_f32 / n_experts as f32;
        let mut col_sums = vec![0.0_f32; n_experts];
        for t in 0..n_tokens {
            for (e, col) in col_sums.iter_mut().enumerate() {
                *col += res.plan[t * n_experts + e];
            }
        }
        for (e, &cs) in col_sums.iter().enumerate() {
            assert!(
                (cs - target_c).abs() < 1e-2,
                "expert {e} column mass {cs}, target {target_c}"
            );
        }
    }

    #[test]
    fn more_iterations_reduce_marginal_deviation() {
        let n_tokens = 10;
        let n_experts = 5;
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for t in 0..n_tokens {
            logits[t * n_experts + 1] = 5.0; // all prefer expert 1
        }
        let cfg_few = SinkhornRouteConfig {
            n_experts,
            lambda: 1.0,
            n_iter: 1,
            ..SinkhornRouteConfig::default()
        };
        let cfg_many = SinkhornRouteConfig {
            n_experts,
            lambda: 1.0,
            n_iter: 80,
            ..SinkhornRouteConfig::default()
        };
        let dev_few = marginal_deviation(
            &sinkhorn_route(&logits, n_tokens, &cfg_few)
                .expect("sinkhorn_route should succeed")
                .plan,
            n_tokens,
            n_experts,
        );
        let dev_many = marginal_deviation(
            &sinkhorn_route(&logits, n_tokens, &cfg_many)
                .expect("sinkhorn_route should succeed")
                .plan,
            n_tokens,
            n_experts,
        );
        assert!(
            dev_many <= dev_few + 1e-6,
            "more iters should not worsen balance: few={dev_few}, many={dev_many}"
        );
    }

    #[test]
    fn weights_rows_sum_to_one() {
        let n_tokens = 7;
        let n_experts = 4;
        let cfg = SinkhornRouteConfig {
            n_experts,
            ..SinkhornRouteConfig::default()
        };
        let logits = make_logits(n_tokens, n_experts, 1.5);
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        for t in 0..n_tokens {
            let row_sum: f32 = res.weights[t * n_experts..(t + 1) * n_experts].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "token {t} weight sum {row_sum}"
            );
        }
    }

    #[test]
    fn assignments_in_valid_range() {
        let n_tokens = 9;
        let n_experts = 6;
        let cfg = SinkhornRouteConfig {
            n_experts,
            ..SinkhornRouteConfig::default()
        };
        let logits = make_logits(n_tokens, n_experts, 2.5);
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        assert_eq!(res.expert_assignments.len(), n_tokens);
        assert!(res.expert_assignments.iter().all(|&e| e < n_experts));
    }

    #[test]
    fn deterministic_same_inputs() {
        let n_tokens = 8;
        let n_experts = 4;
        let cfg = SinkhornRouteConfig {
            n_experts,
            ..SinkhornRouteConfig::default()
        };
        let logits = make_logits(n_tokens, n_experts, 2.0);
        let a = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        let b = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        assert_eq!(a.expert_assignments, b.expert_assignments);
        for (x, y) in a.plan.iter().zip(b.plan.iter()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn transport_cost_finite_and_prefers_high_logit() {
        // With a strongly preferred expert per token, the plan should place mass
        // on those entries, giving a positive transport cost.
        let n_tokens = 6;
        let n_experts = 3;
        let cfg = SinkhornRouteConfig {
            n_experts,
            lambda: 2.0,
            n_iter: 60,
            ..SinkhornRouteConfig::default()
        };
        // Balanced preferences: tokens split evenly across experts.
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for t in 0..n_tokens {
            logits[t * n_experts + (t % n_experts)] = 5.0;
        }
        let res = sinkhorn_route(&logits, n_tokens, &cfg).expect("sinkhorn_route should succeed");
        assert!(res.transport_cost.is_finite());
        assert!(
            res.transport_cost > 0.0,
            "expected positive cost, got {}",
            res.transport_cost
        );
    }

    #[test]
    fn marginal_deviation_perfect_plan_zero() {
        // T=2, E=2 balanced plan: each entry = 1/(T*E) = 0.25.
        let plan = [0.25_f32, 0.25, 0.25, 0.25];
        let dev = marginal_deviation(&plan, 2, 2);
        assert!(dev < 1e-6, "deviation {dev}");
    }

    #[test]
    fn default_config_values() {
        let cfg = SinkhornRouteConfig::default();
        assert_eq!(cfg.n_experts, 8);
        assert_eq!(cfg.n_iter, 20);
        assert!((cfg.lambda - 1.0).abs() < 1e-9);
    }
}
