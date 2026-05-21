//! Stable MoE routing: sigmoid gating, z-loss, and load-balance auxiliary loss.
//!
//! Combines ideas from:
//! - Zuo et al. 2022: "Taming Sparsely Activated Transformer with Stochastic Experts"
//! - Dai et al. 2022: "StableMoE: Stable Routing Strategy for Mixture of Experts"
//!
//! Core contributions over vanilla Top-K routing:
//!  1. **Sigmoid gating**: each expert gate is `sigmoid(logit_e)`, then normalised
//!     over experts.  This avoids the cross-expert competition of softmax and
//!     produces more stable gradients.
//!  2. **Z-loss**: `L_z = (1/T) Σ_t log²(logsumexp(logits_t))` penalises large
//!     logit magnitudes, preventing routing collapse.
//!  3. **Expert dropout**: randomly masks expert contributions during training.
//!  4. **Load-balance auxiliary loss**: identical to Switch Transformer.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Gating function choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableMoeGating {
    /// Sigmoid gates: `g_e = sigmoid(logit_e)`, then normalised so Σ_e g_e = 1.
    Sigmoid,
    /// Standard softmax gates (baseline comparison).
    Softmax,
}

/// Configuration for Stable MoE routing.
#[derive(Debug, Clone)]
pub struct StableMoeConfig {
    /// Number of experts E.
    pub n_experts: usize,
    /// Number of experts to route to per token (top-K; default 1).
    pub top_k: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Z-loss coefficient (default 0.001; penalises large logits).
    pub z_loss_coef: f32,
    /// Load-balance auxiliary loss coefficient (default 0.01).
    pub load_balance_coef: f32,
    /// Expert dropout rate ∈ [0, 1): fraction of experts to randomly zero
    /// per token during training (default 0.0 = no dropout).
    pub expert_dropout: f32,
    /// Gating function.
    pub gating: StableMoeGating,
}

impl Default for StableMoeConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            top_k: 1,
            input_dim: 256,
            z_loss_coef: 0.001,
            load_balance_coef: 0.01,
            expert_dropout: 0.0,
            gating: StableMoeGating::Sigmoid,
        }
    }
}

/// Result of Stable MoE routing.
#[derive(Debug, Clone)]
pub struct StableMoeResult {
    /// Top-K expert assignments per token: `[n_tokens × top_k]`.
    pub expert_assignments: Vec<usize>,
    /// Routing scores for top-K experts: `[n_tokens × top_k]`.
    pub routing_scores: Vec<f32>,
    /// Z-loss value for this batch (scalar, unweighted by `z_loss_coef`).
    pub z_loss: f32,
    /// Load-balance auxiliary loss (unweighted by `load_balance_coef`).
    pub load_balance_loss: f32,
    /// Total auxiliary loss = `z_loss_coef * z_loss + load_balance_coef * load_balance_loss`.
    pub aux_loss: f32,
}

/// Stable MoE router.
pub struct StableMoeRouter {
    /// Routing configuration.
    pub config: StableMoeConfig,
}

impl StableMoeRouter {
    /// Create a new Stable MoE router.
    ///
    /// # Errors
    /// - `n_experts == 0`
    /// - `top_k > n_experts`
    /// - `expert_dropout >= 1.0`
    pub fn new(config: StableMoeConfig) -> MoeResult<Self> {
        if config.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: config.n_experts,
            });
        }
        if config.top_k > config.n_experts {
            return Err(MoeError::InvalidTopK {
                k: config.top_k,
                n_experts: config.n_experts,
            });
        }
        if config.expert_dropout >= 1.0 {
            return Err(MoeError::Internal {
                msg: format!(
                    "expert_dropout must be < 1.0, got {}",
                    config.expert_dropout
                ),
            });
        }
        Ok(Self { config })
    }

    /// Compute gating weights for a **single** token's logit vector.
    ///
    /// Returns gate weights `[n_experts]` that sum to 1.
    #[must_use]
    pub fn gate(&self, logits: &[f32]) -> Vec<f32> {
        let n_e = self.config.n_experts;
        let eps = 1e-7_f32;
        match self.config.gating {
            StableMoeGating::Sigmoid => {
                let mut g: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
                let total: f32 = g.iter().sum();
                if total < eps {
                    // Degenerate: return uniform.
                    let uniform = 1.0 / n_e as f32;
                    g.iter_mut().for_each(|v| *v = uniform);
                } else {
                    g.iter_mut().for_each(|v| *v /= total);
                }
                g
            }
            StableMoeGating::Softmax => {
                // Numerically stable max-shift softmax.
                let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exps.iter().sum::<f32>() + eps;
                exps.iter_mut().for_each(|v| *v /= sum);
                exps
            }
        }
    }

    /// Route a batch of tokens.
    ///
    /// # Arguments
    /// - `logits`: router logits `[n_tokens × n_experts]`, row-major.
    /// - `n_tokens`: T.
    /// - `rng`: random state for expert dropout (only used when `expert_dropout > 0`).
    ///
    /// # Returns
    /// `StableMoeResult` with assignments, scores, and auxiliary losses.
    pub fn route(
        &self,
        logits: &[f32],
        n_tokens: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<StableMoeResult> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected = n_tokens * cfg.n_experts;
        if logits.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: logits.len(),
            });
        }

        let n_e = cfg.n_experts;
        let k = cfg.top_k;

        let mut expert_assignments = vec![0_usize; n_tokens * k];
        let mut routing_scores = vec![0.0_f32; n_tokens * k];
        // Pre-dropout gating probs, shape [n_tokens × n_experts], for load balance.
        let mut gating_probs = vec![0.0_f32; n_tokens * n_e];

        for t in 0..n_tokens {
            let raw = &logits[t * n_e..(t + 1) * n_e];
            let g_pre = self.gate(raw);

            // Store pre-dropout probabilities for load-balance loss.
            gating_probs[t * n_e..(t + 1) * n_e].copy_from_slice(&g_pre);

            // Apply expert dropout.
            let mut g = g_pre;
            if cfg.expert_dropout > 0.0 {
                let mut any_nonzero = false;
                for gv in g.iter_mut() {
                    if rng.next_f32() < cfg.expert_dropout {
                        *gv = 0.0;
                    } else {
                        any_nonzero = true;
                    }
                }
                if !any_nonzero {
                    // All dropped — restore uniform to avoid all-zero gate.
                    let uniform = 1.0 / n_e as f32;
                    g.iter_mut().for_each(|v| *v = uniform);
                } else {
                    // Re-normalise survivors.
                    let total: f32 = g.iter().sum();
                    if total > 1e-12 {
                        g.iter_mut().for_each(|v| *v /= total);
                    }
                }
            }

            // Top-K selection: simple O(K·E) for small K.
            let tok_assignments = &mut expert_assignments[t * k..(t + 1) * k];
            let tok_scores = &mut routing_scores[t * k..(t + 1) * k];

            // We build a sorted list of (score, expert_index) for top-K.
            for slot in 0..k {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0_usize;
                'inner: for (e, &gval) in g.iter().enumerate() {
                    // Skip already-selected experts.
                    let already_chosen = (0..slot).any(|prev| tok_assignments[prev] == e);
                    if already_chosen {
                        continue 'inner;
                    }
                    if gval > best_val {
                        best_val = gval;
                        best_idx = e;
                    }
                }
                tok_assignments[slot] = best_idx;
                tok_scores[slot] = best_val;
            }
        }

        // Compute z-loss from raw logits (no dropout involved).
        let z = z_loss(logits, n_tokens, n_e)?;

        // Primary assignments (first top-k slot) for load balance.
        let primary_assignments: Vec<usize> =
            (0..n_tokens).map(|t| expert_assignments[t * k]).collect();
        let lb = load_balance_loss(&primary_assignments, &gating_probs, n_tokens, n_e)?;

        let aux = cfg.z_loss_coef * z + cfg.load_balance_coef * lb;

        Ok(StableMoeResult {
            expert_assignments,
            routing_scores,
            z_loss: z,
            load_balance_loss: lb,
            aux_loss: aux,
        })
    }
}

/// Compute z-loss: `L_z = (1/T) Σ_t log²(Σ_e exp(logit_{t,e}))`.
///
/// Uses numerically stable logsumexp:
/// `log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))`.
pub fn z_loss(logits: &[f32], n_tokens: usize, n_experts: usize) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    let expected = n_tokens * n_experts;
    if logits.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: logits.len(),
        });
    }

    let mut total = 0.0_f32;
    for t in 0..n_tokens {
        let row = &logits[t * n_experts..(t + 1) * n_experts];
        let lse = logsumexp(row);
        total += lse * lse;
    }

    Ok(total / n_tokens as f32)
}

/// Compute load-balance auxiliary loss (same formula as Switch Transformer):
/// `L_lb = n_experts * Σ_e f_e * P_e`
/// where `f_e` = fraction of tokens routed to expert e (from hard assignments)
/// and `P_e` = mean gating probability for expert e.
pub fn load_balance_loss(
    expert_assignments: &[usize],
    gating_probs: &[f32],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if expert_assignments.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: expert_assignments.len(),
        });
    }
    let expected_probs = n_tokens * n_experts;
    if gating_probs.len() != expected_probs {
        return Err(MoeError::DimensionMismatch {
            expected: expected_probs,
            got: gating_probs.len(),
        });
    }

    // f_e: fraction of tokens routed to expert e.
    let mut counts = vec![0_usize; n_experts];
    for &a in expert_assignments {
        if a < n_experts {
            counts[a] += 1;
        }
    }
    let t_inv = 1.0 / n_tokens as f32;
    let f_e: Vec<f32> = counts.iter().map(|&c| c as f32 * t_inv).collect();

    // P_e: mean gating probability for expert e.
    let mut p_e = vec![0.0_f32; n_experts];
    for t in 0..n_tokens {
        for e in 0..n_experts {
            p_e[e] += gating_probs[t * n_experts + e];
        }
    }
    for pe in p_e.iter_mut() {
        *pe *= t_inv;
    }

    // L_lb = n_experts * Σ_e f_e * P_e.
    let dot: f32 = f_e.iter().zip(p_e.iter()).map(|(f, p)| f * p).sum();
    Ok(n_experts as f32 * dot)
}

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
///
/// Avoids overflow for large `|x|` using the identity
/// `sigmoid(x) = 1 / (1 + exp(-x))` for x ≥ 0,
/// `sigmoid(x) = exp(x) / (1 + exp(x))` for x < 0.
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Numerically stable `logsumexp` over a slice.
#[inline]
fn logsumexp(v: &[f32]) -> f32 {
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum_exp: f32 = v.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // -----------------------------------------------------------------------
    // sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_at_zero_is_half() {
        let v = sigmoid(0.0);
        assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) = {v}");
    }

    #[test]
    fn sigmoid_large_positive_near_one() {
        let v = sigmoid(100.0);
        assert!(v > 0.999, "sigmoid(100) = {v}");
    }

    #[test]
    fn sigmoid_large_negative_near_zero() {
        let v = sigmoid(-100.0);
        assert!(v < 0.001, "sigmoid(-100) = {v}");
    }

    #[test]
    fn sigmoid_is_monotone_and_bounded() {
        let vals = [-10.0_f32, -1.0, 0.0, 1.0, 10.0];
        let sigs: Vec<f32> = vals.iter().map(|&x| sigmoid(x)).collect();
        for &s in &sigs {
            assert!((0.0..=1.0).contains(&s), "sigmoid out of [0,1]: {s}");
        }
        for w in sigs.windows(2) {
            assert!(w[0] <= w[1], "not monotone: {}", w[0]);
        }
    }

    // -----------------------------------------------------------------------
    // z_loss
    // -----------------------------------------------------------------------

    #[test]
    fn z_loss_single_token_two_experts_uniform() {
        // logits=[0,0], LSE = log(e^0 + e^0) = log(2)
        let logits = [0.0_f32, 0.0];
        let zl = z_loss(&logits, 1, 2).unwrap();
        let expected = 2.0_f32.ln().powi(2);
        assert!(
            (zl - expected).abs() < 1e-5,
            "z_loss={zl}, expected={expected}"
        );
    }

    #[test]
    fn z_loss_large_logits_numerically_stable() {
        // logits=[100, 0], LSE ≈ 100 (dominated by first entry)
        let logits = [100.0_f32, 0.0];
        let zl = z_loss(&logits, 1, 2).unwrap();
        // LSE ≈ 100; z_loss ≈ 100² = 10000
        assert!(zl > 9000.0 && zl < 10100.0, "z_loss={zl}");
    }

    #[test]
    fn z_loss_non_negative() {
        let mut rng = LcgRng::new(77);
        let n_tokens = 16;
        let n_experts = 8;
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        rng.fill_normal_scaled(&mut logits, 2.0);
        let zl = z_loss(&logits, n_tokens, n_experts).unwrap();
        assert!(zl >= 0.0, "z_loss must be >= 0, got {zl}");
        assert!(zl.is_finite(), "z_loss must be finite, got {zl}");
    }

    #[test]
    fn z_loss_empty_tokens_returns_err() {
        assert!(z_loss(&[], 0, 4).is_err());
    }

    // -----------------------------------------------------------------------
    // load_balance_loss
    // -----------------------------------------------------------------------

    #[test]
    fn load_balance_loss_all_same_expert() {
        // All 4 tokens → expert 0; uniform gating probs (1/2 each).
        // f_0 = 1, f_1 = 0
        // P_0 = 0.5, P_1 = 0.5
        // L_lb = 2 * (1*0.5 + 0*0.5) = 2 * 0.5 = 1.0
        let assignments = vec![0_usize; 4];
        let gating_probs = vec![0.5_f32; 4 * 2]; // [4 tokens × 2 experts], all 0.5
        let lb = load_balance_loss(&assignments, &gating_probs, 4, 2).unwrap();
        assert!((lb - 1.0).abs() < 1e-5, "load_balance_loss={lb}");
    }

    #[test]
    fn load_balance_loss_balanced_lower_than_imbalanced() {
        // The load-balance loss is n_e * Σ_e f_e * P_e.
        // To make the comparison meaningful, use gating probs that reflect
        // the actual assignment: concentrated gating probs make the imbalanced
        // case have a higher loss.
        let n_tokens = 8_usize;
        let n_experts = 4_usize;

        // Balanced: round-robin assignment.
        // Gating probs: each token spreads weight evenly → P_e = 1/4.
        // f_e = 1/4 for all e → L_lb = 4 * (4 * 1/4 * 1/4) = 1.
        let balanced_assignments: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let uniform_probs = vec![1.0_f32 / n_experts as f32; n_tokens * n_experts];
        let lb_balanced =
            load_balance_loss(&balanced_assignments, &uniform_probs, n_tokens, n_experts).unwrap();

        // Imbalanced: all tokens to expert 0.
        // Gating probs: all weight on expert 0 → P_0 = 1.0, P_1..3 = 0.
        // f_0 = 1, f_1..3 = 0 → L_lb = 4 * (1 * 1 + 0 + 0 + 0) = 4.
        let imbalanced_assignments = vec![0_usize; n_tokens];
        let concentrated_probs: Vec<f32> = (0..n_tokens * n_experts)
            .map(|i| if i % n_experts == 0 { 1.0_f32 } else { 0.0 })
            .collect();
        let lb_imbalanced = load_balance_loss(
            &imbalanced_assignments,
            &concentrated_probs,
            n_tokens,
            n_experts,
        )
        .unwrap();

        assert!(
            lb_balanced < lb_imbalanced,
            "balanced={lb_balanced} should be < imbalanced={lb_imbalanced}"
        );
    }

    #[test]
    fn load_balance_loss_empty_tokens_returns_err() {
        assert!(load_balance_loss(&[], &[], 0, 4).is_err());
    }

    // -----------------------------------------------------------------------
    // StableMoeRouter::new
    // -----------------------------------------------------------------------

    #[test]
    fn new_with_zero_experts_returns_err() {
        let cfg = StableMoeConfig {
            n_experts: 0,
            ..StableMoeConfig::default()
        };
        assert!(StableMoeRouter::new(cfg).is_err());
    }

    #[test]
    fn new_with_top_k_greater_than_experts_returns_err() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            top_k: 8,
            ..StableMoeConfig::default()
        };
        assert!(StableMoeRouter::new(cfg).is_err());
    }

    #[test]
    fn new_with_expert_dropout_one_returns_err() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            top_k: 1,
            expert_dropout: 1.0,
            ..StableMoeConfig::default()
        };
        assert!(StableMoeRouter::new(cfg).is_err());
    }

    #[test]
    fn new_with_valid_config_succeeds() {
        let cfg = StableMoeConfig::default();
        assert!(StableMoeRouter::new(cfg).is_ok());
    }

    // -----------------------------------------------------------------------
    // StableMoeRouter::gate
    // -----------------------------------------------------------------------

    #[test]
    fn gate_sigmoid_sums_to_one() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            gating: StableMoeGating::Sigmoid,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let logits = [0.5_f32, -0.5, 1.0, -1.0];
        let g = router.gate(&logits);
        let total: f32 = g.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "sum={total}");
    }

    #[test]
    fn gate_softmax_sums_to_one() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            gating: StableMoeGating::Softmax,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let logits = [0.5_f32, -0.5, 1.0, -1.0];
        let g = router.gate(&logits);
        let total: f32 = g.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "sum={total}");
    }

    #[test]
    fn gate_sigmoid_non_negative() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            gating: StableMoeGating::Sigmoid,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let logits = [10.0_f32, -10.0, 0.0, 3.0];
        for &v in router.gate(&logits).iter() {
            assert!(v >= 0.0, "negative gate value: {v}");
        }
    }

    #[test]
    fn gate_sigmoid_uniform_logits_gives_uniform_output() {
        // sigmoid([1,1,1,1]) = [0.731...]*4; after normalisation → [0.25]*4
        let cfg = StableMoeConfig {
            n_experts: 4,
            gating: StableMoeGating::Sigmoid,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let logits = [1.0_f32; 4];
        let g = router.gate(&logits);
        for &v in &g {
            assert!((v - 0.25).abs() < 1e-5, "expected 0.25, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // StableMoeRouter::route
    // -----------------------------------------------------------------------

    #[test]
    fn route_wrong_logits_length_returns_err() {
        let cfg = StableMoeConfig::default();
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(1);
        let logits = vec![0.0_f32; 5]; // wrong
        assert!(router.route(&logits, 3, &mut rng).is_err());
    }

    #[test]
    fn route_zero_tokens_returns_err() {
        let cfg = StableMoeConfig::default();
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(2);
        assert!(router.route(&[], 0, &mut rng).is_err());
    }

    #[test]
    fn route_expert_assignments_length_equals_n_tokens_times_k() {
        let n_tokens = 6_usize;
        let k = 2_usize;
        let n_e = 4_usize;
        let cfg = StableMoeConfig {
            n_experts: n_e,
            top_k: k,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(3);
        let logits = vec![0.1_f32; n_tokens * n_e];
        let result = router.route(&logits, n_tokens, &mut rng).unwrap();
        assert_eq!(result.expert_assignments.len(), n_tokens * k);
    }

    #[test]
    fn route_top_k_scores_are_largest_gates() {
        // With k=1 the score should be the max gate value.
        let n_tokens = 4_usize;
        let n_e = 4_usize;
        let cfg = StableMoeConfig {
            n_experts: n_e,
            top_k: 1,
            expert_dropout: 0.0,
            gating: StableMoeGating::Softmax,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(4);
        let logits: Vec<f32> = (0..n_tokens * n_e).map(|i| (i as f32) * 0.3).collect();
        let result = router.route(&logits, n_tokens, &mut rng).unwrap();
        for t in 0..n_tokens {
            let raw_gates = router.gate(&logits[t * n_e..(t + 1) * n_e]);
            let max_gate = raw_gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let chosen_score = result.routing_scores[t];
            assert!(
                (chosen_score - max_gate).abs() < 1e-5,
                "token {t}: chosen={chosen_score}, max={max_gate}"
            );
        }
    }

    #[test]
    fn route_aux_loss_finite_and_non_negative() {
        let cfg = StableMoeConfig::default();
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(5);
        let n_tokens = 8;
        let n_e = 8;
        let logits = vec![0.5_f32; n_tokens * n_e];
        let result = router.route(&logits, n_tokens, &mut rng).unwrap();
        assert!(result.aux_loss.is_finite(), "aux_loss not finite");
        assert!(result.aux_loss >= 0.0, "aux_loss < 0: {}", result.aux_loss);
    }

    #[test]
    fn route_no_dropout_all_tokens_get_expert() {
        let cfg = StableMoeConfig {
            n_experts: 4,
            top_k: 1,
            expert_dropout: 0.0,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(6);
        let n_tokens = 16;
        let logits = vec![1.0_f32; n_tokens * 4];
        let result = router.route(&logits, n_tokens, &mut rng).unwrap();
        // Every assignment must be a valid expert index.
        for &a in &result.expert_assignments {
            assert!(a < 4, "invalid assignment: {a}");
        }
    }

    #[test]
    fn route_expert_dropout_zero_point_nine_still_valid() {
        // dropout=0.9: most experts are dropped per token, but we should still
        // have valid non-NaN routing scores and at least one expert per token.
        let cfg = StableMoeConfig {
            n_experts: 8,
            top_k: 1,
            expert_dropout: 0.9,
            ..StableMoeConfig::default()
        };
        let router = StableMoeRouter::new(cfg).unwrap();
        let mut rng = LcgRng::new(7);
        let n_tokens = 32;
        let logits = vec![0.5_f32; n_tokens * 8];
        let result = router.route(&logits, n_tokens, &mut rng).unwrap();
        for &a in &result.expert_assignments {
            assert!(a < 8, "invalid assignment: {a}");
        }
        assert!(result.aux_loss.is_finite());
    }
}
