//! Mixtral-style sparse Mixture-of-Experts layer (Jiang et al. 2024).
//!
//! Implements the sparse MoE block from:
//! Jiang et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
//!
//! Each token is routed to its **top-2** experts by a linear gating network. The
//! router computes a softmax over *all* `n_experts` gate logits, keeps the `top_k`
//! highest-probability experts, and **renormalises** those `top_k` probabilities
//! to sum to `1`. Every expert is a SwiGLU feed-forward network (as in
//! Mixtral / LLaMA), and the layer output for a token is the gate-weighted sum of
//! its selected experts:
//!
//! ```text
//! g       = softmax(W_gate · x)                  // over all experts
//! (w, e)  = top_k(g);   w ← w / Σ w              // selected + renormalised
//! y       = Σ_j w_j · SwiGLU_{e_j}(x)
//! ```
//!
//! The layer also returns the Switch-style load-balancing auxiliary loss
//! (generalised to top-k), which encourages tokens to spread evenly across the
//! expert pool.

use crate::error::{MoeError, MoeResult};
use crate::expert::ffn::SwiGluExpert;
use crate::handle::LcgRng;
use crate::moe::topk_balance_loss;
use crate::routing::top_k::{TopKConfig, TopKRouter};

/// Configuration for a [`MixtralMoeLayer`].
#[derive(Debug, Clone)]
pub struct MixtralConfig {
    /// Model (input/output) dimension `d_model`.
    pub d_model: usize,
    /// Hidden dimension of every expert SwiGLU FFN.
    pub ffn_dim: usize,
    /// Number of experts in the pool (`8` in Mixtral-8x7B).
    pub n_experts: usize,
    /// Experts activated per token (`2` in Mixtral); must satisfy `1 ≤ top_k ≤ n_experts`.
    pub top_k: usize,
    /// Coefficient applied to the load-balancing auxiliary loss (`0.02` in Mixtral).
    pub load_balance_coef: f32,
}

impl Default for MixtralConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            ffn_dim: 1024,
            n_experts: 8,
            top_k: 2,
            load_balance_coef: 0.02,
        }
    }
}

/// Routing decisions produced by a Mixtral forward pass.
#[derive(Debug, Clone)]
pub struct MixtralRoutingInfo {
    /// Selected expert indices per token, shape `[n_tokens · top_k]`.
    pub expert_indices: Vec<usize>,
    /// Renormalised combine weights aligned with `expert_indices`,
    /// shape `[n_tokens · top_k]`; each token's `top_k` weights sum to `1`.
    pub combine_weights: Vec<f32>,
    /// Raw gate logits before softmax, shape `[n_tokens · n_experts]`.
    pub router_logits: Vec<f32>,
}

/// Output of a [`MixtralMoeLayer`] forward pass.
#[derive(Debug)]
pub struct MixtralOutput {
    /// Output hidden states, shape `[n_tokens · d_model]`.
    pub hidden: Vec<f32>,
    /// Load-balancing auxiliary loss, already scaled by `load_balance_coef`.
    pub aux_loss: f32,
    /// Routing decisions for inspection / downstream losses.
    pub routing: MixtralRoutingInfo,
}

/// Mixtral-style top-`k` sparse MoE layer with SwiGLU experts.
pub struct MixtralMoeLayer {
    router: TopKRouter,
    experts: Vec<SwiGluExpert>,
    /// Number of experts in the pool.
    pub n_experts: usize,
    /// Experts activated per token (2 for canonical Mixtral).
    pub top_k: usize,
    /// Model (input/output) dimension.
    pub d_model: usize,
    /// Expert FFN hidden dimension.
    pub ffn_dim: usize,
    /// Coefficient applied to the auxiliary load-balancing loss.
    pub load_balance_coef: f32,
}

impl MixtralMoeLayer {
    /// Build a new Mixtral MoE layer with randomly initialised gate and experts.
    ///
    /// # Errors
    /// Returns [`MoeError`] for a zero `d_model` / `ffn_dim` / `n_experts`, or a
    /// `top_k` outside `1 ..= n_experts`.
    pub fn new(cfg: MixtralConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.d_model == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.d_model });
        }
        if cfg.ffn_dim == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: cfg.ffn_dim });
        }
        if cfg.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_experts,
            });
        }
        if cfg.top_k == 0 || cfg.top_k > cfg.n_experts {
            return Err(MoeError::InvalidTopK {
                k: cfg.top_k,
                n_experts: cfg.n_experts,
            });
        }

        let router = TopKRouter::new(
            TopKConfig {
                k: cfg.top_k,
                n_experts: cfg.n_experts,
                input_dim: cfg.d_model,
                noise_std: 0.0,
            },
            rng,
        )?;
        let experts: Vec<SwiGluExpert> = (0..cfg.n_experts)
            .map(|_| SwiGluExpert::new(cfg.d_model, cfg.ffn_dim, rng))
            .collect();

        Ok(Self {
            router,
            experts,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            d_model: cfg.d_model,
            ffn_dim: cfg.ffn_dim,
            load_balance_coef: cfg.load_balance_coef,
        })
    }

    /// Run the full sparse-MoE forward pass.
    ///
    /// # Arguments
    /// * `tokens` — input activations, row-major `[n_tokens · d_model]`.
    /// * `n_tokens` — number of tokens `T`.
    /// * `d_model` — feature dimension, validated against the layer's `d_model`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input, a `d_model` mismatch, or a `tokens`
    /// length that is not `n_tokens · d_model`.
    pub fn forward(
        &self,
        tokens: &[f32],
        n_tokens: usize,
        d_model: usize,
    ) -> MoeResult<MixtralOutput> {
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        if d_model != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: d_model,
            });
        }
        let expected = n_tokens * self.d_model;
        if tokens.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: tokens.len(),
            });
        }

        // Top-k softmax routing (softmax over all experts, then renormalised top-k).
        let routing = self.router.route(tokens, n_tokens)?;
        let k = self.top_k;

        let mut hidden = vec![0.0_f32; n_tokens * self.d_model];
        for tok in 0..n_tokens {
            let x_tok = &tokens[tok * self.d_model..(tok + 1) * self.d_model];
            let out_slice = &mut hidden[tok * self.d_model..(tok + 1) * self.d_model];
            for slot in 0..k {
                let expert_idx = routing.indices[tok * k + slot];
                if expert_idx >= self.n_experts {
                    return Err(MoeError::ExpertIndexOutOfRange {
                        idx: expert_idx,
                        n_experts: self.n_experts,
                    });
                }
                let weight = routing.scores[tok * k + slot];
                let expert_out = self.experts[expert_idx].forward(x_tok)?;
                for (acc, &val) in out_slice.iter_mut().zip(expert_out.iter()) {
                    *acc += weight * val;
                }
            }
        }

        let raw_aux = topk_balance_loss(
            &routing.router_logits,
            &routing.indices,
            n_tokens,
            self.n_experts,
        )?;

        Ok(MixtralOutput {
            hidden,
            aux_loss: self.load_balance_coef * raw_aux,
            routing: MixtralRoutingInfo {
                expert_indices: routing.indices,
                combine_weights: routing.scores,
                router_logits: routing.router_logits,
            },
        })
    }

    /// Total trainable parameter count (gate + all expert FFNs).
    #[must_use]
    pub fn param_count(&self) -> usize {
        let gate = self.router.param_count();
        // SwiGLU: w1, w3 are [ffn·d], w2 is [d·ffn] -> 3·d·ffn per expert.
        let per_expert = 3 * self.d_model * self.ffn_dim;
        gate + self.n_experts * per_expert
    }
}

/// Standalone Switch-style load-balancing loss for Mixtral top-k routing.
///
/// `L = n_experts · Σ_i f_i · P_i` (see `crate::moe::topk_balance_loss`); it is
/// `≈ 1` for balanced routing and approaches `n_experts` when routing collapses
/// onto a single expert.
///
/// # Arguments
/// * `router_logits` — raw gate logits, shape `[n_tokens · n_experts]`.
/// * `expert_indices` — selected expert indices, shape `[n_tokens · top_k]`.
///
/// # Errors
/// Propagates the errors of `crate::moe::topk_balance_loss`.
pub fn mixtral_load_balance_loss(
    router_logits: &[f32],
    expert_indices: &[usize],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    topk_balance_loss(router_logits, expert_indices, n_tokens, n_experts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn small_layer(n_experts: usize, top_k: usize) -> MixtralMoeLayer {
        let mut rng = LcgRng::new(123);
        let cfg = MixtralConfig {
            d_model: 8,
            ffn_dim: 16,
            n_experts,
            top_k,
            load_balance_coef: 0.02,
        };
        MixtralMoeLayer::new(cfg, &mut rng).expect("new should succeed")
    }

    /// (a) Each token is routed to exactly top_k=2 experts whose combine weights sum to 1.
    #[test]
    fn top2_combine_weights_sum_to_one() {
        let layer = small_layer(4, 2);
        let n_tokens = 6;
        let x: Vec<f32> = (0..n_tokens * 8).map(|i| (i as f32 * 0.13).sin()).collect();
        let out = layer
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        assert_eq!(out.routing.expert_indices.len(), n_tokens * 2);
        assert_eq!(out.routing.combine_weights.len(), n_tokens * 2);
        for tok in 0..n_tokens {
            let w = &out.routing.combine_weights[tok * 2..tok * 2 + 2];
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "token {tok} weight sum {sum}");
            // The two chosen experts must be distinct.
            let e = &out.routing.expert_indices[tok * 2..tok * 2 + 2];
            assert_ne!(e[0], e[1], "token {tok} selected the same expert twice");
            assert!(e[0] < 4 && e[1] < 4);
        }
    }

    /// (b) Output shape is [T · d_model] and every value is finite.
    #[test]
    fn output_shape_and_finite() {
        let layer = small_layer(4, 2);
        let n_tokens = 5;
        let x = vec![0.4_f32; n_tokens * 8];
        let out = layer
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        assert_eq!(out.hidden.len(), n_tokens * 8);
        assert!(out.hidden.iter().all(|v| v.is_finite()));
        assert!(out.aux_loss.is_finite() && out.aux_loss >= 0.0);
    }

    /// (c) Aux loss is >= 0, small (≈1·coef) for uniform routing, larger when collapsed.
    #[test]
    fn aux_loss_minimised_by_uniform_routing() {
        let n_tokens = 8;
        let n_experts = 4;
        let top_k = 1;
        // Uniform: equal logits -> P_i = 1/4; round-robin selection -> f_i = 1/4.
        let uniform_logits = vec![0.0_f32; n_tokens * n_experts];
        let uniform_sel: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let uniform = mixtral_load_balance_loss(&uniform_logits, &uniform_sel, n_tokens, n_experts)
            .expect("mixtral_load_balance_loss should succeed");

        // Collapsed: all tokens to expert 0 with strong bias.
        let mut collapsed_logits = vec![0.0_f32; n_tokens * n_experts];
        for tok in 0..n_tokens {
            collapsed_logits[tok * n_experts] = 20.0;
        }
        let collapsed_sel = vec![0_usize; n_tokens];
        let collapsed =
            mixtral_load_balance_loss(&collapsed_logits, &collapsed_sel, n_tokens, n_experts)
                .expect("value should be present");

        assert!(uniform >= 0.0, "uniform {uniform} negative");
        assert!((uniform - 1.0).abs() < 1e-4, "uniform {uniform} not ≈ 1");
        assert!(
            collapsed > uniform,
            "collapsed {collapsed} should exceed uniform {uniform}"
        );
        let _ = top_k;
    }

    /// (d) With a single expert the layer degenerates to that expert exactly.
    #[test]
    fn single_expert_degenerates() {
        let mut rng = LcgRng::new(7);
        let cfg = MixtralConfig {
            d_model: 8,
            ffn_dim: 16,
            n_experts: 1,
            top_k: 1,
            load_balance_coef: 0.02,
        };
        let layer = MixtralMoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let n_tokens = 3;
        let x: Vec<f32> = (0..n_tokens * 8).map(|i| (i as f32 * 0.07).cos()).collect();
        let out = layer
            .forward(&x, n_tokens, 8)
            .expect("forward should succeed");
        // softmax over a single logit is 1.0, so combine weight is exactly 1.
        for &w in &out.routing.combine_weights {
            assert!((w - 1.0).abs() < 1e-6, "weight {w} != 1");
        }
        // Output must equal the lone expert applied to each token.
        for tok in 0..n_tokens {
            let x_tok = &x[tok * 8..(tok + 1) * 8];
            let direct = layer.experts[0]
                .forward(x_tok)
                .expect("forward should succeed");
            let got = &out.hidden[tok * 8..(tok + 1) * 8];
            for (g, d) in got.iter().zip(direct.iter()) {
                assert!((g - d).abs() < 1e-5, "got {g} != expert {d}");
            }
        }
    }

    /// (e) d_model / shape mismatches return errors.
    #[test]
    fn shape_mismatch_errors() {
        let layer = small_layer(4, 2);
        // Wrong d_model.
        assert!(matches!(
            layer.forward(&[0.0_f32; 5 * 8], 5, 7),
            Err(MoeError::DimensionMismatch { .. })
        ));
        // Wrong token buffer length.
        assert!(matches!(
            layer.forward(&[0.0_f32; 5 * 8 + 1], 5, 8),
            Err(MoeError::DimensionMismatch { .. })
        ));
        // Empty input.
        assert!(matches!(
            layer.forward(&[], 0, 8),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_config_errors() {
        let mut rng = LcgRng::new(1);
        let base = MixtralConfig {
            d_model: 8,
            ffn_dim: 16,
            n_experts: 4,
            top_k: 2,
            load_balance_coef: 0.02,
        };
        let bad_k = MixtralConfig {
            top_k: 5,
            ..base.clone()
        };
        assert!(matches!(
            MixtralMoeLayer::new(bad_k, &mut rng),
            Err(MoeError::InvalidTopK { .. })
        ));
        let bad_experts = MixtralConfig {
            n_experts: 0,
            ..base
        };
        assert!(matches!(
            MixtralMoeLayer::new(bad_experts, &mut rng),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn param_count_positive() {
        let layer = small_layer(4, 2);
        assert!(layer.param_count() > 0);
    }
}
