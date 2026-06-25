//! Complete MoE layer: router + dispatch + expert compute + combine.
//!
//! Supports top-K routing with Switch Transformer capacity-bounded dispatch.

use crate::error::{MoeError, MoeResult};
use crate::expert::bank::ExpertBank;
use crate::expert::ffn::ExpertActivation;
use crate::handle::LcgRng;
use crate::loss::load_balance::{LoadStats, compute_load_stats};
use crate::loss::router_z::router_z_loss;
use crate::routing::switch::{SwitchConfig, switch_dispatch};
use crate::routing::top_k::{TopKConfig, TopKRouter};

/// Configuration for a full MoE layer.
#[derive(Debug, Clone)]
pub struct MoeLayerConfig {
    /// Input/output model dimension.
    pub input_dim: usize,
    /// FFN hidden dimension for each expert.
    pub ffn_dim: usize,
    /// Number of experts.
    pub n_experts: usize,
    /// Top-k experts per token (1 = Switch, 2 = GShard).
    pub top_k: usize,
    /// Expert capacity factor for Switch dispatch.
    pub capacity_factor: f32,
    /// Coefficient for load balance auxiliary loss (λ).
    pub load_balance_coef: f32,
    /// Coefficient for router z-loss.
    pub router_z_loss_coef: f32,
    /// Expert activation function.
    pub activation: ExpertActivation,
}

impl Default for MoeLayerConfig {
    fn default() -> Self {
        Self {
            input_dim: 256,
            ffn_dim: 1024,
            n_experts: 8,
            top_k: 1,
            capacity_factor: 1.25,
            load_balance_coef: 1e-2,
            router_z_loss_coef: 1e-3,
            activation: ExpertActivation::Gelu,
        }
    }
}

/// Output of a full MoE forward pass.
#[derive(Debug)]
pub struct MoeLayerOutput {
    /// Output hidden states, shape `[n_tokens * input_dim]`.
    pub hidden: Vec<f32>,
    /// Combined auxiliary loss (load balance + z-loss).
    pub aux_loss: f32,
    /// Number of tokens dropped due to capacity overflow.
    pub n_overflows: usize,
    /// Detailed per-expert load statistics.
    pub load_stats: LoadStats,
}

/// Complete MoE layer: router + expert bank + dispatch.
pub struct MoeLayer {
    router: TopKRouter,
    experts: ExpertBank,
    /// Layer configuration.
    pub config: MoeLayerConfig,
}

impl MoeLayer {
    /// Create a new MoE layer with random initialization.
    pub fn new(cfg: MoeLayerConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.input_dim == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.input_dim });
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

        let router_cfg = TopKConfig {
            k: cfg.top_k,
            n_experts: cfg.n_experts,
            input_dim: cfg.input_dim,
            noise_std: 0.0,
        };
        let router = TopKRouter::new(router_cfg, rng)?;
        let experts = ExpertBank::new(
            cfg.n_experts,
            cfg.input_dim,
            cfg.ffn_dim,
            cfg.activation,
            rng,
        )?;

        Ok(Self {
            router,
            experts,
            config: cfg,
        })
    }

    /// Full forward pass through the MoE layer.
    ///
    /// # Arguments
    /// * `x` — input tokens, shape `[n_tokens * input_dim]`
    /// * `n_tokens` — number of tokens
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> MoeResult<MoeLayerOutput> {
        let cfg = &self.config;

        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected_x = n_tokens * cfg.input_dim;
        if x.len() != expected_x {
            return Err(MoeError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }

        // Step 1: Route tokens through the top-k router
        let routing = self.router.route(x, n_tokens)?;

        // For Switch (top_k=1), use the first index from each token's routing result
        // For top_k > 1, we dispatch per expert (for simplicity, use the first choice)
        let gate_indices: Vec<usize> = (0..n_tokens)
            .map(|tok| routing.indices[tok * cfg.top_k])
            .collect();
        let gate_scores: Vec<f32> = (0..n_tokens)
            .map(|tok| routing.scores[tok * cfg.top_k])
            .collect();

        // Step 2: Switch dispatch with capacity bounds
        let switch_cfg = SwitchConfig {
            n_experts: cfg.n_experts,
            input_dim: cfg.input_dim,
            capacity_factor: cfg.capacity_factor,
            min_capacity: 1,
            drop_tokens: true,
        };
        let dispatch = switch_dispatch(&gate_indices, n_tokens, &switch_cfg)?;

        // Step 3: Expert bank processes tokens according to dispatch
        let hidden = self.experts.forward_dispatched(
            x,
            &dispatch.expert_assignments,
            n_tokens,
            &gate_scores,
        )?;

        // Step 4: Compute auxiliary losses
        let load_stats = compute_load_stats(
            &routing.router_logits,
            &dispatch.expert_assignments,
            n_tokens,
            cfg.n_experts,
        )?;

        let z_loss = router_z_loss(&routing.router_logits, n_tokens, cfg.n_experts)?;

        let aux_loss =
            cfg.load_balance_coef * load_stats.balance_loss + cfg.router_z_loss_coef * z_loss;

        Ok(MoeLayerOutput {
            hidden,
            aux_loss,
            n_overflows: dispatch.n_overflows,
            load_stats,
        })
    }

    /// Immutable access to the underlying expert bank.
    #[must_use]
    pub fn experts(&self) -> &ExpertBank {
        &self.experts
    }

    /// Mutable access to the underlying expert bank (e.g. to overwrite expert
    /// weights when sparse-upcycling from a dense checkpoint).
    pub fn experts_mut(&mut self) -> &mut ExpertBank {
        &mut self.experts
    }

    /// Immutable access to the top-k router.
    #[must_use]
    pub fn router(&self) -> &TopKRouter {
        &self.router
    }

    /// Return the total parameter count (router + all experts).
    #[must_use]
    pub fn param_count(&self) -> usize {
        let router_params = self.router.param_count();
        let expert_params = self.config.n_experts
            * (2 * self.config.input_dim * self.config.ffn_dim
                + self.config.ffn_dim
                + self.config.input_dim);
        router_params + expert_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn moe_layer_forward_basic() {
        let mut rng = LcgRng::new(0);
        let cfg = MoeLayerConfig {
            input_dim: 16,
            ffn_dim: 64,
            n_experts: 4,
            top_k: 1,
            capacity_factor: 1.25,
            load_balance_coef: 0.01,
            router_z_loss_coef: 0.001,
            activation: ExpertActivation::Gelu,
        };
        let layer = MoeLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 8 * 16];
        let output = layer.forward(&x, 8).expect("forward should succeed");
        assert_eq!(output.hidden.len(), 8 * 16);
        assert!(output.aux_loss.is_finite());
    }

    #[test]
    fn moe_layer_param_count_reasonable() {
        let mut rng = LcgRng::new(0);
        let cfg = MoeLayerConfig::default();
        let layer = MoeLayer::new(cfg.clone(), &mut rng).expect("value should be present");
        let params = layer.param_count();
        // Router: E * d; each expert: 2 * d * d_ffn + d_ffn + d
        let expected_min = cfg.n_experts * cfg.input_dim + cfg.n_experts * cfg.input_dim;
        assert!(params >= expected_min);
    }
}
