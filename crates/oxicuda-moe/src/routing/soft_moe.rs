//! Soft MoE: fully differentiable routing via slot averaging.
//!
//! Implements the routing from:
//! Puigcerver et al. "From Sparse to Soft Mixtures of Experts." ICLR 2024.
//!
//! Each expert has `n_slots`; slots aggregate tokens, then experts process slots.
//! `D = softmax(X · Φ / sqrt(d))` where Φ is `[d, E*S]` learned slot parameters.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::routing::top_k::stable_softmax;

/// Configuration for Soft MoE routing.
#[derive(Debug, Clone)]
pub struct SoftMoeConfig {
    /// Number of experts.
    pub n_experts: usize,
    /// Number of slots per expert (S).
    pub n_slots_per_expert: usize,
    /// Input feature dimension.
    pub input_dim: usize,
}

/// Soft MoE router: holds learned slot parameters Φ.
#[derive(Debug, Clone)]
pub struct SoftMoeRouter {
    /// Slot parameters, shape `[input_dim * (n_experts * n_slots_per_expert)]`.
    pub phi: Vec<f32>,
    /// Routing configuration.
    pub config: SoftMoeConfig,
}

impl SoftMoeRouter {
    /// Create a new soft MoE router with random initialization.
    pub fn new(cfg: SoftMoeConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_experts,
            });
        }
        if cfg.n_slots_per_expert == 0 {
            return Err(MoeError::Internal {
                msg: "n_slots_per_expert must be >= 1".to_string(),
            });
        }
        if cfg.input_dim == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.input_dim });
        }
        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        let phi_len = cfg.input_dim * n_slots;
        let std_dev = 1.0 / (cfg.input_dim as f32).sqrt();
        let mut phi = vec![0.0_f32; phi_len];
        rng.fill_normal_scaled(&mut phi, std_dev);
        Ok(Self { phi, config: cfg })
    }

    /// Compute dispatch weights `D` of shape `[n_tokens, n_experts * n_slots_per_expert]`.
    ///
    /// `D = softmax(X · Φ / sqrt(input_dim), dim=-1)`
    pub fn dispatch_weights(&self, x: &[f32], n_tokens: usize) -> MoeResult<Vec<f32>> {
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

        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        let scale = 1.0 / (cfg.input_dim as f32).sqrt();
        let mut logits = vec![0.0_f32; n_tokens * n_slots];

        // logits[t, j] = scale * dot(x[t], phi[:, j])
        // phi layout: phi[d * n_slots + j] = phi[d][j]
        // But we store phi as [input_dim * n_slots] row-major with dim as outer:
        // phi[slot_idx * input_dim + dim_idx]
        // Let's interpret phi as [n_slots * input_dim] for matrix multiply friendliness
        for tok in 0..n_tokens {
            let x_row = &x[tok * cfg.input_dim..(tok + 1) * cfg.input_dim];
            for slot in 0..n_slots {
                let phi_col = &self.phi[slot * cfg.input_dim..(slot + 1) * cfg.input_dim];
                let dot: f32 = x_row
                    .iter()
                    .zip(phi_col.iter())
                    .map(|(&xi, &pi)| xi * pi)
                    .sum();
                logits[tok * n_slots + slot] = dot * scale;
            }
        }

        // Softmax over slot dimension for each token
        let mut dispatch = vec![0.0_f32; n_tokens * n_slots];
        for tok in 0..n_tokens {
            let logit_row = &logits[tok * n_slots..(tok + 1) * n_slots];
            let probs = stable_softmax(logit_row);
            dispatch[tok * n_slots..(tok + 1) * n_slots].copy_from_slice(&probs);
        }

        Ok(dispatch)
    }

    /// Create expert inputs from dispatch weights.
    ///
    /// For slot `j`: `x_j = Σ_t (D[t,j] / Σ_t D[t,j]) · x_t`
    ///
    /// Returns aggregated inputs of shape `[n_slots * input_dim]`.
    pub fn aggregate_inputs(
        &self,
        x: &[f32],
        dispatch: &[f32],
        n_tokens: usize,
    ) -> MoeResult<Vec<f32>> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        let expected_d = n_tokens * n_slots;
        if dispatch.len() != expected_d {
            return Err(MoeError::DimensionMismatch {
                expected: expected_d,
                got: dispatch.len(),
            });
        }
        let expected_x = n_tokens * cfg.input_dim;
        if x.len() != expected_x {
            return Err(MoeError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }

        // Compute column sums of dispatch for normalization
        let mut col_sums = vec![0.0_f32; n_slots];
        for tok in 0..n_tokens {
            for slot in 0..n_slots {
                col_sums[slot] += dispatch[tok * n_slots + slot];
            }
        }

        let mut aggregated = vec![0.0_f32; n_slots * cfg.input_dim];

        for slot in 0..n_slots {
            let denom = col_sums[slot] + 1e-12;
            let agg_row = &mut aggregated[slot * cfg.input_dim..(slot + 1) * cfg.input_dim];
            for tok in 0..n_tokens {
                let weight = dispatch[tok * n_slots + slot] / denom;
                let x_row = &x[tok * cfg.input_dim..(tok + 1) * cfg.input_dim];
                for (agg_val, &xi) in agg_row.iter_mut().zip(x_row.iter()) {
                    *agg_val += weight * xi;
                }
            }
        }

        Ok(aggregated)
    }

    /// Combine expert outputs: `y_t = Σ_j D[t,j] · out_j`.
    ///
    /// # Arguments
    /// * `expert_out` — expert FFN outputs, shape `[n_slots * d_model]`
    /// * `dispatch` — dispatch weights, shape `[n_tokens * n_slots]`
    /// * `n_tokens` — number of tokens
    /// * `d_model` — output dimension
    pub fn combine_outputs(
        &self,
        expert_out: &[f32],
        dispatch: &[f32],
        n_tokens: usize,
        d_model: usize,
    ) -> MoeResult<Vec<f32>> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        if d_model == 0 {
            return Err(MoeError::InvalidHiddenDim { dim: d_model });
        }
        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        let expected_expert_out = n_slots * d_model;
        if expert_out.len() != expected_expert_out {
            return Err(MoeError::DimensionMismatch {
                expected: expected_expert_out,
                got: expert_out.len(),
            });
        }
        let expected_dispatch = n_tokens * n_slots;
        if dispatch.len() != expected_dispatch {
            return Err(MoeError::DimensionMismatch {
                expected: expected_dispatch,
                got: dispatch.len(),
            });
        }

        let mut output = vec![0.0_f32; n_tokens * d_model];

        for tok in 0..n_tokens {
            let out_row = &mut output[tok * d_model..(tok + 1) * d_model];
            for slot in 0..n_slots {
                let weight = dispatch[tok * n_slots + slot];
                let exp_row = &expert_out[slot * d_model..(slot + 1) * d_model];
                for (out_val, &exp_val) in out_row.iter_mut().zip(exp_row.iter()) {
                    *out_val += weight * exp_val;
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn dispatch_weights_sum_to_one() {
        let mut rng = LcgRng::new(7);
        let cfg = SoftMoeConfig {
            n_experts: 4,
            n_slots_per_expert: 2,
            input_dim: 16,
        };
        let n_tokens = 8;
        let router = SoftMoeRouter::new(cfg.clone(), &mut rng).expect("value should be present");
        let x = vec![0.5_f32; n_tokens * cfg.input_dim];
        let dispatch = router
            .dispatch_weights(&x, n_tokens)
            .expect("dispatch_weights should succeed");
        let n_slots = cfg.n_experts * cfg.n_slots_per_expert;
        for tok in 0..n_tokens {
            let row_sum: f32 = dispatch[tok * n_slots..(tok + 1) * n_slots].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row_sum={row_sum}");
        }
    }
}
