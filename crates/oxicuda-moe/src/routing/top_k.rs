//! Top-K routing: compute softmax(W_g · x) and select top-k experts per token.
//!
//! Implements the routing mechanism from:
//! Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer."
//! ICLR 2017.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Configuration for top-k routing.
#[derive(Debug, Clone)]
pub struct TopKConfig {
    /// Number of experts per token (1 for Switch, 2 for GShard).
    pub k: usize,
    /// Total number of experts.
    pub n_experts: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Standard deviation for additive jitter noise (0.0 = no noise).
    pub noise_std: f32,
}

/// Output of a top-k routing pass.
#[derive(Debug, Clone)]
pub struct TopKResult {
    /// Gate scores after softmax, normalized within each token's top-k selection.
    /// Shape: `[n_tokens * k]`.
    pub scores: Vec<f32>,
    /// Expert indices for each token's top-k selection.
    /// Shape: `[n_tokens * k]`.
    pub indices: Vec<usize>,
    /// Raw gate logits before softmax.
    /// Shape: `[n_tokens * n_experts]`.
    pub router_logits: Vec<f32>,
}

/// Top-k router: holds gate weight matrix and config.
#[derive(Debug, Clone)]
pub struct TopKRouter {
    /// Gate weight matrix, shape `[n_experts * input_dim]`.
    pub weights: Vec<f32>,
    /// Routing configuration.
    pub config: TopKConfig,
}

impl TopKRouter {
    /// Create a new router with random weight initialization (N(0, 0.01)).
    pub fn new(cfg: TopKConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        if cfg.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: cfg.n_experts,
            });
        }
        if cfg.input_dim == 0 {
            return Err(MoeError::InvalidInputDim { dim: cfg.input_dim });
        }
        if cfg.k == 0 || cfg.k > cfg.n_experts {
            return Err(MoeError::InvalidTopK {
                k: cfg.k,
                n_experts: cfg.n_experts,
            });
        }
        let weight_count = cfg.n_experts * cfg.input_dim;
        let mut weights = vec![0.0_f32; weight_count];
        rng.fill_normal_scaled(&mut weights, 0.01);
        Ok(Self {
            weights,
            config: cfg,
        })
    }

    /// Route `n_tokens` tokens from input `x` (shape `[n_tokens * input_dim]`).
    ///
    /// Returns `TopKResult` with scores, indices, and raw logits.
    pub fn route(&self, x: &[f32], n_tokens: usize) -> MoeResult<TopKResult> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected_len = n_tokens * cfg.input_dim;
        if x.len() != expected_len {
            return Err(MoeError::DimensionMismatch {
                expected: expected_len,
                got: x.len(),
            });
        }

        let mut router_logits = vec![0.0_f32; n_tokens * cfg.n_experts];

        // Compute logits = x · W_g^T: for each token t and expert e,
        // logit[t * E + e] = dot(x[t*d..(t+1)*d], weights[e*d..(e+1)*d])
        for tok in 0..n_tokens {
            let x_row = &x[tok * cfg.input_dim..(tok + 1) * cfg.input_dim];
            for exp_idx in 0..cfg.n_experts {
                let w_row = &self.weights[exp_idx * cfg.input_dim..(exp_idx + 1) * cfg.input_dim];
                let dot: f32 = x_row
                    .iter()
                    .zip(w_row.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum();
                router_logits[tok * cfg.n_experts + exp_idx] = dot;
            }
        }

        // Optionally add jitter noise
        if cfg.noise_std > 0.0 {
            // Cannot add noise here without mut rng — noise is applied via add_noise helper.
            // Users should call add_noise separately if needed.
        }

        // Compute softmax and top-k per token
        let mut scores = vec![0.0_f32; n_tokens * cfg.k];
        let mut indices = vec![0_usize; n_tokens * cfg.k];

        for tok in 0..n_tokens {
            let logit_slice = &router_logits[tok * cfg.n_experts..(tok + 1) * cfg.n_experts];
            let soft = stable_softmax(logit_slice);
            let (top_vals, top_idx) = topk(&soft, cfg.k)?;

            // Normalize top-k scores to sum=1 when k > 1
            let score_sum: f32 = top_vals.iter().sum();
            let normalized_sum = if score_sum > 1e-12 { score_sum } else { 1.0 };

            for slot in 0..cfg.k {
                let normalized = if cfg.k > 1 {
                    top_vals[slot] / normalized_sum
                } else {
                    top_vals[slot]
                };
                scores[tok * cfg.k + slot] = normalized;
                indices[tok * cfg.k + slot] = top_idx[slot];
            }
        }

        // Check for NaN
        if scores.iter().any(|v| v.is_nan()) {
            return Err(MoeError::NanEncountered {
                context: "top_k router scores".to_string(),
            });
        }

        Ok(TopKResult {
            scores,
            indices,
            router_logits,
        })
    }

    /// Route with additive jitter noise for training (noisy top-k).
    pub fn route_with_noise(
        &self,
        x: &[f32],
        n_tokens: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<TopKResult> {
        let cfg = &self.config;
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }
        let expected_len = n_tokens * cfg.input_dim;
        if x.len() != expected_len {
            return Err(MoeError::DimensionMismatch {
                expected: expected_len,
                got: x.len(),
            });
        }

        let mut router_logits = vec![0.0_f32; n_tokens * cfg.n_experts];

        for tok in 0..n_tokens {
            let x_row = &x[tok * cfg.input_dim..(tok + 1) * cfg.input_dim];
            for exp_idx in 0..cfg.n_experts {
                let w_row = &self.weights[exp_idx * cfg.input_dim..(exp_idx + 1) * cfg.input_dim];
                let dot: f32 = x_row
                    .iter()
                    .zip(w_row.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum();
                router_logits[tok * cfg.n_experts + exp_idx] = dot;
            }
        }

        // Add N(0, noise_std) noise
        if cfg.noise_std > 0.0 {
            let mut noise = vec![0.0_f32; n_tokens * cfg.n_experts];
            rng.fill_normal_scaled(&mut noise, cfg.noise_std);
            for (logit, n) in router_logits.iter_mut().zip(noise.iter()) {
                *logit += n;
            }
        }

        let mut scores = vec![0.0_f32; n_tokens * cfg.k];
        let mut indices = vec![0_usize; n_tokens * cfg.k];

        for tok in 0..n_tokens {
            let logit_slice = &router_logits[tok * cfg.n_experts..(tok + 1) * cfg.n_experts];
            let soft = stable_softmax(logit_slice);
            let (top_vals, top_idx) = topk(&soft, cfg.k)?;

            let score_sum: f32 = top_vals.iter().sum();
            let normalized_sum = if score_sum > 1e-12 { score_sum } else { 1.0 };

            for slot in 0..cfg.k {
                let normalized = if cfg.k > 1 {
                    top_vals[slot] / normalized_sum
                } else {
                    top_vals[slot]
                };
                scores[tok * cfg.k + slot] = normalized;
                indices[tok * cfg.k + slot] = top_idx[slot];
            }
        }

        if scores.iter().any(|v| v.is_nan()) {
            return Err(MoeError::NanEncountered {
                context: "top_k router scores (noisy)".to_string(),
            });
        }

        Ok(TopKResult {
            scores,
            indices,
            router_logits,
        })
    }

    /// Return the number of trainable parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.weights.len()
    }
}

/// Numerically stable softmax over a slice of logits.
///
/// Uses the max-subtraction trick to prevent overflow.
#[must_use]
pub fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&val| (val - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter()
        .map(|&exp_val| exp_val / (sum + 1e-12))
        .collect()
}

/// Select the top-k values and their indices from a slice, sorted descending.
///
/// Uses a linear scan for small k (k=1 or k=2) and a partial sort otherwise.
pub fn topk(values: &[f32], k: usize) -> MoeResult<(Vec<f32>, Vec<usize>)> {
    if values.is_empty() {
        return Err(MoeError::EmptyInput);
    }
    if k == 0 || k > values.len() {
        return Err(MoeError::InvalidTopK {
            k,
            n_experts: values.len(),
        });
    }

    match k {
        1 => {
            // argmax via linear scan
            let (best_idx, &best_val) = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &f32::NEG_INFINITY));
            Ok((vec![best_val], vec![best_idx]))
        }
        2 => {
            // One-pass for top-2
            let mut first_val = f32::NEG_INFINITY;
            let mut second_val = f32::NEG_INFINITY;
            let mut first_idx = 0usize;
            let mut second_idx = 0usize;
            for (idx, &val) in values.iter().enumerate() {
                if val > first_val {
                    second_val = first_val;
                    second_idx = first_idx;
                    first_val = val;
                    first_idx = idx;
                } else if val > second_val {
                    second_val = val;
                    second_idx = idx;
                }
            }
            Ok((vec![first_val, second_val], vec![first_idx, second_idx]))
        }
        _ => {
            // General case: partial sort with a heap-like approach
            let mut indexed: Vec<(f32, usize)> = values
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();
            // Partial sort: bring top-k to front
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_slice = &mut indexed[..k];
            top_slice.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_vals: Vec<f32> = top_slice.iter().map(|(v, _)| *v).collect();
            let top_idx: Vec<usize> = top_slice.iter().map(|(_, i)| *i).collect();
            Ok((top_vals, top_idx))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn topk_k1_returns_max() {
        let vals = [0.1_f32, 0.9, 0.3, 0.5];
        let (top_vals, top_idx) = topk(&vals, 1).expect("topk should succeed");
        assert_eq!(top_vals.len(), 1);
        assert_eq!(top_idx[0], 1);
        assert!((top_vals[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn topk_k2_returns_top2() {
        let vals = [0.1_f32, 0.9, 0.3, 0.5];
        let (top_vals, top_idx) = topk(&vals, 2).expect("topk should succeed");
        assert_eq!(top_vals.len(), 2);
        assert_eq!(top_idx[0], 1);
        assert_eq!(top_idx[1], 3);
        assert!((top_vals[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn stable_softmax_sum_one() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0];
        let probs = stable_softmax(&logits);
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn router_new_and_route() {
        let mut rng = LcgRng::new(0);
        let cfg = TopKConfig {
            k: 1,
            n_experts: 4,
            input_dim: 8,
            noise_std: 0.0,
        };
        let router = TopKRouter::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 8];
        let result = router.route(&x, 1).expect("route should succeed");
        assert_eq!(result.scores.len(), 1);
        assert!(result.indices[0] < 4);
    }
}
