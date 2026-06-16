//! Mixture-of-Experts CPU layer (Shazeer et al., 2017).
//!
//! A pure-Rust, CPU-only simulation of Sparse MoE that routes each input token
//! to the `top_k` highest-scoring experts and blends their outputs.  This is
//! intentionally kept simple so it can be used without a GPU; the
//! GPU-accelerated version lives in `crate::moe`.

use crate::LcgRng;
use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`MoeLayer`].
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of expert networks.
    pub n_experts: usize,
    /// How many experts each token is routed to.
    pub top_k: usize,
    /// Input / output model dimension.
    pub d_model: usize,
    /// Hidden dimension inside each expert FFN.
    pub d_ffn: usize,
    /// Coefficient for the load-balancing auxiliary loss.
    pub aux_loss_coeff: f32,
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// CPU Mixture-of-Experts layer.
///
/// Each expert is a two-layer FFN:  `y = W2 · relu(W1 · x + b1) + b2`.
/// A learned gating network produces per-expert weights, and only the top-k
/// experts contribute to the final output for each token.
pub struct MoeLayer {
    /// Gating projection weights `[n_experts × d_model]`.
    gate_w: Vec<f32>,
    /// Per-expert first-layer weights `[n_experts][d_ffn × d_model]`.
    expert_w1: Vec<Vec<f32>>,
    /// Per-expert first-layer biases `[n_experts][d_ffn]`.
    expert_b1: Vec<Vec<f32>>,
    /// Per-expert second-layer weights `[n_experts][d_model × d_ffn]`.
    expert_w2: Vec<Vec<f32>>,
    /// Per-expert second-layer biases `[n_experts][d_model]`.
    expert_b2: Vec<Vec<f32>>,
    config: MoeConfig,
}

impl MoeLayer {
    /// Construct a new [`MoeLayer`] with randomly-initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] when any dimension is zero or
    /// `top_k > n_experts`.
    pub fn new(config: MoeConfig, rng: &mut LcgRng) -> DnnResult<Self> {
        if config.n_experts == 0 {
            return Err(DnnError::InvalidArgument(
                "n_experts must be > 0".to_owned(),
            ));
        }
        if config.top_k == 0 {
            return Err(DnnError::InvalidArgument("top_k must be > 0".to_owned()));
        }
        if config.d_model == 0 {
            return Err(DnnError::InvalidArgument("d_model must be > 0".to_owned()));
        }
        if config.d_ffn == 0 {
            return Err(DnnError::InvalidArgument("d_ffn must be > 0".to_owned()));
        }
        if config.top_k > config.n_experts {
            return Err(DnnError::InvalidArgument(format!(
                "top_k ({}) must be <= n_experts ({})",
                config.top_k, config.n_experts
            )));
        }

        let n_e = config.n_experts;
        let d_m = config.d_model;
        let d_f = config.d_ffn;

        let gate_w: Vec<f32> = (0..n_e * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 0.1)
            .collect();

        let expert_w1: Vec<Vec<f32>> = (0..n_e)
            .map(|_| {
                (0..d_f * d_m)
                    .map(|_| (rng.next_f64() as f32 - 0.5) * 0.1)
                    .collect()
            })
            .collect();

        let expert_b1: Vec<Vec<f32>> = (0..n_e)
            .map(|_| {
                (0..d_f)
                    .map(|_| (rng.next_f64() as f32 - 0.5) * 0.1)
                    .collect()
            })
            .collect();

        let expert_w2: Vec<Vec<f32>> = (0..n_e)
            .map(|_| {
                (0..d_m * d_f)
                    .map(|_| (rng.next_f64() as f32 - 0.5) * 0.1)
                    .collect()
            })
            .collect();

        let expert_b2: Vec<Vec<f32>> = (0..n_e)
            .map(|_| {
                (0..d_m)
                    .map(|_| (rng.next_f64() as f32 - 0.5) * 0.1)
                    .collect()
            })
            .collect();

        Ok(Self {
            gate_w,
            expert_w1,
            expert_b1,
            expert_w2,
            expert_b2,
            config,
        })
    }

    /// Number of experts in this layer.
    #[inline]
    pub fn n_experts(&self) -> usize {
        self.config.n_experts
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------

    /// Run the MoE layer on `n_tokens` tokens.
    ///
    /// `x` must have length `n_tokens × d_model`.  Returns a flat vector of
    /// length `n_tokens × d_model`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] when `x.len()` does not match
    /// the expected shape.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> DnnResult<Vec<f32>> {
        let d_m = self.config.d_model;
        let d_f = self.config.d_ffn;
        let n_e = self.config.n_experts;
        let top_k = self.config.top_k;

        if x.len() != n_tokens * d_m {
            return Err(DnnError::InvalidDimension(format!(
                "expected x.len() = {} ({} tokens × d_model {}), got {}",
                n_tokens * d_m,
                n_tokens,
                d_m,
                x.len()
            )));
        }

        // ------------------------------------------------------------------
        // Step 1: gate logits  [n_tokens × n_experts]
        // ------------------------------------------------------------------
        let mut gate_logits: Vec<f32> = vec![0.0f32; n_tokens * n_e];
        for t in 0..n_tokens {
            let x_t = &x[t * d_m..(t + 1) * d_m];
            for e in 0..n_e {
                let s: f32 = x_t
                    .iter()
                    .zip(self.gate_w[e * d_m..(e + 1) * d_m].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                gate_logits[t * n_e + e] = s;
            }
        }

        // ------------------------------------------------------------------
        // Step 2: softmax per token  → gate_probs  [n_tokens × n_experts]
        // ------------------------------------------------------------------
        let mut gate_probs: Vec<f32> = gate_logits.clone();
        for t in 0..n_tokens {
            let row = &mut gate_probs[t * n_e..(t + 1) * n_e];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            let inv_sum = if sum > 1e-30 { 1.0 / sum } else { 1.0 };
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        }

        // ------------------------------------------------------------------
        // Step 3–5: route and accumulate
        // ------------------------------------------------------------------
        let mut output = vec![0.0f32; n_tokens * d_m];

        for t in 0..n_tokens {
            let x_t = &x[t * d_m..(t + 1) * d_m];
            let probs_t = &gate_probs[t * n_e..(t + 1) * n_e];

            // Partial sort: pick top_k experts by probability.
            let mut ranked: Vec<(usize, f32)> = probs_t.iter().copied().enumerate().collect();
            // Sort descending by probability.
            ranked.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let selected = &ranked[..top_k];

            // Normalise by sum of selected weights.
            let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
            let inv_weight = if weight_sum > 1e-30 {
                1.0 / weight_sum
            } else {
                0.0
            };

            let out_t = &mut output[t * d_m..(t + 1) * d_m];

            for &(e, gate_w) in selected {
                let norm_w = gate_w * inv_weight;
                if norm_w == 0.0 {
                    continue;
                }

                // hidden = relu(W1_e · x_t + b1_e)  [d_ffn]
                let w1 = &self.expert_w1[e];
                let b1 = &self.expert_b1[e];
                let mut hidden = vec![0.0f32; d_f];
                for j in 0..d_f {
                    let mut v = b1[j];
                    for d in 0..d_m {
                        v += w1[j * d_m + d] * x_t[d];
                    }
                    hidden[j] = v.max(0.0); // relu
                }

                // out_e = W2_e · hidden + b2_e  [d_model]
                let w2 = &self.expert_w2[e];
                let b2 = &self.expert_b2[e];
                for i in 0..d_m {
                    let mut v = b2[i];
                    for j in 0..d_f {
                        v += w2[i * d_f + j] * hidden[j];
                    }
                    out_t[i] += norm_w * v;
                }
            }
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Auxiliary loss
    // -----------------------------------------------------------------------

    /// Compute the load-balancing auxiliary loss (Switch Transformer style).
    ///
    /// `gate_probs` is the flat `[n_tokens × n_experts]` softmax output
    /// obtained during a forward pass.  The caller is responsible for
    /// computing this; see [`Self::forward`] for the expected layout.
    ///
    /// Returns `aux_loss_coeff × n_experts × Σ_e (fraction_e × load_e)`.
    pub fn aux_loss(&self, gate_probs: &[f32], n_tokens: usize) -> f32 {
        let n_e = self.config.n_experts;
        if n_tokens == 0 || n_e == 0 {
            return 0.0;
        }

        // fraction_e: fraction of tokens whose argmax expert is e.
        let mut counts = vec![0usize; n_e];
        for t in 0..n_tokens {
            let row = &gate_probs[t * n_e..(t + 1) * n_e];
            let best_e = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            counts[best_e] += 1;
        }

        let inv_tokens = 1.0 / n_tokens as f32;
        let mut aux = 0.0f32;
        for e in 0..n_e {
            let fraction_e = counts[e] as f32 * inv_tokens;
            // load_e = mean gate prob for expert e across all tokens.
            let load_e: f32 =
                (0..n_tokens).map(|t| gate_probs[t * n_e + e]).sum::<f32>() * inv_tokens;
            aux += fraction_e * load_e;
        }

        self.config.aux_loss_coeff * (n_e as f32) * aux
    }

    // -----------------------------------------------------------------------
    // Internal: expose gate probs for tests
    // -----------------------------------------------------------------------

    /// Compute softmax gate probabilities for `x` without running the full
    /// forward pass.  Useful for testing auxiliary-loss behaviour.
    pub fn gate_probs(&self, x: &[f32], n_tokens: usize) -> DnnResult<Vec<f32>> {
        let d_m = self.config.d_model;
        let n_e = self.config.n_experts;

        if x.len() != n_tokens * d_m {
            return Err(DnnError::InvalidDimension(format!(
                "expected {}, got {}",
                n_tokens * d_m,
                x.len()
            )));
        }

        let mut gate_probs = vec![0.0f32; n_tokens * n_e];
        for t in 0..n_tokens {
            let x_t = &x[t * d_m..(t + 1) * d_m];
            let row = &mut gate_probs[t * n_e..(t + 1) * n_e];
            for (e, slot) in row.iter_mut().enumerate() {
                *slot = x_t
                    .iter()
                    .zip(self.gate_w[e * d_m..(e + 1) * d_m].iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            let inv_sum = if sum > 1e-30 { 1.0 / sum } else { 1.0 };
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        }

        Ok(gate_probs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(n_experts: usize, top_k: usize, d_model: usize, d_ffn: usize) -> MoeLayer {
        let cfg = MoeConfig {
            n_experts,
            top_k,
            d_model,
            d_ffn,
            aux_loss_coeff: 0.01,
        };
        let mut rng = LcgRng::new(42);
        MoeLayer::new(cfg, &mut rng).expect("valid config")
    }

    fn random_input(n_tokens: usize, d_model: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n_tokens * d_model)
            .map(|_| rng.next_f64() as f32)
            .collect()
    }

    // 1. Output shape
    #[test]
    fn output_shape() {
        let layer = make_layer(4, 2, 16, 32);
        let x = random_input(8, 16, 1);
        let out = layer.forward(&x, 8).expect("forward ok");
        assert_eq!(out.len(), 8 * 16);
    }

    // 2. All outputs are finite
    #[test]
    fn output_finite() {
        let layer = make_layer(4, 2, 16, 32);
        let x = random_input(6, 16, 2);
        let out = layer.forward(&x, 6).expect("forward ok");
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v}");
        }
    }

    // 3. top_k=1 always picks the single best expert
    #[test]
    fn top_k_1_always_best_expert() {
        let layer = make_layer(4, 1, 8, 16);
        let x = random_input(4, 8, 3);
        // Should succeed and produce valid output.
        let out = layer.forward(&x, 4).expect("top_k=1 forward ok");
        assert_eq!(out.len(), 4 * 8);
    }

    // 4. top_k > n_experts → error
    #[test]
    fn top_k_gt_n_experts_error() {
        let cfg = MoeConfig {
            n_experts: 3,
            top_k: 5,
            d_model: 8,
            d_ffn: 16,
            aux_loss_coeff: 0.01,
        };
        let mut rng = LcgRng::new(0);
        let result = MoeLayer::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(DnnError::InvalidArgument(_))),
            "expected InvalidArgument error"
        );
    }

    // 5. Gate probs per token sum to 1.0
    #[test]
    fn gate_sums_to_1() {
        let layer = make_layer(4, 2, 16, 32);
        let n_tokens = 5;
        let x = random_input(n_tokens, 16, 5);
        let probs = layer.gate_probs(&x, n_tokens).expect("gate_probs ok");
        for t in 0..n_tokens {
            let sum: f32 = probs[t * 4..(t + 1) * 4].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "token {t}: gate probs sum = {sum}"
            );
        }
    }

    // 6. Aux loss is non-negative
    #[test]
    fn aux_loss_nonneg() {
        let layer = make_layer(4, 2, 16, 32);
        let x = random_input(8, 16, 6);
        let probs = layer.gate_probs(&x, 8).expect("ok");
        let loss = layer.aux_loss(&probs, 8);
        assert!(loss >= 0.0, "aux_loss should be >= 0, got {loss}");
    }

    // 7. Different tokens get different routing with sufficiently distinct inputs
    #[test]
    fn different_tokens_different_routes() {
        // Use a large n_experts to make collisions less likely.
        let layer = make_layer(8, 1, 16, 32);
        let n_experts = 8;
        let d_model = 16;
        let n_tokens = n_experts; // one token per expert slot

        // Build inputs that are one-hot-like along different dimensions so that
        // the gate (a random linear map) is very likely to route them differently.
        let mut x = vec![0.0f32; n_tokens * d_model];
        for t in 0..n_tokens {
            // Token t is dominated by dimension (t * 2) % d_model.
            let hot_dim = (t * 2) % d_model;
            x[t * d_model + hot_dim] = 10.0;
        }

        let probs = layer.gate_probs(&x, n_tokens).expect("ok");
        let argmaxes: Vec<usize> = (0..n_tokens)
            .map(|t| {
                probs[t * n_experts..(t + 1) * n_experts]
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        // With n_experts=8 and 8 maximally-distinct tokens, at least 2
        // different experts should be chosen.
        let distinct_count = {
            let mut seen = [false; 8];
            for &e in &argmaxes {
                seen[e] = true;
            }
            seen.iter().filter(|&&b| b).count()
        };
        assert!(
            distinct_count >= 2,
            "expected multiple distinct routing decisions; argmaxes={argmaxes:?}"
        );
    }

    // 8. Output is non-zero for non-zero input
    #[test]
    fn output_nonzero() {
        let layer = make_layer(4, 2, 16, 32);
        let x = random_input(4, 16, 8);
        let out = layer.forward(&x, 4).expect("ok");
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            norm > 0.0,
            "output should not be all-zero for nonzero input"
        );
    }

    // 9. Aux loss is finite
    #[test]
    fn aux_loss_finite() {
        let layer = make_layer(4, 2, 16, 32);
        let x = random_input(8, 16, 9);
        let probs = layer.gate_probs(&x, 8).expect("ok");
        let loss = layer.aux_loss(&probs, 8);
        assert!(loss.is_finite(), "aux_loss should be finite, got {loss}");
    }

    // 10. n_experts=0 → error
    #[test]
    fn n_experts_0_error() {
        let cfg = MoeConfig {
            n_experts: 0,
            top_k: 1,
            d_model: 8,
            d_ffn: 16,
            aux_loss_coeff: 0.01,
        };
        let mut rng = LcgRng::new(0);
        let result = MoeLayer::new(cfg, &mut rng);
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }

    // 11. d_model=0 → error
    #[test]
    fn d_model_0_error() {
        let cfg = MoeConfig {
            n_experts: 4,
            top_k: 1,
            d_model: 0,
            d_ffn: 16,
            aux_loss_coeff: 0.01,
        };
        let mut rng = LcgRng::new(0);
        let result = MoeLayer::new(cfg, &mut rng);
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }
}
