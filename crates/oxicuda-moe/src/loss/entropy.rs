//! Entropy regularization for routing.
//!
//! Encourages diverse token-to-expert assignment by maximizing routing entropy.
//!
//! `L_ent = -(1/T) * Σ_t Σ_e p_{t,e} * log(p_{t,e} + ε)`
//!
//! High entropy → tokens spread across experts (beneficial for utilization).

use crate::error::{MoeError, MoeResult};

/// Compute the mean routing entropy over all tokens.
///
/// # Arguments
/// * `router_probs` — gate probabilities after softmax, shape `[n_tokens * n_experts]`
/// * `n_tokens` — number of tokens
/// * `n_experts` — number of experts
pub fn routing_entropy(router_probs: &[f32], n_tokens: usize, n_experts: usize) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected = n_tokens * n_experts;
    if router_probs.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: router_probs.len(),
        });
    }

    let eps = 1e-10_f32;
    let mut total_entropy = 0.0_f32;

    for tok in 0..n_tokens {
        let prob_row = &router_probs[tok * n_experts..(tok + 1) * n_experts];
        let token_entropy: f32 = prob_row.iter().map(|&prob| -prob * (prob + eps).ln()).sum();
        total_entropy += token_entropy;
    }

    Ok(total_entropy / n_tokens as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::top_k::stable_softmax;

    #[test]
    fn entropy_uniform_is_max() {
        let n_tokens = 2_usize;
        let n_experts = 4_usize;
        // Uniform distribution: p = 1/E for all experts
        let probs = vec![0.25_f32; n_tokens * n_experts];
        let entropy = routing_entropy(&probs, n_tokens, n_experts).unwrap();
        let max_entropy = (n_experts as f32).ln();
        assert!(
            (entropy - max_entropy).abs() < 1e-3,
            "entropy={entropy}, max={max_entropy}"
        );
    }

    #[test]
    fn entropy_concentrated_is_low() {
        let n_tokens = 1_usize;
        let n_experts = 4_usize;
        // One-hot: p = [1, 0, 0, 0]
        let probs = [1.0_f32, 0.0, 0.0, 0.0];
        let entropy = routing_entropy(&probs, n_tokens, n_experts).unwrap();
        // Entropy should be near 0
        assert!(entropy < 0.01, "entropy={entropy}");
    }

    #[test]
    fn entropy_nonneg_for_softmax_output() {
        let logits = [0.5_f32, 1.5, -0.3, 2.0, 0.1, 0.9, 1.2, -0.5];
        let probs = stable_softmax(&logits);
        let entropy = routing_entropy(&probs, 1, logits.len()).unwrap();
        assert!(entropy >= 0.0, "entropy must be >= 0, got {entropy}");
    }
}
