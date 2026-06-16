//! Router z-loss: prevents router collapse by penalizing large logits.
//!
//! From: Zoph et al. "ST-MoE: Designing Stable and Transferable Sparse Expert Models."
//!
//! `L_z = (1/T) * Σ_{t=1}^{T} log²(Σ_{e=1}^{E} exp(logit_{t,e}))`
//!
//! Uses logsumexp for numerical stability:
//! `lse_t = max_e(logit_t) + log(Σ_e exp(logit_t_e - max_e))`
//! `L_z = (1/T) * Σ_t lse_t²`

use crate::error::{MoeError, MoeResult};

/// Compute the router z-loss.
///
/// # Arguments
/// * `router_logits` — raw gate logits, shape `[n_tokens * n_experts]`
/// * `n_tokens` — number of tokens
/// * `n_experts` — number of experts
pub fn router_z_loss(router_logits: &[f32], n_tokens: usize, n_experts: usize) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected = n_tokens * n_experts;
    if router_logits.len() != expected {
        return Err(MoeError::DimensionMismatch {
            expected,
            got: router_logits.len(),
        });
    }

    let mut loss_acc = 0.0_f32;

    for tok in 0..n_tokens {
        let logit_row = &router_logits[tok * n_experts..(tok + 1) * n_experts];

        // Find max for numerical stability
        let max_logit = logit_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute log-sum-exp: lse = max + log(Σ exp(logit - max))
        let sum_exp: f32 = logit_row.iter().map(|&lg| (lg - max_logit).exp()).sum();
        let lse = max_logit + (sum_exp + 1e-12).ln();

        loss_acc += lse * lse;
    }

    Ok(loss_acc / n_tokens as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_loss_nonneg() {
        let logits = [1.0_f32, 2.0, 0.5, -0.3, 1.5, 0.8];
        let loss = router_z_loss(&logits, 2, 3).expect("router_z_loss should succeed");
        assert!(loss >= 0.0, "z-loss must be non-negative, got {loss}");
        assert!(loss.is_finite(), "z-loss must be finite, got {loss}");
    }

    #[test]
    fn z_loss_zero_logits() {
        // logits all zero → lse = log(E) for each token → loss = log(E)^2
        let n_experts = 4_usize;
        let n_tokens = 3_usize;
        let logits = vec![0.0_f32; n_tokens * n_experts];
        let loss =
            router_z_loss(&logits, n_tokens, n_experts).expect("router_z_loss should succeed");
        let expected = (n_experts as f32).ln().powi(2);
        assert!(
            (loss - expected).abs() < 1e-4,
            "z_loss={loss}, expected={expected}"
        );
    }
}
