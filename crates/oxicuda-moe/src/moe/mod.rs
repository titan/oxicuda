//! Complete sparse Mixture-of-Experts *layer* implementations.
//!
//! Where [`crate::routing`] provides routing primitives and [`crate::expert`]
//! provides the expert feed-forward networks, this module assembles them into
//! end-to-end MoE layers used by modern large language models:
//!
//! * [`mixtral`] — Mixtral-style top-2 sparse MoE (Jiang et al. 2024).
//! * [`lora_moe`] — LoRAMoE: a mixture of low-rank LoRA adapters (Sheng et al. 2024).
//! * [`hierarchical`] — Hierarchical MoE: two-level group → expert routing
//!   (Jordan & Jacobs 1994; hierarchical sparse routing).
//! * [`upcycle`] — Sparse upcycling: warm-start an MoE from a dense FFN
//!   checkpoint (Komatsuzaki et al. 2023).

pub mod hierarchical;
pub mod lora_moe;
pub mod mixtral;
pub mod upcycle;

use crate::error::{MoeError, MoeResult};
use crate::routing::top_k::stable_softmax;

/// Dense matrix–vector product `y = W · x`.
///
/// `weight` is row-major with `cols` columns (`rows = weight.len() / cols`),
/// `x` has length `cols`, and the returned vector has length `rows`.
///
/// # Errors
/// Returns [`MoeError::DimensionMismatch`] when `x.len() != cols`, and
/// [`MoeError::Internal`] when `weight.len()` is not a multiple of `cols`.
pub(crate) fn matvec(weight: &[f32], x: &[f32], cols: usize) -> MoeResult<Vec<f32>> {
    if x.len() != cols {
        return Err(MoeError::DimensionMismatch {
            expected: cols,
            got: x.len(),
        });
    }
    if cols == 0 || !weight.len().is_multiple_of(cols) {
        return Err(MoeError::Internal {
            msg: format!(
                "matvec weight length {} not divisible by cols {cols}",
                weight.len()
            ),
        });
    }
    Ok(weight
        .chunks_exact(cols)
        .map(|row| row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum())
        .collect())
}

/// Switch-style load-balance auxiliary loss generalised to top-k (multi-slot)
/// routing.
///
/// `L = n_experts · Σ_i f_i · P_i`, where
/// * `f_i` is the fraction of routing *slots* assigned to expert `i`
///   (`Σ_i f_i = 1`, counting every selected `(token, slot)` pair once), and
/// * `P_i = (1/T) Σ_t softmax(logits_t)[i]` is the mean router probability for
///   expert `i` (`Σ_i P_i = 1`).
///
/// The loss equals `1` when both `f` and `P` are uniform (`1/n_experts`) and
/// grows toward `n_experts` as routing collapses onto a single expert, so it is
/// minimised by balanced routing and is always non-negative.
///
/// # Arguments
/// * `router_logits` — raw gate logits, shape `[n_tokens · n_experts]`.
/// * `selected_indices` — selected expert indices, shape `[n_tokens · slots]`
///   (every entry in `[0, n_experts)`); `slots = len / n_tokens`.
///
/// # Errors
/// Returns [`MoeError`] on empty input, zero experts, a `router_logits` length
/// mismatch, a `selected_indices` length not divisible by `n_tokens`, or an
/// out-of-range expert index.
pub(crate) fn topk_balance_loss(
    router_logits: &[f32],
    selected_indices: &[usize],
    n_tokens: usize,
    n_experts: usize,
) -> MoeResult<f32> {
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if n_experts == 0 {
        return Err(MoeError::InvalidExpertCount { n_experts });
    }
    let expected_logits = n_tokens * n_experts;
    if router_logits.len() != expected_logits {
        return Err(MoeError::DimensionMismatch {
            expected: expected_logits,
            got: router_logits.len(),
        });
    }
    if selected_indices.is_empty() || !selected_indices.len().is_multiple_of(n_tokens) {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: selected_indices.len(),
        });
    }

    // f_i: fraction of routing slots assigned to each expert.
    let mut slot_counts = vec![0_usize; n_experts];
    for &idx in selected_indices {
        if idx >= n_experts {
            return Err(MoeError::ExpertIndexOutOfRange { idx, n_experts });
        }
        slot_counts[idx] += 1;
    }
    let total_slots = selected_indices.len() as f32;
    let fraction: Vec<f32> = slot_counts
        .iter()
        .map(|&c| c as f32 / total_slots)
        .collect();

    // P_i: mean router softmax probability for each expert.
    let mut prob_sum = vec![0.0_f32; n_experts];
    for tok in 0..n_tokens {
        let probs = stable_softmax(&router_logits[tok * n_experts..(tok + 1) * n_experts]);
        for (acc, &p) in prob_sum.iter_mut().zip(probs.iter()) {
            *acc += p;
        }
    }
    let token_count = n_tokens as f32;
    let mean_prob: Vec<f32> = prob_sum.iter().map(|&s| s / token_count).collect();

    let loss = n_experts as f32
        * fraction
            .iter()
            .zip(mean_prob.iter())
            .map(|(&f, &p)| f * p)
            .sum::<f32>();

    if !loss.is_finite() {
        return Err(MoeError::NanEncountered {
            context: "topk_balance_loss".to_string(),
        });
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        // 2x2 identity times [3, 5] = [3, 5].
        let w = [1.0_f32, 0.0, 0.0, 1.0];
        let y = matvec(&w, &[3.0, 5.0], 2).expect("matvec should succeed");
        assert_eq!(y, vec![3.0, 5.0]);
    }

    #[test]
    fn matvec_rectangular() {
        // 3x2 matrix times [1, 1] sums each row.
        let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = matvec(&w, &[1.0, 1.0], 2).expect("matvec should succeed");
        assert_eq!(y, vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn matvec_dim_mismatch_errors() {
        let w = [1.0_f32, 2.0, 3.0, 4.0];
        assert!(matches!(
            matvec(&w, &[1.0, 1.0, 1.0], 2),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn balance_loss_uniform_is_one() {
        // 4 experts, 8 tokens, top-1; uniform logits -> P_i = 1/4.
        let n_tokens = 8;
        let n_experts = 4;
        let logits = vec![0.0_f32; n_tokens * n_experts];
        // Round-robin selection -> f_i = 1/4 each.
        let selected: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let loss = topk_balance_loss(&logits, &selected, n_tokens, n_experts)
            .expect("topk_balance_loss should succeed");
        assert!((loss - 1.0).abs() < 1e-4, "uniform loss {loss} != 1");
    }

    #[test]
    fn balance_loss_collapsed_is_large() {
        // All tokens collapse onto expert 0 with a strong logit bias.
        let n_tokens = 8;
        let n_experts = 4;
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for tok in 0..n_tokens {
            logits[tok * n_experts] = 20.0; // expert 0 dominates softmax.
        }
        let selected = vec![0_usize; n_tokens];
        let loss = topk_balance_loss(&logits, &selected, n_tokens, n_experts)
            .expect("topk_balance_loss should succeed");
        // f_0 = 1, P_0 ~ 1 -> L ~ n_experts.
        assert!(
            loss > 3.5,
            "collapsed loss {loss} should approach n_experts"
        );
    }

    #[test]
    fn balance_loss_collapsed_exceeds_uniform() {
        let n_tokens = 8;
        let n_experts = 4;
        let uniform_logits = vec![0.0_f32; n_tokens * n_experts];
        let uniform_sel: Vec<usize> = (0..n_tokens).map(|t| t % n_experts).collect();
        let uniform = topk_balance_loss(&uniform_logits, &uniform_sel, n_tokens, n_experts)
            .expect("topk_balance_loss should succeed");

        let mut collapsed_logits = vec![0.0_f32; n_tokens * n_experts];
        for tok in 0..n_tokens {
            collapsed_logits[tok * n_experts] = 20.0;
        }
        let collapsed_sel = vec![0_usize; n_tokens];
        let collapsed = topk_balance_loss(&collapsed_logits, &collapsed_sel, n_tokens, n_experts)
            .expect("topk_balance_loss should succeed");
        assert!(collapsed > uniform, "{collapsed} !> {uniform}");
        assert!(uniform >= 0.0 && collapsed >= 0.0);
    }

    #[test]
    fn balance_loss_out_of_range_errors() {
        let logits = vec![0.0_f32; 2 * 3];
        let selected = [0_usize, 9]; // 9 >= 3
        assert!(matches!(
            topk_balance_loss(&logits, &selected, 2, 3),
            Err(MoeError::ExpertIndexOutOfRange { .. })
        ));
    }
}
