/// Compute the fraction of parameters that are trainable.
///
/// Returns `trainable / total` as an `f32`. Returns `0.0` if `total == 0`.
#[must_use]
pub fn param_efficiency_ratio(trainable: usize, total: usize) -> f32 {
    if total == 0 {
        return 0.0;
    }
    trainable as f32 / total as f32
}

/// Compute the energy-based effective rank of a set of singular values.
///
/// Effective rank = `(Σ s_i)² / (Σ s_i²)`.  Returns `0.0` if all values are zero.
#[must_use]
pub fn effective_rank(singular_values: &[f32]) -> f32 {
    let sum: f32 = singular_values.iter().sum();
    let sum_sq: f32 = singular_values.iter().map(|&v| v * v).sum();
    if sum_sq == 0.0 {
        return 0.0;
    }
    (sum * sum) / sum_sq
}

/// Count the trainable parameters introduced by a LoRA adapter.
///
/// The formula is `rank * (in_dim + out_dim)` (A: rank×in, B: out×rank).
#[must_use]
pub fn lora_param_count(in_dim: usize, out_dim: usize, rank: usize) -> usize {
    rank * (in_dim + out_dim)
}

/// Compute the compression ratio: `original_params / peft_params`.
///
/// Returns `0.0` if `peft_params == 0`.
#[must_use]
pub fn compression_ratio(original_params: usize, peft_params: usize) -> f32 {
    if peft_params == 0 {
        return 0.0;
    }
    original_params as f32 / peft_params as f32
}
