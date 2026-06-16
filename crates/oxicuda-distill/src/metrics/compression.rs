//! Model compression metrics: parameter ratios, FLOPs, and latency speedup.

use crate::error::{DistillError, DistillResult};

/// Ratio of original to compressed parameter counts.
#[must_use]
pub fn param_ratio(original: usize, compressed: usize) -> f32 {
    if compressed == 0 {
        return f32::INFINITY;
    }
    original as f32 / compressed as f32
}

/// Ratio of original to compressed FLOPs counts.
#[must_use]
pub fn flops_ratio(original_flops: u64, compressed_flops: u64) -> f32 {
    if compressed_flops == 0 {
        return f32::INFINITY;
    }
    original_flops as f32 / compressed_flops as f32
}

/// Latency speedup: original_ms / compressed_ms.
pub fn latency_speedup(original_ms: f32, compressed_ms: f32) -> DistillResult<f32> {
    if compressed_ms <= 0.0 {
        return Err(DistillError::InvalidConfig {
            msg: format!("compressed_ms must be > 0, got {compressed_ms}"),
        });
    }
    Ok(original_ms / compressed_ms)
}

/// Estimate FLOPs for a LoRA-factored linear layer.
///
/// Full layer: `2 · seq_len · in_dim · out_dim`
/// LoRA overhead: `2 · seq_len · (in_dim · rank + rank · out_dim)`
///
/// Returns the LoRA FLOPs overhead (not the full-layer cost).
#[must_use]
pub fn estimate_lora_flops(in_dim: usize, out_dim: usize, rank: usize, seq_len: usize) -> u64 {
    let a = 2u64
        .saturating_mul(seq_len as u64)
        .saturating_mul(in_dim as u64)
        .saturating_mul(rank as u64);
    let b = 2u64
        .saturating_mul(seq_len as u64)
        .saturating_mul(rank as u64)
        .saturating_mul(out_dim as u64);
    a.saturating_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_ratio_ten_x() {
        assert!((param_ratio(1_000_000, 100_000) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn latency_speedup_ok() {
        let sp = latency_speedup(100.0, 25.0).expect("latency_speedup should succeed");
        assert!((sp - 4.0).abs() < 1e-4);
    }

    #[test]
    fn latency_speedup_zero_compressed_is_err() {
        assert!(latency_speedup(100.0, 0.0).is_err());
    }

    #[test]
    fn estimate_lora_flops_nonzero() {
        let flops = estimate_lora_flops(512, 512, 8, 128);
        assert!(flops > 0);
    }
}
