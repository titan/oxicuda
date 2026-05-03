//! Error types for the `oxicuda-gen` crate.

use thiserror::Error;

/// All errors that can arise from generative AI operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GenError {
    /// Tensor or slice dimension does not match expectation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// An input that must be non-empty was empty.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// Timestep index is out of valid range.
    #[error("invalid timestep {t}: must be in [0, {max_t})")]
    InvalidTimestep { t: usize, max_t: usize },

    /// Beta schedule values are out of the valid (0, 1) range.
    #[error("invalid beta schedule: beta values must be in (0, 1)")]
    InvalidBetaSchedule,

    /// Classifier-free guidance scale must be >= 1.0.
    #[error("invalid guidance scale {0}: must be >= 1.0")]
    InvalidGuidanceScale(f32),

    /// LoRA rank must be >= 1.
    #[error("invalid LoRA rank {0}: must be >= 1")]
    InvalidLoraRank(usize),

    /// LoRA alpha must be > 0.
    #[error("invalid LoRA alpha {0}: must be > 0")]
    InvalidLoraAlpha(f32),

    /// Codebook size must be a power of two and >= 2.
    #[error("codebook size {0} must be a power of two and >= 2")]
    InvalidCodebookSize(usize),

    /// VQ-VAE commitment loss is not finite.
    #[error("VQ-VAE commitment loss not finite: {0}")]
    NonFiniteCommitmentLoss(f32),

    /// Weight shape is incompatible with the input shape.
    #[error("shape mismatch: weight shape {weight:?} incompatible with input {input:?}")]
    WeightShapeMismatch {
        weight: Vec<usize>,
        input: Vec<usize>,
    },

    /// SM version is too old (minimum SM 7.5 required).
    #[error("unsupported SM version {0}: minimum SM 7.5 required")]
    UnsupportedSmVersion(u32),

    /// DPM-Solver order must be 1, 2, or 3.
    #[error("DPM-Solver order {0} not supported: must be 1, 2, or 3")]
    UnsupportedDpmOrder(usize),

    /// Flow matching time t must be in [0, 1].
    #[error("flow matching: time t={0} out of range [0, 1]")]
    InvalidFlowTime(f32),

    /// Catch-all for internal invariant violations.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias for `Result<T, GenError>`.
pub type GenResult<T> = Result<T, GenError>;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = GenError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let s = e.to_string();
        assert!(s.contains("128"), "expected 128 in: {s}");
        assert!(s.contains("64"), "expected 64 in: {s}");
    }

    #[test]
    fn error_display_empty_input() {
        let e = GenError::EmptyInput("noise buffer");
        let s = e.to_string();
        assert!(s.contains("noise buffer"), "got: {s}");
    }

    #[test]
    fn error_display_invalid_timestep() {
        let e = GenError::InvalidTimestep {
            t: 1000,
            max_t: 999,
        };
        let s = e.to_string();
        assert!(s.contains("1000"), "got: {s}");
        assert!(s.contains("999"), "got: {s}");
    }

    #[test]
    fn error_display_invalid_beta_schedule() {
        let e = GenError::InvalidBetaSchedule;
        let s = e.to_string();
        assert!(s.contains("beta"), "got: {s}");
        assert!(s.contains("(0, 1)"), "got: {s}");
    }

    #[test]
    fn error_display_invalid_guidance_scale() {
        let e = GenError::InvalidGuidanceScale(0.5);
        let s = e.to_string();
        assert!(s.contains("0.5"), "got: {s}");
    }

    #[test]
    fn error_display_invalid_lora_rank() {
        let e = GenError::InvalidLoraRank(0);
        let s = e.to_string();
        assert!(s.contains('0'), "got: {s}");
    }

    #[test]
    fn error_display_invalid_lora_alpha() {
        let e = GenError::InvalidLoraAlpha(-1.0);
        let s = e.to_string();
        assert!(s.contains("-1"), "got: {s}");
    }

    #[test]
    fn error_display_invalid_codebook_size() {
        let e = GenError::InvalidCodebookSize(3);
        let s = e.to_string();
        assert!(s.contains('3'), "got: {s}");
        assert!(s.contains("power of two"), "got: {s}");
    }

    #[test]
    fn error_display_non_finite_commitment_loss() {
        let e = GenError::NonFiniteCommitmentLoss(f32::NAN);
        let s = e.to_string();
        assert!(s.contains("NaN") || s.contains("commitment"), "got: {s}");
    }

    #[test]
    fn error_display_weight_shape_mismatch() {
        let e = GenError::WeightShapeMismatch {
            weight: vec![64, 128],
            input: vec![32, 64],
        };
        let s = e.to_string();
        assert!(s.contains("[64, 128]"), "got: {s}");
        assert!(s.contains("[32, 64]"), "got: {s}");
    }

    #[test]
    fn error_display_unsupported_sm_version() {
        let e = GenError::UnsupportedSmVersion(70);
        let s = e.to_string();
        assert!(s.contains("70"), "got: {s}");
        assert!(s.contains("7.5"), "got: {s}");
    }

    #[test]
    fn error_display_unsupported_dpm_order() {
        let e = GenError::UnsupportedDpmOrder(4);
        let s = e.to_string();
        assert!(s.contains('4'), "got: {s}");
    }

    #[test]
    fn error_display_invalid_flow_time() {
        let e = GenError::InvalidFlowTime(1.5);
        let s = e.to_string();
        assert!(s.contains("1.5") || s.contains("1"), "got: {s}");
    }

    #[test]
    fn error_display_internal() {
        let e = GenError::Internal("invariant violated".into());
        let s = e.to_string();
        assert!(s.contains("invariant violated"), "got: {s}");
    }

    #[test]
    fn error_clone_and_eq() {
        let a = GenError::InvalidLoraRank(0);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(GenError::Internal("test".into()));
        assert!(e.to_string().contains("test"));
    }
}
