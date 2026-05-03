//! Error types for `oxicuda-vision`.

use thiserror::Error;

/// Errors returned by `oxicuda-vision` operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum VisionError {
    /// Tensor dimension does not match the expected value.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Shape mismatch between two tensors.
    #[error("shape mismatch: lhs {lhs:?} vs rhs {rhs:?}")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    /// The input slice or tensor is empty.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// Image spatial dimensions are zero or inconsistent.
    #[error("invalid image size: height={height}, width={width}, channels={channels}")]
    InvalidImageSize {
        height: usize,
        width: usize,
        channels: usize,
    },

    /// Patch size is zero, negative, or does not divide the image dimension.
    #[error("invalid patch size {patch_size}: image size {img_size} is not divisible")]
    InvalidPatchSize { patch_size: usize, img_size: usize },

    /// Embedding dimension is zero or otherwise invalid.
    #[error("invalid embed dim: {0}")]
    InvalidEmbedDim(usize),

    /// Number of attention heads is zero or invalid.
    #[error("invalid number of heads: {0}")]
    InvalidNumHeads(usize),

    /// Head dimension does not divide the embedding dimension.
    #[error("head count {n_heads} does not divide embed dim {embed_dim}")]
    HeadDimMismatch { n_heads: usize, embed_dim: usize },

    /// Number of output classes is zero.
    #[error("invalid number of classes: {0}")]
    InvalidNumClasses(usize),

    /// Projection dimension is zero.
    #[error("invalid projection dim: {0}")]
    InvalidProjDim(usize),

    /// Contrastive loss temperature is non-positive.
    #[error("non-positive temperature: {0}")]
    NonPositiveTemperature(f32),

    /// RoI box coordinates are invalid (e.g., x1 >= x2).
    #[error("invalid RoI box [{x1}, {y1}, {x2}, {y2}]")]
    InvalidRoiBox { x1: f32, y1: f32, x2: f32, y2: f32 },

    /// Weight tensor has wrong shape.
    #[error("weight shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        name: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// NaN or infinity encountered in intermediate values.
    #[error("non-finite value encountered: {0}")]
    NonFinite(&'static str),

    /// Internal logic error (should not occur in correct usage).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias for `Result<T, VisionError>`.
pub type VisionResult<T> = Result<T, VisionError>;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = VisionError::DimensionMismatch {
            expected: 64,
            got: 128,
        };
        assert!(e.to_string().contains("64"));
        assert!(e.to_string().contains("128"));
    }

    #[test]
    fn error_display_shape_mismatch() {
        let e = VisionError::ShapeMismatch {
            lhs: vec![2, 4],
            rhs: vec![2, 8],
        };
        let s = e.to_string();
        assert!(s.contains("4"));
        assert!(s.contains("8"));
    }

    #[test]
    fn error_display_empty_input() {
        let e = VisionError::EmptyInput("image tensor");
        assert!(e.to_string().contains("image tensor"));
    }

    #[test]
    fn error_display_invalid_image_size() {
        let e = VisionError::InvalidImageSize {
            height: 0,
            width: 32,
            channels: 3,
        };
        let s = e.to_string();
        assert!(s.contains("0"));
        assert!(s.contains("32"));
    }

    #[test]
    fn error_display_invalid_patch_size() {
        let e = VisionError::InvalidPatchSize {
            patch_size: 5,
            img_size: 32,
        };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("32"));
    }

    #[test]
    fn error_display_head_dim_mismatch() {
        let e = VisionError::HeadDimMismatch {
            n_heads: 3,
            embed_dim: 64,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("64"));
    }

    #[test]
    fn error_display_non_positive_temperature() {
        let e = VisionError::NonPositiveTemperature(-0.1);
        assert!(e.to_string().contains("non-positive"));
    }

    #[test]
    fn error_display_invalid_roi_box() {
        let e = VisionError::InvalidRoiBox {
            x1: 5.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
        };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("3"));
    }

    #[test]
    fn error_display_weight_shape_mismatch() {
        let e = VisionError::WeightShapeMismatch {
            name: "patch_kernel",
            expected: vec![64, 3, 4, 4],
            got: vec![64, 3, 8, 8],
        };
        let s = e.to_string();
        assert!(s.contains("patch_kernel"));
        assert!(s.contains("64"));
    }

    #[test]
    fn error_display_non_finite() {
        let e = VisionError::NonFinite("attention logits");
        assert!(e.to_string().contains("attention logits"));
    }

    #[test]
    fn error_display_internal() {
        let e = VisionError::Internal("unexpected state".into());
        assert!(e.to_string().contains("unexpected state"));
    }

    #[test]
    fn error_clone_eq() {
        let a = VisionError::InvalidEmbedDim(0);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn result_alias_ok() {
        fn make_ok() -> VisionResult<u32> {
            Ok(42)
        }
        assert_eq!(make_ok().expect("ok result"), 42);
    }

    #[test]
    fn result_alias_err() {
        let r: VisionResult<u32> = Err(VisionError::EmptyInput("test"));
        assert!(r.is_err());
    }
}
