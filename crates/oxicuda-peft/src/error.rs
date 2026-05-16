use thiserror::Error;

/// All error variants produced by the `oxicuda-peft` crate.
#[derive(Debug, Error)]
pub enum PeftError {
    /// Input slice or tensor is empty.
    #[error("empty input")]
    EmptyInput,
    /// Dimension mismatch between tensors.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Requested rank exceeds the available dimension.
    #[error("rank {rank} exceeds dimension {dim}")]
    RankTooLarge { rank: usize, dim: usize },
    /// Target rank must be ≤ full rank.
    #[error("target rank {target_r} must not exceed rank {r}")]
    InvalidTargetRank { target_r: usize, r: usize },
    /// Layer index is out of range.
    #[error("layer index {idx} out of range (num_layers={num_layers})")]
    LayerOutOfRange { idx: usize, num_layers: usize },
    /// Block size must be positive.
    #[error("block size must be positive")]
    ZeroBlockSize,
    /// Adapter dimension must divide the feature dimension evenly.
    #[error("bottleneck dim {bot} does not divide in_dim {in_dim}")]
    UnalignedDimension { bot: usize, in_dim: usize },
    /// Density must be in (0, 1].
    #[error("density {density} must be in (0.0, 1.0]")]
    InvalidDensity { density: f32 },
    /// Weight vectors must have identical length.
    #[error("weight count {weights} does not match adapter count {adapters}")]
    WeightCountMismatch { weights: usize, adapters: usize },
    /// Internal arithmetic failure.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias for `Result<T, PeftError>`.
pub type PeftResult<T> = std::result::Result<T, PeftError>;
