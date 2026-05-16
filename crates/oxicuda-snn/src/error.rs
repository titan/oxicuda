use thiserror::Error;

/// All error variants produced by the `oxicuda-snn` crate.
#[derive(Debug, Error)]
pub enum SnnError {
    /// Input slice or tensor is empty.
    #[error("empty input")]
    EmptyInput,
    /// Tensor shape does not match expected dimensions.
    #[error("bad shape: expected {expected}, got {got}")]
    BadShape { expected: usize, got: usize },
    /// Membrane time constant must be strictly positive.
    #[error("invalid time constant tau: {tau} (must be > 0)")]
    BadTau { tau: f32 },
    /// Spike threshold must be finite.
    #[error("invalid spike threshold: {v_th}")]
    BadThreshold { v_th: f32 },
    /// Time step dt must be strictly positive.
    #[error("invalid dt: {dt} (must be > 0)")]
    BadDt { dt: f32 },
    /// Two buffers have incompatible lengths.
    #[error("incompatible length: {a} vs {b}")]
    IncompatibleLength { a: usize, b: usize },
    /// Layer index is out of range.
    #[error("layer index {idx} out of range (num_layers={num_layers})")]
    LayerOutOfRange { idx: usize, num_layers: usize },
    /// Parameter value out of valid range.
    #[error("{name} out of range: {val}")]
    OutOfRange { name: String, val: f32 },
    /// Number of timesteps must be positive.
    #[error("invalid timesteps: {got} (must be > 0)")]
    BadTimesteps { got: usize },
    /// Bottleneck or hidden dimension must be positive.
    #[error("invalid dimension: {got} (must be > 0)")]
    BadDim { got: usize },
    /// Internal arithmetic failure.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias for `Result<T, SnnError>`.
pub type SnnResult<T> = std::result::Result<T, SnnError>;
