//! Error types for `oxicuda-nerf`.

/// All errors that can be returned by `oxicuda-nerf` operations.
#[derive(Debug, thiserror::Error)]
pub enum NerfError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input")]
    EmptyInput,

    #[error("invalid frequency levels: {levels}")]
    InvalidFreqLevels { levels: usize },

    #[error("invalid hash grid config: {msg}")]
    InvalidHashConfig { msg: String },

    #[error("NaN encountered in {context}")]
    NanEncountered { context: String },

    #[error("invalid near/far bounds: near={near}, far={far}")]
    InvalidBounds { near: f32, far: f32 },

    #[error("invalid sample count: {n}")]
    InvalidSampleCount { n: usize },

    #[error("ray direction is zero or near-zero")]
    ZeroRayDirection,

    #[error("invalid camera intrinsics: {msg}")]
    InvalidCameraIntrinsics { msg: String },

    #[error("invalid grid resolution: {res}")]
    InvalidGridResolution { res: usize },

    #[error("hash level out of range: {level}")]
    HashLevelOutOfRange { level: usize },

    #[error("invalid feature dimension: {dim}")]
    InvalidFeatureDim { dim: usize },

    #[error("tensor decomp rank error: {msg}")]
    TensorDecompError { msg: String },

    #[error("volume render failed: {msg}")]
    VolumeRenderError { msg: String },

    #[error("invalid octree configuration: {msg}")]
    InvalidOctreeConfig { msg: String },

    #[error("invalid embedding configuration: {msg}")]
    InvalidEmbeddingConfig { msg: String },

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias.
pub type NerfResult<T> = Result<T, NerfError>;
