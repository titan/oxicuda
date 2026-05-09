/// Errors produced by ANN operations.
#[derive(Debug, thiserror::Error)]
pub enum AnnError {
    #[error("empty input")]
    EmptyInput,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid vector dimension: {dim}")]
    InvalidVectorDim { dim: usize },

    #[error("invalid k={k} for n={n} vectors")]
    InvalidK { k: usize, n: usize },

    #[error("invalid nprobe={nprobe} for nlist={nlist}")]
    InvalidNumProbes { nprobe: usize, nlist: usize },

    #[error("invalid number of subspaces m={m} for dim={dim}")]
    InvalidNumSubspaces { m: usize, dim: usize },

    #[error("index not fitted (call train() first)")]
    NotFitted,

    #[error("index is empty")]
    IndexEmpty,

    #[error("k-means did not converge after {iter} iterations")]
    KmeansDidNotConverge { iter: usize },

    #[error("LSH hash collision in internal structure")]
    LshHashCollision,

    #[error("invalid layer count: {n}")]
    InvalidLayerCount { n: usize },

    #[error("HNSW graph is not connected")]
    GraphNotConnected,

    #[error("id {id} out of range [0, {n})")]
    IdOutOfRange { id: usize, n: usize },

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type AnnResult<T> = std::result::Result<T, AnnError>;
