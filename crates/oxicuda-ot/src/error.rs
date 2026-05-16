use thiserror::Error;

/// All error variants produced by the `oxicuda-ot` crate.
#[derive(Debug, Error)]
pub enum OtError {
    /// Input slice or tensor is empty.
    #[error("empty input")]
    EmptyInput,
    /// Cost matrix shape does not match marginals.
    #[error("marginal mismatch: cost is {m}x{n}, marginals are {a_len},{b_len}")]
    MarginalMismatch {
        m: usize,
        n: usize,
        a_len: usize,
        b_len: usize,
    },
    /// Mass total of source and target marginals differ beyond tolerance.
    #[error("mass imbalance: sum_a={sum_a}, sum_b={sum_b}")]
    MassImbalance { sum_a: f32, sum_b: f32 },
    /// One or more weights are negative.
    #[error("negative weight encountered")]
    NegativeWeight,
    /// Algorithm failed to converge within `max_iter`.
    #[error("did not converge after {iter} iterations (tol={tol})")]
    NotConverged { iter: usize, tol: f32 },
    /// Probability simplex violation: a value outside [0, 1] or marginals do not sum to one.
    #[error("not a valid probability distribution")]
    NotProbability,
    /// Regularisation parameter must be strictly positive.
    #[error("invalid epsilon: {eps} (must be > 0)")]
    BadEpsilon { eps: f32 },
    /// KL relaxation parameter τ must be strictly positive.
    #[error("invalid tau: {tau} (must be > 0)")]
    BadTau { tau: f32 },
    /// Number of dimensions or projections must be positive.
    #[error("invalid dimension: {got} (must be > 0)")]
    BadDim { got: usize },
    /// Number of clusters or partitions must be positive.
    #[error("invalid cluster count: {got} (must be > 0)")]
    BadCount { got: usize },
    /// Sample buffers have incompatible lengths.
    #[error("incompatible sample lengths: {a} vs {b}")]
    IncompatibleLength { a: usize, b: usize },
    /// Internal arithmetic failure.
    #[error("internal error: {msg}")]
    Internal { msg: String },
}

/// Convenience alias for `Result<T, OtError>`.
pub type OtResult<T> = std::result::Result<T, OtError>;
