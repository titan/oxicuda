/// Errors produced by quantum simulation operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("empty input")]
    EmptyInput,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid qubit count: {n}")]
    InvalidQubitCount { n: usize },

    #[error("qubit index {index} out of range for {n_qubits} qubits")]
    QubitIndexOutOfRange { index: usize, n_qubits: usize },

    #[error("state not normalized: norm = {norm}")]
    NonNormalizedState { norm: f32 },

    #[error("Hamiltonian is not Hermitian")]
    NonHermitianHamiltonian,

    #[error("Kraus operators not complete: residual = {residual}")]
    KrausNotComplete { residual: f32 },

    #[error("invalid parameter '{name}'")]
    InvalidParameter { name: String },

    #[error("invalid Pauli operator '{op}'")]
    InvalidPauliOp { op: String },

    #[error("incompatible ansatz configuration")]
    IncompatibleAnsatz,

    #[error("optimization diverged after {iter} iterations")]
    OptimizationDiverged { iter: usize },

    #[error("model not fitted (call fit() first)")]
    NotFitted,

    #[error("measurement failed")]
    MeasurementFailed,

    #[error("internal error: {msg}")]
    Internal { msg: String },
}

pub type QuantumResult<T> = std::result::Result<T, QuantumError>;
