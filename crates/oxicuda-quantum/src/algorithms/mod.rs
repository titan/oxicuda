//! Advanced quantum algorithms built on the state-vector simulator.
//!
//! Complementing the textbook oracle algorithms in [`crate::algorithm`], this
//! module collects modern matrix-arithmetic and dynamics primitives:
//!
//! * [`mod@qsvt`] — Quantum Signal Processing, the single-qubit core of Quantum
//!   Singular Value Transformation (Gilyén–Su–Low–Wiebe 2019).
//! * [`mod@quantum_walk`] — discrete-time coined quantum walk on a cycle, exhibiting
//!   ballistic spreading (Aharonov 1993 / Kempe 2003).
//! * [`mod@amplitude_estimation`] — Quantum Amplitude Estimation via QPE on the
//!   Grover amplitude operator (Brassard–Høyer–Mosca–Tapp 2002).
//! * [`mod@vqls`] — Variational Quantum Linear Solver for `A = Σ_l c_l P_l`
//!   (Bravo-Prieto et al. 2019).
//! * [`mod@iterative_qpe`] — single-ancilla iterative (Kitaev) phase estimation
//!   with feedback (Kitaev 1995 / Dobšíček 2007).

pub mod amplitude_estimation;
pub mod iterative_qpe;
pub mod qsvt;
pub mod quantum_walk;
pub mod vqls;

pub use amplitude_estimation::{AmplitudeEstimationResult, StatePreparation, amplitude_estimation};
pub use iterative_qpe::{IterativeQpeResult, iterative_phase_estimation};
pub use qsvt::{
    Mat2, chebyshev_qsp_angles, chebyshev_t, qsp_top_left, qsp_unitary, signal_operator,
};
pub use quantum_walk::{CoinInit, CoinedWalk, position_std_about};
pub use vqls::{
    HardwareEfficientAnsatz, LcuOperator, Pauli, PauliTerm, VqlsResult, VqlsSolver, fidelity,
};
