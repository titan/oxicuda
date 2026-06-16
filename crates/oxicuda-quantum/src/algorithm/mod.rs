//! Canonical quantum algorithms built on the state-vector simulator.
//!
//! This module collects textbook oracle-based and communication algorithms that
//! showcase quantum advantage on top of the primitives in [`crate::statevec`]
//! and [`crate::gates`]:
//!
//! * [`mod@grover`] — Grover's unstructured search (quadratic query speed-up).
//! * [`mod@deutsch_jozsa`] — single-query constant-vs-balanced decision.
//! * [`mod@bernstein_vazirani`] — single-query hidden-string recovery.
//! * [`mod@teleportation`] — single-qubit state transfer via a Bell pair + 2 cbits.
//! * [`mod@superdense`] — two classical bits sent through one qubit + a Bell pair.
//! * [`mod@simon`] — exponential-speed-up hidden-period recovery (Shor precursor).
//! * [`mod@shor`] — order-finding subroutine + classical factoring reduction.
//! * [`mod@error_correction`] — GHZ/W state prep + 3-qubit bit/phase-flip codes.

pub mod bernstein_vazirani;
pub mod deutsch_jozsa;
pub mod error_correction;
pub mod grover;
pub mod shor;
pub mod simon;
pub mod superdense;
pub mod swap_test;
pub mod teleportation;

pub use bernstein_vazirani::{BernsteinVaziraniResult, bernstein_vazirani};
pub use deutsch_jozsa::{DeutschJozsaResult, FunctionKind, deutsch_jozsa};
pub use error_correction::{
    bit_flip_correct, bit_flip_encode, phase_flip_correct, phase_flip_encode, prepare_ghz,
    prepare_w,
};
pub use grover::{GroverResult, grover_search, marked_probability, optimal_iterations};
pub use shor::{
    OrderFindingResult, ShorResult, classical_order, continued_fraction_convergents,
    factor_from_order, gcd, mod_exp, order_finding, shor_factor,
};
pub use simon::{SimonResult, recover_period_from_constraints, simon};
pub use superdense::{SuperdenseResult, superdense_decode};
pub use swap_test::{hadamard_test, swap_test};
pub use teleportation::{TeleportResult, teleport};
