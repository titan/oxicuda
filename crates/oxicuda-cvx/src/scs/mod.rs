//! Splitting Conic Solver (SCS).
//!
//! Operator-splitting (ADMM) solver for standard-form conic programs, following
//! O'Donoghue, Chu, Parikh & Boyd (2016), "Conic Optimization via Operator
//! Splitting and Homogeneous Self-Dual Embedding".

pub mod scs_solver;

pub use scs_solver::{Cone, ScsConfig, ScsResult, ScsStatus, scs_solve};
