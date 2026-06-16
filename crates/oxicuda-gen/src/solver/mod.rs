//! Solver module for fast generative model inference.
//!
//! Provides DPM-Solver++ 2M (multi-step), UniPC predictor-corrector, PNDM/PLMS
//! linear-multistep, and Conditional Flow Matching (CFM) Euler integrators for
//! deterministic, high-quality sampling.

pub mod conditional_flow_matching;
pub mod dpm_solver_pp;
pub mod pndm;
pub mod unipc;

pub use conditional_flow_matching::{CfmConfig, ConditionalFlowMatching};
pub use dpm_solver_pp::{DpmAlgorithm, DpmSolverPp, DpmSolverPpConfig};
pub use pndm::{PndmConfig, PndmSolver};
pub use unipc::{UniPc, UniPcConfig, UniPcOrder};
