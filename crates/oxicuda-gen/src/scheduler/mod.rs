//! Diffusion schedulers module.
//!
//! Provides DDPM, DDIM, DPM-Solver++, Flow Matching, EDM, and Consistency
//! Model schedulers for generative diffusion model inference.

pub mod beta_schedule;
pub mod consistency;
pub mod ddim;
pub mod ddpm;
pub mod dpm_solver;
pub mod edm;
pub mod flow_matching;
pub mod rectified_flow;
pub mod stochastic_interpolant;
pub mod v_prediction;

pub use beta_schedule::{BetaSchedule, BetaScheduleType};
pub use consistency::{ConsistencyConfig, ConsistencyScheduler};
pub use ddim::DdimScheduler;
pub use ddpm::DdpmScheduler;
pub use dpm_solver::{DpmOrder, DpmSolverScheduler};
pub use edm::{EdmConfig, EdmScheduler};
pub use flow_matching::{FlowMatchingPath, FlowMatchingScheduler};
pub use rectified_flow::{RectifiedFlow, RectifiedFlowConfig};
pub use stochastic_interpolant::{InterpolantConfig, InterpolantKind, StochasticInterpolant};
pub use v_prediction::{VPrediction, VPredictionConfig};
