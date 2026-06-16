//! Constrained convex optimisation methods (Frank-Wolfe, conditional-gradient
//! sliding, projected variants).

pub mod cond_grad_sliding;
pub mod frank_wolfe;

pub use cond_grad_sliding::{CgsConfig, CgsResult, conditional_gradient_sliding};
pub use frank_wolfe::{FrankWolfeConfig, FwResult, frank_wolfe, l1_ball_lmo, simplex_lmo};
