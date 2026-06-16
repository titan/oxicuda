//! Least-squares solvers.
//!
//! - [`mod@levenberg_marquardt`] — nonlinear least-squares via the
//!   Levenberg-Marquardt PECE algorithm with analytic Jacobian.
//! - [`levenberg_marquardt_numerical`] — same algorithm using a
//!   forward-difference numerical Jacobian.

pub mod levenberg_marquardt;
pub use levenberg_marquardt::{
    LmConfig, LmResult, levenberg_marquardt, levenberg_marquardt_numerical,
};
