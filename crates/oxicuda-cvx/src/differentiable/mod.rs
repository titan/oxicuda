//! Differentiable convex optimization layers.
//!
//! These layers expose the solution of a convex program as a differentiable
//! function of its parameters, enabling end-to-end training (cvxpylayers-style).
//!
//! * [`kkt_diff`] — OptNet (Amos & Kolter 2017): differentiate a QP solution
//!   `z*(Q, q, A, b, G, h)` by applying the implicit function theorem to the
//!   KKT system and solving the transposed KKT linear system on the backward
//!   pass.

pub mod kkt_diff;

pub use kkt_diff::{OptNetConfig, OptNetLayer, QpParamGrads, QpProblem, QpSolution};
