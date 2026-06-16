//! Nonsmooth (semismooth) Newton methods.
//!
//! This module hosts Newton-type solvers for *nonsmooth* root-finding problems
//! `F(x) = 0`, where `F` is only locally Lipschitz and semismooth rather than
//! continuously differentiable.  The flagship routine is the generalized
//! (Clarke / B-subdifferential) Newton method of Qi & Sun (1993), applied to a
//! Linear Complementarity Problem reformulated through the Fischer-Burmeister
//! NCP function.
//!
//! See [`semismooth`] for the full development.

pub mod semismooth;

pub use semismooth::{
    SemismoothConfig, SemismoothNewtonResult, SemismoothStatus, fischer_burmeister,
    fischer_burmeister_gradient, lcp_residual, semismooth_newton_lcp,
};
