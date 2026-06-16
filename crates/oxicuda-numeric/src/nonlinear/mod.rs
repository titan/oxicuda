//! Nonlinear system solvers and unconstrained optimisers for `F : ℝⁿ → ℝⁿ`
//! and `φ : ℝⁿ → ℝ`.
//!
//! - [`mod@newton_krylov`] — Jacobian-free Newton-Krylov (Newton + matrix-free
//!   GMRES inner solve) for large or analytically-awkward systems.
//! - [`mod@broyden`] — Broyden's "good" quasi-Newton root finder with rank-1
//!   inverse-Jacobian updates.
//! - [`bfgs_minimize`] / [`bfgs_minimize_numerical`] — BFGS quasi-Newton
//!   unconstrained minimisation with Armijo backtracking.
//! - [`lbfgs_minimize`] / [`lbfgs_minimize_numerical`] — limited-memory BFGS
//!   (two-loop recursion) for large-scale smooth minimisation.
//! - [`conjugate_gradient_minimize`] — nonlinear conjugate gradient
//!   (Fletcher-Reeves / Polak-Ribière / Hestenes-Stiefel) with strong-Wolfe
//!   line search.
//! - [`mod@nelder_mead`] — derivative-free downhill-simplex direct search.

pub mod bfgs;
pub mod broyden;
pub mod conjugate_gradient;
pub mod lbfgs;
pub mod nelder_mead;
pub mod newton_krylov;

pub use bfgs::{BfgsConfig, BfgsResult, bfgs_minimize, bfgs_minimize_numerical};
pub use broyden::broyden;
pub use conjugate_gradient::{
    CgConfig, CgResult, CgVariant, conjugate_gradient_minimize,
    conjugate_gradient_minimize_numerical,
};
pub use lbfgs::{LbfgsConfig, LbfgsResult, lbfgs_minimize, lbfgs_minimize_numerical};
pub use nelder_mead::{NelderMeadConfig, NelderMeadResult, nelder_mead};
pub use newton_krylov::{NewtonKrylovConfig, newton_krylov};
