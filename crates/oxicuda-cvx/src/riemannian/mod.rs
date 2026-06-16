//! Riemannian optimisation on smooth matrix manifolds (Absil, Mahony & Sepulchre 2008).
//!
//! Provides Riemannian gradient descent with retraction-based updates and a
//! Riemannian Armijo line search for three canonical manifolds:
//!
//! - **Sphere** `Sⁿ⁻¹ = { x ∈ ℝⁿ : ‖x‖ = 1 }`,
//! - **Symmetric positive-definite** matrices `S⁺⁺(n)` with the affine-invariant metric,
//! - **Stiefel** `St(n, p) = { X ∈ ℝⁿˣᵖ : XᵀX = I_p }`.
//!
//! See [`riemannian_cvx`] for the algorithm and [`Manifold`] for the geometric
//! primitives (projection of the Euclidean gradient onto the tangent space and the
//! retraction).

pub mod riemannian_cvx;

pub use riemannian_cvx::{
    Manifold, RiemannianConfig, RiemannianResult, riemannian_gradient_descent,
};
