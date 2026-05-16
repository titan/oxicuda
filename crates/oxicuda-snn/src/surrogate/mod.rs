//! Surrogate gradients for spike functions.
//!
//! Each surrogate exposes a `*_grad(v, v_th, alpha, grad_out)` function that
//! writes the derivative of a smooth approximation to the Heaviside step at
//! `v_th`. Usage: replace the non-differentiable `dS/dv = δ(v−v_th)` term in
//! BPTT/STBP/SLAYER with `α · σ'(α(v−v_th))` (or the analogous shape).

/// Smooth `arctan` surrogate.
pub mod atan;
/// Fast-sigmoid surrogate `α / (1 + |α(v−v_th)|)²`.
pub mod fast_sigmoid;
/// Logistic-sigmoid surrogate `α · σ · (1−σ)`.
pub mod sigmoid;
/// Zenke-Ganguli "SuperSpike" surrogate.
pub mod super_spike;
/// Triangular surrogate.
pub mod triangle;
