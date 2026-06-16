//! Operator-splitting methods for structured monotone inclusions.
//!
//! These routines minimize sums of (possibly nonsmooth) convex functions, or
//! more generally find a zero of a sum of maximal-monotone operators, by
//! activating each operator separately through its resolvent (proximal map) or
//! a forward (gradient) step.
//!
//! * [`three_operator`] — Davis-Yin three-operator splitting (2017) for
//!   `min f + g + h` with `f, g` proximable and `h` smooth.
//! * [`mod@tseng_fbf`] — Tseng's forward-backward-forward splitting (2000) for
//!   `0 ∈ A x + B x` with `B` monotone-Lipschitz but *not* necessarily
//!   cocoercive (e.g. skew-symmetric / saddle-point operators).

pub mod three_operator;
pub mod tseng_fbf;

pub use three_operator::{
    DavisYinConfig, DavisYinResult, DavisYinStatus, davis_yin_three_operator,
};
pub use tseng_fbf::{TsengConfig, TsengResult, TsengStatus, tseng_fbf};
