//! Action-space abstractions.
//!
//! * [`crate::spaces::Discrete`] — single categorical action space.
//! * [`crate::spaces::MultiDiscrete`] — vector of independent discrete
//!   sub-actions with factorised log-probability / entropy.
//! * [`crate::spaces::TupleSpace`] — ordered tuple of sub-spaces.
//! * [`crate::spaces::Space`] — the common samplable-space trait.

pub mod multi_discrete;

pub use multi_discrete::{Discrete, MultiDiscrete, Space, TupleSpace};
