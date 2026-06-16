//! Differentiable feature selection for tabular data.
//!
//! Currently provides STG: stochastic gates (Yamada et al., 2020) with an
//! `L0`-surrogate regulariser and learned per-feature importances.

pub mod stg;

pub use stg::{StgConfig, StgModel};
