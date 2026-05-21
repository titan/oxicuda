//! Zero-cost proxies for predictor-free architecture ranking.
//!
//! - [`zero_cost`] — NASWOT logdet kernel + SNIP / GraSP / SynFlow saliencies,
//!   plus the [`ZeroCostProxy`] selector and [`rank_architectures`] helper.
//!
//! Unlike the surrogate predictors in [`crate::predictor`], these proxies need
//! no training data and no fitted model: they score an architecture directly
//! from forward-only (and, for SNIP / GraSP / SynFlow, single-backward) signals
//! gathered on an untrained network.

pub mod zero_cost;

pub use zero_cost::{
    NASWOT_RIDGE, ZeroCostProxy, grasp_score, naswot_score, rank_architectures, snip_score,
    synflow_score,
};
