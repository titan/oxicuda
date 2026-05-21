//! Sensitivity-analysis primitives for causal estimands.
//!
//! Closed-form bounds on how strong an unobserved confounder would need to
//! be to nullify an observed effect estimate. Currently exposes:
//!
//! - [`e_value`] — VanderWeele & Ding (2017) E-value bound, applicable to
//!   risk ratios, odds ratios, hazard ratios and risk differences.
//! - [`rosenbaum_bounds`] — Rosenbaum (1987 / 2002) sensitivity bounds for
//!   matched-pair Wilcoxon signed-rank tests under bias Γ.
//! - [`manski_bounds`] — Manski (1990) / Manski-Pepper (2000) partial-
//!   identification ATE bounds under four assumptions.

pub mod cinelli_hazlett;
pub mod e_value;
pub mod manski_bounds;
pub mod rosenbaum_bounds;
#[cfg(test)]
mod rosenbaum_bounds_tests;

pub use cinelli_hazlett::{
    BenchmarkResult, CinelliHazlett, CinelliHazlettConfig, CinelliHazlettResult, OvbInput,
};
pub use e_value::{EValue, EValueConfig, EValueResult, EffectType};
pub use manski_bounds::{ManskiAssumption, ManskiBounds, ManskiConfig, ManskiResult};
pub use rosenbaum_bounds::{RosenbaumBounds, RosenbaumConfig, RosenbaumResult};
