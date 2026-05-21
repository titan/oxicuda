//! Data primitives for survival analysis: observations, datasets, and risk sets.

pub mod dataset;
pub mod observation;
pub mod risk_set;
pub mod truncation;

pub use dataset::Dataset;
pub use observation::Observation;
pub use risk_set::RiskSet;
pub use truncation::{
    IntervalObs, RightTruncatedObs, TruncatedKmResult, TruncatedObs, TurnbullResult,
    conditional_survival, effective_sample_size, to_counting_process, truncated_km, turnbull_em,
    validate_truncated,
};
