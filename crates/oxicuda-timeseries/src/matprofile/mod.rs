//! Matrix Profile — STOMP-based all-pairs similarity join for time series.

pub mod stomp;

pub use stomp::{
    MatProfileConfig, MatProfileResult, matrix_profile, matrix_profile_ab, sliding_stats,
    znorm_distance,
};
