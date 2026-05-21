//! Hypothesis tests for survival data: log-rank, stratified log-rank, Peto, Gehan,
//! and power/sample-size calculations for survival trials.

pub mod gehan_breslow;
pub mod log_rank;
pub mod peto_peto;
pub mod ph_lr_test;
pub mod power_sample_size;
pub mod stratified_log_rank;

pub use gehan_breslow::gehan_breslow_test;
pub use log_rank::{LogRankResult, log_rank_test};
pub use peto_peto::peto_peto_test;
pub use ph_lr_test::{PhLrTestResult, PhWaldResult, ph_lr_test, ph_score_test, ph_wald_test};
pub use power_sample_size::{
    FreedmanConfig, FreedmanResult, PowerFromEventsConfig, SchoenefeldConfig, SchoenefeldResult,
    expected_events, freedman_sample_size, power_from_events, schoenfeld_sample_size,
};
pub use stratified_log_rank::stratified_log_rank_test;
