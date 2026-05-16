//! Hypothesis tests for survival data: log-rank, stratified log-rank, Peto, Gehan.

pub mod gehan_breslow;
pub mod log_rank;
pub mod peto_peto;
pub mod stratified_log_rank;

pub use gehan_breslow::gehan_breslow_test;
pub use log_rank::{LogRankResult, log_rank_test};
pub use peto_peto::peto_peto_test;
pub use stratified_log_rank::stratified_log_rank_test;
