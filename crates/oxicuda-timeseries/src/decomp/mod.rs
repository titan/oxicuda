//! Time-series decomposition into trend and seasonal components.

pub mod mint_reconcile;
pub mod moving_avg;
pub mod series_decomp;
pub mod stl;
pub mod sts;

pub use mint_reconcile::{MintMethod, MintReconciler};
pub use moving_avg::MovingAvg;
pub use series_decomp::{DecompResult, SeriesDecomp};
pub use stl::{
    StlConfig, StlResult, stl_decompose, stl_naive_forecast, stl_seasonal_strength,
    stl_trend_strength,
};
pub use sts::{StsConfig, StsDecomposer};
