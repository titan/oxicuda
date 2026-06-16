//! Trend detection and estimation for `oxicuda-stats`.
//!
//! Currently provides the Mann-Kendall family of nonparametric trend tests:
//!
//! - [`mod@mann_kendall`] — the classical Mann-Kendall trend test (Mann 1945, Kendall
//!   1975) with continuity-corrected normal approximation and tie correction.
//! - [`sens_slope`] — Sen's robust slope estimator (median pairwise slope).
//! - [`seasonal_mann_kendall`] — the seasonal Mann-Kendall test (Hirsch 1982).

pub mod mann_kendall;

pub use mann_kendall::{
    MannKendallResult, TrendDirection, mann_kendall, seasonal_mann_kendall, sens_slope,
};
