//! Foundation time-series models: universal / pretrained-style forecasters.
//!
//! * [`moirai`] — Moirai any-variate masked-encoder forecaster (Woo et al. 2024).
//! * [`chronos`] — Chronos quantisation-tokenised LM forecaster (Ansari et al. 2024).

pub mod chronos;
pub mod moirai;

pub use chronos::{ChronosConfig, ChronosForecast, ChronosPredictor};
pub use moirai::{MoiraiConfig, MoiraiForecast, MoiraiForecaster};
