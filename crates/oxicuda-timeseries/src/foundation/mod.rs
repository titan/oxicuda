//! Foundation time-series models: universal / pretrained-style forecasters.
//!
//! * [`moirai`] — Moirai any-variate masked-encoder forecaster (Woo et al. 2024).
//! * [`chronos`] — Chronos quantisation-tokenised LM forecaster (Ansari et al. 2024).
//! * [`adapter`] — checkpoint export / import interface ([`FoundationAdapter`],
//!   [`WeightStore`]) for loading pretrained weights into a forecaster.

pub mod adapter;
pub mod chronos;
pub mod moirai;

pub use adapter::{FoundationAdapter, WeightStore};
pub use chronos::{ChronosConfig, ChronosForecast, ChronosPredictor};
pub use moirai::{MoiraiConfig, MoiraiForecast, MoiraiForecaster};
