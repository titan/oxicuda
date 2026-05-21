//! DARTS — Differentiable Architecture Search.
//!
//! - [`cell`] — `DartsCell`: normal/reduction cells with edge MixedOps.
//! - [`network`] — `DartsNetwork`: stacked cells → global avg pool → classifier.
//! - [`bilevel`] — `BilevelOptimizer`: inner weight SGD + outer arch Adam.
//! - [`mod@derive`] — `derive_discrete_cell`, `derive_network`.
//! - [`pc_darts`] — `PcDarts`: partial-channel sampling + edge normalization.
//! - [`darts_plus`] — `DartsPlusState`: skip-collapse early-stopping (Liang 2019).

pub mod bilevel;
pub mod cell;
pub mod darts_plus;
pub mod derive;
pub mod network;
pub mod pc_darts;

pub use bilevel::{BilevelConfig, BilevelOptimizer};
pub use cell::DartsCell;
pub use darts_plus::{DartsPlusConfig, DartsPlusState};
pub use derive::{DiscretizedCell, DiscretizedNetwork, derive_discrete_cell, derive_network};
pub use network::DartsNetwork;
pub use pc_darts::{PcDarts, PcDartsConfig};
