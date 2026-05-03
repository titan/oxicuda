//! DARTS — Differentiable Architecture Search.
//!
//! - [`cell`] — `DartsCell`: normal/reduction cells with edge MixedOps.
//! - [`network`] — `DartsNetwork`: stacked cells → global avg pool → classifier.
//! - [`bilevel`] — `BilevelOptimizer`: inner weight SGD + outer arch Adam.
//! - [`mod@derive`] — `derive_discrete_cell`, `derive_network`.

pub mod bilevel;
pub mod cell;
pub mod derive;
pub mod network;

pub use bilevel::{BilevelConfig, BilevelOptimizer};
pub use cell::DartsCell;
pub use derive::{DiscretizedCell, DiscretizedNetwork, derive_discrete_cell, derive_network};
pub use network::DartsNetwork;
