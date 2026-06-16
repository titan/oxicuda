//! Online / continual meta-learning algorithms.
//!
//! Currently exposes [`Oml`] — Online-aware Meta-Learning (Javed & White, 2019),
//! a representation/prediction-network factorisation whose meta-objective is
//! aware of catastrophic forgetting under online (streaming) adaptation.

pub mod oml;

pub use oml::{Oml, OmlConfig, OmlLinear};
