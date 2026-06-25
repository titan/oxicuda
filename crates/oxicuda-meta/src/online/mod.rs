//! Online / continual meta-learning algorithms.
//!
//! Exposes:
//!
//! * [`Oml`] — Online-aware Meta-Learning (Javed & White, 2019), a
//!   representation/prediction-network factorisation whose meta-objective is
//!   aware of catastrophic forgetting under online (streaming) adaptation;
//! * [`Anml`] — A Neuromodulated Meta-Learning algorithm (Beaulieu et al. 2020),
//!   which extends the OML factorisation with a learned neuromodulatory gate
//!   that multiplicatively controls selective plasticity of the representation.

pub mod anml;
pub mod oml;

pub use anml::{Anml, AnmlConfig, AnmlLinear};
pub use oml::{Oml, OmlConfig, OmlLinear};
