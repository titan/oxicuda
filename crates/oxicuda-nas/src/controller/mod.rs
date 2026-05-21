//! Reinforcement-learning architecture controllers.
//!
//! - [`enas`] — the ENAS LSTM controller (Pham et al., 2018): autoregressive
//!   categorical sampling of an architecture, trained by REINFORCE with an EMA
//!   reward baseline and full back-propagation-through-time.

pub mod enas;

pub use enas::{EnasConfig, EnasController};
