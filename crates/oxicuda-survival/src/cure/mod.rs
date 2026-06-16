//! Cure models — mixture cure and related methods.

/// Mixture cure model (Berkson-Gage 1952, Farewell 1982).
pub mod mixture_cure;

pub use mixture_cure::{CureModelConfig, CureModelFit, cure_predict_survival, mixture_cure_fit};
