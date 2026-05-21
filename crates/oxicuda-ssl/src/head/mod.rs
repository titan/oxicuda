//! Projection and predictor MLP heads used by SSL pipelines.

pub mod linear_probe;
pub mod predictor;
pub mod projector;

pub use linear_probe::{
    FittedLinearProbe, LinearProbeConfig, LinearProbeResult, linear_probe_eval, linear_probe_fit,
    linear_probe_predict,
};
