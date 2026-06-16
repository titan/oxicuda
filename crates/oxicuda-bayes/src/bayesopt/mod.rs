//! Bayesian Optimization with GP surrogate and EI/UCB/PI acquisition.
pub mod bo;
pub use bo::{
    AcquisitionFn, BayesOptConfig, BayesOptResult, GprKernelReexport as GprKernel,
    acquisition_value, bayesopt,
};
