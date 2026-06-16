//! Nonparametric density estimation.
//!
//! Currently provides 1-D and 2-D **kernel density estimation** (Silverman,
//! 1986) with Gaussian / Epanechnikov kernels and Silverman / Scott bandwidth
//! selection.

pub mod kde;

pub use kde::{
    BandwidthRule, Kernel, KernelDensity, KernelDensity2d, scott_bandwidth, silverman_bandwidth,
};
