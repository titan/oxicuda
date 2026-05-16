//! Compressed-sensing recovery metrics.

pub mod metrics;

pub use metrics::{
    mean_squared_error, normalized_mse, psnr, recovery_error, snr, sparsity, support_recovery_rate,
};
