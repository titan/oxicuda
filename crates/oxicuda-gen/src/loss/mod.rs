//! Loss functions for generative models.
//!
//! This module provides DDPM denoising loss and related utilities
//! for training diffusion models following Ho et al. (2020).

pub mod ddpm_loss;

pub use ddpm_loss::{DdpmLoss, DdpmLossConfig, DdpmLossType};
