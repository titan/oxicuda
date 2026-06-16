//! Utility functions for the OxiCUDA training engine.
//!
//! This module provides standalone free functions that operate on raw `f32`
//! slices, complementing the higher-level abstractions in the rest of the
//! crate.

/// Standalone gradient-clipping utilities for raw `f32` slices.
pub mod grad_clip;

pub use grad_clip::{adaptive_grad_clip, clip_grad_norm, clip_grad_value, global_grad_norm};
