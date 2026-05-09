//! Shared architecture feature extraction utilities for predictors.
//!
//! An architecture is described by:
//! - `ops`: a vector of [`OpKind`] (one per edge / cell location)
//! - `in_channels` / `out_channels`: layer width
//! - `spatial`: `(H, W)` of the activations the op consumes
//!
//! [`ArchFeatures`] packs these into a fixed-length numerical vector that
//! down-stream regression models (latency, accuracy) can consume directly.

use crate::error::{NasError, NasResult};
use crate::ops::OpKind;

/// One layer of an architecture: its op kind, in/out channels, and spatial size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerSpec {
    /// Operation kind.
    pub op: OpKind,
    /// Input-channel count.
    pub in_channels: usize,
    /// Output-channel count.
    pub out_channels: usize,
    /// Input spatial height.
    pub h: usize,
    /// Input spatial width.
    pub w: usize,
}

impl LayerSpec {
    /// Convenience constructor.
    #[must_use]
    pub fn new(op: OpKind, in_channels: usize, out_channels: usize, h: usize, w: usize) -> Self {
        Self {
            op,
            in_channels,
            out_channels,
            h,
            w,
        }
    }
}

/// Numerical feature vector for an architecture.
///
/// The layout is: for each layer, append the one-hot encoding of `OpKind`
/// (length 8) plus four scalar features `[in_ch, out_ch, h, w]`. The final
/// feature vector has length `n_layers · (8 + 4) = 12·n_layers`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchFeatures {
    /// Flat feature vector.
    pub data: Vec<f32>,
    /// Number of layers in the source architecture.
    pub n_layers: usize,
}

impl ArchFeatures {
    /// Per-layer feature dimension (8 op-kind + 4 width/spatial).
    pub const PER_LAYER_DIM: usize = 8 + 4;

    /// Total feature dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Build a feature vector from a layer list.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] when `layers.is_empty()`.
    /// - [`NasError::DimensionMismatch`] if any layer has zero spatial dim or channels.
    pub fn from_layers(layers: &[LayerSpec]) -> NasResult<Self> {
        if layers.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let mut data = Vec::with_capacity(layers.len() * Self::PER_LAYER_DIM);
        for layer in layers {
            if layer.in_channels == 0 || layer.out_channels == 0 || layer.h == 0 || layer.w == 0 {
                return Err(NasError::DimensionMismatch {
                    expected: 1,
                    got: 0,
                });
            }
            // One-hot OpKind
            for op in OpKind::all() {
                data.push(if *op == layer.op { 1.0 } else { 0.0 });
            }
            // Width/spatial scalars (kept in raw counts; downstream models can normalise).
            data.push(layer.in_channels as f32);
            data.push(layer.out_channels as f32);
            data.push(layer.h as f32);
            data.push(layer.w as f32);
        }
        Ok(Self {
            data,
            n_layers: layers.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_layers() -> Vec<LayerSpec> {
        vec![
            LayerSpec::new(OpKind::SepConv3x3, 3, 16, 32, 32),
            LayerSpec::new(OpKind::SepConv5x5, 16, 16, 32, 32),
            LayerSpec::new(OpKind::AvgPool3x3, 16, 16, 32, 32),
        ]
    }

    #[test]
    fn arch_features_dim_matches_layers() {
        let layers = sample_layers();
        let f = ArchFeatures::from_layers(&layers).unwrap();
        assert_eq!(f.n_layers, 3);
        assert_eq!(f.dim(), 3 * ArchFeatures::PER_LAYER_DIM);
    }

    #[test]
    fn arch_features_one_hot_op() {
        let layers = vec![LayerSpec::new(OpKind::Identity, 4, 4, 8, 8)];
        let f = ArchFeatures::from_layers(&layers).unwrap();
        // Identity is index 1 in OpKind::all().
        assert!((f.data[1] - 1.0).abs() < 1e-6);
        // Other op-kind slots zero.
        for (i, &v) in f.data.iter().enumerate().take(OpKind::n_ops()) {
            if i == 1 {
                continue;
            }
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn arch_features_rejects_empty() {
        let r = ArchFeatures::from_layers(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn arch_features_rejects_zero_spatial() {
        let layers = vec![LayerSpec::new(OpKind::SepConv3x3, 4, 4, 0, 8)];
        let r = ArchFeatures::from_layers(&layers);
        assert!(r.is_err());
    }

    #[test]
    fn arch_features_rejects_zero_channels() {
        let layers = vec![LayerSpec::new(OpKind::SepConv3x3, 0, 4, 8, 8)];
        let r = ArchFeatures::from_layers(&layers);
        assert!(r.is_err());
    }
}
