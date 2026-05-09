//! Latency surrogate predictors for hardware-aware NAS.
//!
//! Two complementary models:
//! - [`LatencyLut`] — measurement lookup table indexed by `(OpKind, in_ch, out_ch, h, w)`.
//!   Returns calibrated latency in seconds. Use for known device-specific
//!   benchmarks where the search space is small enough to memoise.
//! - [`LatencyMlp`] — small two-layer ReLU MLP that consumes
//!   [`ArchFeatures`] and predicts a scalar.
//!   Use for generalising across unseen `(op, shape)` combinations from a
//!   profiled training set.
//!
//! Both models can be calibrated against measured data with
//! [`LatencyLut::insert`] / [`LatencyMlp::fit`], and queried with `predict`.

use std::collections::HashMap;

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::OpKind;
use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};

/// Lookup-table latency model.
#[derive(Debug, Default, Clone)]
pub struct LatencyLut {
    table: HashMap<LatencyKey, f32>,
    /// Default latency returned for unknown layers.
    pub default_latency: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LatencyKey {
    op: OpKind,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
}

impl LatencyLut {
    /// Create an empty LUT with default latency `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a measured latency for a `(op, cin, cout, h, w)` configuration.
    pub fn insert(&mut self, layer: &LayerSpec, latency_seconds: f32) {
        self.table.insert(
            LatencyKey {
                op: layer.op,
                cin: layer.in_channels,
                cout: layer.out_channels,
                h: layer.h,
                w: layer.w,
            },
            latency_seconds,
        );
    }

    /// Look up a single layer's latency. Falls back to `default_latency` if absent.
    #[must_use]
    pub fn lookup(&self, layer: &LayerSpec) -> f32 {
        let k = LatencyKey {
            op: layer.op,
            cin: layer.in_channels,
            cout: layer.out_channels,
            h: layer.h,
            w: layer.w,
        };
        self.table.get(&k).copied().unwrap_or(self.default_latency)
    }

    /// Sum the latencies of an architecture's layers.
    ///
    /// # Errors
    /// [`NasError::EmptySearchSpace`] when `layers.is_empty()`.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if layers.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let mut total = 0.0_f32;
        for layer in layers {
            total += self.lookup(layer);
        }
        Ok(total)
    }

    /// Number of measured `(op, shape)` entries in the table.
    #[must_use]
    pub fn n_entries(&self) -> usize {
        self.table.len()
    }
}

/// Small MLP latency surrogate.
#[derive(Debug, Clone)]
pub struct LatencyMlp {
    /// Hidden layer weights `[hidden_dim × in_dim]` (row-major).
    pub w1: Vec<f32>,
    /// Hidden layer bias `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Output weights `[hidden_dim]`.
    pub w2: Vec<f32>,
    /// Output bias scalar.
    pub b2: f32,
    /// Input dimension (must match `ArchFeatures::dim()` of fitted samples).
    pub in_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// True once [`LatencyMlp::fit`] has run successfully.
    pub fitted: bool,
}

impl LatencyMlp {
    /// Create an unfitted MLP with Kaiming-initialised hidden layer.
    #[must_use]
    pub fn new(in_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        rng.fill_normal(&mut w1);
        for v in w1.iter_mut() {
            *v *= scale;
        }
        let b1 = vec![0.0_f32; hidden_dim];
        let mut w2 = vec![0.0_f32; hidden_dim];
        rng.fill_normal(&mut w2);
        for v in w2.iter_mut() {
            *v *= (2.0_f32 / hidden_dim as f32).sqrt();
        }
        Self {
            w1,
            b1,
            w2,
            b2: 0.0,
            in_dim,
            hidden_dim,
            fitted: false,
        }
    }

    /// One forward pass: returns scalar prediction.
    fn forward(&self, x: &[f32]) -> NasResult<f32> {
        if x.len() != self.in_dim {
            return Err(NasError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let mut h = vec![0.0_f32; self.hidden_dim];
        for ((hj, b), row) in h
            .iter_mut()
            .zip(self.b1.iter())
            .zip(self.w1.chunks(self.in_dim))
        {
            let mut acc = *b;
            for (w, &xi) in row.iter().zip(x.iter()) {
                acc += w * xi;
            }
            *hj = acc.max(0.0); // ReLU
        }
        let mut y = self.b2;
        for (wi, &hi) in self.w2.iter().zip(h.iter()) {
            y += wi * hi;
        }
        Ok(y)
    }

    /// Predict the latency of an architecture.
    ///
    /// # Errors
    /// - [`NasError::LatencyModelNotFitted`] if `fit` has not been called.
    /// - [`NasError::DimensionMismatch`] if features don't match `in_dim`.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if !self.fitted {
            return Err(NasError::LatencyModelNotFitted);
        }
        let f = ArchFeatures::from_layers(layers)?;
        self.forward(&f.data)
    }

    /// Fit the MLP via simple per-sample gradient descent on MSE.
    ///
    /// `samples` is a list of `(features, latency)` pairs; all features must
    /// have the same length equal to `self.in_dim`.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `samples.is_empty()`.
    /// - [`NasError::DimensionMismatch`] if any feature length disagrees with `in_dim`.
    pub fn fit(
        &mut self,
        samples: &[(Vec<f32>, f32)],
        epochs: usize,
        learning_rate: f32,
    ) -> NasResult<f32> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        for (x, _) in samples {
            if x.len() != self.in_dim {
                return Err(NasError::DimensionMismatch {
                    expected: self.in_dim,
                    got: x.len(),
                });
            }
        }
        let mut last_loss = f32::INFINITY;
        for _ in 0..epochs {
            let mut total_loss = 0.0_f64;
            for (x, target) in samples {
                let target = *target;
                let mut h_pre = vec![0.0_f32; self.hidden_dim];
                let mut h = vec![0.0_f32; self.hidden_dim];
                for ((((hp, hh), b), row), _) in h_pre
                    .iter_mut()
                    .zip(h.iter_mut())
                    .zip(self.b1.iter())
                    .zip(self.w1.chunks(self.in_dim))
                    .zip(0..self.hidden_dim)
                {
                    let mut acc = *b;
                    for (w, &xi) in row.iter().zip(x.iter()) {
                        acc += w * xi;
                    }
                    *hp = acc;
                    *hh = acc.max(0.0);
                }
                let mut y = self.b2;
                for (wi, &hi) in self.w2.iter().zip(h.iter()) {
                    y += wi * hi;
                }
                let err = y - target;
                total_loss += (err * err) as f64;
                // Gradients
                let dy = 2.0 * err;
                // Output bias and weights
                self.b2 -= learning_rate * dy;
                for (wi, &hi) in self.w2.iter_mut().zip(h.iter()) {
                    *wi -= learning_rate * dy * hi;
                }
                // Hidden gradients
                for (((hp, b), row), w2) in h_pre
                    .iter()
                    .zip(self.b1.iter_mut())
                    .zip(self.w1.chunks_mut(self.in_dim))
                    .zip(self.w2.iter())
                {
                    if *hp <= 0.0 {
                        continue;
                    }
                    let dh = dy * w2;
                    *b -= learning_rate * dh;
                    for (w, &xi) in row.iter_mut().zip(x.iter()) {
                        *w -= learning_rate * dh * xi;
                    }
                }
            }
            last_loss = (total_loss / samples.len() as f64) as f32;
            if !last_loss.is_finite() {
                return Err(NasError::Internal(
                    "non-finite loss during latency MLP fit".into(),
                ));
            }
        }
        self.fitted = true;
        Ok(last_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_returns_default_for_unknown() {
        let mut lut = LatencyLut::new();
        lut.default_latency = 1e-3;
        let layer = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        assert!((lut.lookup(&layer) - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn lut_returns_inserted_value() {
        let mut lut = LatencyLut::new();
        let layer = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        lut.insert(&layer, 0.005);
        assert!((lut.lookup(&layer) - 0.005).abs() < 1e-7);
    }

    #[test]
    fn lut_predict_sums() {
        let mut lut = LatencyLut::new();
        let l1 = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        let l2 = LayerSpec::new(OpKind::AvgPool3x3, 4, 4, 8, 8);
        lut.insert(&l1, 0.001);
        lut.insert(&l2, 0.0001);
        let total = lut.predict(&[l1, l2]).unwrap();
        assert!((total - 0.0011).abs() < 1e-6);
    }

    #[test]
    fn lut_predict_rejects_empty() {
        let lut = LatencyLut::new();
        let r = lut.predict(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn mlp_predict_before_fit_errors() {
        let mut rng = LcgRng::new(0);
        let mlp = LatencyMlp::new(ArchFeatures::PER_LAYER_DIM, 16, &mut rng);
        let layer = LayerSpec::new(OpKind::Identity, 4, 4, 8, 8);
        let r = mlp.predict(&[layer]);
        assert!(r.is_err());
    }

    #[test]
    fn mlp_fit_reduces_loss_on_constant_target() {
        let mut rng = LcgRng::new(0);
        let in_dim = ArchFeatures::PER_LAYER_DIM;
        let mut mlp = LatencyMlp::new(in_dim, 16, &mut rng);
        // Synthetic samples with target = 1.0
        let layer = LayerSpec::new(OpKind::Identity, 4, 4, 8, 8);
        let f = ArchFeatures::from_layers(&[layer]).unwrap();
        let samples = vec![(f.data.clone(), 1.0_f32); 16];
        let loss0 = mlp.fit(&samples, 1, 1e-4).unwrap();
        let loss1 = mlp.fit(&samples, 200, 1e-4).unwrap();
        assert!(
            loss1 <= loss0 + 1e-3,
            "loss did not decrease: {loss0} -> {loss1}"
        );
        assert!(mlp.fitted);
        let pred = mlp.predict(&[layer]).unwrap();
        assert!(pred.is_finite());
    }

    #[test]
    fn mlp_fit_rejects_empty() {
        let mut rng = LcgRng::new(0);
        let mut mlp = LatencyMlp::new(4, 4, &mut rng);
        assert!(mlp.fit(&[], 1, 1e-3).is_err());
    }

    #[test]
    fn mlp_fit_rejects_wrong_in_dim() {
        let mut rng = LcgRng::new(0);
        let mut mlp = LatencyMlp::new(4, 4, &mut rng);
        let r = mlp.fit(&[(vec![0.0_f32; 5], 1.0)], 1, 1e-3);
        assert!(r.is_err());
    }
}
