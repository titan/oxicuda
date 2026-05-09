//! DeepSVDD (Ruff et al. 2018) — One-Class Deep Learning anomaly detector.
//!
//! Learns a hypersphere in feature space that encloses normal data.
//!
//! * **Center `c`**: computed once as the mean of encoder outputs on training
//!   data; never updated during training (hypersphere-collapse prevention).
//! * **Loss**: `(1/N) * Σ_i ||φ(x_i; W) - c||²`
//! * **Score**: `||φ(x; W) - c||²`

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Xavier weight initialisation ────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32();
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── SvddEncoder ─────────────────────────────────────────────────────────────

/// Lightweight MLP encoder for DeepSVDD.
///
/// All intermediate layers use ReLU activation; the final (representation)
/// layer is **linear with no bias** — required to prevent hypersphere collapse.
pub struct SvddEncoder {
    /// Per-layer `(weight [out*in], bias [out])`.
    /// The last layer bias is always `vec![0.0; rep_dim]` and never updated.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Layer dimensions including input: `[input, h1, h2, ..., rep_dim]`.
    dims: Vec<usize>,
    /// Representation (output) dimensionality.
    pub rep_dim: usize,
}

impl SvddEncoder {
    /// Build encoder from dimension spec and initialise weights (Xavier).
    pub fn new(dims: &[usize], rng: &mut LcgRng) -> AnomalyResult<Self> {
        if dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "need at least [input_dim, rep_dim]".into(),
            });
        }
        for &d in dims {
            if d == 0 {
                return Err(AnomalyError::InvalidLayerDims {
                    msg: "zero dimension in layer spec".into(),
                });
            }
        }
        let n_layers = dims.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let fan_in = dims[l];
            let fan_out = dims[l + 1];
            let w = xavier_init(fan_in, fan_out, rng);
            // Last layer: no bias (zero, never updated)
            let b = vec![0.0_f32; fan_out];
            layers.push((w, b));
        }
        let rep_dim = *dims.last().unwrap_or(&1);
        Ok(Self {
            layers,
            dims: dims.to_vec(),
            rep_dim,
        })
    }

    /// Forward pass for a single sample `x` of length `dims[0]`.
    pub fn forward(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        let input_dim = self.dims[0];
        if x.len() != input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: input_dim,
                got: x.len(),
            });
        }
        let n_layers = self.layers.len();
        let mut activation: Vec<f32> = x.to_vec();
        for (layer_idx, (w, b)) in self.layers.iter().enumerate() {
            let fan_in = self.dims[layer_idx];
            let fan_out = self.dims[layer_idx + 1];
            let mut out = vec![0.0_f32; fan_out];
            for o in 0..fan_out {
                let mut acc = b[o];
                for i in 0..fan_in {
                    acc += w[o * fan_in + i] * activation[i];
                }
                // ReLU on all but the last layer
                out[o] = if layer_idx < n_layers - 1 {
                    acc.max(0.0)
                } else {
                    acc
                };
            }
            activation = out;
        }
        Ok(activation)
    }

    /// Batch forward: `x` is `[n * input_dim]`; returns `[n * rep_dim]`.
    pub fn forward_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        let input_dim = self.dims[0];
        if x.len() != n * input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * input_dim,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(n * self.rep_dim);
        for i in 0..n {
            let sample = &x[i * input_dim..(i + 1) * input_dim];
            let rep = self.forward(sample)?;
            out.extend_from_slice(&rep);
        }
        Ok(out)
    }
}

// ─── DeepSvdd ────────────────────────────────────────────────────────────────

/// DeepSVDD anomaly detector.
pub struct DeepSvdd {
    encoder: SvddEncoder,
    center: Option<Vec<f32>>,
    pub input_dim: usize,
    pub rep_dim: usize,
}

impl DeepSvdd {
    /// Create a new DeepSVDD from a layer spec `[input, h1, ..., rep_dim]`.
    pub fn new(dims: &[usize], rng: &mut LcgRng) -> AnomalyResult<Self> {
        if dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "dims must have at least 2 entries".into(),
            });
        }
        let input_dim = dims[0];
        let rep_dim = *dims.last().unwrap_or(&1);
        let encoder = SvddEncoder::new(dims, rng)?;
        Ok(Self {
            encoder,
            center: None,
            input_dim,
            rep_dim,
        })
    }

    /// Initialise the hypersphere center as the mean of encoder outputs.
    ///
    /// If any center component is within `[-0.01, 0.01]` it is shifted to
    /// `0.01` to avoid the degenerate origin collapse.
    pub fn fit(&mut self, x: &[f32], n_samples: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        let reps = self.encoder.forward_batch(x, n_samples)?;
        let mut center = vec![0.0_f32; self.rep_dim];
        for i in 0..n_samples {
            for j in 0..self.rep_dim {
                center[j] += reps[i * self.rep_dim + j];
            }
        }
        let inv_n = 1.0 / n_samples as f32;
        for v in &mut center {
            *v *= inv_n;
            if v.abs() < 0.01 {
                *v = 0.01;
            }
        }
        self.center = Some(center);
        Ok(())
    }

    /// Compute anomaly score `||φ(x) - c||²` for a single sample.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let c = self.center.as_ref().ok_or(AnomalyError::NotFitted)?;
        let rep = self.encoder.forward(x)?;
        let score = rep
            .iter()
            .zip(c.iter())
            .map(|(r, ci)| (r - ci).powi(2))
            .sum();
        Ok(score)
    }

    /// Batch scoring; `x` is `[n * input_dim]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        let c = self.center.as_ref().ok_or(AnomalyError::NotFitted)?;
        let reps = self.encoder.forward_batch(x, n)?;
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let s: f32 = (0..self.rep_dim)
                .map(|j| (reps[i * self.rep_dim + j] - c[j]).powi(2))
                .sum();
            scores.push(s);
        }
        Ok(scores)
    }

    /// Mean SVDD loss `(1/N) Σ_i ||φ(x_i) - c||²` over a batch.
    pub fn svdd_loss(&self, x: &[f32], n: usize) -> AnomalyResult<f32> {
        let scores = self.score_batch(x, n)?;
        let total: f32 = scores.iter().sum();
        Ok(total / n as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_forward_shape() {
        let mut rng = LcgRng::new(1);
        let enc = SvddEncoder::new(&[4, 8, 4], &mut rng).unwrap();
        let x = vec![1.0_f32; 4];
        let out = enc.forward(&x).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn deep_svdd_fit_score() {
        let mut rng = LcgRng::new(2);
        let mut svdd = DeepSvdd::new(&[4, 8, 4], &mut rng).unwrap();
        let x = vec![0.1_f32; 40]; // 10 samples
        svdd.fit(&x, 10).unwrap();
        let s = svdd.score(&[0.1_f32, 0.1, 0.1, 0.1]).unwrap();
        assert!(s.is_finite(), "score={s}");
    }

    #[test]
    fn deep_svdd_not_fitted_error() {
        let mut rng = LcgRng::new(3);
        let svdd = DeepSvdd::new(&[4, 4], &mut rng).unwrap();
        assert!(svdd.score(&[0.0_f32; 4]).is_err());
    }
}
