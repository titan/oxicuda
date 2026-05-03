//! Attention-weighted temporal statistics pooling.
//!
//! A learned attention mechanism assigns a scalar score to each time frame,
//! then the weighted mean and weighted standard deviation are concatenated
//! to form a fixed-dimensional utterance-level embedding.
//!
//! Architecture:
//! ```text
//! scores[t] = v^T · tanh(W · x[t] + b)    # [T] scalar scores
//! attn[t]   = softmax(scores)[t]
//! mean_out  = Σ_t attn[t] * x[t]          # [C]
//! std_out   = sqrt(Σ_t attn[t] * (x[t] - mean_out)^2)  # [C]
//! output    = concat(mean_out, std_out)    # [2C]
//! ```

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Numerically stable in-place softmax.
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

// ─── AttentivePool ───────────────────────────────────────────────────────────

/// Attention-weighted temporal statistics pooling layer.
///
/// Takes a `[T, C]` feature tensor and produces a `[2 * C]` utterance
/// embedding consisting of the attention-weighted mean and standard deviation
/// over the time axis.
#[derive(Debug, Clone)]
pub struct AttentivePool {
    /// Linear projection `[bottleneck, C]`.
    pub attention_w: Vec<f32>,
    /// Bias vector `[bottleneck]`.
    pub attention_b: Vec<f32>,
    /// Scoring vector `[bottleneck]`.
    pub attention_v: Vec<f32>,
    /// Input channel dimension.
    pub c: usize,
    /// Bottleneck dimension (`max(16, C / 4)`).
    pub bottleneck: usize,
}

impl AttentivePool {
    /// Construct an `AttentivePool` with randomly initialised weights.
    ///
    /// `bottleneck = max(16, c / 4)`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidEmbedDim`] when `c == 0`.
    pub fn new(c: usize, rng: &mut LcgRng) -> AudioResult<Self> {
        if c == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        let bottleneck = (c / 4).max(16);

        let w_scale = 1.0 / (c as f32).sqrt();
        let v_scale = 1.0 / (bottleneck as f32).sqrt();

        let mut attention_w = vec![0.0_f32; bottleneck * c];
        rng.fill_normal(&mut attention_w);
        for val in &mut attention_w {
            *val *= w_scale;
        }

        let attention_b = vec![0.0_f32; bottleneck];

        let mut attention_v = vec![0.0_f32; bottleneck];
        rng.fill_normal(&mut attention_v);
        for val in &mut attention_v {
            *val *= v_scale;
        }

        Ok(Self {
            attention_w,
            attention_b,
            attention_v,
            c,
            bottleneck,
        })
    }

    /// Compute the attention-weighted mean and std pooling.
    ///
    /// # Arguments
    ///
    /// * `features` — `[T, C]` row-major tensor.
    /// * `t` — sequence length.
    ///
    /// Returns `[2 * C]`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidSequenceLength`] when `t == 0`.
    /// - [`AudioError::DimensionMismatch`] when `features.len() != t * self.c`.
    pub fn forward(&self, features: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        if t == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }
        let c = self.c;
        let bn = self.bottleneck;
        let expected = t * c;
        if features.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        // ── Compute attention scores [T] ─────────────────────────────────────
        // For each frame: hidden[bn] = tanh(W·x + b), score = v^T·hidden
        let mut scores = vec![0.0_f32; t];
        for tok in 0..t {
            let x_tok = &features[tok * c..(tok + 1) * c];
            let mut hidden = vec![0.0_f32; bn];
            // hidden[j] = Σ_i W[j, i] * x[i] + b[j]
            for (j, hv) in hidden.iter_mut().enumerate() {
                let mut acc = self.attention_b[j];
                let w_row = &self.attention_w[j * c..(j + 1) * c];
                for (wi, &xi) in w_row.iter().zip(x_tok.iter()) {
                    acc += wi * xi;
                }
                *hv = acc.tanh();
            }
            // score = v^T · hidden
            let score: f32 = self
                .attention_v
                .iter()
                .zip(hidden.iter())
                .map(|(&vi, &hi)| vi * hi)
                .sum();
            scores[tok] = score;
        }

        // ── Softmax over time ─────────────────────────────────────────────────
        softmax_inplace(&mut scores);

        // ── Attention-weighted mean [C] ───────────────────────────────────────
        let mut mean_out = vec![0.0_f32; c];
        for tok in 0..t {
            let x_tok = &features[tok * c..(tok + 1) * c];
            let w = scores[tok];
            for (m, &xv) in mean_out.iter_mut().zip(x_tok.iter()) {
                *m += w * xv;
            }
        }

        // ── Attention-weighted std [C] ────────────────────────────────────────
        let mut var_out = vec![0.0_f32; c];
        for tok in 0..t {
            let x_tok = &features[tok * c..(tok + 1) * c];
            let w = scores[tok];
            for (var, (&xv, &mv)) in var_out.iter_mut().zip(x_tok.iter().zip(mean_out.iter())) {
                let diff = xv - mv;
                *var += w * diff * diff;
            }
        }

        let min_std: f32 = 1e-10;
        let mut output = Vec::with_capacity(2 * c);
        output.extend_from_slice(&mean_out);
        for v in &var_out {
            output.push(v.sqrt().max(min_std));
        }

        Ok(output)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn attentive_pool_output_shape() {
        let c = 32_usize;
        let t = 10_usize;
        let mut rng = make_rng();
        let pool = AttentivePool::new(c, &mut rng).expect("new ok");
        let features = vec![1.0_f32; t * c];
        let out = pool.forward(&features, t).expect("forward ok");
        assert_eq!(out.len(), 2 * c);
    }

    #[test]
    fn attentive_pool_output_finite() {
        let c = 16_usize;
        let t = 8_usize;
        let mut rng = make_rng();
        let pool = AttentivePool::new(c, &mut rng).expect("new ok");
        let mut features = vec![0.0_f32; t * c];
        rng.fill_normal(&mut features);
        let out = pool.forward(&features, t).expect("forward ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite in output");
    }

    #[test]
    fn attentive_pool_attn_sums_to_one() {
        // With all equal scores, softmax produces uniform weights.
        // Uniform weights over T frames → each weight = 1/T.
        // We verify the output matches manual uniform averaging.
        let c = 4_usize;
        let t = 3_usize;
        let mut rng = make_rng();
        // Build a pool but zero out the score weights so all scores = 0 → uniform attn.
        let mut pool = AttentivePool::new(c, &mut rng).expect("new ok");
        pool.attention_w.fill(0.0);
        pool.attention_b.fill(0.0);
        pool.attention_v.fill(1.0);
        // features: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        let features = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let out = pool.forward(&features, t).expect("forward ok");
        // Uniform weights → mean = [5, 6, 7, 8]
        let expected_mean = [5.0_f32, 6.0, 7.0, 8.0];
        for (i, &em) in expected_mean.iter().enumerate() {
            assert!(
                (out[i] - em).abs() < 1e-5,
                "mean[{i}]: got={} exp={em}",
                out[i]
            );
        }
    }

    #[test]
    fn attentive_pool_single_frame() {
        let c = 8_usize;
        let mut rng = make_rng();
        let pool = AttentivePool::new(c, &mut rng).expect("new ok");
        let features: Vec<f32> = (0..c).map(|i| i as f32).collect();
        let out = pool.forward(&features, 1).expect("single frame ok");
        assert_eq!(out.len(), 2 * c);
        // With T=1: mean = x[0], std = sqrt(0) → clamped to 1e-10
        for (i, &f_val) in features.iter().enumerate() {
            assert!(
                (out[i] - f_val).abs() < 1e-5,
                "mean[{i}]: got={} exp={f_val}",
                out[i]
            );
        }
        for (idx, &val) in out[c..].iter().enumerate() {
            assert!(val >= 1e-10, "std[{idx}]={val} should be >= 1e-10");
        }
    }

    #[test]
    fn attentive_pool_zero_t_error() {
        let mut rng = make_rng();
        let pool = AttentivePool::new(8, &mut rng).expect("new ok");
        let features = vec![1.0_f32; 8];
        let err = pool.forward(&features, 0).unwrap_err();
        assert!(matches!(err, AudioError::InvalidSequenceLength(0)));
    }

    #[test]
    fn attentive_pool_dim_mismatch_error() {
        let mut rng = make_rng();
        let pool = AttentivePool::new(8, &mut rng).expect("new ok");
        let features = vec![1.0_f32; 10]; // wrong size
        let err = pool.forward(&features, 2).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    #[test]
    fn attentive_pool_weights_finite() {
        let c = 64_usize;
        let mut rng = make_rng();
        let pool = AttentivePool::new(c, &mut rng).expect("new ok");
        assert!(pool.attention_w.iter().all(|v| v.is_finite()));
        assert!(pool.attention_b.iter().all(|v| v.is_finite()));
        assert!(pool.attention_v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attentive_pool_zero_c_error() {
        let mut rng = make_rng();
        let err = AttentivePool::new(0, &mut rng).unwrap_err();
        assert!(matches!(err, AudioError::InvalidEmbedDim(0)));
    }
}
