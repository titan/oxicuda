//! RWKV linear-attention recurrent layer (Peng et al., 2023).
//!
//! RWKV replaces the quadratic self-attention of a Transformer with a linear
//! recurrence that can be computed in O(T·D) time while retaining competitive
//! modelling performance.  Each layer consists of two sub-layers:
//!
//! * **Time-mixing** — a WKV recurrence that generalises linear attention.
//! * **Space-mixing** — a token-shift FFN (channel-mixing).
//!
//! Both sub-layers use residual connections, so the overall layer maps
//! `x_seq [T × D]` → `y_seq [T × D]` with the same shape.
//!
//! # Reference
//!
//! Peng, B. et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
//! arXiv:2305.13048.

use crate::LcgRng;
use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Numerically-stable sigmoid.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Softplus: ln(1 + exp(x)).
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // avoid overflow: softplus(x) ≈ x for large x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Dense matrix-vector product: `y = W · x` where `W` is `rows × cols`.
fn matmul_vec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut s = 0.0f32;
        for j in 0..cols {
            s += w[i * cols + j] * x[j];
        }
        y[i] = s;
    }
    y
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`RwkvLayer`].
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    /// Input / output model dimension.
    pub d_model: usize,
    /// FFN hidden dimension for the space-mixing sub-layer (typically 4×d_model).
    pub d_ffn: usize,
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// RWKV layer implementing time-mixing (WKV recurrence) and space-mixing (FFN).
pub struct RwkvLayer {
    // ---- Time-mixing weights ----
    /// Receptance projection `[d_model × d_model]`.
    w_r: Vec<f32>,
    /// Key projection `[d_model × d_model]`.
    w_k: Vec<f32>,
    /// Value projection `[d_model × d_model]`.
    w_v: Vec<f32>,
    /// Output projection `[d_model × d_model]`.
    w_o: Vec<f32>,
    /// Per-channel positive decay values `[d_model]` (applied as `exp(-decay)`).
    decay: Vec<f32>,

    // ---- Space-mixing weights ----
    /// FFN key projection `[d_ffn × d_model]`.
    w_k_ffn: Vec<f32>,
    /// FFN value projection `[d_model × d_ffn]`.
    w_v_ffn: Vec<f32>,
    /// FFN receptance projection `[d_model × d_model]`.
    w_r_ffn: Vec<f32>,
    /// Token-shift mix parameter for WKV keys `[d_model]`.
    mix_k_time: Vec<f32>,
    /// Token-shift mix parameter for space-mixing `[d_model]`.
    mix_k_ffn: Vec<f32>,

    config: RwkvConfig,
}

impl RwkvLayer {
    /// Construct a new [`RwkvLayer`] with random weight initialisation.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] when `d_model` or `d_ffn` is
    /// zero.
    pub fn new(config: RwkvConfig, rng: &mut LcgRng) -> DnnResult<Self> {
        if config.d_model == 0 {
            return Err(DnnError::InvalidArgument("d_model must be > 0".to_owned()));
        }
        if config.d_ffn == 0 {
            return Err(DnnError::InvalidArgument("d_ffn must be > 0".to_owned()));
        }

        let d_m = config.d_model;
        let d_f = config.d_ffn;
        let scale = 0.01_f32;

        // Inline random-vector generators to avoid conflicting closure borrows.
        let w_r: Vec<f32> = (0..d_m * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();
        let w_k: Vec<f32> = (0..d_m * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();
        let w_v: Vec<f32> = (0..d_m * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();
        let w_o: Vec<f32> = (0..d_m * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();

        // Decay must be positive (we store w > 0, use exp(-w) as the decay factor).
        let decay: Vec<f32> = (0..d_m).map(|_| softplus(rng.next_f64() as f32)).collect();

        let w_k_ffn: Vec<f32> = (0..d_f * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();
        let w_v_ffn: Vec<f32> = (0..d_m * d_f)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();
        let w_r_ffn: Vec<f32> = (0..d_m * d_m)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * scale)
            .collect();

        // Mix parameters initialised near 0.5 so both the current token and
        // the previous token have roughly equal weight at the start.
        let mix_k_time: Vec<f32> = (0..d_m)
            .map(|_| 0.5 + (rng.next_f64() as f32 - 0.5) * 0.1)
            .collect();
        let mix_k_ffn: Vec<f32> = (0..d_m)
            .map(|_| 0.5 + (rng.next_f64() as f32 - 0.5) * 0.1)
            .collect();

        Ok(Self {
            w_r,
            w_k,
            w_v,
            w_o,
            decay,
            w_k_ffn,
            w_v_ffn,
            w_r_ffn,
            mix_k_time,
            mix_k_ffn,
            config,
        })
    }

    /// Input / output model dimension.
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    // -----------------------------------------------------------------------
    // Time-mixing (WKV recurrence)
    // -----------------------------------------------------------------------

    /// Compute the WKV time-mixing sub-layer for the full input sequence.
    ///
    /// `x_seq` is `[seq_len × d_model]`.  Returns `[seq_len × d_model]`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] when `x_seq.len()` is
    /// inconsistent.
    pub fn forward_time_mix(&self, x_seq: &[f32], seq_len: usize) -> DnnResult<Vec<f32>> {
        let d_m = self.config.d_model;

        if x_seq.len() != seq_len * d_m {
            return Err(DnnError::InvalidDimension(format!(
                "x_seq.len() expected {}, got {}",
                seq_len * d_m,
                x_seq.len()
            )));
        }

        let mut y_seq = vec![0.0f32; seq_len * d_m];

        // Numerator and denominator of the WKV recurrence.
        let mut num = vec![0.0f32; d_m];
        let mut den = vec![0.0f32; d_m];

        // Pre-compute decay factors (per channel).
        let exp_neg_decay: Vec<f32> = self.decay.iter().map(|&w| (-w).exp()).collect();

        // Previous token embedding for time-shift (token-shift in time dimension).
        let mut prev_x = vec![0.0f32; d_m];

        for t in 0..seq_len {
            let x_t = &x_seq[t * d_m..(t + 1) * d_m];

            // r_t = sigmoid(W_r · x_t)
            let r_t_pre = matmul_vec(&self.w_r, d_m, d_m, x_t);
            let r_t: Vec<f32> = r_t_pre.iter().map(|&v| sigmoid(v)).collect();

            // Apply time-shift mix for key computation:
            // x_key = mix_k_time ⊙ x_t + (1 - mix_k_time) ⊙ prev_x
            let x_key: Vec<f32> = (0..d_m)
                .map(|c| self.mix_k_time[c] * x_t[c] + (1.0 - self.mix_k_time[c]) * prev_x[c])
                .collect();

            // k_t = W_k · x_key  (uses time-shifted input)
            let k_t = matmul_vec(&self.w_k, d_m, d_m, &x_key);

            // v_t = W_v · x_t
            let v_t = matmul_vec(&self.w_v, d_m, d_m, x_t);

            // WKV recurrence update + compute wkv_t.
            let mut wkv_t = vec![0.0f32; d_m];
            for c in 0..d_m {
                let ek = k_t[c].exp();
                num[c] = exp_neg_decay[c] * num[c] + ek * v_t[c];
                den[c] = exp_neg_decay[c] * den[c] + ek;
                wkv_t[c] = num[c] / den[c].max(1e-10);
            }

            // output_t = r_t ⊙ wkv_t
            let output_t: Vec<f32> = r_t.iter().zip(wkv_t.iter()).map(|(r, w)| r * w).collect();

            // y_t = W_o · output_t
            let y_t = matmul_vec(&self.w_o, d_m, d_m, &output_t);
            y_seq[t * d_m..(t + 1) * d_m].copy_from_slice(&y_t);

            prev_x.copy_from_slice(x_t);
        }

        Ok(y_seq)
    }

    // -----------------------------------------------------------------------
    // Space-mixing (channel-mixing FFN with token-shift)
    // -----------------------------------------------------------------------

    /// Compute the space-mixing sub-layer for the full input sequence.
    ///
    /// `x_seq` is `[seq_len × d_model]`.  Returns `[seq_len × d_model]`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] when `x_seq.len()` is
    /// inconsistent.
    pub fn forward_space_mix(&self, x_seq: &[f32], seq_len: usize) -> DnnResult<Vec<f32>> {
        let d_m = self.config.d_model;
        let d_f = self.config.d_ffn;

        if x_seq.len() != seq_len * d_m {
            return Err(DnnError::InvalidDimension(format!(
                "x_seq.len() expected {}, got {}",
                seq_len * d_m,
                x_seq.len()
            )));
        }

        let mut y_seq = vec![0.0f32; seq_len * d_m];
        let mut prev_x = vec![0.0f32; d_m]; // x_{t-1}, initialised to zero.

        for t in 0..seq_len {
            let x_t = &x_seq[t * d_m..(t + 1) * d_m];

            // x_mix = mix_k_ffn ⊙ x_t + (1 - mix_k_ffn) ⊙ x_{t-1}
            let x_mix: Vec<f32> = (0..d_m)
                .map(|c| self.mix_k_ffn[c] * x_t[c] + (1.0 - self.mix_k_ffn[c]) * prev_x[c])
                .collect();

            // k_ffn = relu(W_k_ffn · x_mix)^2  (squared ReLU)  [d_ffn]
            let k_pre = matmul_vec(&self.w_k_ffn, d_f, d_m, &x_mix);
            let k_ffn: Vec<f32> = k_pre.iter().map(|&v| v.max(0.0).powi(2)).collect();

            // v_ffn = W_v_ffn · k_ffn  [d_model]
            let v_ffn = matmul_vec(&self.w_v_ffn, d_m, d_f, &k_ffn);

            // r_ffn = sigmoid(W_r_ffn · x_mix)  [d_model]
            let r_pre = matmul_vec(&self.w_r_ffn, d_m, d_m, &x_mix);
            let r_ffn: Vec<f32> = r_pre.iter().map(|&v| sigmoid(v)).collect();

            // y_t = r_ffn ⊙ v_ffn
            let y_t: Vec<f32> = r_ffn.iter().zip(v_ffn.iter()).map(|(r, v)| r * v).collect();
            y_seq[t * d_m..(t + 1) * d_m].copy_from_slice(&y_t);

            prev_x.copy_from_slice(x_t);
        }

        Ok(y_seq)
    }

    // -----------------------------------------------------------------------
    // Combined forward pass
    // -----------------------------------------------------------------------

    /// Run the full RWKV layer (time-mixing then space-mixing, each with a
    /// residual connection).
    ///
    /// `x_seq` is `[seq_len × d_model]`.  Returns `[seq_len × d_model]`.
    ///
    /// # Errors
    ///
    /// Propagates any error from the sub-layers.
    pub fn forward(&self, x_seq: &[f32], seq_len: usize) -> DnnResult<Vec<f32>> {
        let d_m = self.config.d_model;

        if x_seq.len() != seq_len * d_m {
            return Err(DnnError::InvalidDimension(format!(
                "x_seq.len() expected {}, got {}",
                seq_len * d_m,
                x_seq.len()
            )));
        }

        // 1. Time-mixing + residual.
        let time_out = self.forward_time_mix(x_seq, seq_len)?;
        let mut x_after_time = vec![0.0f32; seq_len * d_m];
        for i in 0..x_after_time.len() {
            x_after_time[i] = x_seq[i] + time_out[i];
        }

        // 2. Space-mixing + residual.
        let space_out = self.forward_space_mix(&x_after_time, seq_len)?;
        let mut output = x_after_time;
        for i in 0..output.len() {
            output[i] += space_out[i];
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(d_model: usize, d_ffn: usize) -> RwkvLayer {
        let cfg = RwkvConfig { d_model, d_ffn };
        let mut rng = LcgRng::new(42);
        RwkvLayer::new(cfg, &mut rng).expect("valid config")
    }

    fn random_seq(seq_len: usize, d_model: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..seq_len * d_model)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0)
            .collect()
    }

    // 1. Forward output shape
    #[test]
    fn forward_output_shape() {
        let layer = make_layer(16, 64);
        let x = random_seq(8, 16, 1);
        let out = layer.forward(&x, 8).expect("ok");
        assert_eq!(out.len(), 8 * 16);
    }

    // 2. All forward outputs are finite
    #[test]
    fn forward_finite() {
        let layer = make_layer(16, 64);
        let x = random_seq(6, 16, 2);
        let out = layer.forward(&x, 6).expect("ok");
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v}");
        }
    }

    // 3. Single-token sequence
    #[test]
    fn single_token() {
        let layer = make_layer(8, 32);
        let x = random_seq(1, 8, 3);
        let out = layer.forward(&x, 1).expect("ok");
        assert_eq!(out.len(), 8);
    }

    // 4. d_model=0 → error
    #[test]
    fn d_model_0_error() {
        let cfg = RwkvConfig {
            d_model: 0,
            d_ffn: 32,
        };
        let mut rng = LcgRng::new(0);
        let result = RwkvLayer::new(cfg, &mut rng);
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }

    // 5. d_ffn=0 → error
    #[test]
    fn d_ffn_0_error() {
        let cfg = RwkvConfig {
            d_model: 8,
            d_ffn: 0,
        };
        let mut rng = LcgRng::new(0);
        let result = RwkvLayer::new(cfg, &mut rng);
        assert!(matches!(result, Err(DnnError::InvalidArgument(_))));
    }

    // 6. Time-mix output shape
    #[test]
    fn time_mix_output_shape() {
        let layer = make_layer(16, 64);
        let x = random_seq(5, 16, 6);
        let out = layer.forward_time_mix(&x, 5).expect("ok");
        assert_eq!(out.len(), 5 * 16);
    }

    // 7. Space-mix output shape
    #[test]
    fn space_mix_output_shape() {
        let layer = make_layer(16, 64);
        let x = random_seq(5, 16, 7);
        let out = layer.forward_space_mix(&x, 5).expect("ok");
        assert_eq!(out.len(), 5 * 16);
    }

    // 8. Nonzero input → nonzero output
    #[test]
    fn recurrence_nonzero() {
        let layer = make_layer(8, 32);
        let x = random_seq(4, 8, 8);
        let out = layer.forward(&x, 4).expect("ok");
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "output should be nonzero for nonzero input");
    }

    // 9. With uniform input the output varies across timesteps (recurrence).
    #[test]
    fn sequence_varies_over_time() {
        let layer = make_layer(8, 32);
        let d_m = 8;
        let seq_len = 6;
        // Use non-trivial uniform-ish input.
        let x: Vec<f32> = (0..seq_len * d_m)
            .map(|i| 0.1 * (i % d_m) as f32 + 0.05)
            .collect();
        let out = layer.forward(&x, seq_len).expect("ok");
        // The WKV state evolves so outputs at different timesteps should differ.
        let out_t0 = &out[..d_m];
        let out_last = &out[(seq_len - 1) * d_m..seq_len * d_m];
        let diff: f32 = out_t0
            .iter()
            .zip(out_last.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "output should vary across timesteps due to recurrence (diff={diff})"
        );
    }

    // 10. No NaN in output
    #[test]
    fn forward_not_nan() {
        let layer = make_layer(12, 48);
        let x = random_seq(10, 12, 10);
        let out = layer.forward(&x, 10).expect("ok");
        assert!(out.iter().all(|v| !v.is_nan()), "output contains NaN");
    }
}
