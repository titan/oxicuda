//! 8-bit Adam optimizer with block-wise dynamic quantization of the moment buffers.
//!
//! Reference: Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2022).
//! *8-bit Optimizers via Block-wise Quantization*. <https://arxiv.org/abs/2110.02861>
//! (the `bitsandbytes` 8-bit Adam).
//!
//! Adam keeps two full-precision state buffers per parameter — the first moment `m` and the
//! second moment `v`. Storing them in `f32` doubles the optimizer memory relative to the
//! parameters. 8-bit Adam quantizes each buffer to `INT8`, splitting it into contiguous
//! *blocks* of `block_size` elements and giving every block its own `absmax` scale:
//!
//! ```text
//!   scale_b = absmax_b / 127
//!   code    = round( value / scale_b )  ∈ [-127, 127]
//!   value'  = code · scale_b
//! ```
//!
//! Because the scale is computed per block, a single outlier only inflates the scale of its
//! own block, keeping quantization error small everywhere else ("dynamic" / block-wise
//! quantization). Each [`Adam8bit::step`] dequantizes the stored state, performs an exact
//! `f32` Adam update, then re-quantizes the new moments back to `INT8`.

use crate::error::{PeftError, PeftResult};

/// Default block size used by `bitsandbytes`-style 8-bit optimizers.
pub const DEFAULT_BLOCK_SIZE: usize = 64;

/// Largest magnitude code retained, keeping the encoding symmetric (`-127..=127`).
const INT8_MAX: f32 = 127.0;

/// A buffer quantized block-wise to signed 8-bit integers with per-block `absmax` scales.
#[derive(Debug, Clone)]
pub struct BlockwiseInt8 {
    /// Quantized codes, one per element, each in `[-127, 127]`.
    pub codes: Vec<i8>,
    /// Per-block `absmax` scales; `absmax[b]` covers elements `[b·block_size, (b+1)·block_size)`.
    pub absmax: Vec<f32>,
    /// Number of elements per block.
    pub block_size: usize,
    /// Number of quantized elements.
    pub len: usize,
}

impl BlockwiseInt8 {
    /// Construct an all-zero quantized buffer of `len` elements.
    ///
    /// # Errors
    ///
    /// [`PeftError::ZeroBlockSize`] when `block_size == 0`.
    pub fn zeros(len: usize, block_size: usize) -> PeftResult<Self> {
        if block_size == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        let num_blocks = len.div_ceil(block_size).max(1);
        Ok(Self {
            codes: vec![0_i8; len],
            absmax: vec![0.0_f32; num_blocks],
            block_size,
            len,
        })
    }

    /// Quantize `data` block-wise to signed `INT8`.
    ///
    /// # Errors
    ///
    /// [`PeftError::ZeroBlockSize`] when `block_size == 0`.
    pub fn quantize(data: &[f32], block_size: usize) -> PeftResult<Self> {
        if block_size == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        let len = data.len();
        let num_blocks = len.div_ceil(block_size).max(1);
        let mut codes = vec![0_i8; len];
        let mut absmax = vec![0.0_f32; num_blocks];
        for (bi, chunk) in data.chunks(block_size).enumerate() {
            let amax = chunk.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
            absmax[bi] = amax;
            let inv = if amax > 0.0 { INT8_MAX / amax } else { 0.0 };
            let base = bi * block_size;
            for (off, &v) in chunk.iter().enumerate() {
                let q = (v * inv).round().clamp(-INT8_MAX, INT8_MAX);
                codes[base + off] = q as i8;
            }
        }
        Ok(Self {
            codes,
            absmax,
            block_size,
            len,
        })
    }

    /// Dequantize back to `f32`.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.len];
        for (bi, chunk) in out.chunks_mut(self.block_size).enumerate() {
            let scale = self.absmax[bi] / INT8_MAX;
            let base = bi * self.block_size;
            for (off, o) in chunk.iter_mut().enumerate() {
                *o = f32::from(self.codes[base + off]) * scale;
            }
        }
        out
    }

    /// Number of quantization blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.absmax.len()
    }
}

/// Hyper-parameters for [`Adam8bit`].
#[derive(Debug, Clone)]
pub struct Adam8bitConfig {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay rate for the first moment.
    pub beta1: f32,
    /// Exponential decay rate for the second moment.
    pub beta2: f32,
    /// Numerical-stability constant added to the denominator.
    pub eps: f32,
    /// Quantization block size for the moment buffers.
    pub block_size: usize,
}

impl Default for Adam8bitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }
}

/// Adam optimizer whose first/second moment state is stored block-wise in `INT8`.
#[derive(Debug, Clone)]
pub struct Adam8bit {
    cfg: Adam8bitConfig,
    m: BlockwiseInt8,
    v: BlockwiseInt8,
    t: u64,
    n: usize,
}

impl Adam8bit {
    /// Create an optimizer for `n_params` parameters with zero-initialised moment state.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] when `n_params == 0`.
    /// - [`PeftError::ZeroBlockSize`] when `cfg.block_size == 0`.
    pub fn new(n_params: usize, cfg: Adam8bitConfig) -> PeftResult<Self> {
        if n_params == 0 {
            return Err(PeftError::EmptyInput);
        }
        let m = BlockwiseInt8::zeros(n_params, cfg.block_size)?;
        let v = BlockwiseInt8::zeros(n_params, cfg.block_size)?;
        Ok(Self {
            cfg,
            m,
            v,
            t: 0,
            n: n_params,
        })
    }

    /// Number of parameters this optimizer tracks.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.n
    }

    /// Current step count (number of [`Self::step`] calls performed).
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Borrow the quantized first-moment buffer.
    #[must_use]
    pub fn m_state(&self) -> &BlockwiseInt8 {
        &self.m
    }

    /// Borrow the quantized second-moment buffer.
    #[must_use]
    pub fn v_state(&self) -> &BlockwiseInt8 {
        &self.v
    }

    /// Perform one Adam update of `params` in place using `grads`.
    ///
    /// The stored 8-bit moments are dequantized, updated in `f32` with bias correction, the
    /// parameters are stepped, and the new moments are re-quantized to `INT8`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `params.len()` or `grads.len()` differ from
    /// the configured parameter count.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> PeftResult<()> {
        if params.len() != self.n {
            return Err(PeftError::DimensionMismatch {
                expected: self.n,
                got: params.len(),
            });
        }
        if grads.len() != self.n {
            return Err(PeftError::DimensionMismatch {
                expected: self.n,
                got: grads.len(),
            });
        }
        self.t += 1;
        let b1 = self.cfg.beta1;
        let b2 = self.cfg.beta2;
        let lr = self.cfg.lr;
        let eps = self.cfg.eps;
        let exp = i32::try_from(self.t).unwrap_or(i32::MAX);
        let bias1 = 1.0 - b1.powi(exp);
        let bias2 = 1.0 - b2.powi(exp);

        let mut m = self.m.dequantize();
        let mut v = self.v.dequantize();
        for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let m_i = b1 * m[i] + (1.0 - b1) * g;
            let v_i = b2 * v[i] + (1.0 - b2) * g * g;
            m[i] = m_i;
            v[i] = v_i;
            let m_hat = m_i / bias1;
            let v_hat = v_i / bias2;
            *p -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        self.m = BlockwiseInt8::quantize(&m, self.cfg.block_size)?;
        self.v = BlockwiseInt8::quantize(&v, self.cfg.block_size)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference full-precision Adam used as the ground truth in comparison tests.
    struct FullAdam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        m: Vec<f32>,
        v: Vec<f32>,
        t: u64,
    }

    impl FullAdam {
        fn new(n: usize, cfg: &Adam8bitConfig) -> Self {
            Self {
                lr: cfg.lr,
                beta1: cfg.beta1,
                beta2: cfg.beta2,
                eps: cfg.eps,
                m: vec![0.0; n],
                v: vec![0.0; n],
                t: 0,
            }
        }

        fn step(&mut self, params: &mut [f32], grads: &[f32]) {
            self.t += 1;
            let bias1 = 1.0 - self.beta1.powi(self.t as i32);
            let bias2 = 1.0 - self.beta2.powi(self.t as i32);
            for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[i] / bias1;
                let v_hat = self.v[i] / bias2;
                *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    #[test]
    fn quant_dequant_roundtrip_within_block_step() {
        let data: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.37 + 5.0).collect();
        let block_size = 8;
        let q = BlockwiseInt8::quantize(&data, block_size)
            .expect("quantization should succeed with valid data and non-zero block size");
        let dq = q.dequantize();
        assert_eq!(dq.len(), data.len());
        for (bi, chunk) in data.chunks(block_size).enumerate() {
            let step = q.absmax[bi] / 127.0;
            for (off, &orig) in chunk.iter().enumerate() {
                let recon = dq[bi * block_size + off];
                assert!(
                    (orig - recon).abs() <= step + 1e-6,
                    "block {bi} elem {off}: |{orig} - {recon}| exceeds step {step}"
                );
            }
        }
    }

    #[test]
    fn int8_range_respected() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32).sin() * 13.0).collect();
        let q = BlockwiseInt8::quantize(&data, 16)
            .expect("quantization should succeed with valid data and block size");
        for &c in &q.codes {
            assert!(
                (-127..=127).contains(&c),
                "code {c} outside INT8 symmetric range"
            );
        }
    }

    #[test]
    fn zeros_dequantize_to_zero() {
        let q = BlockwiseInt8::zeros(20, 8)
            .expect("BlockwiseInt8::zeros should succeed with non-zero block size");
        assert_eq!(q.num_blocks(), 3);
        assert!(q.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn eight_bit_adam_tracks_full_precision_on_quadratic() {
        // Minimise f(w) = Σ (w_i - target_i)²  ⇒  grad = 2 (w - target).
        let target = [1.0_f32, -2.0, 0.5, 3.0, -0.75];
        let n = target.len();
        let cfg = Adam8bitConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            block_size: 2, // exercise multiple (incl. a partial) blocks
        };
        let steps = 300;
        let mut opt8 = Adam8bit::new(n, cfg.clone())
            .expect("Adam8bit::new should succeed with valid config and non-zero params");
        let mut full = FullAdam::new(n, &cfg);
        let mut w8 = vec![0.0_f32; n];
        let mut wf = vec![0.0_f32; n];
        for _ in 0..steps {
            let g8: Vec<f32> = w8
                .iter()
                .zip(target.iter())
                .map(|(w, t)| 2.0 * (w - t))
                .collect();
            let gf: Vec<f32> = wf
                .iter()
                .zip(target.iter())
                .map(|(w, t)| 2.0 * (w - t))
                .collect();
            opt8.step(&mut w8, &g8)
                .expect("Adam8bit step should succeed with matching gradient dimensions");
            full.step(&mut wf, &gf);
        }
        for i in 0..n {
            // 8-bit Adam stays close to full-precision Adam (the headline guarantee) ...
            assert!(
                (w8[i] - wf[i]).abs() < 5e-2,
                "8-bit vs full Adam diverged at {i}: {} vs {}",
                w8[i],
                wf[i]
            );
            // ... and the quantized optimizer still converges toward the optimum.
            assert!(
                (w8[i] - target[i]).abs() < 5e-2,
                "8-bit Adam did not converge at {i}: {} vs target {}",
                w8[i],
                target[i]
            );
        }
        // The full-precision reference must converge tightly too.
        let full_loss: f32 = wf
            .iter()
            .zip(target.iter())
            .map(|(w, t)| (w - t).powi(2))
            .sum();
        assert!(
            full_loss < 1e-3,
            "reference Adam did not converge: loss {full_loss}"
        );
        assert_eq!(opt8.step_count(), steps as u64);
    }

    #[test]
    fn state_stays_finite_and_quantized() {
        let cfg = Adam8bitConfig::default();
        let n = 10;
        let mut opt = Adam8bit::new(n, cfg)
            .expect("Adam8bit::new should succeed with valid config and non-zero params");
        let mut w = vec![0.5_f32; n];
        for s in 0..15 {
            let g: Vec<f32> = (0..n).map(|i| ((i + s) as f32).cos() * 0.3).collect();
            opt.step(&mut w, &g)
                .expect("Adam8bit step should succeed with matching gradient dimensions");
        }
        assert!(w.iter().all(|v| v.is_finite()));
        assert!(opt.m_state().dequantize().iter().all(|v| v.is_finite()));
        assert!(opt.v_state().dequantize().iter().all(|v| v.is_finite()));
        for &c in &opt.m_state().codes {
            assert!((-127..=127).contains(&c));
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut opt = Adam8bit::new(4, Adam8bitConfig::default())
            .expect("Adam8bit::new should succeed with default config and non-zero params");
        let mut w = vec![0.0_f32; 4];
        assert!(opt.step(&mut w, &[0.0; 3]).is_err());
        let mut w_bad = vec![0.0_f32; 5];
        assert!(opt.step(&mut w_bad, &[0.0; 5]).is_err());
    }
}
