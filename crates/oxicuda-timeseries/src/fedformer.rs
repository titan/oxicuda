//! FEDformer: Frequency Enhanced Decomposed Transformer (Zhou et al. 2022 ICML).
//!
//! Reference: "FEDformer: Frequency Enhanced Decomposed Transformer for
//! Long-term Series Forecasting", Zhou et al., ICML 2022.
//!
//! FEDformer combines seasonal-trend series decomposition with a frequency
//! domain mixing block. The series is first split into a slowly-varying trend
//! (moving average) and a seasonal residual. The seasonal component is then
//! processed by a **Frequency Enhanced Block (FEB)**: each channel is
//! transformed to the frequency domain via a Discrete Fourier Transform, only
//! the `M` lowest-frequency modes are kept (the rest are zeroed), and each kept
//! mode is mixed across the channel dimension by a learnable complex linear map.
//! An inverse DFT returns to the time domain. The trend (optionally projected by
//! a learnable linear map) is added back to recover the full series.
//!
//! This pure-Rust CPU reference uses a direct O(n²) real-input DFT/iDFT so the
//! transform is exact (no power-of-two restriction). All tensors use the
//! crate-wide time-major `[seq_len, d_model]` row-major layout.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for a FEDformer model.
#[derive(Debug, Clone)]
pub struct FedformerConfig {
    /// Input sequence length (time axis).
    pub seq_len: usize,
    /// Channel / feature dimension.
    pub d_model: usize,
    /// Number of selected (lowest) frequency modes kept in the FEB.
    pub n_modes: usize,
    /// Moving-average kernel size for series decomposition (odd recommended).
    pub moving_avg_kernel: usize,
}

impl FedformerConfig {
    /// Small configuration: `d_model = 16`, `n_modes = 4`, `kernel = 5`.
    #[must_use]
    pub fn tiny(seq_len: usize) -> Self {
        Self {
            seq_len,
            d_model: 16,
            n_modes: 4,
            moving_avg_kernel: 5,
        }
    }
}

// ─── Frequency Enhanced Block ────────────────────────────────────────────────

/// Frequency Enhanced Block (FEB).
///
/// Transforms each channel of a `[seq_len, d_model]` signal to the frequency
/// domain, keeps only the `n_modes` lowest-frequency complex coefficients,
/// applies a per-mode learnable complex linear map across the `d_model` channel
/// dimension, then inverse-transforms back to the time domain.
///
/// The complex linear map for mode `m` is parameterised by two real
/// `d_model × d_model` weight matrices (`weight_real`, `weight_imag`) so that
/// for an input mode coefficient `(a + i·b)` per channel the mixed output is
/// `out_real = W_r · a − W_i · b` and `out_imag = W_r · b + W_i · a`.
#[derive(Debug, Clone)]
pub struct FrequencyEnhancedBlock {
    /// Sequence length (time axis).
    seq_len: usize,
    /// Channel dimension.
    d_model: usize,
    /// Number of kept low-frequency modes.
    n_modes: usize,
    /// Selected mode indices (the `n_modes` lowest frequencies, i.e. `0..n_modes`).
    mode_indices: Vec<usize>,
    /// Per-mode real weight matrices, each `d_model × d_model` (row-major).
    weight_real: Vec<Vec<f32>>,
    /// Per-mode imaginary weight matrices, each `d_model × d_model` (row-major).
    weight_imag: Vec<Vec<f32>>,
}

impl FrequencyEnhancedBlock {
    /// Build a Frequency Enhanced Block with randomly initialised complex weights.
    ///
    /// The `n_modes` lowest-frequency mode indices `0, 1, …, n_modes − 1` are
    /// selected (DC at index 0 upwards). Each per-mode complex weight matrix is
    /// initialised with a small Glorot-style scale so the spectral mixing starts
    /// near identity-magnitude.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidTopK`] when `n_modes == 0` or `n_modes > seq_len / 2 + 1`.
    pub fn new(seq_len: usize, d_model: usize, n_modes: usize, rng: &mut LcgRng) -> TsResult<Self> {
        if seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        let max_modes = seq_len / 2 + 1;
        if n_modes == 0 || n_modes > max_modes {
            return Err(TsError::InvalidTopK(n_modes));
        }

        // Select the n_modes lowest frequencies: 0 (DC), 1, …, n_modes − 1.
        let mode_indices: Vec<usize> = (0..n_modes).collect();

        // Glorot-style scale; fan-in == fan-out == d_model.
        let scale = (1.0_f32 / d_model as f32).sqrt();
        let mut init_mat = || -> Vec<f32> {
            let mut v = vec![0.0_f32; d_model * d_model];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= scale;
            }
            v
        };

        let mut weight_real = Vec::with_capacity(n_modes);
        let mut weight_imag = Vec::with_capacity(n_modes);
        for _ in 0..n_modes {
            weight_real.push(init_mat());
            weight_imag.push(init_mat());
        }

        Ok(Self {
            seq_len,
            d_model,
            n_modes,
            mode_indices,
            weight_real,
            weight_imag,
        })
    }

    /// Number of kept low-frequency modes.
    #[must_use]
    #[inline]
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// The selected (lowest) frequency mode indices.
    #[must_use]
    #[inline]
    pub fn mode_indices(&self) -> &[usize] {
        &self.mode_indices
    }

    /// Frequency-domain forward pass.
    ///
    /// Input and output are `[seq_len, d_model]` row-major (time-major, channels
    /// innermost). Each channel is treated as a length-`seq_len` real signal:
    /// DFT → keep `n_modes` lowest modes → per-mode complex channel mixing →
    /// inverse DFT → real output.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let n = self.seq_len;
        let d = self.d_model;
        let expected = n * d;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let n_freq = n / 2 + 1; // non-redundant rfft bins: 0 ..= n/2.

        // 1. Forward DFT per channel → spectra[freq][channel] = (re, im).
        //    Only the kept modes are needed; all other bins map to zero output.
        //    spec_re/spec_im are laid out [n_modes, d] for the kept modes.
        let two_pi_over_n = 2.0 * std::f32::consts::PI / n as f32;
        let mut spec_re = vec![0.0_f32; self.n_modes * d];
        let mut spec_im = vec![0.0_f32; self.n_modes * d];

        for (mi, &f) in self.mode_indices.iter().enumerate() {
            for ci in 0..d {
                let mut re = 0.0_f32;
                let mut im = 0.0_f32;
                for ti in 0..n {
                    let angle = two_pi_over_n * (f * ti) as f32;
                    let xv = x[ti * d + ci];
                    re += xv * angle.cos();
                    im -= xv * angle.sin(); // exp(-2πi f t / n)
                }
                spec_re[mi * d + ci] = re;
                spec_im[mi * d + ci] = im;
            }
        }

        // 2. Per-mode complex linear mixing across the channel dimension.
        //    out = W_r·a − W_i·b  (real),  W_r·b + W_i·a  (imag).
        let mut mixed_re = vec![0.0_f32; self.n_modes * d];
        let mut mixed_im = vec![0.0_f32; self.n_modes * d];
        for mi in 0..self.n_modes {
            let wr = &self.weight_real[mi];
            let wi = &self.weight_imag[mi];
            let a = &spec_re[mi * d..(mi + 1) * d];
            let b = &spec_im[mi * d..(mi + 1) * d];
            for oi in 0..d {
                let row = oi * d;
                let mut acc_re = 0.0_f32;
                let mut acc_im = 0.0_f32;
                for ki in 0..d {
                    let r = wr[row + ki];
                    let i = wi[row + ki];
                    acc_re += r * a[ki] - i * b[ki];
                    acc_im += r * b[ki] + i * a[ki];
                }
                mixed_re[mi * d + oi] = acc_re;
                mixed_im[mi * d + oi] = acc_im;
            }
        }

        // 3. Inverse DFT per channel using only the kept (non-redundant) modes.
        //    Real iDFT: y[t] = (1/n) Σ_f w_f · (Re·cos(θ) − Im·sin(θ)) where the
        //    Hermitian-conjugate bins double every coefficient except DC and (for
        //    even n) the Nyquist bin at n/2.
        let inv_n = 1.0_f32 / n as f32;
        let mut out = vec![0.0_f32; n * d];
        for ti in 0..n {
            for ci in 0..d {
                let mut acc = 0.0_f32;
                for (mi, &f) in self.mode_indices.iter().enumerate() {
                    let is_self_conjugate = f == 0 || (n % 2 == 0 && f == n / 2);
                    let weight = if is_self_conjugate { 1.0_f32 } else { 2.0_f32 };
                    let angle = two_pi_over_n * (f * ti) as f32;
                    let re = mixed_re[mi * d + ci];
                    let im = mixed_im[mi * d + ci];
                    acc += weight * (re * angle.cos() - im * angle.sin());
                }
                out[ti * d + ci] = acc * inv_n;
            }
        }

        // n_freq is used to document the spectral layout; assert the invariant
        // that we never select more modes than available bins.
        debug_assert!(self.n_modes <= n_freq);

        Ok(out)
    }
}

// ─── FEDformer model ─────────────────────────────────────────────────────────

/// FEDformer forecasting backbone.
///
/// Decomposes the input into seasonal and trend components, mixes the seasonal
/// part in the frequency domain via a [`FrequencyEnhancedBlock`], optionally
/// projects the trend through a learnable linear map, and recombines the two
/// to produce a `[seq_len, d_model]` representation.
#[derive(Debug, Clone)]
pub struct Fedformer {
    /// Frequency Enhanced Block applied to the seasonal component.
    feb: FrequencyEnhancedBlock,
    /// Moving-average kernel size for the series decomposition.
    moving_avg_kernel: usize,
    /// Trend projection weight `[d_model, d_model]` (row-major).
    trend_proj: Vec<f32>,
    /// Model configuration.
    cfg: FedformerConfig,
}

impl Fedformer {
    /// Build a FEDformer model, initialising the FEB and trend projection.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `seq_len == 0`.
    /// - [`TsError::InvalidEmbedDim`] when `d_model == 0`.
    /// - [`TsError::InvalidTopK`] when `n_modes == 0` or `n_modes > seq_len / 2 + 1`.
    /// - [`TsError::InvalidKernelSize`] when `moving_avg_kernel == 0`.
    pub fn new(cfg: FedformerConfig, rng: &mut LcgRng) -> TsResult<Self> {
        if cfg.seq_len == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if cfg.d_model == 0 {
            return Err(TsError::InvalidEmbedDim(0));
        }
        let max_modes = cfg.seq_len / 2 + 1;
        if cfg.n_modes == 0 || cfg.n_modes > max_modes {
            return Err(TsError::InvalidTopK(cfg.n_modes));
        }
        if cfg.moving_avg_kernel == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }

        let feb = FrequencyEnhancedBlock::new(cfg.seq_len, cfg.d_model, cfg.n_modes, rng)?;

        // Trend projection initialised near identity so the trend is preserved
        // by default while remaining learnable.
        let d = cfg.d_model;
        let scale = (1.0_f32 / d as f32).sqrt();
        let mut trend_proj = vec![0.0_f32; d * d];
        rng.fill_normal(&mut trend_proj);
        for w in &mut trend_proj {
            *w *= scale;
        }
        for i in 0..d {
            trend_proj[i * d + i] += 1.0_f32;
        }

        Ok(Self {
            feb,
            moving_avg_kernel: cfg.moving_avg_kernel,
            trend_proj,
            cfg,
        })
    }

    /// Access the model configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &FedformerConfig {
        &self.cfg
    }

    /// Access the underlying Frequency Enhanced Block.
    #[must_use]
    #[inline]
    pub fn frequency_block(&self) -> &FrequencyEnhancedBlock {
        &self.feb
    }

    /// Series decomposition into `(seasonal, trend)`.
    ///
    /// The trend is the centred moving average of `x` along the time axis using
    /// `moving_avg_kernel` with replicate (edge) boundary padding so the length
    /// is preserved. The seasonal component is `x − trend`. Both outputs are
    /// `[seq_len, d_model]` row-major.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn series_decomp(&self, x: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        let n = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let expected = n * d;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let trend = moving_average(x, n, d, self.moving_avg_kernel);
        let seasonal: Vec<f32> = x
            .iter()
            .zip(trend.iter())
            .map(|(&xv, &tv)| xv - tv)
            .collect();
        Ok((seasonal, trend))
    }

    /// Full forward pass.
    ///
    /// Decomposes `x` into seasonal + trend, mixes the seasonal part in the
    /// frequency domain with the FEB, projects the trend through the learnable
    /// linear map, and returns `seasonal_out + trend_proj` as a `[seq_len,
    /// d_model]` representation.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != seq_len * d_model`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let n = self.cfg.seq_len;
        let d = self.cfg.d_model;

        let (seasonal, trend) = self.series_decomp(x)?;
        let seasonal_out = self.feb.forward(&seasonal)?;

        // Trend projection: out[t, :] = trend[t, :] @ trend_proj^T.
        let mut out = vec![0.0_f32; n * d];
        for ti in 0..n {
            for oi in 0..d {
                let row = oi * d;
                let mut acc = 0.0_f32;
                for ki in 0..d {
                    acc += trend[ti * d + ki] * self.trend_proj[row + ki];
                }
                out[ti * d + oi] = acc + seasonal_out[ti * d + oi];
            }
        }
        Ok(out)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Centred moving average over the time axis of a `[t, c]` tensor.
///
/// `y[i, c] = (1/K) Σ_{k=-h}^{h} x[clamp(i + k, 0, t − 1), c]` where
/// `h = (K − 1) / 2`. Replicate (edge) padding keeps the length equal to `t`.
fn moving_average(x: &[f32], t: usize, c: usize, kernel: usize) -> Vec<f32> {
    let half = kernel / 2;
    let inv_k = 1.0_f32 / kernel as f32;
    let mut out = vec![0.0_f32; t * c];
    for ti in 0..t {
        for ci in 0..c {
            let mut sum = 0.0_f32;
            for ki in 0..kernel {
                // Position offset by half; clamp to [0, t − 1] (edge padding).
                let pos = (ti + ki) as isize - half as isize;
                let src = pos.clamp(0, t as isize - 1) as usize;
                sum += x[src * c + ci];
            }
            out[ti * c + ci] = sum * inv_k;
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    fn tiny_cfg(seq_len: usize) -> FedformerConfig {
        FedformerConfig {
            seq_len,
            d_model: 4,
            n_modes: 3,
            moving_avg_kernel: 5,
        }
    }

    // 1. series_decomp reconstruction: seasonal + trend == x.
    #[test]
    fn series_decomp_reconstructs_input() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(24);
        let model = Fedformer::new(cfg.clone(), &mut rng).expect("build");
        let x: Vec<f32> = (0..cfg.seq_len * cfg.d_model)
            .map(|i| (i as f32 * 0.13).sin() * 2.0 + i as f32 * 0.01)
            .collect();
        let (seasonal, trend) = model.series_decomp(&x).expect("decomp");
        for i in 0..x.len() {
            assert!(
                (x[i] - (seasonal[i] + trend[i])).abs() < 1e-4,
                "idx={i}: x={} s+t={}",
                x[i],
                seasonal[i] + trend[i]
            );
        }
    }

    // 2. FEB forward output shape == seq_len * d_model.
    #[test]
    fn feb_forward_shape() {
        let mut rng = make_rng();
        let seq_len = 20;
        let d_model = 6;
        let feb = FrequencyEnhancedBlock::new(seq_len, d_model, 4, &mut rng).expect("build");
        let x = vec![0.3_f32; seq_len * d_model];
        let out = feb.forward(&x).expect("forward");
        assert_eq!(out.len(), seq_len * d_model);
    }

    // 3. DFT → iDFT round-trip ≈ identity when all modes kept and weights = identity.
    #[test]
    fn dft_idft_round_trip_identity() {
        let mut rng = make_rng();
        let seq_len = 16;
        let d_model = 1;
        let max_modes = seq_len / 2 + 1;
        let mut feb =
            FrequencyEnhancedBlock::new(seq_len, d_model, max_modes, &mut rng).expect("build");
        // Replace each per-mode complex weight with identity (real I, imag 0)
        // so the FEB reduces to a pure DFT → iDFT round-trip.
        for m in 0..feb.n_modes() {
            for r in 0..d_model {
                for c in 0..d_model {
                    feb.weight_real[m][r * d_model + c] = if r == c { 1.0 } else { 0.0 };
                    feb.weight_imag[m][r * d_model + c] = 0.0;
                }
            }
        }
        let x: Vec<f32> = (0..seq_len).map(|i| (i as f32 * 0.7).sin() + 0.5).collect();
        let out = feb.forward(&x).expect("forward");
        for i in 0..x.len() {
            assert!(
                (x[i] - out[i]).abs() < 1e-3,
                "idx={i}: x={} out={}",
                x[i],
                out[i]
            );
        }
    }

    // 4. FEB keeps exactly n_modes modes: a high-frequency-only input is attenuated.
    #[test]
    fn feb_attenuates_high_frequency() {
        let mut rng = make_rng();
        let seq_len = 32;
        let d_model = 1;
        let mut feb = FrequencyEnhancedBlock::new(seq_len, d_model, 2, &mut rng).expect("build");
        // Identity complex weights so the only effect is mode truncation.
        for m in 0..feb.n_modes() {
            feb.weight_real[m][0] = 1.0;
            feb.weight_imag[m][0] = 0.0;
        }
        // Pure high-frequency signal (Nyquist-ish, far above the kept modes 0,1).
        let x: Vec<f32> = (0..seq_len)
            .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / seq_len as f32).sin())
            .collect();
        let out = feb.forward(&x).expect("forward");
        let in_energy: f32 = x.iter().map(|v| v * v).sum();
        let out_energy: f32 = out.iter().map(|v| v * v).sum();
        assert!(
            out_energy < in_energy * 0.1,
            "high freq not attenuated: in={in_energy} out={out_energy}"
        );
    }

    // 5. FEB selects exactly the n_modes lowest indices.
    #[test]
    fn feb_mode_indices_lowest() {
        let mut rng = make_rng();
        let feb = FrequencyEnhancedBlock::new(40, 3, 5, &mut rng).expect("build");
        assert_eq!(feb.mode_indices(), &[0, 1, 2, 3, 4]);
        assert_eq!(feb.n_modes(), 5);
    }

    // 6. forward output shape == input shape.
    #[test]
    fn forward_output_shape() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(30);
        let model = Fedformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![0.5_f32; cfg.seq_len * cfg.d_model];
        let out = model.forward(&x).expect("forward");
        assert_eq!(out.len(), cfg.seq_len * cfg.d_model);
    }

    // 7. Constant input → near-zero seasonal (trend captures the mean).
    #[test]
    fn constant_input_zero_seasonal() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(20);
        let model = Fedformer::new(cfg.clone(), &mut rng).expect("build");
        let x = vec![3.5_f32; cfg.seq_len * cfg.d_model];
        let (seasonal, _trend) = model.series_decomp(&x).expect("decomp");
        for &s in &seasonal {
            assert!(
                s.abs() < 1e-4,
                "seasonal should be ~0 for constant, got {s}"
            );
        }
    }

    // 8. Deterministic given the same seed.
    #[test]
    fn deterministic_given_seed() {
        let cfg = tiny_cfg(24);
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let model_a = Fedformer::new(cfg.clone(), &mut rng_a).expect("build");
        let model_b = Fedformer::new(cfg.clone(), &mut rng_b).expect("build");
        let x: Vec<f32> = (0..cfg.seq_len * cfg.d_model)
            .map(|i| (i as f32 * 0.21).cos())
            .collect();
        let out_a = model_a.forward(&x).expect("forward");
        let out_b = model_b.forward(&x).expect("forward");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-7, "non-deterministic: {a} vs {b}");
        }
    }

    // 9. Trend of a linear ramp ≈ the ramp (moving average preserves linear trend).
    #[test]
    fn trend_of_linear_ramp() {
        let mut rng = make_rng();
        let seq_len = 40;
        let d_model = 1;
        let cfg = FedformerConfig {
            seq_len,
            d_model,
            n_modes: 3,
            moving_avg_kernel: 5,
        };
        let model = Fedformer::new(cfg, &mut rng).expect("build");
        let x: Vec<f32> = (0..seq_len).map(|i| i as f32 * 0.5 + 1.0).collect();
        let (_seasonal, trend) = model.series_decomp(&x).expect("decomp");
        // Interior points (away from clamped edges) should match the ramp closely.
        for i in 3..seq_len - 3 {
            assert!(
                (trend[i] - x[i]).abs() < 1e-3,
                "idx={i}: trend={} ramp={}",
                trend[i],
                x[i]
            );
        }
    }

    // 10. n_modes == 1 keeps only the DC/lowest mode.
    #[test]
    fn single_mode_keeps_dc() {
        let mut rng = make_rng();
        let seq_len = 24;
        let d_model = 1;
        let mut feb = FrequencyEnhancedBlock::new(seq_len, d_model, 1, &mut rng).expect("build");
        assert_eq!(feb.mode_indices(), &[0]);
        // Identity weight → output is the DC component (constant = signal mean).
        feb.weight_real[0][0] = 1.0;
        feb.weight_imag[0][0] = 0.0;
        let x: Vec<f32> = (0..seq_len).map(|i| (i as f32 * 0.9).sin() + 2.0).collect();
        let out = feb.forward(&x).expect("forward");
        let mean: f32 = x.iter().sum::<f32>() / seq_len as f32;
        for &v in &out {
            assert!(
                (v - mean).abs() < 1e-3,
                "DC-only output should equal mean {mean}, got {v}"
            );
        }
    }

    // 11. err: seq_len == 0.
    #[test]
    fn err_seq_len_zero() {
        let mut rng = make_rng();
        let cfg = FedformerConfig {
            seq_len: 0,
            d_model: 4,
            n_modes: 1,
            moving_avg_kernel: 3,
        };
        assert!(matches!(
            Fedformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    // 12. err: d_model == 0.
    #[test]
    fn err_d_model_zero() {
        let mut rng = make_rng();
        let cfg = FedformerConfig {
            seq_len: 16,
            d_model: 0,
            n_modes: 1,
            moving_avg_kernel: 3,
        };
        assert!(matches!(
            Fedformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidEmbedDim(0)
        ));
    }

    // 13. err: n_modes == 0 and n_modes > seq_len / 2 + 1.
    #[test]
    fn err_n_modes_invalid() {
        let mut rng = make_rng();
        let cfg_zero = FedformerConfig {
            seq_len: 16,
            d_model: 4,
            n_modes: 0,
            moving_avg_kernel: 3,
        };
        assert!(matches!(
            Fedformer::new(cfg_zero, &mut rng).unwrap_err(),
            TsError::InvalidTopK(0)
        ));
        // seq_len = 16 → max modes = 9; 10 is too many.
        let cfg_big = FedformerConfig {
            seq_len: 16,
            d_model: 4,
            n_modes: 10,
            moving_avg_kernel: 3,
        };
        assert!(matches!(
            Fedformer::new(cfg_big, &mut rng).unwrap_err(),
            TsError::InvalidTopK(10)
        ));
    }

    // 14. err: moving_avg_kernel == 0.
    #[test]
    fn err_kernel_zero() {
        let mut rng = make_rng();
        let cfg = FedformerConfig {
            seq_len: 16,
            d_model: 4,
            n_modes: 2,
            moving_avg_kernel: 0,
        };
        assert!(matches!(
            Fedformer::new(cfg, &mut rng).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    // 15. err: x wrong length (both forward and series_decomp).
    #[test]
    fn err_wrong_input_length() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(20);
        let model = Fedformer::new(cfg, &mut rng).expect("build");
        let bad = vec![0.0_f32; 13];
        assert!(matches!(
            model.forward(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            model.series_decomp(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
        // FEB forward also validates length.
        let feb = FrequencyEnhancedBlock::new(20, 4, 2, &mut rng).expect("build");
        assert!(matches!(
            feb.forward(&bad).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    // 16. FEB forward output is finite.
    #[test]
    fn feb_forward_finite() {
        let mut rng = make_rng();
        let seq_len = 28;
        let d_model = 5;
        let feb = FrequencyEnhancedBlock::new(seq_len, d_model, 4, &mut rng).expect("build");
        let mut x = vec![0.0_f32; seq_len * d_model];
        rng.fill_normal(&mut x);
        let out = feb.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "FEB produced non-finite");
    }

    // 17. Moving average of a constant == that constant.
    #[test]
    fn moving_average_constant() {
        let t = 20;
        let c = 3;
        let x = vec![4.2_f32; t * c];
        let out = moving_average(&x, t, c, 5);
        for &v in &out {
            assert!((v - 4.2).abs() < 1e-5, "expected 4.2, got {v}");
        }
    }

    // 18. Full forward output is finite.
    #[test]
    fn forward_finite() {
        let mut rng = make_rng();
        let cfg = tiny_cfg(36);
        let model = Fedformer::new(cfg.clone(), &mut rng).expect("build");
        let mut x = vec![0.0_f32; cfg.seq_len * cfg.d_model];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("forward");
        assert!(out.iter().all(|v| v.is_finite()), "forward non-finite");
    }

    // 19. n_modes == seq_len / 2 + 1 (max) is accepted.
    #[test]
    fn max_modes_accepted() {
        let mut rng = make_rng();
        let seq_len = 18;
        let max_modes = seq_len / 2 + 1;
        let feb = FrequencyEnhancedBlock::new(seq_len, 2, max_modes, &mut rng).expect("build");
        assert_eq!(feb.n_modes(), max_modes);
    }
}
