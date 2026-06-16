//! Wavelet Neural Operator (1D Haar).
//!
//! Reference:
//!
//! - Tripura, T. & Chakraborty, S. (2022). *Wavelet Neural Operator for
//!   solving parametric partial differential equations in computational
//!   mechanics problems*. arXiv:2205.02191.
//!
//! Mirrors the FNO architecture but replaces the **Fourier** transform with a
//! multi-level **wavelet** transform. For a 1-D real input feature field of
//! shape `(in_channels, seq_len)`:
//!
//! 1. Compute the `n_levels`-level forward Haar wavelet transform per channel,
//!    yielding one approximation buffer at the coarsest level and one detail
//!    buffer per level.
//! 2. At each level (including the approximation), apply a learnable
//!    `(in_channels × out_channels)` real linear map per position across the
//!    channel axis.
//! 3. Inverse-transform back to `(out_channels, seq_len)`.
//! 4. Add a 1×1 (per-position) linear residual `W · x + b` with
//!    `W ∈ ℝ^{out × in}`.
//!
//! The Haar transform used here is the orthonormal one:
//!
//! ```text
//! approx_{l+1}[i] = (approx_l[2i] + approx_l[2i + 1]) / √2
//! detail_{l+1}[i] = (approx_l[2i] − approx_l[2i + 1]) / √2
//! ```
//!
//! with reconstruction
//!
//! ```text
//! approx_l[2i]     = (approx_{l+1}[i] + detail_{l+1}[i]) / √2
//! approx_l[2i + 1] = (approx_{l+1}[i] − detail_{l+1}[i]) / √2 .
//! ```
//!
//! Both passes preserve the L²-norm of the channel (Parseval-style), so the
//! whole spectral block is well-conditioned in the same sense as the
//! FNO/FFT-based block but tuned to localised features.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the 1-D Wavelet Neural Operator.
#[derive(Debug, Clone)]
pub struct WnoConfig {
    /// Number of input channels (`≥ 1`).
    pub in_channels: usize,
    /// Number of output channels (`≥ 1`).
    pub out_channels: usize,
    /// Spatial length (must be a power of two, `≥ 2`).
    pub seq_len: usize,
    /// Number of Haar decomposition levels (`≥ 1`, with `2^n_levels ≤ seq_len`).
    pub n_levels: usize,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

#[inline]
fn is_power_of_two(n: usize) -> bool {
    n >= 2 && (n & (n - 1)) == 0
}

/// One-step 1-D Haar decomposition of a flat row-major buffer
/// `(in_channels × len)` along the last axis. Returns
/// `(approx_next, detail_next)`, each shaped `(in_channels × len/2)`.
fn haar_step(buf: &[f32], in_channels: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = len / 2;
    let sqrt2_inv = 1.0_f32 / 2.0_f32.sqrt();
    let mut approx = vec![0.0_f32; in_channels * half];
    let mut detail = vec![0.0_f32; in_channels * half];
    for c in 0..in_channels {
        let off = c * len;
        let off_h = c * half;
        for i in 0..half {
            let a = buf[off + 2 * i];
            let b = buf[off + 2 * i + 1];
            approx[off_h + i] = (a + b) * sqrt2_inv;
            detail[off_h + i] = (a - b) * sqrt2_inv;
        }
    }
    (approx, detail)
}

/// One-step inverse Haar reconstruction of channel-major flat buffers
/// `(channels × len)`. Returns the longer buffer `(channels × 2·len)`.
///
/// Both `approx` and `detail` must have identical shape `(channels × len)`.
fn haar_step_inverse(approx: &[f32], detail: &[f32], channels: usize, len: usize) -> Vec<f32> {
    let sqrt2_inv = 1.0_f32 / 2.0_f32.sqrt();
    let total_len = 2 * len;
    let mut out = vec![0.0_f32; channels * total_len];
    for c in 0..channels {
        let src = c * len;
        let dst = c * total_len;
        for i in 0..len {
            let a = approx[src + i];
            let d = detail[src + i];
            out[dst + 2 * i] = (a + d) * sqrt2_inv;
            out[dst + 2 * i + 1] = (a - d) * sqrt2_inv;
        }
    }
    out
}

/// Apply an `(in_channels × out_channels)` channel-mixing matrix per position
/// to a `(in_channels × len)` row-major buffer, returning an
/// `(out_channels × len)` buffer.
///
/// `weight[i * out + o]` is the coefficient connecting input channel `i` to
/// output channel `o`.
fn channel_linear(
    x: &[f32],
    weight: &[f32],
    in_channels: usize,
    out_channels: usize,
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_channels * len];
    for o in 0..out_channels {
        for p in 0..len {
            let mut acc = 0.0_f32;
            for i in 0..in_channels {
                let xi = x[i * len + p];
                let w = weight[i * out_channels + o];
                acc += w * xi;
            }
            out[o * len + p] = acc;
        }
    }
    out
}

// ─── Wno ─────────────────────────────────────────────────────────────────────

/// 1-D Wavelet Neural Operator.
///
/// Stores:
///
/// - `detail_weights`: a vector of `n_levels` weight matrices, one per detail
///   level. Each is `(in_channels × out_channels)` flattened row-major as
///   `weight[i * out_channels + o]`.
/// - `approx_weight`: the `(in_channels × out_channels)` channel-mixing matrix
///   applied to the coarsest approximation level.
/// - `residual_w`: 1×1 residual matrix `(out_channels × in_channels)`, flattened
///   as `residual_w[o * in_channels + i]`.
/// - `residual_b`: per-output-channel bias of length `out_channels`.
pub struct Wno {
    cfg: WnoConfig,
    detail_weights: Vec<Vec<f32>>,
    approx_weight: Vec<f32>,
    residual_w: Vec<f32>,
    residual_b: Vec<f32>,
}

impl Wno {
    /// Construct a new `Wno` with Gaussian-initialised spectral weights and
    /// He-uniform residual weights.
    ///
    /// # Errors
    ///
    /// - [`PinnError::InvalidLayerWidth`] if `in_channels == 0` or
    ///   `out_channels == 0`.
    /// - [`PinnError::InvalidGridResolution`] if `seq_len < 2` or is not a
    ///   power of two.
    /// - [`PinnError::InvalidNetworkDepth`] if `n_levels == 0`.
    /// - [`PinnError::TooManyFourierModes`] if `2^n_levels > seq_len`.
    pub fn new(cfg: WnoConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        validate_cfg(&cfg)?;

        let in_c = cfg.in_channels;
        let out_c = cfg.out_channels;
        let n_mat = in_c * out_c;

        // Spectral scale: 1 / (in · out) keeps initial magnitudes bounded.
        let denom = n_mat.max(1) as f32;
        let spec_scale = 1.0_f32 / denom;

        let mut detail_weights: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_levels);
        for _ in 0..cfg.n_levels {
            let mut w = Vec::with_capacity(n_mat);
            // Fill via Box-Muller normal pairs.
            let mut k = 0;
            while k + 1 < n_mat {
                let (a, b) = rng.next_normal_pair();
                w.push(a * spec_scale);
                w.push(b * spec_scale);
                k += 2;
            }
            if k < n_mat {
                let (a, _) = rng.next_normal_pair();
                w.push(a * spec_scale);
            }
            detail_weights.push(w);
        }

        let mut approx_weight = Vec::with_capacity(n_mat);
        let mut k = 0;
        while k + 1 < n_mat {
            let (a, b) = rng.next_normal_pair();
            approx_weight.push(a * spec_scale);
            approx_weight.push(b * spec_scale);
            k += 2;
        }
        if k < n_mat {
            let (a, _) = rng.next_normal_pair();
            approx_weight.push(a * spec_scale);
        }

        // He-style residual: scale = sqrt(2 / in_channels), uniform on
        // [-scale, +scale]. next_u32 ∈ [0, 2^31), divide by 2^31 → [0, 1).
        let res_scale = (2.0_f32 / (in_c as f32).max(1.0)).sqrt();
        let mut residual_w = Vec::with_capacity(out_c * in_c);
        for _ in 0..(out_c * in_c) {
            let u = (rng.next_u32() as f32) / ((1u64 << 32) as f32);
            residual_w.push((u * 2.0 - 1.0) * res_scale);
        }
        let residual_b = vec![0.0_f32; out_c];

        Ok(Self {
            cfg,
            detail_weights,
            approx_weight,
            residual_w,
            residual_b,
        })
    }

    /// Borrow the configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &WnoConfig {
        &self.cfg
    }

    /// Total trainable-parameter count: `(n_levels + 1) · in · out` spectral
    /// reals plus the residual `(out · in)` plus the bias `out`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let n_mat = self.cfg.in_channels * self.cfg.out_channels;
        let spec = (self.cfg.n_levels + 1) * n_mat;
        let res_w = self.cfg.out_channels * self.cfg.in_channels;
        let res_b = self.cfg.out_channels;
        spec + res_w + res_b
    }

    /// Multi-level 1-D Haar wavelet decomposition of an `(in_channels × seq_len)`
    /// row-major input.
    ///
    /// Returns the pair `(approx, details)` where:
    ///
    /// - `approx` is the coarsest approximation, of length
    ///   `in_channels · seq_len / 2^n_levels`, row-major.
    /// - `details[l]` (`l = 0, …, n_levels − 1`) is the detail at level `l + 1`,
    ///   of length `in_channels · seq_len / 2^{l + 1}`, row-major. `details[0]`
    ///   is the finest level and `details[n_levels − 1]` is the coarsest detail.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if
    ///   `x.len() != in_channels * seq_len`.
    pub fn haar_decompose(&self, x: &[f32]) -> PinnResult<(Vec<f32>, Vec<Vec<f32>>)> {
        let in_c = self.cfg.in_channels;
        let n = self.cfg.seq_len;
        if x.len() != in_c * n {
            return Err(PinnError::DimensionMismatch {
                expected: in_c * n,
                got: x.len(),
            });
        }
        let mut current = x.to_vec();
        let mut current_len = n;
        let mut details = Vec::with_capacity(self.cfg.n_levels);
        for _ in 0..self.cfg.n_levels {
            let (approx_next, detail_next) = haar_step(&current, in_c, current_len);
            details.push(detail_next);
            current = approx_next;
            current_len /= 2;
        }
        Ok((current, details))
    }

    /// Multi-level inverse 1-D Haar reconstruction.
    ///
    /// Inputs must be exactly the structure returned by `haar_decompose`:
    /// `approx` of length `channels · seq_len / 2^n_levels` and `details` of
    /// length `n_levels`, with `details[l]` of length `channels · seq_len /
    /// 2^{l + 1}` where `channels` is inferred from `approx` and the
    /// configured `seq_len`/`n_levels`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if any buffer has an unexpected
    ///   length, or if `details.len() != n_levels`.
    pub fn haar_reconstruct(&self, approx: &[f32], details: &[Vec<f32>]) -> PinnResult<Vec<f32>> {
        let n_levels = self.cfg.n_levels;
        let seq_len = self.cfg.seq_len;
        if details.len() != n_levels {
            return Err(PinnError::DimensionMismatch {
                expected: n_levels,
                got: details.len(),
            });
        }
        let coarsest_len = seq_len >> n_levels;
        if coarsest_len == 0 {
            return Err(PinnError::InvalidGridResolution { n: coarsest_len });
        }
        if approx.len() % coarsest_len != 0 {
            return Err(PinnError::DimensionMismatch {
                expected: coarsest_len,
                got: approx.len(),
            });
        }
        let channels = approx.len() / coarsest_len;
        if channels == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }

        // Walk from coarsest detail to finest: details[n_levels - 1], ...,
        // details[0]. At each step the running length doubles.
        let mut current = approx.to_vec();
        let mut current_len = coarsest_len;
        for l in (0..n_levels).rev() {
            let expected_detail = channels * current_len;
            if details[l].len() != expected_detail {
                return Err(PinnError::DimensionMismatch {
                    expected: expected_detail,
                    got: details[l].len(),
                });
            }
            current = haar_step_inverse(&current, &details[l], channels, current_len);
            current_len *= 2;
        }
        if current_len != seq_len {
            return Err(PinnError::DimensionMismatch {
                expected: seq_len,
                got: current_len,
            });
        }
        Ok(current)
    }

    /// Forward pass: `(in_channels × seq_len)` → `(out_channels × seq_len)`,
    /// both row-major.
    ///
    /// Implements
    /// ```text
    /// û = inv_haar( W_a · approx, [W_l · detail_l]_l ) + W_res · x + b
    /// ```
    /// where `W_a` and each `W_l` are `(in_channels × out_channels)` channel
    /// mixing matrices and `W_res` is a 1×1 residual of shape
    /// `(out_channels × in_channels)`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `x.len() != in_channels · seq_len`.
    /// - [`PinnError::NanEncountered`] if a non-finite value is produced.
    pub fn forward(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        let in_c = self.cfg.in_channels;
        let out_c = self.cfg.out_channels;
        let n = self.cfg.seq_len;
        if x.len() != in_c * n {
            return Err(PinnError::DimensionMismatch {
                expected: in_c * n,
                got: x.len(),
            });
        }

        // 1. Multi-level Haar decompose.
        let (approx_in, details_in) = self.haar_decompose(x)?;

        // 2. Per-level (in × out) channel mixing.
        let coarsest_len = n >> self.cfg.n_levels;
        let approx_out = channel_linear(&approx_in, &self.approx_weight, in_c, out_c, coarsest_len);

        let mut details_out: Vec<Vec<f32>> = Vec::with_capacity(self.cfg.n_levels);
        // details_in[l] has length in_c · (seq_len >> (l+1)); index 0 is the
        // finest level, n_levels − 1 the coarsest.
        for (l, det_in) in details_in.iter().enumerate() {
            let len = n >> (l + 1);
            let mixed = channel_linear(det_in, &self.detail_weights[l], in_c, out_c, len);
            details_out.push(mixed);
        }

        // 3. Build a Wno-like inverse: reconstruct using the same multi-level
        // inverse pipeline but with `out_channels` channels.
        let mut current = approx_out;
        let mut current_len = coarsest_len;
        for l in (0..self.cfg.n_levels).rev() {
            current = haar_step_inverse(&current, &details_out[l], out_c, current_len);
            current_len *= 2;
        }
        if current_len != n {
            return Err(PinnError::DimensionMismatch {
                expected: n,
                got: current_len,
            });
        }
        let spectral_out = current;

        // 4. Residual: W · x per position (1×1 over channel axis) + bias.
        let mut output = vec![0.0_f32; out_c * n];
        for o in 0..out_c {
            let b = self.residual_b[o];
            for p in 0..n {
                let mut acc = b;
                for i in 0..in_c {
                    acc += self.residual_w[o * in_c + i] * x[i * n + p];
                }
                output[o * n + p] = spectral_out[o * n + p] + acc;
            }
        }

        for v in &output {
            if !v.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "wno::forward",
                });
            }
        }

        Ok(output)
    }
}

// ─── Test-only mutators ──────────────────────────────────────────────────────
//
// Available only in tests so unit tests can independently exercise the
// spectral path and the residual path (identity test, linearity test).

#[cfg(test)]
impl Wno {
    /// Zero every spectral weight (both detail levels and the approx level).
    pub(crate) fn zero_spectral(&mut self) {
        for level in &mut self.detail_weights {
            for v in level.iter_mut() {
                *v = 0.0;
            }
        }
        for v in &mut self.approx_weight {
            *v = 0.0;
        }
    }

    /// Zero the residual `W` and bias `b`.
    pub(crate) fn zero_residual(&mut self) {
        for v in &mut self.residual_w {
            *v = 0.0;
        }
        for v in &mut self.residual_b {
            *v = 0.0;
        }
    }

    /// Set the residual to the identity matrix `W = I`, `b = 0`. Requires
    /// `in_channels == out_channels`.
    pub(crate) fn residual_identity(&mut self) {
        self.zero_residual();
        let in_c = self.cfg.in_channels;
        let out_c = self.cfg.out_channels;
        let n = in_c.min(out_c);
        for k in 0..n {
            self.residual_w[k * in_c + k] = 1.0;
        }
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_cfg(cfg: &WnoConfig) -> PinnResult<()> {
    if cfg.in_channels == 0 || cfg.out_channels == 0 {
        return Err(PinnError::InvalidLayerWidth);
    }
    if !is_power_of_two(cfg.seq_len) {
        return Err(PinnError::InvalidGridResolution { n: cfg.seq_len });
    }
    if cfg.n_levels == 0 {
        return Err(PinnError::InvalidNetworkDepth { depth: 0 });
    }
    let max_levels = cfg.seq_len.trailing_zeros() as usize;
    if cfg.n_levels > max_levels {
        return Err(PinnError::TooManyFourierModes {
            k_max: cfg.n_levels,
            n_half: max_levels,
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> WnoConfig {
        WnoConfig {
            in_channels: 2,
            out_channels: 2,
            seq_len: 8,
            n_levels: 2,
        }
    }

    fn make(seed: u64, cfg: WnoConfig) -> Wno {
        let mut rng = LcgRng::new(seed);
        Wno::new(cfg, &mut rng).expect("Wno construction failed in test helper")
    }

    // ── Haar round-trip ─────────────────────────────────────────────────────

    #[test]
    fn haar_roundtrip_random_input() {
        let mut rng = LcgRng::new(1);
        let cfg = WnoConfig {
            in_channels: 3,
            out_channels: 1,
            seq_len: 16,
            n_levels: 3,
        };
        let wno = Wno::new(cfg.clone(), &mut rng)
            .expect("WNO construction with valid config should succeed");
        // Generate a random input independent of model weights.
        let mut x = vec![0.0_f32; cfg.in_channels * cfg.seq_len];
        for v in x.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let (approx, details) = wno
            .haar_decompose(&x)
            .expect("Haar decomposition should succeed for valid input");
        let back = wno
            .haar_reconstruct(&approx, &details)
            .expect("Haar reconstruction should succeed after decomposition");
        assert_eq!(back.len(), x.len());
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "Haar round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn haar_decompose_approx_length_correct() {
        let wno = make(2, default_cfg());
        let cfg = wno.config().clone();
        let x = vec![0.5_f32; cfg.in_channels * cfg.seq_len];
        let (approx, _details) = wno
            .haar_decompose(&x)
            .expect("Haar decomposition should succeed for constant input");
        let coarsest = cfg.seq_len >> cfg.n_levels;
        assert_eq!(approx.len(), cfg.in_channels * coarsest);
    }

    #[test]
    fn haar_decompose_details_lengths_shrink() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 32,
            n_levels: 4,
        };
        let wno = make(3, cfg.clone());
        let x = vec![1.0_f32; cfg.in_channels * cfg.seq_len];
        let (_a, details) = wno
            .haar_decompose(&x)
            .expect("Haar decomposition should return details at correct lengths");
        assert_eq!(details.len(), cfg.n_levels);
        // Index 0 is the finest level → length seq_len / 2.
        for (l, d) in details.iter().enumerate() {
            let expected = cfg.in_channels * (cfg.seq_len >> (l + 1));
            assert_eq!(d.len(), expected, "Detail level {l} length");
        }
    }

    #[test]
    fn haar_constant_input_has_zero_details() {
        // For a constant signal c, every detail (a − b)/√2 = 0.
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 16,
            n_levels: 3,
        };
        let wno = make(4, cfg.clone());
        let x = vec![1.25_f32; cfg.in_channels * cfg.seq_len];
        let (_a, details) = wno
            .haar_decompose(&x)
            .expect("Haar decomposition of constant input should produce zero details");
        for (l, d) in details.iter().enumerate() {
            for &v in d.iter() {
                assert!(
                    v.abs() < 1e-5,
                    "Detail level {l} should be 0 for constant input, got {v}"
                );
            }
        }
    }

    // ── Forward shape & finiteness ─────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let cfg = WnoConfig {
            in_channels: 2,
            out_channels: 3,
            seq_len: 8,
            n_levels: 2,
        };
        let wno = make(5, cfg.clone());
        let x = vec![0.1_f32; cfg.in_channels * cfg.seq_len];
        let y = wno
            .forward(&x)
            .expect("WNO forward pass should succeed for valid input");
        assert_eq!(y.len(), cfg.out_channels * cfg.seq_len);
    }

    #[test]
    fn forward_finite() {
        let wno = make(6, default_cfg());
        let cfg = wno.config().clone();
        let mut x = vec![0.0_f32; cfg.in_channels * cfg.seq_len];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f32) * 0.13).sin();
        }
        let y = wno
            .forward(&x)
            .expect("WNO forward pass should produce finite values");
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── Identity test (residual-only path) ─────────────────────────────────

    #[test]
    fn residual_identity_with_zero_spectral_is_identity() {
        let cfg = WnoConfig {
            in_channels: 2,
            out_channels: 2,
            seq_len: 8,
            n_levels: 2,
        };
        let mut wno = make(7, cfg.clone());
        wno.zero_spectral();
        wno.residual_identity();
        let mut x = vec![0.0_f32; cfg.in_channels * cfg.seq_len];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 5.0) * 0.3;
        }
        let y = wno
            .forward(&x)
            .expect("WNO forward with identity residual and zero spectral should succeed");
        for i in 0..x.len() {
            assert!(
                (y[i] - x[i]).abs() < 1e-5,
                "Identity violated at {i}: {} vs {}",
                y[i],
                x[i]
            );
        }
    }

    // ── Sensitivity to spectral weights ────────────────────────────────────

    #[test]
    fn changing_spectral_weights_changes_output() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 8,
            n_levels: 2,
        };
        let mut wno = make(8, cfg.clone());
        wno.zero_residual();
        // Pure constant input: details vanish, so only the approx-level
        // weight matters; bump it and the output magnitude must change.
        let x = vec![0.7_f32; cfg.in_channels * cfg.seq_len];
        let y0 = wno
            .forward(&x)
            .expect("WNO forward before spectral weight change should succeed");
        wno.approx_weight[0] += 1.5;
        let y1 = wno
            .forward(&x)
            .expect("WNO forward after spectral weight change should succeed");
        let diff: f32 = y0.iter().zip(y1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "Forward should change when spectral weights change; diff = {diff}"
        );
    }

    // ── n_params formula ───────────────────────────────────────────────────

    #[test]
    fn n_params_formula() {
        let cfg = WnoConfig {
            in_channels: 3,
            out_channels: 4,
            seq_len: 16,
            n_levels: 2,
        };
        let wno = make(9, cfg.clone());
        // (n_levels + 1) · in · out + (out · in) + out = 3 · 12 + 12 + 4 = 52.
        let expected = (cfg.n_levels + 1) * cfg.in_channels * cfg.out_channels
            + cfg.out_channels * cfg.in_channels
            + cfg.out_channels;
        assert_eq!(wno.n_params(), expected);
        assert!(wno.n_params() > 0);
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn deterministic_given_seed() {
        let cfg = WnoConfig {
            in_channels: 2,
            out_channels: 2,
            seq_len: 8,
            n_levels: 2,
        };
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        let a = Wno::new(cfg.clone(), &mut rng_a)
            .expect("WNO construction with seed 123 should succeed");
        let b = Wno::new(cfg.clone(), &mut rng_b)
            .expect("WNO construction with same seed 123 should succeed");
        let x: Vec<f32> = (0..cfg.in_channels * cfg.seq_len)
            .map(|i| (i as f32) * 0.07)
            .collect();
        let ya = a
            .forward(&x)
            .expect("WNO forward pass should be deterministic for given seed");
        let yb = b
            .forward(&x)
            .expect("WNO forward pass should produce identical result for same seed");
        for i in 0..ya.len() {
            assert!((ya[i] - yb[i]).abs() < 1e-8, "Determinism at {i}");
        }
    }

    // ── Linearity (zero residual) ──────────────────────────────────────────

    #[test]
    fn forward_is_linear_with_zero_residual_bias() {
        // The spectral path is linear in x; with bias = 0 the whole forward
        // is linear too. Confirm additivity and homogeneity.
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 8,
            n_levels: 2,
        };
        let mut wno = make(10, cfg.clone());
        // Zero the bias only; keep the residual W to exercise the linear
        // residual path together with the linear spectral path.
        for v in &mut wno.residual_b {
            *v = 0.0;
        }
        let len = cfg.in_channels * cfg.seq_len;
        let a: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.11).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.07).cos()).collect();
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let scaled: Vec<f32> = a.iter().map(|x| x * 2.5).collect();

        let ya = wno
            .forward(&a)
            .expect("WNO forward on input a should succeed");
        let yb = wno
            .forward(&b)
            .expect("WNO forward on input b should succeed");
        let ysum = wno
            .forward(&sum)
            .expect("WNO forward on sum of inputs should succeed");
        let yscaled = wno
            .forward(&scaled)
            .expect("WNO forward on scaled input should succeed");

        for i in 0..ya.len() {
            let lhs = ysum[i];
            let rhs = ya[i] + yb[i];
            assert!(
                (lhs - rhs).abs() < 1e-4,
                "Additivity violated at {i}: {} vs {}",
                lhs,
                rhs
            );
            let lhs2 = yscaled[i];
            let rhs2 = ya[i] * 2.5;
            assert!(
                (lhs2 - rhs2).abs() < 1e-4,
                "Homogeneity violated at {i}: {} vs {}",
                lhs2,
                rhs2
            );
        }
    }

    // ── Minimal-case ───────────────────────────────────────────────────────

    #[test]
    fn minimal_case_seq_len_two_one_level() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 2,
            n_levels: 1,
        };
        let wno = make(11, cfg.clone());
        let x = vec![1.0_f32, -1.0];
        let y = wno
            .forward(&x)
            .expect("WNO forward on minimal seq_len=2 input should succeed");
        assert_eq!(y.len(), 2);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── Errors ─────────────────────────────────────────────────────────────

    #[test]
    fn err_seq_len_not_power_of_two() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 6,
            n_levels: 1,
        };
        let mut rng = LcgRng::new(12);
        let r = Wno::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidGridResolution { .. })));
    }

    #[test]
    fn err_n_levels_too_large() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 8,
            n_levels: 4, // 2^4 = 16 > 8
        };
        let mut rng = LcgRng::new(13);
        let r = Wno::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::TooManyFourierModes { .. })));
    }

    #[test]
    fn err_in_channels_zero() {
        let cfg = WnoConfig {
            in_channels: 0,
            out_channels: 1,
            seq_len: 4,
            n_levels: 1,
        };
        let mut rng = LcgRng::new(14);
        let r = Wno::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidLayerWidth)));
    }

    #[test]
    fn err_out_channels_zero() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 0,
            seq_len: 4,
            n_levels: 1,
        };
        let mut rng = LcgRng::new(15);
        let r = Wno::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidLayerWidth)));
    }

    #[test]
    fn err_n_levels_zero() {
        let cfg = WnoConfig {
            in_channels: 1,
            out_channels: 1,
            seq_len: 4,
            n_levels: 0,
        };
        let mut rng = LcgRng::new(16);
        let r = Wno::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidNetworkDepth { .. })));
    }

    #[test]
    fn err_forward_wrong_length() {
        let wno = make(17, default_cfg());
        let r = wno.forward(&[0.0_f32; 3]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_forward_empty_input() {
        let wno = make(18, default_cfg());
        let r = wno.forward(&[]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }
}
