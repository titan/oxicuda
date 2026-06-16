//! StyleGAN3 alias-free generator operations.
//!
//! Implements the *alias-free* building blocks from Karras et al. (2021),
//! "Alias-Free Generative Adversarial Networks" (StyleGAN3, NeurIPS 2021).
//! The defining property of these blocks is **translation equivariance**:
//! shifting the (band-limited) input signal produces a correspondingly
//! shifted output, with no aliasing artefacts leaking across the
//! up/down-sampling boundary.
//!
//! ## 1-D reference
//!
//! For clarity and unit-testability the operations here work on **1-D**
//! signals (a single spatial axis).  This is sufficient to demonstrate the
//! alias-free / translation-equivariant property cleanly; the 2-D generator
//! used in the paper is the separable tensor product of these 1-D operators.
//!
//! All convolutions are *zero-phase circular* convolutions: the symmetric FIR
//! is centred on the current sample (introducing no net delay) and the signal
//! is treated periodically.  Under this convention every operator commutes
//! with an integer circular shift, so the *filtered nonlinearity*
//!
//! ```text
//!   x ─► upsample 2× ─► pointwise nonlinearity ─► downsample 2× ─► y
//! ```
//!
//! satisfies `F(shift(x, k)) = shift(F(x), k)` exactly for integer `k`, and to
//! a tight tolerance for fractional shifts of band-limited inputs (the
//! headline alias-free property).
//!
//! ## Why the nonlinearity is sandwiched between resampling
//!
//! A pointwise nonlinearity applied at the native rate generates high-frequency
//! harmonics that alias back into the signal band (the "texture sticking"
//! artefact StyleGAN3 fixes).  By temporarily upsampling, applying the
//! nonlinearity at the higher rate, then low-pass filtering before decimating,
//! the harmonics above the original Nyquist frequency are suppressed before
//! they can fold back, restoring equivariance.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── RNG helper (full-range uniform) ───────────────────────────────────────────

/// Draw a uniform sample in `[-scale, scale)`.
///
/// NOTE: the crate's [`LcgRng::next_f32`] only spans `[0, 0.5)` because
/// `next_u32` keeps the top 31 bits.  We therefore rebuild a true `[0, 1)`
/// uniform from the 31-bit integer directly (`/ 2^31`) and recentre it.
#[inline]
fn uniform_sym(rng: &mut LcgRng, scale: f32) -> f32 {
    let u = rng.next_u32() as f32 / 4_294_967_296.0_f32; // 2^31  → [0, 1)
    (u - 0.5) * 2.0 * scale
}

// ─── Activation ────────────────────────────────────────────────────────────────

/// Leaky ReLU pointwise nonlinearity.
#[inline]
#[must_use]
pub fn leaky_relu(v: f32, alpha: f32) -> f32 {
    if v >= 0.0 { v } else { alpha * v }
}

// ─── FIR filter design ──────────────────────────────────────────────────────────

/// Modified Bessel function of the first kind, order 0 (`I₀`).
///
/// Evaluated by its rapidly-converging power series
/// `I₀(x) = Σ_{m≥0} (x/2)^{2m} / (m!)²`.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for m in 1..=64 {
        let mf = f64::from(m);
        let ratio = half / mf;
        term *= ratio * ratio;
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Design a Kaiser-windowed-sinc low-pass FIR filter.
///
/// * `num_taps`  — requested length (rounded up to the next odd number, min 3).
/// * `cutoff`    — cutoff frequency in cycles/sample, `0 < cutoff < 0.5`.
/// * `beta`      — Kaiser window shape parameter (larger ⇒ wider main lobe but
///   deeper stop-band).
///
/// The returned taps are **normalised to unit DC gain** (they sum to `1`), so a
/// constant input is passed through unchanged.
///
/// # Errors
/// * [`GenError::EmptyInput`] if `num_taps == 0`.
/// * [`GenError::InvalidFlowTime`] (reused as a range error) if `cutoff` is not
///   in the open interval `(0, 0.5)`.
/// * [`GenError::Internal`] if the taps sum to ~0 (degenerate design).
pub fn kaiser_lowpass_fir(num_taps: usize, cutoff: f32, beta: f32) -> GenResult<Vec<f32>> {
    if num_taps == 0 {
        return Err(GenError::EmptyInput("num_taps must be > 0"));
    }
    if !(cutoff > 0.0 && cutoff < 0.5) {
        return Err(GenError::InvalidFlowTime(cutoff));
    }
    let mut n = num_taps.max(3);
    if n % 2 == 0 {
        n += 1;
    }
    let center = (n - 1) as f64 / 2.0;
    let fc = f64::from(cutoff);
    let beta = f64::from(beta);
    let i0_beta = bessel_i0(beta);

    let mut taps = vec![0.0_f32; n];
    for (i, tap) in taps.iter_mut().enumerate() {
        let x = i as f64 - center;
        // Ideal low-pass impulse response: sin(2π·fc·x) / (π·x), value 2·fc at x=0.
        let sinc = if x.abs() < 1e-12 {
            2.0 * fc
        } else {
            (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x)
        };
        // Kaiser window over the normalised position r = x / center ∈ [-1, 1].
        let r = x / center;
        let arg = 1.0 - r * r;
        let window = if arg <= 0.0 {
            0.0
        } else {
            bessel_i0(beta * arg.sqrt()) / i0_beta
        };
        *tap = (sinc * window) as f32;
    }

    let sum: f32 = taps.iter().sum();
    if sum.abs() < 1e-20 {
        return Err(GenError::Internal("degenerate FIR (zero DC gain)".into()));
    }
    for tap in &mut taps {
        *tap /= sum;
    }
    Ok(taps)
}

// ─── AliasFreeOps ────────────────────────────────────────────────────────────────

/// Bundle of alias-free resampling operators sharing one FIR low-pass filter.
///
/// The same symmetric low-pass (cutoff `0.25` cycles/sample, i.e. half the
/// original Nyquist frequency) is used both to suppress the imaging replicas
/// introduced by 2× zero-stuffing and to band-limit before 2× decimation.
#[derive(Debug, Clone)]
pub struct AliasFreeOps {
    taps: Vec<f32>,
    leaky_alpha: f32,
}

impl AliasFreeOps {
    /// Default cutoff (cycles/sample) for the 2× resampling filter.
    pub const DEFAULT_CUTOFF: f32 = 0.25;
    /// Default Kaiser window `β`.
    pub const DEFAULT_BETA: f32 = 8.0;

    /// Build with a `num_taps`-long Kaiser low-pass (cutoff `0.25`, `β = 8`).
    ///
    /// # Errors
    /// Propagates [`kaiser_lowpass_fir`] errors.
    pub fn new(num_taps: usize, leaky_alpha: f32) -> GenResult<Self> {
        Self::with_filter(
            num_taps,
            Self::DEFAULT_CUTOFF,
            Self::DEFAULT_BETA,
            leaky_alpha,
        )
    }

    /// Build with an explicit cutoff and Kaiser `β`.
    ///
    /// # Errors
    /// Propagates [`kaiser_lowpass_fir`] errors.
    pub fn with_filter(
        num_taps: usize,
        cutoff: f32,
        beta: f32,
        leaky_alpha: f32,
    ) -> GenResult<Self> {
        let taps = kaiser_lowpass_fir(num_taps, cutoff, beta)?;
        Ok(Self { taps, leaky_alpha })
    }

    /// Read-only view of the FIR taps (sum to `1`).
    #[must_use]
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    /// Leaky-ReLU negative slope.
    #[must_use]
    pub fn leaky_alpha(&self) -> f32 {
        self.leaky_alpha
    }

    /// Zero-phase **circular** convolution of `signal` with the FIR taps.
    ///
    /// The symmetric, odd-length filter is centred on the current sample, so
    /// the operation introduces no net delay and commutes with circular shifts.
    fn circular_conv(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let l = self.taps.len();
        let c = (l - 1) / 2;
        let n_isize = n as isize;
        let mut out = vec![0.0_f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (j, &tap) in self.taps.iter().enumerate() {
                let raw = i as isize + j as isize - c as isize;
                let idx = raw.rem_euclid(n_isize) as usize;
                acc += tap * signal[idx];
            }
            *o = acc;
        }
        out
    }

    /// 2× upsampling: insert a zero after every sample, then low-pass filter.
    ///
    /// A gain of `2` (the upsampling factor) is applied so that the DC level is
    /// preserved through zero-stuffing.  Output length is `2 · signal.len()`.
    ///
    /// # Errors
    /// [`GenError::EmptyInput`] if `signal` is empty.
    pub fn upsample_2x(&self, signal: &[f32]) -> GenResult<Vec<f32>> {
        if signal.is_empty() {
            return Err(GenError::EmptyInput("signal is empty"));
        }
        let n = signal.len();
        let mut stuffed = vec![0.0_f32; 2 * n];
        for (i, &s) in signal.iter().enumerate() {
            stuffed[2 * i] = s;
        }
        let mut filtered = self.circular_conv(&stuffed);
        for v in &mut filtered {
            *v *= 2.0;
        }
        Ok(filtered)
    }

    /// 2× downsampling: low-pass filter (anti-alias) then keep every 2nd sample.
    ///
    /// Output length is `signal.len() / 2`.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `signal` is empty.
    /// * [`GenError::DimensionMismatch`] if `signal.len()` is odd.
    pub fn downsample_2x(&self, signal: &[f32]) -> GenResult<Vec<f32>> {
        if signal.is_empty() {
            return Err(GenError::EmptyInput("signal is empty"));
        }
        if signal.len() % 2 != 0 {
            return Err(GenError::DimensionMismatch {
                expected: signal.len() + 1,
                got: signal.len(),
            });
        }
        let filtered = self.circular_conv(signal);
        let out = filtered.iter().step_by(2).copied().collect();
        Ok(out)
    }

    /// The alias-free **filtered nonlinearity** block.
    ///
    /// `upsample 2× → leaky-ReLU at the higher rate → downsample 2×`.
    /// Input and output have the same length.
    ///
    /// # Errors
    /// [`GenError::EmptyInput`] if `signal` is empty.
    pub fn filtered_nonlinearity(&self, signal: &[f32]) -> GenResult<Vec<f32>> {
        let up = self.upsample_2x(signal)?;
        let alpha = self.leaky_alpha;
        let activated: Vec<f32> = up.iter().map(|&v| leaky_relu(v, alpha)).collect();
        self.downsample_2x(&activated)
    }
}

// ─── StyleGan3Config ─────────────────────────────────────────────────────────────

/// Configuration for the (1-D reference) StyleGAN3 generator.
#[derive(Debug, Clone)]
pub struct StyleGan3Config {
    /// Input latent `z` dimensionality.
    pub z_dim: usize,
    /// Conditioning label `c` dimensionality (`0` ⇒ unconditional).
    pub c_dim: usize,
    /// Intermediate latent `w` dimensionality.
    pub w_dim: usize,
    /// Number of fully-connected layers in the mapping network (`≥ 1`).
    pub num_mapping_layers: usize,
    /// Feature channels carried through the synthesis network.
    pub num_channels: usize,
    /// Length of the 1-D feature signal.
    pub signal_length: usize,
    /// FIR filter length for the alias-free resampling (`≥ 3`).
    pub filter_taps: usize,
    /// Leaky-ReLU negative slope.
    pub leaky_alpha: f32,
    /// Number of synthesis layers.
    pub num_synthesis_layers: usize,
}

impl Default for StyleGan3Config {
    fn default() -> Self {
        Self {
            z_dim: 16,
            c_dim: 0,
            w_dim: 16,
            num_mapping_layers: 2,
            num_channels: 4,
            signal_length: 32,
            filter_taps: 13,
            leaky_alpha: 0.2,
            num_synthesis_layers: 2,
        }
    }
}

impl StyleGan3Config {
    /// Validate that all dimensions are usable.
    ///
    /// # Errors
    /// [`GenError::EmptyInput`] if any required dimension is zero / too small.
    pub fn validate(&self) -> GenResult<()> {
        if self.z_dim == 0 {
            return Err(GenError::EmptyInput("z_dim must be > 0"));
        }
        if self.w_dim == 0 {
            return Err(GenError::EmptyInput("w_dim must be > 0"));
        }
        if self.num_mapping_layers == 0 {
            return Err(GenError::EmptyInput("num_mapping_layers must be > 0"));
        }
        if self.num_channels == 0 {
            return Err(GenError::EmptyInput("num_channels must be > 0"));
        }
        if self.signal_length < 2 {
            return Err(GenError::EmptyInput("signal_length must be >= 2"));
        }
        if self.filter_taps < 3 {
            return Err(GenError::EmptyInput("filter_taps must be >= 3"));
        }
        if self.num_synthesis_layers == 0 {
            return Err(GenError::EmptyInput("num_synthesis_layers must be > 0"));
        }
        Ok(())
    }
}

// ─── MappingNetwork ──────────────────────────────────────────────────────────────

/// Mapping network `z (+ c) ↦ w`.
///
/// A small MLP with leaky-ReLU activations.  The input latent is first
/// normalised onto the unit hypersphere (PixelNorm), matching StyleGAN.
#[derive(Debug, Clone)]
pub struct MappingNetwork {
    /// Per-layer weights, row-major `[out × in]`.
    weights: Vec<Vec<f32>>,
    /// Per-layer biases `[out]`.
    biases: Vec<Vec<f32>>,
    z_dim: usize,
    c_dim: usize,
    w_dim: usize,
    leaky_alpha: f32,
}

impl MappingNetwork {
    /// Build a mapping network with random weights.
    ///
    /// # Errors
    /// Propagates [`StyleGan3Config::validate`].
    pub fn new(config: &StyleGan3Config, rng: &mut LcgRng) -> GenResult<Self> {
        config.validate()?;
        let mut weights = Vec::with_capacity(config.num_mapping_layers);
        let mut biases = Vec::with_capacity(config.num_mapping_layers);
        let mut in_dim = config.z_dim + config.c_dim;
        for _ in 0..config.num_mapping_layers {
            let out_dim = config.w_dim;
            let scale = 1.0 / (in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| uniform_sym(rng, scale))
                .collect();
            let b = vec![0.0_f32; out_dim];
            weights.push(w);
            biases.push(b);
            in_dim = out_dim;
        }
        Ok(Self {
            weights,
            biases,
            z_dim: config.z_dim,
            c_dim: config.c_dim,
            w_dim: config.w_dim,
            leaky_alpha: config.leaky_alpha,
        })
    }

    /// Map `z` (and optional label `c`) to the intermediate latent `w`.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `z` / `c` lengths disagree with the
    /// configured dimensions.
    pub fn forward(&self, z: &[f32], c: &[f32]) -> GenResult<Vec<f32>> {
        if z.len() != self.z_dim {
            return Err(GenError::DimensionMismatch {
                expected: self.z_dim,
                got: z.len(),
            });
        }
        if c.len() != self.c_dim {
            return Err(GenError::DimensionMismatch {
                expected: self.c_dim,
                got: c.len(),
            });
        }
        // PixelNorm on z: z ← z / sqrt(mean(z²) + ε).
        let mean_sq = z.iter().map(|&v| v * v).sum::<f32>() / (self.z_dim as f32);
        let inv = 1.0 / (mean_sq + 1e-8).sqrt();
        let mut activ: Vec<f32> = z.iter().map(|&v| v * inv).collect();
        activ.extend_from_slice(c);

        let last = self.weights.len() - 1;
        for (layer, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let in_dim = activ.len();
            let out_dim = b.len();
            let mut next = vec![0.0_f32; out_dim];
            for (o, (next_o, &bias)) in next.iter_mut().zip(b).enumerate() {
                let mut acc = bias;
                let row = &w[o * in_dim..(o + 1) * in_dim];
                for (&wi, &xi) in row.iter().zip(&activ) {
                    acc += wi * xi;
                }
                // Apply leaky-ReLU on every layer except the final projection.
                *next_o = if layer == last {
                    acc
                } else {
                    leaky_relu(acc, self.leaky_alpha)
                };
            }
            activ = next;
        }
        Ok(activ)
    }

    /// Intermediate latent dimensionality.
    #[must_use]
    pub fn w_dim(&self) -> usize {
        self.w_dim
    }
}

// ─── SynthesisLayer ──────────────────────────────────────────────────────────────

/// A single alias-free synthesis layer.
///
/// Performs a style-modulated `1×1` channel-mixing convolution with weight
/// demodulation, followed by the alias-free filtered nonlinearity on every
/// output channel.  The style is an affine projection of `w`.
#[derive(Debug, Clone)]
pub struct SynthesisLayer {
    in_channels: usize,
    out_channels: usize,
    length: usize,
    /// Style affine `w ↦ per-input-channel scale`, row-major `[in_channels × w_dim]`.
    affine_w: Vec<f32>,
    /// Style affine bias `[in_channels]` (initialised to `1` ⇒ unit style).
    affine_b: Vec<f32>,
    /// `1×1` convolution weights, row-major `[out_channels × in_channels]`.
    conv_w: Vec<f32>,
    /// Shared alias-free operators.
    ops: AliasFreeOps,
    w_dim: usize,
}

impl SynthesisLayer {
    /// Build a synthesis layer with random weights.
    ///
    /// # Errors
    /// [`GenError::EmptyInput`] on zero dimensions; propagates FIR design errors.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        config: &StyleGan3Config,
        rng: &mut LcgRng,
    ) -> GenResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(GenError::EmptyInput("channel counts must be > 0"));
        }
        config.validate()?;
        let w_dim = config.w_dim;
        let affine_scale = 1.0 / (w_dim as f32).sqrt();
        let affine_w: Vec<f32> = (0..in_channels * w_dim)
            .map(|_| uniform_sym(rng, affine_scale))
            .collect();
        let affine_b = vec![1.0_f32; in_channels];
        let conv_scale = 1.0 / (in_channels as f32).sqrt();
        let conv_w: Vec<f32> = (0..out_channels * in_channels)
            .map(|_| uniform_sym(rng, conv_scale))
            .collect();
        let ops = AliasFreeOps::new(config.filter_taps, config.leaky_alpha)?;
        Ok(Self {
            in_channels,
            out_channels,
            length: config.signal_length,
            affine_w,
            affine_b,
            conv_w,
            ops,
            w_dim,
        })
    }

    /// Forward pass.
    ///
    /// * `features` — input signal, row-major `[in_channels × length]`.
    /// * `w`        — intermediate latent `[w_dim]`.
    ///
    /// Returns `[out_channels × length]`.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] on shape mismatch.
    pub fn forward(&self, features: &[f32], w: &[f32]) -> GenResult<Vec<f32>> {
        if features.len() != self.in_channels * self.length {
            return Err(GenError::DimensionMismatch {
                expected: self.in_channels * self.length,
                got: features.len(),
            });
        }
        if w.len() != self.w_dim {
            return Err(GenError::DimensionMismatch {
                expected: self.w_dim,
                got: w.len(),
            });
        }

        // 1. Style: per-input-channel scale s_c = affine_w · w + affine_b.
        let mut style = vec![0.0_f32; self.in_channels];
        for (c, (style_c, &bias)) in style.iter_mut().zip(&self.affine_b).enumerate() {
            let row = &self.affine_w[c * self.w_dim..(c + 1) * self.w_dim];
            let mut acc = bias;
            for (&wi, &si) in row.iter().zip(w) {
                acc += wi * si;
            }
            *style_c = acc;
        }

        // 2. Weight modulation + demodulation (StyleGAN2/3): for output channel o
        //    the effective weight is W[o,c]·s_c, demodulated by its L2 norm.
        let mut out = vec![0.0_f32; self.out_channels * self.length];
        for o in 0..self.out_channels {
            let conv_row = &self.conv_w[o * self.in_channels..(o + 1) * self.in_channels];
            let mut norm_sq = 0.0_f32;
            for (&wc, &sc) in conv_row.iter().zip(&style) {
                let m = wc * sc;
                norm_sq += m * m;
            }
            let demod = 1.0 / (norm_sq + 1e-8).sqrt();
            for t in 0..self.length {
                let mut acc = 0.0_f32;
                for (c, (&wc, &sc)) in conv_row.iter().zip(&style).enumerate() {
                    acc += (wc * sc) * features[c * self.length + t];
                }
                out[o * self.length + t] = acc * demod;
            }
        }

        // 3. Alias-free filtered nonlinearity, per output channel.
        let mut result = vec![0.0_f32; self.out_channels * self.length];
        for o in 0..self.out_channels {
            let channel = &out[o * self.length..(o + 1) * self.length];
            let activated = self.ops.filtered_nonlinearity(channel)?;
            result[o * self.length..(o + 1) * self.length].copy_from_slice(&activated);
        }
        Ok(result)
    }

    /// Output channel count.
    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Signal length.
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
}

// ─── StyleGan3Generator ──────────────────────────────────────────────────────────

/// End-to-end (1-D reference) StyleGAN3 generator stub.
///
/// Maps a latent `z` to a style `w`, then runs a stack of alias-free synthesis
/// layers (all with `num_channels` in/out) starting from a learned constant /
/// supplied input signal.
#[derive(Debug, Clone)]
pub struct StyleGan3Generator {
    mapping: MappingNetwork,
    layers: Vec<SynthesisLayer>,
    num_channels: usize,
    signal_length: usize,
}

impl StyleGan3Generator {
    /// Build the generator with random weights.
    ///
    /// # Errors
    /// Propagates configuration / sub-module construction errors.
    pub fn new(config: &StyleGan3Config, rng: &mut LcgRng) -> GenResult<Self> {
        config.validate()?;
        let mapping = MappingNetwork::new(config, rng)?;
        let mut layers = Vec::with_capacity(config.num_synthesis_layers);
        for _ in 0..config.num_synthesis_layers {
            layers.push(SynthesisLayer::new(
                config.num_channels,
                config.num_channels,
                config,
                rng,
            )?);
        }
        Ok(Self {
            mapping,
            layers,
            num_channels: config.num_channels,
            signal_length: config.signal_length,
        })
    }

    /// Generate features from latent `z`, label `c`, and an input feature map.
    ///
    /// `init_features` is row-major `[num_channels × signal_length]`.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] on shape mismatch; propagates layer errors.
    pub fn forward(&self, z: &[f32], c: &[f32], init_features: &[f32]) -> GenResult<Vec<f32>> {
        let expected = self.num_channels * self.signal_length;
        if init_features.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: init_features.len(),
            });
        }
        let w = self.mapping.forward(z, c)?;
        let mut features = init_features.to_vec();
        for layer in &self.layers {
            features = layer.forward(&features, &w)?;
        }
        Ok(features)
    }

    /// Number of feature channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn ops() -> AliasFreeOps {
        AliasFreeOps::new(21, 0.2)
            .expect("AliasFreeOps::new should succeed with 21 taps and leaky_alpha 0.2")
    }

    /// A length-`n` band-limited signal: a sum of low-frequency cosines whose
    /// integer cycle counts keep it exactly periodic.  Evaluated at a continuous
    /// offset `delta` so the same routine yields the exact continuous shift.
    fn band_limited(n: usize, comps: &[(usize, f32, f32)], delta: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 - delta;
                comps
                    .iter()
                    .map(|&(k, amp, phase)| {
                        amp * (2.0 * PI * (k as f32) * t / (n as f32) + phase).cos()
                    })
                    .sum()
            })
            .collect()
    }

    /// Integer circular shift by `k` (positive ⇒ rightwards).
    fn circ_shift(signal: &[f32], k: usize) -> Vec<f32> {
        let n = signal.len();
        (0..n).map(|i| signal[(i + n - (k % n)) % n]).collect()
    }

    /// DFT-based continuous (fractional) circular shift of an arbitrary signal.
    fn spectral_shift(signal: &[f32], delta: f32) -> Vec<f32> {
        let n = signal.len();
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for k in 0..n {
            for (t, &s) in signal.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                re[k] += f64::from(s) * ang.cos();
                im[k] += f64::from(s) * ang.sin();
            }
        }
        for k in 0..n {
            // Symmetric frequency index so the shift stays a real operation.
            let kk = if k <= n / 2 {
                k as f64
            } else {
                k as f64 - n as f64
            };
            let ph = -2.0 * std::f64::consts::PI * kk * f64::from(delta) / (n as f64);
            let (c, s) = (ph.cos(), ph.sin());
            let nr = re[k] * c - im[k] * s;
            let ni = re[k] * s + im[k] * c;
            re[k] = nr;
            im[k] = ni;
        }
        let mut out = vec![0.0_f32; n];
        for (t, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for k in 0..n {
                let ang = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                acc += re[k] * ang.cos() - im[k] * ang.sin();
            }
            *o = (acc / n as f64) as f32;
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn fir_taps_sum_to_one() {
        let taps = kaiser_lowpass_fir(21, 0.25, 8.0)
            .expect("kaiser_lowpass_fir should succeed with 21 taps, cutoff 0.25, beta 8.0");
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "FIR DC gain should be 1: {sum}");
    }

    #[test]
    fn fir_is_symmetric_and_odd() {
        let taps = kaiser_lowpass_fir(20, 0.25, 8.0)
            .expect("kaiser_lowpass_fir should succeed with 20 taps (rounded to odd), cutoff 0.25, beta 8.0");
        assert_eq!(taps.len() % 2, 1, "taps length should be odd");
        let l = taps.len();
        for i in 0..l / 2 {
            assert!(
                (taps[i] - taps[l - 1 - i]).abs() < 1e-6,
                "filter must be symmetric"
            );
        }
    }

    #[test]
    fn fir_rejects_bad_cutoff() {
        assert!(kaiser_lowpass_fir(21, 0.0, 8.0).is_err());
        assert!(kaiser_lowpass_fir(21, 0.5, 8.0).is_err());
        assert!(kaiser_lowpass_fir(21, 0.6, 8.0).is_err());
        assert!(kaiser_lowpass_fir(0, 0.25, 8.0).is_err());
    }

    #[test]
    fn upsample_downsample_shapes() {
        let o = ops();
        let x = vec![0.3_f32; 32];
        let up = o
            .upsample_2x(&x)
            .expect("upsample_2x should succeed on a non-empty 32-element signal");
        assert_eq!(up.len(), 64);
        let down = o
            .downsample_2x(&up)
            .expect("downsample_2x should succeed on the even-length upsampled signal");
        assert_eq!(down.len(), 32);
        assert!(o.downsample_2x(&x[..31]).is_err(), "odd length must error");
    }

    #[test]
    fn upsample_preserves_dc() {
        // Zero-stuffing halves the mean; the ×2 gain restores it.
        let o = ops();
        let x = vec![1.5_f32; 32];
        let up = o
            .upsample_2x(&x)
            .expect("upsample_2x should succeed on a non-empty constant DC signal");
        let mean = up.iter().sum::<f32>() / up.len() as f32;
        assert!((mean - 1.5).abs() < 1e-3, "DC not preserved: mean={mean}");
    }

    #[test]
    fn up_then_down_is_near_identity_for_band_limited() {
        let o = ops();
        let n = 32;
        let comps = [(1_usize, 0.6_f32, 0.4_f32), (2, 0.3, 1.1)];
        let x = band_limited(n, &comps, 0.0);
        let up = o
            .upsample_2x(&x)
            .expect("upsample_2x should succeed on a non-empty band-limited signal");
        let recon = o
            .downsample_2x(&up)
            .expect("downsample_2x should succeed on the even-length upsampled output");
        let err = max_abs_diff(&x, &recon);
        assert!(err < 5e-2, "up→down should ≈ identity, max err={err}");
    }

    #[test]
    fn filtered_nonlinearity_shape_and_finite() {
        let o = ops();
        let x = band_limited(32, &[(1, 0.8, 0.0)], 0.0);
        let y = o
            .filtered_nonlinearity(&x)
            .expect("filtered_nonlinearity should succeed on a non-empty band-limited signal");
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn translation_equivariance_integer_shift() {
        // Headline property: F(shift(x, k)) == shift(F(x), k) for integer k.
        // Exact (to float) under zero-phase circular convolution.
        let o = ops();
        let n = 32;
        let comps = [(1_usize, 0.7_f32, 0.2_f32), (3, 0.25, 0.9)];
        let x = band_limited(n, &comps, 0.0);
        let y = o.filtered_nonlinearity(&x)
            .expect("filtered_nonlinearity should succeed on the base band-limited signal for integer-shift equivariance test");
        for k in [1_usize, 5, 11, 16] {
            let x_shift = circ_shift(&x, k);
            let y_of_shift = o.filtered_nonlinearity(&x_shift).expect(
                "filtered_nonlinearity should succeed on the integer-shifted band-limited input",
            );
            let shift_of_y = circ_shift(&y, k);
            let err = max_abs_diff(&y_of_shift, &shift_of_y);
            assert!(
                err < 1e-4,
                "integer-shift equivariance broken (k={k}): err={err}"
            );
        }
    }

    #[test]
    fn translation_equivariance_fractional_shift() {
        // The alias-free claim: for a band-limited input, a *fractional* shift of
        // the input yields the same fractional shift of the output.
        let o = AliasFreeOps::new(31, 0.2)
            .expect("AliasFreeOps::new should succeed with 31 taps and leaky_alpha 0.2 for fractional-shift test");
        let n = 32;
        // Single low-frequency tone keeps the leaky-ReLU harmonics inside the
        // pass-band so they are not aliased on decimation.
        let comps = [(1_usize, 0.9_f32, 0.3_f32)];
        let x = band_limited(n, &comps, 0.0);
        let y = o.filtered_nonlinearity(&x)
            .expect("filtered_nonlinearity should succeed on the base band-limited signal for fractional-shift equivariance test");
        for &delta in &[0.5_f32, 1.5, 0.25] {
            let x_shift = band_limited(n, &comps, delta); // exact continuous shift of x
            let y_of_shift = o.filtered_nonlinearity(&x_shift)
                .expect("filtered_nonlinearity should succeed on the continuously-shifted band-limited input");
            let shift_of_y = spectral_shift(&y, delta);
            let err = max_abs_diff(&y_of_shift, &shift_of_y);
            assert!(
                err < 3e-2,
                "fractional-shift equivariance broken (δ={delta}): err={err}"
            );
        }
    }

    #[test]
    fn leaky_relu_basic() {
        assert!((leaky_relu(2.0, 0.2) - 2.0).abs() < 1e-7);
        assert!((leaky_relu(-2.0, 0.2) - (-0.4)).abs() < 1e-7);
        assert!((leaky_relu(0.0, 0.2)).abs() < 1e-7);
    }

    #[test]
    fn mapping_network_shape_and_finite() {
        let cfg = StyleGan3Config::default();
        let mut rng = LcgRng::new(1);
        let net = MappingNetwork::new(&cfg, &mut rng)
            .expect("MappingNetwork::new should succeed with default StyleGan3Config");
        let z = vec![0.5_f32; cfg.z_dim];
        let w = net.forward(&z, &[])
            .expect("MappingNetwork::forward should succeed with correctly-sized z and empty label for unconditional config");
        assert_eq!(w.len(), cfg.w_dim);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mapping_network_conditional() {
        let cfg = StyleGan3Config {
            c_dim: 3,
            ..StyleGan3Config::default()
        };
        let mut rng = LcgRng::new(2);
        let net = MappingNetwork::new(&cfg, &mut rng).expect("new should succeed");
        let z = vec![0.1_f32; cfg.z_dim];
        let c = vec![0.7_f32; cfg.c_dim];
        let w = net.forward(&z, &c).expect("forward should succeed");
        assert_eq!(w.len(), cfg.w_dim);
        assert!(net.forward(&z, &[]).is_err(), "missing label must error");
    }

    #[test]
    fn synthesis_layer_shape_and_finite() {
        let cfg = StyleGan3Config::default();
        let mut rng = LcgRng::new(3);
        let layer = SynthesisLayer::new(cfg.num_channels, cfg.num_channels, &cfg, &mut rng)
            .expect("new should succeed");
        let mut rng2 = LcgRng::new(4);
        let mut feats = vec![0.0_f32; cfg.num_channels * cfg.signal_length];
        for f in &mut feats {
            *f = uniform_sym(&mut rng2, 1.0);
        }
        let w = vec![0.2_f32; cfg.w_dim];
        let out = layer.forward(&feats, &w).expect("forward should succeed");
        assert_eq!(out.len(), cfg.num_channels * cfg.signal_length);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn generator_end_to_end() {
        let cfg = StyleGan3Config::default();
        let mut rng = LcgRng::new(5);
        let generator = StyleGan3Generator::new(&cfg, &mut rng).expect("new should succeed");
        let z = vec![0.3_f32; cfg.z_dim];
        let init = vec![0.1_f32; cfg.num_channels * cfg.signal_length];
        let out = generator
            .forward(&z, &[], &init)
            .expect("forward should succeed");
        assert_eq!(out.len(), cfg.num_channels * cfg.signal_length);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(generator.num_channels(), cfg.num_channels);
    }

    #[test]
    fn config_validation() {
        let bad = StyleGan3Config {
            z_dim: 0,
            ..StyleGan3Config::default()
        };
        assert!(bad.validate().is_err());
        let bad2 = StyleGan3Config {
            filter_taps: 2,
            ..StyleGan3Config::default()
        };
        assert!(bad2.validate().is_err());
    }
}
