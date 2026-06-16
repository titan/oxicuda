//! Mel-spectrogram and Mel-Frequency Cepstral Coefficients (MFCC).
//!
//! This module computes the classic ASR front-end from a raw waveform:
//!
//! ```text
//! signal → framing (Hann) → magnitude DFT → mel filterbank → log → DCT-II → MFCC
//! ```
//!
//! The mel filterbank is constructed via [`super::mel_filterbank::MelFilterbank`]
//! (triangular, HTK 2595·log10 convention). The cepstral transform is a
//! type-II DCT (orthonormal) keeping the lowest `n_mfcc` coefficients.
//!
//! ## Stages
//!
//! 1. **Framing** — `(n_samples - n_fft) / hop_length + 1` Hann-windowed frames
//!    (zero frames when `n_samples < n_fft`).
//! 2. **Magnitude spectrum** — direct DFT (O(N²) per frame, pure-Rust, no FFT
//!    crate); `|X(t, k)|` for `k ∈ [0, n_fft/2]`.
//! 3. **Mel energies** — triangular mel filterbank applied to the magnitude (or
//!    power) spectrum.
//! 4. **Log compression** — `log(mel + ε)` (natural log).
//! 5. **DCT-II** — orthonormal type-II DCT over the mel axis, keeping the first
//!    `n_mfcc` coefficients.
//!
//! ## References
//! - Davis, S. & Mermelstein, P. (1980). "Comparison of parametric
//!   representations for monosyllabic word recognition." IEEE TASSP 28(4).
//! - Slaney, M. (1998). "Auditory Toolbox v2."

use std::f32::consts::PI;

use crate::error::{AudioError, AudioResult};
use crate::features::mel_filterbank::{MelFilterbank, MelFilterbankConfig};

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`mfcc`] / [`mel_spectrogram`].
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Audio sample rate in Hz (must be > 0).
    pub sample_rate: f32,
    /// FFT length (frame length); must be ≥ 2.
    pub n_fft: usize,
    /// Hop length between successive frames in samples; must be ≥ 1.
    pub hop_length: usize,
    /// Number of mel filters; must be ≥ 1.
    pub n_mels: usize,
    /// Number of cepstral coefficients to keep; must satisfy `1 ≤ n_mfcc ≤ n_mels`.
    pub n_mfcc: usize,
    /// Lower frequency cutoff in Hz (≥ 0).
    pub f_min: f32,
    /// Upper frequency cutoff in Hz (`f_min < f_max ≤ sample_rate/2`).
    pub f_max: f32,
    /// If `true`, use the power spectrum `|X|²`; if `false`, the magnitude `|X|`.
    pub use_power: bool,
    /// Floor added before the logarithm for numerical stability (`> 0`).
    pub log_floor: f32,
}

impl MfccConfig {
    /// A small default suitable for tests and 16 kHz speech.
    #[must_use]
    pub fn default_16k() -> Self {
        Self {
            sample_rate: 16_000.0,
            n_fft: 400,
            hop_length: 160,
            n_mels: 40,
            n_mfcc: 13,
            f_min: 0.0,
            f_max: 8_000.0,
            use_power: true,
            log_floor: 1e-10,
        }
    }
}

// ─── Mel filterbank helper ──────────────────────────────────────────────────────

/// Construct the triangular mel filterbank described by `cfg`.
///
/// This is a thin convenience wrapper exposing the filterbank used internally by
/// [`mfcc`] / [`mel_spectrogram`].
///
/// # Errors
/// Propagates [`MelFilterbank::new`] validation errors (invalid sample rate,
/// `n_mels == 0`, `n_fft < 2`, or frequency-range violations).
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> AudioResult<MelFilterbank> {
    MelFilterbank::new(MelFilterbankConfig {
        sample_rate,
        n_fft,
        n_mels,
        f_min,
        f_max,
    })
}

// ─── Framing + DFT ──────────────────────────────────────────────────────────────

/// Number of frames produced for `n_samples` with the given window/hop.
#[must_use]
#[inline]
pub fn frame_count(n_samples: usize, n_fft: usize, hop_length: usize) -> usize {
    if hop_length == 0 || n_samples < n_fft {
        0
    } else {
        (n_samples - n_fft) / hop_length + 1
    }
}

/// Hann window of length `n` (`w[k] = 0.5·(1 − cos(2πk/(n−1)))`, `[1.0]` for `n == 1`).
fn hann_window(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0_f32];
    }
    (0..n)
        .map(|k| 0.5_f32 * (1.0_f32 - (2.0_f32 * PI * k as f32 / (n - 1) as f32).cos()))
        .collect()
}

/// Compute the magnitude (or power) spectrogram `[n_frames, n_fft/2 + 1]`.
fn magnitude_spectrogram(
    signal: &[f32],
    n_samples: usize,
    n_fft: usize,
    hop_length: usize,
    use_power: bool,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let n_frames = frame_count(n_samples, n_fft, hop_length);
    let window = hann_window(n_fft);
    let mut spec = vec![0.0_f32; n_frames * n_bins];

    for frame in 0..n_frames {
        let start = frame * hop_length;
        let row = &mut spec[frame * n_bins..(frame + 1) * n_bins];
        for (k, out_v) in row.iter_mut().enumerate() {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let omega = -2.0_f32 * PI * k as f32 / n_fft as f32;
            for (j, &w) in window.iter().enumerate() {
                let sample = signal[start + j] * w;
                let angle = omega * j as f32;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }
            let power = re * re + im * im;
            *out_v = if use_power { power } else { power.sqrt() };
        }
    }
    spec
}

// ─── DCT-II ─────────────────────────────────────────────────────────────────────

/// Build an orthonormal type-II DCT matrix `[n_mfcc, n_mels]` (row-major).
///
/// `D[c, m] = α(c) · cos(π·(2m+1)·c / (2·n_mels))` with `α(0) = √(1/n_mels)` and
/// `α(c>0) = √(2/n_mels)`. This is the DCT-II "ortho" normalisation used by
/// librosa / scipy.
fn dct2_matrix(n_mfcc: usize, n_mels: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; n_mfcc * n_mels];
    let norm0 = (1.0_f32 / n_mels as f32).sqrt();
    let normk = (2.0_f32 / n_mels as f32).sqrt();
    for c in 0..n_mfcc {
        let alpha = if c == 0 { norm0 } else { normk };
        let row_off = c * n_mels;
        for m in 0..n_mels {
            let angle = PI * (2 * m + 1) as f32 * c as f32 / (2 * n_mels) as f32;
            mat[row_off + m] = alpha * angle.cos();
        }
    }
    mat
}

// ─── Validation ─────────────────────────────────────────────────────────────────

fn validate(cfg: &MfccConfig, n_samples: usize) -> AudioResult<()> {
    if cfg.n_fft < 2 {
        return Err(AudioError::ShapeMismatch {
            msg: format!("mfcc: n_fft must be ≥ 2, got {}", cfg.n_fft),
        });
    }
    if cfg.hop_length == 0 {
        return Err(AudioError::InvalidStride(0));
    }
    if cfg.n_mels == 0 {
        return Err(AudioError::InvalidNumMels(0));
    }
    if cfg.n_mfcc == 0 || cfg.n_mfcc > cfg.n_mels {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "mfcc: n_mfcc ({}) must satisfy 1 ≤ n_mfcc ≤ n_mels ({})",
                cfg.n_mfcc, cfg.n_mels
            ),
        });
    }
    if n_samples == 0 {
        return Err(AudioError::EmptyInput {
            msg: "mfcc: empty signal".into(),
        });
    }
    if n_samples < cfg.n_fft {
        return Err(AudioError::InvalidSequenceLength(n_samples));
    }
    if cfg.log_floor <= 0.0 {
        return Err(AudioError::Internal(format!(
            "mfcc: log_floor must be > 0, got {}",
            cfg.log_floor
        )));
    }
    Ok(())
}

// ─── Public API ─────────────────────────────────────────────────────────────────

/// Compute the **log-mel spectrogram** `[n_frames, n_mels]` (row-major).
///
/// Frames the `signal` with a Hann window, computes the magnitude (or power)
/// spectrum via direct DFT, projects onto the triangular mel filterbank, and
/// applies `log(mel + log_floor)`.
///
/// # Errors
/// - [`AudioError::ShapeMismatch`] on `n_fft < 2` / `n_mfcc` out of range.
/// - [`AudioError::InvalidStride`] on `hop_length == 0`.
/// - [`AudioError::InvalidNumMels`] on `n_mels == 0`.
/// - [`AudioError::EmptyInput`] on an empty signal.
/// - [`AudioError::InvalidSequenceLength`] when `signal.len() < n_fft`.
/// - Filterbank construction errors from [`mel_filterbank`].
pub fn log_mel_spectrogram(signal: &[f32], cfg: &MfccConfig) -> AudioResult<Vec<f32>> {
    let n_samples = signal.len();
    validate(cfg, n_samples)?;

    let fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.f_min, cfg.f_max)?;
    let spec = magnitude_spectrogram(signal, n_samples, cfg.n_fft, cfg.hop_length, cfg.use_power);
    let n_frames = frame_count(n_samples, cfg.n_fft, cfg.hop_length);
    let mut mel = fb.apply_batch(&spec, n_frames)?;
    for v in mel.iter_mut() {
        *v = (*v + cfg.log_floor).ln();
    }
    Ok(mel)
}

/// Compute the **(linear) mel spectrogram** `[n_frames, n_mels]` (no log).
///
/// # Errors
/// Same as [`log_mel_spectrogram`] (validation + filterbank construction).
pub fn mel_spectrogram(signal: &[f32], cfg: &MfccConfig) -> AudioResult<Vec<f32>> {
    let n_samples = signal.len();
    validate(cfg, n_samples)?;

    let fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.f_min, cfg.f_max)?;
    let spec = magnitude_spectrogram(signal, n_samples, cfg.n_fft, cfg.hop_length, cfg.use_power);
    let n_frames = frame_count(n_samples, cfg.n_fft, cfg.hop_length);
    fb.apply_batch(&spec, n_frames)
}

/// Compute **MFCC** features `[n_frames, n_mfcc]` (row-major).
///
/// Computes the log-mel spectrogram, then applies an orthonormal DCT-II across
/// the mel axis, keeping the lowest `n_mfcc` coefficients per frame.
///
/// # Errors
/// Same as [`log_mel_spectrogram`].
pub fn mfcc(signal: &[f32], cfg: &MfccConfig) -> AudioResult<Vec<f32>> {
    let log_mel = log_mel_spectrogram(signal, cfg)?;
    let n_mels = cfg.n_mels;
    let n_mfcc = cfg.n_mfcc;
    // `log_mel_spectrogram` already validated `n_mels ≥ 1`; guard defensively.
    let n_frames = log_mel.len().checked_div(n_mels).unwrap_or(0);
    let dct = dct2_matrix(n_mfcc, n_mels);

    let mut out = vec![0.0_f32; n_frames * n_mfcc];
    for frame in 0..n_frames {
        let lm = &log_mel[frame * n_mels..(frame + 1) * n_mels];
        for c in 0..n_mfcc {
            let row = &dct[c * n_mels..(c + 1) * n_mels];
            let mut acc = 0.0_f32;
            for (d, &x) in row.iter().zip(lm.iter()) {
                acc += d * x;
            }
            out[frame * n_mfcc + c] = acc;
        }
    }
    Ok(out)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn default_cfg() -> MfccConfig {
        MfccConfig {
            sample_rate: 16_000.0,
            n_fft: 256,
            hop_length: 128,
            n_mels: 26,
            n_mfcc: 13,
            f_min: 0.0,
            f_max: 8_000.0,
            use_power: true,
            log_floor: 1e-10,
        }
    }

    /// Sine wave at `freq` Hz, `n` samples at `fs`.
    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn filterbank_shape() {
        let fb = mel_filterbank(26, 256, 16_000.0, 0.0, 8_000.0).expect("ok");
        assert_eq!(fb.n_mels(), 26);
        assert_eq!(fb.filter_weights().len(), 26 * (256 / 2 + 1));
    }

    #[test]
    fn filterbank_nonneg() {
        let fb = mel_filterbank(40, 512, 16_000.0, 0.0, 8_000.0).expect("ok");
        assert!(fb.filter_weights().iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn triangular_filters() {
        // Each filter rises then falls (triangular): monotone up to peak, down after.
        let fb = mel_filterbank(20, 256, 16_000.0, 0.0, 8_000.0).expect("ok");
        let n_bins = 256 / 2 + 1;
        let w = fb.filter_weights();
        for m in 0..20 {
            let row = &w[m * n_bins..(m + 1) * n_bins];
            let mut peak = 0usize;
            let mut peak_v = row[0];
            for (i, &v) in row.iter().enumerate() {
                if v > peak_v {
                    peak_v = v;
                    peak = i;
                }
            }
            assert!(peak_v > 0.0, "row {m} flat");
            for k in 1..=peak {
                assert!(row[k] + 1e-6 >= row[k - 1], "not rising at {k}");
            }
            for k in peak + 1..n_bins {
                assert!(row[k - 1] + 1e-6 >= row[k], "not falling at {k}");
            }
        }
    }

    #[test]
    fn mfcc_shape() {
        let cfg = default_cfg();
        let sig = sine(440.0, cfg.sample_rate, 2000);
        let out = mfcc(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        assert_eq!(out.len(), n_frames * cfg.n_mfcc);
        assert!(n_frames > 0);
    }

    #[test]
    fn dc_signal() {
        // A constant (DC) signal: MFCCs must be finite (Hann window kills the
        // DC-edge discontinuity; energy concentrates at bin 0).
        let cfg = default_cfg();
        let sig = vec![1.0_f32; 2000];
        let out = mfcc(&sig, &cfg).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn n_mfcc_le_n_mels() {
        let mut cfg = default_cfg();
        cfg.n_mfcc = cfg.n_mels + 1;
        let sig = sine(440.0, cfg.sample_rate, 1000);
        assert!(matches!(
            mfcc(&sig, &cfg).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn deterministic() {
        let cfg = default_cfg();
        let sig = sine(523.25, cfg.sample_rate, 1500);
        let a = mfcc(&sig, &cfg).expect("ok");
        let b = mfcc(&sig, &cfg).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn fmin_ge_fmax_error() {
        let mut cfg = default_cfg();
        cfg.f_min = 4000.0;
        cfg.f_max = 1000.0;
        let sig = sine(440.0, cfg.sample_rate, 1000);
        assert!(mfcc(&sig, &cfg).is_err());
    }

    #[test]
    fn n_mels_0_error() {
        let mut cfg = default_cfg();
        cfg.n_mels = 0;
        // n_mfcc must also stay ≤ n_mels checks; n_mels==0 triggers first.
        cfg.n_mfcc = 0;
        let sig = sine(440.0, cfg.sample_rate, 1000);
        assert_eq!(mfcc(&sig, &cfg).unwrap_err(), AudioError::InvalidNumMels(0));
    }

    #[test]
    fn finite() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 3000);
        let out = mfcc(&sig, &cfg).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
        let lm = log_mel_spectrogram(&sig, &cfg).expect("ok");
        assert!(lm.iter().all(|v| v.is_finite()));
        let ms = mel_spectrogram(&sig, &cfg).expect("ok");
        assert!(ms.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn frame_count_matches() {
        // Cross-check the helper against the produced row count.
        let cfg = default_cfg();
        let sig = sine(300.0, cfg.sample_rate, 1777);
        let expected = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        let lm = log_mel_spectrogram(&sig, &cfg).expect("ok");
        assert_eq!(lm.len() / cfg.n_mels, expected);
        // Closed-form: (1777 - 256)/128 + 1 = 11.
        assert_eq!(expected, (1777 - 256) / 128 + 1);
    }

    #[test]
    fn empty_signal_error() {
        let cfg = default_cfg();
        let sig: Vec<f32> = Vec::new();
        assert!(matches!(
            mfcc(&sig, &cfg).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn signal_shorter_than_fft_error() {
        let cfg = default_cfg();
        let sig = sine(440.0, cfg.sample_rate, 100); // < n_fft = 256
        assert!(matches!(
            mfcc(&sig, &cfg).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn hop_zero_error() {
        let mut cfg = default_cfg();
        cfg.hop_length = 0;
        let sig = sine(440.0, cfg.sample_rate, 1000);
        assert_eq!(mfcc(&sig, &cfg).unwrap_err(), AudioError::InvalidStride(0));
    }

    #[test]
    fn dct2_orthonormal_rows() {
        // Orthonormal DCT-II: rows are orthonormal vectors → D·Dᵀ = I.
        let n_mels = 16usize;
        let dct = dct2_matrix(n_mels, n_mels);
        for a in 0..n_mels {
            for b in 0..n_mels {
                let mut dot = 0.0_f32;
                for m in 0..n_mels {
                    dot += dct[a * n_mels + m] * dct[b * n_mels + m];
                }
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-4, "D Dᵀ[{a},{b}] = {dot}");
            }
        }
    }

    #[test]
    fn magnitude_vs_power_differ() {
        // Power and magnitude front-ends produce different MFCCs in general.
        let mut cfg = default_cfg();
        let sig = sine(700.0, cfg.sample_rate, 1500);
        cfg.use_power = true;
        let p = mfcc(&sig, &cfg).expect("ok");
        cfg.use_power = false;
        let m = mfcc(&sig, &cfg).expect("ok");
        assert_eq!(p.len(), m.len());
        let differ = p.iter().zip(m.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differ, "power and magnitude MFCC should differ");
    }

    #[test]
    fn tone_energy_in_expected_mel_band() {
        // A pure tone should put most mel energy in the band covering its freq.
        let cfg = MfccConfig {
            sample_rate: 16_000.0,
            n_fft: 512,
            hop_length: 256,
            n_mels: 40,
            n_mfcc: 13,
            f_min: 0.0,
            f_max: 8_000.0,
            use_power: true,
            log_floor: 1e-10,
        };
        let freq = 1000.0_f32;
        let sig = sine(freq, cfg.sample_rate, 4096);
        let ms = mel_spectrogram(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        // Average mel energy across frames.
        let mut avg = vec![0.0_f32; cfg.n_mels];
        for f in 0..n_frames {
            for m in 0..cfg.n_mels {
                avg[m] += ms[f * cfg.n_mels + m];
            }
        }
        let mut peak = 0usize;
        let mut peak_v = avg[0];
        for (m, &v) in avg.iter().enumerate() {
            if v > peak_v {
                peak_v = v;
                peak = m;
            }
        }
        // The peak mel band's centre frequency should be near 1000 Hz.
        let mel_min = MelFilterbank::hz_to_mel(cfg.f_min);
        let mel_max = MelFilterbank::hz_to_mel(cfg.f_max);
        let step = (mel_max - mel_min) / (cfg.n_mels as f32 + 1.0);
        let centre_hz = MelFilterbank::mel_to_hz(mel_min + step * (peak as f32 + 1.0));
        assert!(
            (centre_hz - freq).abs() < 400.0,
            "peak band centre {centre_hz} Hz far from {freq} Hz"
        );
    }
}
