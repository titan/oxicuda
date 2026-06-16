//! Low-level spectral and temporal descriptors.
//!
//! Per-frame scalar features widely used for music/environmental-sound
//! classification and audio segmentation:
//!
//! - **Spectral centroid** — energy-weighted mean frequency (Hz), the spectral
//!   "centre of mass".
//! - **Spectral bandwidth** — energy-weighted spread (p-th order, default p = 2)
//!   around the centroid.
//! - **Spectral rolloff** — frequency below which a given fraction (default
//!   85 %) of the total spectral energy lies.
//! - **Spectral flatness** — geometric-mean / arithmetic-mean ratio (Wiener
//!   entropy); 1.0 ≈ white noise, → 0 for a pure tone.
//! - **Zero-crossing rate (ZCR)** — fraction of sign changes per frame (time
//!   domain).
//! - **RMS energy** — root-mean-square amplitude per frame (time domain).
//!
//! Spectral descriptors share a Hann-windowed direct DFT (pure-Rust); time-domain
//! descriptors operate directly on the framed samples.

use std::f32::consts::PI;

use crate::error::{AudioError, AudioResult};

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the spectral/temporal descriptors.
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Audio sample rate in Hz (> 0).
    pub sample_rate: f32,
    /// FFT length (frame length); must be ≥ 2.
    pub n_fft: usize,
    /// Hop length in samples; must be ≥ 1.
    pub hop_length: usize,
}

impl SpectralConfig {
    /// Default 16 kHz front-end (25 ms / 10 ms framing).
    #[must_use]
    pub fn default_16k() -> Self {
        Self {
            sample_rate: 16_000.0,
            n_fft: 400,
            hop_length: 160,
        }
    }
}

// ─── Framing helpers ────────────────────────────────────────────────────────────

/// Number of frames produced for `n_samples`.
#[must_use]
#[inline]
pub fn frame_count(n_samples: usize, n_fft: usize, hop_length: usize) -> usize {
    if hop_length == 0 || n_samples < n_fft {
        0
    } else {
        (n_samples - n_fft) / hop_length + 1
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0_f32];
    }
    (0..n)
        .map(|k| 0.5_f32 * (1.0_f32 - (2.0_f32 * PI * k as f32 / (n - 1) as f32).cos()))
        .collect()
}

fn validate(cfg: &SpectralConfig, n_samples: usize) -> AudioResult<()> {
    if cfg.sample_rate <= 0.0 {
        return Err(AudioError::Internal(format!(
            "spectral: sample_rate must be > 0, got {}",
            cfg.sample_rate
        )));
    }
    if cfg.n_fft < 2 {
        return Err(AudioError::ShapeMismatch {
            msg: format!("spectral: n_fft must be ≥ 2, got {}", cfg.n_fft),
        });
    }
    if cfg.hop_length == 0 {
        return Err(AudioError::InvalidStride(0));
    }
    if n_samples == 0 {
        return Err(AudioError::EmptyInput {
            msg: "spectral: empty signal".into(),
        });
    }
    if n_samples < cfg.n_fft {
        return Err(AudioError::InvalidSequenceLength(n_samples));
    }
    Ok(())
}

/// Magnitude spectrogram `[n_frames, n_fft/2 + 1]` via Hann-windowed direct DFT.
fn magnitude_spectrogram(signal: &[f32], cfg: &SpectralConfig) -> Vec<f32> {
    let n_bins = cfg.n_fft / 2 + 1;
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let window = hann_window(cfg.n_fft);
    let mut spec = vec![0.0_f32; n_frames * n_bins];
    for frame in 0..n_frames {
        let start = frame * cfg.hop_length;
        let row = &mut spec[frame * n_bins..(frame + 1) * n_bins];
        for (k, out_v) in row.iter_mut().enumerate() {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let omega = -2.0_f32 * PI * k as f32 / cfg.n_fft as f32;
            for (j, &w) in window.iter().enumerate() {
                let sample = signal[start + j] * w;
                let angle = omega * j as f32;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }
            *out_v = (re * re + im * im).sqrt();
        }
    }
    spec
}

/// Centre frequency (Hz) of FFT bin `k`.
#[inline]
fn bin_freq(k: usize, cfg: &SpectralConfig) -> f32 {
    k as f32 * cfg.sample_rate / cfg.n_fft as f32
}

// ─── Spectral descriptors ──────────────────────────────────────────────────────

/// Spectral **centroid** per frame, `[n_frames]` (Hz).
///
/// `centroid(t) = Σ_k f_k |X(t,k)| / Σ_k |X(t,k)|` (0 for a silent frame).
///
/// # Errors
/// See module-level validation: invalid `sample_rate` / `n_fft` / `hop_length`,
/// empty signal, or `signal.len() < n_fft`.
pub fn spectral_centroid(signal: &[f32], cfg: &SpectralConfig) -> AudioResult<Vec<f32>> {
    validate(cfg, signal.len())?;
    let n_bins = cfg.n_fft / 2 + 1;
    let spec = magnitude_spectrogram(signal, cfg);
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    for (t, out_v) in out.iter_mut().enumerate() {
        let row = &spec[t * n_bins..(t + 1) * n_bins];
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for (k, &mag) in row.iter().enumerate() {
            num += bin_freq(k, cfg) * mag;
            den += mag;
        }
        *out_v = if den > 1e-12 { num / den } else { 0.0 };
    }
    Ok(out)
}

/// Spectral **bandwidth** (order `p`, default 2) per frame, `[n_frames]` (Hz).
///
/// `bw(t) = (Σ_k |X(t,k)| · |f_k − centroid(t)|^p / Σ_k |X(t,k)|)^(1/p)`.
///
/// # Errors
/// As [`spectral_centroid`]; additionally [`AudioError::Internal`] if `p < 1`.
pub fn spectral_bandwidth(signal: &[f32], cfg: &SpectralConfig, p: u32) -> AudioResult<Vec<f32>> {
    if p < 1 {
        return Err(AudioError::Internal(
            "spectral: bandwidth p must be ≥ 1".into(),
        ));
    }
    validate(cfg, signal.len())?;
    let n_bins = cfg.n_fft / 2 + 1;
    let spec = magnitude_spectrogram(signal, cfg);
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    for (t, out_v) in out.iter_mut().enumerate() {
        let row = &spec[t * n_bins..(t + 1) * n_bins];
        let mut den = 0.0_f32;
        let mut num_c = 0.0_f32;
        for (k, &mag) in row.iter().enumerate() {
            num_c += bin_freq(k, cfg) * mag;
            den += mag;
        }
        if den <= 1e-12 {
            *out_v = 0.0;
            continue;
        }
        let centroid = num_c / den;
        let mut acc = 0.0_f32;
        for (k, &mag) in row.iter().enumerate() {
            acc += mag * (bin_freq(k, cfg) - centroid).abs().powi(p as i32);
        }
        *out_v = (acc / den).powf(1.0_f32 / p as f32);
    }
    Ok(out)
}

/// Spectral **rolloff** per frame, `[n_frames]` (Hz).
///
/// The lowest frequency `f_r` such that `Σ_{f_k ≤ f_r} |X| ≥ roll · Σ_k |X|`.
///
/// # Errors
/// As [`spectral_centroid`]; additionally [`AudioError::Internal`] if `roll`
/// is not in `(0, 1]`.
pub fn spectral_rolloff(signal: &[f32], cfg: &SpectralConfig, roll: f32) -> AudioResult<Vec<f32>> {
    if !(roll > 0.0 && roll <= 1.0) {
        return Err(AudioError::Internal(format!(
            "spectral: rolloff fraction must be in (0, 1], got {roll}"
        )));
    }
    validate(cfg, signal.len())?;
    let n_bins = cfg.n_fft / 2 + 1;
    let spec = magnitude_spectrogram(signal, cfg);
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    for (t, out_v) in out.iter_mut().enumerate() {
        let row = &spec[t * n_bins..(t + 1) * n_bins];
        let total: f32 = row.iter().sum();
        if total <= 1e-12 {
            *out_v = 0.0;
            continue;
        }
        let threshold = roll * total;
        let mut cumulative = 0.0_f32;
        let mut found = bin_freq(n_bins - 1, cfg);
        for (k, &mag) in row.iter().enumerate() {
            cumulative += mag;
            if cumulative >= threshold {
                found = bin_freq(k, cfg);
                break;
            }
        }
        *out_v = found;
    }
    Ok(out)
}

/// Spectral **flatness** (Wiener entropy) per frame, `[n_frames]` in `[0, 1]`.
///
/// `flatness(t) = exp(mean_k ln |X(t,k)|) / mean_k |X(t,k)|`. Returns 0 for a
/// silent frame. Computed in log-domain for numerical stability.
///
/// # Errors
/// As [`spectral_centroid`].
pub fn spectral_flatness(signal: &[f32], cfg: &SpectralConfig) -> AudioResult<Vec<f32>> {
    validate(cfg, signal.len())?;
    let n_bins = cfg.n_fft / 2 + 1;
    let spec = magnitude_spectrogram(signal, cfg);
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    let eps = 1e-10_f32;
    for (t, out_v) in out.iter_mut().enumerate() {
        let row = &spec[t * n_bins..(t + 1) * n_bins];
        let mut log_sum = 0.0_f32;
        let mut arith = 0.0_f32;
        for &mag in row.iter() {
            log_sum += (mag + eps).ln();
            arith += mag + eps;
        }
        let n = n_bins as f32;
        let geo = (log_sum / n).exp();
        let am = arith / n;
        *out_v = if am > eps {
            (geo / am).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    Ok(out)
}

// ─── Temporal descriptors ───────────────────────────────────────────────────────

/// **Zero-crossing rate** per frame, `[n_frames]` in `[0, 1]`.
///
/// The fraction of adjacent-sample sign changes within each (un-windowed) frame:
/// `zcr(t) = (1 / (n_fft − 1)) · Σ_{j} [sign(x_j) ≠ sign(x_{j+1})]`.
///
/// # Errors
/// As [`spectral_centroid`] (no `sample_rate` dependence, but framing validity
/// is still checked).
pub fn zero_crossing_rate(signal: &[f32], cfg: &SpectralConfig) -> AudioResult<Vec<f32>> {
    validate(cfg, signal.len())?;
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    let denom = (cfg.n_fft - 1) as f32;
    for (t, out_v) in out.iter_mut().enumerate() {
        let start = t * cfg.hop_length;
        let frame = &signal[start..start + cfg.n_fft];
        let mut crossings = 0usize;
        for w in frame.windows(2) {
            // A crossing occurs when the two samples lie on opposite sides of 0.
            let a = w[0];
            let b = w[1];
            if (a >= 0.0) != (b >= 0.0) {
                crossings += 1;
            }
        }
        *out_v = crossings as f32 / denom;
    }
    Ok(out)
}

/// **RMS energy** per frame, `[n_frames]`.
///
/// `rms(t) = sqrt(mean_j x_j²)` over the `n_fft` samples of frame `t`.
///
/// # Errors
/// As [`spectral_centroid`].
pub fn rms_energy(signal: &[f32], cfg: &SpectralConfig) -> AudioResult<Vec<f32>> {
    validate(cfg, signal.len())?;
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let mut out = vec![0.0_f32; n_frames];
    for (t, out_v) in out.iter_mut().enumerate() {
        let start = t * cfg.hop_length;
        let frame = &signal[start..start + cfg.n_fft];
        let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
        *out_v = (sum_sq / cfg.n_fft as f32).sqrt();
    }
    Ok(out)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn default_cfg() -> SpectralConfig {
        SpectralConfig {
            sample_rate: 16_000.0,
            n_fft: 256,
            hop_length: 128,
        }
    }

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn centroid_shape_and_finite() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 2000);
        let c = spectral_centroid(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        assert_eq!(c.len(), n_frames);
        assert!(c.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn centroid_tracks_tone() {
        // Centroid of a pure tone is near its frequency.
        let cfg = SpectralConfig {
            n_fft: 1024,
            hop_length: 512,
            ..default_cfg()
        };
        let freq = 2000.0_f32;
        let sig = sine(freq, cfg.sample_rate, 8192);
        let c = spectral_centroid(&sig, &cfg).expect("ok");
        let mean: f32 = c.iter().sum::<f32>() / c.len() as f32;
        assert!(
            (mean - freq).abs() < 300.0,
            "centroid mean {mean} vs {freq}"
        );
    }

    #[test]
    fn centroid_low_vs_high() {
        // A higher tone has a larger centroid than a lower tone.
        let cfg = SpectralConfig {
            n_fft: 1024,
            hop_length: 512,
            ..default_cfg()
        };
        let low = sine(500.0, cfg.sample_rate, 8192);
        let high = sine(3000.0, cfg.sample_rate, 8192);
        let cl = spectral_centroid(&low, &cfg).expect("ok");
        let ch = spectral_centroid(&high, &cfg).expect("ok");
        let ml: f32 = cl.iter().sum::<f32>() / cl.len() as f32;
        let mh: f32 = ch.iter().sum::<f32>() / ch.len() as f32;
        assert!(mh > ml, "high {mh} should exceed low {ml}");
    }

    #[test]
    fn bandwidth_nonneg() {
        let cfg = default_cfg();
        let sig = sine(800.0, cfg.sample_rate, 2000);
        let bw = spectral_bandwidth(&sig, &cfg, 2).expect("ok");
        assert!(bw.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn bandwidth_p_zero_error() {
        let cfg = default_cfg();
        let sig = sine(800.0, cfg.sample_rate, 2000);
        assert!(spectral_bandwidth(&sig, &cfg, 0).is_err());
    }

    #[test]
    fn rolloff_monotone_in_fraction() {
        // A higher rolloff fraction yields a rolloff frequency ≥ a lower one.
        let cfg = SpectralConfig {
            n_fft: 1024,
            hop_length: 512,
            ..default_cfg()
        };
        let sig = sine(1500.0, cfg.sample_rate, 8192);
        let lo = spectral_rolloff(&sig, &cfg, 0.5).expect("ok");
        let hi = spectral_rolloff(&sig, &cfg, 0.95).expect("ok");
        for (a, b) in lo.iter().zip(hi.iter()) {
            assert!(b + 1e-3 >= *a, "rolloff(0.95)={b} < rolloff(0.5)={a}");
        }
    }

    #[test]
    fn rolloff_within_nyquist() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 2000);
        let r = spectral_rolloff(&sig, &cfg, 0.85).expect("ok");
        let nyquist = cfg.sample_rate / 2.0;
        assert!(r.iter().all(|&v| v >= 0.0 && v <= nyquist + 1e-3));
    }

    #[test]
    fn rolloff_bad_fraction_error() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 2000);
        assert!(spectral_rolloff(&sig, &cfg, 0.0).is_err());
        assert!(spectral_rolloff(&sig, &cfg, 1.5).is_err());
    }

    #[test]
    fn flatness_in_unit_range() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 2000);
        let f = spectral_flatness(&sig, &cfg).expect("ok");
        assert!(f.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn flatness_tone_below_noise() {
        // A pure tone is far less flat than white noise.
        let cfg = SpectralConfig {
            n_fft: 512,
            hop_length: 256,
            ..default_cfg()
        };
        let tone = sine(1000.0, cfg.sample_rate, 8192);
        // Deterministic pseudo-noise via a simple LCG-ish sequence.
        let mut state = 12345u64;
        let noise: Vec<f32> = (0..8192)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();
        let ft = spectral_flatness(&tone, &cfg).expect("ok");
        let fn_ = spectral_flatness(&noise, &cfg).expect("ok");
        let mt: f32 = ft.iter().sum::<f32>() / ft.len() as f32;
        let mn: f32 = fn_.iter().sum::<f32>() / fn_.len() as f32;
        assert!(mn > mt, "noise flatness {mn} should exceed tone {mt}");
    }

    #[test]
    fn zcr_in_unit_range() {
        let cfg = default_cfg();
        let sig = sine(1000.0, cfg.sample_rate, 2000);
        let z = zero_crossing_rate(&sig, &cfg).expect("ok");
        assert!(z.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn zcr_higher_for_higher_freq() {
        // A higher-frequency sine crosses zero more often.
        let cfg = SpectralConfig {
            n_fft: 512,
            hop_length: 256,
            ..default_cfg()
        };
        let low = sine(200.0, cfg.sample_rate, 8192);
        let high = sine(4000.0, cfg.sample_rate, 8192);
        let zl = zero_crossing_rate(&low, &cfg).expect("ok");
        let zh = zero_crossing_rate(&high, &cfg).expect("ok");
        let ml: f32 = zl.iter().sum::<f32>() / zl.len() as f32;
        let mh: f32 = zh.iter().sum::<f32>() / zh.len() as f32;
        assert!(mh > ml, "high ZCR {mh} should exceed low {ml}");
    }

    #[test]
    fn rms_matches_amplitude() {
        // RMS of A·sin is ≈ A/√2 for a frame spanning many periods.
        let cfg = SpectralConfig {
            n_fft: 1024,
            hop_length: 512,
            ..default_cfg()
        };
        let amp = 0.7_f32;
        let sig: Vec<f32> = (0..8192)
            .map(|i| amp * (2.0 * PI_F32 * 1000.0 * i as f32 / cfg.sample_rate).sin())
            .collect();
        let r = rms_energy(&sig, &cfg).expect("ok");
        let mean: f32 = r.iter().sum::<f32>() / r.len() as f32;
        let expected = amp / 2.0_f32.sqrt();
        assert!((mean - expected).abs() < 0.05, "rms {mean} vs {expected}");
    }

    #[test]
    fn rms_zero_for_silence() {
        let cfg = default_cfg();
        let sig = vec![0.0_f32; 2000];
        let r = rms_energy(&sig, &cfg).expect("ok");
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn empty_signal_error() {
        let cfg = default_cfg();
        assert!(matches!(
            spectral_centroid(&[], &cfg).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn short_signal_error() {
        let cfg = default_cfg();
        let sig = sine(440.0, cfg.sample_rate, 100); // < n_fft
        assert!(matches!(
            rms_energy(&sig, &cfg).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn hop_zero_error() {
        let mut cfg = default_cfg();
        cfg.hop_length = 0;
        let sig = sine(440.0, cfg.sample_rate, 2000);
        assert_eq!(
            spectral_centroid(&sig, &cfg).unwrap_err(),
            AudioError::InvalidStride(0)
        );
    }

    #[test]
    fn deterministic() {
        let cfg = default_cfg();
        let sig = sine(660.0, cfg.sample_rate, 2000);
        let a = spectral_centroid(&sig, &cfg).expect("ok");
        let b = spectral_centroid(&sig, &cfg).expect("ok");
        assert_eq!(a, b);
    }
}
