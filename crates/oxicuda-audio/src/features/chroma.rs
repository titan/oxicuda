//! Chroma features — 12-bin pitch-class profile (chromagram).
//!
//! A chroma vector collapses the power spectrum onto the 12 pitch classes of the
//! equal-tempered scale (C, C#, …, B), folding all octaves together. It is the
//! standard front-end for chord recognition, key estimation, and cover-song
//! detection.
//!
//! ## Method
//!
//! 1. **Framing + magnitude DFT** — Hann-windowed frames; magnitude (or power)
//!    spectrum `|X(t, k)|` for `k ∈ [0, n_fft/2]` (pure-Rust direct DFT).
//! 2. **Bin → pitch-class map** — each FFT bin `k` has centre frequency
//!    `f_k = k·fs/n_fft`; its MIDI pitch is `p = 69 + 12·log2(f_k / 440)` and its
//!    pitch class is `p mod 12`. Bins below `f_min` (and the DC bin) are ignored.
//! 3. **Accumulation** — sum spectral energy into the 12 pitch-class bins.
//! 4. **Normalisation** — optional per-frame L2 (or L∞) normalisation so frames
//!    are comparable regardless of loudness.
//!
//! Output layout: `[n_frames, 12]` row-major, pitch class 0 = C.
//!
//! ## References
//! - Fujishima, T. (1999). "Realtime chord recognition of musical sound."
//! - Müller, M. (2015). "Fundamentals of Music Processing", Ch. 3.

use std::f32::consts::PI;

use crate::error::{AudioError, AudioResult};

/// Number of pitch classes (chroma bins) in 12-tone equal temperament.
pub const N_CHROMA: usize = 12;

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Per-frame normalisation strategy for the chromagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaNorm {
    /// No normalisation (raw summed energy).
    None,
    /// L2 (Euclidean) normalisation per frame.
    L2,
    /// L∞ (max) normalisation per frame.
    LInf,
}

/// Configuration for [`chroma`].
#[derive(Debug, Clone)]
pub struct ChromaConfig {
    /// Audio sample rate in Hz (> 0).
    pub sample_rate: f32,
    /// FFT length (frame length); must be ≥ 2.
    pub n_fft: usize,
    /// Hop length in samples; must be ≥ 1.
    pub hop_length: usize,
    /// Reference tuning frequency for A4 in Hz (typically 440).
    pub tuning_a4: f32,
    /// Lowest frequency (Hz) considered; bins below this are ignored (≥ 0).
    pub f_min: f32,
    /// If `true`, accumulate power `|X|²`; otherwise magnitude `|X|`.
    pub use_power: bool,
    /// Per-frame normalisation.
    pub norm: ChromaNorm,
}

impl ChromaConfig {
    /// A reasonable default for music at 22.05 kHz.
    #[must_use]
    pub fn default_22k() -> Self {
        Self {
            sample_rate: 22_050.0,
            n_fft: 2048,
            hop_length: 512,
            tuning_a4: 440.0,
            f_min: 32.70, // C1
            use_power: true,
            norm: ChromaNorm::L2,
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

// ─── Validation ─────────────────────────────────────────────────────────────────

fn validate(cfg: &ChromaConfig, n_samples: usize) -> AudioResult<()> {
    if cfg.sample_rate <= 0.0 {
        return Err(AudioError::Internal(format!(
            "chroma: sample_rate must be > 0, got {}",
            cfg.sample_rate
        )));
    }
    if cfg.n_fft < 2 {
        return Err(AudioError::ShapeMismatch {
            msg: format!("chroma: n_fft must be ≥ 2, got {}", cfg.n_fft),
        });
    }
    if cfg.hop_length == 0 {
        return Err(AudioError::InvalidStride(0));
    }
    if cfg.tuning_a4 <= 0.0 {
        return Err(AudioError::Internal(format!(
            "chroma: tuning_a4 must be > 0, got {}",
            cfg.tuning_a4
        )));
    }
    if cfg.f_min < 0.0 {
        return Err(AudioError::Internal(format!(
            "chroma: f_min must be ≥ 0, got {}",
            cfg.f_min
        )));
    }
    if n_samples == 0 {
        return Err(AudioError::EmptyInput {
            msg: "chroma: empty signal".into(),
        });
    }
    if n_samples < cfg.n_fft {
        return Err(AudioError::InvalidSequenceLength(n_samples));
    }
    Ok(())
}

// ─── Bin → pitch-class map ──────────────────────────────────────────────────────

/// Map each FFT bin to a pitch class in `[0, 12)`, or `usize::MAX` if the bin
/// should be ignored (DC, or below `f_min`).
fn pitch_class_map(cfg: &ChromaConfig, n_bins: usize) -> Vec<usize> {
    let mut map = vec![usize::MAX; n_bins];
    let bin_hz = cfg.sample_rate / cfg.n_fft as f32;
    for (k, slot) in map.iter_mut().enumerate() {
        if k == 0 {
            continue; // DC bin has no pitch
        }
        let f_k = k as f32 * bin_hz;
        if f_k < cfg.f_min || f_k <= 0.0 {
            continue;
        }
        // MIDI note number relative to A4; pitch class = round(p) mod 12.
        let p = 69.0_f32 + 12.0_f32 * (f_k / cfg.tuning_a4).log2();
        let pc = p.round();
        // Rust's `rem_euclid` keeps the result in [0, 12).
        let pc_mod = (pc as i64).rem_euclid(N_CHROMA as i64) as usize;
        *slot = pc_mod;
    }
    map
}

// ─── Public API ─────────────────────────────────────────────────────────────────

/// Compute the **chromagram** `[n_frames, 12]` (row-major, pitch class 0 = C).
///
/// # Errors
/// - [`AudioError::Internal`] on non-positive `sample_rate` / `tuning_a4`, or
///   negative `f_min`.
/// - [`AudioError::ShapeMismatch`] on `n_fft < 2`.
/// - [`AudioError::InvalidStride`] on `hop_length == 0`.
/// - [`AudioError::EmptyInput`] on an empty signal.
/// - [`AudioError::InvalidSequenceLength`] when `signal.len() < n_fft`.
pub fn chroma(signal: &[f32], cfg: &ChromaConfig) -> AudioResult<Vec<f32>> {
    let n_samples = signal.len();
    validate(cfg, n_samples)?;

    let n_bins = cfg.n_fft / 2 + 1;
    let n_frames = frame_count(n_samples, cfg.n_fft, cfg.hop_length);
    let window = hann_window(cfg.n_fft);
    let pc_map = pitch_class_map(cfg, n_bins);

    let mut out = vec![0.0_f32; n_frames * N_CHROMA];

    for frame in 0..n_frames {
        let start = frame * cfg.hop_length;
        let chroma_row = &mut out[frame * N_CHROMA..(frame + 1) * N_CHROMA];
        for (k, &pc) in pc_map.iter().enumerate() {
            if pc == usize::MAX {
                continue;
            }
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let omega = -2.0_f32 * PI * k as f32 / cfg.n_fft as f32;
            for (j, &w) in window.iter().enumerate() {
                let sample = signal[start + j] * w;
                let angle = omega * j as f32;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }
            let power = re * re + im * im;
            let energy = if cfg.use_power { power } else { power.sqrt() };
            chroma_row[pc] += energy;
        }
        normalise_frame(chroma_row, cfg.norm);
    }
    Ok(out)
}

fn normalise_frame(row: &mut [f32], norm: ChromaNorm) {
    match norm {
        ChromaNorm::None => {}
        ChromaNorm::L2 => {
            let nrm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if nrm > 1e-12 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        }
        ChromaNorm::LInf => {
            let nrm = row.iter().copied().fold(0.0_f32, f32::max);
            if nrm > 1e-12 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn default_cfg() -> ChromaConfig {
        ChromaConfig {
            sample_rate: 22_050.0,
            n_fft: 1024,
            hop_length: 512,
            tuning_a4: 440.0,
            f_min: 55.0,
            use_power: true,
            norm: ChromaNorm::L2,
        }
    }

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn chroma_shape() {
        let cfg = default_cfg();
        let sig = sine(440.0, cfg.sample_rate, 4096);
        let out = chroma(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        assert_eq!(out.len(), n_frames * N_CHROMA);
        assert!(n_frames > 0);
    }

    #[test]
    fn twelve_bins() {
        assert_eq!(N_CHROMA, 12);
        let cfg = default_cfg();
        let sig = sine(261.63, cfg.sample_rate, 2048);
        let out = chroma(&sig, &cfg).expect("ok");
        assert_eq!(out.len() % N_CHROMA, 0);
    }

    #[test]
    fn a440_peaks_at_pitch_class_a() {
        // A4 = 440 Hz → MIDI 69 → pitch class 69 mod 12 = 9 (A).
        let cfg = ChromaConfig {
            n_fft: 4096,
            hop_length: 2048,
            ..default_cfg()
        };
        let sig = sine(440.0, cfg.sample_rate, 16384);
        let out = chroma(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        let mut avg = [0.0_f32; N_CHROMA];
        for f in 0..n_frames {
            for c in 0..N_CHROMA {
                avg[c] += out[f * N_CHROMA + c];
            }
        }
        let mut peak = 0usize;
        let mut peak_v = avg[0];
        for (c, &v) in avg.iter().enumerate() {
            if v > peak_v {
                peak_v = v;
                peak = c;
            }
        }
        assert_eq!(peak, 9, "A440 should peak at pitch class A (9)");
    }

    #[test]
    fn c_note_peaks_at_pitch_class_c() {
        // C4 ≈ 261.63 Hz → pitch class 0 (C).
        let cfg = ChromaConfig {
            n_fft: 4096,
            hop_length: 2048,
            ..default_cfg()
        };
        let sig = sine(261.63, cfg.sample_rate, 16384);
        let out = chroma(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        let mut avg = [0.0_f32; N_CHROMA];
        for f in 0..n_frames {
            for c in 0..N_CHROMA {
                avg[c] += out[f * N_CHROMA + c];
            }
        }
        let mut peak = 0usize;
        let mut peak_v = avg[0];
        for (c, &v) in avg.iter().enumerate() {
            if v > peak_v {
                peak_v = v;
                peak = c;
            }
        }
        assert_eq!(peak, 0, "C4 should peak at pitch class C (0)");
    }

    #[test]
    fn octave_invariance() {
        // 440 Hz (A4) and 880 Hz (A5) should both peak at pitch class A.
        let cfg = ChromaConfig {
            n_fft: 4096,
            hop_length: 2048,
            ..default_cfg()
        };
        for &freq in &[440.0_f32, 880.0_f32] {
            let sig = sine(freq, cfg.sample_rate, 16384);
            let out = chroma(&sig, &cfg).expect("ok");
            let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
            let mut avg = [0.0_f32; N_CHROMA];
            for f in 0..n_frames {
                for c in 0..N_CHROMA {
                    avg[c] += out[f * N_CHROMA + c];
                }
            }
            let peak = avg
                .iter()
                .enumerate()
                .fold(
                    (0usize, avg[0]),
                    |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    },
                )
                .0;
            assert_eq!(peak, 9, "freq {freq} should map to pitch class A");
        }
    }

    #[test]
    fn l2_normalisation_unit_norm() {
        let cfg = ChromaConfig {
            norm: ChromaNorm::L2,
            ..default_cfg()
        };
        let sig = sine(330.0, cfg.sample_rate, 4096);
        let out = chroma(&sig, &cfg).expect("ok");
        let n_frames = frame_count(sig.len(), cfg.n_fft, cfg.hop_length);
        for f in 0..n_frames {
            let row = &out[f * N_CHROMA..(f + 1) * N_CHROMA];
            let nrm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            // Either a silent frame (norm 0) or unit L2 norm.
            assert!(nrm < 1e-6 || (nrm - 1.0).abs() < 1e-4, "norm = {nrm}");
        }
    }

    #[test]
    fn nonneg_and_finite() {
        let cfg = default_cfg();
        let sig = sine(523.25, cfg.sample_rate, 4096);
        let out = chroma(&sig, &cfg).expect("ok");
        assert!(out.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn deterministic() {
        let cfg = default_cfg();
        let sig = sine(196.0, cfg.sample_rate, 3000);
        let a = chroma(&sig, &cfg).expect("ok");
        let b = chroma(&sig, &cfg).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn empty_signal_error() {
        let cfg = default_cfg();
        assert!(matches!(
            chroma(&[], &cfg).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn short_signal_error() {
        let cfg = default_cfg();
        let sig = sine(440.0, cfg.sample_rate, 100); // < n_fft
        assert!(matches!(
            chroma(&sig, &cfg).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn hop_zero_error() {
        let mut cfg = default_cfg();
        cfg.hop_length = 0;
        let sig = sine(440.0, cfg.sample_rate, 2048);
        assert_eq!(
            chroma(&sig, &cfg).unwrap_err(),
            AudioError::InvalidStride(0)
        );
    }

    #[test]
    fn bad_sample_rate_error() {
        let mut cfg = default_cfg();
        cfg.sample_rate = 0.0;
        let sig = vec![0.1_f32; 2048];
        assert!(matches!(
            chroma(&sig, &cfg).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn norm_none_keeps_magnitude() {
        // With no normalisation, a louder signal yields a larger total chroma.
        let cfg = ChromaConfig {
            norm: ChromaNorm::None,
            ..default_cfg()
        };
        let quiet = sine(440.0, cfg.sample_rate, 2048);
        let loud: Vec<f32> = quiet.iter().map(|&v| v * 4.0).collect();
        let cq = chroma(&quiet, &cfg).expect("ok");
        let cl = chroma(&loud, &cfg).expect("ok");
        let sq: f32 = cq.iter().sum();
        let sl: f32 = cl.iter().sum();
        assert!(sl > sq * 2.0, "louder input should have larger chroma sum");
    }
}
