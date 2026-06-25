//! End-to-end log-mel spectrogram extractor (STFT → mel filterbank → log).
//!
//! Unlike [`super::log_mel_adapter::LogMelInput`] — which merely *validates* a
//! log-mel tensor produced by an upstream crate — this module performs the full
//! front-end **inside** `oxicuda-audio` by composing the crate's own primitives:
//!
//! ```text
//! raw_pcm  →  stft_hann  →  |·|²  →  MelFilterbank  →  log(mel + eps)  →  [T, F]
//! ```
//!
//! The STFT is the direct-DFT Hann-windowed transform from
//! [`crate::vocoder::griffin_lim::stft_hann`] (O(N²) per frame, pure-Rust, no
//! external FFT). The triangular mel filterbank is
//! [`crate::features::mel_filterbank::MelFilterbank`] (HTK 2595·log10
//! convention). The final non-linearity is a natural-log compression with a
//! configurable floor, matching the convention used by Whisper / ESPnet
//! front-ends (`log(max(mel, eps))`).
//!
//! ## Frame count
//!
//! For an input of `n_samples` samples the number of frames is
//! `(n_samples - n_fft) / hop_length + 1` (zero when `n_samples < n_fft`),
//! identical to [`stft_hann`].
//!
//! ## References
//! - Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale Weak
//!   Supervision" (Whisper) — log-mel front-end `log10(max(mel, 1e-10))`.
//! - Davis & Mermelstein (1980) — mel-frequency cepstral front-end.

use crate::error::{AudioError, AudioResult};
use crate::features::log_mel_adapter::LogMelInput;
use crate::features::mel_filterbank::{MelFilterbank, MelFilterbankConfig};
use crate::vocoder::griffin_lim::stft_hann;

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`LogMelExtractor`].
#[derive(Debug, Clone)]
pub struct LogMelExtractorConfig {
    /// Audio sample rate in Hz (must be > 0).
    pub sample_rate: f32,
    /// FFT length in samples (must be ≥ 2).
    pub n_fft: usize,
    /// Hop length in samples (must be ≥ 1).
    pub hop_length: usize,
    /// Number of mel bands (must be ≥ 1).
    pub n_mels: usize,
    /// Lower frequency cutoff in Hz (≥ 0).
    pub f_min: f32,
    /// Upper frequency cutoff in Hz (`f_min < f_max ≤ sample_rate/2`).
    pub f_max: f32,
    /// If `true`, the mel filterbank is applied to the **power** spectrum
    /// (`|STFT|²`); if `false`, to the **magnitude** spectrum (`|STFT|`).
    pub power: bool,
    /// Floor added inside the logarithm to avoid `log(0)` (must be > 0).
    pub log_eps: f32,
}

impl LogMelExtractorConfig {
    /// A Whisper-like 16 kHz / 80-band front end (`n_fft = 400`, `hop = 160`).
    #[must_use]
    pub fn whisper_like() -> Self {
        Self {
            sample_rate: 16_000.0,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8_000.0,
            power: true,
            log_eps: 1e-10,
        }
    }

    /// A small configuration suitable for unit tests
    /// (`n_fft = 64`, `hop = 16`, 16 bands).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            sample_rate: 16_000.0,
            n_fft: 64,
            hop_length: 16,
            n_mels: 16,
            f_min: 0.0,
            f_max: 8_000.0,
            power: true,
            log_eps: 1e-10,
        }
    }
}

// ─── Extractor ─────────────────────────────────────────────────────────────────

/// End-to-end log-mel spectrogram extractor.
///
/// Pre-computes the mel filterbank at construction time so that repeated calls
/// to [`LogMelExtractor::extract`] avoid re-building the triangular weights.
#[derive(Debug, Clone)]
pub struct LogMelExtractor {
    config: LogMelExtractorConfig,
    filterbank: MelFilterbank,
}

impl LogMelExtractor {
    /// Construct a log-mel extractor, validating the configuration and building
    /// the mel filterbank.
    ///
    /// # Errors
    ///
    /// - [`AudioError::ShapeMismatch`] when `n_fft < 2`.
    /// - [`AudioError::InvalidStride`] when `hop_length == 0`.
    /// - [`AudioError::NonFinite`] when `log_eps ≤ 0` or not finite.
    /// - Any error propagated from [`MelFilterbank::new`] (sample rate, mel
    ///   count, frequency-band validation).
    pub fn new(config: LogMelExtractorConfig) -> AudioResult<Self> {
        if config.n_fft < 2 {
            return Err(AudioError::ShapeMismatch {
                msg: format!("LogMelExtractor: n_fft must be ≥ 2, got {}", config.n_fft),
            });
        }
        if config.hop_length == 0 {
            return Err(AudioError::InvalidStride(0));
        }
        if config.log_eps <= 0.0 || !config.log_eps.is_finite() {
            return Err(AudioError::NonFinite {
                msg: format!(
                    "LogMelExtractor: log_eps must be finite and > 0, got {}",
                    config.log_eps
                ),
            });
        }
        let filterbank = MelFilterbank::new(MelFilterbankConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            n_mels: config.n_mels,
            f_min: config.f_min,
            f_max: config.f_max,
        })?;
        Ok(Self { config, filterbank })
    }

    /// Number of FFT bins (`n_fft / 2 + 1`).
    #[must_use]
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.config.n_fft / 2 + 1
    }

    /// Number of mel bands.
    #[must_use]
    #[inline]
    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    /// Reference to the construction configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &LogMelExtractorConfig {
        &self.config
    }

    /// Number of frames produced for a signal of `n_samples` samples.
    #[must_use]
    #[inline]
    pub fn output_frames(&self, n_samples: usize) -> usize {
        if n_samples < self.config.n_fft {
            0
        } else {
            (n_samples - self.config.n_fft) / self.config.hop_length + 1
        }
    }

    /// Extract a `[T, n_mels]` log-mel spectrogram from a raw waveform.
    ///
    /// `signal` is mono PCM (length `signal.len()`); the number of frames is
    /// [`Self::output_frames`]. The returned buffer is row-major `[T, n_mels]`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `signal` is shorter than `n_fft`
    ///   (no frames can be formed).
    /// - Any error propagated from the STFT or the mel filterbank.
    pub fn extract(&self, signal: &[f32]) -> AudioResult<Vec<f32>> {
        let n_samples = signal.len();
        let n_frames = self.output_frames(n_samples);
        if n_frames == 0 {
            return Err(AudioError::EmptyInput {
                msg: format!(
                    "LogMelExtractor: signal length {} < n_fft {} — no frames",
                    n_samples, self.config.n_fft
                ),
            });
        }

        // STFT is computed in f64 for numerical robustness, then reduced to a
        // per-frame magnitude / power spectrum.
        let signal_f64: Vec<f64> = signal.iter().map(|&v| v as f64).collect();
        let stft = stft_hann(
            &signal_f64,
            n_samples,
            self.config.n_fft,
            self.config.hop_length,
        )?;

        let n_bins = self.n_bins();
        let mut spectra = vec![0.0_f32; n_frames * n_bins];
        for t in 0..n_frames {
            let base_s = t * n_bins * 2;
            let base_m = t * n_bins;
            for k in 0..n_bins {
                let re = stft[base_s + k * 2];
                let im = stft[base_s + k * 2 + 1];
                let power = re * re + im * im;
                let value = if self.config.power {
                    power
                } else {
                    power.sqrt()
                };
                spectra[base_m + k] = value as f32;
            }
        }

        // Apply the triangular mel filterbank → `[T, n_mels]` mel energies.
        let mel = self.filterbank.apply_batch(&spectra, n_frames)?;

        // Log compression with floor.
        let eps = self.config.log_eps;
        let mut log_mel = vec![0.0_f32; mel.len()];
        for (out, &m) in log_mel.iter_mut().zip(mel.iter()) {
            *out = m.max(eps).ln();
        }
        Ok(log_mel)
    }

    /// Extract directly into a validated [`LogMelInput`] (`[T, F]` wrapper).
    ///
    /// # Errors
    ///
    /// Same as [`Self::extract`], plus any validation error from
    /// [`LogMelInput::from_mel`].
    pub fn extract_input(&self, signal: &[f32]) -> AudioResult<LogMelInput> {
        let data = self.extract(signal)?;
        let n_frames = self.output_frames(signal.len());
        LogMelInput::from_mel(&data, n_frames, self.config.n_mels)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::f32::consts::PI;

    fn tiny() -> LogMelExtractor {
        LogMelExtractor::new(LogMelExtractorConfig::tiny()).expect("build tiny")
    }

    #[test]
    fn output_frames_matches_stft() {
        let ext = tiny();
        // n_fft = 64, hop = 16: (256 - 64)/16 + 1 = 13.
        assert_eq!(ext.output_frames(256), 13);
        assert_eq!(ext.output_frames(64), 1);
        assert_eq!(ext.output_frames(63), 0);
    }

    #[test]
    fn extract_shape_is_t_by_mels() {
        let ext = tiny();
        let n = 512usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let out = ext.extract(&signal).expect("extract");
        let frames = ext.output_frames(n);
        assert_eq!(out.len(), frames * ext.n_mels());
        assert!(out.iter().all(|v| v.is_finite()), "non-finite log-mel");
    }

    #[test]
    fn extract_input_validates_shape() {
        let ext = tiny();
        let n = 512usize;
        let signal: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 * 0.01).sin()).collect();
        let lm = ext.extract_input(&signal).expect("extract_input");
        assert_eq!(lm.mels, ext.n_mels());
        assert_eq!(lm.time, ext.output_frames(n));
        assert_eq!(lm.data.len(), lm.time * lm.mels);
    }

    #[test]
    fn silence_hits_log_floor() {
        let ext = tiny();
        let signal = vec![0.0_f32; 512];
        let out = ext.extract(&signal).expect("extract");
        // A zero signal → zero mel energy → log(eps).
        let floor = ext.config().log_eps.ln();
        for &v in &out {
            assert!((v - floor).abs() < 1e-3, "expected log-floor, got {v}");
        }
    }

    #[test]
    fn pure_tone_peaks_in_expected_band() {
        // A pure sinusoid should concentrate energy in a single mel band; that
        // band's log energy must exceed the mean across all bands.
        let cfg = LogMelExtractorConfig {
            sample_rate: 16_000.0,
            n_fft: 512,
            hop_length: 128,
            n_mels: 40,
            f_min: 0.0,
            f_max: 8_000.0,
            power: true,
            log_eps: 1e-10,
        };
        let ext = LogMelExtractor::new(cfg).expect("build");
        let freq = 1_000.0_f32;
        let n = 4096usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / 16_000.0).sin())
            .collect();
        let out = ext.extract(&signal).expect("extract");
        let frames = ext.output_frames(n);
        let n_mels = ext.n_mels();
        // Average the log-mel over time per band.
        let mut band_mean = vec![0.0_f32; n_mels];
        for t in 0..frames {
            for m in 0..n_mels {
                band_mean[m] += out[t * n_mels + m];
            }
        }
        for v in band_mean.iter_mut() {
            *v /= frames as f32;
        }
        let overall: f32 = band_mean.iter().sum::<f32>() / n_mels as f32;
        let peak = band_mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(peak > overall + 1.0, "tone did not produce a clear peak");
    }

    #[test]
    fn power_vs_magnitude_differ() {
        let mut cfg = LogMelExtractorConfig::tiny();
        cfg.power = true;
        let ext_p = LogMelExtractor::new(cfg.clone()).expect("p");
        cfg.power = false;
        let ext_m = LogMelExtractor::new(cfg).expect("m");

        let mut rng = LcgRng::new(123);
        let mut signal = vec![0.0_f32; 512];
        rng.fill_normal(&mut signal);
        let out_p = ext_p.extract(&signal).expect("p");
        let out_m = ext_m.extract(&signal).expect("m");
        // Power and magnitude logs cannot be identical for a non-trivial signal.
        let diff: f32 = out_p
            .iter()
            .zip(out_m.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "power and magnitude paths were identical");
    }

    #[test]
    fn deterministic_across_runs() {
        let ext = tiny();
        let mut rng = LcgRng::new(7);
        let mut signal = vec![0.0_f32; 400];
        rng.fill_normal(&mut signal);
        let a = ext.extract(&signal).expect("a");
        let b = ext.extract(&signal).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn err_signal_too_short() {
        let ext = tiny(); // n_fft = 64
        let signal = vec![0.1_f32; 32];
        let r = ext.extract(&signal);
        assert!(matches!(r.unwrap_err(), AudioError::EmptyInput { .. }));
    }

    #[test]
    fn err_n_fft_too_small() {
        let mut cfg = LogMelExtractorConfig::tiny();
        cfg.n_fft = 1;
        let r = LogMelExtractor::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn err_hop_zero() {
        let mut cfg = LogMelExtractorConfig::tiny();
        cfg.hop_length = 0;
        let r = LogMelExtractor::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::InvalidStride(0)));
    }

    #[test]
    fn err_log_eps_non_positive() {
        let mut cfg = LogMelExtractorConfig::tiny();
        cfg.log_eps = 0.0;
        let r = LogMelExtractor::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::NonFinite { .. }));
    }

    #[test]
    fn whisper_like_config_builds() {
        let ext = LogMelExtractor::new(LogMelExtractorConfig::whisper_like()).expect("whisper");
        assert_eq!(ext.n_mels(), 80);
        assert_eq!(ext.n_bins(), 201);
    }
}
