//! Triangular mel-scale filterbank computation.
//!
//! This module **computes** the mel filterbank from a magnitude or power
//! spectrogram (distinct from [`super::log_mel_adapter::LogMelInput`], which
//! merely validates an externally-computed log-mel tensor).
//!
//! ## Mel ↔ Hz conversion (HTK / Slaney 2595 convention)
//!
//! ```text
//! mel(f)   = 2595 · log10(1 + f / 700)
//! hz(m)    = 700  · (10^(m / 2595) − 1)
//! ```
//!
//! ## Triangular bank
//!
//! Given `n_mels` filters covering `[f_min, f_max]`, the centres are linearly
//! spaced on the mel scale (then mapped back to Hz, then to FFT bin indices).
//! Each filter is a triangle whose value is `1.0` at its centre bin, decays
//! linearly to `0.0` at the previous / next centre, and is `0` elsewhere.
//!
//! References:
//! - Slaney 1998, "Auditory Toolbox v2"
//! - HTK book §5.4 (Young et al.)

use crate::error::{AudioError, AudioResult};

// ─── Public configuration ────────────────────────────────────────────────────

/// Configuration for [`MelFilterbank`].
#[derive(Debug, Clone)]
pub struct MelFilterbankConfig {
    /// Audio sample rate in Hz (must be > 0).
    pub sample_rate: f32,
    /// FFT length used to produce the input spectrogram (must be ≥ 2).
    pub n_fft: usize,
    /// Number of mel filters (must be ≥ 1).
    pub n_mels: usize,
    /// Lower frequency cutoff in Hz (≥ 0).
    pub f_min: f32,
    /// Upper frequency cutoff in Hz (must satisfy `f_min < f_max ≤ sample_rate/2`).
    pub f_max: f32,
}

// ─── Public type ─────────────────────────────────────────────────────────────

/// Triangular mel-scale filterbank.
///
/// Stores a `n_mels × (n_fft/2 + 1)` matrix of non-negative weights (row-major,
/// row `m`, column `k` is the weight applied to FFT bin `k` for mel band `m`).
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Filter weights `[n_mels, n_fft/2 + 1]`, row-major.
    weights: Vec<f32>,
    /// Construction configuration.
    cfg: MelFilterbankConfig,
}

impl MelFilterbank {
    /// Convert a frequency in Hz to the mel scale (HTK 2595·log10 convention).
    ///
    /// ```text
    /// mel(f) = 2595 · log10(1 + f / 700)
    /// ```
    #[must_use]
    #[inline]
    pub fn hz_to_mel(f_hz: f32) -> f32 {
        2595.0_f32 * (1.0_f32 + f_hz / 700.0_f32).log10()
    }

    /// Inverse of [`Self::hz_to_mel`]:
    ///
    /// ```text
    /// hz(m) = 700 · (10^(m / 2595) − 1)
    /// ```
    #[must_use]
    #[inline]
    pub fn mel_to_hz(m: f32) -> f32 {
        700.0_f32 * (10.0_f32.powf(m / 2595.0_f32) - 1.0_f32)
    }

    /// Number of FFT bins (`n_fft / 2 + 1`).
    #[must_use]
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.cfg.n_fft / 2 + 1
    }

    /// Return a reference to the flat `n_mels × n_bins` weight matrix
    /// (row-major).
    #[must_use]
    #[inline]
    pub fn filter_weights(&self) -> &[f32] {
        &self.weights
    }

    /// Number of mel filters.
    #[must_use]
    #[inline]
    pub fn n_mels(&self) -> usize {
        self.cfg.n_mels
    }

    /// Number of FFT samples.
    #[must_use]
    #[inline]
    pub fn n_fft(&self) -> usize {
        self.cfg.n_fft
    }

    /// Sample rate the filterbank was constructed for.
    #[must_use]
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.cfg.sample_rate
    }

    /// Construct a triangular mel-scale filterbank.
    ///
    /// # Errors
    ///
    /// - [`AudioError::Internal`] when `sample_rate ≤ 0` or `f_min < 0`.
    /// - [`AudioError::InvalidNumMels`] when `n_mels == 0`.
    /// - [`AudioError::ShapeMismatch`] when `n_fft < 2`, `f_max ≤ f_min`,
    ///   or `f_max > sample_rate / 2` (Nyquist), or any boundary inconsistency.
    pub fn new(cfg: MelFilterbankConfig) -> AudioResult<Self> {
        // ── Validation ───────────────────────────────────────────────────────
        if cfg.sample_rate <= 0.0 {
            return Err(AudioError::Internal(format!(
                "MelFilterbank: sample_rate must be > 0, got {}",
                cfg.sample_rate
            )));
        }
        if cfg.n_fft < 2 {
            return Err(AudioError::ShapeMismatch {
                msg: format!("MelFilterbank: n_fft must be ≥ 2, got {}", cfg.n_fft),
            });
        }
        if cfg.n_mels == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        if cfg.f_min < 0.0 {
            return Err(AudioError::Internal(format!(
                "MelFilterbank: f_min must be ≥ 0, got {}",
                cfg.f_min
            )));
        }
        if cfg.f_max <= cfg.f_min {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "MelFilterbank: f_max ({}) must be > f_min ({})",
                    cfg.f_max, cfg.f_min
                ),
            });
        }
        let nyquist = cfg.sample_rate / 2.0_f32;
        // Allow a tiny epsilon for floating-point representation of f_max==nyquist.
        let nyquist_tol = nyquist * (1.0_f32 + 1e-6_f32);
        if cfg.f_max > nyquist_tol {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "MelFilterbank: f_max ({}) must be ≤ sample_rate/2 ({})",
                    cfg.f_max, nyquist
                ),
            });
        }

        // ── Mel-scale centres ────────────────────────────────────────────────
        // n_mels + 2 mel-spaced points: lower edge, n_mels centres, upper edge.
        let n_mels = cfg.n_mels;
        let n_bins = cfg.n_fft / 2 + 1;

        let mel_min = Self::hz_to_mel(cfg.f_min);
        let mel_max = Self::hz_to_mel(cfg.f_max);

        let mut hz_points = vec![0.0_f32; n_mels + 2];
        let step = (mel_max - mel_min) / ((n_mels + 1) as f32);
        for (idx, point) in hz_points.iter_mut().enumerate() {
            let m = mel_min + step * (idx as f32);
            *point = Self::mel_to_hz(m);
        }

        // ── Map each Hz centre to a fractional FFT bin index ─────────────────
        // bin_freq(k) = k * sample_rate / n_fft  →  k(f) = f * n_fft / sample_rate
        let bin_scale = (cfg.n_fft as f32) / cfg.sample_rate;
        let mut bin_points = vec![0.0_f32; n_mels + 2];
        for (idx, &hz) in hz_points.iter().enumerate() {
            bin_points[idx] = hz * bin_scale;
        }

        // ── Triangular weights ───────────────────────────────────────────────
        // For each filter m, weights[m, k] is a triangle peaking at bin_points[m+1].
        let mut weights = vec![0.0_f32; n_mels * n_bins];
        for m in 0..n_mels {
            let lower = bin_points[m];
            let centre = bin_points[m + 1];
            let upper = bin_points[m + 2];

            // Avoid division by zero on degenerate spacings.
            let left_width = (centre - lower).max(1e-12_f32);
            let right_width = (upper - centre).max(1e-12_f32);

            let row_off = m * n_bins;
            for k in 0..n_bins {
                let k_f = k as f32;
                let w = if k_f <= lower || k_f >= upper {
                    0.0_f32
                } else if k_f <= centre {
                    (k_f - lower) / left_width
                } else {
                    (upper - k_f) / right_width
                };
                weights[row_off + k] = w.max(0.0_f32);
            }
        }

        Ok(Self { weights, cfg })
    }

    /// Apply the filterbank to a single magnitude (or power) spectrum frame.
    ///
    /// `spectrum` must have length `n_fft / 2 + 1`. Returns a `n_mels`-length
    /// vector of mel energies (one summed weighted dot-product per filter).
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] when `spectrum.len() != n_bins()`.
    pub fn apply(&self, spectrum: &[f32]) -> AudioResult<Vec<f32>> {
        let n_bins = self.n_bins();
        if spectrum.len() != n_bins {
            return Err(AudioError::DimensionMismatch {
                expected: n_bins,
                got: spectrum.len(),
            });
        }
        let n_mels = self.cfg.n_mels;
        let mut out = vec![0.0_f32; n_mels];
        for (m, out_v) in out.iter_mut().enumerate().take(n_mels) {
            let row_off = m * n_bins;
            let row = &self.weights[row_off..row_off + n_bins];
            let mut acc = 0.0_f32;
            for (w, &s) in row.iter().zip(spectrum.iter()) {
                acc += w * s;
            }
            *out_v = acc;
        }
        Ok(out)
    }

    /// Apply the filterbank to a batch of `n_frames` spectra in a flat buffer.
    ///
    /// Input layout: `n_frames × (n_fft/2 + 1)` row-major.
    /// Output layout: `n_frames × n_mels` row-major.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] when `spectra.len() != n_frames * n_bins()`.
    pub fn apply_batch(&self, spectra: &[f32], n_frames: usize) -> AudioResult<Vec<f32>> {
        let n_bins = self.n_bins();
        let n_mels = self.cfg.n_mels;
        let expected = n_frames * n_bins;
        if spectra.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: spectra.len(),
            });
        }
        let mut out = vec![0.0_f32; n_frames * n_mels];
        for f in 0..n_frames {
            let src = &spectra[f * n_bins..(f + 1) * n_bins];
            for m in 0..n_mels {
                let row_off = m * n_bins;
                let row = &self.weights[row_off..row_off + n_bins];
                let mut acc = 0.0_f32;
                for (w, &s) in row.iter().zip(src.iter()) {
                    acc += w * s;
                }
                out[f * n_mels + m] = acc;
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> MelFilterbankConfig {
        MelFilterbankConfig {
            sample_rate: 16_000.0,
            n_fft: 512,
            n_mels: 40,
            f_min: 0.0,
            f_max: 8_000.0,
        }
    }

    #[test]
    fn hz_to_mel_zero_is_zero() {
        let m = MelFilterbank::hz_to_mel(0.0);
        assert!(m.abs() < 1e-6_f32, "hz_to_mel(0) = {m}");
    }

    #[test]
    fn hz_to_mel_monotone() {
        let mut prev = MelFilterbank::hz_to_mel(0.0);
        for k in 1..200 {
            let hz = (k as f32) * 50.0;
            let cur = MelFilterbank::hz_to_mel(hz);
            assert!(
                cur > prev,
                "non-monotone at {hz} Hz: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    #[test]
    fn mel_hz_round_trip() {
        for &hz in &[
            0.0_f32, 50.0, 200.0, 700.0, 1_000.0, 4_000.0, 8_000.0, 16_000.0,
        ] {
            let m = MelFilterbank::hz_to_mel(hz);
            let back = MelFilterbank::mel_to_hz(m);
            let err = (back - hz).abs();
            // Use relative tolerance for non-zero values.
            let tol = if hz == 0.0 { 1e-3 } else { hz * 1e-3 + 1e-3 };
            assert!(err <= tol, "round-trip hz={hz} → {back}, err={err}");
        }
    }

    #[test]
    fn hz_to_mel_known_value() {
        // mel(8000) ≈ 2840.02 (within the literature)
        let m = MelFilterbank::hz_to_mel(8_000.0);
        assert!((m - 2840.0).abs() < 5.0, "mel(8000) = {m}");
    }

    #[test]
    fn new_default_ok() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        assert_eq!(fb.n_bins(), 257);
        assert_eq!(fb.n_mels(), 40);
    }

    #[test]
    fn filter_weights_length() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let n_bins = fb.n_bins();
        let expected = fb.n_mels() * n_bins;
        assert_eq!(fb.filter_weights().len(), expected);
    }

    #[test]
    fn filter_weights_non_negative() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        assert!(
            fb.filter_weights().iter().all(|&v| v >= 0.0),
            "found a negative filter weight"
        );
    }

    #[test]
    fn filter_weights_triangular_shape() {
        // For each row, the maximum must be unique-ish (or tied), and from the
        // peak bin the values should monotonically decay outward to zero.
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let n_bins = fb.n_bins();
        let n_mels = fb.n_mels();
        let w = fb.filter_weights();

        for m in 0..n_mels {
            let row = &w[m * n_bins..(m + 1) * n_bins];
            // Find argmax.
            let mut peak_idx = 0_usize;
            let mut peak_val = row[0];
            for (idx, &v) in row.iter().enumerate() {
                if v > peak_val {
                    peak_val = v;
                    peak_idx = idx;
                }
            }
            assert!(peak_val > 0.0, "row {m} has zero peak");

            // To the left of the peak: non-decreasing toward the peak.
            for k in 1..=peak_idx {
                assert!(
                    row[k] + 1e-7 >= row[k - 1],
                    "row {m} not non-decreasing left of peak at k={k}: {} → {}",
                    row[k - 1],
                    row[k]
                );
            }
            // To the right of the peak: non-increasing away from the peak.
            for k in peak_idx + 1..n_bins {
                assert!(
                    row[k - 1] + 1e-7 >= row[k],
                    "row {m} not non-increasing right of peak at k={k}: {} → {}",
                    row[k - 1],
                    row[k]
                );
            }
        }
    }

    #[test]
    fn apply_output_length() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let spectrum = vec![1.0_f32; fb.n_bins()];
        let out = fb.apply(&spectrum).expect("apply");
        assert_eq!(out.len(), fb.n_mels());
    }

    #[test]
    fn apply_constant_finite() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let spectrum = vec![0.5_f32; fb.n_bins()];
        let out = fb.apply(&spectrum).expect("apply");
        assert!(out.iter().all(|v| v.is_finite()));
        // Every filter should produce a positive value for a positive constant
        // spectrum (assuming each filter has at least one non-zero weight).
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn apply_zero_gives_zero() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let spectrum = vec![0.0_f32; fb.n_bins()];
        let out = fb.apply(&spectrum).expect("apply");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn apply_deterministic() {
        let fb_a = MelFilterbank::new(default_cfg()).expect("ok");
        let fb_b = MelFilterbank::new(default_cfg()).expect("ok");
        let spectrum: Vec<f32> = (0..fb_a.n_bins()).map(|i| (i as f32) * 0.01).collect();
        let out_a = fb_a.apply(&spectrum).expect("apply");
        let out_b = fb_b.apply(&spectrum).expect("apply");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn apply_batch_length() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let n_frames = 7usize;
        let n_bins = fb.n_bins();
        let spectra = vec![0.5_f32; n_frames * n_bins];
        let out = fb.apply_batch(&spectra, n_frames).expect("batch");
        assert_eq!(out.len(), n_frames * fb.n_mels());
    }

    #[test]
    fn apply_batch_matches_apply() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let n_frames = 3usize;
        let n_bins = fb.n_bins();
        let mut spectra = vec![0.0_f32; n_frames * n_bins];
        for (i, v) in spectra.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.02;
        }
        let batch = fb.apply_batch(&spectra, n_frames).expect("batch");
        for f in 0..n_frames {
            let frame = &spectra[f * n_bins..(f + 1) * n_bins];
            let single = fb.apply(frame).expect("apply");
            let target = &batch[f * fb.n_mels()..(f + 1) * fb.n_mels()];
            for (a, b) in single.iter().zip(target.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn err_sample_rate_zero() {
        let cfg = MelFilterbankConfig {
            sample_rate: 0.0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::Internal(_)));
    }

    #[test]
    fn err_sample_rate_negative() {
        let cfg = MelFilterbankConfig {
            sample_rate: -16_000.0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(r.is_err());
    }

    #[test]
    fn err_n_fft_too_small() {
        let cfg = MelFilterbankConfig {
            n_fft: 1,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn err_n_mels_zero() {
        let cfg = MelFilterbankConfig {
            n_mels: 0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert_eq!(r.unwrap_err(), AudioError::InvalidNumMels(0));
    }

    #[test]
    fn err_f_min_negative() {
        let cfg = MelFilterbankConfig {
            f_min: -100.0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::Internal(_)));
    }

    #[test]
    fn err_f_max_le_f_min() {
        let cfg = MelFilterbankConfig {
            f_min: 1_000.0,
            f_max: 500.0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn err_f_max_above_nyquist() {
        let cfg = MelFilterbankConfig {
            f_max: 12_000.0,
            ..default_cfg()
        };
        let r = MelFilterbank::new(cfg);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn err_apply_wrong_spectrum_length() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let bad = vec![0.5_f32; fb.n_bins() + 3];
        let r = fb.apply(&bad);
        assert!(matches!(
            r.unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn err_apply_batch_wrong_length() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let n_frames = 4usize;
        // intentionally wrong length
        let bad = vec![0.5_f32; n_frames * fb.n_bins() + 1];
        let r = fb.apply_batch(&bad, n_frames);
        assert!(matches!(
            r.unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn single_mel_filter() {
        let cfg = MelFilterbankConfig {
            sample_rate: 16_000.0,
            n_fft: 256,
            n_mels: 1,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let fb = MelFilterbank::new(cfg).expect("ok");
        assert_eq!(fb.n_mels(), 1);
        let spectrum = vec![1.0_f32; fb.n_bins()];
        let out = fb.apply(&spectrum).expect("apply");
        assert_eq!(out.len(), 1);
        assert!(out[0] > 0.0);
    }

    #[test]
    fn n_fft_odd_ok() {
        // n_fft ≥ 2 is the only constraint (not "even" per spec).
        let cfg = MelFilterbankConfig {
            sample_rate: 16_000.0,
            n_fft: 5,
            n_mels: 2,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let fb = MelFilterbank::new(cfg).expect("ok");
        assert_eq!(fb.n_bins(), 3); // 5/2 + 1 = 3
        let spec = vec![1.0_f32; 3];
        let out = fb.apply(&spec).expect("apply");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn weights_clone_independent() {
        let fb = MelFilterbank::new(default_cfg()).expect("ok");
        let fb_clone = fb.clone();
        assert_eq!(fb.filter_weights(), fb_clone.filter_weights());
        assert_eq!(fb.n_mels(), fb_clone.n_mels());
    }

    #[test]
    fn f_max_at_nyquist_ok() {
        let cfg = MelFilterbankConfig {
            sample_rate: 22_050.0,
            n_fft: 1024,
            n_mels: 80,
            f_min: 0.0,
            f_max: 11_025.0,
        };
        let fb = MelFilterbank::new(cfg).expect("ok at Nyquist");
        assert_eq!(fb.n_mels(), 80);
    }
}
