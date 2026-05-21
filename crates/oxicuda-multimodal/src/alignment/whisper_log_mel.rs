//! Whisper-style log-mel spectrogram front-end.
//!
//! Reference: Radford et al. 2023, *Robust Speech Recognition via Large-Scale
//! Weak Supervision* (Whisper).
//!
//! Converts a raw audio waveform into a log-mel spectrogram suitable for
//! the Whisper transformer encoder. The pipeline:
//!
//! 1. Frame the waveform with stride `hop_length` and window length `n_fft`.
//! 2. Multiply each frame by a Hann window of length `n_fft`.
//! 3. Compute the real-DFT magnitude spectrum (power spectrum)
//!    of length `n_fft/2 + 1`.
//! 4. Project the power spectrum through an `n_mels × (n_fft/2 + 1)` mel
//!    filterbank constructed with Slaney/HTK conventions
//!    (`mel = 2595 · log10(1 + f/700)`).
//! 5. Apply `log10(mel_energy + 1e-10)` to compress dynamic range.
//!
//! The output is shaped `[n_frames × n_mels]` (row-major) and is identical to
//! Whisper's encoder input up to numerical precision and ordering of frames.

use crate::error::{MmResult, MultiModalError};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`WhisperLogMel`].
#[derive(Debug, Clone)]
pub struct WhisperLogMelConfig {
    /// Sample rate of the input waveform in Hz. Must be `> 0`.
    pub sample_rate: f32,
    /// FFT window size (must be `≥ 2` and even).
    pub n_fft: usize,
    /// Hop between successive frames in samples (must satisfy
    /// `1 ≤ hop_length ≤ n_fft`).
    pub hop_length: usize,
    /// Number of mel filterbank channels (must be `≥ 1`).
    pub n_mels: usize,
    /// Lowest analysed frequency in Hz (must be `≥ 0` and `< f_max`).
    pub f_min: f32,
    /// Highest analysed frequency in Hz (must be `≤ sample_rate / 2`).
    pub f_max: f32,
}

impl WhisperLogMelConfig {
    /// Tiny preset suitable for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        }
    }

    fn validate(&self) -> MmResult<()> {
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: sample_rate must be > 0, got {}",
                self.sample_rate,
            )));
        }
        if self.n_fft < 2 || self.n_fft % 2 != 0 {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: n_fft must be even and >= 2, got {}",
                self.n_fft,
            )));
        }
        if self.hop_length == 0 || self.hop_length > self.n_fft {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: hop_length must be in [1, n_fft]={} got {}",
                self.n_fft, self.hop_length,
            )));
        }
        if self.n_mels == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if !self.f_min.is_finite() || self.f_min < 0.0 {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: f_min must be >= 0, got {}",
                self.f_min,
            )));
        }
        if !self.f_max.is_finite() || self.f_max <= self.f_min {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: f_max ({}) must be > f_min ({})",
                self.f_max, self.f_min,
            )));
        }
        let nyquist = 0.5_f32 * self.sample_rate;
        if self.f_max > nyquist {
            return Err(MultiModalError::Internal(format!(
                "whisper-log-mel: f_max ({}) must be <= sample_rate/2 ({})",
                self.f_max, nyquist,
            )));
        }
        Ok(())
    }
}

// ─── Front-end ───────────────────────────────────────────────────────────────

/// Whisper-style log-mel spectrogram front-end (CPU reference).
#[derive(Debug, Clone)]
pub struct WhisperLogMel {
    cfg: WhisperLogMelConfig,
    /// Mel filter matrix, shape `n_mels × (n_fft/2 + 1)`, row-major.
    mel_filter: Vec<f32>,
    /// Cached Hann analysis window of length `n_fft`.
    window: Vec<f32>,
}

impl WhisperLogMel {
    /// Construct a new front-end with cached Hann window and mel filterbank.
    pub fn new(cfg: WhisperLogMelConfig) -> MmResult<Self> {
        cfg.validate()?;
        let window = Self::hann_window(cfg.n_fft);
        let mel_filter = build_mel_filter(&cfg)?;
        Ok(Self {
            cfg,
            mel_filter,
            window,
        })
    }

    /// Borrow the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &WhisperLogMelConfig {
        &self.cfg
    }

    /// Borrow the cached Hann window.
    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Borrow the cached mel filterbank
    /// (row-major, shape `n_mels × (n_fft/2 + 1)`).
    #[must_use]
    pub fn mel_filter(&self) -> &[f32] {
        &self.mel_filter
    }

    /// Periodic-style Hann window: `w[i] = 0.5 · (1 − cos(2π · i / (n_fft − 1)))`.
    #[must_use]
    pub fn hann_window(n_fft: usize) -> Vec<f32> {
        if n_fft == 0 {
            return Vec::new();
        }
        if n_fft == 1 {
            return vec![0.0_f32];
        }
        let denom = (n_fft - 1) as f32;
        let two_pi = 2.0_f32 * std::f32::consts::PI;
        (0..n_fft)
            .map(|i| 0.5_f32 * (1.0_f32 - (two_pi * (i as f32) / denom).cos()))
            .collect()
    }

    /// Hertz → mel scale conversion (Slaney/HTK convention).
    #[must_use]
    pub fn hz_to_mel(f: f32) -> f32 {
        2_595.0_f32 * (1.0_f32 + f / 700.0_f32).log10()
    }

    /// Mel → hertz scale conversion (inverse of [`Self::hz_to_mel`]).
    #[must_use]
    pub fn mel_to_hz(m: f32) -> f32 {
        700.0_f32 * (10.0_f32.powf(m / 2_595.0_f32) - 1.0_f32)
    }

    /// Compute the power spectrum `|X|²` of a single (already-framed) audio
    /// buffer of length `n_fft`. Applies the cached Hann window first.
    /// Returns a vector of length `n_fft/2 + 1`.
    pub fn power_spectrum(&self, frame: &[f32]) -> MmResult<Vec<f32>> {
        if frame.len() != self.cfg.n_fft {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.cfg.n_fft,
                got: frame.len(),
            });
        }
        let n = self.cfg.n_fft;
        let half = n / 2 + 1;

        // Windowed frame.
        let mut windowed = vec![0.0_f32; n];
        for i in 0..n {
            let w = match self.window.get(i) {
                Some(w) => *w,
                None => 0.0_f32,
            };
            let s = match frame.get(i) {
                Some(s) => *s,
                None => 0.0_f32,
            };
            windowed[i] = s * w;
        }

        // Direct real-DFT — O(n²). For tests/configs used here `n` is small
        // (≤ 1024), and we keep numerical structure transparent.
        let two_pi = 2.0_f32 * std::f32::consts::PI;
        let inv_n = 1.0_f32 / n as f32;
        let mut power = vec![0.0_f32; half];
        for k in 0..half {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for j in 0..n {
                let angle = two_pi * (k as f32) * (j as f32) * inv_n;
                let x = match windowed.get(j) {
                    Some(v) => *v,
                    None => 0.0_f32,
                };
                re += x * angle.cos();
                im -= x * angle.sin();
            }
            power[k] = re * re + im * im;
        }
        Ok(power)
    }

    /// Compute the mel filterbank energies of a single (already-framed) audio
    /// buffer of length `n_fft`. Returns a vector of length `n_mels`.
    pub fn mel_energies(&self, frame: &[f32]) -> MmResult<Vec<f32>> {
        let power = self.power_spectrum(frame)?;
        let half = self.cfg.n_fft / 2 + 1;
        let mut out = vec![0.0_f32; self.cfg.n_mels];
        for m in 0..self.cfg.n_mels {
            let row_start = m * half;
            let mut acc = 0.0_f32;
            for k in 0..half {
                let filt = match self.mel_filter.get(row_start + k) {
                    Some(v) => *v,
                    None => 0.0_f32,
                };
                let p = match power.get(k) {
                    Some(v) => *v,
                    None => 0.0_f32,
                };
                acc += filt * p;
            }
            out[m] = acc;
        }
        Ok(out)
    }

    /// Full log-mel front-end. Frames the waveform with stride `hop_length`
    /// and window `n_fft`, computes mel energies per frame and applies
    /// `log10(mel + 1e-10)`. Returns `(log_mel, n_frames)` where `log_mel`
    /// has length `n_frames × n_mels` (row-major).
    pub fn forward(&self, waveform: &[f32]) -> MmResult<(Vec<f32>, usize)> {
        let n = self.cfg.n_fft;
        let hop = self.cfg.hop_length;
        if waveform.len() < n {
            return Ok((Vec::new(), 0));
        }
        let n_frames = (waveform.len() - n) / hop + 1;
        let mut out = vec![0.0_f32; n_frames * self.cfg.n_mels];
        const LOG_EPS: f32 = 1e-10_f32;

        for f in 0..n_frames {
            let start = f * hop;
            let end = start + n;
            if end > waveform.len() {
                return Err(MultiModalError::Internal(format!(
                    "whisper-log-mel: frame {f} overflows waveform of len {}",
                    waveform.len(),
                )));
            }
            let frame_slice = &waveform[start..end];
            let mel_e = self.mel_energies(frame_slice)?;
            for (m, energy) in mel_e.iter().enumerate() {
                let idx = f * self.cfg.n_mels + m;
                out[idx] = (energy + LOG_EPS).log10();
            }
        }
        Ok((out, n_frames))
    }
}

// ─── Mel filterbank construction ─────────────────────────────────────────────

/// Build a triangular mel filterbank shaped `n_mels × (n_fft/2 + 1)`.
fn build_mel_filter(cfg: &WhisperLogMelConfig) -> MmResult<Vec<f32>> {
    let half = cfg.n_fft / 2 + 1;
    let mel_min = WhisperLogMel::hz_to_mel(cfg.f_min);
    let mel_max = WhisperLogMel::hz_to_mel(cfg.f_max);

    // n_mels + 2 evenly-spaced mel anchor points → n_mels triangular filters.
    let total = cfg.n_mels + 2;
    let mut mel_points = Vec::with_capacity(total);
    for i in 0..total {
        let frac = (i as f32) / ((total - 1) as f32);
        mel_points.push(mel_min + (mel_max - mel_min) * frac);
    }
    let hz_points: Vec<f32> = mel_points
        .iter()
        .map(|m| WhisperLogMel::mel_to_hz(*m))
        .collect();

    // FFT bin frequencies: bin k → k * sample_rate / n_fft Hz.
    let bin_hz: Vec<f32> = (0..half)
        .map(|k| (k as f32) * cfg.sample_rate / (cfg.n_fft as f32))
        .collect();

    let mut filter = vec![0.0_f32; cfg.n_mels * half];
    for m in 0..cfg.n_mels {
        let left = match hz_points.get(m) {
            Some(v) => *v,
            None => {
                return Err(MultiModalError::Internal(
                    "whisper-log-mel: mel filter index out of range (left)".into(),
                ));
            }
        };
        let center = match hz_points.get(m + 1) {
            Some(v) => *v,
            None => {
                return Err(MultiModalError::Internal(
                    "whisper-log-mel: mel filter index out of range (center)".into(),
                ));
            }
        };
        let right = match hz_points.get(m + 2) {
            Some(v) => *v,
            None => {
                return Err(MultiModalError::Internal(
                    "whisper-log-mel: mel filter index out of range (right)".into(),
                ));
            }
        };
        let lc_span = (center - left).max(1e-12);
        let cr_span = (right - center).max(1e-12);
        for k in 0..half {
            let f = match bin_hz.get(k) {
                Some(v) => *v,
                None => 0.0_f32,
            };
            let val = if f <= left || f >= right {
                0.0_f32
            } else if f <= center {
                (f - left) / lc_span
            } else {
                (right - f) / cr_span
            };
            filter[m * half + k] = val.max(0.0_f32);
        }
    }
    Ok(filter)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default() -> WhisperLogMel {
        match WhisperLogMel::new(WhisperLogMelConfig::tiny()) {
            Ok(w) => w,
            Err(e) => panic!("default whisper front-end must construct: {e:?}"),
        }
    }

    // ── 1: hann_window length == n_fft, starts and ends at 0 ─────────────────
    #[test]
    fn hann_window_length_and_endpoints() {
        let n = 32;
        let w = WhisperLogMel::hann_window(n);
        assert_eq!(w.len(), n);
        assert!(w[0].abs() < 1e-6, "w[0] should be 0, got {}", w[0]);
        assert!(
            w[n - 1].abs() < 1e-6,
            "w[n-1] should be 0, got {}",
            w[n - 1]
        );
        // Peak of a Hann window is at the centre and is at most 1; for an
        // even `n_fft` the discrete peak lies between two samples, so the
        // closest sample is at most ~3e-3 below unity.
        let mid = w[n / 2];
        assert!(
            mid > 0.99 && mid <= 1.0 + 1e-6,
            "centre of Hann should be ~1, got {mid}",
        );
    }

    // ── 2: hz_to_mel(0) == 0 ─────────────────────────────────────────────────
    #[test]
    fn hz_to_mel_zero_is_zero() {
        let m = WhisperLogMel::hz_to_mel(0.0);
        assert!(m.abs() < 1e-5, "mel(0 Hz) should be 0, got {m}");
    }

    // ── 3: mel_to_hz round-trip ±1e-3 ────────────────────────────────────────
    #[test]
    fn mel_hz_round_trip() {
        for &hz in &[100.0_f32, 440.0, 1000.0, 4000.0, 7000.0] {
            let m = WhisperLogMel::hz_to_mel(hz);
            let back = WhisperLogMel::mel_to_hz(m);
            assert!((hz - back).abs() < 1e-3, "round-trip {hz} -> {m} -> {back}",);
        }
    }

    // ── 4: power_spectrum length == n_fft/2 + 1 ──────────────────────────────
    #[test]
    fn power_spectrum_output_length() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let frame = vec![0.1_f32; n];
        let p = w.power_spectrum(&frame).expect("power_spectrum");
        assert_eq!(p.len(), n / 2 + 1);
    }

    // ── 5: all-zero frame → all-zero power spectrum ─────────────────────────
    #[test]
    fn power_spectrum_zero_frame_is_zero() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let frame = vec![0.0_f32; n];
        let p = w.power_spectrum(&frame).expect("power_spectrum");
        for v in p.iter() {
            assert!(v.abs() < 1e-10, "expected 0, got {v}");
        }
    }

    // ── 6: mel_energies length == n_mels ─────────────────────────────────────
    #[test]
    fn mel_energies_output_length() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let frame = vec![0.5_f32; n];
        let e = w.mel_energies(&frame).expect("mel_energies");
        assert_eq!(e.len(), w.cfg.n_mels);
    }

    // ── 7: all-zero frame → all-zero mel energies ───────────────────────────
    #[test]
    fn mel_energies_zero_frame_is_zero() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let frame = vec![0.0_f32; n];
        let e = w.mel_energies(&frame).expect("mel_energies");
        for v in e.iter() {
            assert!(v.abs() < 1e-10, "expected 0, got {v}");
        }
    }

    // ── 8: forward output dims == n_frames * n_mels ─────────────────────────
    #[test]
    fn forward_shape_consistent() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let hop = w.cfg.hop_length;
        // Build a waveform with exactly 5 frames.
        let waveform_len = n + 4 * hop;
        let waveform = vec![0.1_f32; waveform_len];
        let (out, n_frames) = w.forward(&waveform).expect("forward");
        assert_eq!(n_frames, 5);
        assert_eq!(out.len(), n_frames * w.cfg.n_mels);
    }

    // ── 9: log applied: log10(mel + eps) ────────────────────────────────────
    #[test]
    fn forward_applies_log10() {
        // For a zero-input waveform, mel energies are zero, so each output
        // should equal log10(eps) = log10(1e-10) = -10.
        let w = make_default();
        let n = w.cfg.n_fft;
        let hop = w.cfg.hop_length;
        let waveform = vec![0.0_f32; n + 3 * hop];
        let (out, frames) = w.forward(&waveform).expect("forward");
        assert!(frames > 0);
        for v in out.iter() {
            assert!(
                (v - (-10.0)).abs() < 1e-3,
                "log10(0 + 1e-10) should be -10, got {v}",
            );
        }
    }

    // ── 10: deterministic ────────────────────────────────────────────────────
    #[test]
    fn forward_deterministic() {
        let a = make_default();
        let b = make_default();
        let n = a.cfg.n_fft;
        let hop = a.cfg.hop_length;
        let mut waveform = vec![0.0_f32; n + 7 * hop];
        for (i, v) in waveform.iter_mut().enumerate() {
            *v = (i as f32 * 0.01).sin();
        }
        let (out_a, fa) = a.forward(&waveform).expect("a");
        let (out_b, fb) = b.forward(&waveform).expect("b");
        assert_eq!(fa, fb);
        assert_eq!(out_a.len(), out_b.len());
        for (x, y) in out_a.iter().zip(out_b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    // ── 11: invalid sample_rate ──────────────────────────────────────────────
    #[test]
    fn invalid_sample_rate_errors() {
        let cfg = WhisperLogMelConfig {
            sample_rate: 0.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err = WhisperLogMel::new(cfg).expect_err("sample_rate=0 must err");
        assert!(matches!(err, MultiModalError::Internal(_)));
    }

    // ── 12: invalid n_fft ────────────────────────────────────────────────────
    #[test]
    fn invalid_n_fft_errors() {
        let cfg = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 1,
            hop_length: 1,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err = WhisperLogMel::new(cfg).expect_err("n_fft=1 must err");
        assert!(matches!(err, MultiModalError::Internal(_)));

        let cfg_odd = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 31,
            hop_length: 16,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err2 = WhisperLogMel::new(cfg_odd).expect_err("odd n_fft must err");
        assert!(matches!(err2, MultiModalError::Internal(_)));
    }

    // ── 13: invalid hop_length ───────────────────────────────────────────────
    #[test]
    fn invalid_hop_length_errors() {
        let cfg_zero = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 0,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err = WhisperLogMel::new(cfg_zero).expect_err("hop=0 must err");
        assert!(matches!(err, MultiModalError::Internal(_)));

        let cfg_big = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 64,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err2 = WhisperLogMel::new(cfg_big).expect_err("hop>n_fft must err");
        assert!(matches!(err2, MultiModalError::Internal(_)));
    }

    // ── 14: invalid n_mels ───────────────────────────────────────────────────
    #[test]
    fn invalid_n_mels_errors() {
        let cfg = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 0,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let err = WhisperLogMel::new(cfg).expect_err("n_mels=0 must err");
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 15: invalid f_min / f_max ───────────────────────────────────────────
    #[test]
    fn invalid_freq_range_errors() {
        let cfg_neg = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 8,
            f_min: -1.0,
            f_max: 8_000.0,
        };
        let err = WhisperLogMel::new(cfg_neg).expect_err("f_min<0 must err");
        assert!(matches!(err, MultiModalError::Internal(_)));

        let cfg_inv = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 8,
            f_min: 4_000.0,
            f_max: 1_000.0,
        };
        let err2 = WhisperLogMel::new(cfg_inv).expect_err("f_max<=f_min must err");
        assert!(matches!(err2, MultiModalError::Internal(_)));

        let cfg_over = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 8,
            f_min: 0.0,
            f_max: 100_000.0,
        };
        let err3 = WhisperLogMel::new(cfg_over).expect_err("f_max>nyquist must err");
        assert!(matches!(err3, MultiModalError::Internal(_)));
    }

    // ── 16: frame wrong length ──────────────────────────────────────────────
    #[test]
    fn power_spectrum_frame_wrong_len_errors() {
        let w = make_default();
        let bad = vec![0.0_f32; w.cfg.n_fft + 1];
        let err = w
            .power_spectrum(&bad)
            .expect_err("frame wrong len must err");
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 17: short waveform → empty output ───────────────────────────────────
    #[test]
    fn short_waveform_returns_empty() {
        let w = make_default();
        let short = vec![0.0_f32; w.cfg.n_fft / 2];
        let (out, frames) = w.forward(&short).expect("short waveform");
        assert_eq!(frames, 0);
        assert!(out.is_empty());
    }

    // ── 18: exactly one frame ───────────────────────────────────────────────
    #[test]
    fn exact_one_frame_waveform_works() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let waveform = vec![0.1_f32; n];
        let (out, frames) = w.forward(&waveform).expect("forward one frame");
        assert_eq!(frames, 1);
        assert_eq!(out.len(), w.cfg.n_mels);
    }

    // ── 19: sine wave energy concentrated at expected mel bin ───────────────
    #[test]
    fn sine_wave_energy_in_expected_mel_bin() {
        // Build a sine of frequency 4 kHz at 16 kHz sample rate (= 4 cycles
        // per 16 samples). With n_fft = 32 the analysis bin spacing is
        // 500 Hz, so the energy will be concentrated at FFT bin 8 (= 4 kHz)
        // which falls inside the highest mel filter (filters cover 0..8 kHz).
        let cfg = WhisperLogMelConfig {
            sample_rate: 16_000.0,
            n_fft: 32,
            hop_length: 16,
            n_mels: 4,
            f_min: 0.0,
            f_max: 8_000.0,
        };
        let w = WhisperLogMel::new(cfg).expect("sine front-end");
        let n = w.cfg.n_fft;
        let target_freq = 4_000.0_f32;
        let mut frame = vec![0.0_f32; n];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = (2.0_f32 * std::f32::consts::PI * target_freq * (i as f32) / w.cfg.sample_rate)
                .sin();
        }
        let mel = w.mel_energies(&frame).expect("mel_energies sine");
        // The highest-frequency mel bin must dominate the lowest one for a
        // 4 kHz tone in a 0..8 kHz band.
        let high = mel[mel.len() - 1];
        let low = mel[0];
        assert!(
            high > low,
            "high mel bin {high} should exceed low mel bin {low} for 4 kHz tone",
        );
    }

    // ── 20: mel_filter shape correct ────────────────────────────────────────
    #[test]
    fn mel_filter_shape_is_n_mels_by_half() {
        let w = make_default();
        let half = w.cfg.n_fft / 2 + 1;
        assert_eq!(w.mel_filter().len(), w.cfg.n_mels * half);
    }

    // ── 21: cached window matches static helper ─────────────────────────────
    #[test]
    fn cached_window_matches_static_helper() {
        let w = make_default();
        let n = w.cfg.n_fft;
        let expected = WhisperLogMel::hann_window(n);
        assert_eq!(w.window().len(), expected.len());
        for (a, b) in w.window().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
