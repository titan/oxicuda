//! Onset detection and tempo (beat-rate) estimation.
//!
//! The pipeline follows the standard music-information-retrieval recipe:
//!
//! 1. **Onset strength envelope** — frame the signal with a Hann-windowed DFT
//!    and compute the *spectral flux*, the half-wave-rectified sum of positive
//!    bin-to-bin magnitude increases:
//!    `flux(t) = Σ_k max(0, |X(t,k)| − |X(t−1,k)|)`.
//!    Rising spectral energy (a new note / percussive hit) produces a peak.
//! 2. **Peak picking** — locate local maxima of the (mean-subtracted) onset
//!    envelope that exceed an adaptive threshold `δ + λ·moving_mean`, separated
//!    by a refractory gap.
//! 3. **Tempo** — autocorrelate the onset envelope; the lag of the dominant
//!    peak (restricted to a plausible BPM window) gives the beat period, and
//!    `tempo = 60 · fps / lag` (BPM), where `fps = sample_rate / hop_length`.
//!
//! Everything is pure Rust with a direct DFT (no FFT dependency); this keeps
//! the module self-contained at the cost of `O(n_fft²)` per frame.
//!
//! ## References
//! - Bello, J. P. et al. (2005). "A tutorial on onset detection in music
//!   signals." IEEE TSALP 13(5).
//! - Ellis, D. P. W. (2007). "Beat tracking by dynamic programming." J. New
//!   Music Research 36(1).

use std::f32::consts::PI;

use crate::error::{AudioError, AudioResult};

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration shared by the onset / tempo routines.
#[derive(Debug, Clone)]
pub struct OnsetConfig {
    /// Audio sample rate in Hz (> 0).
    pub sample_rate: f32,
    /// FFT / frame length (≥ 2).
    pub n_fft: usize,
    /// Hop length in samples (≥ 1).
    pub hop_length: usize,
}

impl OnsetConfig {
    /// Default 22.05 kHz music front-end (≈ 46 ms / 11.6 ms framing).
    #[must_use]
    pub fn default_music() -> Self {
        Self {
            sample_rate: 22_050.0,
            n_fft: 1024,
            hop_length: 256,
        }
    }

    /// Frames per second of the onset envelope (`sample_rate / hop_length`).
    #[must_use]
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate / self.hop_length as f32
    }
}

// ─── Framing / spectrum helpers ─────────────────────────────────────────────────

fn frame_count(n_samples: usize, n_fft: usize, hop_length: usize) -> usize {
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
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / (n - 1) as f32).cos()))
        .collect()
}

fn validate(cfg: &OnsetConfig, n_samples: usize) -> AudioResult<()> {
    if cfg.sample_rate <= 0.0 {
        return Err(AudioError::Internal(format!(
            "onset: sample_rate must be > 0, got {}",
            cfg.sample_rate
        )));
    }
    if cfg.n_fft < 2 {
        return Err(AudioError::ShapeMismatch {
            msg: format!("onset: n_fft must be ≥ 2, got {}", cfg.n_fft),
        });
    }
    if cfg.hop_length == 0 {
        return Err(AudioError::InvalidStride(0));
    }
    if n_samples == 0 {
        return Err(AudioError::EmptyInput {
            msg: "onset: empty signal".into(),
        });
    }
    if n_samples < cfg.n_fft {
        return Err(AudioError::InvalidSequenceLength(n_samples));
    }
    Ok(())
}

/// Magnitude spectrogram `[n_frames, n_fft/2+1]` via Hann-windowed direct DFT.
fn magnitude_spectrogram(signal: &[f32], cfg: &OnsetConfig) -> Vec<f32> {
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
            let omega = -2.0 * PI * k as f32 / cfg.n_fft as f32;
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

// ─── Onset strength envelope ────────────────────────────────────────────────────

/// Compute the **spectral-flux onset strength envelope**, length `n_frames`.
///
/// `flux(t) = Σ_k max(0, |X(t,k)| − |X(t−1,k)|)`, with `flux(0) = 0`. The
/// envelope is non-negative; larger values indicate stronger note onsets.
///
/// # Errors
/// See module validation (`sample_rate` / `n_fft` / `hop_length`, empty signal,
/// `signal.len() < n_fft`).
pub fn onset_strength(signal: &[f32], cfg: &OnsetConfig) -> AudioResult<Vec<f32>> {
    validate(cfg, signal.len())?;
    let n_bins = cfg.n_fft / 2 + 1;
    let n_frames = frame_count(signal.len(), cfg.n_fft, cfg.hop_length);
    let spec = magnitude_spectrogram(signal, cfg);
    let mut env = vec![0.0_f32; n_frames];
    for t in 1..n_frames {
        let cur = &spec[t * n_bins..(t + 1) * n_bins];
        let prev = &spec[(t - 1) * n_bins..t * n_bins];
        let mut flux = 0.0_f32;
        for (c, p) in cur.iter().zip(prev.iter()) {
            let d = c - p;
            if d > 0.0 {
                flux += d;
            }
        }
        env[t] = flux;
    }
    Ok(env)
}

// ─── Peak picking ───────────────────────────────────────────────────────────────

/// Parameters governing adaptive peak picking on an onset envelope.
#[derive(Debug, Clone)]
pub struct PeakPickConfig {
    /// Half-width (in frames) of the moving-average window for the local mean.
    pub window: usize,
    /// Multiplicative factor on the local mean (`λ`).
    pub mean_factor: f32,
    /// Absolute threshold offset added to `λ·local_mean` (`δ`).
    pub delta: f32,
    /// Minimum gap (in frames) between successive accepted peaks.
    pub min_separation: usize,
}

impl Default for PeakPickConfig {
    fn default() -> Self {
        Self {
            window: 6,
            mean_factor: 1.0,
            delta: 0.0,
            min_separation: 3,
        }
    }
}

/// Pick onset peaks from an onset-strength envelope.
///
/// A frame `t` is accepted as a peak when it is a strict local maximum within
/// `±1` frame, exceeds the adaptive threshold `δ + λ · mean(env[t−w .. t+w])`,
/// and is at least `min_separation` frames after the previous accepted peak.
/// Returns the accepted **frame indices** in ascending order.
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `env` is empty.
pub fn pick_peaks(env: &[f32], cfg: &PeakPickConfig) -> AudioResult<Vec<usize>> {
    if env.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "onset: empty envelope".into(),
        });
    }
    let n = env.len();
    let w = cfg.window;
    let mut peaks = Vec::new();
    let mut last_peak: Option<usize> = None;
    for t in 0..n {
        // Local maximum (treat boundaries as one-sided).
        let left_ok = t == 0 || env[t] > env[t - 1];
        let right_ok = t + 1 >= n || env[t] >= env[t + 1];
        if !(left_ok && right_ok) {
            continue;
        }
        // Adaptive threshold from the local window mean.
        let lo = t.saturating_sub(w);
        let hi = (t + w + 1).min(n);
        let mean = env[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        let threshold = cfg.delta + cfg.mean_factor * mean;
        if env[t] <= threshold {
            continue;
        }
        if let Some(prev) = last_peak {
            if t - prev < cfg.min_separation {
                // Keep the stronger of the two within the refractory window.
                if env[t] > env[prev] {
                    peaks.pop();
                    peaks.push(t);
                    last_peak = Some(t);
                }
                continue;
            }
        }
        peaks.push(t);
        last_peak = Some(t);
    }
    Ok(peaks)
}

/// Convenience: detect onset frame indices directly from a signal.
///
/// # Errors
/// As [`onset_strength`] and [`pick_peaks`].
pub fn detect_onsets(
    signal: &[f32],
    cfg: &OnsetConfig,
    pick: &PeakPickConfig,
) -> AudioResult<Vec<usize>> {
    let env = onset_strength(signal, cfg)?;
    pick_peaks(&env, pick)
}

/// Convert onset **frame indices** to onset **times** in seconds.
#[must_use]
pub fn onset_times(frames: &[usize], cfg: &OnsetConfig) -> Vec<f32> {
    let fps = cfg.frames_per_second();
    frames.iter().map(|&f| f as f32 / fps).collect()
}

// ─── Tempo estimation ───────────────────────────────────────────────────────────

/// Result of a tempo (beat-rate) estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    /// Estimated tempo in beats per minute.
    pub bpm: f32,
    /// Beat period in frames (the dominant autocorrelation lag).
    pub period_frames: usize,
    /// Normalised strength of the chosen autocorrelation peak in `[0, 1]`.
    pub strength: f32,
}

/// Estimate **tempo** (BPM) from an onset-strength envelope by autocorrelation.
///
/// The envelope is mean-removed and autocorrelated; the dominant lag within
/// the BPM range `[min_bpm, max_bpm]` is selected. `tempo = 60 · fps / lag`.
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `env` is empty.
/// - [`AudioError::Internal`] on invalid `fps` or BPM range, or when the
///   envelope is too short for the requested range.
pub fn tempo_from_envelope(
    env: &[f32],
    fps: f32,
    min_bpm: f32,
    max_bpm: f32,
) -> AudioResult<TempoEstimate> {
    if env.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "tempo: empty envelope".into(),
        });
    }
    if fps <= 0.0 {
        return Err(AudioError::Internal(format!(
            "tempo: frames-per-second must be > 0, got {fps}"
        )));
    }
    if !(min_bpm > 0.0 && max_bpm > min_bpm) {
        return Err(AudioError::Internal(format!(
            "tempo: require 0 < min_bpm < max_bpm, got {min_bpm}, {max_bpm}"
        )));
    }
    let n = env.len();
    // Mean-remove the envelope so the autocorrelation peaks at periodicities.
    let mean = env.iter().sum::<f32>() / n as f32;
    let centred: Vec<f32> = env.iter().map(|&v| v - mean).collect();

    // Lag range in frames from the BPM range (higher BPM ⇒ smaller lag).
    let lag_min = ((60.0 * fps / max_bpm).floor() as usize).max(1);
    let lag_max = ((60.0 * fps / min_bpm).ceil() as usize).min(n - 1);
    if lag_max <= lag_min {
        return Err(AudioError::Internal(
            "tempo: envelope too short for the requested BPM range".into(),
        ));
    }

    let r0: f32 = centred.iter().map(|&v| v * v).sum::<f32>().max(1e-12);
    let mut best_lag = lag_min;
    let mut best_val = f32::NEG_INFINITY;
    for lag in lag_min..=lag_max {
        let mut acc = 0.0_f32;
        for j in 0..(n - lag) {
            acc += centred[j] * centred[j + lag];
        }
        if acc > best_val {
            best_val = acc;
            best_lag = lag;
        }
    }
    let bpm = 60.0 * fps / best_lag as f32;
    let strength = (best_val / r0).clamp(0.0, 1.0);
    Ok(TempoEstimate {
        bpm,
        period_frames: best_lag,
        strength,
    })
}

/// Convenience: estimate tempo directly from a signal.
///
/// # Errors
/// As [`onset_strength`] and [`tempo_from_envelope`].
pub fn estimate_tempo(
    signal: &[f32],
    cfg: &OnsetConfig,
    min_bpm: f32,
    max_bpm: f32,
) -> AudioResult<TempoEstimate> {
    let env = onset_strength(signal, cfg)?;
    tempo_from_envelope(&env, cfg.frames_per_second(), min_bpm, max_bpm)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn cfg() -> OnsetConfig {
        OnsetConfig {
            sample_rate: 22_050.0,
            n_fft: 512,
            hop_length: 128,
        }
    }

    /// A click train: short broadband bursts every `interval` samples.
    fn click_train(interval: usize, n: usize) -> Vec<f32> {
        let mut x = vec![0.0_f32; n];
        let mut t = interval; // first click not at index 0 so flux can rise
        while t < n {
            // A short decaying burst (a few samples) to create a clear onset.
            for d in 0..20usize {
                if t + d < n {
                    let env = (-(d as f32) / 5.0).exp();
                    // pseudo-broadband content
                    x[t + d] += env * (((d * 53 + t) % 7) as f32 / 3.0 - 1.0);
                }
            }
            t += interval;
        }
        x
    }

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn onset_strength_nonneg_and_shape() {
        let c = cfg();
        let sig = click_train(2048, 16_384);
        let env = onset_strength(&sig, &c).expect("ok");
        let expected = frame_count(sig.len(), c.n_fft, c.hop_length);
        assert_eq!(env.len(), expected);
        assert!(env.iter().all(|&v| v >= 0.0 && v.is_finite()));
        assert_eq!(env[0], 0.0, "first frame flux must be 0");
    }

    #[test]
    fn onset_strength_steady_tone_low_flux() {
        // A steady sine has near-zero flux after the first couple of frames.
        let c = cfg();
        let sig = sine(440.0, c.sample_rate, 16_384);
        let env = onset_strength(&sig, &c).expect("ok");
        let tail_mean = env[5..].iter().sum::<f32>() / (env.len() - 5) as f32;
        let peak = env.iter().fold(0.0_f32, |m, &v| m.max(v));
        assert!(
            tail_mean < 0.5 * peak.max(1e-6) + 1e-3,
            "tail {tail_mean} peak {peak}"
        );
    }

    #[test]
    fn onset_strength_detects_energy_increase() {
        // Concatenate silence then a tone: flux should spike at the boundary.
        let c = cfg();
        let mut sig = vec![0.0_f32; 4096];
        sig.extend(sine(880.0, c.sample_rate, 4096));
        let env = onset_strength(&sig, &c).expect("ok");
        let peak = env.iter().fold(0.0_f32, |m, &v| m.max(v));
        assert!(peak > 0.0, "expected a positive flux peak");
    }

    #[test]
    fn onset_strength_empty_error() {
        let c = cfg();
        assert!(matches!(
            onset_strength(&[], &c).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn onset_strength_short_error() {
        let c = cfg();
        let sig = sine(440.0, c.sample_rate, 100);
        assert!(matches!(
            onset_strength(&sig, &c).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn pick_peaks_finds_periodic_onsets() {
        let c = cfg();
        let interval = 2048usize;
        let sig = click_train(interval, 16_384);
        let env = onset_strength(&sig, &c).expect("ok");
        let pp = PeakPickConfig {
            window: 4,
            mean_factor: 1.0,
            delta: 0.0,
            min_separation: 3,
        };
        let peaks = pick_peaks(&env, &pp).expect("ok");
        // ~7 clicks in 16384 samples; expect several detected onsets.
        assert!(peaks.len() >= 3, "got {} peaks", peaks.len());
        // Peaks must be in ascending order and respect the separation.
        for w in peaks.windows(2) {
            assert!(w[1] > w[0]);
            assert!(w[1] - w[0] >= pp.min_separation);
        }
    }

    #[test]
    fn pick_peaks_high_threshold_suppresses() {
        let c = cfg();
        let sig = click_train(2048, 16_384);
        let env = onset_strength(&sig, &c).expect("ok");
        let strict = PeakPickConfig {
            window: 4,
            mean_factor: 100.0,
            delta: 1e9,
            min_separation: 3,
        };
        let peaks = pick_peaks(&env, &strict).expect("ok");
        assert!(peaks.is_empty(), "huge threshold should reject all peaks");
    }

    #[test]
    fn pick_peaks_empty_error() {
        let pp = PeakPickConfig::default();
        assert!(matches!(
            pick_peaks(&[], &pp).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn pick_peaks_single_clear_peak() {
        // A single spike in an otherwise flat envelope must be picked.
        let mut env = vec![0.1_f32; 50];
        env[25] = 10.0;
        let pp = PeakPickConfig {
            window: 5,
            mean_factor: 1.0,
            delta: 0.0,
            min_separation: 2,
        };
        let peaks = pick_peaks(&env, &pp).expect("ok");
        assert!(peaks.contains(&25), "peak at 25 missing: {peaks:?}");
    }

    #[test]
    fn detect_onsets_end_to_end() {
        let c = cfg();
        let sig = click_train(2048, 16_384);
        let pp = PeakPickConfig::default();
        let frames = detect_onsets(&sig, &c, &pp).expect("ok");
        assert!(!frames.is_empty());
        let times = onset_times(&frames, &c);
        // Times are non-decreasing and start at ≥ 0.
        assert!(times.iter().all(|&t| t >= 0.0));
        for w in times.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn tempo_recovers_click_rate() {
        // Clicks every 2048 samples at 22050 Hz → period 2048/22050 s →
        // ≈ 10.77 Hz → ≈ 646 BPM (sub-harmonics also valid). Use a wide range
        // and check the detected period matches the click spacing in frames.
        let c = cfg();
        let interval = 4096usize; // ~5.38 Hz click → ~323 BPM
        let sig = click_train(interval, 32_768);
        let t = estimate_tempo(&sig, &c, 60.0, 400.0).expect("ok");
        // Click spacing in frames = interval / hop_length.
        let expected_period = interval / c.hop_length; // 32 frames
        // The detected period should be the click period or a small multiple.
        let ratio = t.period_frames as f32 / expected_period as f32;
        let near_integer = (ratio - ratio.round()).abs() < 0.2
            || ((1.0 / ratio) - (1.0 / ratio).round()).abs() < 0.2;
        assert!(
            near_integer,
            "period {} not harmonically related to {}",
            t.period_frames, expected_period
        );
        assert!(t.bpm > 0.0 && t.bpm.is_finite());
    }

    #[test]
    fn tempo_strength_in_unit_range() {
        let c = cfg();
        let sig = click_train(4096, 32_768);
        let t = estimate_tempo(&sig, &c, 60.0, 300.0).expect("ok");
        assert!((0.0..=1.0).contains(&t.strength));
    }

    #[test]
    fn tempo_from_envelope_period_in_range() {
        // Construct a synthetic periodic envelope with period 20 frames.
        let period = 20usize;
        let env: Vec<f32> = (0..400)
            .map(|i| if i % period == 0 { 1.0 } else { 0.05 })
            .collect();
        let fps = 100.0_f32;
        // period 20 frames at 100 fps → 5 s? no: 0.2 s → 300 BPM.
        let t = tempo_from_envelope(&env, fps, 100.0, 400.0).expect("ok");
        assert_eq!(t.period_frames, period, "detected {}", t.period_frames);
        let expected_bpm = 60.0 * fps / period as f32;
        assert!((t.bpm - expected_bpm).abs() < 1.0, "bpm {}", t.bpm);
    }

    #[test]
    fn tempo_empty_error() {
        assert!(matches!(
            tempo_from_envelope(&[], 100.0, 60.0, 200.0).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn tempo_bad_bpm_range_error() {
        let env = vec![1.0_f32; 200];
        assert!(matches!(
            tempo_from_envelope(&env, 100.0, 200.0, 60.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn tempo_bad_fps_error() {
        let env = vec![1.0_f32; 200];
        assert!(matches!(
            tempo_from_envelope(&env, 0.0, 60.0, 200.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn frames_per_second_matches() {
        let c = cfg();
        assert!((c.frames_per_second() - 22_050.0 / 128.0).abs() < 1e-3);
    }

    #[test]
    fn onset_deterministic() {
        let c = cfg();
        let sig = click_train(2048, 16_384);
        let a = onset_strength(&sig, &c).expect("ok");
        let b = onset_strength(&sig, &c).expect("ok");
        assert_eq!(a, b);
    }
}
