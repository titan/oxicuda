//! YIN fundamental-frequency (pitch) estimator.
//!
//! Implements the YIN algorithm of de Cheveigné & Kawahara (2002), a robust
//! time-domain pitch tracker. The full pipeline has six steps; this module
//! implements the four that determine the estimate plus parabolic refinement:
//!
//! 1. **Difference function**
//!    `d(τ) = Σ_{j} (x_j − x_{j+τ})²` for `τ ∈ [0, W]`.
//! 2. **Cumulative mean normalised difference (CMND)**
//!    `d'(0) = 1`, `d'(τ) = d(τ) / ((1/τ) · Σ_{k=1}^{τ} d(k))`.
//! 3. **Absolute threshold** — pick the smallest `τ` with `d'(τ) < threshold`
//!    that is a local minimum (a dip below the threshold); if none, take the
//!    global minimum of `d'`.
//! 4. **Parabolic interpolation** — refine the integer lag `τ` to sub-sample
//!    precision using the neighbouring CMND values.
//!
//! The estimated pitch is `f0 = fs / τ*`. A frame is declared **unvoiced**
//! (returns `None`) when the best CMND value exceeds the threshold *and* the
//! aperiodicity is high, or when `τ*` falls outside `[f_min, f_max]`.
//!
//! ## References
//! - de Cheveigné, A. & Kawahara, H. (2002). "YIN, a fundamental frequency
//!   estimator for speech and music." JASA 111(4), 1917–1930.

use crate::error::{AudioError, AudioResult};

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for [`yin_pitch`].
#[derive(Debug, Clone)]
pub struct YinConfig {
    /// Audio sample rate in Hz (> 0).
    pub sample_rate: f32,
    /// Absolute YIN threshold on the CMND function (typically 0.1–0.15, in `(0, 1)`).
    pub threshold: f32,
    /// Minimum detectable fundamental in Hz (> 0).
    pub f_min: f32,
    /// Maximum detectable fundamental in Hz (`f_min < f_max ≤ sample_rate/2`).
    pub f_max: f32,
}

impl YinConfig {
    /// A reasonable default for speech / music at 16 kHz (≈ 65 Hz–2 kHz).
    #[must_use]
    pub fn default_16k() -> Self {
        Self {
            sample_rate: 16_000.0,
            threshold: 0.1,
            f_min: 65.0,
            f_max: 2_000.0,
        }
    }
}

/// Result of a single-frame YIN estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YinEstimate {
    /// Estimated fundamental frequency in Hz.
    pub f0: f32,
    /// Estimated lag (period) in samples (sub-sample, after interpolation).
    pub period: f32,
    /// The CMND value at the chosen lag (lower ⇒ more periodic; `≈ 1 − clarity`).
    pub cmnd: f32,
}

// ─── Validation ─────────────────────────────────────────────────────────────────

fn validate(cfg: &YinConfig, n: usize) -> AudioResult<()> {
    if cfg.sample_rate <= 0.0 {
        return Err(AudioError::Internal(format!(
            "yin: sample_rate must be > 0, got {}",
            cfg.sample_rate
        )));
    }
    if !(cfg.threshold > 0.0 && cfg.threshold < 1.0) {
        return Err(AudioError::Internal(format!(
            "yin: threshold must be in (0, 1), got {}",
            cfg.threshold
        )));
    }
    if cfg.f_min <= 0.0 || cfg.f_max <= cfg.f_min {
        return Err(AudioError::Internal(format!(
            "yin: require 0 < f_min < f_max, got f_min={}, f_max={}",
            cfg.f_min, cfg.f_max
        )));
    }
    if cfg.f_max > cfg.sample_rate / 2.0 + 1e-3 {
        return Err(AudioError::Internal(format!(
            "yin: f_max ({}) exceeds Nyquist ({})",
            cfg.f_max,
            cfg.sample_rate / 2.0
        )));
    }
    if n == 0 {
        return Err(AudioError::EmptyInput {
            msg: "yin: empty signal".into(),
        });
    }
    // Need at least two full periods of the lowest detectable frequency.
    let max_tau = (cfg.sample_rate / cfg.f_min).ceil() as usize;
    if n < 2 * max_tau {
        return Err(AudioError::InvalidSequenceLength(n));
    }
    Ok(())
}

// ─── YIN core ───────────────────────────────────────────────────────────────────

/// Estimate the fundamental frequency of a single analysis frame.
///
/// Returns `Some(estimate)` for a voiced frame, or `None` when no pitch is
/// detected (silence / aperiodic, or the period leaves `[f_min, f_max]`).
///
/// # Errors
/// - [`AudioError::Internal`] on invalid `sample_rate` / `threshold` /
///   frequency range.
/// - [`AudioError::EmptyInput`] on an empty frame.
/// - [`AudioError::InvalidSequenceLength`] when the frame is shorter than two
///   periods of `f_min`.
pub fn yin_pitch(frame: &[f32], cfg: &YinConfig) -> AudioResult<Option<YinEstimate>> {
    let n = frame.len();
    validate(cfg, n)?;

    // Lag search bounds from the frequency range.
    let tau_min = ((cfg.sample_rate / cfg.f_max).floor() as usize).max(1);
    let tau_max = ((cfg.sample_rate / cfg.f_min).ceil() as usize).min(n / 2);
    if tau_max <= tau_min {
        return Ok(None);
    }

    // Silence guard: if the frame has negligible energy, declare unvoiced.
    let energy: f32 = frame.iter().map(|&x| x * x).sum();
    if energy < 1e-10 {
        return Ok(None);
    }

    // ── Step 1: difference function d(τ) for τ ∈ [0, tau_max] ────────────────
    let mut diff = vec![0.0_f32; tau_max + 1];
    let window = n - tau_max; // number of comparison samples
    for tau in 1..=tau_max {
        let mut sum = 0.0_f32;
        for j in 0..window {
            let delta = frame[j] - frame[j + tau];
            sum += delta * delta;
        }
        diff[tau] = sum;
    }

    // ── Step 2: cumulative mean normalised difference d'(τ) ──────────────────
    let mut cmnd = vec![1.0_f32; tau_max + 1];
    let mut running = 0.0_f32;
    for tau in 1..=tau_max {
        running += diff[tau];
        cmnd[tau] = if running > 1e-12 {
            diff[tau] * tau as f32 / running
        } else {
            1.0
        };
    }

    // ── Step 3: absolute threshold within [tau_min, tau_max] ─────────────────
    let mut best_tau = 0usize;
    let mut tau = tau_min;
    while tau <= tau_max {
        if cmnd[tau] < cfg.threshold {
            // Descend to the local minimum of this dip.
            let mut t = tau;
            while t < tau_max && cmnd[t + 1] < cmnd[t] {
                t += 1;
            }
            best_tau = t;
            break;
        }
        tau += 1;
    }

    // Fallback: no dip below threshold → global minimum of the CMND in range.
    if best_tau == 0 {
        let mut min_val = f32::INFINITY;
        for (t, &c) in cmnd.iter().enumerate().take(tau_max + 1).skip(tau_min) {
            if c < min_val {
                min_val = c;
                best_tau = t;
            }
        }
        // If even the best is far from periodic, declare unvoiced.
        if best_tau == 0 || min_val > 1.0 - 1e-6 {
            return Ok(None);
        }
        // High aperiodicity (CMND well above threshold) ⇒ treat as unvoiced.
        if min_val >= 0.5 {
            return Ok(None);
        }
    }

    // ── Step 4: parabolic interpolation around best_tau ──────────────────────
    let refined = parabolic_interpolation(&cmnd, best_tau, tau_max);
    let period = refined;
    if period <= 0.0 {
        return Ok(None);
    }
    let f0 = cfg.sample_rate / period;
    if f0 < cfg.f_min || f0 > cfg.f_max {
        return Ok(None);
    }

    Ok(Some(YinEstimate {
        f0,
        period,
        cmnd: cmnd[best_tau],
    }))
}

/// Parabolic interpolation of the lag minimum using the CMND triplet
/// `(τ−1, τ, τ+1)`. Falls back to the integer lag at the array edges.
fn parabolic_interpolation(cmnd: &[f32], tau: usize, tau_max: usize) -> f32 {
    if tau == 0 || tau >= tau_max {
        return tau as f32;
    }
    let s0 = cmnd[tau - 1];
    let s1 = cmnd[tau];
    let s2 = cmnd[tau + 1];
    let denom = s0 + s2 - 2.0 * s1;
    if denom.abs() < 1e-12 {
        return tau as f32;
    }
    let shift = 0.5 * (s0 - s2) / denom;
    // The vertex shift should be within ±1 sample; clamp defensively.
    tau as f32 + shift.clamp(-1.0, 1.0)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn cfg() -> YinConfig {
        YinConfig {
            sample_rate: 16_000.0,
            threshold: 0.15,
            f_min: 65.0,
            f_max: 2_000.0,
        }
    }

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn detects_sine_frequency() {
        let c = cfg();
        let freq = 220.0_f32;
        let sig = sine(freq, c.sample_rate, 2048);
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        let rel_err = (est.f0 - freq).abs() / freq;
        assert!(rel_err < 0.02, "estimated {} vs {freq}", est.f0);
    }

    #[test]
    fn output_positive() {
        let c = cfg();
        let sig = sine(330.0, c.sample_rate, 2048);
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        assert!(est.f0 > 0.0 && est.period > 0.0);
    }

    #[test]
    fn silence_no_pitch() {
        let c = cfg();
        let sig = vec![0.0_f32; 2048];
        let est = yin_pitch(&sig, &c).expect("ok");
        assert!(est.is_none(), "silence should not yield a pitch");
    }

    #[test]
    fn threshold_respected() {
        // With a low threshold a clean tone is still detected; the chosen lag's
        // CMND must lie below the configured threshold (true dip case).
        let mut c = cfg();
        c.threshold = 0.1;
        let sig = sine(440.0, c.sample_rate, 2048);
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        assert!(
            est.cmnd < c.threshold + 1e-3,
            "cmnd {} not < threshold",
            est.cmnd
        );
    }

    #[test]
    fn octave_handling() {
        // YIN should track the true fundamental, not the octave above/below.
        let c = cfg();
        let freq = 200.0_f32;
        let sig = sine(freq, c.sample_rate, 4096);
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        // It must not lock onto 100 Hz (octave below) or 400 Hz (octave above).
        assert!((est.f0 - freq).abs() < 20.0, "octave error: {}", est.f0);
    }

    #[test]
    fn harmonic_signal() {
        // Fundamental + harmonics; YIN should report the fundamental.
        let c = cfg();
        let fs = c.sample_rate;
        let f0 = 150.0_f32;
        let n = 4096;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                (2.0 * PI_F32 * f0 * t).sin()
                    + 0.5 * (2.0 * PI_F32 * 2.0 * f0 * t).sin()
                    + 0.3 * (2.0 * PI_F32 * 3.0 * f0 * t).sin()
            })
            .collect();
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        let rel_err = (est.f0 - f0).abs() / f0;
        assert!(rel_err < 0.03, "harmonic f0 {} vs {f0}", est.f0);
    }

    #[test]
    fn fs_affects_result() {
        // A fixed sample-lag period maps to different frequencies depending on
        // `sample_rate`: f0 = fs / τ. Hold the waveform's *sample* period fixed
        // (40 samples) and change fs → the detected f0 must scale with fs.
        let period_samples = 40usize;
        let n = 4096;
        // A tone with exactly `period_samples` samples per cycle (fs-independent
        // shape); only the interpretation as a frequency depends on `sample_rate`.
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI_F32 * i as f32 / period_samples as f32).sin())
            .collect();

        let mut ca = cfg();
        ca.sample_rate = 16_000.0;
        let est_a = yin_pitch(&sig, &ca).expect("ok").expect("voiced");

        let mut cb = cfg();
        cb.sample_rate = 8_000.0;
        let est_b = yin_pitch(&sig, &cb).expect("ok").expect("voiced");

        // Same sample-lag → ~equal detected periods, but f0 scales 2:1 with fs.
        assert!(
            (est_a.period - est_b.period).abs() < 2.0,
            "periods {} vs {}",
            est_a.period,
            est_b.period
        );
        assert!(
            (est_a.f0 - 2.0 * est_b.f0).abs() < 25.0,
            "f0 should scale with fs: {} vs {}",
            est_a.f0,
            est_b.f0
        );
    }

    #[test]
    fn window_too_short_error() {
        let c = cfg();
        // f_min = 65 Hz at 16 kHz → max lag ≈ 246; need ≥ 2× → ~492 samples.
        let sig = sine(440.0, c.sample_rate, 100);
        assert!(matches!(
            yin_pitch(&sig, &c).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn finite() {
        let c = cfg();
        let sig = sine(500.0, c.sample_rate, 2048);
        let est = yin_pitch(&sig, &c).expect("ok").expect("voiced");
        assert!(est.f0.is_finite() && est.period.is_finite() && est.cmnd.is_finite());
    }

    #[test]
    fn frequency_in_range() {
        // Detected pitch must lie within [f_min, f_max].
        let c = cfg();
        for &freq in &[80.0_f32, 261.63, 440.0, 880.0, 1500.0] {
            let sig = sine(freq, c.sample_rate, 4096);
            if let Some(est) = yin_pitch(&sig, &c).expect("ok") {
                assert!(
                    est.f0 >= c.f_min - 1e-3 && est.f0 <= c.f_max + 1e-3,
                    "f0 {} out of [{}, {}]",
                    est.f0,
                    c.f_min,
                    c.f_max
                );
            }
        }
    }

    #[test]
    fn empty_signal_error() {
        let c = cfg();
        assert!(matches!(
            yin_pitch(&[], &c).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn bad_threshold_error() {
        let mut c = cfg();
        c.threshold = 1.5;
        let sig = sine(440.0, c.sample_rate, 2048);
        assert!(matches!(
            yin_pitch(&sig, &c).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn fmin_ge_fmax_error() {
        let mut c = cfg();
        c.f_min = 2000.0;
        c.f_max = 500.0;
        let sig = sine(440.0, c.sample_rate, 2048);
        assert!(matches!(
            yin_pitch(&sig, &c).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn deterministic() {
        let c = cfg();
        let sig = sine(300.0, c.sample_rate, 2048);
        let a = yin_pitch(&sig, &c).expect("ok");
        let b = yin_pitch(&sig, &c).expect("ok");
        assert_eq!(a, b);
    }
}
