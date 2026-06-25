//! FFT-based period (seasonality) detection via the Wiener–Khinchin theorem.
//!
//! The dominant period of a real time-series is recovered from the **lag of the
//! strongest peak of its autocorrelation function** (ACF). Computing the ACF
//! directly in the time domain costs `O(T²)` (this is the path used inside
//! [`crate::timesnet`]). The Wiener–Khinchin theorem gives an `O(T log T)`
//! route instead:
//!
//! ```text
//! ACF(x) = IDFT( |DFT(x)|² )
//! ```
//!
//! i.e. the autocorrelation is the inverse transform of the power spectrum. We
//! zero-pad the (mean-removed) series to a power-of-two length `N ≥ 2T` so the
//! *circular* correlation produced by the FFT equals the *linear* correlation,
//! forward-transform with a real FFT, square the magnitude bin-wise, then
//! inverse-transform. The transform itself is routed through the pure-CPU real
//! FFT of the sibling [`oxicuda_fft`] crate ([`oxicuda_fft::rfft`] /
//! [`oxicuda_fft::irfft`], radix-2 with a Bluestein fallback — no CUDA driver
//! is required to evaluate the math).
//!
//! The resulting linear autocorrelation is **bit-for-bit equivalent** (to f64
//! round-off, ≤ 1e-8) to the direct `O(T²)` time-domain sum
//! `r[τ] = Σ_t x[t]·x[t+τ]` — this equivalence is asserted in the unit tests and
//! is what proves the FFT routing is mathematically correct rather than merely
//! "runs without panicking".
//!
//! ## Period selection
//!
//! The dominant period is the **lag `τ > 0` of the tallest local maximum** of
//! the ACF. A local maximum (rather than the global argmax over lags) is
//! required because for smooth signals the ACF is largest at tiny lags — that
//! is high *correlation*, not *periodicity*. The descending shoulder leaving
//! `r[0]` carries no local maximum, so the first genuine peak sits at the true
//! fundamental period; the `(T−τ)/T` triangular taper of the unbiased estimator
//! makes the fundamental taller than its harmonics, so the strongest peak wins.
//!
//! The peak strength is reported as the normalised autocorrelation
//! `confidence = r[τ] / r[0] ∈ [−1, 1]`; a pure-noise or aperiodic series yields
//! only small peaks and falls below [`PeriodConfig::min_confidence`], so
//! [`detect_period_fft`] returns `None`.

use crate::error::{TsError, TsResult};
use oxicuda_fft::{FftError, irfft, rfft};

// ─── Public types ──────────────────────────────────────────────────────────

/// A ranked period-detection candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodCandidate {
    /// Candidate period (autocorrelation lag, in samples). Always `≥ 2`.
    pub period: usize,
    /// Normalised autocorrelation strength at that lag, `r[period] / r[0]`.
    ///
    /// Lies in `[−1, 1]`; larger means a stronger, more confident period.
    pub confidence: f64,
}

/// Tunable parameters for [`detect_period_fft_with`].
#[derive(Debug, Clone, Copy)]
pub struct PeriodConfig {
    /// Smallest lag considered a candidate period (default `2`).
    pub min_period: usize,
    /// Largest lag considered. `None` (default) means `T / 2`, beyond which the
    /// autocorrelation estimate has too few overlapping samples to be reliable.
    pub max_period: Option<usize>,
    /// Minimum normalised autocorrelation a peak must reach to be reported as a
    /// confident period (default `0.3`).
    pub min_confidence: f64,
}

impl Default for PeriodConfig {
    fn default() -> Self {
        Self {
            min_period: 2,
            max_period: None,
            min_confidence: 0.3,
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Detect the dominant period of `series` with the default [`PeriodConfig`].
///
/// Returns `Some(period)` when a peak clears [`PeriodConfig::min_confidence`],
/// otherwise `None` (too short, constant, or aperiodic / noise-dominated).
#[must_use]
pub fn detect_period_fft(series: &[f32]) -> Option<usize> {
    detect_period_fft_with(series, &PeriodConfig::default())
}

/// Detect the dominant period of `series` under an explicit [`PeriodConfig`].
///
/// Returns `Some(period)` only when the strongest autocorrelation peak in the
/// `[min_period, max_period]` band reaches `cfg.min_confidence`.
#[must_use]
pub fn detect_period_fft_with(series: &[f32], cfg: &PeriodConfig) -> Option<usize> {
    let candidates = ranked_candidates(series, cfg).ok()?;
    let best = candidates.into_iter().next()?;
    (best.confidence >= cfg.min_confidence).then_some(best.period)
}

/// Return up to `max_candidates` period candidates, strongest first.
///
/// Unlike [`detect_period_fft`] this does **not** apply a confidence threshold,
/// so callers can inspect [`PeriodCandidate::confidence`] themselves. An empty
/// vector is returned for degenerate input (empty, constant, or too short).
#[must_use]
pub fn detect_period_fft_ranked(series: &[f32], max_candidates: usize) -> Vec<PeriodCandidate> {
    let mut candidates = ranked_candidates(series, &PeriodConfig::default()).unwrap_or_default();
    candidates.truncate(max_candidates);
    candidates
}

/// Mean-removed linear autocorrelation `r[τ]` for `τ = 0..T`, computed in
/// `O(T log T)` via [`oxicuda_fft`].
///
/// `r[0]` is the (zero-lag) signal energy; the series mean is removed first so
/// the DC component does not swamp the periodic structure.
///
/// # Errors
///
/// [`TsError::EmptyInput`] for an empty series, [`TsError::NonFinite`] if any
/// sample is non-finite, or [`TsError::Internal`] if the underlying FFT fails.
pub fn autocorrelation_fft(series: &[f32]) -> TsResult<Vec<f64>> {
    let centred = mean_removed_f64(series)?;
    autocorr_via_fft(&centred)
}

// ─── Core: Wiener–Khinchin autocorrelation ──────────────────────────────────

/// Linear autocorrelation of `x` for lags `0..x.len()` via
/// `IDFT(|DFT(x)|²)`, zero-padded so the circular result equals the linear one.
///
/// The forward/inverse transforms are the pure-CPU real FFTs of `oxicuda-fft`.
fn autocorr_via_fft(x: &[f64]) -> TsResult<Vec<f64>> {
    let t = x.len();
    if t == 0 {
        return Err(TsError::EmptyInput {
            msg: "autocorrelation of empty series".into(),
        });
    }
    if t == 1 {
        return Ok(vec![x[0] * x[0]]);
    }

    // Pad to a power-of-two length ≥ 2T: this both removes the circular
    // wrap-around (lags 0..T see no aliasing) and guarantees the radix-2
    // O(N log N) path inside oxicuda-fft.
    let n = (2 * t).next_power_of_two();
    let mut padded = vec![0.0_f64; n];
    padded[..t].copy_from_slice(x);

    // X = rfft(padded): half spectrum [(re, im); N/2 + 1].
    let spectrum = rfft(&padded, n).map_err(fft_err)?;
    // Power spectrum S = |X|² is real and (when Hermitian-extended) even, so its
    // inverse transform is the real autocorrelation. Feed S as a real spectrum.
    let power: Vec<(f64, f64)> = spectrum
        .iter()
        .map(|&(re, im)| (re * re + im * im, 0.0))
        .collect();
    let mut acf = irfft(&power, n).map_err(fft_err)?;
    acf.truncate(t);
    Ok(acf)
}

/// Map an [`oxicuda_fft`] error into a [`TsError`].
fn fft_err(e: FftError) -> TsError {
    TsError::Internal(format!("oxicuda-fft: {e}"))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert to `f64` and subtract the mean. Errors on empty / non-finite input.
fn mean_removed_f64(series: &[f32]) -> TsResult<Vec<f64>> {
    if series.is_empty() {
        return Err(TsError::EmptyInput {
            msg: "period detection on empty series".into(),
        });
    }
    if series.iter().any(|v| !v.is_finite()) {
        return Err(TsError::NonFinite);
    }
    let t = series.len();
    let mean = series.iter().map(|&v| v as f64).sum::<f64>() / t as f64;
    Ok(series.iter().map(|&v| v as f64 - mean).collect())
}

/// Compute and rank the autocorrelation-peak candidates of `series`.
fn ranked_candidates(series: &[f32], cfg: &PeriodConfig) -> TsResult<Vec<PeriodCandidate>> {
    let centred = mean_removed_f64(series)?;
    let acf = autocorr_via_fft(&centred)?;
    Ok(peak_candidates(&acf, cfg))
}

/// Find every local-maximum lag of `acf` within the configured band and return
/// them as candidates sorted by descending confidence (`r[τ] / r[0]`).
fn peak_candidates(acf: &[f64], cfg: &PeriodConfig) -> Vec<PeriodCandidate> {
    let t = acf.len();
    let r0 = acf[0];
    // No variance (constant / zero series) or too short to host a lag-1 shoulder
    // plus an interior peak ⇒ no period.
    if r0 <= 0.0 || t < 4 {
        return Vec::new();
    }

    let min_p = cfg.min_period.max(2);
    // `max_p ≤ t − 2` keeps the `τ + 1` neighbour in range for the local-max test.
    let max_p = cfg.max_period.unwrap_or(t / 2).min(t - 2);
    if max_p < min_p {
        return Vec::new();
    }

    let inv_r0 = 1.0 / r0;
    let mut candidates: Vec<PeriodCandidate> = Vec::new();
    // Slide a width-3 window so neighbours are checked without manual indexing.
    // Window i covers acf[min_p-1 + i ..= min_p+1 + i]; its centre lag is min_p+i.
    for (i, w) in acf[(min_p - 1)..=(max_p + 1)].windows(3).enumerate() {
        let (left, centre, right) = (w[0], w[1], w[2]);
        // Strict rise on the left rejects the monotone shoulder leaving r[0];
        // `>=` on the right collapses flat-topped peaks to a single candidate.
        if centre > left && centre >= right {
            candidates.push(PeriodCandidate {
                period: min_p + i,
                confidence: centre * inv_r0,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Direct `O(T²)` time-domain autocorrelation oracle:
    /// `r[τ] = Σ_{t=0}^{T-1-τ} x[t]·x[t+τ]`.
    fn autocorr_direct(x: &[f64]) -> Vec<f64> {
        let t = x.len();
        (0..t)
            .map(|tau| (0..t - tau).map(|i| x[i] * x[i + tau]).sum())
            .collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // ── Integrity test 1: FFT autocorrelation == direct O(T²) ────────────────

    #[test]
    fn fft_autocorr_matches_direct_reference() {
        // Several lengths, including non-power-of-two and prime, all checked
        // ELEMENTWISE against the direct time-domain sum to ≤ 1e-8.
        let mut worst = 0.0_f64;
        for &t in &[8_usize, 16, 64, 100, 128, 173, 251, 256] {
            let x: Vec<f64> = (0..t)
                .map(|i| {
                    let i = i as f64;
                    (0.30 * i).sin() + 0.5 * (0.07 * i).cos() + 0.01 * i
                })
                .collect();
            let fft = autocorr_via_fft(&x).expect("fft autocorr");
            let direct = autocorr_direct(&x);
            assert_eq!(fft.len(), direct.len(), "length mismatch at T={t}");
            let d = max_abs_diff(&fft, &direct);
            assert!(d <= 1e-8, "T={t}: max|fft − direct| = {d:e} exceeds 1e-8");
            worst = worst.max(d);
        }
        // Surface the achieved precision (visible with --nocapture).
        eprintln!("fft-vs-direct autocorrelation max abs error = {worst:e}");
    }

    #[test]
    fn fft_autocorr_matches_direct_on_random() {
        // Deterministic pseudo-random series (fixed seed) — still exact.
        let mut rng = LcgRng::new(2026);
        let mut x32 = vec![0.0_f32; 200];
        rng.fill_normal(&mut x32);
        let x: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let fft = autocorr_via_fft(&x).expect("fft autocorr");
        let direct = autocorr_direct(&x);
        let d = max_abs_diff(&fft, &direct);
        assert!(d <= 1e-8, "random series: max abs error {d:e} exceeds 1e-8");
    }

    // ── Integrity test 2: known-period recovery ──────────────────────────────

    fn sine(period: usize, t: usize) -> Vec<f32> {
        (0..t)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / period as f32).sin())
            .collect()
    }

    #[test]
    fn recovers_sine_period_12() {
        let x = sine(12, 144);
        assert_eq!(detect_period_fft(&x), Some(12));
    }

    #[test]
    fn recovers_square_wave_period_7() {
        // Period-7 square wave: +1 for three samples, −1 for the next four.
        let t = 140;
        let x: Vec<f32> = (0..t).map(|i| if i % 7 < 3 { 1.0 } else { -1.0 }).collect();
        assert_eq!(detect_period_fft(&x), Some(7));
    }

    #[test]
    fn recovers_sawtooth_period_20() {
        let t = 200;
        let x: Vec<f32> = (0..t).map(|i| (i % 20) as f32).collect();
        assert_eq!(detect_period_fft(&x), Some(20));
    }

    #[test]
    fn multi_component_strongest_period_wins() {
        // A strong period-7 tone (amplitude 3) plus a weaker, non-harmonic
        // period-31 tone (amplitude 1): the stronger period-7 component must win
        // both the single-answer detection and the top of the ranked list.
        //
        // Note: a strong fundamental's own harmonics (14, 21, 28, …) outrank a
        // weaker component's fundamental in the ACF, so the weak period need not
        // appear in the leading candidates — that is the honest, measured
        // behaviour of autocorrelation-based detection.
        let t = 252;
        let x: Vec<f32> = (0..t)
            .map(|i| {
                let i = i as f32;
                let tau = 2.0 * std::f32::consts::PI;
                3.0 * (tau * i / 7.0).sin() + 1.0 * (tau * i / 31.0).sin()
            })
            .collect();
        assert_eq!(detect_period_fft(&x), Some(7));

        let ranked = detect_period_fft_ranked(&x, 8);
        assert_eq!(ranked.first().map(|c| c.period), Some(7));
        // The strong tone's peak is unambiguous, not a near-tie.
        assert!(ranked[0].confidence > 0.6, "weak top peak: {ranked:?}");
    }

    #[test]
    fn clean_sine_confidence_is_high() {
        let x = sine(16, 256);
        let ranked = detect_period_fft_ranked(&x, 1);
        let top = ranked.first().expect("a candidate");
        assert_eq!(top.period, 16);
        assert!(
            top.confidence > 0.5,
            "clean sine confidence unexpectedly low: {}",
            top.confidence
        );
    }

    // ── Integrity test 3: robustness ─────────────────────────────────────────

    #[test]
    fn pure_noise_returns_none() {
        // Deterministic white noise (fixed seed) is aperiodic → no confident
        // period.
        let mut rng = LcgRng::new(12345);
        let mut x = vec![0.0_f32; 400];
        rng.fill_normal(&mut x);
        assert_eq!(detect_period_fft(&x), None);

        // Even the strongest spurious peak stays well below a clean signal's.
        let ranked = detect_period_fft_ranked(&x, 1);
        if let Some(top) = ranked.first() {
            assert!(
                top.confidence < 0.3,
                "noise produced a suspiciously strong peak: {}",
                top.confidence
            );
        }
    }

    #[test]
    fn constant_series_is_handled() {
        let x = vec![5.0_f32; 100];
        // No variance ⇒ no period, and crucially no NaN/Inf anywhere.
        assert_eq!(detect_period_fft(&x), None);
        let acf = autocorrelation_fft(&x).expect("constant acf ok");
        assert!(
            acf.iter().all(|v| v.is_finite()),
            "constant series produced non-finite acf"
        );
        assert_eq!(acf[0], 0.0, "mean-removed constant must have zero energy");
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let mut rng_a = LcgRng::new(7);
        let mut a = vec![0.0_f32; 256];
        rng_a.fill_normal(&mut a);

        let mut rng_b = LcgRng::new(7);
        let mut b = vec![0.0_f32; 256];
        rng_b.fill_normal(&mut b);

        assert_eq!(detect_period_fft(&a), detect_period_fft(&b));
        assert_eq!(
            detect_period_fft_ranked(&a, 5),
            detect_period_fft_ranked(&b, 5)
        );
    }

    #[test]
    fn handles_non_power_of_two_length() {
        // T = 130 is not a power of two (pads to N = 512 internally); a period-13
        // sine must still be recovered exactly.
        let x = sine(13, 130);
        assert_eq!(x.len(), 130);
        assert!(!x.len().is_power_of_two());
        assert_eq!(detect_period_fft(&x), Some(13));

        // And a prime length T = 211 with a period-11 sine.
        let y = sine(11, 211);
        assert!(!y.len().is_power_of_two());
        assert_eq!(detect_period_fft(&y), Some(11));
    }

    #[test]
    fn empty_and_tiny_series_are_graceful() {
        assert_eq!(detect_period_fft(&[]), None);
        assert_eq!(detect_period_fft(&[1.0]), None);
        assert_eq!(detect_period_fft(&[1.0, 2.0, 3.0]), None);
        assert!(autocorrelation_fft(&[]).is_err());
        assert!(detect_period_fft_ranked(&[], 4).is_empty());
    }

    #[test]
    fn non_finite_input_errors() {
        assert!(autocorrelation_fft(&[1.0, f32::NAN, 2.0]).is_err());
        assert_eq!(detect_period_fft(&[1.0, f32::INFINITY, 2.0, 3.0]), None);
    }

    #[test]
    fn config_min_confidence_gates_detection() {
        let x = sine(10, 200);
        // A demanding threshold still admits a clean sine …
        let strict = PeriodConfig {
            min_confidence: 0.5,
            ..Default::default()
        };
        assert_eq!(detect_period_fft_with(&x, &strict), Some(10));
        // … but an impossible (> 1) threshold rejects everything.
        let impossible = PeriodConfig {
            min_confidence: 1.5,
            ..Default::default()
        };
        assert_eq!(detect_period_fft_with(&x, &impossible), None);
    }

    #[test]
    fn config_period_band_restricts_search() {
        // A pure period-10 sine: the ACF peaks at 10 and its harmonics 20, 30, …
        // Default detection finds the fundamental; raising `min_period` past it
        // makes the detector select the next harmonic peak (20) instead.
        let x = sine(10, 200);
        assert_eq!(detect_period_fft(&x), Some(10));
        let band = PeriodConfig {
            min_period: 15,
            ..Default::default()
        };
        assert_eq!(detect_period_fft_with(&x, &band), Some(20));
    }
}
