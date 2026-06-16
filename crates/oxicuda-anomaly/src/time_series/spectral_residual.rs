//! Spectral-Residual anomaly detection for univariate time series (Ren et al. 2019).
//!
//! The Spectral-Residual (SR) method, used in Microsoft's time-series anomaly service,
//! treats anomalies as *salient* regions of the signal. Given a series `x ∈ ℝ^n`:
//!
//! 1. Compute the Fourier amplitude spectrum `A = |F x|` and phase `P = ∠F x`.
//! 2. Form the **log spectrum** `L = log A`.
//! 3. Smooth it with an averaging filter `h_q` of length `q`: `AL = h_q ∗ L`.
//! 4. The **spectral residual** is `R = L − AL`.
//! 5. Invert with the original phase to obtain the **saliency map**
//!    `S = | F⁻¹( exp(R + iP) ) |`.
//! 6. Smooth `S` and flag points whose saliency exceeds `τ · S̄_local`, where `S̄_local`
//!    is a trailing moving average; the score is `(S − S̄_local) / S̄_local`.
//!
//! Following the paper we optionally **extrapolate** the series by `m` estimated points
//! before the transform so the most recent observations are not penalised by edge
//! effects (the gradient-extension trick `x_{n} = x_{n-1} + ḡ` with `ḡ` the average
//! slope over the last `look_ahead` points).
//!
//! All transforms are a self-contained O(n²) DFT, so the module depends only on `std`.
//!
//! # References
//!
//! - H. Ren, B. Xu, Y. Wang, C. Yi, C. Huang, X. Kou, T. Xing, M. Yang, J. Tong & Q. Zhang
//!   (2019), "Time-Series Anomaly Detection Service at Microsoft", KDD.
//! - X. Hou & L. Zhang (2007), "Saliency Detection: A Spectral Residual Approach", CVPR.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the [`spectral_residual`] detector.
#[derive(Debug, Clone)]
pub struct SpectralResidualConfig {
    /// Length `q` of the averaging filter on the log spectrum (default `3`).
    pub series_window: usize,
    /// Length of the trailing moving-average window for the saliency threshold
    /// (default `21`).
    pub score_window: usize,
    /// Threshold multiplier `τ` on the local saliency average (default `3.0`).
    pub threshold: f32,
    /// Number of points to extrapolate before the transform (default `5`; `0` disables).
    pub extend_points: usize,
    /// Number of trailing points used to estimate the extrapolation slope (default `5`).
    pub look_ahead: usize,
}

impl Default for SpectralResidualConfig {
    fn default() -> Self {
        Self {
            series_window: 3,
            score_window: 21,
            threshold: 3.0,
            extend_points: 5,
            look_ahead: 5,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Output of the spectral-residual detector.
#[derive(Debug, Clone)]
pub struct SpectralResidualResult {
    /// Saliency map `S` aligned to the input series (`[n]`).
    pub saliency: Vec<f32>,
    /// Per-point anomaly score `(S − S̄_local) / S̄_local` (`[n]`).
    pub scores: Vec<f32>,
    /// Binary anomaly flags from thresholding the scores (`[n]`).
    pub anomalies: Vec<bool>,
}

// ─── Detector ─────────────────────────────────────────────────────────────────

/// Run spectral-residual anomaly detection on a univariate series `x`.
///
/// # Errors
/// * [`AnomalyError::EmptyInput`] if `x` is empty.
/// * [`AnomalyError::InsufficientSamples`] if `x.len() < 4` (the transform needs a few
///   points to be meaningful).
/// * [`AnomalyError::Internal`] for a zero window size.
pub fn spectral_residual(
    x: &[f32],
    config: &SpectralResidualConfig,
) -> AnomalyResult<SpectralResidualResult> {
    if x.is_empty() {
        return Err(AnomalyError::EmptyInput);
    }
    if x.len() < 4 {
        return Err(AnomalyError::InsufficientSamples {
            need: 4,
            got: x.len(),
        });
    }
    if config.series_window == 0 || config.score_window == 0 {
        return Err(AnomalyError::Internal {
            msg: "spectral_residual: window sizes must be > 0".into(),
        });
    }

    let n = x.len();

    // ── Optional extrapolation to mitigate edge effects on recent points. ──
    let extended = extend_series(x, config.extend_points, config.look_ahead);
    let n_ext = extended.len();

    // ── DFT (real input ⇒ interleaved complex output). ──
    let spec = dft_real(&extended);

    // Amplitude and phase.
    let mut amp = vec![0.0_f64; n_ext];
    let mut phase = vec![0.0_f64; n_ext];
    for k in 0..n_ext {
        let re = spec[2 * k];
        let im = spec[2 * k + 1];
        amp[k] = (re * re + im * im).sqrt();
        phase[k] = im.atan2(re);
    }

    // Log spectrum, with a floor to avoid log(0).
    let log_amp: Vec<f64> = amp.iter().map(|&a| a.max(1e-12).ln()).collect();

    // Smoothed log spectrum (averaging filter of length q).
    let avg_log = moving_average_f64(&log_amp, config.series_window);

    // Spectral residual R = L − AL.
    let residual: Vec<f64> = log_amp
        .iter()
        .zip(avg_log.iter())
        .map(|(l, al)| l - al)
        .collect();

    // Reconstruct: S = | F⁻¹( exp(R) · e^{iP} ) |.
    let mut recon = vec![0.0_f64; 2 * n_ext];
    for k in 0..n_ext {
        let mag = residual[k].exp();
        recon[2 * k] = mag * phase[k].cos();
        recon[2 * k + 1] = mag * phase[k].sin();
    }
    let inv = idft(&recon, n_ext);
    let mut saliency_ext = vec![0.0_f64; n_ext];
    for k in 0..n_ext {
        let re = inv[2 * k];
        let im = inv[2 * k + 1];
        saliency_ext[k] = (re * re + im * im).sqrt();
    }

    // Trim the extrapolated tail and convert to f32.
    let saliency: Vec<f32> = saliency_ext[..n].iter().map(|&v| v as f32).collect();

    // ── Scoring against a trailing local average. ──
    let mut scores = vec![0.0_f32; n];
    let w = config.score_window;
    for i in 0..n {
        let start = i.saturating_sub(w);
        let window = &saliency[start..i];
        let cnt = window.len();
        let sum: f32 = window.iter().sum();
        let local_avg = if cnt > 0 {
            sum / cnt as f32
        } else {
            saliency[i]
        };
        let denom = local_avg.max(1e-8);
        scores[i] = (saliency[i] - local_avg) / denom;
    }

    let anomalies: Vec<bool> = scores.iter().map(|&s| s > config.threshold).collect();

    Ok(SpectralResidualResult {
        saliency,
        scores,
        anomalies,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Extend the series by `m` points using the average trailing slope.
fn extend_series(x: &[f32], m: usize, look_ahead: usize) -> Vec<f64> {
    let n = x.len();
    let mut out: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    if m == 0 || n < 2 {
        return out;
    }
    let look = look_ahead.clamp(1, n - 1);
    // Average gradient over the last `look` steps.
    let mut grad = 0.0_f64;
    for k in 1..=look {
        grad += out[n - k] - out[n - k - 1];
    }
    grad /= look as f64;
    let last = out[n - 1];
    for j in 1..=m {
        out.push(last + grad * j as f64);
    }
    out
}

/// Symmetric moving average of length `q` (odd or even) over a real signal.
fn moving_average_f64(x: &[f64], q: usize) -> Vec<f64> {
    let n = x.len();
    let q = q.min(n).max(1);
    let half = q / 2;
    let mut out = vec![0.0_f64; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(n - 1);
        let mut sum = 0.0_f64;
        for v in x.iter().take(hi + 1).skip(lo) {
            sum += v;
        }
        *slot = sum / ((hi - lo + 1) as f64);
    }
    out
}

#[inline]
fn cmul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// DFT of a real signal; returns interleaved complex `[2n]`.
fn dft_real(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0_f64; 2 * n];
    let two_pi = std::f64::consts::TAU;
    for k in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            let angle = -two_pi * (j as f64) * (k as f64) / (n as f64);
            sr += xj * angle.cos();
            si += xj * angle.sin();
        }
        out[2 * k] = sr;
        out[2 * k + 1] = si;
    }
    out
}

/// Inverse DFT (`1/n`-scaled) of an interleaved complex signal `[2n]`.
fn idft(x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; 2 * n];
    let two_pi = std::f64::consts::TAU;
    let inv_n = 1.0 / n as f64;
    for k in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for j in 0..n {
            let angle = two_pi * (j as f64) * (k as f64) / (n as f64);
            let (wr, wi) = (angle.cos(), angle.sin());
            let (pr, pi) = cmul(x[2 * j], x[2 * j + 1], wr, wi);
            sr += pr;
            si += pi;
        }
        out[2 * k] = sr * inv_n;
        out[2 * k + 1] = si * inv_n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clean sinusoid with a single injected spike at `spike_idx`.
    fn sine_with_spike(n: usize, spike_idx: usize, spike: f32) -> Vec<f32> {
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        x[spike_idx] += spike;
        x
    }

    #[test]
    fn runs_on_simple_series() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        let res = spectral_residual(&x, &SpectralResidualConfig::default()).expect("ok");
        assert_eq!(res.saliency.len(), 64);
        assert_eq!(res.scores.len(), 64);
        assert_eq!(res.anomalies.len(), 64);
        assert!(res.saliency.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn detects_injected_spike() {
        let n = 100usize;
        let spike_idx = 60usize;
        let x = sine_with_spike(n, spike_idx, 8.0);
        let res = spectral_residual(&x, &SpectralResidualConfig::default()).expect("ok");
        // The spike location should have an above-average saliency score.
        let spike_score = res.scores[spike_idx];
        let mean_score: f32 = res.scores.iter().sum::<f32>() / n as f32;
        assert!(
            spike_score > mean_score,
            "spike score {spike_score} not above mean {mean_score}"
        );
    }

    #[test]
    fn spike_flagged_as_anomaly() {
        let n = 120usize;
        let spike_idx = 75usize;
        let x = sine_with_spike(n, spike_idx, 10.0);
        let cfg = SpectralResidualConfig {
            threshold: 2.0,
            ..Default::default()
        };
        let res = spectral_residual(&x, &cfg).expect("ok");
        // A neighbourhood around the spike should contain at least one flag.
        let flagged =
            (spike_idx.saturating_sub(2)..=(spike_idx + 2).min(n - 1)).any(|i| res.anomalies[i]);
        assert!(flagged, "spike not flagged near idx {spike_idx}");
    }

    #[test]
    fn clean_signal_few_anomalies() {
        // Pure sinusoid: SR should flag very few points at a high threshold.
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.25).sin()).collect();
        let cfg = SpectralResidualConfig {
            threshold: 5.0,
            ..Default::default()
        };
        let res = spectral_residual(&x, &cfg).expect("ok");
        let n_anom = res.anomalies.iter().filter(|&&b| b).count();
        assert!(n_anom <= 10, "too many anomalies on clean signal: {n_anom}");
    }

    #[test]
    fn higher_spike_higher_score() {
        let n = 100usize;
        let idx = 50usize;
        let small = spectral_residual(
            &sine_with_spike(n, idx, 3.0),
            &SpectralResidualConfig::default(),
        )
        .expect("ok");
        let large = spectral_residual(
            &sine_with_spike(n, idx, 12.0),
            &SpectralResidualConfig::default(),
        )
        .expect("ok");
        assert!(
            large.scores[idx] > small.scores[idx],
            "large {} should exceed small {}",
            large.scores[idx],
            small.scores[idx]
        );
    }

    #[test]
    fn extrapolation_changes_tail_handling() {
        // With and without extension the result is still valid and finite.
        let x: Vec<f32> = (0..80)
            .map(|i| (i as f32 * 0.2).sin() + 0.01 * i as f32)
            .collect();
        let with_ext = spectral_residual(
            &x,
            &SpectralResidualConfig {
                extend_points: 5,
                ..Default::default()
            },
        )
        .expect("ok");
        let no_ext = spectral_residual(
            &x,
            &SpectralResidualConfig {
                extend_points: 0,
                ..Default::default()
            },
        )
        .expect("ok");
        assert!(with_ext.saliency.iter().all(|s| s.is_finite()));
        assert!(no_ext.saliency.iter().all(|s| s.is_finite()));
        assert_eq!(with_ext.saliency.len(), no_ext.saliency.len());
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            spectral_residual(&[], &SpectralResidualConfig::default()),
            Err(AnomalyError::EmptyInput)
        ));
    }

    #[test]
    fn too_short_errors() {
        assert!(matches!(
            spectral_residual(&[1.0, 2.0, 3.0], &SpectralResidualConfig::default()),
            Err(AnomalyError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn zero_window_errors() {
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = SpectralResidualConfig {
            series_window: 0,
            ..Default::default()
        };
        assert!(matches!(
            spectral_residual(&x, &cfg),
            Err(AnomalyError::Internal { .. })
        ));
    }

    #[test]
    fn moving_average_constant_signal() {
        let x = vec![5.0_f64; 10];
        let avg = moving_average_f64(&x, 3);
        assert!(avg.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn dft_idft_round_trip() {
        let x = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 2.0, 0.0, 4.0];
        let back = idft(&dft_real(&x), x.len());
        for (k, &xk) in x.iter().enumerate() {
            assert!((back[2 * k] - xk).abs() < 1e-9, "k={k}");
            assert!(back[2 * k + 1].abs() < 1e-9, "imag k={k}");
        }
    }

    #[test]
    fn extend_series_uses_slope() {
        // A pure ramp ⇒ extension should continue the ramp.
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ext = extend_series(&x, 3, 5);
        assert_eq!(ext.len(), 13);
        assert!((ext[10] - 10.0).abs() < 1e-6, "{}", ext[10]);
        assert!((ext[11] - 11.0).abs() < 1e-6, "{}", ext[11]);
        assert!((ext[12] - 12.0).abs() < 1e-6, "{}", ext[12]);
    }
}
