//! Power spectral density (PSD) estimation: periodogram, Welch, Bartlett, and
//! sine-taper multitaper methods.
//!
//! All estimators return a **one-sided** PSD over the frequency axis
//! `f ∈ [0, fs/2]`, with the non-DC / non-Nyquist bins doubled so that the
//! integral of the PSD over frequency recovers the signal power (Parseval).
//!
//! ## Methods
//!
//! - [`periodogram`] — single windowed FFT magnitude², the maximum-resolution
//!   (but high-variance) estimator.
//! - [`welch`] — average of windowed periodograms over overlapping segments,
//!   trading frequency resolution for reduced variance (Welch 1967).
//! - [`bartlett_psd`] — Welch with a rectangular window and **no** overlap
//!   (Bartlett 1948).
//! - [`multitaper_psd`] — average of eigenspectra computed with orthogonal
//!   **sine tapers** (Riedel & Sidorenko 1995), a closed-form low-bias
//!   multitaper estimator that avoids the DPSS eigenproblem.
//!
//! ## Scaling
//!
//! [`PsdScaling::Density`] yields a power spectral *density* (units V²/Hz):
//! the periodogram is divided by `fs · Σ w[n]²`.  [`PsdScaling::Spectrum`]
//! yields a power *spectrum* (units V²): division by `(Σ w[n])²`.  The Welch
//! `Density` scaling integrates (with the bin spacing `Δf = fs / nfft`) to the
//! signal power.

use std::f64::consts::TAU;

use crate::{
    audio::stft::make_window,
    error::{SignalError, SignalResult},
    types::WindowType,
};

/// Normalisation convention for the returned PSD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsdScaling {
    /// Power spectral density in V²/Hz: integrates to total power over `df`.
    Density,
    /// Power spectrum in V²: each bin is the power at that frequency.
    Spectrum,
}

// ─────────────────────────────────────────────────────────────────── FFT ────

/// Cooley-Tukey radix-2 FFT, in-place.  `re`/`im` must have power-of-2 length.
///
/// Mirrors the self-contained transform used elsewhere in the crate
/// (`transform::czt`) so PSD estimation has no external FFT dependency in the
/// CPU reference path.
fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(
        n.is_power_of_two(),
        "fft_inplace requires power-of-2 length"
    );

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages (forward transform: sign = −1).
    let mut len = 2usize;
    while len <= n {
        let ang = -TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let half = i + k + len / 2;
                let (vr, vi) = (cr * re[half] - ci * im[half], cr * im[half] + ci * re[half]);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[half] = ur - vr;
                im[half] = ui - vi;
                let tmp_r = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp_r;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Smallest power of two `≥ v` (with a floor of 1).
fn next_pow2(v: usize) -> usize {
    let mut p = 1usize;
    while p < v {
        p <<= 1;
    }
    p
}

// ───────────────────────────────────────────────── one-sided periodogram ────

/// Compute one windowed periodogram of a segment and accumulate `|FFT|²` into
/// `accum` (length `nfft/2 + 1`).
///
/// `seg` is the raw (un-windowed) segment of length `nperseg`; `win` is the
/// window of the same length.  The segment is zero-padded to `nfft` (a power
/// of two ≥ `nperseg`).  `scale` is the per-periodogram normalisation applied
/// to `|X[k]|²` before accumulation.
fn accumulate_periodogram(seg: &[f64], win: &[f64], nfft: usize, scale: f64, accum: &mut [f64]) {
    let mut re = vec![0.0_f64; nfft];
    let mut im = vec![0.0_f64; nfft];
    for (i, (&s, &w)) in seg.iter().zip(win.iter()).enumerate() {
        re[i] = s * w;
    }
    let _ = &mut im; // im starts at zero (real input)
    fft_inplace(&mut re, &mut im);

    let half = nfft / 2;
    for (k, slot) in accum.iter_mut().enumerate().take(half + 1) {
        let mag2 = re[k] * re[k] + im[k] * im[k];
        // One-sided doubling for all but DC and Nyquist.
        let factor = if k == 0 || k == half { 1.0 } else { 2.0 };
        *slot += factor * scale * mag2;
    }
}

/// Build the one-sided frequency axis `[0, fs/2]` for `nfft` bins.
fn one_sided_freqs(nfft: usize, fs: f64) -> Vec<f64> {
    let half = nfft / 2;
    (0..=half).map(|k| k as f64 * fs / nfft as f64).collect()
}

/// Per-periodogram scale factor for the requested PSD convention.
fn psd_scale(win: &[f64], fs: f64, scaling: PsdScaling) -> f64 {
    match scaling {
        PsdScaling::Density => {
            let sum_sq: f64 = win.iter().map(|w| w * w).sum();
            1.0 / (fs * sum_sq)
        }
        PsdScaling::Spectrum => {
            let sum: f64 = win.iter().sum();
            1.0 / (sum * sum)
        }
    }
}

// ───────────────────────────────────────────────────────── periodogram ──────

/// Single-segment one-sided periodogram PSD (density scaling).
///
/// Returns `(freqs, psd)` with `freqs.len() == psd.len() == nfft/2 + 1`, where
/// `nfft` is the next power of two ≥ `x.len()`.
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `x` is empty.
pub fn periodogram(x: &[f32], fs: f64, window: WindowType) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::InvalidSize(
            "periodogram input must be non-empty".to_owned(),
        ));
    }
    if fs <= 0.0 || fs.is_nan() {
        return Err(SignalError::InvalidParameter(format!(
            "sampling rate fs ({fs}) must be > 0"
        )));
    }
    let n = x.len();
    let nfft = next_pow2(n);
    let win = make_window(n, window);
    let scale = psd_scale(&win, fs, PsdScaling::Density);

    let seg: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut psd = vec![0.0_f64; nfft / 2 + 1];
    accumulate_periodogram(&seg, &win, nfft, scale, &mut psd);

    Ok((one_sided_freqs(nfft, fs), psd))
}

// ─────────────────────────────────────────────────────────────── Welch ──────

/// Welch's method: averaged periodogram of overlapping, windowed segments.
///
/// The signal is split into segments of length `nperseg` advancing by
/// `hop = nperseg − noverlap`.  Each segment has its **mean removed** (the
/// `scipy` `detrend="constant"` default), is windowed, FFT'd (zero-padded to
/// the next power of two ≥ `nperseg`), squared, and averaged.
///
/// Returns `(freqs, psd)` with `freqs.len() == psd.len() == nfft/2 + 1`.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `nperseg == 0`,
/// `nperseg > x.len()`, or `noverlap >= nperseg`.
pub fn welch(
    x: &[f32],
    fs: f64,
    nperseg: usize,
    noverlap: usize,
    window: WindowType,
    scaling: PsdScaling,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if fs <= 0.0 || fs.is_nan() {
        return Err(SignalError::InvalidParameter(format!(
            "sampling rate fs ({fs}) must be > 0"
        )));
    }
    if nperseg == 0 {
        return Err(SignalError::InvalidParameter(
            "welch nperseg must be ≥ 1".to_owned(),
        ));
    }
    if nperseg > x.len() {
        return Err(SignalError::InvalidParameter(format!(
            "welch nperseg ({nperseg}) must be ≤ signal length ({})",
            x.len()
        )));
    }
    if noverlap >= nperseg {
        return Err(SignalError::InvalidParameter(format!(
            "welch noverlap ({noverlap}) must be < nperseg ({nperseg})"
        )));
    }

    let hop = nperseg - noverlap;
    let nfft = next_pow2(nperseg);
    let win = make_window(nperseg, window);
    let scale = psd_scale(&win, fs, scaling);

    let mut psd = vec![0.0_f64; nfft / 2 + 1];
    let mut n_segs = 0usize;

    let mut start = 0usize;
    while start + nperseg <= x.len() {
        // Mean-removed segment.
        let mut seg: Vec<f64> = x[start..start + nperseg]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mean = seg.iter().sum::<f64>() / nperseg as f64;
        for s in seg.iter_mut() {
            *s -= mean;
        }
        accumulate_periodogram(&seg, &win, nfft, scale, &mut psd);
        n_segs += 1;
        start += hop;
    }

    if n_segs == 0 {
        return Err(SignalError::InvalidSize(
            "welch produced no segments".to_owned(),
        ));
    }
    let inv = 1.0 / n_segs as f64;
    for v in psd.iter_mut() {
        *v *= inv;
    }

    Ok((one_sided_freqs(nfft, fs), psd))
}

/// Bartlett's method: Welch with a rectangular window and **no** overlap.
///
/// # Errors
/// See [`welch`].
pub fn bartlett_psd(x: &[f32], fs: f64, nperseg: usize) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    welch(
        x,
        fs,
        nperseg,
        0,
        WindowType::Rectangular,
        PsdScaling::Density,
    )
}

// ──────────────────────────────────────────────────────────── multitaper ────

/// Multitaper PSD estimate using orthogonal **sine tapers**.
///
/// The `k`-th sine taper (Riedel & Sidorenko 1995) over `N` samples is
/// ```text
/// v_k[n] = sqrt(2 / (N + 1)) · sin(π (k+1)(n+1) / (N+1)),   n = 0..N-1.
/// ```
/// These tapers are mutually orthogonal and have well-localised spectral
/// concentration without requiring a DPSS eigensolve.  The estimate averages
/// the `n_tapers` eigenspectra `|FFT(v_k · x)|²`.
///
/// Returns `(freqs, psd)` (density scaling, one-sided).
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `n_tapers == 0` or `n_tapers`
/// exceeds the signal length; [`SignalError::InvalidSize`] if `x` is empty.
pub fn multitaper_psd(x: &[f32], fs: f64, n_tapers: usize) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::InvalidSize(
            "multitaper input must be non-empty".to_owned(),
        ));
    }
    if fs <= 0.0 || fs.is_nan() {
        return Err(SignalError::InvalidParameter(format!(
            "sampling rate fs ({fs}) must be > 0"
        )));
    }
    let n = x.len();
    if n_tapers == 0 || n_tapers > n {
        return Err(SignalError::InvalidParameter(format!(
            "multitaper n_tapers ({n_tapers}) must be in 1..={n}"
        )));
    }
    let nfft = next_pow2(n);
    let np1 = (n + 1) as f64;
    let amp = (2.0 / np1).sqrt();

    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut psd = vec![0.0_f64; nfft / 2 + 1];

    for k in 0..n_tapers {
        // Build the k-th sine taper.
        let taper: Vec<f64> = (0..n)
            .map(|nn| amp * (std::f64::consts::PI * (k + 1) as f64 * (nn + 1) as f64 / np1).sin())
            .collect();
        // Density scaling per taper: Σ v_k² ≈ 1 by construction, but compute it
        // exactly so the integral is calibrated regardless of N.
        let scale = psd_scale(&taper, fs, PsdScaling::Density);
        accumulate_periodogram(&xf, &taper, nfft, scale, &mut psd);
    }

    let inv = 1.0 / n_tapers as f64;
    for v in psd.iter_mut() {
        *v *= inv;
    }

    Ok((one_sided_freqs(nfft, fs), psd))
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// LCG white-noise generator matching the crate-wide convention.
    fn lcg_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Centre to ~zero-mean in [-1, 1].
            let u = (s >> 33) as f64 / (u32::MAX as f64);
            v.push((2.0 * u - 1.0) as f32);
        }
        v
    }

    fn integrate(freqs: &[f64], psd: &[f64]) -> f64 {
        // Rectangular-rule integration over the (uniform) frequency axis.
        let df = freqs[1] - freqs[0];
        psd.iter().sum::<f64>() * df
    }

    #[test]
    fn test_periodogram_parseval_sinusoid() {
        // A unit-amplitude cosine has power 0.5; the integral of its one-sided
        // density PSD should recover that within a few percent.
        let fs = 1000.0;
        let n = 1024;
        let f0 = 100.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) = periodogram(&x, fs, WindowType::Hann).expect("periodogram");
        let power = integrate(&freqs, &psd);
        assert!((power - 0.5).abs() < 0.05, "recovered power {power} != 0.5");
    }

    #[test]
    fn test_periodogram_peak_bin() {
        let fs = 1000.0;
        let n = 1024;
        let f0 = 125.0; // lands near a bin for nfft=1024
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) = periodogram(&x, fs, WindowType::Hann).expect("periodogram");
        let (peak_k, _) = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty psd");
        let peak_f = freqs[peak_k];
        assert!(
            (peak_f - f0).abs() < fs / n as f64 * 2.0,
            "peak at {peak_f}, want {f0}"
        );
    }

    #[test]
    fn test_welch_parseval_sinusoid() {
        let fs = 1000.0;
        let n = 4096;
        let f0 = 100.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) =
            welch(&x, fs, 512, 256, WindowType::Hann, PsdScaling::Density).expect("welch");
        let power = integrate(&freqs, &psd);
        assert!((power - 0.5).abs() < 0.1, "welch power {power} != 0.5");
    }

    #[test]
    fn test_welch_peak_frequency() {
        let fs = 2000.0;
        let n = 8192;
        let f0 = 300.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) =
            welch(&x, fs, 1024, 512, WindowType::Hann, PsdScaling::Density).expect("welch");
        let (peak_k, _) = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty");
        assert!(
            (freqs[peak_k] - f0).abs() < 5.0,
            "peak {} != {f0}",
            freqs[peak_k]
        );
    }

    #[test]
    fn test_welch_reduces_variance_vs_periodogram() {
        // Welch with many segments should have lower PSD variance than a single
        // periodogram on the same white noise.
        let fs = 1.0;
        let n = 8192;
        let x = lcg_noise(n, 0xC0FFEE);

        let (_, pg) = periodogram(&x, fs, WindowType::Rectangular).expect("pg");
        let (_, w) = welch(
            &x,
            fs,
            256,
            128,
            WindowType::Rectangular,
            PsdScaling::Density,
        )
        .expect("welch");

        // Compare coefficient of variation in the interior of the band.
        let cov = |p: &[f64]| -> f64 {
            let lo = p.len() / 8;
            let hi = p.len() - p.len() / 8;
            let slice = &p[lo..hi];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / slice.len() as f64;
            var.sqrt() / mean.max(1e-30)
        };
        let cov_pg = cov(&pg);
        let cov_w = cov(&w);
        assert!(
            cov_w < cov_pg,
            "Welch CoV ({cov_w}) should be < periodogram CoV ({cov_pg})"
        );
    }

    #[test]
    fn test_welch_white_noise_flatish() {
        // White-noise PSD should be roughly flat: the mean of the lower half of
        // the band and the mean of the upper half agree within a factor.
        let fs = 1.0;
        let n = 16384;
        let x = lcg_noise(n, 0x1234_5678);
        let (_, psd) =
            welch(&x, fs, 512, 256, WindowType::Hann, PsdScaling::Density).expect("welch");
        let mid = psd.len() / 2;
        let lower = &psd[1..mid];
        let upper = &psd[mid..psd.len() - 1];
        let m_lo = lower.iter().sum::<f64>() / lower.len() as f64;
        let m_hi = upper.iter().sum::<f64>() / upper.len() as f64;
        let ratio = m_lo.max(m_hi) / m_lo.min(m_hi).max(1e-30);
        assert!(ratio < 1.5, "white PSD not flat: lo={m_lo} hi={m_hi}");
    }

    #[test]
    fn test_bartlett_psd_power() {
        let fs = 1000.0;
        let n = 4096;
        let f0 = 80.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) = bartlett_psd(&x, fs, 512).expect("bartlett");
        let power = integrate(&freqs, &psd);
        assert!((power - 0.5).abs() < 0.15, "bartlett power {power} != 0.5");
    }

    #[test]
    fn test_multitaper_parseval() {
        let fs = 1000.0;
        let n = 2048;
        let f0 = 150.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) = multitaper_psd(&x, fs, 5).expect("multitaper");
        let power = integrate(&freqs, &psd);
        assert!((power - 0.5).abs() < 0.1, "multitaper power {power} != 0.5");
    }

    #[test]
    fn test_multitaper_peak() {
        let fs = 1000.0;
        let n = 2048;
        let f0 = 200.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (freqs, psd) = multitaper_psd(&x, fs, 4).expect("multitaper");
        let (peak_k, _) = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty");
        assert!(
            (freqs[peak_k] - f0).abs() < 5.0,
            "peak {} != {f0}",
            freqs[peak_k]
        );
    }

    #[test]
    fn test_onesided_sums_to_power() {
        // Direct check that one-sided density integrates to total signal power
        // for a two-tone signal.
        let fs = 800.0;
        let n = 2048;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                ((2.0 * PI * 100.0 * t).cos() + 0.5 * (2.0 * PI * 250.0 * t).cos()) as f32
            })
            .collect();
        // Total power of cos(a) + 0.5 cos(b) = 0.5 + 0.5*0.25 = 0.625.
        let (freqs, psd) = periodogram(&x, fs, WindowType::Hann).expect("periodogram");
        let power = integrate(&freqs, &psd);
        assert!(
            (power - 0.625).abs() < 0.06,
            "two-tone power {power} != 0.625"
        );
    }

    #[test]
    fn test_periodogram_empty_error() {
        assert!(periodogram(&[], 1.0, WindowType::Hann).is_err());
    }

    #[test]
    fn test_welch_nperseg_too_large_error() {
        let x = vec![0.0_f32; 10];
        assert!(welch(&x, 1.0, 20, 0, WindowType::Hann, PsdScaling::Density).is_err());
    }

    #[test]
    fn test_welch_overlap_too_large_error() {
        let x = vec![0.0_f32; 100];
        assert!(welch(&x, 1.0, 32, 32, WindowType::Hann, PsdScaling::Density).is_err());
        assert!(welch(&x, 1.0, 32, 40, WindowType::Hann, PsdScaling::Density).is_err());
    }

    #[test]
    fn test_multitaper_invalid_tapers_error() {
        let x = vec![0.0_f32; 16];
        assert!(multitaper_psd(&x, 1.0, 0).is_err());
        assert!(multitaper_psd(&x, 1.0, 100).is_err());
    }

    #[test]
    fn test_spectrum_scaling_peak_amplitude() {
        // Spectrum scaling: the peak of a unit cosine's power spectrum is ~0.25
        // (the two-sided lines of amplitude 0.5 are 0.25 each; one-sided doubles
        // the non-DC bin to ~0.5 total — split across leakage). Check the peak
        // is on the order of the analytic single-sided line power.
        let fs = 1000.0;
        let n = 1024;
        let f0 = 125.0;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).cos() as f32)
            .collect();
        let (_freqs, psd) =
            welch(&x, fs, 1024, 0, WindowType::FlatTop, PsdScaling::Spectrum).expect("welch");
        let peak = psd.iter().cloned().fold(0.0_f64, f64::max);
        // Flat-top has excellent amplitude flatness; the one-sided line power of
        // a unit cosine is 0.5, so the peak should approach 0.5.
        assert!((peak - 0.5).abs() < 0.1, "spectrum peak {peak} != ~0.5");
    }
}
