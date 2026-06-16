//! Continuous Wavelet Transform (CWT) via frequency-domain convolution.
//!
//! Implements the CWT using the Fourier-domain approach (O(N log N) per scale)
//! described in Torrence & Compo 1998 BAMS "A Practical Guide to Wavelet Analysis"
//! and Mallat 1999 "A Wavelet Tour of Signal Processing".
//!
//! Supported mother wavelets:
//! - **Morlet** (complex analytic) — good time-frequency localization
//! - **Mexican Hat** (real, second derivative of Gaussian) — good frequency resolution

use crate::error::{SignalError, SignalResult};
use std::f64::consts::{PI, TAU};

// ─────────────────────────────────────────────────────────────────── FFT ────

/// Cooley-Tukey radix-2 in-place FFT.  Both slices must have power-of-2 length.
fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

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

    // Butterfly stages.
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    cr * re[i + k + len / 2] - ci * im[i + k + len / 2],
                    cr * im[i + k + len / 2] + ci * re[i + k + len / 2],
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let tmp_r = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp_r;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for (r, v) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *v *= scale;
        }
    }
}

/// Return the smallest power of two that is ≥ `v` (and ≥ 1).
fn next_pow2(v: usize) -> usize {
    if v <= 1 {
        return 1;
    }
    if v.is_power_of_two() {
        v
    } else {
        v.next_power_of_two()
    }
}

// ─────────────────────────────────────────────────────────── Wavelet types ───

/// Mother wavelet choice for the CWT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CwtWavelet {
    /// Complex Morlet wavelet.  `omega0` is the non-dimensional frequency
    /// (default 6.0 for good time-frequency balance; must satisfy ω₀ > 5 for
    /// the admissibility condition to be approximately met).
    Morlet {
        /// Non-dimensional centre frequency ω₀.
        omega0: f64,
    },
    /// Real Mexican Hat (Ricker) wavelet — second derivative of Gaussian.
    MexicanHat,
}

// ──────────────────────────────────────────────────────────────── Config ─────

/// Configuration for the Continuous Wavelet Transform.
#[derive(Debug, Clone)]
pub struct CwtConfig {
    /// Mother wavelet.
    pub wavelet: CwtWavelet,
    /// Number of scales (voices) to compute.  Default: 32.
    pub n_scales: usize,
    /// Smallest scale in units of the sampling interval `dt`.  Default: 2.0.
    pub min_scale: f64,
    /// Fractional octave step between consecutive scales (sub-octave resolution).
    /// 0.25 means 4 voices per octave.  Default: 0.25.
    pub dj: f64,
    /// Sampling interval in physical time units.  Default: 1.0.
    pub dt: f64,
}

impl Default for CwtConfig {
    fn default() -> Self {
        Self {
            wavelet: CwtWavelet::Morlet { omega0: 6.0 },
            n_scales: 32,
            min_scale: 2.0,
            dj: 0.25,
            dt: 1.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────── Output ─────

/// Output of the CWT computation.
#[derive(Debug, Clone)]
pub struct CwtOutput {
    /// Real parts of the wavelet coefficients, row-major `[n_scales × n_time]`.
    pub coeffs_re: Vec<f64>,
    /// Imaginary parts of the wavelet coefficients, row-major `[n_scales × n_time]`.
    /// Always zero for the (real) Mexican Hat wavelet.
    pub coeffs_im: Vec<f64>,
    /// Scale values a_j = min_scale × 2^(j·dj), length `n_scales`.
    pub scales: Vec<f64>,
    /// Equivalent Fourier frequency for each scale, length `n_scales`.
    pub frequencies: Vec<f64>,
    /// Number of scales.
    pub n_scales: usize,
    /// Number of time samples (equals `signal.len()`).
    pub n_time: usize,
}

// ─────────────────────────────────────────────────────────────── Core CWT ────

/// Compute the Continuous Wavelet Transform of `signal`.
///
/// Uses the frequency-domain formulation: for each scale aⱼ the wavelet
/// coefficients are obtained via W(aⱼ, b) = √(dt/|aⱼ|) · IFFT(X̂(ω) · Ψ̂*(aⱼ·ω)).
///
/// The signal is zero-padded to the next power-of-two ≥ 2N before the FFT
/// to reduce circular-wrap artefacts.
///
/// # Errors
/// - [`SignalError::InvalidParameter`] if the signal is shorter than 4 samples,
///   `n_scales == 0`, `min_scale ≤ 0`, `dj ≤ 0`, or `dt ≤ 0`.
pub fn cwt(signal: &[f64], config: &CwtConfig) -> SignalResult<CwtOutput> {
    let n = signal.len();
    if n < 4 {
        return Err(SignalError::InvalidParameter(
            "signal too short for CWT (minimum 4 samples)".into(),
        ));
    }
    if config.n_scales == 0 {
        return Err(SignalError::InvalidParameter("n_scales must be > 0".into()));
    }
    if config.min_scale <= 0.0 {
        return Err(SignalError::InvalidParameter(
            "min_scale must be positive".into(),
        ));
    }
    if config.dj <= 0.0 {
        return Err(SignalError::InvalidParameter("dj must be positive".into()));
    }
    if config.dt <= 0.0 {
        return Err(SignalError::InvalidParameter("dt must be positive".into()));
    }

    // --- Build geometric scale sequence ---
    let scales: Vec<f64> = (0..config.n_scales)
        .map(|j| config.min_scale * 2f64.powf(j as f64 * config.dj))
        .collect();

    // --- Equivalent Fourier frequencies ---
    let frequencies: Vec<f64> = scales
        .iter()
        .map(|&s| fourier_frequency(config.wavelet, s, config.dt))
        .collect();

    // --- Pad signal to next power-of-two >= 2N ---
    let npad = next_pow2(2 * n);

    // FFT of zero-padded signal.
    let mut xr: Vec<f64> = signal.to_vec();
    xr.resize(npad, 0.0);
    let mut xi: Vec<f64> = vec![0.0; npad];
    fft_inplace(&mut xr, &mut xi, false);

    // Angular frequency array omega_k (rad / time-unit) for each FFT bin.
    // Bins 0..npad/2 are positive; bins npad/2+1..npad are negative.
    let omega: Vec<f64> = (0..npad)
        .map(|k| {
            if k <= npad / 2 {
                TAU * k as f64 / (npad as f64 * config.dt)
            } else {
                -TAU * (npad - k) as f64 / (npad as f64 * config.dt)
            }
        })
        .collect();

    let mut coeffs_re = vec![0.0_f64; config.n_scales * n];
    let mut coeffs_im = vec![0.0_f64; config.n_scales * n];

    for (j, &scale) in scales.iter().enumerate() {
        // Build conj(Psi_hat(scale * omega)) for all FFT bins.
        let mut psi_re = vec![0.0_f64; npad];
        let psi_im = vec![0.0_f64; npad];

        match config.wavelet {
            CwtWavelet::Morlet { omega0 } => {
                // Normalisation factor: (2*pi*scale/dt)^(1/2) * pi^(-1/4)
                let norm = (TAU * scale / config.dt).sqrt() * PI.powf(-0.25);
                for k in 0..npad {
                    // Analytic wavelet: only positive frequencies (Heaviside).
                    if omega[k] > 0.0 {
                        let w = scale * omega[k];
                        let gauss = (-0.5 * (w - omega0).powi(2)).exp();
                        psi_re[k] = norm * gauss;
                        // psi_im stays 0 — Morlet hat is real-valued for positive omega
                    }
                    // omega[k] <= 0 -> zero contribution
                }
            }
            CwtWavelet::MexicanHat => {
                // Psi_hat(omega) = -sqrt(2*pi) * omega^2 * exp(-omega^2/2),
                // normalised so ||psi|| = 1, absorbing (2*pi*scale/dt)^(1/2).
                let norm = (TAU * scale / config.dt).sqrt() * 2.0 * PI.powf(0.25) / 3f64.sqrt();
                for k in 0..npad {
                    let w = scale * omega[k];
                    psi_re[k] = -norm * w * w * (-0.5 * w * w).exp();
                    // psi_im[k] = 0.0 — real, even wavelet
                }
            }
        }

        // Product X_hat(omega) * conj(Psi_hat(scale*omega)).
        // Since psi_im == 0 in both cases (or Morlet is real on +ve side):
        // conj(Psi_hat) = psi_re - i*psi_im
        let mut gr: Vec<f64> = (0..npad)
            .map(|k| xr[k] * psi_re[k] + xi[k] * psi_im[k])
            .collect();
        let mut gi: Vec<f64> = (0..npad)
            .map(|k| xi[k] * psi_re[k] - xr[k] * psi_im[k])
            .collect();

        // Inverse FFT -> wavelet coefficients in time domain.
        fft_inplace(&mut gr, &mut gi, true);

        // Extract first n samples and apply sqrt(dt/scale) normalisation.
        let norm_factor = (config.dt / scale).sqrt();
        let row = j * n;
        for t in 0..n {
            coeffs_re[row + t] = gr[t] * norm_factor;
            coeffs_im[row + t] = gi[t] * norm_factor;
        }
    }

    Ok(CwtOutput {
        coeffs_re,
        coeffs_im,
        scales,
        frequencies,
        n_scales: config.n_scales,
        n_time: n,
    })
}

// ────────────────────────────────────────────────── Derived quantities ────────

/// Compute the scalogram |W(a,b)|² from CWT output.
///
/// Returns a flat `Vec<f64>` of length `n_scales × n_time` (row-major).
#[must_use]
pub fn cwt_scalogram(output: &CwtOutput) -> Vec<f64> {
    output
        .coeffs_re
        .iter()
        .zip(output.coeffs_im.iter())
        .map(|(&r, &i)| r * r + i * i)
        .collect()
}

/// Compute the global wavelet power spectrum (time-averaged scalogram).
///
/// Returns a `Vec<f64>` of length `n_scales`.  Each entry is the mean power
/// at that scale: `P(a) = (1/N) * sum_t |W(a,t)|^2`.
#[must_use]
pub fn cwt_global_power(output: &CwtOutput) -> Vec<f64> {
    let n = output.n_time;
    let inv_n = if n == 0 { 1.0 } else { 1.0 / n as f64 };
    (0..output.n_scales)
        .map(|j| {
            let row = j * n;
            (0..n)
                .map(|t| {
                    let r = output.coeffs_re[row + t];
                    let i = output.coeffs_im[row + t];
                    r * r + i * i
                })
                .sum::<f64>()
                * inv_n
        })
        .collect()
}

/// For each time step, return the scale index with maximum power.
///
/// Returns a `Vec<usize>` of length `n_time`.
#[must_use]
pub fn cwt_ridge(output: &CwtOutput) -> Vec<usize> {
    let n = output.n_time;
    let ns = output.n_scales;
    (0..n)
        .map(|t| {
            let mut best_j = 0usize;
            let mut best_power = f64::NEG_INFINITY;
            for j in 0..ns {
                let r = output.coeffs_re[j * n + t];
                let i = output.coeffs_im[j * n + t];
                let p = r * r + i * i;
                if p > best_power {
                    best_power = p;
                    best_j = j;
                }
            }
            best_j
        })
        .collect()
}

/// Compute the cone of influence (COI) — the e-folding boundary in scale
/// units for each time step.
///
/// Coefficients at scales larger than the COI value at that time step are
/// affected by edge effects and should be interpreted with caution.
///
/// Returns a `Vec<f64>` of length `n` giving the COI boundary scale
/// (in time units) at each time step.
pub fn cwt_cone_of_influence(wavelet: CwtWavelet, _scales: &[f64], n: usize, dt: f64) -> Vec<f64> {
    // e-folding factor per wavelet type (Torrence & Compo 1998 Table 1).
    let e_fold = match wavelet {
        CwtWavelet::Morlet { omega0 } => omega0 / 2f64.sqrt(),
        CwtWavelet::MexicanHat => 2f64.sqrt(),
    };
    // COI at time t: scale limit = distance_from_nearest_edge / e_fold
    (0..n)
        .map(|t| {
            let dist_left = t as f64 * dt;
            let dist_right = (n.saturating_sub(1).saturating_sub(t)) as f64 * dt;
            dist_left.min(dist_right) / e_fold
        })
        .collect()
}

// ────────────────────────────────────────── Equivalent Fourier frequency ─────

/// Convert a scale to its equivalent Fourier frequency.
fn fourier_frequency(wavelet: CwtWavelet, scale: f64, dt: f64) -> f64 {
    match wavelet {
        CwtWavelet::Morlet { omega0 } => {
            // f = (omega0 + sqrt(2 + omega0^2)) / (4*pi*scale*dt)
            (omega0 + (2.0 + omega0 * omega0).sqrt()) / (4.0 * PI * scale * dt)
        }
        CwtWavelet::MexicanHat => {
            // f = sqrt(2.5) / (pi*scale*dt)
            2.5f64.sqrt() / (PI * scale * dt)
        }
    }
}

// ─────────────────────────────────────────────────────────────────── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    fn default_morlet() -> CwtConfig {
        CwtConfig::default()
    }

    fn make_sine(n: usize, freq_norm: f64) -> Vec<f64> {
        (0..n).map(|i| (TAU * freq_norm * i as f64).sin()).collect()
    }

    // 1. Morlet CWT on a zero signal -> all coefficients exactly zero
    // A zero input has a zero FFT spectrum, so all wavelet coefficients are zero.
    #[test]
    fn test_morlet_constant_signal_zero_coeffs() {
        let signal = vec![0.0f64; 64];
        let config = default_morlet();
        let out = cwt(&signal, &config).expect("cwt should succeed");
        let scalo = cwt_scalogram(&out);
        let max_power: f64 = scalo.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_power < 1e-20,
            "zero signal should have exactly zero CWT power, got {max_power}"
        );
    }

    // 2. Morlet CWT on pure sine: power peak at correct scale
    #[test]
    fn test_morlet_sine_power_peak_correct_scale() {
        let n = 128;
        let freq_norm = 0.1;
        let signal = make_sine(n, freq_norm);
        let config = CwtConfig {
            n_scales: 32,
            min_scale: 2.0,
            dj: 0.25,
            dt: 1.0,
            wavelet: CwtWavelet::Morlet { omega0: 6.0 },
        };
        let out = cwt(&signal, &config).expect("cwt should succeed");
        let global_power = cwt_global_power(&out);
        let peak_j = global_power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("value should be present");
        let freq_diff: Vec<f64> = out
            .frequencies
            .iter()
            .map(|&f| (f - freq_norm).abs())
            .collect();
        let expected_j = freq_diff
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("value should be present");
        assert!(
            (peak_j as isize - expected_j as isize).unsigned_abs() <= 4,
            "peak at scale index {peak_j}, expected near {expected_j}"
        );
    }

    // 3. MexHat CWT completes without error
    #[test]
    fn test_mexhat_cwt_no_error() {
        let signal = make_sine(64, 0.1);
        let config = CwtConfig {
            wavelet: CwtWavelet::MexicanHat,
            ..Default::default()
        };
        let result = cwt(&signal, &config);
        assert!(
            result.is_ok(),
            "MexHat CWT should succeed: {:?}",
            result.err()
        );
    }

    // 4. Output size: coeffs_re.len() == n_scales * n
    #[test]
    fn test_output_size_coeffs_re() {
        let n = 64;
        let config = default_morlet();
        let out = cwt(&vec![0.5f64; n], &config).expect("cwt should succeed");
        assert_eq!(out.coeffs_re.len(), config.n_scales * n);
        assert_eq!(out.coeffs_im.len(), config.n_scales * n);
        assert_eq!(out.n_time, n);
        assert_eq!(out.n_scales, config.n_scales);
    }

    // 5. scales.len() == n_scales
    #[test]
    fn test_scales_length() {
        let config = default_morlet();
        let out = cwt(&vec![0.0f64; 32], &config).expect("cwt should succeed");
        assert_eq!(out.scales.len(), config.n_scales);
    }

    // 6. frequencies.len() == n_scales
    #[test]
    fn test_frequencies_length() {
        let config = default_morlet();
        let out = cwt(&vec![0.0f64; 32], &config).expect("cwt should succeed");
        assert_eq!(out.frequencies.len(), config.n_scales);
    }

    // 7. scales are monotonically increasing
    #[test]
    fn test_scales_monotone_increasing() {
        let config = default_morlet();
        let out = cwt(&vec![0.0f64; 32], &config).expect("cwt should succeed");
        for i in 1..out.scales.len() {
            assert!(
                out.scales[i] > out.scales[i - 1],
                "scales not increasing at index {i}: {} vs {}",
                out.scales[i],
                out.scales[i - 1]
            );
        }
    }

    // 8. frequencies are monotonically decreasing
    #[test]
    fn test_frequencies_monotone_decreasing() {
        let config = default_morlet();
        let out = cwt(&vec![0.0f64; 32], &config).expect("cwt should succeed");
        for i in 1..out.frequencies.len() {
            assert!(
                out.frequencies[i] < out.frequencies[i - 1],
                "frequencies not decreasing at index {i}: {} vs {}",
                out.frequencies[i],
                out.frequencies[i - 1]
            );
        }
    }

    // 9. scalogram = |W|^2 >= 0 everywhere
    #[test]
    fn test_scalogram_non_negative() {
        let signal = make_sine(64, 0.1);
        let out = cwt(&signal, &default_morlet()).expect("value should be present");
        for (i, &v) in cwt_scalogram(&out).iter().enumerate() {
            assert!(v >= 0.0, "scalogram[{i}] = {v} < 0");
        }
    }

    // 10. global_power sums to positive value on non-zero signal
    #[test]
    fn test_global_power_positive() {
        let signal = make_sine(64, 0.1);
        let out = cwt(&signal, &default_morlet()).expect("value should be present");
        let total: f64 = cwt_global_power(&out).iter().sum();
        assert!(
            total > 0.0,
            "global power should be positive for sine, got {total}"
        );
    }

    // 11. cwt_ridge returns vec of len n
    #[test]
    fn test_ridge_length() {
        let n = 64;
        let signal = make_sine(n, 0.1);
        let out = cwt(&signal, &default_morlet()).expect("value should be present");
        assert_eq!(cwt_ridge(&out).len(), n);
    }

    // 12. cone_of_influence returns vec of len n
    #[test]
    fn test_coi_length() {
        let n = 64;
        let config = default_morlet();
        let out = cwt(&vec![1.0f64; n], &config).expect("cwt should succeed");
        let coi = cwt_cone_of_influence(config.wavelet, &out.scales, n, config.dt);
        assert_eq!(coi.len(), n);
    }

    // 13. MexHat: coeffs_im all zero (real wavelet)
    #[test]
    fn test_mexhat_imaginary_zero() {
        let signal = make_sine(64, 0.1);
        let config = CwtConfig {
            wavelet: CwtWavelet::MexicanHat,
            ..Default::default()
        };
        let out = cwt(&signal, &config).expect("cwt should succeed");
        for (i, &v) in out.coeffs_im.iter().enumerate() {
            assert!(v.abs() < 1e-12, "MexHat coeffs_im[{i}] = {v} (expected 0)");
        }
    }

    // 14. Morlet: power at sine frequency scale > 5x mean power
    #[test]
    fn test_morlet_frequency_localization() {
        let n = 256;
        let freq_norm = 0.05;
        let signal = make_sine(n, freq_norm);
        let config = CwtConfig {
            n_scales: 40,
            min_scale: 2.0,
            dj: 0.2,
            dt: 1.0,
            wavelet: CwtWavelet::Morlet { omega0: 6.0 },
        };
        let out = cwt(&signal, &config).expect("cwt should succeed");
        let global_power = cwt_global_power(&out);
        let peak_power: f64 = global_power.iter().cloned().fold(0.0_f64, f64::max);
        let mean_power: f64 = global_power.iter().sum::<f64>() / global_power.len() as f64;
        assert!(
            peak_power > 5.0 * mean_power,
            "peak {peak_power} should be > 5x mean {mean_power}"
        );
    }

    // 15. n_scales=1: single scale works
    #[test]
    fn test_single_scale() {
        let config = CwtConfig {
            n_scales: 1,
            ..Default::default()
        };
        let out = cwt(&vec![1.0f64; 32], &config).expect("cwt should succeed");
        assert_eq!(out.n_scales, 1);
        assert_eq!(out.scales.len(), 1);
        assert_eq!(out.frequencies.len(), 1);
    }

    // 16. min_scale=1.0: small scale works
    #[test]
    fn test_min_scale_one() {
        let config = CwtConfig {
            min_scale: 1.0,
            n_scales: 16,
            ..Default::default()
        };
        let result = cwt(&make_sine(64, 0.1), &config);
        assert!(
            result.is_ok(),
            "min_scale=1.0 should succeed: {:?}",
            result.err()
        );
    }

    // 17. dt=0.5: different sampling rate gives higher frequencies
    #[test]
    fn test_custom_dt() {
        let config_half = CwtConfig {
            dt: 0.5,
            ..Default::default()
        };
        let config_one = CwtConfig::default();
        let signal = make_sine(64, 0.1);
        let out_half = cwt(&signal, &config_half).expect("cwt should succeed");
        let out_one = cwt(&signal, &config_one).expect("cwt should succeed");
        // Smaller dt -> higher equivalent frequency
        assert!(
            out_half.frequencies[0] > out_one.frequencies[0] * 1.5,
            "dt=0.5 should give higher frequencies: {} vs {}",
            out_half.frequencies[0],
            out_one.frequencies[0]
        );
    }

    // 18. dj=0.5: larger step works
    #[test]
    fn test_larger_dj() {
        let config = CwtConfig {
            dj: 0.5,
            n_scales: 16,
            ..Default::default()
        };
        let result = cwt(&make_sine(64, 0.1), &config);
        assert!(result.is_ok(), "dj=0.5 should succeed: {:?}", result.err());
    }

    // 19. signal length=4 (minimum): works
    #[test]
    fn test_minimum_signal_length() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        let config = CwtConfig {
            n_scales: 2,
            min_scale: 1.5,
            ..Default::default()
        };
        let result = cwt(&signal, &config);
        assert!(result.is_ok(), "n=4 should succeed: {:?}", result.err());
    }

    // 20. n_scales=0 -> InvalidParameter
    #[test]
    fn test_error_n_scales_zero() {
        let config = CwtConfig {
            n_scales: 0,
            ..Default::default()
        };
        assert!(matches!(
            cwt(&vec![1.0f64; 32], &config),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    // 21. dt<=0 -> InvalidParameter
    #[test]
    fn test_error_dt_nonpositive() {
        let config_zero = CwtConfig {
            dt: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cwt(&vec![1.0f64; 32], &config_zero),
            Err(SignalError::InvalidParameter(_))
        ));
        let config_neg = CwtConfig {
            dt: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cwt(&vec![1.0f64; 32], &config_neg),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    // 22. min_scale<=0 -> InvalidParameter
    #[test]
    fn test_error_min_scale_nonpositive() {
        let config_zero = CwtConfig {
            min_scale: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cwt(&vec![1.0f64; 32], &config_zero),
            Err(SignalError::InvalidParameter(_))
        ));
        let config_neg = CwtConfig {
            min_scale: -2.0,
            ..Default::default()
        };
        assert!(matches!(
            cwt(&vec![1.0f64; 32], &config_neg),
            Err(SignalError::InvalidParameter(_))
        ));
    }
}
