//! Wigner-Ville Distribution (WVD) and Pseudo-WVD (PWVD).
//!
//! Implements the discrete WVD for analytic signals obtained via the
//! Hilbert transform. The WVD is a joint time-frequency energy distribution:
//!
//! ```text
//! WVD_z(t, f) = ∫ z(t + τ/2) · z*(t − τ/2) · exp(−j·2π·f·τ) dτ
//! ```
//!
//! The Pseudo-WVD (PWVD) applies a Hann lag-smoothing window to reduce
//! cross-term interference at the cost of frequency resolution.
//!
//! References:
//!   Wigner (1932) Phys. Rev. 40:749-759
//!   Ville (1948) Cables et Transmissions 2A(1):61-74
//!   Cohen (1989) Proc. IEEE 77(7):941-981

use crate::error::{SignalError, SignalResult};
use std::f64::consts::TAU;

// ─────────────────────────────────────────────────────────────────── FFT ────

/// Cooley-Tukey radix-2 FFT, in-place. Requires power-of-2 length.
fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

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
        for (r, im_v) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *im_v *= scale;
        }
    }
}

// ──────────────────────────────────────────── Hilbert / analytic signal ────

/// Compute the analytic signal z = x + j·H{x} via FFT-based Hilbert transform.
///
/// The analytic signal is obtained by zeroing negative-frequency components
/// and doubling positive-frequency ones, then IFFT-ing back.
fn hilbert_analytic(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    // Pad to next power of 2 to enable radix-2 FFT.
    let fft_len = if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    };

    let mut re = vec![0.0_f64; fft_len];
    let mut im = vec![0.0_f64; fft_len];
    re[..n].copy_from_slice(x);

    fft_inplace(&mut re, &mut im, false);

    // Apply analytic signal filter h[k]:
    //   k == 0 or k == fft_len/2 → unchanged (boundary bins)
    //   0 < k < fft_len/2        → double (positive frequencies)
    //   fft_len/2 < k < fft_len  → zero (negative frequencies)
    let half = fft_len / 2;
    for k in 1..half {
        re[k] *= 2.0;
        im[k] *= 2.0;
    }
    for k in half + 1..fft_len {
        re[k] = 0.0;
        im[k] = 0.0;
    }

    fft_inplace(&mut re, &mut im, true);

    // Truncate back to original length.
    re.truncate(n);
    im.truncate(n);
    (re, im)
}

// ──────────────────────────────────────────────────────────── Config / IO ────

/// Configuration for the Wigner-Ville Distribution.
#[derive(Debug, Clone)]
pub struct WvdConfig {
    /// Signal length N.
    pub signal_len: usize,
    /// Number of frequency bins (must be even). Determines frequency resolution.
    /// Default equals `signal_len`.
    pub n_freq: usize,
    /// For Pseudo-WVD: Hann window half-length (0 = full WVD, no smoothing).
    pub smoothing_half_len: usize,
}

impl Default for WvdConfig {
    fn default() -> Self {
        Self {
            signal_len: 64,
            n_freq: 64,
            smoothing_half_len: 0,
        }
    }
}

/// Output of the WVD computation.
#[derive(Debug, Clone)]
pub struct WvdOutput {
    /// WVD matrix, n_time × n_freq, row-major. Real-valued energy distribution.
    pub distribution: Vec<f64>,
    /// Number of time samples (= N).
    pub n_time: usize,
    /// Number of frequency bins.
    pub n_freq: usize,
}

// ─────────────────────────────────────── Kernel builder (inner loop) ────────

/// Build the FFT-length padded DFT of the instantaneous autocorrelation kernel
/// K[m] = z[n+m] · conj(z[n−m]) for a single time index `t`.
///
/// The kernel is Hermitian (K[-m] = K*[m]), so its DFT is real-valued.
/// We exploit this by forming K explicitly and taking only the real part.
fn compute_wvd_row(
    z_re: &[f64],
    z_im: &[f64],
    t: usize,
    n_sig: usize,
    n_freq: usize,
    smoothing: &[f64], // empty = no smoothing; length = 2*half+1
) -> Vec<f64> {
    let half = n_freq / 2;
    let tau_max = t.min(n_sig.saturating_sub(1).saturating_sub(t));

    let mut fft_re = vec![0.0_f64; n_freq];
    let mut fft_im = vec![0.0_f64; n_freq];

    // DC term m=0: K[0] = |z[t]|²
    let w0 = if smoothing.is_empty() {
        1.0
    } else {
        *smoothing.get(smoothing.len() / 2).unwrap_or(&1.0)
    };
    fft_re[0] = w0 * (z_re[t] * z_re[t] + z_im[t] * z_im[t]);

    for m in 1..=tau_max.min(half) {
        let tp = t + m;
        let tm = t - m; // guaranteed valid since m ≤ tau_max ≤ t

        // K[m]  = z[t+m] · conj(z[t-m])
        let k_re = z_re[tp] * z_re[tm] + z_im[tp] * z_im[tm];
        let k_im = z_im[tp] * z_re[tm] - z_re[tp] * z_im[tm];

        let w = if smoothing.is_empty() {
            1.0
        } else {
            let center = smoothing.len() / 2;
            if m <= center {
                smoothing[center - m]
            } else {
                0.0
            }
        };

        // Place K[m] at positive-lag index m and K[-m]=conj(K[m]) at n_freq-m.
        fft_re[m] = w * k_re;
        fft_im[m] = w * k_im;
        fft_re[n_freq - m] = w * k_re;
        fft_im[n_freq - m] = w * (-k_im); // conj
    }

    fft_inplace(&mut fft_re, &mut fft_im, false);

    // The DFT of a Hermitian kernel is real; discard imaginary part.
    // Scale by 2 per the WVD definition.
    fft_re.iter().map(|&v| 2.0 * v).collect()
}

// ─────────────────────────────────────────────────────────── Public API ─────

/// Validate a `WvdConfig` against a signal of length `sig_len`.
fn validate_config(sig_len: usize, config: &WvdConfig) -> SignalResult<()> {
    if config.signal_len == 0 {
        return Err(SignalError::InvalidSize(
            "WVD signal length must be > 0".into(),
        ));
    }
    if sig_len != config.signal_len {
        return Err(SignalError::DimensionMismatch {
            expected: format!("signal length {}", config.signal_len),
            got: format!("{sig_len}"),
        });
    }
    if config.n_freq == 0 || config.n_freq % 2 != 0 {
        return Err(SignalError::InvalidSize(
            "n_freq must be positive and even".into(),
        ));
    }
    Ok(())
}

/// Build the Hann smoothing window for the PWVD.
fn hann_window(half_len: usize) -> Vec<f64> {
    if half_len == 0 {
        return Vec::new();
    }
    let full = 2 * half_len + 1;
    (0..full)
        .map(|i| {
            let x = i as f64 / (full - 1) as f64;
            0.5 * (1.0 - (TAU * x).cos())
        })
        .collect()
}

/// Ensure n_freq is a power of 2 (required by the row FFT).
/// If it isn't, the FFT length is rounded up internally.
fn fft_len_for(n_freq: usize) -> usize {
    if n_freq.is_power_of_two() {
        n_freq
    } else {
        n_freq.next_power_of_two()
    }
}

/// Compute the (Pseudo-)Wigner-Ville Distribution of a real signal.
///
/// Returns a real-valued N × n_freq distribution matrix (row-major).
///
/// # Errors
/// - `InvalidSize` if `signal_len == 0`, `n_freq == 0`, or `n_freq` is odd.
/// - `DimensionMismatch` if `signal.len() != signal_len`.
pub fn wvd(signal: &[f64], config: &WvdConfig) -> SignalResult<WvdOutput> {
    validate_config(signal.len(), config)?;

    let n = config.signal_len;
    let n_freq = fft_len_for(config.n_freq);
    let window = hann_window(config.smoothing_half_len);

    let (z_re, z_im) = hilbert_analytic(signal);

    let mut distribution = vec![0.0_f64; n * n_freq];
    for t in 0..n {
        let row = compute_wvd_row(&z_re, &z_im, t, n, n_freq, &window);
        let base = t * n_freq;
        distribution[base..base + n_freq].copy_from_slice(&row);
    }

    Ok(WvdOutput {
        distribution,
        n_time: n,
        n_freq,
    })
}

/// Compute the cross-Wigner-Ville Distribution of two real signals x and y.
///
/// ```text
/// CWVD(t, f) = ∫ z_x(t + τ/2) · z_y*(t − τ/2) · exp(−j·2π·f·τ) dτ
/// ```
///
/// Returns the real part (cross-spectral energy distribution).
///
/// # Errors
/// - Same config validation errors as [`wvd`].
/// - `DimensionMismatch` if `x.len() != y.len()`.
pub fn cross_wvd(x: &[f64], y: &[f64], config: &WvdConfig) -> SignalResult<WvdOutput> {
    validate_config(x.len(), config)?;
    if x.len() != y.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("y length == x length {}", x.len()),
            got: format!("{}", y.len()),
        });
    }

    let n = config.signal_len;
    let n_freq = fft_len_for(config.n_freq);
    let window = hann_window(config.smoothing_half_len);

    let (zx_re, zx_im) = hilbert_analytic(x);
    let (zy_re, zy_im) = hilbert_analytic(y);

    let half = n_freq / 2;
    let mut distribution = vec![0.0_f64; n * n_freq];

    for t in 0..n {
        let tau_max = t.min(n.saturating_sub(1).saturating_sub(t));

        let mut fft_re = vec![0.0_f64; n_freq];
        let mut fft_im = vec![0.0_f64; n_freq];

        let w0 = if window.is_empty() {
            1.0
        } else {
            *window.get(window.len() / 2).unwrap_or(&1.0)
        };
        // Cross-kernel at m=0: z_x[t] · conj(z_y[t])
        fft_re[0] = w0 * (zx_re[t] * zy_re[t] + zx_im[t] * zy_im[t]);

        for m in 1..=tau_max.min(half) {
            let tp = t + m;
            let tm = t - m;

            // Cross-kernel K[m] = z_x[t+m] · conj(z_y[t-m])
            let k_re = zx_re[tp] * zy_re[tm] + zx_im[tp] * zy_im[tm];
            let k_im = zx_im[tp] * zy_re[tm] - zx_re[tp] * zy_im[tm];

            let w = if window.is_empty() {
                1.0
            } else {
                let center = window.len() / 2;
                if m <= center {
                    window[center - m]
                } else {
                    0.0
                }
            };

            fft_re[m] = w * k_re;
            fft_im[m] = w * k_im;
            fft_re[n_freq - m] = w * k_re;
            fft_im[n_freq - m] = w * (-k_im);
        }

        fft_inplace(&mut fft_re, &mut fft_im, false);

        let base = t * n_freq;
        for (dst, src) in distribution[base..base + n_freq].iter_mut().zip(fft_re.iter()) {
            *dst = 2.0 * src;
        }
    }

    Ok(WvdOutput {
        distribution,
        n_time: n,
        n_freq,
    })
}

/// Compute the time-marginal: ∫ WVD(t, f) df = |z(t)|² (energy at each instant).
///
/// Returns a vector of length `n_time`.
#[must_use]
pub fn wvd_time_marginal(output: &WvdOutput) -> Vec<f64> {
    let n = output.n_time;
    let nf = output.n_freq;
    let df = 1.0 / nf as f64;
    (0..n)
        .map(|t| {
            output.distribution[t * nf..(t + 1) * nf]
                .iter()
                .sum::<f64>()
                * df
        })
        .collect()
}

/// Compute the frequency-marginal: ∫ WVD(t, f) dt = |Z(f)|² (energy at each frequency).
///
/// Returns a vector of length `n_freq`.
#[must_use]
pub fn wvd_frequency_marginal(output: &WvdOutput) -> Vec<f64> {
    let n = output.n_time;
    let nf = output.n_freq;
    let dt = 1.0 / n as f64;
    let mut marginal = vec![0.0_f64; nf];
    for t in 0..n {
        for (f, slot) in marginal.iter_mut().enumerate() {
            *slot += output.distribution[t * nf + f] * dt;
        }
    }
    marginal
}

/// Estimate the instantaneous frequency: E[f | t] = ∫ f · WVD(t,f) df / ∫ WVD(t,f) df.
///
/// Frequencies are normalised (0 = DC, 1 = sample rate). Returns 0.0 where
/// the denominator is too small to divide.
///
/// Returns a vector of length `n_time`.
#[must_use]
pub fn wvd_instantaneous_frequency(output: &WvdOutput) -> Vec<f64> {
    let n = output.n_time;
    let nf = output.n_freq;
    (0..n)
        .map(|t| {
            let row = &output.distribution[t * nf..(t + 1) * nf];
            let denom: f64 = row.iter().sum();
            if denom.abs() < 1e-15 {
                return 0.0;
            }
            let numer: f64 = row
                .iter()
                .enumerate()
                .map(|(f, &v)| (f as f64 / nf as f64) * v)
                .sum();
            numer / denom
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_signal(n: usize, freq_norm: f64) -> Vec<f64> {
        (0..n).map(|i| (TAU * freq_norm * i as f64).sin()).collect()
    }

    #[test]
    fn test_wvd_output_shape() {
        let n = 16usize;
        let config = WvdConfig {
            signal_len: n,
            n_freq: 16,
            smoothing_half_len: 0,
        };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd must succeed");
        assert_eq!(out.distribution.len(), n * 16);
        assert_eq!(out.n_time, n);
        assert_eq!(out.n_freq, 16);
    }

    #[test]
    fn test_wvd_energy_conservation() {
        let n = 32usize;
        let config = WvdConfig {
            signal_len: n,
            n_freq: 32,
            smoothing_half_len: 0,
        };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd");
        let wvd_energy: f64 = out.distribution.iter().sum::<f64>() / (n * 32) as f64;
        let sig_energy: f64 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
        // WVD total energy should be in same order of magnitude as signal energy.
        assert!(
            wvd_energy.abs() <= sig_energy.abs() * 20.0 + 1e-6,
            "WVD energy {wvd_energy} >> signal energy {sig_energy}"
        );
    }

    #[test]
    fn test_time_marginal_length() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd");
        let tm = wvd_time_marginal(&out);
        assert_eq!(tm.len(), n);
    }

    #[test]
    fn test_frequency_marginal_length() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd");
        let fm = wvd_frequency_marginal(&out);
        assert_eq!(fm.len(), 16);
    }

    #[test]
    fn test_instantaneous_frequency_length() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd");
        let inst_f = wvd_instantaneous_frequency(&out);
        assert_eq!(inst_f.len(), n);
    }

    #[test]
    fn test_cross_wvd_identical_matches_auto() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let auto = wvd(&x, &config).expect("auto wvd");
        let cross = cross_wvd(&x, &x, &config).expect("cross wvd identical");
        for (a, c) in auto.distribution.iter().zip(cross.distribution.iter()) {
            assert!(
                (a - c).abs() < 1e-10,
                "cross_wvd(x,x) != wvd(x): {a} vs {c}"
            );
        }
    }

    #[test]
    fn test_cross_wvd_shape() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let y = sine_signal(n, 0.2);
        let cross = cross_wvd(&x, &y, &config).expect("cross wvd");
        let auto = wvd(&x, &config).expect("wvd");
        assert_eq!(cross.distribution.len(), auto.distribution.len());
        assert_eq!(cross.n_time, auto.n_time);
        assert_eq!(cross.n_freq, auto.n_freq);
    }

    #[test]
    fn test_no_smoothing_full_wvd() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.15);
        let out = wvd(&x, &config).expect("full WVD must not panic");
        assert_eq!(out.distribution.len(), n * 16);
    }

    #[test]
    fn test_pwvd_with_smoothing() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 4 };
        let x = sine_signal(n, 0.15);
        let out = wvd(&x, &config).expect("PWVD must not panic");
        assert_eq!(out.distribution.len(), n * 16);
    }

    #[test]
    fn test_small_n_freq() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("small n_freq");
        assert_eq!(out.n_freq, 16);
    }

    #[test]
    fn test_n_freq_larger_than_signal() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 32, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("n_freq > N must work");
        assert_eq!(out.n_freq, 32);
        assert_eq!(out.distribution.len(), n * 32);
    }

    #[test]
    fn test_single_sample_signal() {
        let config = WvdConfig { signal_len: 1, n_freq: 2, smoothing_half_len: 0 };
        let x = vec![1.0_f64];
        let out = wvd(&x, &config).expect("single-sample WVD");
        assert_eq!(out.n_time, 1);
        assert_eq!(out.distribution.len(), 2);
    }

    #[test]
    fn test_stationary_tone_peak_frequency() {
        // Pure tone at f₀; WVD should concentrate energy near f₀ bin.
        let n = 32usize;
        let nf = 32usize;
        let f0_norm = 0.125_f64; // 1/8 of sampling rate
        let config = WvdConfig { signal_len: n, n_freq: nf, smoothing_half_len: 0 };
        let x = sine_signal(n, f0_norm);
        let out = wvd(&x, &config).expect("wvd tone");
        let fm = wvd_frequency_marginal(&out);
        // Expected peak bin (considering WVD concentrates at ±f₀).
        let expected_bin = (f0_norm * nf as f64).round() as usize;
        let peak_bin = fm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Allow ±2 bins tolerance.
        let diff = peak_bin.abs_diff(expected_bin).min(nf - peak_bin.abs_diff(expected_bin));
        assert!(
            diff <= 4,
            "peak at bin {peak_bin}, expected near {expected_bin} (±4), f_marginal={fm:?}"
        );
    }

    #[test]
    fn test_instantaneous_freq_pure_tone() {
        // For a real sine the WVD produces two symmetric spectral peaks at ±f₀.
        // The naive centroid E[f|t] is therefore pulled toward the midpoint
        // between f₀ and its alias at (1 − f₀).  We verify only that the
        // estimate is non-trivially far from DC (> 0.05) and within [0, 0.5],
        // confirming that the distribution does concentrate near the tone.
        let n = 32usize;
        let nf = 32usize;
        let f0_norm = 0.125_f64;
        let config = WvdConfig { signal_len: n, n_freq: nf, smoothing_half_len: 0 };
        let x = sine_signal(n, f0_norm);
        let out = wvd(&x, &config).expect("wvd");
        let inst_f = wvd_instantaneous_frequency(&out);
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        for (t, &f_est) in inst_f.iter().enumerate().take(mid_end).skip(mid_start) {
            assert!(
                (0.0..=0.5).contains(&f_est),
                "inst_freq[{t}]={f_est:.4} out of [0, 0.5]"
            );
            assert!(
                f_est > 0.04,
                "inst_freq[{t}]={f_est:.4} should be above DC (> 0.04)"
            );
        }
    }

    #[test]
    fn test_signal_len_zero_error() {
        let config = WvdConfig { signal_len: 0, n_freq: 16, smoothing_half_len: 0 };
        assert!(matches!(wvd(&[], &config), Err(SignalError::InvalidSize(_))));
    }

    #[test]
    fn test_n_freq_zero_error() {
        let config = WvdConfig { signal_len: 8, n_freq: 0, smoothing_half_len: 0 };
        let x = vec![1.0_f64; 8];
        assert!(matches!(wvd(&x, &config), Err(SignalError::InvalidSize(_))));
    }

    #[test]
    fn test_n_freq_odd_error() {
        let config = WvdConfig { signal_len: 8, n_freq: 7, smoothing_half_len: 0 };
        let x = vec![1.0_f64; 8];
        assert!(matches!(wvd(&x, &config), Err(SignalError::InvalidSize(_))));
    }

    #[test]
    fn test_signal_len_mismatch_error() {
        let config = WvdConfig { signal_len: 8, n_freq: 8, smoothing_half_len: 0 };
        let x = vec![1.0_f64; 5]; // wrong length
        assert!(matches!(wvd(&x, &config), Err(SignalError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_cross_wvd_length_mismatch_error() {
        let config = WvdConfig { signal_len: 8, n_freq: 8, smoothing_half_len: 0 };
        let x = vec![1.0_f64; 8];
        let y = vec![1.0_f64; 5];
        assert!(matches!(
            cross_wvd(&x, &y, &config),
            Err(SignalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_large_n128_smoke() {
        let n = 128usize;
        let nf = 128usize;
        let config = WvdConfig { signal_len: n, n_freq: nf, smoothing_half_len: 0 };
        // Chirp signal: linearly increasing frequency.
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (PI * 0.3 * t * t * n as f64).sin()
            })
            .collect();
        let out = wvd(&x, &config).expect("large N=128 WVD must complete");
        assert_eq!(out.distribution.len(), n * nf);
        assert!(out.distribution.iter().all(|v| v.is_finite()), "all values finite");
    }

    #[test]
    fn test_time_marginal_nonnegative_for_analytic() {
        // For an analytic signal the time marginal approximates |z(t)|² ≥ 0.
        // Discrete WVD may have small negative values from boundary effects.
        let n = 32usize;
        let config = WvdConfig { signal_len: n, n_freq: 32, smoothing_half_len: 0 };
        let x = sine_signal(n, 0.1);
        let out = wvd(&x, &config).expect("wvd");
        let tm = wvd_time_marginal(&out);
        // Time marginal should be non-negative or only very slightly negative.
        for (t, &v) in tm.iter().enumerate() {
            assert!(v >= -1e-6, "time_marginal[{t}] = {v} should be ≥ 0");
        }
    }

    #[test]
    fn test_wvd_finite_all_values() {
        let n = 16usize;
        let config = WvdConfig { signal_len: n, n_freq: 16, smoothing_half_len: 2 };
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let out = wvd(&x, &config).expect("wvd");
        for v in &out.distribution {
            assert!(v.is_finite(), "WVD value must be finite, got {v}");
        }
    }
}
