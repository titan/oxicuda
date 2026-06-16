//! Chirp Z-Transform (CZT) via Bluestein's algorithm.
//!
//! Evaluates the Z-transform on a spiral contour z_k = A · W^{-k},
//! k = 0..M-1, using a length-L convolution so that non-power-of-2
//! input/output lengths are handled at the cost of one extra FFT size
//! roundup.
//!
//! References:
//!   Rabiner, Schafer & Rader (1969) IEEE Trans. Audio Electroacoust. 17(2):86–92
//!   Bluestein (1970) IEEE Trans. Audio Electroacoust. 18(4):451–455

use crate::error::{SignalError, SignalResult};
use std::f64::consts::TAU;

// ─────────────────────────────────────────────────────────────────── FFT ────

/// Cooley-Tukey radix-2 FFT, in-place. `re` and `im` must have power-of-2 length.
fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two(), "fft_inplace requires power-of-2 length");

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
        for (r, im_v) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *im_v *= scale;
        }
    }
}

// ──────────────────────────────────────────────────────────────── Config ────

/// Configuration for the Chirp Z-Transform.
#[derive(Debug, Clone)]
pub struct CztConfig {
    /// Input signal length N.
    pub input_len: usize,
    /// Number of output frequency samples M (can differ from N).
    pub output_len: usize,
    /// Starting point magnitude |A| (usually 1.0 for unit circle).
    pub a_mag: f64,
    /// Starting point angle θ_A in radians (start frequency = θ_A / (2π)).
    pub a_angle: f64,
    /// Step ratio magnitude |W| (usually 1.0 to stay on unit circle).
    pub w_mag: f64,
    /// Step angle ψ_W in radians (frequency step = ψ_W / (2π)).
    pub w_angle: f64,
}

impl Default for CztConfig {
    fn default() -> Self {
        Self {
            input_len: 256,
            output_len: 256,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / 256.0,
        }
    }
}

/// Output of the Chirp Z-Transform.
#[derive(Debug, Clone)]
pub struct CztOutput {
    /// Real parts of X[k], length = output_len.
    pub re: Vec<f64>,
    /// Imaginary parts of X[k], length = output_len.
    pub im: Vec<f64>,
    /// Number of output samples M.
    pub output_len: usize,
}

// ───────────────────────────────────────────────── helper: next power of 2 ──

#[must_use]
fn next_pow2(v: usize) -> usize {
    if v.is_power_of_two() {
        v
    } else {
        v.next_power_of_two()
    }
}

// ──────────────────────────────────────────────────────── Core algorithm ────

/// Compute the Chirp Z-Transform of a complex signal.
///
/// # Errors
/// - `InvalidSize` if `input_len == 0` or `output_len == 0`.
/// - `DimensionMismatch` if slice lengths disagree with config.
/// - `InvalidParameter` if `a_mag <= 0` or `w_mag <= 0`.
pub fn czt(input_re: &[f64], input_im: &[f64], config: &CztConfig) -> SignalResult<CztOutput> {
    let n = config.input_len;
    let m = config.output_len;

    if n == 0 {
        return Err(SignalError::InvalidSize(
            "CZT input length must be > 0".into(),
        ));
    }
    if m == 0 {
        return Err(SignalError::InvalidSize(
            "CZT output length must be > 0".into(),
        ));
    }
    if input_re.len() != n {
        return Err(SignalError::DimensionMismatch {
            expected: format!("input_re length {n}"),
            got: format!("{}", input_re.len()),
        });
    }
    if input_im.len() != n {
        return Err(SignalError::DimensionMismatch {
            expected: format!("input_im length {n}"),
            got: format!("{}", input_im.len()),
        });
    }
    if config.a_mag <= 0.0 {
        return Err(SignalError::InvalidParameter("a_mag must be > 0".into()));
    }
    if config.w_mag <= 0.0 {
        return Err(SignalError::InvalidParameter("w_mag must be > 0".into()));
    }

    let l = next_pow2(n + m - 1);
    let psi = config.w_angle;
    let w_mag = config.w_mag;
    let a_mag = config.a_mag;
    let theta_a = config.a_angle;

    // Step 1: build modified input g[n] = x[n] · A^{-n} · W^{n²/2}
    let mut g_re = vec![0.0_f64; l];
    let mut g_im = vec![0.0_f64; l];
    for idx in 0..n {
        let n2h = (idx * idx) as f64 / 2.0; // n²/2
        let angle = -(idx as f64) * theta_a + n2h * psi;
        let mag = a_mag.powi(-(idx as i32)) * w_mag.powf(n2h);
        let (s, c) = angle.sin_cos();
        let phi_re = mag * c;
        let phi_im = mag * s;
        g_re[idx] = input_re[idx] * phi_re - input_im[idx] * phi_im;
        g_im[idx] = input_re[idx] * phi_im + input_im[idx] * phi_re;
    }
    // g[n..l] is already zero.

    // Step 2: build chirp filter h.
    // h[0] = 1, h[n] = W^{-n²/2} for n=1..M-1, h[L-n] = h[n] (conjugate symmetry).
    let mut h_re = vec![0.0_f64; l];
    let mut h_im = vec![0.0_f64; l];
    h_re[0] = 1.0;
    for idx in 1..m {
        let n2h = (idx * idx) as f64 / 2.0;
        let angle = -n2h * psi;
        let mag = w_mag.powf(-n2h);
        let (s, c) = angle.sin_cos();
        h_re[idx] = mag * c;
        h_im[idx] = mag * s;
        // Wrap negative lag via circular convolution: h[-n] = conj(h[n]) for
        // Hermitian kernel, but here h[-n] defined as same as h[n] because the
        // kernel satisfies h[-m] = h[m] (the chirp phase is even in lag index).
        h_re[l - idx] = h_re[idx];
        h_im[l - idx] = h_im[idx];
    }

    // Step 3: FFT of g and h.
    fft_inplace(&mut g_re, &mut g_im, false);
    fft_inplace(&mut h_re, &mut h_im, false);

    // Step 4: pointwise multiply G * H.
    let mut y_re = vec![0.0_f64; l];
    let mut y_im = vec![0.0_f64; l];
    for idx in 0..l {
        y_re[idx] = g_re[idx] * h_re[idx] - g_im[idx] * h_im[idx];
        y_im[idx] = g_re[idx] * h_im[idx] + g_im[idx] * h_re[idx];
    }

    // Step 5: IFFT → convolution result.
    fft_inplace(&mut y_re, &mut y_im, true);

    // Step 6: multiply output by chirp taper W^{k²/2} and trim to M samples.
    let mut out_re = vec![0.0_f64; m];
    let mut out_im = vec![0.0_f64; m];
    for k in 0..m {
        let k2h = (k * k) as f64 / 2.0;
        let angle = k2h * psi;
        let mag = w_mag.powf(k2h);
        let (s, c) = angle.sin_cos();
        out_re[k] = mag * (c * y_re[k] - s * y_im[k]);
        out_im[k] = mag * (c * y_im[k] + s * y_re[k]);
    }

    Ok(CztOutput {
        re: out_re,
        im: out_im,
        output_len: m,
    })
}

/// Compute the CZT of a real signal (imaginary input treated as zero).
///
/// # Errors
/// Propagates errors from [`czt`].
pub fn czt_real(signal: &[f64], config: &CztConfig) -> SignalResult<CztOutput> {
    let im = vec![0.0_f64; signal.len()];
    czt(signal, &im, config)
}

/// Zoom FFT: compute DFT on the normalised frequency band `[f_lo, f_hi]`
/// with `m_points` output bins.
///
/// Frequencies are normalised so that 0 = DC and 0.5 = Nyquist (sample/2).
///
/// # Errors
/// - `InvalidParameter` if `f_lo >= f_hi` or `f_hi > 0.5`.
/// - `InvalidSize` if `m_points == 0`.
pub fn zoom_fft(
    signal: &[f64],
    f_lo: f64,
    f_hi: f64,
    m_points: usize,
) -> SignalResult<CztOutput> {
    if f_lo >= f_hi {
        return Err(SignalError::InvalidParameter(
            "f_lo must be < f_hi".into(),
        ));
    }
    if f_hi > 0.5 {
        return Err(SignalError::InvalidParameter(
            "f_hi must be <= 0.5 (Nyquist)".into(),
        ));
    }
    if m_points == 0 {
        return Err(SignalError::InvalidSize("m_points must be > 0".into()));
    }

    let n = signal.len();
    // A = exp(j·2π·f_lo): start point on the unit circle at f_lo.
    let a_angle = TAU * f_lo;
    // W = exp(-j·2π·(f_hi - f_lo)/M): frequency step.
    let w_angle = -TAU * (f_hi - f_lo) / m_points as f64;

    let config = CztConfig {
        input_len: n,
        output_len: m_points,
        a_mag: 1.0,
        a_angle,
        w_mag: 1.0,
        w_angle,
    };
    czt_real(signal, &config)
}

/// Compute the standard DFT via CZT (useful for equivalence verification).
///
/// Equivalent to the N-point DFT: A=1, W=exp(-j·2π/N), M=N.
///
/// # Errors
/// Propagates errors from [`czt_real`]. Returns `InvalidSize` if `signal` is empty.
pub fn dft_via_czt(signal: &[f64]) -> SignalResult<CztOutput> {
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::InvalidSize(
            "CZT input length must be > 0".into(),
        ));
    }
    let config = CztConfig {
        input_len: n,
        output_len: n,
        a_mag: 1.0,
        a_angle: 0.0,
        w_mag: 1.0,
        w_angle: -TAU / n as f64,
    };
    czt_real(signal, &config)
}

/// Compute the magnitude spectrum from a `CztOutput`.
#[must_use]
pub fn czt_magnitude(output: &CztOutput) -> Vec<f64> {
    output
        .re
        .iter()
        .zip(output.im.iter())
        .map(|(&r, &i)| (r * r + i * i).sqrt())
        .collect()
}

/// Compute the power spectrum from a `CztOutput`.
#[must_use]
pub fn czt_power(output: &CztOutput) -> Vec<f64> {
    output
        .re
        .iter()
        .zip(output.im.iter())
        .map(|(&r, &i)| r * r + i * i)
        .collect()
}

// ─────────────────────────────────────────────────────────────────── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Naive O(N²) DFT for ground-truth reference.
    fn naive_dft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut re = vec![0.0_f64; n];
        let mut im = vec![0.0_f64; n];
        for k in 0..n {
            for (nn, &xn) in x.iter().enumerate() {
                let angle = -TAU * k as f64 * nn as f64 / n as f64;
                re[k] += xn * angle.cos();
                im[k] += xn * angle.sin();
            }
        }
        (re, im)
    }

    #[test]
    fn test_dft_equiv_n16_sine() {
        let n = 16usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let czt_out = dft_via_czt(&x).expect("dft_via_czt must succeed");
        let (ref_re, ref_im) = naive_dft(&x);
        for k in 0..n {
            assert!(
                (czt_out.re[k] - ref_re[k]).abs() < 1e-8,
                "re[{k}]: {} vs {}",
                czt_out.re[k],
                ref_re[k]
            );
            assert!(
                (czt_out.im[k] - ref_im[k]).abs() < 1e-8,
                "im[{k}]: {} vs {}",
                czt_out.im[k],
                ref_im[k]
            );
        }
    }

    #[test]
    fn test_dft_equiv_n32() {
        let n = 32usize;
        // deterministic pseudo-random-looking signal via closed-form
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 3.0 * i as f64 / n as f64).sin()
                    + 0.5 * (2.0 * PI * 7.0 * i as f64 / n as f64).cos()
            })
            .collect();
        let czt_out = dft_via_czt(&x).expect("dft_via_czt must succeed for n=32");
        let (ref_re, ref_im) = naive_dft(&x);
        for k in 0..n {
            assert!(
                (czt_out.re[k] - ref_re[k]).abs() < 1e-8,
                "re[{k}] mismatch"
            );
            assert!(
                (czt_out.im[k] - ref_im[k]).abs() < 1e-8,
                "im[{k}] mismatch"
            );
        }
    }

    #[test]
    fn test_output_length_matches_config() {
        let signal = vec![1.0_f64; 8];
        let config = CztConfig {
            input_len: 8,
            output_len: 5,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / 8.0,
        };
        let out = czt_real(&signal, &config).expect("czt_real must succeed");
        assert_eq!(out.output_len, 5);
        assert_eq!(out.re.len(), 5);
        assert_eq!(out.im.len(), 5);
    }

    #[test]
    fn test_parseval() {
        let n = 16usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let out = dft_via_czt(&x).expect("dft_via_czt must succeed");
        let power = czt_power(&out);
        let spectral_energy: f64 = power.iter().sum::<f64>() / n as f64;
        let time_energy: f64 = x.iter().map(|v| v * v).sum::<f64>();
        assert!(
            (spectral_energy - time_energy).abs() < 1e-6 * (time_energy.abs() + 1.0),
            "Parseval: spectral={spectral_energy}, time={time_energy}"
        );
    }

    #[test]
    fn test_linearity() {
        let n = 16usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let a = 2.3_f64;
        let b = -1.1_f64;
        let combined: Vec<f64> = x.iter().zip(y.iter()).map(|(&xi, &yi)| a * xi + b * yi).collect();

        let cx = dft_via_czt(&x).expect("czt x");
        let cy = dft_via_czt(&y).expect("czt y");
        let cc = dft_via_czt(&combined).expect("czt combined");

        for k in 0..n {
            let expected_re = a * cx.re[k] + b * cy.re[k];
            let expected_im = a * cx.im[k] + b * cy.im[k];
            assert!(
                (cc.re[k] - expected_re).abs() < 1e-10,
                "linearity re[{k}]"
            );
            assert!(
                (cc.im[k] - expected_im).abs() < 1e-10,
                "linearity im[{k}]"
            );
        }
    }

    #[test]
    fn test_single_tone_peak() {
        let n = 32usize;
        let freq_bin = 4usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * freq_bin as f64 * i as f64 / n as f64).sin())
            .collect();
        let out = dft_via_czt(&x).expect("dft_via_czt");
        let mag = czt_magnitude(&out);
        let peak = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // For a pure sine at bin 4, energy appears at bins 4 and N-4.
        assert!(
            peak == freq_bin || peak == n - freq_bin,
            "peak at {peak}, expected {freq_bin} or {}",
            n - freq_bin
        );
    }

    #[test]
    fn test_zoom_fft_narrows_bandwidth() {
        let n = 64usize;
        // Tone at 0.1 (normalised)
        let f0 = 0.1_f64;
        let x: Vec<f64> = (0..n).map(|i| (TAU * f0 * i as f64).sin()).collect();
        let m = 32usize;
        let out = zoom_fft(&x, 0.05, 0.2, m).expect("zoom_fft");
        // Peak should be somewhere in output (narrow window around f0).
        let mag = czt_magnitude(&out);
        let peak_val = mag.iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak_val > 1e-3, "zoom_fft should resolve the tone, got peak={peak_val}");
    }

    #[test]
    fn test_zoom_fft_output_length() {
        let x = vec![1.0_f64; 32];
        let out = zoom_fft(&x, 0.0, 0.4, 20).expect("zoom_fft");
        assert_eq!(out.output_len, 20);
        assert_eq!(out.re.len(), 20);
    }

    #[test]
    fn test_czt_magnitude_non_negative() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).sin()).collect();
        let out = dft_via_czt(&x).expect("dft_via_czt");
        let mag = czt_magnitude(&out);
        for v in &mag {
            assert!(*v >= 0.0, "magnitude must be non-negative");
        }
    }

    #[test]
    fn test_czt_power_equals_magnitude_squared() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).cos()).collect();
        let out = dft_via_czt(&x).expect("dft_via_czt");
        let mag = czt_magnitude(&out);
        let pwr = czt_power(&out);
        for (m, p) in mag.iter().zip(pwr.iter()) {
            assert!((m * m - p).abs() < 1e-12, "power != magnitude²: {m}²={} vs {p}", m * m);
        }
    }

    #[test]
    fn test_zero_input_zero_output() {
        let x = vec![0.0_f64; 16];
        let out = dft_via_czt(&x).expect("dft_via_czt zero input");
        for &v in &out.re {
            assert!(v.abs() < 1e-14, "re should be ~0 for zero input");
        }
        for &v in &out.im {
            assert!(v.abs() < 1e-14, "im should be ~0 for zero input");
        }
    }

    #[test]
    fn test_dc_input_only_dc_bin() {
        let n = 16usize;
        let x = vec![1.0_f64; n];
        let out = dft_via_czt(&x).expect("dft_via_czt DC");
        // DC bin should be N, all others ~0.
        assert!(
            (out.re[0] - n as f64).abs() < 1e-8,
            "DC bin: {}",
            out.re[0]
        );
        for k in 1..n {
            assert!(
                out.re[k].abs() < 1e-8 && out.im[k].abs() < 1e-8,
                "non-DC bin [{k}] should be ~0"
            );
        }
    }

    #[test]
    fn test_real_matches_complex_zero_im() {
        let n = 16usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.4).sin()).collect();
        let config = CztConfig {
            input_len: n,
            output_len: n,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / n as f64,
        };
        let out_real = czt_real(&x, &config).expect("czt_real");
        let im_zeros = vec![0.0_f64; n];
        let out_complex = czt(&x, &im_zeros, &config).expect("czt");
        for k in 0..n {
            assert!((out_real.re[k] - out_complex.re[k]).abs() < 1e-14);
            assert!((out_real.im[k] - out_complex.im[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_n1_single_point() {
        let x = vec![3.7_f64];
        let config = CztConfig {
            input_len: 1,
            output_len: 1,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU,
        };
        let out = czt_real(&x, &config).expect("single-point CZT");
        assert!((out.re[0] - 3.7).abs() < 1e-12, "single-point: {}", out.re[0]);
        assert!(out.im[0].abs() < 1e-12);
    }

    #[test]
    fn test_n2_dft() {
        let x = vec![1.0_f64, -1.0];
        let out = dft_via_czt(&x).expect("n=2 DFT via CZT");
        // X[0] = sum = 0, X[1] = 1 - exp(-jπ)*(-1) = 1 + (-1) = 2
        // Actually: X[0] = x[0]+x[1] = 0, X[1] = x[0]-x[1] = 2
        assert!((out.re[0] - 0.0).abs() < 1e-12, "DC: {}", out.re[0]);
        assert!((out.re[1] - 2.0).abs() < 1e-12, "Nyquist: {}", out.re[1]);
    }

    #[test]
    fn test_input_len_zero_error() {
        let config = CztConfig {
            input_len: 0,
            output_len: 8,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / 8.0,
        };
        let result = czt(&[], &[], &config);
        assert!(matches!(result, Err(SignalError::InvalidSize(_))));
    }

    #[test]
    fn test_output_len_zero_error() {
        let config = CztConfig {
            input_len: 8,
            output_len: 0,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / 8.0,
        };
        let x = vec![1.0_f64; 8];
        let result = czt_real(&x, &config);
        assert!(matches!(result, Err(SignalError::InvalidSize(_))));
    }

    #[test]
    fn test_a_mag_zero_error() {
        let config = CztConfig {
            input_len: 8,
            output_len: 8,
            a_mag: 0.0,
            a_angle: 0.0,
            w_mag: 1.0,
            w_angle: -TAU / 8.0,
        };
        let x = vec![1.0_f64; 8];
        assert!(matches!(czt_real(&x, &config), Err(SignalError::InvalidParameter(_))));
    }

    #[test]
    fn test_w_mag_zero_error() {
        let config = CztConfig {
            input_len: 8,
            output_len: 8,
            a_mag: 1.0,
            a_angle: 0.0,
            w_mag: 0.0,
            w_angle: -TAU / 8.0,
        };
        let x = vec![1.0_f64; 8];
        assert!(matches!(czt_real(&x, &config), Err(SignalError::InvalidParameter(_))));
    }

    #[test]
    fn test_zoom_fft_f_lo_ge_f_hi_error() {
        let x = vec![1.0_f64; 16];
        assert!(matches!(
            zoom_fft(&x, 0.3, 0.1, 8),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_zoom_fft_f_hi_above_nyquist_error() {
        let x = vec![1.0_f64; 16];
        assert!(matches!(
            zoom_fft(&x, 0.0, 0.6, 8),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_large_n512_dft_accuracy() {
        let n = 512usize;
        let freq_bin = 10usize;
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * freq_bin as f64 * i as f64 / n as f64).cos())
            .collect();
        let czt_out = dft_via_czt(&x).expect("dft_via_czt n=512");
        let (ref_re, ref_im) = naive_dft(&x);
        for k in 0..n {
            assert!(
                (czt_out.re[k] - ref_re[k]).abs() < 1e-6,
                "re[{k}] diff={}",
                (czt_out.re[k] - ref_re[k]).abs()
            );
            assert!(
                (czt_out.im[k] - ref_im[k]).abs() < 1e-6,
                "im[{k}] diff={}",
                (czt_out.im[k] - ref_im[k]).abs()
            );
        }
    }
}
