//! Real-input FFT (`rfft`) and its inverse (`irfft`) — CPU reference.
//!
//! For a length-`n` real signal the DFT is conjugate-symmetric:
//! `X[n − k] = conj(X[k])`.  Only the first `n / 2 + 1` bins are independent,
//! so [`rfft`] returns the *half spectrum* and [`irfft`] reconstructs the full
//! real signal from it.
//!
//! These routines accept arbitrary `n` (not restricted to powers of two): the
//! core complex transform falls back to the in-crate [`crate::transforms::bluestein`]
//! Bluestein FFT for non-power-of-two lengths, and uses a direct radix-2 path
//! otherwise.  Output is the standard `[(re, im); n / 2 + 1]` complex layout.

use crate::error::{FftError, FftResult};
use crate::transforms::bluestein::bluestein_fft;

const TAU: f64 = std::f64::consts::TAU;

/// Forward real FFT: returns the half spectrum `[(re, im); n / 2 + 1]`.
///
/// `signal` must have length `n`.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or `signal.len() != n`.
pub fn rfft(signal: &[f64], n: usize) -> FftResult<Vec<(f64, f64)>> {
    if n == 0 {
        return Err(FftError::InvalidSize("rfft size must be > 0".into()));
    }
    if signal.len() != n {
        return Err(FftError::InvalidSize(format!(
            "signal length {} != n {n}",
            signal.len()
        )));
    }

    let im_in = vec![0.0_f64; n];
    let (re, im) = full_fft(signal, &im_in, false)?;

    let half = n / 2 + 1;
    let mut out = Vec::with_capacity(half);
    for k in 0..half {
        out.push((re[k], im[k]));
    }
    Ok(out)
}

/// Inverse real FFT: reconstructs the length-`n` real signal from a half
/// spectrum of length `n / 2 + 1`.
///
/// The imaginary part of the reconstructed signal is discarded (it is zero up
/// to round-off for a Hermitian input).
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or
/// `spectrum.len() != n / 2 + 1`.
pub fn irfft(spectrum: &[(f64, f64)], n: usize) -> FftResult<Vec<f64>> {
    if n == 0 {
        return Err(FftError::InvalidSize("irfft size must be > 0".into()));
    }
    let half = n / 2 + 1;
    if spectrum.len() != half {
        return Err(FftError::InvalidSize(format!(
            "spectrum length {} != n/2+1 = {half}",
            spectrum.len()
        )));
    }

    // Rebuild the full Hermitian-symmetric spectrum of length n.
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for (k, &(r, i)) in spectrum.iter().enumerate() {
        re[k] = r;
        im[k] = i;
    }
    // Mirror: X[n-k] = conj(X[k]) for k = 1 .. (n-1)/2 (and Nyquist if present).
    for k in 1..half {
        let mirror = n - k;
        if mirror >= half {
            re[mirror] = re[k];
            im[mirror] = -im[k];
        }
    }

    let (out_re, _out_im) = full_fft(&re, &im, true)?;
    Ok(out_re)
}

/// Computes the full complex DFT of `(re, im)` using a power-of-two radix-2
/// path when possible, falling back to Bluestein otherwise.
fn full_fft(re: &[f64], im: &[f64], inverse: bool) -> FftResult<(Vec<f64>, Vec<f64>)> {
    let n = re.len();
    if n.is_power_of_two() {
        let mut r = re.to_vec();
        let mut i = im.to_vec();
        fft_radix2(&mut r, &mut i, inverse);
        Ok((r, i))
    } else {
        bluestein_fft(re, im, n, inverse)
    }
}

/// In-place radix-2 FFT on split buffers; scales by `1 / n` when `inverse`.
fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
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
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let half = i + k + len / 2;
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (cr * re[half] - ci * im[half], cr * im[half] + ci * re[half]);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[half] = ur - vr;
                im[half] = ui - vi;
                let tmp = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv = 1.0 / n as f64;
        for k in 0..n {
            re[k] *= inv;
            im[k] *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft_re(signal: &[f64]) -> Vec<(f64, f64)> {
        let n = signal.len();
        (0..n)
            .map(|k| {
                let mut sr = 0.0_f64;
                let mut si = 0.0_f64;
                for (t, &x) in signal.iter().enumerate() {
                    let ang = -TAU * (k * t) as f64 / n as f64;
                    sr += x * ang.cos();
                    si += x * ang.sin();
                }
                (sr, si)
            })
            .collect()
    }

    #[test]
    fn output_len_half_plus_1() {
        for n in [2, 4, 7, 8, 16, 17, 32] {
            let sig = vec![1.0_f64; n];
            let spec = rfft(&sig, n).expect("ok");
            assert_eq!(spec.len(), n / 2 + 1);
        }
    }

    #[test]
    fn roundtrip() {
        let n = 16;
        let sig: Vec<f64> = (0..n).map(|k| (k as f64 * 0.4).sin() + 0.3).collect();
        let spec = rfft(&sig, n).expect("fwd");
        let rec = irfft(&spec, n).expect("inv");
        assert_eq!(rec.len(), n);
        for (a, b) in sig.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn dc_signal() {
        let n = 8;
        let sig = vec![3.0_f64; n];
        let spec = rfft(&sig, n).expect("ok");
        assert!((spec[0].0 - 3.0 * n as f64).abs() < 1e-9);
        assert!(spec[0].1.abs() < 1e-9);
        for s in &spec[1..] {
            assert!(s.0.abs() < 1e-8 && s.1.abs() < 1e-8);
        }
    }

    #[test]
    fn real_tone() {
        // A pure cosine at bin 2 in an N=16 frame puts all energy in bin 2.
        let n = 16;
        let sig: Vec<f64> = (0..n)
            .map(|t| (TAU * 2.0 * t as f64 / n as f64).cos())
            .collect();
        let spec = rfft(&sig, n).expect("ok");
        let mag2: Vec<f64> = spec.iter().map(|(r, i)| r * r + i * i).collect();
        let peak = mag2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(k, _)| k)
            .expect("value should be present");
        assert_eq!(peak, 2);
        assert!((mag2[2].sqrt() - n as f64 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn irfft_real_output() {
        // irfft always returns exactly n real samples.
        let n = 12;
        let half = n / 2 + 1;
        let spec: Vec<(f64, f64)> = (0..half).map(|k| (k as f64, 0.0)).collect();
        // Force a valid Hermitian spectrum: zero out imag of DC/Nyquist already.
        let out = irfft(&spec, n).expect("ok");
        assert_eq!(out.len(), n);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn hermitian_symmetry_implied() {
        // rfft of a real signal must match the half of the full DFT.
        let n = 16;
        let sig: Vec<f64> = (0..n)
            .map(|k| (k as f64 * 0.9).cos() - 0.2 * k as f64)
            .collect();
        let spec = rfft(&sig, n).expect("ok");
        let full = naive_dft_re(&sig);
        for k in 0..(n / 2 + 1) {
            assert!((spec[k].0 - full[k].0).abs() < 1e-7, "re bin {k}");
            assert!((spec[k].1 - full[k].1).abs() < 1e-7, "im bin {k}");
        }
    }

    #[test]
    fn n_not_pow2_error_or_handled() {
        // Non-power-of-two length is handled (via Bluestein), round-trips cleanly.
        let n = 12;
        let sig: Vec<f64> = (0..n).map(|k| k as f64 - 5.0).collect();
        let spec = rfft(&sig, n).expect("rfft handles non-pow2");
        assert_eq!(spec.len(), n / 2 + 1);
        let rec = irfft(&spec, n).expect("irfft handles non-pow2");
        for (a, b) in sig.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn linearity() {
        let n = 8;
        let a: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let b: Vec<f64> = (0..n).map(|k| (k as f64 * 0.5).sin()).collect();
        let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let sa = rfft(&a, n).expect("a");
        let sb = rfft(&b, n).expect("b");
        let ss = rfft(&sum, n).expect("sum");
        for k in 0..(n / 2 + 1) {
            assert!((ss[k].0 - (sa[k].0 + sb[k].0)).abs() < 1e-9);
            assert!((ss[k].1 - (sa[k].1 + sb[k].1)).abs() < 1e-9);
        }
    }

    #[test]
    fn impulse() {
        // rfft of impulse at 0 is flat with unit magnitude across all bins.
        let n = 16;
        let mut sig = vec![0.0_f64; n];
        sig[0] = 1.0;
        let spec = rfft(&sig, n).expect("ok");
        for (r, i) in &spec {
            assert!((r - 1.0).abs() < 1e-9, "re={r}");
            assert!(i.abs() < 1e-9, "im={i}");
        }
    }

    #[test]
    fn rejects_bad_sizes() {
        assert!(rfft(&[], 0).is_err());
        assert!(rfft(&[1.0, 2.0], 3).is_err());
        assert!(irfft(&[(1.0, 0.0)], 0).is_err());
        // wrong half length
        assert!(irfft(&[(1.0, 0.0)], 8).is_err());
    }
}
