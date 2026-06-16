//! Bluestein (chirp-Z) FFT for arbitrary lengths — CPU reference.
//!
//! Computes an `n`-point DFT for *any* `n` (including primes and lengths with
//! large prime factors) by re-expressing the transform as a linear convolution
//! that is evaluated with a power-of-two FFT.
//!
//! # Algorithm
//!
//! The DFT is rewritten using the identity `2 j k = j² + k² − (k − j)²`:
//!
//! ```text
//! X[k] = w[k] · Σ_j ( x[j] · w[j] ) · v[k − j]
//! ```
//!
//! where `w[j] = exp(−sπ i j² / n)` is the chirp sequence (`s = −1` for the
//! forward transform, `s = +1` for the inverse) and `v` is the conjugate
//! chirp arranged for circular convolution.  The inner convolution of length
//! `m = next_pow2(2 n − 1)` is computed as `IFFT(FFT(a) · FFT(b))`, reusing a
//! self-contained radix-2 FFT.
//!
//! Unlike the GPU-oriented [`crate::radix::bluestein`] module (which works with
//! [`crate::types::Complex`] and emits plan metadata), this routine operates on
//! split real / imaginary `f64` slices and returns the spectrum directly,
//! making it convenient as a CPU fallback or correctness oracle.

use crate::error::{FftError, FftResult};

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// Computes the arbitrary-length DFT of a complex signal via Bluestein's
/// algorithm.
///
/// The input is supplied as parallel real (`re`) and imaginary (`im`) slices,
/// each of length `n`.  When `inverse` is `true`, the inverse transform is
/// computed including the `1 / n` normalisation, so that
/// `bluestein_fft(forward, n, true)` recovers the original signal.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or if `re` / `im` do not both
/// have length `n`.
pub fn bluestein_fft(
    re: &[f64],
    im: &[f64],
    n: usize,
    inverse: bool,
) -> FftResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(FftError::InvalidSize("Bluestein size must be > 0".into()));
    }
    if re.len() != n || im.len() != n {
        return Err(FftError::InvalidSize(format!(
            "input length mismatch: n={n}, re={}, im={}",
            re.len(),
            im.len()
        )));
    }
    if n == 1 {
        return Ok((vec![re[0]], vec![im[0]]));
    }

    // Convolution length: next power of two >= 2n - 1.
    let m = (2 * n - 1).next_power_of_two();

    // Chirp sign: forward DFT uses exp(-i 2π jk / n); the j² chirp carries
    // the same sign.  Inverse uses the conjugate.
    let sign = if inverse { 1.0 } else { -1.0 };

    // Chirp w[j] = exp(sign * i * π * j² / n). Use j² mod 2n to keep the angle
    // small and accurate for large n (since exp is 2π-periodic in π j² / n with
    // period 2n in j²).
    let mut chirp_re = vec![0.0_f64; n];
    let mut chirp_im = vec![0.0_f64; n];
    for j in 0..n {
        // (j*j) mod (2n) avoids precision loss in the argument for large j.
        let jj = ((j as u128 * j as u128) % (2 * n as u128)) as f64;
        let angle = sign * PI * jj / n as f64;
        chirp_re[j] = angle.cos();
        chirp_im[j] = angle.sin();
    }

    // a[j] = x[j] * w[j], zero-padded to m.
    let mut a_re = vec![0.0_f64; m];
    let mut a_im = vec![0.0_f64; m];
    for j in 0..n {
        // (re + i im) * (cr + i ci)
        let cr = chirp_re[j];
        let ci = chirp_im[j];
        a_re[j] = re[j] * cr - im[j] * ci;
        a_im[j] = re[j] * ci + im[j] * cr;
    }

    // b[k] = conj(w[k]) arranged for circular convolution:
    //   b[0..n]      = conj(w[k])
    //   b[m-k]       = conj(w[k]) for k = 1..n  (negative-index wrap-around)
    let mut b_re = vec![0.0_f64; m];
    let mut b_im = vec![0.0_f64; m];
    for k in 0..n {
        b_re[k] = chirp_re[k];
        b_im[k] = -chirp_im[k];
    }
    for k in 1..n {
        b_re[m - k] = chirp_re[k];
        b_im[m - k] = -chirp_im[k];
    }

    // Convolve via FFT: c = IFFT(FFT(a) .* FFT(b)).
    fft_radix2(&mut a_re, &mut a_im, false);
    fft_radix2(&mut b_re, &mut b_im, false);
    for k in 0..m {
        let pr = a_re[k] * b_re[k] - a_im[k] * b_im[k];
        let pi = a_re[k] * b_im[k] + a_im[k] * b_re[k];
        a_re[k] = pr;
        a_im[k] = pi;
    }
    fft_radix2(&mut a_re, &mut a_im, true); // includes 1/m scaling

    // Out[k] = w[k] * c[k], then optional 1/n inverse normalisation.
    let norm = if inverse { 1.0 / n as f64 } else { 1.0 };
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        let cr = chirp_re[k];
        let ci = chirp_im[k];
        let yr = a_re[k] * cr - a_im[k] * ci;
        let yi = a_re[k] * ci + a_im[k] * cr;
        out_re[k] = yr * norm;
        out_im[k] = yi * norm;
    }

    Ok((out_re, out_im))
}

/// In-place Cooley-Tukey radix-2 FFT on split real / imaginary buffers.
///
/// `re` and `im` must share a power-of-two length.  When `inverse` is `true`
/// the result is scaled by `1 / n`.
fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }

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

    /// Naive O(n²) DFT used as the correctness oracle.
    fn naive_dft(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut out_re = vec![0.0_f64; n];
        let mut out_im = vec![0.0_f64; n];
        for k in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..n {
                let ang = sign * TAU * (k * t) as f64 / n as f64;
                let (c, s) = (ang.cos(), ang.sin());
                sr += re[t] * c - im[t] * s;
                si += re[t] * s + im[t] * c;
            }
            if inverse {
                out_re[k] = sr / n as f64;
                out_im[k] = si / n as f64;
            } else {
                out_re[k] = sr;
                out_im[k] = si;
            }
        }
        (out_re, out_im)
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "idx {i}: {x} vs {y}");
        }
    }

    #[test]
    fn matches_dft_size_3() {
        let re = [1.0, 2.0, -3.0];
        let im = [0.5, -1.0, 0.25];
        let (gr, gi) = bluestein_fft(&re, &im, 3, false).expect("ok");
        let (nr, ni) = naive_dft(&re, &im, false);
        assert_close(&gr, &nr, 1e-9);
        assert_close(&gi, &ni, 1e-9);
    }

    #[test]
    fn matches_dft_size_5() {
        let re = [1.0, 0.0, -2.0, 3.0, 0.5];
        let im = [0.0, 1.0, 0.0, -1.0, 2.0];
        let (gr, gi) = bluestein_fft(&re, &im, 5, false).expect("ok");
        let (nr, ni) = naive_dft(&re, &im, false);
        assert_close(&gr, &nr, 1e-9);
        assert_close(&gi, &ni, 1e-9);
    }

    #[test]
    fn matches_dft_size_7() {
        let re: Vec<f64> = (0..7).map(|k| (k as f64).sin()).collect();
        let im: Vec<f64> = (0..7).map(|k| (k as f64 * 0.3).cos()).collect();
        let (gr, gi) = bluestein_fft(&re, &im, 7, false).expect("ok");
        let (nr, ni) = naive_dft(&re, &im, false);
        assert_close(&gr, &nr, 1e-9);
        assert_close(&gi, &ni, 1e-9);
    }

    #[test]
    fn roundtrip() {
        let re: Vec<f64> = (0..11).map(|k| k as f64 * 0.7 - 2.0).collect();
        let im: Vec<f64> = (0..11).map(|k| (k as f64).cos()).collect();
        let (fr, fi) = bluestein_fft(&re, &im, 11, false).expect("fwd");
        let (rr, ri) = bluestein_fft(&fr, &fi, 11, true).expect("inv");
        assert_close(&rr, &re, 1e-9);
        assert_close(&ri, &im, 1e-9);
    }

    #[test]
    fn impulse() {
        // DFT of a unit impulse at index 0 is a flat unit spectrum.
        let n = 13;
        let mut re = vec![0.0_f64; n];
        let im = vec![0.0_f64; n];
        re[0] = 1.0;
        let (gr, gi) = bluestein_fft(&re, &im, n, false).expect("ok");
        for k in 0..n {
            assert!((gr[k] - 1.0).abs() < 1e-9, "re[{k}]={}", gr[k]);
            assert!(gi[k].abs() < 1e-9, "im[{k}]={}", gi[k]);
        }
    }

    #[test]
    fn linearity() {
        let n = 6;
        let a_re: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let a_im: Vec<f64> = (0..n).map(|k| -(k as f64)).collect();
        let b_re: Vec<f64> = (0..n).map(|k| (k as f64 * 2.0).sin()).collect();
        let b_im: Vec<f64> = (0..n).map(|k| (k as f64).cos()).collect();

        let (fa_re, fa_im) = bluestein_fft(&a_re, &a_im, n, false).expect("a");
        let (fb_re, fb_im) = bluestein_fft(&b_re, &b_im, n, false).expect("b");

        let sum_re: Vec<f64> = a_re.iter().zip(&b_re).map(|(x, y)| x + y).collect();
        let sum_im: Vec<f64> = a_im.iter().zip(&b_im).map(|(x, y)| x + y).collect();
        let (fs_re, fs_im) = bluestein_fft(&sum_re, &sum_im, n, false).expect("sum");

        for k in 0..n {
            assert!((fs_re[k] - (fa_re[k] + fb_re[k])).abs() < 1e-9);
            assert!((fs_im[k] - (fa_im[k] + fb_im[k])).abs() < 1e-9);
        }
    }

    #[test]
    fn dc_signal() {
        // Constant signal -> energy concentrated in bin 0.
        let n = 9;
        let re = vec![2.0_f64; n];
        let im = vec![0.0_f64; n];
        let (gr, gi) = bluestein_fft(&re, &im, n, false).expect("ok");
        assert!((gr[0] - 2.0 * n as f64).abs() < 1e-8);
        assert!(gi[0].abs() < 1e-8);
        for k in 1..n {
            assert!(gr[k].abs() < 1e-8, "re[{k}]={}", gr[k]);
            assert!(gi[k].abs() < 1e-8, "im[{k}]={}", gi[k]);
        }
    }

    #[test]
    fn prime_length() {
        // A large prime length that needs Bluestein (101 > 7).
        let n = 101;
        let re: Vec<f64> = (0..n).map(|k| (k as f64 * 0.11).sin()).collect();
        let im: Vec<f64> = (0..n).map(|k| (k as f64 * 0.07).cos()).collect();
        let (gr, gi) = bluestein_fft(&re, &im, n, false).expect("ok");
        let (nr, ni) = naive_dft(&re, &im, false);
        assert_close(&gr, &nr, 1e-7);
        assert_close(&gi, &ni, 1e-7);
    }

    #[test]
    fn inverse_scaling() {
        // IDFT of a flat unit spectrum is an impulse at index 0.
        let n = 7;
        let re = vec![1.0_f64; n];
        let im = vec![0.0_f64; n];
        let (gr, gi) = bluestein_fft(&re, &im, n, true).expect("ok");
        assert!((gr[0] - 1.0).abs() < 1e-9, "re[0]={}", gr[0]);
        assert!(gi[0].abs() < 1e-9);
        for k in 1..n {
            assert!(gr[k].abs() < 1e-9, "re[{k}]={}", gr[k]);
            assert!(gi[k].abs() < 1e-9, "im[{k}]={}", gi[k]);
        }
    }

    #[test]
    fn output_len() {
        for n in [2, 3, 5, 6, 10, 17, 31] {
            let re = vec![1.0_f64; n];
            let im = vec![0.0_f64; n];
            let (gr, gi) = bluestein_fft(&re, &im, n, false).expect("ok");
            assert_eq!(gr.len(), n);
            assert_eq!(gi.len(), n);
        }
    }

    #[test]
    fn rejects_bad_size() {
        assert!(bluestein_fft(&[], &[], 0, false).is_err());
        assert!(bluestein_fft(&[1.0], &[0.0, 0.0], 2, false).is_err());
    }

    #[test]
    fn size_one_identity() {
        let (gr, gi) = bluestein_fft(&[3.0], &[-1.0], 1, false).expect("ok");
        assert_eq!(gr, vec![3.0]);
        assert_eq!(gi, vec![-1.0]);
    }
}
