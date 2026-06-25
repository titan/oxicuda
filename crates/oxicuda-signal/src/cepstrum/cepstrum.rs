//! Cepstral analysis — real, power, and complex cepstrum plus liftering.
//!
//! The *cepstrum* maps the multiplicative interaction of an excitation source
//! and a filter (e.g. glottal pulse train × vocal-tract response) into an
//! additive one, by taking the (inverse) Fourier transform of the **log
//! spectrum**.  This is the foundation of homomorphic deconvolution, pitch /
//! fundamental-frequency estimation (the "rahmonic" peak at the pitch period)
//! and echo detection.
//!
//! Definitions used here (for a length-`N` real signal `x[n]`, with `X = DFT(x)`):
//!
//! ```text
//! power cepstrum   : c_p[n] = abs( IDFT( log |X|^2 ) )^2
//! real cepstrum    : c_r[n] = Re{ IDFT( log |X| ) }
//! complex cepstrum : ĉ[n]   = IDFT( log|X| + j·arg(X) )
//! ```
//!
//! The complex cepstrum is *invertible* (an inverse complex-cepstrum recovers
//! the original signal), provided the phase is unwrapped and a linear-phase
//! term is removed.  Liftering (cepstral windowing) then separates source from
//! filter: low-quefrency liftering keeps the slowly-varying spectral envelope,
//! high-quefrency liftering keeps the fine excitation structure.
//!
//! References:
//!   Bogert, Healy & Tukey (1963) "The Quefrency Alanysis of Time Series for
//!   Echoes", Proc. Symp. Time Series Analysis.
//!   Oppenheim & Schafer (2010) "Discrete-Time Signal Processing", ch. 13.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::{PI, TAU};

// --------------------------------------------------------------------------- //
//  Self-contained complex FFT (radix-2 / zero-padded mixed) — matches the
//  crate-wide convention used in `transform/czt.rs` and `spectral/welch.rs`.
// --------------------------------------------------------------------------- //

/// In-place radix-2 Cooley-Tukey FFT. `re`/`im` must have power-of-two length.
fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
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
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let half = len / 2;
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    cr * re[i + k + half] - ci * im[i + k + half],
                    cr * im[i + k + half] + ci * re[i + k + half],
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
                let tmp = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for (r, iv) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *iv *= scale;
        }
    }
}

/// Length used for cepstral FFTs — the next power of two ≥ the signal length.
/// A small floor of 2 keeps even 1-sample inputs valid.
fn fft_len(n: usize) -> usize {
    n.max(1).next_power_of_two().max(2)
}

// --------------------------------------------------------------------------- //
//  Power & real cepstrum
// --------------------------------------------------------------------------- //

/// Compute the **real cepstrum** of a real signal.
///
/// `c_r[n] = Re{ IDFT( log|X[k]| ) }`, returned at the FFT length
/// (next power of two ≥ `signal.len()`).  A small floor `eps` is added to the
/// magnitude before the logarithm to keep spectral nulls finite.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `eps` is negative or non-finite.
pub fn real_cepstrum(signal: &[f64], eps: f64) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::InvalidSize(
            "real_cepstrum input must be non-empty".to_owned(),
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(SignalError::InvalidParameter(
            "eps must be finite and >= 0".to_owned(),
        ));
    }
    let nfft = fft_len(signal.len());
    let mut re = vec![0.0_f64; nfft];
    let mut im = vec![0.0_f64; nfft];
    re[..signal.len()].copy_from_slice(signal);
    fft_radix2(&mut re, &mut im, false);
    // Replace spectrum with log-magnitude (real), zero imaginary.
    for k in 0..nfft {
        let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
        re[k] = (mag + eps).ln();
        im[k] = 0.0;
    }
    fft_radix2(&mut re, &mut im, true);
    Ok(re)
}

/// Compute the **power cepstrum** of a real signal.
///
/// `c_p[n] = |IDFT( log|X[k]|² )|²`.  This is the classic Bogert-Healy-Tukey
/// definition whose dominant peak (beyond quefrency 0) locates the pitch period
/// / echo delay.  `eps` floors the squared magnitude.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `eps` is negative or non-finite.
pub fn power_cepstrum(signal: &[f64], eps: f64) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::InvalidSize(
            "power_cepstrum input must be non-empty".to_owned(),
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(SignalError::InvalidParameter(
            "eps must be finite and >= 0".to_owned(),
        ));
    }
    let nfft = fft_len(signal.len());
    let mut re = vec![0.0_f64; nfft];
    let mut im = vec![0.0_f64; nfft];
    re[..signal.len()].copy_from_slice(signal);
    fft_radix2(&mut re, &mut im, false);
    for k in 0..nfft {
        let p = re[k] * re[k] + im[k] * im[k];
        re[k] = (p + eps).ln();
        im[k] = 0.0;
    }
    fft_radix2(&mut re, &mut im, true);
    Ok(re.iter().map(|&v| v * v).collect())
}

// --------------------------------------------------------------------------- //
//  Phase unwrapping + complex cepstrum (invertible)
// --------------------------------------------------------------------------- //

/// Unwrap a phase sequence so successive samples never jump by more than π.
///
/// Adds ±2π multiples to remove the principal-value discontinuities; required
/// for a well-defined complex cepstrum.
#[must_use]
pub fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(phase.len());
    out.push(phase[0]);
    let mut offset = 0.0_f64;
    for k in 1..phase.len() {
        let mut delta = phase[k] - phase[k - 1];
        // Reduce delta into (-π, π].
        while delta > PI {
            delta -= TAU;
        }
        while delta <= -PI {
            delta += TAU;
        }
        offset += delta - (phase[k] - phase[k - 1]);
        out.push(phase[k] + offset);
    }
    out
}

/// Compute the **complex cepstrum** of a real signal.
///
/// `ĉ[n] = IDFT( log|X| + j·unwrap(arg X) )` after removing the linear-phase
/// term (so the unwrapped phase passes through the origin with an integer
/// winding count).  Unlike the real/power cepstrum this transform is
/// *invertible* — see [`inverse_complex_cepstrum`].
///
/// Returns `(cepstrum, linear_phase_slope)` where `linear_phase_slope` (the
/// integer winding / sample-shift `r`) must be supplied back to the inverse to
/// reconstruct the signal exactly.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `eps` is negative or non-finite.
pub fn complex_cepstrum(signal: &[f64], eps: f64) -> SignalResult<(Vec<f64>, i64)> {
    if signal.is_empty() {
        return Err(SignalError::InvalidSize(
            "complex_cepstrum input must be non-empty".to_owned(),
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(SignalError::InvalidParameter(
            "eps must be finite and >= 0".to_owned(),
        ));
    }
    let nfft = fft_len(signal.len());
    let mut re = vec![0.0_f64; nfft];
    let mut im = vec![0.0_f64; nfft];
    re[..signal.len()].copy_from_slice(signal);
    fft_radix2(&mut re, &mut im, false);

    // Log-magnitude and raw phase.
    let mut log_mag = vec![0.0_f64; nfft];
    let mut phase = vec![0.0_f64; nfft];
    for k in 0..nfft {
        let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
        log_mag[k] = (mag + eps).ln();
        phase[k] = im[k].atan2(re[k]);
    }
    let mut uphase = unwrap_phase(&phase);

    // Remove the linear-phase component: round phase[N] / 2π ≈ winding r, and
    // subtract a ramp r·2πk/N so the resulting cepstrum is (anti)causal.
    let r = (uphase[nfft - 1] / TAU).round() as i64;
    if r != 0 {
        for (k, p) in uphase.iter_mut().enumerate() {
            *p -= TAU * r as f64 * k as f64 / nfft as f64;
        }
    }

    // Inverse-transform (log_mag + j·uphase).
    let mut cr = log_mag;
    let mut ci = uphase;
    fft_radix2(&mut cr, &mut ci, true);
    // The complex cepstrum of a real signal is real; the imaginary part is
    // numerical dust — discard it.
    Ok((cr, r))
}

/// Reconstruct a signal from its complex cepstrum.
///
/// Exact inverse of [`complex_cepstrum`]; `linear_phase_slope` is the second
/// returned value of that call. The reconstruction length equals
/// `min(nfft, original_len)` significant samples — pass `signal_len` to trim.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `cepstrum` is empty or `signal_len` is 0.
/// - [`SignalError::InvalidParameter`] if `cepstrum.len()` is not a power of two.
pub fn inverse_complex_cepstrum(
    cepstrum: &[f64],
    linear_phase_slope: i64,
    signal_len: usize,
) -> SignalResult<Vec<f64>> {
    let nfft = cepstrum.len();
    if nfft == 0 || signal_len == 0 {
        return Err(SignalError::InvalidSize(
            "inverse_complex_cepstrum needs non-empty cepstrum and signal_len > 0".to_owned(),
        ));
    }
    if !nfft.is_power_of_two() {
        return Err(SignalError::InvalidParameter(
            "cepstrum length must be a power of two".to_owned(),
        ));
    }
    // Forward DFT of the (real) cepstrum -> log_mag + j·uphase.
    let mut re = cepstrum.to_vec();
    let mut im = vec![0.0_f64; nfft];
    fft_radix2(&mut re, &mut im, false);
    // Re-apply the linear-phase ramp that was removed, then exponentiate:
    //   X[k] = exp(log_mag + j·phase) = e^{re} (cos·, sin·).
    for k in 0..nfft {
        let mag = re[k].exp();
        let phase = im[k] + TAU * linear_phase_slope as f64 * k as f64 / nfft as f64;
        re[k] = mag * phase.cos();
        im[k] = mag * phase.sin();
    }
    fft_radix2(&mut re, &mut im, true);
    re.truncate(signal_len.min(nfft));
    Ok(re)
}

// --------------------------------------------------------------------------- //
//  Liftering
// --------------------------------------------------------------------------- //

/// Low-time (low-quefrency) lifter — keep the first `cutoff` cepstral samples
/// (and their mirror in the upper half), zeroing the rest.
///
/// This isolates the slowly-varying spectral envelope (vocal-tract / filter
/// response) and is the homomorphic way to estimate a smooth log-spectrum.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `cutoff == 0` or
/// `cutoff > cepstrum.len()/2`.
pub fn lowpass_lifter(cepstrum: &[f64], cutoff: usize) -> SignalResult<Vec<f64>> {
    let n = cepstrum.len();
    if cutoff == 0 || cutoff > n / 2 {
        return Err(SignalError::InvalidParameter(format!(
            "lowpass lifter cutoff must be in 1..={}",
            n / 2
        )));
    }
    let mut out = vec![0.0_f64; n];
    // Keep DC..cutoff and the symmetric tail (n-cutoff+1..n).
    out[0] = cepstrum[0];
    for i in 1..cutoff {
        out[i] = cepstrum[i];
        out[n - i] = cepstrum[n - i];
    }
    out[cutoff] = cepstrum[cutoff];
    Ok(out)
}

/// High-time (high-quefrency) lifter — zero the first `cutoff` cepstral samples
/// (and their mirror), keeping the rest.
///
/// This isolates the fast-varying excitation structure (the source / pitch
/// harmonics), the complement of [`lowpass_lifter`].
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `cutoff == 0` or
/// `cutoff > cepstrum.len()/2`.
pub fn highpass_lifter(cepstrum: &[f64], cutoff: usize) -> SignalResult<Vec<f64>> {
    let n = cepstrum.len();
    if cutoff == 0 || cutoff > n / 2 {
        return Err(SignalError::InvalidParameter(format!(
            "highpass lifter cutoff must be in 1..={}",
            n / 2
        )));
    }
    let mut out = cepstrum.to_vec();
    out[0] = 0.0;
    for i in 1..cutoff {
        out[i] = 0.0;
        out[n - i] = 0.0;
    }
    out[cutoff] = 0.0;
    Ok(out)
}

/// Sinusoidal lifter `w[n] = 1 + (L/2)·sin(π n / L)` for `n = 0..L`, as used in
/// HTK/Kaldi-style MFCC liftering to equalise cepstral dynamic range.
///
/// Returns the liftered cepstrum (the input is multiplied element-wise by the
/// lifter; samples beyond `n_coeffs` are left untouched).
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `l == 0` or
/// `n_coeffs > cepstrum.len()`.
pub fn sinusoidal_lifter(cepstrum: &[f64], l: usize, n_coeffs: usize) -> SignalResult<Vec<f64>> {
    if l == 0 {
        return Err(SignalError::InvalidParameter(
            "sinusoidal lifter L must be > 0".to_owned(),
        ));
    }
    if n_coeffs > cepstrum.len() {
        return Err(SignalError::InvalidParameter(
            "n_coeffs exceeds cepstrum length".to_owned(),
        ));
    }
    let mut out = cepstrum.to_vec();
    let lf = l as f64;
    for (i, v) in out.iter_mut().take(n_coeffs).enumerate() {
        let w = 1.0 + 0.5 * lf * (PI * i as f64 / lf).sin();
        *v *= w;
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Pitch / fundamental-frequency estimation via the cepstral peak
// --------------------------------------------------------------------------- //

/// Estimate the fundamental frequency (pitch) of a voiced frame from the
/// dominant peak of its power cepstrum.
///
/// Searches the quefrency band `[fs/max_hz, fs/min_hz]` (in samples) for the
/// largest cepstral peak; the corresponding quefrency `q` gives `f0 = fs / q`.
/// Returns `Some(f0_hz)` when a peak is found in band, else `None` (unvoiced).
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `sample_rate_hz <= 0`,
///   `min_hz <= 0`, or `max_hz <= min_hz`.
pub fn cepstral_pitch(
    signal: &[f64],
    sample_rate_hz: f64,
    min_hz: f64,
    max_hz: f64,
) -> SignalResult<Option<f64>> {
    if sample_rate_hz <= 0.0 || min_hz <= 0.0 || max_hz <= min_hz {
        return Err(SignalError::InvalidParameter(
            "require sample_rate_hz>0, min_hz>0, max_hz>min_hz".to_owned(),
        ));
    }
    let cep = power_cepstrum(signal, 1e-12)?;
    let n = cep.len();
    // Quefrency search bounds (in samples): q = fs / f.
    let q_lo = (sample_rate_hz / max_hz).floor() as usize;
    let q_hi = (sample_rate_hz / min_hz).ceil() as usize;
    let lo = q_lo.max(1);
    let hi = q_hi.min(n / 2);
    if lo >= hi {
        return Ok(None);
    }
    let mut best_q = lo;
    let mut best_v = cep[lo];
    for (q, &v) in cep.iter().enumerate().take(hi).skip(lo + 1) {
        if v > best_v {
            best_v = v;
            best_q = q;
        }
    }
    Ok(Some(sample_rate_hz / best_q as f64))
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG noise in [-1, 1) (crate convention: full-range / 2³²).
    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 32) as u32) as f64 / (u32::MAX as f64);
            v.push(2.0 * u - 1.0);
        }
        v
    }

    #[test]
    fn test_empty_errors() {
        assert!(real_cepstrum(&[], 1e-9).is_err());
        assert!(power_cepstrum(&[], 1e-9).is_err());
        assert!(complex_cepstrum(&[], 1e-9).is_err());
        assert!(real_cepstrum(&[1.0], -1.0).is_err());
    }

    #[test]
    fn test_real_cepstrum_length_pow2() {
        let c = real_cepstrum(&[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12).expect("ok");
        assert_eq!(c.len(), 8); // next pow2 of 5
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_unwrap_phase_monotone() {
        // Wrapping a +0.9π-per-step ramp; unwrap must recover the smooth ramp.
        let raw: Vec<f64> = (0..10)
            .map(|k| {
                let mut p = 0.9 * PI * k as f64;
                while p > PI {
                    p -= TAU;
                }
                while p <= -PI {
                    p += TAU;
                }
                p
            })
            .collect();
        let uw = unwrap_phase(&raw);
        for k in 1..uw.len() {
            assert!((uw[k] - uw[k - 1] - 0.9 * PI).abs() < 1e-9, "step {k}");
        }
    }

    #[test]
    fn test_complex_cepstrum_roundtrip() {
        // The complex cepstrum is invertible: inverse(forward(x)) == x.
        let x = vec![1.0_f64, 0.5, -0.25, 0.75, 0.1, -0.6, 0.3, 0.2];
        let (cep, r) = complex_cepstrum(&x, 0.0).expect("forward ok");
        let recon = inverse_complex_cepstrum(&cep, r, x.len()).expect("inverse ok");
        assert_eq!(recon.len(), x.len());
        for (a, b) in x.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-7, "roundtrip {a} vs {b}");
        }
    }

    #[test]
    fn test_complex_cepstrum_roundtrip_random() {
        let x = lcg_noise(16, 12345);
        let (cep, r) = complex_cepstrum(&x, 0.0).expect("forward ok");
        let recon = inverse_complex_cepstrum(&cep, r, x.len()).expect("inverse ok");
        for (a, b) in x.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip {a} vs {b}");
        }
    }

    #[test]
    fn test_power_cepstrum_detects_echo() {
        // x[n] = s[n] + α s[n-D] has a cepstral peak at quefrency D.
        let n = 256usize;
        let delay = 40usize;
        let alpha = 0.6_f64;
        let base = lcg_noise(n, 999);
        let mut x = base.clone();
        for i in delay..n {
            x[i] += alpha * base[i - delay];
        }
        let cep = power_cepstrum(&x, 1e-12).expect("ok");
        // The largest peak in the plausible echo band must sit at the delay.
        let search = &cep[10..120];
        let (mut bi, mut bv) = (0usize, search[0]);
        for (i, &v) in search.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        let peak_q = bi + 10;
        assert!(
            (peak_q as i64 - delay as i64).abs() <= 1,
            "echo peak at q={peak_q}, expected ~{delay}"
        );
    }

    #[test]
    fn test_cepstral_pitch_detects_f0() {
        // Synthetic voiced signal: pulse train at f0 with formant shaping.
        let fs = 8000.0_f64;
        let f0 = 125.0_f64; // period = 64 samples
        let n = 1024usize;
        let period = (fs / f0).round() as usize;
        let mut x = vec![0.0_f64; n];
        // Impulse train.
        let mut idx = 0usize;
        while idx < n {
            x[idx] = 1.0;
            idx += period;
        }
        // Shape with a short FIR (vocal-tract-ish) to create harmonics.
        let h = [1.0_f64, 0.8, 0.5, 0.2];
        let mut shaped = vec![0.0_f64; n];
        for i in 0..n {
            for (j, &hc) in h.iter().enumerate() {
                if i >= j {
                    shaped[i] += hc * x[i - j];
                }
            }
        }
        let est = cepstral_pitch(&shaped, fs, 80.0, 300.0)
            .expect("pitch ok")
            .expect("voiced");
        assert!((est - f0).abs() < 8.0, "f0 est={est}, true={f0}");
        // Parameter validation.
        assert!(cepstral_pitch(&shaped, -1.0, 80.0, 300.0).is_err());
    }

    #[test]
    fn test_lowpass_highpass_lifter_complement() {
        // low + high lifter should reconstruct the original cepstrum.
        let cep = real_cepstrum(&lcg_noise(32, 7), 1e-12).expect("ok");
        let cutoff = 5usize;
        let lo = lowpass_lifter(&cep, cutoff).expect("lp");
        let hi = highpass_lifter(&cep, cutoff).expect("hp");
        for i in 0..cep.len() {
            assert!((lo[i] + hi[i] - cep[i]).abs() < 1e-12, "sum at {i}");
        }
        assert!(lowpass_lifter(&cep, 0).is_err());
        assert!(highpass_lifter(&cep, cep.len()).is_err());
    }

    #[test]
    fn test_sinusoidal_lifter() {
        let cep = vec![1.0_f64; 13];
        let out = sinusoidal_lifter(&cep, 22, 13).expect("ok");
        // w[0] = 1 (sin 0), monotone rise then symmetric within 0..L.
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!(out[1] > 1.0);
        assert!(sinusoidal_lifter(&cep, 0, 13).is_err());
        assert!(sinusoidal_lifter(&cep, 22, 99).is_err());
    }
}
