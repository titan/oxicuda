//! Goertzel algorithm — single-bin DFT evaluation.
//!
//! The Goertzel algorithm computes one DFT coefficient `X[k]` (or the power at
//! a target frequency) in `O(n)` time and `O(1)` memory, without forming the
//! whole transform.  It is the method of choice for tone detection (DTMF), goal
//! frequency tracking, and any task that needs only a handful of bins.
//!
//! The recurrence (second-order IIR section) is
//!
//! ```text
//! coeff = 2 cos(2π k / n)
//! s[i]  = x[i] + coeff · s[i-1] − s[i-2]
//! ```
//!
//! After processing all `n` samples,
//!
//! ```text
//! X[k] = s[n-1] − exp(−i 2π k / n) · s[n-2]
//! power = s[n-1]² + s[n-2]² − coeff · s[n-1] · s[n-2]
//! ```

use crate::error::{FftError, FftResult};

const TAU: f64 = std::f64::consts::TAU;

/// Computes the complex DFT coefficient `X[k]` at integer bin `k` for a real
/// `signal` of length `n`, returning `(re, im)`.
///
/// Equivalent to `Σ_t x[t] · exp(−i 2π k t / n)` but evaluated with the
/// Goertzel recurrence.  `k` is taken modulo `n`.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or `signal.len() != n`.
pub fn goertzel(signal: &[f64], n: usize, k: usize) -> FftResult<(f64, f64)> {
    validate(signal, n)?;
    let omega = TAU * (k % n) as f64 / n as f64;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let coeff = 2.0 * cos_w;

    let (mut s_prev, mut s_prev2) = (0.0_f64, 0.0_f64);
    for &x in signal {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    // X[k] = Σ x[t] exp(-i 2π k t / n) = s_prev · cos ω − s_prev2 + i · s_prev · sin ω
    let re = s_prev * cos_w - s_prev2;
    let im = s_prev * sin_w;
    Ok((re, im))
}

/// Computes the squared magnitude `|X[k]|²` at bin `k` using the Goertzel
/// power form (one fewer trig multiply than squaring [`goertzel`]).
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or `signal.len() != n`.
pub fn goertzel_power(signal: &[f64], n: usize, k: usize) -> FftResult<f64> {
    validate(signal, n)?;
    let omega = TAU * (k % n) as f64 / n as f64;
    let coeff = 2.0 * omega.cos();

    let (mut s_prev, mut s_prev2) = (0.0_f64, 0.0_f64);
    for &x in signal {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    Ok(s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2)
}

/// Computes the DFT coefficient at an *arbitrary* (non-integer) normalised
/// frequency `freq` cycles-per-sample in `[0, 1)` via the generalised Goertzel
/// algorithm.  Returns `(re, im)` of `Σ_t x[t] · exp(−i 2π freq · t)`.
///
/// This is the "zoom" variant useful when the frequency of interest does not
/// land on a DFT grid point.
///
/// # Errors
///
/// Returns [`FftError::InvalidSize`] if `n == 0` or `signal.len() != n`.
/// Returns [`FftError::InvalidSize`] if `freq` is not finite.
pub fn goertzel_freq(signal: &[f64], n: usize, freq: f64) -> FftResult<(f64, f64)> {
    validate(signal, n)?;
    if !freq.is_finite() {
        return Err(FftError::InvalidSize(format!("non-finite freq {freq}")));
    }
    let omega = TAU * freq;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let coeff = 2.0 * cos_w;

    let (mut s_prev, mut s_prev2) = (0.0_f64, 0.0_f64);
    for &x in signal {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    // Same closed form as the integer-bin case, evaluated at the continuous ω:
    // X(freq) = Σ x[t] exp(-i 2π freq t).
    let re = s_prev * cos_w - s_prev2;
    let im = s_prev * sin_w;
    Ok((re, im))
}

fn validate(signal: &[f64], n: usize) -> FftResult<()> {
    if n == 0 {
        return Err(FftError::InvalidSize("Goertzel size must be > 0".into()));
    }
    if signal.len() != n {
        return Err(FftError::InvalidSize(format!(
            "signal length {} != n {n}",
            signal.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dft_bin(signal: &[f64], k: usize) -> (f64, f64) {
        let n = signal.len();
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for (t, &x) in signal.iter().enumerate() {
            let ang = -TAU * (k * t) as f64 / n as f64;
            sr += x * ang.cos();
            si += x * ang.sin();
        }
        (sr, si)
    }

    #[test]
    fn matches_dft_bin_dc() {
        let sig = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (gr, gi) = goertzel(&sig, sig.len(), 0).expect("ok");
        let (dr, di) = dft_bin(&sig, 0);
        assert!((gr - dr).abs() < 1e-9, "{gr} vs {dr}");
        assert!((gi - di).abs() < 1e-9);
    }

    #[test]
    fn matches_dft_bin_mid() {
        let sig: Vec<f64> = (0..16).map(|t| (t as f64 * 0.6).sin() + 0.4).collect();
        for k in [1usize, 2, 3, 7, 8] {
            let (gr, gi) = goertzel(&sig, sig.len(), k).expect("ok");
            let (dr, di) = dft_bin(&sig, k);
            assert!((gr - dr).abs() < 1e-7, "bin {k}: re {gr} vs {dr}");
            assert!((gi - di).abs() < 1e-7, "bin {k}: im {gi} vs {di}");
        }
    }

    #[test]
    fn matches_all_bins() {
        let sig: Vec<f64> = (0..13).map(|t| (t as f64).cos()).collect();
        for k in 0..sig.len() {
            let (gr, gi) = goertzel(&sig, sig.len(), k).expect("ok");
            let (dr, di) = dft_bin(&sig, k);
            assert!((gr - dr).abs() < 1e-7, "bin {k}");
            assert!((gi - di).abs() < 1e-7, "bin {k}");
        }
    }

    #[test]
    fn power_matches_magnitude() {
        let sig: Vec<f64> = (0..20).map(|t| (t as f64 * 0.31).sin()).collect();
        for k in [0usize, 1, 4, 9] {
            let p = goertzel_power(&sig, sig.len(), k).expect("ok");
            let (r, i) = goertzel(&sig, sig.len(), k).expect("ok");
            assert!((p - (r * r + i * i)).abs() < 1e-6, "bin {k}: {p}");
        }
    }

    #[test]
    fn detects_tone() {
        // A pure tone at bin 5 should have far more power there than elsewhere.
        let n = 32;
        let sig: Vec<f64> = (0..n)
            .map(|t| (TAU * 5.0 * t as f64 / n as f64).cos())
            .collect();
        let p5 = goertzel_power(&sig, n, 5).expect("ok");
        let p3 = goertzel_power(&sig, n, 3).expect("ok");
        assert!(p5 > 100.0 * (p3 + 1e-12), "p5={p5} p3={p3}");
    }

    #[test]
    fn k_modulo_n() {
        // Bin k and k+n must give identical results.
        let sig: Vec<f64> = (0..9).map(|t| t as f64 - 4.0).collect();
        let (r1, i1) = goertzel(&sig, sig.len(), 2).expect("ok");
        let (r2, i2) = goertzel(&sig, sig.len(), 2 + 9).expect("ok");
        assert!((r1 - r2).abs() < 1e-9);
        assert!((i1 - i2).abs() < 1e-9);
    }

    #[test]
    fn impulse_flat() {
        let n = 10;
        let mut sig = vec![0.0_f64; n];
        sig[0] = 1.0;
        for k in 0..n {
            let p = goertzel_power(&sig, n, k).expect("ok");
            assert!((p - 1.0).abs() < 1e-9, "bin {k}: {p}");
        }
    }

    #[test]
    fn generalised_freq_matches_grid() {
        // At a grid frequency k/n the generalised form equals the integer form.
        let sig: Vec<f64> = (0..16).map(|t| (t as f64 * 0.5).sin() + 1.0).collect();
        let k = 3usize;
        let (gr, gi) = goertzel(&sig, sig.len(), k).expect("ok");
        let (fr, fi) = goertzel_freq(&sig, sig.len(), k as f64 / sig.len() as f64).expect("ok");
        assert!((gr - fr).abs() < 1e-7, "{gr} vs {fr}");
        assert!((gi - fi).abs() < 1e-7, "{gi} vs {fi}");
    }

    #[test]
    fn finite_outputs() {
        let sig: Vec<f64> = (0..64).map(|t| (t as f64 * 1.3).sin()).collect();
        for k in 0..10 {
            let (r, i) = goertzel(&sig, sig.len(), k).expect("ok");
            assert!(r.is_finite() && i.is_finite());
            let p = goertzel_power(&sig, sig.len(), k).expect("ok");
            assert!(p.is_finite() && p >= -1e-9);
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(goertzel(&[], 0, 0).is_err());
        assert!(goertzel(&[1.0, 2.0], 3, 0).is_err());
        assert!(goertzel_power(&[1.0], 2, 0).is_err());
        assert!(goertzel_freq(&[1.0], 1, f64::NAN).is_err());
    }
}
