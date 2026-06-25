//! Deconvolution — recovering an input signal from its blurred / filtered
//! observation.
//!
//! Given an observation `y = x * h + noise` (linear convolution of an unknown
//! signal `x` with a known kernel / point-spread function `h`), this module
//! provides the two workhorse deconvolution estimators:
//!
//! * [`wiener_deconvolve`] — the linear minimum-mean-square-error (MMSE)
//!   frequency-domain inverse, regularised by the noise-to-signal ratio so the
//!   inverse stays well-conditioned at spectral nulls of `H`.
//! * [`richardson_lucy`] — the iterative, non-negative, flux-conserving
//!   maximum-likelihood deconvolution for Poisson-noise data (the standard
//!   astronomy / microscopy image-restoration method).
//!
//! Both operate on 1-D real signals; the convolution uses the crate's
//! self-contained radix-2 FFT for the frequency-domain method and a direct
//! correlation for the iterative method.
//!
//! References:
//!   Wiener (1949) "Extrapolation, Interpolation, and Smoothing of Stationary
//!   Time Series".
//!   Richardson (1972) JOSA 62(1):55; Lucy (1974) Astron. J. 79:745.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::TAU;

// --------------------------------------------------------------------------- //
//  Self-contained radix-2 FFT (matches crate convention).
// --------------------------------------------------------------------------- //

fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
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
            let half = len / 2;
            for k in 0..half {
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

// --------------------------------------------------------------------------- //
//  Wiener (regularised) deconvolution
// --------------------------------------------------------------------------- //

/// Frequency-domain Wiener deconvolution of `observed` by kernel `kernel`.
///
/// Computes the MMSE inverse filter
///
/// ```text
/// G[k] = conj(H[k]) / ( |H[k]|² + nsr )
/// X̂[k] = G[k] · Y[k]
/// ```
///
/// where `nsr` is the noise-to-signal power ratio (a single scalar acting as
/// Tikhonov regularisation — larger values suppress noise amplification at
/// frequencies where `|H|` is small).  Set `nsr = 0` for the (ill-conditioned)
/// exact inverse.
///
/// ## Observation model
///
/// `observed` is treated as the (full) linear convolution `y = x ⊛ h`; for a
/// length-`N` signal and length-`M` kernel that record has length `N + M − 1`.
/// The transform length is `nfft = next_pow2(observed.len())` and both
/// `observed` and `kernel` are zero-padded to it, so when `observed` carries the
/// complete convolution the circular- and linear-convolution models coincide and
/// the `nsr = 0` inverse recovers `x` to machine precision.  The result is
/// truncated to `observed.len()` samples (the first `N` of which are the signal
/// estimate).
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `observed` or `kernel` is empty.
/// - [`SignalError::InvalidParameter`] if `nsr` is negative or non-finite.
pub fn wiener_deconvolve(observed: &[f64], kernel: &[f64], nsr: f64) -> SignalResult<Vec<f64>> {
    if observed.is_empty() || kernel.is_empty() {
        return Err(SignalError::InvalidSize(
            "wiener_deconvolve requires non-empty observed and kernel".to_owned(),
        ));
    }
    if !nsr.is_finite() || nsr < 0.0 {
        return Err(SignalError::InvalidParameter(
            "nsr must be finite and >= 0".to_owned(),
        ));
    }
    // Size the transform to the observation length so a full-convolution record
    // is deconvolved exactly under the circular model.
    let nfft = observed.len().next_power_of_two().max(2);

    // Y = FFT(observed), H = FFT(kernel), both zero-padded to nfft.
    let mut yr = vec![0.0_f64; nfft];
    let mut yi = vec![0.0_f64; nfft];
    yr[..observed.len()].copy_from_slice(observed);
    let mut hr = vec![0.0_f64; nfft];
    let mut hi = vec![0.0_f64; nfft];
    hr[..kernel.len()].copy_from_slice(kernel);
    fft_radix2(&mut yr, &mut yi, false);
    fft_radix2(&mut hr, &mut hi, false);

    // X̂[k] = Y[k] · conj(H[k]) / (|H[k]|² + nsr).
    let mut xr = vec![0.0_f64; nfft];
    let mut xi = vec![0.0_f64; nfft];
    for k in 0..nfft {
        let h2 = hr[k] * hr[k] + hi[k] * hi[k] + nsr;
        // numerator = Y · conj(H) = (yr + j yi)(hr − j hi).
        let nr = yr[k] * hr[k] + yi[k] * hi[k];
        let ni = yi[k] * hr[k] - yr[k] * hi[k];
        // Guard against an all-zero kernel bin with nsr == 0.
        if h2 > 0.0 {
            xr[k] = nr / h2;
            xi[k] = ni / h2;
        }
    }
    fft_radix2(&mut xr, &mut xi, true);
    xr.truncate(observed.len());
    Ok(xr)
}

// --------------------------------------------------------------------------- //
//  Richardson-Lucy iterative deconvolution
// --------------------------------------------------------------------------- //

/// Linear (full) convolution of `a` with `b` (direct, `O(|a|·|b|)`).
fn convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut out = vec![0.0_f64; n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// "Same"-length convolution of `signal` with `kernel`, centred so the output
/// has the same length as `signal` (the standard PSF-blur convention).
fn convolve_same(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let full = convolve_full(signal, kernel);
    let start = (kernel.len() - 1) / 2;
    full[start..start + signal.len()].to_vec()
}

/// Correlation of `signal` with `kernel` matched to [`convolve_same`] (i.e.
/// convolution with the time-reversed kernel, same centring).
fn correlate_same(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let flipped: Vec<f64> = kernel.iter().rev().copied().collect();
    // For a length-L kernel, the matched flip-centre offset is L/2.
    let full = convolve_full(signal, &flipped);
    let start = kernel.len() / 2;
    full[start..start + signal.len()].to_vec()
}

/// Richardson-Lucy iterative (maximum-likelihood, Poisson) deconvolution.
///
/// Restores a non-negative signal `x` from `observed ≈ x ⊛ kernel` by the
/// multiplicative update
///
/// ```text
/// x_{t+1} = x_t · [ kernelᵀ ⊛ ( observed / (x_t ⊛ kernel) ) ]
/// ```
///
/// The estimate stays non-negative and conserves total flux (for a normalised
/// kernel), which is why R-L is the standard method for image / spectral
/// restoration under photon-counting noise.  `iterations` controls the
/// regularisation implicitly (more iterations → sharper but noisier).
///
/// The kernel is internally normalised to unit sum.  The "same"-length
/// convolution convention is used, so the output has the same length as
/// `observed`.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `observed` or `kernel` is empty.
/// - [`SignalError::InvalidParameter`] if `iterations == 0` or the kernel sums
///   to zero (or a non-positive total).
pub fn richardson_lucy(
    observed: &[f64],
    kernel: &[f64],
    iterations: usize,
) -> SignalResult<Vec<f64>> {
    if observed.is_empty() || kernel.is_empty() {
        return Err(SignalError::InvalidSize(
            "richardson_lucy requires non-empty observed and kernel".to_owned(),
        ));
    }
    if iterations == 0 {
        return Err(SignalError::InvalidParameter(
            "richardson_lucy iterations must be >= 1".to_owned(),
        ));
    }
    let ksum: f64 = kernel.iter().sum();
    if ksum <= 0.0 || !ksum.is_finite() {
        return Err(SignalError::InvalidParameter(
            "richardson_lucy kernel must have positive, finite sum".to_owned(),
        ));
    }
    let kn: Vec<f64> = kernel.iter().map(|&k| k / ksum).collect();
    // Initialise with a flat estimate equal to the observation mean (a common,
    // robust choice that conserves the initial flux).
    let mean = observed.iter().sum::<f64>() / observed.len() as f64;
    let init = if mean > 0.0 { mean } else { 1.0 };
    let mut est = vec![init; observed.len()];
    let eps = 1e-12_f64;

    for _ in 0..iterations {
        // Forward model: blurred = est ⊛ kernel.
        let blurred = convolve_same(&est, &kn);
        // Ratio = observed / blurred (guarded).
        let ratio: Vec<f64> = observed
            .iter()
            .zip(blurred.iter())
            .map(|(&o, &b)| o / (b + eps))
            .collect();
        // Correction = kernelᵀ ⊛ ratio.
        let correction = correlate_same(&ratio, &kn);
        for (e, c) in est.iter_mut().zip(correction.iter()) {
            *e *= c;
            if *e < 0.0 {
                *e = 0.0;
            }
        }
    }
    Ok(est)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
    fn test_errors() {
        assert!(wiener_deconvolve(&[], &[1.0], 0.0).is_err());
        assert!(wiener_deconvolve(&[1.0], &[], 0.0).is_err());
        assert!(wiener_deconvolve(&[1.0], &[1.0], -1.0).is_err());
        assert!(richardson_lucy(&[], &[1.0], 5).is_err());
        assert!(richardson_lucy(&[1.0], &[1.0], 0).is_err());
        assert!(richardson_lucy(&[1.0], &[0.0, 0.0], 5).is_err());
    }

    #[test]
    fn test_wiener_deconvolve_recovers_signal_noise_free() {
        // Observation = full linear convolution signal ⊛ kernel; the exact
        // (nsr=0) inverse must recover the signal to machine precision.
        let signal: Vec<f64> = (0..32)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / 32.0).sin())
            .collect();
        let kernel = [0.25_f64, 0.5, 0.25];
        let observed = convolve_full(&signal, &kernel); // length 34
        let est = wiener_deconvolve(&observed, &kernel, 0.0).expect("ok");
        for i in 0..signal.len() {
            assert!(
                (est[i] - signal[i]).abs() < 1e-9,
                "i={i}: est={} sig={}",
                est[i],
                signal[i]
            );
        }
    }

    #[test]
    fn test_wiener_deconvolve_regularised_with_noise() {
        // A blurred + noisy observation; regularised Wiener must be much closer
        // to the truth than the noisy observation itself (in the interior).
        let n = 64usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let kernel = [0.2_f64, 0.3, 0.3, 0.2];
        let full = convolve_full(&signal, &kernel);
        let noise = lcg_noise(full.len(), 4242);
        let observed: Vec<f64> = full
            .iter()
            .zip(noise.iter())
            .map(|(&v, &z)| v + 0.02 * z)
            .collect();
        let est = wiener_deconvolve(&observed, &kernel, 1e-2).expect("ok");
        let err_est: f64 = (8..n - 8)
            .map(|i| (est[i] - signal[i]).powi(2))
            .sum::<f64>();
        // Compare against the noisy *blurred* observation aligned to the signal.
        let err_obs: f64 = (8..n - 8)
            .map(|i| (observed[i] - signal[i]).powi(2))
            .sum::<f64>();
        assert!(
            err_est < err_obs,
            "deconv err {err_est} should beat observed err {err_obs}"
        );
    }

    #[test]
    fn test_richardson_lucy_nonnegative_and_flux() {
        // Two well-separated positive pulses blurred by a Gaussian PSF; R-L must
        // stay non-negative and roughly conserve total flux.
        let n = 80usize;
        let mut signal = vec![0.0_f64; n];
        signal[25] = 5.0;
        signal[55] = 3.0;
        // Unit-sum Gaussian PSF, so the blur conserves total flux and R-L can
        // be checked against the true input flux.
        let raw: Vec<f64> = (-3i32..=3)
            .map(|d| (-(d as f64).powi(2) / (2.0 * 1.2_f64.powi(2))).exp())
            .collect();
        let ksum: f64 = raw.iter().sum();
        let psf: Vec<f64> = raw.iter().map(|&v| v / ksum).collect();
        let observed = convolve_same(&signal, &psf);
        let restored = richardson_lucy(&observed, &psf, 50).expect("ok");
        assert_eq!(restored.len(), n);
        // Non-negativity.
        assert!(restored.iter().all(|&v| v >= 0.0));
        // Flux conservation (normalised PSF => sum preserved within a few %).
        let flux_in: f64 = signal.iter().sum();
        let flux_out: f64 = restored.iter().sum();
        assert!(
            (flux_out - flux_in).abs() < 0.15 * flux_in,
            "flux in={flux_in} out={flux_out}"
        );
    }

    #[test]
    fn test_richardson_lucy_localizes_peaks() {
        // After enough iterations, R-L should concentrate energy back near the
        // original pulse locations (sharper than the blurred observation).
        let n = 100usize;
        let mut signal = vec![0.0_f64; n];
        signal[40] = 10.0;
        let psf: Vec<f64> = (-4i32..=4)
            .map(|d| (-(d as f64).powi(2) / (2.0 * 1.5_f64.powi(2))).exp())
            .collect();
        let observed = convolve_same(&signal, &psf);
        let restored = richardson_lucy(&observed, &psf, 80).expect("ok");
        // The argmax of the restoration should be at (or adjacent to) index 40.
        let (mut bi, mut bv) = (0usize, restored[0]);
        for (i, &v) in restored.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        assert!((bi as i64 - 40).abs() <= 1, "peak at {bi}, expected 40");
        // Restoration peak must be sharper than the blurred observation peak.
        let obs_peak = observed.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            bv > obs_peak,
            "restored peak {bv} not sharper than {obs_peak}"
        );
    }
}
