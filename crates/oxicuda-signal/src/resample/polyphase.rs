//! Polyphase rational resampling (`scipy.signal.resample_poly` semantics).
//!
//! Resampling a signal by the rational factor `up / down` conceptually
//! proceeds in three steps:
//!
//! 1. **Upsample** by `up` — insert `up − 1` zeros between samples.
//! 2. **Anti-alias / interpolation lowpass** — convolve with an FIR filter of
//!    cutoff `0.5 / max(up, down)` (normalised) and gain `up` so that the
//!    inserted spectral images are removed and the surviving spectrum keeps
//!    unit amplitude.
//! 3. **Downsample** by `down` — keep every `down`-th sample.
//!
//! Materialising the zero-stuffed signal is wasteful (a factor `up` more
//! samples, most of them zero).  Instead this module uses the classic
//! **polyphase decomposition**: the prototype lowpass `h` of length `M` is
//! split into `up` sub-filters (phases) `h_p[m] = h[p + m·up]`, and each output
//! sample selects exactly one phase and accumulates it against the *original*
//! (non-stuffed) input samples.  This computes the same result as the naive
//! pipeline at `1 / up` of the cost.
//!
//! The prototype FIR is designed with the crate's own Kaiser-windowed-sinc
//! routine ([`crate::filter::design_lowpass`] + a Kaiser window), so no new
//! design machinery is introduced.

use crate::{
    error::{SignalError, SignalResult},
    filter::design_lowpass,
    types::WindowType,
};

/// Greatest common divisor (Euclid).
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Default Kaiser β shape parameter for the anti-alias prototype.
const DEFAULT_BETA: f64 = 5.0;

/// Design the anti-alias / interpolation prototype lowpass of length
/// `num_taps` (odd), cutoff `0.5/max(up,down)` normalised, gain `up`, using a
/// Kaiser window of shape `beta`.
fn design_prototype(up: usize, down: usize, num_taps: usize, beta: f64) -> SignalResult<Vec<f64>> {
    let max_rate = up.max(down);
    // `design_lowpass` cutoff is normalised so that 1.0 == Nyquist (π rad).  The
    // half-band cutoff in cycles/sample is 0.5/max_rate; as a fraction of
    // Nyquist that is (0.5/max_rate)/0.5 = 1/max_rate.
    let cutoff = 1.0 / max_rate as f64;
    let window = crate::audio::stft::make_window(num_taps, WindowType::Kaiser { beta });
    let mut h = design_lowpass(num_taps, cutoff, &window)?;
    // Apply the interpolation gain `up` so the passband amplitude is preserved
    // after discarding `up − 1` of every `up` samples.
    let gain = up as f64;
    for v in h.iter_mut() {
        *v *= gain;
    }
    Ok(h)
}

/// Resample `x` by the rational factor `up / down` using polyphase filtering.
///
/// # Parameters
/// - `up`, `down` — resampling ratio (reduced internally by their GCD); both
///   must be `≥ 1`.
/// - `num_taps` — optional prototype length; default `10·max(up,down) + 1`
///   (forced odd).  Larger values sharpen the anti-alias transition.
/// - `beta` — optional Kaiser β shape parameter; default `5.0`.
///
/// The output length is `ceil(len(x) · up / down)`, and the filter's group
/// delay `(M − 1)/2` is compensated so the output aligns in time with the
/// input.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if `up == 0` or `down == 0`.
pub fn resample_poly(
    x: &[f32],
    up: usize,
    down: usize,
    num_taps: Option<usize>,
    beta: Option<f64>,
) -> SignalResult<Vec<f32>> {
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidParameter(format!(
            "resample factors must be ≥ 1 (up={up}, down={down})"
        )));
    }

    // Reduce the ratio.
    let g = gcd(up, down);
    let up = up / g;
    let down = down / g;

    // Output length is defined on the *original* sample count.
    let out_len = x.len().saturating_mul(up).div_ceil(down);

    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }

    // Prototype filter.
    let max_rate = up.max(down);
    let n_taps = num_taps.unwrap_or(10 * max_rate + 1);
    // Force odd length for a symmetric (linear-phase) prototype.
    let n_taps = if n_taps % 2 == 0 { n_taps + 1 } else { n_taps };
    let n_taps = n_taps.max(3);
    let beta = beta.unwrap_or(DEFAULT_BETA);
    let h = design_prototype(up, down, n_taps, beta)?;
    let m = h.len();

    // Group delay of the prototype (samples, in the *upsampled* grid).
    let half_delay = (m - 1) / 2;

    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let n_in = xf.len();

    // Polyphase resampling via the commutator.  For output index `j`:
    //   * the corresponding position in the upsampled+filtered+delay-compensated
    //     stream is  t = j·down + half_delay  (index into the zero-stuffed,
    //     length n_in·up signal — but we never build it);
    //   * the active polyphase phase is  p = t mod up;
    //   * the convolution sum runs over the prototype taps of that phase against
    //     the original input samples  x[n_in_idx]  where
    //         n_in_idx = (t − (p + r·up)) / up = floor(t/up) − r,   r = 0,1,2,…
    //
    // Concretely: y[j] = Σ_r h[p + r·up] · x[(t/up) − r], for valid indices.
    let mut out = vec![0.0_f64; out_len];
    for (j, yj) in out.iter_mut().enumerate() {
        let t = j * down + half_delay;
        let phase = t % up;
        let base_in = t / up; // floor; index of the most-recent input sample
        let mut acc = 0.0_f64;
        // Taps of this phase: r = 0, 1, …, while (phase + r·up) < m.
        let mut r = 0usize;
        loop {
            let tap = phase + r * up;
            if tap >= m {
                break;
            }
            // Input index: base_in − r.
            let in_idx = base_in as isize - r as isize;
            if in_idx >= 0 && (in_idx as usize) < n_in {
                acc += h[tap] * xf[in_idx as usize];
            }
            r += 1;
        }
        *yj = acc;
    }

    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// Resample `x` from `in_rate` to `out_rate` by reducing the rate ratio to a
/// rational `up / down` via the GCD of the (rounded) rates.
///
/// # Errors
/// Returns [`SignalError::InvalidParameter`] if either rate is non-finite,
/// non-positive, or rounds to zero.
pub fn resample_rate(x: &[f32], in_rate: f64, out_rate: f64) -> SignalResult<Vec<f32>> {
    if !(in_rate.is_finite() && out_rate.is_finite() && in_rate > 0.0 && out_rate > 0.0) {
        return Err(SignalError::InvalidParameter(format!(
            "resample rates must be finite and > 0 (in={in_rate}, out={out_rate})"
        )));
    }
    // Express the ratio as integers.  Round to the nearest Hz-level integer;
    // for fractional rates a common scale keeps the ratio faithful.
    let scale = 1_000.0_f64;
    let up = (out_rate * scale).round() as usize;
    let down = (in_rate * scale).round() as usize;
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidParameter(format!(
            "resample rates round to zero (in={in_rate}, out={out_rate})"
        )));
    }
    let g = gcd(up, down);
    resample_poly(x, up / g, down / g, None, None)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| (v as f64) * (v as f64)).sum()
    }

    #[test]
    fn test_output_length_various_ratios() {
        let n = 100;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        for (up, down) in [(2, 1), (1, 2), (3, 2), (2, 3), (5, 4), (4, 7)] {
            let y = resample_poly(&x, up, down, None, None).expect("resample");
            let expected = (n * up).div_ceil(down);
            assert_eq!(y.len(), expected, "len for up={up} down={down}");
        }
    }

    #[test]
    fn test_upsample_by_two_length_and_tone() {
        // A low-frequency sinusoid upsampled by 2 should double in length and
        // the interior samples should track the analytic upsampled tone.
        let fs = 100.0;
        let f0 = 5.0;
        let n = 200;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin() as f32)
            .collect();
        let y = resample_poly(&x, 2, 1, None, None).expect("resample");
        assert_eq!(y.len(), 2 * n);

        // New sample rate is 200 Hz; compare the interior (away from edges)
        // against the ideal tone, allowing for filter ripple.
        let fs2 = 200.0;
        let mut max_err = 0.0_f64;
        for (i, &yi) in y.iter().enumerate().take(2 * n - 80).skip(80) {
            let ideal = (2.0 * PI * f0 * i as f64 / fs2).sin();
            max_err = max_err.max((yi as f64 - ideal).abs());
        }
        assert!(max_err < 0.1, "upsample tone interior error {max_err}");
    }

    #[test]
    fn test_downsample_by_two_halves_length() {
        let n = 200usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
        let y = resample_poly(&x, 1, 2, None, None).expect("resample");
        assert_eq!(y.len(), n.div_ceil(2));
    }

    #[test]
    fn test_dc_gain_preserved() {
        // A constant (DC) input must come out as the same constant, away from
        // the warm-up/cool-down edges where the FIR has not fully overlapped.
        let n = 300;
        let c = 2.5_f32;
        let x = vec![c; n];
        for (up, down) in [(2, 1), (1, 2), (3, 2), (2, 3)] {
            let y = resample_poly(&x, up, down, None, None).expect("resample");
            let lo = y.len() / 4;
            let hi = y.len() - y.len() / 4;
            for v in &y[lo..hi] {
                assert!(
                    (*v - c).abs() < 0.05,
                    "DC not preserved (up={up} down={down}): {v} != {c}"
                );
            }
        }
    }

    #[test]
    fn test_antialias_attenuates_high_tone() {
        // A tone above the post-decimation Nyquist must be strongly attenuated.
        // Downsample by 4: new Nyquist = fs/8.  Put a tone well above it.
        let fs = 8000.0;
        let n = 4000;
        let f_high = 2800.0; // > new Nyquist (1000 Hz) ⇒ should be killed
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f_high * i as f64 / fs).sin() as f32)
            .collect();
        let y = resample_poly(&x, 1, 4, None, None).expect("resample");

        // Compare interior energy per sample of input vs output.
        let in_e = energy(&x[n / 4..3 * n / 4]) / (n / 2) as f64;
        let lo = y.len() / 4;
        let hi = y.len() - y.len() / 4;
        let out_e = energy(&y[lo..hi]) / (hi - lo) as f64;
        assert!(
            out_e < 0.1 * in_e,
            "high tone not attenuated: in/sample={in_e}, out/sample={out_e}"
        );
    }

    #[test]
    fn test_passband_tone_survives_downsample() {
        // A tone safely below the new Nyquist should survive downsampling with
        // its amplitude roughly intact.
        let fs = 8000.0;
        let n = 4000;
        let f_low = 300.0; // << new Nyquist (1000 Hz)
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f_low * i as f64 / fs).sin() as f32)
            .collect();
        let y = resample_poly(&x, 1, 4, None, None).expect("resample");
        let lo = y.len() / 4;
        let hi = y.len() - y.len() / 4;
        let out_e = energy(&y[lo..hi]) / (hi - lo) as f64;
        // RMS of a unit sine is 1/√2 ⇒ mean square 0.5.
        assert!(
            (out_e - 0.5).abs() < 0.1,
            "passband tone energy/sample {out_e} != ~0.5"
        );
    }

    #[test]
    fn test_identity_when_up_equals_down() {
        let x: Vec<f32> = (0..50).map(|i| (i as f32 * 0.3).sin()).collect();
        let y = resample_poly(&x, 3, 3, None, None).expect("resample");
        assert_eq!(y.len(), x.len());
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-6, "identity broken: {a} vs {b}");
        }
    }

    #[test]
    fn test_ratio_reduced_by_gcd() {
        // up=4, down=2 reduces to 2/1: output length should match the reduced
        // ratio (2N), not 4N/2 (which numerically equals 2N anyway) — and the
        // result should match a direct 2/1 call exactly.
        let x: Vec<f32> = (0..60).map(|i| (i as f32 * 0.2).cos()).collect();
        let y_reduced = resample_poly(&x, 2, 1, None, None).expect("2/1");
        let y_raw = resample_poly(&x, 4, 2, None, None).expect("4/2");
        assert_eq!(y_reduced.len(), y_raw.len());
        for (a, b) in y_reduced.iter().zip(y_raw.iter()) {
            assert!((a - b).abs() < 1e-6, "gcd reduction mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_resample_rate_helper() {
        // 48000 → 16000 is exactly 1/3.
        let n = 600usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = resample_rate(&x, 48_000.0, 16_000.0).expect("resample_rate");
        assert_eq!(y.len(), (n).div_ceil(3));
    }

    #[test]
    fn test_resample_empty_input() {
        let y = resample_poly(&[], 3, 2, None, None).expect("empty resample");
        assert!(y.is_empty());
    }

    #[test]
    fn test_resample_zero_factor_error() {
        let x = vec![1.0_f32; 10];
        assert!(resample_poly(&x, 0, 2, None, None).is_err());
        assert!(resample_poly(&x, 2, 0, None, None).is_err());
    }

    #[test]
    fn test_resample_rate_invalid_error() {
        let x = vec![1.0_f32; 10];
        assert!(resample_rate(&x, 0.0, 16_000.0).is_err());
        assert!(resample_rate(&x, 48_000.0, -1.0).is_err());
        assert!(resample_rate(&x, f64::NAN, 16_000.0).is_err());
    }

    #[test]
    fn test_custom_taps_and_beta() {
        // Explicit (even) tap count is bumped to odd; design still succeeds and
        // preserves DC.
        let x = vec![1.0_f32; 200];
        let y = resample_poly(&x, 3, 2, Some(64), Some(8.0)).expect("resample");
        let lo = y.len() / 4;
        let hi = y.len() - y.len() / 4;
        for v in &y[lo..hi] {
            assert!(
                (*v - 1.0).abs() < 0.05,
                "DC not preserved with custom taps: {v}"
            );
        }
    }
}
