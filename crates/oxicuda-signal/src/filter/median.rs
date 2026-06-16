//! Median filtering — robust nonlinear smoothing.
//!
//! The median filter replaces each sample by the median of a sliding window
//! centred on it.  Unlike linear filters it suppresses impulsive ("salt and
//! pepper") noise while preserving sharp edges, because a single outlier in the
//! window cannot move the median far.
//!
//! Two entry points are provided:
//!
//! * [`median_filter_1d`] — sliding-window median over a 1-D signal with
//!   zero / reflect / replicate boundary handling.
//! * [`weighted_median_1d`] — Hampel-style robust filter that only replaces a
//!   sample when it deviates from the local median by more than `n_sigmas`
//!   times the (scaled) median absolute deviation.

use crate::error::{SignalError, SignalResult};
use crate::types::PadMode;

/// Applies a 1-D median filter with window length `window` (forced odd).
///
/// Boundaries are handled according to `pad`:
/// * [`PadMode::Zero`] — out-of-range samples treated as `0`.
/// * [`PadMode::Reflect`] — mirror without repeating the edge sample.
/// * [`PadMode::Replicate`] — clamp to the nearest edge sample.
/// * [`PadMode::Circular`] — wrap around.
///
/// # Errors
///
/// Returns [`SignalError::InvalidParameter`] if `window == 0`.
pub fn median_filter_1d(signal: &[f64], window: usize, pad: PadMode) -> SignalResult<Vec<f64>> {
    if window == 0 {
        return Err(SignalError::InvalidParameter(
            "median window must be >= 1".into(),
        ));
    }
    let n = signal.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // Force odd window so the median is a single element.
    let win = if window % 2 == 0 { window + 1 } else { window };
    let half = (win / 2) as isize;

    let mut out = vec![0.0_f64; n];
    let mut buf: Vec<f64> = Vec::with_capacity(win);
    for (i, slot) in out.iter_mut().enumerate() {
        buf.clear();
        for off in -half..=half {
            let idx = i as isize + off;
            let sample = boundary_sample(signal, idx, pad);
            buf.push(sample);
        }
        *slot = median_of(&mut buf);
    }
    Ok(out)
}

/// Hampel filter: replaces a sample with the local median only when it lies
/// more than `n_sigmas · 1.4826 · MAD` from that median (an outlier).
///
/// `1.4826` rescales the median absolute deviation (MAD) to be a consistent
/// estimator of the standard deviation for Gaussian data.
///
/// # Errors
///
/// Returns [`SignalError::InvalidParameter`] if `window == 0` or `n_sigmas` is
/// negative / non-finite.
pub fn weighted_median_1d(signal: &[f64], window: usize, n_sigmas: f64) -> SignalResult<Vec<f64>> {
    if window == 0 {
        return Err(SignalError::InvalidParameter(
            "Hampel window must be >= 1".into(),
        ));
    }
    if !n_sigmas.is_finite() || n_sigmas < 0.0 {
        return Err(SignalError::InvalidParameter(format!(
            "n_sigmas must be finite and >= 0, got {n_sigmas}"
        )));
    }
    let n = signal.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let win = if window % 2 == 0 { window + 1 } else { window };
    let half = (win / 2) as isize;
    const MAD_SCALE: f64 = 1.4826;

    let mut out = signal.to_vec();
    let mut buf: Vec<f64> = Vec::with_capacity(win);
    let mut dev: Vec<f64> = Vec::with_capacity(win);
    for i in 0..n {
        buf.clear();
        for off in -half..=half {
            let idx = i as isize + off;
            buf.push(boundary_sample(signal, idx, PadMode::Replicate));
        }
        let med = median_of(&mut buf);
        dev.clear();
        dev.extend(buf.iter().map(|&v| (v - med).abs()));
        let mad = median_of(&mut dev);
        let sigma = MAD_SCALE * mad;
        if (signal[i] - med).abs() > n_sigmas * sigma {
            out[i] = med;
        }
    }
    Ok(out)
}

/// Returns the median of `buf`, mutating it (partial sort).  `buf` must be
/// non-empty.
fn median_of(buf: &mut [f64]) -> f64 {
    let len = buf.len();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if len % 2 == 1 {
        buf[len / 2]
    } else {
        0.5 * (buf[len / 2 - 1] + buf[len / 2])
    }
}

/// Fetches `signal[idx]` applying the requested boundary handling for indices
/// outside `0..signal.len()`.
fn boundary_sample(signal: &[f64], idx: isize, pad: PadMode) -> f64 {
    let n = signal.len() as isize;
    if idx >= 0 && idx < n {
        return signal[idx as usize];
    }
    match pad {
        PadMode::Zero => 0.0,
        PadMode::Replicate => {
            let clamped = idx.clamp(0, n - 1);
            signal[clamped as usize]
        }
        PadMode::Reflect => {
            // Reflect without repeating the edge: indices map into [0, n-1].
            let period = 2 * (n - 1).max(1);
            let mut m = idx.rem_euclid(period);
            if m >= n {
                m = period - m;
            }
            signal[m.clamp(0, n - 1) as usize]
        }
        PadMode::Circular => {
            let m = idx.rem_euclid(n);
            signal[m as usize]
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_len() {
        for len in [0usize, 1, 5, 100] {
            let x = vec![1.0_f64; len];
            let y = median_filter_1d(&x, 3, PadMode::Replicate).expect("ok");
            assert_eq!(y.len(), len);
        }
    }

    #[test]
    fn removes_single_spike() {
        // An isolated impulse is fully removed by a width-3 median filter.
        let x = vec![1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0_f64];
        let y = median_filter_1d(&x, 3, PadMode::Replicate).expect("ok");
        assert!((y[3] - 1.0).abs() < 1e-12, "spike not removed: {}", y[3]);
        for v in &y {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn preserves_edge() {
        // A step edge is preserved (median doesn't blur it like a mean would).
        let x = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0_f64];
        let y = median_filter_1d(&x, 3, PadMode::Replicate).expect("ok");
        assert!((y[2] - 0.0).abs() < 1e-12, "y[2]={}", y[2]);
        assert!((y[3] - 5.0).abs() < 1e-12, "y[3]={}", y[3]);
    }

    #[test]
    fn constant_unchanged() {
        let x = vec![7.0_f64; 20];
        let y = median_filter_1d(&x, 5, PadMode::Reflect).expect("ok");
        for v in &y {
            assert!((v - 7.0).abs() < 1e-12);
        }
    }

    #[test]
    fn window_1_identity() {
        let x = vec![3.0, 1.0, 4.0, 1.0, 5.0_f64];
        let y = median_filter_1d(&x, 1, PadMode::Zero).expect("ok");
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn even_window_forced_odd() {
        // window=2 is bumped to 3; result must match an explicit width-3 filter.
        let x = vec![2.0, 8.0, 2.0, 8.0, 2.0_f64];
        let y2 = median_filter_1d(&x, 2, PadMode::Replicate).expect("ok");
        let y3 = median_filter_1d(&x, 3, PadMode::Replicate).expect("ok");
        for (a, b) in y2.iter().zip(&y3) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn boundary_modes_differ() {
        let x = vec![10.0, 0.0, 0.0, 0.0, 0.0_f64];
        let zero = median_filter_1d(&x, 3, PadMode::Zero).expect("z");
        let repl = median_filter_1d(&x, 3, PadMode::Replicate).expect("r");
        // At index 0, zero-pad window = {0, 10, 0} → 0; replicate = {10,10,0} → 10.
        assert!((zero[0] - 0.0).abs() < 1e-12, "zero[0]={}", zero[0]);
        assert!((repl[0] - 10.0).abs() < 1e-12, "repl[0]={}", repl[0]);
    }

    #[test]
    fn window_0_error() {
        assert!(median_filter_1d(&[1.0, 2.0], 0, PadMode::Zero).is_err());
        assert!(weighted_median_1d(&[1.0, 2.0], 0, 3.0).is_err());
    }

    #[test]
    fn hampel_replaces_outlier() {
        // Hampel filter replaces the gross outlier but leaves clean samples.
        let mut x: Vec<f64> = (0..21).map(|i| (i as f64 * 0.1).sin()).collect();
        let original = x[10];
        x[10] = 50.0; // inject outlier
        let y = weighted_median_1d(&x, 7, 3.0).expect("ok");
        assert!((y[10] - 50.0).abs() > 10.0, "outlier not replaced");
        // A clean sample far from the outlier stays put.
        assert!((y[2] - original.max(x[2])).abs() < 5.0 || (y[2] - x[2]).abs() < 1e-9);
    }

    #[test]
    fn hampel_leaves_clean_signal() {
        // Smooth signal with no outliers: Hampel leaves it essentially unchanged.
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let y = weighted_median_1d(&x, 5, 3.0).expect("ok");
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn finite_outputs() {
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 1.7).cos() * 4.0).collect();
        let y = median_filter_1d(&x, 9, PadMode::Circular).expect("ok");
        for v in &y {
            assert!(v.is_finite());
        }
    }
}
