//! Harmonic/Percussive Source Separation (HPSS) via median filtering.
//!
//! Implements the Fitzgerald 2010 DAFX algorithm: apply a horizontal median
//! filter along the time axis (captures stable harmonic content) and a vertical
//! median filter along the frequency axis (captures transient percussive
//! content), then form Wiener soft masks or binary hard masks.
//!
//! ## References
//! - Fitzgerald, D. (2010). "Harmonic/Percussive Separation using Median
//!   Filtering." Proc. DAFX.
//! - Ono, N. et al. (2008). "Separation of a monaural audio signal into harmonic
//!   /percussive components by complementary diffusion on spectrogram." EUSIPCO.

use crate::error::{AudioError, AudioResult};

// ─── Public types ────────────────────────────────────────────────────────────

/// Mask type used to combine the harmonic and percussive estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpssMask {
    /// Wiener soft mask: `M_H = H^p / (H^p + P^p)`.
    Soft,
    /// Binary hard mask: `M_H = 1` where `H >= P`, else `0`.
    Binary,
}

/// Configuration for [`hpss`] and [`hpss_masks`].
#[derive(Debug, Clone)]
pub struct HpssConfig {
    /// Median-filter window length along the **time** axis (odd; even values are
    /// rounded up to the next odd number). Captures harmonic content.
    pub harmonic_window: usize,
    /// Median-filter window length along the **frequency** axis (odd; even
    /// values are rounded up). Captures percussive transients.
    pub percussive_window: usize,
    /// Which mask type to apply when building the separated spectrograms.
    pub mask: HpssMask,
    /// Exponent for the Wiener soft mask formula (`power >= 0`).
    pub power: f64,
}

impl Default for HpssConfig {
    fn default() -> Self {
        Self {
            harmonic_window: 31,
            percussive_window: 31,
            mask: HpssMask::Soft,
            power: 2.0,
        }
    }
}

/// Result of [`hpss`]: separated magnitude spectrograms.
#[derive(Debug, Clone)]
pub struct HpssResult {
    /// Harmonic component `[n_frames × n_bins]`, row-major.
    pub harmonic: Vec<f64>,
    /// Percussive component `[n_frames × n_bins]`, row-major.
    pub percussive: Vec<f64>,
    /// Number of time frames.
    pub n_frames: usize,
    /// Number of frequency bins.
    pub n_bins: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Round `w` up to the nearest odd integer (>= 1).
fn to_odd(w: usize) -> usize {
    let w = w.max(1);
    if w % 2 == 0 { w + 1 } else { w }
}

/// Reflect-padded index into a sequence of length `n`.
///
/// For `i < 0` uses `|i|`; for `i >= n` uses `2*(n-1) - i`.
fn reflect_idx(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut idx = i;
    if idx < 0 {
        idx = -idx;
    }
    if idx >= n {
        idx = 2 * (n - 1) - idx;
    }
    // Clamp defensively for very small n or large windows
    idx.clamp(0, n - 1) as usize
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// 1-D median filter with **reflect** boundary padding.
///
/// `window` is rounded up to the nearest odd integer >= 1.
/// Returns a vector of the same length as `signal`.
#[must_use]
pub fn median_filter_1d(signal: &[f64], window: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let w = to_odd(window);
    let half = (w / 2) as isize;

    let mut output = vec![0.0_f64; n];
    let mut buf = vec![0.0_f64; w];

    for (i, out) in output.iter_mut().enumerate() {
        let i = i as isize;
        for (j, b) in buf.iter_mut().enumerate() {
            let offset = j as isize - half;
            *b = signal[reflect_idx(i + offset, n)];
        }
        // Insertion sort (fast for small w, typically <= 63)
        for k in 1..w {
            let key = buf[k];
            let mut m = k;
            while m > 0 && buf[m - 1] > key {
                buf[m] = buf[m - 1];
                m -= 1;
            }
            buf[m] = key;
        }
        *out = buf[w / 2];
    }
    output
}

/// Compute harmonic and percussive **masks** from a magnitude spectrogram.
///
/// Returns `(M_H, M_P)` each of shape `[n_frames × n_bins]`, row-major.
///
/// # Errors
/// Returns [`AudioError`] if dimensions are invalid or `power < 0`.
pub fn hpss_masks(
    magnitude: &[f64],
    n_frames: usize,
    n_bins: usize,
    config: &HpssConfig,
) -> AudioResult<(Vec<f64>, Vec<f64>)> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n_frames == 0 || n_bins == 0 {
        return Err(AudioError::EmptyInput {
            msg: format!("n_frames={n_frames}, n_bins={n_bins} — both must be > 0"),
        });
    }
    let expected = n_frames * n_bins;
    if magnitude.len() != expected {
        return Err(AudioError::DimensionMismatch {
            expected,
            got: magnitude.len(),
        });
    }
    if config.power < 0.0 {
        return Err(AudioError::NonFinite {
            msg: format!("power must be >= 0, got {}", config.power),
        });
    }

    let hw = to_odd(config.harmonic_window);
    let pw = to_odd(config.percussive_window);

    // ── Harmonic: median along TIME axis for each frequency bin ───────────────
    // h_mat[t, f] = median along time of column f
    let mut h_mat = vec![0.0_f64; n_frames * n_bins];
    {
        let mut col = vec![0.0_f64; n_frames];
        for f in 0..n_bins {
            for (t, c) in col.iter_mut().enumerate() {
                *c = magnitude[t * n_bins + f];
            }
            let filtered = median_filter_1d(&col, hw);
            for (t, &v) in filtered.iter().enumerate() {
                h_mat[t * n_bins + f] = v;
            }
        }
    }

    // ── Percussive: median along FREQUENCY axis for each time frame ───────────
    // p_mat[t, f] = median along freq of row t
    let mut p_mat = vec![0.0_f64; n_frames * n_bins];
    for t in 0..n_frames {
        let row = &magnitude[t * n_bins..(t + 1) * n_bins];
        let filtered = median_filter_1d(row, pw);
        p_mat[t * n_bins..(t + 1) * n_bins].copy_from_slice(&filtered);
    }

    // ── Masks ─────────────────────────────────────────────────────────────────
    let len = n_frames * n_bins;
    let mut m_h = vec![0.0_f64; len];
    let mut m_p = vec![0.0_f64; len];

    match config.mask {
        HpssMask::Soft => {
            let p = config.power;
            for i in 0..len {
                let h = h_mat[i];
                let pv = p_mat[i];
                if h == 0.0 && pv == 0.0 {
                    m_h[i] = 0.5;
                    m_p[i] = 0.5;
                } else {
                    let hp = h.powf(p);
                    let pp = pv.powf(p);
                    let denom = hp + pp;
                    m_h[i] = hp / denom;
                    m_p[i] = pp / denom;
                }
            }
        }
        HpssMask::Binary => {
            for i in 0..len {
                if h_mat[i] >= p_mat[i] {
                    m_h[i] = 1.0;
                    m_p[i] = 0.0;
                } else {
                    m_h[i] = 0.0;
                    m_p[i] = 1.0;
                }
            }
        }
    }

    Ok((m_h, m_p))
}

/// Harmonic/Percussive Source Separation on a magnitude spectrogram.
///
/// `magnitude` must be row-major with shape `[n_frames × n_bins]`.
///
/// # Errors
/// Returns [`AudioError`] if dimensions are invalid or `config.power < 0`.
pub fn hpss(
    magnitude: &[f64],
    n_frames: usize,
    n_bins: usize,
    config: &HpssConfig,
) -> AudioResult<HpssResult> {
    let (m_h, m_p) = hpss_masks(magnitude, n_frames, n_bins, config)?;

    let len = n_frames * n_bins;
    let mut harmonic = vec![0.0_f64; len];
    let mut percussive = vec![0.0_f64; len];

    for i in 0..len {
        harmonic[i] = m_h[i] * magnitude[i];
        percussive[i] = m_p[i] * magnitude[i];
    }

    Ok(HpssResult {
        harmonic,
        percussive,
        n_frames,
        n_bins,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers
    fn default_soft() -> HpssConfig {
        HpssConfig::default()
    }

    fn binary_cfg() -> HpssConfig {
        HpssConfig {
            mask: HpssMask::Binary,
            ..HpssConfig::default()
        }
    }

    // ── median_filter_1d ──────────────────────────────────────────────────────

    #[test]
    fn median_filter_removes_spike() {
        let sig = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0];
        let out = median_filter_1d(&sig, 3);
        assert_eq!(out.len(), 7);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-12, "spike not removed: {v}");
        }
    }

    #[test]
    fn median_filter_constant_signal() {
        let sig = vec![5.0_f64; 20];
        let out = median_filter_1d(&sig, 7);
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn median_filter_window_one() {
        let sig = [3.0, 1.0, 4.0, 1.0, 5.0];
        let out = median_filter_1d(&sig, 1);
        assert_eq!(out, sig);
    }

    #[test]
    fn median_filter_even_window_rounded_to_odd() {
        // window=4 -> odd 5; should not panic
        let sig: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let out = median_filter_1d(&sig, 4);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn median_filter_window_larger_than_signal() {
        let sig = [2.0, 4.0, 6.0];
        let out = median_filter_1d(&sig, 11);
        assert_eq!(out.len(), 3);
        // Each window includes all 3 elements (with reflections), median of some multiset
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn median_filter_empty_signal() {
        let out = median_filter_1d(&[], 5);
        assert!(out.is_empty());
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn hpss_error_empty_frames() {
        let cfg = default_soft();
        let err = hpss(&[], 0, 4, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput { .. }));
    }

    #[test]
    fn hpss_error_empty_bins() {
        let cfg = default_soft();
        let err = hpss(&[], 4, 0, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput { .. }));
    }

    #[test]
    fn hpss_error_dimension_mismatch() {
        let cfg = default_soft();
        let mag = vec![1.0_f64; 10];
        let err = hpss(&mag, 3, 4, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    #[test]
    fn hpss_error_negative_power() {
        let cfg = HpssConfig {
            power: -1.0,
            ..HpssConfig::default()
        };
        let mag = vec![1.0_f64; 12];
        let err = hpss(&mag, 3, 4, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::NonFinite { .. }));
    }

    // ── Soft mask properties ──────────────────────────────────────────────────

    #[test]
    fn soft_masks_sum_to_one() {
        let n_frames = 8;
        let n_bins = 6;
        let mag: Vec<f64> = (0..n_frames * n_bins)
            .map(|i| ((i as f64) * 0.3 + 1.0).sin().abs() + 0.1)
            .collect();
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..default_soft()
        };
        let (m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        for i in 0..n_frames * n_bins {
            let sum = m_h[i] + m_p[i];
            assert!((sum - 1.0).abs() < 1e-10, "mask sum={sum} at i={i}");
        }
    }

    #[test]
    fn binary_masks_partition_exactly() {
        let n_frames = 6;
        let n_bins = 5;
        let mag: Vec<f64> = (0..n_frames * n_bins).map(|i| i as f64 + 1.0).collect();
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..binary_cfg()
        };
        let (m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        for i in 0..n_frames * n_bins {
            let sum = m_h[i] + m_p[i];
            assert!((sum - 1.0).abs() < 1e-14, "binary sum={sum} at i={i}");
            assert!(m_h[i] == 0.0 || m_h[i] == 1.0);
        }
    }

    #[test]
    fn soft_result_partitions_magnitude() {
        let n_frames = 5;
        let n_bins = 4;
        let mag: Vec<f64> = (0..n_frames * n_bins).map(|i| i as f64 + 1.0).collect();
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..default_soft()
        };
        let res = hpss(&mag, n_frames, n_bins, &cfg).expect("hpss should succeed");
        assert_eq!(res.n_frames, n_frames);
        assert_eq!(res.n_bins, n_bins);
        for (i, ((&h, &p), &m)) in res
            .harmonic
            .iter()
            .zip(res.percussive.iter())
            .zip(mag.iter())
            .enumerate()
        {
            let reconstructed = h + p;
            assert!(
                (reconstructed - m).abs() < 1e-10,
                "partition failed at i={i}: {reconstructed} != {m}"
            );
        }
    }

    #[test]
    fn harmonic_dominates_constant_frequency_band() {
        // Build spectrogram: row f=2 is all 1.0, all other rows zero.
        let n_frames = 16;
        let n_bins = 8;
        let mut mag = vec![0.0_f64; n_frames * n_bins];
        for t in 0..n_frames {
            mag[t * n_bins + 2] = 1.0;
        }
        let cfg = HpssConfig {
            harmonic_window: 7,
            percussive_window: 7,
            ..default_soft()
        };
        let (m_h, _m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        // At the constant band (f=2) the harmonic mask should dominate
        let mid = n_frames / 2;
        let mh_val = m_h[mid * n_bins + 2];
        assert!(
            mh_val > 0.6,
            "harmonic mask at constant band should be > 0.6, got {mh_val}"
        );
    }

    #[test]
    fn percussive_dominates_single_frame_spike() {
        // Spectrogram: frame t=8 has all bins = 1.0, others are zero.
        let n_frames = 16;
        let n_bins = 8;
        let mut mag = vec![0.0_f64; n_frames * n_bins];
        let spike_t = 8;
        for f in 0..n_bins {
            mag[spike_t * n_bins + f] = 1.0;
        }
        let cfg = HpssConfig {
            harmonic_window: 7,
            percussive_window: 7,
            ..default_soft()
        };
        let (_m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        // At the spike frame, percussive mask should dominate
        let mid_bin = n_bins / 2;
        let mp_val = m_p[spike_t * n_bins + mid_bin];
        assert!(
            mp_val > 0.6,
            "percussive mask at spike frame should be > 0.6, got {mp_val}"
        );
    }

    #[test]
    fn zero_magnitude_gives_half_soft_masks() {
        let n_frames = 4;
        let n_bins = 4;
        let mag = vec![0.0_f64; n_frames * n_bins];
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..default_soft()
        };
        let (m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        for i in 0..n_frames * n_bins {
            assert!((m_h[i] - 0.5).abs() < 1e-14, "m_h[{i}]={}", m_h[i]);
            assert!((m_p[i] - 0.5).abs() < 1e-14, "m_p[{i}]={}", m_p[i]);
        }
    }

    #[test]
    fn zero_magnitude_hpss_result_is_zero() {
        let n_frames = 4;
        let n_bins = 4;
        let mag = vec![0.0_f64; n_frames * n_bins];
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..default_soft()
        };
        let res = hpss(&mag, n_frames, n_bins, &cfg).expect("hpss should succeed");
        assert!(res.harmonic.iter().all(|&v| v == 0.0));
        assert!(res.percussive.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn power_zero_gives_equal_masks() {
        // With p=0: H^0 = 1 (when H>0), P^0 = 1, so masks are both 0.5
        // Corner: when both H,P > 0: each mask = 0.5
        let n_frames = 4;
        let n_bins = 4;
        let mag: Vec<f64> = (0..n_frames * n_bins).map(|i| i as f64 + 1.0).collect();
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            power: 0.0,
            mask: HpssMask::Soft,
        };
        let (m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        for i in 0..n_frames * n_bins {
            let sum = m_h[i] + m_p[i];
            assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
        }
    }

    #[test]
    fn hpss_single_frame_single_bin() {
        let mag = vec![3.0_f64];
        let cfg = HpssConfig {
            harmonic_window: 1,
            percussive_window: 1,
            ..default_soft()
        };
        let res = hpss(&mag, 1, 1, &cfg).expect("hpss should succeed");
        assert_eq!(res.harmonic.len(), 1);
        assert_eq!(res.percussive.len(), 1);
        let total = res.harmonic[0] + res.percussive[0];
        assert!((total - 3.0).abs() < 1e-10, "total={total}");
    }

    #[test]
    fn binary_result_partitions_magnitude() {
        let n_frames = 6;
        let n_bins = 5;
        let mag: Vec<f64> = (0..n_frames * n_bins)
            .map(|i| (i as f64).sin().abs() + 0.5)
            .collect();
        let cfg = HpssConfig {
            harmonic_window: 3,
            percussive_window: 3,
            ..binary_cfg()
        };
        let res = hpss(&mag, n_frames, n_bins, &cfg).expect("hpss should succeed");
        for (i, ((&h, &p), &m)) in res
            .harmonic
            .iter()
            .zip(res.percussive.iter())
            .zip(mag.iter())
            .enumerate()
        {
            let total = h + p;
            assert!((total - m).abs() < 1e-12, "i={i}: {total} != {m}");
        }
    }

    #[test]
    fn output_shapes_correct() {
        let n_frames = 10;
        let n_bins = 6;
        let mag = vec![1.0_f64; n_frames * n_bins];
        let cfg = default_soft();
        let res = hpss(&mag, n_frames, n_bins, &cfg).expect("hpss should succeed");
        assert_eq!(res.harmonic.len(), n_frames * n_bins);
        assert_eq!(res.percussive.len(), n_frames * n_bins);
        assert_eq!(res.n_frames, n_frames);
        assert_eq!(res.n_bins, n_bins);
    }

    #[test]
    fn soft_masks_all_finite() {
        let n_frames = 12;
        let n_bins = 10;
        let mag: Vec<f64> = (0..n_frames * n_bins)
            .map(|i| ((i as f64) * 0.1).sin().powi(2))
            .collect();
        let cfg = HpssConfig {
            harmonic_window: 5,
            percussive_window: 5,
            ..default_soft()
        };
        let (m_h, m_p) =
            hpss_masks(&mag, n_frames, n_bins, &cfg).expect("hpss_masks should succeed");
        assert!(m_h.iter().all(|v| v.is_finite()));
        assert!(m_p.iter().all(|v| v.is_finite()));
    }
}
