//! Constant-Q Transform (CQT) — Brown 1991.
//!
//! The CQT provides a time-frequency representation with logarithmically spaced
//! frequency bins, making it particularly well-suited for music analysis where
//! notes are related by multiplicative frequency ratios (octaves).
//!
//! ## Core Parameters
//!
//! ```text
//! Q = 1 / (2^(1/B) - 1)                         B = bins_per_octave
//! f_k = f_min · 2^(k/B)                          k = 0, 1, …, K-1
//! N_k = ⌈Q · f_s / f_k⌉                          window length at bin k
//! ```
//!
//! ## CQT Kernel
//!
//! ```text
//! ψ_k[n] = w_k[n] · exp(-j · 2π · Q · n / N_k) / N_k
//! ```
//!
//! where `w_k` is the analysis window of length `N_k`.
//!
//! ## Output Layout
//!
//! The flat output vector stores interleaved `[Re, Im]` pairs in frame-major
//! order:
//!
//! ```text
//! out[frame * K * 2 + k * 2 + 0] = Re{X[frame, k]}
//! out[frame * K * 2 + k * 2 + 1] = Im{X[frame, k]}
//! ```
//!
//! ## Reference
//!
//! Brown, J. C. (1991). "Calculation of a constant Q spectral transform."
//! *Journal of the Acoustical Society of America*, 89(1), 425–434.

use crate::audio::stft::make_window;
use crate::error::{SignalError, SignalResult};
use crate::types::WindowType;
use std::f64::consts::PI;

// --------------------------------------------------------------------------- //
//  CQT configuration
// --------------------------------------------------------------------------- //

/// Configuration for the Constant-Q Transform.
#[derive(Debug, Clone)]
pub struct CqtConfig {
    /// Sample rate of the input signal in Hz.
    pub sample_rate: f64,
    /// Minimum frequency (centre of bin 0) in Hz.
    pub f_min: f64,
    /// Total number of CQT bins.
    pub n_bins: usize,
    /// Number of bins per octave (e.g. 12 for semitone resolution).
    pub bins_per_octave: usize,
    /// Hop length in samples between successive frames.
    pub hop_length: usize,
    /// Window function applied to each analysis frame.
    pub window: WindowType,
}

impl CqtConfig {
    /// Create a new CQT configuration and validate all parameters.
    ///
    /// # Errors
    /// - `InvalidParameter` — `sample_rate <= 0`, `f_min <= 0`, `bins_per_octave == 0`,
    ///   `hop_length == 0`
    /// - `InvalidSize`       — `n_bins == 0`
    pub fn new(
        sample_rate: f64,
        f_min: f64,
        n_bins: usize,
        bins_per_octave: usize,
        hop_length: usize,
        window: WindowType,
    ) -> SignalResult<Self> {
        if sample_rate <= 0.0_f64 {
            return Err(SignalError::InvalidParameter(format!(
                "sample_rate ({sample_rate}) must be > 0"
            )));
        }
        if f_min <= 0.0_f64 {
            return Err(SignalError::InvalidParameter(format!(
                "f_min ({f_min}) must be > 0"
            )));
        }
        if n_bins == 0 {
            return Err(SignalError::InvalidSize("n_bins must be ≥ 1".to_owned()));
        }
        if bins_per_octave == 0 {
            return Err(SignalError::InvalidParameter(
                "bins_per_octave must be ≥ 1".to_owned(),
            ));
        }
        if hop_length == 0 {
            return Err(SignalError::InvalidParameter(
                "hop_length must be ≥ 1".to_owned(),
            ));
        }
        Ok(Self {
            sample_rate,
            f_min,
            n_bins,
            bins_per_octave,
            hop_length,
            window,
        })
    }

    /// Compute the constant Q factor: `Q = 1 / (2^(1/B) - 1)`.
    #[must_use]
    pub fn q_factor(&self) -> f64 {
        1.0_f64 / (2.0_f64.powf(1.0_f64 / self.bins_per_octave as f64) - 1.0_f64)
    }

    /// Centre frequency of bin `k`: `f_k = f_min · 2^(k / B)`.
    #[must_use]
    pub fn freq_at_bin(&self, k: usize) -> f64 {
        self.f_min * 2.0_f64.powf(k as f64 / self.bins_per_octave as f64)
    }

    /// Analysis window length at bin `k`: `N_k = ⌈Q · f_s / f_k⌉`.
    ///
    /// Always at least 1 to avoid degenerate zero-length windows.
    #[must_use]
    pub fn window_length_at_bin(&self, k: usize) -> usize {
        let f_k = self.freq_at_bin(k);
        let q = self.q_factor();
        ((q * self.sample_rate / f_k).ceil() as usize).max(1)
    }

    /// Total number of CQT frequency bins (same as `self.n_bins`).
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.n_bins
    }

    /// Number of analysis frames for a signal of `n_samples` samples.
    ///
    /// Returns 0 for empty signals; otherwise `1 + (n_samples - 1) / hop_length`.
    #[must_use]
    pub fn num_frames(&self, n_samples: usize) -> usize {
        if n_samples == 0 {
            0
        } else {
            1 + (n_samples - 1) / self.hop_length
        }
    }
}

// --------------------------------------------------------------------------- //
//  CPU reference CQT (O(N²) DFT, no FFT)
// --------------------------------------------------------------------------- //

/// Compute the Constant-Q Transform via a direct (O(N²)) DFT kernel.
///
/// This is the CPU reference implementation: no FFT is used.  Each bin k at
/// each frame applies the CQT kernel `ψ_k` directly by summing over the N_k
/// windowed samples.
///
/// Output is flat interleaved `[Re, Im]` in frame-major, bin-minor order.
/// Total length = `2 * num_frames(signal.len()) * n_bins`.
///
/// # Errors
/// Returns `SignalError::InvalidSize` when `n_bins == 0` (already caught by
/// [`CqtConfig::new`], but guarded defensively here as well).
pub fn cqt_reference(signal: &[f64], config: &CqtConfig) -> SignalResult<Vec<f64>> {
    if config.n_bins == 0 {
        return Err(SignalError::InvalidSize("n_bins must be ≥ 1".to_owned()));
    }

    let n_samples = signal.len();
    let num_frames = config.num_frames(n_samples);
    let n_bins = config.n_bins;
    let hop_length = config.hop_length;
    let q = config.q_factor();

    let mut out = vec![0.0_f64; num_frames * n_bins * 2];

    for frame in 0..num_frames {
        for k in 0..n_bins {
            let nk = config.window_length_at_bin(k);
            let window = make_window(nk, config.window);
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;

            for (n, &win_coef) in window.iter().enumerate() {
                let src_idx = frame * hop_length + n;
                let x_val = if src_idx < signal.len() {
                    signal[src_idx]
                } else {
                    0.0_f64
                };
                let angle = -2.0_f64 * PI * q * n as f64 / nk as f64;
                re += x_val * win_coef * angle.cos();
                im += x_val * win_coef * angle.sin();
            }

            re /= nk as f64;
            im /= nk as f64;

            out[frame * n_bins * 2 + k * 2] = re;
            out[frame * n_bins * 2 + k * 2 + 1] = im;
        }
    }

    Ok(out)
}

// --------------------------------------------------------------------------- //
//  Magnitude and power from CQT output
// --------------------------------------------------------------------------- //

/// Compute CQT magnitude (absolute value) from interleaved `[Re, Im]` output.
///
/// Input: flat `[Re₀, Im₀, Re₁, Im₁, …]` from [`cqt_reference`].
/// Output: `[|X₀|, |X₁|, …]` of length `cqt_out.len() / 2`.
#[must_use]
pub fn cqt_magnitude(cqt_out: &[f64]) -> Vec<f64> {
    cqt_out
        .chunks_exact(2)
        .map(|pair| (pair[0] * pair[0] + pair[1] * pair[1]).sqrt())
        .collect()
}

/// Compute CQT power (squared magnitude) from interleaved `[Re, Im]` output.
///
/// Returns `Re² + Im²` for each complex coefficient.
#[must_use]
pub fn cqt_power(cqt_out: &[f64]) -> Vec<f64> {
    cqt_out
        .chunks_exact(2)
        .map(|pair| pair[0] * pair[0] + pair[1] * pair[1])
        .collect()
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 8_000.0_f64;
    const F_MIN: f64 = 100.0_f64;
    const N_BINS: usize = 12;
    const BPO: usize = 12;
    const HOP: usize = 512;

    fn default_config() -> CqtConfig {
        CqtConfig::new(SR, F_MIN, N_BINS, BPO, HOP, WindowType::Hann)
            .expect("valid default CQT config")
    }

    fn make_sine(freq_hz: f64, n_samples: usize, sr: f64) -> Vec<f64> {
        (0..n_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sr).sin())
            .collect()
    }

    // ---- Test 1: freq_at_bin(0) == f_min ----
    #[test]
    fn test_freq_at_bin_zero_equals_fmin() {
        let cfg = default_config();
        assert_eq!(cfg.freq_at_bin(0), F_MIN, "bin 0 must have frequency f_min");
    }

    // ---- Test 2: freq_at_bin(bpo) ≈ 2*f_min (one octave) ----
    #[test]
    fn test_freq_at_bin_octave() {
        let cfg = default_config();
        let f_octave = cfg.freq_at_bin(BPO);
        let expected = 2.0_f64 * F_MIN;
        assert!(
            (f_octave - expected).abs() < 1e-10,
            "bin {BPO} should be one octave above f_min; got {f_octave}, expected {expected}"
        );
    }

    // ---- Test 3: q_factor formula ----
    #[test]
    fn test_q_factor_formula() {
        let cfg = default_config();
        let q = cfg.q_factor();
        let expected = 1.0_f64 / (2.0_f64.powf(1.0_f64 / BPO as f64) - 1.0_f64);
        assert!(
            (q - expected).abs() < 1e-10,
            "Q factor mismatch: got {q}, expected {expected}"
        );
    }

    // ---- Test 4: window_length monotone decreasing with bin index ----
    #[test]
    fn test_window_length_monotone_decreasing() {
        let cfg = default_config();
        let nk0 = cfg.window_length_at_bin(0);
        let nk1 = cfg.window_length_at_bin(1);
        assert!(
            nk0 >= nk1,
            "lower freq bin should have longer window: N_0={nk0}, N_1={nk1}"
        );
    }

    // ---- Test 5: num_frames * hop covers at least first num_frames frames ----
    #[test]
    fn test_num_frames_covers_signal() {
        let cfg = default_config();
        let n_samples = 4096usize;
        let nf = cfg.num_frames(n_samples);
        // Frame 0 starts at 0, frame nf-1 starts at (nf-1)*hop.
        let last_frame_start = (nf - 1) * HOP;
        assert!(
            last_frame_start < n_samples,
            "last frame start {last_frame_start} should be within signal of len {n_samples}"
        );
    }

    // ---- Test 6: output length = 2 * num_frames * n_bins ----
    #[test]
    fn test_output_length_correct() {
        let cfg = default_config();
        let signal = vec![0.0_f64; 4096];
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed on zero signal");
        let expected_len = 2 * cfg.num_frames(signal.len()) * N_BINS;
        assert_eq!(out.len(), expected_len, "output length mismatch");
    }

    // ---- Test 7: zero signal → all zeros output ----
    #[test]
    fn test_zero_signal_all_zeros() {
        let cfg = default_config();
        let signal = vec![0.0_f64; 2048];
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed on zero signal");
        assert!(
            out.iter().all(|&v| v == 0.0_f64),
            "zero signal should produce all-zero CQT output"
        );
    }

    // ---- Test 8: sine at f_min localises to bin 0 ----
    #[test]
    fn test_sine_fmin_localises_to_bin0() {
        let n_samples = 4096usize;
        let cfg = default_config();
        let signal = make_sine(F_MIN, n_samples, SR);
        let out = cqt_reference(&signal, &cfg).expect("CQT of sine should succeed");
        let mag = cqt_magnitude(&out);
        // In the first frame, bin 0 should have larger magnitude than bin 5.
        let n_bins = cfg.num_bins();
        let mag_bin0 = mag[0]; // frame 0, bin 0
        let mag_bin5 = mag[5]; // frame 0, bin 5
        assert!(
            mag_bin0 > mag_bin5,
            "f_min sine should localise to bin 0: mag[0]={mag_bin0}, mag[5]={mag_bin5}; n_bins={n_bins}"
        );
    }

    // ---- Test 9: DC signal → imaginary parts ≈ 0 ----
    //
    // With bins_per_octave=1, Q=1 exactly. Then the CQT kernel sums
    // sin(-2π·1·n/N_k) over n=0..N_k-1, which equals 0 (full-cycle
    // geometric series), so Im ≈ 0 for a DC input with a Rectangular window.
    #[test]
    fn test_dc_signal_imaginary_near_zero() {
        // bpo=1 → Q=1 (exact integer) → full-cycle kernel → Im=0 for DC.
        let cfg = CqtConfig::new(SR, F_MIN, 1, 1, HOP, WindowType::Rectangular)
            .expect("valid config bpo=1");
        let signal = vec![1.0_f64; 2048];
        let out = cqt_reference(&signal, &cfg).expect("CQT of DC signal should succeed");
        // Check imaginary parts (indices 1, 3, 5, …).
        for (idx, chunk) in out.chunks_exact(2).enumerate() {
            let im = chunk[1];
            assert!(
                im.abs() < 1e-9,
                "DC signal (bpo=1): Im at pair {idx} = {im} (should be ≈ 0)"
            );
        }
    }

    // ---- Test 10: cqt_magnitude non-negative ----
    #[test]
    fn test_cqt_magnitude_non_negative() {
        let cfg = default_config();
        let signal = make_sine(F_MIN, 4096, SR);
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed");
        let mag = cqt_magnitude(&out);
        assert!(
            mag.iter().all(|&v| v >= 0.0_f64),
            "cqt_magnitude should be non-negative"
        );
    }

    // ---- Test 11: cqt_power = cqt_magnitude² ----
    #[test]
    fn test_cqt_power_equals_magnitude_squared() {
        let cfg = default_config();
        let signal = make_sine(F_MIN, 4096, SR);
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed");
        let mag = cqt_magnitude(&out);
        let pow = cqt_power(&out);
        for (i, (&m, &p)) in mag.iter().zip(pow.iter()).enumerate() {
            let diff = (m * m - p).abs();
            assert!(
                diff < 1e-10,
                "cqt_power[{i}] = {p} != cqt_magnitude[{i}]² = {}; diff={diff}",
                m * m
            );
        }
    }

    // ---- Test 12: 12 bpo — bin 12 freq ≈ 2 * bin 0 freq ----
    #[test]
    fn test_12bpo_bin12_is_octave() {
        let cfg = CqtConfig::new(SR, F_MIN, 24, 12, HOP, WindowType::Hann).expect("valid config");
        let f0 = cfg.freq_at_bin(0);
        let f12 = cfg.freq_at_bin(12);
        let ratio = f12 / f0;
        assert!(
            (ratio - 2.0_f64).abs() < 0.001_f64,
            "bin 12 should be one octave (×2) above bin 0; ratio={ratio}"
        );
    }

    // ---- Test 13: cqt_magnitude and cqt_power lengths = half of cqt_reference output ----
    #[test]
    fn test_magnitude_power_half_length() {
        let cfg = default_config();
        let signal = vec![1.0_f64; 2048];
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed");
        let mag = cqt_magnitude(&out);
        let pow = cqt_power(&out);
        assert_eq!(mag.len(), out.len() / 2);
        assert_eq!(pow.len(), out.len() / 2);
    }

    // ---- Test 14: all output values finite ----
    #[test]
    fn test_output_all_finite() {
        let cfg = default_config();
        let signal = make_sine(440.0_f64, 4096, SR);
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    // ---- Test 15: different hop_lengths → proportionally different num_frames ----
    #[test]
    fn test_hop_proportional_num_frames() {
        let n_samples = 4096usize;
        let cfg_hop256 = CqtConfig::new(SR, F_MIN, N_BINS, BPO, 256, WindowType::Hann)
            .expect("valid config hop=256");
        let cfg_hop512 = CqtConfig::new(SR, F_MIN, N_BINS, BPO, 512, WindowType::Hann)
            .expect("valid config hop=512");
        let nf256 = cfg_hop256.num_frames(n_samples);
        let nf512 = cfg_hop512.num_frames(n_samples);
        // hop=256 should produce approximately 2x as many frames as hop=512.
        assert!(
            nf256 >= 2 * nf512 - 2,
            "hop=256 frames ({nf256}) should be ~2x hop=512 frames ({nf512})"
        );
    }

    // ---- Test 16: single-sample signal produces 1 frame ----
    #[test]
    fn test_single_sample_one_frame() {
        let cfg = default_config();
        let signal = vec![1.0_f64];
        let out = cqt_reference(&signal, &cfg).expect("CQT of single sample should not crash");
        assert_eq!(cfg.num_frames(1), 1);
        assert_eq!(out.len(), 2 * N_BINS);
    }

    // ---- Test 17: energy concentrates at dominant frequency bin ----
    #[test]
    fn test_energy_concentrates_at_dominant_frequency() {
        let cfg = default_config();
        let signal = make_sine(F_MIN, 4096, SR);
        let out = cqt_reference(&signal, &cfg).expect("CQT should succeed");
        let pow = cqt_power(&out);
        // Sum power at bin 0 across all frames vs total power.
        let n_frames = cfg.num_frames(signal.len());
        let n_bins = cfg.num_bins();
        let power_bin0: f64 = (0..n_frames).map(|f| pow[f * n_bins]).sum();
        let total_power: f64 = pow.iter().sum();
        let fraction = power_bin0 / (total_power + 1e-30);
        assert!(
            fraction > 0.3_f64,
            "f_min sine energy should concentrate at bin 0; fraction={fraction:.3}"
        );
    }

    // ---- Test 18: sample_rate <= 0 → InvalidParameter ----
    #[test]
    fn test_invalid_sample_rate() {
        let result = CqtConfig::new(0.0_f64, F_MIN, N_BINS, BPO, HOP, WindowType::Hann);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "sample_rate=0 should return InvalidParameter"
        );
        let result2 = CqtConfig::new(-1.0_f64, F_MIN, N_BINS, BPO, HOP, WindowType::Hann);
        assert!(
            matches!(result2, Err(SignalError::InvalidParameter(_))),
            "sample_rate<0 should return InvalidParameter"
        );
    }

    // ---- Test 19: f_min <= 0 → InvalidParameter ----
    #[test]
    fn test_invalid_fmin() {
        let result = CqtConfig::new(SR, 0.0_f64, N_BINS, BPO, HOP, WindowType::Hann);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "f_min=0 should return InvalidParameter"
        );
        let result2 = CqtConfig::new(SR, -50.0_f64, N_BINS, BPO, HOP, WindowType::Hann);
        assert!(
            matches!(result2, Err(SignalError::InvalidParameter(_))),
            "f_min<0 should return InvalidParameter"
        );
    }

    // ---- Test 20: n_bins=0 → InvalidSize ----
    #[test]
    fn test_invalid_nbins() {
        let result = CqtConfig::new(SR, F_MIN, 0, BPO, HOP, WindowType::Hann);
        assert!(
            matches!(result, Err(SignalError::InvalidSize(_))),
            "n_bins=0 should return InvalidSize"
        );
    }

    // ---- Test 21: hop_length=0 → InvalidParameter ----
    #[test]
    fn test_invalid_hop_length() {
        let result = CqtConfig::new(SR, F_MIN, N_BINS, BPO, 0, WindowType::Hann);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "hop_length=0 should return InvalidParameter"
        );
    }

    // ---- Test 22: bins_per_octave=0 → InvalidParameter ----
    #[test]
    fn test_invalid_bins_per_octave() {
        let result = CqtConfig::new(SR, F_MIN, N_BINS, 0, HOP, WindowType::Hann);
        assert!(
            matches!(result, Err(SignalError::InvalidParameter(_))),
            "bins_per_octave=0 should return InvalidParameter"
        );
    }
}
