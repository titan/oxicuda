//! Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
//!
//! The STFT segments a signal into overlapping frames, multiplies each frame
//! by a window function, and applies a DFT.  This produces a time-frequency
//! representation of the signal (a *spectrogram*).
//!
//! The inverse operation (ISTFT) uses overlap-add (OLA) reconstruction to
//! recover (an approximation of) the original signal.
//!
//! # References
//! - Gabor, D. (1946). "Theory of communication". J. IEE.
//! - Griffin & Lim (1984). "Signal estimation from modified short-time
//!   Fourier transform". IEEE TASSP.

use std::f64::consts::PI;

use crate::error::{FftError, FftResult};

// ---------------------------------------------------------------------------
// Window type
// ---------------------------------------------------------------------------

/// Supported analysis window functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowType {
    /// Rectangular (boxcar) window — no weighting.
    Rectangular,
    /// Hann (von Hann) window: `0.5 (1 − cos(2πn/(N−1)))`.
    Hann,
    /// Hamming window: `0.54 − 0.46 cos(2πn/(N−1))`.
    Hamming,
    /// Blackman window: three-term cosine window with very low sidelobes.
    Blackman,
}

impl WindowType {
    /// Compute the coefficient `w[n]` for a window of length `N`.
    #[inline]
    pub fn coefficient(self, n: usize, len: usize) -> f64 {
        let n = n as f64;
        let nm1 = (len - 1) as f64;
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => 0.5 * (1.0 - (2.0 * PI * n / nm1).cos()),
            Self::Hamming => 0.54 - 0.46 * (2.0 * PI * n / nm1).cos(),
            Self::Blackman => {
                0.42 - 0.5 * (2.0 * PI * n / nm1).cos() + 0.08 * (4.0 * PI * n / nm1).cos()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// STFT configuration and handle
// ---------------------------------------------------------------------------

/// Configuration for the STFT / ISTFT.
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// Analysis window length `N` (samples).
    pub window_size: usize,
    /// Hop size `H` between successive frames (samples).
    pub hop_size: usize,
    /// Window function type.
    pub window_type: WindowType,
}

/// STFT / ISTFT handle.
///
/// Constructed from a [`StftConfig`].  The window coefficients are
/// pre-computed once and reused across `compute`, `spectrogram`, and `istft`
/// calls.
#[derive(Debug, Clone)]
pub struct Stft {
    window: Vec<f64>,
    config: StftConfig,
}

impl Stft {
    /// Create a new STFT handle from the given configuration.
    ///
    /// Pre-computes the window vector.
    ///
    /// # Errors
    ///
    /// - [`FftError::InvalidSize`] if `window_size == 0`.
    /// - [`FftError::InvalidSize`] if `hop_size == 0`.
    /// - [`FftError::InvalidSize`] if `window_size` is not a power of two
    ///   (required for the internal Cooley-Tukey FFT).
    pub fn new(config: StftConfig) -> FftResult<Self> {
        if config.window_size == 0 {
            return Err(FftError::InvalidSize("window_size must be > 0".to_string()));
        }
        if config.hop_size == 0 {
            return Err(FftError::InvalidSize("hop_size must be > 0".to_string()));
        }
        if !config.window_size.is_power_of_two() {
            return Err(FftError::InvalidSize(format!(
                "window_size must be a power of 2 for the internal FFT, got {}",
                config.window_size
            )));
        }

        let window: Vec<f64> = (0..config.window_size)
            .map(|n| config.window_type.coefficient(n, config.window_size))
            .collect();

        Ok(Self { window, config })
    }

    // -----------------------------------------------------------------------
    // Shape helpers
    // -----------------------------------------------------------------------

    /// Number of output frequency bins: `window_size / 2 + 1`.
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.config.window_size / 2 + 1
    }

    /// Number of STFT frames for a signal of length `signal_len`.
    ///
    /// Returns `0` if the signal is shorter than the window.
    #[inline]
    pub fn n_frames(&self, signal_len: usize) -> usize {
        if signal_len < self.config.window_size {
            return 0;
        }
        (signal_len - self.config.window_size) / self.config.hop_size + 1
    }

    // -----------------------------------------------------------------------
    // Forward STFT
    // -----------------------------------------------------------------------

    /// Compute the STFT of `signal`.
    ///
    /// Returns a `Vec` of `n_frames` rows, each containing `n_bins` complex
    /// pairs `(Re, Im)` (one-sided spectrum, DC to Nyquist inclusive).
    ///
    /// # Errors
    ///
    /// - [`FftError::InvalidSize`] if `signal.len() < window_size`.
    pub fn compute(&self, signal: &[f64]) -> FftResult<Vec<Vec<(f64, f64)>>> {
        let n_win = self.config.window_size;
        if signal.len() < n_win {
            return Err(FftError::InvalidSize(format!(
                "signal length {} < window_size {}",
                signal.len(),
                n_win
            )));
        }

        let n_frames = self.n_frames(signal.len());
        let n_bins = self.n_bins();
        let mut frames = Vec::with_capacity(n_frames);

        for t in 0..n_frames {
            let start = t * self.config.hop_size;
            let frame = &signal[start..start + n_win];

            // Apply window
            let mut windowed: Vec<f64> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(s, w)| s * w)
                .collect();

            // FFT (in-place on complex representation)
            let mut cx: Vec<(f64, f64)> = windowed.iter_mut().map(|&mut x| (x, 0.0)).collect();
            fft_inplace(&mut cx, false);

            // Keep only positive frequencies [0, N/2]
            frames.push(cx[..n_bins].to_vec());
        }

        Ok(frames)
    }

    // -----------------------------------------------------------------------
    // Spectrogram (magnitude squared)
    // -----------------------------------------------------------------------

    /// Compute the power spectrogram: `|STFT(t, f)|^2`.
    ///
    /// Returns a `Vec` of `n_frames` rows, each with `n_bins` non-negative
    /// power values.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Stft::compute`].
    pub fn spectrogram(&self, signal: &[f64]) -> FftResult<Vec<Vec<f64>>> {
        let stft = self.compute(signal)?;
        Ok(stft
            .into_iter()
            .map(|frame| frame.iter().map(|(re, im)| re * re + im * im).collect())
            .collect())
    }

    // -----------------------------------------------------------------------
    // Inverse STFT (overlap-add)
    // -----------------------------------------------------------------------

    /// Inverse STFT via overlap-add reconstruction.
    ///
    /// Each STFT frame is converted back to a time-domain frame by IFFT,
    /// multiplied by the synthesis window (same as analysis window), and
    /// accumulated into the output buffer with the configured hop.  The result
    /// is then normalised by the OLA normalisation factor.
    ///
    /// # Arguments
    ///
    /// * `stft_frames` — The STFT output produced by [`Stft::compute`]: each
    ///   inner vec must have exactly `n_bins = window_size/2 + 1` complex bins.
    ///
    /// # Errors
    ///
    /// - [`FftError::InvalidSize`] if any frame has the wrong number of bins.
    pub fn istft(&self, stft_frames: &[Vec<(f64, f64)>]) -> FftResult<Vec<f64>> {
        let n_win = self.config.window_size;
        let n_bins = self.n_bins();
        let n_frames = stft_frames.len();

        if n_frames == 0 {
            return Ok(Vec::new());
        }

        for (t, frame) in stft_frames.iter().enumerate() {
            if frame.len() != n_bins {
                return Err(FftError::InvalidSize(format!(
                    "frame {t} has {} bins, expected {n_bins}",
                    frame.len()
                )));
            }
        }

        let signal_len = (n_frames - 1) * self.config.hop_size + n_win;
        let mut output = vec![0.0_f64; signal_len];
        let mut norm = vec![0.0_f64; signal_len];

        for (t, frame) in stft_frames.iter().enumerate() {
            let start = t * self.config.hop_size;

            // Reconstruct full spectrum (mirror positive-frequency half)
            let mut full_cx: Vec<(f64, f64)> = vec![(0.0, 0.0); n_win];
            full_cx[0] = frame[0]; // DC
            for k in 1..n_bins - 1 {
                full_cx[k] = frame[k];
                full_cx[n_win - k] = (frame[k].0, -frame[k].1); // conjugate mirror
            }
            full_cx[n_win / 2] = frame[n_bins - 1]; // Nyquist (real only)

            // IFFT
            fft_inplace(&mut full_cx, true);

            // Overlap-add with synthesis window
            for (j, (cx_re, _cx_im)) in full_cx.iter().enumerate() {
                let w = self.window[j];
                output[start + j] += cx_re * w;
                norm[start + j] += w * w;
            }
        }

        // Normalise by OLA weights (avoid division by zero)
        for (out, n) in output.iter_mut().zip(norm.iter()) {
            if *n > 1e-12 {
                *out /= n;
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Internal: Cooley-Tukey radix-2 FFT on f64 complex pairs
// ---------------------------------------------------------------------------

/// In-place iterative Cooley-Tukey FFT/IFFT.
/// Length must be a power of 2.
fn fft_inplace(a: &mut [(f64, f64)], inverse: bool) {
    let n = a.len();
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_rev(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly stages
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2_usize;
    while len <= n {
        let half = len >> 1;
        let ang = sign * PI / half as f64;
        let (w_re, w_im) = (ang.cos(), ang.sin());
        let mut start = 0;
        while start < n {
            let (mut u_re, mut u_im) = (1.0_f64, 0.0_f64);
            for j in 0..half {
                let (a_re, a_im) = a[start + j];
                let (b_re, b_im) = a[start + j + half];
                let t_re = u_re * b_re - u_im * b_im;
                let t_im = u_re * b_im + u_im * b_re;
                a[start + j] = (a_re + t_re, a_im + t_im);
                a[start + j + half] = (a_re - t_re, a_im - t_im);
                let new_re = u_re * w_re - u_im * w_im;
                let new_im = u_re * w_im + u_im * w_re;
                u_re = new_re;
                u_im = new_im;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for (re, im) in a.iter_mut() {
            *re *= scale;
            *im *= scale;
        }
    }
}

/// Bit-reverse an index with `bits` significant bits.
#[inline]
fn bit_rev(mut x: usize, bits: usize) -> usize {
    let mut r = 0_usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn hann_stft(window_size: usize, hop_size: usize) -> Stft {
        Stft::new(StftConfig {
            window_size,
            hop_size,
            window_type: WindowType::Hann,
        })
        .expect("Stft::new")
    }

    fn rect_stft(window_size: usize, hop_size: usize) -> Stft {
        Stft::new(StftConfig {
            window_size,
            hop_size,
            window_type: WindowType::Rectangular,
        })
        .expect("Stft::new")
    }

    #[test]
    fn n_bins_correct() {
        let stft = hann_stft(512, 128);
        assert_eq!(stft.n_bins(), 257, "n_bins = window_size/2 + 1");
    }

    #[test]
    fn n_frames_correct() {
        let stft = hann_stft(256, 64);
        let signal_len = 1024;
        // (1024 - 256) / 64 + 1 = 768/64 + 1 = 12 + 1 = 13
        assert_eq!(stft.n_frames(signal_len), 13);
    }

    #[test]
    fn compute_output_shape() {
        let window_size = 64;
        let hop_size = 16;
        let stft = hann_stft(window_size, hop_size);
        let signal = vec![0.0_f64; 256];
        let out = stft.compute(&signal).expect("compute");
        let expected_frames = stft.n_frames(signal.len());
        assert_eq!(out.len(), expected_frames);
        for frame in &out {
            assert_eq!(frame.len(), stft.n_bins());
        }
    }

    #[test]
    fn spectrogram_nonneg() {
        let stft = hann_stft(64, 16);
        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let spec = stft.spectrogram(&signal).expect("spectrogram");
        for row in &spec {
            for &v in row {
                assert!(v >= 0.0, "spectrogram values must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn spectrogram_shape() {
        let window_size = 64;
        let hop_size = 32;
        let stft = hann_stft(window_size, hop_size);
        let signal = vec![1.0_f64; 256];
        let spec = stft.spectrogram(&signal).expect("spectrogram");
        assert_eq!(spec.len(), stft.n_frames(signal.len()));
        for row in &spec {
            assert_eq!(row.len(), stft.n_bins());
        }
    }

    #[test]
    fn hann_window_nonzero() {
        let stft = hann_stft(64, 32);
        // Interior window coefficients should be significantly > 0
        let mid = stft.window.len() / 2;
        assert!(
            stft.window[mid] > 0.4,
            "mid-point Hann coefficient should be > 0.4"
        );
        // First and last coefficients are (near) zero for Hann
        assert!(stft.window[0].abs() < 0.01, "Hann window starts near 0");
    }

    #[test]
    fn dc_signal_peaks_at_bin0() {
        // A constant (DC) signal has all its energy in bin 0.
        let window_size = 64;
        let stft = rect_stft(window_size, window_size);
        let signal = vec![1.0_f64; window_size * 3];
        let spec = stft.spectrogram(&signal).expect("spectrogram");
        for (t, row) in spec.iter().enumerate() {
            let dc = row[0];
            let rest_max = row[1..].iter().cloned().fold(0.0_f64, f64::max);
            assert!(
                dc > rest_max * 2.0,
                "frame {t}: DC bin {dc:.2} should dominate rest_max={rest_max:.2}"
            );
        }
    }

    #[test]
    fn tone_peaks_at_correct_bin() {
        // A sine at k cycles per window peaks at bin k.
        let window_size = 64;
        let k = 5_usize;
        let hop_size = window_size; // non-overlapping for clean frames
        let stft = rect_stft(window_size, hop_size);
        let signal: Vec<f64> = (0..window_size * 2)
            .map(|i| (2.0 * PI * k as f64 * i as f64 / window_size as f64).sin())
            .collect();
        let spec = stft.spectrogram(&signal).expect("spectrogram");
        for (t, row) in spec.iter().enumerate() {
            let peak_bin = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("cmp"))
                .map(|(idx, _)| idx)
                .expect("non-empty");
            assert_eq!(
                peak_bin, k,
                "frame {t}: peak at bin {peak_bin}, expected {k}"
            );
        }
    }

    #[test]
    fn istft_shape() {
        let window_size = 64;
        let hop_size = 16;
        let stft = hann_stft(window_size, hop_size);
        let signal = vec![0.5_f64; 256];
        let frames = stft.compute(&signal).expect("compute");
        let reconstructed = stft.istft(&frames).expect("istft");
        let expected_len = (frames.len() - 1) * hop_size + window_size;
        assert_eq!(reconstructed.len(), expected_len);
    }

    #[test]
    fn istft_roundtrip_approx() {
        // With 75% overlap (hop = N/4) and Hann window, OLA reconstruct≈original.
        let window_size = 64;
        let hop_size = 16; // 75% overlap
        let stft = hann_stft(window_size, hop_size);

        // Use a longer signal so boundary effects are minor in the middle.
        let total = 512;
        let signal: Vec<f64> = (0..total)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / window_size as f64).sin())
            .collect();

        let frames = stft.compute(&signal).expect("compute");
        let recon = stft.istft(&frames).expect("istft");

        // Check the middle portion (avoid boundary effects at start/end).
        let margin = window_size;
        let check_end = recon.len().min(signal.len()) - margin;
        let mut max_err = 0.0_f64;
        for i in margin..check_end {
            let err = (recon[i] - signal[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.1,
            "ISTFT roundtrip max error = {max_err:.4}, expected < 0.1"
        );
    }

    #[test]
    fn hop_size_zero_error() {
        let result = Stft::new(StftConfig {
            window_size: 64,
            hop_size: 0,
            window_type: WindowType::Hann,
        });
        assert!(result.is_err(), "hop_size=0 should return error");
    }

    #[test]
    fn window_size_zero_error() {
        let result = Stft::new(StftConfig {
            window_size: 0,
            hop_size: 16,
            window_type: WindowType::Hann,
        });
        assert!(result.is_err(), "window_size=0 should return error");
    }
}
