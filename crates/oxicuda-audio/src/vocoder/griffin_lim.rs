//! Griffin-Lim Algorithm (GLA) and Fast Griffin-Lim (FGLA) for phase
//! reconstruction from a magnitude spectrogram.
//!
//! Iteratively reconstructs a time-domain signal whose STFT magnitude matches a
//! target magnitude spectrogram. The Fast Griffin-Lim variant (Perraudin 2013)
//! adds a momentum term that accelerates convergence.
//!
//! All STFT/ISTFT operations use a Hann analysis window and an overlap-add
//! synthesis window; the DFT is computed directly (O(N²) per frame), keeping
//! the implementation pure-Rust and free of external FFT crates.
//!
//! ## References
//! - Griffin, D. W. & Lim, J. S. (1984). "Signal Estimation from Modified
//!   Short-Time Fourier Transform." IEEE TASLP 32(2), 236–243.
//! - Perraudin, N. et al. (2013). "A fast Griffin-Lim algorithm." WASPAA.

use std::f64::consts::PI;

use crate::error::{AudioError, AudioResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`griffin_lim`].
#[derive(Debug, Clone)]
pub struct GriffinLimConfig {
    /// FFT size in samples (must be >= 1).
    pub n_fft: usize,
    /// Hop length in samples (must be >= 1).
    pub hop_length: usize,
    /// Number of Griffin-Lim iterations.
    pub n_iter: usize,
    /// FGLA momentum `β ∈ [0, 1)`. Set to `0.0` for standard Griffin-Lim.
    pub momentum: f64,
}

impl Default for GriffinLimConfig {
    fn default() -> Self {
        Self {
            n_fft: 256,
            hop_length: 64,
            n_iter: 32,
            momentum: 0.0,
        }
    }
}

// ─── Hann window ─────────────────────────────────────────────────────────────

/// Build a length-`n` Hann window.
///
/// For `n == 1` returns `[1.0]`; otherwise `w[k] = 0.5 * (1 - cos(2πk/(n-1)))`.
fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos()))
        .collect()
}

// ─── STFT ─────────────────────────────────────────────────────────────────────

/// Hann-windowed Short-Time Fourier Transform.
///
/// Computes DFT directly (O(N²) per frame). Output is flat interleaved
/// `[Re(t,0), Im(t,0), Re(t,1), Im(t,1), …]` with length `n_frames * n_bins * 2`.
///
/// Number of frames: `(n_samples.saturating_sub(n_fft)) / hop_length + 1`
/// (zero frames if `n_samples < n_fft`).
///
/// # Errors
/// Returns [`AudioError`] if `n_fft == 0` or `hop_length == 0`.
pub fn stft_hann(
    signal: &[f64],
    n_samples: usize,
    n_fft: usize,
    hop_length: usize,
) -> AudioResult<Vec<f64>> {
    validate_stft_params(n_fft, hop_length, 0.0)?;

    let n_bins = n_fft / 2 + 1;
    let n_frames = if n_samples < n_fft {
        0
    } else {
        (n_samples - n_fft) / hop_length + 1
    };

    let window = hann_window(n_fft);
    let mut out = vec![0.0_f64; n_frames * n_bins * 2];

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame_slice = &signal[start..start + n_fft];
        let base = t * n_bins * 2;

        for k in 0..n_bins {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let angle_step = 2.0 * PI * k as f64 / n_fft as f64;
            for (n, (&x, &w)) in frame_slice.iter().zip(window.iter()).enumerate() {
                let angle = angle_step * n as f64;
                re += x * w * angle.cos();
                im -= x * w * angle.sin();
            }
            out[base + k * 2] = re;
            out[base + k * 2 + 1] = im;
        }
    }

    Ok(out)
}

// ─── ISTFT ────────────────────────────────────────────────────────────────────

/// Hann-windowed Inverse Short-Time Fourier Transform via overlap-add.
///
/// Input is flat interleaved `[Re(t,0), Im(t,0), …]` of length `n_frames * n_bins * 2`.
/// Output length: `n_frames * hop_length + n_fft`.
///
/// # Errors
/// Returns [`AudioError`] if `n_fft == 0` or `hop_length == 0`, or if the
/// input length does not match `n_frames * (n_fft/2+1) * 2`.
pub fn istft_hann(
    stft_complex: &[f64],
    n_frames: usize,
    n_fft: usize,
    hop_length: usize,
) -> AudioResult<Vec<f64>> {
    validate_stft_params(n_fft, hop_length, 0.0)?;

    let n_bins = n_fft / 2 + 1;
    let expected_len = n_frames * n_bins * 2;
    if stft_complex.len() != expected_len {
        return Err(AudioError::DimensionMismatch {
            expected: expected_len,
            got: stft_complex.len(),
        });
    }

    let out_len = n_frames * hop_length + n_fft;
    let mut out = vec![0.0_f64; out_len];
    let mut win_sum = vec![0.0_f64; out_len];

    let window = hann_window(n_fft);
    let inv_n = 1.0 / n_fft as f64;

    for t in 0..n_frames {
        let base = t * n_bins * 2;
        let start = t * hop_length;

        // DC and Nyquist bins (imaginary parts are zero for real signals)
        let re_dc = stft_complex[base];
        let re_ny = stft_complex[base + (n_fft / 2) * 2]; // n_bins-1 = n_fft/2

        // IDFT for each sample in the frame using conjugate symmetry
        for n in 0..n_fft {
            let mut x = re_dc; // k=0
            let angle_step_base = 2.0 * PI * n as f64 / n_fft as f64;
            for k in 1..n_fft / 2 {
                let re_k = stft_complex[base + k * 2];
                let im_k = stft_complex[base + k * 2 + 1];
                let angle = angle_step_base * k as f64;
                // 2 * (Re[k] * cos - Im[k] * (-sin)) = 2*(Re*cos + Im*sin)?
                // Im from STFT has sign: Im = -Σ x*w*sin, so IDFT reconstruction:
                // x[n] = (1/N)[Re[0] + 2*Σ_{k=1}^{N/2-1}(Re[k]*cos - Im[k]*(-sin))
                //                                                     wait...
                // With the sign convention Im[k] = -Σ x*w*sin(2πkn/N):
                // Re[k] + j*Im[k] = Σ x*w*(cos - j*sin) = DFT(x*w)
                // IDFT: x[n] = (1/N)*Σ_k (Re[k]+j*Im[k]) * e^{j2πkn/N}
                //            = (1/N)*Σ_k (Re[k]*cos - Im[k]*sin + j*(...))
                // For real x, Im part vanishes.
                // Using conjugate symmetry (k and N-k are conjugates):
                // x[n] = (1/N)*[Re[0] + 2*Σ_{k=1}^{N/2-1}(Re[k]*cos(2πkn/N) - Im[k]*sin(2πkn/N)) + Re[N/2]*cos(πn)]
                x += 2.0 * (re_k * angle.cos() - im_k * angle.sin());
            }
            // Nyquist: cos(π*n) = (-1)^n
            let nyquist_cos = if n % 2 == 0 { 1.0 } else { -1.0 };
            x += re_ny * nyquist_cos;
            x *= inv_n;

            // Multiply by synthesis (Hann) window and overlap-add
            let w = window[n];
            out[start + n] += w * x;
            win_sum[start + n] += w * w;
        }
    }

    // Normalize by window power accumulation
    for (o, ws) in out.iter_mut().zip(win_sum.iter()) {
        *o /= ws.max(1e-8);
    }

    Ok(out)
}

// ─── Magnitude extraction ─────────────────────────────────────────────────────

/// Compute the magnitude spectrogram from a raw audio signal.
///
/// Returns a flat array of shape `[n_frames × n_bins]`, row-major.
///
/// # Errors
/// Returns [`AudioError`] if `n_fft == 0` or `hop_length == 0`.
pub fn magnitude_from_signal(
    signal: &[f64],
    n_samples: usize,
    n_fft: usize,
    hop_length: usize,
) -> AudioResult<Vec<f64>> {
    let stft = stft_hann(signal, n_samples, n_fft, hop_length)?;
    let n_bins = n_fft / 2 + 1;
    let n_frames = if n_samples < n_fft {
        0
    } else {
        (n_samples - n_fft) / hop_length + 1
    };
    let mut mag = vec![0.0_f64; n_frames * n_bins];
    for t in 0..n_frames {
        let base_s = t * n_bins * 2;
        let base_m = t * n_bins;
        for k in 0..n_bins {
            let re = stft[base_s + k * 2];
            let im = stft[base_s + k * 2 + 1];
            mag[base_m + k] = (re * re + im * im).sqrt();
        }
    }
    Ok(mag)
}

// ─── Validation helper ────────────────────────────────────────────────────────

fn validate_stft_params(n_fft: usize, hop_length: usize, momentum: f64) -> AudioResult<()> {
    if n_fft == 0 {
        return Err(AudioError::InvalidKernelSize(n_fft));
    }
    if hop_length == 0 {
        return Err(AudioError::InvalidStride(hop_length));
    }
    if !(0.0..1.0).contains(&momentum) {
        return Err(AudioError::NonFinite {
            msg: format!("momentum must be in [0, 1), got {momentum}"),
        });
    }
    Ok(())
}

fn validate_gl_config(config: &GriffinLimConfig) -> AudioResult<()> {
    if config.n_fft == 0 {
        return Err(AudioError::InvalidKernelSize(config.n_fft));
    }
    if config.hop_length == 0 {
        return Err(AudioError::InvalidStride(config.hop_length));
    }
    if config.momentum < 0.0 || config.momentum >= 1.0 {
        return Err(AudioError::NonFinite {
            msg: format!("momentum must be in [0, 1), got {}", config.momentum),
        });
    }
    Ok(())
}

// ─── Griffin-Lim ─────────────────────────────────────────────────────────────

/// Reconstruct a time-domain signal from a magnitude spectrogram via the
/// Griffin-Lim Algorithm (GLA) or Fast Griffin-Lim (FGLA with momentum > 0).
///
/// `magnitude` must have shape `[n_frames × (n_fft/2+1)]`, row-major.
///
/// # Errors
/// Returns [`AudioError`] if `n_fft == 0`, `hop_length == 0`,
/// `momentum ∉ [0, 1)`, or `magnitude.len() != n_frames * (n_fft/2+1)`.
pub fn griffin_lim(
    magnitude: &[f64],
    n_frames: usize,
    config: &GriffinLimConfig,
) -> AudioResult<Vec<f64>> {
    validate_gl_config(config)?;

    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let n_bins = n_fft / 2 + 1;
    let expected = n_frames * n_bins;

    if magnitude.len() != expected {
        return Err(AudioError::DimensionMismatch {
            expected,
            got: magnitude.len(),
        });
    }

    if n_frames == 0 {
        return Ok(Vec::new());
    }

    let stft_len = n_frames * n_bins * 2;

    // Initialise phases to zero: build complex STFT with magnitude * e^{j*0}
    let mut phase_re = vec![0.0_f64; n_frames * n_bins];
    let mut phase_im = vec![0.0_f64; n_frames * n_bins];
    phase_re.copy_from_slice(magnitude); // cos(0) = 1; sin(0) = 0

    // FGLA momentum buffers (previous iteration's Re/Im)
    let mut prev_re = vec![0.0_f64; n_frames * n_bins];
    let mut prev_im = vec![0.0_f64; n_frames * n_bins];

    let mut stft_buf = vec![0.0_f64; stft_len];

    for _iter in 0..config.n_iter {
        // Build complex STFT buffer from current phase estimates
        for t in 0..n_frames {
            let base_s = t * n_bins * 2;
            let base_p = t * n_bins;
            for k in 0..n_bins {
                stft_buf[base_s + k * 2] = phase_re[base_p + k];
                stft_buf[base_s + k * 2 + 1] = phase_im[base_p + k];
            }
        }

        // Reconstruct signal via ISTFT
        let out_len = n_frames * hop + n_fft;
        let signal = istft_hann(&stft_buf, n_frames, n_fft, hop)?;

        // Re-analyse to get consistent STFT
        let new_stft = stft_hann(&signal, out_len, n_fft, hop)?;

        // Update phase estimates
        let new_n_frames = if out_len < n_fft {
            0
        } else {
            (out_len - n_fft) / hop + 1
        };
        let use_frames = n_frames.min(new_n_frames);

        let beta = config.momentum;

        for t in 0..use_frames {
            let base_s = t * n_bins * 2;
            let base_p = t * n_bins;
            for k in 0..n_bins {
                let re_new = new_stft[base_s + k * 2];
                let im_new = new_stft[base_s + k * 2 + 1];

                let (re_mom, im_mom) = if beta > 0.0 {
                    let pr = prev_re[base_p + k];
                    let pi = prev_im[base_p + k];
                    (re_new + beta * (re_new - pr), im_new + beta * (im_new - pi))
                } else {
                    (re_new, im_new)
                };

                // Extract new phase angle and set phase_re/im = magnitude * exp(j*phi)
                let norm = (re_mom * re_mom + im_mom * im_mom).sqrt();
                let mag_val = magnitude[base_p + k];
                if norm > 1e-12 {
                    phase_re[base_p + k] = mag_val * re_mom / norm;
                    phase_im[base_p + k] = mag_val * im_mom / norm;
                } else {
                    phase_re[base_p + k] = mag_val;
                    phase_im[base_p + k] = 0.0;
                }

                prev_re[base_p + k] = re_new;
                prev_im[base_p + k] = im_new;
            }
        }
        // Frames beyond new_n_frames keep their previous phase
    }

    // Final synthesis
    for t in 0..n_frames {
        let base_s = t * n_bins * 2;
        let base_p = t * n_bins;
        for k in 0..n_bins {
            stft_buf[base_s + k * 2] = phase_re[base_p + k];
            stft_buf[base_s + k * 2 + 1] = phase_im[base_p + k];
        }
    }
    istft_hann(&stft_buf, n_frames, n_fft, hop)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── stft_hann / istft_hann ────────────────────────────────────────────────

    #[test]
    fn stft_output_length() {
        let n_fft = 32;
        let hop = 8;
        let n_samples = 200;
        let signal = vec![0.5_f64; n_samples];
        let stft = stft_hann(&signal, n_samples, n_fft, hop).expect("stft_hann should succeed");
        let n_bins = n_fft / 2 + 1;
        let n_frames = (n_samples - n_fft) / hop + 1;
        assert_eq!(stft.len(), n_frames * n_bins * 2);
    }

    #[test]
    fn stft_zero_signal_gives_zero_spectrum() {
        let n_fft = 16;
        let hop = 4;
        let signal = vec![0.0_f64; 64];
        let stft = stft_hann(&signal, 64, n_fft, hop).expect("stft_hann should succeed");
        for &v in &stft {
            assert!(v.abs() < 1e-12, "non-zero for zero signal: {v}");
        }
    }

    #[test]
    fn istft_output_length() {
        let n_fft = 32;
        let hop = 8;
        let n_frames = 10;
        let n_bins = n_fft / 2 + 1;
        let stft = vec![0.0_f64; n_frames * n_bins * 2];
        let signal = istft_hann(&stft, n_frames, n_fft, hop).expect("istft_hann should succeed");
        assert_eq!(signal.len(), n_frames * hop + n_fft);
    }

    #[test]
    fn stft_istft_roundtrip_interior_sine() {
        // Round-trip on a sine wave; trim edges to avoid boundary artifacts
        let n_fft = 64;
        let hop = 16;
        let sr = 1000.0_f64;
        let freq = 50.0_f64;
        let n_samples = 600;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();
        let stft = stft_hann(&signal, n_samples, n_fft, hop).expect("stft_hann should succeed");
        let n_frames = (n_samples - n_fft) / hop + 1;
        let reconstructed =
            istft_hann(&stft, n_frames, n_fft, hop).expect("istft_hann should succeed");

        // Compare interior region (skip first and last n_fft samples)
        let trim = n_fft;
        let inner_len = reconstructed.len().saturating_sub(2 * trim);
        if inner_len > 0 {
            let orig_inner = &signal[trim..trim + inner_len.min(signal.len() - trim)];
            let rec_inner = &reconstructed[trim..trim + inner_len];
            let use_len = orig_inner.len().min(rec_inner.len());
            let mut max_err = 0.0_f64;
            let mut max_abs = 0.0_f64;
            for (&o, &r) in orig_inner[..use_len]
                .iter()
                .zip(rec_inner[..use_len].iter())
            {
                max_err = max_err.max((o - r).abs());
                max_abs = max_abs.max(o.abs());
            }
            let rel_err = if max_abs > 0.0 {
                max_err / max_abs
            } else {
                0.0
            };
            assert!(
                rel_err < 0.2,
                "round-trip relative error too large: {rel_err:.4}"
            );
        }
    }

    #[test]
    fn stft_error_n_fft_zero() {
        let err = stft_hann(&[1.0], 1, 0, 4).unwrap_err();
        assert!(matches!(err, AudioError::InvalidKernelSize(0)));
    }

    #[test]
    fn stft_error_hop_zero() {
        let err = stft_hann(&[1.0; 16], 16, 8, 0).unwrap_err();
        assert!(matches!(err, AudioError::InvalidStride(0)));
    }

    #[test]
    fn istft_error_wrong_length() {
        let n_fft = 16;
        let hop = 4;
        let n_frames = 5;
        let n_bins = n_fft / 2 + 1;
        let wrong = vec![0.0_f64; n_frames * n_bins * 2 + 1];
        let err = istft_hann(&wrong, n_frames, n_fft, hop).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    // ── griffin_lim ───────────────────────────────────────────────────────────

    #[test]
    fn griffin_lim_zero_magnitude_gives_zero() {
        let n_fft = 32;
        let hop = 8;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 10;
        let mag = vec![0.0_f64; n_frames * n_bins];
        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 5,
            momentum: 0.0,
        };
        let out = griffin_lim(&mag, n_frames, &cfg).expect("griffin_lim should succeed");
        for &v in &out {
            assert!(v.abs() < 1e-8, "non-zero output for zero magnitude: {v}");
        }
    }

    #[test]
    fn griffin_lim_output_length() {
        let n_fft = 32;
        let hop = 8;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 12;
        let mag = vec![1.0_f64; n_frames * n_bins];
        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 3,
            momentum: 0.0,
        };
        let out = griffin_lim(&mag, n_frames, &cfg).expect("griffin_lim should succeed");
        assert_eq!(out.len(), n_frames * hop + n_fft);
    }

    #[test]
    fn griffin_lim_error_n_fft_zero() {
        let cfg = GriffinLimConfig {
            n_fft: 0,
            hop_length: 8,
            n_iter: 5,
            momentum: 0.0,
        };
        let err = griffin_lim(&[1.0], 1, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::InvalidKernelSize(0)));
    }

    #[test]
    fn griffin_lim_error_hop_zero() {
        let cfg = GriffinLimConfig {
            n_fft: 16,
            hop_length: 0,
            n_iter: 5,
            momentum: 0.0,
        };
        let mag = vec![1.0_f64; 16 / 2 + 1];
        let err = griffin_lim(&mag, 1, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::InvalidStride(0)));
    }

    #[test]
    fn griffin_lim_error_momentum_negative() {
        let cfg = GriffinLimConfig {
            n_fft: 16,
            hop_length: 4,
            n_iter: 5,
            momentum: -0.1,
        };
        let n_bins = 16 / 2 + 1;
        let mag = vec![1.0_f64; 4 * n_bins];
        let err = griffin_lim(&mag, 4, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::NonFinite { .. }));
    }

    #[test]
    fn griffin_lim_error_momentum_one() {
        let cfg = GriffinLimConfig {
            n_fft: 16,
            hop_length: 4,
            n_iter: 5,
            momentum: 1.0,
        };
        let n_bins = 16 / 2 + 1;
        let mag = vec![1.0_f64; 4 * n_bins];
        let err = griffin_lim(&mag, 4, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::NonFinite { .. }));
    }

    #[test]
    fn griffin_lim_error_dimension_mismatch() {
        let n_fft = 32;
        let hop = 8;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 5;
        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 3,
            momentum: 0.0,
        };
        let wrong_len_mag = vec![1.0_f64; n_frames * n_bins + 1];
        let err = griffin_lim(&wrong_len_mag, n_frames, &cfg).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    #[test]
    fn griffin_lim_magnitude_consistency_after_convergence() {
        // After enough iterations, STFT(GL(|S|)) should be close to |S|
        let n_fft = 64;
        let hop = 16;
        let n_bins = n_fft / 2 + 1;
        let sr = 8000.0_f64;
        let freq = 440.0_f64;
        let n_samples = 512;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();
        let target_mag = magnitude_from_signal(&signal, n_samples, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        let n_frames = target_mag.len() / n_bins;

        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 50,
            momentum: 0.0,
        };
        let out = griffin_lim(&target_mag, n_frames, &cfg).expect("griffin_lim should succeed");
        let out_len = out.len();
        let gl_mag = magnitude_from_signal(&out, out_len, n_fft, hop)
            .expect("magnitude_from_signal should succeed");

        // Compare magnitudes over the frames that exist in both
        let compare_frames = n_frames.min(gl_mag.len() / n_bins);
        let total = compare_frames * n_bins;
        if total > 0 {
            let mean_target: f64 = target_mag[..total].iter().sum::<f64>() / total as f64;
            let mean_err: f64 = target_mag[..total]
                .iter()
                .zip(gl_mag[..total].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / total as f64;
            let rel = mean_err / (mean_target + 1e-12);
            assert!(
                rel < 0.5,
                "GL magnitude consistency too poor: rel={rel:.4}, mean_target={mean_target:.4}"
            );
        }
    }

    #[test]
    fn griffin_lim_reconstruction_fidelity_sine() {
        // Test via normalized cross-correlation of output vs target magnitude
        let n_fft = 64;
        let hop = 16;
        let n_bins = n_fft / 2 + 1;
        let sr = 8000.0_f64;
        let freq = 440.0_f64;
        let n_samples = 400;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();
        let target_mag = magnitude_from_signal(&signal, n_samples, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        let n_frames = target_mag.len() / n_bins;

        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 50,
            momentum: 0.0,
        };
        let out = griffin_lim(&target_mag, n_frames, &cfg).expect("griffin_lim should succeed");
        let out_len = out.len();
        let gl_mag = magnitude_from_signal(&out, out_len, n_fft, hop)
            .expect("magnitude_from_signal should succeed");

        let len = target_mag.len().min(gl_mag.len());
        if len > 0 {
            let dot: f64 = target_mag[..len]
                .iter()
                .zip(gl_mag[..len].iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f64 = target_mag[..len].iter().map(|a| a * a).sum::<f64>().sqrt();
            let norm_b: f64 = gl_mag[..len].iter().map(|b| b * b).sum::<f64>().sqrt();
            let corr = dot / (norm_a * norm_b + 1e-12);
            assert!(
                corr > 0.7,
                "magnitude correlation too low: {corr:.4} (need > 0.7)"
            );
        }
    }

    #[test]
    fn fgla_momentum_not_worse_than_gl() {
        // FGLA with momentum should achieve similar or better reconstruction
        let n_fft = 64;
        let hop = 16;
        let n_bins = n_fft / 2 + 1;
        let sr = 8000.0_f64;
        let freq = 200.0_f64;
        let n_samples = 400;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();
        let target_mag = magnitude_from_signal(&signal, n_samples, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        let n_frames = target_mag.len() / n_bins;

        let cfg_gl = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 20,
            momentum: 0.0,
        };
        let cfg_fgla = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 20,
            momentum: 0.99,
        };

        let out_gl =
            griffin_lim(&target_mag, n_frames, &cfg_gl).expect("griffin_lim should succeed");
        let out_fgla =
            griffin_lim(&target_mag, n_frames, &cfg_fgla).expect("griffin_lim should succeed");

        // Compute mean absolute magnitude error for each
        let len_gl = out_gl.len();
        let len_fgla = out_fgla.len();
        let mag_gl = magnitude_from_signal(&out_gl, len_gl, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        let mag_fgla = magnitude_from_signal(&out_fgla, len_fgla, n_fft, hop)
            .expect("magnitude_from_signal should succeed");

        let compare = target_mag.len().min(mag_gl.len()).min(mag_fgla.len());
        let err_gl: f64 = target_mag[..compare]
            .iter()
            .zip(mag_gl[..compare].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        let err_fgla: f64 = target_mag[..compare]
            .iter()
            .zip(mag_fgla[..compare].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();

        // FGLA should be at most 50% worse than GL (usually better; allow slack)
        assert!(
            err_fgla <= err_gl * 1.5 + 1e-6,
            "FGLA err={err_fgla:.4} should not be much worse than GL err={err_gl:.4}"
        );
    }

    #[test]
    fn griffin_lim_zero_frames() {
        let cfg = GriffinLimConfig {
            n_fft: 32,
            hop_length: 8,
            n_iter: 3,
            momentum: 0.0,
        };
        let mag: Vec<f64> = Vec::new();
        let out = griffin_lim(&mag, 0, &cfg).expect("griffin_lim should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn magnitude_from_signal_shape() {
        let n_fft = 32;
        let hop = 8;
        let n_bins = n_fft / 2 + 1;
        let n_samples = 200;
        let signal = vec![0.1_f64; n_samples];
        let mag = magnitude_from_signal(&signal, n_samples, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        let n_frames = (n_samples - n_fft) / hop + 1;
        assert_eq!(mag.len(), n_frames * n_bins);
        assert!(mag.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn magnitude_from_signal_non_negative() {
        let n_fft = 16;
        let hop = 4;
        let n_samples = 100;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 * 0.1).sin() - 0.5)
            .collect();
        let mag = magnitude_from_signal(&signal, n_samples, n_fft, hop)
            .expect("magnitude_from_signal should succeed");
        for &v in &mag {
            assert!(v >= 0.0 && v.is_finite());
        }
    }

    #[test]
    fn griffin_lim_all_output_finite() {
        let n_fft = 32;
        let hop = 8;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 8;
        let mag: Vec<f64> = (0..n_frames * n_bins)
            .map(|i| ((i as f64) * 0.07).sin().abs() + 0.5)
            .collect();
        let cfg = GriffinLimConfig {
            n_fft,
            hop_length: hop,
            n_iter: 10,
            momentum: 0.0,
        };
        let out = griffin_lim(&mag, n_frames, &cfg).expect("griffin_lim should succeed");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite in GL output");
    }
}
