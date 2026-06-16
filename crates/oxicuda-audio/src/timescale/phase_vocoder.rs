//! Phase vocoder: STFT-based time-stretching and pitch-shifting
//! (Flanagan & Golden 1966; Laroche & Dolson, "Improved phase vocoder
//! time-scale modification of audio", IEEE TSAP 1999).
//!
//! A phase vocoder changes the **duration** of a signal without altering its
//! **pitch** (time-stretch), and — by combining a stretch with resampling —
//! changes pitch without altering duration (pitch-shift).
//!
//! # Algorithm
//!
//! 1. **Analysis STFT.** The input is analysed with a Hann-windowed Short-Time
//!    Fourier Transform at analysis hop `H_a` (reusing
//!    [`stft_hann`]).  Frame `t` gives a
//!    magnitude `|X_t(k)|` and phase `∠X_t(k)` for every bin `k`.
//!
//! 2. **Instantaneous frequency via phase unwrapping.** For each bin `k`, the
//!    *expected* phase advance between consecutive analysis frames is
//!    `ω_k · H_a` with `ω_k = 2π k / N`.  Subtracting it from the measured phase
//!    difference and wrapping the remainder into `(−π, π]` yields the
//!    *heterodyned phase increment* `Δφ`; the true instantaneous angular
//!    frequency of the partial captured by bin `k` is
//!    ```text
//!    ω̂_k = ω_k + Δφ / H_a .
//!    ```
//!
//! 3. **Phase propagation at the synthesis hop.** Resynthesis uses hop
//!    `H_s = round(r · H_a)` for a time-stretch factor `r` (`r > 1` lengthens,
//!    `r < 1` shortens).  The synthesis phase is accumulated so that each
//!    partial advances by its instantaneous frequency over the *synthesis* hop:
//!    ```text
//!    ψ_t(k) = ψ_{t−1}(k) + ω̂_k · H_s ,
//!    ```
//!    while the magnitude `|X_t(k)|` is carried over unchanged.  Keeping `ω̂_k`
//!    (not `ω_k`) is what preserves pitch: the partials are reproduced at their
//!    true frequencies, only the rate at which frames are laid down changes.
//!
//! 4. **Overlap-add resynthesis.** The modified spectra `|X_t(k)| e^{jψ_t(k)}`
//!    are inverted with a Hann-windowed ISTFT / overlap-add at hop `H_s`
//!    (reusing [`istft_hann`]).  The
//!    output length is `≈ r ×` the input length.
//!
//! 5. **Pitch-shift.** A pitch shift by factor `s` is a time-stretch by `s`
//!    followed by **resampling** by `1/s` (linear interpolation): stretching
//!    makes the signal longer at the same pitch, then resampling back to the
//!    original length compresses time and scales every frequency by `s`.
//!
//! All STFT/ISTFT operations and the Hann window follow the conventions of the
//! [`mod@crate::vocoder::griffin_lim`] module (interleaved `[Re, Im]` spectra,
//! `n_bins = n_fft/2 + 1`, sign convention `Im = −Σ x·w·sin`), so the two
//! modules interoperate.

use std::f64::consts::PI;

use crate::error::{AudioError, AudioResult};
use crate::vocoder::griffin_lim::{istft_hann, stft_hann};

/// Configuration for the [`phase_vocoder_stretch`] / [`pitch_shift`] operations.
#[derive(Debug, Clone)]
pub struct PhaseVocoderConfig {
    /// FFT size in samples (window length). Must be `≥ 2`.
    pub n_fft: usize,
    /// Analysis hop in samples. Must satisfy `1 ≤ hop_analysis ≤ n_fft`
    /// (sufficient overlap is required for a faithful reconstruction; for a Hann
    /// window an overlap factor of at least 2, i.e. `hop ≤ n_fft/2`, is
    /// recommended and enforced as a soft floor).
    pub hop_analysis: usize,
}

impl Default for PhaseVocoderConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_analysis: 256,
        }
    }
}

impl PhaseVocoderConfig {
    /// Validate the FFT size / hop and the requested stretch factor.
    ///
    /// # Errors
    /// - [`AudioError::InvalidKernelSize`] if `n_fft < 2`.
    /// - [`AudioError::InvalidStride`] if `hop_analysis == 0` or
    ///   `hop_analysis > n_fft` (no overlap ⇒ no phase continuity).
    /// - [`AudioError::NonFinite`] if `stretch` is non-finite or `≤ 0`.
    fn validate(&self, stretch: f64) -> AudioResult<()> {
        if self.n_fft < 2 {
            return Err(AudioError::InvalidKernelSize(self.n_fft));
        }
        if self.hop_analysis == 0 || self.hop_analysis > self.n_fft {
            return Err(AudioError::InvalidStride(self.hop_analysis));
        }
        if !stretch.is_finite() || stretch <= 0.0 {
            return Err(AudioError::NonFinite {
                msg: format!("stretch factor must be finite and positive, got {stretch}"),
            });
        }
        Ok(())
    }
}

/// Wrap an angle into the principal interval `(−π, π]`.
#[inline]
fn princ_arg(phase: f64) -> f64 {
    // phase - 2π·round(phase / 2π)
    let two_pi = 2.0 * PI;
    let wrapped = phase - two_pi * (phase / two_pi).round();
    // Map -π exactly to +π for a canonical (−π, π] half-open interval.
    if wrapped <= -PI {
        wrapped + two_pi
    } else {
        wrapped
    }
}

/// Time-stretch a signal by `stretch` using the phase vocoder, preserving pitch.
///
/// `signal` is the mono input of length `n_samples`.  A factor `stretch > 1`
/// makes the output **longer** (slower), `stretch < 1` makes it **shorter**
/// (faster); `stretch == 1` reconstructs the input (up to STFT round-trip
/// error).  The output length is approximately `round(stretch · n_samples)`.
///
/// # Errors
/// Propagates `PhaseVocoderConfig::validate` errors and any STFT error.
pub fn phase_vocoder_stretch(
    signal: &[f64],
    n_samples: usize,
    stretch: f64,
    config: &PhaseVocoderConfig,
) -> AudioResult<Vec<f64>> {
    config.validate(stretch)?;

    let n_fft = config.n_fft;
    let hop_a = config.hop_analysis;
    let n_bins = n_fft / 2 + 1;

    // Synthesis hop: round(stretch · hop_analysis), at least 1.
    let hop_s = ((stretch * hop_a as f64).round() as usize).max(1);

    // Too-short signals (fewer than one full window) cannot be analysed.
    if n_samples < n_fft {
        return Ok(Vec::new());
    }

    // Analysis STFT (interleaved [Re, Im] per bin).
    let stft = stft_hann(signal, n_samples, n_fft, hop_a)?;
    let n_frames = (n_samples - n_fft) / hop_a + 1;
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    // Per-bin centre angular frequency ω_k = 2π k / N.
    let omega: Vec<f64> = (0..n_bins)
        .map(|k| 2.0 * PI * k as f64 / n_fft as f64)
        .collect();

    // Running analysis phase (previous frame) and accumulated synthesis phase.
    let mut prev_phase = vec![0.0_f64; n_bins];
    let mut synth_phase = vec![0.0_f64; n_bins];

    // Output STFT buffer (same frame count; only the hop changes on inversion).
    let mut out_stft = vec![0.0_f64; n_frames * n_bins * 2];

    for t in 0..n_frames {
        let base = t * n_bins * 2;
        for k in 0..n_bins {
            let re = stft[base + k * 2];
            let im = stft[base + k * 2 + 1];
            let mag = (re * re + im * im).sqrt();
            let phase = im.atan2(re);

            if t == 0 {
                // First frame: synthesis phase = analysis phase (identity start).
                synth_phase[k] = phase;
            } else {
                // Heterodyned phase increment: measured Δφ minus expected ω_k·H_a,
                // wrapped to the principal interval.
                let delta = phase - prev_phase[k] - omega[k] * hop_a as f64;
                let delta_wrapped = princ_arg(delta);
                // True instantaneous angular frequency of this bin's partial.
                let inst_freq = omega[k] + delta_wrapped / hop_a as f64;
                // Advance the synthesis phase by inst_freq over the synthesis hop.
                synth_phase[k] += inst_freq * hop_s as f64;
            }
            prev_phase[k] = phase;

            // Modified spectrum: original magnitude, propagated synthesis phase.
            out_stft[base + k * 2] = mag * synth_phase[k].cos();
            out_stft[base + k * 2 + 1] = mag * synth_phase[k].sin();
        }
    }

    // Overlap-add resynthesis at the synthesis hop.
    istft_hann(&out_stft, n_frames, n_fft, hop_s)
}

/// Resample `signal` to `out_len` samples by linear interpolation.
///
/// Used by [`pitch_shift`] to undo the time-stretch.  Linear interpolation is a
/// standard, artefact-light choice for the integer-ish ratios encountered in
/// pitch shifting.  Returns an empty vector if `signal` is empty or `out_len`
/// is `0`.
#[must_use]
pub fn resample_linear(signal: &[f64], out_len: usize) -> Vec<f64> {
    let in_len = signal.len();
    if in_len == 0 || out_len == 0 {
        return Vec::new();
    }
    if in_len == 1 {
        return vec![signal[0]; out_len];
    }
    let mut out = vec![0.0_f64; out_len];
    // Map output index j -> source position in [0, in_len-1].
    let scale = (in_len - 1) as f64 / (out_len - 1).max(1) as f64;
    for (j, o) in out.iter_mut().enumerate() {
        let src = j as f64 * scale;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(in_len - 1);
        let frac = src - i0 as f64;
        *o = signal[i0] * (1.0 - frac) + signal[i1] * frac;
    }
    out
}

/// Pitch-shift `signal` by a frequency factor `shift` while keeping its
/// duration, via *time-stretch by `shift` then resample by `1/shift`*.
///
/// `shift > 1` raises the pitch (e.g. `2.0` = one octave up), `shift < 1` lowers
/// it (`0.5` = one octave down).  The output has (approximately) the same length
/// as the input — every frequency component is multiplied by `shift`.
///
/// # Errors
/// Propagates `PhaseVocoderConfig::validate` errors (with `shift` as the
/// stretch) and any STFT error.
pub fn pitch_shift(
    signal: &[f64],
    n_samples: usize,
    shift: f64,
    config: &PhaseVocoderConfig,
) -> AudioResult<Vec<f64>> {
    config.validate(shift)?;
    if n_samples < config.n_fft {
        return Ok(Vec::new());
    }
    // 1. Time-stretch by `shift` (longer, same pitch).
    let stretched = phase_vocoder_stretch(signal, n_samples, shift, config)?;
    if stretched.is_empty() {
        return Ok(Vec::new());
    }
    // 2. Resample back to the original length: compresses time by `shift`, which
    //    scales every frequency by `shift` ⇒ pitch up by `shift`, duration ≈
    //    original.
    Ok(resample_linear(&stretched, n_samples))
}

/// Estimate the instantaneous frequency (in Hz) of every STFT bin between two
/// consecutive analysis frames of a signal.
///
/// This is the core measurement that drives [`phase_vocoder_stretch`], exposed
/// directly so callers (and tests) can verify that a pure sinusoid's dominant
/// bin reports its true frequency.  `sample_rate` is in Hz.
///
/// Returns one frequency per bin (`n_fft/2 + 1` values) computed from frames
/// `frame` and `frame + 1`.
///
/// # Errors
/// - [`AudioError::InvalidKernelSize`] / [`AudioError::InvalidStride`] for bad
///   FFT/hop parameters.
/// - [`AudioError::NonFinite`] if `sample_rate ≤ 0`.
/// - [`AudioError::EmptyInput`] if the signal has fewer than `frame + 2` frames.
pub fn instantaneous_frequency(
    signal: &[f64],
    n_samples: usize,
    sample_rate: f64,
    frame: usize,
    config: &PhaseVocoderConfig,
) -> AudioResult<Vec<f64>> {
    if config.n_fft < 2 {
        return Err(AudioError::InvalidKernelSize(config.n_fft));
    }
    if config.hop_analysis == 0 || config.hop_analysis > config.n_fft {
        return Err(AudioError::InvalidStride(config.hop_analysis));
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AudioError::NonFinite {
            msg: format!("sample_rate must be finite and positive, got {sample_rate}"),
        });
    }
    let n_fft = config.n_fft;
    let hop_a = config.hop_analysis;
    let n_bins = n_fft / 2 + 1;
    let n_frames = if n_samples < n_fft {
        0
    } else {
        (n_samples - n_fft) / hop_a + 1
    };
    if frame + 1 >= n_frames {
        return Err(AudioError::EmptyInput {
            msg: format!(
                "need frames {frame} and {} but only {n_frames} available",
                frame + 1
            ),
        });
    }
    let stft = stft_hann(signal, n_samples, n_fft, hop_a)?;
    let mut freqs = vec![0.0_f64; n_bins];
    for (k, f_out) in freqs.iter_mut().enumerate() {
        let base0 = frame * n_bins * 2 + k * 2;
        let base1 = (frame + 1) * n_bins * 2 + k * 2;
        let phase0 = stft[base0 + 1].atan2(stft[base0]);
        let phase1 = stft[base1 + 1].atan2(stft[base1]);
        let omega_k = 2.0 * PI * k as f64 / n_fft as f64;
        let delta = phase1 - phase0 - omega_k * hop_a as f64;
        let inst_omega = omega_k + princ_arg(delta) / hop_a as f64;
        // Convert angular frequency (rad/sample) to Hz.
        *f_out = inst_omega * sample_rate / (2.0 * PI);
    }
    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave of `freq` Hz at `sample_rate` over `n` samples.
    fn sine(freq: f64, sample_rate: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    /// Dominant FFT-bin frequency (Hz) of a signal via a single global DFT
    /// magnitude peak. Independent of the phase-vocoder machinery.
    ///
    /// To keep the O(N²) DFT bounded regardless of input length, the analysis is
    /// restricted to a centred window of at most 4096 samples — at 8 kHz that is
    /// ~2 Hz bin resolution, ample for the pitch checks here.
    fn dominant_frequency(signal: &[f64], sample_rate: f64) -> f64 {
        const MAX_WIN: usize = 4096;
        let full = signal.len();
        if full == 0 {
            return 0.0;
        }
        let window = full.min(MAX_WIN);
        let start = (full - window) / 2;
        let seg = &signal[start..start + window];
        let n = seg.len();
        let mut best_k = 0usize;
        let mut best_mag = -1.0_f64;
        // Only search up to Nyquist.
        for k in 1..n / 2 {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (idx, &x) in seg.iter().enumerate() {
                let ang = 2.0 * PI * k as f64 * idx as f64 / n as f64;
                re += x * ang.cos();
                im -= x * ang.sin();
            }
            let mag = re * re + im * im;
            if mag > best_mag {
                best_mag = mag;
                best_k = k;
            }
        }
        best_k as f64 * sample_rate / n as f64
    }

    /// Rate of sign changes per sample (a cheap pitch proxy).
    fn zero_crossing_rate(signal: &[f64]) -> f64 {
        if signal.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        for w in signal.windows(2) {
            if (w[0] >= 0.0) != (w[1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f64 / (signal.len() - 1) as f64
    }

    // (a) IDENTITY r = 1 reconstructs the input closely (small round-trip error
    //     on the steady interior of a windowed sinusoid).
    #[test]
    fn identity_stretch_reconstructs() {
        let sr = 8000.0;
        let freq = 440.0;
        let n = 4096;
        let cfg = PhaseVocoderConfig {
            n_fft: 1024,
            hop_analysis: 256,
        };
        let signal = sine(freq, sr, n);
        let out = phase_vocoder_stretch(&signal, n, 1.0, &cfg).expect("stretch");

        // Compare interior (skip a window at each edge for overlap-add warm-up).
        let trim = cfg.n_fft;
        let use_len = out.len().min(signal.len()).saturating_sub(2 * trim);
        assert!(use_len > 0, "no interior to compare");
        let mut max_err = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for i in trim..trim + use_len {
            max_err = max_err.max((signal[i] - out[i]).abs());
            max_abs = max_abs.max(signal[i].abs());
        }
        let rel = max_err / max_abs.max(1e-9);
        assert!(rel < 0.15, "identity round-trip rel error too large: {rel}");
    }

    // (b) Time-stretch by r changes output length by ≈ r while PRESERVING pitch:
    //     a pure tone's dominant frequency is unchanged after stretching.
    #[test]
    fn stretch_changes_length_preserves_pitch() {
        let sr = 8000.0;
        let freq = 500.0;
        let n = 6000;
        let cfg = PhaseVocoderConfig {
            n_fft: 1024,
            hop_analysis: 256,
        };
        let signal = sine(freq, sr, n);

        for &r in &[1.5_f64, 2.0, 0.75] {
            let out = phase_vocoder_stretch(&signal, n, r, &cfg).expect("stretch");
            // Output length should scale by ≈ r (within ~10% incl. windowing edge).
            let expected = (r * n as f64) as usize;
            let ratio = out.len() as f64 / expected as f64;
            assert!(
                (0.8..1.2).contains(&ratio),
                "length ratio {ratio} off for r={r} (got {}, expected ≈ {expected})",
                out.len()
            );

            // Pitch preserved: dominant frequency stays ≈ freq.
            // Analyse the steady interior only.
            let trim = cfg.n_fft;
            if out.len() > 3 * trim {
                let interior = &out[trim..out.len() - trim];
                let dom = dominant_frequency(interior, sr);
                let err = (dom - freq).abs() / freq;
                assert!(
                    err < 0.08,
                    "pitch drifted under stretch r={r}: dominant {dom} Hz vs {freq} Hz (rel {err})"
                );
            }
        }
    }

    // (c) PITCH-SHIFT by an octave ≈ doubles the dominant frequency.
    #[test]
    fn pitch_shift_octave_doubles_frequency() {
        let sr = 8000.0;
        let freq = 300.0;
        let n = 6000;
        let cfg = PhaseVocoderConfig {
            n_fft: 1024,
            hop_analysis: 256,
        };
        let signal = sine(freq, sr, n);

        // Up an octave: factor 2.0.
        let up = pitch_shift(&signal, n, 2.0, &cfg).expect("shift up");
        let trim = cfg.n_fft;
        assert!(up.len() > 3 * trim, "pitch-shift output too short");
        let interior = &up[trim..up.len() - trim];
        let dom = dominant_frequency(interior, sr);
        let err = (dom - 2.0 * freq).abs() / (2.0 * freq);
        assert!(
            err < 0.1,
            "octave-up dominant {dom} Hz should be ≈ {} Hz (rel {err})",
            2.0 * freq
        );

        // Length ≈ preserved.
        let len_ratio = up.len() as f64 / n as f64;
        assert!(
            (0.8..1.2).contains(&len_ratio),
            "pitch-shift changed duration too much: ratio {len_ratio}"
        );

        // Cross-check with the zero-crossing-rate proxy (doubles too).
        let zcr_in = zero_crossing_rate(&signal);
        let zcr_out = zero_crossing_rate(interior);
        assert!(
            zcr_out > 1.5 * zcr_in,
            "ZCR did not rise for octave up: in {zcr_in}, out {zcr_out}"
        );
    }

    // (d) The instantaneous-frequency estimate of a pure sinusoid equals its true
    //     frequency (within bin resolution).
    #[test]
    fn instantaneous_frequency_matches_truth() {
        let sr = 8000.0;
        let cfg = PhaseVocoderConfig {
            n_fft: 1024,
            hop_analysis: 256,
        };
        let n = 4096;
        let bin_hz = sr / cfg.n_fft as f64;

        for &freq in &[250.0_f64, 440.0, 1000.0] {
            let signal = sine(freq, sr, n);
            // Use a steady interior frame.
            let frame = 5;
            let freqs = instantaneous_frequency(&signal, n, sr, frame, &cfg).expect("inst freq");
            // The dominant bin is round(freq / bin_hz); its instantaneous-frequency
            // estimate must be within (much less than) one bin of the truth.
            let dom_bin = (freq / bin_hz).round() as usize;
            let est = freqs[dom_bin];
            let err = (est - freq).abs();
            assert!(
                err < bin_hz,
                "instantaneous frequency {est} Hz off from {freq} Hz by {err} (bin {bin_hz} Hz)"
            );
        }
    }

    // (e) Phase coherence on a steady tone: the stretched output has no large
    //     amplitude modulation (the envelope stays roughly flat in the interior).
    #[test]
    fn phase_coherence_no_amplitude_modulation() {
        let sr = 8000.0;
        let freq = 440.0;
        let n = 8000;
        let cfg = PhaseVocoderConfig {
            n_fft: 1024,
            hop_analysis: 256,
        };
        let signal = sine(freq, sr, n);
        let out = phase_vocoder_stretch(&signal, n, 2.0, &cfg).expect("stretch");

        // Compute a short-window RMS envelope over the interior and check that it
        // does not swing wildly (which would indicate phase-incoherent partials
        // beating against each other / the "phasiness" artefact in the extreme).
        let trim = cfg.n_fft;
        assert!(out.len() > 3 * trim);
        let interior = &out[trim..out.len() - trim];
        let win = 256usize;
        let mut envelopes = Vec::new();
        let mut i = 0;
        while i + win <= interior.len() {
            let rms = (interior[i..i + win].iter().map(|v| v * v).sum::<f64>() / win as f64).sqrt();
            envelopes.push(rms);
            i += win;
        }
        assert!(envelopes.len() > 4, "too few envelope windows");
        let mean: f64 = envelopes.iter().sum::<f64>() / envelopes.len() as f64;
        assert!(mean > 1e-6, "stretched tone is silent");
        let max = envelopes.iter().cloned().fold(0.0_f64, f64::max);
        let min = envelopes.iter().cloned().fold(f64::INFINITY, f64::min);
        // The peak-to-trough envelope swing should be moderate (a coherent tone
        // is near-constant; allow generous slack for window edges).
        let swing = (max - min) / mean;
        assert!(
            swing < 1.0,
            "excessive amplitude modulation (swing {swing}): min {min}, max {max}, mean {mean}"
        );
    }

    // (f) Invalid hop / overlap / FFT parameters error.
    #[test]
    fn invalid_parameters_error() {
        let sig = vec![0.0_f64; 64];
        // n_fft < 2.
        let bad_fft = PhaseVocoderConfig {
            n_fft: 1,
            hop_analysis: 1,
        };
        assert!(matches!(
            phase_vocoder_stretch(&sig, sig.len(), 1.0, &bad_fft),
            Err(AudioError::InvalidKernelSize(1))
        ));
        // hop == 0.
        let bad_hop = PhaseVocoderConfig {
            n_fft: 32,
            hop_analysis: 0,
        };
        assert!(matches!(
            phase_vocoder_stretch(&sig, sig.len(), 1.0, &bad_hop),
            Err(AudioError::InvalidStride(0))
        ));
        // hop > n_fft (no overlap).
        let no_overlap = PhaseVocoderConfig {
            n_fft: 32,
            hop_analysis: 64,
        };
        assert!(matches!(
            phase_vocoder_stretch(&sig, sig.len(), 1.0, &no_overlap),
            Err(AudioError::InvalidStride(64))
        ));
        // Non-positive / non-finite stretch.
        let ok_cfg = PhaseVocoderConfig {
            n_fft: 32,
            hop_analysis: 8,
        };
        assert!(matches!(
            phase_vocoder_stretch(&sig, sig.len(), 0.0, &ok_cfg),
            Err(AudioError::NonFinite { .. })
        ));
        assert!(matches!(
            phase_vocoder_stretch(&sig, sig.len(), f64::NAN, &ok_cfg),
            Err(AudioError::NonFinite { .. })
        ));
        // pitch_shift shares validation.
        assert!(matches!(
            pitch_shift(&sig, sig.len(), -1.0, &ok_cfg),
            Err(AudioError::NonFinite { .. })
        ));
        // instantaneous_frequency: bad sample rate and frame out of range.
        assert!(matches!(
            instantaneous_frequency(&sig, sig.len(), 0.0, 0, &ok_cfg),
            Err(AudioError::NonFinite { .. })
        ));
        let short = sine(440.0, 8000.0, 40);
        assert!(matches!(
            instantaneous_frequency(&short, short.len(), 8000.0, 100, &ok_cfg),
            Err(AudioError::EmptyInput { .. })
        ));
    }

    // princ_arg wraps into (−π, π].
    #[test]
    fn principal_argument_wraps() {
        assert!((princ_arg(0.0)).abs() < 1e-12);
        assert!((princ_arg(2.0 * PI)).abs() < 1e-9);
        assert!((princ_arg(3.0 * PI) - PI).abs() < 1e-9);
        let w = princ_arg(PI + 0.1);
        assert!(w <= PI && w > -PI, "wrapped value {w} out of (−π, π]");
        // A value just above π wraps to just above −π.
        assert!((princ_arg(PI + 0.001) - (-(PI - 0.001))).abs() < 1e-9);
    }

    // resample_linear changes length and (roughly) preserves a ramp's shape.
    #[test]
    fn resample_linear_basic() {
        // A linear ramp resampled to a different length stays a linear ramp.
        let ramp: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let up = resample_linear(&ramp, 199);
        assert_eq!(up.len(), 199);
        // Endpoints preserved.
        assert!((up[0] - 0.0).abs() < 1e-9);
        assert!((up[198] - 99.0).abs() < 1e-9);
        // Midpoint ≈ 49.5.
        assert!((up[99] - 49.5).abs() < 0.5, "midpoint {} off", up[99]);
        // Degenerate inputs.
        assert!(resample_linear(&[], 10).is_empty());
        assert!(resample_linear(&ramp, 0).is_empty());
        assert_eq!(resample_linear(&[5.0], 3), vec![5.0, 5.0, 5.0]);
    }

    // Signals shorter than one window produce an empty (well-defined) result.
    #[test]
    fn short_signal_returns_empty() {
        let cfg = PhaseVocoderConfig {
            n_fft: 256,
            hop_analysis: 64,
        };
        let sig = vec![1.0_f64; 100]; // < n_fft
        assert!(
            phase_vocoder_stretch(&sig, sig.len(), 2.0, &cfg)
                .expect("ok")
                .is_empty()
        );
        assert!(
            pitch_shift(&sig, sig.len(), 2.0, &cfg)
                .expect("ok")
                .is_empty()
        );
    }

    // Output is finite for a non-trivial broadband-ish signal.
    #[test]
    fn output_all_finite() {
        let sr = 8000.0;
        let n = 4096;
        let cfg = PhaseVocoderConfig {
            n_fft: 512,
            hop_analysis: 128,
        };
        // Sum of two tones.
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 220.0 * t).sin() + 0.5 * (2.0 * PI * 660.0 * t).sin()
            })
            .collect();
        let out = phase_vocoder_stretch(&signal, n, 1.3, &cfg).expect("stretch");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite sample");
        let shifted = pitch_shift(&signal, n, 1.5, &cfg).expect("shift");
        assert!(shifted.iter().all(|v| v.is_finite()), "non-finite sample");
    }
}
