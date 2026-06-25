//! Goertzel algorithm — efficient single-bin (selective) DFT evaluation.
//!
//! The Goertzel algorithm evaluates one (or a handful of) DFT bins in `O(N)`
//! time per bin without the `O(N log N)` cost of a full FFT, and without
//! storing intermediate complex twiddle tables.  It is the standard technique
//! for tone detection (e.g. DTMF telephony decoding, FSK demodulation, single
//! frequency power estimation).
//!
//! ## Recurrence
//!
//! For a target normalised frequency `k = f · N / fs` (need not be integer for
//! the *generalised* Goertzel), define `ω = 2π k / N` and `coeff = 2 cos ω`.
//! The second-order IIR recurrence
//!
//! ```text
//! s[n] = x[n] + coeff · s[n-1] − s[n-2]
//! ```
//!
//! is run over the `N` input samples (with `s[-1] = s[-2] = 0`).  The complex
//! DFT value of that bin is then recovered from the two final states:
//!
//! ```text
//! y    = s[N-1] − e^{−jω} · s[N-2]
//! X[k] = e^{−jω(N−1)} · y           (linear-phase correction)
//! ```
//!
//! The phase correction makes `X[k]` equal the exact DFT coefficient
//! `Σ x[n] e^{−j 2π k n / N}`.  The (real) power of the bin is invariant under
//! that rotation and is computed directly from the states:
//! `|X[k]|² = s[N-1]² + s[N-2]² − coeff · s[N-1] · s[N-2]`.
//!
//! References:
//!   Goertzel (1958) "An algorithm for the evaluation of finite trigonometric
//!   series", Amer. Math. Monthly 65(1):34–35.
//!   Sysel & Rajmic (2012) "Goertzel algorithm generalized to non-integer
//!   multiples of fundamental frequency", EURASIP J. Adv. Signal Process.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::TAU;

/// Result of a single Goertzel bin evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoertzelBin {
    /// Real part of the complex DFT value `X[k]`.
    pub re: f64,
    /// Imaginary part of the complex DFT value `X[k]`.
    pub im: f64,
    /// Power `|X[k]|²` (computed directly from the recurrence states, which is
    /// numerically more robust than `re·re + im·im` for large `N`).
    pub power: f64,
}

impl GoertzelBin {
    /// Magnitude `|X[k]| = sqrt(power)`.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.power.max(0.0).sqrt()
    }

    /// Phase `arg(X[k])` in radians, in `(-π, π]`.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }
}

/// Generalised Goertzel: evaluate the DFT at an arbitrary (possibly
/// non-integer) bin index `k` of a length-`N` transform.
///
/// `k = f · N / fs` selects analog frequency `f`. Integer `k` reproduces the
/// exact DFT coefficient `X[k] = Σ x[n] e^{−j 2π k n / N}`; fractional `k`
/// gives the off-grid spectral value (generalised Goertzel of Sysel & Rajmic).
///
/// # Errors
/// Returns [`SignalError::InvalidSize`] if `signal` is empty.
pub fn goertzel(signal: &[f64], k: f64) -> SignalResult<GoertzelBin> {
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::InvalidSize(
            "Goertzel input must be non-empty".to_owned(),
        ));
    }
    let omega = TAU * k / n as f64;
    let (cos_w, sin_w) = (omega.cos(), omega.sin());
    let coeff = 2.0 * cos_w;

    // Second-order IIR recurrence — only the two most recent states are kept.
    let mut s_prev = 0.0_f64; // s[n-1]
    let mut s_prev2 = 0.0_f64; // s[n-2]
    for &x in signal {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    // Intermediate complex output: y = s[N-1] − e^{−jω} s[N-2].
    // The classic Goertzel recovers the DFT coefficient only up to a linear
    // phase factor e^{−jω(N−1)}; applying that correction makes `re`/`im` equal
    // the exact DFT value X[k] = Σ x[n] e^{−j 2π k n / N} (generalised Goertzel,
    // Sysel & Rajmic 2012). The power is invariant under this unit-modulus
    // rotation, so it is computed directly from the recurrence states.
    let y_re = s_prev - s_prev2 * cos_w;
    let y_im = s_prev2 * sin_w;
    let phase = -omega * (n as f64 - 1.0);
    let (cr, ci) = (phase.cos(), phase.sin());
    let re = y_re * cr - y_im * ci;
    let im = y_re * ci + y_im * cr;
    // Power directly from states (avoids catastrophic cancellation for big N).
    let power = s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2;

    Ok(GoertzelBin { re, im, power })
}

/// Convenience wrapper: evaluate the Goertzel bin for an analog frequency
/// `target_hz` of a signal sampled at `sample_rate_hz`.
///
/// The continuous frequency is mapped to the fractional bin
/// `k = target_hz · N / sample_rate_hz` and the generalised Goertzel is used,
/// so `target_hz` need not align with an exact DFT bin.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `sample_rate_hz <= 0`.
pub fn goertzel_hz(
    signal: &[f64],
    target_hz: f64,
    sample_rate_hz: f64,
) -> SignalResult<GoertzelBin> {
    if sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidParameter(
            "sample_rate_hz must be > 0".to_owned(),
        ));
    }
    let k = target_hz * signal.len() as f64 / sample_rate_hz;
    goertzel(signal, k)
}

/// Evaluate the Goertzel power at several analog frequencies at once.
///
/// Each requested frequency is run as an independent `O(N)` recurrence, so the
/// total cost is `O(N · M)` for `M` frequencies — cheaper than a full FFT when
/// `M ≪ log₂ N`.  Returns the power `|X|²` for each requested frequency, in the
/// same order.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `sample_rate_hz <= 0`.
pub fn goertzel_power_spectrum(
    signal: &[f64],
    target_hz: &[f64],
    sample_rate_hz: f64,
) -> SignalResult<Vec<f64>> {
    if sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidParameter(
            "sample_rate_hz must be > 0".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(target_hz.len());
    for &f in target_hz {
        out.push(goertzel_hz(signal, f, sample_rate_hz)?.power);
    }
    Ok(out)
}

/// Standard DTMF (dual-tone multi-frequency) row/column frequencies, in Hz.
///
/// The first four are the low ("row") group and the last four are the high
/// ("column") group of the telephone keypad.
pub const DTMF_FREQUENCIES_HZ: [f64; 8] =
    [697.0, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0, 1697.0];

/// Decode the most likely DTMF keypad symbol present in `signal`.
///
/// Runs Goertzel power detection on the eight DTMF tones, picks the strongest
/// low-group and high-group tone, and maps the pair to its keypad symbol.
/// Returns `Some(symbol)` only when both selected tones exceed
/// `relative_threshold` times the *total* DTMF energy (a simple twist-and-noise
/// guard) and dominate their own group.
///
/// # Errors
/// - [`SignalError::InvalidSize`] if `signal` is empty.
/// - [`SignalError::InvalidParameter`] if `sample_rate_hz <= 0` or
///   `relative_threshold` is not finite/positive.
pub fn dtmf_decode(
    signal: &[f64],
    sample_rate_hz: f64,
    relative_threshold: f64,
) -> SignalResult<Option<char>> {
    if !relative_threshold.is_finite() || relative_threshold <= 0.0 {
        return Err(SignalError::InvalidParameter(
            "relative_threshold must be finite and > 0".to_owned(),
        ));
    }
    let powers = goertzel_power_spectrum(signal, &DTMF_FREQUENCIES_HZ, sample_rate_hz)?;
    // Indices 0..4 = low group, 4..8 = high group.
    let (low_idx, low_pow) = arg_max(&powers[0..4]);
    let (high_idx_off, high_pow) = arg_max(&powers[4..8]);
    let total: f64 = powers.iter().sum();
    if total <= 0.0 {
        return Ok(None);
    }
    // Each chosen tone must carry a meaningful fraction of the total energy,
    // and must dominate the rest of its own group (reject mid-band noise).
    let low_group_total: f64 = powers[0..4].iter().sum();
    let high_group_total: f64 = powers[4..8].iter().sum();
    let low_ok = low_pow >= relative_threshold * total
        && low_pow >= 0.5 * low_group_total.max(f64::MIN_POSITIVE);
    let high_ok = high_pow >= relative_threshold * total
        && high_pow >= 0.5 * high_group_total.max(f64::MIN_POSITIVE);
    if low_ok && high_ok {
        Ok(Some(DTMF_KEYPAD[low_idx][high_idx_off]))
    } else {
        Ok(None)
    }
}

/// Keypad lookup indexed by `[low_group][high_group]`.
const DTMF_KEYPAD: [[char; 4]; 4] = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
];

/// Index and value of the maximum element of a non-empty slice.
fn arg_max(values: &[f64]) -> (usize, f64) {
    let mut best_idx = 0usize;
    let mut best_val = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Naïve DFT bin for cross-checking the Goertzel recurrence.
    fn dft_bin(signal: &[f64], k: usize) -> (f64, f64) {
        let n = signal.len();
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (idx, &x) in signal.iter().enumerate() {
            let ang = -TAU * k as f64 * idx as f64 / n as f64;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        (re, im)
    }

    #[test]
    fn test_empty_input_errors() {
        assert!(goertzel(&[], 1.0).is_err());
    }

    #[test]
    fn test_goertzel_matches_dft_integer_bin() {
        // Sum of two cosines; verify several integer bins match the naïve DFT.
        let n = 64usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 5.0 * i as f64 / n as f64).cos()
                    + 0.5 * (2.0 * PI * 11.0 * i as f64 / n as f64).cos()
            })
            .collect();
        for k in 0..n / 2 {
            let g = goertzel(&signal, k as f64).expect("goertzel ok");
            let (re, im) = dft_bin(&signal, k);
            assert!(
                (g.re - re).abs() < 1e-8 && (g.im - im).abs() < 1e-8,
                "bin {k}: goertzel=({},{}) dft=({re},{im})",
                g.re,
                g.im
            );
        }
    }

    #[test]
    fn test_goertzel_power_consistency() {
        // Power from states must equal re² + im².
        let n = 50usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 7.3 * i as f64 / n as f64).sin())
            .collect();
        let g = goertzel(&signal, 7.3).expect("goertzel ok");
        let p2 = g.re * g.re + g.im * g.im;
        assert!(
            (g.power - p2).abs() < 1e-6,
            "power={} re²+im²={p2}",
            g.power
        );
        assert!((g.magnitude() - p2.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_goertzel_peaks_at_tone() {
        // A pure tone should produce far more power at its bin than off-bin.
        let n = 128usize;
        let f_bin = 20.0_f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f_bin * i as f64 / n as f64).sin())
            .collect();
        let on = goertzel(&signal, f_bin).expect("on-bin").power;
        let off = goertzel(&signal, f_bin + 10.0).expect("off-bin").power;
        assert!(on > 100.0 * off, "on={on} off={off}");
    }

    #[test]
    fn test_goertzel_hz_mapping() {
        // 1 kHz tone at 8 kHz sample rate, N = 205 frame.
        let fs = 8000.0_f64;
        let n = 205usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let on = goertzel_hz(&signal, 1000.0, fs).expect("on").power;
        let off = goertzel_hz(&signal, 1500.0, fs).expect("off").power;
        assert!(on > 50.0 * off, "on={on} off={off}");
        assert!(goertzel_hz(&signal, 1000.0, -1.0).is_err());
    }

    #[test]
    fn test_goertzel_power_spectrum() {
        let n = 256usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 30.0 * i as f64 / n as f64).cos())
            .collect();
        // Sample three bins; the on-grid 30 should dominate.
        let p = goertzel_power_spectrum(&signal, &[10.0, 30.0, 50.0], n as f64).expect("ps");
        assert_eq!(p.len(), 3);
        assert!(p[1] > p[0] * 100.0 && p[1] > p[2] * 100.0);
    }

    #[test]
    fn test_dtmf_decode_digit_5() {
        // '5' = low 770 Hz + high 1336 Hz.
        let fs = 8000.0_f64;
        let n = 410usize; // ~51 ms frame
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 770.0 * t).sin() + (2.0 * PI * 1336.0 * t).sin()
            })
            .collect();
        let decoded = dtmf_decode(&signal, fs, 0.05).expect("decode ok");
        assert_eq!(decoded, Some('5'));
    }

    #[test]
    fn test_dtmf_decode_digit_9_and_hash() {
        let fs = 8000.0_f64;
        let n = 410usize;
        // '9' = 852 + 1477.
        let nine: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 852.0 * t).sin() + (2.0 * PI * 1477.0 * t).sin()
            })
            .collect();
        assert_eq!(dtmf_decode(&nine, fs, 0.05).expect("ok"), Some('9'));
        // '#' = 941 + 1477.
        let hash: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 941.0 * t).sin() + (2.0 * PI * 1477.0 * t).sin()
            })
            .collect();
        assert_eq!(dtmf_decode(&hash, fs, 0.05).expect("ok"), Some('#'));
    }

    #[test]
    fn test_dtmf_decode_silence_none() {
        let signal = vec![0.0_f64; 400];
        assert_eq!(dtmf_decode(&signal, 8000.0, 0.05).expect("ok"), None);
        assert!(dtmf_decode(&signal, 8000.0, -0.1).is_err());
    }
}
