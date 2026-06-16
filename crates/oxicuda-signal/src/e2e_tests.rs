//! End-to-end integration tests for `oxicuda-signal`.
//!
//! Cross-module tests that chain together the newer DSP building blocks —
//! Parks-McClellan (Remez) equiripple FIR design, Welch / multitaper PSD
//! estimation, and polyphase rational resampling — verifying that they
//! interoperate correctly (e.g. design a filter, then confirm its stopband
//! with our own PSD estimator; resample a tone, then locate it with a
//! periodogram).

use std::f64::consts::PI;

use crate::filter::fir::fir_apply;
use crate::filter::remez::{magnitude_at, remez_lowpass};
use crate::resample::polyphase::resample_poly;
use crate::spectral::welch::{PsdScaling, periodogram, welch};
use crate::types::{PadMode, WindowType};

/// Deterministic LCG white-noise matching the crate-wide convention.
fn lcg_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (s >> 33) as f64 / (u32::MAX as f64);
        v.push((2.0 * u - 1.0) as f32);
    }
    v
}

fn integrate(freqs: &[f64], psd: &[f64], lo_hz: f64, hi_hz: f64) -> f64 {
    let df = freqs[1] - freqs[0];
    freqs
        .iter()
        .zip(psd.iter())
        .filter(|(f, _)| **f >= lo_hz && **f <= hi_hz)
        .map(|(_, p)| *p)
        .sum::<f64>()
        * df
}

#[test]
fn remez_lowpass_stopband_confirmed_by_welch_psd() {
    // 1. Design an equiripple lowpass (pass ≤ 0.18, stop ≥ 0.28 of fs).
    let h = remez_lowpass(63, 0.18, 0.28, 1.0, 4.0).expect("remez lowpass");
    let h64: Vec<f64> = h.iter().map(|&v| v as f64).collect();

    // 2. Filter white noise through it.
    let fs = 1.0;
    let n = 16384;
    let x = lcg_noise(n, 0xBEEF);
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y = fir_apply(&xf, &h64, PadMode::Zero).expect("fir apply");
    let yf: Vec<f32> = y.iter().map(|&v| v as f32).collect();

    // 3. Estimate the output PSD with Welch and confirm the stopband (f > 0.32)
    //    carries far less power per unit bandwidth than the passband (f < 0.15).
    let (freqs, psd) =
        welch(&yf, fs, 1024, 512, WindowType::Hann, PsdScaling::Density).expect("welch");
    let pass_power = integrate(&freqs, &psd, 0.02, 0.15);
    let stop_power = integrate(&freqs, &psd, 0.32, 0.5);
    let pass_bw = 0.15 - 0.02;
    let stop_bw = 0.5 - 0.32;
    let pass_density = pass_power / pass_bw;
    let stop_density = stop_power / stop_bw;
    assert!(
        stop_density < 0.05 * pass_density,
        "Remez stopband not suppressed: pass={pass_density}, stop={stop_density}"
    );
}

#[test]
fn resampled_tone_located_by_periodogram() {
    // Generate a 50 Hz tone at fs = 1000, resample to fs = 1500 (up=3, down=2),
    // then confirm a periodogram of the output peaks at 50 Hz on the new axis.
    let fs_in = 1000.0;
    let f0 = 50.0;
    let n = 3000;
    let x: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * f0 * i as f64 / fs_in).sin() as f32)
        .collect();

    let y = resample_poly(&x, 3, 2, None, None).expect("resample");
    let fs_out = fs_in * 3.0 / 2.0; // 1500 Hz

    let (freqs, psd) = periodogram(&y, fs_out, WindowType::Hann).expect("periodogram");
    let (peak_k, _) = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty psd");
    assert!(
        (freqs[peak_k] - f0).abs() < 5.0,
        "resampled tone peak at {} Hz, expected {f0} Hz",
        freqs[peak_k]
    );
}

#[test]
fn remez_filter_then_downsample_preserves_passband_tone() {
    // A two-tone signal (one in-band, one out-of-band for the post-decimation
    // Nyquist) is anti-alias filtered by a Remez lowpass, then decimated.  The
    // in-band tone should survive; the periodogram should peak at it.
    let fs = 4000.0;
    let n = 8000;
    let f_keep = 200.0; // well below the /2 decimated Nyquist (1000 Hz)
    let f_drop = 1500.0; // above it ⇒ would alias without filtering
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            ((2.0 * PI * f_keep * t).sin() + (2.0 * PI * f_drop * t).sin()) as f32
        })
        .collect();

    // Pre-filter with a Remez lowpass cutting at ~0.2 (of fs) before /2 decimation.
    let h = remez_lowpass(81, 0.2, 0.24, 1.0, 8.0).expect("remez");
    let h64: Vec<f64> = h.iter().map(|&v| v as f64).collect();
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let filt = fir_apply(&xf, &h64, PadMode::Zero).expect("fir");
    let filt_f: Vec<f32> = filt.iter().map(|&v| v as f32).collect();

    let y = resample_poly(&filt_f, 1, 2, None, None).expect("decimate");
    let fs_out = fs / 2.0;

    let (freqs, psd) =
        welch(&y, fs_out, 512, 256, WindowType::Hann, PsdScaling::Density).expect("welch");
    let (peak_k, _) = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty");
    assert!(
        (freqs[peak_k] - f_keep).abs() < 8.0,
        "kept tone peak at {} Hz, expected {f_keep} Hz",
        freqs[peak_k]
    );

    // The dropped tone's image must be negligible: integrate near where it would
    // alias (1500 Hz folds to |1500 − 2000| = 500 Hz at fs_out=2000 — but it is
    // removed by the pre-filter, so the band around 500 Hz should be quiet).
    let alias_power = integrate(&freqs, &psd, 450.0, 550.0);
    let keep_power = integrate(&freqs, &psd, 150.0, 250.0);
    assert!(
        alias_power < 0.05 * keep_power,
        "alias leakage {alias_power} too large vs kept {keep_power}"
    );
}

#[test]
fn remez_magnitude_matches_fir_apply_response() {
    // The analytic magnitude response of the Remez taps must agree with the
    // empirical response measured by feeding tones through `fir_apply`.
    let h = remez_lowpass(41, 0.2, 0.3, 1.0, 1.0).expect("remez");
    let h64: Vec<f64> = h.iter().map(|&v| v as f64).collect();

    for &f in &[0.05_f64, 0.1, 0.15] {
        // Steady-state tone; measure interior amplitude ratio.
        let n = 2000usize;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * f * i as f64).cos()).collect();
        let y = fir_apply(&x, &h64, PadMode::Zero).expect("fir");
        // Interior peak amplitude.
        let amp = y[n / 2..n - 100]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let analytic = magnitude_at(&h, f);
        assert!(
            (amp - analytic).abs() < 0.08,
            "f={f}: empirical {amp} vs analytic {analytic}"
        );
    }
}
