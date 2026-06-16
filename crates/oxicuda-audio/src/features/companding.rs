//! Amplitude companding and (de-)emphasis filters.
//!
//! Two classic non-linear amplitude transforms used in telephony and neural
//! audio codecs (e.g. WaveNet's 8-bit μ-law quantisation):
//!
//! - **μ-law** (ITU-T G.711, North America / Japan):
//!   `F(x) = sign(x) · ln(1 + μ|x|) / ln(1 + μ)` with companding parameter
//!   `μ = 255` for 8-bit audio.
//! - **A-law** (ITU-T G.711, Europe): a piecewise log law with parameter
//!   `A = 87.6`.
//!
//! Both map a signal in `[−1, 1]` to `[−1, 1]` while allocating more resolution
//! to low-amplitude samples, and both are exactly invertible (expansion).
//! Helpers are provided to quantise the companded value to `q` levels (e.g.
//! 256) and to recover a float from a quantised index.
//!
//! The module also provides the **pre-emphasis** first-difference filter
//! `y_n = x_n − α·x_{n−1}` (a one-zero high-pass that flattens the ~−6 dB/oct
//! spectral tilt of voiced speech) and its exact inverse, **de-emphasis**
//! `y_n = x_n + α·y_{n−1}` (a one-pole leaky integrator).
//!
//! ## References
//! - ITU-T Recommendation G.711 (1988). "Pulse code modulation (PCM) of voice
//!   frequencies."

use crate::error::{AudioError, AudioResult};

/// Default μ-law companding parameter for 8-bit audio.
pub const MU_LAW_MU: f32 = 255.0;

/// Default A-law companding parameter.
pub const A_LAW_A: f32 = 87.6;

// ─── μ-law ────────────────────────────────────────────────────────────────────

/// Encode a single sample `x ∈ [−1, 1]` with the μ-law companding curve.
///
/// Out-of-range inputs are clamped to `[−1, 1]`. The output lies in `[−1, 1]`.
#[must_use]
pub fn mu_law_encode_sample(x: f32, mu: f32) -> f32 {
    let xc = x.clamp(-1.0, 1.0);
    let sign = if xc < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 + mu * xc.abs()).ln() / (1.0 + mu).ln()
}

/// Decode (expand) a single μ-law-companded sample `y ∈ [−1, 1]`.
#[must_use]
pub fn mu_law_decode_sample(y: f32, mu: f32) -> f32 {
    let yc = y.clamp(-1.0, 1.0);
    let sign = if yc < 0.0 { -1.0 } else { 1.0 };
    sign * ((1.0 + mu).powf(yc.abs()) - 1.0) / mu
}

/// μ-law-encode an entire signal (values clamped to `[−1, 1]`).
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `signal` is empty.
/// - [`AudioError::Internal`] if `mu ≤ 0`.
pub fn mu_law_encode(signal: &[f32], mu: f32) -> AudioResult<Vec<f32>> {
    validate(signal, mu, "mu_law")?;
    Ok(signal
        .iter()
        .map(|&x| mu_law_encode_sample(x, mu))
        .collect())
}

/// μ-law-decode (expand) an entire signal.
///
/// # Errors
/// As [`mu_law_encode`].
pub fn mu_law_decode(signal: &[f32], mu: f32) -> AudioResult<Vec<f32>> {
    validate(signal, mu, "mu_law")?;
    Ok(signal
        .iter()
        .map(|&y| mu_law_decode_sample(y, mu))
        .collect())
}

/// μ-law-encode and quantise a signal to integer indices in `[0, levels−1]`.
///
/// This is the standard WaveNet 8-bit front-end (`levels = 256`). The companded
/// value in `[−1, 1]` is affinely mapped to `[0, levels−1]` and rounded.
///
/// # Errors
/// As [`mu_law_encode`]; also [`AudioError::Internal`] if `levels < 2`.
pub fn mu_law_quantize(signal: &[f32], mu: f32, levels: usize) -> AudioResult<Vec<usize>> {
    validate(signal, mu, "mu_law")?;
    if levels < 2 {
        return Err(AudioError::Internal(format!(
            "mu_law: levels must be ≥ 2, got {levels}"
        )));
    }
    let scale = (levels - 1) as f32;
    Ok(signal
        .iter()
        .map(|&x| {
            let y = mu_law_encode_sample(x, mu); // [-1, 1]
            let idx = ((y + 1.0) * 0.5 * scale).round();
            idx.clamp(0.0, scale) as usize
        })
        .collect())
}

/// Recover a float signal in `[−1, 1]` from μ-law quantised indices.
///
/// Inverse of [`mu_law_quantize`].
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `indices` is empty.
/// - [`AudioError::Internal`] if `mu ≤ 0` or `levels < 2`.
pub fn mu_law_dequantize(indices: &[usize], mu: f32, levels: usize) -> AudioResult<Vec<f32>> {
    if indices.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "mu_law: empty indices".into(),
        });
    }
    if mu <= 0.0 {
        return Err(AudioError::Internal("mu_law: mu must be > 0".into()));
    }
    if levels < 2 {
        return Err(AudioError::Internal(format!(
            "mu_law: levels must be ≥ 2, got {levels}"
        )));
    }
    let scale = (levels - 1) as f32;
    Ok(indices
        .iter()
        .map(|&i| {
            let y = (i.min(levels - 1) as f32 / scale) * 2.0 - 1.0; // [-1, 1]
            mu_law_decode_sample(y, mu)
        })
        .collect())
}

// ─── A-law ────────────────────────────────────────────────────────────────────

/// Encode a single sample `x ∈ [−1, 1]` with the A-law companding curve.
#[must_use]
pub fn a_law_encode_sample(x: f32, a: f32) -> f32 {
    let xc = x.clamp(-1.0, 1.0);
    let sign = if xc < 0.0 { -1.0 } else { 1.0 };
    let abs = xc.abs();
    let denom = 1.0 + a.ln();
    let mag = if abs < 1.0 / a {
        a * abs / denom
    } else {
        (1.0 + (a * abs).ln()) / denom
    };
    sign * mag
}

/// Decode (expand) a single A-law-companded sample `y ∈ [−1, 1]`.
#[must_use]
pub fn a_law_decode_sample(y: f32, a: f32) -> f32 {
    let yc = y.clamp(-1.0, 1.0);
    let sign = if yc < 0.0 { -1.0 } else { 1.0 };
    let abs = yc.abs();
    let denom = 1.0 + a.ln();
    let threshold = 1.0 / denom; // value of F(1/A)
    let mag = if abs < threshold {
        abs * denom / a
    } else {
        ((abs * denom) - 1.0).exp() / a
    };
    sign * mag
}

/// A-law-encode an entire signal.
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `signal` is empty.
/// - [`AudioError::Internal`] if `a ≤ 1` (the law requires `A > 1`).
pub fn a_law_encode(signal: &[f32], a: f32) -> AudioResult<Vec<f32>> {
    validate_a_law(signal, a)?;
    Ok(signal.iter().map(|&x| a_law_encode_sample(x, a)).collect())
}

/// A-law-decode (expand) an entire signal.
///
/// # Errors
/// As [`a_law_encode`].
pub fn a_law_decode(signal: &[f32], a: f32) -> AudioResult<Vec<f32>> {
    validate_a_law(signal, a)?;
    Ok(signal.iter().map(|&y| a_law_decode_sample(y, a)).collect())
}

// ─── Pre-/de-emphasis ─────────────────────────────────────────────────────────

/// Apply a **pre-emphasis** first-difference high-pass filter:
/// `y_0 = x_0`, `y_n = x_n − α·x_{n−1}`.
///
/// `α` is typically `0.95`–`0.97`. The first sample is passed through
/// unchanged (zero initial condition on `x_{−1}`).
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `signal` is empty.
/// - [`AudioError::Internal`] if `α` is not in `[0, 1)`.
pub fn pre_emphasis(signal: &[f32], alpha: f32) -> AudioResult<Vec<f32>> {
    if signal.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "pre_emphasis: empty signal".into(),
        });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(AudioError::Internal(format!(
            "pre_emphasis: alpha must be in [0, 1), got {alpha}"
        )));
    }
    let mut out = vec![0.0_f32; signal.len()];
    out[0] = signal[0];
    for n in 1..signal.len() {
        out[n] = signal[n] - alpha * signal[n - 1];
    }
    Ok(out)
}

/// Apply **de-emphasis**, the exact inverse of [`pre_emphasis`]:
/// `y_0 = x_0`, `y_n = x_n + α·y_{n−1}` (a one-pole leaky integrator).
///
/// `de_emphasis(pre_emphasis(x, α), α) ≈ x` up to floating-point error.
///
/// # Errors
/// As [`pre_emphasis`].
pub fn de_emphasis(signal: &[f32], alpha: f32) -> AudioResult<Vec<f32>> {
    if signal.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "de_emphasis: empty signal".into(),
        });
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(AudioError::Internal(format!(
            "de_emphasis: alpha must be in [0, 1), got {alpha}"
        )));
    }
    let mut out = vec![0.0_f32; signal.len()];
    out[0] = signal[0];
    for n in 1..signal.len() {
        out[n] = signal[n] + alpha * out[n - 1];
    }
    Ok(out)
}

// ─── Validation helpers ───────────────────────────────────────────────────────

fn validate(signal: &[f32], param: f32, name: &str) -> AudioResult<()> {
    if signal.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: format!("{name}: empty signal"),
        });
    }
    if param <= 0.0 {
        return Err(AudioError::Internal(format!(
            "{name}: companding parameter must be > 0, got {param}"
        )));
    }
    Ok(())
}

fn validate_a_law(signal: &[f32], a: f32) -> AudioResult<()> {
    if signal.is_empty() {
        return Err(AudioError::EmptyInput {
            msg: "a_law: empty signal".into(),
        });
    }
    if a <= 1.0 {
        return Err(AudioError::Internal(format!(
            "a_law: parameter A must be > 1, got {a}"
        )));
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.8 * (2.0 * PI * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn mu_law_roundtrip_recovers_signal() {
        let sig = sine(440.0, 16_000.0, 256);
        let enc = mu_law_encode(&sig, MU_LAW_MU).expect("ok");
        let dec = mu_law_decode(&enc, MU_LAW_MU).expect("ok");
        for (a, b) in sig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn mu_law_preserves_zero_and_extremes() {
        assert!((mu_law_encode_sample(0.0, MU_LAW_MU)).abs() < 1e-7);
        assert!((mu_law_encode_sample(1.0, MU_LAW_MU) - 1.0).abs() < 1e-5);
        assert!((mu_law_encode_sample(-1.0, MU_LAW_MU) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn mu_law_output_in_unit_range() {
        let sig = sine(1000.0, 16_000.0, 256);
        let enc = mu_law_encode(&sig, MU_LAW_MU).expect("ok");
        assert!(enc.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn mu_law_expands_small_amplitudes() {
        // μ-law gives more codespace to small values: F(0.01) >> 0.01.
        let small = 0.01_f32;
        let companded = mu_law_encode_sample(small, MU_LAW_MU);
        assert!(companded > small * 5.0, "companded={companded}");
    }

    #[test]
    fn mu_law_quantize_range() {
        let sig = sine(500.0, 16_000.0, 256);
        let q = mu_law_quantize(&sig, MU_LAW_MU, 256).expect("ok");
        assert!(q.iter().all(|&i| i < 256));
        assert_eq!(q.len(), sig.len());
    }

    #[test]
    fn mu_law_quantize_dequantize_roundtrip() {
        let sig = sine(330.0, 16_000.0, 512);
        let q = mu_law_quantize(&sig, MU_LAW_MU, 256).expect("ok");
        let rec = mu_law_dequantize(&q, MU_LAW_MU, 256).expect("ok");
        // 8-bit μ-law quantisation error is small for moderate amplitudes.
        let mse: f32 = sig
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / sig.len() as f32;
        assert!(mse < 1e-3, "mse={mse}");
    }

    #[test]
    fn mu_law_quantize_levels_error() {
        let sig = sine(440.0, 16_000.0, 64);
        assert!(matches!(
            mu_law_quantize(&sig, MU_LAW_MU, 1).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn mu_law_empty_error() {
        assert!(matches!(
            mu_law_encode(&[], MU_LAW_MU).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn mu_law_bad_mu_error() {
        let sig = sine(440.0, 16_000.0, 64);
        assert!(matches!(
            mu_law_encode(&sig, 0.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn a_law_roundtrip_recovers_signal() {
        let sig = sine(440.0, 16_000.0, 256);
        let enc = a_law_encode(&sig, A_LAW_A).expect("ok");
        let dec = a_law_decode(&enc, A_LAW_A).expect("ok");
        for (a, b) in sig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn a_law_output_in_unit_range() {
        let sig = sine(800.0, 16_000.0, 256);
        let enc = a_law_encode(&sig, A_LAW_A).expect("ok");
        assert!(enc.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn a_law_preserves_sign() {
        assert!(a_law_encode_sample(0.5, A_LAW_A) > 0.0);
        assert!(a_law_encode_sample(-0.5, A_LAW_A) < 0.0);
        assert!((a_law_encode_sample(0.0, A_LAW_A)).abs() < 1e-6);
    }

    #[test]
    fn a_law_bad_param_error() {
        let sig = sine(440.0, 16_000.0, 64);
        assert!(matches!(
            a_law_encode(&sig, 1.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn a_law_empty_error() {
        assert!(matches!(
            a_law_decode(&[], A_LAW_A).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn pre_de_emphasis_roundtrip() {
        let sig = sine(440.0, 16_000.0, 512);
        let pre = pre_emphasis(&sig, 0.97).expect("ok");
        let rec = de_emphasis(&pre, 0.97).expect("ok");
        for (a, b) in sig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn pre_emphasis_first_sample_unchanged() {
        let sig = vec![0.3_f32, 0.5, -0.2, 0.1];
        let pre = pre_emphasis(&sig, 0.95).expect("ok");
        assert!((pre[0] - sig[0]).abs() < 1e-7);
        // y_1 = x_1 - 0.95 x_0
        assert!((pre[1] - (0.5 - 0.95 * 0.3)).abs() < 1e-6);
    }

    #[test]
    fn pre_emphasis_boosts_high_freq() {
        // Pre-emphasis is high-pass: a high tone keeps more energy than a low one.
        let fs = 16_000.0;
        let low = sine(200.0, fs, 1024);
        let high = sine(4000.0, fs, 1024);
        let pl = pre_emphasis(&low, 0.97).expect("ok");
        let ph = pre_emphasis(&high, 0.97).expect("ok");
        let el: f32 = pl.iter().map(|&x| x * x).sum();
        let eh: f32 = ph.iter().map(|&x| x * x).sum();
        assert!(eh > el, "high energy {eh} should exceed low {el}");
    }

    #[test]
    fn pre_emphasis_bad_alpha_error() {
        let sig = vec![0.1_f32; 8];
        assert!(matches!(
            pre_emphasis(&sig, 1.0).unwrap_err(),
            AudioError::Internal(_)
        ));
        assert!(matches!(
            pre_emphasis(&sig, -0.1).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn pre_emphasis_empty_error() {
        assert!(matches!(
            pre_emphasis(&[], 0.97).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn de_emphasis_empty_error() {
        assert!(matches!(
            de_emphasis(&[], 0.97).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn companding_deterministic() {
        let sig = sine(660.0, 16_000.0, 128);
        let a = mu_law_encode(&sig, MU_LAW_MU).expect("ok");
        let b = mu_law_encode(&sig, MU_LAW_MU).expect("ok");
        assert_eq!(a, b);
    }
}
