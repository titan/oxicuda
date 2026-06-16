//! Linear Predictive Coding (LPC) and LPC-based formant estimation.
//!
//! Linear prediction models a sample as a linear combination of its `p`
//! predecessors, `x̂_n = Σ_{k=1}^{p} a_k · x_{n−k}`, by minimising the
//! short-time prediction error. The coefficients `a_k` are obtained from the
//! signal autocorrelation `r(0..=p)` via the **Levinson–Durbin** recursion,
//! which exploits the Toeplitz structure of the normal equations in `O(p²)`
//! time and additionally yields the reflection (PARCOR) coefficients and the
//! final prediction-error energy.
//!
//! From the LPC coefficients we can recover the **formants** — the resonant
//! frequencies of the vocal tract — as the angles of the roots of the LPC
//! polynomial `A(z) = 1 − Σ_k a_k z^{−k}` that lie inside the unit circle.
//! Roots are found by deflating a companion-matrix-free Durand–Kerner
//! (Weierstrass) iteration in complex arithmetic (no external dependencies).
//!
//! ## References
//! - Makhoul, J. (1975). "Linear prediction: A tutorial review." Proc. IEEE
//!   63(4), 561–580.
//! - Markel, J. D. & Gray, A. H. (1976). *Linear Prediction of Speech*.
//!   Springer.

use std::f32::consts::PI;

use crate::error::{AudioError, AudioResult};

// ─── Autocorrelation ──────────────────────────────────────────────────────────

/// Biased short-time autocorrelation `r(k) = Σ_{n} x_n · x_{n+k}` for
/// `k ∈ [0, max_lag]`.
///
/// The *biased* estimator (no `1/(N−k)` normalisation) is used because it
/// guarantees a positive-semidefinite Toeplitz matrix, which keeps the
/// Levinson recursion stable.
///
/// # Errors
/// - [`AudioError::EmptyInput`] if `signal` is empty.
/// - [`AudioError::InvalidSequenceLength`] if `max_lag >= signal.len()`.
pub fn autocorrelation(signal: &[f32], max_lag: usize) -> AudioResult<Vec<f32>> {
    let n = signal.len();
    if n == 0 {
        return Err(AudioError::EmptyInput {
            msg: "lpc: empty signal".into(),
        });
    }
    if max_lag >= n {
        return Err(AudioError::InvalidSequenceLength(max_lag));
    }
    let mut r = vec![0.0_f32; max_lag + 1];
    for (lag, r_v) in r.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for j in 0..(n - lag) {
            acc += signal[j] * signal[j + lag];
        }
        *r_v = acc;
    }
    Ok(r)
}

// ─── LPC via Levinson–Durbin ──────────────────────────────────────────────────

/// Result of an LPC analysis of one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct LpcResult {
    /// Prediction coefficients `a_1..a_p` (length `p`); the implicit `a_0 = 1`
    /// is *not* stored. The all-pole model is `1 / (1 − Σ a_k z^{−k})`.
    pub coeffs: Vec<f32>,
    /// Reflection (PARCOR) coefficients `k_1..k_p` (length `p`); each lies in
    /// `(−1, 1)` for a stable filter.
    pub reflection: Vec<f32>,
    /// Residual prediction-error energy after the final order.
    pub error: f32,
}

/// Compute order-`p` LPC coefficients from an autocorrelation sequence via the
/// Levinson–Durbin recursion.
///
/// `r` must have length `≥ p + 1`. Returns coefficients in the convention
/// `x̂_n = Σ_{k=1}^{p} a_k x_{n−k}` (so `coeffs[k-1] = a_k`).
///
/// # Errors
/// - [`AudioError::Internal`] if `p == 0` or `r.len() < p + 1`.
/// - [`AudioError::NonFinite`] if `r(0) ≤ 0` (silent / degenerate frame).
pub fn levinson_durbin(r: &[f32], p: usize) -> AudioResult<LpcResult> {
    if p == 0 {
        return Err(AudioError::Internal("lpc: order must be ≥ 1".into()));
    }
    if r.len() < p + 1 {
        return Err(AudioError::Internal(format!(
            "lpc: autocorrelation length {} < order+1 {}",
            r.len(),
            p + 1
        )));
    }
    if r[0] <= 0.0 {
        return Err(AudioError::NonFinite {
            msg: "lpc: non-positive zero-lag autocorrelation (silent frame)".into(),
        });
    }

    let mut a = vec![0.0_f32; p + 1]; // a[0] = 1 implicit; a[1..=p] are the LPC coeffs
    let mut reflection = vec![0.0_f32; p];
    let mut err = r[0];

    for i in 1..=p {
        // Reflection coefficient k_i = −(r_i + Σ_{j=1}^{i−1} a_j r_{i−j}) / err.
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = if err.abs() > 1e-12 { -acc / err } else { 0.0 };
        reflection[i - 1] = k;

        // Update coefficients in place using the symmetric update.
        let half = i / 2;
        for j in 1..=half {
            let tmp = a[j] + k * a[i - j];
            a[i - j] += k * a[j];
            a[j] = tmp;
        }
        a[i] = k;

        err *= 1.0 - k * k;
        if err <= 0.0 {
            // Degenerate update: clamp to a tiny positive energy and stop refining.
            err = err.max(1e-12);
        }
    }

    // Convert sign convention: our recursion stored A(z) = 1 + Σ a_k z^{−k};
    // the prediction form x̂_n = Σ b_k x_{n−k} uses b_k = −a_k.
    let coeffs: Vec<f32> = a[1..=p].iter().map(|&v| -v).collect();
    Ok(LpcResult {
        coeffs,
        reflection,
        error: err,
    })
}

/// Convenience: window the frame with a Hamming taper and compute order-`p` LPC
/// coefficients directly from the time-domain samples.
///
/// # Errors
/// As [`autocorrelation`] and [`levinson_durbin`]; also
/// [`AudioError::InvalidSequenceLength`] if `frame.len() <= p`.
pub fn lpc(frame: &[f32], p: usize) -> AudioResult<LpcResult> {
    if p == 0 {
        return Err(AudioError::Internal("lpc: order must be ≥ 1".into()));
    }
    let n = frame.len();
    if n <= p {
        return Err(AudioError::InvalidSequenceLength(n));
    }
    // Hamming window for spectral leakage suppression.
    let windowed: Vec<f32> = frame
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
            x * w
        })
        .collect();
    let r = autocorrelation(&windowed, p)?;
    levinson_durbin(&r, p)
}

// ─── Complex helper for root finding ──────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Cplx {
    re: f32,
    im: f32,
}

impl Cplx {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        if d < 1e-30 {
            return Self::new(0.0, 0.0);
        }
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    #[inline]
    fn abs(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    #[inline]
    fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }
}

/// Evaluate the monic polynomial with coefficients `c` (highest degree first)
/// at point `z` via Horner's scheme.
fn poly_eval(c: &[Cplx], z: Cplx) -> Cplx {
    let mut acc = Cplx::new(0.0, 0.0);
    for &ci in c {
        acc = acc.mul(z).add(ci);
    }
    acc
}

/// Find all complex roots of a real polynomial with coefficients `coeffs`
/// (highest degree first) via the Durand–Kerner (Weierstrass) iteration.
fn durand_kerner(coeffs: &[f32]) -> Vec<Cplx> {
    // Strip leading zeros and normalise to monic.
    let mut start = 0;
    while start < coeffs.len() && coeffs[start].abs() < 1e-20 {
        start += 1;
    }
    let trimmed = &coeffs[start..];
    if trimmed.len() < 2 {
        return Vec::new();
    }
    let lead = trimmed[0];
    let monic: Vec<Cplx> = trimmed.iter().map(|&c| Cplx::new(c / lead, 0.0)).collect();
    let degree = monic.len() - 1;

    // Initial guesses on a spiral around the unit circle (classic seed).
    let seed = Cplx::new(0.4, 0.9);
    let mut roots: Vec<Cplx> = (0..degree)
        .map(|i| {
            let mut z = Cplx::new(1.0, 0.0);
            for _ in 0..i {
                z = z.mul(seed);
            }
            z
        })
        .collect();

    for _ in 0..200 {
        let mut max_delta = 0.0_f32;
        for i in 0..degree {
            let num = poly_eval(&monic, roots[i]);
            let mut den = Cplx::new(1.0, 0.0);
            for j in 0..degree {
                if j != i {
                    den = den.mul(roots[i].sub(roots[j]));
                }
            }
            let delta = num.div(den);
            roots[i] = roots[i].sub(delta);
            max_delta = max_delta.max(delta.abs());
        }
        if max_delta < 1e-7 {
            break;
        }
    }
    roots
}

// ─── Formant estimation ───────────────────────────────────────────────────────

/// A single estimated formant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formant {
    /// Centre frequency in Hz.
    pub frequency: f32,
    /// −3 dB bandwidth in Hz (derived from the root radius).
    pub bandwidth: f32,
}

/// Estimate vocal-tract **formants** from LPC coefficients.
///
/// The LPC polynomial `A(z) = 1 − Σ a_k z^{−k}` (here represented with
/// highest-degree-first coefficients `[1, −a_1, …, −a_p]`) has roots that come
/// in complex-conjugate pairs. Each pair with positive imaginary part and a
/// radius inside the unit circle maps to a formant:
/// `F = (fs / 2π) · arg(z)`,  `BW = −(fs / π) · ln|z|`.
///
/// Formants are returned sorted by ascending frequency; only those above
/// `min_freq` Hz and with bandwidth below `max_bandwidth` Hz are kept.
///
/// # Errors
/// - [`AudioError::Internal`] if `sample_rate ≤ 0` or `coeffs` is empty.
pub fn formants_from_lpc(
    coeffs: &[f32],
    sample_rate: f32,
    min_freq: f32,
    max_bandwidth: f32,
) -> AudioResult<Vec<Formant>> {
    if sample_rate <= 0.0 {
        return Err(AudioError::Internal(format!(
            "formants: sample_rate must be > 0, got {sample_rate}"
        )));
    }
    if coeffs.is_empty() {
        return Err(AudioError::Internal("formants: empty LPC coeffs".into()));
    }
    // Build A(z) coefficients (highest degree first): [1, -a_1, ..., -a_p].
    let mut poly = Vec::with_capacity(coeffs.len() + 1);
    poly.push(1.0_f32);
    for &a in coeffs {
        poly.push(-a);
    }
    let roots = durand_kerner(&poly);

    let mut formants = Vec::new();
    for z in roots {
        // Take only the upper half-plane to avoid duplicating conjugate pairs.
        if z.im <= 0.0 {
            continue;
        }
        let radius = z.abs();
        if !(1e-6..1.0).contains(&radius) {
            continue; // outside the unit circle / numerical junk
        }
        let freq = sample_rate * z.arg() / (2.0 * PI);
        let bw = -sample_rate * radius.ln() / PI;
        if freq < min_freq || bw > max_bandwidth || !freq.is_finite() || !bw.is_finite() {
            continue;
        }
        formants.push(Formant {
            frequency: freq,
            bandwidth: bw,
        });
    }
    formants.sort_by(|a, b| {
        a.frequency
            .partial_cmp(&b.frequency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(formants)
}

/// End-to-end formant estimation directly from a time-domain frame.
///
/// Computes order-`p` LPC (Hamming-windowed) and extracts formants. A typical
/// order is `p ≈ 2 + sample_rate / 1000` (≈ one pole pair per kHz plus two for
/// spectral shaping).
///
/// # Errors
/// As [`lpc`] and [`formants_from_lpc`].
pub fn formants(
    frame: &[f32],
    sample_rate: f32,
    order: usize,
    min_freq: f32,
    max_bandwidth: f32,
) -> AudioResult<Vec<Formant>> {
    let lpc_result = lpc(frame, order)?;
    formants_from_lpc(&lpc_result.coeffs, sample_rate, min_freq, max_bandwidth)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI_F32;

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / fs).sin())
            .collect()
    }

    /// Drive a two-formant all-pole system with an impulse train to get a
    /// speech-like vowel signal whose formants are known.
    fn synth_two_formants(f1: f32, f2: f32, fs: f32, n: usize) -> Vec<f32> {
        // Each resonator: y_n = x_n + 2 r cos(ω) y_{n−1} − r² y_{n−2}.
        let make_res = |f: f32, r: f32| {
            let w = 2.0 * PI_F32 * f / fs;
            (2.0 * r * w.cos(), -r * r)
        };
        let (a1a, a1b) = make_res(f1, 0.97);
        let (a2a, a2b) = make_res(f2, 0.97);
        // Impulse train at 120 Hz fundamental.
        let period = (fs / 120.0).round() as usize;
        let mut excitation = vec![0.0_f32; n];
        let mut i = 0;
        while i < n {
            excitation[i] = 1.0;
            i += period;
        }
        let mut y1 = vec![0.0_f32; n];
        for k in 0..n {
            let p1 = if k >= 1 { y1[k - 1] } else { 0.0 };
            let p2 = if k >= 2 { y1[k - 2] } else { 0.0 };
            y1[k] = excitation[k] + a1a * p1 + a1b * p2;
        }
        let mut y2 = vec![0.0_f32; n];
        for k in 0..n {
            let p1 = if k >= 1 { y2[k - 1] } else { 0.0 };
            let p2 = if k >= 2 { y2[k - 2] } else { 0.0 };
            y2[k] = y1[k] + a2a * p1 + a2b * p2;
        }
        // Normalise to avoid huge magnitudes.
        let peak = y2.iter().fold(0.0_f32, |m, &v| m.max(v.abs())).max(1e-6);
        y2.iter().map(|&v| v / peak).collect()
    }

    #[test]
    fn autocorr_lag0_is_energy() {
        let sig = sine(440.0, 16_000.0, 512);
        let r = autocorrelation(&sig, 16).expect("ok");
        let energy: f32 = sig.iter().map(|&x| x * x).sum();
        assert!(
            (r[0] - energy).abs() < 1e-2,
            "r0={} energy={}",
            r[0],
            energy
        );
    }

    #[test]
    fn autocorr_symmetric_peak_at_zero() {
        let sig = sine(440.0, 16_000.0, 512);
        let r = autocorrelation(&sig, 32).expect("ok");
        // Zero-lag is the maximum of |r|.
        assert!(r.iter().all(|&v| v.abs() <= r[0] + 1e-3));
    }

    #[test]
    fn autocorr_empty_error() {
        assert!(matches!(
            autocorrelation(&[], 4).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn autocorr_lag_too_large_error() {
        let sig = vec![1.0_f32; 8];
        assert!(matches!(
            autocorrelation(&sig, 8).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn levinson_reflection_in_unit_range() {
        let sig = synth_two_formants(700.0, 1800.0, 16_000.0, 1024);
        let r = autocorrelation(&sig, 12).expect("ok");
        let res = levinson_durbin(&r, 12).expect("ok");
        assert_eq!(res.coeffs.len(), 12);
        assert_eq!(res.reflection.len(), 12);
        // A stable all-pole filter has |k_i| < 1.
        for (i, &k) in res.reflection.iter().enumerate() {
            assert!(k.abs() < 1.0 + 1e-3, "reflection[{i}]={k} not in (-1,1)");
        }
    }

    #[test]
    fn levinson_error_decreases() {
        let sig = synth_two_formants(650.0, 1700.0, 16_000.0, 1024);
        let r = autocorrelation(&sig, 14).expect("ok");
        // Residual error must be positive and no larger than r(0).
        let res = levinson_durbin(&r, 14).expect("ok");
        assert!(res.error > 0.0 && res.error <= r[0] + 1e-3);
    }

    #[test]
    fn levinson_predicts_signal() {
        // The LPC predictor should reconstruct a frame with small residual.
        let sig = synth_two_formants(600.0, 1600.0, 16_000.0, 1024);
        let p = 12;
        let res = lpc(&sig, p).expect("ok");
        // Mean-squared prediction error should be much smaller than signal power.
        let mut sse = 0.0_f32;
        let mut power = 0.0_f32;
        for n in p..sig.len() {
            let mut pred = 0.0_f32;
            for (k, &a) in res.coeffs.iter().enumerate() {
                pred += a * sig[n - 1 - k];
            }
            let e = sig[n] - pred;
            sse += e * e;
            power += sig[n] * sig[n];
        }
        assert!(sse < 0.5 * power, "sse={sse} power={power}");
    }

    #[test]
    fn levinson_zero_order_error() {
        let r = vec![1.0_f32, 0.5, 0.2];
        assert!(matches!(
            levinson_durbin(&r, 0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn levinson_short_autocorr_error() {
        let r = vec![1.0_f32, 0.5];
        assert!(matches!(
            levinson_durbin(&r, 5).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn levinson_silent_frame_error() {
        let r = vec![0.0_f32, 0.0, 0.0];
        assert!(matches!(
            levinson_durbin(&r, 2).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
    }

    #[test]
    fn lpc_short_frame_error() {
        let frame = vec![0.1_f32; 4];
        assert!(matches!(
            lpc(&frame, 8).unwrap_err(),
            AudioError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn formants_recover_synthetic_resonances() {
        // Synthesise a vowel with formants at 700 / 1800 Hz; LPC should recover them.
        let fs = 16_000.0;
        let sig = synth_two_formants(700.0, 1800.0, fs, 2048);
        let f = formants(&sig, fs, 12, 90.0, 600.0).expect("ok");
        assert!(f.len() >= 2, "expected ≥2 formants, got {}", f.len());
        // The two lowest formants should be near the synthesis targets.
        let f1 = f[0].frequency;
        let f2 = f[1].frequency;
        assert!((f1 - 700.0).abs() < 150.0, "F1={f1}");
        assert!((f2 - 1800.0).abs() < 200.0, "F2={f2}");
    }

    #[test]
    fn formants_sorted_ascending() {
        let fs = 16_000.0;
        let sig = synth_two_formants(500.0, 2200.0, fs, 2048);
        let f = formants(&sig, fs, 14, 90.0, 700.0).expect("ok");
        for w in f.windows(2) {
            assert!(w[0].frequency <= w[1].frequency);
        }
    }

    #[test]
    fn formants_positive_bandwidth() {
        let fs = 16_000.0;
        let sig = synth_two_formants(800.0, 1500.0, fs, 2048);
        let f = formants(&sig, fs, 12, 90.0, 800.0).expect("ok");
        assert!(
            f.iter()
                .all(|fm| fm.bandwidth >= 0.0 && fm.frequency >= 90.0)
        );
    }

    #[test]
    fn formants_bad_sample_rate_error() {
        assert!(matches!(
            formants_from_lpc(&[0.1, 0.2], 0.0, 90.0, 600.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn formants_empty_coeffs_error() {
        assert!(matches!(
            formants_from_lpc(&[], 16_000.0, 90.0, 600.0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn formants_within_nyquist() {
        let fs = 16_000.0;
        let sig = synth_two_formants(600.0, 2000.0, fs, 2048);
        let f = formants(&sig, fs, 12, 90.0, 600.0).expect("ok");
        assert!(f.iter().all(|fm| fm.frequency <= fs / 2.0 + 1.0));
    }

    #[test]
    fn lpc_deterministic() {
        let sig = synth_two_formants(700.0, 1800.0, 16_000.0, 1024);
        let a = lpc(&sig, 12).expect("ok");
        let b = lpc(&sig, 12).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn lpc_coeffs_finite() {
        let sig = synth_two_formants(550.0, 1650.0, 16_000.0, 1024);
        let res = lpc(&sig, 16).expect("ok");
        assert!(res.coeffs.iter().all(|v| v.is_finite()));
        assert!(res.error.is_finite() && res.error > 0.0);
    }
}
