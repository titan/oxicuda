//! Butterworth IIR filter design via the bilinear transform.
//!
//! Designs a general-order Butterworth low-pass or high-pass digital filter and
//! returns its transfer function as flat numerator (`b`) and denominator (`a`)
//! coefficient vectors, i.e.
//!
//! ```text
//! H(z) = (b[0] + b[1] z⁻¹ + … + b[N] z⁻ᴺ) /
//!        (a[0] + a[1] z⁻¹ + … + a[N] z⁻ᴺ),   a[0] = 1.
//! ```
//!
//! # Method
//!
//! 1. Place the `N` analog Butterworth poles on the unit circle in the left
//!    half plane: `s_k = exp(i π (2k + N + 1) / (2N))`.
//! 2. Pre-warp the cutoff: `Ω_c = tan(π f_c / f_s)` (bilinear frequency map).
//! 3. Scale the analog prototype to the cutoff (low-pass) or apply the
//!    low-pass → high-pass transform `s → Ω_c / s` (high-pass).
//! 4. Apply the bilinear transform `s = (1 − z⁻¹)/(1 + z⁻¹)` to every pole and
//!    to the zeros (at `s = ∞` for low-pass → `z = −1`; at `s = 0` for
//!    high-pass → `z = +1`).
//! 5. Expand the factored form into polynomial coefficients and normalise the
//!    DC (low-pass) or Nyquist (high-pass) gain to unity.
//!
//! Unlike the second-order-section design in [`crate::filter::iir`], this module
//! yields a single flat difference equation evaluated in Direct Form II.

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

/// Butterworth response type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Low-pass: passes frequencies below the cutoff.
    LowPass,
    /// High-pass: passes frequencies above the cutoff.
    HighPass,
}

/// Configuration for a Butterworth filter design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButterworthConfig {
    /// Filter order `N >= 1`.
    pub order: usize,
    /// Cutoff frequency in the same units as `fs` (must satisfy `0 < cutoff < fs/2`).
    pub cutoff: f64,
    /// Sampling frequency (must be positive).
    pub fs: f64,
    /// Low-pass or high-pass.
    pub filter_type: FilterType,
}

/// A designed Butterworth filter as flat transfer-function coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct ButterworthFilter {
    /// Numerator coefficients `[b0, b1, …, bN]`.
    b: Vec<f64>,
    /// Denominator coefficients `[1, a1, …, aN]` (`a[0]` normalised to 1).
    a: Vec<f64>,
}

/// Minimal complex helper (avoids pulling extra deps for the design math).
#[derive(Clone, Copy)]
struct Cpx {
    re: f64,
    im: f64,
}

impl Cpx {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Cpx) -> Cpx {
        Cpx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cpx) -> Cpx {
        Cpx::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Cpx) -> Cpx {
        Cpx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: Cpx) -> Cpx {
        let d = o.re * o.re + o.im * o.im;
        Cpx::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn scale(self, s: f64) -> Cpx {
        Cpx::new(self.re * s, self.im * s)
    }
}

impl ButterworthFilter {
    /// Designs a Butterworth filter from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::InvalidParameter`] if `order == 0`, `fs <= 0`, or
    /// the cutoff is not strictly inside `(0, fs/2)`.
    pub fn design(config: &ButterworthConfig) -> SignalResult<Self> {
        let n = config.order;
        if n == 0 {
            return Err(SignalError::InvalidParameter(
                "Butterworth order must be >= 1".into(),
            ));
        }
        if config.fs <= 0.0 || !config.fs.is_finite() {
            return Err(SignalError::InvalidParameter(format!(
                "sampling frequency must be positive, got {}",
                config.fs
            )));
        }
        let nyquist = config.fs / 2.0;
        if config.cutoff <= 0.0 || config.cutoff >= nyquist || !config.cutoff.is_finite() {
            return Err(SignalError::InvalidParameter(format!(
                "cutoff {} must satisfy 0 < cutoff < fs/2 = {nyquist}",
                config.cutoff
            )));
        }

        // Pre-warp cutoff to analog frequency (bilinear).
        let omega_c = (PI * config.cutoff / config.fs).tan();

        // Analog Butterworth poles on the left-half unit circle, scaled to ω_c.
        // s_k = ω_c · exp(i·θ_k), θ_k = π(2k + N + 1)/(2N).
        let mut analog_poles = Vec::with_capacity(n);
        for k in 0..n {
            let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
            let p = Cpx::new(theta.cos(), theta.sin()).scale(omega_c);
            analog_poles.push(p);
        }

        // For high-pass, transform s → ω_c²/s (low-pass prototype already has
        // gain ω_c^N built in; recompute via the standard lp→hp pole map
        // p_hp = ω_c / p_lp using the *unscaled* prototype). Re-derive poles on
        // the unit circle then map.
        let (poles, zeros): (Vec<Cpx>, Vec<Cpx>) = match config.filter_type {
            FilterType::LowPass => {
                // Low-pass zeros all at s = ∞ → after bilinear they map to z = −1.
                (analog_poles, Vec::new())
            }
            FilterType::HighPass => {
                // Unit-circle prototype poles (ω_c = 1), then hp map p → ω_c/p.
                let mut hp_poles = Vec::with_capacity(n);
                for k in 0..n {
                    let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
                    let proto = Cpx::new(theta.cos(), theta.sin());
                    hp_poles.push(Cpx::new(omega_c, 0.0).div(proto));
                }
                // High-pass zeros all at s = 0 → after bilinear map to z = +1.
                (hp_poles, vec![Cpx::new(0.0, 0.0); n])
            }
        };

        // Bilinear transform each analog pole / zero: z = (1 + s) / (1 − s).
        let one = Cpx::new(1.0, 0.0);
        let z_poles: Vec<Cpx> = poles.iter().map(|&s| one.add(s).div(one.sub(s))).collect();

        // Discrete zeros:
        //  - low-pass: N zeros at z = −1
        //  - high-pass: zeros are the bilinear image of s = 0 → z = +1
        let z_zeros: Vec<Cpx> = match config.filter_type {
            FilterType::LowPass => vec![Cpx::new(-1.0, 0.0); n],
            FilterType::HighPass => zeros.iter().map(|&s| one.add(s).div(one.sub(s))).collect(),
        };

        // Expand factored polynomials ∏ (1 − r z⁻¹) into coefficient vectors.
        let mut b = poly_from_roots(&z_zeros);
        let a = poly_from_roots(&z_poles);

        // Normalise gain: low-pass at DC (z = 1, evaluate at ω = 0), high-pass
        // at Nyquist (z = −1).
        let eval_point = match config.filter_type {
            FilterType::LowPass => 1.0,
            FilterType::HighPass => -1.0,
        };
        let num_gain = poly_eval_at(&b, eval_point);
        let den_gain = poly_eval_at(&a, eval_point);
        let target = den_gain / num_gain;
        for c in &mut b {
            *c *= target;
        }

        Ok(Self { b, a })
    }

    /// Returns the numerator coefficients `[b0 … bN]`.
    #[must_use]
    pub fn numerator(&self) -> &[f64] {
        &self.b
    }

    /// Returns the denominator coefficients `[1, a1 … aN]`.
    #[must_use]
    pub fn denominator(&self) -> &[f64] {
        &self.a
    }

    /// Applies the filter to `signal` using the Direct-Form-II difference
    /// equation, returning an output of the same length.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::InvalidParameter`] if the filter coefficients are
    /// empty (which cannot happen for a filter produced by [`Self::design`]).
    pub fn apply(&self, signal: &[f64]) -> SignalResult<Vec<f64>> {
        let order =
            self.a.len().checked_sub(1).ok_or_else(|| {
                SignalError::InvalidParameter("filter has no coefficients".into())
            })?;
        if self.b.len() != self.a.len() {
            return Err(SignalError::InvalidParameter(
                "numerator/denominator length mismatch".into(),
            ));
        }
        // Direct Form II: w[n] = x[n] − Σ a_i w[n-i]; y[n] = Σ b_i w[n-i].
        // `w` holds [w[n-1], w[n-2], …, w[n-order]] (most recent first).
        let mut w = vec![0.0_f64; order];
        let mut out = Vec::with_capacity(signal.len());
        for &x in signal {
            // w[n] = x − Σ_{i=1..order} a_i · w[n-i]
            let feedback: f64 = self.a[1..]
                .iter()
                .zip(w.iter())
                .map(|(&ai, &wi)| ai * wi)
                .sum();
            let wn = x - feedback;
            // y[n] = b_0·w[n] + Σ_{i=1..order} b_i · w[n-i]
            let feedforward: f64 = self.b[1..]
                .iter()
                .zip(w.iter())
                .map(|(&bi, &wi)| bi * wi)
                .sum();
            let yn = self.b[0] * wn + feedforward;
            out.push(yn);
            // Shift the delay line: drop the oldest, prepend w[n].
            if order > 0 {
                w.copy_within(0..order - 1, 1);
                w[0] = wn;
            }
        }
        Ok(out)
    }
}

/// Expands `∏_j (1 − r_j z⁻¹)` for complex roots `r_j` into real coefficients.
/// Conjugate-pair roots make the imaginary parts cancel up to round-off, which
/// is then discarded.
fn poly_from_roots(roots: &[Cpx]) -> Vec<f64> {
    let mut coeffs = vec![Cpx::new(1.0, 0.0)];
    for &r in roots {
        // Multiply current polynomial by (1 − r z⁻¹).
        let mut next = vec![Cpx::new(0.0, 0.0); coeffs.len() + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            next[i] = next[i].add(c); // · 1
            next[i + 1] = next[i + 1].sub(c.mul(r)); // · (−r z⁻¹)
        }
        coeffs = next;
    }
    coeffs.into_iter().map(|c| c.re).collect()
}

/// Evaluates a polynomial `Σ c_i z^{-i}` at the real point `z = z_val`,
/// i.e. `Σ c_i / z_val^i`.  Used at `z = ±1`, so this reduces to `Σ c_i z_val^{-i}`
/// with `z_val^{-1} = z_val` for `±1`.
fn poly_eval_at(coeffs: &[f64], z_val: f64) -> f64 {
    // For z = ±1, z⁻¹ = z, so Σ c_i (z)^i with alternating sign for −1.
    let mut acc = 0.0_f64;
    let mut zpow = 1.0_f64;
    for &c in coeffs {
        acc += c * zpow;
        zpow *= z_val;
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(order: usize, cutoff: f64, fs: f64) -> ButterworthConfig {
        ButterworthConfig {
            order,
            cutoff,
            fs,
            filter_type: FilterType::LowPass,
        }
    }
    fn hp(order: usize, cutoff: f64, fs: f64) -> ButterworthConfig {
        ButterworthConfig {
            order,
            cutoff,
            fs,
            filter_type: FilterType::HighPass,
        }
    }

    /// Steady-state gain of the filter at normalised frequency `f/fs` by
    /// running a long sinusoid and measuring output amplitude.
    fn steady_gain(filt: &ButterworthFilter, f_over_fs: f64) -> f64 {
        let n = 4096;
        let x: Vec<f64> = (0..n)
            .map(|t| (2.0 * PI * f_over_fs * t as f64).sin())
            .collect();
        let y = filt.apply(&x).expect("apply");
        // Measure RMS over the last half (after transient).
        let tail = &y[n / 2..];
        let rms: f64 = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        rms * std::f64::consts::SQRT_2 // sinusoid RMS → amplitude
    }

    #[test]
    fn design_finite() {
        let f = ButterworthFilter::design(&lp(4, 100.0, 1000.0)).expect("ok");
        for &c in f.numerator().iter().chain(f.denominator()) {
            assert!(c.is_finite(), "coeff not finite: {c}");
        }
    }

    #[test]
    fn coeffs_len() {
        for order in 1..=6 {
            let f = ButterworthFilter::design(&lp(order, 100.0, 1000.0)).expect("ok");
            assert_eq!(f.numerator().len(), order + 1);
            assert_eq!(f.denominator().len(), order + 1);
            assert!((f.denominator()[0] - 1.0).abs() < 1e-12, "a[0] must be 1");
        }
    }

    #[test]
    fn lowpass_passes_dc() {
        let f = ButterworthFilter::design(&lp(4, 100.0, 1000.0)).expect("ok");
        // DC gain ~ 1.
        let x = vec![1.0_f64; 2000];
        let y = f.apply(&x).expect("apply");
        let last = *y.last().expect("last should succeed");
        assert!((last - 1.0).abs() < 1e-6, "DC gain={last}");
    }

    #[test]
    fn highpass_blocks_dc() {
        let f = ButterworthFilter::design(&hp(4, 100.0, 1000.0)).expect("ok");
        let x = vec![1.0_f64; 2000];
        let y = f.apply(&x).expect("apply");
        let last = *y.last().expect("last should succeed");
        assert!(last.abs() < 1e-6, "HP DC gain should be ~0, got {last}");
    }

    #[test]
    fn apply_output_len() {
        let f = ButterworthFilter::design(&lp(3, 50.0, 1000.0)).expect("ok");
        for len in [0usize, 1, 10, 333] {
            let x = vec![0.5_f64; len];
            let y = f.apply(&x).expect("apply");
            assert_eq!(y.len(), len);
        }
    }

    #[test]
    fn order_0_error() {
        assert!(ButterworthFilter::design(&lp(0, 100.0, 1000.0)).is_err());
    }

    #[test]
    fn cutoff_gt_nyquist_error() {
        // cutoff above Nyquist (fs/2 = 500) must error.
        assert!(ButterworthFilter::design(&lp(4, 600.0, 1000.0)).is_err());
        // cutoff exactly at Nyquist must error.
        assert!(ButterworthFilter::design(&lp(4, 500.0, 1000.0)).is_err());
        // cutoff <= 0 must error.
        assert!(ButterworthFilter::design(&lp(4, 0.0, 1000.0)).is_err());
    }

    #[test]
    fn stable_output() {
        // A stable filter produces a bounded response to a bounded input.
        let f = ButterworthFilter::design(&lp(6, 80.0, 1000.0)).expect("ok");
        let x: Vec<f64> = (0..5000).map(|t| (t as f64 * 0.05).sin()).collect();
        let y = f.apply(&x).expect("apply");
        for v in &y {
            assert!(v.is_finite() && v.abs() < 10.0, "unstable: {v}");
        }
    }

    #[test]
    fn fs_positive() {
        assert!(ButterworthFilter::design(&lp(4, 100.0, 0.0)).is_err());
        assert!(ButterworthFilter::design(&lp(4, 100.0, -1000.0)).is_err());
    }

    #[test]
    fn lowpass_attenuates_high_freq() {
        // Below cutoff: ~unity gain; well above cutoff: strong attenuation.
        let f = ButterworthFilter::design(&lp(4, 100.0, 1000.0)).expect("ok");
        let pass = steady_gain(&f, 20.0 / 1000.0); // well in band
        let stop = steady_gain(&f, 400.0 / 1000.0); // well out of band
        assert!(pass > 0.9, "passband gain too low: {pass}");
        assert!(stop < 0.1, "stopband gain too high: {stop}");
    }

    #[test]
    fn cutoff_minus_3db() {
        // At the cutoff the magnitude should be ≈ 1/sqrt(2) (−3 dB).
        let f = ButterworthFilter::design(&lp(4, 100.0, 1000.0)).expect("ok");
        let g = steady_gain(&f, 100.0 / 1000.0);
        assert!(
            (g - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.05,
            "cutoff gain {g} not ≈ 0.707"
        );
    }
}
