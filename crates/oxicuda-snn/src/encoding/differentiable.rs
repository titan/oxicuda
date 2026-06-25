//! Differentiable (learnable) spike encoders.
//!
//! Classic spike encoders (Bernoulli rate, TTFS latency) have *fixed* transfer
//! functions, so the input→spike map cannot be trained. The encoders here insert
//! a learnable affine pre-activation `a = scale · x + bias` inside a smooth
//! squashing non-linearity, giving a surrogate gradient through the otherwise
//! non-differentiable spike-sampling / latency-rounding step. They follow the
//! surrogate-gradient training methodology of Neftci, Mostafa & Zenke 2019
//! ("Surrogate Gradient Learning in Spiking Neural Networks") and the learnable
//! input-coding ideas of e.g. Zenke & Vogels 2021.
//!
//! Learnable rate encoder. The per-step firing probability is the logistic
//!
//! ```text
//! p(x) = σ(scale · x + bias),     σ(a) = 1 / (1 + e^{−a}).
//! ```
//!
//! Each timestep emits a Bernoulli spike `1[u < p]`, `u ~ U[0, 1)`. The forward
//! sampling is non-differentiable, so we use a **straight-through** estimator:
//! the backward pass differentiates the *expected* spike count
//! `E = T · Σ_i p(x_i)` analytically,
//!
//! ```text
//! ∂E/∂scale = T · Σ_i σ'(a_i) · x_i,   ∂E/∂bias = T · Σ_i σ'(a_i),
//! σ'(a) = σ(a)·(1 − σ(a)).
//! ```
//!
//! [`crate::encoding::differentiable::DifferentiableRateEncoder::update`] performs gradient **ascent** on `E`
//! (`scale ← scale + lr · ∂E/∂scale`), so a positive learning rate raises the
//! firing rate for positive-mean inputs — the natural convention when the
//! returned values are gradients of the spike count being maximised.
//!
//! Learnable TTFS encoder. The latency is
//!
//! ```text
//! g(x) = σ(scale · x + bias),
//! t*(x) = round((1 − clamp(g(x), 0, 1)) · (T − 1)).
//! ```
//!
//! Larger inputs ⇒ larger `g` ⇒ smaller latency (earlier spike). The discrete
//! `round` is replaced for the backward pass by the smooth pre-round latency
//! `t̃(x) = (1 − g(x)) · (T − 1)`, whose gradient is
//!
//! ```text
//! ∂t*/∂x ≈ ∂t̃/∂x = −(T − 1) · g'(x),   g'(x) = σ'(scale·x + bias) · scale.
//! ```
//!
//! Layout. `encode` returns a flat `(t_steps × n)` row-major `Vec<f32>` of `0/1`
//! spikes (the rate encoder) or a one-spike-per-neuron latency train (the TTFS
//! encoder), matching the conventions of the fixed rate / TTFS encoders.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Numerically-stable logistic sigmoid `σ(a) = 1 / (1 + e^{−a})`.
#[inline]
fn sigmoid(a: f32) -> f32 {
    if a >= 0.0 {
        let z = (-a).exp();
        1.0 / (1.0 + z)
    } else {
        let z = a.exp();
        z / (1.0 + z)
    }
}

/// Derivative of the logistic sigmoid, `σ'(a) = σ(a) · (1 − σ(a))`.
#[inline]
fn sigmoid_prime(a: f32) -> f32 {
    let s = sigmoid(a);
    s * (1.0 - s)
}

/// A learnable Bernoulli rate encoder with firing probability
/// `p = σ(scale · x + bias)`.
///
/// The `scale` and `bias` parameters are trainable via the expected-spike
/// gradient returned by [`DifferentiableRateEncoder::grad_params`].
#[derive(Debug, Clone, Copy)]
pub struct DifferentiableRateEncoder {
    /// Multiplicative gain on the input in the logistic pre-activation.
    pub scale: f32,
    /// Additive bias in the logistic pre-activation.
    pub bias: f32,
}

impl Default for DifferentiableRateEncoder {
    fn default() -> Self {
        // Identity-ish mapping: σ(x) centred at 0.5 for x = 0.
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

impl DifferentiableRateEncoder {
    /// Build a learnable rate encoder with the given affine parameters.
    ///
    /// Returns [`SnnError::OutOfRange`] if `scale` or `bias` is non-finite.
    pub fn new(scale: f32, bias: f32) -> SnnResult<Self> {
        if !scale.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "scale".into(),
                val: scale,
            });
        }
        if !bias.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "bias".into(),
                val: bias,
            });
        }
        Ok(Self { scale, bias })
    }

    /// Per-step firing probability `p(x) = σ(scale · x + bias)`.
    #[must_use]
    pub fn spike_prob(&self, x: f32) -> f32 {
        sigmoid(self.scale * x + self.bias)
    }

    /// Encode `input` into a Bernoulli rate-coded spike train of length
    /// `t_steps`, returned flat `(t_steps × n)` row-major.
    ///
    /// At every step each neuron `i` fires with probability
    /// `σ(scale · input_i + bias)`. Returns [`SnnError::EmptyInput`] for empty
    /// input, [`SnnError::BadTimesteps`] for `t_steps == 0`, and
    /// [`SnnError::OutOfRange`] for non-finite input values.
    pub fn encode(&self, input: &[f32], t_steps: usize, rng: &mut LcgRng) -> SnnResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if t_steps == 0 {
            return Err(SnnError::BadTimesteps { got: t_steps });
        }
        for &v in input {
            if !v.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "input".into(),
                    val: v,
                });
            }
        }
        let n = input.len();
        // Precompute per-neuron probabilities once (they do not change over time).
        let mut probs = vec![0.0_f32; n];
        for (p, &x) in probs.iter_mut().zip(input.iter()) {
            *p = self.spike_prob(x);
        }
        let mut out = vec![0.0_f32; t_steps * n];
        for t in 0..t_steps {
            let off = t * n;
            let row = out.get_mut(off..off + n).ok_or(SnnError::Internal {
                msg: "rate-encode output row out of range".into(),
            })?;
            for (slot, &p) in row.iter_mut().zip(probs.iter()) {
                let u = rng.next_f32();
                *slot = if u < p { 1.0_f32 } else { 0.0_f32 };
            }
        }
        Ok(out)
    }

    /// Gradient of the total expected spike count `E = t_steps · Σ_i σ(a_i)`
    /// (with `a_i = scale · x_i + bias`) w.r.t. the parameters.
    ///
    /// Returns `(∂E/∂scale, ∂E/∂bias)`:
    ///
    /// ```text
    /// ∂E/∂scale = t_steps · Σ_i σ'(a_i) · x_i,
    /// ∂E/∂bias  = t_steps · Σ_i σ'(a_i).
    /// ```
    ///
    /// Returns [`SnnError::EmptyInput`] / [`SnnError::BadTimesteps`] on bad
    /// arguments and [`SnnError::OutOfRange`] for non-finite inputs.
    pub fn grad_params(&self, input: &[f32], t_steps: usize) -> SnnResult<(f32, f32)> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if t_steps == 0 {
            return Err(SnnError::BadTimesteps { got: t_steps });
        }
        for &v in input {
            if !v.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "input".into(),
                    val: v,
                });
            }
        }
        let mut d_scale = 0.0_f32;
        let mut d_bias = 0.0_f32;
        for &x in input {
            let a = self.scale * x + self.bias;
            let sp = sigmoid_prime(a);
            d_scale += sp * x;
            d_bias += sp;
        }
        let t = t_steps as f32;
        Ok((t * d_scale, t * d_bias))
    }

    /// Apply one gradient-**ascent** step on the expected spike count:
    /// `scale ← scale + lr · d_scale`, `bias ← bias + lr · d_bias`.
    ///
    /// Returns [`SnnError::OutOfRange`] if `lr` or either gradient is non-finite.
    pub fn update(&mut self, d_scale: f32, d_bias: f32, lr: f32) -> SnnResult<()> {
        if !lr.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: lr,
            });
        }
        if !d_scale.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "d_scale".into(),
                val: d_scale,
            });
        }
        if !d_bias.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "d_bias".into(),
                val: d_bias,
            });
        }
        self.scale += lr * d_scale;
        self.bias += lr * d_bias;
        Ok(())
    }
}

/// A learnable / differentiable Time-To-First-Spike (TTFS) encoder with
/// `g(x) = σ(scale · x + bias)` and latency
/// `t*(x) = round((1 − clamp(g(x))) · (T − 1))`.
#[derive(Debug, Clone, Copy)]
pub struct DifferentiableTtfsEncoder {
    /// Multiplicative gain on the input in the logistic pre-activation.
    pub scale: f32,
    /// Additive bias in the logistic pre-activation.
    pub bias: f32,
}

impl Default for DifferentiableTtfsEncoder {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

impl DifferentiableTtfsEncoder {
    /// Build a learnable TTFS encoder with the given affine parameters.
    ///
    /// Returns [`SnnError::OutOfRange`] if `scale` or `bias` is non-finite.
    pub fn new(scale: f32, bias: f32) -> SnnResult<Self> {
        if !scale.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "scale".into(),
                val: scale,
            });
        }
        if !bias.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "bias".into(),
                val: bias,
            });
        }
        Ok(Self { scale, bias })
    }

    /// Squashed activation `g(x) = σ(scale · x + bias) ∈ (0, 1)`.
    #[must_use]
    pub fn activation(&self, x: f32) -> f32 {
        sigmoid(self.scale * x + self.bias)
    }

    /// Smooth (pre-round) latency surrogate `t̃(x) = (1 − g(x)) · (T − 1)`.
    #[must_use]
    pub fn smooth_latency(&self, x: f32, t_steps: usize) -> f32 {
        let scale = (t_steps as f32 - 1.0).max(0.0);
        (1.0 - self.activation(x)) * scale
    }

    /// Discrete latency `t*(x) = round((1 − g(x)) · (T − 1))`, clamped to
    /// `[0, T − 1]`.
    #[must_use]
    pub fn latency(&self, x: f32, t_steps: usize) -> usize {
        if t_steps == 0 {
            return 0;
        }
        let raw = self.smooth_latency(x, t_steps).round();
        if raw < 0.0 {
            0
        } else if raw as usize >= t_steps {
            t_steps - 1
        } else {
            raw as usize
        }
    }

    /// Encode `input` into a one-spike-per-neuron TTFS train of length
    /// `t_steps`, returned flat `(t_steps × n)` row-major.
    ///
    /// Returns [`SnnError::EmptyInput`] for empty input,
    /// [`SnnError::BadTimesteps`] for `t_steps == 0`, and
    /// [`SnnError::OutOfRange`] for non-finite input values.
    pub fn encode(&self, input: &[f32], t_steps: usize) -> SnnResult<Vec<f32>> {
        if input.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if t_steps == 0 {
            return Err(SnnError::BadTimesteps { got: t_steps });
        }
        for &v in input {
            if !v.is_finite() {
                return Err(SnnError::OutOfRange {
                    name: "input".into(),
                    val: v,
                });
            }
        }
        let n = input.len();
        let mut out = vec![0.0_f32; t_steps * n];
        for (i, &x) in input.iter().enumerate() {
            let t_spike = self.latency(x, t_steps);
            let idx = t_spike * n + i;
            if let Some(slot) = out.get_mut(idx) {
                *slot = 1.0_f32;
            }
        }
        Ok(out)
    }

    /// Surrogate gradient of the latency w.r.t. the input,
    /// `∂t*/∂x ≈ −(T − 1) · g'(x)` with `g'(x) = σ'(scale·x + bias) · scale`.
    ///
    /// Negative for positive `scale` (larger inputs spike earlier). Returns
    /// [`SnnError::BadTimesteps`] for `t_steps == 0` and [`SnnError::OutOfRange`]
    /// for a non-finite input.
    pub fn latency_grad(&self, x: f32, t_steps: usize) -> SnnResult<f32> {
        if t_steps == 0 {
            return Err(SnnError::BadTimesteps { got: t_steps });
        }
        if !x.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "input".into(),
                val: x,
            });
        }
        let scale_t = (t_steps as f32 - 1.0).max(0.0);
        let a = self.scale * x + self.bias;
        let g_prime = sigmoid_prime(a) * self.scale;
        Ok(-scale_t * g_prime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empirical mean spike rate of a flat `(t_steps × n)` train.
    fn mean_rate(train: &[f32]) -> f32 {
        if train.is_empty() {
            return 0.0;
        }
        let fired = train.iter().filter(|&&s| s != 0.0).count();
        fired as f32 / train.len() as f32
    }

    #[test]
    fn rate_encode_shape_correct() {
        let enc = DifferentiableRateEncoder::new(2.0, 0.0).expect("ctor");
        let input = vec![0.1_f32, -0.2, 0.5, 0.9];
        let t_steps = 20;
        let mut rng = LcgRng::new(1);
        let out = enc.encode(&input, t_steps, &mut rng).expect("encode");
        assert_eq!(out.len(), t_steps * input.len());
        for &s in &out {
            assert!(s == 0.0 || s == 1.0, "non-binary spike: {s}");
        }
    }

    #[test]
    fn higher_input_higher_rate() {
        // Monotonic: a larger input value must (in expectation) fire more often.
        let enc = DifferentiableRateEncoder::new(3.0, 0.0).expect("ctor");
        let t_steps = 5000;
        let mut rng = LcgRng::new(2);
        let low = enc.encode(&[-1.0_f32], t_steps, &mut rng).expect("low");
        let high = enc.encode(&[1.0_f32], t_steps, &mut rng).expect("high");
        let r_low = mean_rate(&low);
        let r_high = mean_rate(&high);
        assert!(
            r_high > r_low,
            "expected higher input to raise rate: low={r_low} high={r_high}"
        );
    }

    #[test]
    fn spike_prob_matches_sigmoid() {
        let enc = DifferentiableRateEncoder::new(2.0, 0.5).expect("ctor");
        let p = enc.spike_prob(0.25);
        let expected = sigmoid(2.0 * 0.25 + 0.5);
        assert!((p - expected).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn empirical_rate_matches_prob() {
        let enc = DifferentiableRateEncoder::new(1.0, 0.0).expect("ctor");
        let x = 0.7_f32;
        let p = enc.spike_prob(x);
        let t_steps = 20000;
        let mut rng = LcgRng::new(3);
        let train = enc.encode(&[x], t_steps, &mut rng).expect("encode");
        let r = mean_rate(&train);
        // Binomial std sqrt(p(1-p)/T); allow ~5 sigma.
        let std = (p * (1.0 - p) / t_steps as f32).sqrt().max(1e-6);
        assert!((r - p).abs() < 5.0 * std, "r={r} p={p} std={std}");
    }

    #[test]
    fn grad_scale_sign_matches_finite_difference() {
        let enc = DifferentiableRateEncoder::new(0.5, -0.1).expect("ctor");
        let input = vec![0.3_f32, 0.8, -0.4, 0.6];
        let t_steps = 7;
        let (d_scale, d_bias) = enc.grad_params(&input, t_steps).expect("grad");

        // Finite-difference of E(scale, bias) = t_steps * sum_i sigma(a_i).
        let expected_spikes = |scale: f32, bias: f32| -> f32 {
            let mut e = 0.0_f32;
            for &x in &input {
                e += sigmoid(scale * x + bias);
            }
            e * t_steps as f32
        };
        let h = 1e-3_f32;
        let fd_scale = (expected_spikes(enc.scale + h, enc.bias)
            - expected_spikes(enc.scale - h, enc.bias))
            / (2.0 * h);
        let fd_bias = (expected_spikes(enc.scale, enc.bias + h)
            - expected_spikes(enc.scale, enc.bias - h))
            / (2.0 * h);
        assert!(
            (d_scale - fd_scale).abs() < 1e-2,
            "d_scale={d_scale} fd={fd_scale}"
        );
        assert!(
            (d_bias - fd_bias).abs() < 1e-2,
            "d_bias={d_bias} fd={fd_bias}"
        );
    }

    #[test]
    fn update_raises_rate_for_positive_input() {
        // Gradient ascent on expected spikes with positive-mean input should
        // increase the firing probability at that input.
        let mut enc = DifferentiableRateEncoder::new(0.2, 0.0).expect("ctor");
        let input = vec![1.0_f32; 4];
        let p_before = enc.spike_prob(1.0);
        let (d_scale, d_bias) = enc.grad_params(&input, 1).expect("grad");
        enc.update(d_scale, d_bias, 0.5).expect("update");
        let p_after = enc.spike_prob(1.0);
        assert!(
            p_after > p_before,
            "ascent should raise rate: before={p_before} after={p_after}"
        );
    }

    #[test]
    fn update_changes_encoded_rate_direction() {
        // End-to-end: ascent step raises the empirical rate at a positive input.
        let mut enc = DifferentiableRateEncoder::new(0.2, 0.0).expect("ctor");
        let x = vec![1.0_f32];
        let t_steps = 8000;
        let mut rng = LcgRng::new(4);
        let before = mean_rate(&enc.encode(&x, t_steps, &mut rng).expect("b"));
        let (ds, db) = enc.grad_params(&[1.0_f32], 1).expect("grad");
        enc.update(ds, db, 1.0).expect("update");
        let mut rng2 = LcgRng::new(4);
        let after = mean_rate(&enc.encode(&x, t_steps, &mut rng2).expect("a"));
        assert!(
            after > before,
            "rate should rise: before={before} after={after}"
        );
    }

    #[test]
    fn ttfs_encode_shape_and_one_spike() {
        let enc = DifferentiableTtfsEncoder::new(2.0, 0.0).expect("ctor");
        let input = vec![0.0_f32, 0.5, -0.5, 1.0];
        let t_steps = 12;
        let out = enc.encode(&input, t_steps).expect("encode");
        assert_eq!(out.len(), t_steps * input.len());
        let n = input.len();
        for i in 0..n {
            let mut count = 0_usize;
            for t in 0..t_steps {
                if out[t * n + i] == 1.0 {
                    count += 1;
                }
            }
            assert_eq!(count, 1, "neuron {i} fired {count} times");
        }
    }

    #[test]
    fn ttfs_latency_decreases_as_input_increases() {
        let enc = DifferentiableTtfsEncoder::new(3.0, 0.0).expect("ctor");
        let t_steps = 32;
        let lat_low = enc.latency(-1.0, t_steps);
        let lat_mid = enc.latency(0.0, t_steps);
        let lat_high = enc.latency(1.0, t_steps);
        assert!(
            lat_low > lat_mid && lat_mid > lat_high,
            "latency must decrease with input: {lat_low} > {lat_mid} > {lat_high}"
        );
    }

    #[test]
    fn ttfs_smooth_latency_decreases_monotonically() {
        let enc = DifferentiableTtfsEncoder::new(2.0, 0.1).expect("ctor");
        let t_steps = 16;
        let a = enc.smooth_latency(-0.5, t_steps);
        let b = enc.smooth_latency(0.5, t_steps);
        assert!(a > b, "smooth latency should decrease: {a} > {b}");
    }

    #[test]
    fn ttfs_latency_grad_negative_for_positive_scale() {
        let enc = DifferentiableTtfsEncoder::new(2.0, 0.0).expect("ctor");
        let g = enc.latency_grad(0.0, 16).expect("grad");
        assert!(g < 0.0, "latency grad should be negative, got {g}");
    }

    #[test]
    fn ttfs_latency_grad_matches_finite_difference() {
        let enc = DifferentiableTtfsEncoder::new(1.5, -0.2).expect("ctor");
        let t_steps = 20;
        let x = 0.3_f32;
        let analytic = enc.latency_grad(x, t_steps).expect("grad");
        // Finite difference of the smooth (pre-round) latency surrogate.
        let h = 1e-3_f32;
        let fd =
            (enc.smooth_latency(x + h, t_steps) - enc.smooth_latency(x - h, t_steps)) / (2.0 * h);
        assert!((analytic - fd).abs() < 1e-2, "analytic={analytic} fd={fd}");
    }

    #[test]
    fn ttfs_high_input_fires_early_low_input_fires_late() {
        let enc = DifferentiableTtfsEncoder::new(4.0, 0.0).expect("ctor");
        let t_steps = 16;
        // Strongly positive input -> g near 1 -> latency near 0.
        let high = enc.latency(2.0, t_steps);
        // Strongly negative input -> g near 0 -> latency near T-1.
        let low = enc.latency(-2.0, t_steps);
        assert!(high <= 1, "high input should spike very early, got {high}");
        assert!(
            low >= t_steps - 2,
            "low input should spike very late, got {low}"
        );
    }

    #[test]
    fn negative_scale_reverses_ttfs_monotonicity() {
        // With negative scale, larger inputs lower g -> later spikes; grad > 0.
        let enc = DifferentiableTtfsEncoder::new(-2.0, 0.0).expect("ctor");
        let t_steps = 32;
        let lo = enc.latency(-1.0, t_steps);
        let hi = enc.latency(1.0, t_steps);
        assert!(hi > lo, "negative scale should reverse latency order");
        let g = enc.latency_grad(0.0, t_steps).expect("grad");
        assert!(g > 0.0, "grad should be positive for negative scale: {g}");
    }

    #[test]
    fn rate_encoder_rejects_bad_args() {
        let enc = DifferentiableRateEncoder::default();
        let mut rng = LcgRng::new(5);
        assert!(matches!(
            enc.encode(&[], 4, &mut rng),
            Err(SnnError::EmptyInput)
        ));
        assert!(matches!(
            enc.encode(&[0.5_f32], 0, &mut rng),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            enc.encode(&[f32::NAN], 4, &mut rng),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(enc.grad_params(&[], 4), Err(SnnError::EmptyInput)));
    }

    #[test]
    fn ttfs_encoder_rejects_bad_args() {
        let enc = DifferentiableTtfsEncoder::default();
        assert!(matches!(enc.encode(&[], 4), Err(SnnError::EmptyInput)));
        assert!(matches!(
            enc.encode(&[0.5_f32], 0),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            enc.encode(&[f32::INFINITY], 4),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            enc.latency_grad(0.0, 0),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn constructors_reject_non_finite() {
        assert!(matches!(
            DifferentiableRateEncoder::new(f32::NAN, 0.0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            DifferentiableRateEncoder::new(1.0, f32::INFINITY),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            DifferentiableTtfsEncoder::new(f32::NAN, 0.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn update_rejects_non_finite() {
        let mut enc = DifferentiableRateEncoder::default();
        assert!(matches!(
            enc.update(1.0, 1.0, f32::NAN),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            enc.update(f32::INFINITY, 1.0, 0.1),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn encode_deterministic_given_seed() {
        let enc = DifferentiableRateEncoder::new(1.5, 0.2).expect("ctor");
        let input = vec![0.1_f32, 0.6, -0.3];
        let mut r1 = LcgRng::new(77);
        let mut r2 = LcgRng::new(77);
        let a = enc.encode(&input, 10, &mut r1).expect("a");
        let b = enc.encode(&input, 10, &mut r2).expect("b");
        assert_eq!(a, b, "encoding must be deterministic per seed");
    }
}
