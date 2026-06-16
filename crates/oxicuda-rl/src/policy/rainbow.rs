//! # Rainbow components — NoisyNet layers and dueling-Q aggregation.
//!
//! Two of the orthogonal improvements combined in Rainbow (Hessel et al. 2018,
//! AAAI) that are not specific to a particular loss:
//!
//! * [`NoisyLinear`] — **NoisyNets** (Fortunato et al. 2018,
//!   <https://arxiv.org/abs/1706.10295>): a linear layer whose weights and
//!   biases carry learnable Gaussian noise, providing state-dependent
//!   exploration that replaces ε-greedy. Uses **factorized** Gaussian noise
//!   `εʷ_{ij} = f(εᵢ)·f(εⱼ)`, `f(x) = sgn(x)·√|x|`, so only `in + out`
//!   noise variables are sampled per forward pass.
//!
//! * [`dueling_q_values`] — **Dueling networks** (Wang et al. 2016,
//!   <https://arxiv.org/abs/1511.06581>): aggregate a scalar state-value `V(s)`
//!   and per-action advantages `A(s, a)` into action values via the
//!   identifiable form
//!   ```text
//!   Q(s, a) = V(s) + ( A(s, a) − mean_{a'} A(s, a') ).
//!   ```
//!
//! Both operate on flat `&[f32]` slices; the caller owns the parameter storage.

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

// ─── NoisyLinear ────────────────────────────────────────────────────────────────

/// A noisy linear layer `y = (μ_w + σ_w ⊙ εʷ)·x + (μ_b + σ_b ⊙ ε_b)`.
///
/// Parameters are stored row-major with shape `[out_features × in_features]`
/// for the weights and `[out_features]` for the biases. The `σ` parameters are
/// initialised to `σ₀ / √in_features` (factorized-noise convention,
/// Fortunato et al. 2018, σ₀ = 0.5 by default).
#[derive(Debug, Clone)]
pub struct NoisyLinear {
    /// Input dimensionality.
    pub in_features: usize,
    /// Output dimensionality.
    pub out_features: usize,
    /// Weight means `μ_w`, shape `[out × in]`.
    pub weight_mu: Vec<f32>,
    /// Weight noise scales `σ_w`, shape `[out × in]`.
    pub weight_sigma: Vec<f32>,
    /// Bias means `μ_b`, shape `[out]`.
    pub bias_mu: Vec<f32>,
    /// Bias noise scales `σ_b`, shape `[out]`.
    pub bias_sigma: Vec<f32>,
}

impl NoisyLinear {
    /// Create a layer with the standard NoisyNet initialisation.
    ///
    /// `μ` parameters are drawn uniformly from `[−1/√in, 1/√in]`; `σ`
    /// parameters are set to `σ₀ / √in`.
    ///
    /// # Errors
    /// * [`RlError::InvalidHyperparameter`] if `in_features` or `out_features`
    ///   is zero, or `sigma0 <= 0`.
    pub fn new(
        in_features: usize,
        out_features: usize,
        sigma0: f32,
        rng: &mut LcgRng,
    ) -> RlResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "features".into(),
                msg: "in_features and out_features must be > 0".into(),
            });
        }
        if sigma0 <= 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "sigma0".into(),
                msg: "must be > 0".into(),
            });
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let sigma_init = sigma0 / (in_features as f32).sqrt();

        let n_w = out_features * in_features;
        let mut weight_mu = Vec::with_capacity(n_w);
        for _ in 0..n_w {
            // Uniform in [-bound, bound].
            weight_mu.push((rng.next_f32() * 2.0 - 1.0) * bound);
        }
        let weight_sigma = vec![sigma_init; n_w];

        let mut bias_mu = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            bias_mu.push((rng.next_f32() * 2.0 - 1.0) * bound);
        }
        let bias_sigma = vec![sigma_init; out_features];

        Ok(Self {
            in_features,
            out_features,
            weight_mu,
            weight_sigma,
            bias_mu,
            bias_sigma,
        })
    }

    /// Deterministic (noise-free) forward pass `y = μ_w·x + μ_b`.
    ///
    /// Used at evaluation time when exploration noise is disabled.
    ///
    /// # Errors
    /// * [`RlError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward_mean(&self, x: &[f32]) -> RlResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(RlError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(self.out_features);
        for (row, &b) in self
            .weight_mu
            .chunks_exact(self.in_features)
            .zip(&self.bias_mu)
        {
            let mut acc = b;
            for (&w, &xi) in row.iter().zip(x) {
                acc += w * xi;
            }
            out.push(acc);
        }
        Ok(out)
    }

    /// Noisy forward pass with freshly sampled factorized Gaussian noise.
    ///
    /// `y_o = Σ_i (μ_w[o,i] + σ_w[o,i]·f(ε_i)·f(ε_o))·x_i
    ///        + μ_b[o] + σ_b[o]·f(ε_o)`,
    /// where `f(x) = sgn(x)·√|x|`, `ε_i`/`ε_o` are standard normals.
    ///
    /// # Errors
    /// * [`RlError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward_noisy(&self, x: &[f32], rng: &mut LcgRng) -> RlResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(RlError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        // Factorized noise: one variable per input and per output.
        let eps_in: Vec<f32> = (0..self.in_features)
            .map(|_| factorized_noise(sample_standard_normal(rng)))
            .collect();
        let eps_out: Vec<f32> = (0..self.out_features)
            .map(|_| factorized_noise(sample_standard_normal(rng)))
            .collect();

        let mut out = Vec::with_capacity(self.out_features);
        let rows_mu = self.weight_mu.chunks_exact(self.in_features);
        let rows_sigma = self.weight_sigma.chunks_exact(self.in_features);
        for (((row_mu, row_sigma), (&b_mu, &b_sigma)), &eo) in rows_mu
            .zip(rows_sigma)
            .zip(self.bias_mu.iter().zip(&self.bias_sigma))
            .zip(&eps_out)
        {
            let mut acc = b_mu + b_sigma * eo;
            for ((&w_mu, &w_sigma), (&ei, &xi)) in
                row_mu.iter().zip(row_sigma).zip(eps_in.iter().zip(x))
            {
                let noisy_w = w_mu + w_sigma * ei * eo;
                acc += noisy_w * xi;
            }
            out.push(acc);
        }
        Ok(out)
    }
}

/// Factorized-noise transform `f(x) = sgn(x)·√|x|`.
#[inline]
fn factorized_noise(x: f32) -> f32 {
    x.signum() * x.abs().sqrt()
}

/// Sample one standard normal via Box–Muller (matches the crate convention).
#[inline]
fn sample_standard_normal(rng: &mut LcgRng) -> f32 {
    let u1 = (rng.next_f32() + 1e-10).min(1.0 - 1e-10);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ─── Dueling aggregation ────────────────────────────────────────────────────────

/// Combine a scalar state-value and per-action advantages into Q-values using
/// the identifiable dueling form (Wang et al. 2016):
///
/// ```text
/// Q(s, a) = V(s) + ( A(s, a) − (1/|A|)·Σ_{a'} A(s, a') ).
/// ```
///
/// Subtracting the **mean** advantage (rather than the max) keeps the
/// decomposition identifiable while preserving training stability.
///
/// # Arguments
/// * `value`      — scalar `V(s)`.
/// * `advantages` — `[n_actions]` raw advantage stream `A(s, ·)`.
///
/// # Errors
/// * [`RlError::DimensionMismatch`] if `advantages` is empty.
pub fn dueling_q_values(value: f32, advantages: &[f32]) -> RlResult<Vec<f32>> {
    if advantages.is_empty() {
        return Err(RlError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let mean_adv = advantages.iter().copied().sum::<f32>() / advantages.len() as f32;
    Ok(advantages.iter().map(|&a| value + (a - mean_adv)).collect())
}

/// Batched dueling aggregation.
///
/// * `values`     — `[B]` per-sample state values.
/// * `advantages` — `[B × n_actions]` advantage streams (row-major).
/// * `n_actions`  — number of actions per sample.
///
/// Returns `[B × n_actions]` Q-values.
///
/// # Errors
/// * [`RlError::InvalidHyperparameter`] if `n_actions == 0`.
/// * [`RlError::DimensionMismatch`] if shapes are inconsistent.
pub fn dueling_q_values_batch(
    values: &[f32],
    advantages: &[f32],
    n_actions: usize,
) -> RlResult<Vec<f32>> {
    if n_actions == 0 {
        return Err(RlError::InvalidHyperparameter {
            name: "n_actions".into(),
            msg: "must be > 0".into(),
        });
    }
    let b = values.len();
    if advantages.len() != b * n_actions {
        return Err(RlError::DimensionMismatch {
            expected: b * n_actions,
            got: advantages.len(),
        });
    }
    let mut out = Vec::with_capacity(b * n_actions);
    for i in 0..b {
        let row = &advantages[i * n_actions..(i + 1) * n_actions];
        let q = dueling_q_values(values[i], row)?;
        out.extend_from_slice(&q);
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> LcgRng {
        LcgRng::new(7)
    }

    #[test]
    fn noisy_shapes_correct() {
        let layer = NoisyLinear::new(4, 3, 0.5, &mut rng()).expect("ok");
        assert_eq!(layer.weight_mu.len(), 12);
        assert_eq!(layer.weight_sigma.len(), 12);
        assert_eq!(layer.bias_mu.len(), 3);
        assert_eq!(layer.bias_sigma.len(), 3);
    }

    #[test]
    fn noisy_sigma_init_value() {
        let layer = NoisyLinear::new(4, 2, 0.5, &mut rng()).expect("ok");
        let expected = 0.5 / 2.0; // sigma0 / sqrt(in)
        for &s in &layer.weight_sigma {
            assert!((s - expected).abs() < 1e-6, "sigma init {s} != {expected}");
        }
    }

    #[test]
    fn forward_mean_shape_and_value() {
        // Build a layer with known means: identity-ish.
        let mut layer = NoisyLinear::new(2, 2, 0.5, &mut rng()).expect("ok");
        layer.weight_mu = vec![1.0, 0.0, 0.0, 1.0]; // identity
        layer.bias_mu = vec![0.5, -0.5];
        let out = layer.forward_mean(&[3.0, 4.0]).expect("ok");
        assert!((out[0] - 3.5).abs() < 1e-6, "out0={}", out[0]);
        assert!((out[1] - 3.5).abs() < 1e-6, "out1={}", out[1]);
    }

    #[test]
    fn forward_noisy_shape_and_finite() {
        let layer = NoisyLinear::new(8, 5, 0.5, &mut rng()).expect("ok");
        let x = vec![0.1_f32; 8];
        let mut r = rng();
        let out = layer.forward_noisy(&x, &mut r).expect("ok");
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_noisy_differs_across_draws() {
        let layer = NoisyLinear::new(8, 5, 0.5, &mut rng()).expect("ok");
        let x = vec![1.0_f32; 8];
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(2);
        let o1 = layer.forward_noisy(&x, &mut r1).expect("ok");
        let o2 = layer.forward_noisy(&x, &mut r2).expect("ok");
        let diff: f32 = o1.iter().zip(&o2).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-5, "different noise draws should differ: {diff}");
    }

    #[test]
    fn forward_noisy_zero_sigma_equals_mean() {
        let mut layer = NoisyLinear::new(4, 3, 0.5, &mut rng()).expect("ok");
        // Zero out all noise scales ⇒ noisy forward == mean forward.
        for s in layer.weight_sigma.iter_mut() {
            *s = 0.0;
        }
        for s in layer.bias_sigma.iter_mut() {
            *s = 0.0;
        }
        let x = vec![0.5_f32, -0.3, 0.2, 0.7];
        let mut r = rng();
        let noisy = layer.forward_noisy(&x, &mut r).expect("ok");
        let mean = layer.forward_mean(&x).expect("ok");
        for (&n, &m) in noisy.iter().zip(&mean) {
            assert!((n - m).abs() < 1e-6, "zero-σ noisy != mean: {n} vs {m}");
        }
    }

    #[test]
    fn factorized_noise_transform() {
        assert!((factorized_noise(4.0) - 2.0).abs() < 1e-6);
        assert!((factorized_noise(-9.0) - (-3.0)).abs() < 1e-6);
        assert!((factorized_noise(0.0)).abs() < 1e-6);
    }

    #[test]
    fn dueling_basic_identity() {
        // Q = V + (A - mean(A)). With A summing such that mean is subtracted.
        let q = dueling_q_values(10.0, &[1.0, 2.0, 3.0]).expect("ok");
        // mean = 2 ⇒ Q = [10 + (1-2), 10 + (2-2), 10 + (3-2)] = [9, 10, 11]
        assert!((q[0] - 9.0).abs() < 1e-6);
        assert!((q[1] - 10.0).abs() < 1e-6);
        assert!((q[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn dueling_advantages_mean_centered() {
        // Sum of (Q - V) over actions should be 0 (mean-centered advantage).
        let v = 5.0_f32;
        let q = dueling_q_values(v, &[0.5, -2.0, 3.5, 1.0]).expect("ok");
        let sum_adv: f32 = q.iter().map(|&qi| qi - v).sum();
        assert!(
            sum_adv.abs() < 1e-5,
            "centered advantages sum to 0: {sum_adv}"
        );
    }

    #[test]
    fn dueling_constant_advantages_equal_value() {
        // If all advantages equal, Q == V for every action.
        let q = dueling_q_values(7.0, &[2.0, 2.0, 2.0]).expect("ok");
        for &qi in &q {
            assert!((qi - 7.0).abs() < 1e-6, "Q should equal V, got {qi}");
        }
    }

    #[test]
    fn dueling_batch_shape_and_values() {
        let values = vec![10.0_f32, 20.0];
        let advantages = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]; // B=2, n=3
        let q = dueling_q_values_batch(&values, &advantages, 3).expect("ok");
        assert_eq!(q.len(), 6);
        // Sample 1: [9,10,11]; sample 2: all == 20.
        assert!((q[0] - 9.0).abs() < 1e-6);
        assert!((q[2] - 11.0).abs() < 1e-6);
        assert!((q[3] - 20.0).abs() < 1e-6);
        assert!((q[5] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn err_noisy_zero_features() {
        assert!(matches!(
            NoisyLinear::new(0, 4, 0.5, &mut rng()),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_noisy_bad_sigma() {
        assert!(matches!(
            NoisyLinear::new(4, 4, 0.0, &mut rng()),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_forward_dim_mismatch() {
        let layer = NoisyLinear::new(4, 2, 0.5, &mut rng()).expect("ok");
        let mut r = rng();
        assert!(matches!(
            layer.forward_mean(&[1.0, 2.0]),
            Err(RlError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.forward_noisy(&[1.0, 2.0], &mut r),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_dueling_empty() {
        assert!(matches!(
            dueling_q_values(1.0, &[]),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_dueling_batch_bad_shape() {
        assert!(matches!(
            dueling_q_values_batch(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3),
            Err(RlError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            dueling_q_values_batch(&[1.0], &[1.0], 0),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }
}
