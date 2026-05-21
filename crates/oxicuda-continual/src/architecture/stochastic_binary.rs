//! Stochastic-binary PiggybackMask forward with straight-through estimator.
//!
//! Implements the **straight-through estimator (STE)** variant of Piggyback
//! masks combining
//!
//! - Bengio, Léonard, Courville, "Estimating or Propagating Gradients Through
//!   Stochastic Neurons for Conditional Computation," arXiv:1308.3432 (2013),
//! - Mallya, Davis, Lazebnik, "Piggyback: Adapting a Single Network to
//!   Multiple Tasks by Learning to Mask Weights," ECCV 2018, and
//! - the closely-related Mandziuk-style stochastic-binarisation
//!   formulation that draws each binary mask entry as
//!   `b_i ~ Bernoulli(σ(m_real_i / T))`.
//!
//! The deterministic threshold-based forward already lives in
//! [`super::piggyback`]. This module exposes a *temperature-controlled
//! stochastic* forward whose gradient flows back to the underlying real-valued
//! mask via the straight-through estimator (the binarisation step is treated
//! as the identity on the backward pass, with an optional gradient clip to
//! tame outliers).

#![forbid(unsafe_code)]

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

// ─── Numerically stable sigmoid ──────────────────────────────────────────────

/// Numerically stable logistic sigmoid for `f64`.
///
/// The branch on the sign of `x` avoids overflow of `(-x).exp()` for large
/// negative inputs (which would otherwise yield `+inf`).
#[inline]
#[must_use]
pub fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyper-parameters for the stochastic-binary Piggyback forward.
///
/// * `temperature` — logit scale `T` applied as `σ(m_real / T)`. Lower `T`
///   sharpens the Bernoulli distribution towards a hard 0/1; larger `T`
///   softens it. Must be strictly positive and finite.
/// * `clip_grad`  — optional symmetric clip `|g| ≤ c` applied to the
///   straight-through backward gradient. `None` disables clipping; `Some(c)`
///   requires `c > 0` and finite.
/// * `seed`       — deterministic seed for the internal LCG RNG used to draw
///   the Bernoulli samples.
#[derive(Debug, Clone, Copy)]
pub struct StochasticBinaryConfig {
    pub temperature: f64,
    pub clip_grad: Option<f64>,
    pub seed: u64,
}

impl Default for StochasticBinaryConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            clip_grad: None,
            seed: 42,
        }
    }
}

impl StochasticBinaryConfig {
    fn validate(&self) -> ContinualResult<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(ContinualError::Internal(format!(
                "stochastic_binary: temperature must be > 0 and finite, got {}",
                self.temperature
            )));
        }
        match self.clip_grad {
            Some(c) if !c.is_finite() || c <= 0.0 => Err(ContinualError::Internal(format!(
                "stochastic_binary: clip_grad must be > 0 and finite when set, got {c}"
            ))),
            _ => Ok(()),
        }
    }
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Persistent state for stochastic-binary forward: the validated config and a
/// deterministic LCG RNG seeded from `cfg.seed`.
#[derive(Debug, Clone)]
pub struct StochasticBinaryState {
    cfg: StochasticBinaryConfig,
    rng: LcgRng,
}

impl StochasticBinaryState {
    /// Construct a new state, validating the config eagerly.
    pub fn new(cfg: StochasticBinaryConfig) -> ContinualResult<Self> {
        cfg.validate()?;
        let rng = LcgRng::new(cfg.seed);
        Ok(Self { cfg, rng })
    }

    /// Return the validated config.
    #[must_use]
    pub fn config(&self) -> &StochasticBinaryConfig {
        &self.cfg
    }

    /// Stochastic binary forward.
    ///
    /// For each element of `m_real`, draws `b_i ~ Bernoulli(σ(m_real_i / T))`
    /// and returns the binary mask as `Vec<f64>` whose entries lie in `{0.0,
    /// 1.0}`. An empty input yields an empty output.
    pub fn forward(&mut self, m_real: &[f64]) -> ContinualResult<Vec<f64>> {
        let inv_t = 1.0 / self.cfg.temperature;
        let mut out = Vec::with_capacity(m_real.len());
        for &m in m_real {
            let p = stable_sigmoid(m * inv_t);
            let u = self.next_uniform_unit();
            out.push(if u < p { 1.0 } else { 0.0 });
        }
        Ok(out)
    }

    /// Draw a `f64` uniformly in `[0, 1)` by combining two LCG draws.
    ///
    /// `LcgRng::next_u32` keeps only 31 usable bits (the source LCG shifts the
    /// state right by 33), so the bundled `next_f32` is restricted to
    /// `[0, 0.5)`. Two consecutive draws are concatenated into a 53-bit
    /// mantissa and divided by `2^53` to recover a full-range double-precision
    /// uniform sample without modifying the shared RNG type.
    #[inline]
    fn next_uniform_unit(&mut self) -> f64 {
        let hi = u64::from(self.rng.next_u32()) & 0x7FFF_FFFF;
        let lo = u64::from(self.rng.next_u32()) & 0x003F_FFFF;
        let bits = (hi << 22) | lo;
        (bits as f64) / ((1u64 << 53) as f64)
    }

    /// Straight-through backward.
    ///
    /// Treats the binarisation step as the identity, so the gradient w.r.t.
    /// the real-valued mask equals the incoming `grad_binary` (optionally
    /// clipped symmetrically to `±clip_grad`).
    pub fn backward(&self, grad_binary: &[f64]) -> ContinualResult<Vec<f64>> {
        match self.cfg.clip_grad {
            None => Ok(grad_binary.to_vec()),
            Some(c) => Ok(grad_binary.iter().map(|&g| g.clamp(-c, c)).collect()),
        }
    }

    /// Backward variant that additionally validates length against an
    /// expected mask size, returning `DimensionMismatch` on mismatch. Useful
    /// when chaining with a known forward output length.
    pub fn backward_checked(
        &self,
        grad_binary: &[f64],
        expected_len: usize,
    ) -> ContinualResult<Vec<f64>> {
        if grad_binary.len() != expected_len {
            return Err(ContinualError::DimensionMismatch {
                expected: expected_len,
                got: grad_binary.len(),
            });
        }
        self.backward(grad_binary)
    }

    /// Expected-value (deterministic) forward: returns `σ(m_real_i / T)` for
    /// every element without sampling. Useful for evaluation and for
    /// expectation-based smooth surrogate losses.
    pub fn forward_expected(&self, m_real: &[f64]) -> ContinualResult<Vec<f64>> {
        let inv_t = 1.0 / self.cfg.temperature;
        Ok(m_real.iter().map(|&m| stable_sigmoid(m * inv_t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state(seed: u64) -> StochasticBinaryState {
        StochasticBinaryState::new(StochasticBinaryConfig {
            temperature: 1.0,
            clip_grad: None,
            seed,
        })
        .expect("config must validate")
    }

    #[test]
    fn empty_input_returns_empty_ok() {
        let mut s = default_state(1);
        let out = s.forward(&[]).unwrap();
        assert!(out.is_empty());
        let exp = s.forward_expected(&[]).unwrap();
        assert!(exp.is_empty());
        let back = s.backward(&[]).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn temperature_zero_is_invalid_config() {
        let cfg = StochasticBinaryConfig {
            temperature: 0.0,
            clip_grad: None,
            seed: 0,
        };
        assert!(matches!(
            StochasticBinaryState::new(cfg),
            Err(ContinualError::Internal(_))
        ));
    }

    #[test]
    fn temperature_negative_is_invalid_config() {
        let cfg = StochasticBinaryConfig {
            temperature: -0.5,
            clip_grad: None,
            seed: 0,
        };
        assert!(StochasticBinaryState::new(cfg).is_err());
    }

    #[test]
    fn temperature_nan_or_inf_is_invalid_config() {
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let cfg = StochasticBinaryConfig {
                temperature: t,
                clip_grad: None,
                seed: 0,
            };
            assert!(
                StochasticBinaryState::new(cfg).is_err(),
                "temperature {t} must be rejected"
            );
        }
    }

    #[test]
    fn clip_grad_zero_is_invalid_config() {
        let cfg = StochasticBinaryConfig {
            temperature: 1.0,
            clip_grad: Some(0.0),
            seed: 0,
        };
        assert!(matches!(
            StochasticBinaryState::new(cfg),
            Err(ContinualError::Internal(_))
        ));
    }

    #[test]
    fn clip_grad_negative_is_invalid_config() {
        let cfg = StochasticBinaryConfig {
            temperature: 1.0,
            clip_grad: Some(-1e-3),
            seed: 0,
        };
        assert!(StochasticBinaryState::new(cfg).is_err());
    }

    #[test]
    fn clip_grad_nan_is_invalid_config() {
        let cfg = StochasticBinaryConfig {
            temperature: 1.0,
            clip_grad: Some(f64::NAN),
            seed: 0,
        };
        assert!(StochasticBinaryState::new(cfg).is_err());
    }

    #[test]
    fn forward_expected_matches_sigmoid_at_temperature() {
        let cfg = StochasticBinaryConfig {
            temperature: 2.0,
            clip_grad: None,
            seed: 0,
        };
        let s = StochasticBinaryState::new(cfg).unwrap();
        let m = vec![-4.0_f64, -1.0, 0.0, 1.5, 3.0];
        let out = s.forward_expected(&m).unwrap();
        for (i, &m_i) in m.iter().enumerate() {
            let expected = stable_sigmoid(m_i / 2.0);
            assert!(
                (out[i] - expected).abs() < 1e-15,
                "forward_expected[{i}] = {} expected {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn forward_expected_is_symmetric_around_zero_for_signed_pair() {
        let s = default_state(123);
        let out = s.forward_expected(&[-2.5_f64, 2.5]).unwrap();
        assert!(
            (out[0] + out[1] - 1.0).abs() < 1e-15,
            "σ(-x) + σ(x) must equal 1, got {} + {} = {}",
            out[0],
            out[1],
            out[0] + out[1]
        );
    }

    #[test]
    fn stable_sigmoid_extremes_do_not_overflow() {
        let very_neg = stable_sigmoid(-1.0e6);
        let very_pos = stable_sigmoid(1.0e6);
        assert!(very_neg.is_finite() && (0.0..=1.0).contains(&very_neg));
        assert!(very_pos.is_finite() && (0.0..=1.0).contains(&very_pos));
        assert!(very_neg < 1e-300, "σ(-1e6) ≈ 0, got {very_neg}");
        assert!(
            (very_pos - 1.0).abs() < 1e-12,
            "σ(+1e6) ≈ 1, got {very_pos}"
        );
    }

    #[test]
    fn forward_extreme_positive_samples_mostly_one() {
        let mut s = default_state(7);
        let m = vec![20.0_f64; 256];
        let out = s.forward(&m).unwrap();
        let n_one = out.iter().filter(|&&v| v == 1.0).count();
        assert!(
            n_one >= 250,
            "expected near-all 1.0 for large +m, got {n_one}/256"
        );
        for v in out {
            assert!(v == 0.0 || v == 1.0, "binary output must be 0 or 1");
        }
    }

    #[test]
    fn forward_extreme_negative_samples_mostly_zero() {
        let mut s = default_state(11);
        let m = vec![-20.0_f64; 256];
        let out = s.forward(&m).unwrap();
        let n_zero = out.iter().filter(|&&v| v == 0.0).count();
        assert!(
            n_zero >= 250,
            "expected near-all 0.0 for large -m, got {n_zero}/256"
        );
    }

    #[test]
    fn forward_zero_mask_samples_about_half_one() {
        let mut s = default_state(2024);
        let n = 2048;
        let m = vec![0.0_f64; n];
        let out = s.forward(&m).unwrap();
        let n_one = out.iter().filter(|&&v| v == 1.0).count() as f64;
        let frac = n_one / n as f64;
        assert!(
            (frac - 0.5).abs() < 0.06,
            "Bernoulli(0.5) sample fraction should be near 0.5, got {frac}"
        );
    }

    #[test]
    fn backward_identity_passthrough_without_clip() {
        let s = default_state(0);
        let g = vec![-3.0_f64, 0.0, 1.5, 100.0, -50.0];
        let out = s.backward(&g).unwrap();
        assert_eq!(out, g);
    }

    #[test]
    fn backward_clip_applies_symmetrically() {
        let cfg = StochasticBinaryConfig {
            temperature: 1.0,
            clip_grad: Some(2.0),
            seed: 0,
        };
        let s = StochasticBinaryState::new(cfg).unwrap();
        let g = vec![-5.0_f64, -1.0, 0.0, 1.0, 3.0, 2.0, -2.0];
        let out = s.backward(&g).unwrap();
        assert_eq!(out, vec![-2.0, -1.0, 0.0, 1.0, 2.0, 2.0, -2.0]);
    }

    #[test]
    fn determinism_with_same_seed() {
        let cfg = StochasticBinaryConfig {
            temperature: 0.7,
            clip_grad: None,
            seed: 4242,
        };
        let mut a = StochasticBinaryState::new(cfg).unwrap();
        let mut b = StochasticBinaryState::new(cfg).unwrap();
        let m = vec![-1.0_f64, 0.5, 2.0, -0.3, 0.0, 4.0, -7.0];
        let out_a = a.forward(&m).unwrap();
        let out_b = b.forward(&m).unwrap();
        assert_eq!(out_a, out_b, "same seed must produce identical samples");
    }

    #[test]
    fn backward_checked_length_mismatch_errors() {
        let s = default_state(0);
        let g = vec![1.0_f64, 2.0, 3.0];
        let err = s.backward_checked(&g, 5).unwrap_err();
        assert!(matches!(
            err,
            ContinualError::DimensionMismatch {
                expected: 5,
                got: 3
            }
        ));
    }

    #[test]
    fn round_trip_large_vector_forward_then_backward() {
        let mut s = StochasticBinaryState::new(StochasticBinaryConfig {
            temperature: 1.5,
            clip_grad: Some(10.0),
            seed: 99,
        })
        .unwrap();
        let n = 4096;
        let m: Vec<f64> = (0..n)
            .map(|i| ((i as f64) - (n as f64) * 0.5) * 0.01)
            .collect();
        let bin = s.forward(&m).unwrap();
        assert_eq!(bin.len(), n);
        for v in &bin {
            assert!(*v == 0.0 || *v == 1.0);
        }
        let upstream: Vec<f64> = bin
            .iter()
            .map(|&b| if b > 0.5 { 0.25 } else { -0.25 })
            .collect();
        let grad = s.backward_checked(&upstream, n).unwrap();
        assert_eq!(grad.len(), n);
        for &g in &grad {
            assert!((-10.0..=10.0).contains(&g));
        }
    }

    #[test]
    fn forward_with_temperature_sharpens_distribution() {
        let cfg_cold = StochasticBinaryConfig {
            temperature: 0.1,
            clip_grad: None,
            seed: 3,
        };
        let cfg_warm = StochasticBinaryConfig {
            temperature: 10.0,
            clip_grad: None,
            seed: 3,
        };
        let s_cold = StochasticBinaryState::new(cfg_cold).unwrap();
        let s_warm = StochasticBinaryState::new(cfg_warm).unwrap();
        let m = vec![1.0_f64; 4];
        let exp_cold = s_cold.forward_expected(&m).unwrap();
        let exp_warm = s_warm.forward_expected(&m).unwrap();
        for (c, w) in exp_cold.iter().zip(exp_warm.iter()) {
            assert!(
                *c > *w,
                "low T should drive σ closer to 1 (sharp); got cold={c} warm={w}"
            );
            assert!(*w > 0.5 && *w < *c);
        }
    }
}
