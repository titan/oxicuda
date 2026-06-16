//! # Discrete Action Exploration Strategies
//!
//! Sutton & Barto (2018), "Reinforcement Learning: An Introduction", §2.2
//! (ε-greedy) and §2.8 (softmax / Boltzmann action selection).
//!
//! These strategies turn a vector of action values `Q[s, ·]` into a concrete
//! action, trading off **exploitation** (picking the current best action)
//! against **exploration** (sampling sub-optimal actions to gather
//! information).
//!
//! * [`EpsilonGreedy`] — with probability `ε` pick a uniformly random action,
//!   otherwise the greedy `argmax`.  Supports linear ε-decay.
//! * [`Boltzmann`] — sample from `softmax(Q / τ)`; temperature `τ`
//!   interpolates between greedy (`τ → 0`) and uniform (`τ → ∞`).

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

// ─── ε-greedy ──────────────────────────────────────────────────────────────

/// ε-greedy action selector with optional linear decay.
#[derive(Debug, Clone)]
pub struct EpsilonGreedy {
    /// Current exploration probability ε ∈ [0, 1].
    epsilon: f32,
    /// Lower bound that ε decays toward.
    epsilon_min: f32,
    /// Additive amount subtracted from ε on each [`EpsilonGreedy::decay`] call.
    decay_step: f32,
}

impl EpsilonGreedy {
    /// Create an ε-greedy selector with a fixed ε (no decay).
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `epsilon ∉ [0, 1]`.
    pub fn new(epsilon: f32) -> RlResult<Self> {
        Self::with_decay(epsilon, epsilon, 0.0)
    }

    /// Create an ε-greedy selector that linearly decays ε toward
    /// `epsilon_min` by `decay_step` each call to [`EpsilonGreedy::decay`].
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `epsilon` or `epsilon_min` lie
    ///   outside `[0, 1]`, if `epsilon_min > epsilon`, or if `decay_step < 0`.
    pub fn with_decay(epsilon: f32, epsilon_min: f32, decay_step: f32) -> RlResult<Self> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(RlError::InvalidHyperparameter {
                name: "epsilon".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if !(0.0..=1.0).contains(&epsilon_min) {
            return Err(RlError::InvalidHyperparameter {
                name: "epsilon_min".into(),
                msg: "must be in [0, 1]".into(),
            });
        }
        if epsilon_min > epsilon {
            return Err(RlError::InvalidHyperparameter {
                name: "epsilon_min".into(),
                msg: "must be <= epsilon".into(),
            });
        }
        if decay_step < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "decay_step".into(),
                msg: "must be >= 0".into(),
            });
        }
        Ok(Self {
            epsilon,
            epsilon_min,
            decay_step,
        })
    }

    /// Current exploration probability.
    #[must_use]
    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Decay ε one step toward `epsilon_min` (clamped, never below the floor).
    pub fn decay(&mut self) {
        self.epsilon = (self.epsilon - self.decay_step).max(self.epsilon_min);
    }

    /// Select an action for the given action-value row `q`.
    ///
    /// With probability ε returns a uniformly random action index; otherwise
    /// the greedy `argmax` (ties broken toward the lowest index).
    ///
    /// # Errors
    ///
    /// * [`RlError::EmptyDistribution`] if `q` is empty.
    pub fn select(&self, q: &[f32], rng: &mut LcgRng) -> RlResult<usize> {
        if q.is_empty() {
            return Err(RlError::EmptyDistribution);
        }
        if rng.next_f32() < self.epsilon {
            Ok(rng.next_usize(q.len()))
        } else {
            Ok(argmax(q))
        }
    }
}

// ─── Boltzmann / softmax exploration ─────────────────────────────────────────

/// Boltzmann (softmax) action selector parameterised by temperature `τ`.
///
/// Action `a` is sampled with probability proportional to `exp(Q[a] / τ)`.
/// Low `τ` concentrates mass on the greedy action; high `τ` flattens toward
/// uniform.
#[derive(Debug, Clone)]
pub struct Boltzmann {
    /// Temperature τ > 0.
    temperature: f32,
}

impl Boltzmann {
    /// Create a Boltzmann selector with temperature `τ`.
    ///
    /// # Errors
    ///
    /// * [`RlError::InvalidHyperparameter`] if `temperature <= 0` or non-finite.
    pub fn new(temperature: f32) -> RlResult<Self> {
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(RlError::InvalidHyperparameter {
                name: "temperature".into(),
                msg: "must be finite and > 0".into(),
            });
        }
        Ok(Self { temperature })
    }

    /// Current temperature.
    #[must_use]
    #[inline]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Return the softmax action-selection probabilities for row `q`.
    ///
    /// Computed in a numerically-stable manner by subtracting `max(q)` before
    /// exponentiating.
    ///
    /// # Errors
    ///
    /// * [`RlError::EmptyDistribution`] if `q` is empty.
    pub fn probabilities(&self, q: &[f32]) -> RlResult<Vec<f32>> {
        if q.is_empty() {
            return Err(RlError::EmptyDistribution);
        }
        let inv_t = 1.0 / self.temperature;
        let max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = q.iter().map(|&v| ((v - max) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            // Degenerate fallback: uniform distribution.
            let u = 1.0 / q.len() as f32;
            probs.fill(u);
            return Ok(probs);
        }
        let inv = 1.0 / sum;
        for p in &mut probs {
            *p *= inv;
        }
        Ok(probs)
    }

    /// Sample an action from the Boltzmann distribution over row `q`.
    ///
    /// # Errors
    ///
    /// * [`RlError::EmptyDistribution`] if `q` is empty.
    pub fn select(&self, q: &[f32], rng: &mut LcgRng) -> RlResult<usize> {
        let probs = self.probabilities(q)?;
        let u = rng.next_f32();
        let mut cumsum = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > u {
                return Ok(i);
            }
        }
        // Floating-point fallback: last positive-probability index.
        Ok(probs.iter().rposition(|&p| p > 0.0).unwrap_or(0))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Argmax with ties broken toward the lowest index. Assumes `q` non-empty.
fn argmax(q: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_val = q[0];
    for (i, &v) in q.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_zero_is_greedy() {
        let eg = EpsilonGreedy::new(0.0).expect("valid epsilon");
        let mut rng = LcgRng::new(1);
        let q = vec![0.1_f32, 0.9, 0.3];
        for _ in 0..100 {
            assert_eq!(eg.select(&q, &mut rng).expect("non-empty"), 1);
        }
    }

    #[test]
    fn epsilon_one_is_uniform() {
        let eg = EpsilonGreedy::new(1.0).expect("valid epsilon");
        let mut rng = LcgRng::new(7);
        let q = vec![0.0_f32, 100.0, 0.0, 0.0];
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[eg.select(&q, &mut rng).expect("non-empty")] += 1;
        }
        // With ε=1 every action (incl. non-greedy) should be explored.
        assert!(
            counts.iter().all(|&c| c > 500),
            "ε=1 should explore all actions, got {counts:?}"
        );
    }

    #[test]
    fn epsilon_invalid_range_error() {
        assert!(EpsilonGreedy::new(-0.1).is_err());
        assert!(EpsilonGreedy::new(1.1).is_err());
    }

    #[test]
    fn epsilon_decays_to_floor() {
        let mut eg = EpsilonGreedy::with_decay(1.0, 0.1, 0.3).expect("valid");
        assert!((eg.epsilon() - 1.0).abs() < 1e-6);
        eg.decay(); // 0.7
        eg.decay(); // 0.4
        eg.decay(); // 0.1 (floor)
        eg.decay(); // stays 0.1
        assert!((eg.epsilon() - 0.1).abs() < 1e-6, "got {}", eg.epsilon());
    }

    #[test]
    fn epsilon_decay_invalid_config_error() {
        // epsilon_min > epsilon
        assert!(EpsilonGreedy::with_decay(0.2, 0.5, 0.01).is_err());
        // negative decay
        assert!(EpsilonGreedy::with_decay(0.5, 0.1, -0.01).is_err());
    }

    #[test]
    fn epsilon_empty_error() {
        let eg = EpsilonGreedy::new(0.5).expect("valid");
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            eg.select(&[], &mut rng),
            Err(RlError::EmptyDistribution)
        ));
    }

    #[test]
    fn epsilon_select_in_range() {
        let eg = EpsilonGreedy::new(0.5).expect("valid");
        let mut rng = LcgRng::new(99);
        let q = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        for _ in 0..1000 {
            let a = eg.select(&q, &mut rng).expect("non-empty");
            assert!(a < q.len());
        }
    }

    #[test]
    fn boltzmann_probabilities_sum_to_one() {
        let b = Boltzmann::new(1.0).expect("valid temp");
        let probs = b.probabilities(&[1.0, 2.0, 3.0]).expect("non-empty");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn boltzmann_low_temp_favours_max() {
        let b = Boltzmann::new(0.01).expect("valid temp");
        let probs = b.probabilities(&[0.0, 1.0, 0.0]).expect("non-empty");
        assert!(
            probs[1] > 0.99,
            "low τ should concentrate on max: {probs:?}"
        );
    }

    #[test]
    fn boltzmann_high_temp_near_uniform() {
        let b = Boltzmann::new(1000.0).expect("valid temp");
        let probs = b.probabilities(&[0.0, 1.0, 2.0]).expect("non-empty");
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 0.01, "high τ ≈ uniform: {probs:?}");
        }
    }

    #[test]
    fn boltzmann_invalid_temp_error() {
        assert!(Boltzmann::new(0.0).is_err());
        assert!(Boltzmann::new(-1.0).is_err());
        assert!(Boltzmann::new(f32::INFINITY).is_err());
    }

    #[test]
    fn boltzmann_empty_error() {
        let b = Boltzmann::new(1.0).expect("valid temp");
        assert!(matches!(
            b.probabilities(&[]),
            Err(RlError::EmptyDistribution)
        ));
    }

    #[test]
    fn boltzmann_select_in_range_and_favours_max() {
        let b = Boltzmann::new(0.1).expect("valid temp");
        let mut rng = LcgRng::new(3);
        let q = vec![0.0_f32, 5.0, 0.0];
        let mut counts = [0usize; 3];
        for _ in 0..2000 {
            let a = b.select(&q, &mut rng).expect("non-empty");
            assert!(a < 3);
            counts[a] += 1;
        }
        assert!(counts[1] > counts[0] && counts[1] > counts[2]);
    }
}
