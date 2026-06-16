//! # Mirostat Sampling (v2)
//!
//! Basu et al. (2021), "Mirostat: A Neural Text Decoding Algorithm that
//! Directly Controls Perplexity", ICLR 2021.
//!
//! Mirostat directly controls the **perplexity** (equivalently, the average
//! per-token *surprise* `−log₂ p`) of generated text via a feedback loop,
//! avoiding both the repetition of low-temperature decoding and the incoherence
//! of high-temperature decoding.
//!
//! ## Mirostat v2 algorithm
//!
//! A running threshold `μ` (initialised to `2·τ`, twice the target surprise)
//! bounds the maximum surprise of admissible tokens.  At every step:
//!
//! 1. Convert logits to probabilities and their surprises `S(i) = −log₂ p_i`.
//! 2. **Truncate** to the candidate set `{ i : S(i) < μ }` (always keep at
//!    least the most-probable token).
//! 3. Sample a token `x` from the renormalised candidate distribution.
//! 4. Measure the observed surprise `S(x)` and update the threshold using the
//!    error feedback `μ ← μ − η · (S(x) − τ)`.
//!
//! Over a sequence this drives the *average* observed surprise toward the
//! target `τ`, and hence the perplexity toward `2^τ`.

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, softmax};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`Mirostat`].
#[derive(Debug, Clone)]
pub struct MirostatConfig {
    /// Target surprise `τ` in bits (controls perplexity = `2^τ`).
    pub target_surprise: f32,
    /// Learning rate `η` of the feedback controller (must be ≥ 0).
    pub lr: f32,
    /// Initial threshold `μ` (typically `2·τ`).
    pub tau_init: f32,
}

impl Default for MirostatConfig {
    fn default() -> Self {
        let target = 3.0;
        Self {
            target_surprise: target,
            lr: 0.1,
            tau_init: 2.0 * target,
        }
    }
}

// ─── Sampler ─────────────────────────────────────────────────────────────────

/// Mirostat-v2 perplexity-controlled sampler.
///
/// The sampler is **stateful**: it carries the adaptive threshold `μ` across
/// calls so that the average surprise converges to the configured target.
pub struct Mirostat {
    mu: f32,
    config: MirostatConfig,
}

impl Mirostat {
    /// Create a new Mirostat sampler.
    ///
    /// # Errors
    ///
    /// * [`InferError::InvalidConfig`] if `target_surprise <= 0`, `lr < 0`, or
    ///   `tau_init <= 0` (or any is non-finite).
    pub fn new(config: MirostatConfig) -> InferResult<Self> {
        if !config.target_surprise.is_finite() || config.target_surprise <= 0.0 {
            return Err(InferError::InvalidConfig(
                "Mirostat target_surprise must be finite and > 0",
            ));
        }
        if !config.lr.is_finite() || config.lr < 0.0 {
            return Err(InferError::InvalidConfig(
                "Mirostat lr must be finite and >= 0",
            ));
        }
        if !config.tau_init.is_finite() || config.tau_init <= 0.0 {
            return Err(InferError::InvalidConfig(
                "Mirostat tau_init must be finite and > 0",
            ));
        }
        Ok(Self {
            mu: config.tau_init,
            config,
        })
    }

    /// Current adaptive threshold `μ`.
    #[must_use]
    #[inline]
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Target surprise `τ`.
    #[must_use]
    #[inline]
    pub fn target_surprise(&self) -> f32 {
        self.config.target_surprise
    }

    /// Sample a single token id from `logits`, updating the internal threshold.
    ///
    /// # Errors
    ///
    /// * [`InferError::EmptyBatch`] if `logits` is empty.
    /// * [`InferError::NanLogits`] if any logit is NaN.
    pub fn sample(&mut self, logits: &[f32], rng: &mut Rng) -> InferResult<usize> {
        if logits.is_empty() {
            return Err(InferError::EmptyBatch);
        }
        for &v in logits {
            if v.is_nan() {
                return Err(InferError::NanLogits);
            }
        }

        let probs = softmax(logits);

        // Sort indices by probability descending for stable truncation.
        let n = probs.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to candidates with surprise S(i) = −log2(p_i) < μ.
        // Always keep at least the top-1 token.
        let mut candidates: Vec<usize> = Vec::with_capacity(n);
        let mut mass = 0.0_f32;
        for &idx in &order {
            let p = probs[idx];
            let surprise = if p > 0.0 { -p.log2() } else { f32::INFINITY };
            if candidates.is_empty() || surprise < self.mu {
                candidates.push(idx);
                mass += p;
            } else {
                break;
            }
        }

        // Sample from the renormalised candidate distribution.
        let chosen = if mass > 0.0 {
            let u = rng.next_f32() * mass;
            let mut cumsum = 0.0_f32;
            let mut pick = candidates[candidates.len() - 1];
            for &idx in &candidates {
                cumsum += probs[idx];
                if cumsum > u {
                    pick = idx;
                    break;
                }
            }
            pick
        } else {
            // Degenerate: all candidate mass is zero → fall back to top-1.
            candidates[0]
        };

        // Observed surprise of the chosen token and threshold feedback update.
        let p_chosen = probs[chosen];
        let observed = if p_chosen > 0.0 {
            -p_chosen.log2()
        } else {
            self.mu
        };
        let error = observed - self.config.target_surprise;
        self.mu -= self.config.lr * error;
        // Keep μ strictly positive so the candidate set never collapses.
        if self.mu < f32::MIN_POSITIVE {
            self.mu = f32::MIN_POSITIVE;
        }

        Ok(chosen)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MirostatConfig {
        MirostatConfig {
            target_surprise: 3.0,
            lr: 0.1,
            tau_init: 6.0,
        }
    }

    /// Build logits for a Zipf-like distribution over `n` tokens.
    fn zipf_logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| -((i + 1) as f32).ln()).collect()
    }

    #[test]
    fn sample_in_range() {
        let mut m = Mirostat::new(config()).expect("valid config");
        let mut rng = Rng::new(42);
        let logits = zipf_logits(50);
        for _ in 0..200 {
            let t = m.sample(&logits, &mut rng).expect("valid logits");
            assert!(t < 50, "token out of range: {t}");
        }
    }

    #[test]
    fn mu_updates() {
        let mut m = Mirostat::new(config()).expect("valid");
        let mut rng = Rng::new(1);
        let logits = zipf_logits(40);
        let mu0 = m.mu();
        m.sample(&logits, &mut rng).expect("valid");
        // After at least one step the threshold should generally move.
        // Run a few steps to be robust to a single zero-error step.
        for _ in 0..10 {
            m.sample(&logits, &mut rng).expect("valid");
        }
        assert!((m.mu() - mu0).abs() > 1e-6, "mu should adapt over steps");
    }

    #[test]
    fn target_surprise_tracked() {
        // Over many steps the running average observed surprise should be
        // near the target. We approximate observed surprise by recomputing it.
        let mut m = Mirostat::new(config()).expect("valid");
        let mut rng = Rng::new(7);
        let logits = zipf_logits(256);
        let probs = softmax(&logits);
        let mut total_surprise = 0.0_f64;
        let steps = 2000;
        for _ in 0..steps {
            let t = m.sample(&logits, &mut rng).expect("valid");
            total_surprise += (-probs[t].log2()) as f64;
        }
        let avg = (total_surprise / steps as f64) as f32;
        assert!(
            (avg - 3.0).abs() < 1.0,
            "avg surprise {avg} should be near target 3.0"
        );
    }

    #[test]
    fn lr_zero_static_mu() {
        let cfg = MirostatConfig {
            lr: 0.0,
            ..config()
        };
        let mut m = Mirostat::new(cfg).expect("valid");
        let mut rng = Rng::new(3);
        let logits = zipf_logits(32);
        let mu0 = m.mu();
        for _ in 0..100 {
            m.sample(&logits, &mut rng).expect("valid");
        }
        assert!((m.mu() - mu0).abs() < 1e-9, "lr=0 should freeze mu");
    }

    #[test]
    fn deterministic_seed() {
        let logits = zipf_logits(64);
        let mut m1 = Mirostat::new(config()).expect("valid");
        let mut m2 = Mirostat::new(config()).expect("valid");
        let mut r1 = Rng::new(123);
        let mut r2 = Rng::new(123);
        for _ in 0..100 {
            let a = m1.sample(&logits, &mut r1).expect("valid");
            let b = m2.sample(&logits, &mut r2).expect("valid");
            assert_eq!(a, b, "same seed should give same tokens");
        }
        assert!((m1.mu() - m2.mu()).abs() < 1e-9);
    }

    #[test]
    fn logits_empty_error() {
        let mut m = Mirostat::new(config()).expect("valid");
        let mut rng = Rng::new(0);
        assert!(matches!(
            m.sample(&[], &mut rng),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn nan_logits_error() {
        let mut m = Mirostat::new(config()).expect("valid");
        let mut rng = Rng::new(0);
        let logits = vec![1.0_f32, f32::NAN, 2.0];
        assert!(matches!(
            m.sample(&logits, &mut rng),
            Err(InferError::NanLogits)
        ));
    }

    #[test]
    fn mu_positive() {
        let mut m = Mirostat::new(config()).expect("valid");
        let mut rng = Rng::new(9);
        // A very peaked distribution gives near-zero observed surprise, which
        // pushes mu down; it must stay positive.
        let logits = vec![100.0_f32, 0.0, 0.0, 0.0];
        for _ in 0..500 {
            m.sample(&logits, &mut rng).expect("valid");
            assert!(m.mu() > 0.0, "mu must remain positive, got {}", m.mu());
        }
    }

    #[test]
    fn invalid_config_errors() {
        assert!(
            Mirostat::new(MirostatConfig {
                target_surprise: 0.0,
                ..config()
            })
            .is_err()
        );
        assert!(
            Mirostat::new(MirostatConfig {
                lr: -0.1,
                ..config()
            })
            .is_err()
        );
        assert!(
            Mirostat::new(MirostatConfig {
                tau_init: -1.0,
                ..config()
            })
            .is_err()
        );
    }

    #[test]
    fn high_target_more_diverse() {
        // A higher target surprise should admit a larger candidate set, so the
        // sampler explores more distinct tokens over a fixed budget.
        let logits = zipf_logits(128);
        let mut low = Mirostat::new(MirostatConfig {
            target_surprise: 1.0,
            tau_init: 2.0,
            lr: 0.1,
        })
        .expect("valid");
        let mut high = Mirostat::new(MirostatConfig {
            target_surprise: 6.0,
            tau_init: 12.0,
            lr: 0.1,
        })
        .expect("valid");
        let mut r1 = Rng::new(5);
        let mut r2 = Rng::new(5);
        let mut seen_low = std::collections::HashSet::new();
        let mut seen_high = std::collections::HashSet::new();
        for _ in 0..1000 {
            seen_low.insert(low.sample(&logits, &mut r1).expect("valid"));
            seen_high.insert(high.sample(&logits, &mut r2).expect("valid"));
        }
        assert!(
            seen_high.len() >= seen_low.len(),
            "higher target should be at least as diverse: {} vs {}",
            seen_high.len(),
            seen_low.len()
        );
    }

    #[test]
    fn sample_finite_path() {
        // Uniform logits: every token equally likely, surprise = log2(n).
        let mut m = Mirostat::new(MirostatConfig {
            target_surprise: 4.0,
            tau_init: 8.0,
            lr: 0.05,
        })
        .expect("valid");
        let mut rng = Rng::new(11);
        let logits = vec![0.0_f32; 16];
        for _ in 0..200 {
            let t = m.sample(&logits, &mut rng).expect("valid");
            assert!(t < 16);
            assert!(m.mu().is_finite());
        }
    }
}
