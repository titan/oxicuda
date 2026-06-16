//! Expert dropout: stochastic expert masking for MoE regularisation.
//!
//! During training, randomly masking a fraction of experts forces the router to
//! spread token assignments across the remaining experts, improving robustness
//! and preventing the collapse of routing onto a small subset of experts
//! ("expert specialisation collapse"). This mirrors the role of standard dropout
//! but operates on the *expert* axis of the gate distribution rather than on
//! individual activations.
//!
//! Two complementary mechanisms are provided:
//!
//! * [`ExpertDropout::sample_mask`] — draws a boolean keep-mask over experts
//!   (each expert dropped i.i.d. with probability `p`), with a safeguard
//!   guaranteeing at least one expert survives.
//! * [`ExpertDropout::apply`] — applies a keep-mask to a `[T×E]` gate matrix:
//!   dropped-expert columns are zeroed, surviving columns are renormalised per
//!   token and rescaled by the inverted-dropout factor `1/(1−p)` so the expected
//!   gate magnitude is preserved between train and eval.
//!
//! At evaluation time the mask is the identity (keep all experts), so the layer
//! is a no-op — matching the standard "inverted dropout" convention.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;

/// Configuration for expert dropout.
#[derive(Debug, Clone)]
pub struct ExpertDropoutConfig {
    /// Number of experts `E`.
    pub n_experts: usize,
    /// Per-expert drop probability `p ∈ [0, 1)`.
    pub drop_prob: f32,
}

/// Stateless expert-dropout operator.
#[derive(Debug, Clone)]
pub struct ExpertDropout {
    /// Routing / dropout configuration.
    pub config: ExpertDropoutConfig,
}

impl ExpertDropout {
    /// Create a new expert-dropout operator.
    ///
    /// # Errors
    /// Returns [`MoeError`] when `n_experts == 0` or `drop_prob` is not in
    /// `[0, 1)` / is non-finite.
    pub fn new(config: ExpertDropoutConfig) -> MoeResult<Self> {
        if config.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: config.n_experts,
            });
        }
        if !config.drop_prob.is_finite() || config.drop_prob < 0.0 || config.drop_prob >= 1.0 {
            return Err(MoeError::Internal {
                msg: format!("invalid drop_prob {}: must be in [0, 1)", config.drop_prob),
            });
        }
        Ok(Self { config })
    }

    /// Inverted-dropout scale `1 / (1 − p)`.
    #[must_use]
    pub fn keep_scale(&self) -> f32 {
        1.0 / (1.0 - self.config.drop_prob)
    }

    /// Sample a boolean keep-mask of length `n_experts`.
    ///
    /// Each expert is kept with probability `1 − p`. If every expert would be
    /// dropped, a single uniformly-chosen expert is forced to survive so the
    /// gate never becomes all-zero.
    pub fn sample_mask(&self, rng: &mut LcgRng) -> Vec<bool> {
        let n_e = self.config.n_experts;
        let p = self.config.drop_prob;
        let mut mask = vec![true; n_e];
        for keep in mask.iter_mut() {
            // Drop when the uniform draw falls below p.
            *keep = rng.next_f32() >= p;
        }
        if mask.iter().all(|&k| !k) {
            let idx = rng.next_usize(n_e);
            mask[idx] = true;
        }
        mask
    }

    /// Apply a keep-mask to a `[T×E]` gate matrix (row-major), in place.
    ///
    /// Dropped columns are zeroed; per token the surviving entries are
    /// renormalised to sum to `1` and then scaled by [`Self::keep_scale`].
    /// A token whose mass is entirely on dropped experts is left as a zero row
    /// (it contributes no expert output for this step).
    ///
    /// # Errors
    /// Returns [`MoeError`] on a `gates`/`T·E` length mismatch or a wrong-length
    /// mask.
    pub fn apply(&self, gates: &mut [f32], n_tokens: usize, mask: &[bool]) -> MoeResult<()> {
        let n_e = self.config.n_experts;
        if mask.len() != n_e {
            return Err(MoeError::DimensionMismatch {
                expected: n_e,
                got: mask.len(),
            });
        }
        let expected = n_tokens * n_e;
        if gates.len() != expected {
            return Err(MoeError::DimensionMismatch {
                expected,
                got: gates.len(),
            });
        }
        if n_tokens == 0 {
            return Err(MoeError::EmptyInput);
        }

        let scale = self.keep_scale();
        for t in 0..n_tokens {
            let base = t * n_e;
            // Zero dropped columns and accumulate surviving mass.
            let mut surviving = 0.0_f32;
            for e in 0..n_e {
                if mask[e] {
                    surviving += gates[base + e];
                } else {
                    gates[base + e] = 0.0;
                }
            }
            if surviving > 1e-12 {
                let renorm = scale / surviving;
                for e in 0..n_e {
                    if mask[e] {
                        gates[base + e] *= renorm;
                    }
                }
            }
        }
        Ok(())
    }

    /// Convenience: sample a mask and apply it, returning the mask used.
    ///
    /// # Errors
    /// Propagates errors from [`Self::apply`].
    pub fn forward(
        &self,
        gates: &mut [f32],
        n_tokens: usize,
        rng: &mut LcgRng,
    ) -> MoeResult<Vec<bool>> {
        let mask = self.sample_mask(rng);
        self.apply(gates, n_tokens, &mask)?;
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_experts_errors() {
        let cfg = ExpertDropoutConfig {
            n_experts: 0,
            drop_prob: 0.1,
        };
        assert!(matches!(
            ExpertDropout::new(cfg),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn new_invalid_prob_errors() {
        for p in [-0.1_f32, 1.0, 1.5, f32::NAN] {
            let cfg = ExpertDropoutConfig {
                n_experts: 4,
                drop_prob: p,
            };
            assert!(ExpertDropout::new(cfg).is_err(), "p={p} should error");
        }
    }

    #[test]
    fn keep_scale_correct() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.5,
        })
        .expect("value should be present");
        assert!((d.keep_scale() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn mask_length_equals_n_experts() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 8,
            drop_prob: 0.3,
        })
        .expect("value should be present");
        let mut rng = LcgRng::new(1);
        let mask = d.sample_mask(&mut rng);
        assert_eq!(mask.len(), 8);
    }

    #[test]
    fn mask_zero_prob_keeps_all() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 6,
            drop_prob: 0.0,
        })
        .expect("value should be present");
        let mut rng = LcgRng::new(2);
        let mask = d.sample_mask(&mut rng);
        assert!(mask.iter().all(|&k| k), "p=0 must keep every expert");
    }

    #[test]
    fn mask_never_all_dropped() {
        // High drop prob still must leave ≥1 expert.
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.99,
        })
        .expect("value should be present");
        let mut rng = LcgRng::new(3);
        for _ in 0..200 {
            let mask = d.sample_mask(&mut rng);
            assert!(mask.iter().any(|&k| k), "at least one expert must survive");
        }
    }

    #[test]
    fn apply_zeros_dropped_columns() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.5,
        })
        .expect("value should be present");
        let mut gates = vec![0.25_f32; 2 * 4];
        let mask = vec![true, false, true, false];
        d.apply(&mut gates, 2, &mask).expect("apply should succeed");
        for t in 0..2 {
            assert_eq!(gates[t * 4 + 1], 0.0);
            assert_eq!(gates[t * 4 + 3], 0.0);
            assert!(gates[t * 4] > 0.0);
            assert!(gates[t * 4 + 2] > 0.0);
        }
    }

    #[test]
    fn apply_renormalises_with_inverted_scale() {
        // Two surviving experts each 0.25 → renorm to sum 1, then ×scale (=2 here).
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.5,
        })
        .expect("value should be present");
        let mut gates = vec![0.25_f32; 4];
        let mask = vec![true, true, false, false];
        d.apply(&mut gates, 1, &mask).expect("apply should succeed");
        // surviving=0.5; renorm = scale/surviving = 2/0.5 = 4; each 0.25*4 = 1.0
        assert!((gates[0] - 1.0).abs() < 1e-5, "g0={}", gates[0]);
        assert!((gates[1] - 1.0).abs() < 1e-5, "g1={}", gates[1]);
    }

    #[test]
    fn apply_mask_length_mismatch_errors() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.2,
        })
        .expect("value should be present");
        let mut gates = vec![0.25_f32; 4];
        let mask = vec![true, false]; // wrong length
        assert!(matches!(
            d.apply(&mut gates, 1, &mask),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn apply_gates_length_mismatch_errors() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.2,
        })
        .expect("value should be present");
        let mut gates = vec![0.25_f32; 10]; // not 2*4
        let mask = vec![true; 4];
        assert!(matches!(
            d.apply(&mut gates, 2, &mask),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn apply_all_kept_zero_prob_is_identity_in_distribution() {
        // p=0 → scale=1; a normalised row stays normalised.
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.0,
        })
        .expect("value should be present");
        let mut gates = vec![0.1_f32, 0.2, 0.3, 0.4];
        let mask = vec![true; 4];
        d.apply(&mut gates, 1, &mask).expect("apply should succeed");
        let sum: f32 = gates.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row sum {sum} should stay 1");
    }

    #[test]
    fn forward_returns_mask_and_modifies_gates() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 4,
            drop_prob: 0.5,
        })
        .expect("value should be present");
        let mut rng = LcgRng::new(7);
        let mut gates = vec![0.25_f32; 3 * 4];
        let mask = d
            .forward(&mut gates, 3, &mut rng)
            .expect("forward should succeed");
        assert_eq!(mask.len(), 4);
        // Dropped columns must be zero in every row.
        for (e, &keep) in mask.iter().enumerate() {
            if !keep {
                for t in 0..3 {
                    assert_eq!(gates[t * 4 + e], 0.0);
                }
            }
        }
    }

    #[test]
    fn forward_deterministic_for_same_seed() {
        let d = ExpertDropout::new(ExpertDropoutConfig {
            n_experts: 8,
            drop_prob: 0.4,
        })
        .expect("value should be present");
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let mut g_a = vec![0.125_f32; 4 * 8];
        let mut g_b = vec![0.125_f32; 4 * 8];
        let m_a = d
            .forward(&mut g_a, 4, &mut rng_a)
            .expect("forward should succeed");
        let m_b = d
            .forward(&mut g_b, 4, &mut rng_b)
            .expect("forward should succeed");
        assert_eq!(m_a, m_b);
        assert_eq!(g_a, g_b);
    }
}
