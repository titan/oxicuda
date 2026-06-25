//! Adaptive negative sampler with importance weighting.
//!
//! Reference: Steffen Rendle and Christoph Freudenthaler, "Improving Pairwise
//! Learning for Item Recommendation from Implicit Feedback", WSDM 2014; and the
//! *adaptive sampled importance resampling* (AdaSIR) family (Chen et al. 2022,
//! "Learning Recommenders for Implicit Feedback with Importance Resampling",
//! WWW).
//!
//! # Idea
//!
//! Uniform negative sampling wastes most updates on *easy* negatives whose
//! pairwise gradient is already ≈ 0. An adaptive sampler instead draws negatives
//! roughly in proportion to how *informative* they are under the current model
//! score `s(u, j)`.
//!
//! A naive way to draw `j ∝ exp(s(u, j) / τ)` would require scanning every item
//! per update. Sampled-importance-resampling (SIR) avoids that:
//!
//! 1. **Proposal.** Draw a small pool of `pool_size` candidate negatives
//!    `j_1 … j_m` from a cheap proposal distribution `q` (uniform over
//!    non-positive items here).
//! 2. **Importance weights.** Give each candidate weight
//!    `w_k = exp(s(u, j_k) / τ) / q(j_k)`. With a uniform proposal the `q(j_k)`
//!    factor is constant and cancels in the normalisation.
//! 3. **Resample.** Draw the returned negative `j_k` with probability
//!    `w_k / Σ_l w_l`. This makes the marginal sampling distribution approach the
//!    target `∝ exp(s/τ)` as `pool_size → n_items`.
//!
//! The sampler also returns a *self-normalised importance weight*
//! `ŵ = (Σ_l w_l) / (pool_size · w_chosen)` that can be used to debias the
//! pairwise gradient (importance-weighted SGD): an easy pool (all weights
//! similar) yields `ŵ ≈ 1`, while a pool that contains one dominant hard
//! negative yields `ŵ < 1` for that negative, down-weighting its (large but
//! rare) gradient so the estimator stays unbiased for the *uniform* objective.
//!
//! All randomness flows through the crate [`LcgRng`]; the sampler holds no model
//! state of its own — scores are supplied per call so it composes with any
//! factorisation model.

use std::collections::BTreeSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// A negative drawn by the adaptive sampler, together with its self-normalised
/// importance weight for debiasing the pairwise update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveNegative {
    /// The sampled item id (guaranteed `∉ user_positives`).
    pub item: usize,
    /// Self-normalised importance weight `ŵ ∈ (0, 1]` for gradient debiasing.
    pub weight: f32,
}

/// Configuration for the adaptive importance-resampling negative sampler.
#[derive(Debug, Clone)]
pub struct AdaptiveNegConfig {
    /// Number of items in the catalogue.
    pub n_items: usize,
    /// Candidate pool size drawn from the uniform proposal per sample.
    pub pool_size: usize,
    /// Softmax temperature `τ > 0`; larger ⇒ closer to uniform, smaller ⇒
    /// greedier toward the single highest-scoring candidate.
    pub temperature: f32,
}

impl AdaptiveNegConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumItems`] if `n_items == 0`.
    /// - [`RecsysError::InvalidConfig`] if `pool_size == 0` or `temperature <= 0`.
    pub fn validate(&self) -> RecsysResult<()> {
        if self.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: self.n_items });
        }
        if self.pool_size == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "pool_size must be >= 1".into(),
            });
        }
        if self.temperature <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "temperature must be > 0".into(),
            });
        }
        Ok(())
    }
}

/// Adaptive sampler that resamples a uniform candidate pool in proportion to a
/// model score, returning the negative and an importance weight.
#[derive(Debug, Clone)]
pub struct AdaptiveNegSampler {
    cfg: AdaptiveNegConfig,
}

impl AdaptiveNegSampler {
    /// Builds a sampler from a validated configuration.
    ///
    /// # Errors
    /// Propagates [`AdaptiveNegConfig::validate`].
    pub fn new(cfg: AdaptiveNegConfig) -> RecsysResult<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Number of catalogue items.
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.cfg.n_items
    }

    /// Draw one uniform non-positive candidate via rejection (≤ 100 tries).
    fn propose(&self, user_positives: &BTreeSet<usize>, rng: &mut LcgRng) -> Option<usize> {
        for _ in 0..100 {
            let candidate = (rng.next_u32() as usize) % self.cfg.n_items;
            if !user_positives.contains(&candidate) {
                return Some(candidate);
            }
        }
        None
    }

    /// Sample an adaptive negative for `user`, scoring candidates with `score_fn`.
    ///
    /// `score_fn(item)` must return the (higher-is-more-relevant) model score
    /// `s(u, item)`; the sampler internally subtracts the pool maximum before
    /// exponentiation for numerical stability, so unbounded scores are fine.
    ///
    /// # Errors
    /// - [`RecsysError::NoNegativeAvailable`] if no non-positive item could be
    ///   drawn for the candidate pool (the user has positives over essentially
    ///   the whole catalogue).
    pub fn sample_with<F>(
        &self,
        user: usize,
        user_positives: &BTreeSet<usize>,
        rng: &mut LcgRng,
        mut score_fn: F,
    ) -> RecsysResult<AdaptiveNegative>
    where
        F: FnMut(usize) -> f32,
    {
        let mut items: Vec<usize> = Vec::with_capacity(self.cfg.pool_size);
        let mut logits: Vec<f32> = Vec::with_capacity(self.cfg.pool_size);
        for _ in 0..self.cfg.pool_size {
            match self.propose(user_positives, rng) {
                Some(candidate) => {
                    items.push(candidate);
                    logits.push(score_fn(candidate) / self.cfg.temperature);
                }
                None => break,
            }
        }
        if items.is_empty() {
            return Err(RecsysError::NoNegativeAvailable { user });
        }

        // Stable softmax over the candidate logits → unnormalised weights.
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            // Degenerate scores (all -inf / NaN): fall back to uniform pick.
            let idx = (rng.next_u32() as usize) % items.len();
            return Ok(AdaptiveNegative {
                item: items[idx],
                weight: 1.0,
            });
        }
        for w in &mut weights {
            *w /= weight_sum;
        }

        // Resample one candidate ∝ normalised weight via CDF inversion.
        let u01 = rng.next_u32() as f64 / 2f64.powi(32);
        let mut cumulative = 0.0_f64;
        let mut chosen = items.len() - 1;
        for (k, &w) in weights.iter().enumerate() {
            cumulative += f64::from(w);
            if u01 < cumulative {
                chosen = k;
                break;
            }
        }

        // Self-normalised importance weight for debiasing the uniform objective:
        //   ŵ = (mean target weight) / (chosen target weight)
        //     = (Σ_l ŵ_l / m) / ŵ_chosen          with ŵ_l the normalised probs.
        // Easy pool (uniform weights) ⇒ ŵ = 1; dominant hard negative ⇒ ŵ < 1.
        let n = items.len() as f32;
        let chosen_prob = weights[chosen].max(f32::MIN_POSITIVE);
        let debias = (1.0 / n) / chosen_prob;
        let weight = debias.clamp(f32::MIN_POSITIVE, 1.0);

        Ok(AdaptiveNegative {
            item: items[chosen],
            weight,
        })
    }

    /// Convenience wrapper scoring candidates as `dot(user_emb, item_emb_row)`
    /// over a flat `[n_items × dim]` item-embedding matrix.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidEmbeddingDim`] if `user_emb` is empty.
    /// - [`RecsysError::DimensionMismatch`] if `item_embs.len() != n_items · dim`.
    /// - Propagates [`Self::sample_with`].
    pub fn sample_dot(
        &self,
        user: usize,
        user_positives: &BTreeSet<usize>,
        user_emb: &[f32],
        item_embs: &[f32],
        rng: &mut LcgRng,
    ) -> RecsysResult<AdaptiveNegative> {
        let dim = user_emb.len();
        if dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: dim });
        }
        if item_embs.len() != self.cfg.n_items * dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.cfg.n_items * dim,
                got: item_embs.len(),
            });
        }
        self.sample_with(user, user_positives, rng, |item| {
            let row = &item_embs[item * dim..(item + 1) * dim];
            user_emb.iter().zip(row.iter()).map(|(&u, &e)| u * e).sum()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positives() -> BTreeSet<usize> {
        let mut p = BTreeSet::new();
        p.insert(0usize);
        p.insert(1);
        p
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(
            AdaptiveNegSampler::new(AdaptiveNegConfig {
                n_items: 0,
                pool_size: 4,
                temperature: 1.0,
            })
            .is_err()
        );
        assert!(
            AdaptiveNegSampler::new(AdaptiveNegConfig {
                n_items: 10,
                pool_size: 0,
                temperature: 1.0,
            })
            .is_err()
        );
        assert!(
            AdaptiveNegSampler::new(AdaptiveNegConfig {
                n_items: 10,
                pool_size: 4,
                temperature: 0.0,
            })
            .is_err()
        );
    }

    #[test]
    fn never_samples_a_positive() {
        let sampler = AdaptiveNegSampler::new(AdaptiveNegConfig {
            n_items: 50,
            pool_size: 8,
            temperature: 0.5,
        })
        .expect("config ok");
        let pos = positives();
        let mut rng = LcgRng::new(7);
        for _ in 0..200 {
            let neg = sampler
                .sample_with(0, &pos, &mut rng, |item| item as f32)
                .expect("sample ok");
            assert!(!pos.contains(&neg.item), "drew positive {}", neg.item);
            assert!(neg.weight > 0.0 && neg.weight <= 1.0);
        }
    }

    #[test]
    fn prefers_high_scoring_negatives() {
        // Item 49 has by far the highest score; with a low temperature the
        // adaptive sampler should pick it far more often than uniform (≈ 1/48).
        let sampler = AdaptiveNegSampler::new(AdaptiveNegConfig {
            n_items: 50,
            pool_size: 16,
            temperature: 0.1,
        })
        .expect("config ok");
        let pos = positives();
        let mut rng = LcgRng::new(123);
        let target = 49usize;
        let mut hits = 0usize;
        let trials = 4000usize;
        for _ in 0..trials {
            let neg = sampler
                .sample_with(
                    0,
                    &pos,
                    &mut rng,
                    |item| {
                        if item == target { 10.0 } else { 0.0 }
                    },
                )
                .expect("sample ok");
            if neg.item == target {
                hits += 1;
            }
        }
        let freq = hits as f32 / trials as f32;
        // Uniform would give ≈ 1/48 ≈ 0.021; adaptive must be dramatically larger.
        assert!(
            freq > 0.2,
            "adaptive sampler should favour the hard negative, freq={freq}"
        );
    }

    #[test]
    fn uniform_scores_give_unit_weight() {
        // With identical scores every normalised prob is 1/m, so the debias
        // weight ŵ = (1/m)/(1/m) = 1 exactly.
        let sampler = AdaptiveNegSampler::new(AdaptiveNegConfig {
            n_items: 30,
            pool_size: 8,
            temperature: 1.0,
        })
        .expect("config ok");
        let pos = positives();
        let mut rng = LcgRng::new(55);
        for _ in 0..100 {
            let neg = sampler
                .sample_with(0, &pos, &mut rng, |_| 3.0)
                .expect("sample ok");
            assert!(
                (neg.weight - 1.0).abs() < 1e-4,
                "uniform-score weight must be 1, got {}",
                neg.weight
            );
        }
    }

    #[test]
    fn hard_negative_is_down_weighted() {
        // A single dominant negative is chosen almost always, but its importance
        // weight must be < 1 so its (large) gradient is debiased.
        let sampler = AdaptiveNegSampler::new(AdaptiveNegConfig {
            n_items: 40,
            pool_size: 12,
            temperature: 0.05,
        })
        .expect("config ok");
        let pos = positives();
        let mut rng = LcgRng::new(900);
        let target = 39usize;
        let mut saw_target = false;
        for _ in 0..500 {
            let neg = sampler
                .sample_with(
                    0,
                    &pos,
                    &mut rng,
                    |item| {
                        if item == target { 20.0 } else { 0.0 }
                    },
                )
                .expect("sample ok");
            if neg.item == target {
                saw_target = true;
                assert!(
                    neg.weight < 1.0,
                    "dominant hard negative must be down-weighted, got {}",
                    neg.weight
                );
            }
        }
        assert!(saw_target, "expected to draw the dominant hard negative");
    }

    #[test]
    fn dot_helper_matches_manual_score() {
        let sampler = AdaptiveNegSampler::new(AdaptiveNegConfig {
            n_items: 4,
            pool_size: 4,
            temperature: 1.0,
        })
        .expect("config ok");
        let mut pos = BTreeSet::new();
        pos.insert(0usize);
        let user_emb = vec![1.0_f32, 0.0];
        // rows: item0 unused (positive), item1 = (2,0), item2 = (0,5), item3=(3,0)
        let item_embs = vec![0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 3.0, 0.0];
        let mut rng = LcgRng::new(11);
        // item3 has dot=3, highest among non-positives ⇒ most frequent at τ=1.
        let mut counts = [0usize; 4];
        for _ in 0..3000 {
            let neg = sampler
                .sample_dot(0, &pos, &user_emb, &item_embs, &mut rng)
                .expect("sample ok");
            counts[neg.item] += 1;
        }
        assert_eq!(counts[0], 0, "positive item never sampled");
        assert!(
            counts[3] > counts[1] && counts[3] > counts[2],
            "highest-dot negative should dominate: {counts:?}"
        );
    }
}
