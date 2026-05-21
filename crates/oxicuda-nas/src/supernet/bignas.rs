//! BigNAS: training a single big supernet with the sandwich rule.
//!
//! Reference: Yu, Jin, Liu, Bender, Kindermans, Tan, Huang, Song & Le,
//! "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage
//! Models", ECCV 2020.
//!
//! # Background
//!
//! Classic one-shot NAS supernets train every weight-shared sub-network with
//! the same `Uniform` strategy that [`crate::supernet::path_sample::PathSampler`]
//! implements. Empirically this is *unbalanced*: the very small and very large
//! sub-networks rarely get the same gradient signal as the average ones, so
//! after training the supernet's ranking of sub-networks decorrelates from a
//! standalone-trained reference. BigNAS proposes to fix this with the
//! **sandwich rule**: on every gradient step, the supernet is trained on the
//! *same* mini-batch using
//!
//! 1. the **MAX** sub-network (all-largest-choice blocks),
//! 2. the **MIN** sub-network (all-smallest-choice blocks), and
//! 3. exactly `S` uniformly-random sub-networks,
//!
//! and the per-loss gradients are summed. The MAX/MIN pair explicitly forces
//! the supernet to reserve capacity for both extremes; the random samples
//! provide unbiased coverage of the in-between configurations. Adding this
//! rule was shown to substantially improve supernet-vs-standalone Kendall
//! rank correlation and to enable single-stage supernets that are good enough
//! to deploy *without* per-sub-network finetuning.
//!
//! # Search-space encoding
//!
//! A BigNAS-style search space is a sequence of **blocks** (e.g. inverted
//! residual blocks in a MobileNet-style backbone). Each block has a fixed
//! number of choices (e.g. expansion ratio 3/4/6, kernel 3/5/7). A
//! sub-network is therefore a vector of choice indices, one per block, where
//! choice `0` is the *smallest* option and choice `n_choices_per_block - 1`
//! is the *largest*. This crate keeps the encoding intentionally minimal so
//! that BigNAS can be plugged into any sequential-block supernet; the actual
//! block weights live elsewhere ([`crate::supernet::weight_share::Supernet`]).
//!
//! # Cost proxy
//!
//! [`BigNasSampler::flops_proxy`] returns the *sum of choice indices* across
//! all blocks. This is intentionally a placeholder ordering function — its
//! sole guarantee is that `flops_proxy(max_subnet) >= flops_proxy(min_subnet)`
//! and that it is monotonic in each block's choice index, which is enough to
//! check the sandwich rule's "MAX uses more compute than MIN" property in
//! unit tests. Production cost models should wire in the calibrated FLOP /
//! latency predictors in [`crate::predictor`].

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── BigNasConfig ─────────────────────────────────────────────────────────────

/// Configuration for a [`BigNasSampler`].
///
/// All three fields are validated at construction time by
/// [`BigNasSampler::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BigNasConfig {
    /// Number of sequential blocks in the supernet. Must be `>= 1`.
    pub n_blocks: usize,
    /// Number of discrete choices per block, indexed `0..n_choices_per_block`
    /// with `0` = smallest and `n_choices_per_block - 1` = largest. Must be
    /// `>= 1`.
    pub n_choices_per_block: usize,
    /// Number of uniformly-random sub-networks to draw in every sandwich
    /// batch *in addition to* the MAX and MIN. May be `0`, in which case
    /// every sandwich batch is exactly `[max, min]`.
    pub sandwich_samples: usize,
}

impl BigNasConfig {
    fn validate(&self) -> NasResult<()> {
        if self.n_blocks == 0 {
            return Err(NasError::InvalidNumNodes { min: 1, got: 0 });
        }
        if self.n_choices_per_block == 0 {
            return Err(NasError::InvalidNumOps);
        }
        // sandwich_samples >= 0 is always true for usize.
        Ok(())
    }
}

// ─── BigNasSampler ───────────────────────────────────────────────────────────

/// BigNAS sub-network sampler implementing the **sandwich rule**.
///
/// The sampler is stateless apart from its configuration: every call takes a
/// mutable [`LcgRng`] so that callers can fully control reproducibility.
#[derive(Debug, Clone, Copy)]
pub struct BigNasSampler {
    /// Read-only configuration. Public so callers can inspect the search-space
    /// dimensions without going through accessor methods.
    pub cfg: BigNasConfig,
}

impl BigNasSampler {
    /// Construct a new sampler after validating `cfg`.
    ///
    /// # Errors
    /// * [`NasError::InvalidNumNodes`] if `cfg.n_blocks == 0`,
    /// * [`NasError::InvalidNumOps`] if `cfg.n_choices_per_block == 0`.
    pub fn new(cfg: BigNasConfig) -> NasResult<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// The all-largest-choice sub-network: every block is set to
    /// `n_choices_per_block - 1`.
    #[must_use]
    pub fn max_subnet(&self) -> Vec<usize> {
        let top = self.cfg.n_choices_per_block.saturating_sub(1);
        vec![top; self.cfg.n_blocks]
    }

    /// The all-smallest-choice sub-network: every block is `0`.
    #[must_use]
    pub fn min_subnet(&self) -> Vec<usize> {
        vec![0usize; self.cfg.n_blocks]
    }

    /// Sample a uniformly-random sub-network: each block's choice is drawn
    /// independently from `Uniform({0, 1, …, n_choices_per_block - 1})` via
    /// [`LcgRng::next_usize`].
    ///
    /// # Errors
    /// * [`NasError::InvalidNumNodes`] / [`NasError::InvalidNumOps`] if the
    ///   sampler was constructed with an invalid `cfg` (these cannot happen
    ///   if `cfg` went through [`BigNasSampler::new`], but the check is
    ///   re-run defensively so direct field construction is also safe).
    pub fn sample_uniform(&self, rng: &mut LcgRng) -> NasResult<Vec<usize>> {
        self.cfg.validate()?;
        let mut subnet = Vec::with_capacity(self.cfg.n_blocks);
        for _ in 0..self.cfg.n_blocks {
            subnet.push(rng.next_usize(self.cfg.n_choices_per_block));
        }
        Ok(subnet)
    }

    /// Produce one **sandwich batch** of sub-networks: `[max, min, R_0, R_1,
    /// …, R_{S-1}]`, where each `R_i` is an independent
    /// [`BigNasSampler::sample_uniform`]. The total length is therefore
    /// `2 + sandwich_samples`.
    ///
    /// The convention `[max, min, randoms…]` matches the reference paper's
    /// pseudocode (Algorithm 1) and lets callers index into the result by a
    /// fixed offset without re-scanning.
    ///
    /// # Errors
    /// Re-runs `BigNasConfig::validate` and propagates any error.
    pub fn sandwich_batch(&self, rng: &mut LcgRng) -> NasResult<Vec<Vec<usize>>> {
        self.cfg.validate()?;
        let mut batch = Vec::with_capacity(2 + self.cfg.sandwich_samples);
        batch.push(self.max_subnet());
        batch.push(self.min_subnet());
        for _ in 0..self.cfg.sandwich_samples {
            batch.push(self.sample_uniform(rng)?);
        }
        Ok(batch)
    }

    /// Placeholder FLOPs proxy: sum of choice indices.
    ///
    /// See module docs for the rationale — this is *not* a calibrated cost
    /// model, only a monotone ordering function that guarantees
    /// `flops_proxy(max_subnet) >= flops_proxy(min_subnet)` for unit-test
    /// invariants. For real cost prediction use
    /// [`crate::predictor::flops::total_cost`] / [`crate::predictor::latency`].
    ///
    /// # Errors
    /// * [`NasError::DimensionMismatch`] if `subnet.len() != cfg.n_blocks`,
    /// * [`NasError::InvalidArchEncoding`] if any entry is
    ///   `>= cfg.n_choices_per_block`.
    pub fn flops_proxy(&self, subnet: &[usize]) -> NasResult<usize> {
        if subnet.len() != self.cfg.n_blocks {
            return Err(NasError::DimensionMismatch {
                expected: self.cfg.n_blocks,
                got: subnet.len(),
            });
        }
        let mut total = 0usize;
        for &c in subnet {
            if c >= self.cfg.n_choices_per_block {
                return Err(NasError::InvalidArchEncoding);
            }
            total = total.saturating_add(c);
        }
        Ok(total)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(n_blocks: usize, n_choices: usize, sandwich: usize) -> BigNasSampler {
        BigNasSampler::new(BigNasConfig {
            n_blocks,
            n_choices_per_block: n_choices,
            sandwich_samples: sandwich,
        })
        .expect("test invariant: valid bignas cfg")
    }

    #[test]
    fn max_subnet_length_and_value() {
        let s = mk(6, 4, 2);
        let m = s.max_subnet();
        assert_eq!(m.len(), 6);
        assert!(m.iter().all(|&c| c == 3));
    }

    #[test]
    fn min_subnet_length_and_value() {
        let s = mk(5, 4, 2);
        let m = s.min_subnet();
        assert_eq!(m.len(), 5);
        assert!(m.iter().all(|&c| c == 0));
    }

    #[test]
    fn sample_uniform_length_and_range() {
        let s = mk(7, 5, 0);
        let mut rng = LcgRng::new(42);
        for _ in 0..50 {
            let sub = s
                .sample_uniform(&mut rng)
                .expect("test invariant: uniform sample");
            assert_eq!(sub.len(), 7);
            assert!(sub.iter().all(|&c| c < 5));
        }
    }

    #[test]
    fn sandwich_batch_total_count() {
        let s = mk(4, 3, 5);
        let mut rng = LcgRng::new(7);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        // 2 + sandwich_samples = 2 + 5 = 7
        assert_eq!(batch.len(), 7);
    }

    #[test]
    fn sandwich_batch_first_is_max() {
        let s = mk(4, 3, 5);
        let mut rng = LcgRng::new(7);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        assert_eq!(batch[0], s.max_subnet());
    }

    #[test]
    fn sandwich_batch_second_is_min() {
        let s = mk(4, 3, 5);
        let mut rng = LcgRng::new(7);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        assert_eq!(batch[1], s.min_subnet());
    }

    #[test]
    fn sandwich_batch_zero_random_is_exactly_max_min() {
        let s = mk(3, 4, 0);
        let mut rng = LcgRng::new(7);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], s.max_subnet());
        assert_eq!(batch[1], s.min_subnet());
    }

    #[test]
    fn sandwich_batch_three_random_total_five() {
        let s = mk(3, 4, 3);
        let mut rng = LcgRng::new(7);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        assert_eq!(batch.len(), 5);
        for sub in &batch[2..] {
            assert_eq!(sub.len(), 3);
            assert!(sub.iter().all(|&c| c < 4));
        }
    }

    #[test]
    fn uniform_distribution_is_roughly_balanced_per_block() {
        // With n_choices_per_block = 4 and 2000 samples, each choice should
        // appear ~500 times in each block. Tolerance ±150 (≈30%) is large
        // enough to be robust to the LCG's modest quality but still rules
        // out a stuck-on-one-choice sampler.
        let s = mk(4, 4, 0);
        let mut rng = LcgRng::new(1234);
        let n_draws = 2000usize;
        let mut counts = vec![vec![0usize; 4]; 4];
        for _ in 0..n_draws {
            let sub = s
                .sample_uniform(&mut rng)
                .expect("test invariant: uniform sample");
            for (b, &c) in sub.iter().enumerate() {
                counts[b][c] += 1;
            }
        }
        for row in &counts {
            for &c in row {
                assert!(c > 350 && c < 650, "imbalanced choice count: {c} of 500");
            }
        }
    }

    #[test]
    fn flops_proxy_max_at_least_min() {
        let s = mk(8, 5, 0);
        let max_f = s
            .flops_proxy(&s.max_subnet())
            .expect("test invariant: flops_proxy max");
        let min_f = s
            .flops_proxy(&s.min_subnet())
            .expect("test invariant: flops_proxy min");
        assert!(max_f >= min_f);
        assert_eq!(max_f, 8 * 4);
        assert_eq!(min_f, 0);
    }

    #[test]
    fn flops_proxy_monotone_uniform_samples() {
        // For any uniform sample, flops_proxy lies in [min, max]. This
        // catches an off-by-one in the index/choice mapping.
        let s = mk(5, 4, 0);
        let mut rng = LcgRng::new(99);
        let max_f = s.flops_proxy(&s.max_subnet()).expect("test invariant: max");
        let min_f = s.flops_proxy(&s.min_subnet()).expect("test invariant: min");
        for _ in 0..200 {
            let sub = s.sample_uniform(&mut rng).expect("test invariant: sample");
            let f = s.flops_proxy(&sub).expect("test invariant: flops_proxy");
            assert!(f >= min_f && f <= max_f);
        }
    }

    #[test]
    fn deterministic_given_same_seed() {
        let s = mk(4, 4, 3);
        let mut rng_a = LcgRng::new(2025);
        let mut rng_b = LcgRng::new(2025);
        let a = s
            .sandwich_batch(&mut rng_a)
            .expect("test invariant: sandwich a");
        let b = s
            .sandwich_batch(&mut rng_b)
            .expect("test invariant: sandwich b");
        assert_eq!(a, b);
    }

    #[test]
    fn err_n_blocks_zero() {
        let r = BigNasSampler::new(BigNasConfig {
            n_blocks: 0,
            n_choices_per_block: 4,
            sandwich_samples: 1,
        });
        assert!(matches!(
            r,
            Err(NasError::InvalidNumNodes { min: 1, got: 0 })
        ));
    }

    #[test]
    fn err_n_choices_per_block_zero() {
        let r = BigNasSampler::new(BigNasConfig {
            n_blocks: 3,
            n_choices_per_block: 0,
            sandwich_samples: 1,
        });
        assert!(matches!(r, Err(NasError::InvalidNumOps)));
    }

    #[test]
    fn err_flops_proxy_wrong_length() {
        let s = mk(4, 4, 0);
        let bad = vec![0usize, 1, 2];
        let r = s.flops_proxy(&bad);
        assert!(matches!(
            r,
            Err(NasError::DimensionMismatch {
                expected: 4,
                got: 3,
            })
        ));
    }

    #[test]
    fn err_flops_proxy_out_of_range_choice() {
        let s = mk(3, 4, 0);
        let bad = vec![0usize, 4, 2]; // 4 == n_choices, illegal
        let r = s.flops_proxy(&bad);
        assert!(matches!(r, Err(NasError::InvalidArchEncoding)));
    }

    #[test]
    fn err_sample_uniform_with_bad_cfg_direct_construction() {
        // Directly construct a BigNasSampler bypassing `new`, then verify
        // that the lazy validation in `sample_uniform` still catches the
        // illegal configuration.
        let bad = BigNasSampler {
            cfg: BigNasConfig {
                n_blocks: 0,
                n_choices_per_block: 4,
                sandwich_samples: 0,
            },
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            bad.sample_uniform(&mut rng),
            Err(NasError::InvalidNumNodes { min: 1, got: 0 })
        ));
    }

    #[test]
    fn n_choices_per_block_one_max_equals_min_all_zero() {
        let s = mk(5, 1, 3);
        assert_eq!(s.max_subnet(), vec![0usize; 5]);
        assert_eq!(s.min_subnet(), vec![0usize; 5]);
        let mut rng = LcgRng::new(13);
        let batch = s
            .sandwich_batch(&mut rng)
            .expect("test invariant: sandwich batch");
        assert_eq!(batch.len(), 5);
        for sub in &batch {
            assert_eq!(sub, &vec![0usize; 5]);
        }
        // flops_proxy of any of them is 0.
        let f = s
            .flops_proxy(&batch[0])
            .expect("test invariant: flops_proxy");
        assert_eq!(f, 0);
    }

    #[test]
    fn err_sandwich_batch_with_bad_cfg() {
        let bad = BigNasSampler {
            cfg: BigNasConfig {
                n_blocks: 3,
                n_choices_per_block: 0,
                sandwich_samples: 2,
            },
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            bad.sandwich_batch(&mut rng),
            Err(NasError::InvalidNumOps)
        ));
    }

    #[test]
    fn cfg_validate_accepts_sandwich_zero() {
        // sandwich_samples == 0 is allowed (the [max, min] degenerate batch).
        let r = BigNasSampler::new(BigNasConfig {
            n_blocks: 1,
            n_choices_per_block: 1,
            sandwich_samples: 0,
        });
        assert!(r.is_ok());
    }
}
