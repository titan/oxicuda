//! Differentiable expert capacity: a learnable per-expert capacity scale.
//!
//! Classic Switch / GShard dispatch gives **every** expert the same fixed
//! capacity `C = ⌈T/E · capacity_factor⌉`. That is wasteful when the token
//! distribution is skewed — popular experts overflow while rare experts sit
//! half-empty. Here each expert `i` owns a learnable scalar `s_i`; the
//! per-expert capacity is allocated *proportionally* to a positive, smooth
//! function of those scales while keeping the **total** capacity budget fixed:
//!
//! ```text
//! a_i      = softplus(s_i)                       (positive, differentiable)
//! share_i  = a_i / Σ_j a_j                        (Σ share = 1)
//! C_i      = round(share_i · E · base_capacity)   (Σ C_i ≈ E · base_capacity)
//! ```
//!
//! so capacity flows toward experts whose scale grows, and the scales receive a
//! gradient through `softplus`. Each `C_i` is clamped to `[min_capacity, T]`.
//!
//! For the *assignment* itself we provide a **soft capacity gate**: instead of a
//! hard "in/out at rank `C_i`" cutoff, a token at within-expert rank `r`
//! (`0`-based, by descending gate score) receives a differentiable keep weight
//!
//! ```text
//! keep(r) = σ( (C_i − r − 0.5) / τ )
//! ```
//!
//! which is `≈1` well inside capacity, `≈0` well past it, and smoothly `0.5` at
//! the boundary — letting capacity itself be optimised end-to-end (`τ` is a
//! temperature). As `τ → 0` this recovers the hard Switch cutoff.

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::routing::conditional::sigmoid;

/// `softplus(x) = ln(1 + e^x)`, numerically stable for large `|x|`.
#[inline]
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Configuration for [`DifferentiableCapacity`].
#[derive(Debug, Clone)]
pub struct DiffCapacityConfig {
    /// Number of experts (`> 0`).
    pub n_experts: usize,
    /// Per-expert base capacity (the uniform capacity each expert would get).
    /// The total budget is `n_experts · base_capacity`.
    pub base_capacity: usize,
    /// Minimum capacity floor per expert (`≥ 1`).
    pub min_capacity: usize,
    /// Soft-gate temperature `τ` (`> 0`); smaller is sharper.
    pub temperature: f32,
}

impl DiffCapacityConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`MoeError`] for zero experts, a zero `base_capacity`, a zero
    /// `min_capacity`, or a non-positive temperature.
    pub fn validate(&self) -> MoeResult<()> {
        if self.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: self.n_experts,
            });
        }
        if self.base_capacity == 0 {
            return Err(MoeError::InvalidCapacityFactor { factor: 0.0 });
        }
        if self.min_capacity == 0 {
            return Err(MoeError::Internal {
                msg: "min_capacity must be >= 1".to_string(),
            });
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(MoeError::Internal {
                msg: format!("invalid temperature {}", self.temperature),
            });
        }
        Ok(())
    }
}

/// A learnable allocator of per-expert capacity.
#[derive(Debug, Clone)]
pub struct DifferentiableCapacity {
    /// Per-expert learnable capacity scales `s_i`, shape `[n_experts]`.
    pub scales: Vec<f32>,
    /// Configuration.
    pub config: DiffCapacityConfig,
}

impl DifferentiableCapacity {
    /// Create an allocator with all scales `0` (⇒ uniform capacity).
    ///
    /// # Errors
    /// Propagates [`DiffCapacityConfig::validate`].
    pub fn new(cfg: DiffCapacityConfig) -> MoeResult<Self> {
        cfg.validate()?;
        let scales = vec![0.0_f32; cfg.n_experts];
        Ok(Self {
            scales,
            config: cfg,
        })
    }

    /// Create an allocator with small random scales (`N(0, init_std²)`).
    ///
    /// # Errors
    /// Propagates [`DiffCapacityConfig::validate`].
    pub fn new_random(cfg: DiffCapacityConfig, init_std: f32, rng: &mut LcgRng) -> MoeResult<Self> {
        cfg.validate()?;
        let mut scales = vec![0.0_f32; cfg.n_experts];
        rng.fill_normal_scaled(&mut scales, init_std);
        Ok(Self {
            scales,
            config: cfg,
        })
    }

    /// Positive capacity *shares* (`softplus(s_i)/Σ softplus(s_j)`); sums to `1`.
    #[must_use]
    pub fn shares(&self) -> Vec<f32> {
        let act: Vec<f32> = self.scales.iter().map(|&s| softplus(s)).collect();
        let total: f32 = act.iter().sum::<f32>().max(1e-12);
        act.iter().map(|&a| a / total).collect()
    }

    /// Integer per-expert capacities, each clamped to `[min_capacity, n_tokens]`.
    ///
    /// The total budget targeted is `n_experts · base_capacity`.
    #[must_use]
    pub fn capacities(&self, n_tokens: usize) -> Vec<usize> {
        let cfg = &self.config;
        let budget = (cfg.n_experts * cfg.base_capacity) as f32;
        let shares = self.shares();
        let upper = n_tokens.max(cfg.min_capacity);
        shares
            .iter()
            .map(|&sh| {
                let raw = (sh * budget).round() as i64;
                let raw = raw.max(0) as usize;
                raw.clamp(cfg.min_capacity, upper)
            })
            .collect()
    }

    /// Differentiable keep-weight for a token at within-expert rank `r` of
    /// expert `expert_idx`, given `n_tokens`.
    ///
    /// `keep = σ((C_i − r − 0.5)/τ)` — smooth around the capacity boundary.
    ///
    /// # Errors
    /// Returns [`MoeError::ExpertIndexOutOfRange`] for an invalid expert.
    pub fn soft_keep_weight(
        &self,
        expert_idx: usize,
        rank: usize,
        n_tokens: usize,
    ) -> MoeResult<f32> {
        if expert_idx >= self.config.n_experts {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: expert_idx,
                n_experts: self.config.n_experts,
            });
        }
        let cap = self.capacities(n_tokens)[expert_idx] as f32;
        let margin = (cap - rank as f32 - 0.5) / self.config.temperature;
        Ok(sigmoid(margin))
    }

    /// Compute a soft keep-weight for every `(token, expert)` assignment.
    ///
    /// `assignments[t]` is the expert chosen for token `t` (or `usize::MAX` for a
    /// dropped/unrouted token, which gets weight `0`). Within each expert tokens
    /// are ranked by descending `gate_scores[t]`; the returned weight applies the
    /// soft capacity gate at that rank. Output shape `[n_tokens]`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on length mismatches or an out-of-range expert.
    pub fn soft_capacity_gate(
        &self,
        assignments: &[usize],
        gate_scores: &[f32],
        n_tokens: usize,
    ) -> MoeResult<Vec<f32>> {
        if assignments.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: assignments.len(),
            });
        }
        if gate_scores.len() != n_tokens {
            return Err(MoeError::DimensionMismatch {
                expected: n_tokens,
                got: gate_scores.len(),
            });
        }

        // Group token indices by expert.
        let mut per_expert: Vec<Vec<usize>> = vec![Vec::new(); self.config.n_experts];
        for (tok, &a) in assignments.iter().enumerate() {
            if a == usize::MAX {
                continue;
            }
            if a >= self.config.n_experts {
                return Err(MoeError::ExpertIndexOutOfRange {
                    idx: a,
                    n_experts: self.config.n_experts,
                });
            }
            per_expert[a].push(tok);
        }

        let caps = self.capacities(n_tokens);
        let tau = self.config.temperature;
        let mut weights = vec![0.0_f32; n_tokens];
        for (expert_idx, toks) in per_expert.iter_mut().enumerate() {
            // Rank within expert by descending gate score (stable on ties).
            toks.sort_by(|&a, &b| {
                gate_scores[b]
                    .partial_cmp(&gate_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let cap = caps[expert_idx] as f32;
            for (rank, &tok) in toks.iter().enumerate() {
                let margin = (cap - rank as f32 - 0.5) / tau;
                weights[tok] = sigmoid(margin);
            }
        }
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> DiffCapacityConfig {
        DiffCapacityConfig {
            n_experts: 4,
            base_capacity: 8,
            min_capacity: 1,
            temperature: 0.5,
        }
    }

    #[test]
    fn softplus_properties() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert!(softplus(30.0) > 29.9); // ~= x for large x
        assert!(softplus(-30.0) >= 0.0 && softplus(-30.0) < 1e-6);
        assert!(softplus(1.0) > softplus(-1.0));
    }

    #[test]
    fn zero_scales_give_uniform_capacity() {
        let alloc = DifferentiableCapacity::new(base_cfg()).expect("new should succeed");
        let shares = alloc.shares();
        for &s in &shares {
            assert!((s - 0.25).abs() < 1e-6, "share {s} != 1/4");
        }
        // Total budget = 4·8 = 32; uniform ⇒ each gets 8.
        let caps = alloc.capacities(100);
        assert_eq!(caps, vec![8, 8, 8, 8]);
    }

    #[test]
    fn larger_scale_gets_more_capacity() {
        let mut alloc = DifferentiableCapacity::new(base_cfg()).expect("new should succeed");
        alloc.scales = vec![3.0, 0.0, 0.0, 0.0]; // expert 0 favoured
        let shares = alloc.shares();
        assert!(shares[0] > shares[1], "favoured expert share not larger");
        let caps = alloc.capacities(200);
        assert!(
            caps[0] > caps[1],
            "favoured expert capacity {} !> {}",
            caps[0],
            caps[1]
        );
        // Budget roughly preserved (rounding tolerance ±n_experts).
        let total: usize = caps.iter().sum();
        assert!((28..=36).contains(&total), "budget drifted: {total}");
    }

    #[test]
    fn shares_sum_to_one() {
        let mut rng = LcgRng::new(11);
        let alloc = DifferentiableCapacity::new_random(base_cfg(), 1.0, &mut rng)
            .expect("new should succeed");
        let s: f32 = alloc.shares().iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "shares sum {s} != 1");
    }

    #[test]
    fn capacity_clamped_to_min() {
        let cfg = DiffCapacityConfig {
            n_experts: 3,
            base_capacity: 10,
            min_capacity: 4,
            temperature: 0.3,
        };
        let mut alloc = DifferentiableCapacity::new(cfg).expect("new should succeed");
        // Starve experts 1,2 so their raw capacity would be ~0.
        alloc.scales = vec![10.0, -10.0, -10.0];
        let caps = alloc.capacities(100);
        assert!(
            caps[1] >= 4 && caps[2] >= 4,
            "min_capacity floor not applied"
        );
    }

    #[test]
    fn soft_keep_weight_decreases_with_rank() {
        let mut alloc = DifferentiableCapacity::new(base_cfg()).expect("new should succeed");
        alloc.scales = vec![0.0; 4]; // capacity 8 each at n_tokens=100
        let inside = alloc
            .soft_keep_weight(0, 0, 100)
            .expect("soft_keep_weight should succeed");
        let boundary = alloc
            .soft_keep_weight(0, 8, 100)
            .expect("soft_keep_weight should succeed");
        let outside = alloc
            .soft_keep_weight(0, 20, 100)
            .expect("soft_keep_weight should succeed");
        assert!(inside > 0.99, "deep-inside keep {inside} should be ~1");
        // rank 8 vs cap 8: margin = (8-8-0.5)/0.5 = -1 ⇒ σ(-1) ≈ 0.27.
        assert!(boundary > 0.2 && boundary < 0.5, "boundary keep {boundary}");
        assert!(outside < 0.01, "far-outside keep {outside} should be ~0");
        assert!(inside > boundary && boundary > outside);
    }

    #[test]
    fn soft_capacity_gate_ranks_by_score() {
        // One expert, capacity 1 (tiny budget), temperature small: only the
        // highest-scoring token should keep ~1, the rest ~0.
        let cfg = DiffCapacityConfig {
            n_experts: 1,
            base_capacity: 1,
            min_capacity: 1,
            temperature: 0.1,
        };
        let alloc = DifferentiableCapacity::new(cfg).expect("new should succeed");
        let n_tokens = 4;
        let assignments = vec![0_usize; n_tokens];
        let gate_scores = vec![0.1_f32, 0.9, 0.4, 0.2]; // token 1 wins
        let w = alloc
            .soft_capacity_gate(&assignments, &gate_scores, n_tokens)
            .expect("soft_capacity_gate should succeed");
        // Token 1 (rank 0) keeps ~1; all others past capacity ~0.
        assert!(w[1] > 0.99, "top token keep {} should be ~1", w[1]);
        for &t in &[0_usize, 2, 3] {
            assert!(w[t] < 0.01, "overflow token {t} keep {} should be ~0", w[t]);
        }
    }

    #[test]
    fn dropped_tokens_get_zero_weight() {
        let alloc = DifferentiableCapacity::new(base_cfg()).expect("new should succeed");
        let assignments = vec![0_usize, usize::MAX, 1, usize::MAX];
        let scores = vec![0.5_f32, 0.0, 0.5, 0.0];
        let w = alloc
            .soft_capacity_gate(&assignments, &scores, 4)
            .expect("soft_capacity_gate should succeed");
        assert_eq!(w[1], 0.0);
        assert_eq!(w[3], 0.0);
        assert!(w[0] > 0.0 && w[2] > 0.0);
    }

    #[test]
    fn invalid_config_rejected() {
        let cfg = DiffCapacityConfig {
            n_experts: 0,
            base_capacity: 4,
            min_capacity: 1,
            temperature: 0.5,
        };
        assert!(matches!(
            DifferentiableCapacity::new(cfg),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }
}
