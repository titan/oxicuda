//! # SimHash count-based exploration (`#Exploration`).
//!
//! Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel
//! (2017), "#Exploration: A Study of Count-Based Exploration for Deep
//! Reinforcement Learning", NeurIPS 2017, <https://arxiv.org/abs/1611.04717>.
//!
//! Continuous states are discretised with a **static random projection
//! SimHash**: a fixed Gaussian matrix `A ∈ ℝ^{k×S}` maps a state `s` to a
//! `k`-bit code
//!
//! ```text
//! φ(s) = sign(A · s) ∈ {0, 1}^k
//! ```
//!
//! A hash table counts how often each code `φ(s)` has been visited, and the
//! intrinsic exploration bonus follows the MBIE-EB form
//!
//! ```text
//! r⁺(s) = β / √( max(1, n(φ(s))) )         (≥ 0)
//! ```
//!
//! so *novel* states (small/zero visit counts) receive a large bonus while
//! frequently-visited states receive a vanishing one. The projection is seeded,
//! making the hash deterministic across runs.

use std::collections::HashMap;

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

/// Uniform sample in `[0, 1)` (works around the crate `next_f32` `[0, 0.5)` range).
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
}

/// One standard-normal variate via the Box–Muller transform.
#[inline]
fn sample_standard_normal(rng: &mut LcgRng) -> f32 {
    let u1 = unit_uniform(rng).max(1e-7_f32);
    let u2 = unit_uniform(rng);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

// ─── SimHashCount ───────────────────────────────────────────────────────────────

/// Static-hashing count-based exploration bonus (Tang et al. 2017).
#[derive(Debug, Clone)]
pub struct SimHashCount {
    /// State dimensionality `S`.
    state_dim: usize,
    /// Number of hash bits `k` (1..=64).
    n_bits: usize,
    /// Bonus coefficient `β`.
    beta: f32,
    /// Fixed random projection `A` of shape `[k × S]`, row-major.
    projection: Vec<f32>,
    /// Visit counts keyed by the `k`-bit SimHash code.
    counts: HashMap<u64, u64>,
}

impl SimHashCount {
    /// Create a new estimator with a seeded random projection.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if `state_dim == 0`, `n_bits`
    /// is not in `1..=64`, or `beta` is negative / non-finite.
    pub fn new(state_dim: usize, n_bits: usize, beta: f32, rng: &mut LcgRng) -> RlResult<Self> {
        if state_dim == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "state_dim".into(),
                msg: "must be > 0".into(),
            });
        }
        if n_bits == 0 || n_bits > 64 {
            return Err(RlError::InvalidHyperparameter {
                name: "n_bits".into(),
                msg: "must be in 1..=64".into(),
            });
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(RlError::InvalidHyperparameter {
                name: "beta".into(),
                msg: "must be finite and >= 0".into(),
            });
        }
        let projection = (0..n_bits * state_dim)
            .map(|_| sample_standard_normal(rng))
            .collect();
        Ok(Self {
            state_dim,
            n_bits,
            beta,
            projection,
            counts: HashMap::new(),
        })
    }

    /// State dimensionality `S`.
    #[must_use]
    #[inline]
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Number of hash bits `k`.
    #[must_use]
    #[inline]
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Bonus coefficient `β`.
    #[must_use]
    #[inline]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Number of distinct hash buckets observed so far.
    #[must_use]
    #[inline]
    pub fn unique_buckets(&self) -> usize {
        self.counts.len()
    }

    /// Total number of observations across all buckets.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Compute the `k`-bit SimHash code `φ(s) = sign(A · s)`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `state.len() != state_dim`.
    pub fn hash(&self, state: &[f32]) -> RlResult<u64> {
        if state.len() != self.state_dim {
            return Err(RlError::DimensionMismatch {
                expected: self.state_dim,
                got: state.len(),
            });
        }
        let mut code = 0_u64;
        for (bit, row) in self.projection.chunks_exact(self.state_dim).enumerate() {
            if dot(row, state) >= 0.0 {
                code |= 1_u64 << bit;
            }
        }
        Ok(code)
    }

    /// Current visit count of the bucket containing `state` (0 if unseen).
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a state shape error.
    pub fn count(&self, state: &[f32]) -> RlResult<u64> {
        let code = self.hash(state)?;
        Ok(self.counts.get(&code).copied().unwrap_or(0))
    }

    /// Record a visit to `state`, returning the **new** bucket count.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a state shape error.
    pub fn observe(&mut self, state: &[f32]) -> RlResult<u64> {
        let code = self.hash(state)?;
        let entry = self.counts.entry(code).or_insert(0);
        *entry += 1;
        Ok(*entry)
    }

    /// Intrinsic exploration bonus `β / √(max(1, n(φ(s))))` using the *current*
    /// stored count (does not record a visit). Always ≥ 0 and finite.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a state shape error.
    pub fn intrinsic_reward(&self, state: &[f32]) -> RlResult<f32> {
        let n = self.count(state)?.max(1);
        Ok(self.beta / (n as f32).sqrt())
    }

    /// Record a visit to `state` and return the resulting bonus
    /// `β / √(n(φ(s)))` for the post-increment count. Always ≥ 0 and finite.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a state shape error.
    pub fn observe_and_reward(&mut self, state: &[f32]) -> RlResult<f32> {
        let n = self.observe(state)?;
        Ok(self.beta / (n as f32).sqrt())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(seed: u64) -> SimHashCount {
        SimHashCount::new(4, 16, 1.0, &mut LcgRng::new(seed)).expect("valid config")
    }

    #[test]
    fn new_ok_and_accessors() {
        let h = make(1);
        assert_eq!(h.state_dim(), 4);
        assert_eq!(h.n_bits(), 16);
        assert!((h.beta() - 1.0).abs() < 1e-6);
        assert_eq!(h.unique_buckets(), 0);
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn hash_is_stable() {
        let h = make(2);
        let s = [0.5, -1.0, 2.0, 0.25];
        let c1 = h.hash(&s).expect("ok");
        let c2 = h.hash(&s).expect("ok");
        assert_eq!(c1, c2, "same state must hash identically");
    }

    #[test]
    fn deterministic_same_seed() {
        let a = make(7);
        let b = make(7);
        let s = [1.0, 2.0, -3.0, 0.5];
        assert_eq!(a.hash(&s).expect("ok"), b.hash(&s).expect("ok"));
    }

    #[test]
    fn observe_increments_count() {
        let mut h = make(3);
        let s = [1.0, 0.0, -1.0, 0.5];
        assert_eq!(h.observe(&s).expect("ok"), 1);
        assert_eq!(h.observe(&s).expect("ok"), 2);
        assert_eq!(h.count(&s).expect("ok"), 2);
        assert_eq!(h.unique_buckets(), 1);
        assert_eq!(h.total_count(), 2);
    }

    #[test]
    fn reward_decreases_on_repeated_state() {
        let mut h = make(4);
        let s = [0.7, -0.3, 1.2, -0.8];
        h.observe(&s).expect("ok");
        let r1 = h.intrinsic_reward(&s).expect("ok"); // count 1
        for _ in 0..9 {
            h.observe(&s).expect("ok");
        }
        let r2 = h.intrinsic_reward(&s).expect("ok"); // count 10
        assert!(
            r2 < r1,
            "bonus must shrink with repeated visits: {r1} -> {r2}"
        );
        assert!(r2 >= 0.0 && r2.is_finite());
    }

    #[test]
    fn novel_state_higher_than_trained() {
        let mut h = make(5);
        let a = [1.0, -2.0, 0.5, 1.5];
        let b = [-1.0, 2.0, -0.5, -1.5]; // negation ⇒ complementary code ⇒ different bucket
        assert_ne!(
            h.hash(&a).expect("ok"),
            h.hash(&b).expect("ok"),
            "negated state must occupy a different bucket"
        );
        for _ in 0..10 {
            h.observe(&a).expect("ok");
        }
        let r_seen = h.intrinsic_reward(&a).expect("ok"); // count 10
        let r_novel = h.intrinsic_reward(&b).expect("ok"); // count 0 -> max bonus
        assert!(
            r_novel > r_seen,
            "novel state bonus {r_novel} must exceed visited {r_seen}"
        );
    }

    #[test]
    fn reward_nonneg_and_finite() {
        let mut h = make(6);
        let s = [3.0, 3.0, 3.0, 3.0];
        for _ in 0..25 {
            let r = h.observe_and_reward(&s).expect("ok");
            assert!(
                r >= 0.0 && r.is_finite(),
                "reward must be non-negative & finite: {r}"
            );
        }
    }

    #[test]
    fn err_dim_mismatch() {
        let h = make(8);
        assert!(matches!(
            h.hash(&[0.0, 1.0]),
            Err(RlError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_zero_state_dim() {
        let mut rng = LcgRng::new(9);
        assert!(matches!(
            SimHashCount::new(0, 8, 1.0, &mut rng),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_bad_n_bits() {
        let mut rng = LcgRng::new(10);
        assert!(SimHashCount::new(4, 0, 1.0, &mut rng).is_err());
        assert!(SimHashCount::new(4, 65, 1.0, &mut rng).is_err());
    }

    #[test]
    fn err_negative_beta() {
        let mut rng = LcgRng::new(11);
        assert!(matches!(
            SimHashCount::new(4, 8, -1.0, &mut rng),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }
}
