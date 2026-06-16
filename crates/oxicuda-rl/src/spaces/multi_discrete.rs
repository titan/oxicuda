//! # Discrete action-space abstractions — `Discrete`, `MultiDiscrete`, `Tuple`.
//!
//! A small, dependency-free family of action spaces mirroring the Gymnasium
//! taxonomy:
//!
//! * [`Discrete`] — a single categorical choice in `{0, …, n−1}`.
//! * [`MultiDiscrete`] — a *vector* of independent categorical sub-actions, one
//!   per dimension, with per-dimension cardinalities `nvec`.
//! * [`TupleSpace`] — an ordered tuple of [`MultiDiscrete`] sub-spaces (a single
//!   [`Discrete`] is the special case `MultiDiscrete([n])`).
//!
//! Because the sub-actions of a `MultiDiscrete`/`Tuple` are **independent**, the
//! joint distribution **factorises**: the joint log-probability is the *sum* of
//! the per-dimension log-probabilities and the joint entropy is the *sum* of the
//! per-dimension entropies,
//!
//! ```text
//! log π(a | s) = Σ_i log π_i(a_i | s)
//! H(π)         = Σ_i H(π_i)
//! ```
//!
//! Probabilities are supplied as a single flat `&[f32]` of length
//! `Σ_i nvec[i]` (the per-dimension categorical distributions laid out
//! consecutively), consistent with the rest of the crate.

use crate::error::{RlError, RlResult};
use crate::handle::LcgRng;

/// Uniform sample in `[0, 1)` (works around the crate `next_f32` `[0, 0.5)` range).
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
}

/// Sample an index from a (sub-)categorical via inverse-CDF.
fn sample_categorical(probs: &[f32], rng: &mut LcgRng) -> usize {
    let u = unit_uniform(rng);
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u <= cumulative {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Shannon entropy `−Σ p log p` of a single categorical slice.
fn categorical_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

// ─── Space trait ────────────────────────────────────────────────────────────────

/// A samplable action space.
///
/// The concrete action representation differs per space ([`usize`] for
/// [`Discrete`], `Vec<usize>` for [`MultiDiscrete`], `Vec<Vec<usize>>` for
/// [`TupleSpace`]), so it is carried as an associated type.
pub trait Space {
    /// Concrete action representation produced by this space.
    type Action;

    /// Draw a uniformly-random valid action.
    fn sample(&self, rng: &mut LcgRng) -> Self::Action;

    /// Total number of categorical entries (sum of all sub-action
    /// cardinalities) — the expected length of a flat probability slice.
    fn flat_dim(&self) -> usize;
}

// ─── Discrete ─────────────────────────────────────────────────────────────────

/// A single categorical action space `{0, …, n−1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discrete {
    /// Number of choices `n`.
    n: usize,
}

impl Discrete {
    /// Create a discrete space with `n` choices.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if `n == 0`.
    pub fn new(n: usize) -> RlResult<Self> {
        if n == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "n".into(),
                msg: "must be > 0".into(),
            });
        }
        Ok(Self { n })
    }

    /// Number of choices `n`.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// `true` iff `action < n`.
    #[must_use]
    #[inline]
    pub fn contains(&self, action: usize) -> bool {
        action < self.n
    }

    /// Validate that `action` is in range.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `action >= n`.
    pub fn validate(&self, action: usize) -> RlResult<()> {
        if action >= self.n {
            return Err(RlError::DimensionMismatch {
                expected: self.n,
                got: action,
            });
        }
        Ok(())
    }

    /// Log-probability `log p(action)` for a categorical distribution `probs`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `probs.len() != n` or the action
    /// is out of range.
    pub fn log_prob(&self, probs: &[f32], action: usize) -> RlResult<f32> {
        if probs.len() != self.n {
            return Err(RlError::DimensionMismatch {
                expected: self.n,
                got: probs.len(),
            });
        }
        self.validate(action)?;
        Ok(probs[action].max(1e-10).ln())
    }

    /// Shannon entropy `H = −Σ p log p`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] if `probs.len() != n`.
    pub fn entropy(&self, probs: &[f32]) -> RlResult<f32> {
        if probs.len() != self.n {
            return Err(RlError::DimensionMismatch {
                expected: self.n,
                got: probs.len(),
            });
        }
        Ok(categorical_entropy(probs))
    }
}

impl Space for Discrete {
    type Action = usize;

    fn sample(&self, rng: &mut LcgRng) -> usize {
        rng.next_usize(self.n)
    }

    fn flat_dim(&self) -> usize {
        self.n
    }
}

// ─── MultiDiscrete ──────────────────────────────────────────────────────────────

/// A vector of independent categorical sub-actions with cardinalities `nvec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiDiscrete {
    /// Per-dimension number of choices (`nvec[i]` ≥ 1).
    nvec: Vec<usize>,
}

impl MultiDiscrete {
    /// Create a multi-discrete space from per-dimension cardinalities.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if `nvec` is empty or contains
    /// a zero.
    pub fn new(nvec: Vec<usize>) -> RlResult<Self> {
        if nvec.is_empty() {
            return Err(RlError::InvalidHyperparameter {
                name: "nvec".into(),
                msg: "must be non-empty".into(),
            });
        }
        if nvec.contains(&0) {
            return Err(RlError::InvalidHyperparameter {
                name: "nvec".into(),
                msg: "every entry must be > 0".into(),
            });
        }
        Ok(Self { nvec })
    }

    /// Number of sub-action dimensions.
    #[must_use]
    #[inline]
    pub fn n_dims(&self) -> usize {
        self.nvec.len()
    }

    /// Per-dimension cardinalities.
    #[must_use]
    #[inline]
    pub fn nvec(&self) -> &[usize] {
        &self.nvec
    }

    /// Total number of categorical entries `Σ_i nvec[i]`.
    #[must_use]
    pub fn flat_dim(&self) -> usize {
        self.nvec.iter().sum()
    }

    /// `true` iff `action` has the right length and every entry is in range.
    #[must_use]
    pub fn contains(&self, action: &[usize]) -> bool {
        action.len() == self.nvec.len() && action.iter().zip(&self.nvec).all(|(&a, &n)| a < n)
    }

    /// Validate `action` membership.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] for a length mismatch or an
    /// out-of-range sub-action.
    pub fn validate(&self, action: &[usize]) -> RlResult<()> {
        if action.len() != self.nvec.len() {
            return Err(RlError::DimensionMismatch {
                expected: self.nvec.len(),
                got: action.len(),
            });
        }
        for (&a, &n) in action.iter().zip(&self.nvec) {
            if a >= n {
                return Err(RlError::DimensionMismatch {
                    expected: n,
                    got: a,
                });
            }
        }
        Ok(())
    }

    /// Validate a flat probability slice and the action, returning per-dimension
    /// `(slice, n, a)` triples via a closure-free internal layout check.
    fn check_flat_probs(&self, flat_probs: &[f32]) -> RlResult<()> {
        let expected = self.flat_dim();
        if flat_probs.len() != expected {
            return Err(RlError::DimensionMismatch {
                expected,
                got: flat_probs.len(),
            });
        }
        Ok(())
    }

    /// Per-dimension log-probabilities `log π_i(a_i)`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a shape error or out-of-range
    /// sub-action.
    pub fn per_dim_log_probs(&self, flat_probs: &[f32], action: &[usize]) -> RlResult<Vec<f32>> {
        self.validate(action)?;
        self.check_flat_probs(flat_probs)?;
        let mut out = Vec::with_capacity(self.nvec.len());
        let mut offset = 0;
        for (&n, &a) in self.nvec.iter().zip(action) {
            let probs = &flat_probs[offset..offset + n];
            out.push(probs[a].max(1e-10).ln());
            offset += n;
        }
        Ok(out)
    }

    /// Joint (factorised) log-probability `Σ_i log π_i(a_i)`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a shape error or out-of-range
    /// sub-action.
    pub fn joint_log_prob(&self, flat_probs: &[f32], action: &[usize]) -> RlResult<f32> {
        Ok(self.per_dim_log_probs(flat_probs, action)?.iter().sum())
    }

    /// Per-dimension Shannon entropies.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a flat-probability shape error.
    pub fn per_dim_entropies(&self, flat_probs: &[f32]) -> RlResult<Vec<f32>> {
        self.check_flat_probs(flat_probs)?;
        let mut out = Vec::with_capacity(self.nvec.len());
        let mut offset = 0;
        for &n in &self.nvec {
            out.push(categorical_entropy(&flat_probs[offset..offset + n]));
            offset += n;
        }
        Ok(out)
    }

    /// Joint (factorised) entropy `Σ_i H(π_i)`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a flat-probability shape error.
    pub fn joint_entropy(&self, flat_probs: &[f32]) -> RlResult<f32> {
        Ok(self.per_dim_entropies(flat_probs)?.iter().sum())
    }

    /// Sample one action per dimension from the supplied flat categorical
    /// distributions.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a flat-probability shape error.
    pub fn sample_from_flat_probs(
        &self,
        flat_probs: &[f32],
        rng: &mut LcgRng,
    ) -> RlResult<Vec<usize>> {
        self.check_flat_probs(flat_probs)?;
        let mut out = Vec::with_capacity(self.nvec.len());
        let mut offset = 0;
        for &n in &self.nvec {
            out.push(sample_categorical(&flat_probs[offset..offset + n], rng));
            offset += n;
        }
        Ok(out)
    }
}

impl Space for MultiDiscrete {
    type Action = Vec<usize>;

    fn sample(&self, rng: &mut LcgRng) -> Vec<usize> {
        self.nvec.iter().map(|&n| rng.next_usize(n)).collect()
    }

    fn flat_dim(&self) -> usize {
        MultiDiscrete::flat_dim(self)
    }
}

// ─── TupleSpace ─────────────────────────────────────────────────────────────────

/// An ordered tuple of [`MultiDiscrete`] sub-spaces.
///
/// A tuple action is a `Vec<Vec<usize>>` (one sub-action per sub-space). The
/// joint distribution still factorises across *all* dimensions of *all*
/// sub-spaces — [`TupleSpace::flatten`] collapses the tuple into the equivalent
/// flat [`MultiDiscrete`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupleSpace {
    /// Ordered sub-spaces.
    spaces: Vec<MultiDiscrete>,
}

impl TupleSpace {
    /// Build a tuple space from its ordered sub-spaces.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::InvalidHyperparameter`] if `spaces` is empty.
    pub fn new(spaces: Vec<MultiDiscrete>) -> RlResult<Self> {
        if spaces.is_empty() {
            return Err(RlError::InvalidHyperparameter {
                name: "spaces".into(),
                msg: "tuple must contain at least one sub-space".into(),
            });
        }
        Ok(Self { spaces })
    }

    /// Number of sub-spaces in the tuple.
    #[must_use]
    #[inline]
    pub fn num_spaces(&self) -> usize {
        self.spaces.len()
    }

    /// The ordered sub-spaces.
    #[must_use]
    #[inline]
    pub fn spaces(&self) -> &[MultiDiscrete] {
        &self.spaces
    }

    /// Collapse the tuple into the equivalent flat [`MultiDiscrete`] (the
    /// concatenation of every sub-space's `nvec`).
    #[must_use]
    pub fn flatten(&self) -> MultiDiscrete {
        let nvec = self
            .spaces
            .iter()
            .flat_map(|s| s.nvec.iter().copied())
            .collect();
        // Valid by construction: every sub-space already guarantees `nvec[i] > 0`.
        MultiDiscrete { nvec }
    }

    /// `true` iff `action` matches the tuple structure and every sub-action is
    /// valid.
    #[must_use]
    pub fn contains(&self, action: &[Vec<usize>]) -> bool {
        action.len() == self.spaces.len()
            && action.iter().zip(&self.spaces).all(|(a, s)| s.contains(a))
    }

    /// Validate `action` against the tuple structure.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a structural or sub-action
    /// error.
    pub fn validate(&self, action: &[Vec<usize>]) -> RlResult<()> {
        if action.len() != self.spaces.len() {
            return Err(RlError::DimensionMismatch {
                expected: self.spaces.len(),
                got: action.len(),
            });
        }
        for (a, s) in action.iter().zip(&self.spaces) {
            s.validate(a)?;
        }
        Ok(())
    }

    /// Joint (factorised) log-probability over the whole tuple,
    /// `Σ_subspace Σ_dim log π(a)`.
    ///
    /// `flat_probs` is the concatenation of every sub-space's flat probability
    /// slice.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a shape error or out-of-range
    /// sub-action.
    pub fn joint_log_prob(&self, flat_probs: &[f32], action: &[Vec<usize>]) -> RlResult<f32> {
        self.validate(action)?;
        let flat_action: Vec<usize> = action.iter().flatten().copied().collect();
        self.flatten().joint_log_prob(flat_probs, &flat_action)
    }

    /// Joint (factorised) entropy over the whole tuple, `Σ_subspace Σ_dim H`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::DimensionMismatch`] on a flat-probability shape error.
    pub fn joint_entropy(&self, flat_probs: &[f32]) -> RlResult<f32> {
        self.flatten().joint_entropy(flat_probs)
    }
}

impl Space for TupleSpace {
    type Action = Vec<Vec<usize>>;

    fn sample(&self, rng: &mut LcgRng) -> Vec<Vec<usize>> {
        self.spaces.iter().map(|s| s.sample(rng)).collect()
    }

    fn flat_dim(&self) -> usize {
        self.spaces.iter().map(MultiDiscrete::flat_dim).sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Discrete ─────────────────────────────────────────────────────────────

    #[test]
    fn discrete_sample_in_range() {
        let d = Discrete::new(5).expect("ok");
        let mut rng = LcgRng::new(1);
        for _ in 0..200 {
            assert!(d.sample(&mut rng) < 5);
        }
    }

    #[test]
    fn discrete_log_prob_and_entropy() {
        let d = Discrete::new(4).expect("ok");
        let probs = vec![0.25_f32; 4];
        let lp = d.log_prob(&probs, 0).expect("ok");
        assert!((lp - 0.25_f32.ln()).abs() < 1e-6);
        let h = d.entropy(&probs).expect("ok");
        assert!((h - 4.0_f32.ln()).abs() < 1e-5, "uniform entropy = ln(4)");
    }

    #[test]
    fn discrete_invalid_action_errors() {
        let d = Discrete::new(3).expect("ok");
        assert!(d.validate(3).is_err());
        assert!(!d.contains(3));
        assert!(d.contains(2));
    }

    #[test]
    fn err_discrete_zero() {
        assert!(matches!(
            Discrete::new(0),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    // ── MultiDiscrete ────────────────────────────────────────────────────────

    #[test]
    fn multi_sample_shape_and_bounds() {
        let md = MultiDiscrete::new(vec![3, 2, 4]).expect("ok");
        let mut rng = LcgRng::new(2);
        for _ in 0..200 {
            let a = md.sample(&mut rng);
            assert_eq!(a.len(), 3, "one sub-action per dimension");
            assert!(
                a[0] < 3 && a[1] < 2 && a[2] < 4,
                "sub-actions in range: {a:?}"
            );
        }
    }

    #[test]
    fn multi_flat_dim() {
        let md = MultiDiscrete::new(vec![2, 3, 4]).expect("ok");
        assert_eq!(md.flat_dim(), 9);
        assert_eq!(md.n_dims(), 3);
    }

    #[test]
    fn multi_joint_log_prob_is_sum_of_per_dim() {
        let md = MultiDiscrete::new(vec![2, 3]).expect("ok");
        // dim0: [0.3, 0.7]; dim1: [0.2, 0.5, 0.3]
        let flat = vec![0.3, 0.7, 0.2, 0.5, 0.3];
        let action = vec![1_usize, 2];
        let per = md.per_dim_log_probs(&flat, &action).expect("ok");
        let joint = md.joint_log_prob(&flat, &action).expect("ok");
        let sum: f32 = per.iter().sum();
        assert!((joint - sum).abs() < 1e-6, "joint must equal Σ per-dim");
        let hand = 0.7_f32.ln() + 0.3_f32.ln();
        assert!((joint - hand).abs() < 1e-5, "joint={joint}, hand={hand}");
    }

    #[test]
    fn multi_joint_entropy_is_sum_of_per_dim() {
        let md = MultiDiscrete::new(vec![2, 3]).expect("ok");
        let flat = vec![0.3, 0.7, 0.2, 0.5, 0.3];
        let per = md.per_dim_entropies(&flat).expect("ok");
        let joint = md.joint_entropy(&flat).expect("ok");
        let sum: f32 = per.iter().sum();
        assert!(
            (joint - sum).abs() < 1e-6,
            "joint entropy must equal Σ per-dim"
        );
    }

    #[test]
    fn multi_sample_from_flat_probs_peaked() {
        let md = MultiDiscrete::new(vec![3, 3]).expect("ok");
        // dim0 peaked on idx2, dim1 peaked on idx0
        let flat = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let mut rng = LcgRng::new(3);
        for _ in 0..50 {
            let a = md.sample_from_flat_probs(&flat, &mut rng).expect("ok");
            assert_eq!(a, vec![2, 0], "peaked distributions must pick the peaks");
        }
    }

    #[test]
    fn multi_validate_errors() {
        let md = MultiDiscrete::new(vec![2, 3]).expect("ok");
        assert!(md.validate(&[0, 1]).is_ok());
        assert!(md.validate(&[0]).is_err(), "wrong length");
        assert!(md.validate(&[0, 3]).is_err(), "out of range");
        assert!(!md.contains(&[2, 0]), "first dim out of range");
    }

    #[test]
    fn err_multi_empty() {
        assert!(matches!(
            MultiDiscrete::new(vec![]),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn err_multi_zero_entry() {
        assert!(matches!(
            MultiDiscrete::new(vec![2, 0, 3]),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    // ── TupleSpace ───────────────────────────────────────────────────────────

    #[test]
    fn tuple_sample_and_flatten() {
        let t = TupleSpace::new(vec![
            MultiDiscrete::new(vec![2, 3]).expect("ok"),
            MultiDiscrete::new(vec![4]).expect("ok"),
        ])
        .expect("ok");
        assert_eq!(t.num_spaces(), 2);
        assert_eq!(t.flatten().nvec(), &[2, 3, 4]);
        assert_eq!(t.flat_dim(), 9);

        let mut rng = LcgRng::new(4);
        let a = t.sample(&mut rng);
        assert_eq!(a.len(), 2, "one sub-action vector per sub-space");
        assert_eq!(a[0].len(), 2);
        assert_eq!(a[1].len(), 1);
        assert!(t.contains(&a), "sampled action must be valid: {a:?}");
    }

    #[test]
    fn tuple_joint_factorises_over_subspaces() {
        let s0 = MultiDiscrete::new(vec![2, 3]).expect("ok");
        let s1 = MultiDiscrete::new(vec![4]).expect("ok");
        let t = TupleSpace::new(vec![s0.clone(), s1.clone()]).expect("ok");

        let probs0 = vec![0.3, 0.7, 0.2, 0.5, 0.3]; // for s0
        let probs1 = vec![0.1, 0.2, 0.3, 0.4]; // for s1
        let mut flat = probs0.clone();
        flat.extend_from_slice(&probs1);

        let action = vec![vec![1_usize, 2], vec![3_usize]];

        let joint = t.joint_log_prob(&flat, &action).expect("ok");
        let expected = s0.joint_log_prob(&probs0, &action[0]).expect("ok")
            + s1.joint_log_prob(&probs1, &action[1]).expect("ok");
        assert!(
            (joint - expected).abs() < 1e-6,
            "tuple joint must equal Σ sub-space joints: {joint} vs {expected}"
        );

        let joint_h = t.joint_entropy(&flat).expect("ok");
        let expected_h =
            s0.joint_entropy(&probs0).expect("ok") + s1.joint_entropy(&probs1).expect("ok");
        assert!(
            (joint_h - expected_h).abs() < 1e-6,
            "tuple entropy must factorise"
        );
    }

    #[test]
    fn tuple_validate_errors() {
        let t = TupleSpace::new(vec![MultiDiscrete::new(vec![2, 3]).expect("ok")]).expect("ok");
        assert!(t.validate(&[vec![0, 1]]).is_ok());
        assert!(t.validate(&[]).is_err(), "wrong number of sub-actions");
        assert!(
            t.validate(&[vec![0, 9]]).is_err(),
            "out of range sub-action"
        );
    }

    #[test]
    fn err_tuple_empty() {
        assert!(matches!(
            TupleSpace::new(vec![]),
            Err(RlError::InvalidHyperparameter { .. })
        ));
    }

    // ── Space trait abstraction ──────────────────────────────────────────────

    #[test]
    fn space_trait_object_friendly() {
        fn flat_of<S: Space>(s: &S) -> usize {
            s.flat_dim()
        }
        let d = Discrete::new(6).expect("ok");
        let md = MultiDiscrete::new(vec![2, 2, 2]).expect("ok");
        assert_eq!(flat_of(&d), 6);
        assert_eq!(flat_of(&md), 6);
    }
}
