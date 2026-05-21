//! Sparse Block-Codes (SBC) hypervectors.
//!
//! Reference:
//! - M. Laiho, J. H. Poikonen, P. Kanerva *et al.*,
//!   "High-dimensional computing with sparse vectors," *IEEE Biomedical
//!   Circuits and Systems Conference* (2015).
//! - E. P. Frady, S. J. Kent, B. A. Olshausen, F. T. Sommer,
//!   "A theory of sequence indexing and working memory in recurrent neural
//!   networks," (2020).
//!
//! A Sparse Block-Code (SBC) hypervector partitions the `D` dimensions into
//! `n_blocks` blocks of equal `block_size = D / n_blocks`; each block holds
//! exactly **one** active unit (a one-hot code per block). The compact storage
//! is therefore a length-`n_blocks` vector of active indices, each in
//! `[0, block_size)`.
//!
//! Operators:
//! - **Bind**: block-wise modular addition of active indices.
//!   `result.active[b] = (a.active[b] + b.active[b]) mod block_size`.
//!   Bind is associative, commutative, and has identity `0` (per block).
//! - **Unbind**: block-wise modular subtraction.
//!   `result.active[b] = (a.active[b] − b.active[b]) mod block_size`.
//!   `unbind(bind(x, y), y) == x`.
//! - **Bundle**: per block, sum unit counts across the bundle and pick
//!   `argmax` (resparsify by one-hot). Ties are broken to the lowest index.
//!
//! The Hamming similarity is the count of blocks in which two SBC vectors
//! agree on the active unit (an integer in `[0, n_blocks]`).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for an [`SbcVec`].
///
/// `dim` is the total ambient dimension, `n_blocks` is the number of disjoint
/// blocks; `dim` must be a positive multiple of `n_blocks`. The block size is
/// derived as `dim / n_blocks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SbcConfig {
    /// Total ambient dimension (must satisfy `dim >= 1` and `dim % n_blocks == 0`).
    pub dim: usize,
    /// Number of equally-sized blocks (must be `>= 1` and divide `dim`).
    pub n_blocks: usize,
}

impl SbcConfig {
    /// Validate that `dim >= 1`, `n_blocks >= 1`, and `dim % n_blocks == 0`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0` or `n_blocks == 0`.
    /// - [`HdcError::DimensionMismatch`] if `dim` is not a multiple of `n_blocks`.
    pub fn validate(&self) -> HdcResult<()> {
        if self.dim == 0 || self.n_blocks == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if !self.dim.is_multiple_of(self.n_blocks) {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: self.n_blocks,
            });
        }
        Ok(())
    }

    /// Block size = `dim / n_blocks`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SbcConfig::validate`].
    pub fn block_size(&self) -> HdcResult<usize> {
        self.validate()?;
        Ok(self.dim / self.n_blocks)
    }
}

// ── SBC Vector ────────────────────────────────────────────────────────────────

/// A Sparse Block-Code hypervector.
///
/// Compact storage: `active[b]` is the active unit in block `b`, in
/// `[0, block_size)`. The dense representation has a `1.0` at position
/// `b * block_size + active[b]` for each block `b`, and `0.0` elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SbcVec {
    dim: usize,
    n_blocks: usize,
    block_size: usize,
    active: Vec<usize>,
}

impl SbcVec {
    /// Construct an SBC vector from explicit per-block active indices.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `cfg` is invalid.
    /// - [`HdcError::DimensionMismatch`] if `active.len() != cfg.n_blocks` or
    ///   `cfg.dim % cfg.n_blocks != 0`.
    /// - [`HdcError::FeatureIndexOutOfRange`] if any `active[b] >= block_size`.
    pub fn from_active(cfg: &SbcConfig, active: Vec<usize>) -> HdcResult<Self> {
        let block_size = cfg.block_size()?;
        if active.len() != cfg.n_blocks {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.n_blocks,
                got: active.len(),
            });
        }
        for &a in &active {
            if a >= block_size {
                return Err(HdcError::FeatureIndexOutOfRange {
                    feat: a,
                    max: block_size,
                });
            }
        }
        Ok(Self {
            dim: cfg.dim,
            n_blocks: cfg.n_blocks,
            block_size,
            active,
        })
    }

    /// Generate a uniformly-random SBC vector.
    ///
    /// Each block's active index is drawn uniformly from `[0, block_size)`
    /// using [`LcgRng::next_usize`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SbcConfig::block_size`].
    pub fn random(cfg: &SbcConfig, rng: &mut LcgRng) -> HdcResult<Self> {
        let block_size = cfg.block_size()?;
        let mut active = Vec::with_capacity(cfg.n_blocks);
        for _ in 0..cfg.n_blocks {
            active.push(rng.next_usize(block_size));
        }
        Ok(Self {
            dim: cfg.dim,
            n_blocks: cfg.n_blocks,
            block_size,
            active,
        })
    }

    /// Ambient (dense) dimension.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of disjoint blocks.
    #[must_use]
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Size of each block (`dim / n_blocks`).
    #[must_use]
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Borrow the per-block active indices (length `n_blocks`).
    #[must_use]
    #[inline]
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// Return the dense expansion as a length-`dim` `Vec<f32>` with a `1.0`
    /// in each block's active slot and `0.0` elsewhere.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0f32; self.dim];
        for (b, &a) in self.active.iter().enumerate() {
            dense[b * self.block_size + a] = 1.0;
        }
        dense
    }

    /// Hamming similarity: the number of blocks in which `self` and `other`
    /// agree on the active unit. Result is in `[0, n_blocks]`.
    ///
    /// # Errors
    ///
    /// [`HdcError::DimensionMismatch`] if the two operands have different
    /// `(dim, n_blocks, block_size)` triples.
    pub fn hamming(&self, other: &Self) -> HdcResult<usize> {
        Self::check_compatible(self, other)?;
        let mut matches = 0usize;
        for (&a, &b) in self.active.iter().zip(other.active.iter()) {
            if a == b {
                matches += 1;
            }
        }
        Ok(matches)
    }

    /// SBC bind: block-wise modular addition of active indices.
    ///
    /// `result.active[b] = (self.active[b] + other.active[b]) mod block_size`.
    ///
    /// # Errors
    ///
    /// [`HdcError::DimensionMismatch`] if the two operands have incompatible
    /// shapes.
    pub fn bind(&self, other: &Self) -> HdcResult<Self> {
        Self::check_compatible(self, other)?;
        let bs = self.block_size;
        let mut active = Vec::with_capacity(self.n_blocks);
        for (&a, &b) in self.active.iter().zip(other.active.iter()) {
            active.push((a + b) % bs);
        }
        Ok(Self {
            dim: self.dim,
            n_blocks: self.n_blocks,
            block_size: bs,
            active,
        })
    }

    /// SBC unbind: block-wise modular subtraction of active indices.
    ///
    /// `result.active[b] = (self.active[b] − other.active[b]) mod block_size`.
    ///
    /// # Errors
    ///
    /// [`HdcError::DimensionMismatch`] if the two operands have incompatible
    /// shapes.
    pub fn unbind(&self, other: &Self) -> HdcResult<Self> {
        Self::check_compatible(self, other)?;
        let bs = self.block_size;
        let mut active = Vec::with_capacity(self.n_blocks);
        for (&a, &b) in self.active.iter().zip(other.active.iter()) {
            // (a − b) mod bs with non-negative result.
            let diff = (a + bs - (b % bs)) % bs;
            active.push(diff);
        }
        Ok(Self {
            dim: self.dim,
            n_blocks: self.n_blocks,
            block_size: bs,
            active,
        })
    }

    /// SBC bundle by per-block argmax of unit counts (resparsify the
    /// superposition). For each block `b`, count how many input vectors
    /// have unit `u` active and take the lowest `u` achieving the maximum.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `vecs` is empty.
    /// - [`HdcError::ZeroDimension`] if `cfg` is invalid.
    /// - [`HdcError::DimensionMismatch`] if any vector's shape disagrees
    ///   with `cfg`.
    pub fn bundle(vecs: &[Self], cfg: &SbcConfig) -> HdcResult<Self> {
        if vecs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let block_size = cfg.block_size()?;
        for v in vecs {
            if v.dim != cfg.dim || v.n_blocks != cfg.n_blocks || v.block_size != block_size {
                return Err(HdcError::DimensionMismatch {
                    expected: cfg.dim,
                    got: v.dim,
                });
            }
        }
        let mut active = Vec::with_capacity(cfg.n_blocks);
        let mut counts = vec![0usize; block_size];
        for b in 0..cfg.n_blocks {
            for slot in counts.iter_mut() {
                *slot = 0;
            }
            for v in vecs {
                counts[v.active[b]] += 1;
            }
            // argmax with lowest-index tie break.
            let mut best_idx = 0usize;
            let mut best_count = counts[0];
            for (idx, &c) in counts.iter().enumerate().skip(1) {
                if c > best_count {
                    best_count = c;
                    best_idx = idx;
                }
            }
            active.push(best_idx);
        }
        Ok(Self {
            dim: cfg.dim,
            n_blocks: cfg.n_blocks,
            block_size,
            active,
        })
    }

    /// The all-zero SBC vector (every block's active unit is `0`).
    /// Acts as the identity element for [`SbcVec::bind`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`SbcConfig::block_size`].
    pub fn identity(cfg: &SbcConfig) -> HdcResult<Self> {
        let block_size = cfg.block_size()?;
        Ok(Self {
            dim: cfg.dim,
            n_blocks: cfg.n_blocks,
            block_size,
            active: vec![0usize; cfg.n_blocks],
        })
    }

    /// Verify that two operands share `(dim, n_blocks, block_size)`.
    fn check_compatible(a: &Self, b: &Self) -> HdcResult<()> {
        if a.dim != b.dim || a.n_blocks != b.n_blocks || a.block_size != b.block_size {
            return Err(HdcError::DimensionMismatch {
                expected: a.dim,
                got: b.dim,
            });
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0x000C_0FFE_E5BC_0000)
    }

    fn cfg(dim: usize, n_blocks: usize) -> SbcConfig {
        SbcConfig { dim, n_blocks }
    }

    // ── random / shape ─────────────────────────────────────────────────────

    #[test]
    fn random_yields_n_blocks_active_indices_in_range() {
        let mut r = rng();
        let c = cfg(64, 8);
        let v = SbcVec::random(&c, &mut r).expect("random");
        assert_eq!(v.active().len(), 8);
        assert_eq!(v.n_blocks(), 8);
        assert_eq!(v.block_size(), 8);
        assert_eq!(v.dim(), 64);
        for &a in v.active() {
            assert!(a < v.block_size(), "active {a} out of range");
        }
    }

    #[test]
    fn random_n_blocks_equals_one() {
        let mut r = rng();
        let c = cfg(16, 1);
        let v = SbcVec::random(&c, &mut r).expect("random");
        assert_eq!(v.active().len(), 1);
        assert!(v.active()[0] < 16);
    }

    #[test]
    fn random_n_blocks_equals_dim_block_size_one() {
        // n_blocks == dim → block_size = 1 → every active index must be 0.
        let mut r = rng();
        let c = cfg(32, 32);
        let v = SbcVec::random(&c, &mut r).expect("random");
        assert!(v.active().iter().all(|&a| a == 0));
    }

    // ── to_dense ───────────────────────────────────────────────────────────

    #[test]
    fn dense_length_equals_dim_and_one_hot_per_block() {
        let mut r = rng();
        let c = cfg(24, 4); // block_size = 6
        let v = SbcVec::random(&c, &mut r).expect("random");
        let dense = v.to_dense();
        assert_eq!(dense.len(), 24);
        let ones = dense.iter().filter(|&&x| x == 1.0).count();
        let zeros = dense.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(ones, 4);
        assert_eq!(zeros, 20);
        // Verify exactly one active per block at the expected slot.
        for (b, &a) in v.active().iter().enumerate() {
            assert!((dense[b * 6 + a] - 1.0).abs() < 1e-7);
        }
    }

    #[test]
    fn dense_explicit_known_pattern() {
        // dim=6, n_blocks=2, block_size=3, active=[1, 2] ⇒ dense=[0,1,0, 0,0,1].
        let v = SbcVec::from_active(&cfg(6, 2), vec![1, 2]).expect("from_active");
        let dense = v.to_dense();
        assert_eq!(dense, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // ── hamming ────────────────────────────────────────────────────────────

    #[test]
    fn hamming_self_equals_n_blocks() {
        let mut r = rng();
        let c = cfg(64, 8);
        let v = SbcVec::random(&c, &mut r).expect("random");
        assert_eq!(v.hamming(&v).expect("hamming"), 8);
    }

    #[test]
    fn hamming_symmetric() {
        let mut r = rng();
        let c = cfg(64, 8);
        let a = SbcVec::random(&c, &mut r).expect("a");
        let b = SbcVec::random(&c, &mut r).expect("b");
        assert_eq!(a.hamming(&b).expect("a∼b"), b.hamming(&a).expect("b∼a"));
    }

    #[test]
    fn hamming_disjoint_returns_zero() {
        // Two SBC vectors with completely different active slots in every block.
        let c = cfg(8, 4); // block_size = 2
        let a = SbcVec::from_active(&c, vec![0, 0, 0, 0]).expect("a");
        let b = SbcVec::from_active(&c, vec![1, 1, 1, 1]).expect("b");
        assert_eq!(a.hamming(&b).expect("hamming"), 0);
    }

    // ── bind ───────────────────────────────────────────────────────────────

    #[test]
    fn bind_commutative_and_modular() {
        let c = cfg(12, 3); // block_size = 4
        let a = SbcVec::from_active(&c, vec![1, 2, 3]).expect("a");
        let b = SbcVec::from_active(&c, vec![2, 3, 1]).expect("b");
        let ab = a.bind(&b).expect("bind ab");
        let ba = b.bind(&a).expect("bind ba");
        assert_eq!(ab.active(), ba.active());
        // Expected modular sums under block_size = 4: [3, 1, 0].
        assert_eq!(ab.active(), &[3usize, 1, 0]);
    }

    #[test]
    fn bind_associative() {
        let c = cfg(20, 5); // block_size = 4
        let a = SbcVec::from_active(&c, vec![3, 2, 1, 0, 2]).expect("a");
        let b = SbcVec::from_active(&c, vec![1, 1, 2, 3, 2]).expect("b");
        let d = SbcVec::from_active(&c, vec![2, 0, 3, 1, 1]).expect("d");
        let lhs = a.bind(&b).expect("a*b").bind(&d).expect("(a*b)*d");
        let rhs = a.bind(&b.bind(&d).expect("b*d")).expect("a*(b*d)");
        assert_eq!(lhs.active(), rhs.active());
    }

    #[test]
    fn bind_identity_is_zero_block() {
        let c = cfg(12, 3);
        let a = SbcVec::from_active(&c, vec![1, 3, 2]).expect("a");
        let id = SbcVec::identity(&c).expect("identity");
        let bound = a.bind(&id).expect("bind a id");
        assert_eq!(bound.active(), a.active());
    }

    // ── unbind ─────────────────────────────────────────────────────────────

    #[test]
    fn unbind_inverts_bind() {
        let c = cfg(24, 6); // block_size = 4
        let x = SbcVec::from_active(&c, vec![1, 2, 0, 3, 1, 2]).expect("x");
        let y = SbcVec::from_active(&c, vec![3, 0, 2, 1, 2, 3]).expect("y");
        let bound = x.bind(&y).expect("bind");
        let recovered = bound.unbind(&y).expect("unbind");
        assert_eq!(recovered.active(), x.active());
    }

    #[test]
    fn unbind_random_roundtrip() {
        let mut r = rng();
        let c = cfg(128, 16); // block_size = 8
        let x = SbcVec::random(&c, &mut r).expect("x");
        let y = SbcVec::random(&c, &mut r).expect("y");
        let bound = x.bind(&y).expect("bind");
        let recovered = bound.unbind(&y).expect("unbind");
        assert_eq!(recovered.active(), x.active());
    }

    // ── bundle ─────────────────────────────────────────────────────────────

    #[test]
    fn bundle_of_single_vec_equals_that_vec() {
        let mut r = rng();
        let c = cfg(64, 8);
        let v = SbcVec::random(&c, &mut r).expect("v");
        let bundled = SbcVec::bundle(std::slice::from_ref(&v), &c).expect("bundle");
        assert_eq!(bundled.active(), v.active());
    }

    #[test]
    fn bundle_of_two_copies_equals_that_vec() {
        // For each block, the only candidate has count 2 → argmax → that index.
        let mut r = rng();
        let c = cfg(64, 8);
        let v = SbcVec::random(&c, &mut r).expect("v");
        let bundled = SbcVec::bundle(&[v.clone(), v.clone()], &c).expect("bundle");
        assert_eq!(bundled.active(), v.active());
    }

    #[test]
    fn bundle_argmax_ties_break_to_lowest_index() {
        // n_blocks=1, block_size=4, two inputs activate units 3 and 1 in the
        // single block. Both counts are 1 ⇒ argmax → lowest index = 1.
        let c = cfg(4, 1);
        let a = SbcVec::from_active(&c, vec![3]).expect("a");
        let b = SbcVec::from_active(&c, vec![1]).expect("b");
        let bundled = SbcVec::bundle(&[a, b], &c).expect("bundle");
        assert_eq!(bundled.active(), &[1usize]);
    }

    #[test]
    fn bundle_all_zero_yields_all_zero() {
        // n_blocks copies of the zero vector. Counts at unit 0 are maximal in
        // every block ⇒ argmax stays at 0.
        let c = cfg(12, 4); // block_size = 3
        let zero = SbcVec::from_active(&c, vec![0, 0, 0, 0]).expect("zero");
        let bundled =
            SbcVec::bundle(&[zero.clone(), zero.clone(), zero.clone(), zero], &c).expect("bundle");
        assert!(bundled.active().iter().all(|&a| a == 0));
    }

    #[test]
    fn bundle_majority_wins_when_three_of_four_agree() {
        // n_blocks=1, block_size=4. Three vectors with unit 2, one with unit 1.
        // Counts: [0,1,3,0] ⇒ argmax = 2.
        let c = cfg(4, 1);
        let dominant = SbcVec::from_active(&c, vec![2]).expect("d");
        let dissident = SbcVec::from_active(&c, vec![1]).expect("dis");
        let bundled = SbcVec::bundle(
            &[
                dominant.clone(),
                dominant.clone(),
                dominant.clone(),
                dissident,
            ],
            &c,
        )
        .expect("bundle");
        assert_eq!(bundled.active(), &[2usize]);
    }

    // ── n_blocks = 1 (degenerate config) ──────────────────────────────────

    #[test]
    fn n_blocks_one_works_for_bind_unbind() {
        // n_blocks=1, block_size=8. Bind = mod-8 addition.
        let c = cfg(8, 1);
        let a = SbcVec::from_active(&c, vec![5]).expect("a");
        let b = SbcVec::from_active(&c, vec![6]).expect("b");
        let bound = a.bind(&b).expect("bind"); // (5+6) mod 8 = 3
        assert_eq!(bound.active(), &[3usize]);
        let recovered = bound.unbind(&b).expect("unbind");
        assert_eq!(recovered.active(), a.active());
    }

    // ── errors ─────────────────────────────────────────────────────────────

    #[test]
    fn err_dim_not_multiple_of_n_blocks() {
        // dim=7 is not divisible by n_blocks=2.
        let c = SbcConfig {
            dim: 7,
            n_blocks: 2,
        };
        let res = c.validate();
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_zero_dim_or_zero_n_blocks() {
        assert!(matches!(
            SbcConfig {
                dim: 0,
                n_blocks: 1
            }
            .validate(),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            SbcConfig {
                dim: 4,
                n_blocks: 0
            }
            .validate(),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn err_bind_on_mismatched_configs() {
        let a = SbcVec::from_active(&cfg(8, 2), vec![0, 1]).expect("a");
        let b = SbcVec::from_active(&cfg(12, 3), vec![0, 1, 2]).expect("b");
        let res = a.bind(&b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_unbind_on_mismatched_configs() {
        let a = SbcVec::from_active(&cfg(8, 2), vec![0, 1]).expect("a");
        let b = SbcVec::from_active(&cfg(8, 4), vec![0, 1, 0, 1]).expect("b");
        let res = a.unbind(&b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_bundle_empty() {
        let c = cfg(8, 2);
        let res = SbcVec::bundle(&[], &c);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn err_bundle_mismatched_member() {
        let c = cfg(8, 2);
        let a = SbcVec::from_active(&c, vec![0, 1]).expect("a");
        let b = SbcVec::from_active(&cfg(12, 3), vec![0, 1, 2]).expect("b");
        let res = SbcVec::bundle(&[a, b], &c);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_from_active_wrong_length() {
        let c = cfg(8, 2);
        let res = SbcVec::from_active(&c, vec![0, 1, 2]); // length 3 != n_blocks 2
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_from_active_out_of_range() {
        let c = cfg(8, 2); // block_size = 4
        let res = SbcVec::from_active(&c, vec![0, 4]); // 4 not in [0,4)
        assert!(matches!(res, Err(HdcError::FeatureIndexOutOfRange { .. })));
    }

    #[test]
    fn err_hamming_on_mismatched_configs() {
        let a = SbcVec::from_active(&cfg(8, 2), vec![0, 1]).expect("a");
        let b = SbcVec::from_active(&cfg(8, 4), vec![0, 1, 0, 1]).expect("b");
        let res = a.hamming(&b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── determinism ────────────────────────────────────────────────────────

    #[test]
    fn deterministic_given_seed() {
        let c = cfg(64, 8);
        let mut r1 = LcgRng::new(0x0DEA_D5BC);
        let mut r2 = LcgRng::new(0x0DEA_D5BC);
        let v1 = SbcVec::random(&c, &mut r1).expect("v1");
        let v2 = SbcVec::random(&c, &mut r2).expect("v2");
        assert_eq!(v1.active(), v2.active());
    }
}
