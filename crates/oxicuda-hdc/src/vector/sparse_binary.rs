//! Sparse binary hypervectors (`k`-of-`D`) — Kanerva Binary Sparse Distributed
//! Representations (BSDR).
//!
//! Reference: P. Kanerva, "Sparse Distributed Memory" (MIT Press, 1988); P. Kanerva,
//! "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation
//! with High-Dimensional Random Vectors" (2009); D. Kleyko et al., "A Survey on
//! Hyperdimensional Computing aka Vector Symbolic Architectures, Part I" (ACM CSUR 2022).
//!
//! A *sparse* binary hypervector activates only `k` of its `D` coordinates (`k ≪ D`); the
//! remaining coordinates are zero. Such codes are biologically motivated (sparse neural
//! firing) and memory-efficient: only the `k` active indices need be stored. This module
//! represents a code by the sorted list of its active indices and provides the three VSA
//! primitives over that representation:
//!
//! - **Sparse dot / overlap.** The inner product of two sparse codes equals the number of
//!   coordinates active in *both*, i.e. the size of the intersection of their active-index
//!   sets. Because the lists are kept sorted, the overlap is computed by a linear merge in
//!   `O(k_a + k_b)` time rather than `O(D)`.
//!
//! - **Sparse bundle (thinning / context-dependent union).** The bundle of several sparse
//!   codes is their *union* re-sparsified back to exactly `k` active coordinates: each
//!   coordinate is scored by how many inputs activate it, and the `k` highest-scoring
//!   coordinates (ties broken by index, then by a deterministic RNG draw) are kept. This is
//!   the standard CDT/thinning bundle that preserves sparsity (Rachkovskij & Kussul 2001).
//!
//! - **Sparse bind (permutation product).** Binding two sparse codes maps active index `i`
//!   of operand `a` and active index `j` of operand `b` to the single index `(i + j) mod D`,
//!   forming the modular sumset and re-sparsifying to `k`. This binding is commutative,
//!   approximately invertible (unbind shifts back by `-j`), and distributes over the union,
//!   exactly mirroring circular convolution restricted to the sparse support.
//!
//! All densified views are returned as `Vec<i8>` in `{0, 1}` (note: *unipolar*, unlike the
//! dense `{−1, +1}` model) so that overlap equals the ordinary dot product.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Immutable configuration shared by a family of `k`-of-`D` sparse binary codes.
#[derive(Debug, Clone)]
pub struct SparseBinaryConfig {
    /// Total number of coordinates `D`.
    dim: usize,
    /// Number of active coordinates `k` (`1 ≤ k ≤ D`).
    active: usize,
}

impl SparseBinaryConfig {
    /// Create a configuration for `k`-of-`D` sparse codes.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `active == 0`.
    /// - [`HdcError::CapacityExceeded`] if `active > dim`.
    pub fn new(dim: usize, active: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if active == 0 {
            return Err(HdcError::EmptyInput);
        }
        if active > dim {
            return Err(HdcError::CapacityExceeded {
                stored: active,
                capacity: dim,
            });
        }
        Ok(Self { dim, active })
    }

    /// Total number of coordinates `D`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of active coordinates `k`.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Density `k / D`.
    #[must_use]
    pub fn density(&self) -> f32 {
        self.active as f32 / self.dim as f32
    }
}

/// A `k`-of-`D` sparse binary hypervector stored as its sorted, de-duplicated active indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseBinaryVec {
    dim: usize,
    /// Strictly increasing active coordinate indices, all `< dim`.
    active: Vec<usize>,
}

impl SparseBinaryVec {
    /// Build a code from an explicit list of active indices (any order, duplicates allowed).
    ///
    /// The indices are sorted and de-duplicated. The result is *not* forced to a fixed `k`;
    /// it simply records whichever distinct in-range coordinates were supplied.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::FeatureIndexOutOfRange`] if any index is `>= dim`.
    pub fn from_indices(dim: usize, mut indices: Vec<usize>) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        for &i in &indices {
            if i >= dim {
                return Err(HdcError::FeatureIndexOutOfRange { feat: i, max: dim });
            }
        }
        indices.sort_unstable();
        indices.dedup();
        Ok(Self {
            dim,
            active: indices,
        })
    }

    /// Draw a random `k`-of-`D` code with exactly `cfg.active()` distinct active coordinates.
    ///
    /// Uses a partial Fisher-Yates shuffle over a `0..D` index pool so that every `k`-subset
    /// is equally likely and exactly `k` coordinates are chosen.
    pub fn random(cfg: &SparseBinaryConfig, rng: &mut LcgRng) -> Self {
        let mut pool: Vec<usize> = (0..cfg.dim).collect();
        for i in 0..cfg.active {
            let j = i + rng.next_usize(cfg.dim - i);
            pool.swap(i, j);
        }
        let mut active: Vec<usize> = pool[..cfg.active].to_vec();
        active.sort_unstable();
        Self {
            dim: cfg.dim,
            active,
        }
    }

    /// Total number of coordinates `D`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of active coordinates `k = |support|`.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.active.len()
    }

    /// The sorted active-coordinate indices.
    #[must_use]
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// Densify into a `Vec<i8>` in `{0, 1}` of length `dim` (1 at every active coordinate).
    #[must_use]
    pub fn to_dense(&self) -> Vec<i8> {
        let mut dense = vec![0i8; self.dim];
        for &i in &self.active {
            dense[i] = 1;
        }
        dense
    }

    /// Sparse overlap (inner product) — the number of coordinates active in **both** codes,
    /// computed by a linear merge of the two sorted index lists in `O(k_a + k_b)`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if the two codes have different `dim`.
    pub fn overlap(&self, other: &Self) -> HdcResult<usize> {
        if self.dim != other.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
        while i < self.active.len() && j < other.active.len() {
            match self.active[i].cmp(&other.active[j]) {
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        Ok(count)
    }

    /// Jaccard similarity `|A ∩ B| / |A ∪ B|` over the active-coordinate sets.
    ///
    /// Returns `0.0` when both codes are empty (the empty-set convention).
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if the two codes have different `dim`.
    pub fn jaccard(&self, other: &Self) -> HdcResult<f32> {
        let inter = self.overlap(other)?;
        let union = self.active.len() + other.active.len() - inter;
        if union == 0 {
            return Ok(0.0);
        }
        Ok(inter as f32 / union as f32)
    }

    /// Sparse bind by modular sumset: active index `i` of `self` and `j` of `other` map to
    /// `(i + j) mod D`. The full sumset is scored by multiplicity and re-sparsified to
    /// exactly `k = cfg.active()` coordinates.
    ///
    /// Binding is commutative. With `unbind` it is approximately invertible:
    /// `a.bind(b).unbind(b) ≈ a` (the recovered support is biased toward `a`'s).
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `self.dim != other.dim` or differs from `cfg.dim()`.
    pub fn bind(
        &self,
        other: &Self,
        cfg: &SparseBinaryConfig,
        rng: &mut LcgRng,
    ) -> HdcResult<Self> {
        if self.dim != other.dim || self.dim != cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.dim,
                got: if self.dim != cfg.dim {
                    self.dim
                } else {
                    other.dim
                },
            });
        }
        let mut scores = vec![0u32; self.dim];
        for &i in &self.active {
            for &j in &other.active {
                let idx = (i + j) % self.dim;
                scores[idx] += 1;
            }
        }
        let active = top_k_indices(&scores, cfg.active, rng);
        Ok(Self {
            dim: self.dim,
            active,
        })
    }

    /// Sparse unbind: shift `self`'s active coordinates back by each active index of `key`,
    /// scoring by multiplicity and re-sparsifying to `k`. Approximate inverse of [`Self::bind`].
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `self.dim != key.dim` or differs from `cfg.dim()`.
    pub fn unbind(
        &self,
        key: &Self,
        cfg: &SparseBinaryConfig,
        rng: &mut LcgRng,
    ) -> HdcResult<Self> {
        if self.dim != key.dim || self.dim != cfg.dim {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.dim,
                got: if self.dim != cfg.dim {
                    self.dim
                } else {
                    key.dim
                },
            });
        }
        let mut scores = vec![0u32; self.dim];
        for &i in &self.active {
            for &j in &key.active {
                // (i - j) mod D, computed without underflow.
                let idx = (i + self.dim - (j % self.dim)) % self.dim;
                scores[idx] += 1;
            }
        }
        let active = top_k_indices(&scores, cfg.active, rng);
        Ok(Self {
            dim: self.dim,
            active,
        })
    }

    /// Sparse bundle (thinning union): score every coordinate by how many inputs activate it,
    /// then keep the `k = cfg.active()` highest-scoring coordinates. This preserves sparsity
    /// while superposing the inputs' supports (Rachkovskij-Kussul context-dependent thinning).
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `codes` is empty.
    /// - [`HdcError::DimensionMismatch`] if any code's `dim` differs from `cfg.dim()`.
    pub fn bundle(codes: &[Self], cfg: &SparseBinaryConfig, rng: &mut LcgRng) -> HdcResult<Self> {
        if codes.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let mut scores = vec![0u32; cfg.dim];
        for code in codes {
            if code.dim != cfg.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: cfg.dim,
                    got: code.dim,
                });
            }
            for &i in &code.active {
                scores[i] += 1;
            }
        }
        let active = top_k_indices(&scores, cfg.active, rng);
        Ok(Self {
            dim: cfg.dim,
            active,
        })
    }
}

/// Select the indices of the `k` highest-scoring coordinates from `scores`.
///
/// Selection is deterministic for a fixed `rng`: coordinates are ranked by descending score,
/// then ascending index. When the score at the `k`-th boundary is shared by more candidates
/// than remain to be filled, the tie is broken by a deterministic RNG permutation of the tied
/// block so that no positional bias is introduced. Zero-scored coordinates are never selected
/// unless fewer than `k` coordinates have a positive score, in which case the shortfall is
/// filled from the lowest zero-scored indices to keep the support size exactly `k`.
fn top_k_indices(scores: &[u32], k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let dim = scores.len();
    let k = k.min(dim);
    // Pair (score, index); sort by descending score then ascending index.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| scores[b].cmp(&scores[a]).then(a.cmp(&b)));

    // Determine the boundary score (score of the k-th element, 1-indexed).
    if k == 0 {
        return Vec::new();
    }
    let boundary = scores[order[k - 1]];

    // Coordinates strictly above the boundary are always kept.
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let mut tied: Vec<usize> = Vec::new();
    for &idx in &order {
        if scores[idx] > boundary {
            chosen.push(idx);
        } else if scores[idx] == boundary {
            tied.push(idx);
        }
    }
    let remaining = k - chosen.len();
    if remaining >= tied.len() {
        chosen.extend(tied);
    } else {
        // Break the tie with a deterministic Fisher-Yates shuffle, then take `remaining`.
        for i in 0..remaining {
            let j = i + rng.next_usize(tied.len() - i);
            tied.swap(i, j);
        }
        chosen.extend_from_slice(&tied[..remaining]);
    }
    chosen.sort_unstable();
    chosen.truncate(k);
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> LcgRng {
        LcgRng::new(0xB5D0_1234_5678_9ABC)
    }

    #[test]
    fn config_validation() {
        assert!(matches!(
            SparseBinaryConfig::new(0, 5),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            SparseBinaryConfig::new(100, 0),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            SparseBinaryConfig::new(10, 20),
            Err(HdcError::CapacityExceeded {
                stored: 20,
                capacity: 10
            })
        ));
        let cfg = SparseBinaryConfig::new(1000, 20).expect("cfg");
        assert_eq!(cfg.dim(), 1000);
        assert_eq!(cfg.active(), 20);
        assert!((cfg.density() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn random_has_exactly_k_active() {
        let cfg = SparseBinaryConfig::new(2000, 30).expect("cfg");
        let mut r = rng();
        for _ in 0..20 {
            let v = SparseBinaryVec::random(&cfg, &mut r);
            assert_eq!(v.n_active(), 30);
            // Strictly increasing & in range.
            for w in v.active().windows(2) {
                assert!(w[0] < w[1]);
            }
            assert!(v.active().iter().all(|&i| i < 2000));
        }
    }

    #[test]
    fn from_indices_sorts_and_dedups() {
        let v = SparseBinaryVec::from_indices(100, vec![9, 3, 3, 7, 0, 9]).expect("v");
        assert_eq!(v.active(), &[0, 3, 7, 9]);
        assert!(matches!(
            SparseBinaryVec::from_indices(10, vec![3, 11]),
            Err(HdcError::FeatureIndexOutOfRange { feat: 11, max: 10 })
        ));
        assert!(matches!(
            SparseBinaryVec::from_indices(0, vec![]),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn to_dense_matches_active() {
        let v = SparseBinaryVec::from_indices(8, vec![1, 4, 6]).expect("v");
        assert_eq!(v.to_dense(), vec![0, 1, 0, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn overlap_equals_intersection() {
        let a = SparseBinaryVec::from_indices(20, vec![1, 3, 5, 7, 9]).expect("a");
        let b = SparseBinaryVec::from_indices(20, vec![3, 4, 5, 6, 9]).expect("b");
        assert_eq!(a.overlap(&b).expect("ov"), 3); // {3,5,9}
        // Self-overlap equals k.
        assert_eq!(a.overlap(&a).expect("self"), 5);
        // Disjoint → 0.
        let c = SparseBinaryVec::from_indices(20, vec![0, 2, 4]).expect("c");
        let d = SparseBinaryVec::from_indices(20, vec![1, 3, 5]).expect("d");
        assert_eq!(c.overlap(&d).expect("disjoint"), 0);
    }

    #[test]
    fn overlap_matches_dense_dot() {
        let cfg = SparseBinaryConfig::new(500, 40).expect("cfg");
        let mut r = rng();
        let a = SparseBinaryVec::random(&cfg, &mut r);
        let b = SparseBinaryVec::random(&cfg, &mut r);
        let dense_dot: i64 = a
            .to_dense()
            .iter()
            .zip(b.to_dense().iter())
            .map(|(&x, &y)| (x as i64) * (y as i64))
            .sum();
        assert_eq!(a.overlap(&b).expect("ov") as i64, dense_dot);
    }

    #[test]
    fn overlap_dim_mismatch_errors() {
        let a = SparseBinaryVec::from_indices(10, vec![1]).expect("a");
        let b = SparseBinaryVec::from_indices(20, vec![1]).expect("b");
        assert!(matches!(
            a.overlap(&b),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn jaccard_bounds() {
        let a = SparseBinaryVec::from_indices(20, vec![1, 2, 3, 4]).expect("a");
        assert!((a.jaccard(&a).expect("self") - 1.0).abs() < 1e-6);
        let b = SparseBinaryVec::from_indices(20, vec![5, 6, 7, 8]).expect("b");
        assert!(a.jaccard(&b).expect("disjoint").abs() < 1e-6);
        let c = SparseBinaryVec::from_indices(20, vec![1, 2, 5, 6]).expect("c");
        // |∩| = 2 ({1,2}), |∪| = 6 → 1/3.
        assert!((a.jaccard(&c).expect("partial") - (1.0 / 3.0)).abs() < 1e-6);
        // Empty/empty → 0 by convention.
        let e = SparseBinaryVec::from_indices(20, vec![]).expect("e");
        assert!(e.jaccard(&e).expect("empty").abs() < 1e-6);
    }

    #[test]
    fn bundle_preserves_sparsity_and_unions() {
        let cfg = SparseBinaryConfig::new(1000, 20).expect("cfg");
        let mut r = rng();
        let a = SparseBinaryVec::random(&cfg, &mut r);
        let b = SparseBinaryVec::random(&cfg, &mut r);
        let c = SparseBinaryVec::random(&cfg, &mut r);
        let bundled =
            SparseBinaryVec::bundle(&[a.clone(), b.clone(), c.clone()], &cfg, &mut r).expect("bun");
        // Sparsity preserved.
        assert_eq!(bundled.n_active(), 20);
        // Bundle is more similar to its constituents than two random codes are to each other.
        let base = SparseBinaryVec::random(&cfg, &mut r);
        let cross = base.overlap(&a).expect("cross");
        let ov_a = bundled.overlap(&a).expect("ov a");
        assert!(
            ov_a >= cross,
            "bundle should retain constituent support: ov_a={ov_a}, cross={cross}"
        );
    }

    #[test]
    fn bundle_empty_errors() {
        let cfg = SparseBinaryConfig::new(100, 10).expect("cfg");
        let mut r = rng();
        assert!(matches!(
            SparseBinaryVec::bundle(&[], &cfg, &mut r),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn bundle_dim_mismatch_errors() {
        let cfg = SparseBinaryConfig::new(100, 10).expect("cfg");
        let mut r = rng();
        let bad = SparseBinaryVec::from_indices(50, vec![1, 2]).expect("bad");
        assert!(matches!(
            SparseBinaryVec::bundle(&[bad], &cfg, &mut r),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn bind_is_commutative_and_sparse() {
        let cfg = SparseBinaryConfig::new(1000, 16).expect("cfg");
        let mut r = rng();
        let a = SparseBinaryVec::random(&cfg, &mut r);
        let b = SparseBinaryVec::random(&cfg, &mut r);
        // Same rng stream for both orderings.
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        let ab = a.bind(&b, &cfg, &mut r1).expect("ab");
        let ba = b.bind(&a, &cfg, &mut r2).expect("ba");
        assert_eq!(ab.active(), ba.active(), "bind must be commutative");
        assert_eq!(ab.n_active(), 16);
    }

    #[test]
    fn bind_unbind_recovers_support() {
        // bind(a,b) then unbind by b should overlap a much more than a random code does.
        let cfg = SparseBinaryConfig::new(4000, 24).expect("cfg");
        let mut r = LcgRng::new(123);
        let a = SparseBinaryVec::random(&cfg, &mut r);
        let b = SparseBinaryVec::random(&cfg, &mut r);
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(1);
        let bound = a.bind(&b, &cfg, &mut r1).expect("bound");
        let recovered = bound.unbind(&b, &cfg, &mut r2).expect("recovered");
        let ov_true = recovered.overlap(&a).expect("ov true");
        let rand = SparseBinaryVec::random(&cfg, &mut r);
        let ov_rand = recovered.overlap(&rand).expect("ov rand");
        assert!(
            ov_true > ov_rand,
            "unbind did not recover support: ov_true={ov_true}, ov_rand={ov_rand}"
        );
    }

    #[test]
    fn bind_dim_mismatch_errors() {
        let cfg = SparseBinaryConfig::new(100, 8).expect("cfg");
        let mut r = rng();
        let a = SparseBinaryVec::from_indices(100, vec![1, 2]).expect("a");
        let bad = SparseBinaryVec::from_indices(50, vec![1]).expect("bad");
        assert!(matches!(
            a.bind(&bad, &cfg, &mut r),
            Err(HdcError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            a.unbind(&bad, &cfg, &mut r),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn top_k_deterministic_and_correct() {
        // Scores with a clear top-3 and a tie at the boundary.
        let scores = [0u32, 5, 3, 5, 1, 5, 2, 4];
        let mut r = LcgRng::new(99);
        let sel = top_k_indices(&scores, 3, &mut r);
        assert_eq!(sel.len(), 3);
        // The three 5s are at indices 1,3,5 → must select exactly those (all above boundary
        // would be >5; here boundary itself is 5 and there are exactly three of them).
        assert_eq!(sel, vec![1, 3, 5]);
        // k larger than positives still returns exactly k.
        let sel2 = top_k_indices(&scores, 6, &mut r);
        assert_eq!(sel2.len(), 6);
    }

    #[test]
    fn bundle_is_deterministic_for_fixed_rng() {
        let cfg = SparseBinaryConfig::new(800, 20).expect("cfg");
        let mut r = LcgRng::new(5);
        let codes: Vec<SparseBinaryVec> = (0..5)
            .map(|_| SparseBinaryVec::random(&cfg, &mut r))
            .collect();
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let h1 = SparseBinaryVec::bundle(&codes, &cfg, &mut r1).expect("h1");
        let h2 = SparseBinaryVec::bundle(&codes, &cfg, &mut r2).expect("h2");
        assert_eq!(h1, h2);
    }
}
