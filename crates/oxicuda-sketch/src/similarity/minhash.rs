//! MinHash signatures for Jaccard-similarity estimation.
//!
//! Use `K` independent hash functions. `Signature[i]` = min over set of h_i(x).
//! Jaccard(A, B) ≈ (1/K) * #{i : `sig_A[i]` == `sig_B[i]`}.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;
use crate::hash::twouniv::TwoUniversal;

/// MinHash signature.
#[derive(Debug, Clone)]
pub struct MinHash {
    pub k: usize,
    pub signature: Vec<u64>,
    pub hashes: Vec<TwoUniversal>,
}

impl MinHash {
    /// Create an empty MinHash with `k` hash functions.
    pub fn new(k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let hashes = TwoUniversal::many(rng, k, u32::MAX as u64);
        Ok(Self {
            k,
            signature: vec![u64::MAX; k],
            hashes,
        })
    }

    /// Insert a single element of the set.
    pub fn add(&mut self, x: u64) {
        for i in 0..self.k {
            let h = self.hashes[i].hash(x);
            if h < self.signature[i] {
                self.signature[i] = h;
            }
        }
    }

    /// Estimate the Jaccard similarity to another MinHash with identical hashes.
    pub fn jaccard(&self, other: &MinHash) -> SketchResult<f64> {
        if self.k != other.k {
            return Err(SketchError::DimensionMismatch {
                a: self.k,
                b: other.k,
            });
        }
        let matches = (0..self.k)
            .filter(|&i| self.signature[i] == other.signature[i])
            .count();
        Ok(matches as f64 / self.k as f64)
    }

    /// Estimate Jaccard similarity treating each item as part of a single multiset.
    /// Build a new MinHash from `set` and compare.
    pub fn from_set(set: &[u64], k: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        let mut mh = Self::new(k, rng)?;
        for &x in set {
            mh.add(x);
        }
        Ok(mh)
    }

    /// Estimate Jaccard via direct computation of true Jaccard between two arbitrary multisets.
    /// Useful as a reference/test helper.
    #[must_use]
    pub fn true_jaccard(a: &[u64], b: &[u64]) -> f64 {
        use std::collections::BTreeSet;
        let sa: BTreeSet<u64> = a.iter().copied().collect();
        let sb: BTreeSet<u64> = b.iter().copied().collect();
        let inter: usize = sa.intersection(&sb).count();
        let uni: usize = sa.union(&sb).count();
        if uni == 0 {
            0.0
        } else {
            inter as f64 / uni as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minhash_constructs() {
        let mut rng = LcgRng::new(11);
        let mh = MinHash::new(64, &mut rng).expect("ok");
        assert_eq!(mh.k, 64);
        assert_eq!(mh.signature.len(), 64);
    }

    #[test]
    fn minhash_invalid_k() {
        let mut rng = LcgRng::new(0);
        assert!(MinHash::new(0, &mut rng).is_err());
    }

    #[test]
    fn minhash_jaccard_identical() {
        let rng = LcgRng::new(11);
        let a: Vec<u64> = (0..1000).collect();
        let mh1 = MinHash::from_set(&a, 64, &mut rng.clone()).expect("ok");
        let mh2 = MinHash::from_set(&a, 64, &mut rng.clone()).expect("ok");
        let j = mh1.jaccard(&mh2).expect("ok");
        assert!((j - 1.0).abs() < 1e-9);
    }

    #[test]
    fn minhash_jaccard_estimate_close() {
        let rng = LcgRng::new(11);
        let a: Vec<u64> = (0..500).collect();
        let b: Vec<u64> = (250..750).collect();
        let true_j = MinHash::true_jaccard(&a, &b);
        let mh_a = MinHash::from_set(&a, 256, &mut rng.clone()).expect("ok");
        let mh_b = MinHash::from_set(&b, 256, &mut rng.clone()).expect("ok");
        let est = mh_a.jaccard(&mh_b).expect("ok");
        assert!((est - true_j).abs() < 0.1, "true {true_j} vs est {est}");
    }

    #[test]
    fn minhash_jaccard_disjoint() {
        let rng = LcgRng::new(7);
        let a: Vec<u64> = (0..500).collect();
        let b: Vec<u64> = (1000..1500).collect();
        let mh_a = MinHash::from_set(&a, 128, &mut rng.clone()).expect("ok");
        let mh_b = MinHash::from_set(&b, 128, &mut rng.clone()).expect("ok");
        let est = mh_a.jaccard(&mh_b).expect("ok");
        assert!(est < 0.1, "disjoint J estimate = {est}");
    }
}
