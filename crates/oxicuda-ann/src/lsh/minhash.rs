use crate::handle::LcgRng;

/// MinHash LSH for estimating Jaccard similarity of sets.
pub struct MinHash {
    pub n_hashes: usize,
    a: Vec<u64>,
    b: Vec<u64>,
    /// Mersenne prime 2^31 - 1.
    prime: u64,
}

impl MinHash {
    /// Create a new MinHash with `n_hashes` independent hash functions.
    #[must_use]
    pub fn new(n_hashes: usize, rng: &mut LcgRng) -> Self {
        let prime: u64 = (1u64 << 31) - 1;
        let mut a = Vec::with_capacity(n_hashes);
        let mut b = Vec::with_capacity(n_hashes);
        for _ in 0..n_hashes {
            // a must be in [1, prime-1], b in [0, prime-1]
            let av = (rng.next_u64() % (prime - 1)) + 1;
            let bv = rng.next_u64() % prime;
            a.push(av);
            b.push(bv);
        }
        Self {
            n_hashes,
            a,
            b,
            prime,
        }
    }

    /// Compute the MinHash signature for `set` (slice of u32 element IDs).
    #[must_use]
    pub fn hash(&self, set: &[u32]) -> Vec<u32> {
        if set.is_empty() {
            return vec![u32::MAX; self.n_hashes];
        }
        let mut sig = vec![u64::MAX; self.n_hashes];
        for &elem in set {
            let x = elem as u64;
            for (h, sh) in sig.iter_mut().enumerate() {
                let hv = (self.a[h].wrapping_mul(x).wrapping_add(self.b[h])) % self.prime;
                if hv < *sh {
                    *sh = hv;
                }
            }
        }
        sig.iter().map(|&v| v as u32).collect()
    }

    /// Estimate Jaccard similarity from two signatures.
    #[must_use]
    pub fn jaccard_estimate(sig1: &[u32], sig2: &[u32]) -> f32 {
        if sig1.is_empty() {
            return 0.0;
        }
        let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
        matches as f32 / sig1.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_sets() {
        let mut rng = LcgRng::new(7);
        let mh = MinHash::new(256, &mut rng);
        let s1 = vec![1u32, 2, 3, 4, 5];
        let sig1 = mh.hash(&s1);
        let sig2 = mh.hash(&s1);
        let j = MinHash::jaccard_estimate(&sig1, &sig2);
        assert!((j - 1.0).abs() < 1e-6, "j={j}");
    }
}
