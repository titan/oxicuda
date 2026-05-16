//! N-gram encoding for sequential data using cyclic shift and binding.
//!
//! For tokens [t_0, ..., t_{L-1}] with n-gram order N:
//! Each n-gram HV h_t = Π_{j=0..N-1} ρ^{N-1-j}(hv(t_{t+j}))  (bind with j shifts)
//! Final encoding = Bundle(h_0, ..., h_{L-N}).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::cyclic_shift;
use crate::vector::binary::random_binary;

/// N-gram encoder for sequential token data.
pub struct NgramEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// N-gram order (n ≥ 1).
    n: usize,
    /// HV per vocabulary item (vocab_size entries).
    vocab_hvs: Vec<Vec<i8>>,
}

impl NgramEncoder {
    /// Create a new N-gram encoder.
    ///
    /// - `vocab_size`: size of token vocabulary.
    /// - `n`: n-gram order (must be ≥ 1).
    /// - `dim`: hypervector dimension.
    /// - `rng`: random number generator.
    pub fn new(vocab_size: usize, n: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if n == 0 {
            return Err(HdcError::InvalidNgramOrder(n));
        }
        if vocab_size == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut vocab_hvs = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            vocab_hvs.push(random_binary(dim, rng)?);
        }
        Ok(Self { dim, n, vocab_hvs })
    }

    /// Encode a sequence of token IDs.
    ///
    /// Returns a single binary HV representing the sequence via n-gram bundling.
    pub fn encode(&self, tokens: &[usize], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        if tokens.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if tokens.len() < self.n {
            // Sequence shorter than n-gram order: encode as single item bundle
            let mut hvs: Vec<Vec<i8>> = Vec::with_capacity(tokens.len());
            for &tok in tokens {
                if tok >= self.vocab_hvs.len() {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: tok,
                        max: self.vocab_hvs.len(),
                    });
                }
                hvs.push(self.vocab_hvs[tok].clone());
            }
            return bundle_binary(&hvs, rng);
        }
        let n_grams = tokens.len() - self.n + 1;
        let mut ngram_hvs: Vec<Vec<i8>> = Vec::with_capacity(n_grams);
        for t in 0..n_grams {
            // h_t = bind over j=0..N-1 of ρ^{N-1-j}(hv(tokens[t+j]))
            // Start with an all-+1 identity for binding
            let mut bound: Vec<i8> = vec![1i8; self.dim];
            for j in 0..self.n {
                let tok = tokens[t + j];
                if tok >= self.vocab_hvs.len() {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: tok,
                        max: self.vocab_hvs.len(),
                    });
                }
                let shift_amt = self.n - 1 - j;
                let shifted = if shift_amt == 0 {
                    self.vocab_hvs[tok].clone()
                } else {
                    cyclic_shift(&self.vocab_hvs[tok], shift_amt)?
                };
                bound = binary_bind(&bound, &shifted)?;
            }
            ngram_hvs.push(bound);
        }
        bundle_binary(&ngram_hvs, rng)
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_hvs.len()
    }

    /// Return the n-gram order.
    pub fn n(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::hamming::hamming_frac;
    use crate::handle::LcgRng;

    #[test]
    fn ngram_same_sequence_same_hv() {
        let mut rng = LcgRng::new(100);
        let enc = NgramEncoder::new(10, 2, 256, &mut rng).expect("new");
        let tokens = vec![0, 1, 2, 3, 4];
        // Use separate RNG instances with the same seed for both encode calls
        // so tie-breaking is identical → same output for same input.
        let mut rng_a = LcgRng::new(999);
        let mut rng_b = LcgRng::new(999);
        let hv1 = enc.encode(&tokens, &mut rng_a).expect("encode");
        let hv2 = enc.encode(&tokens, &mut rng_b).expect("encode");
        let dist = hamming_frac(&hv1, &hv2).expect("hamming");
        assert!(
            dist < 0.01,
            "same sequence with same rng seed should produce identical HVs, dist={dist:.3}"
        );
    }

    #[test]
    fn ngram_different_sequences_differ() {
        let mut rng = LcgRng::new(101);
        let enc = NgramEncoder::new(10, 2, 512, &mut rng).expect("new");
        let t1 = vec![0, 1, 2, 3];
        let t2 = vec![5, 6, 7, 8];
        let hv1 = enc.encode(&t1, &mut rng).expect("hv1");
        let hv2 = enc.encode(&t2, &mut rng).expect("hv2");
        let dist = hamming_frac(&hv1, &hv2).expect("hamming");
        // Different sequences should give different HVs (Hamming > 0.3)
        assert!(
            dist > 0.3,
            "different sequences too similar: dist={dist:.3}"
        );
    }
}
