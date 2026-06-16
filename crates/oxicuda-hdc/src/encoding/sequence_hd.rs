//! Permutation-based sequence encoding for hyperdimensional computing (Kanerva 2009).
//!
//! A variable-length sequence of symbol hypervectors `[s₀, s₁, …, s_{L-1}]` is encoded so
//! that the *order* of the symbols — not merely the multiset of symbols — is reflected in the
//! resulting hypervector. This is the canonical VSA construction for sequences and time
//! series (Kanerva 2009; Rahimi 2016): position is represented by repeated application of a
//! fixed permutation `ρ` (here the unit circular shift), and the bundled superposition of the
//! position-bound symbols yields a single fixed-width hypervector.
//!
//! Two encodings are provided, both producing `Vec<i8>` in `{−1, +1}` to match the crate-wide
//! binary hypervector representation:
//!
//! - **Position-bundle encoding** ([`SequenceHdEncoder::encode`]). The sequence is mapped to
//!   `⨁_{i} ρ^{i}(sᵢ)` where `ρ^{i}` is the circular shift by `i` and `⨁` is the binary
//!   majority bundle. Because `ρ^{i}(s)` is (nearly) orthogonal to `s` for `i ≠ 0`, permuting
//!   the order of the same symbols changes which shifted copies are superposed, so two
//!   sequences over the same multiset but in a different order encode to (nearly) orthogonal
//!   hypervectors.
//!
//! - **N-gram encoding** ([`SequenceHdEncoder::encode_ngrams`]). For window size `k` each
//!   length-`k` window `[s_w, …, s_{w+k-1}]` is bound into a single hypervector
//!   `ρ^{0}(s_w) ⊗ ρ^{1}(s_{w+1}) ⊗ … ⊗ ρ^{k-1}(s_{w+k-1})` (element-wise `±1` product), and
//!   all `L − k + 1` window hypervectors are bundled. This captures local ordered context and
//!   is the representation used by HD language/text classifiers.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::cyclic_shift;

/// Permutation-based encoder for sequences of binary symbol hypervectors.
///
/// The encoder is stateless apart from the fixed hypervector dimension it validates against;
/// the symbol hypervectors are supplied by the caller (typically drawn from an
/// [`crate::memory::item_memory`] item memory or generated with
/// [`crate::vector::binary::random_binary`]).
pub struct SequenceHdEncoder {
    /// Hypervector dimension every symbol must match.
    dim: usize,
}

impl SequenceHdEncoder {
    /// Create a new sequence encoder for hypervectors of dimension `dim`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn new(dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self { dim })
    }

    /// Hypervector dimension expected for every symbol.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Validate that every symbol matches the configured dimension.
    fn check_symbols(&self, symbols: &[Vec<i8>]) -> HdcResult<()> {
        if symbols.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        for s in symbols {
            if s.len() != self.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.dim,
                    got: s.len(),
                });
            }
        }
        Ok(())
    }

    /// Encode a sequence as the bundle of position-shifted symbol hypervectors.
    ///
    /// The result is `⨁_{i=0}^{L-1} ρ^{i}(sᵢ)`, i.e. each symbol is circularly shifted by its
    /// position index and the shifted copies are combined by binary majority vote. Reversing or
    /// otherwise reordering the symbols yields a (nearly) orthogonal hypervector, so order is
    /// preserved by the encoding.
    ///
    /// `rng` is only consulted to break ties in the majority bundle (which occur solely when an
    /// even number of symbols cancel exactly at a component); for odd-length sequences the
    /// output is independent of `rng`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `symbols` is empty.
    /// - [`HdcError::DimensionMismatch`] if any symbol's length differs from [`Self::dim`].
    pub fn encode(&self, symbols: &[Vec<i8>], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        self.check_symbols(symbols)?;
        let mut shifted: Vec<Vec<i8>> = Vec::with_capacity(symbols.len());
        for (i, s) in symbols.iter().enumerate() {
            if i == 0 {
                shifted.push(s.clone());
            } else {
                shifted.push(cyclic_shift(s, i)?);
            }
        }
        bundle_binary(&shifted, rng)
    }

    /// Encode a sequence as the bundle of bound `k`-grams.
    ///
    /// Each contiguous length-`k` window `[s_w, …, s_{w+k-1}]` is bound into a single
    /// hypervector by element-wise product of its position-shifted members
    /// `ρ^{0}(s_w) ⊗ … ⊗ ρ^{k-1}(s_{w+k-1})`, and the `L − k + 1` window hypervectors are
    /// bundled by majority vote.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `symbols` is empty.
    /// - [`HdcError::DimensionMismatch`] if any symbol's length differs from [`Self::dim`].
    /// - [`HdcError::InvalidNgramOrder`] if `k == 0` or `k` exceeds the sequence length `L`.
    pub fn encode_ngrams(
        &self,
        symbols: &[Vec<i8>],
        k: usize,
        rng: &mut LcgRng,
    ) -> HdcResult<Vec<i8>> {
        self.check_symbols(symbols)?;
        if k == 0 || k > symbols.len() {
            return Err(HdcError::InvalidNgramOrder(k));
        }
        let n_windows = symbols.len() - k + 1;
        let mut window_hvs: Vec<Vec<i8>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            // Bind the k shifted members; start from the all-+1 identity of `⊗`.
            let mut bound: Vec<i8> = vec![1i8; self.dim];
            for j in 0..k {
                let s = &symbols[w + j];
                let shifted = if j == 0 {
                    s.clone()
                } else {
                    cyclic_shift(s, j)?
                };
                bound = binary_bind(&bound, &shifted)?;
            }
            window_hvs.push(bound);
        }
        bundle_binary(&window_hvs, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::cosine_binary;
    use crate::vector::binary::random_binary;

    fn three_symbols(seed: u64, dim: usize) -> (Vec<i8>, Vec<i8>, Vec<i8>) {
        let mut rng = LcgRng::new(seed);
        let a = random_binary(dim, &mut rng).expect("a");
        let b = random_binary(dim, &mut rng).expect("b");
        let c = random_binary(dim, &mut rng).expect("c");
        (a, b, c)
    }

    #[test]
    fn order_changes_encoding() {
        // (a) Same multiset, different order → (nearly) orthogonal encodings.
        let dim = 1024;
        let (a, b, c) = three_symbols(11, dim);
        let enc = SequenceHdEncoder::new(dim).expect("new");
        let seq1 = vec![a.clone(), b.clone(), c.clone()];
        let seq2 = vec![b.clone(), c.clone(), a.clone()]; // cyclic reorder, no shared (sym,pos)
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        let h1 = enc.encode(&seq1, &mut r1).expect("h1");
        let h2 = enc.encode(&seq2, &mut r2).expect("h2");
        let sim = cosine_binary(&h1, &h2).expect("cos");
        assert!(
            sim.abs() < 0.4,
            "reordered sequences too similar: sim={sim:.3}"
        );
    }

    #[test]
    fn identical_sequences_identical_encoding() {
        // (b) Identical input → identical output (odd length ⇒ rng-independent).
        let dim = 512;
        let (a, b, c) = three_symbols(22, dim);
        let enc = SequenceHdEncoder::new(dim).expect("new");
        let seq = vec![a, b, c];
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(999); // different rng on purpose: odd length is deterministic
        let h1 = enc.encode(&seq, &mut r1).expect("h1");
        let h2 = enc.encode(&seq, &mut r2).expect("h2");
        assert_eq!(h1, h2);
    }

    #[test]
    fn palindrome_reverse_invariant_but_general_reverse_differs() {
        // (c) Symbol-palindrome [A,B,A] reverses to itself → identical encoding;
        //     a non-palindrome [A,B,C] reverses to [C,B,A] → strictly different encoding.
        let dim = 1024;
        let (a, b, c) = three_symbols(33, dim);
        let enc = SequenceHdEncoder::new(dim).expect("new");

        let pal = vec![a.clone(), b.clone(), a.clone()];
        let pal_rev = vec![a.clone(), b.clone(), a.clone()];
        let mut r1 = LcgRng::new(5);
        let mut r2 = LcgRng::new(5);
        let hp = enc.encode(&pal, &mut r1).expect("hp");
        let hpr = enc.encode(&pal_rev, &mut r2).expect("hpr");
        assert_eq!(
            hp, hpr,
            "palindrome and its reverse must encode identically"
        );

        let seq = vec![a.clone(), b.clone(), c.clone()];
        let rev = vec![c, b, a];
        let mut r3 = LcgRng::new(6);
        let mut r4 = LcgRng::new(6);
        let hs = enc.encode(&seq, &mut r3).expect("hs");
        let hr = enc.encode(&rev, &mut r4).expect("hr");
        let sim = cosine_binary(&hs, &hr).expect("cos");
        assert!(
            sim < 0.7,
            "non-palindrome reverse should differ clearly: sim={sim:.3}"
        );
    }

    #[test]
    fn ngram_encode_and_order_sensitivity() {
        let dim = 1024;
        let (a, b, c) = three_symbols(44, dim);
        let enc = SequenceHdEncoder::new(dim).expect("new");
        let seq = vec![a.clone(), b.clone(), c.clone()];
        let reordered = vec![a.clone(), c.clone(), b.clone()];
        let mut r1 = LcgRng::new(3);
        let mut r2 = LcgRng::new(3);
        let g1 = enc.encode_ngrams(&seq, 2, &mut r1).expect("g1");
        let g2 = enc.encode_ngrams(&reordered, 2, &mut r2).expect("g2");
        assert_eq!(g1.len(), dim);
        let sim = cosine_binary(&g1, &g2).expect("cos");
        assert!(sim < 0.9, "n-grams should be order sensitive: sim={sim:.3}");
    }

    #[test]
    fn ngram_window_too_large_errors() {
        // (d) k > L → error; also k == 0 → error.
        let dim = 256;
        let (a, b, _c) = three_symbols(55, dim);
        let enc = SequenceHdEncoder::new(dim).expect("new");
        let seq = vec![a, b];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            enc.encode_ngrams(&seq, 3, &mut rng),
            Err(HdcError::InvalidNgramOrder(3))
        ));
        assert!(matches!(
            enc.encode_ngrams(&seq, 0, &mut rng),
            Err(HdcError::InvalidNgramOrder(0))
        ));
    }

    #[test]
    fn dimension_mismatch_among_symbols_errors() {
        // (e) Symbols of inconsistent dimension → DimensionMismatch.
        let enc = SequenceHdEncoder::new(256).expect("new");
        let mut rng = LcgRng::new(1);
        let good = random_binary(256, &mut rng).expect("good");
        let bad = random_binary(128, &mut rng).expect("bad");
        let seq = vec![good, bad];
        assert!(matches!(
            enc.encode(&seq, &mut rng),
            Err(HdcError::DimensionMismatch {
                expected: 256,
                got: 128
            })
        ));
    }

    #[test]
    fn empty_sequence_errors() {
        // (f) Empty sequence → EmptyInput.
        let enc = SequenceHdEncoder::new(256).expect("new");
        let mut rng = LcgRng::new(1);
        let empty: Vec<Vec<i8>> = Vec::new();
        assert!(matches!(
            enc.encode(&empty, &mut rng),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            enc.encode_ngrams(&empty, 1, &mut rng),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn new_rejects_zero_dim() {
        assert!(matches!(
            SequenceHdEncoder::new(0),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn single_symbol_roundtrips() {
        let dim = 512;
        let mut rng = LcgRng::new(77);
        let a = random_binary(dim, &mut rng).expect("a");
        let enc = SequenceHdEncoder::new(dim).expect("new");
        let h = enc
            .encode(std::slice::from_ref(&a), &mut rng)
            .expect("encode");
        // ρ^0(a) bundled alone is a itself.
        assert_eq!(h, a);
    }
}
