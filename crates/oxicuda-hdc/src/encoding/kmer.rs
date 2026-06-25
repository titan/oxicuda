//! k-mer hypervector encoding for alignment-free genomic sequence comparison.
//!
//! This module realises the hyperdimensional-computing (HDC) approach to genomics popularised
//! by GenieHD (Kim, Imani, Kim & Rosing, *"GenieHD: Efficient DNA Pattern Matching Accelerator
//! Using Hyperdimensional Computing"*, DATE 2020) and the earlier DNA pattern-matching work of
//! Imani et al. (2018). It is the genomic specialisation of Kanerva's n-gram text-classification
//! construction (Joshi, Halseth & Kanerva 2016): a biological sequence over a fixed alphabet
//! (canonically DNA `{A, C, G, T}`) is turned into a single fixed-width binary hypervector whose
//! geometry encodes the *multiset of ordered k-mers* the sequence contains. Cosine similarity
//! between two such hypervectors then approximates their shared-k-mer content, giving an
//! **alignment-free** measure of sequence relatedness: two sequences that share many length-`k`
//! sub-words (because they are homologous, or differ only by a handful of point mutations) encode
//! to nearly parallel hypervectors, whereas unrelated sequences encode to nearly orthogonal ones.
//!
//! # Construction
//!
//! 1. **Symbol hypervectors.** Each of the `alphabet_size` symbols is assigned an independent
//!    random `±1` hypervector `sₐ` of dimension `D`. Distinct symbols are (with high probability
//!    in high dimension) mutually quasi-orthogonal.
//!
//! 2. **Order-sensitive k-mer binding.** A length-`k` window (k-mer) `w = (c₀, c₁, …, c_{k−1})`
//!    is bound into one hypervector
//!
//!    ```text
//!    g(w) = ρ⁰(s_{c₀}) ⊗ ρ¹(s_{c₁}) ⊗ … ⊗ ρ^{k−1}(s_{c_{k−1}})
//!    ```
//!
//!    where `ρʲ` is the circular shift by `j` positions and `⊗` is the element-wise `±1` product
//!    (binding). Position-dependent shifting is what makes the k-mer *order-sensitive*: because
//!    `ρʲ(s)` is quasi-orthogonal to `s` for `j ≠ 0`, the k-mers `ACG` and `GCA` bind to
//!    near-orthogonal hypervectors even though they share the same multiset of symbols. Without
//!    the shifts the product would be commutative and every permutation of a k-mer would collide.
//!
//! 3. **Sequence bundling.** The `L − k + 1` k-mer hypervectors obtained by sliding the window
//!    one symbol at a time across a length-`L` sequence are bundled by binary majority vote into
//!    the sequence hypervector. Bundling is a similarity-preserving superposition, so the result
//!    is similar to every k-mer it contains; two sequences sharing many k-mers therefore overlap
//!    in the bundle and score a high cosine similarity.
//!
//! All hypervectors are `Vec<i8>` with entries in `{−1, +1}`, matching the crate-wide binary
//! hypervector representation. The encoder is deterministic given its symbol hypervectors and the
//! tie-breaking RNG; for sequences whose number of k-mers is odd the bundle has no ties and the
//! output is fully RNG-independent.

use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::cyclic_shift;
use crate::vector::binary::random_binary;

/// k-mer encoder for alignment-free comparison of symbolic (e.g. genomic) sequences.
///
/// The encoder owns one random `±1` hypervector per alphabet symbol and turns an arbitrary
/// sequence of symbol indices into a single binary hypervector via order-sensitive k-mer binding
/// followed by majority bundling (see the [module documentation](self) for the full construction
/// and the GenieHD reference). For the canonical DNA alphabet use [`KmerEncoder::dna`] together
/// with [`KmerEncoder::encode_chars`].
#[derive(Debug, Clone)]
pub struct KmerEncoder {
    /// Hypervector dimension `D` shared by every symbol and every produced hypervector.
    dim: usize,
    /// k-mer length (window size); always `≥ 1`.
    k: usize,
    /// Number of distinct alphabet symbols (valid symbol indices are `0..alphabet_size`).
    alphabet_size: usize,
    /// One random `±1` hypervector per alphabet symbol (`alphabet_size` entries, each length `dim`).
    symbol_hvs: Vec<Vec<i8>>,
}

impl KmerEncoder {
    /// Build a k-mer encoder over an arbitrary alphabet of `alphabet_size` symbols.
    ///
    /// `alphabet_size` independent random `±1` hypervectors of dimension `dim` are drawn from
    /// `rng`, one per symbol. Valid symbol indices are then `0..alphabet_size`.
    ///
    /// # Arguments
    ///
    /// - `alphabet_size`: number of distinct symbols (must be `≥ 1`).
    /// - `k`: k-mer (window) length (must be `≥ 1`).
    /// - `dim`: hypervector dimension (must be `> 0`).
    /// - `rng`: deterministic random source used to generate the symbol hypervectors.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `alphabet_size == 0`.
    /// - [`HdcError::InvalidNgramOrder`] (carrying `k`) if `k == 0`.
    pub fn new(alphabet_size: usize, k: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if alphabet_size == 0 {
            return Err(HdcError::EmptyInput);
        }
        if k == 0 {
            return Err(HdcError::InvalidNgramOrder(k));
        }
        let mut symbol_hvs = Vec::with_capacity(alphabet_size);
        for _ in 0..alphabet_size {
            symbol_hvs.push(random_binary(dim, rng)?);
        }
        Ok(Self {
            dim,
            k,
            alphabet_size,
            symbol_hvs,
        })
    }

    /// Build a DNA k-mer encoder over the 4-symbol nucleotide alphabet `{A, C, G, T}`.
    ///
    /// This is a convenience wrapper around [`KmerEncoder::new`] with `alphabet_size == 4`,
    /// using the fixed symbol assignment `A = 0`, `C = 1`, `G = 2`, `T = 3`. The resulting
    /// encoder accepts nucleotide strings through [`KmerEncoder::encode_chars`].
    ///
    /// # Arguments
    ///
    /// - `k`: k-mer (window) length (must be `≥ 1`).
    /// - `dim`: hypervector dimension (must be `> 0`).
    /// - `rng`: deterministic random source used to generate the four symbol hypervectors.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`KmerEncoder::new`]: [`HdcError::ZeroDimension`] if `dim == 0`
    /// and [`HdcError::InvalidNgramOrder`] if `k == 0`.
    pub fn dna(k: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        Self::new(4, k, dim, rng)
    }

    /// Hypervector dimension `D` produced by this encoder.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// k-mer (window) length used by this encoder.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of distinct alphabet symbols (valid symbol indices are `0..alphabet_size`).
    #[must_use]
    pub fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    /// Borrow the random hypervector assigned to alphabet symbol `s`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `s >= alphabet_size`.
    pub fn symbol_hv(&self, s: usize) -> HdcResult<&[i8]> {
        if s >= self.alphabet_size {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: s,
                max: self.alphabet_size,
            });
        }
        Ok(&self.symbol_hvs[s])
    }

    /// Bind a single length-`k` window of symbol indices into one order-sensitive k-mer hypervector.
    ///
    /// The result is `ρ⁰(s_{w₀}) ⊗ ρ¹(s_{w₁}) ⊗ … ⊗ ρ^{k−1}(s_{w_{k−1}})`, i.e. each symbol
    /// hypervector is circularly shifted by its offset *within the window* and the shifted copies
    /// are bound by element-wise `±1` product. The accumulation starts from the all-`+1` identity
    /// of binding, so a `k == 1` window simply returns the (unshifted) symbol hypervector. Because
    /// the per-position shifts break the commutativity of the product, permuting the window's
    /// symbols generally yields a near-orthogonal hypervector — this is what makes the encoding
    /// order-sensitive.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] (with `expected == k`, `got == window.len()`) if the
    ///   window does not have exactly `k` symbols.
    /// - [`HdcError::FeatureIndexOutOfRange`] if any symbol index is `>= alphabet_size`.
    pub fn kmer_hv(&self, window: &[usize]) -> HdcResult<Vec<i8>> {
        if window.len() != self.k {
            return Err(HdcError::DimensionMismatch {
                expected: self.k,
                got: window.len(),
            });
        }
        // Start from the all-+1 identity of the ±1 binding product.
        let mut bound: Vec<i8> = vec![1i8; self.dim];
        for (j, &sym) in window.iter().enumerate() {
            if sym >= self.alphabet_size {
                return Err(HdcError::FeatureIndexOutOfRange {
                    feat: sym,
                    max: self.alphabet_size,
                });
            }
            // ρʲ(s_sym): shift by the within-window offset j (j == 0 leaves the HV unchanged).
            let shifted = if j == 0 {
                self.symbol_hvs[sym].clone()
            } else {
                cyclic_shift(&self.symbol_hvs[sym], j)?
            };
            bound = binary_bind(&bound, &shifted)?;
        }
        Ok(bound)
    }

    /// Encode a sequence of symbol indices into a single bundled k-mer hypervector.
    ///
    /// A length-`k` window is slid one symbol at a time across `sequence`, each window is bound
    /// into an order-sensitive k-mer hypervector via [`KmerEncoder::kmer_hv`], and the resulting
    /// `sequence.len() − k + 1` k-mer hypervectors are bundled by binary majority vote. `rng` is
    /// consulted only to break ties in the majority bundle (which arise only when an even number
    /// of k-mers cancel exactly at a component); when the number of k-mers is odd the output is
    /// independent of `rng`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::InvalidNgramOrder`] (carrying `k`) if `sequence.len() < k` (no full window
    ///   fits).
    /// - [`HdcError::FeatureIndexOutOfRange`] if any symbol index is `>= alphabet_size`.
    pub fn encode(&self, sequence: &[usize], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        if sequence.len() < self.k {
            return Err(HdcError::InvalidNgramOrder(self.k));
        }
        // Validate all symbol indices up front so a bad symbol fails fast and uniformly.
        for &sym in sequence {
            if sym >= self.alphabet_size {
                return Err(HdcError::FeatureIndexOutOfRange {
                    feat: sym,
                    max: self.alphabet_size,
                });
            }
        }
        let n_kmers = sequence.len() - self.k + 1;
        let mut kmer_hvs: Vec<Vec<i8>> = Vec::with_capacity(n_kmers);
        for start in 0..n_kmers {
            let window = &sequence[start..start + self.k];
            kmer_hvs.push(self.kmer_hv(window)?);
        }
        bundle_binary(&kmer_hvs, rng)
    }

    /// Encode a DNA string into a single bundled k-mer hypervector.
    ///
    /// Each character of `dna` is mapped case-insensitively (`A`/`a → 0`, `C`/`c → 1`,
    /// `G`/`g → 2`, `T`/`t → 3`) to a symbol index and the resulting index sequence is encoded
    /// with [`KmerEncoder::encode`]. This convenience entry point is meaningful **only for the
    /// 4-symbol DNA alphabet**: if the encoder was not built over four symbols the call is
    /// rejected, because the nucleotide ↔ index mapping is undefined for other alphabets.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] (with `expected == 4`, `got == alphabet_size`) if the
    ///   encoder's alphabet is not the 4-symbol DNA alphabet.
    /// - [`HdcError::EmptyInput`] if `dna` contains no characters.
    /// - [`HdcError::FeatureIndexOutOfRange`] (with `feat == 4`, the first out-of-DNA index)
    ///   if `dna` contains any character other than `A`, `C`, `G`, `T` (any case).
    /// - [`HdcError::InvalidNgramOrder`] if the decoded sequence is shorter than `k`.
    pub fn encode_chars(&self, dna: &str, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        if self.alphabet_size != 4 {
            return Err(HdcError::DimensionMismatch {
                expected: 4,
                got: self.alphabet_size,
            });
        }
        if dna.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let mut sequence: Vec<usize> = Vec::with_capacity(dna.len());
        for ch in dna.chars() {
            let sym = match ch {
                'A' | 'a' => 0usize,
                'C' | 'c' => 1usize,
                'G' | 'g' => 2usize,
                'T' | 't' => 3usize,
                // Any non-nucleotide character is reported as an out-of-alphabet feature index
                // (4 is the first index past the DNA alphabet {0,1,2,3}).
                _ => {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: 4,
                        max: self.alphabet_size,
                    });
                }
            };
            sequence.push(sym);
        }
        self.encode(&sequence, rng)
    }

    /// Alignment-free similarity between two symbol-index sequences.
    ///
    /// Both sequences are encoded with [`KmerEncoder::encode`] and the cosine similarity of the
    /// two sequence hypervectors is returned. The score lies in `[−1, 1]` and grows with the
    /// amount of shared ordered k-mer content: sequences that are identical, homologous, or differ
    /// by only a few point mutations score near `1`, while unrelated sequences score near `0`.
    ///
    /// `rng` is used for the bundle tie-breaks of *both* encodings; pass freshly equally-seeded
    /// generators if a fully reproducible score is required for even-length k-mer counts.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`KmerEncoder::encode`] (e.g. [`HdcError::InvalidNgramOrder`] for
    /// a sequence shorter than `k`, or [`HdcError::FeatureIndexOutOfRange`] for an invalid symbol).
    pub fn similarity(&self, seq_a: &[usize], seq_b: &[usize], rng: &mut LcgRng) -> HdcResult<f32> {
        let hv_a = self.encode(seq_a, rng)?;
        let hv_b = self.encode(seq_b, rng)?;
        cosine_binary(&hv_a, &hv_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn new_rejects_zero_dim() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            KmerEncoder::new(4, 3, 0, &mut rng),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn new_rejects_zero_alphabet() {
        let mut rng = LcgRng::new(2);
        assert!(matches!(
            KmerEncoder::new(0, 3, 1024, &mut rng),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn new_rejects_zero_k() {
        let mut rng = LcgRng::new(3);
        assert!(matches!(
            KmerEncoder::new(4, 0, 1024, &mut rng),
            Err(HdcError::InvalidNgramOrder(0))
        ));
    }

    #[test]
    fn dna_builds_four_symbols() {
        let mut rng = LcgRng::new(4);
        let enc = KmerEncoder::dna(3, 1024, &mut rng).expect("dna encoder");
        assert_eq!(enc.alphabet_size(), 4);
        assert_eq!(enc.k(), 3);
        assert_eq!(enc.dim(), 1024);
        // Four distinct symbol hypervectors are available; index 4 is out of range.
        for s in 0..4 {
            let hv = enc.symbol_hv(s).expect("symbol hv");
            assert_eq!(hv.len(), 1024);
        }
        assert!(matches!(
            enc.symbol_hv(4),
            Err(HdcError::FeatureIndexOutOfRange { feat: 4, max: 4 })
        ));
    }

    #[test]
    fn same_sequence_same_encoding() {
        let mut rng = LcgRng::new(5);
        let enc = KmerEncoder::dna(3, 2048, &mut rng).expect("dna encoder");
        // 8 symbols, k = 3 → 6 k-mers (even). Use identically-seeded RNGs so the bundle
        // tie-breaks match and the encodings are bit-for-bit identical.
        let seq = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
        let mut ra = LcgRng::new(777);
        let mut rb = LcgRng::new(777);
        let hv1 = enc.encode(&seq, &mut ra).expect("encode 1");
        let hv2 = enc.encode(&seq, &mut rb).expect("encode 2");
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn point_mutation_more_similar_than_random() {
        // The defining alignment-free property: a sequence with a single point mutation shares
        // almost all of its k-mers with the original (only the few windows spanning the mutated
        // position change), so it is much more similar to the original than an unrelated random
        // sequence of the same length.
        let mut rng = LcgRng::new(6);
        let dim = 4096;
        // k = 6 over the 4-letter DNA alphabet gives 4^6 = 4096 distinct ordered k-mers, so two
        // unrelated random sequences share very few k-mers by chance and stay near orthogonal,
        // while a single point mutation leaves the great majority of k-mers intact.
        let enc = KmerEncoder::dna(6, dim, &mut rng).expect("dna encoder");

        // Draw random DNA symbols from the HIGH bits of the LCG (`>> 30` → top two bits → 0..3).
        // `next_usize(4)` would take the LOW two bits, which for this MMIX LCG cycle with period
        // four and produce a degenerate periodic sequence — see crate handle.rs (next_u32/next_bool
        // both use the high bits for the same reason).
        let rand_symbol = |g: &mut LcgRng| -> usize { (g.next_u32() >> 30) as usize };

        // A 64-symbol pseudo-random "reference" DNA sequence over {0,1,2,3} (own RNG stream so
        // its k-mers are diverse rather than periodic).
        let mut ref_rng = LcgRng::new(20_260_620);
        let reference: Vec<usize> = (0..64).map(|_| rand_symbol(&mut ref_rng)).collect();

        // Same sequence with one symbol changed (point mutation roughly in the middle).
        let mut mutated = reference.clone();
        mutated[32] = (mutated[32] + 1) % 4;

        // An unrelated random sequence of identical length, drawn from a different RNG stream.
        let mut other_rng = LcgRng::new(424_242);
        let random_seq: Vec<usize> = (0..64).map(|_| rand_symbol(&mut other_rng)).collect();

        let mut r1 = LcgRng::new(11);
        let mut r2 = LcgRng::new(11);
        let mut r3 = LcgRng::new(11);
        let hv_ref = enc.encode(&reference, &mut r1).expect("encode ref");
        let hv_mut = enc.encode(&mutated, &mut r2).expect("encode mut");
        let hv_rand = enc.encode(&random_seq, &mut r3).expect("encode rand");

        let sim_mut = cosine_binary(&hv_ref, &hv_mut).expect("cos mut");
        let sim_rand = cosine_binary(&hv_ref, &hv_rand).expect("cos rand");

        assert!(
            sim_mut > sim_rand,
            "point-mutation similarity ({sim_mut:.3}) must exceed random similarity ({sim_rand:.3})"
        );
        // Quantitative sanity: shared k-mers should keep the mutant clearly correlated while the
        // random sequence stays near orthogonal.
        assert!(
            sim_mut > 0.5,
            "point mutant should remain strongly similar: sim_mut={sim_mut:.3}"
        );
        assert!(
            sim_rand < 0.4,
            "unrelated sequence should be near orthogonal: sim_rand={sim_rand:.3}"
        );
    }

    #[test]
    fn reversing_non_palindrome_changes_encoding() {
        // Order sensitivity: reversing a non-palindromic sequence reorders the symbols inside
        // every k-mer, so (thanks to the position-dependent shifts) the bound k-mers change and
        // the bundled encoding becomes clearly dissimilar from the original.
        let mut rng = LcgRng::new(7);
        let dim = 4096;
        let enc = KmerEncoder::dna(3, dim, &mut rng).expect("dna encoder");

        let seq: Vec<usize> = vec![0, 1, 2, 3, 2, 1, 0, 3, 1, 2, 0, 3, 2, 1, 3, 0, 1];
        let reversed: Vec<usize> = seq.iter().rev().copied().collect();
        assert_ne!(seq, reversed, "test sequence must be non-palindromic");

        let mut r1 = LcgRng::new(13);
        let mut r2 = LcgRng::new(13);
        let hv = enc.encode(&seq, &mut r1).expect("encode");
        let hv_rev = enc.encode(&reversed, &mut r2).expect("encode rev");

        let sim = cosine_binary(&hv, &hv_rev).expect("cos");
        assert!(
            sim < 0.7,
            "reversed non-palindrome should differ clearly: sim={sim:.3}"
        );
    }

    #[test]
    fn encode_chars_matches_index_encode() {
        // encode_chars("ACGTACGT") must equal encode(&[0,1,2,3,0,1,2,3]) with the same RNG seed.
        let mut rng = LcgRng::new(8);
        let enc = KmerEncoder::dna(3, 2048, &mut rng).expect("dna encoder");
        let mut ra = LcgRng::new(321);
        let mut rb = LcgRng::new(321);
        let from_chars = enc.encode_chars("ACGTACGT", &mut ra).expect("encode_chars");
        let from_idx = enc
            .encode(&[0usize, 1, 2, 3, 0, 1, 2, 3], &mut rb)
            .expect("encode");
        assert_eq!(from_chars, from_idx);
        // Case-insensitivity: lowercase input yields the identical encoding.
        let mut rc = LcgRng::new(321);
        let from_lower = enc
            .encode_chars("acgtacgt", &mut rc)
            .expect("encode_chars lower");
        assert_eq!(from_lower, from_idx);
    }

    #[test]
    fn encode_chars_rejects_invalid_char() {
        let mut rng = LcgRng::new(9);
        let enc = KmerEncoder::dna(3, 1024, &mut rng).expect("dna encoder");
        let mut r = LcgRng::new(1);
        // 'N' (an ambiguity code) is not in {A,C,G,T} → out-of-range feature.
        assert!(matches!(
            enc.encode_chars("ACGNACGT", &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 4, max: 4 })
        ));
    }

    #[test]
    fn encode_chars_requires_dna_alphabet() {
        // A non-DNA alphabet cannot use the nucleotide-string entry point.
        let mut rng = LcgRng::new(10);
        let enc = KmerEncoder::new(5, 3, 1024, &mut rng).expect("encoder");
        let mut r = LcgRng::new(1);
        assert!(matches!(
            enc.encode_chars("ACGT", &mut r),
            Err(HdcError::DimensionMismatch {
                expected: 4,
                got: 5
            })
        ));
    }

    #[test]
    fn sequence_shorter_than_k_errors() {
        let mut rng = LcgRng::new(11);
        let enc = KmerEncoder::dna(4, 1024, &mut rng).expect("dna encoder");
        let mut r = LcgRng::new(1);
        // Only 3 symbols but k = 4 → no full window fits.
        assert!(matches!(
            enc.encode(&[0usize, 1, 2], &mut r),
            Err(HdcError::InvalidNgramOrder(4))
        ));
    }

    #[test]
    fn out_of_range_symbol_errors() {
        let mut rng = LcgRng::new(12);
        let enc = KmerEncoder::dna(3, 1024, &mut rng).expect("dna encoder");
        let mut r = LcgRng::new(1);
        // Symbol 4 is outside the DNA alphabet {0,1,2,3}.
        assert!(matches!(
            enc.encode(&[0usize, 1, 4, 2], &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 4, max: 4 })
        ));
    }

    #[test]
    fn kmer_hv_wrong_window_length_errors() {
        let mut rng = LcgRng::new(13);
        let enc = KmerEncoder::dna(3, 1024, &mut rng).expect("dna encoder");
        // Window of length 2 against k == 3.
        assert!(matches!(
            enc.kmer_hv(&[0usize, 1]),
            Err(HdcError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        // Window of length 4 against k == 3.
        assert!(matches!(
            enc.kmer_hv(&[0usize, 1, 2, 3]),
            Err(HdcError::DimensionMismatch {
                expected: 3,
                got: 4
            })
        ));
    }

    #[test]
    fn kmer_hv_is_order_sensitive() {
        // Two k-mers over the same symbol multiset but a different order bind to clearly
        // dissimilar hypervectors thanks to the per-position shifts.
        let mut rng = LcgRng::new(14);
        let enc = KmerEncoder::dna(3, 4096, &mut rng).expect("dna encoder");
        let acg = enc.kmer_hv(&[0usize, 1, 2]).expect("ACG");
        let gca = enc.kmer_hv(&[2usize, 1, 0]).expect("GCA");
        let sim = cosine_binary(&acg, &gca).expect("cos");
        assert!(
            sim.abs() < 0.4,
            "permuted k-mers should be near orthogonal: sim={sim:.3}"
        );
    }

    #[test]
    fn similarity_self_is_high() {
        // A sequence is maximally similar to itself (identical encoding ⇒ cosine 1).
        let mut rng = LcgRng::new(15);
        let enc = KmerEncoder::dna(4, 2048, &mut rng).expect("dna encoder");
        let seq: Vec<usize> = (0..40).map(|i| (i * 3 + 1) % 4).collect();
        let mut r = LcgRng::new(99);
        let sim = enc.similarity(&seq, &seq, &mut r).expect("similarity");
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "self-similarity should be 1.0: sim={sim:.6}"
        );
    }
}
