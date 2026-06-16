//! Burrows–Wheeler transform and FM-index with backward search.
//!
//! References:
//! * Michael Burrows & David J. Wheeler, *"A Block-sorting Lossless Data
//!   Compression Algorithm"*, Digital SRC Research Report 124, 1994 — the BWT
//!   and its LF-mapping inverse.
//! * Paolo Ferragina & Giovanni Manzini, *"Opportunistic Data Structures with
//!   Applications"*, FOCS 2000, pp. 390–398 — the FM-index (`C` array + `Occ`
//!   rank) and backward search for `count`/`locate`.
//!
//! # Construction
//!
//! A unique sentinel `$` — modelled here as a rank smaller than every real byte
//! — is appended to the text. The BWT is then read off the suffix array (built
//! by [`crate::string::SuffixArray`], reused verbatim): for each suffix position
//! `sa[i]`, the BWT character is the byte *preceding* that suffix cyclically,
//! `T[sa[i] − 1]`, with the position `0` mapping to the sentinel.
//!
//! ```text
//! T = "banana", T$ = "banana$"
//! sorted rotations →  BWT = "annb$aa"   (last column)
//! ```
//!
//! # FM-index
//!
//! Two precomputed tables turn the BWT into a searchable index:
//!
//! * `C[c]` — the number of characters in `T$` that are **strictly smaller**
//!   than `c`; equivalently the index of the first row beginning with `c`.
//! * `Occ(c, i)` — the number of occurrences of `c` in `BWT[0..i]` (a rank
//!   query). Stored here as a full `|Σ| × (n+1)` prefix-sum table for `O(1)`
//!   rank, which is `O(n |Σ|)` space — appropriate for the byte alphabet and
//!   exact, deterministic queries.
//!
//! The **LF-mapping** `LF(i) = C[BWT[i]] + Occ(BWT[i], i)` sends row `i` to the
//! row obtained by rotating its first column to the last; iterating it from the
//! sentinel row reconstructs `T` right-to-left, which is exactly the inverse
//! BWT.
//!
//! # Backward search
//!
//! Matching a pattern `p` proceeds from its last character to its first,
//! maintaining the half-open SA range `[lo, hi)` of rows whose suffix is
//! prefixed by the processed pattern tail:
//!
//! ```text
//! lo ← C[c] + Occ(c, lo)
//! hi ← C[c] + Occ(c, hi)
//! ```
//!
//! When the range becomes empty the pattern is absent. [`FmIndex::count`]
//! returns `hi − lo`; [`FmIndex::locate`] maps each row in the final range back
//! to a text position through the stored suffix array.
//!
//! Inputs are raw bytes (`&[u8]`).

use crate::error::{SeqError, SeqResult};
use crate::string::SuffixArray;

/// Alphabet cardinality including the sentinel: 256 byte values plus `$`.
const SIGMA: usize = 257;

/// Symbol code for the sentinel `$` (strictly smallest).
const SENTINEL: usize = 0;

/// Map a data byte to its FM-index symbol code (`1..=256`, leaving `0` for `$`).
fn code_of(byte: u8) -> usize {
    byte as usize + 1
}

/// An FM-index over a byte string: BWT, `C` array, and `Occ` rank table, with
/// the suffix array retained for `locate`.
///
/// Build with [`FmIndex::new`]. The index supports exact-occurrence
/// [`FmIndex::count`], position [`FmIndex::locate`], and full text recovery via
/// [`FmIndex::inverse_bwt`] (the BWT round-trips).
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::FmIndex;
///
/// let fm = FmIndex::new(b"banana").expect("non-empty");
/// assert_eq!(fm.count(b"ana"), 2);
/// assert_eq!(fm.locate(b"ana"), vec![1, 3]);
/// assert_eq!(fm.count(b"xyz"), 0);
/// assert_eq!(fm.inverse_bwt(), b"banana");
/// ```
#[derive(Debug, Clone)]
pub struct FmIndex {
    /// BWT as symbol codes over `T$` (length `n + 1`).
    bwt: Vec<usize>,
    /// Suffix array of `T$` (length `n + 1`); `sa[i]` is a text position into
    /// `T$`, where position `n` denotes the sentinel.
    sa: Vec<usize>,
    /// `C[c]` = number of symbols in `T$` strictly less than `c`.
    c: Vec<usize>,
    /// `occ[c][i]` = occurrences of symbol `c` in `bwt[0..i]`, for `i ≤ n+1`.
    occ: Vec<Vec<usize>>,
    /// Length of the original text (without the sentinel).
    text_len: usize,
}

impl FmIndex {
    /// Build the FM-index of `s`.
    ///
    /// Internally appends a unique sentinel, derives the BWT from the reused
    /// SA-IS suffix array, and precomputes the `C` and `Occ` tables.
    ///
    /// # Errors
    ///
    /// Returns [`SeqError::EmptyInput`] for an empty `s`, consistent with the
    /// sibling string modules.
    pub fn new(s: &[u8]) -> SeqResult<Self> {
        if s.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let n = s.len();

        // Suffix array of T (without sentinel), reusing module 2.
        let sa_no_sentinel = SuffixArray::new(s)?;
        let sa_t = sa_no_sentinel.sa();

        // Build the suffix array of T$ by hand: the sentinel suffix (position n)
        // is lexicographically smallest, so it leads, followed by the suffixes
        // of T in the same relative order (the sentinel only ever appears at the
        // very end, so it cannot change the order among the real suffixes).
        let mut sa: Vec<usize> = Vec::with_capacity(n + 1);
        sa.push(n); // the sentinel suffix "$"
        sa.extend_from_slice(sa_t);

        // BWT over T$: bwt[i] = symbol preceding suffix sa[i] cyclically.
        let mut bwt = vec![0usize; n + 1];
        for (i, &p) in sa.iter().enumerate() {
            bwt[i] = if p == 0 {
                SENTINEL // wraps to the sentinel
            } else {
                code_of(s[p - 1])
            };
        }

        // C array: counts of each symbol in T$, then exclusive prefix sum.
        let mut counts = vec![0usize; SIGMA];
        counts[SENTINEL] += 1; // the single sentinel
        for &b in s {
            counts[code_of(b)] += 1;
        }
        let mut c = vec![0usize; SIGMA];
        let mut acc = 0usize;
        for sym in 0..SIGMA {
            c[sym] = acc;
            acc += counts[sym];
        }

        // Occ table: occ[sym][i] = occurrences of sym in bwt[0..i].
        let mut occ = vec![vec![0usize; n + 2]; SIGMA];
        for i in 0..=n {
            let sym = bwt[i];
            for s_idx in 0..SIGMA {
                occ[s_idx][i + 1] = occ[s_idx][i];
            }
            occ[sym][i + 1] += 1;
        }

        Ok(Self {
            bwt,
            sa,
            c,
            occ,
            text_len: n,
        })
    }

    /// Length of the original text (excluding the sentinel).
    pub fn text_len(&self) -> usize {
        self.text_len
    }

    /// Borrow the BWT as raw bytes, mapping the sentinel to `sentinel_byte`.
    ///
    /// The sentinel has no byte value of its own, so the caller supplies the
    /// placeholder used to render it. The placeholder is *not* required to be
    /// absent from the text; it is purely cosmetic for inspection/printing.
    pub fn bwt_bytes(&self, sentinel_byte: u8) -> Vec<u8> {
        self.bwt
            .iter()
            .map(|&sym| {
                if sym == SENTINEL {
                    sentinel_byte
                } else {
                    (sym - 1) as u8
                }
            })
            .collect()
    }

    /// `Occ(sym, i)`: occurrences of symbol `sym` in `BWT[0..i]`.
    fn occ(&self, sym: usize, i: usize) -> usize {
        self.occ[sym][i]
    }

    /// The LF-mapping `LF(i) = C[BWT[i]] + Occ(BWT[i], i)`.
    fn lf(&self, i: usize) -> usize {
        let sym = self.bwt[i];
        self.c[sym] + self.occ(sym, i)
    }

    /// Recover the original text by inverting the BWT through the LF-mapping.
    ///
    /// Starts at the sentinel row (row `0`, since the sentinel sorts first) and
    /// walks LF backwards, emitting characters right-to-left.
    pub fn inverse_bwt(&self) -> Vec<u8> {
        let n = self.text_len;
        let mut out = vec![0u8; n];
        // Row 0 is the sentinel row (T$ sorted ⇒ "$..." is first). The character
        // BWT[0] is the last real character of T; walking LF reveals the rest.
        let mut row = 0usize;
        for k in (0..n).rev() {
            let sym = self.bwt[row];
            // sym is never the sentinel here for the first n steps because the
            // sentinel appears exactly once and is reached only after all n real
            // characters have been emitted.
            out[k] = (sym - 1) as u8;
            row = self.lf(row);
        }
        out
    }

    /// Backward-search the half-open SA range `[lo, hi)` of rows whose suffix is
    /// prefixed by `pattern`. Returns `None` for an empty pattern (no defined
    /// range) and an empty range `lo == hi` when absent.
    fn backward_search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.sa.len(); // n + 1
        for &b in pattern.iter().rev() {
            let sym = code_of(b);
            lo = self.c[sym] + self.occ(sym, lo);
            hi = self.c[sym] + self.occ(sym, hi);
            if lo >= hi {
                return Some((lo, lo)); // empty range; pattern absent
            }
        }
        Some((lo, hi))
    }

    /// Number of occurrences of `pattern` in the text (backward search).
    ///
    /// Returns `0` for an empty pattern or when the pattern does not occur.
    pub fn count(&self, pattern: &[u8]) -> usize {
        match self.backward_search(pattern) {
            Some((lo, hi)) => hi - lo,
            None => 0,
        }
    }

    /// Sorted text positions of every occurrence of `pattern`.
    ///
    /// Each row in the final backward-search range maps to a text position
    /// through the stored suffix array. Returns an empty vector for an empty or
    /// absent pattern.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        match self.backward_search(pattern) {
            Some((lo, hi)) if lo < hi => {
                let mut positions: Vec<usize> = self.sa[lo..hi].to_vec();
                positions.sort_unstable();
                positions
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_search(p: &[u8], t: &[u8]) -> Vec<usize> {
        let (m, n) = (p.len(), t.len());
        if m == 0 || m > n {
            return Vec::new();
        }
        (0..=(n - m)).filter(|&i| &t[i..i + m] == p).collect()
    }

    fn random_bytes(rng: &mut crate::handle::LcgRng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| alphabet[rng.next_usize(alphabet.len())])
            .collect()
    }

    /// (a) The BWT is invertible: inverse-BWT recovers the original exactly.
    #[test]
    fn bwt_round_trips() {
        for s in [
            b"banana".as_slice(),
            b"mississippi",
            b"abracadabra",
            b"aaaa",
            b"a",
            b"the quick brown fox",
        ] {
            let fm = FmIndex::new(s).expect("non-empty");
            assert_eq!(fm.inverse_bwt(), s, "round-trip for {s:?}");
        }
        let mut rng = crate::handle::LcgRng::new(101);
        for &alphabet in &[b"a".as_slice(), b"ab", b"abc", b"abcd"] {
            for _ in 0..400 {
                let len = 1 + rng.next_usize(40);
                let s = random_bytes(&mut rng, alphabet, len);
                let fm = FmIndex::new(&s).expect("non-empty");
                assert_eq!(fm.inverse_bwt(), s, "round-trip for {s:?}");
            }
        }
    }

    /// (b) Backward-search count equals the true number of occurrences,
    /// including absent patterns (→ 0).
    #[test]
    fn count_matches_naive() {
        let fm = FmIndex::new(b"mississippi").expect("non-empty");
        assert_eq!(fm.count(b"issi"), 2);
        assert_eq!(fm.count(b"ss"), 2);
        assert_eq!(fm.count(b"i"), 4);
        assert_eq!(fm.count(b"mississippi"), 1);
        assert_eq!(fm.count(b"xyz"), 0); // absent
        assert_eq!(fm.count(b"ppp"), 0); // absent
        assert_eq!(fm.count(b""), 0); // empty pattern

        let mut rng = crate::handle::LcgRng::new(202);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..400 {
                let tlen = 1 + rng.next_usize(40);
                let plen = 1 + rng.next_usize(5);
                let t = random_bytes(&mut rng, alphabet, tlen);
                let p = random_bytes(&mut rng, alphabet, plen);
                let fm = FmIndex::new(&t).expect("non-empty");
                assert_eq!(fm.count(&p), naive_search(&p, &t).len(), "p={p:?} t={t:?}");
            }
        }
    }

    /// (c) Locate returns the correct sorted positions.
    #[test]
    fn locate_matches_naive() {
        let fm = FmIndex::new(b"banana").expect("non-empty");
        assert_eq!(fm.locate(b"ana"), vec![1, 3]);
        assert_eq!(fm.locate(b"a"), vec![1, 3, 5]);
        assert_eq!(fm.locate(b"na"), vec![2, 4]);
        assert!(fm.locate(b"xyz").is_empty());
        assert!(fm.locate(b"").is_empty());

        let mut rng = crate::handle::LcgRng::new(303);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..400 {
                let tlen = 1 + rng.next_usize(40);
                let plen = 1 + rng.next_usize(5);
                let t = random_bytes(&mut rng, alphabet, tlen);
                let p = random_bytes(&mut rng, alphabet, plen);
                let fm = FmIndex::new(&t).expect("non-empty");
                let mut want = naive_search(&p, &t);
                want.sort_unstable();
                assert_eq!(fm.locate(&p), want, "p={p:?} t={t:?}");
            }
        }
    }

    /// (d) The LF-mapping is a permutation of the rows.
    #[test]
    fn lf_is_permutation() {
        let mut rng = crate::handle::LcgRng::new(404);
        for _ in 0..300 {
            let len = 1 + rng.next_usize(30);
            let s = random_bytes(&mut rng, b"abc", len);
            let fm = FmIndex::new(&s).expect("non-empty");
            let rows = fm.sa.len(); // n + 1
            let mut seen = vec![false; rows];
            for i in 0..rows {
                let target = fm.lf(i);
                assert!(target < rows, "LF out of range");
                assert!(!seen[target], "LF not injective at {i}");
                seen[target] = true;
            }
            assert!(seen.iter().all(|&b| b), "LF not surjective");
        }
    }

    /// (e) C/Occ consistency: Occ monotone non-decreasing in i, C cumulative.
    #[test]
    fn c_and_occ_consistent() {
        let mut rng = crate::handle::LcgRng::new(505);
        for _ in 0..100 {
            let len = 1 + rng.next_usize(30);
            let s = random_bytes(&mut rng, b"abc", len);
            let fm = FmIndex::new(&s).expect("non-empty");
            let rows = fm.sa.len();

            // C is cumulative (non-decreasing) and starts at 0.
            assert_eq!(fm.c[SENTINEL], 0);
            for sym in 1..SIGMA {
                assert!(fm.c[sym] >= fm.c[sym - 1], "C not cumulative");
            }
            // C[max] + (count of max sym) == rows; equivalently the last bucket
            // boundary plus its size equals the total number of rows.
            assert!(*fm.c.last().expect("non-empty C") <= rows);

            // Occ monotone in i for every symbol, and Occ(sym, rows) totals to
            // the count of sym, whose sum over symbols equals rows.
            let mut total = 0usize;
            for sym in 0..SIGMA {
                for i in 0..rows {
                    assert!(fm.occ(sym, i + 1) >= fm.occ(sym, i), "Occ not monotone");
                }
                total += fm.occ(sym, rows);
            }
            assert_eq!(total, rows, "Occ totals must sum to row count");

            // Cross-check C against Occ at the end: C[sym] equals the number of
            // BWT symbols strictly less than sym, i.e. Σ_{k<sym} Occ(k, rows).
            let mut acc = 0usize;
            for sym in 0..SIGMA {
                assert_eq!(fm.c[sym], acc, "C vs Occ mismatch at {sym}");
                acc += fm.occ(sym, rows);
            }
        }
    }

    /// (f) A sentinel-terminated string round-trips (text containing the byte we
    /// later render the sentinel as — the internal sentinel is distinct, so this
    /// still works).
    #[test]
    fn sentinel_rendering_does_not_collide() {
        // Use a text that contains byte 0 to prove the internal sentinel is a
        // separate symbol from any data byte (the BWT is over symbol codes).
        let s = &[0u8, 1u8, 0u8, 2u8, 0u8];
        let fm = FmIndex::new(s).expect("non-empty");
        assert_eq!(fm.inverse_bwt(), s, "round-trip with embedded zero bytes");
        assert_eq!(fm.count(&[0u8]), 3);
        assert_eq!(fm.locate(&[0u8]), vec![0, 2, 4]);
        assert_eq!(fm.count(&[0u8, 0u8]), 0); // no adjacent zeros

        // The rendered BWT with '$' as a placeholder has the sentinel exactly
        // once even though byte 0 occurs three times in the source.
        let rendered = fm.bwt_bytes(b'$');
        assert_eq!(rendered.iter().filter(|&&b| b == b'$').count(), 1);
    }

    /// Empty input is rejected.
    #[test]
    fn empty_input_errors() {
        assert!(matches!(FmIndex::new(b""), Err(SeqError::EmptyInput)));
    }

    /// The BWT of "banana" is the textbook "annb$aa".
    #[test]
    fn banana_bwt_textbook() {
        let fm = FmIndex::new(b"banana").expect("non-empty");
        let rendered = fm.bwt_bytes(b'$');
        assert_eq!(rendered.as_slice(), b"annb$aa");
    }
}
