//! Myers' bit-vector algorithm for global Levenshtein edit distance.
//!
//! Reference: Gene Myers, *"A Fast Bit-Vector Algorithm for Approximate String
//! Matching Based on Dynamic Programming"*, Journal of the ACM 46(3), 1999.
//! The clean recurrence used here follows Heikki Hyyrö's formulation
//! (*"A Note on Bit-Parallel Alignment Computation"*, 2003), which restates
//! Myers' equations in a form that is easy to verify against the scalar
//! Wagner–Fischer dynamic program.
//!
//! # Idea
//!
//! The classic edit-distance DP fills an `(m+1) × (n+1)` table column by column.
//! Adjacent cells differ by exactly `+1`, `0`, or `-1`, so a whole DP column can
//! be encoded by two bit-vectors of length `m`:
//!
//! * `VP` — bit `j` set iff `D[j+1] - D[j] == +1` (a *positive* vertical delta),
//! * `VN` — bit `j` set iff `D[j+1] - D[j] == -1` (a *negative* vertical delta).
//!
//! Processing one text character then updates `(VP, VN)` with a constant number
//! of word operations, giving an `O(n · ⌈m / w⌉)` algorithm where `w = 64` is the
//! machine word width. For patterns of at most 64 symbols a *single* word holds
//! the entire column (`single_word`); longer patterns are split into 64-bit
//! blocks whose horizontal carries are threaded together (`blocked`).
//!
//! # Symmetry
//!
//! Levenshtein distance is symmetric, but only one of the two strings — the
//! *pattern* — is packed into bit-vectors. [`myers_edit_distance`] always packs
//! the **shorter** of the two inputs, which minimises the number of 64-bit
//! blocks without changing the result.

use crate::error::{SeqError, SeqResult};

/// Machine word width, in bits, used for the bit-parallel columns.
const WORD_BITS: usize = 64;

/// Build the pattern equality table `Peq`.
///
/// `peq[c]` has bit `j` set iff `pattern[j] == c`. For patterns longer than a
/// single word the table is laid out block-major: entry `c * blocks + b` holds
/// the mask for byte `c` restricted to block `b` (covering pattern positions
/// `64*b ..= 64*b + 63`).
fn build_peq(pattern: &[u8], blocks: usize) -> Vec<u64> {
    let mut peq = vec![0u64; 256 * blocks];
    for (j, &c) in pattern.iter().enumerate() {
        let block = j / WORD_BITS;
        let bit = j % WORD_BITS;
        peq[usize::from(c) * blocks + block] |= 1u64 << bit;
    }
    peq
}

/// Single-word Myers scan for patterns with `1 <= m <= 64`.
///
/// `pattern` is the (short) string packed into one 64-bit column; `text` is
/// scanned character by character. Returns the global Levenshtein distance.
///
/// Caller guarantees `1 <= pattern.len() <= 64`.
fn single_word(pattern: &[u8], text: &[u8]) -> usize {
    let m = pattern.len();
    let peq = build_peq(pattern, 1);

    // Low `m` bits all set: every vertical delta starts at +1 (D[j] = j).
    let mut vp: u64 = if m == WORD_BITS {
        u64::MAX
    } else {
        (1u64 << m) - 1
    };
    let mut vn: u64 = 0;

    // Distance of the empty text prefix against the full pattern is `m`.
    let mut score = m;
    let high_bit: u64 = 1u64 << (m - 1);

    for &c in text {
        let eq = peq[usize::from(c)];
        let xv = eq | vn;
        let xh = (((eq & vp).wrapping_add(vp)) ^ vp) | eq;

        let mut ph = vn | !(xh | vp);
        let mut mh = vp & xh;

        if ph & high_bit != 0 {
            score += 1;
        } else if mh & high_bit != 0 {
            score -= 1;
        }

        // Shift the horizontal deltas down into the next column. The injected
        // low bit encodes the boundary gap `D[0]` growing by one per column.
        ph = (ph << 1) | 1;
        mh <<= 1;

        vp = mh | !(xv | ph);
        vn = ph & xv;
    }

    score
}

/// Multi-word ("blocked") Myers scan for patterns with `m > 64`.
///
/// The pattern occupies `blocks = ⌈m / 64⌉` consecutive 64-bit columns. Per
/// column we sweep the blocks from low to high, threading the horizontal delta
/// of the bottom DP row between them as a signed carry: `hin = +1` (carry in
/// `ph`), `hin = -1` (carry in `mh`), or `hin = 0`. The score lives at the
/// bottom-right cell and is adjusted by the carry that leaves the **last**
/// block at the pattern's true high bit.
///
/// Caller guarantees `pattern.len() > 64`.
fn blocked(pattern: &[u8], text: &[u8]) -> usize {
    let m = pattern.len();
    let blocks = m.div_ceil(WORD_BITS);
    let peq = build_peq(pattern, blocks);

    // Vertical-delta vectors, one 64-bit lane per block.
    let mut vp = vec![u64::MAX; blocks];
    let mut vn = vec![0u64; blocks];

    // Position of the pattern's true high bit inside the final block.
    let last_bits = m - (blocks - 1) * WORD_BITS;
    let high_bit: u64 = 1u64 << (last_bits - 1);

    let mut score = m;

    for &c in text {
        // Boundary gap: the bottom row grows by one each column, so the carry
        // entering block 0 is `+1`.
        let mut hp: u64 = 1;
        let mut hn: u64 = 0;

        for b in 0..blocks {
            let eq = peq[usize::from(c) * blocks + b];
            // Fold the incoming horizontal carry into this block's equality
            // mask so a `-1` carry behaves like a match at bit 0.
            let eq = eq | hn;

            let vp_b = vp[b];
            let vn_b = vn[b];

            let xv = eq | vn_b;
            let xh = (((eq & vp_b).wrapping_add(vp_b)) ^ vp_b) | eq;

            let mut ph = vn_b | !(xh | vp_b);
            let mut mh = vp_b & xh;

            // Carry leaving the top of this block (bit 63), to be fed into the
            // next higher block, or — for the last block — into the score.
            let ph_out = (ph >> (WORD_BITS - 1)) & 1;
            let mh_out = (mh >> (WORD_BITS - 1)) & 1;

            if b + 1 == blocks {
                // Score adjustment uses the pattern's genuine high bit, which
                // may sit below bit 63 in a partial final block.
                if ph & high_bit != 0 {
                    score += 1;
                } else if mh & high_bit != 0 {
                    score -= 1;
                }
            }

            // Shift down, re-injecting the carry that entered this block at the
            // freshly vacated low bit.
            ph = (ph << 1) | hp;
            mh = (mh << 1) | hn;

            vp[b] = mh | !(xv | ph);
            vn[b] = ph & xv;

            // Propagate this block's carry-out to the next block.
            hp = ph_out;
            hn = mh_out;
        }
    }

    score
}

/// Compute the global Levenshtein (edit) distance between `pattern` and `text`.
///
/// The returned value is the minimum number of single-character insertions,
/// deletions, and substitutions that transform one byte string into the other.
/// The function is **symmetric** in its arguments; internally the shorter input
/// is the one packed into bit-vectors, which is purely an efficiency choice and
/// does not affect the result.
///
/// Bytes are compared for raw equality, so the inputs are treated as `&[u8]`,
/// *not* as sequences of Unicode scalar values. For ASCII this coincides with
/// the usual character distance.
///
/// Empty inputs are handled directly: an empty pattern costs `text.len()`
/// insertions and an empty text costs `pattern.len()` deletions.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::distance::myers_edit_distance;
///
/// assert_eq!(myers_edit_distance(b"kitten", b"sitting"), 3);
/// assert_eq!(myers_edit_distance(b"", b"abc"), 3);
/// assert_eq!(myers_edit_distance(b"abc", b"abc"), 0);
/// ```
pub fn myers_edit_distance(pattern: &[u8], text: &[u8]) -> usize {
    // Pack the shorter string; the longer one is streamed as the "text".
    let (short, long) = if pattern.len() <= text.len() {
        (pattern, text)
    } else {
        (text, pattern)
    };

    let m = short.len();
    let n = long.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    if m <= WORD_BITS {
        single_word(short, long)
    } else {
        blocked(short, long)
    }
}

/// Fallible wrapper around [`myers_edit_distance`].
///
/// The Levenshtein distance is always well defined — including for empty inputs
/// — so this never actually fails; it exists only for callers that prefer the
/// crate-wide [`SeqResult`] convention. The `Err` arm is reserved for inputs
/// whose combined size would overflow `usize` index arithmetic, which cannot
/// occur for in-memory slices and is therefore unreachable in practice.
pub fn myers_edit_distance_checked(pattern: &[u8], text: &[u8]) -> SeqResult<usize> {
    pattern.len().checked_add(text.len()).ok_or_else(|| {
        SeqError::InvalidConfiguration("combined input length overflows usize".to_string())
    })?;
    Ok(myers_edit_distance(pattern, text))
}

/// Convenience wrapper computing the byte-wise edit distance between two `&str`s.
///
/// The strings are compared by their UTF-8 bytes, **not** by Unicode scalar
/// values, so multi-byte characters contribute according to their encoded
/// length. For ASCII text this is exactly the character-level Levenshtein
/// distance.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::distance::myers_distance_str;
///
/// assert_eq!(myers_distance_str("flaw", "lawn"), 2);
/// ```
pub fn myers_distance_str(a: &str, b: &str) -> usize {
    myers_edit_distance(a.as_bytes(), b.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::metrics::metrics::edit_distance;

    /// Tiny self-contained Wagner–Fischer DP, independent of the crate oracle,
    /// used as a second cross-check.
    fn dp_levenshtein(a: &[u8], b: &[u8]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut prev: Vec<usize> = (0..=n).collect();
        let mut curr = vec![0usize; n + 1];
        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n]
    }

    /// Build a deterministic random byte string over `alphabet` of `len` bytes.
    fn random_bytes(rng: &mut LcgRng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| alphabet[rng.next_usize(alphabet.len())])
            .collect()
    }

    /// (a) CENTRAL ORACLE: Myers must equal the reference DP on random and
    /// hand-picked pairs.
    #[test]
    fn matches_reference_dp_random_and_edge_cases() {
        let explicit: &[(&[u8], &[u8], usize)] = &[
            (b"kitten", b"sitting", 3),
            (b"flaw", b"lawn", 2),
            (b"", b"", 0),
            (b"abc", b"", 3),
            (b"", b"abc", 3),
            (b"book", b"back", 2),
        ];
        for &(a, b, expected) in explicit {
            let got = myers_edit_distance(a, b);
            assert_eq!(got, expected, "explicit pair {a:?} vs {b:?}");
            assert_eq!(got, edit_distance(a, b), "vs crate DP for {a:?} vs {b:?}");
            assert_eq!(got, dp_levenshtein(a, b), "vs local DP for {a:?} vs {b:?}");
        }

        let mut rng = LcgRng::new(42);
        let alphabet = b"ACGT";
        for _ in 0..200 {
            let la = rng.next_usize(40);
            let lb = rng.next_usize(40);
            let a = random_bytes(&mut rng, alphabet, la);
            let b = random_bytes(&mut rng, alphabet, lb);
            let got = myers_edit_distance(&a, &b);
            assert_eq!(got, edit_distance(&a, &b), "crate DP: {a:?} vs {b:?}");
            assert_eq!(got, dp_levenshtein(&a, &b), "local DP: {a:?} vs {b:?}");
            // Symmetry.
            assert_eq!(got, myers_edit_distance(&b, &a), "symmetry: {a:?} vs {b:?}");
        }
    }

    /// (b) Identical strings have distance zero.
    #[test]
    fn identical_strings_are_zero() {
        for s in [b"abc".as_slice(), b"hello world", b"", b"AAAAAAAA"] {
            assert_eq!(myers_edit_distance(s, s), 0);
        }
    }

    /// (c) A single substitution, insertion, or deletion costs exactly one.
    #[test]
    fn single_edits_cost_one() {
        assert_eq!(myers_edit_distance(b"abc", b"abd"), 1, "substitution");
        assert_eq!(myers_edit_distance(b"abc", b"abxc"), 1, "insertion");
        assert_eq!(myers_edit_distance(b"abc", b"ac"), 1, "deletion");
    }

    /// (d) Empty pattern / empty text reduce to the other length.
    #[test]
    fn empty_inputs() {
        assert_eq!(myers_edit_distance(b"", b"hello"), 5);
        assert_eq!(myers_edit_distance(b"hello", b""), 5);
        assert_eq!(myers_edit_distance(b"", b""), 0);
    }

    /// (e) BLOCKED-PATH ORACLE: patterns longer than one word must agree with
    /// the reference DP.
    #[test]
    fn blocked_path_matches_reference_dp() {
        let mut rng = LcgRng::new(7);
        let alphabet = b"ACGT";

        // A 100-byte pattern with exactly five substitutions injected.
        let pattern = random_bytes(&mut rng, alphabet, 100);
        let mut mutated = pattern.clone();
        for pos in [3usize, 17, 41, 68, 95] {
            // Flip to a guaranteed-different symbol of the alphabet.
            let cur = mutated[pos];
            let next = alphabet.iter().copied().find(|&x| x != cur).unwrap_or(b'X');
            mutated[pos] = next;
        }
        let got = myers_edit_distance(&pattern, &mutated);
        assert_eq!(got, 5, "five substitutions on length-100 pattern");
        assert_eq!(got, edit_distance(&pattern, &mutated));
        assert_eq!(got, dp_levenshtein(&pattern, &mutated));

        // A 100-byte pattern with three deletions (text is shorter).
        let mut deleted = pattern.clone();
        for &pos in [80usize, 50, 10].iter() {
            deleted.remove(pos);
        }
        let got_del = myers_edit_distance(&pattern, &deleted);
        assert_eq!(got_del, edit_distance(&pattern, &deleted));
        assert_eq!(got_del, dp_levenshtein(&pattern, &deleted));
        assert_eq!(got_del, 3, "three deletions");

        // Two long, independent random strings (~120 vs ~110).
        for _ in 0..40 {
            let la = 64 + rng.next_usize(64); // 64..=127
            let lb = 64 + rng.next_usize(64);
            let a = random_bytes(&mut rng, alphabet, la);
            let b = random_bytes(&mut rng, alphabet, lb);
            let got = myers_edit_distance(&a, &b);
            assert_eq!(got, edit_distance(&a, &b), "long random crate DP");
            assert_eq!(got, dp_levenshtein(&a, &b), "long random local DP");
            assert_eq!(got, myers_edit_distance(&b, &a), "long random symmetry");
        }

        // Exercise exact word-boundary lengths (63, 64, 65, 128, 129).
        for &len in &[63usize, 64, 65, 127, 128, 129, 192, 200] {
            let a = random_bytes(&mut rng, alphabet, len);
            let mut b = a.clone();
            if !b.is_empty() {
                let cur = b[len / 2];
                b[len / 2] = if cur == b'A' { b'C' } else { b'A' };
            }
            let got = myers_edit_distance(&a, &b);
            assert_eq!(got, edit_distance(&a, &b), "boundary len {len}");
            assert_eq!(got, dp_levenshtein(&a, &b), "boundary len {len} local");
        }
    }

    /// (f) Transpositions cost two under Levenshtein (this is not Damerau).
    #[test]
    fn transposition_costs_two() {
        assert_eq!(myers_edit_distance(b"ab", b"ba"), 2);
        assert_eq!(edit_distance(b"ab".as_slice(), b"ba".as_slice()), 2);
        assert_eq!(myers_edit_distance(b"converse", b"covnerse"), 2);
        assert_eq!(
            myers_edit_distance(b"converse", b"covnerse"),
            edit_distance(b"converse".as_slice(), b"covnerse".as_slice())
        );
    }

    /// (g) Prefix / suffix relationships reduce to the length difference.
    #[test]
    fn prefix_suffix_relationships() {
        assert_eq!(myers_edit_distance(b"hello", b"hello world"), 6);
        assert_eq!(
            myers_edit_distance(b"hello", b"hello world"),
            edit_distance(b"hello".as_slice(), b"hello world".as_slice())
        );

        let base = b"abcdefghij";
        for k in 0..=base.len() {
            let prefix = &base[..k];
            assert_eq!(myers_edit_distance(base, prefix), base.len() - k);
            let suffix = &base[k..];
            assert_eq!(myers_edit_distance(base, suffix), k);
        }
    }

    /// The `&str` and checked wrappers agree with the byte API.
    #[test]
    fn wrappers_agree() {
        assert_eq!(myers_distance_str("kitten", "sitting"), 3);
        assert_eq!(myers_distance_str("flaw", "lawn"), 2);
        assert_eq!(
            myers_edit_distance_checked(b"book", b"back").expect("never fails"),
            2
        );
    }
}
