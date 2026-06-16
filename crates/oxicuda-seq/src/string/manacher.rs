//! Manacher's algorithm for longest palindromic substring in linear time.
//!
//! Reference: Glenn Manacher, *"A New Linear-Time 'On-Line' Algorithm for
//! Finding the Smallest Initial Palindrome of a String"*, Journal of the ACM
//! 22(3), 1975, pp. 346–351. The modern center/radius formulation used here is
//! the standard textbook restatement.
//!
//! # Idea
//!
//! A palindrome is either *odd*-length (centred on a character) or *even*-length
//! (centred between two characters). To treat both uniformly, the input is
//! transformed by interleaving a separator that occurs nowhere in the data:
//!
//! ```text
//! "abba"  ->  ^#a#b#b#a#$
//! ```
//!
//! Every palindrome of the original string corresponds to an odd-length
//! palindrome of the transformed string centred at one of the `#`/character
//! slots, and vice versa. We compute, for each transformed position `i`, the
//! radius `p[i]` = the largest `r` such that `t[i-r..=i+r]` is a palindrome.
//!
//! The linear-time trick maintains the palindrome with the right-most boundary
//! seen so far, `(center, right)`. For a new position `i` inside that
//! palindrome, the **mirror** position `2*center - i` gives a free lower bound
//! on `p[i]`; only the excess beyond the boundary is verified by explicit
//! character comparisons, and each such comparison advances `right`, so the
//! total work is `O(n)`.
//!
//! The sentinels `^` and `$` (chosen here as two *distinct* synthetic markers
//! that never compare equal to any data byte or to each other) remove the need
//! for explicit bounds checks during expansion.
//!
//! # What is returned
//!
//! [`manacher`] returns a [`Manacher`] bundling the longest palindromic
//! substring together with its location and the full per-center radius array,
//! from which every maximal palindrome — and the total count of palindromic
//! substrings — can be recovered ([`Manacher::palindrome_count`],
//! [`Manacher::longest`]).
//!
//! Input is treated as raw bytes (`&[u8]`); for ASCII this is the usual
//! character-level notion of a palindrome.

use crate::error::{SeqError, SeqResult};

/// Result of running [`manacher`] on a byte string.
///
/// The radius array `radii` is indexed by *transformed* position. Position `i`
/// of the transformed string corresponds either to a real character (when `i`
/// is odd in the `# c # c # …` layout) or to a gap between characters (when `i`
/// is even). `radii[i]` is the palindromic radius in transformed coordinates;
/// the length of the palindrome centred there, measured in *original*
/// characters, is exactly `radii[i]`.
#[derive(Debug, Clone)]
pub struct Manacher {
    /// Start index (inclusive) of a longest palindromic substring in the input.
    pub start: usize,
    /// Length in bytes of a longest palindromic substring (0 for empty input).
    pub length: usize,
    /// Per-center radius array in transformed coordinates. Its length is
    /// `2 * n + 1` where `n = input.len()`. `radii[i]` equals the number of
    /// original characters spanned on each side of center `i`, i.e. the length
    /// of the palindrome centred at `i` in original-character units.
    pub radii: Vec<usize>,
}

impl Manacher {
    /// Borrow the longest palindromic substring out of the original `input`.
    ///
    /// Returns `&input[start..start + length]`. For an empty input this is the
    /// empty slice.
    pub fn longest<'a>(&self, input: &'a [u8]) -> &'a [u8] {
        &input[self.start..self.start + self.length]
    }

    /// Total number of palindromic substrings (each distinct occurrence counted
    /// once), derived from the radius array.
    ///
    /// A palindrome centred at transformed position `i` with radius `r` (in
    /// original-character units) contains `⌈r / 2⌉` palindromic substrings that
    /// are themselves centred at `i` (itself, then peeling two characters at a
    /// time). Summing over all centers counts every palindromic substring of
    /// the original string exactly once, because each palindrome has a unique
    /// center in the transformed string.
    pub fn palindrome_count(&self) -> usize {
        self.radii.iter().map(|&r| r.div_ceil(2)).sum()
    }
}

/// Compute the longest palindromic substring of `input` with Manacher's
/// algorithm in `O(n)` time.
///
/// On ties (several palindromes of maximal length) the **left-most** one is
/// returned, which makes the result deterministic. The returned [`Manacher`]
/// also exposes the full radius array for callers that need every palindrome.
///
/// # Errors
///
/// Returns [`SeqError::EmptyInput`] for an empty `input`: there is no
/// palindromic *substring* of the empty string under the convention that a
/// substring is non-empty.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::manacher;
///
/// let m = manacher(b"babad").expect("non-empty");
/// // Either "bab" or "aba"; this implementation returns the left-most, "bab".
/// assert_eq!(m.longest(b"babad"), b"bab");
/// assert_eq!(m.length, 3);
/// ```
pub fn manacher(input: &[u8]) -> SeqResult<Manacher> {
    if input.is_empty() {
        return Err(SeqError::EmptyInput);
    }

    let n = input.len();

    // Transformed alphabet. We map each data byte to a value in 2..=257 so that
    // the two sentinels (LEFT_GUARD, GAP) and the boundary marker are reserved
    // and never collide with any data character.
    const GAP: u16 = 0; // the interleaved separator '#'
    const LEFT_GUARD: u16 = 1; // synthetic '^' to the left of the whole string

    // Build "^ # c0 # c1 # … # c_{n-1} # $" where the trailing region is
    // implicitly guarded: we store an explicit right guard distinct from GAP.
    const RIGHT_GUARD: u16 = 0xFFFF; // synthetic '$'; distinct from LEFT_GUARD.

    // The center/gap layout has length 2*n + 1 (a gap, then char, gap, char, …,
    // gap). We additionally bracket it with the two distinct guards so that the
    // expansion loop never needs a bounds check.
    let trans_len = 2 * n + 1;
    let mut t: Vec<u16> = Vec::with_capacity(trans_len + 2);
    t.push(LEFT_GUARD);
    for &byte in input {
        t.push(GAP);
        // Offset data bytes by 2 to clear LEFT_GUARD(1) and GAP(0).
        t.push(u16::from(byte) + 2);
    }
    t.push(GAP);
    t.push(RIGHT_GUARD);

    // `p[k]` is the radius (in transformed units) at transformed-center k, where
    // k ranges over the inner positions 1..=trans_len of `t` (excluding the two
    // guards). We index `p` by the *inner* coordinate i in 0..trans_len, mapping
    // to t-coordinate i + 1.
    let mut p = vec![0usize; trans_len];

    let mut center: isize = 0; // current palindrome center, inner coordinate
    let mut right: isize = 0; // one past the right edge, inner coordinate

    for i in 0..trans_len as isize {
        // Mirror of i across `center`.
        if i < right {
            let mirror = 2 * center - i;
            // Free lower bound: min(distance to right edge, mirror radius).
            let bound = (right - i) as usize;
            p[i as usize] = bound.min(p[mirror as usize]);
        }

        // Attempt to expand around i. The +1 offsets translate inner coordinate
        // to t-coordinate; guards stop the loop without explicit bounds checks.
        loop {
            let radius = p[i as usize] as isize;
            let left_pos = i - radius - 1 + 1; // = i - radius  (t-coordinate)
            let right_pos = i + radius + 1 + 1; // = i + radius + 2 (t-coordinate)
            // Bounds are guaranteed by the guards: indices stay in 0..t.len().
            if t[left_pos as usize] == t[right_pos as usize] {
                p[i as usize] += 1;
            } else {
                break;
            }
        }

        // Slide the right-most boundary if we expanded past it.
        if i + p[i as usize] as isize > right {
            center = i;
            right = i + p[i as usize] as isize;
        }
    }

    // Recover the longest palindrome. In this transform, p[i] (transformed
    // radius) equals the palindrome length in *original* characters, and its
    // start in original coordinates is (i - p[i]) / 2.
    let mut best_len = 0usize;
    let mut best_center = 0usize;
    for (i, &radius) in p.iter().enumerate() {
        if radius > best_len {
            best_len = radius;
            best_center = i;
        }
    }
    // `best_center` is an inner coordinate; original start = (best_center - p)/2.
    let start = (best_center - best_len) / 2;

    Ok(Manacher {
        start,
        length: best_len,
        radii: p,
    })
}

/// Convenience wrapper returning the longest palindromic substring of a `&str`
/// as an owned `String`.
///
/// The string is processed by its UTF-8 bytes; for inputs containing multi-byte
/// characters the returned slice is guaranteed to fall on byte boundaries only
/// when those characters are not split by a palindrome edge, so this helper is
/// intended for ASCII / byte-oriented use. The result is re-decoded losslessly
/// because palindrome edges in a byte palindrome of valid UTF-8 input land on
/// character boundaries for ASCII text.
///
/// # Errors
///
/// Propagates [`SeqError::EmptyInput`] for an empty string, and returns
/// [`SeqError::InvalidObservation`] if the recovered byte range is not valid
/// UTF-8 (possible only for non-ASCII inputs whose palindrome splits a
/// multi-byte sequence).
pub fn longest_palindrome_str(input: &str) -> SeqResult<String> {
    let m = manacher(input.as_bytes())?;
    let slice = m.longest(input.as_bytes());
    std::str::from_utf8(slice)
        .map(|s| s.to_owned())
        .map_err(|e| {
            SeqError::InvalidObservation(format!("palindrome split a UTF-8 boundary: {e}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force `O(n²)` expansion oracle: returns, for the input, the length
    /// and start of a left-most longest palindrome, plus the total count of
    /// palindromic substrings.
    fn brute_force(input: &[u8]) -> (usize, usize, usize) {
        let n = input.len();
        let mut best_len = 0usize;
        let mut best_start = 0usize;
        let mut count = 0usize;

        let is_pal = |lo: usize, hi: usize| -> bool {
            // Inclusive [lo, hi].
            let (mut a, mut b) = (lo, hi);
            while a < b {
                if input[a] != input[b] {
                    return false;
                }
                a += 1;
                b -= 1;
            }
            true
        };

        for lo in 0..n {
            for hi in lo..n {
                if is_pal(lo, hi) {
                    count += 1;
                    let len = hi - lo + 1;
                    if len > best_len {
                        best_len = len;
                        best_start = lo;
                    }
                }
            }
        }
        (best_len, best_start, count)
    }

    fn random_bytes(rng: &mut crate::handle::LcgRng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| alphabet[rng.next_usize(alphabet.len())])
            .collect()
    }

    /// (a) "babad" → a length-3 palindrome; left-most is "bab".
    #[test]
    fn babad_returns_length_three() {
        let m = manacher(b"babad").expect("non-empty");
        assert_eq!(m.length, 3);
        let got = m.longest(b"babad");
        assert!(got == b"bab" || got == b"aba", "got {got:?}");
        // Deterministic left-most choice.
        assert_eq!(got, b"bab");
    }

    /// (b) "cbbd" → "bb" (an even palindrome).
    #[test]
    fn cbbd_returns_bb() {
        let m = manacher(b"cbbd").expect("non-empty");
        assert_eq!(m.length, 2);
        assert_eq!(m.longest(b"cbbd"), b"bb");
    }

    /// (c) A whole-string palindrome returns itself.
    #[test]
    fn whole_string_palindrome() {
        for s in [b"racecar".as_slice(), b"abba", b"a", b"level"] {
            let m = manacher(s).expect("non-empty");
            assert_eq!(m.length, s.len());
            assert_eq!(m.longest(s), s);
        }
    }

    /// (d) A single character is its own longest palindrome.
    #[test]
    fn single_char() {
        let m = manacher(b"z").expect("non-empty");
        assert_eq!(m.length, 1);
        assert_eq!(m.longest(b"z"), b"z");
        assert_eq!(m.palindrome_count(), 1);
    }

    /// (e) "aaaa" → "aaaa".
    #[test]
    fn all_same_char() {
        let m = manacher(b"aaaa").expect("non-empty");
        assert_eq!(m.length, 4);
        assert_eq!(m.longest(b"aaaa"), b"aaaa");
        // Palindromic substrings of "aaaa": 4 of len1, 3 of len2, 2 of len3, 1
        // of len4 = 10.
        assert_eq!(m.palindrome_count(), 10);
    }

    /// (f) Both even and odd palindromes are handled, and a no-nontrivial case.
    #[test]
    fn even_and_odd_palindromes() {
        // Odd center.
        let odd = manacher(b"xabax").expect("non-empty");
        assert_eq!(odd.length, 5);
        assert_eq!(odd.longest(b"xabax"), b"xabax");

        // Even center.
        let even = manacher(b"xabbax").expect("non-empty");
        assert_eq!(even.length, 6);
        assert_eq!(even.longest(b"xabbax"), b"xabbax");

        // No palindrome longer than a single character.
        let none = manacher(b"abcde").expect("non-empty");
        assert_eq!(none.length, 1);
        // Only the 5 single characters are palindromes.
        assert_eq!(none.palindrome_count(), 5);
    }

    /// (g) The radius array and longest-length match brute force on random
    /// strings (both even- and odd-friendly small alphabets).
    #[test]
    fn radius_array_matches_brute_force() {
        let mut rng = crate::handle::LcgRng::new(123);

        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..400 {
                let len = 1 + rng.next_usize(24);
                let s = random_bytes(&mut rng, alphabet, len);

                let m = manacher(&s).expect("non-empty");
                let (bf_len, _bf_start, bf_count) = brute_force(&s);

                assert_eq!(m.length, bf_len, "length mismatch for {s:?}");

                // The returned palindrome really is a palindrome of that length.
                let got = m.longest(&s);
                assert_eq!(got.len(), bf_len);
                let rev: Vec<u8> = got.iter().rev().copied().collect();
                assert_eq!(got, rev.as_slice(), "returned slice is a palindrome");

                // Independently recompute every maximal palindrome from `radii`
                // and confirm the implied count matches brute force (h).
                assert_eq!(m.palindrome_count(), bf_count, "count mismatch for {s:?}");
            }
        }
    }

    /// (h) Palindrome count matches brute force on the hand-picked strings.
    #[test]
    fn palindrome_count_hand_checked() {
        let cases: &[(&[u8], usize)] = &[
            (b"abc", 3),  // a, b, c
            (b"aaa", 6),  // a×3, aa×2, aaa×1
            (b"aba", 4),  // a, b, a, aba
            (b"abba", 6), // a, b, b, a, bb, abba
            (b"", 0),     // handled separately below
        ];
        for &(s, expected) in cases {
            if s.is_empty() {
                assert!(matches!(manacher(s), Err(SeqError::EmptyInput)));
                continue;
            }
            let m = manacher(s).expect("non-empty");
            assert_eq!(m.palindrome_count(), expected, "for {s:?}");
            let (_, _, bf) = brute_force(s);
            assert_eq!(m.palindrome_count(), bf, "vs brute force for {s:?}");
        }
    }

    /// Empty input is an error.
    #[test]
    fn empty_input_errors() {
        assert!(matches!(manacher(b""), Err(SeqError::EmptyInput)));
        assert!(matches!(
            longest_palindrome_str(""),
            Err(SeqError::EmptyInput)
        ));
    }

    /// The `&str` wrapper returns the expected owned palindrome.
    #[test]
    fn str_wrapper() {
        assert_eq!(longest_palindrome_str("babad").expect("ok"), "bab");
        assert_eq!(longest_palindrome_str("cbbd").expect("ok"), "bb");
        assert_eq!(longest_palindrome_str("racecar").expect("ok"), "racecar");
    }
}
