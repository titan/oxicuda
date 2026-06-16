//! Gusfield's Z-algorithm: the Z-array in linear time, with exact matching.
//!
//! Reference: Dan Gusfield, *"Algorithms on Strings, Trees, and Sequences:
//! Computer Science and Computational Biology"*, Cambridge University Press,
//! 1997, §1.3–1.5. The Z-array (sometimes "Z-function") and its use for
//! linear-time exact pattern matching via the `P $ T` construction are due to
//! that exposition; the underlying fundamental preprocessing predates it.
//!
//! # The Z-array
//!
//! For a string `s` of length `n`, the **Z-array** `z` is defined by
//!
//! ```text
//! z[0] = 0                                    (by convention; see below)
//! z[i] = length of the longest substring starting at i
//!        that is also a prefix of s,          for 1 ≤ i < n.
//! ```
//!
//! Equivalently `z[i] = max { k : s[0..k] == s[i..i+k] }`. So if
//! `s = "aabxaabxcaabxaabxay"` then `z[4] = 4` because `s[4..8] == "aabx" ==
//! s[0..4]` but `s[8] = 'c' != 'a' = s[4]`.
//!
//! ## The `z[0]` convention
//!
//! Position `0` matches the whole prefix trivially (`s[0..n] == s[0..n]`), so a
//! literal reading would give `z[0] = n`. That value carries no information and
//! complicates the matching application, so — following Gusfield and the common
//! competitive-programming convention — **this implementation sets `z[0] = 0`**.
//! Callers that need the "real" value can use `n` directly. This choice is fixed
//! and tested (`z0_convention_is_zero`).
//!
//! # Linear time via the Z-box
//!
//! The algorithm maintains the interval `[l, r)` (the **Z-box**) that is the
//! right-most match of a prefix discovered so far: `s[l..r] == s[0..r-l]`. For a
//! new index `i`:
//!
//! * if `i < r`, the box already tells us that `s[i..r] == s[i-l..r-l]`, so
//!   `z[i] ≥ min(z[i-l], r-i)` for free; only the part beyond `r` needs explicit
//!   comparisons;
//! * otherwise (`i ≥ r`) we compare from scratch starting at `i`.
//!
//! Every explicit character comparison either fails (ending the current `i`) or
//! advances `r`, and `r` only ever increases, so the total comparison count is
//! `O(n)`.
//!
//! # Exact pattern matching
//!
//! To find every occurrence of a pattern `p` (length `m`) in a text `t`
//! (length `n`), form `s = p ++ [sep] ++ t` where `sep` is a byte occurring in
//! neither `p` nor `t`, and run the Z-algorithm. An occurrence of `p` ends-aligns
//! with each index `i` (in `s`) for which `z[i] ≥ m`; the corresponding text
//! offset is `i − (m + 1)`. Because no Z-value may reach across the separator
//! (it equals nothing in `p`), `z[i]` is capped at `m`, so there are **no false
//! positives** — a property exercised directly by `tests`.
//!
//! When no separator byte is available (`p` and `t` between them use all 256
//! byte values) the construction would be unsafe, so [`z_search`] instead falls
//! back to scanning a *prefix-length array* computed on the concatenation
//! without a separator but bounded explicitly at `m`; this keeps the routine
//! total and panic-free. See [`z_search`].
//!
//! Inputs are raw bytes (`&[u8]`).

use crate::error::{SeqError, SeqResult};

/// Compute the Z-array of `s` in `O(n)` time using the Z-box method.
///
/// The returned vector has length `s.len()`. Index `0` holds `0` by the
/// convention documented at the module level; for `1 ≤ i < n`, entry `i` is the
/// length of the longest common prefix of `s` and the suffix `s[i..]`.
///
/// # Errors
///
/// Returns [`SeqError::EmptyInput`] when `s` is empty: the Z-array of the empty
/// string is itself empty, and rejecting it makes the empty/degenerate contract
/// explicit and consistent with the rest of this module.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::z_array;
///
/// let z = z_array(b"aabxaabxcaabxaabxay").expect("non-empty");
/// assert_eq!(z[0], 0); // by convention
/// assert_eq!(z[4], 4); // "aabx" repeats at offset 4
/// ```
pub fn z_array(s: &[u8]) -> SeqResult<Vec<usize>> {
    if s.is_empty() {
        return Err(SeqError::EmptyInput);
    }

    let n = s.len();
    let mut z = vec![0usize; n];

    // [l, r) is the right-most prefix-match window discovered so far.
    let mut l = 0usize;
    let mut r = 0usize;

    for i in 1..n {
        if i < r {
            // Inside the current Z-box: copy from the mirror, capped at the box.
            let mirror = i - l;
            z[i] = z[mirror].min(r - i);
        }
        // Extend the match explicitly beyond whatever was inherited.
        while i + z[i] < n && s[z[i]] == s[i + z[i]] {
            z[i] += 1;
        }
        // If this match reaches further right than the current box, adopt it.
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    }

    Ok(z)
}

/// Brute-force matching that does not depend on a free separator byte.
///
/// Used by [`z_search`] only when `p` and `t` jointly occupy all 256 byte
/// values so that no separator can be chosen; it scans the text directly,
/// remaining `O(n · m)` in that rare degenerate case while keeping the public
/// routine total and panic-free.
fn naive_match(p: &[u8], t: &[u8]) -> Vec<usize> {
    let m = p.len();
    let n = t.len();
    if m == 0 || m > n {
        return Vec::new();
    }
    let mut out = Vec::new();
    for start in 0..=(n - m) {
        if &t[start..start + m] == p {
            out.push(start);
        }
    }
    out
}

/// Pick a byte value that occurs in neither `p` nor `t`, if one exists.
fn free_separator(p: &[u8], t: &[u8]) -> Option<u8> {
    let mut used = [false; 256];
    for &b in p {
        used[b as usize] = true;
    }
    for &b in t {
        used[b as usize] = true;
    }
    (0u16..256).find(|&v| !used[v as usize]).map(|v| v as u8)
}

/// Find every occurrence of `pattern` in `text` using the Z-algorithm on the
/// `pattern ++ sep ++ text` construction, returning start offsets in ascending
/// order.
///
/// Each returned index `j` satisfies `text[j..j + pattern.len()] == pattern`.
/// Overlapping occurrences are all reported (e.g. searching `"aa"` in `"aaaa"`
/// yields `[0, 1, 2]`).
///
/// # Errors
///
/// Returns [`SeqError::EmptyInput`] if `pattern` is empty: an empty pattern has
/// no well-defined occurrence set here (it would "match" at every position),
/// matching the convention used by the Aho–Corasick automaton in this crate.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::z_search;
///
/// assert_eq!(z_search(b"ab", b"abcabxabc").expect("ok"), vec![0, 3, 6]);
/// assert_eq!(z_search(b"aa", b"aaaa").expect("ok"), vec![0, 1, 2]);
/// assert!(z_search(b"z", b"abc").expect("ok").is_empty());
/// ```
pub fn z_search(pattern: &[u8], text: &[u8]) -> SeqResult<Vec<usize>> {
    if pattern.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let m = pattern.len();
    if m > text.len() {
        return Ok(Vec::new());
    }

    // Degenerate alphabet: no separator available → direct scan.
    let sep = match free_separator(pattern, text) {
        Some(sep) => sep,
        None => return Ok(naive_match(pattern, text)),
    };

    // s = pattern ++ sep ++ text.
    let mut s = Vec::with_capacity(m + 1 + text.len());
    s.extend_from_slice(pattern);
    s.push(sep);
    s.extend_from_slice(text);

    let z = z_array(&s)?;

    // Occurrences end-align where z reaches the full pattern length. The
    // separator guarantees z[i] ≤ m, so equality with m is the match test and
    // no value can leak across the boundary (no false positives).
    let mut out = Vec::new();
    let offset = m + 1; // start of the text within s
    for i in offset..s.len() {
        if z[i] >= m {
            out.push(i - offset);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force `O(n²)` Z-array oracle straight from the definition.
    fn brute_force_z(s: &[u8]) -> Vec<usize> {
        let n = s.len();
        let mut z = vec![0usize; n];
        for i in 1..n {
            let mut k = 0;
            while i + k < n && s[k] == s[i + k] {
                k += 1;
            }
            z[i] = k;
        }
        z
    }

    /// Naive all-occurrences search oracle.
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

    /// (a) The Z-array matches brute force on many random strings.
    #[test]
    fn z_array_matches_brute_force() {
        let mut rng = crate::handle::LcgRng::new(0xED);
        for &alphabet in &[b"ab".as_slice(), b"abc", b"abcd"] {
            for _ in 0..500 {
                let len = 1 + rng.next_usize(40);
                let s = random_bytes(&mut rng, alphabet, len);
                let fast = z_array(&s).expect("non-empty");
                let slow = brute_force_z(&s);
                assert_eq!(fast, slow, "Z-array mismatch for {s:?}");
            }
        }
    }

    /// (b) Pattern matching via the P$T construction finds ALL occurrences.
    #[test]
    fn search_finds_all_occurrences() {
        let mut rng = crate::handle::LcgRng::new(7);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..500 {
                let tlen = 1 + rng.next_usize(40);
                let plen = 1 + rng.next_usize(6);
                let t = random_bytes(&mut rng, alphabet, tlen);
                let p = random_bytes(&mut rng, alphabet, plen);
                let got = z_search(&p, &t).expect("non-empty pattern");
                let want = naive_search(&p, &t);
                assert_eq!(got, want, "search mismatch for p={p:?} t={t:?}");
            }
        }
    }

    /// (c) Periodic strings give the expected Z structure.
    #[test]
    fn periodic_strings_have_expected_z() {
        // "aaaa": z[i] = n - i for i >= 1 (every suffix is a prefix run of 'a').
        let z = z_array(b"aaaa").expect("non-empty");
        assert_eq!(z, vec![0, 3, 2, 1]);

        // "abcabc": period 3. z[3] = 3 (whole "abc" repeats), z[1]=z[2]=0,
        // z[4]=z[5]=0 because the tail "bc"/"c" do not start with 'a'.
        let z = z_array(b"abcabc").expect("non-empty");
        assert_eq!(z, vec![0, 0, 0, 3, 0, 0]);

        // "ababab": period 2. z[2]=4 ("abab"), z[4]=2 ("ab"), odd positions 0.
        let z = z_array(b"ababab").expect("non-empty");
        assert_eq!(z, vec![0, 0, 4, 0, 2, 0]);
    }

    /// (d) The z[0] convention is documented and consistently zero.
    #[test]
    fn z0_convention_is_zero() {
        for s in [
            b"a".as_slice(),
            b"abc",
            b"aaaa",
            b"abacabad",
            b"mississippi",
        ] {
            let z = z_array(s).expect("non-empty");
            assert_eq!(z[0], 0, "z[0] must be 0 by convention for {s:?}");
        }
    }

    /// (e) Single char and empty input behave per contract.
    #[test]
    fn single_char_and_empty() {
        let z = z_array(b"x").expect("non-empty");
        assert_eq!(z, vec![0]);

        assert!(matches!(z_array(b""), Err(SeqError::EmptyInput)));
        assert!(matches!(z_search(b"", b"abc"), Err(SeqError::EmptyInput)));

        // Pattern longer than text → no matches, but not an error.
        assert!(z_search(b"abcd", b"ab").expect("ok").is_empty());
    }

    /// (f) No false positives can leak across the separator.
    ///
    /// We construct a case where a naive concatenation without a separator would
    /// spuriously match: pattern "ab", text "...a" then "b..." straddling. The
    /// separator construction must report exactly the genuine in-text matches.
    #[test]
    fn no_false_positives_across_separator() {
        // Pattern equals the boundary of pattern+text if no separator were used.
        // p = "ba", t = "xb" → with naive p++t = "baxb"; the suffix overlap must
        // not be reported. Genuine occurrences of "ba" in "xb" = none.
        assert!(z_search(b"ba", b"xb").expect("ok").is_empty());

        // p = "ab", text ends in 'a', next pattern char 'b' would join across.
        // t = "zzza" has no "ab"; ensure empty.
        assert!(z_search(b"ab", b"zzza").expect("ok").is_empty());

        // Positive control with the same alphabet so the test really exercises
        // matching, not just rejection.
        assert_eq!(z_search(b"ab", b"zzzab").expect("ok"), vec![3]);
    }

    /// Whole-string and prefix patterns at the boundaries.
    #[test]
    fn boundary_patterns() {
        // Pattern == text.
        assert_eq!(z_search(b"abc", b"abc").expect("ok"), vec![0]);
        // Pattern at the very end.
        assert_eq!(z_search(b"c", b"aabc").expect("ok"), vec![3]);
        // Pattern at the very start.
        assert_eq!(z_search(b"aa", b"aab").expect("ok"), vec![0]);
    }

    /// Degenerate full-alphabet fallback path: every byte value used, so no
    /// separator exists and the naive matcher must still be correct.
    #[test]
    fn full_alphabet_fallback() {
        // Text uses all 256 byte values once.
        let text: Vec<u8> = (0u16..256).map(|v| v as u8).collect();
        // Pattern present in the middle.
        let pattern = &text[100..104];
        assert!(free_separator(pattern, &text).is_none());
        let got = z_search(pattern, &text).expect("ok");
        assert_eq!(got, vec![100]);

        // Absent pattern over the full alphabet (reversed pair not contiguous).
        let absent = [200u8, 5u8, 200u8];
        let got = z_search(&absent, &text).expect("ok");
        assert!(got.is_empty());
    }
}
