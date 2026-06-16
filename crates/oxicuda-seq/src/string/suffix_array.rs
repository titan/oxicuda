//! Suffix array (SA-IS) with Kasai's LCP array and SA binary-search.
//!
//! References:
//! * Ge Nong, Sen Zhang & Wai Hong Chan, *"Linear Suffix Array Construction by
//!   Almost Pure Induced-Sorting"*, Data Compression Conference (DCC), 2009,
//!   pp. 193–202 — the **SA-IS** algorithm implemented here.
//! * Toru Kasai, Gunho Lee, Hiroki Arimura, Setsuo Arikawa & Kunsoo Park,
//!   *"Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its
//!   Applications"*, CPM 2001, LNCS 2089, pp. 181–192 — the **LCP** array.
//!
//! # Suffix array
//!
//! The suffix array of a string `s` of length `n` is the permutation
//! `sa[0..n]` of `0..n` that lists the starting positions of all suffixes of
//! `s` in lexicographic order: `s[sa[0]..] < s[sa[1]..] < … < s[sa[n-1]..]`.
//!
//! SA-IS builds it in `O(n)` time and `O(n)` space by:
//!
//! 1. classifying each position as **S-type** (its suffix is lexicographically
//!    smaller than the next position's suffix) or **L-type** (larger);
//! 2. identifying **LMS** ("left-most S") positions — an S-type position whose
//!    predecessor is L-type — which split the string into LMS-substrings;
//! 3. sorting the LMS-substrings by two **induced-sort** passes (sort L-types
//!    from sorted S/LMS, then S-types from sorted L-types);
//! 4. naming the sorted LMS-substrings and recursing on the shorter name string
//!    if any two share a name, then inducing the final order from the recursive
//!    answer.
//!
//! A unique sentinel smaller than every real byte is appended internally so the
//! last suffix is always the unique smallest; the returned array excludes it.
//!
//! # LCP array (Kasai)
//!
//! `lcp[i]` (for `1 ≤ i < n`) is the length of the longest common prefix of the
//! two adjacent suffixes `s[sa[i-1]..]` and `s[sa[i]..]`; `lcp[0] = 0` by
//! convention. Kasai's algorithm computes all of them in `O(n)` using the rank
//! (inverse SA) array and the observation that the LCP of consecutive *text*
//! positions can only shrink by one as we advance the suffix start.
//!
//! # Pattern search
//!
//! Because suffixes are sorted, every occurrence of a pattern `p` corresponds to
//! a contiguous range of the suffix array (the suffixes that have `p` as a
//! prefix). [`SuffixArray::search`] locates that range with two binary searches
//! and returns the sorted text positions.
//!
//! Inputs are raw bytes (`&[u8]`); the alphabet is the full byte range.

use crate::error::{SeqError, SeqResult};

/// A suffix array together with its source length and (lazily-built) LCP array.
///
/// Construct with [`SuffixArray::new`]. The stored `sa` is a permutation of
/// `0..n` (`n = source length`); [`SuffixArray::lcp`] returns the Kasai LCP
/// array, and [`SuffixArray::search`] performs SA binary search.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::SuffixArray;
///
/// let sa = SuffixArray::new(b"banana").expect("non-empty");
/// // Suffixes of "banana" sorted: a, ana, anana, banana, na, nana.
/// assert_eq!(sa.sa(), &[5, 3, 1, 0, 4, 2]);
/// assert_eq!(sa.search(b"ana"), vec![1, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct SuffixArray {
    /// The suffix array proper: a permutation of `0..source_len`.
    sa: Vec<usize>,
    /// The source bytes (owned so search/LCP need no external slice).
    text: Vec<u8>,
}

impl SuffixArray {
    /// Build the suffix array of `s` using SA-IS in linear time.
    ///
    /// # Errors
    ///
    /// Returns [`SeqError::EmptyInput`] for an empty `s`: the suffix array of
    /// the empty string is empty and is rejected to keep the contract explicit
    /// and consistent with the sibling string modules.
    pub fn new(s: &[u8]) -> SeqResult<Self> {
        if s.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let sa = build_sais(s);
        Ok(Self {
            sa,
            text: s.to_vec(),
        })
    }

    /// Borrow the suffix array (a permutation of `0..n`).
    pub fn sa(&self) -> &[usize] {
        &self.sa
    }

    /// Borrow the source text.
    pub fn text(&self) -> &[u8] {
        &self.text
    }

    /// The rank (inverse suffix) array: `rank[sa[i]] == i`.
    pub fn rank(&self) -> Vec<usize> {
        let n = self.sa.len();
        let mut rank = vec![0usize; n];
        for (i, &p) in self.sa.iter().enumerate() {
            rank[p] = i;
        }
        rank
    }

    /// Compute the Kasai LCP array in `O(n)`.
    ///
    /// The returned vector has length `n`; `lcp[0] = 0` and, for `1 ≤ i < n`,
    /// `lcp[i]` is the length of the longest common prefix of the suffixes
    /// `text[sa[i-1]..]` and `text[sa[i]..]`.
    pub fn lcp(&self) -> Vec<usize> {
        let n = self.sa.len();
        let mut lcp = vec![0usize; n];
        if n == 0 {
            return lcp;
        }
        let rank = self.rank();
        let mut h = 0usize; // running LCP length, shrinks by ≤1 per step
        for i in 0..n {
            if rank[i] == 0 {
                // Suffix i is the smallest; no predecessor to compare with.
                h = 0;
                continue;
            }
            let j = self.sa[rank[i] - 1]; // text position of the predecessor
            while i + h < n && j + h < n && self.text[i + h] == self.text[j + h] {
                h += 1;
            }
            lcp[rank[i]] = h;
            h = h.saturating_sub(1);
        }
        lcp
    }

    /// Number of **distinct** non-empty substrings of the source,
    /// `n(n+1)/2 − Σ lcp`.
    ///
    /// Every substring is a prefix of exactly one suffix; summing suffix lengths
    /// counts all substrings with multiplicity, and the LCP between adjacent
    /// suffixes is precisely the number of duplicate prefixes to subtract.
    pub fn distinct_substring_count(&self) -> usize {
        let n = self.sa.len();
        let total = n * (n + 1) / 2;
        let lcp_sum: usize = self.lcp().iter().sum();
        total - lcp_sum
    }

    /// Find all occurrences of `pattern` via two binary searches on the suffix
    /// array, returning the matching text positions in ascending order.
    ///
    /// Returns an empty vector for an empty pattern or when there is no match.
    /// Overlapping occurrences are all reported.
    pub fn search(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }
        let n = self.sa.len();
        // Lower bound: first suffix that is >= pattern (as a prefix-bounded cmp).
        let lo = self.lower_bound(pattern);
        if lo == n {
            return Vec::new();
        }
        // Upper bound: first suffix that does NOT have `pattern` as a prefix.
        let hi = self.upper_bound(pattern);
        let mut out: Vec<usize> = self.sa[lo..hi].to_vec();
        out.sort_unstable();
        out
    }

    /// First index `i` in `0..=n` such that `text[sa[i]..]` is ≥ `pattern` when
    /// compared up to `pattern.len()` bytes (treating a suffix shorter than the
    /// pattern but matching so far as smaller).
    fn lower_bound(&self, pattern: &[u8]) -> usize {
        let n = self.sa.len();
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.suffix_lt_pattern(self.sa[mid], pattern) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// First index `i` such that `text[sa[i]..]` does NOT start with `pattern`
    /// (i.e. is strictly greater than every string having `pattern` as a
    /// prefix).
    fn upper_bound(&self, pattern: &[u8]) -> usize {
        let n = self.sa.len();
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.suffix_le_pattern_prefix(self.sa[mid], pattern) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `true` if the suffix at `start` is strictly less than `pattern` in the
    /// prefix-bounded order (shorter-but-equal suffix counts as less).
    fn suffix_lt_pattern(&self, start: usize, pattern: &[u8]) -> bool {
        let suf = &self.text[start..];
        let m = pattern.len().min(suf.len());
        for k in 0..m {
            if suf[k] != pattern[k] {
                return suf[k] < pattern[k];
            }
        }
        // Equal on the overlap: the shorter is smaller.
        suf.len() < pattern.len()
    }

    /// `true` while the suffix at `start` still has `pattern` as a prefix *or*
    /// is smaller — used to find the exclusive upper bound of the match range.
    fn suffix_le_pattern_prefix(&self, start: usize, pattern: &[u8]) -> bool {
        let suf = &self.text[start..];
        let m = pattern.len().min(suf.len());
        for k in 0..m {
            if suf[k] != pattern[k] {
                return suf[k] < pattern[k];
            }
        }
        // Suffix has pattern as a prefix (suf.len() >= pattern.len()) → still in
        // range; a suffix shorter than the pattern but equal so far is < pattern.
        true
    }
}

/// SA-IS entry point over raw bytes. Maps bytes into `1..=256` and reserves `0`
/// for the sentinel, then dispatches to the generic recursive core.
fn build_sais(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    // Work string with a 0 sentinel appended; alphabet size is 257 (0..=256).
    let mut work: Vec<usize> = Vec::with_capacity(n + 1);
    for &b in s {
        work.push(b as usize + 1);
    }
    work.push(0); // sentinel, strictly smallest

    let sa_full = sais_core(&work, 257);
    // Drop the sentinel suffix (always first) → suffix array of the original.
    sa_full.into_iter().skip(1).collect()
}

/// Suffix type: `true` for S-type, `false` for L-type.
type SType = bool;

/// The generic SA-IS core over an integer alphabet `0..alphabet`, where the last
/// element of `s` is the unique smallest sentinel.
fn sais_core(s: &[usize], alphabet: usize) -> Vec<usize> {
    let n = s.len();
    let mut sa = vec![usize::MAX; n];
    if n == 1 {
        sa[0] = 0;
        return sa;
    }
    if n == 2 {
        // Two elements; the sentinel s[1] is smallest.
        sa[0] = 1;
        sa[1] = 0;
        return sa;
    }

    // 1. Classify S/L types from the right. The sentinel is S-type.
    let t = classify_types(s);

    // Bucket boundaries.
    let bucket_sizes = bucket_sizes(s, alphabet);

    // 2. Place LMS suffixes into their bucket ends (rough order), then induce.
    let lms_positions: Vec<usize> = (1..n).filter(|&i| is_lms(&t, i)).collect();

    place_lms_at_bucket_ends(&mut sa, s, &bucket_sizes, &lms_positions);
    induce_l(&mut sa, s, &t, &bucket_sizes);
    induce_s(&mut sa, s, &t, &bucket_sizes);

    // 3. Name the sorted LMS substrings.
    let (reduced, names_count, lms_order) = name_lms_substrings(&sa, s, &t);

    // 4. Recurse if names are not all distinct, else order directly.
    let lms_sorted: Vec<usize> = if names_count == lms_order.len() {
        // Already distinct: reduced[k] is the rank; invert to get order.
        let mut sorted = vec![0usize; lms_order.len()];
        for (k, &pos) in lms_order.iter().enumerate() {
            sorted[reduced[k]] = pos;
        }
        sorted
    } else {
        let sub_sa = sais_core(&reduced, names_count + 1);
        // Map reduced suffix order back to original LMS positions.
        sub_sa.into_iter().map(|r| lms_order[r]).collect()
    };

    // 5. Final induced sort using the correct LMS order.
    for slot in sa.iter_mut() {
        *slot = usize::MAX;
    }
    place_lms_sorted_at_bucket_ends(&mut sa, s, &bucket_sizes, &lms_sorted);
    induce_l(&mut sa, s, &t, &bucket_sizes);
    induce_s(&mut sa, s, &t, &bucket_sizes);

    sa
}

/// Classify each position as S-type (`true`) or L-type (`false`).
fn classify_types(s: &[usize]) -> Vec<SType> {
    let n = s.len();
    let mut t = vec![false; n];
    t[n - 1] = true; // sentinel is S-type
    for i in (0..n - 1).rev() {
        t[i] = match s[i].cmp(&s[i + 1]) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => t[i + 1],
        };
    }
    t
}

/// `true` if position `i` is an LMS position (S-type with an L-type predecessor).
fn is_lms(t: &[SType], i: usize) -> bool {
    i > 0 && t[i] && !t[i - 1]
}

/// Count occurrences of each symbol → bucket sizes.
fn bucket_sizes(s: &[usize], alphabet: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; alphabet];
    for &c in s {
        sizes[c] += 1;
    }
    sizes
}

/// Exclusive prefix-sum giving each bucket's start index.
fn bucket_heads(sizes: &[usize]) -> Vec<usize> {
    let mut heads = vec![0usize; sizes.len()];
    let mut sum = 0usize;
    for (i, &sz) in sizes.iter().enumerate() {
        heads[i] = sum;
        sum += sz;
    }
    heads
}

/// Inclusive prefix-sum minus one giving each bucket's last index.
fn bucket_tails(sizes: &[usize]) -> Vec<usize> {
    let mut tails = vec![0usize; sizes.len()];
    let mut sum = 0usize;
    for (i, &sz) in sizes.iter().enumerate() {
        sum += sz;
        tails[i] = sum.wrapping_sub(1);
    }
    tails
}

/// Place LMS suffixes (in text order) at the *ends* of their character buckets.
fn place_lms_at_bucket_ends(
    sa: &mut [usize],
    s: &[usize],
    sizes: &[usize],
    lms_positions: &[usize],
) {
    let mut tails = bucket_tails(sizes);
    for &p in lms_positions {
        let c = s[p];
        sa[tails[c]] = p;
        tails[c] = tails[c].wrapping_sub(1);
    }
}

/// Place already-sorted LMS suffixes at the ends of their buckets (right→left so
/// the sorted order is preserved within each bucket).
fn place_lms_sorted_at_bucket_ends(
    sa: &mut [usize],
    s: &[usize],
    sizes: &[usize],
    lms_sorted: &[usize],
) {
    let mut tails = bucket_tails(sizes);
    for &p in lms_sorted.iter().rev() {
        let c = s[p];
        sa[tails[c]] = p;
        tails[c] = tails[c].wrapping_sub(1);
    }
}

/// Induce L-type suffixes from the current (partial) SA by a left-to-right pass.
fn induce_l(sa: &mut [usize], s: &[usize], t: &[SType], sizes: &[usize]) {
    let n = s.len();
    let mut heads = bucket_heads(sizes);
    for i in 0..n {
        let p = sa[i];
        if p == usize::MAX || p == 0 {
            continue;
        }
        let j = p - 1;
        if !t[j] {
            // L-type → place at the head of its bucket.
            let c = s[j];
            sa[heads[c]] = j;
            heads[c] += 1;
        }
    }
}

/// Induce S-type suffixes from the current SA by a right-to-left pass.
fn induce_s(sa: &mut [usize], s: &[usize], t: &[SType], sizes: &[usize]) {
    let n = s.len();
    let mut tails = bucket_tails(sizes);
    for i in (0..n).rev() {
        let p = sa[i];
        if p == usize::MAX || p == 0 {
            continue;
        }
        let j = p - 1;
        if t[j] {
            // S-type → place at the tail of its bucket.
            let c = s[j];
            sa[tails[c]] = j;
            tails[c] = tails[c].wrapping_sub(1);
        }
    }
}

/// Name the sorted LMS substrings.
///
/// Returns `(reduced, names_count, lms_order)` where `lms_order` lists the LMS
/// text positions in their first-appearance order within the array, `reduced` is
/// the name string over those positions in *text* order, and `names_count` is
/// the number of distinct names assigned.
fn name_lms_substrings(sa: &[usize], s: &[usize], t: &[SType]) -> (Vec<usize>, usize, Vec<usize>) {
    let n = s.len();
    // Collect the LMS positions in the order the induced sort produced them.
    let mut lms_in_sa: Vec<usize> = Vec::new();
    for &p in sa {
        if p != usize::MAX && is_lms(t, p) {
            lms_in_sa.push(p);
        }
    }

    // Assign names by comparing consecutive LMS substrings.
    let mut names = vec![usize::MAX; n];
    let mut current_name = 0usize;
    names[lms_in_sa[0]] = current_name;
    let mut prev = lms_in_sa[0];
    for &cur in lms_in_sa.iter().skip(1) {
        if !lms_substrings_equal(s, t, prev, cur) {
            current_name += 1;
        }
        names[cur] = current_name;
        prev = cur;
    }
    let names_count = current_name + 1;

    // Build the reduced string in text order of LMS positions, and the position
    // list for inverse mapping.
    let mut lms_order: Vec<usize> = Vec::new();
    let mut reduced: Vec<usize> = Vec::new();
    for i in 0..n {
        if is_lms(t, i) {
            lms_order.push(i);
            reduced.push(names[i]);
        }
    }
    (reduced, names_count, lms_order)
}

/// Compare two LMS substrings starting at `a` and `b` for equality, including
/// the type sequence (so that boundaries are respected).
fn lms_substrings_equal(s: &[usize], t: &[SType], a: usize, b: usize) -> bool {
    let n = s.len();
    if a == b {
        return true;
    }
    let mut i = 0usize;
    loop {
        let ai = a + i;
        let bi = b + i;
        // Past the sentinel for either → only equal if both ended together.
        if ai >= n || bi >= n {
            return false;
        }
        let a_is_lms = is_lms(t, ai);
        let b_is_lms = is_lms(t, bi);
        if i > 0 && a_is_lms && b_is_lms {
            // Both reached the next LMS boundary simultaneously → equal so far.
            return true;
        }
        if a_is_lms != b_is_lms {
            return false;
        }
        if s[ai] != s[bi] || t[ai] != t[bi] {
            return false;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force suffix array: sort all suffixes lexicographically.
    fn brute_force_sa(s: &[u8]) -> Vec<usize> {
        let n = s.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
        idx
    }

    /// Brute-force LCP from the (correct) suffix array.
    fn brute_force_lcp(s: &[u8], sa: &[usize]) -> Vec<usize> {
        let n = sa.len();
        let mut lcp = vec![0usize; n];
        for i in 1..n {
            let (a, b) = (sa[i - 1], sa[i]);
            let mut k = 0;
            while a + k < s.len() && b + k < s.len() && s[a + k] == s[b + k] {
                k += 1;
            }
            lcp[i] = k;
        }
        lcp
    }

    /// Brute-force distinct substring count over all substrings.
    fn brute_force_distinct(s: &[u8]) -> usize {
        let n = s.len();
        let mut set = std::collections::BTreeSet::new();
        for i in 0..n {
            for j in i + 1..=n {
                set.insert(s[i..j].to_vec());
            }
        }
        set.len()
    }

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

    /// (a) SA equals brute force on random strings and on the canonical cases.
    #[test]
    fn sa_matches_brute_force() {
        // Canonical strings.
        for s in [b"banana".as_slice(), b"mississippi", b"abracadabra"] {
            let sa = SuffixArray::new(s).expect("non-empty");
            assert_eq!(sa.sa(), brute_force_sa(s).as_slice(), "SA for {s:?}");
        }
        // Random.
        let mut rng = crate::handle::LcgRng::new(11);
        for &alphabet in &[b"a".as_slice(), b"ab", b"abc", b"abcd"] {
            for _ in 0..400 {
                let len = 1 + rng.next_usize(40);
                let s = random_bytes(&mut rng, alphabet, len);
                let got = SuffixArray::new(&s).expect("non-empty");
                assert_eq!(got.sa(), brute_force_sa(&s).as_slice(), "SA for {s:?}");
            }
        }
    }

    /// (b) The SA is a permutation of 0..n.
    #[test]
    fn sa_is_permutation() {
        let mut rng = crate::handle::LcgRng::new(22);
        for _ in 0..300 {
            let len = 1 + rng.next_usize(40);
            let s = random_bytes(&mut rng, b"abc", len);
            let sa = SuffixArray::new(&s).expect("non-empty");
            let n = s.len();
            let mut seen = vec![false; n];
            assert_eq!(sa.sa().len(), n);
            for &p in sa.sa() {
                assert!(p < n, "index out of range");
                assert!(!seen[p], "duplicate index {p}");
                seen[p] = true;
            }
            assert!(seen.iter().all(|&b| b), "not all indices present");
        }
    }

    /// (c) The LCP array matches the adjacent-suffix LCPs (brute force).
    #[test]
    fn lcp_matches_brute_force() {
        for s in [b"banana".as_slice(), b"mississippi", b"aaaa"] {
            let sa = SuffixArray::new(s).expect("non-empty");
            let got = sa.lcp();
            let want = brute_force_lcp(s, sa.sa());
            assert_eq!(got, want, "LCP for {s:?}");
        }
        let mut rng = crate::handle::LcgRng::new(33);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..400 {
                let len = 1 + rng.next_usize(40);
                let s = random_bytes(&mut rng, alphabet, len);
                let sa = SuffixArray::new(&s).expect("non-empty");
                assert_eq!(sa.lcp(), brute_force_lcp(&s, sa.sa()), "LCP {s:?}");
            }
        }
    }

    /// (d) Repeated characters handled.
    #[test]
    fn repeated_characters() {
        let sa = SuffixArray::new(b"aaaa").expect("non-empty");
        // Sorted suffixes: "a" < "aa" < "aaa" < "aaaa" → starts 3,2,1,0.
        assert_eq!(sa.sa(), &[3, 2, 1, 0]);
        // LCP between consecutive: 1, 2, 3.
        assert_eq!(sa.lcp(), vec![0, 1, 2, 3]);

        let sa = SuffixArray::new(b"aaaaaaaa").expect("non-empty");
        assert_eq!(sa.sa(), brute_force_sa(b"aaaaaaaa").as_slice());
    }

    /// (e) Pattern search via SA binary search finds occurrences.
    #[test]
    fn search_matches_naive() {
        // Canonical.
        let sa = SuffixArray::new(b"banana").expect("non-empty");
        assert_eq!(sa.search(b"ana"), vec![1, 3]);
        assert_eq!(sa.search(b"a"), vec![1, 3, 5]);
        assert_eq!(sa.search(b"banana"), vec![0]);
        assert!(sa.search(b"xyz").is_empty());
        assert!(sa.search(b"").is_empty());

        // Random cross-check.
        let mut rng = crate::handle::LcgRng::new(44);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..400 {
                let tlen = 1 + rng.next_usize(40);
                let plen = 1 + rng.next_usize(5);
                let t = random_bytes(&mut rng, alphabet, tlen);
                let p = random_bytes(&mut rng, alphabet, plen);
                let sa = SuffixArray::new(&t).expect("non-empty");
                let mut want = naive_search(&p, &t);
                want.sort_unstable();
                assert_eq!(sa.search(&p), want, "search p={p:?} t={t:?}");
            }
        }
    }

    /// (f) #distinct substrings = n(n+1)/2 − ΣLCP matches brute force.
    #[test]
    fn distinct_substring_count_matches_brute_force() {
        for s in [b"banana".as_slice(), b"mississippi", b"aaaa", b"abcabc"] {
            let sa = SuffixArray::new(s).expect("non-empty");
            assert_eq!(
                sa.distinct_substring_count(),
                brute_force_distinct(s),
                "distinct for {s:?}"
            );
        }
        let mut rng = crate::handle::LcgRng::new(55);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..200 {
                let len = 1 + rng.next_usize(24);
                let s = random_bytes(&mut rng, alphabet, len);
                let sa = SuffixArray::new(&s).expect("non-empty");
                assert_eq!(
                    sa.distinct_substring_count(),
                    brute_force_distinct(&s),
                    "distinct for {s:?}"
                );
            }
        }
    }

    /// Empty input is rejected.
    #[test]
    fn empty_input_errors() {
        assert!(matches!(SuffixArray::new(b""), Err(SeqError::EmptyInput)));
    }

    /// Single character.
    #[test]
    fn single_char() {
        let sa = SuffixArray::new(b"x").expect("non-empty");
        assert_eq!(sa.sa(), &[0]);
        assert_eq!(sa.lcp(), vec![0]);
        assert_eq!(sa.search(b"x"), vec![0]);
        assert!(sa.search(b"y").is_empty());
        assert_eq!(sa.distinct_substring_count(), 1);
    }

    /// The rank array is the exact inverse of the SA.
    #[test]
    fn rank_is_inverse() {
        let mut rng = crate::handle::LcgRng::new(66);
        for _ in 0..200 {
            let len = 1 + rng.next_usize(30);
            let s = random_bytes(&mut rng, b"abc", len);
            let sa = SuffixArray::new(&s).expect("non-empty");
            let rank = sa.rank();
            for (i, &p) in sa.sa().iter().enumerate() {
                assert_eq!(rank[p], i);
            }
        }
    }
}
