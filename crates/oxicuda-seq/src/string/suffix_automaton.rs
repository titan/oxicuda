//! Suffix automaton (directed acyclic word graph, DAWG).
//!
//! References:
//! * Anselm Blumer, Janet Blumer, David Haussler, Andrzej Ehrenfeucht, M. T.
//!   Chen & Joel Seiferas, *"The smallest automaton recognizing the subwords of
//!   a text"*, Theoretical Computer Science 40, 1985, pp. 31–55.
//! * Maxime Crochemore, *"Transducers and repetitions"*, TCS 45, 1986 — the
//!   linear online construction with state cloning on split, used here.
//!
//! # What a suffix automaton is
//!
//! The suffix automaton of a string `s` is the smallest deterministic finite
//! automaton (with at most `2|s| − 1` states for `|s| ≥ 2`) whose recognised
//! language is *exactly* the set of substrings of `s`. Every state `v` carries:
//!
//! * `len[v]` — the length of the **longest** string in `v`'s
//!   *endpos-equivalence class* (two substrings are equivalent when they end at
//!   the same set of positions in `s`), and
//! * `link[v]` — the **suffix link** to the state of the longest proper suffix
//!   of `v`'s strings that lies in a *different* endpos class.
//!
//! The strings recognised at state `v` are precisely the suffixes of its
//! longest string whose lengths lie in `(len[link[v]], len[v]]`.
//!
//! # Online construction
//!
//! Characters are appended one at a time. Extending by `c` creates a new state
//! `cur` with `len[cur] = len[last] + 1` and walks the suffix-link chain from
//! `last` adding `c`-transitions to `cur`. The interesting case is **cloning**:
//! when the chain reaches a state `p` that already has a `c`-transition to some
//! `q` with `len[q] != len[p] + 1`, a copy `clone` of `q` is inserted with
//! `len[clone] = len[p] + 1`, redirecting the affected transitions and suffix
//! links. This split is what keeps the automaton minimal; it is exercised
//! directly by the tests (e.g. `"abcbc"`).
//!
//! # Provided queries
//!
//! * [`SuffixAutomaton::contains`] — substring membership.
//! * [`SuffixAutomaton::distinct_substring_count`] — number of distinct
//!   non-empty substrings, `Σ_v (len[v] − len[link[v]])`.
//! * [`SuffixAutomaton::occurrences`] — number of (possibly overlapping)
//!   occurrences of a pattern, via the size of each state's endpos set.
//! * [`longest_common_substring`] — the longest common substring of two strings
//!   by running the automaton of one over the other.
//!
//! Input is treated as raw bytes (`&[u8]`); for ASCII this matches the usual
//! character-level notion of substring.

use crate::error::{SeqError, SeqResult};
use std::collections::BTreeMap;

/// Sentinel suffix link for the initial state (it has no parent).
const NO_LINK: isize = -1;

/// A single state of the suffix automaton.
#[derive(Debug, Clone)]
struct State {
    /// Length of the longest string in this state's endpos class.
    len: usize,
    /// Suffix link, or [`NO_LINK`] for the initial state.
    link: isize,
    /// Outgoing transitions keyed by byte. A `BTreeMap` keeps the structure
    /// compact for sparse alphabets while remaining deterministic to iterate.
    next: BTreeMap<u8, usize>,
    /// `endpos`-set size contribution used for occurrence counting. Original
    /// (non-clone) states are created with `cnt = 1`; clones with `cnt = 0`.
    /// After construction these are summed along suffix links in
    /// reverse-`len` order so that `cnt[v]` becomes `|endpos(v)|`.
    cnt: usize,
    /// `true` if this state was produced by a clone-on-split. Recorded so the
    /// clone path is observable and testable.
    is_clone: bool,
}

impl State {
    fn new(len: usize, link: isize) -> Self {
        Self {
            len,
            link,
            next: BTreeMap::new(),
            cnt: 0,
            is_clone: false,
        }
    }
}

/// A suffix automaton (DAWG) built online from a byte string.
///
/// Construct it with [`SuffixAutomaton::new`]; the build is `O(n log |Σ|)` for
/// the `BTreeMap` transition tables (`O(n |Σ|)` with array tables). The
/// automaton stores the length of the source string so that occurrence counts
/// and the distinct-substring total can be reported directly.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::string::SuffixAutomaton;
///
/// let sam = SuffixAutomaton::new(b"abcbc");
/// assert!(sam.contains(b"bcb"));
/// assert!(!sam.contains(b"acb"));
/// // "abcbc" has 12 distinct non-empty substrings.
/// assert_eq!(sam.distinct_substring_count(), 12);
/// ```
#[derive(Debug, Clone)]
pub struct SuffixAutomaton {
    states: Vec<State>,
    /// Index of the initial state (always 0).
    init: usize,
    /// Index of the state reached after consuming the whole source string.
    last: usize,
    /// Length of the source string in bytes.
    source_len: usize,
    /// Lazily-computed `cnt` finalisation flag for occurrence counting.
    counts_finalised: bool,
}

impl SuffixAutomaton {
    /// Build the suffix automaton of `s`.
    ///
    /// An empty `s` yields an automaton with only the initial state; all
    /// queries then behave consistently (no non-empty substring exists, the
    /// distinct count is zero).
    pub fn new(s: &[u8]) -> Self {
        // Pre-reserve the worst-case 2n state budget.
        let mut states: Vec<State> = Vec::with_capacity(2 * s.len().max(1));
        states.push(State::new(0, NO_LINK)); // initial state

        let mut sam = Self {
            states,
            init: 0,
            last: 0,
            source_len: 0,
            counts_finalised: false,
        };

        for &c in s {
            sam.extend(c);
        }
        sam.source_len = s.len();
        sam
    }

    /// Append a single byte `c`, growing the automaton (Crochemore online step).
    fn extend(&mut self, c: u8) {
        let cur = self.states.len();
        {
            let cur_len = self.states[self.last].len + 1;
            let mut st = State::new(cur_len, NO_LINK);
            st.cnt = 1; // a genuine new prefix state ends at one position
            self.states.push(st);
        }

        // Walk the suffix-link chain from `last`, adding c-transitions to `cur`
        // until we reach the initial state or a state that already has one.
        let mut p: isize = self.last as isize;
        while p != NO_LINK && !self.states[p as usize].next.contains_key(&c) {
            self.states[p as usize].next.insert(c, cur);
            p = self.states[p as usize].link;
        }

        if p == NO_LINK {
            // Reached the initial state without an existing c-edge.
            self.states[cur].link = self.init as isize;
        } else {
            let q = self.states[p as usize].next[&c];
            if self.states[p as usize].len + 1 == self.states[q].len {
                // `q` is the right state already.
                self.states[cur].link = q as isize;
            } else {
                // --- Clone-on-split. ---
                let clone = self.states.len();
                {
                    let q_state = &self.states[q];
                    let mut cl = State::new(self.states[p as usize].len + 1, q_state.link);
                    cl.next = q_state.next.clone();
                    cl.is_clone = true;
                    cl.cnt = 0; // a clone contributes no fresh endpos by itself
                    self.states.push(cl);
                }

                // Redirect the c-edges that pointed at `q` (and lay on the chain
                // continuing from `p`) to the clone.
                while p != NO_LINK {
                    match self.states[p as usize].next.get(&c) {
                        Some(&target) if target == q => {
                            self.states[p as usize].next.insert(c, clone);
                            p = self.states[p as usize].link;
                        }
                        _ => break,
                    }
                }

                self.states[q].link = clone as isize;
                self.states[cur].link = clone as isize;
            }
        }

        self.last = cur;
    }

    /// Finalise the endpos-set sizes (`cnt`) used for occurrence counting.
    ///
    /// Sums `cnt` from longer to shorter states along suffix links. Idempotent:
    /// after the first call the flag short-circuits further work. Implemented as
    /// an internal mutation guarded by `&mut self`; callers reach it through
    /// [`occurrences`](Self::occurrences) which borrows mutably.
    fn finalise_counts(&mut self) {
        if self.counts_finalised {
            return;
        }
        // Counting sort of state indices by `len` (max len = source_len).
        let max_len = self.source_len;
        let mut bucket = vec![0usize; max_len + 2];
        for st in &self.states {
            bucket[st.len] += 1;
        }
        for i in 1..bucket.len() {
            bucket[i] += bucket[i - 1];
        }
        let mut order = vec![0usize; self.states.len()];
        // Stable placement; iterate states in index order, fill from the back.
        for (idx, st) in self.states.iter().enumerate() {
            bucket[st.len] -= 1;
            order[bucket[st.len]] = idx;
        }

        // Process in decreasing `len`, pushing counts up the suffix link.
        for &v in order.iter().rev() {
            let link = self.states[v].link;
            if link != NO_LINK {
                let add = self.states[v].cnt;
                self.states[link as usize].cnt += add;
            }
        }
        self.counts_finalised = true;
    }

    /// Walk the automaton on `pattern`, returning the index of the state reached
    /// after consuming all of it, or `None` if `pattern` is not a substring.
    fn walk(&self, pattern: &[u8]) -> Option<usize> {
        let mut state = self.init;
        for &c in pattern {
            match self.states[state].next.get(&c) {
                Some(&next) => state = next,
                None => return None,
            }
        }
        Some(state)
    }

    /// Return `true` iff `pattern` is a substring of the source string.
    ///
    /// The empty pattern is a substring of every string (including the empty
    /// string), so this returns `true` for an empty `pattern`.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.walk(pattern).is_some()
    }

    /// Number of **distinct** non-empty substrings of the source string.
    ///
    /// Equals `Σ_v (len[v] − len[link[v]])` over all non-initial states `v`,
    /// because each state recognises exactly `len[v] − len[link[v]]` distinct
    /// substrings (the suffixes of its longest string down to, but excluding,
    /// its suffix-link length).
    pub fn distinct_substring_count(&self) -> usize {
        let mut total = 0usize;
        for (idx, st) in self.states.iter().enumerate() {
            if idx == self.init {
                continue;
            }
            let link_len = if st.link == NO_LINK {
                0
            } else {
                self.states[st.link as usize].len
            };
            total += st.len - link_len;
        }
        total
    }

    /// Number of (possibly overlapping) occurrences of `pattern` in the source.
    ///
    /// Returns `0` when `pattern` is not a substring. The empty pattern is
    /// reported as occurring `source_len + 1` times (once at each gap), matching
    /// the usual convention.
    ///
    /// This finalises the endpos counters on first use (and caches them), hence
    /// the `&mut self` receiver.
    pub fn occurrences(&mut self, pattern: &[u8]) -> usize {
        if pattern.is_empty() {
            return self.source_len + 1;
        }
        self.finalise_counts();
        match self.walk(pattern) {
            Some(state) => self.states[state].cnt,
            None => 0,
        }
    }

    /// Number of states in the automaton, including the initial state.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of clone states produced during construction.
    ///
    /// Exposed so tests can confirm that an input which forces a split actually
    /// triggered the clone path.
    pub fn clone_count(&self) -> usize {
        self.states.iter().filter(|s| s.is_clone).count()
    }

    /// Length of the source string the automaton was built from.
    pub fn source_len(&self) -> usize {
        self.source_len
    }
}

/// Longest common substring of `a` and `b`.
///
/// Builds the suffix automaton of `a` and streams `b` through it, tracking the
/// length of the current match and resetting via suffix links on a mismatch.
/// Runs in `O(|a| + |b|)` after the `O(|a|)` construction. Returns the
/// `(start_in_b, length)` of a longest common substring; on ties the match
/// ending earliest in `b` is returned. A length of `0` means the two strings
/// share no character.
///
/// # Errors
///
/// Returns [`SeqError::EmptyInput`] if **either** input is empty, since the
/// longest common substring is then trivially empty and `start` is undefined.
/// Callers that prefer to treat that as length `0` can check emptiness first.
pub fn longest_common_substring(a: &[u8], b: &[u8]) -> SeqResult<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Err(SeqError::EmptyInput);
    }

    let sam = SuffixAutomaton::new(a);

    let mut state = sam.init;
    let mut length = 0usize; // current matched suffix length
    let mut best_len = 0usize;
    let mut best_end = 0usize; // index in `b` one past the best match

    for (i, &c) in b.iter().enumerate() {
        // Try to extend the current match by `c`.
        loop {
            if let Some(&next) = sam.states[state].next.get(&c) {
                state = next;
                length += 1;
                break;
            }
            // Mismatch: follow suffix links until a c-edge exists or we hit the
            // root, shrinking the matched length to that state's `len`.
            if sam.states[state].link == NO_LINK {
                // Back at the initial state; cannot match `c` here.
                length = 0;
                break;
            }
            state = sam.states[state].link as usize;
            length = sam.states[state].len;
        }

        if length > best_len {
            best_len = length;
            best_end = i + 1;
        }
    }

    // Recover the start position in `b`.
    let start = best_end - best_len;
    Ok((start, best_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::collections::HashSet;

    /// Brute-force set of all distinct non-empty substrings of `s`.
    fn brute_distinct(s: &[u8]) -> HashSet<Vec<u8>> {
        let n = s.len();
        let mut set = HashSet::new();
        for i in 0..n {
            for j in (i + 1)..=n {
                set.insert(s[i..j].to_vec());
            }
        }
        set
    }

    /// Brute-force count of (overlapping) occurrences of `pat` in `s`.
    fn brute_occurrences(s: &[u8], pat: &[u8]) -> usize {
        if pat.is_empty() {
            return s.len() + 1;
        }
        if pat.len() > s.len() {
            return 0;
        }
        (0..=(s.len() - pat.len()))
            .filter(|&start| &s[start..start + pat.len()] == pat)
            .count()
    }

    /// Brute-force longest common substring length (DP).
    fn brute_lcs_len(a: &[u8], b: &[u8]) -> usize {
        let (m, n) = (a.len(), b.len());
        if m == 0 || n == 0 {
            return 0;
        }
        let mut prev = vec![0usize; n + 1];
        let mut curr = vec![0usize; n + 1];
        let mut best = 0usize;
        for i in 1..=m {
            for j in 1..=n {
                curr[j] = if a[i - 1] == b[j - 1] {
                    prev[j - 1] + 1
                } else {
                    0
                };
                best = best.max(curr[j]);
            }
            std::mem::swap(&mut prev, &mut curr);
            for v in curr.iter_mut() {
                *v = 0;
            }
        }
        best
    }

    fn random_bytes(rng: &mut LcgRng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| alphabet[rng.next_usize(alphabet.len())])
            .collect()
    }

    /// (a) Distinct-substring count equals brute force, including the
    /// split-forcing "abcbc", "aaaa", and "abracadabra".
    #[test]
    fn distinct_count_matches_brute_force() {
        for s in [
            b"abcbc".as_slice(),
            b"aaaa",
            b"abracadabra",
            b"banana",
            b"mississippi",
            b"a",
            b"ab",
        ] {
            let sam = SuffixAutomaton::new(s);
            let got = sam.distinct_substring_count();
            let expect = brute_distinct(s).len();
            assert_eq!(got, expect, "distinct count for {s:?}");
        }

        // Randomised.
        let mut rng = LcgRng::new(99);
        for &alphabet in &[b"ab".as_slice(), b"abc"] {
            for _ in 0..300 {
                let len = rng.next_usize(18);
                let s = random_bytes(&mut rng, alphabet, len);
                let sam = SuffixAutomaton::new(&s);
                assert_eq!(
                    sam.distinct_substring_count(),
                    brute_distinct(&s).len(),
                    "random distinct count for {s:?}"
                );
            }
        }
    }

    /// (b) Membership: positives AND negatives versus brute force.
    #[test]
    fn membership_positive_and_negative() {
        let s = b"abracadabra";
        let sam = SuffixAutomaton::new(s);
        let all = brute_distinct(s);

        // Every real substring is contained.
        for sub in &all {
            assert!(sam.contains(sub), "should contain {sub:?}");
        }

        // Some non-substrings are rejected.
        for bad in [
            b"xyz".as_slice(),
            b"aa",
            b"brc",
            b"cadabrra",
            b"z",
            b"abrad",
        ] {
            let present = all.contains(bad);
            assert_eq!(sam.contains(bad), present, "membership of {bad:?}");
        }

        // Empty pattern is always contained.
        assert!(sam.contains(b""));

        // Randomised positive/negative probing.
        let mut rng = LcgRng::new(7);
        for _ in 0..200 {
            let len = 1 + rng.next_usize(12);
            let text = random_bytes(&mut rng, b"abc", len);
            let sam = SuffixAutomaton::new(&text);
            let set = brute_distinct(&text);
            // Probe a handful of random short strings.
            for _ in 0..8 {
                let qlen = 1 + rng.next_usize(5);
                let q = random_bytes(&mut rng, b"abcd", qlen); // 'd' forces misses
                assert_eq!(
                    sam.contains(&q),
                    set.contains(&q),
                    "probe {q:?} against {text:?}"
                );
            }
        }
    }

    /// (c) Longest common substring matches the DP oracle.
    #[test]
    fn lcs_matches_dp() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"abcde", b"cdefg"),
            (b"abcbc", b"bcbcd"),
            (b"banana", b"ananas"),
            (b"xxxx", b"yyyy"),
            (b"hello", b"yellow"),
            (b"a", b"a"),
            (b"a", b"b"),
        ];
        for &(a, b) in cases {
            let (start, len) = longest_common_substring(a, b).expect("non-empty");
            let expect = brute_lcs_len(a, b);
            assert_eq!(len, expect, "lcs length for {a:?},{b:?}");
            // The reported slice of `b` is genuinely a substring of `a`.
            let sub = &b[start..start + len];
            let sam = SuffixAutomaton::new(a);
            assert!(sam.contains(sub), "lcs slice {sub:?} must occur in {a:?}");
        }

        // Randomised cross-check.
        let mut rng = LcgRng::new(2024);
        for _ in 0..300 {
            let la = 1 + rng.next_usize(14);
            let lb = 1 + rng.next_usize(14);
            let a = random_bytes(&mut rng, b"abc", la);
            let b = random_bytes(&mut rng, b"abc", lb);
            let (start, len) = longest_common_substring(&a, &b).expect("non-empty");
            assert_eq!(len, brute_lcs_len(&a, &b), "random lcs {a:?},{b:?}");
            let sub = &b[start..start + len];
            let sam = SuffixAutomaton::new(&a);
            assert!(sam.contains(sub), "random lcs slice {sub:?} in {a:?}");
        }
    }

    /// (d) Clone-state correctness: "abcbc" forces a split, and all queries
    /// still agree with brute force.
    #[test]
    fn clone_split_is_exercised() {
        let sam = SuffixAutomaton::new(b"abcbc");
        // The split path must have produced at least one clone.
        assert!(sam.clone_count() >= 1, "abcbc must force a clone");
        // State count stays within the 2n-1 bound.
        assert!(sam.state_count() <= 2 * 5);
        // Distinct count is the canonical 12 for "abcbc".
        assert_eq!(sam.distinct_substring_count(), 12);
        assert_eq!(
            sam.distinct_substring_count(),
            brute_distinct(b"abcbc").len()
        );

        // Another classic splitter.
        let sam2 = SuffixAutomaton::new(b"abcbcba");
        assert!(sam2.clone_count() >= 1);
        assert_eq!(
            sam2.distinct_substring_count(),
            brute_distinct(b"abcbcba").len()
        );
    }

    /// (e) Occurrence counting matches brute force (overlapping occurrences).
    #[test]
    fn occurrence_counting_matches_brute_force() {
        let s = b"abracadabra";
        let mut sam = SuffixAutomaton::new(s);
        for pat in [
            b"a".as_slice(),
            b"ab",
            b"abra",
            b"bra",
            b"ra",
            b"cad",
            b"z",
            b"abrad",
        ] {
            assert_eq!(
                sam.occurrences(pat),
                brute_occurrences(s, pat),
                "occurrences of {pat:?}"
            );
        }
        // Overlapping case: "aa" in "aaaa" occurs 3 times.
        let mut sam_a = SuffixAutomaton::new(b"aaaa");
        assert_eq!(sam_a.occurrences(b"aa"), 3);
        assert_eq!(sam_a.occurrences(b"aa"), brute_occurrences(b"aaaa", b"aa"));
        assert_eq!(sam_a.occurrences(b"aaaa"), 1);

        // Empty pattern occurs source_len + 1 times.
        assert_eq!(sam_a.occurrences(b""), 5);

        // Randomised occurrence cross-check.
        let mut rng = LcgRng::new(555);
        for _ in 0..150 {
            let len = 1 + rng.next_usize(14);
            let text = random_bytes(&mut rng, b"ab", len);
            let mut sam = SuffixAutomaton::new(&text);
            for _ in 0..6 {
                let qlen = 1 + rng.next_usize(4);
                let q = random_bytes(&mut rng, b"abc", qlen);
                assert_eq!(
                    sam.occurrences(&q),
                    brute_occurrences(&text, &q),
                    "occ {q:?} in {text:?}"
                );
            }
        }
    }

    /// (f) Empty-string and single-character edge cases.
    #[test]
    fn edge_cases_empty_and_single() {
        // Empty source: only the initial state, no substrings.
        let sam_empty = SuffixAutomaton::new(b"");
        assert_eq!(sam_empty.state_count(), 1);
        assert_eq!(sam_empty.distinct_substring_count(), 0);
        assert!(sam_empty.contains(b"")); // empty pattern
        assert!(!sam_empty.contains(b"a"));
        assert_eq!(sam_empty.source_len(), 0);

        // Single character.
        let mut sam_one = SuffixAutomaton::new(b"a");
        assert_eq!(sam_one.distinct_substring_count(), 1);
        assert!(sam_one.contains(b"a"));
        assert!(!sam_one.contains(b"b"));
        assert_eq!(sam_one.occurrences(b"a"), 1);
        assert_eq!(sam_one.occurrences(b"b"), 0);

        // Empty inputs to LCS are an error.
        assert!(matches!(
            longest_common_substring(b"", b"abc"),
            Err(SeqError::EmptyInput)
        ));
        assert!(matches!(
            longest_common_substring(b"abc", b""),
            Err(SeqError::EmptyInput)
        ));
    }
}
