//! Aho–Corasick multi-pattern string matching.
//!
//! Reference: Alfred V. Aho & Margaret J. Corasick, *"Efficient string
//! matching: an aid to bibliographic search"*, Communications of the ACM
//! 18(6), 1975, pp. 333–340.
//!
//! # Idea
//!
//! Aho–Corasick generalises the Knuth–Morris–Pratt single-pattern automaton to
//! a *set* of patterns. It first builds the **goto** trie of all patterns, then
//! augments it with two functions computed by a breadth-first sweep:
//!
//! * the **failure (suffix) link** `fail[v]` points to the node spelling the
//!   longest proper suffix of `v`'s string that is itself a trie node — exactly
//!   the state to fall back to when the current character cannot extend the
//!   match, and
//! * the **output function** which, for every node, lists the pattern ids whose
//!   strings end at that node, *including* those reachable by following the
//!   chain of failure links (the **dictionary-suffix** links).
//!
//! Scanning a text of length `n` then visits one automaton state per input
//! character and emits every occurrence of every pattern. The whole scan runs
//! in `O(n + z)` time where `z` is the number of reported matches, after an
//! `O(Σ |pᵢ|)` construction.
//!
//! # Output convention
//!
//! [`AhoCorasick::find_iter`] reports each occurrence as a [`Match`] carrying
//! the matched `pattern_id`, the **end** index `end` (one past the last matched
//! byte, i.e. the standard exclusive bound), and the derived `start` index.
//! Because the dictionary-suffix links are followed, **overlapping** matches are
//! all reported — if both `he` and `she` end at the same text position, both
//! appear. Matches are emitted in increasing order of `end`; ties (several
//! patterns ending at the same position) are emitted in ascending `pattern_id`.
//!
//! Patterns are matched over raw bytes (`&[u8]`); for ASCII this coincides with
//! character matching. A pattern may be added more than once and an empty
//! pattern is rejected at construction time (it would match at every position
//! and has no well-defined occurrence semantics here).

use crate::error::{SeqError, SeqResult};
use std::collections::VecDeque;

/// Sentinel for "no node": the root can never be a failure/goto target via this
/// value, and using `usize::MAX` makes an accidental dereference panic loudly in
/// debug builds rather than silently aliasing the root.
const NONE: usize = usize::MAX;

/// One occurrence reported by the automaton.
///
/// The matched substring is `text[start..end]`; `end` is exclusive and equals
/// `start + len` where `len` is the length of pattern `pattern_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// Index of the matched pattern in the original `patterns` slice.
    pub pattern_id: usize,
    /// Start byte offset of the occurrence within the scanned text.
    pub start: usize,
    /// End byte offset (exclusive) of the occurrence within the scanned text.
    pub end: usize,
}

/// A single trie node of the Aho–Corasick automaton.
#[derive(Debug, Clone)]
struct Node {
    /// Child transitions of the *goto* trie, indexed by byte. `NONE` marks the
    /// absence of an explicit trie edge (filled lazily only when querying).
    next: [usize; 256],
    /// Failure link: the longest proper suffix of this node's string that is a
    /// trie node. The root's failure link is itself.
    fail: usize,
    /// Pattern ids whose string ends exactly at this node (not following
    /// failure links). Stored sorted and deduplicated.
    outputs: Vec<usize>,
    /// Head of the dictionary-suffix chain: the nearest strict ancestor *via
    /// failure links* that is itself the end of some pattern, or `NONE`.
    dict_link: usize,
}

impl Node {
    fn new() -> Self {
        Self {
            next: [NONE; 256],
            fail: 0,
            outputs: Vec::new(),
            dict_link: NONE,
        }
    }
}

/// A compiled Aho–Corasick automaton over a fixed set of byte patterns.
///
/// Build it once with [`AhoCorasick::new`] and reuse it for many texts. The
/// automaton stores, for each pattern, its length, so that occurrences can be
/// reported with both `start` and `end` offsets.
///
/// # Examples
///
/// ```
/// use oxicuda_seq::matching::AhoCorasick;
///
/// let ac = AhoCorasick::new(&["he", "she", "his", "hers"]).expect("non-empty");
/// let hits: Vec<_> = ac
///     .find_iter(b"ushers")
///     .iter()
///     .map(|m| (m.pattern_id, m.start, m.end))
///     .collect();
/// // "she" ends at 3, "he" ends at 3, "hers" ends at 6.
/// assert!(hits.contains(&(1, 1, 4))); // she
/// assert!(hits.contains(&(0, 2, 4))); // he
/// assert!(hits.contains(&(3, 2, 6))); // hers
/// ```
#[derive(Debug, Clone)]
pub struct AhoCorasick {
    nodes: Vec<Node>,
    /// Length in bytes of each pattern, indexed by `pattern_id`.
    pattern_lens: Vec<usize>,
}

impl AhoCorasick {
    /// Build the automaton from a slice of patterns.
    ///
    /// Patterns may be anything convertible to bytes (`&str`, `String`,
    /// `&[u8]`, …) via [`AsRef<[u8]>`]. Duplicate patterns are permitted and
    /// keep distinct ids. An empty pattern slice yields an automaton that never
    /// matches; an *empty individual pattern* is rejected with
    /// [`SeqError::EmptyInput`].
    pub fn new<P: AsRef<[u8]>>(patterns: &[P]) -> SeqResult<Self> {
        let mut nodes = vec![Node::new()];
        let mut pattern_lens = Vec::with_capacity(patterns.len());

        // --- Phase 1: build the goto trie. ---
        for (pattern_id, pattern) in patterns.iter().enumerate() {
            let bytes = pattern.as_ref();
            if bytes.is_empty() {
                return Err(SeqError::EmptyInput);
            }
            pattern_lens.push(bytes.len());

            let mut state = 0usize;
            for &byte in bytes {
                let idx = usize::from(byte);
                let next = nodes[state].next[idx];
                state = if next == NONE {
                    let new_state = nodes.len();
                    nodes.push(Node::new());
                    nodes[state].next[idx] = new_state;
                    new_state
                } else {
                    next
                };
            }
            nodes[state].outputs.push(pattern_id);
        }

        // A pattern can be supplied twice; keep each node's output list tidy.
        for node in &mut nodes {
            node.outputs.sort_unstable();
            node.outputs.dedup();
        }

        let mut automaton = Self {
            nodes,
            pattern_lens,
        };
        automaton.build_failure_links();
        Ok(automaton)
    }

    /// Compute failure links and dictionary-suffix links by BFS over the trie.
    ///
    /// Depth-1 nodes fail to the root. For a node `v` reached from `u` on byte
    /// `c`, its failure target is `goto(fail[u], c)`, computed using the
    /// already-finalised links of the (shallower) BFS frontier. The
    /// dictionary-suffix link of `v` is `fail[v]` if that node is itself an
    /// output, else the dictionary-suffix link of `fail[v]` — a classic
    /// path-compressed chain so that reporting all suffix-matches at a node is
    /// `O(#matches)` rather than `O(depth)`.
    fn build_failure_links(&mut self) {
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Root's children fail to the root itself.
        self.nodes[0].fail = 0;
        for c in 0..256usize {
            let child = self.nodes[0].next[c];
            if child != NONE {
                self.nodes[child].fail = 0;
                queue.push_back(child);
            }
        }

        while let Some(u) = queue.pop_front() {
            // Snapshot the failure target of `u`; it is already final because
            // `u` was dequeued, hence shallower than its children.
            let u_fail = self.nodes[u].fail;

            for c in 0..256usize {
                let child = self.nodes[u].next[c];
                if child == NONE {
                    continue;
                }

                // Failure link of `child`: walk failure links from `u`'s
                // failure node until a goto edge on `c` exists, defaulting to
                // the root.
                let mut f = u_fail;
                loop {
                    let edge = self.nodes[f].next[c];
                    if edge != NONE && edge != child {
                        self.nodes[child].fail = edge;
                        break;
                    }
                    if f == 0 {
                        self.nodes[child].fail = 0;
                        break;
                    }
                    f = self.nodes[f].fail;
                }

                // Dictionary-suffix link: nearest failure-ancestor that ends a
                // pattern, with path compression.
                let cf = self.nodes[child].fail;
                self.nodes[child].dict_link = if !self.nodes[cf].outputs.is_empty() {
                    cf
                } else {
                    self.nodes[cf].dict_link
                };

                queue.push_back(child);
            }
        }
    }

    /// Follow the goto function of the *automaton* (not merely the trie):
    /// from `state` on byte `c`, take the explicit edge if present, otherwise
    /// fall back along failure links until an edge exists or the root is
    /// reached. Never returns `NONE`.
    fn goto(&self, mut state: usize, c: usize) -> usize {
        loop {
            let edge = self.nodes[state].next[c];
            if edge != NONE {
                return edge;
            }
            if state == 0 {
                return 0;
            }
            state = self.nodes[state].fail;
        }
    }

    /// Scan `text`, invoking `report` for every occurrence of every pattern.
    ///
    /// This is the streaming core used by [`find_iter`](Self::find_iter): it
    /// allocates nothing and lets the caller decide what to do with each
    /// [`Match`]. The closure is called once per occurrence; at a given text
    /// position the node's own outputs are reported before those reached via
    /// the dictionary-suffix chain.
    pub fn for_each_match<F: FnMut(Match)>(&self, text: &[u8], mut report: F) {
        let mut state = 0usize;
        for (pos, &byte) in text.iter().enumerate() {
            state = self.goto(state, usize::from(byte));

            // Emit outputs of the current node and every dictionary-suffix
            // ancestor. `end` is one past the current byte.
            let end = pos + 1;
            let mut node = state;
            while node != NONE {
                for &pattern_id in &self.nodes[node].outputs {
                    let len = self.pattern_lens[pattern_id];
                    report(Match {
                        pattern_id,
                        start: end - len,
                        end,
                    });
                }
                node = self.nodes[node].dict_link;
            }
        }
    }

    /// Collect every occurrence of every pattern in `text`.
    ///
    /// The returned vector is sorted by `(end, pattern_id)`: occurrences ending
    /// earlier come first, and several patterns ending at the same position are
    /// ordered by ascending id. Overlapping matches are all included.
    pub fn find_iter(&self, text: &[u8]) -> Vec<Match> {
        // Group by end position so that, for a fixed `end`, the node-local
        // ordering (own outputs, then dictionary-suffix links) is replaced by a
        // deterministic ascending-`pattern_id` order independent of trie shape.
        let mut matches: Vec<Match> = Vec::new();
        self.for_each_match(text, |m| matches.push(m));
        matches.sort_unstable_by(|a, b| {
            a.end
                .cmp(&b.end)
                .then_with(|| a.pattern_id.cmp(&b.pattern_id))
        });
        matches
    }

    /// Return `true` if *any* pattern occurs in `text`.
    ///
    /// Short-circuits at the first match, so it is cheaper than materialising
    /// [`find_iter`](Self::find_iter) when only presence is needed.
    pub fn is_match(&self, text: &[u8]) -> bool {
        let mut state = 0usize;
        for &byte in text {
            state = self.goto(state, usize::from(byte));
            if !self.nodes[state].outputs.is_empty() || self.nodes[state].dict_link != NONE {
                return true;
            }
        }
        false
    }

    /// Number of patterns compiled into the automaton.
    pub fn pattern_count(&self) -> usize {
        self.pattern_lens.len()
    }

    /// Number of states (trie nodes) in the automaton, including the root.
    pub fn state_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Naive `O(n · Σ|pᵢ|)` multi-substring search used as the cross-check
    /// oracle: for each pattern, slide it over every text position.
    fn naive_matches(patterns: &[&[u8]], text: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        for (pattern_id, pat) in patterns.iter().enumerate() {
            if pat.is_empty() {
                continue;
            }
            if pat.len() > text.len() {
                continue;
            }
            for start in 0..=(text.len() - pat.len()) {
                if &text[start..start + pat.len()] == *pat {
                    out.push(Match {
                        pattern_id,
                        start,
                        end: start + pat.len(),
                    });
                }
            }
        }
        out.sort_unstable_by(|a, b| {
            a.end
                .cmp(&b.end)
                .then_with(|| a.pattern_id.cmp(&b.pattern_id))
        });
        out
    }

    fn random_bytes(rng: &mut LcgRng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| alphabet[rng.next_usize(alphabet.len())])
            .collect()
    }

    /// (a) The textbook {he, she, his, hers} over "ushers".
    ///
    /// Verifies both the exact positions *and* which pattern fired:
    /// `she` ends at 3, `he` ends at 3 (overlapping `she`), `hers` ends at 6.
    #[test]
    fn classic_he_she_his_hers() {
        let patterns = ["he", "she", "his", "hers"];
        let ac = AhoCorasick::new(&patterns).expect("non-empty");
        let hits = ac.find_iter(b"ushers");

        // Expected occurrences (pattern_id, start, end), 0=he 1=she 2=his 3=hers.
        // `find_iter` sorts by (end, pattern_id); both `he` and `she` end at 4,
        // so `he` (id 0) precedes `she` (id 1) despite `she` starting earlier.
        let expected = vec![
            Match {
                pattern_id: 0,
                start: 2,
                end: 4,
            }, // he
            Match {
                pattern_id: 1,
                start: 1,
                end: 4,
            }, // she
            Match {
                pattern_id: 3,
                start: 2,
                end: 6,
            }, // hers
        ];
        assert_eq!(hits, expected);

        // "his" must NOT appear in "ushers".
        assert!(hits.iter().all(|m| m.pattern_id != 2), "his must be absent");

        // Cross-check against the naive oracle.
        let pat_bytes: Vec<&[u8]> = patterns.iter().map(|p| p.as_bytes()).collect();
        assert_eq!(hits, naive_matches(&pat_bytes, b"ushers"));
    }

    /// (b) Overlapping matches are *all* reported, not just the leftmost-longest.
    #[test]
    fn overlapping_matches_all_reported() {
        // "aa" and "aaa" over "aaaaa": every starting position of each.
        let patterns = ["aa", "aaa"];
        let ac = AhoCorasick::new(&patterns).expect("non-empty");
        let hits = ac.find_iter(b"aaaaa");

        let pat_bytes: Vec<&[u8]> = patterns.iter().map(|p| p.as_bytes()).collect();
        let oracle = naive_matches(&pat_bytes, b"aaaaa");
        assert_eq!(hits, oracle);

        // "aa" occurs at starts 0,1,2,3 (4 times); "aaa" at starts 0,1,2 (3).
        let aa = hits.iter().filter(|m| m.pattern_id == 0).count();
        let aaa = hits.iter().filter(|m| m.pattern_id == 1).count();
        assert_eq!(aa, 4, "every aa occurrence");
        assert_eq!(aaa, 3, "every aaa occurrence");
    }

    /// (c) A pattern absent from the text yields no matches for it.
    #[test]
    fn absent_pattern_no_matches() {
        let ac = AhoCorasick::new(&["xyz", "qqq"]).expect("non-empty");
        let hits = ac.find_iter(b"the quick brown fox");
        assert!(hits.is_empty(), "no pattern occurs");
        assert!(!ac.is_match(b"the quick brown fox"));

        // One present, one absent.
        let ac2 = AhoCorasick::new(&["fox", "zzz"]).expect("non-empty");
        let hits2 = ac2.find_iter(b"the quick brown fox");
        assert_eq!(hits2.len(), 1);
        assert_eq!(hits2[0].pattern_id, 0);
        assert!(hits2.iter().all(|m| m.pattern_id != 1));
    }

    /// (d) Single-character patterns.
    #[test]
    fn single_character_patterns() {
        let patterns = ["a", "b", "c"];
        let ac = AhoCorasick::new(&patterns).expect("non-empty");
        let hits = ac.find_iter(b"abcabc");

        let pat_bytes: Vec<&[u8]> = patterns.iter().map(|p| p.as_bytes()).collect();
        assert_eq!(hits, naive_matches(&pat_bytes, b"abcabc"));
        // Each of a/b/c appears twice.
        for id in 0..3 {
            assert_eq!(hits.iter().filter(|m| m.pattern_id == id).count(), 2);
        }
    }

    /// (e) A pattern that is a suffix of another forces a dictionary-suffix
    /// link; both must be reported where the longer one ends.
    #[test]
    fn dictionary_suffix_link_both_reported() {
        // "ers" is a suffix of "hers". Scanning "hers", at end=4 both "hers"
        // and "ers" complete and both must fire.
        let patterns = ["hers", "ers"];
        let ac = AhoCorasick::new(&patterns).expect("non-empty");
        let hits = ac.find_iter(b"hers");

        assert!(
            hits.iter()
                .any(|m| m.pattern_id == 0 && m.start == 0 && m.end == 4),
            "hers reported"
        );
        assert!(
            hits.iter()
                .any(|m| m.pattern_id == 1 && m.start == 1 && m.end == 4),
            "ers reported via dictionary-suffix link"
        );

        let pat_bytes: Vec<&[u8]> = patterns.iter().map(|p| p.as_bytes()).collect();
        assert_eq!(hits, naive_matches(&pat_bytes, b"hers"));

        // A deeper chain: "c" ⊂ "bc" ⊂ "abc" all end at the same position.
        let chain = ["abc", "bc", "c"];
        let ac2 = AhoCorasick::new(&chain).expect("non-empty");
        let hits2 = ac2.find_iter(b"abc");
        assert_eq!(hits2.len(), 3, "three nested suffixes all reported");
        let chain_bytes: Vec<&[u8]> = chain.iter().map(|p| p.as_bytes()).collect();
        assert_eq!(hits2, naive_matches(&chain_bytes, b"abc"));
    }

    /// (f) Repeated occurrences of the same pattern are all found.
    #[test]
    fn repeated_occurrences_all_found() {
        let ac = AhoCorasick::new(&["ab"]).expect("non-empty");
        let hits = ac.find_iter(b"ababab");
        assert_eq!(hits.len(), 3);
        let starts: Vec<usize> = hits.iter().map(|m| m.start).collect();
        assert_eq!(starts, vec![0, 2, 4]);

        // Same pattern supplied twice keeps both ids and both fire.
        let ac_dup = AhoCorasick::new(&["xy", "xy"]).expect("non-empty");
        let dup_hits = ac_dup.find_iter(b"xyxy");
        // 2 positions × 2 ids = 4 matches.
        assert_eq!(dup_hits.len(), 4);
        assert_eq!(dup_hits.iter().filter(|m| m.pattern_id == 0).count(), 2);
        assert_eq!(dup_hits.iter().filter(|m| m.pattern_id == 1).count(), 2);
    }

    /// (g) Randomised cross-check against the naive multi-substring search.
    #[test]
    fn random_cross_check_against_naive() {
        let mut rng = LcgRng::new(0xACDC);
        let alphabet = b"abc";

        for _ in 0..300 {
            // 1..=6 patterns, each length 1..=4.
            let num_patterns = 1 + rng.next_usize(6);
            let mut owned: Vec<Vec<u8>> = Vec::with_capacity(num_patterns);
            for _ in 0..num_patterns {
                let plen = 1 + rng.next_usize(4);
                owned.push(random_bytes(&mut rng, alphabet, plen));
            }
            let pat_refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

            let text_len = rng.next_usize(20);
            let text = random_bytes(&mut rng, alphabet, text_len);

            let ac = AhoCorasick::new(&pat_refs).expect("patterns non-empty");
            let got = ac.find_iter(&text);
            let oracle = naive_matches(&pat_refs, &text);
            assert_eq!(got, oracle, "mismatch: patterns={pat_refs:?} text={text:?}");

            // `is_match` must agree with whether the oracle found anything.
            assert_eq!(ac.is_match(&text), !oracle.is_empty());
        }
    }

    /// Empty individual patterns are rejected.
    #[test]
    fn empty_pattern_rejected() {
        let patterns: [&str; 2] = ["ok", ""];
        assert!(matches!(
            AhoCorasick::new(&patterns),
            Err(SeqError::EmptyInput)
        ));
    }

    /// An empty pattern set builds and never matches.
    #[test]
    fn empty_pattern_set_never_matches() {
        let patterns: [&str; 0] = [];
        let ac = AhoCorasick::new(&patterns).expect("empty set is valid");
        assert_eq!(ac.pattern_count(), 0);
        assert!(ac.find_iter(b"anything at all").is_empty());
        assert!(!ac.is_match(b"anything"));
    }

    /// Empty text yields no matches regardless of the patterns.
    #[test]
    fn empty_text_no_matches() {
        let ac = AhoCorasick::new(&["a", "abc"]).expect("non-empty");
        assert!(ac.find_iter(b"").is_empty());
        assert!(!ac.is_match(b""));
    }
}
