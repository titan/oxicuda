//! Multi-pattern and exact string-matching automata.
//!
//! This module hosts matching algorithms that compile a *set* of patterns into
//! a reusable automaton, complementing the single-string distance routines in
//! [`crate::distance`] and the alignment code in [`crate::alignment`].
//!
//! * [`aho_corasick`] — Aho & Corasick's 1975 multi-pattern matcher: a trie of
//!   the patterns augmented with failure (suffix) and dictionary-suffix links,
//!   scanning a text in `O(n + #matches)` and reporting every occurrence of
//!   every pattern, overlaps included.

pub mod aho_corasick;

pub use aho_corasick::{AhoCorasick, Match};
