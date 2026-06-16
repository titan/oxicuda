//! Classical string algorithms.
//!
//! This module collects combinatorial string algorithms that operate on raw
//! byte sequences, complementing the edit-distance routines in
//! [`crate::distance`] and the alignment code in [`crate::alignment`].
//!
//! * [`mod@manacher`] — Manacher's 1975 linear-time longest-palindromic-substring
//!   algorithm, exposing the full per-center radius array.
//! * [`suffix_automaton`] — the suffix automaton (DAWG) of Blumer et al. (1985)
//!   with Crochemore's online clone-on-split construction, supporting substring
//!   membership, distinct-substring counting, occurrence counting, and longest
//!   common substring.
//! * [`z_algorithm`] — Gusfield's 1997 Z-array in linear time via the Z-box,
//!   plus linear-time exact pattern matching through the `P $ T` construction.
//! * [`suffix_array`] — the suffix array via SA-IS (Nong–Zhang–Chan 2009) with
//!   Kasai's LCP array (2001) and binary-search pattern matching.
//! * [`fm_index`] — the Burrows–Wheeler transform (1994) and FM-index
//!   (Ferragina–Manzini 2000) built on the suffix array, with invertible BWT
//!   and backward-search `count`/`locate`.

pub mod fm_index;
pub mod manacher;
pub mod suffix_array;
pub mod suffix_automaton;
pub mod z_algorithm;

pub use fm_index::FmIndex;
pub use manacher::{Manacher, longest_palindrome_str, manacher};
pub use suffix_array::SuffixArray;
pub use suffix_automaton::{SuffixAutomaton, longest_common_substring};
pub use z_algorithm::{z_array, z_search};
