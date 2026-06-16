//! String-distance algorithms.
//!
//! This module collects bit-parallel and dynamic-programming string-distance
//! routines that complement the alignment-oriented code in
//! [`crate::alignment`] and the evaluation metrics in [`crate::metrics`].
//!
//! * [`myers_bitvector`] — Myers' 1999 bit-vector algorithm computing the global
//!   Levenshtein edit distance in `O(n · ⌈m / w⌉)` word operations.

pub mod myers_bitvector;

pub use myers_bitvector::{myers_distance_str, myers_edit_distance, myers_edit_distance_checked};
