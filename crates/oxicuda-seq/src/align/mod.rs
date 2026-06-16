//! Wavefront-based sequence alignment.
//!
//! This module hosts the WFA (Wavefront Alignment) family of algorithms. It is
//! kept separate from the classical dynamic-programming aligners in
//! [`crate::alignment`]; the two share the [`crate::alignment::gotoh::GotohScoring`]
//! scoring scheme so their results can be cross-checked.

pub mod wfa;

pub use wfa::{WfaAlignment, WfaOp, wfa_align};
