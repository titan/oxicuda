//! Sequence-alignment algorithms.

pub mod banded;
pub mod blast;
pub mod gotoh;
pub mod hirschberg;
pub mod needleman_wunsch;
pub mod smith_waterman;

pub use banded::banded_align;
pub use blast::{BlastAligner, BlastConfig, Hsp};
pub use gotoh::{GotohScoring, gotoh_align};
pub use hirschberg::hirschberg_align;
pub use needleman_wunsch::{Alignment, ScoringMatrix, needleman_wunsch};
pub use smith_waterman::smith_waterman;

/// Re-export the local alignment trace direction used by SW.
pub use smith_waterman::LocalAlignment;
