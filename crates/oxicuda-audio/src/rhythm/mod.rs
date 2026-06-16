//! Rhythm analysis for `oxicuda-audio`.
//!
//! Provides:
//! - **`beat_tracker`**: Ellis (2007) / Böck (2011) dynamic-programming beat
//!   tracking over an onset-strength envelope with a tempo prior and an
//!   inter-beat-interval transition penalty.

pub mod beat_tracker;

pub use beat_tracker::{BeatTracker, BeatTrackerConfig, beat_times};
