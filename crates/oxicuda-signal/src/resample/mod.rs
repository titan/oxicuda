//! Sample-rate conversion (resampling).
//!
//! Provides rational (`up / down`) resampling via efficient polyphase
//! filtering with a Kaiser-windowed-sinc anti-alias prototype, plus a
//! convenience wrapper that derives the integer ratio from input/output
//! sampling rates.

pub mod polyphase;

pub use polyphase::{resample_poly, resample_rate};
