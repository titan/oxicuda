//! Pitch (fundamental-frequency) estimation.
//!
//! - [`yin`] — YIN time-domain pitch tracker (de Cheveigné & Kawahara 2002).

pub mod yin;

pub use yin::{YinConfig, YinEstimate, yin_pitch};
