//! Replay-based continual learning methods.
//!
//! These methods store and replay past experiences to prevent forgetting,
//! ranging from simple buffers to gradient-level episodic memory.

pub mod a_gem;
pub mod dark_exp;
pub mod er;
pub mod gem;
