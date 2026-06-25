//! Replay-based continual learning methods.
//!
//! These methods store and replay past experiences to prevent forgetting,
//! ranging from simple buffers to gradient-level episodic memory.

pub mod a_gem;
pub mod dark_exp;
pub mod dark_exp_v2;
pub mod der_plus_plus;
pub mod er;
pub mod gem;
pub mod sharded_buffer;
pub mod vectorised_gem;

// ─── Sharded replay buffer re-exports ─────────────────────────────────────────
pub use sharded_buffer::{
    ReplayShard, ShardPolicy, ShardedReplayBuffer, ShardedReplayConfig, sharded_add,
    sharded_buffer_new, sharded_len, sharded_sample_balanced,
};

// ─── Vectorised GEM re-exports ────────────────────────────────────────────────
pub use vectorised_gem::{VectorisedGemConfig, vectorised_gem_project};

// ─── DER V2 re-exports ────────────────────────────────────────────────────────
pub use dark_exp_v2::{DerV2Buffer, DerV2Config};

// ─── DER++ re-exports ─────────────────────────────────────────────────────────
pub use der_plus_plus::DerPpLoss;
