//! KV cache subsystem — PagedAttention block management.
//!
//! Modules:
//! * [`kv_cache`]       — physical block pool and per-block K/V storage.
//! * [`cache_manager`]  — per-sequence block table management.
//! * [`prefix_cache`]   — LRU prefix-sharing cache for prompt reuse.
//! * [`radix_cache`]    — radix-tree cache for partial (longest-prefix) reuse.
//! * [`sliding_window`] — StreamingLLM attention-sink rolling KV window.
//! * [`kv_quant`]       — per-token INT8/INT4 KV-cache quantization logic.
//! * [`compaction`]     — page-table compaction / defragmentation planner.

pub mod cache_manager;
pub mod compaction;
pub mod kv_cache;
pub mod kv_quant;
pub mod prefix_cache;
pub mod radix_cache;
pub mod sliding_window;

pub use cache_manager::CacheManager;
pub use compaction::{CompactionPlan, plan_compaction, rewrite_block_table};
pub use kv_cache::{BlockId, KvBlock, PagedKvCache};
pub use kv_quant::{
    KvQuantConfig, QuantizedToken, quantization_mse, quantize_dequantize_token, quantize_token,
};
pub use prefix_cache::{PrefixCache, PrefixEntry};
pub use radix_cache::{MatchResult, RadixCache};
pub use sliding_window::{SlidingWindowConfig, SlidingWindowManager};
