//! LoRA (Low-Rank Adaptation) module.
//!
//! Provides LoRA adapter and weight merging utilities for efficient
//! fine-tuning of pre-trained models.

pub mod adapter;
pub mod checkpoint;
pub mod dora;
pub mod merge;
pub mod mixed_rank;
pub mod qlora;

pub use adapter::{LoraConfig, LoraLinear, LoraModel};
pub use checkpoint::{LORA_CKPT_MAGIC, load as load_lora, save as save_lora};
pub use dora::{DoraAdapter, DoraConfig};
pub use merge::{
    compose_adapters, merge_lora, scale_adapter, unmerge_lora, verify_merge_roundtrip,
};
pub use mixed_rank::{BudgetStrategy, LayerSpec, MixedRankLoraModel, RankBudget};
pub use qlora::{NF4_LEVELS, Nf4Tensor, QLoraLinear};
