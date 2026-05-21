//! LoRA (Low-Rank Adaptation) module.
//!
//! Provides LoRA adapter and weight merging utilities for efficient
//! fine-tuning of pre-trained models.

pub mod adapter;
pub mod dora;
pub mod merge;

pub use adapter::{LoraConfig, LoraLinear, LoraModel};
pub use dora::{DoraAdapter, DoraConfig};
pub use merge::{
    compose_adapters, merge_lora, scale_adapter, unmerge_lora, verify_merge_roundtrip,
};
