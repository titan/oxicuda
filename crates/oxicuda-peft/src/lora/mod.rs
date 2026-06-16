/// AdaLoRA: adaptive rank allocation via singular value decomposition structure.
pub mod adalora;
/// AWQ: activation-aware weight quantization (Lin et al. 2024).
pub mod awq;
#[cfg(test)]
mod awq_tests;
/// BOFT: butterfly-factorised orthogonal fine-tuning (Liu et al. 2024).
pub mod boft;
/// DoRA: weight-decomposed low-rank adaptation (column-wise magnitude).
pub mod dora;
/// DoRA (row-wise): weight-decomposed low-rank adaptation with per-output-feature magnitude.
pub mod dora_new;
/// DyLoRA: dynamic search-free low-rank adaptation over a nested rank range (Valipour et al. 2023).
pub mod dylora;
/// Flora: low-rank Adam via random gradient projection (Hao et al. 2024).
pub mod flora;
/// GLoRA: generalized LoRA with five weight/bias support tensors (Chavan et al. 2023).
pub mod glora;
/// GPTQ: activation-aware post-training quantization (Frantar et al. 2023).
pub mod gptq;
#[cfg(test)]
mod gptq_tests;
/// HQQ: half-quadratic quantization for low-bit weights.
pub mod hqq;
/// LoHa: low-rank Hadamard product adapter.
pub mod loha;
#[cfg(test)]
mod loha_tests;
/// LoKr: low-rank Kronecker product adapter.
pub mod lokr;
#[cfg(test)]
mod lokr_tests;
/// Standard LoRA low-rank adapter.
pub mod lora;
/// LoRA-FA: frozen-A variant with trainable B only.
pub mod lora_fa;
/// LoRA+: separate learning rates for A and B.
pub mod lora_plus;
/// MoLoRA: mixture of low-rank adapters routed per token.
pub mod molora;
#[cfg(test)]
mod molora_tests;
/// MoRA: high-rank updating with a square trainable matrix (Jiang et al. 2024).
pub mod mora;
/// MoSA: mixture of sparse low-rank adapters with MoE-style routing (Zeng et al. 2024).
pub mod mosa;
/// OFT: orthogonal fine-tuning via Cayley-parametrised block rotations (Qiu et al. 2023).
pub mod oft;
/// OLoRA: orthonormal-A initialisation via Gram-Schmidt.
pub mod olora;
/// PiSSA: principal-singular-value LoRA initialisation.
pub mod pissa;
/// QA-LoRA: quantization-aware LoRA with group-wise NF4 quantization (Xu et al. 2023 ICLR).
pub mod qa_lora;
/// QLoRA: 4-bit NF4-quantised weights with LoRA adapter.
pub mod qlora;
/// ReLoRA: periodic merge-and-restart low-rank adaptation (Lialin et al. 2024).
pub mod relora;
/// VeRA: vector-based random adaptation with shared frozen projections.
pub mod vera;

pub use awq::{Awq, AwqConfig, AwqQuantized};
pub use boft::{BoftConfig, BoftLinear};
pub use dora_new::{DoraConfig, DoraLayer};
pub use dylora::{DyLoraConfig, DyLoraLinear};
pub use flora::{FloraCompressor, FloraConfig};
pub use glora::{GloraConfig, GloraLinear};
pub use gptq::{Gptq, GptqConfig, GptqQuantized};
pub use hqq::{Hqq, HqqConfig, HqqQuantized};
pub use mora::{MoraConfig, MoraLinear, suggest_square_rank};
pub use mosa::{MosaAdapter, MosaConfig};
pub use oft::{OftConfig, OftLinear};
pub use qa_lora::{QaLoraConfig, QaLoraLayer};
pub use relora::{ReloraConfig, ReloraLinear, ReloraSchedule};
