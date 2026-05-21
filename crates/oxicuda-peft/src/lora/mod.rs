/// AdaLoRA: adaptive rank allocation via singular value decomposition structure.
pub mod adalora;
/// AWQ: activation-aware weight quantization (Lin et al. 2024).
pub mod awq;
#[cfg(test)]
mod awq_tests;
/// DoRA: weight-decomposed low-rank adaptation.
pub mod dora;
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
/// OLoRA: orthonormal-A initialisation via Gram-Schmidt.
pub mod olora;
/// PiSSA: principal-singular-value LoRA initialisation.
pub mod pissa;
/// QA-LoRA: quantization-aware LoRA with group-wise NF4 quantization (Xu et al. 2023 ICLR).
pub mod qa_lora;
/// QLoRA: 4-bit NF4-quantised weights with LoRA adapter.
pub mod qlora;
/// VeRA: vector-based random adaptation with shared frozen projections.
pub mod vera;

pub use awq::{Awq, AwqConfig, AwqQuantized};
pub use gptq::{Gptq, GptqConfig, GptqQuantized};
pub use hqq::{Hqq, HqqConfig, HqqQuantized};
pub use qa_lora::{QaLoraConfig, QaLoraLayer};
