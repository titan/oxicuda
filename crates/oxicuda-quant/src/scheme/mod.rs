//! # Quantization Schemes
//!
//! This module exposes a suite of post-training quantization (PTQ) strategies:
//!
//! | Module        | Scheme                                      | Primary use |
//! |---------------|---------------------------------------------|-------------|
//! | `minmax`      | Min-Max calibration (INT4/INT8)             | General PTQ |
//! | `nf4`         | NormalFloat4 (QLoRA)                        | 4-bit weights |
//! | `fp8`         | FP8 E4M3 / E5M2 (Hopper / Blackwell)        | Training & inference |
//! | `gptq`        | GPTQ Hessian-guided quantization            | LLM weights |
//! | `smooth_quant`| SmoothQuant activation–weight migration     | LLM activations |
//! | `awq`         | AWQ activation-aware weight quantization    | LLM compression |
//! | `ada_round`   | AdaRound adaptive per-weight rounding       | INT4/INT8 PTQ |
//! | `ggml`        | GGML/GGUF block formats (Q8_0/Q4_0/Q4_1/Q4_K)| llama.cpp deploy |
//! | `gguf`        | GGUF v3 container read/write (metadata + tensor directory) | llama.cpp deploy |
//! | `llm_int8`    | LLM.int8() outlier-aware mixed precision     | LLM inference |
//! | `kv_cache`    | INT8/INT4 KV-cache quantization (KIVI-style)| LLM decoding |
//! | `sparse_gptq` | Sparse-GPTQ joint pruning + quantization     | LLM compression |

pub mod ada_round;
pub mod awq;
pub mod fp8;
pub mod ggml;
pub mod gguf;
pub mod gptq;
pub mod kv_cache;
pub mod llm_int8;
pub mod minmax;
pub mod nf4;
pub mod smooth_quant;
pub mod sparse_gptq;

pub use ada_round::{AdaRound, AdaRoundConfig, AdaRoundResult, ada_round};
pub use awq::{AwqConfig, AwqOutput, AwqQuantizer, awq_quantize};
pub use fp8::{Fp8Codec, Fp8Format};
pub use ggml::{
    BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ8_0, GgmlType, dequantize_q4_0, dequantize_q4_1,
    dequantize_q4_k, dequantize_q8_0, f16_round, f16_to_f32, f32_to_f16_bits, fake_quantize,
    quantize_q4_0, quantize_q4_1, quantize_q4_k, quantize_q8_0,
};
pub use gguf::{
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgufArray, GgufFile, GgufHeader,
    GgufMetadataKv, GgufMetadataValue, GgufTensorInfo, GgufValueType, read_gguf, write_gguf,
};
pub use gptq::{GptqConfig, GptqOutput, GptqQuantizer};
pub use kv_cache::{KvAxis, KvCacheConfig, KvCacheQuantizer, QuantizedKvCache};
pub use llm_int8::{LlmInt8Config, LlmInt8Output, LlmInt8Quantizer};
pub use minmax::{MinMaxQuantizer, QuantGranularity, QuantParams, QuantScheme};
pub use nf4::{NF4_LUT, Nf4Quantizer};
pub use smooth_quant::{SmoothQuantConfig, SmoothQuantMigrator};
pub use sparse_gptq::{SparseGptqConfig, SparseGptqOutput, SparseGptqQuantizer, SparsityTarget};
