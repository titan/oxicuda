# oxicuda-lm TODO

Large language model inference primitives: BPE tokenizer, transformer layer building blocks with KV-cache, GPT-2 and LLaMA-2/3 architectures, and PTX kernel generators -- pure Rust, zero CUDA SDK dependency. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.13).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 5,781 SLoC across 22 files (includes Markdown doc-comments) / 4,470 pure Rust SLoC**

Model-layer abstractions for LLM inference: BPE tokenizer, transformer layer building blocks
with KV-cache for incremental decode, complete GPT-2 and LLaMA-2/3 model implementations,
and GPU kernel PTX string generators for 6 SM versions (75/80/86/90/100/120).

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `LmError` (17 variants): DimensionMismatch, InvalidConfig, EmptyInput, OutOfVocab, Utf8Decode, WeightNotFound/ShapeMismatch, LayerIndexOutOfRange, HeadDimMismatch, KvCacheLengthMismatch, SequenceTooLong, InvalidMergePair, VocabSizeMismatch, GqaHeadMismatch, WeightDataLengthMismatch, Internal
- [x] `config.rs` -- `GptConfig` GPT-2 presets: `gpt2_small` (12L/12H/768D), `gpt2_medium` (24L/16H/1024D), `gpt2_large`, `gpt2_xl`, `tiny` (2L/2H/8D); `LlamaConfig` LLaMA presets: `llama2_7b`, `llama2_13b`, `llama3_8b` (GQA 32H/8KV), `mistral_7b`, `phi2`, `tiny` (2L/4H/2KV)
- [x] `handle.rs` -- `LmHandle`, `SmVersion` with `ptx_version_str()`, `target_str()`
- [x] `lib.rs` -- module declarations, top-level re-exports, 10 E2E integration tests

#### Weights
- [x] `weights.rs` -- `WeightTensor { data, shape }` with `zeros()`, `ones()`, `eye()`, `from_data()`, `row_slice()`, `validate_shape()`; `ModelWeights` HashMap-backed -- `get_checked()` with shape validation, `n_params()`, iterators

#### PTX Kernel Generators
- [x] `ptx_kernels.rs` -- 5 GPU kernels x 6 SM versions (75/80/86/90/100/120)
  - `embedding_forward_ptx` -- token embedding table lookup (grid-stride over n_tokens x embed_dim)
  - `rope_apply_ptx` -- RoPE in-place from pre-computed cos/sin tables; grid-stride pair indexing
  - `silu_gate_ptx` -- SwiGLU gate: `out = (g / (1 + exp(-g))) * up`; `ex2.approx.f32` + `rcp.approx.f32`
  - `rms_norm_ptx` -- shared-memory warp butterfly reduction -> normalize + scale; `sqrt.approx.f32`
  - `causal_attn_softmax_ptx` -- per-head causal mask + stable softmax (max -> exp -> sum -> normalize)

#### Tokenizer (`tokenizer/`)
- [x] `tokenizer/mod.rs` -- module organization
- [x] `tokenizer/vocab.rs` -- `Vocab` byte<->id bidirectional map; `gpt2_byte_vocab()` (256 byte tokens); `with_extra_tokens()`, `special_id()`
- [x] `tokenizer/bpe.rs` -- `BpeTokenizer` byte-level BPE; `merge_ranks` (priority table) + `pair_to_merged` (result table); `encode()` vocab-lookup init -> greedy lowest-rank merge loop; `decode()` byte concat -> UTF-8; `BpeBuilder` -- `add_merge()`, `add_special()`, `build()` convenience builder

#### Layers (`layer/`)
- [x] `layer/mod.rs` -- module organization
- [x] `layer/norm.rs` -- `RmsNorm` / `LayerNorm` -- per-token normalize with learnable weight (and bias for LayerNorm)
- [x] `layer/embedding.rs` -- `TokenEmbedding`, `LearnedPositionalEmbedding`, `RotaryEmbedding` -- RoPE with precomputed cos/sin tables, absolute position offset for KV-cache decode
- [x] `layer/ffn.rs` -- `MlpFfn` GPT-2 GELU MLP `W_proj(GELU(W_fc * x + b)) + b_proj`; `SwiGluFfn` LLaMA SwiGLU `W_down(silu(W_gate * x) (*) W_up * x)`, no biases
- [x] `layer/attention.rs` -- `LayerKvCache` / `MultiHeadAttention` -- GQA (`kv_h = q_h / (n_heads / n_kv_heads)`), causal mask at absolute position `past_len + t`, KV append for incremental decode
- [x] `layer/transformer.rs` -- `GptBlock` / `LlamaBlock` / `PastKvCache` -- pre-LN residual blocks; multi-layer KV cache container

#### Models (`model/`)
- [x] `model/mod.rs` -- module organization
- [x] `model/gpt.rs` -- `Gpt2Model` -- token + pos embedding -> N x GptBlock -> LayerNorm -> weight-tied LM head; `next_token()` greedy decode
- [x] `model/llama.rs` -- `LlamaModel` -- TokenEmbedding -> N x LlamaBlock -> RmsNorm -> independent LM head; `next_token()` greedy decode
- [x] `model/weights.rs` -- weight loaders: `load_gpt2_block()` (HuggingFace key convention, packed QKV split), `load_llama_block()` (separate q/k/v proj)

#### Integration Tests
- [x] 10 E2E tests in `lib.rs`:
  - GPT-2 tiny forward (shape, zero-weight -> zero-logits)
  - LLaMA tiny forward (shape validation)
  - GPT-2 incremental decode consistency (full vs token-by-token last-position logit match)
  - LLaMA incremental decode consistency
  - BPE encode/decode round-trip ("hello" -> [259] -> "hello")
  - RMSNorm + LayerNorm numerical correctness
  - PTX kernels x 6 SM versions (target directive presence)
  - LLaMA GQA multi-step decode (prefill 4 + decode 3 -> past_len=7)
  - Vocab special token round-trip (BOS/EOS)
  - GPT-2 greedy decode loop (5 steps, all IDs in vocab range)

### Future Enhancements

#### P0 -- Critical (Model Coverage)
- [x] GPT-2 full architecture (`model/gpt.rs`)
- [x] LLaMA-2 / LLaMA-3 with GQA (`model/llama.rs`)
- [x] Pre-LN residual blocks for both families (`layer/transformer.rs`)
- [x] Weight tying (GPT-2) and independent LM head (LLaMA) (`model/`)

#### P1 -- Important (Inference Throughput)
- [x] Incremental decode with `PastKvCache` (`layer/transformer.rs`)
- [x] RoPE absolute-position offset for KV-cache decode (`layer/embedding.rs`)
- [x] GQA head indexing for LLaMA-3 / Mistral (`layer/attention.rs`)
- [x] BPE tokenizer with byte-level fallback (`tokenizer/bpe.rs`)

#### P2 -- Nice-to-Have (Numerical / Ecosystem)
- [x] RmsNorm + LayerNorm reference implementations (`layer/norm.rs`)
- [x] SwiGLU + GELU FFN variants (`layer/ffn.rs`)
- [x] HuggingFace weight loader convention (`model/weights.rs`)
- [ ] (P2) Mixtral-style MoE block (deferred -- integrates with `oxicuda-dist-infer` expert parallelism)
- [x] (P2) Long-context RoPE scaling (NTK-aware / YaRN) (`layer/rope_scaling.rs`; Peng et al. 2023 YaRN, Chen 2023 PI, bloc97 NTK-aware)
- [x] (P2) Flash-Attention CPU reference (`layer/flash_attention.rs`; Dao et al. 2022 FlashAttention -- tiled online-softmax, equivalence-checked vs naive)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

(No CUDA crate deps -- `oxicuda-lm` is a pure CPU reference layer; PTX kernels are emitted as strings for downstream consumers.)

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 226 passing (root TODO.md count)
- unwrap() calls: 0 (production code; tests use `.expect()` with descriptive messages)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, all CPU reference paths work natively

## Performance Targets

| Operation | Target |
|-----------|--------|
| `embedding_forward_ptx` -- 32k-vocab, 4096-d table | >= 95% bandwidth-limited peak on sm_80+ |
| `rope_apply_ptx` -- 4096-token, 4096-d | >= 90% bandwidth-limited peak |
| `silu_gate_ptx` -- 11008-d SwiGLU (LLaMA-2-7B FFN) | >= 90% bandwidth-limited peak |
| `rms_norm_ptx` -- 4096-d per-token normalize | >= 95% bandwidth-limited peak |
| `causal_attn_softmax_ptx` -- 4096-token causal softmax | >= 90% bandwidth-limited peak |
| `Gpt2Model::next_token` (CPU, tiny config) | sub-millisecond per decode step |
| `LlamaModel::next_token` (CPU, tiny config) | sub-millisecond per decode step |

## Numerical Accuracy Requirements

| Operation | Tolerance |
|-----------|-----------|
| GPT-2 full vs incremental decode (last-position logits) | abs < 1e-4 |
| LLaMA full vs incremental decode (last-position logits) | abs < 1e-4 |
| RmsNorm output rms | abs < 1e-5 vs analytic `1 / sqrt(mean(x^2) + eps)` |
| LayerNorm output (mean, var) | mean abs < 1e-5, var abs < 1e-4 |
| RoPE round-trip (apply + un-apply) | abs < 1e-6 |
| BPE encode/decode round-trip | exact UTF-8 equality |

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86 / sm_89)
- [x] PTX header selection emits `.target sm_80` for cp.async-capable kernels
- [ ] Tensor-core attention QK^T via `mma.sync.aligned.m16n8k8` (waiting on `oxicuda-blas` GEMM path)
- [ ] cp.async-driven RoPE table streaming (deferred)

### Hopper (sm_90 / sm_90a)
- [x] PTX header selection emits `.target sm_90`
- [ ] `wgmma` attention QK^T + `wgmma` SwiGLU FFN (deferred -- waits on `oxicuda-blas` Hopper path)
- [ ] TMA-driven KV-cache load (when integrated with `oxicuda-infer::PagedKvCache`)

### Blackwell (sm_100 / sm_120)
- [x] PTX header generation supports sm_100/sm_120 (matches root SM table)
- [ ] FP8 weight loading for LLaMA-3 inference (waits on `oxicuda-quant` FP8 dequant + `oxicuda-blas` FP8 GEMM)

## Deepening Opportunities

### Verification Gaps
- [x] GPT-2 and LLaMA incremental decode last-token consistency (covered by E2E tests)
- [x] PTX kernels validated for all 6 SM versions (sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120)
- [x] BPE round-trip on "hello" cascade (covered by E2E test)
- [x] LLaMA-3 GQA (4Q / 2KV in tiny) multi-step decode (covered by E2E test)
- [ ] Real HuggingFace checkpoint loading verified against PyTorch reference (loader exists; needs `oxicuda-arc` safetensors plumbing)
- [ ] BPE tokenizer fidelity against `tiktoken` reference on a large corpus

### Implementation Deepening
- [x] `Vocab` supports byte-level encoding + special tokens
- [x] `BpeTokenizer` uses greedy lowest-rank merge loop (standard byte-pair encoding)
- [x] `MultiHeadAttention` handles GQA with `kv_h = q_h / (n_heads / n_kv_heads)`
- [x] `PastKvCache` is a multi-layer container; each layer holds its own `LayerKvCache`
- [x] Weight loaders for both packed-QKV (GPT-2) and separate-QKV (LLaMA) conventions
- [ ] Speculative decoding draft model selection helper (integrates with `oxicuda-infer::speculative_verify`)
- [ ] Quantized weight loading (INT8 / NF4) via `oxicuda-quant` codecs
- [x] Long-context (>4k token) RoPE scaling (NTK-aware / YaRN) (`layer/rope_scaling.rs`; Peng et al. 2023 YaRN, Chen 2023 PI, bloc97 NTK-aware)

## Notes

- All forward passes are pure-Rust CPU reference implementations -- GPU acceleration is provided by the PTX kernel strings (see `ptx_kernels.rs`) once a CUDA driver is available at runtime via `oxicuda-driver` / `oxicuda-launch`.
- The `Gpt2Model` and `LlamaModel` `next_token()` paths are deterministic greedy decode -- richer sampling (top-k / top-p / beam) is provided by `oxicuda-infer::sampling`.
- Benchmarks live in `benches/lm_inference.rs` (Criterion harness) -- CPU forward-pass timing only; GPU benchmarking awaits Linux+NVIDIA hardware.
- Future integration with `oxicuda-infer::ModelRunner` trait will expose `Gpt2Model` and `LlamaModel` as production-grade runners for the continuous batcher.
