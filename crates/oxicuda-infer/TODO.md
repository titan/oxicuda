# oxicuda-infer TODO

vLLM-style continuous batching inference engine with PagedAttention KV cache, speculative decoding, beam search, top-k/top-p sampling, prefix caching, and a pluggable ModelRunner abstraction. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.11).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 7,754 SLoC across 33 files (includes Markdown doc-comments) / 4,060 pure Rust SLoC**

Production-grade GPU inference engine implementing the algorithms required for efficient large
language-model serving: PagedAttention KV cache (Kwon et al., 2023), continuous batching
(vLLM, Yu et al., 2022), speculative decoding (Chen et al., 2023), beam search, and rich
structured sampling.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `InferError` (15 variants): BlockAllocFailed, InvalidSequenceId, EmptyBatch, DimensionMismatch, SamplingError, SchedulerFull, NoPrefillSeqs, CacheManagerError, InvalidSamplingParams, EosTokenMissing, BeamSearchError, SpeculativeError, UnsupportedConfig, ModelRunnerError, Other
- [x] `handle.rs` -- `InferHandle` -- device, sm_version, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, block_size, max_seq_len; `ptx_version_str()`, `attention_scale()`
- [x] `lib.rs` -- module declarations, top-level re-exports, 6 E2E integration tests

#### PTX Kernel Sources
- [x] `ptx_kernels.rs` -- 5 GPU-side inference kernels (14.2 KB of PTX)
  - `paged_attn_ptx` -- online Flash-Attention-style softmax over paged KV blocks; per-block numerically-stable `m_new = max(m, tile_max)`
  - `rope_apply_ptx` -- in-place RoPE with `cos.approx.f32` / `sin.approx.f32`; frequency `theta_i = position * 10000^(-2i/d)`
  - `top_k_filter_ptx` -- sets non-top-K logit positions to NEG_INFINITY; register-shuffle warp sort
  - `logits_softmax_ptx` -- three-pass stable softmax: max -> sum_exp -> normalize using warp butterfly reduces
  - `kv_append_ptx` -- writes K/V into physical block slot; grid-stride across attention heads

#### KV Cache (`cache/`)
- [x] `cache/mod.rs` -- module organization
- [x] `cache/kv_cache.rs` -- `BlockId(u32)` opaque identifier; `KvBlock` with `append()`, `key_slice()`, `value_slice()`, `reset()`; `PagedKvCache` -- `[n_layers][n_blocks]` 2D block pool; O(1) free-list alloc; reference counting for copy-on-write prefix sharing
- [x] `cache/cache_manager.rs` -- `CacheManager` -- per-sequence block tables `HashMap<u64, Vec<BlockId>>`; auto-grow on block fill; `allocate_sequence`, `free_sequence`, `append_token`
- [x] `cache/prefix_cache.rs` -- `PrefixCache` -- FNV-1a token hash -> `PrefixEntry`; LRU eviction; `lookup()`, `insert()`, `hit_rate()`

#### Batch Scheduling (`batch/`)
- [x] `batch/mod.rs` -- module organization
- [x] `batch/sequence.rs` -- `SequenceStatus`: Waiting -> Prefill -> Decode -> Finished(FinishReason) with EosToken(u32) / MaxLength variants; `SamplingParams` (temperature, top_k, top_p, max_new_tokens, eos_token_id, repetition_penalty)
- [x] `batch/scheduler.rs` -- `Scheduler` -- FCFS admission; token-budget decode phase; memory-pressure preemption; `ScheduledBatch{prefill_ids, decode_ids}`; `on_step_complete` / `take_finished`
- [x] `batch/continuous_batcher.rs` -- `ContinuousBatcher` -- orchestrates scheduler + cache_manager + model_fn + Rng; one batched forward pass per `step()`

#### Sampling Suite (`sampling/`)
- [x] `sampling/mod.rs` -- module organization; `Rng` -- 64-bit LCG (Knuth constants); `softmax` + `categorical_sample`
- [x] `sampling/greedy.rs` -- `greedy_sample` / `greedy_sample_batch` -- argmax with NaN guard
- [x] `sampling/top_k.rs` -- `top_k_filter` / `top_k_sample` -- threshold from k-th sorted logit; exactly-k tokens retained
- [x] `sampling/top_p.rs` -- `top_p_filter` / `top_p_sample` -- sorted cumulative-probability nucleus cutoff
- [x] `sampling/beam_search.rs` -- `BeamSearchState::step()` -- log-softmax expansion; keep `beam_width` candidates; EOS -> completed; length-normalised `score / len^alpha`
- [x] `sampling/speculative.rs` -- `speculative_verify()` -- rejection sampling: accept `d_i` if `u < min(1, p_target / p_draft)`; correction token from `max(0, p - q) / Z`; provably identical distribution to target

#### Executor (`executor/`)
- [x] `executor/mod.rs` -- module organization
- [x] `executor/model_runner.rs` -- `ModelRunner` trait -- `vocab_size()`, `decode(token_ids, block_tables, seq_lens)`, `prefill(token_ids, seq_starts, block_tables)`; `MockModelRunner` -- peaks at `(token_id + bias) % vocab_size`; `RunnerStats` -- n_steps, total_tokens, sequences_completed; `avg_batch_size()`
- [x] `executor/attention_backend.rs` -- `paged_attention_cpu` -- reference GQA PagedAttention: load K/V per block, Q*K^T*scale, stable softmax, weighted *V; `kv_h = h / (n_heads / n_kv_heads)`; `AttentionConfig`

#### Integration Tests
- [x] 6 E2E tests in `lib.rs`:
  - `e2e_greedy_until_eos` -- continuous batching generates until EOS token
  - `e2e_max_tokens_termination` -- max_new_tokens=1 path
  - `e2e_beam_search_completes` -- beam_width=2 finishes on EOS in one step
  - `e2e_speculative_all_accepted` -- draft == target -> all k drafts accepted
  - `e2e_paged_attention_single_token` -- Q = V = 1.0 -> output = 1.0
  - `e2e_prefix_cache_hit_rate` -- 3 queries (1 miss, 2 hits) -> hit_rate = 2/3

### Future Enhancements

#### P0 -- Critical (Throughput Path)
- [x] PagedAttention KV cache with O(1) block alloc (`cache/kv_cache.rs`)
- [x] Continuous batching with FCFS + token-budget decode (`batch/`)
- [x] Numerically stable Flash-Attention-style PTX softmax (`ptx_kernels.rs::paged_attn_ptx`)
- [x] Per-sequence block-table management (`cache/cache_manager.rs`)

#### P1 -- Important (Quality + Latency)
- [x] Speculative decoding with provably correct rejection sampling (`sampling/speculative.rs`)
- [x] Beam search with length-normalised scoring (`sampling/beam_search.rs`)
- [x] Top-k and top-p (nucleus) filters (`sampling/top_k.rs`, `sampling/top_p.rs`)
- [x] Prefix cache with FNV-1a hashing + LRU eviction (`cache/prefix_cache.rs`)

#### P2 -- Nice-to-Have (Advanced Sampling)
- [x] Pluggable `ModelRunner` trait + `MockModelRunner` for unit testing (`executor/model_runner.rs`)
- [x] Repetition penalty in `SamplingParams` (`batch/sequence.rs`)
- [x] (P2) Structured output / JSON-constrained sampling (sampling/json_constrained.rs -- char-level pushdown JSON validator state machine + logit masking of structurally-invalid tokens)
- [x] `sampling/logits_processor.rs` / `LogitsProcessor` — composable logits-processing pipeline: `LogitsProcessor` trait with `process()` over raw logit `Vec<f32>`; built-in processors: `TemperatureScaling`, `RepetitionPenalty`, `TopKFilter`, `TopPFilter`, `MinPFilter`; `LogitsProcessorChain` for sequential composition before sampling
- [x] `sampling/beam_search.rs` / `BeamSearch` — extended `BeamSearch` struct wrapping `BeamSearchState` with configurable `BeamSearchConfig { beam_width, length_penalty_alpha, min_length, no_repeat_ngram_size }`; `BeamSearch::run()` drives the multi-step decode loop
- [ ] (P2) Chunked prefill for long-prompt latency reduction (sequence chunking exists but not pipelined)
- [x] (P2) Speculative decoding with verified-tree / Medusa heads (sampling/medusa.rs -- Cai 2024; multi-head top-k candidate tree capped at max_candidates + verify longest-accepted-prefix; extends single-draft SpeculativeDecoder)
- [x] `kv_cache/paged_kv.rs` — Paged KV cache (Kwon 2023 vLLM): physical KV block table + logical page mapping; dynamic allocation for variable-length sequences; O(1) amortised block allocation
- [x] `speculative/drafter.rs` — Speculative decoding (Leviathan 2023): small draft model generates k tokens; target model verifies in one forward pass; acceptance ratio approach with temperature-corrected rejection sampling
- [ ] `quantization/awq.rs` — AWQ activation-aware weight quantization (Lin 2023): per-channel weight scaling based on activation magnitude; protect salient weights from INT4 rounding; no gradient needed
- [x] `serving/continuous_batching.rs` — Continuous batching scheduler (Orca 2022): iteration-level scheduling; add new sequences at any step; preemption via swap/recompute; `ContinuousBatchScheduler`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

(No CUDA crate deps -- `oxicuda-infer` is a pure orchestration layer; PTX kernels are generated as strings and executed by downstream callers via `oxicuda-driver`/`oxicuda-launch`.)

## Quality Status

- Warnings: 0 (clippy clean, `#![forbid(unsafe_code)]`)
- Tests: 297 passing (root TODO.md count)
- unwrap() calls: 0 (production code)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, all CPU reference paths work; runtime GPU executor returns `UnsupportedPlatform`

## Performance Targets

| Operation | Target |
|-----------|--------|
| `paged_attn_ptx` -- 64-block, 16-token-per-block attention | >= 90% of FlashAttention-2 throughput on sm_80+ |
| `kv_append_ptx` -- 16-token K/V write across 32 heads | >= 95% bandwidth-limited peak |
| `top_k_filter_ptx` -- 32k-vocab top-k=50 | >= 85% bandwidth-limited peak |
| `logits_softmax_ptx` -- 32k-vocab softmax | >= 90% bandwidth-limited peak |
| `PagedKvCache::alloc_block` -- O(1) amortised | < 100 ns CPU dispatch |
| `ContinuousBatcher::step` -- 64-sequence batch | sub-millisecond CPU overhead |

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86 / sm_89)
- [x] PTX header selection emits `.target sm_80` for cp.async-capable PagedAttention
- [ ] cp.async-driven KV block streaming for sm_80+ (deferred -- requires GPU verification)
- [ ] mma.sync.aligned.m16n8k8 in attention QK^T (waiting on `oxicuda-blas` tensor-core path)

### Hopper (sm_90 / sm_90a)
- [x] PTX header selection emits `.target sm_90`
- [ ] TMA-driven KV cache block loads with `wgmma` attention (deferred)
- [ ] Warp-specialized PagedAttention producer-consumer split (deferred)

### Blackwell (sm_100 / sm_120)
- [x] PTX header generation supports sm_100/sm_120 (matches root SM table)
- [ ] FP8 PagedAttention via `oxicuda-quant` + `oxicuda-blas` FP8 GEMM path

## Deepening Opportunities

### Verification Gaps
- [x] PagedAttention CPU reference matches a single-block direct attention computation
- [x] Speculative decoding `accepted.len() == k` when draft == target verified
- [x] Beam search EOS termination preserves `completed` candidates
- [x] Prefix cache hit-rate statistically tracked and exposed via `hit_rate()`
- [ ] Multi-head GQA correctness verified against ungrouped MHA reference (kv_heads != n_heads paths)
- [ ] Continuous batcher memory-pressure preemption traced under heavy churn

### Implementation Deepening
- [x] `PagedKvCache` reference counting supports copy-on-write prefix sharing
- [x] `Scheduler` distinguishes Prefill / Decode phases with token-budget admission
- [x] `SamplingParams` includes repetition penalty (often absent in early vLLM clones)
- [x] `BeamSearchState` length normalisation via `score / len^alpha` configurable
- [ ] Page table compaction / defragmentation routine for long-running engines
- [ ] Per-sequence sampling parameter override (currently per-batch via `SamplingParams`)

## Notes

- `oxicuda-infer` is intentionally GPU-agnostic at the API layer: PTX kernel strings are emitted by `ptx_kernels.rs` and executed by downstream consumers via `oxicuda-driver` / `oxicuda-launch`.
- All E2E tests use `MockModelRunner` -- real GPU model runners are expected to be implemented in `oxicuda-lm` or user crates by implementing the `ModelRunner` trait.
- Benchmarks live in `benches/infer_engine.rs` (Criterion harness) -- CPU-side dispatch heuristics only; GPU benchmarking awaits Linux+NVIDIA hardware.
