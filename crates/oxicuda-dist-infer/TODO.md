# oxicuda-dist-infer TODO

Distributed multi-GPU inference engine with three orthogonal parallelism axes (TP x SP x EP = world_size), distributed KV-cache management, and affinity-aware request routing. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.12).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 4,553 SLoC across 19 files (includes Markdown doc-comments) / 3,279 pure Rust SLoC**

Production-grade distributed inference infrastructure for OxiCUDA. Implements three orthogonal
parallelism strategies and the distributed KV-cache / request-routing infrastructure needed to
serve LLMs across GPU clusters.

| Axis | Degree | Description |
|------|--------|-------------|
| TP | `tp` | Tensor parallelism -- shard weight matrices column- or row-wise |
| SP | `sp` | Sequence parallelism -- partition the token sequence |
| EP | `ep` | Expert parallelism -- partition MoE experts across GPUs |

The three degrees multiply to `world_size = tp * sp * ep`.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `DistInferError` (27 variants): InvalidWorldSize, RankOutOfRange, TooFewRanks, TpFeaturesMisaligned, TpInputMisaligned, ShardShapeMismatch, SpSeqLenMisaligned, EmptyChunk, EpExpertsMisaligned, EmptyExpertBatch, SequenceNotOwned, MigrationTargetInvalid, BlockPoolExhausted, AllRanksAtCapacity, EmptyTokenSequence, NoPrefixAffinity, DimensionMismatch, Internal, ...
- [x] `handle.rs` -- `ParallelismConfig { tp, sp, ep }` 3-way decomposition with `world_size()`, `validate()`; `RankCoordinates` 3-D tp/sp/ep coords from flat global rank; `peer_tp/sp/ep()` for ring lookups; `DistInferHandle` lightweight descriptor with device, SM version, config, coords; `single_rank()` for tests
- [x] `lib.rs` -- module declarations, re-exports, 6 E2E integration tests

#### PTX Kernel Sources
- [x] `ptx_kernels.rs` -- 5 GPU-side collective kernels
  - `tp_col_scatter_ptx` -- column-parallel linear scatter: write strided shard into full output buffer
  - `tp_row_all_reduce_ptx` -- row-parallel linear all-reduce: ring partial-sum accumulation
  - `sp_seq_chunk_copy_ptx` -- sequence chunk copy: extract/insert contiguous token slice (direction=0/1)
  - `ep_token_scatter_ptx` -- expert-parallel token scatter: route tokens to expert-local input buffers
  - `ep_token_gather_ptx` -- expert-parallel token gather: collect expert outputs back to original order

#### Tensor Parallelism (`tensor_parallel/`)
- [x] `tensor_parallel/mod.rs` -- module organization
- [x] `tensor_parallel/column_parallel.rs` -- `ColumnLinearShard` weight shard `[local_out x in]`; `forward()` local GEMM; `validate()`; `ColumnLinear` `from_full_weight()` slices rows; `local_forward()`; `all_gather()` simulates collective
- [x] `tensor_parallel/row_parallel.rs` -- `RowLinearShard` weight shard `[out x local_in]`; `forward_partial()` local GEMM; bias only on rank 0; `RowLinear` `from_full_weight()` slices columns; `slice_input()`; `all_reduce()` simulates ring reduce

#### Sequence Parallelism (`sequence_parallel/`)
- [x] `sequence_parallel/mod.rs` -- module organization
- [x] `sequence_parallel/splitter.rs` -- `SeqSplitter` -- `extract_chunk()`, `insert_chunk()`, `all_gather()`, `reduce_scatter()`; validates divisibility; `ChunkInfo` describes rank's token window (start, len, total_tokens, hidden_dim)
- [x] `sequence_parallel/boundary.rs` -- `BoundaryExchange` -- pre-attention all-gather of K/V; post-attention reduce-scatter of outputs; `local_attention()` with causal masking and GQA-compatible head indexing

#### Expert Parallelism (`expert_parallel/`)
- [x] `expert_parallel/mod.rs` -- module organization
- [x] `expert_parallel/router.rs` -- `TopKRouter` top-K selection from gating logits + softmax weight normalisation; `RoutingPlan` with expert_load; `load_balance_cv()` metric; `RoutingEntry` / `RoutingPlan` per-(token, expert) assignment with routing weight
- [x] `expert_parallel/dispatch.rs` -- `LocalExpertBatch` dispatched token batch per expert with token_indices and weights; `ExpertDispatcher` -- `scatter()` -> local expert buffers; `gather()` -> weighted output sum; `dispatch_and_gather()` end-to-end

#### Distributed KV Cache (`distributed_cache/`)
- [x] `distributed_cache/mod.rs` -- module organization
- [x] `distributed_cache/partition.rs` -- `SeqOwnership` / `RankCacheStats` per-sequence owner rank + block count; per-rank utilization stats; `CachePartition` -- least-loaded assignment; `grow()`, `release()`; `rebalance_suggestions()` (utilization-threshold migration hints); `apply_migration()`
- [x] `distributed_cache/migration.rs` -- `BlockData` serialized KV block `[n_layers x 2 x block_size x kv_dim]`; `key_slice(l)` / `value_slice(l)`; `validate()`; `MigrationRequest` / `MigrationStats` cross-rank block transfer descriptor + statistics; `BlockMigrator` -- `receive_block()` -> local staging id; `take_block()`; `validate_target()`

#### Request Routing (`router/`)
- [x] `router/mod.rs` -- module organization
- [x] `router/request.rs` -- `Request` -- token_ids, max_new_tokens, priority; `prefix_hash(len)` FNV-1a for affinity lookup; `RoutingDecision` / `DispatchPolicy` selected rank + policy tag + prefix_hit flag
- [x] `router/policy.rs` -- `RankLoad` (free_blocks, total_blocks, in_flight); `utilization()`; `RouterMetrics` per-policy request counts, total_routed, prefix_hits, `prefix_hit_rate()`; `RoutingPolicy` -- three modes: RoundRobin, LeastLoaded, PrefixAffinity (with fallback + registration)

#### Integration Tests
- [x] 6 E2E tests in `lib.rs`:
  - `e2e_tp_column_row_roundtrip` -- tp=4 column-parallel + all-gather + row-parallel + all-reduce = identity
  - `e2e_sp_attention_pipeline` -- sp=2 extract chunks + all-gather + local_attention (uniform QKV -> output=1.0)
  - `e2e_ep_moe_dispatch_gather` -- ep=2, 4 experts, 4 tokens, top-1 routing + identity experts + gather
  - `e2e_cache_partition_lifecycle` -- 4 ranks, 8 sequences, assign/grow/release lifecycle
  - `e2e_routing_prefix_affinity_pipeline` -- first request misses, second with same prefix hits same rank
  - `e2e_ptx_kernels_all_sm_versions` -- all 5 kernels x 5 SM versions produce valid PTX headers

### Future Enhancements

#### P0 -- Critical (Parallelism Axes)
- [x] Tensor parallelism column + row variants (`tensor_parallel/`)
- [x] Sequence parallelism with K/V boundary exchange (`sequence_parallel/`)
- [x] Expert parallelism top-K router + dispatcher (`expert_parallel/`)
- [x] Per-rank ParallelismConfig validation (`handle.rs`)

#### P1 -- Important (Cache + Routing)
- [x] Distributed KV-cache partition with least-loaded assignment (`distributed_cache/partition.rs`)
- [x] Block migration with staging IDs (`distributed_cache/migration.rs`)
- [x] Three routing policies (RoundRobin / LeastLoaded / PrefixAffinity) (`router/policy.rs`)
- [x] FNV-1a prefix hashing for affinity routing (`router/request.rs`)

#### P2 -- Nice-to-Have (Scaling / Observability)
- [x] Load-balance CV metric for MoE router (`expert_parallel/router.rs::load_balance_cv`)
- [x] Per-policy router metrics with prefix hit-rate (`router/policy.rs::RouterMetrics`)
- [ ] (P2) Real NCCL-equivalent collective backend (currently simulated via in-process function calls)
- [ ] (P2) Pipeline parallelism axis (PP) for very deep models -- intentionally out of scope of v1
- [ ] (P2) Dynamic rebalancing trigger on load imbalance (rebalance_suggestions() exists; no autonomous trigger)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

(No CUDA crate deps -- `oxicuda-dist-infer` is a pure orchestration layer; collective kernels are emitted as PTX strings and executed by downstream callers via `oxicuda-driver`/`oxicuda-launch`. Real collectives require user-supplied NCCL-equivalent backend.)

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 80 passing (root TODO.md count)
- unwrap() calls: 0 (production code; test helpers use `.unwrap()` on infallible handle construction)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, all CPU reference simulations work; runtime collective backend returns `UnsupportedPlatform`

## Performance Targets

| Operation | Target |
|-----------|--------|
| `tp_col_scatter_ptx` -- 4096-hidden scatter on tp=8 | >= 90% bandwidth-limited peak on sm_80+ |
| `tp_row_all_reduce_ptx` -- 4096-hidden reduce on tp=8 | >= 80% of NCCL `ncclAllReduce` |
| `sp_seq_chunk_copy_ptx` -- 4096-token chunk copy | >= 95% bandwidth-limited peak |
| `ep_token_scatter_ptx` -- 256-token, 8-expert dispatch | >= 85% bandwidth-limited peak |
| `CachePartition::assign` -- 1k sequences, 16 ranks | < 10 us per assignment |
| `RoutingPolicy::route` -- PrefixAffinity, 1M-token cache | sub-microsecond lookup |

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86 / sm_89)
- [x] PTX header selection emits `.target sm_80` for cp.async-capable collective kernels
- [ ] cp.async-driven cross-rank K/V boundary exchange (deferred -- requires multi-GPU NCCL bring-up)

### Hopper (sm_90 / sm_90a)
- [x] PTX header selection emits `.target sm_90`
- [ ] TMA-driven multi-CTA all-reduce ring (deferred)
- [ ] Warp-specialized MoE dispatch with overlapped compute/transfer (deferred)

### NVLink / PCIe Bandwidth
- All collective kernels are designed to interoperate with downstream NCCL or UCX backends -- no in-crate NIC code.
- `CachePartition::rebalance_suggestions()` uses utilization-threshold heuristics that map directly to NVLink topology when available.

## Deepening Opportunities

### Verification Gaps
- [x] TP roundtrip identity verified (column-parallel + all-gather + row-parallel + all-reduce)
- [x] SP attention pipeline preserves uniform softmax output
- [x] EP MoE dispatch + gather is round-trip identity for top-1 routing + identity experts
- [x] PTX kernels validated for all 5 SM versions (sm_75 / sm_80 / sm_90 / sm_100 / sm_120)
- [x] Prefix-affinity routing exhibits >0 hit rate after registration
- [ ] Multi-rank end-to-end roundtrip on actual NVLink hardware (deferred -- single-process simulation only)
- [ ] Load-imbalance MoE stress (skewed expert load) verifies `load_balance_cv()` triggers rebalancing

### Implementation Deepening
- [x] `RankCoordinates` 3-D tp/sp/ep decomposition with peer lookups
- [x] `BoundaryExchange::local_attention` supports causal masking and GQA head indexing
- [x] `ExpertDispatcher::dispatch_and_gather` accepts user-supplied expert closure
- [x] `BlockMigrator` validates target rank before staging
- [ ] NCCL-equivalent collective backend (currently in-process simulation; needs real cluster integration)
- [ ] Dynamic per-rank scaling -- elastic add/remove of ranks during serving
- [ ] Pipeline parallelism (PP) axis -- intentionally out of scope of v1

## Notes

- All collective implementations in this crate are *simulations* over in-process buffers. Real multi-GPU execution requires plugging in a NCCL-equivalent collective backend (planned for a future `oxicuda-collective` crate).
- The PTX kernel strings are deployment-ready and exercise `.visible .entry`, `.target sm_*` headers, and ring-style partial-sum accumulation patterns.
- No benchmark harness configured (no `criterion` dev-dep) -- the 27-variant `DistInferError` surface and simulation correctness are the primary verification targets.
- Future integration with `oxicuda-infer` will expose distributed `ContinuousBatcher` via `RoutingDecision` + `CachePartition`.
