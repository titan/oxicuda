# oxicuda-graph TODO

High-level DAG-based computation graph engine that sits above the raw CUDA driver and lowers to `oxicuda_driver::graph::Graph` for low-overhead CUDA graph submission. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.7).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 6,624 SLoC across 17 files**

Vol.7 models GPU workloads as directed acyclic graphs (DAGs) of kernel
launches, memcopies, and event syncs. It runs a suite of analysis and
optimisation passes — topological scheduling, liveness, dominance,
operator fusion, memory planning, stream partitioning — then emits an
`ExecutionPlan` that can be executed by a CPU-side sequential simulator
or captured into a real `oxicuda_driver::graph::Graph`. The crate is
`[COMPLETE]`; outstanding work is GPU-hardware verification and end-to-end
integration with the rest of the Vol.3–Vol.6 kernel crates.

### Completed [x]

#### Core scaffolding
- [x] `lib.rs` -- crate root, prelude, `#![forbid(unsafe_code)]`, ASCII architecture diagram
- [x] `error.rs` -- `GraphError` (10 variants: CycleDetected, NodeNotFound, BufferNotFound, InvalidStream, FusionRejected, MemoryOverflow, PlanCorruption, ExecutorMismatch, DriverError, ValidationFailed), `GraphResult<T>` alias
- [x] `node.rs` -- `NodeId`, `BufferId`, `StreamId`, `KernelConfig`, `MemcpyDir`, `NodeKind` (8 variants: Kernel, Memcpy, Memset, EventRecord, EventWait, StreamSync, GraphHostFn, Empty), `GraphNode`, `BufferDescriptor`
- [x] `graph.rs` -- `ComputeGraph` DAG: adjacency-list representation, eager cycle detection, Kahn topological sort, reachability matrix, DOT export, predecessor / successor queries
- [x] `builder.rs` -- `GraphBuilder` fluent API: `alloc_buffer`, `add_kernel(...).fusible(...).inputs(...).outputs(...).finish()`, `add_memcpy`, `add_memset`, `chain`, auto data-flow edge inference from buffer I/O annotations, `.build() -> GraphResult<ComputeGraph>`

#### Analysis passes (`analysis/`)
- [x] `analysis/topo.rs` -- topological analysis: ASAP / ALAP scheduling, slack computation, critical-path identification, level assignment, priority ordering
- [x] `analysis/liveness.rs` -- buffer live intervals `[def_pos, last_use_pos]` in topological order, interference-pair enumeration, peak-live-bytes computation
- [x] `analysis/dominance.rs` -- Cooper / Harvey / Kennedy dominator tree, `idom`, `dominates()`, `dominated_by()`, lowest-common-ancestor (LCA)

#### Optimisation passes (`optimizer/`)
- [x] `optimizer/fusion.rs` -- operator fusion: greedy chain fusion of compatible element-wise kernels using dominator + per-kernel `fusible` flag + config checks; produces `FusionGroup` clusters
- [x] `optimizer/memory.rs` -- memory planning: live-interval graph colouring (best-fit), 256-byte alignment, pool layout, offset assignment per buffer
- [x] `optimizer/stream.rs` -- stream partitioning: list-scheduling heuristic with predecessor-aware stream assignment, cross-stream event synchronisation detection

#### Executor backends (`executor/`)
- [x] `executor/plan.rs` -- `ExecutionPlan` — full compilation pipeline (topo → liveness → fusion → memory → stream → linearise), `PlanStep` sequence with event record / wait pairs, validation
- [x] `executor/sequential.rs` -- CPU-side `SequentialExecutor` simulator: event validity checking, deterministic step iteration, `ExecutionStats` (`kernels_launched`, `memcopies`, `events`, `streams_used`)
- [x] `executor/cuda_graph.rs` -- `CudaGraphCapture` — converts `ExecutionPlan` to `oxicuda_driver::graph::Graph` with dependency edges, ready for `GraphExec::launch`

### Future Enhancements [ ]

#### P0 — Critical (Already Implemented)
- [x] Eager cycle detection in `GraphBuilder` (rejects invalid graphs at build time)
- [x] Multi-stream partitioning with cross-stream event sync
- [x] Element-wise operator fusion driven by dominator information
- [x] Memory planning with interval-graph colouring (peak-live-bytes minimisation)
- [x] CPU-side sequential simulator for testing without GPU

#### P1 — Important (Already Implemented)
- [x] ASAP / ALAP scheduling for slack-aware scheduling decisions
- [x] Critical-path-aware priority ordering
- [x] DOT export for visual debugging of large graphs
- [x] CUDA Graph capture via `oxicuda_driver::graph::Graph` round-trip
- [x] `serde` feature for (de)serialising `ComputeGraph` / `ExecutionPlan` artifacts

#### P2 — Nice-to-have (Already Implemented)
- [x] Empty node (`NodeKind::Empty`) as fence point
- [x] Memset, host-function, and stream-sync nodes
- [x] `chain(&[node_a, node_b, ...])` convenience for linear pipelines

#### Outstanding — Hardware Verification & Integration
- [ ] (P0) End-to-end test: build a Vol.3 GEMM + Vol.6 softmax pipeline as a `ComputeGraph`, capture to driver graph, launch on Linux + NVIDIA
- [ ] (P1) Benchmark CUDA-graph submission overhead vs. naive stream submission for repeated launches (target: ≥ 5× reduction in launch overhead for graphs with 10+ nodes)
- [ ] (P1) Fusion pass extended beyond element-wise to support reduction-chained patterns (e.g., `LayerNorm` = mean + variance + scale)
- [ ] (P1) Stream partitioner aware of NVLink topology when scheduling cross-GPU edges (depends on `oxicuda-driver::nvlink`)
- [ ] (P2) Conditional / loop nodes (CUDA 12.4+ graph conditional nodes) — currently DAG-only

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper + `graph::Graph` lowering target | Yes (runtime FFI only) |
| oxicuda-ptx | PTX code generation DSL (for fused-kernel emission) | Yes |
| thiserror | Error derive macros | Yes |
| serde (optional, `serde`) | (De)serialisation of `ComputeGraph` / `ExecutionPlan` | Yes |
| criterion (dev-dep) | `graph_build` benchmark harness | Yes |

## Quality Status

- Warnings: 0 (clippy + rustdoc clean)
- Tests: 175 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` enforced crate-wide
- All public APIs return `GraphResult<T>` for fallible paths

## Functional Coverage (derived from `lib.rs`)

| # | Capability | Module | Status |
|---|------------|--------|--------|
| G1 | Ergonomic graph construction with auto data-flow inference | `builder.rs` | [x] |
| G2 | DAG representation with cycle detection + topo sort | `graph.rs` | [x] |
| G3 | Kernel / memcpy / memset / event / host-fn node kinds | `node.rs` | [x] |
| G4 | ASAP / ALAP / slack / critical-path scheduling | `analysis/topo.rs` | [x] |
| G5 | Liveness analysis (intervals, interference, peak bytes) | `analysis/liveness.rs` | [x] |
| G6 | Dominator tree (Cooper et al.) | `analysis/dominance.rs` | [x] |
| G7 | Greedy element-wise operator fusion | `optimizer/fusion.rs` | [x] |
| G8 | Buffer memory planning via interval-graph colouring | `optimizer/memory.rs` | [x] |
| G9 | Multi-stream partitioning with cross-stream sync | `optimizer/stream.rs` | [x] |
| G10 | Full compilation: topo → liveness → fusion → memory → stream → linearise | `executor/plan.rs` | [x] |
| G11 | CPU-side sequential simulator | `executor/sequential.rs` | [x] |
| G12 | CUDA-graph capture lowering | `executor/cuda_graph.rs` | [x] |
| G13 | DOT export for visualisation | `graph.rs` | [x] |

## Performance Targets

| # | Operation | Target | Status |
|---|-----------|--------|--------|
| T1 | Graph build (100 nodes) | < 100 µs | [x] (criterion `graph_build`) |
| T2 | Full compile (100 nodes, all passes) | < 1 ms | [x] (criterion `graph_build`) |
| T3 | CUDA-graph submission overhead vs. stream submission | ≥ 5× reduction for ≥ 10-node graphs | [~] Awaiting Linux + NVIDIA verification |
| T4 | Peak-live-bytes reduction vs. naive allocation | ≥ 30% on transformer-block workloads | [~] Awaiting integration with oxicuda-blas pipelines |

## Algorithmic Notes

- **Topological sort**: Kahn's algorithm with stable iteration order for reproducible plans
- **Dominator tree**: Cooper / Harvey / Kennedy iterative algorithm (preferred over Lengauer-Tarjan for small CFGs and simpler implementation)
- **Memory planning**: live-interval colouring with first-fit decreasing on interval length, 256-byte alignment for coalesced loads, returns `(buffer_id, offset, size)` triples
- **Fusion**: greedy chain merging — two nodes fuse iff (a) both `fusible(true)`, (b) `b` is the only consumer of `a`'s outputs, (c) `a` dominates `b`, (d) `b`'s only predecessor (data-wise) is `a`
- **Stream partitioning**: list-scheduling heuristic; node assigned to predecessor's stream when possible, otherwise lowest-load stream, otherwise new stream up to `max_streams`

## Architecture-Specific Deepening Opportunities

### Hopper (sm_90)
- [ ] Cluster-launch nodes (`cuLaunchKernelEx` with cluster dims) as a new `NodeKind::ClusterKernel`
- [ ] TMA descriptor lifetime tracked as a `BufferDescriptor`

### Blackwell (sm_100 / sm_120)
- [ ] Distributed-shared-memory (DSMEM) lifetimes modelled as cross-CTA-cluster live intervals
- [ ] Graph conditional nodes (CUDA 12.4+) for runtime-data-dependent control flow

## Deepening Opportunities

> Items marked `[x]` above represent API-surface coverage. These represent the gap
> between current implementation depth and production-grade graph-engine requirements.

### Verification Gaps
- [x] Round-trip property: `build(plan(g)) ≡ g` for random DAGs (proptest-style)
- [x] Cycle-detection rejects every cyclic permutation tested
- [x] Dominator-tree invariants checked against textbook examples
- [x] Sequential executor matches manually-computed `ExecutionStats` on canonical pipelines
- [ ] CUDA-graph capture round-trip verified on Linux + NVIDIA hardware (`cudaGraphLaunch` → memory contents match `SequentialExecutor` simulation)
- [ ] Stress test: 10K-node graph builds, compiles, and captures in < 1 s
- [ ] Fusion correctness verified by lowering fused groups to PTX via `oxicuda-ptx` and round-tripping outputs

### Coverage
- [x] DOT export for debugging large graphs
- [x] `serde` round-trip for `ComputeGraph` and `ExecutionPlan` (when `serde` feature enabled)
- [ ] Integration tests that build graphs out of `oxicuda-blas` GEMM + `oxicuda-signal` STFT + `oxicuda-dnn` activation chains
- [ ] Multi-GPU graph partitioning (post-Vol.7; depends on `oxicuda-driver::multi_gpu` + `nvlink`)
- [ ] Graph-level autotuning: select tile configs for kernels based on graph-wide memory pressure (depends on `oxicuda-autotune`)
