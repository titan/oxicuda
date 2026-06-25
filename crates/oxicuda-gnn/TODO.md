# oxicuda-gnn TODO

Pure-Rust Graph Neural Network primitives for OxiCUDA: sparse graph representations
(CSR/COO/heterogeneous), message passing, GCN/GAT/GATv2/GraphSAGE/GIN layers, global
and hierarchical pooling, Set2Set readout. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.18).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Files:** 54 Rust source files in `src/`
- **Tests:** 662 passing (#[test] count in src/)
- **Crate:** `oxicuda-gnn` -- Vol.18 Graph Neural Network Primitives

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `GnnError` (14 variants): `EmptyGraph`, `NodeIndexOutOfRange`,
      `EdgeIndexOutOfRange`, `InvalidLayerConfig`, `FeatureDimensionMismatch`,
      `InvalidEdgeWeight`, `InvalidPoolingK`, `SamplingError`, etc.; `GnnResult<T>`
- [x] `handle.rs` -- `SmVersion`, `GnnHandle`, `LcgRng`
- [x] `lib.rs` -- crate root with `prelude` module and 12 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 7 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `csr_spmv_ptx` -- `y[i] = sum A[i,j]*x[j]`; warp-per-row with `shfl.sync.down`
      butterfly reduction
- [x] `scatter_add_ptx` -- `out[idx[i]] += in[i]`; `atom.global.add.f32`
- [x] `gat_attention_ptx` -- `LeakyReLU(a^T [W x_i || W x_j])` per edge
- [x] `softmax_edge_ptx` -- numerically-stable per-source softmax over outgoing edges
- [x] `aggregate_mean_ptx` -- accumulator / `degree[i]` mean reduction
- [x] `gin_combine_ptx` -- `(1+eps)*x_i + sum x_j` self-loop GIN aggregator
- [x] `topk_score_ptx` -- `tanh(p^T x / ||p||)` scoring for Top-K node selection

#### Graph representations (`graph/`, 4 files + mod)
- [x] `graph/csr.rs` -- `CsrGraph`: `row_ptr`/`col_idx`/`edge_weight`; `from_edges()`,
      `neighbors()`, `degrees()`, `normalized_adjacency()` (`D_hat^-1/2 A_hat D_hat^-1/2`)
- [x] `graph/coo.rs` -- `CooGraph`: COO format with `to_csr()` conversion;
      symmetry detection
- [x] `graph/heterogeneous.rs` -- `HeterogeneousGraph`: multi-type node/edge relations
      (R-GCN style)
- [x] `graph/sampling.rs` -- `KHopSubgraph`, uniform random walk, Node2Vec biased walk
      (return-parameter p, in-out parameter q)

#### Message Passing (`message_passing/`, 3 files + mod)
- [x] `message_passing/aggregate.rs` -- sum / mean / max / min / softmax aggregations
      over neighbour messages
- [x] `message_passing/scatter.rs` -- `scatter_add`, `scatter_max`, `scatter_min`,
      `scatter_mul`, `scatter_softmax`
- [x] `message_passing/update.rs` -- update functions: MLP (2-layer), identity, ReLU,
      SiLU, LeakyReLU

#### GNN Layers (`layers/`, 5 files + mod)
- [x] `layers/gcn.rs` -- `GcnLayer`:
      `H^(l+1) = sigma(D_hat^-1/2 A_hat D_hat^-1/2 H^(l) W^(l))`
      (Kipf & Welling 2017)
- [x] `layers/gat.rs` -- `GatLayer`:
      `alpha_ij = softmax(LeakyReLU(a^T [W x_i || W x_j]))`, multi-head concat or mean
      (Velickovic 2018)
- [x] `layers/gat_v2.rs` -- `GatV2Layer`: dynamic attention
      `a^T LeakyReLU(W [x_i || x_j])` (Brody 2022)
- [x] `layers/sage.rs` -- `SageLayer`: mean / MaxPool / LSTM aggregators; optional
      L2-norm output (Hamilton 2017)
- [x] `layers/gin.rs` -- `GinLayer`: `(1+eps)*h_v + sum h_u` with 2-layer MLP;
      BatchNorm; trainable eps (Xu 2019)

#### Pooling (`pooling/`, 3 files + mod)
- [x] `pooling/global_pool.rs` -- `GlobalPool`: mean / max / sum / attention pooling
      to graph-level representation; batched graphs
- [x] `pooling/topk_pool.rs` -- `TopKPool`: Gao & Ji top-k node selection with
      `tanh(p^T x / ||p||)` scoring
- [x] `pooling/diff_pool.rs` -- `DiffPool`: differentiable hierarchical pooling
      `S = softmax(GNN(A,X))`, `X' = S^T X`, `A' = S^T A S`; LP + entropy regularisation
      losses

#### Readout (`readout/`, 1 file + mod)
- [x] `readout/set2set.rs` -- `Set2Set`: LSTM-based permutation-invariant readout
      `q_t = LSTM(q*_{t-1})`, `alpha_it = softmax(x_i^T q_t)`, `q*_t = [q_t || r_t]`
      (Vinyals 2016)

#### Integration tests (`lib.rs::tests`)
- [x] 12 E2E tests covering CSR, COO, scatter, GCN, SAGE, GIN, DiffPool, Top-K,
      sampling, Set2Set, plus PTX generation across 6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Performance and Correctness)
- [x] CSR-balanced SpMV (merge-based / merge-path, Merrill & Garland SC 2016) for highly
      skewed degree distributions (`ops/balanced_spmv.rs` -- `balanced_spmv` +
      `BalancedSpmvConfig`; CUB-style merge-path endpoint search + carry-out serial fix-up;
      bit-stable vs `CsrGraph::spmv`, validated on star / power-law graphs across segment counts)
- [x] Edge-parallel GAT (one logical warp per edge) for high-degree graphs
      (`ops/edge_parallel_gat.rs` -- `EdgeParallelGat` + `EdgeParallelGatConfig`; flat per-edge
      score / per-source scatter-softmax / scatter-aggregate; multi-head concat|mean; numerically
      identical to `GatLayer::forward`, validated on hub graphs)
- [x] Sparse-tensor backend integration with `oxicuda-sparse` for SpMM-based GCN
      (`layers/gcn.rs` -- `GcnLayer::forward_sparse` + `propagation_matrix`; aggregation
      routed through `oxicuda_sparse::host_csr::HostCsr` SpMV per feature column, bit-exact
      vs the dense `forward` on dyadic operators)
- [x] `scatter_softmax` numerical-stability test on >1M edges

#### P1 -- Important (Architecture Coverage)
- [x] Transformer-based GNN: `GraphTransformer` / Graphormer with edge-feature bias
- [x] PNA (Principal Neighbourhood Aggregation, Corso 2020)
- [x] EdgeConv (DGCNN, Wang 2019) point-cloud style layer
- [x] R-GCN multi-relational layer (layers/rgcn.rs -- Schlichtkrull 2018; per-relation message passing with in-degree normalization + basis decomposition W_r=Σ_b a_rb V_b + self-loop; one CsrGraph per relation)
- [x] Neighbour sampling for mini-batch training (GraphSAGE inductive style)
- [x] Cluster-GCN partitioning helper for very large graphs (`sampling/cluster_gcn.rs` --
      `ClusterGcn` / `Partition` / `BatchSubgraph`; deterministic balanced-BFS clustering, Chiang 2019)

#### P2 -- Nice-to-Have (Research / Advanced)
- [x] SIGN scalable inception graph network (`layers/sign.rs`) — Rossi 2020 ICML workshop: multi-hop diffusion pre-computation Aᵏ X followed by MLP over concatenated hop features; `SignConv`
- [x] GRAND graph random neural diffusion (`layers/grand.rs`) — Chamberlain 2021 ICML: GRAND-l linear-diffusion PDE message passing `x_{k+1}=(1-Δ)x_k+Δ·A_att·x_k` with row-stochastic scaled-dot-product attention operator; `GrandLayer` + `GrandConfig` (15 tests)
- [x] k-WL expressive GNN (`layers/k_wl_gnn.rs`) — Maron 2019 NeurIPS: order-2 invariant/equivariant network over node-pair tensors with the Bell(4)=15 equivariant basis ops + permutation-invariant graph readout; `KWlGnn` + `KWlConfig` / `PairOp` (18 tests)
- [x] SGC (Simple Graph Convolution, Wu 2019) closed-form precomputation
- [x] APPNP (predict-then-propagate) personalised PageRank propagation
- [x] JK-Net Jumping Knowledge aggregator (layers/jk_net.rs -- Xu 2018 ICML; Concat / MaxPool / forward-LSTM-attention aggregation of per-node representations across all layers)
- [x] DGI (Deep Graph Infomax) contrastive pre-training loss
- [ ] Mixed-precision GAT softmax (FP16 attention with FP32 accumulation) (requires GPU hardware: FP16 tensor path)
- [ ] CUDA-Graph capture of multi-layer GCN forward for inference (requires GPU hardware: CUDA-Graph capture)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-sparse | Host CSR SpMM (`HostCsr`) for the GCN sparse-tensor path | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader.

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 662 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

GNN workloads are dominated by SpMV / SpMM and atomic-add scatter. Per-kernel targets:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `csr_spmv_ptx` | OGB-arxiv (N=170K, E=1.2M), OGB-products (N=2.4M, E=62M) | P0 |
| `scatter_add_ptx` | E in {1M, 10M, 100M} | P0 |
| `gat_attention_ptx` | heads in {4, 8}, hidden in {64, 128} | P1 |
| `softmax_edge_ptx` | E up to 100M | P1 |
| `aggregate_mean_ptx` | mixed-degree distributions | P1 |
| `gin_combine_ptx` | MUTAG, PROTEINS, ENZYMES benchmark sizes | P2 |
| `topk_score_ptx` | N up to 10K, K in {0.1N, 0.5N} | P2 |

Target: >=85% peak DRAM throughput on SpMV, >=80% peak atomic throughput on scatter.

## Notes

- Graphs are CSR-native; COO is supported via `CooGraph::to_csr()` conversion
- `HeterogeneousGraph` stores per-relation `(src_type, edge_type, dst_type)` triples
- Sampling uses `LcgRng` for deterministic test reproducibility
- `DiffPool` LP + entropy regularisation losses returned alongside pooled `(X', A')`
- All edge-parallel kernels use `atom.global.add.f32` (loss of strict-order
  determinism is acceptable per standard GNN literature)
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `csr_spmv_ptx` warp-shuffle butterfly reduction (no shared-memory atomics)
- [x] `scatter_add_ptx` uses `atom.global.add.f32` (Ampere has 2x faster atomics vs Volta)
- [x] PTX × SM 80, 86 generation verified in integration tests
- [ ] `cp.async` prefetch of `col_idx` / `edge_weight` for SpMV (requires GPU hardware)
- [ ] Vectorised `ld.global.v4.f32` for dense node-feature loads in GAT (requires GPU hardware)

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for dense node-feature tile staging (requires GPU hardware)
- [ ] Distributed-shared-memory cluster aggregation for very high-degree nodes (requires GPU hardware)
- [ ] `wgmma.mma_async` for GAT QK^T attention path (requires GPU hardware)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) node-feature representation for inference (requires GPU hardware)
- [ ] Tensor-Memory (TMEM) staged dense-feature reduction (requires GPU hardware)

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] CSR / COO round-trip and symmetry-detection tests
- [x] `normalized_adjacency` correctness vs hand-computed
      `D^-1/2 A D^-1/2` on small graphs
- [x] GCN / GAT / GATv2 / SAGE / GIN forward-pass shape and finiteness tests
- [x] Multi-head GAT concat vs mean-aggregation tests
- [x] DiffPool soft-assignment row-stochastic property test
- [x] Top-K pooling node-selection ordering verified
- [x] Set2Set permutation invariance test
- [x] Random walk / Node2Vec biased walk distributional tests
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`) (requires GPU hardware)
- [ ] Numerical agreement with `torch_geometric` reference within 1e-4 relative (requires PyG reference / GPU)
- [ ] OGB-arxiv full-graph GCN inference accuracy match (within 0.5 % accuracy) (requires OGB dataset / GPU)
- [ ] Scalability test on OGB-products (2.4 M nodes) on multi-GPU (requires multi-GPU hardware)

### Implementation Deepening
- [ ] Multi-GPU partitioning helper (Metis-style or hash-partition) with edge replication (requires multi-GPU hardware; single-device balanced partitioning exists in `sampling/cluster_gcn.rs`)
- [x] Sparse-tensor backend: route `GcnLayer::forward` through `oxicuda-sparse` SpMM (`layers/gcn.rs::GcnLayer::forward_sparse` + `propagation_matrix`/`coo_to_host_csr` -- builds the normalised (or raw) adjacency Â as `oxicuda_sparse::host_csr::HostCsr` and aggregates Â·(HW) via `HostCsr::matvec` per feature column; HostCsr SpMM path == dense `forward`, elementwise ≤1e-9 (bit-exact, max err 0.0 on dyadic operators); GPU-SpMM / tensor-core variant still `[ ]`)
- [ ] Mini-batch training loop helper (random-walk sampling + neighbour-batch assembly) (sampling primitives exist in `sampling/`; full train-loop orchestration is application-level)
- [x] Heterogeneous-graph message-passing dispatch (per-relation weight matrices)
      (`layers/hetero_conv.rs` -- `HeteroConv` / `HeteroConvConfig` / `HeteroConvWeights`;
      PyG-`HeteroConv`-style dispatch over a `HeteroGraph`'s named typed relations -- per-relation
      `H_s·W_r` projection scattered source→destination with R-GCN in-degree mean normalisation +
      optional per-dst-type self-loop; output map keyed by destination type)
- [x] Edge-feature support in `GatLayer` and `MessagePassing` interface
      (`layers/gat.rs` -- `GatLayer::forward_with_edges` / `forward_inner` / `EdgeAttention`:
      attention logit `e_ij = LeakyReLU(a_src^T Wx_i + a_dst^T Wx_j + a_edge^T (W_e edge_ij))`,
      per-head `W_e` edge projection into head space, edges indexed in CSR order; the edge-free
      `forward` delegates to the shared core unchanged -- bit-for-bit regression-safe.
      `message_passing/aggregate.rs` -- `build_edge_messages`: MPNN `message(x_src, edge_attr) =
      W_x·x_src [+ W_e·edge]` step threading an optional edge attribute through the message
      function before any `aggregate_*` / `scatter_*` reducer. GATv2 keeps its own dynamic-attention
      path and is out of scope for this item. `EdgeConv` / `GraphTransformer` already consume edge
      features.)

### Benchmark Coverage
- [x] `benches/gnn_ops.rs` Criterion harness wired (CPU-side PTX generation + layer
      forward on small graphs)
- [ ] GPU-side throughput numbers vs reference (cuSPARSE SpMV, DGL/PyG layer) once
      Linux+NVIDIA harness is available (requires GPU hardware)
- [ ] Degree-distribution sensitivity sweep on power-law graphs (bench-only; CPU `balanced_spmv`
      already validated across skewed/star graphs in `ops/balanced_spmv.rs` tests)
