# oxicuda-gnn

Graph Neural Network primitives for OxiCUDA -- sparse graph representations,
message passing, GCN / GAT / GraphSAGE / GIN layers, global and hierarchical
graph pooling, in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-gnn` provides the structural primitives needed for graph deep
learning: CSR / COO / heterogeneous graph storage with neighbourhood
sampling, the scatter / gather / aggregate / segment-softmax foundation that
sits underneath every message-passing layer, four canonical GNN layers
(GCN, GAT, GAT-v2, GraphSAGE, GIN), three pooling strategies (global, Top-K,
DiffPool), a Set2Set readout, and the corresponding PTX kernels for SM 7.5
through SM 12.0.

The reference implementation runs entirely on CPU `Vec<f32>` so the same
code paths can be exercised in tests, benchmarks, and CPU-only deployments.
The only crate dependency is `thiserror` -- there is **no CUDA SDK
requirement** at build time.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `GnnError` / `GnnResult` |
| `handle` | `GnnHandle`, `SmVersion`, `LcgRng` |
| `graph::csr` | `CsrGraph` -- compressed sparse row, `spmv`, `degree` |
| `graph::coo` | `CooGraph` -- COO format, `to_csr` round-trip |
| `graph::heterogeneous` | `HeteroGraph` multi-relation graphs |
| `graph::sampling` | `NeighborhoodSampler`, `random_walk`, `biased_walk` |
| `message_passing::scatter` | `scatter_add`/`max`/`min`/`mul`, `gather`, `segment_softmax` |
| `message_passing::aggregate` | `aggregate_mean`/`max`/`sum`/`softmax`, degree-normalised aggregation |
| `message_passing::update` | `LinearUpdate`, `MlpUpdate`, ReLU / LeakyReLU / PReLU / ELU |
| `layers::gcn` | `GcnLayer`, `GcnConfig` -- normalised Kipf-Welling GCN |
| `layers::gat` / `layers::gat_v2` | `GatLayer`, `GatV2Layer` -- multi-head attention |
| `layers::sage` | `SageLayer`, `SageAggregator::{Mean, Max, Pool}` |
| `layers::gin` | `GinLayer`, `GinConfig` -- learnable epsilon |
| `pooling::global_pool` | global mean / max / sum / attention pooling |
| `pooling::topk_pool` | `TopKPool` learnable node selection |
| `pooling::diff_pool` | `DiffPool` differentiable hierarchical clustering |
| `readout::set2set` | `Set2Set` permutation-invariant readout |
| `ptx_kernels` | PTX for `csr_spmv`, `scatter_add`, `gat_attention`, `softmax_edge`, `aggregate_mean`, `gin_combine`, `topk_score` |

## Quick Start

```rust,no_run
use oxicuda_gnn::prelude::*;

// Triangle graph 0-1-2-0.
let g = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)])?;

// Sparse matrix-vector multiply on a 1-D feature.
let x = vec![1.0_f32, 2.0, 3.0];
let y = g.spmv(&x, 1)?;
assert_eq!(y.len(), 3);

// Scatter-add four messages onto two destination nodes (feat_dim = 2).
let messages = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let dest_idx = vec![0_usize, 0, 1, 1];
let pooled = scatter_add(&messages, &dest_idx, 2, 2)?;
assert_eq!(pooled.len(), 4);
# Ok::<(), GnnError>(())
```

## Status

| Item | Value |
|------|-------|
| Version | 0.2.0 |
| Release date | 2026-06-16 |
| Default features | Pure Rust (`thiserror` only) |
| `unwrap()` | 0 in production code |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
