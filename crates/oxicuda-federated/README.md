# oxicuda-federated

Federated learning primitives for OxiCUDA -- FedAvg / FedProx / SCAFFOLD /
FedAdam server algorithms, communication compression, differential privacy,
secure aggregation, and client selection, all in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-federated` covers the algorithmic core of federated learning: server
aggregation, gradient compression for bandwidth-limited clients, formal
differential-privacy mechanisms with RDP / Moments accountants, Shamir-based
secure aggregation, and client-selection strategies. The crate is a CPU
reference plus PTX kernel emitter -- there is **no CUDA SDK dependency** at
build time and the only crate dependency is `thiserror`.

Algorithms are exposed as plain functions and stateful structs operating on
flat `Vec<f32>` gradients, so they compose cleanly with any tensor library.
PTX kernels (`fedavg_weighted_sum`, `dp_clip_gradient`, `gaussian_noise`,
`qsgd_quantize`, `topk_mask`, `pairwise_mask`, `aggregate_mean`) are emitted
for SM 7.5 through SM 12.0.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `FedError` / `FedResult` |
| `handle` | `FedHandle`, `SmVersion`, `LcgRng` |
| `algorithm::fedavg` | `FedAvgState`, `FedAvgConfig` weighted server aggregation |
| `algorithm::fedprox` | `FedProxConfig`, `proximal_loss`, `proximal_gradient`, client correction |
| `algorithm::scaffold` | `ScaffoldState`, `ScaffoldClientState`, control-variate update |
| `algorithm::fedadam` | `FedAdamState` adaptive server optimizer (Yogi / Adam variants) |
| `compression::powersgd` | `PowerSgdCompressor` low-rank approximation with residual feedback |
| `compression::quantize` | QSGD `stochastic_quantize` / `dequantize`, error bounds |
| `compression::randomk` | `random_sparsify`, compression-ratio helper |
| `compression::topk` | `topk_sparsify`, `error_feedback` accumulator |
| `privacy::gaussian` | `GaussianMechanism` (clip + noise) |
| `privacy::laplacian` | `LaplacianMechanism`, `add_laplacian_noise` |
| `privacy::moments` | `MomentsAccountant` for DP-SGD |
| `privacy::rdp` | `rdp_gaussian`, `compose_rdp`, `rdp_to_dp`, `optimal_epsilon` |
| `privacy::pate` | `PateConfig`, `noisy_voting`, data-dependent epsilon |
| `secure_agg::shamir` | Shamir secret sharing with `share_scalar` / `reconstruct_*` |
| `secure_agg::masking` | Pairwise / single masking primitives |
| `secure_agg::aggregator` | High-level `SecureAggregator` orchestrator |
| `selection::random` | `random_select`, `stratified_select` client samplers |
| `ptx_kernels` | GPU PTX strings for the seven kernels above |

## Quick Start

```rust,no_run
use oxicuda_federated::prelude::*;

// Server holds 3 global parameters; two clients submit (params, weight) updates.
let mut state = FedAvgState::new(3);

let client_a = (vec![0.10_f32, 0.20, 0.30], 1.0);
let client_b = (vec![0.30_f32, 0.20, 0.10], 1.0);

state.aggregate(&[client_a, client_b])?;
assert_eq!(state.global_params.len(), 3);
# Ok::<(), FedError>(())
```

## Status

| Item | Value |
|------|-------|
| Version | 0.3.0 |
| Release date | 2026-06-25 |
| Default features | Pure Rust (`thiserror` only) |
| `unwrap()` | 0 in production code |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
