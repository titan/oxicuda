# oxicuda-mamba

State Space Model primitives for OxiCUDA -- S4 (HiPPO-LegS / DPLR), Mamba
selective scan (S6), Mamba-2 (SSD), and RWKV time-mixing, in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-mamba` provides the four canonical state-space architectures from
the recent SSM literature, plus the underlying numerical primitives every
SSM needs: Zero-Order-Hold and bilinear discretization, an associative
parallel scan that respects the `(A, b)` operator, and a tiny SSM kernel for
unit testing. Each architecture is offered both as a single-block module
(layer-level) and as full model definitions where applicable, so that the
same code can drive both kernel-level benchmarks and end-to-end language /
sequence model decoding tests.

PTX strings (`selective_scan`, `parallel_scan`, `depthwise_conv1d`,
`wkv_forward`, `ssd_chunk`, `hippo_legendre`, `rms_norm_silu`) are emitted
for SM 7.5 through SM 12.0. The crate is 100 % pure Rust by default; the
only required dependency is `thiserror`.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `MambaError` / `MambaResult` |
| `handle` | `MambaHandle`, `SmVersion`, `LcgRng` |
| `ssm::discretize` | `discretize`, `Discretization::{Zoh, Bilinear, Euler}` |
| `ssm::parallel_scan` | `ScanPair`, `inclusive_scan`, `exclusive_scan`, `ssm_state_scan` |
| `ssm::ssm_kernel` | `SsmKernel`, `SsmConfig` reference forward pass |
| `s4::hippo` | `hippo_legs`, `hippo_legs_diag`, `hippo_nplr` matrices |
| `s4::dplr` | `Dplr` diagonal-plus-low-rank parameterization |
| `s4::s4_layer` | `S4Layer`, `S4Config`, `S4Weights`, `naive_conv1d` reference |
| `mamba::selective_scan` | `selective_scan` (S6), `SelectiveScanConfig`, `softplus` |
| `mamba::mamba_block` | `MambaBlock`: causal depthwise conv + gating + SSM |
| `mamba::mamba_model` | `MambaModel`, `MambaConfig::tiny()`, `next_token` |
| `mamba2::ssd` | `ssd_naive`, `ssd_recurrent`, `verify_ssd_equivalence` |
| `mamba2::chunk_scan` | `chunk_scan`, `ChunkConfig` chunk-wise SSD |
| `mamba2::mamba2_block` | `Mamba2Block`, `Mamba2BlockConfig`, multi-head SSD |
| `rwkv::time_mixing` | `TimeMixingLayer`, `WkvState`, numerically stable WKV recurrence |
| `rwkv::channel_mixing` | `ChannelMixingLayer`, `square_relu` gated FFN |
| `rwkv::rwkv_block` | `RwkvBlock` complete pre-norm residual block |
| `ptx_kernels` | PTX strings for the seven SSM/Mamba/RWKV kernels |

## Quick Start

```rust,no_run
use oxicuda_mamba::prelude::*;

// HiPPO-LegS NPLR decomposition: stable initialization for an S4 layer.
let (lambda, p, q) = hippo_nplr(8)?;
assert!(lambda.iter().all(|&v| v < 0.0)); // stable eigenvalues

// Mamba (S6) selective scan over a [B=2, L=8, D=4] input with state size N=4.
let mut rng = LcgRng::new(42);
let cfg = SelectiveScanConfig::new(2, 8, 4, 4)?;
let len = 2 * 8 * 4;
let (mut u, mut delta) = (vec![0.0_f32; len], vec![0.0_f32; len]);
rng.fill_normal(&mut u);
rng.fill_normal(&mut delta);
let a_log = vec![0.1_f32; 4 * 4];
let y = selective_scan(&u, &delta, &a_log, &u, &u, &cfg)?;
assert_eq!(y.len(), len);
# Ok::<(), MambaError>(())
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
