# oxicuda-tn

Tensor Networks -- a pure Rust library for MPS, MPO, PEPS, DMRG, TEBD, and
tensor decompositions.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-tn` is a pure-Rust tensor-network library covering the
one-dimensional and two-dimensional families used throughout quantum
many-body physics, quantum simulation, and machine-learning compression.
All linear algebra (including SVD) is implemented in-crate with no external
linear-algebra dependencies, and random sampling uses the workspace `LcgRng`
(MMIX LCG) for deterministic reproducibility.

The MPS layer provides finite and quantum-number-symmetric matrix product
states, left/right/mixed canonicalisation, truncation, and the iTEBD
infinite-bond evolver. MPOs support construction, auto-compression, and
contraction against MPS. Ground-state solvers include single-site DMRG,
two-site DMRG (with Lanczos), iDMRG, and two-site excited-state DMRG. Time
evolution covers TEBD with configurable Trotter splittings, mixed-state /
density-MPO TEBD, and an iTEBD driver for infinite chains. A high-level
`Circuit` interface compiles quantum-gate sequences to TEBD gates.

Two-dimensional PEPS is supported with simple-update and CTMRG boundary
contraction. Decompositions include CP/PARAFAC via ALS (and a non-negative
variant), Tucker via HOSVD/HOOI/ST-HOSVD, Tensor-Train via TT-SVD (Oseledets)
and TT-cross with maxvol pivoting, plus TT-ALS regression. The contraction
engine provides binary einsum, optimal contraction-path search by dynamic
programming with greedy comparison, and network simplification (traces,
pairwise contraction). SVD is offered as both Jacobi rotation and Householder
bidiagonalisation followed by implicit-QR (Golub-Reinsch).

## Modules

| Module | Description |
|--------|-------------|
| `svd` | Jacobi and Householder/Golub-Reinsch SVD, truncated SVD |
| `mps` | `Mps`, `MpsTensor`, canonicalisation, truncation, symmetric MPS, iTEBD |
| `mpo` | Matrix Product Operators with auto-compression and MPO-MPS contraction |
| `dmrg` | Single-site, two-site, two-site-excited, infinite DMRG, finite-T purification, Lanczos solvers |
| `tebd` | TEBD with Trotter splittings, mixed-state / density-MPO TEBD |
| `peps` | 2D PEPS with simple update and CTMRG boundary contraction |
| `tt` | Tensor-Train (Oseledets): TT-SVD, TT-cross with maxvol, TT-ALS regression |
| `tucker` | HOSVD, HOOI, ST-HOSVD Tucker decompositions |
| `cp` | CP / PARAFAC alternating least squares and non-negative CP |
| `contraction` | Binary einsum, optimal contraction-path DP search, network simplification |
| `circuits` | Quantum-circuit interface compiled to TEBD gates |
| `metrics` | Bond dimensions, entanglement entropy, Schmidt spectrum, fidelity, structure factor, Loschmidt echo |
| `handle` | `TnHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernel strings for tensor-network operations |
| `error` | `TnError` / `TnResult` |

## Quick Start

```rust,no_run
use oxicuda_tn::mps::mps::Mps;
use oxicuda_tn::error::TnResult;

fn main() -> TnResult<()> {
    // Build a 4-site, d=2 product state |1010>.
    let local = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let mps = Mps::from_product_state(&local)?;

    println!("n_sites = {}", mps.n_sites());
    println!("norm    = {}", mps.norm()?);

    // Pauli-Z on every site -> total Sz_total = sum of ±1 amplitudes.
    let pauli_z: Vec<Vec<f64>> = (0..mps.n_sites())
        .map(|_| vec![1.0, 0.0, 0.0, -1.0])
        .collect();
    let sz = mps.expectation_local(&pauli_z)?;
    println!("<Z⊗...⊗Z> = {sz}");
    Ok(())
}
```

## Status

**Alpha** -- 28,138 SLoC, 540 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
