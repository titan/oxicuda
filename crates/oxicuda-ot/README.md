# oxicuda-ot

Optimal Transport primitives in pure Rust, paired with PTX kernels emitted at runtime for OxiCUDA.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-ot` covers the canonical Optimal Transport algorithm spectrum:
entropic OT (Sinkhorn-Knopp and a family of stabilised / accelerated
variants), exact OT via the network simplex method (plus the closed-form
1D EMD), Wasserstein-1 / 2 distances and sliced / max-sliced
approximations, Gromov-Wasserstein and Fused-GW for unaligned domains,
KL-relaxed Unbalanced OT and TV / L2-relaxed Partial OT, Wasserstein
barycenters (free and fixed support), JKO gradient flow, Schrödinger
Bridge / IPF, multi-marginal OT, Wasserstein k-means clustering, and
OT-based domain adaptation (barycentric mapping).

The implementation is from-scratch in safe Rust, with no external numerical
libraries. The Sinkhorn family alone includes log-stabilised, debiased,
low-rank, screened, momentum-accelerated, EMA, Greenkhorn, anchor-partial,
and conditional-gradient-Wasserstein variants. Each domain module is
paired with PTX kernels emitted at runtime, parametric in the device SM
version (SM 7.5 through SM 10.0).

## Modules

| Module | Description |
|--------|-------------|
| `sinkhorn` | Entropic OT: Sinkhorn-Knopp (log-stab), divergence, low-rank, screened, momentum, EMA, Greenkhorn, anchor-partial, CG-Wasserstein |
| `exact` | Exact OT via network simplex method (`emd`); closed-form 1D EMD |
| `wasserstein` | W1, W2, sliced and max-sliced Wasserstein distances |
| `gromov` | Entropic Gromov-Wasserstein, Fused-GW, batched-GW |
| `unbalanced` | KL-relaxed unbalanced OT and TV / L2-relaxed partial OT |
| `barycenter` | Free-support and fixed-support Wasserstein barycenters |
| `jko` | Jordan-Kinderlehrer-Otto proximal scheme for Wasserstein gradient flows |
| `bridge` | Schrödinger Bridge / Iterative Proportional Fitting (IPF) |
| `multi` | Multi-marginal OT via tensor scaling |
| `clustering` | Wasserstein k-means and Sinkhorn k-means |
| `domain` | OT-based domain adaptation via barycentric mapping |
| `metrics` | Marginal violation, transport cost, entropy, KL diagnostics |
| `handle` | `OtHandle`, `SmVersion`, `LcgRng` |
| `error` | `OtError` / `OtResult` |
| `ptx_kernels` | Runtime PTX strings for Sinkhorn / Gromov / barycenter per SM version |

## Quick Start

```rust,no_run
use oxicuda_ot::sinkhorn::sinkhorn::{sinkhorn, SinkhornConfig};
use oxicuda_ot::error::OtResult;

fn main() -> OtResult<()> {
    // Cost matrix (m × n), row-major, plus source/target histograms.
    let m: usize = unimplemented!();
    let n: usize = unimplemented!();
    let c: Vec<f32> = unimplemented!();
    let a: Vec<f32> = unimplemented!();
    let b: Vec<f32> = unimplemented!();

    let cfg = SinkhornConfig::default();
    let out = sinkhorn(&c, &a, &b, m, n, &cfg)?;
    println!("transport cost = {}", out.cost);
    Ok(())
}
```

## Design Notes

- Pure Rust, no external numerical / linear-algebra dependencies. Network
  simplex, log-domain Sinkhorn, and the GW alternating minimisation are
  all written from scratch.
- The Sinkhorn family is implemented with numerical-stability primitives
  (log-sum-exp updates, dual-potential stabilisation, low-rank
  factorisations, screening) so high-regularisation (small `eps`) regimes
  remain well-conditioned.
- Each domain module is paired with PTX kernels emitted at runtime,
  parametric in the device SM version (SM 7.5 through SM 10.0). The CPU
  implementation is the reference oracle for the matching GPU kernel.

## Status

**Alpha** — 26,462 SLoC, 657 passing tests. API may evolve before v1.0.

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
