# oxicuda-nas

Neural Architecture Search primitives for OxiCUDA -- DARTS, one-shot
supernets, evolutionary NSGA-II search, and hardware-aware predictors, all
in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-nas` covers the three dominant families of modern Neural
Architecture Search:

* **Differentiable (DARTS)** -- a continuous relaxation over a fixed
  primitive set with a bilevel optimizer over weights and architecture
  parameters. Discrete cells / networks are derived from the soft mixture
  weights at the end of search.
* **Evolutionary** -- multi-objective NSGA-II with non-dominated sorting,
  crowding-distance ranking, tournament selection, and an `ArchEncoding`
  representation suitable for crossover and mutation.
* **One-shot supernets** -- a single weight-shared `Supernet` that supports
  per-iteration path sampling and Slimmable-Net width scaling, alongside
  `BnStats` and width multipliers for hardware-aware deployment.

Eight DARTS primitive kinds are provided in `ops`, and seven PTX kernels
(`mixed_op_blend`, `arch_softmax`, `arch_grad`, `gumbel_softmax`,
`flops_accumulate`, `pareto_dominate`, `crossover_uniform`) are emitted for
SM 7.5 through SM 12.0. The only crate dependency is `thiserror`.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `NasError` / `NasResult` |
| `handle` | `NasHandle`, `SmVersion`, `LcgRng` |
| `ops::primitives` | `OpKind`, `OpWeights` -- the 8 DARTS primitive ops |
| `ops::mixed_op` | `MixedOp` continuous relaxation over primitives |
| `ops::search_space` | `SearchSpace`, `CellSpace`, `NetworkSpace` |
| `darts::cell` | `DartsCell` searchable cell |
| `darts::network` | `DartsNetwork` stack of cells |
| `darts::bilevel` | `BilevelOptimizer`, `BilevelConfig` |
| `darts::derive` | `derive_discrete_cell`, `derive_network`, `DiscretizedCell`, `DiscretizedNetwork` |
| `evolution::encoding` | `ArchEncoding` flat representation for crossover / mutation |
| `evolution::nsga2` | `Individual`, `fast_non_dominated_sort`, `crowding_distance`, `nsga2_select`, `tournament_select` |
| `evolution::population` | `Population` evolutionary container |
| `supernet::weight_share` | `Supernet` weight-shared supernet |
| `supernet::path_sample` | `PathSampler`, `SamplingStrategy` |
| `supernet::slimmable` | `SlimmableNet`, `BnStats`, `WIDTH_MULTIPLIERS` |
| `ptx_kernels` | PTX strings for the seven NAS kernels |

## Quick Start

```rust,no_run
use oxicuda_nas::prelude::*;

// Mixed-op blend over all 8 DARTS primitives.
let mut rng = LcgRng::new(42);
let mixed = MixedOp::new(OpKind::all().to_vec(), &mut rng);
assert_eq!(mixed.n_ops(), 8);

// Numerically stable softmax over the architecture logits.
let weights = mixed.weights();
let sum: f32 = weights.iter().sum();
assert!((sum - 1.0).abs() < 1e-5);

// PTX kernel string for an Ampere (SM 8.0) target.
let _ptx = arch_softmax_ptx(80);
# Ok::<(), NasError>(())
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
