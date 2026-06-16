# oxicuda-timeseries

Time-series forecasting and classification primitives for OxiCUDA --
TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN, series decomposition,
in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-timeseries` collects the building blocks of modern time-series
deep learning into a single crate. Classical reversible normalisation
(`RevIN`, `InstanceNorm1d`), additive trend / seasonal decomposition, 1-D
patch embedding, and forecasting heads (`LinearHead`, `MlpHead`) are
combined with five complete forecaster architectures:

* **TCN** -- causal dilated temporal convolutions (Bai et al. 2018).
* **NHiTS** -- hierarchical multi-rate forecasting (Challu et al. 2022).
* **PatchTST** -- patched-Transformer encoder (Nie et al. 2023).
* **TimesNet** -- 2-D variation modelling via FFT period detection
  (Wu et al. 2023).
* **iTransformer** -- inverted variate-as-token Transformer
  (Liu et al. 2024).

All tensors use a **time-major `[T, C]` layout** throughout. PTX kernels
(`moving_average`, `patch_embed_1d`, `causal_temporal_conv`,
`auto_correlation`, `revin_normalize`, `multirate_pool`, `period_detect`)
are emitted for SM 7.5 through SM 12.0. The only crate dependency is
`thiserror`.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `TsError` / `TsResult` |
| `handle` | `TsHandle`, `SmVersion`, `LcgRng` |
| `decomp` | `MovingAvg`, `SeriesDecomp`, `DecompResult` |
| `norm` | `RevIn` (reversible instance norm), `InstanceNorm1d` |
| `patch` | `PatchEmbed1d` strided 1-D patch embedder |
| `head` | `LinearHead`, `MlpHead` forecast / classification heads |
| `tcn` | `TcnConfig`, `TcnBlock`, `TcnEncoder` |
| `nhits` | `NHitsConfig`, `NHitsBlock`, `NHits`, `MultiRateSampler` |
| `patchtst` | `PatchTstConfig`, `PatchTst` |
| `timesnet` | `TimesNetConfig`, `TimesBlock`, `TimesNet` |
| `itransformer` | `ITransformerConfig`, `InvertedBlock`, `ITransformer` |
| `ptx_kernels` | PTX for the seven kernels listed above |

## Quick Start

```rust,no_run
use oxicuda_timeseries::prelude::*;

let mut rng = LcgRng::new(42);

// 96-step look-back, 4 channels, 24-step horizon, tiny PatchTST config.
let t = 96;
let c = 4;
let horizon = 24;
let cfg = PatchTstConfig::tiny(c, t, horizon);
let model = PatchTst::new(cfg, &mut rng)?;

// Time-major [T, C] input.
let x = vec![0.1_f32; t * c];
let forecast = model.forward(&x)?;
assert_eq!(forecast.len(), horizon * c);
# Ok::<(), TsError>(())
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
