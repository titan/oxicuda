# oxicuda-bayes

Bayesian deep learning primitives for OxiCUDA -- pure Rust implementation of
variational inference, Bayesian layers, MC Dropout, Deep Ensembles, SWAG,
Laplace approximation, and uncertainty calibration.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-bayes` provides building blocks for Bayesian neural networks and
post-hoc uncertainty estimation, designed to run as a CPU reference and to
emit PTX kernels for GPU execution. There is **zero CUDA SDK dependency** at
build time -- the only runtime crate dependency is `thiserror`.

The variational layers (`BayesLinear`, `BayesConv2d`, `FlipoutLinear`,
`FlipoutConv2d`) carry both mean and rho parameters; `softplus(rho)` recovers
the per-weight standard deviation. Sampling uses the standard reparameterization
trick (or Flipout perturbations for low-variance gradient estimators). The
`variational` module supplies ELBO / IWAE bounds, Gaussian KL helpers,
Planar/Radial normalizing flows, and mean-field distribution objects. PTX
kernels are emitted for SM 7.5 through SM 12.0.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `BayesError` / `BayesResult` with `thiserror` variants |
| `handle` | `BayesHandle`, `SmVersion`, deterministic `LcgRng` |
| `layers::bayes_linear` | `BayesLinear` (mean + rho), `softplus` helper |
| `layers::bayes_conv` | `BayesConv2d` Bayesian convolution |
| `layers::flipout` | `FlipoutLinear`, `FlipoutConv2d` low-variance perturbation |
| `variational::elbo` | ELBO / IWAE bounds, `kl_gaussian`, `kl_gaussian_vec` |
| `variational::flows` | `PlanarFlow`, `RadialFlow` normalizing-flow blocks |
| `variational::mean_field` | `MeanFieldDist` Gaussian variational posterior |
| `variational::reparam` | Gaussian / Laplacian sampling and log-prob, straight-through estimator |
| `ptx_kernels` | PTX strings for KL, local reparam, MC Dropout, Flipout, ECE, temperature scaling, ensemble aggregation |

## Quick Start

```rust,no_run
use oxicuda_bayes::prelude::*;

let handle = BayesHandle::default_handle();
let mut rng = LcgRng::new(42);

// A 4x8 Bayesian linear layer with prior sigma = 1.0.
let layer = BayesLinear::new(4, 8, 1.0, &mut rng)?;

// Closed-form KL(N(mu, sigma^2) || N(0, 1)) for mu = 0, log_var = 0
// (i.e., sigma^2 = 1) is exactly 0.
let kl = kl_gaussian(0.0, 0.0)?;
assert!(kl.abs() < 1e-6);

// PTX kernel string for the handle's SM target (default = SM 8.0 / Ampere).
let _ptx = kl_gaussian_ptx(handle.sm_version().as_u32());
# Ok::<(), BayesError>(())
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
