# oxicuda-privacy

Differential Privacy primitives -- selection mechanisms, advanced accountants, local-DP encodings, and DP optimisers in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-privacy` provides the differential-privacy primitives that sit
above the standard `Gaussian` / `Laplace` / Moments-Accountant / RDP-accountant
pieces already shipped in `oxicuda-federated::privacy`. The two crates are
designed to be complementary: this one focuses on selection mechanisms,
numerical / analytical accountants beyond RDP, local-DP encodings, and a
broader set of DP optimisers and sensitivity tools.

All algorithms are implemented in pure Rust with no external linear-algebra
dependencies. The crate also emits GPU PTX kernel strings through
`ptx_kernels`, parameterised on SM compute capability, for the inner loops
that map cleanly to GPU execution.

Accounting coverage is the deepest part of the crate: f-DP / GDP, zCDP, tCDP,
the PRV accountant (including an FFT-based numerical composition variant),
CTD, PLD, RDP subsampling amplification, and shuffle-DP composition are all
provided as separate sub-modules.

## Modules

| Module | Description |
|--------|-------------|
| `mechanism` | Exponential, Report-Noisy-Max, Propose-Test-Release, discrete Gaussian / Laplace, Skellam, DP k-means, DP PCA |
| `selection` | Sparse Vector Technique, AboveThreshold, adaptive / numeric SVT, private histogram, private tuning |
| `accounting` | f-DP / GDP, zCDP, tCDP, PRV (analytical + FFT), CTD, PLD, RDP subsampling, shuffle-DP |
| `composition` | Strong (advanced) composition, subsampling and shuffling amplification |
| `optimizer` | DP-FTRL with tree aggregation, DP-Adam, DP-Adagrad, DP-Adadelta, DP-LAMB, DP-SGD with micro-batches |
| `local` | GRR, OUE, RAPPOR, Hadamard response, heavy hitters, mean estimation, piecewise, subset selection |
| `sensitivity` | Local sensitivity and smooth sensitivity helpers |
| `metrics` | Budget tracking, MSE, SNR, utility |
| `handle` | `PrivacyHandle`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernel generators for DP primitives |
| `error` | `PrivacyError` / `PrivacyResult` |

## Method Coverage

### Selection mechanisms (`selection`)
- Sparse Vector Technique (SVT) -- classical, adaptive, and numeric variants
- AboveThreshold -- single-shot threshold query
- Private histogram and private hyperparameter tuning

### Output-perturbation mechanisms (`mechanism`)
- Exponential mechanism for arbitrary utility scores
- Report-Noisy-Max for top-1 selection
- Propose-Test-Release for high-sensitivity queries
- Discrete Gaussian, discrete Laplace, Skellam mechanisms
- DP k-means and DP PCA

### Local-DP encodings (`local`)
- GRR (Generalised Randomised Response)
- OUE (Optimised Unary Encoding)
- RAPPOR (Bloom-filter-based)
- Hadamard response, subset selection, piecewise mechanism
- Heavy hitters and DP mean estimation under local DP

### Accountants (`accounting`)
- f-DP / GDP -- Gaussian Differential Privacy
- zCDP, tCDP -- (truncated) concentrated DP
- PRV -- Privacy Random Variable accountant (analytical + FFT)
- CTD, PLD -- characteristic function / privacy loss distribution
- RDP subsampling amplification, shuffle-DP composition

### DP optimisers (`optimizer`)
- DP-FTRL with binary-tree noise aggregation
- DP-Adam, DP-Adagrad, DP-Adadelta, DP-LAMB
- DP-SGD with micro-batch gradient clipping

## Quick Start

```rust,no_run
use oxicuda_privacy::handle::LcgRng;
use oxicuda_privacy::optimizer::dp_ftrl::{DpFtrlConfig, DpFtrlState};
use oxicuda_privacy::PrivacyResult;

fn main() -> PrivacyResult<()> {
    // Configure DP-FTRL with tree aggregation.
    let cfg = DpFtrlConfig {
        sigma: 1.0,
        grad_clip: 1.0,
        learning_rate: 0.1,
        l2_reg: 0.0,
        max_depth: 10, // supports up to 2^10 = 1024 steps
    };

    let mut rng = LcgRng::new(0);
    let n_params = 1024;
    let mut state = DpFtrlState::new(n_params, &cfg, &mut rng)?;

    // Per-step: clip gradient, then call state.step(...) with the noisy update.
    let _grad: Vec<f64> = unimplemented!();
    Ok(())
}
```

## Status

**Alpha** -- 16,590 SLoC, 823 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
