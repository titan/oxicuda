# oxicuda-seq

Sequence models and structured prediction -- HMMs, CRFs, Kalman filters, alignment, beam search, and stochastic decoders in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-seq` is the sequence-modelling volume of the OxiCUDA stack. It
collects the canonical algorithms for discrete-state sequence models
(HMMs, CRFs, MEMMs, structured SVMs), continuous-state filtering
(Kalman / EKF / UKF, particle filters, RTS smoothing), pairwise / grid MRFs,
classical alignment, and the decoding strategies (Viterbi, beam search,
top-k / nucleus / typical sampling) that sit on top of them.

All algorithms are implemented in pure Rust with no external linear-algebra
dependencies. GPU PTX kernel generators in `ptx_kernels` provide kernel
strings for the operations whose inner loops are amenable to direct kernel
mapping, parameterised on SM compute capability.

The numerical kernels in this crate prefer straight integer-indexed loops
over iterator chains because most bodies touch multiple parallel arrays
indexed by the same `(state, observation, time)` triplets, which is why
`clippy::needless_range_loop` is allowed crate-wide.

## Modules

| Module | Description |
|--------|-------------|
| `hmm` | Discrete and Gaussian HMMs, forward-backward, Viterbi, Baum-Welch, variational EM, semi-Markov |
| `crf` | Linear-chain Conditional Random Fields with L-BFGS-B training, Viterbi decoding, skip-chain extension |
| `memm` | Maximum-Entropy Markov Models |
| `ssvm` | Structured SVM (linear-chain) with cutting-plane optimisation |
| `beam` | Generic beam search with length normalisation and diverse-beam decoding |
| `decoders` | Stochastic decoders: top-k, nucleus (top-p), typical sampling |
| `alignment` | Needleman-Wunsch, Smith-Waterman, Gotoh affine-gap, Hirschberg |
| `grid_crf` | Pairwise 2D CRF with mean-field variational inference |
| `kalman` | Linear / Extended / Unscented Kalman filter, RTS smoother, EM parameter learning, particle filter |
| `mrf` | General MRF + Ising model, Gibbs sampler, loopy belief propagation |
| `metrics` | Token / sequence accuracy, edit distance, BLEU, chrF, perplexity, log-loss |
| `handle` | `SeqHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernels for sequence-model operations |
| `error` | `SeqError` / `SeqResult` |

## Method Coverage

### Hidden Markov Models (`hmm`)
- Discrete and Gaussian emissions
- Forward-backward with scaling for numerical stability
- Viterbi decoding
- Baum-Welch (EM) parameter learning
- Variational Bayes EM with Dirichlet priors
- Hidden semi-Markov extension

### Conditional Random Fields (`crf`)
- Linear-chain CRF: forward-backward in score space, Viterbi decoding
- L-BFGS-B training
- Skip-chain extension for long-range dependencies

### State-space filters (`kalman`)
- Linear Kalman filter and RTS smoother
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle filter
- EM parameter learning for linear-Gaussian models

### Sequence alignment (`alignment`)
- Needleman-Wunsch (global)
- Smith-Waterman (local)
- Gotoh (affine-gap penalty)
- Hirschberg (linear-space)

### Decoding (`beam` + `decoders`)
- Beam search with length normalisation and diverse-beam penalty
- Top-k sampling, nucleus / top-p sampling, typical sampling

### Graphical models (`grid_crf` + `mrf`)
- Pairwise 2D CRF with mean-field variational inference
- General MRF + Ising model
- Gibbs sampler and loopy belief propagation

## Quick Start

```rust,no_run
use oxicuda_seq::hmm::hmm::HmmDiscrete;
use oxicuda_seq::hmm::viterbi::viterbi;
use oxicuda_seq::SeqResult;

fn main() -> SeqResult<()> {
    // Build a discrete-emission HMM.
    let n_states = 2;
    let n_obs = 3;
    let pi: Vec<f64> = unimplemented!(); // length = n_states, sums to 1
    let a: Vec<f64> = unimplemented!();  // length = n_states^2, row-stochastic
    let b: Vec<f64> = unimplemented!();  // length = n_states * n_obs, row-stochastic
    let hmm = HmmDiscrete::new(n_states, n_obs, pi, a, b)?;

    // Observation sequence (indices into [0, n_obs)).
    let obs: Vec<usize> = vec![0, 1, 2, 1, 0];

    // Decode the most-likely state path.
    let _result = viterbi(&hmm, &obs)?;
    Ok(())
}
```

## Status

**Alpha** -- 20,887 SLoC, 706 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
