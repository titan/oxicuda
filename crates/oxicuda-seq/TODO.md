### oxicuda-seq TODO

Sequence Models & Structured Prediction for OxiCUDA — a pure Rust toolkit covering
HMM, linear-chain CRF, MEMM, structured SVM, beam search, classical pairwise sequence
alignment (Needleman-Wunsch / Smith-Waterman / Gotoh / Hirschberg), 2D grid CRFs,
Kalman / RTS / EKF / EM, and general MRFs with Gibbs sampling and loopy belief
propagation. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.51).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 6,021 lines / 5,124 SLoC (41 files)** — implements every major sequence /
structured-prediction algorithm in pure Rust with log-space numerics for stability,
including L-BFGS for CRF training and Hirschberg O(min(m, n))-memory alignment.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` — `SeqError` enum (13 variants: `ShapeMismatch`, `DimensionMismatch`,
  `NotConverged`, `InvalidConfiguration`, `EmptyInput`, `IndexOutOfBounds`,
  `NumericalInstability`, `ProbabilityOutOfRange`, `InvalidParameter`,
  `UnsupportedSmVersion`, `InvalidObservation`, `LengthMismatch`,
  `GraphInvariantViolated`) plus `SeqResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 boolean trick,
  Box-Muller normal, `sample_categorical`), `SeqHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions (sm_70/75/80/86/89/90):
  `forward_pass`, `viterbi_step`, `crf_features`, `beam_topk`, `edit_dist`,
  `kalman_predict`, `mrf_gibbs` (string concatenation; inline LCG inside the
  Gibbs-sampling kernel)

#### Hidden Markov Models
- [x] `hmm/hmm.rs` — `HmmDiscrete { n_states, n_obs, pi, A, B }`,
  `HmmGaussian { n_states, pi, A, means, covs }` (diagonal covariance)
- [x] `hmm/forward_backward.rs` — Log-space `forward` / `backward`, `log_likelihood`,
  posterior tensors γ and ξ; matches exhaustive enumeration to 1e-9 on short chains
- [x] `hmm/viterbi.rs` — Log-space dynamic programming with explicit backpointers;
  recovers the ground-truth path on deterministic chains
- [x] `hmm/baum_welch.rs` — Full EM training: E-step computes (γ, ξ), M-step updates
  π via γ_0, A via Σξ / Σγ, B via class-conditioned γ; log-likelihood is
  non-decreasing across iterations

#### Linear-Chain Conditional Random Fields
- [x] `crf/linear_chain_crf.rs` — `LinearChainCrf { n_labels, n_features, transitions,
  emissions }` score-form model
- [x] `crf/crf_train.rs` — Score-space forward-backward for partition function /
  expected features; full L-BFGS two-loop recursion with Armijo backtracking line
  search and L2 regularisation; gradient validated to 1e-3 by finite-difference check
- [x] `crf/viterbi_decode.rs` — Log-space Viterbi specialised for the CRF score form

#### Maximum-Entropy Markov Models
- [x] `memm/memm.rs` — MEMM with per-state softmax over features, greedy + beam decoding

#### Structured SVM
- [x] `ssvm/ssvm.rs` — `StructuredSvm` with loss-augmented Viterbi inference and
  sub-gradient training
- [x] `ssvm/cutting_plane.rs` — Constraint-based optimiser scaffold (placeholder for
  full cutting-plane QP solver)

#### Beam Search
- [x] `beam/beam.rs` — Generic `BeamSearch` over a scoring callback with
  length-normalisation and a diversity penalty; matches exhaustive top-1 on tiny vocab

#### Pairwise Sequence Alignment
- [x] `alignment/needleman_wunsch.rs` — Global alignment with linear gaps; classic
  "GATTACA" / "GCATGCU" example
- [x] `alignment/smith_waterman.rs` — Local alignment; embedded-substring detection
- [x] `alignment/gotoh.rs` — Affine-gap 3-state DP (M / X / Y) with gap-open and
  gap-extend penalties
- [x] `alignment/hirschberg.rs` — O(min(m, n))-memory NW via divide-and-conquer;
  identical score to standard NW (validated by e2e test)

#### 2D Grid Conditional Random Fields
- [x] `grid_crf/grid_crf.rs` — 4-connected pairwise CRF for image labelling
- [x] `grid_crf/mean_field.rs` — Damped mean-field variational inference with
  neighbour-message accumulation

#### Kalman Filter Family
- [x] `kalman/kalman_filter.rs` — Linear Kalman filter: innovation,
  S = H · P · H^T + R, K = P · H^T · S⁻¹, posterior update
- [x] `kalman/rts_smoother.rs` — Rauch-Tung-Striebel backward smoother; variance is
  always ≤ filter variance (test-enforced invariant)
- [x] `kalman/ekf.rs` — Extended Kalman filter accepting boxed-closure Jacobians for
  the state-transition `f` and observation `h`
- [x] `kalman/kalman_em.rs` — Shumway-Stoffer EM for Q / R covariance refit from data
- [x] `kalman/linalg.rs` — Local helpers (matmul, Gauss-Jordan inverse, Cholesky,
  determinant) kept private to the kalman module

#### Markov Random Fields
- [x] `mrf/mrf.rs` — General `Mrf` over a graph plus `IsingModel`; potential functions
- [x] `mrf/gibbs.rs` — Gibbs sampler with simulated-annealing temperature schedule;
  recovers low-temperature magnetisation
- [x] `mrf/belief_prop.rs` — Log-space loopy belief propagation: sum-product marginals
  and max-product MAP estimates over factor graphs

#### Sequence Metrics
- [x] `metrics/metrics.rs` — `token_accuracy`, `sequence_accuracy`, Levenshtein
  `edit_distance` (`"kitten" → "sitting" = 3`), BLEU-n with add-1 smoothing and
  brevity penalty, `perplexity`, `log_loss`

#### Tests & Benchmarks
- [x] `e2e_tests.rs` — 15 cross-module tests: HMM forward-backward vs. enumeration,
  Viterbi on deterministic chain, Baum-Welch monotone log-likelihood, CRF Viterbi
  sanity, CRF gradient finite-difference check, NW / SW / Gotoh / Hirschberg
  correctness, edit distance kitten → sitting, Kalman 1σ recovery, RTS smoother
  variance reduction, Ising Gibbs magnetisation, beam top-1 vs. enumeration,
  BLEU-1 identical = 1.0, PTX × 6 SM versions
- [x] `benches/seq_ops.rs` — Criterion suite: 7 PTX kernels × all SM versions plus
  Viterbi / alignment algorithm benches

### Future Enhancements [ ]

#### P0 — Correctness & Foundation
- [x] `hmm/scaling.rs` — Explicit scaling (Rabiner 1989) for `HmmDiscrete` forward
  pass on very long sequences (alternative to log-space)
- [x] `crf/lbfgs_b.rs` — L-BFGS-B with box constraints for non-negativity-constrained
  CRF training
  (crf/lbfgs_b.rs -- Byrd-Lu-Nocedal 1995; L-BFGS two-loop + projected line search + box-projection; box-constrained CRF training)
- [ ] `crf/sgd.rs` — Stochastic gradient training with mini-batches and AdaGrad /
  Adam variants
- [x] `ssvm/cutting_plane_full.rs` — Full cutting-plane SVM optimiser with QP
  inner solver (current implementation is a scaffold)
  (ssvm/cutting_plane_full.rs -- Joachims 2009; n-slack cutting-plane separation-oracle + per-example working sets + projected-gradient QP master dual)
- [x] `mrf/junction_tree.rs` — Exact inference via junction-tree algorithm for low-
  treewidth graphs (current loopy BP is approximate)
  (mrf/junction_tree.rs -- triangulation + max-spanning clique tree + Hugin two-pass message passing, exact marginals + log partition)

#### P1 — Algorithmic Coverage
- [x] `hmm/variational.rs` — Variational Bayes EM for HMMs with Dirichlet priors
- [x] `hmm/semimarkov.rs` — Semi-Markov / segmental HMM with explicit duration models
- [x] `crf/skip_chain.rs` — Skip-chain CRF for long-range label dependencies
  (crf/skip_chain.rs -- linear-chain CRF + long-range skip edges, loopy sum-product/max-product BP; exact = forward-backward when no skip edges)
- [ ] `crf/general_graph.rs` — General-graph CRF training (not just linear chain) via
  loopy belief propagation
- [ ] `memm/lbfgs.rs` — L-BFGS training for MEMM (currently sub-gradient only)
- [ ] `ssvm/n_slack.rs` — n-slack formulation alternative to 1-slack
- [x] `beam/diverse.rs` — Diverse beam search (Vijayakumar 2018) with group penalties
- [ ] `beam/length_penalty.rs` — GNMT-style length penalty
  ((5 + |Y|) / 6)^alpha for translation
- [ ] `alignment/blast.rs` — Heuristic BLAST-style seed-and-extend alignment for
  database search
- [ ] `alignment/banded.rs` — Banded DP for sequences known to be near-identical
- [ ] `grid_crf/loopy_bp.rs` — Loopy BP on grid CRF (alternative to mean-field)
- [x] `kalman/ukf.rs` — Unscented Kalman Filter with sigma-point propagation
- [x] `kalman/particle.rs` — Particle filter / sequential Monte Carlo for non-linear
  non-Gaussian state-space models
- [ ] `mrf/swendsen_wang.rs` — Swendsen-Wang cluster sampling for Ising-like models
  (faster mixing than Gibbs at low T)
- [ ] `mrf/wolff.rs` — Wolff single-cluster algorithm for ferromagnetic Ising

#### P1 — Modern Neural Decoders
- [x] `decoders/nucleus.rs` — Top-p (nucleus) sampling (Holtzman et al. 2020) with min_tokens floor + numerically-stable softmax
- [x] `decoders/top_k.rs` — Top-k sampling (Fan-Lewis-Dauphin 2018) with temperature
- [x] `decoders/typical.rs` — Typical sampling (Meister 2022) with entropy-distance ordering
- [x] `decoders/contrastive.rs` — Contrastive search decoding (Su 2022)

#### P2 — Advanced & Hybrid
- [x] `metrics/chrf.rs` — Character n-gram F-score (chrF / chrF++) for translation
  quality
- [ ] `metrics/ter.rs` — Translation Edit Rate
- [ ] `metrics/wer.rs` — Word Error Rate for ASR-style outputs
- [ ] `metrics/bertscore.rs` — Embedding-based metric scaffold (requires external
  embeddings; placeholder)
- [ ] `benches/algo_bench.rs` — Extended algorithm benches: standard NLP corpora
  (CoNLL-2003 NER, Penn Treebank POS)

#### P2 — GPU / Architecture-Specific
- [ ] PTX kernel for batched forward-backward across many short sequences in parallel
- [ ] PTX kernel for warp-cooperative Viterbi over wide label sets (≥ 64 labels)
- [ ] PTX kernel for parallel beam-search top-k via bitonic sort

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA SDK, no `nvcc`, no link-time `libcuda`, no external linear-algebra or
optimiser dependencies. Kalman linear algebra (matmul, inverse, Cholesky) is
implemented from scratch; L-BFGS is implemented from scratch with two-loop recursion.

## Quality Status

- Warnings: 0 (clippy clean; `needless_range_loop` and `type_complexity` are
  intentionally allowed crate-wide because numerical kernels index multiple parallel
  arrays per iteration body and rewriting them in iterator form would obscure the
  math)
- Tests: 15 e2e tests passing (host-side); PTX kernel strings validated per SM version
- `unwrap()` calls in production code: 0
- `unsafe` code: 0
- macOS: compiles; GPU integration paths return `UnsupportedPlatform` at runtime
- Linux + NVIDIA driver 525+: GPU paths exercised behind `#[cfg(feature = "gpu-tests")]`

## Performance Targets

| Workload | Size | Target |
|----------|------|--------|
| HMM forward-backward | T = 1024, n_states = 64 | ≤ 5 ms on sm_80 |
| Viterbi decoding | T = 1024, n_states = 64 | ≤ 2 ms on sm_80 |
| Baum-Welch single iteration | T = 512, n_states = 32 | ≤ 50 ms on sm_80 |
| CRF L-BFGS training step | T = 256, n_labels = 16, n_feat = 1000 | ≤ 100 ms per epoch |
| NW alignment | m, n ≤ 1024 | ≤ 20 ms host-side |
| Hirschberg alignment | m, n ≤ 4096 | O(min(m, n)) memory; ≤ 200 ms |
| Kalman filter step | state ≤ 16, obs ≤ 16 | ≤ 1 ms per step |
| Ising Gibbs sweep | 128 × 128 lattice | ≤ 10 ms per sweep |
| Beam search top-k | beam = 16, vocab = 32K | ≤ 50 ms per step |

The current GPU paths are PTX-string scaffolds; the algorithmic core runs on the host
CPU. Performance numbers above are targets for the fully GPU-integrated pipeline.

## Notes

- All forward / backward / Viterbi computations are in log-space for numerical
  stability across long sequences and small probabilities.
- The CRF L-BFGS implementation uses a two-loop recursion with Armijo backtracking
  line search and is finite-difference-checked to 1e-3 in `e2e_tests.rs`.
- Hirschberg's algorithm provides O(min(m, n))-memory alignment matching the standard
  NW score; this is essential for very long DNA / protein sequences.
- The Kalman EM (Shumway-Stoffer) refits Q and R covariance matrices given an observed
  trajectory; the implementation tracks log-likelihood across iterations.
- The Ising Gibbs sampler uses a simulated-annealing temperature schedule and recovers
  ferromagnetic magnetisation at low T as a sanity check in `e2e_tests.rs`.

---

## Architecture-Specific Deepening

### Volta / Turing (sm_70, sm_75)
- [x] `forward_pass` and `viterbi_step` PTX emit warp-cooperative reductions for
  the per-state max / sum over predecessors
- [ ] Verified on T4 / V100 against host reference for HMM forward-backward

### Ampere (sm_80, sm_86)
- [x] PTX kernels use `cp.async` for emission-matrix tile prefetch
- [ ] `crf_features` 3-stage software pipeline benchmarked vs. naive load on A100
- [ ] Kalman matrix updates use tensor cores for state dim ≥ 16

### Ada (sm_89)
- [x] Reuses Ampere code path with `cp.async` prefetch
- [ ] Investigate L2 persistence policy for hot Viterbi backpointer arrays

### Hopper (sm_90)
- [x] PTX strings emit `wgmma` warp-group MMA for Kalman matrix products where shapes
  are amenable
- [ ] TMA-based asynchronous load of HMM emission matrices for very long sequences
- [ ] Distributed shared-memory cooperative Viterbi across CTAs for large label sets

---

## Deepening Opportunities

### Verification Gaps
- [x] HMM forward-backward matches exhaustive enumeration on short chains to 1e-9
- [x] Viterbi recovers the ground-truth path on deterministic chains
- [x] Baum-Welch log-likelihood is non-decreasing across iterations (monotone EM)
- [x] CRF gradient is finite-difference-checked to 1e-3
- [x] NW / SW / Gotoh / Hirschberg correctness validated against textbook examples
- [x] Hirschberg score matches standard NW exactly
- [x] Edit distance "kitten" → "sitting" = 3 (textbook example)
- [x] Kalman filter recovers a true state within 1σ; RTS smoother variance ≤ filter
  variance
- [x] BLEU-1 self-comparison returns 1.0
- [ ] CRF training on CoNLL-2003 NER reaches reported F1 baseline
- [ ] HMM Gaussian on speech-MFCC reaches reported word-error-rate baseline

### Implementation Deepening
- [x] L-BFGS two-loop recursion + Armijo backtracking line search + L2 regularisation
- [x] Hirschberg divide-and-conquer producing identical score to NW
- [x] Gotoh affine-gap 3-state (M / X / Y) DP with separate open / extend penalties
- [x] EKF accepting boxed-closure Jacobians for arbitrary non-linear f / h
- [x] Loopy BP in log-space with sum-product / max-product modes
- [ ] UKF with sigma-point propagation for highly non-linear systems
- [ ] Particle filter for non-Gaussian state-space models
- [ ] Junction-tree algorithm for exact inference on low-treewidth graphs

### Documentation Gaps
- [x] Each public type carries a doc comment summarising its semantics
- [ ] Worked example: training a CRF chunker on a small CoNLL-format corpus
- [ ] Worked example: Kalman tracking of a 2D position from noisy measurements
- [ ] Worked example: HMM Gaussian recovering mixture parameters from synthetic data
