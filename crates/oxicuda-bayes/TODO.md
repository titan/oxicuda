# oxicuda-bayes TODO

Pure-Rust Bayesian deep learning primitives for OxiCUDA: variational inference,
Bayesian layers, MC Dropout, Deep Ensembles, SWAG, last-layer Laplace,
calibration metrics and post-hoc recalibration. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.23).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~28,400 (61 files)
- **PTX kernels:** 7 kernel generators emitted for 6 SM targets (sm_75 / 80 / 86 / 90 / 100 / 120)
- **Coverage:** CPU reference implementation + PTX string generation for GPU execution

### Completed

#### Core Inference Expansion (gradient-free MCMC, model selection, conjugate, spike-slab)
- [x] `mcmc/metropolis.rs` -- Random-walk Metropolis-Hastings (`MetropolisSampler`) with adaptive per-coordinate proposal scaling (Haario 2001 / Robbins-Monro toward a target acceptance) + univariate coordinate-wise slice sampler (`SliceSampler`, Neal 2003 stepping-out + shrinkage); `−∞` log-density encodes hard constraints; recovers Gaussian / correlated-Gaussian / truncated / bimodal targets; `sample_mean`/`sample_variance` helpers
- [x] `mc/model_selection.rs` -- Predictive model selection: WAIC (Watanabe 2010; lppd − posterior log-lik variance), PSIS-LOO (Vehtari-Gelman-Gabry 2017; self-normalised LOO importance sampling with an exact-likelihood generalized-Pareto tail fit + per-point Pareto-k̂ diagnostic), DIC (Spiegelhalter 2002), and `compare_elpd` paired model comparison with SE
- [x] `mc/conjugate.rs` -- Closed-form conjugate updates + posterior-predictives: Beta-Binomial, Gamma-Poisson (→ Negative-Binomial predictive), Normal-Normal (known variance), Normal-Inverse-Gamma (unknown mean+variance → Student-t predictive), Dirichlet-Multinomial; full-precision (`f64`) Lanczos log-gamma; predictive PMFs verified to normalise
- [x] `sparse/spike_slab.rs` -- Point-mass spike-and-slab Bayesian variable selection (George-McCulloch 1997) via collapsed Gibbs; per-coordinate inclusion log-odds with the slab Bayes factor, Beta-Bernoulli inclusion-probability hyperprior, Inverse-Gamma noise; returns posterior inclusion probabilities + the median-probability model; Marsaglia-Tsang Gamma / Beta / Inverse-Gamma draws (full ÷2³² uniforms); recovers the true sparse support and coefficient magnitudes

#### Core Infrastructure
- [x] `error.rs` -- `BayesError` (16 variants: DimensionMismatch, EmptyInputs, InvalidDropoutRate, InvalidTemperature, InvalidPriorVariance, NonPositiveSigma, InsufficientSamples, InsufficientEnsembleMembers, CalibrationSetEmpty, NCalibBinsTooSmall, IsotonicNotMonotone, PlattFitFailed, TemperatureNotFinite, FlowDimensionMismatch, NanEncountered, Internal) + `BayesResult<T>`
- [x] `handle.rs` -- `SmVersion` with `ptx_version_str()` (sm>=100 -> "8.7" / sm>=90 -> "8.4" / sm>=80 -> "8.0" / else "7.5"); `LcgRng` with Knuth MMIX 64-bit LCG + Box-Muller `next_normal_pair`/`fill_normal`/`shuffle`; `BayesHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- Module exports + `prelude` re-exports + 12 E2E integration tests

#### PTX Kernels (ptx_kernels.rs, 897 LoC)
- [x] `kl_gaussian_ptx` -- Per-element KL(N(mu, sigma^2) || N(0,1)) using `ex2.approx.f32` / `lg2.approx.f32`, accumulated via `atom.global.add.f32`
- [x] `mc_dropout_mask_ptx` -- Bernoulli dropout mask via inline LCG with `(rand > drop) ? 1/keep : 0` selected via `selp.f32`
- [x] `local_reparam_ptx` -- Local reparameterisation with Box-Muller sampling and `fma.rn.f32` for `z = mu + sqrt(var) * eps`
- [x] `ece_bucket_ptx` -- ECE histogram binning with `atom.global.add.u32` count + `atom.global.add.f32` confidence sum
- [x] `ensemble_aggregate_ptx` -- Ensemble mean / variance over M member logits with `fma.rn.f32` accumulation
- [x] `flipout_perturb_ptx` -- Flipout (Wen 2018) +-1 sign perturbation via `selp.f32` for in-batch decorrelation
- [x] `temp_scale_logits_ptx` -- Temperature scaling of logits (divide by `T`)

#### Bayesian Layers (layers/)
- [x] `bayes_linear.rs` (245 LoC) -- `BayesLinear` Bayes-by-Backprop linear with `softplus(rho)` sigma parameterisation; `forward_sample` + `forward_kl`; per-weight prior N(0, sigma_prior^2) (Blundell 2015)
- [x] `bayes_conv.rs` (269 LoC) -- `BayesConv2d` BBB scheme for spatial conv2d kernels with stride and padding
- [x] `flipout.rs` (327 LoC) -- `FlipoutLinear` / `FlipoutConv2d` with sign-flip perturbations to lower per-batch gradient variance

#### Variational Inference (variational/)
- [x] `elbo.rs` (208 LoC) -- `kl_gaussian` / `kl_gaussian_vec` closed-form KL(q || N(0,1)); `elbo` + `iwae` (importance-weighted) objectives via `ElboConfig`
- [x] `mean_field.rs` (173 LoC) -- `MeanFieldDist` factored Gaussian with entropy, KL, ELBO, `sample`, `sample_n` helpers
- [x] `reparam.rs` (234 LoC) -- `gaussian_sample` / `laplacian_sample` + corresponding log-prob and `straight_through` estimator
- [x] `flows.rs` (278 LoC) -- `PlanarFlow`, `RadialFlow` invertible 1-step normalising flows with log-det Jacobian

#### Calibration (calibration/, 4 files)
- [x] `metrics.rs` (482 LoC) -- `expected_calibration_error` (ECE), `maximum_calibration_error` (MCE), `adaptive_calibration_error` (ACE, quantile bins), `brier_score`, `negative_log_likelihood`, `top1_confidences`, `ReliabilityDiagram` + `ReliabilityBin`
- [x] `temperature.rs` (353 LoC) -- `TemperatureScaler` with golden-section search NLL minimisation; argmax-preserving recalibration (Guo 2017)
- [x] `isotonic.rs` (262 LoC) -- `IsotonicRegressor` Pool-Adjacent-Violators with weighted variant for non-parametric monotone recalibration
- [x] `platt.rs` (328 LoC) -- `PlattScaler` two-parameter logistic recalibration with Lin et al. 2007 stable-target Newton + line search

#### Uncertainty Quantification (uncertainty/, 5 files)
- [x] `mc_dropout.rs` (246 LoC) -- `mc_dropout_predict` + `McDropoutPredictor` with Welford online mean / variance over T forward passes (Gal & Ghahramani 2016)
- [x] `deep_ensemble.rs` (236 LoC) -- `DeepEnsemble::aggregate()` / `aggregate_probabilities()` (mean + sample variance with Bessel correction); `EnsembleStats`
- [x] `swag.rs` (245 LoC) -- `SwagPosterior` running first / second moments + FIFO low-rank deviation buffer; sampling `theta_tilde = mu + (1/sqrt(2)) * sigma_diag . z1 + (1/sqrt(2(K-1))) * D . z2` (Maddox 2019)
- [x] `laplace.rs` (286 LoC) -- `LastLayerLaplace` diagonal-Hessian fit for binary logistic head; closed-form predictive logit and probit-approximated marginal probability (MacKay 1992; Daxberger 2021)
- [x] `entropy.rs` (178 LoC) -- `predictive_entropy`, `aleatoric_entropy`, `mutual_information` (BALD), `epistemic_entropy` (Houlsby 2011 decomposition)

#### Integration Tests (lib.rs)
- [x] 12 E2E tests: temperature scaling preserves argmax, isotonic monotonicity, Platt scaling sign, MC Dropout variance band, Deep Ensemble disagreement, SWAG round-trip, Laplace marginal, BALD identity (`H_pred = H_aleatoric + I`), Brier+NLL on perfect predictor, reliability diagram, PTX kernels x 6 SM versions

#### Benchmarks (benches/bayes_ops.rs)
- [x] 7 PTX kernel generator groups x 4 SM versions + `temperature_scaling_fit` + `isotonic_pav_fit` + `ece_compute` + `swag_sample` + `deep_ensemble_aggregate`

### Future Enhancements

#### P0 -- Critical (Algorithm Coverage Gaps)
- [x] Full-covariance Laplace approximation -- current `LastLayerLaplace` uses a diagonal Hessian; add KFAC / dense block fits to capture parameter correlations (Daxberger 2021 full-Laplace)
- [x] Functional Laplace / linearised-Laplace predictive -- exact posterior predictive via Jacobian linearisation rather than probit approximation (uncertainty/functional_laplace.rs -- Immer 2021; GGN posterior precision H=prior_prec·I+Σ JᵀJ, Σ=H⁻¹, predictive var=J Σ Jᵀ, mean=MAP output via local linearization)
- [x] BayesGRU -- Bayesian Gated Recurrent Unit via BBB (BayesLSTM / BayesAttention remain for a future wave)
- [x] Sparse GP FITC/PITC (Titsias 2009): inducing-points variational lower bound ELBO=log 𝒩(y; KₙₘK_mm⁻¹μ, σ²I+Qₙₙ-diag(Kₙₙ-Qₙₙ)) + KL(q(f_m)‖p(f_m)); O(nm²) per gradient step vs O(n³) full GP — **already implemented in `gp/sparse_gp.rs`** (FITC; Snelson-Ghahramani 2006 + Titsias 2009 free energy; Cholesky-based `sparse_gp_fit`/`sparse_gp_predict`/`sparse_gp_elbo`, `InducingInit`)
- [x] `layers/bayes_lstm.rs` — Bayesian LSTM (BBB, Fortunato 2017): weight mean+log-var reparameterised; forward = deterministic LSTM with noise + KL penalty; `BayesLstm` struct with `sample()` + `kl_divergence()`

#### P1 -- Important (Variational Inference Depth)
- [x] Real NVP / IAF / MAF normalising flows — variational/real_nvp.rs (affine coupling layers with alternating masks, exact inverse, log-det-Jacobian; IAF/MAF remain future)
- [x] Stein Variational Gradient Descent (SVGD) — variational/svgd.rs (RBF kernel, median heuristic, Stein operator with score + kernel-gradient, particle update; Liu & Wang 2016 NeurIPS)
- [x] Hamiltonian Monte Carlo + NUTS posterior sampler -- gradient-based MCMC for small-network reference posteriors
- [x] Variational continual learning helpers -- online VI updates with prior replacement for sequential tasks (Nguyen 2018) (variational/vcl.rs -- Nguyen 2018; mean-field Gaussian online VI, closed-form KL(q‖prior), ELBO step, consolidate posterior→next prior, reparameterized sample)
- [x] `gp/deep_gp.rs` — Deep Gaussian Processes (Damianou-Lawrence 2013): doubly-stochastic VI; GP layers f^(l+1)=GP(f^(l)); DSVI ELBO = Σ E_q[log p(y|f_L)] - Σ_l KL(q(u_l)‖p(u_l)); inducing-point approx per layer — **already implemented in `gp/deep_gp.rs`** (`DeepGp`/`DeepGpConfig`/`DeepGpLayer`; Salimbeni-Deisenroth 2017 doubly-stochastic forward_sample + per-layer KL + Titsias output posterior)
- [x] `variational/nvae.rs` — Nouveau VAE (Vahdat-Kautz 2020): hierarchical VAE with residual normalising flows at each level; bidirectional encoder + top-down decoder; KL balancing via free-bits heuristic — **already implemented in `variational/nvae.rs`** (`NVae`/`NVaeConfig`/`NVaeOutput`; hierarchical top-down conditional prior p(z_l|z_<l), per-group `kl_gaussian_diag` + `apply_free_bits` floor, single-sample MC ELBO)
- [x] `variational/iaf_flow.rs` — Inverse Autoregressive Flow (Kingma 2016): IAF posterior q(z|x) = T_t∘…∘T_1(ε); each T invertible autoregressive; exact density via change-of-variables; complements existing `real_nvp.rs`

#### P2 -- Nice-to-Have (Calibration & Reporting Extensions)
- [x] Histogram binning calibration -- non-parametric bin-wise recalibration as a complement to isotonic / Platt
- [x] Beta calibration -- three-parameter calibrator generalising Platt scaling for skewed score distributions
- [x] Multi-class temperature with vector scaling / matrix scaling -- per-class temperature and full affine recalibration heads
- [x] Class-conditional calibration metrics -- per-class ECE / reliability diagrams in addition to the existing top-1 metrics — **already implemented in `calibration/ece_classwise.rs`** (Kull 2019 one-vs-rest class-wise ECE, `BinningScheme::Adaptive` equal-mass bins, `ClassReliability`/`ReliabilityPoint`/`per_class_reliability`, `BrierDecomposition` reliability−resolution+uncertainty, `top_label_calibration`)
- [x] Conformal prediction wrappers -- split / inductive conformal intervals using the existing top-1 score machinery
- [x] `calibration/ece_classwise.rs` — Class-wise ECE (Kull 2019): per-class calibration curve, adaptive equal-mass binning, multiclass reliability diagram; complements existing top-1 ECE
- [x] `uncertainty/evidential.rs` — Evidential Deep Learning (Sensoy 2018): Dirichlet output parameterisation; uncertainty = total variance decomposed into aleatoric + epistemic via Dir(α); NIG prior for regression
- [x] `mc/convergence_diagnostics.rs` — MCMC convergence diagnostics: R-hat (Gelman-Rubin 1992) for multi-chain runs; effective sample size (ESS) via autocorrelation; Geweke Z-test; integrate with `variational/hmc.rs`

#### GPU Launcher Wiring
- [ ] Wire `ptx_kernels::*` strings through `oxicuda-launch::Kernel::from_module` for end-to-end GPU execution (currently only the PTX strings are emitted; CPU paths are the authoritative reference)
- [ ] GPU-resident `BayesLinear::forward_sample` using `local_reparam_ptx` instead of CPU LCG draws

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device / Host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 12 E2E in `lib.rs` + module unit tests (see root TODO.md Vol.23 reference for the workspace-wide count)
- `unwrap()` calls: 0 in library code (test code may use `unwrap`)
- macOS: compiles, returns `UnsupportedPlatform` from any actual GPU launch
- PTX targets covered: sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120

## Performance Targets

| Operation | Size | Target |
|-----------|------|--------|
| PTX kernel string generation | per call | < 100 us (string concat only) |
| `TemperatureScaler::fit` (golden-section, 200 evals) | N = 10K, K = 1K | < 50 ms CPU |
| `IsotonicRegressor::fit` (PAV) | N = 100K | < 20 ms CPU |
| `mc_dropout_predict` (Welford accumulate) | T = 100, D = 1024 | < 100 ms (CPU closure-driven) |
| `DeepEnsemble::aggregate_probabilities` | M = 10, K = 1000 | < 1 ms |
| `SwagPosterior::sample` (K = 30 columns, P = 1M) | -- | < 50 ms CPU |

Targets are CPU-reference budgets; once `[ ]` GPU launcher wiring lands, the
PTX kernels (`kl_gaussian_ptx`, `local_reparam_ptx`, `ensemble_aggregate_ptx`)
should run within 5% of a hand-tuned cuDNN dropout / reduction launch.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/bayes_ops.rs`) -- PTX generation + CPU calibration / ensemble pipelines

## Notes

- All PTX kernels emit `.target sm_<version>` and use a grid-stride loop pattern.
- LCG seed is fixed (42) by `BayesHandle::default_handle()` for reproducibility; callers can `rng_mut()` to reseed.
- The Knuth MMIX 64-bit LCG has visible correlation between consecutive normals -- the `e2e_mc_dropout_quantifies_uncertainty` test discards every other Box-Muller pair as a workaround.
- Calibration code returns `BayesError::PlattFitPailed` / `TemperatureNotFinite` rather than panicking on degenerate input -- prefer `?` over `unwrap` in caller code.
- The PTX kernels target scalar f32 paths; no Tensor Core (wgmma / mma.sync) usage -- Bayesian layer GEMMs are expected to delegate to `oxicuda-blas`.

---

## Architecture-Specific Deepening

### PTX Generation by SM Version

| SM Version | PTX Version | Notes |
|------------|-------------|-------|
| sm_75 (Turing) | 7.5 | Baseline; `atom.global.add.f32` supported |
| sm_80 / sm_86 (Ampere) | 8.0 | Default target for `BayesHandle::default_handle()` |
| sm_89 (Ada) | 8.0 | Treated as sm_80 by `ptx_version_str()` |
| sm_90 / sm_90a (Hopper) | 8.4 | No `wgmma` usage -- kernels are scalar |
| sm_100 / sm_120 (Blackwell) | 8.7 | Same scalar-FMA pattern -- no `cp.async.bulk` yet |

The 7 generators all dispatch on the SM string and emit identical scalar PTX
modulo the `.target` directive. Tensor-Core specialisation is intentionally
left out: Bayesian layers compose with `oxicuda-blas` GEMMs and only use
these kernels for reparameterisation, atomics, and dropout.

### Deepening Opportunities

- [ ] Hopper warp-specialised KL reduction using `redux.sync.add.f32` for atomic-free accumulation across a warp
- [ ] Blackwell (sm_100+) `cp.async.bulk.tensor` for fetching the SWAG low-rank deviation buffer into shared memory in `sample()`
- [ ] PTX FP16 / BF16 variants of `temp_scale_logits_ptx` and `ensemble_aggregate_ptx` for mixed-precision inference

---

## Functional Quality Gates (Vol.23)

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| B1 | Bayesian linear / conv layers (`forward_sample` + KL) | P0 | [x] |
| B2 | Flipout decorrelation for in-batch variance reduction | P0 | [x] |
| B3 | Closed-form Gaussian KL + ELBO / IWAE | P0 | [x] |
| B4 | Mean-field variational distribution | P0 | [x] |
| B5 | Reparameterisation + straight-through estimator | P0 | [x] |
| B6 | Planar / radial normalising flows | P1 | [x] |
| B7 | MC Dropout predictive mean + variance | P0 | [x] |
| B8 | Deep Ensemble aggregation | P0 | [x] |
| B9 | SWAG low-rank posterior sampling | P0 | [x] |
| B10 | Last-layer Laplace (diagonal Hessian) | P0 | [x] |
| B11 | BALD / mutual-information decomposition | P0 | [x] |
| B12 | ECE / MCE / ACE + reliability diagram | P0 | [x] |
| B13 | Brier score + NLL + top-1 confidences | P0 | [x] |
| B14 | Temperature scaling (golden-section NLL) | P0 | [x] |
| B15 | Isotonic (PAV) + Platt recalibration | P0 | [x] |
| B16 | PTX generators for 7 kernels x 6 SM versions | P0 | [x] |

## Performance Verification Harness Status

- All performance numbers above are CPU-side targets achievable on the build host.
- GPU end-to-end harnesses await the [ ] GPU launcher wiring item above plus a
  Linux+NVIDIA test runner; the PTX strings themselves are already covered by
  string-content unit tests inside `ptx_kernels.rs`.
