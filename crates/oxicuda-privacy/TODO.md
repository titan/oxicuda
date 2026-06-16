# oxicuda-privacy TODO

Pure Rust Differential Privacy primitives covering mechanisms (exponential / report-noisy-max / propose-test-release), selection (sparse-vector technique / above-threshold), accounting (f-DP / GDP, zCDP / tCDP, PRV), composition (advanced, subsampling and shuffling amplification), private optimisers (DP-FTRL, DP-Adam), local DP (GRR, OUE, RAPPOR), and sensitivity analyses (local, smooth), with PTX kernel templates for SM 7.5 through SM 10.0. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.46).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 16,590 SLoC (74 files)**

Current implementation provides DP primitives that complement `oxicuda-federated::privacy` (which owns `GaussianMechanism`, `LaplacianMechanism`, `MomentsAccountant`, `PateConfig`, and the RDP accountant). This crate focuses on selection mechanisms, advanced accountants, local DP, private optimisers, and sensitivity analyses.

### Completed [x]

#### Core Infrastructure
- [x] `lib.rs` — Crate root, module declarations
- [x] `error.rs` — `PrivacyError` enum (13 variants: `InvalidParameter`, `EmptyInput`, `DimensionMismatch`, `BudgetExhausted`, `NonPositiveSensitivity`, `NonPositiveEpsilon`, `InvalidDelta`, `IndexOutOfRange`, `ConvergenceFailed`, `EmptyMechanismList`, `SvtQueryLimitExceeded`, `TreeDepthExceeded`) + `PrivacyResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit + Box-Muller), `PrivacyHandle` with `generate_gaussian_noise` / `generate_laplace_noise`
- [x] `ptx_kernels.rs` — 7 GPU kernels × 6 SM versions (75 / 80 / 86 / 89 / 90 / 100)
- [x] `e2e_tests.rs` — 18 cross-module integration tests

#### Selection Mechanisms (mechanism/)
- [x] `mechanism/exponential.rs` — McSherry-Talwar (2007): `P(i) ∝ exp(ε·q_i / (2·Δq))`, numerically-stable shifted softmax + cumulative-weight sampling
- [x] `mechanism/report_noisy_max.rs` — `Lap(Δq/ε)` noise per score + argmax
- [x] `mechanism/propose_release.rs` — Propose-Test-Release (PTR): `c = ln(1/2δ)/ε`, Lap test on local sensitivity, release with `Lap(Δ/ε)` noise or abstain

#### Query Selection (selection/)
- [x] `selection/sparse_vector.rs` — Streaming SVT (AboveThreshold): `ε₁ = ε/2` threshold noise, `ε₂ = ε/2` query noise, k-true-response budget via `SvtState`
- [x] `selection/above_threshold.rs` — Batch above-threshold: return indices of queries exceeding noisy threshold

#### Privacy Accounting (accounting/)
- [x] `accounting/fdp.rs` — f-DP / GDP (Dong-Roth-Su 2022): trade-off function `T_μ(α) = Φ(Φ⁻¹(1−α) − μ)`, `gdp_compose(mus) = √Σμᵢ²`, `gdp_to_epsilon_delta` via Φ approximation (Abramowitz-Stegun 7.1.26), `gaussian_mechanism_mu(Δ, σ) = Δ/σ`
- [x] `accounting/zcdp.rs` — zCDP (Bun-Steinke 2016): `ρ = Δ²/(2σ²)`, composition `ρ_total = Σρᵢ`, `(ε, δ)` conversion `ε = ρ + 2√(ρ · ln(1/δ))`; tCDP with truncation ω
- [x] `accounting/prv.rs` — PRV accountant (Gopi et al. 2021): Gaussian PRV pmf on uniform grid `[grid_lo, grid_hi]`, O(n²) discrete convolution, `prv_delta(ε)`, `prv_epsilon(δ)` via binary search

#### Composition Theorems (composition/)
- [x] `composition/advanced.rs` — Basic `k·ε₀ / k·δ₀`; strong composition (Dwork-Rothblum-Vadhan 2010): `ε₀√(2k·ln(1/δ')) + k·ε₀(e^ε₀ − 1)`; heterogeneous composition
- [x] `composition/amplification_subsampling.rs` — Poisson subsampling: `ln(1 + q(e^ε − 1))`; uniform without-replacement: exact Balle et al. bound
- [x] `composition/amplification_shuffling.rs` — Erlingsson et al. (2019) shuffling bound: `ε ≤ log(1 + (e^ε₀ − 1)/(e^ε₀ + 1) · 8√(2 ln(4/δ)/n))`

#### Private Optimisers (optimizer/)
- [x] `optimizer/dp_ftrl.rs` — DP-FTRL with binary tree aggregation (Kairouz et al. 2021): noise tree of depth `max_depth`, path-based accumulation per step, FTRL update with L2 regularisation
- [x] `optimizer/dp_adam.rs` — DP-Adam: per-sample L2 gradient clip, aggregate + Gaussian noise, Adam `β₁ / β₂ / ε` moment updates

#### Local Differential Privacy (local/)
- [x] `local/grr.rs` — GRR (Generalised Randomised Response) k-ary: `P(output = v | input = v) = e^ε / (e^ε + k − 1)`, unbiased frequency estimator
- [x] `local/oue.rs` — OUE (Optimised Unary Encoding, Wang et al. 2017): one-hot encoding, per-bit Bernoulli flip with `p = 1/(e^ε + 1)`, unbiased frequency estimator
- [x] `local/rappor.rs` — RAPPOR simplified: Bloom-filter hash to k positions, per-bit flip at rate `1/(e^{ε/k} + 1)`, frequency decode

#### Sensitivity Analyses (sensitivity/)
- [x] `sensitivity/local_sensitivity.rs` — `LS_mean`, `LS_median`, `LS_sum`; calibrated noise addition via `Lap(LS/ε)`
- [x] `sensitivity/smooth_sensitivity.rs` — β-smooth sensitivity for mean (`= 1/n` global) and median (order-statistics walk); noise at scale `S^β / (ε − β)`; `β < ε` validation

#### Diagnostics (metrics/)
- [x] `metrics/metrics.rs` — `PrivacyBudget` tracker (spend / remaining / fraction), `gaussian_mse`, `snr_db`, `gaussian_utility`, `subsampling_amplification_factor`

#### GPU PTX Kernels
- [x] `exponential_sample` — Cumulative-weight sampling for exponential mechanism
- [x] `laplace_noise` — Laplace noise generation
- [x] `gaussian_noise` — Box-Muller Gaussian noise generation
- [x] `clip_gradient` — Per-sample L2 gradient clipping for DP-SGD
- [x] `svt_threshold` — SVT threshold noising
- [x] `prv_convolve` — Discrete PRV convolution
- [x] `oue_encode` — OUE per-bit Bernoulli encoding

### Future Enhancements [ ]

#### P0 — Verification on GPU Hardware
- [ ] End-to-end GPU verification of all PTX kernels under Linux + NVIDIA driver 525+
- [ ] Criterion benchmark suite executed on real hardware
- [ ] Numerical equivalence between CPU reference and GPU PTX path for noise distributions (KS-test on samples)

#### P1 — Algorithm Coverage
- [x] Discrete Gaussian mechanism (`mechanism/discrete_gaussian.rs` -- Canonne-Kamath-Steinke 2020) with zCDP → (ε, δ) conversion
- [x] Discrete Laplace mechanism (`mechanism/discrete_laplace.rs` -- Ghosh-Roughgarden-Sundararajan 2012 geometric mechanism)
- [x] Privacy-loss-distribution (PLD) accountant for arbitrary mechanisms (`accounting/pld.rs` + `accounting/pld_tests.rs` -- Meiser-Mohammadi 2018 / Koskela-Honkela 2020; discrete-grid PLD with Gaussian closed-form construction, convolution composition, repeated-squaring `compose_self(k)`, bisection ε(δ))
- [x] Connect-the-dots (CTD) accountant (`accounting/ctd.rs` + `accounting/ctd_tests.rs` -- Doroshenko-Ghazi-Kamath-Kumar-Manurangsi 2022; pessimistic/optimistic PLD bracketing on uniform grid with Gaussian closed-form construction via Φ-difference, O(n²) discrete convolution composition with boundary clamping, repeated-squaring `compose_self(k)`, pessimistic δ(ε) and bisection ε(δ))
- [x] FFT-accelerated PRV convolution (`accounting/prv_fft.rs` -- Gopi-Komargodski-Manurangsi-Shenfeld-Sherali-Yu 2021 NeurIPS + Cooley-Tukey 1965; inline radix-2 Cooley-Tukey FFT with bit-reversal permutation on `Cplx{re,im}` pairs, zero-pad to next-power-of-two, pointwise complex multiply, inverse FFT, real-part extract; O(n log n) vs O(n²) for PRV accountant composition; `compose_self_fft` repeated-squaring `log₂(n)` levels with grid re-projection)
- [x] Truncated Concentrated DP (Bun-Steinke) full implementation (`accounting/tcdp.rs` -- Bun-Steinke 2016 TCC LNCS 9985:635; TcdpMechanism with ρ and ω∈(0,∞], optimal-α minimization of ρ·α+ln(1/δ)/(α−1) over (1,ω], interior α*=1+√(ln(1/δ)/ρ) or boundary α=ω, δ↔ε inversion via geometric-bisection, Poisson subsampling ρ_sub=q²ρ, TcdpAccountant with sequential rho addition)
- [x] Renyi DP for Skellam mechanism (integer Gaussian alternative) (`mechanism/skellam.rs` -- Agarwal-Suresh-Yu-Kumar-McMahan 2021 NeurIPS; Pois(μ)−Pois(μ) via Knuth or normal-approx, closed-form RDP `ε_R(α) ≤ α·Δ²/(2μ) + …` with L2 refinement, α-grid (ε,δ) conversion)
- [x] Tight subsampling amplification (Wang-Balle-Kasiviswanathan 2019) under Renyi DP — accounting/rdp_subsampling.rs (binomial-sum formula, Gaussian/Laplace/Custom RDP, optimal RDP→(ε,δ) conversion)
- [x] Shuffle-DP tighter Feldman-McMillan-Talwar bound — accounting/shuffle_dp.rs (FMT 2022 Theorem 1, amplify/multi/compose/min_users)
- [x] Sparse-vector with budget-aware adaptive thresholding (`selection/adaptive_svt.rs` -- Lyu-Su-Li 2017 PVLDB + Kaplan-Mansour-Nissim 2023 ALT; one-time threshold noise `Lap(2Δ/ε_T)` with `ε_T = threshold_budget_frac·ε_total`, per-query Laplace `Lap(4k·Δ/ε_Q)`, soft adaptation of public `current_threshold` diagnostic via `T_new = T_old + adapt_rate·(q − T_old)` while comparison preserves SVT proof with fixed noisy_threshold)
- [x] Numeric SVT (return noisy values, not just indicators) (`selection/numeric_svt.rs` -- Lyu-Su-Li 2017 PVLDB Algorithm 3; one-time `Lap(2Δ/ε₁)` threshold, per-query `Lap(4kΔ/ε₂)`, per-released-value `Lap(2kΔ/ε₃)`, k-response budget halt)
- [x] DP-PCA (covariance-perturbation method) — Analyze-Gauss mechanism (Dwork-Talwar-Thakurta-Zhang 2014) with symmetric Gaussian noise on the Gram matrix and inline cyclic-Jacobi eigendecomposition
- [x] DP-KMeans with private centroid update — Su-Cao-Wang-Li (2016) with per-iteration Gaussian-mechanism on cluster sums/counts and basic composition across rounds
- [x] PrivateHistogram with stability-based release (`selection/private_histogram.rs` -- Vadhan 2017 Foundations of DP Lecture 12 Thm 12.4; per-bin Laplace noise `Lap(1/ε)`, stability threshold `T = 1 + (2/ε)·ln(2/(δ·k))`, only bins with noisy count > T are released, `Suppressed` variant when no bins clear threshold, `release_top_k` restricts to top-k by noisy count before threshold test)

#### P1 — Local DP Coverage
- [x] Hadamard response (Acharya-Sun 2019) for frequency estimation (`local/hadamard_response.rs` -- single-bit LDP frequency oracle with Sylvester-Hadamard encoding, in-place fast Walsh-Hadamard inverse, bias-corrected aggregation)
- [x] Subset-selection mechanism (`local/subset_selection.rs` -- Ye-Barg 2017 IEEE-IT 63:6957; optimal LDP frequency oracle via uniformly-random k-subset selection, `p = k·e^ε/(k·e^ε + d−k)` inclusion probability for true input, partial Fisher-Yates sampling without replacement, unbiased estimator `f̂_j = (c_j/n − q)/(p − q)` where `q = p·(k−1)/(d−1) + (1−p)·k/(d−1)`, `optimal_k = round(d/(e^ε+1))` clamped to [1, d−1])
- [x] Local DP for heavy-hitters (PrivateHeavyHitter, TreeHist) (`local/heavy_hitters.rs` -- Bassily-Nissim-Stemmer-Thakurta-Thakkar 2017 NeurIPS TreeHist; binary prefix-tree descent over `domain_bits` levels, per-report `(ε,0)`-LDP randomiser via GRR over the full domain `2^{domain_bits}`, prefixes recovered by truncation and scored with the GRR unbiased frequency oracle `f̂ = (count/n − q)/(p − q)` over the coarsened prefix domain, candidates with estimated count ≤ `threshold·n` pruned level-by-level, surviving leaves returned as `(item, est_count)` sorted by descending count with ascending-item tie-break; LDP guarantee inherited from the single per-user GRR report by post-processing)
- [x] Mean estimation (Duchi-Jordan-Wainwright) (`local/mean_estimation.rs` -- Duchi-Jordan-Wainwright 2018 JASA; bounded scalar/vector LDP unbiased mechanism `Z ∈ {±B}` with `B = radius·(e^ε+1)/(e^ε−1)` bias-correction, per-coordinate composition for vector inputs)
- [x] Vector aggregation with bit-encoding (Piecewise mechanism) (`local/piecewise.rs` -- Wang-Xiao-Yang-Yi 2019 ICDE; piecewise distribution `C = (e^{ε/2}+1)/(e^{ε/2}-1)` with high-density region `[L(t), R(t)]` of width `C-1` sampled with prob `e^{ε/2}/(e^{ε/2}+1)`, two-piece low-density elsewhere, per-coordinate composition for vector inputs, lower variance than Duchi at same ε)

#### P2 — Optimiser Coverage
- [x] DP-SGD with microbatching (`optimizer/dp_sgd_microbatch.rs` -- Abadi et al. 2016 CCS + McMahan et al. 2018; per-microbatch averaging + L2 clip + Gaussian noise `N(0, σ²·C²·I)` + optional momentum, partial-microbatch handled with ceil division for divisor)
- [x] DP-SGD-MA (Moments-Accountant variant) end-to-end
- [x] DP-AdaGrad + DP-AdaDelta (`optimizer/dp_adagrad.rs` -- Duchi-Hazan-Singer 2011 + Abadi et al. 2016 CCS; per-sample L2 clip + Gaussian noise + coordinate-wise accumulator, adaptive step `θ[j] -= lr·g_priv[j]/(√accumulator[j] + ε)`) + (`optimizer/dp_adadelta.rs` -- Zeiler 2012 arXiv:1212.5701 + Abadi et al. 2016; per-sample L2 clip + Gaussian noise + EMA-tracked `E[g²]_t = ρ·E[g²]_{t-1} + (1−ρ)·g²`, `Δθ = -√(E[Δθ²]+ε)/√(E[g²]+ε)·g_priv`, `E[Δθ²]_t = ρ·E[Δθ²]_{t-1} + (1−ρ)·Δθ²`)
- [x] DP-LAMB for large-batch DP training (`optimizer/dp_lamb.rs` -- You-Li-Reddi-Hseu-Kumar-Bhojanapalli-Song-Demmel-Keutzer-Hsieh 2020 ICLR + Abadi et al. 2016 CCS; per-sample L2 clip C, Gaussian noise N(0,σ²C²/batch·I), Adam moments β₁/β₂ with bias correction, decoupled weight decay, LAMB trust ratio φ/ψ=‖θ‖/‖r‖ clamped to [min,max])
- [ ] DP-MASR (Adaptive sensitivity refinement)
- [ ] Tree-DP-FTRL with adaptive tree depth

#### P2 — Optimisations and Tooling
- [x] PATE private aggregation of teacher ensembles (`mechanism/pate.rs`) — Papernot 2017 ICLR: noisy argmax aggregation of teacher ensemble votes with Gaussian noise for student label privacy; `PateAggregator`
- [ ] Rényi DP accounting (`accountant/renyi_dp.rs`) — Mironov 2017 CSF: tight closed-form Rényi divergence composition for the Gaussian mechanism; `RenyiDpAccountant`
- [x] Sampled Gaussian mechanism (`mechanism/sampled_gaussian.rs`) — Balle 2020 NeurIPS: amplification-by-subsampling bound for Rényi DP of Poisson-subsampled Gaussian; `SampledGaussianMechanism`
- [x] Data sanitisation via local suppression (`sanitisation/suppression.rs`) — Sweeney 2002: k-anonymity-compliant quasi-identifier suppression + generalisation with bottom-up lattice traversal; `KAnonymiseSuppressor`
- [ ] Fused gradient-clip + Gaussian-noise kernel (saves one global-memory pass)
- [ ] Persistent CTA scheduling for repeated DP-Adam steps
- [ ] CUDA-graph capture for DP-SGD training loop
- [ ] Mixed-precision noise generation (FP16 sample, FP32 accumulate)
- [ ] On-device Philox / ChaCha20 random stream for deterministic noise replay
- [ ] Privacy-budget runtime monitor with circuit-breaker

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 696 passing (unit + 18 e2e integration tests in `e2e_tests.rs`)
- Warnings: 0 (clippy clean)
- `unwrap()` in production code: 0
- macOS: compiles, runtime returns `UnsupportedPlatform` for GPU launches
- All PTX kernels validated as non-empty strings for SM 75 / 80 / 86 / 89 / 90 / 100

## Performance Targets

DP kernels are dominated by RNG cost (noise generation) and reduction (clipping, aggregation). PRV-accountant convolution is currently O(n²); FFT replacement is P1.

| Operation | Target Reference | Notes |
|-----------|------------------|-------|
| Gaussian noise (N=10K) | ≥ 90% of cuRAND Philox-Normal | RNG-bound |
| Laplace noise (N=10K) | ≥ 90% of cuRAND Philox-Uniform + log | RNG-bound |
| Gradient clip (B=1024, D=1M) | ≥ 95% of cuBLAS nrm2 + scal | reduction-bound |
| Exponential sample (k=256) | ≥ 95% of cuBLAS softmax + scan | softmax + CDF |
| PRV convolve (n=1000, O(n²)) | n/a (CPU-side accountant) | replace with FFT for n > 4K |
| OUE encode (B=1024, k=256) | ≥ 90% of N×k Bernoulli sample | RNG-bound |

## Notes

- This crate is **complementary** to `oxicuda-federated::privacy` and intentionally does **not** duplicate `GaussianMechanism`, `LaplacianMechanism`, `MomentsAccountant`, `PateConfig`, or the RDP accountant — those live in the federated crate.
- All randomness flows through `LcgRng` for deterministic replay; production deployments should consider Philox / ChaCha20 / hardware RNG via a future `PrivacyHandle::with_rng` constructor.
- The PRV accountant uses an uniform discretisation grid; accuracy depends on `grid_size` and `grid_lo`/`grid_hi` choices (default 1000 points).
- Smooth-sensitivity supports mean and median; extending to other order statistics requires per-statistic bound derivation (P1).
- Local DP frequency estimators (GRR, OUE, RAPPOR) emit *unbiased* estimates with closed-form variance; production usage should aggregate over many users before releasing.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [ ] Validate Gaussian / Laplace noise kernels on T4 (FP16 storage)
- [ ] Block-size autotuning for per-sample gradient clip at small batch sizes

### Ampere (sm_80 / sm_86)
- [ ] `cp.async` staging of per-sample gradients for fused clip + noise on A100
- [ ] Tensor-Core (mma.sync) acceleration of OUE encoding when k is large

### Ada (sm_89)
- [ ] FP8 (e4m3 / e5m2) gradient storage with FP32 noise accumulation
- [ ] Sparse Tensor-Core path for top-k gradient release

### Hopper (sm_90)
- [ ] TMA-based bulk gradient-tensor staging for very large batch DP-SGD
- [ ] Cluster-wide reduce for population-scale DP aggregation
- [ ] Asynchronous transaction barrier for overlapping clip with noise generation

### Blackwell (sm_100)
- [ ] `tcgen05` tensor memory layout for FP4 / FP6 DP-Adam
- [ ] 5th-generation Tensor Core for low-precision DP training

---

## Deepening Opportunities

### Verification Gaps
- [ ] All 7 PTX kernels executed end-to-end on GPU hardware (currently only string-content verified)
- [ ] KS-test on Gaussian / Laplace noise samples (CPU vs GPU PTX path)
- [ ] PRV accountant accuracy vs Renyi-DP composition for representative Gaussian-mechanism sequences
- [ ] DP-Adam convergence on a private MNIST or CIFAR-10 task with empirical (ε, δ) report
- [ ] SVT k-budget exhaustion behaviour verified for k = 100 / 1K / 10K queries

### Algorithmic Deepening
- [ ] Exponential mechanism with alias method for very large output sets (k > 10K)
- [ ] PTR with multiple-round local-sensitivity refinement
- [ ] SVT with composition across multiple sparse-vector streams
- [ ] f-DP graphical composition (PLD-based numerical f-composition)
- [ ] zCDP-to-RDP conversion verified for representative Gaussian sequences
- [ ] PRV accountant with adaptive grid refinement for tight (ε, δ) reports
- [ ] DP-FTRL with momentum and bias correction

### Coverage Gaps vs Literature
- [x] Concentrated DP (CDP) original definition by Dwork-Rothblum (2016)
- [x] Privacy amplification by iteration (Feldman-Mironov-Talwar 2018)
- [ ] Local DP under intermittent communication
- [ ] Histogram release with stability-based threshold release
- [ ] Private synthetic data generation (PATE-GAN, DP-GAN with our DP-Adam)
- [x] Private hyperparameter tuning (Liu-Talwar 2019) (`selection/private_tuning.rs` -- Liu-Talwar 2019 STOC "Private Selection from Private Candidates"; random-stopping selection over base `(ε₀,δ₀)`-DP candidates supplied as a closure `Fn(usize, &mut LcgRng) -> (f32, T)`, hidden trial count `K` drawn from `StoppingRule::{Geometric(γ) P(K=k)=(1−γ)^{k−1}γ, Poisson(λ) shifted K=1+Pois(λ), Fixed(n)}`, returns argmax-score candidate (first on ties); privacy transform `tuning_epsilon` = `3·ε₀` for Geometric (Thm 3.1, constant independent of γ) and Poisson (smooth-stopping §4), `n·ε₀` for Fixed; `tuning_delta` = `3·e^{ε₀}·δ₀` for random-stopping, `n·δ₀` for Fixed)
- [ ] DP-aware learning-rate schedulers
- [ ] Renyi-DP central / RDP-accountant integration with `oxicuda-federated`
