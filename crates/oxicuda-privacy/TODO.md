# oxicuda-privacy TODO

Pure Rust Differential Privacy primitives covering mechanisms (exponential / report-noisy-max / propose-test-release), selection (sparse-vector technique / above-threshold), accounting (f-DP / GDP, zCDP / tCDP, PRV), composition (advanced, subsampling and shuffling amplification), private optimisers (DP-FTRL, DP-Adam), local DP (GRR, OUE, RAPPOR), and sensitivity analyses (local, smooth), with PTX kernel templates for SM 7.5 through SM 10.0. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.46).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 28,952 SLoC (89 files)**

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
- [x] DP-MASR (Adaptive sensitivity refinement) (`optimizer/dp_masr.rs` -- Andrew-Thakkar-McMahan-Ramaswamy 2021 NeurIPS adaptive clipping + Pichapati-Suresh-Yu-Reddi-Kumar 2019 AdaCliP; private γ-quantile feedback via Gaussian-mechanism on the per-example bit-mean `𝟙[‖g‖≤C]` (sensitivity 1/m, noise σ_b/m), geometric log-space clip refinement `C ← C·exp(−η_C·(b̃−γ))`, refined-sensitivity Gaussian gradient step `N(0,σ²C²/m²)`, per-step ρ-zCDP `1/(2σ_b²)+1/(2σ²)` accumulated)
- [x] Tree-DP-FTRL with adaptive tree depth (`optimizer/dp_ftrl_adaptive.rs` -- Kairouz et al. 2021 ICML + Chan-Shi-Song 2011 streaming binary mechanism; online tree-doubling forest of dyadic blocks indexed by the binary representation of the step count, per-completed-block independent Gaussian node noise drawn on the binary carry, noisy prefix sum = sum of held block sums, effective depth `⌈log₂(t+1)⌉` = popcount(t) with NO pre-set horizon `T`, FTRL update on the noisy prefix sum)

#### P2 — Optimisations and Tooling
- [x] PATE private aggregation of teacher ensembles (`mechanism/pate.rs`) — Papernot 2017 ICLR: noisy argmax aggregation of teacher ensemble votes with Gaussian noise for student label privacy; `PateAggregator`
- [x] Rényi DP accounting — Mironov 2017 CSF: tight closed-form Rényi divergence composition for the Gaussian mechanism; `RenyiDpAccountant` (ALREADY EXISTS as `accounting/rdp_gaussian.rs` -- closed-form `ε_RDP(α)=α/(2σ²)`, additive composition `add_gaussian_step`/`compose`, dual Mironov + Canonne-Kamath-Steinke (ε,δ) conversion minimised over an α-grid). Bridged to zCDP by new `accounting/zcdp_rdp.rs` (`zcdp_to_rdp_curve`/`rdp_curve_to_zcdp`/`zcdp_epsilon_via_rdp`)
- [x] Sampled Gaussian mechanism (`mechanism/sampled_gaussian.rs`) — Balle 2020 NeurIPS: amplification-by-subsampling bound for Rényi DP of Poisson-subsampled Gaussian; `SampledGaussianMechanism`
- [x] Data sanitisation via local suppression (`sanitisation/suppression.rs`) — Sweeney 2002: k-anonymity-compliant quasi-identifier suppression + generalisation with bottom-up lattice traversal; `KAnonymiseSuppressor`
- [x] Fused gradient-clip + Gaussian-noise kernel (saves one global-memory pass) (`noise/fused_clip_noise.rs` -- single-pass per-vector L2 clip + `N(0,σ²C²)` noise (`fused_clip_and_noise` / `fused_clip_and_noise_in_place`) after the unavoidable norm reduction, fusing scale + noise into one multiply-add without materialising the clipped intermediate; verified BIT-FOR-BIT identical to the two-pass `sequential_clip_then_noise` reference for the same RNG state, plus noise-scale/clip-bound checks) [CPU reference; on-device GPU fusion stays GPU-gated]
- [ ] Persistent CTA scheduling for repeated DP-Adam steps
- [ ] CUDA-graph capture for DP-SGD training loop
- [x] Mixed-precision noise generation (FP16 sample, FP32 accumulate) (`noise/mixed_precision.rs` -- complete pure-Rust IEEE-754 binary16 round-trip (`f32_to_f16_bits`/`f16_bits_to_f32`/`quantize_f16`) with round-to-nearest-even, gradual-underflow subnormals, overflow→±∞, NaN/∞ propagation (full 65536-pattern idempotence sweep); `mixed_precision_gaussian` samples Gaussian noise, FP16-quantises each draw and Welford-accumulates mean/variance in FP32 (variance within 3% of σ²); `add_fp16_noise_fp32_accumulate` adds FP16-stored noise into an FP32 accumulator)
- [x] On-device Philox / ChaCha20 random stream for deterministic noise replay (`rng/philox.rs` + `rng/chacha20.rs` -- pure-Rust counter-based RNGs: Philox 4×32-10 (Salmon et al. 2011 Random123, verified against canonical KAT vectors + an independent reference) and the RFC 8439 ChaCha20 block function (verified against the RFC 8439 §2.3.2 block KAT), each a seekable `(key, counter)→stream` with `next_u32`/`u64`/`f64`/`f32`/`normal_pair` and O(1) `seek` for replay; full [0,1) uniforms via ÷2³²) [CPU counter-based RNG done; on-device GPU port stays GPU-gated]
- [x] Privacy-budget runtime monitor with circuit-breaker (ALREADY EXISTS as `accounting/budget_monitor.rs` -- `BudgetMonitor`/`CompositionMode` tracks cumulative (ε,δ) and `try_spend` refuses-without-committing any query that would exceed the total budget; supports Basic and Advanced (Dwork-Rothblum-Vadhan) composition modes)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 816 passing (unit + 20 e2e integration tests in `e2e_tests.rs`; includes 22 counter-based-RNG tests in `rng/`, 19 mixed-precision/fused-noise tests in `noise/`, 6 DP-Adam convergence-harness tests in `optimizer/dp_adam_harness.rs`, and 9 DP synthetic-data (PATE-GAN / DP-GAN) tests in `mechanism/synthetic_data.rs`)
- Warnings: 0 (clippy clean, `--all-features --all-targets -D warnings`)
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
- The default noise path uses `LcgRng` for deterministic replay; for reproducible counter-based noise replay (seek to an arbitrary `(key, counter)` and reproduce a draw bit-for-bit) the crate now also ships pure-Rust `rng::PhiloxRng` (Philox 4×32-10) and `rng::ChaCha20Rng` (RFC 8439 ChaCha20), parallel CBRNGs with the same `next_*`/`normal_pair` surface and O(1) `seek`.
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
- [x] PRV accountant accuracy vs Renyi-DP composition for representative Gaussian-mechanism sequences (verified in `e2e_tests.rs::prv_accountant_matches_renyi_dp_composition` -- composes k=8 identical σ=2 Gaussian steps via both `adaptive_epsilon` (PRV) and `RenyiDpAccountant` (RDP) and asserts PRV ε ≤ RDP ε + slack and |PRV−RDP| < 0.5 at δ=1e-5)
- [x] DP-Adam convergence with empirical (ε, δ) report (`optimizer/dp_adam_harness.rs` -- self-contained CPU harness `DpAdamHarness`: synthetic linear-regression dataset generated in-process via the crate `LcgRng` (NO external MNIST/CIFAR download), minibatch-subsampled DP-Adam training over several epochs, with the Sampled-Gaussian moments accountant composing one RDP term/step; `run` reports per-epoch full-dataset loss + per-epoch spent ε at the target δ; test asserts the loss at least halves AND ε is finite, positive and monotonically grows with steps, plus determinism and ground-truth recovery). [GPU/MNIST-dataset variant remains out of scope]
- [x] SVT k-budget exhaustion behaviour verified for k = 100 / 1K / 10K queries (verified in `e2e_tests.rs::svt_k_budget_exhaustion_behaviour` -- drives an always-above stream into `SvtState` for each k and asserts exactly k True answers then halt/error on the next query)

### Algorithmic Deepening
- [x] Exponential mechanism with alias method for very large output sets (k > 10K) (ALREADY EXISTS as `mechanism/exponential_alias.rs` -- Walker-Vose alias table, O(1) per-sample McSherry-Talwar exponential mechanism, `ExponentialAlias::{new,sample,probabilities}`)
- [x] PTR with multiple-round local-sensitivity refinement (`mechanism/ptr_multiround.rs` -- Dwork-Lei 2009 PTR + Nissim-Raskhodnikova-Smith 2007 local-sensitivity-at-distance; descending sensitivity ladder `b₁>…>b_R`, budget split `ε_test`/`ε_rel`, per-round PTR test `s_r+Lap(R/ε_test) ≤ ln(R/(2δ))·R/ε_test` over `R` composed rounds, release `output+Lap(b_r*/ε_rel)` at the TIGHTEST passing rung else abstain; total `(ε_test+ε_rel, δ)`-DP; `geometric_ladder` helper)
- [x] SVT with composition across multiple sparse-vector streams (`selection/svt_multistream.rs` -- Dwork-Roth §3.6 + Lyu-Su-Li 2017; `MultiStreamSvt` holds S independent `SvtState` streams each `(ε_s,0)`-DP with its own k_s-True cap, routes queries per-stream and halts each at its cap, reports composed total via `SvtCompositionMode::{Basic ε=Σε_s, Advanced strong-composition over ε₀=maxₛε_s}`)
- [x] f-DP graphical composition (PLD-based numerical f-composition) (`accounting/fdp_composition.rs` -- Dong-Roth-Su 2022 f-DP + Koskela-Jälkö-Honkela 2020 FFT; `FdpPld` discrete privacy-loss distribution, Gaussian dominating-pair construction on a shared grid, O(n·m) lattice convolution `compose_two`/`compose_many`/`compose_self` (repeated-squaring), δ(ε) hockey-stick map, trade-off recovery `f(α)=sup_ε[1−δ(ε)−e^ε·α]` via Legendre duality `tradeoff_from_pld`, verified against analytic Gaussian `√k·μ`-GDP)
- [x] zCDP-to-RDP conversion verified for representative Gaussian sequences (`accounting/zcdp_rdp.rs` -- Bun-Steinke 2016 Prop 1.4 + Mironov 2017; `zcdp_to_rdp_curve` samples ε_R(α)=ρα, `rdp_curve_to_zcdp` inverts a general curve to tightest ρ=maxₐ ε_R(α)/α (exact for the linear Gaussian curve, test-verified to recover ρ=Δ²/2σ²), `zcdp_epsilon_via_rdp` optimises ε=ρα+ln(1/δ)/(α−1) over the α-grid + analytic optimum α*=1+√(ln(1/δ)/ρ) and is test-verified equal to the closed-form ρ+2√(ρ·ln(1/δ)))
- [x] PRV accountant with adaptive grid refinement for tight (ε, δ) reports (`accounting/prv_adaptive.rs` -- Gopi-Lee-Wutschitz 2021; analytic grid placement centred on composed mean `k·μ_Z` with half-width `grid_sigmas·√k·σ_Z`, grid-density doubling until successive δ(ε)/ε(δ) estimates agree within `tol` or `max_grid_size`, O(n log n) FFT composition via `compose_gaussian_prv_fft`, `AdaptivePrvResult` reports convergence + refinements)
- [x] DP-FTRL with momentum and bias correction (`optimizer/dp_ftrl_momentum.rs` -- Kairouz et al. 2021 ICML Algorithm 2; binary-tree prefix-sum Gaussian noise on the cumulative gradient sum, heavy-ball momentum `m_t=β·m_{t-1}+(1−β)·S̃_t` over the NOISY cumulative sum, Kingma-Ba bias correction `m̂_t=m_t/(1−βᵗ)` (test-verified first step unscaled), FTRL update `θ_t=θ_{t-1}−η·m̂_t/t`; momentum/bias-correction are post-processing → no extra privacy cost)

### Coverage Gaps vs Literature
- [x] Concentrated DP (CDP) original definition by Dwork-Rothblum (2016)
- [x] Privacy amplification by iteration (Feldman-Mironov-Talwar 2018)
- [ ] Local DP under intermittent communication
- [x] Histogram release with stability-based threshold release (ALREADY EXISTS as `selection/private_histogram.rs` -- Vadhan 2017 Thm 12.4; per-bin `Lap(1/ε)`, stability threshold `T=1+(2/ε)·ln(2/(δ·k))`, releases only bins with noisy count > T, `Suppressed` when none clear, `release_top_k` restricts to top-k first)
- [x] Private synthetic data generation (PATE-GAN, DP-GAN with our DP-Adam) (`mechanism/synthetic_data.rs` -- Jordon-Yoon-van-der-Schaar 2019 PATE-GAN + Xie et al. 2018 DP-GAN; genuine two-layer MLP generator (tanh hidden + tanh-bounded output) and MLP discriminators with hand-written analytic forward/back-prop (incl. back-prop of the discriminator signal through to the generator's parameters -- no stubbed loop). `pate_gan_train`: teachers trained on disjoint private partitions vs current fakes, generated samples labelled through the EXISTING `pate::pate_aggregate` LNMax noisy-argmax (scale `2/ε_q`, L1 vote sensitivity 2), student trained on the DP labels, generator updated against the student; cumulative pure-ε spend composed with the crate's `BudgetMonitor` (basic composition). `dp_gan_train`: discriminator optimised with the EXISTING `DpAdamState` (per-sample clip + Gaussian noise) on per-sample real/fake BCE gradients, generator updated from the DP discriminator signal, privacy accounted with the EXISTING `SampledGaussianMechanism` moments accountant (q=batch/n_rows, one RDP term/step). `SyntheticGenerator::sample(n)` returns n finite tanh-bounded rows, deterministic under a fixed seed; `dp_label_from_votes` exposes the DP voting primitive. 9 tests: accounting consistency vs `BudgetMonitor`/`SampledGaussianMechanism` (pulled not hardcoded), tighter-budget⇒more-noise + larger-σ⇒smaller-ε monotonicity, sample structure/finiteness/determinism, unanimous-vote correctness degrading toward random as ε→0, real-gradient sanity, input validation. [Distributional fidelity NOT asserted -- requires un-verifiable training scale])
- [x] Private hyperparameter tuning (Liu-Talwar 2019) (`selection/private_tuning.rs` -- Liu-Talwar 2019 STOC "Private Selection from Private Candidates"; random-stopping selection over base `(ε₀,δ₀)`-DP candidates supplied as a closure `Fn(usize, &mut LcgRng) -> (f32, T)`, hidden trial count `K` drawn from `StoppingRule::{Geometric(γ) P(K=k)=(1−γ)^{k−1}γ, Poisson(λ) shifted K=1+Pois(λ), Fixed(n)}`, returns argmax-score candidate (first on ties); privacy transform `tuning_epsilon` = `3·ε₀` for Geometric (Thm 3.1, constant independent of γ) and Poisson (smooth-stopping §4), `n·ε₀` for Fixed; `tuning_delta` = `3·e^{ε₀}·δ₀` for random-stopping, `n·δ₀` for Fixed)
- [x] DP-aware learning-rate schedulers (`optimizer/lr_scheduler.rs` -- Loshchilov-Hutter 2017 cosine + Goyal et al. 2017 warmup + DP-SGD SNR literature; `LrSchedule` enum with classic Constant/StepDecay/Exponential/CosineWarmup plus DP-specific `BudgetAware` η=η₀·(1−ρ_spent/ρ_total)^p annealing keyed to zCDP expenditure and `NoiseAware` η=η₀/(1+κ·σ·C) damping by injected DP-noise magnitude; `lr_at_step`/`lr_at_budget`/`lr_at_noise`)
- [ ] Renyi-DP central / RDP-accountant integration with `oxicuda-federated`
