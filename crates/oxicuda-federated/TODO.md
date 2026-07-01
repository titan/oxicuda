# oxicuda-federated TODO

Pure-Rust federated learning primitives for OxiCUDA: server algorithms
(FedAvg / FedProx / SCAFFOLD / FedAdam), gradient compression, differential
privacy mechanisms with RDP / Moments accountants, secure aggregation
(Shamir + pairwise masking), and client selection. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.24).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 12,084 (44 files)
- **PTX kernels:** 7 kernel generators emitted for 6 SM targets (sm_75 / 80 / 86 / 90 / 100 / 120)
- **Coverage:** CPU reference implementation + PTX string generation for GPU execution

### Completed

#### Core Infrastructure
- [x] `error.rs` (130 LoC) -- `FedError` (17 variants: NoClients, EmptyGradient, DimensionMismatch, InvalidEpsilon, InvalidNoiseScale, InvalidThreshold, ShamirReconstructFailed, InsufficientShares, InvalidLearningRate, InvalidComprRank, NumberOfClientsBelowMinimum, NanEncountered, Internal, ...) + `FedResult<T>`
- [x] `handle.rs` (318 LoC) -- `SmVersion`, `LcgRng` with Knuth MMIX 64-bit core + Box-Muller normals, Fisher-Yates shuffle, Gaussian / Laplace samplers; `FedHandle::default_handle()`
- [x] `lib.rs` (229 LoC) -- Module exports + `prelude` re-exports + 10 E2E integration tests

#### PTX Kernels (ptx_kernels.rs, 628 LoC)
- [x] `aggregate_mean_ptx` -- Element-wise average across K client gradient buffers (`fma.rn.f32` accumulation)
- [x] `dp_clip_gradient_ptx` -- Per-sample L2-norm gradient clipping for DP-SGD with `fma.rn.f32` partial-sum reduction
- [x] `fedavg_weighted_sum_ptx` -- Sample-count-weighted FedAvg server update
- [x] `gaussian_noise_ptx` -- Box-Muller Gaussian noise from inline LCG using `lg2.approx.f32` / `ex2.approx.f32`
- [x] `pairwise_mask_ptx` -- Additive pairwise mask `(seed_i XOR seed_j)`-driven for secure aggregation cancellation
- [x] `qsgd_quantize_ptx` -- QSGD stochastic quantisation with dithering and `selp.f32` rounding decisions
- [x] `topk_mask_ptx` -- Top-K sparsification mask via threshold comparison

#### Server Algorithms (algorithm/)
- [x] `fedavg.rs` (214 LoC) -- `FedAvgConfig` / `FedAvgState` sample-weighted parameter averaging (McMahan 2017); `aggregate(&[(grad, weight)])` mutates `global_params` and bumps `round`
- [x] `fedprox.rs` (190 LoC) -- `FedProxConfig` proximal regularisation `mu/2 * ||theta - theta_global||^2`; `proximal_loss`, `proximal_gradient`, `fedprox_client_loss_correction` (Li 2020)
- [x] `scaffold.rs` (269 LoC) -- `ScaffoldClientState` / `ScaffoldState` with control variates `c_i, c` correcting client drift; `scaffold_client_update` + `scaffold_server_aggregate` (Karimireddy 2020)
- [x] `fedadam.rs` (255 LoC) -- `FedAdamState` server-side Adam with momentum / second-moment / `ServerOptimizerKind` (Adam / Yogi / AMSGrad) (Reddi 2021)

#### Compression (compression/)
- [x] `powersgd.rs` (277 LoC) -- `PowerSgdCompressor` low-rank power-iteration compression with error feedback and `frobenius_norm` + `residual` helpers (Vogels 2019)
- [x] `quantize.rs` (164 LoC) -- `stochastic_quantize` / `dequantize` QSGD bit-budget quantisation with dithering; `gradient_norm`, `max_quantization_error`
- [x] `randomk.rs` (150 LoC) -- `random_sparsify` with deterministic compression ratio; `compression_ratio` helper
- [x] `topk.rs` (151 LoC) -- `topk_sparsify` magnitude sparsification + `error_feedback` residual update

#### Differential Privacy (privacy/)
- [x] `gaussian.rs` (197 LoC) -- `GaussianMechanism::new(epsilon, sensitivity, delta)` -> calibrated Gaussian noise for L2-bounded queries
- [x] `laplacian.rs` (164 LoC) -- `LaplacianMechanism` + free function `add_laplacian_noise` for L1-bounded queries (CDF-inversion sampler)
- [x] `moments.rs` (187 LoC) -- `MomentsAccountant` epsilon tracking for DP-SGD across rounds (Abadi 2016)
- [x] `rdp.rs` (157 LoC) -- `rdp_gaussian`, `rdp_to_dp`, `compose_rdp`, `optimal_epsilon` Renyi-DP composition with conversion to (eps, delta)-DP
- [x] `pate.rs` (233 LoC) -- `PateConfig`, `noisy_voting`, `data_dependent_epsilon` for PATE student-teacher voting (Papernot 2017)

#### Secure Aggregation (secure_agg/)
- [x] `shamir.rs` (340 LoC) -- `ShamirConfig::new(k, n)`, `share_scalar` / `share_gradient` / `reconstruct_scalar` / `reconstruct_gradient` over a Mersenne-prime field (`PRIME`)
- [x] `masking.rs` (194 LoC) -- `generate_mask`, `apply_mask`, `apply_pairwise_masks`, `unmask` Bonawitz-style additive masking that cancels in aggregation
- [x] `aggregator.rs` (152 LoC) -- `SecureAggregator` orchestrates mask-then-aggregate flow across a fixed cohort

#### Client Selection (selection/)
- [x] `random.rs` (152 LoC) -- `random_select` (uniform without-replacement) + `stratified_select` (stratum-balanced) using the handle RNG

#### Integration Tests (lib.rs)
- [x] 10 E2E tests: FedAvg mean recovery, FedProx proximal term magnitude, Top-K + error feedback compensation, QSGD unbiased-estimator empirical mean, Gaussian DP noise variance, RDP linear composition, Shamir scalar + gradient round-trip, `random_select` uniqueness, PTX kernels x 6 SM versions

#### Benchmarks (benches/fed_ops.rs)
- [x] 7 PTX kernel generator groups x 4 SM versions + `fedavg_aggregate` + `topk_sparsify` + `qsgd_quantize` + `shamir_share` / `shamir_reconstruct`

### Future Enhancements

#### P0 -- Critical (Algorithm Coverage Gaps)
- [x] Byzantine-robust aggregators — algorithm/robust_agg.rs (Krum, Multi-Krum, Trimmed-Mean, Median, Bulyan; Blanchard 2017 NeurIPS + Yin 2018 ICML + El Mhamdi 2018 NeurIPS)
- [x] FedNova — algorithm/fednova.rs (normalized averaging, heterogeneous local steps, momentum τ-correction; Wang 2020 NeurIPS)
- [x] Personalised FL (Per-FedAvg / pFedMe / Ditto) -- bi-level meta-learning style personalisation alongside global model
- [x] DP-FTRL accountant -- tree-aggregation-based DP-SGD alternative to Moments / RDP for per-round amplification

#### P1 -- Important (Communication & Privacy Depth)
- [x] Signed-SGD compression -- 1-bit gradient sign communication with majority-vote server reconstruction
- [x] Atomo / TernGrad ternary quantisation -- 2-bit gradient codes for ultra-low-bandwidth links
- [x] Sketched updates (compression/sketch.rs -- Count-Sketch depth×width sign-hash table with median estimator + linear merge + heavy-hitter top-k; orthonormal Fast Walsh-Hadamard transform with ±1 diagonal and exact inverse, pow2 padding)
- [x] Diffie-Hellman key agreement helpers for secure_agg (secure_agg/key_exchange.rs -- finite-field DH over the Mersenne prime p=2^61-1; `DhKeyPair::{generate, from_private, public, shared_field_element, shared_seed}` with SplitMix64 seed diffusion; certified primitive root g=37; `pairwise_seed_matrix` builds the symmetric n×n shared-seed table feeding `masking::apply_pairwise_masks` -- closes the caller-supplied-seed gap)
- [x] DP-SGD client-side gradient clipping helper -- ALREADY EXISTS: `GaussianMechanism::clip_gradient` (privacy/gaussian.rs), `DpFtrl::clip_gradient` (privacy/dp_ftrl.rs), `clip_l2` (privacy/ldp_fl.rs) all mirror `dp_clip_gradient_ptx` on the CPU side (g ← g·min(1, C/‖g‖))
- [x] Local-DP randomised response for categorical metadata (privacy/randomized_response.rs -- Warner 1965 extended to k-RR: p=e^ε/(e^ε+k−1) truth/uniform-other; unbiased frequency aggregator inverting the perturbation)
- [x] Asynchronous FedBuff / FedAsync schedulers (algorithm/fedbuff.rs -- Nguyen 2022; server buffer of K most-recent client updates, staleness-weighted 1/(1+α·s) average applied with η_g when buffer fills; non-blocking async client submissions)

#### P2 -- Nice-to-Have (Operational Polish)
- [x] Client drift diagnostics (algorithm/scaffold.rs -- `control_variate_drift` computes `‖c_i − c‖`; `gradient_norm_histogram` bins per-client norms into a `DriftDiagnostics { values, bins, min, max, mean, std }`; `scaffold_drift_diagnostics` runs both across a client cohort)
- [x] Adaptive client selection (selection/power_of_choice.rs -- Cho et al. 2020; Efraimidis-Spirakis weighted-without-replacement candidate sampling ∝ data size + top-m by local loss; LossBased / AvailabilityAware / Random variants)
- [x] Cohort fairness metrics (selection/fairness.rs -- `StratumMetrics` per-cohort accuracy/loss; `fairness_summary` reports mean/min/max accuracy, std, max−min gap, max loss, Jain's fairness index; `CohortFairnessTracker` accumulates across rounds + `worst_stratum`; Li q-FFL + Jain 1984)
- [x] PATE Confident-GNMax voter (privacy/pate.rs -- `confident_gnmax` + `ConfidentGnMaxConfig { threshold T, sigma_threshold σ₁, sigma_answer σ₂ }`; noisy plurality `max+N(0,σ₁²)` confidence check abstains (returns `None`) below T, else GNMax noisy-argmax with σ₂; Papernot 2018 ICLR §4.1)
- [x] Renyi-DP zCDP conversion (privacy/zcdp.rs -- `zcdp_gaussian` ρ=1/(2σ²); `rdp_to_zcdp`/`zcdp_to_rdp` exact (α,ρα)-RDP correspondence; `zcdp_to_dp` optimised ε=ρ+2√(ρ·ln(1/δ)); `ZcdpAccountant` additive ρ composition; Bun-Steinke TCC 2016)
- [x] `federated/moon.rs` — MOON (Li 2021): Model-Contrastive Federated Learning; contrastive loss between current model + global model (positive) vs previous model (negative); representation alignment; `MoonConfig { mu: f32, temperature: f32 }`
- [x] `federated/feddf.rs` — FedDF (Lin 2020) — ALREADY EXISTS at algorithm/feddf.rs (`FedDf`, `FedDfConfig`, `LinearModel`, ensemble logit distillation on public data via `softmax_with_temperature`/`argmax`; exported from prelude)
- [x] `federated/flute.rs` — FLUTE (Dimitriadis 2022) — ALREADY EXISTS at algorithm/flute.rs (`Flute`, `FluteConfig`, `FluteModel` shared body + personalised heads, `FluteClientUpdate`/`FluteSample`; exported from prelude)
- [x] `privacy/ldp_fl.rs` — Local DP for Federated Learning (Truex 2020): Gaussian/Laplace mechanism on client updates before transmission; privacy amplification via subsampling; `LdpFlConfig { epsilon, delta, clip_norm }`

#### GPU Launcher Wiring
- [ ] Wire `ptx_kernels::*` strings through `oxicuda-launch::Kernel::from_module` for end-to-end GPU execution (PTX strings are emitted but currently only CPU paths are exercised end-to-end) (requires GPU hardware)
- [ ] GPU-resident `aggregate_mean_ptx` launch fused with `dp_clip_gradient_ptx` for a single-kernel DP-FedAvg server step (requires GPU hardware)
- [ ] Multi-stream pairwise-mask generation for large cohorts using `pairwise_mask_ptx` (requires GPU hardware)

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
- Tests: 502 passing (10 E2E in `lib.rs` + module unit tests)
- `unwrap()` calls: 0 in library code
- macOS: compiles, returns `UnsupportedPlatform` from any actual GPU launch
- PTX targets covered: sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120

## Performance Targets

| Operation | Size | Target |
|-----------|------|--------|
| PTX kernel string generation | per call | < 100 us |
| `FedAvgState::aggregate` (CPU reference) | K = 100 clients, P = 1M params | < 50 ms |
| `topk_sparsify` (heap-based) | P = 1M, k = 10K | < 30 ms |
| `stochastic_quantize` (QSGD, 8-bit) | P = 1M | < 20 ms |
| `share_scalar` (Shamir, t = 3, n = 5) | per scalar | < 50 us |
| `share_gradient` (Shamir, t = 3, n = 5) | P = 100K | < 500 ms |
| `compose_rdp` (linear composition) | T = 1000 rounds | < 1 ms |
| `MomentsAccountant::compose` | T = 1000 rounds | < 5 ms |

Targets are CPU-reference budgets. GPU launches via `oxicuda-launch` (once
wired) should bring `aggregate_mean_ptx` and `qsgd_quantize_ptx` close to
memory-bandwidth bound for large parameter counts.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/fed_ops.rs`) -- PTX generation + CPU aggregation / compression / Shamir pipelines

## Notes

- All PTX kernels emit `.target sm_<version>` and use a grid-stride loop pattern.
- Shamir secret sharing uses a Mersenne prime `PRIME` (re-exported from the prelude) for closed-form Lagrange interpolation; floats are encoded into u64 before sharing.
- `random_select` and `stratified_select` consume entropy from the handle RNG -- pass `handle.rng_mut()` rather than constructing fresh LCGs per call to keep selections reproducible.
- DP mechanisms validate `epsilon > 0`, `0 < delta < 1`, and finite sensitivity at construction; downstream code can rely on the absence of NaN noise contributions.
- The PTX kernels target scalar f32 paths; Tensor-Core (wgmma / mma.sync) usage is intentionally absent -- federated workloads are dominated by elementwise reductions, not GEMM.

---

## Architecture-Specific Deepening

### PTX Generation by SM Version

| SM Version | PTX Version | Notes |
|------------|-------------|-------|
| sm_75 (Turing) | 7.5 | Baseline; `selp.f32`, `fma.rn.f32` supported |
| sm_80 / sm_86 (Ampere) | 8.0 | Default target for `FedHandle::default_handle()` |
| sm_89 (Ada) | 8.0 | Treated as sm_80 by `ptx_version_str()` |
| sm_90 / sm_90a (Hopper) | 8.4 | No `wgmma` usage -- federated kernels are scalar reductions |
| sm_100 / sm_120 (Blackwell) | 8.7 | Same scalar-FMA / LCG-Box-Muller pattern |

The 7 generators all dispatch on the SM string and emit identical scalar PTX
modulo the `.target` directive. Tensor-Core specialisation is intentionally
left out: federated workloads are gradient reductions and elementwise noise
addition, not GEMM.

### Deepening Opportunities

- [ ] Hopper warp-specialised `aggregate_mean_ptx` using `redux.sync.add.f32` for atomic-free per-warp reduction (requires GPU hardware)
- [ ] Blackwell (sm_100+) cluster-launch path for `share_gradient` so a single grid handles many client gradients in parallel (requires GPU hardware)
- [ ] FP16 / BF16 variants of `aggregate_mean_ptx` and `gaussian_noise_ptx` for low-precision federated training (requires GPU hardware)

---

## Functional Quality Gates (Vol.24)

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F1 | FedAvg sample-weighted aggregation | P0 | [x] |
| F2 | FedProx proximal regularisation | P0 | [x] |
| F3 | SCAFFOLD control-variate correction | P0 | [x] |
| F4 | FedAdam / Yogi / AMSGrad server optimisers | P1 | [x] |
| F5 | PowerSGD low-rank compression with error feedback | P1 | [x] |
| F6 | QSGD quantisation (unbiased) | P0 | [x] |
| F7 | Random-K + Top-K sparsification | P0 | [x] |
| F8 | Gaussian / Laplace DP mechanisms | P0 | [x] |
| F9 | Moments + Renyi-DP composition | P0 | [x] |
| F10 | PATE noisy voting + data-dependent epsilon | P1 | [x] |
| F11 | Shamir secret sharing (k, n) over Mersenne field | P0 | [x] |
| F12 | Pairwise masking + secure aggregator | P0 | [x] |
| F13 | Random + stratified client selection | P0 | [x] |
| F14 | PTX generators for 7 kernels x 6 SM versions | P0 | [x] |

## Performance Verification Harness Status

- All performance numbers above are CPU-side targets achievable on the build host.
- GPU end-to-end harnesses await the [ ] GPU launcher wiring item plus a
  Linux+NVIDIA test runner; the PTX strings themselves are covered by
  string-content unit tests inside `ptx_kernels.rs`.
