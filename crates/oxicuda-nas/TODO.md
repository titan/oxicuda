# oxicuda-nas TODO

Pure-Rust Neural Architecture Search primitives for OxiCUDA: differentiable
architecture search (DARTS), evolutionary multi-objective search (NSGA-II),
one-shot supernets with weight-sharing, slimmable networks, and FLOP / latency
/ accuracy predictors. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.25).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 12,155 (43 files)
- **PTX kernels:** 7 kernel generators emitted for 6 SM targets (sm_75 / 80 / 86 / 90 / 100 / 120)
- **Coverage:** CPU reference implementation + PTX string generation for GPU execution

### Completed

#### Core Infrastructure
- [x] `error.rs` (112 LoC) -- `NasError` (14 variants: EmptyPopulation, InvalidArchEncoding, OpKindOutOfRange, InvalidArchitectureWeights, InvalidGumbelTemperature, InvalidWidthMultiplier, MixedOpDimensionMismatch, MissingPrimitive, NumObjectivesMismatch, InvalidPopulationSize, NanEncountered, Internal, ...) + `NasResult<T>`
- [x] `handle.rs` (290 LoC) -- `SmVersion`, `LcgRng` (Knuth MMIX core + Box-Muller normals + Fisher-Yates shuffle), `NasHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` (137 LoC) -- Module exports + `prelude` re-exports + 5 E2E integration tests

#### PTX Kernels (ptx_kernels.rs, 726 LoC)
- [x] `arch_grad_ptx` -- Architecture parameter gradient accumulation with `fma.rn.f32`
- [x] `arch_softmax_ptx` -- Stable softmax over K operation-mixing weights using `lg2.approx.f32` / `ex2.approx.f32`
- [x] `crossover_uniform_ptx` -- Uniform crossover for evolutionary genome operators using `selp.u32` to pick between parents
- [x] `flops_accumulate_ptx` -- FLOP-cost accumulation across operations via `atom.global.add.f32`
- [x] `gumbel_softmax_ptx` -- Gumbel-softmax differentiable categorical sampling with `-log(-log(u))` Gumbel-noise injection
- [x] `mixed_op_blend_ptx` -- Convex combination `out = sum_k softmax(alpha)_k * op_k(x)` with `fma.rn.f32` accumulation
- [x] `pareto_dominate_ptx` -- Pairwise Pareto dominance check for multi-objective sorting

#### Operations (ops/)
- [x] `primitives.rs` (472 LoC) -- `OpKind` (8 standard DARTS primitives: Skip, SepConv3x3, SepConv5x5, DilConv3x3, DilConv5x5, MaxPool3x3, AvgPool3x3, None), `OpWeights` per-edge architecture vector
- [x] `mixed_op.rs` (177 LoC) -- `MixedOp` differentiable mixture-of-ops with forward / backward and Gumbel-Softmax temperature gating
- [x] `search_space.rs` (165 LoC) -- `SearchSpace`, `CellSpace`, `NetworkSpace` DARTS-style cell + network spaces
- [x] `transformer_nas.rs` -- `BlockSpec` / `TransformerArch` / `TransformerSearchSpace` elastic transformer (AutoFormer / V-MoE) axes with exact attention + FFN + MoE MAC / parameter accounting

#### DARTS (darts/)
- [x] `cell.rs` (176 LoC) -- `DartsCell` multi-step cell with K candidate ops on each edge
- [x] `network.rs` (137 LoC) -- `DartsNetwork` stacked cells (normal + reduction) with auxiliary head
- [x] `bilevel.rs` (164 LoC) -- `BilevelOptimizer` / `BilevelConfig` bi-level w/alpha optimisation: weights on inner train loss, architecture on outer val loss
- [x] `derive.rs` (142 LoC) -- `DiscretizedCell`, `DiscretizedNetwork`, `derive_discrete_cell`, `derive_network` top-2 op selection and architecture derivation

#### Evolutionary (evolution/)
- [x] `encoding.rs` (142 LoC) -- `ArchEncoding` discrete genome representation
- [x] `population.rs` (99 LoC) -- `Population` container with crossover and mutation operators
- [x] `nsga2.rs` (363 LoC) -- `Individual`, `fast_non_dominated_sort`, `crowding_distance`, `nsga2_select`, `tournament_select` NSGA-II multi-objective EA (Deb 2002)

#### Supernet (supernet/)
- [x] `weight_share.rs` (123 LoC) -- `Supernet` weight-shared one-shot supernet (Bender 2018)
- [x] `path_sample.rs` (110 LoC) -- `PathSampler` / `SamplingStrategy` uniform / fairness-aware path sampling (SPOS / FairNAS)
- [x] `slimmable.rs` (234 LoC) -- `SlimmableNet`, `BnStats`, `WIDTH_MULTIPLIERS` slimmable network with per-width batch-norm statistics (Yu 2019)

#### Predictor (predictor/)
- [x] `predictor_io.rs` (154 LoC) -- `LayerSpec`, `ArchFeatures::from_layers()` shared `[op-one-hot || in_ch || out_ch || h || w]` feature extractor
- [x] `flops.rs` (222 LoC) -- `OpCost`, `op_cost`, `total_cost` analytic FLOP + parameter accountant (sep / dilated conv `2*K^2*C_in*HW + 2*C_in*C_out*HW`, pooling `9*C_out*HW`)
- [x] `latency.rs` (350 LoC) -- `LatencyLut` hardware-calibrated `(op, c_in, c_out, h, w)` lookup with default fallback; `LatencyMlp` two-layer ReLU MLP latency surrogate trained via per-sample MSE gradient descent
- [x] `accuracy.rs` (356 LoC) -- `KnnAccuracyPredictor` inverse-distance-weighted k-NN regression; `RbfAccuracyPredictor` Gaussian-kernel ridge regressor with closed-form Gauss-Jordan solve
- [x] `bayesian_gp.rs` (`GaussianProcess`) -- exact Gaussian-Process accuracy regressor: RBF / Matérn-5/2 kernel, closed-form posterior mean + variance via Cholesky solve of `K + σ_n² I`; `Acquisition::{Ucb, ExpectedImprovement}` for sample-efficient proposal
- [x] `gnn_predictor.rs` (`GnnPredictor`, `PathEncodedPredictor`) -- NPENAS message-passing GNN over the cell DAG + BANANAS path-encoded MLP accuracy predictors

#### Integration Tests (lib.rs)
- [x] 5 E2E tests: FLOP accountant finite cost on `sample_arch`, LUT calibrated predict, MLP train + predict, k-NN constant-target round-trip, RBF constant-target

### Future Enhancements

#### P0 -- Critical (NAS Algorithm Coverage Gaps)
- [x] PC-DARTS partial-channel sampling -- mitigates DARTS memory blowup; sample a fraction of channels per edge each step (Xu 2020) (darts/pc_darts.rs -- Xu 2020; partial-channel mask 1/K through ops + edge normalization softmax β per destination node)
- [x] DARTS+ early-stopping criterion -- detect skip-connection collapse and freeze architecture parameters (Liang 2019) (darts/darts_plus.rs -- Liang 2019; detect skip-connection collapse via argmax-over-α skip-count; freeze architecture parameters after `patience` consecutive epochs above threshold; reset on sub-threshold epoch)
- [x] ProxylessNAS gradient estimator -- binary gates with hard-gated activations for memory-efficient direct hardware-aware search (Cai 2019)
- [x] ENAS RL controller -- LSTM policy with REINFORCE for the supernet sampling distribution (Pham 2018) (controller/enas.rs -- LSTM controller autoregressive sampling + REINFORCE EMA-baseline BPTT update)

#### P1 -- Important (Search-Space & Search-Strategy Depth)
- [x] MobileNet-V2 / V3 search space -- inverted-residual blocks with SE modules as an alternative to the DARTS primitives
- [x] `darts/darts_ops.rs` / `DartsMixedOp` — extended DARTS operation set with `DartsMixedOp` supporting runtime op-weight blending, Gumbel-softmax temperature annealing, and gradient-free hard-gate derivation; `darts_ops` registry for custom primitive registration
- [x] `predictor/latency_predictor.rs` / `LatencyPredictor` — unified `LatencyPredictor` trait abstracting `LatencyLut` and `LatencyMlp`; auto-calibration from hardware measurements; per-device profile serialisation; `LatencyPredictor::predict_arch()` over `ArchFeatures`
- [x] Transformer NAS primitives (AutoFormer / V-MoE) -- multi-head attention / FFN-width / num-layers as searchable axes (ops/transformer_nas.rs -- `BlockSpec` / `TransformerArch` / `TransformerSearchSpace`: elastic embed-dim, head-count, FFN-width (mlp_ratio), depth and MoE expert layout as searchable axes; exact attention + FFN + MoE MAC / parameter accounting; deterministic sampling; max/min subnet; now declared in ops/mod.rs + prelude, 14 tests)
- [x] Once-for-All supernet (Cai 2020) -- elastic depth + width + kernel size in one supernet (supernet/once_for_all.rs -- Cai 2020 ICLR; `OfaSpace` (per-unit stage layout + kernel/width/depth choice sets), `OfaSubnet` weight-sliced subnet with MBConv MAC/param accounting via `ops::mbconv_ops`, `ShrinkSchedule` / `ShrinkPhase` progressive-shrinking (FullNetwork -> ElasticKernel -> ElasticDepth -> ElasticWidth) with per-axis `admissible()` axis-collapse, deterministic `sample()` + max/min subnet; was an ORPHANED module (file present, never declared) -- now declared in supernet/mod.rs + prelude; latent bug fixed: added `PartialEq, Eq` derives to `ShrinkSchedule` + `OfaSpace` so the orphan's `assert_eq!(Result<..>, Err(..))` tests compile; 10 tests now run + pass)
- [x] BigNAS uniform sampling + sandwich rule -- improves supernet ranking correlation (supernet/bignas.rs -- Yu 2020 ECCV; uniform sub-net sampling + sandwich rule (max + min + sandwich_samples random subnets) per training step for supernet ranking correlation; flops_proxy ordering)
- [x] Regularized Evolution (Real 2019) -- aging-based EA as an alternative to NSGA-II for single-objective search (evolution/regularized_evolution.rs -- aging-based EA: tournament select + mutate best + add child + remove oldest; reuses ArchEncoding)
- [x] Multi-trial NAS-Bench-style reproducibility hooks -- deterministic search seeds and per-arch result caches (evolution/nas_bench.rs -- NAS-Bench-101/201; `derive_arch_seed` / `arch_rng` SplitMix64-finalised pure `(arch, trial, base_seed) -> seed`, `arch_key` canonical little-endian genome bytes, `NasBenchCache` per-`(arch, trial)` `TrialResult` cache with lazy evaluate-once `query()` + `unique_queries` / `cache_hits` accounting + `mean_val_accuracy`; was an ORPHANED module (file present, never declared) -- now declared in evolution/mod.rs + prelude; fixed `clippy::manual_inspect` in a test oracle closure (`.map(|r| { let _ = a; r })` -> unused `_` param); 11 tests now run + pass)

#### P2 -- Nice-to-Have (Predictor & Evaluation Extensions)
- [x] HAT hardware-aware transformer NAS (`search/hat.rs`) — Wang 2020 ACL: multi-objective search in a weight-shared transformer supernet using Pareto-front evolution with latency LUT for each target device; `HatSearcher` (search/hat.rs -- `HatSearcher` Pareto-front evolution over `TransformerSearchSpace`; per-device `BlockLatencyLut` (recorded measurements, never fabricated); crossover + per-axis mutation + environmental selection; caller-supplied accuracy `loss_proxy` closure; now declared in search/mod.rs + prelude, 9 tests)
- [x] Local search NAS (`search/local_search.rs`) — White 2021 ICLR: hill-climbing on architecture space with single-op perturbations + zero-cost proxy ranking to avoid supernet training; `LocalSearchNas`
- [x] Graph Neural Network architecture predictor -- replace MLP / RBF / k-NN predictors with a GNN over the DAG (`NPENAS`, `BANANAS`) (predictor/gnn_predictor.rs -- `GnnPredictor` message-passing GNN over the cell DAG (op-gated neighbour aggregation + per-layer linear update/ReLU + mean-pool readout, end-to-end reverse-mode SGD) and `PathEncodedPredictor` BANANAS truncated-path-encoding MLP; `CellTopology` + `PathEncoder`; now declared in predictor/mod.rs + prelude, 12 tests)
- [x] Bayesian-optimisation accuracy predictor -- Gaussian Process with uncertainty for sample-efficient search (predictor/bayesian_gp.rs -- `GaussianProcess` exact GP regressor: RBF / Matérn-5/2 kernel, closed-form posterior mean + variance via Cholesky `L Lᵀ = K + σ_n² I` + forward/back triangular solves (no explicit inverse), target centring; `Acquisition::{Ucb, ExpectedImprovement}` with erf-based normal CDF; `propose()` acquisition argmax; 18 tests proving interpolation, ~0 variance at observed points / growth in gaps, EI/UCB peaking in the uncertain promising gap)
- [x] Multi-fidelity NAS -- early-stopping based on partial training (`Hyperband`, `BOHB`-style)
- [x] Zero-cost proxies (NASWOT, SNIP, GraSP) -- predictor-free architecture ranking via untrained-network signals (proxy/zero_cost.rs -- NASWOT logdet kernel + SNIP/GraSP/SynFlow saliencies)
- [x] Hardware-aware predictor calibration -- per-device LUT serialisation / deserialisation helpers

#### GPU Launcher Wiring
- [ ] Wire `ptx_kernels::*` strings through `oxicuda-launch::Kernel::from_module` for end-to-end GPU execution (currently only PTX strings are emitted)
- [ ] GPU-resident `MixedOp::forward` using `mixed_op_blend_ptx` instead of the CPU loop
- [ ] GPU-resident `fast_non_dominated_sort` using `pareto_dominate_ptx` for large populations (N > 1024)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device / Host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0 (clippy clean, `--all-features --all-targets -- -D warnings`)
- Tests: 389 passing (`cargo nextest run -p oxicuda-nas --all-features`) -- 5 E2E in `lib.rs` + module unit tests, incl. 18 (bayesian_gp) + 14 (transformer_nas) + 12 (gnn_predictor) + 11 (nas_bench) + 10 (once_for_all) + 9 (hat)
- `unwrap()` calls: 0 in library code
- macOS: compiles, returns `UnsupportedPlatform` from any actual GPU launch
- PTX targets covered: sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120

## Performance Targets

| Operation | Size | Target |
|-----------|------|--------|
| PTX kernel string generation | per call | < 100 us |
| `total_cost` (FLOP accountant) | 50-layer arch | < 100 us |
| `LatencyLut::predict` (linear scan) | 50-layer arch, LUT = 10K | < 1 ms |
| `LatencyMlp::fit` (200 epochs SGD) | N = 1K samples, hidden = 32 | < 500 ms CPU |
| `KnnAccuracyPredictor::predict` (k = 5) | N = 1K archs, dim = 100 | < 5 ms |
| `RbfAccuracyPredictor::fit` (Gauss-Jordan) | N = 256 archs, dim = 100 | < 100 ms |
| `fast_non_dominated_sort` (NSGA-II) | N = 100, M = 3 objectives | < 5 ms |
| `nsga2_select` (NSGA-II survivor selection) | N = 200 -> 100 | < 10 ms |

Targets are CPU-reference budgets. Once GPU wiring lands, `mixed_op_blend_ptx`
and `pareto_dominate_ptx` should approach memory-bandwidth bound on the
candidate-list size.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/nas_ops.rs`) -- PTX generation + `population_random` + `nsga2_select` + `path_sample` + `mixed_op_blend`

## Notes

- All PTX kernels emit `.target sm_<version>` and use a grid-stride loop pattern.
- The Gauss-Jordan solve inside `RbfAccuracyPredictor::fit` runs in O(N^3); for N > 500 prefer chunked / blockwise solves rather than the default path.
- `SlimmableNet` keeps a per-width `BnStats` (running mean / variance) keyed by `WIDTH_MULTIPLIERS`; switching width without calling `set_width` first leaves the BN stats stale.
- `PathSampler` exposes a `SamplingStrategy::FairNAS` mode that guarantees each candidate op gets sampled exactly once per round across `K` workers -- prefer this for supernet training to avoid op-bias.
- The PTX kernels target scalar f32 paths; no Tensor-Core (wgmma / mma.sync) usage -- supernet GEMMs delegate to `oxicuda-blas`.

---

## Architecture-Specific Deepening

### PTX Generation by SM Version

| SM Version | PTX Version | Notes |
|------------|-------------|-------|
| sm_75 (Turing) | 7.5 | Baseline; `atom.global.add.f32`, `ex2.approx.f32` supported |
| sm_80 / sm_86 (Ampere) | 8.0 | Default target for `NasHandle::default_handle()` |
| sm_89 (Ada) | 8.0 | Treated as sm_80 by `ptx_version_str()` |
| sm_90 / sm_90a (Hopper) | 8.4 | No `wgmma` usage -- kernels are scalar |
| sm_100 / sm_120 (Blackwell) | 8.7 | Same scalar-FMA / atomic pattern |

The 7 generators all dispatch on the SM string and emit identical scalar PTX
modulo the `.target` directive. NAS kernels are dominated by softmax,
small-K elementwise blends, and pairwise dominance checks -- none benefits
from Tensor Core specialisation.

### Deepening Opportunities

- [ ] Hopper `arch_softmax_ptx` rewrite using `redux.sync.add.f32` to remove the inner-loop `atom.global.add`
- [ ] Blackwell (sm_100+) cluster launch for `pareto_dominate_ptx` to make NSGA-II non-dominated sorting scale past N = 1024
- [ ] FP16 / BF16 variant of `mixed_op_blend_ptx` for memory-bound supernet forward passes

---

## Functional Quality Gates (Vol.25)

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| N1 | 8 DARTS primitives (`OpKind`) with `op_cost` | P0 | [x] |
| N2 | `MixedOp` differentiable mixture | P0 | [x] |
| N3 | DARTS cell + network + bi-level optimiser | P0 | [x] |
| N4 | Discrete architecture derivation (top-2 ops per edge) | P0 | [x] |
| N5 | `ArchEncoding` genome + `Population` operators | P0 | [x] |
| N6 | NSGA-II non-dominated sort + crowding distance | P0 | [x] |
| N7 | Tournament selection | P1 | [x] |
| N8 | Weight-shared supernet | P0 | [x] |
| N9 | Path sampler (uniform + FairNAS) | P0 | [x] |
| N10 | Slimmable network with per-width BN | P1 | [x] |
| N11 | FLOP / parameter analytic accountant | P0 | [x] |
| N12 | Latency LUT + MLP predictor | P0 | [x] |
| N13 | k-NN + RBF accuracy predictor | P1 | [x] |
| N14 | PTX generators for 7 kernels x 6 SM versions | P0 | [x] |

## Performance Verification Harness Status

- All performance numbers above are CPU-side targets achievable on the build host.
- GPU end-to-end harnesses await the [ ] GPU launcher wiring item plus a
  Linux+NVIDIA test runner; the PTX strings themselves are covered by
  string-content unit tests inside `ptx_kernels.rs`.
