# oxicuda-autotune TODO

Automatic GPU kernel parameter optimization engine. Measurement-based autotuning with search space pruning, GPU event-timed benchmarking, persistent result storage, and 3-tier runtime dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 16,198** across **33 files** (estimated 32K-50K for oxicuda-autotune portion of Vol.2)

The autotune crate provides the optimization loop that makes OxiCUDA kernels competitive: define a search space, benchmark candidates, persist the best, and dispatch at runtime with fallback tiers.

### Completed [x]

- [x] `config.rs` -- Config struct (tile_m, tile_n, tile_k, pipeline_depth, threads, smem usage, estimated GFLOPS)
- [x] `search_space.rs` -- SearchSpace definition with per-dimension candidate values, SearchSpaceBuilder, prune() for hardware constraints, gemm_default() preset
- [x] `benchmark.rs` -- BenchmarkEngine with warmup + event-timed measurement runs, BenchmarkConfig (warmup/iterations), BenchmarkResult (median, min, max, GFLOPS)
- [x] `result_db.rs` -- ResultDb JSON-backed persistent storage keyed by (GPU, kernel, problem), ProblemKey type
- [x] `dispatch.rs` -- Dispatcher with 3-tier fallback (exact match, nearest-neighbor interpolation, default config), DispatchTier enum
- [x] `tunable.rs` -- TunableKernel trait for kernels that participate in autotuning
- [x] `error.rs` -- AutotuneError enum, AutotuneResult type alias
- [x] `lib.rs` -- Prelude module, re-exports of all public types
- [x] `early_stopping.rs` -- EarlyStoppingConfig, EarlyStoppingTracker, StopReason, EarlyStoppingSummary (patience-based, time-budget, convergence detection; ~17 tests)

### Future Enhancements [ ]

**Search Strategies (P0)**
- [x] Bayesian optimization (bayesian.rs) -- GP surrogate model + acquisition functions (EI, UCB, PI)
- [x] Simulated annealing (simulated_annealing.rs) -- temperature-based exploration for large search spaces
- [x] Genetic algorithm search (genetic.rs) -- crossover/mutation on config populations
- [x] Early stopping -- abort unpromising configurations after partial measurement (early_stopping.rs; EarlyStoppingConfig with patience, time-budget, convergence threshold; EarlyStoppingTracker with StopReason and EarlyStoppingSummary)

**Multi-objective (P1)**
- [x] Multi-objective optimization -- Pareto front for throughput vs latency vs power
- [x] Power-aware tuning -- integrate nvidia-smi power readings into objective (power_aware.rs)
- [x] Memory-constrained tuning -- maximize throughput within VRAM budget

**Scalability (P1)**
- [x] Distributed autotune across GPU cluster -- coordinate tuning across multiple nodes (distributed.rs)
- [x] Parallel benchmark execution -- run multiple configs on different streams/GPUs (parallel_bench.rs)
- [x] Incremental re-tuning -- update results when hardware/driver changes detected (incremental.rs)

**Intelligence (P2)**
- [x] Analytical cost model -- roofline-model analytical predictor: arithmetic-intensity computation, memory-bandwidth-bound vs compute-bound classification, theoretical peak GFLOPS. ALREADY EXISTS as `cost_model/roofline.rs` (`Roofline`/`RooflineClassification`/`Bound`/`BandwidthCeiling`; ridge point, attainable(), estimated_runtime()), complemented by `cost_model/latency_predictor.rs` (learned ridge-regression surrogate). Williams (2009) model.
- [x] Halide-style schedule search -- tile-size + loop-order schedule space inspired by Halide autoscheduler; split/vectorise/unroll axes with cost-model-guided pruning. ALREADY EXISTS as `search/halide_schedule.rs` (`ScheduleSearcher`, `LoopNest`/`Loop` IR, `Transform` tile/reorder/vectorize/parallelize/unroll, `FeatureCostModel`, greedy/beam search). Adams et al. (2019).
- [x] Persistent tuning cache with versioned schema migration (`cache/persistent_cache.rs`) -- on-disk LRU cache keyed by (kernel-hash, GPU-arch, CUDA-driver-version) distinct from the existing `export.rs` export-only path; `PersistentTuneCache` (FNV-1a kernel hash, recency-ordered LRU eviction, `CacheSchemaVersion` envelope + forward migration from legacy bare-array layout, `CacheStats` hit/miss/eviction, save_at/load_at). NEW 2026-06-21.
- [x] Transfer learning between architectures -- warm-start sm_90 tuning from sm_80 results (transfer_learning.rs)
- [x] Problem size interpolation (interpolation.rs) -- nearest-neighbor and inverse-distance-weighted interpolation for unseen matrix sizes
- [x] Kernel similarity detection -- reuse results for structurally similar kernels

**Runtime (P1)**
- [x] Real-time adaptive tuning -- dynamically switch configs based on runtime metrics (adaptive.rs)
- [x] Autotune result sharing/export -- portable format for sharing across machines (export.rs)
- [x] Result database migration -- versioned schema for forward compatibility (db_migration.rs)
- [x] CLI tool for offline autotuning -- batch tuning with progress reporting (cli.rs)

**Integration (P1)**
- [x] PTX template integration (ptx_integration.rs) -- direct SearchSpace generation from GemmTemplate parameters
- [x] Autotune-guided PTX generation -- feed results back into template instantiation (guided_ptx.rs)
- [x] Benchmark visualization -- export timing data for plotting and analysis (visualization.rs)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | GPU device info, stream/event for timing | Yes |
| oxicuda-ptx | (optional, `ptx` feature) Template parameter extraction | Yes |
| thiserror | Derive macro for AutotuneError | Yes |
| tracing | Structured logging for benchmark progress | Yes |
| serde | Serialization for Config, BenchmarkResult, ResultDb | Yes |
| serde_json | JSON persistence for result database | Yes |

## Quality Status

- Warnings: 0
- Tests: 467 unit + 27 doctests passing (was 448 unit + 26 doctests; +18 unit / +1 doctest from `cache/persistent_cache.rs`)
- unwrap() calls: 0
- ResultDb uses JSON for human-readable, debuggable persistence

## Performance Targets

Autotuning overhead is amortized -- invest time once, benefit at every subsequent launch:
- Search space pruning: < 1ms for 1000 candidate configs
- Single benchmark iteration: depends on kernel (typically 0.1-100ms)
- Full GEMM autotune (40 configs, 10 sizes): approximately 80 seconds
- Dispatch lookup (exact match): < 1us from in-memory ResultDb
- Dispatch lookup (nearest-neighbor): < 10us with interpolation

## Notes

- The 3-tier dispatch ensures a config is always available: exact > nearest > default
- ResultDb is keyed by GPU name string, so results are per-GPU-model
- SearchSpace::gemm_default() provides sensible tile/thread ranges for matrix multiply
- Benchmark uses CUDA events for GPU-side timing (not wall-clock), ensuring accuracy
- The `ptx` feature enables direct integration with oxicuda-ptx template parameters
- serde/serde_json are the only non-driver dependencies -- keeps the crate lightweight

---

## Blueprint Quality Gates (Vol.2 Sec 4.2)

### Autotuner Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| A1 | Search space constraint pruning correctness — verified vs manual calculation | P0 | [x] Done |
| A2 | Benchmark result statistical stability — variance < 5% for same config repeated | P0 | [x] Done |
| A3 | `ResultDb` persistence round-trip — save results, reload, verify identical | P0 | [x] Done |
| A4 | `Dispatcher` 3-tier fallback tested independently (exact match, nearest match, default config) | P0 | [x] Done |
| A5 | CLI end-to-end GEMM autotune on real GPU | P1 | [ ] Verify |
| A6 | Best autotuned config achieves **≥ 80% of cuBLAS GEMM throughput** on A100 | P0 | [ ] Verify |

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Statistical Rigor
- [x] Benchmark variance < 5% verified across 20 runs per config (A2)
- [ ] Warmup run count (5) sufficient for GPU clock stabilization — verify on variable-frequency GPU (requires GPU hardware — on-device clock-stabilization measurement)
- [x] GFLOPS calculation verified: `(2 * M * N * K) / median_us / 1e6` formula validated

### Search Strategy Deepening
- [x] Bayesian optimization convergence: find near-optimal config in < 50% of exhaustive search time
- [x] Simulated annealing cooling schedule tuned for typical GEMM search spaces
- [x] Genetic algorithm crossover operator verified doesn't generate invalid configs
- [x] Early stopping: verify correct early termination when variance criterion met

### Database / Dispatch Deepening
- [x] `lookup_nearest()` distance metric verified: correct interpolation for untested problem sizes
- [x] `ResultDb` concurrent write safety (multiple autotuner processes on same machine)
- [x] Export format (JSON) schema validated against cuBLAS autotuning result format

### Performance
- [ ] A6 measurement methodology: GEMM F16 4096x4096x4096 on A100, compare OxiCUDA best config vs cuBLAS default
- [ ] A6 target: achieve ≥ 80% of cuBLAS for this specific problem before considering other sizes
- [x] Search space constraint pruning verified against manual calculation (A1) — 14 new tests
- [x] ResultDb persistence round-trip test (save, reload, verify identical) (A3) — 3 round-trip tests
- [x] Dispatcher 3-tier fallback tested independently (exact, nearest, default) (A4) — 4 tests
- [x] Genetic algorithm crossover never violates search space constraints — 3 crossover tests

- [x] autotune-adaptive-warmup (planned 2026-05-01; DONE -- verified 2026-06-21)
  - **Status:** Fully implemented. `WarmupStrategy { Fixed(usize), Adaptive { min_iterations, max_iterations, tolerance } }` enum in `benchmark.rs` (default `Fixed(5)`), `warmup_converged()` relative-delta helper, adaptive loop in both timed and untimed benchmark paths; wired through `cli.rs` (`parse_warmup_strategy` parsing `fixed:N` / `adaptive:TOL[,min=N,max=N]`), `power_aware.rs`, and `parallel_bench.rs` ctors. All planned tests present (`cli_warmup_parser_roundtrip`, `parse_tune_warmup_*`, etc.).
  - **Goal:** Replace the fixed 5-iteration warmup in `BenchmarkConfig` with an adaptive convergence detector that warms up until the relative delta between two consecutive timing samples is below a configurable tolerance, or a max-iteration cap is hit.
  - **Design:** New `WarmupStrategy { Fixed(usize), Adaptive { min_iterations, max_iterations, tolerance } }` enum. Replace `warmup_runs: usize` field with `warmup: WarmupStrategy` (default `Fixed(5)` for parity). Adaptive loop: iterate, compare consecutive `Duration` samples by relative delta `|t - t_prev| / t_prev`; break on convergence or hit `max_iterations` with `tracing::warn!`. Wire CLI `--warmup` flag at `cli.rs:464` (currently dead `_warmup` param) with a `clap` value_parser parsing `fixed:N` or `adaptive:TOL[,min=N,max=N]`.
  - **Files:** `src/benchmark.rs` (enum + field + loop at lines ~157, ~229), `src/cli.rs:~464`, `src/power_aware.rs:~385`, `src/parallel_bench.rs` (test ctor updates)
  - **Tests:** `warmup_strategy_fixed_runs_exact_count`, `warmup_strategy_adaptive_converges_on_stable_input`, `warmup_strategy_adaptive_caps_at_max`, `warmup_strategy_adaptive_respects_min`, `cli_warmup_parser_roundtrip`
  - **Risk:** Adaptive tolerance too tight on noisy hardware → runaway warmup. Mitigated by `max_iterations` cap (default 20) + `tracing::warn!` on cap-hit. Default stays `Fixed(5)` so existing behavior is preserved unless callers opt in.
