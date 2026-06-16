# oxicuda-timeseries TODO

Pure-Rust time-series forecasting & classification architectures for OxiCUDA: TCN,
NHiTS, PatchTST, TimesNet, iTransformer, RevIN, series decomposition. Time-major
`[T, C]` layout throughout; all variates channels-last. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.22).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 20,036 (73 files, Rust 4,791 code + 870 comments + 829 blanks)
- **Tests:** 615 passing (#[test] count in src/)
- **Crate:** `oxicuda-timeseries` -- Vol.22 Time-Series Forecasting Architectures

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `TsError` (18 variants): `DimensionMismatch`, `ShapeMismatch`,
      `EmptyInput`, `InvalidSequenceLength`, `InvalidNumVariates`, `InvalidPatchLen`,
      `InvalidStride`, `InvalidKernelSize`, `InvalidDilation`, `InvalidNumHeads`,
      `HeadDimMismatch`, `InvalidEmbedDim`, `InvalidHorizon`, `InvalidPoolSize`,
      `InvalidTopK`, `WeightShapeMismatch`, `NonFinite`, `Internal`;
      `TsResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle),
      `TsHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- crate root with `prelude` module and 20 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 7 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `moving_average_ptx` -- strided centred moving average over time axis
- [x] `patch_embed_1d_ptx` -- extract overlapping 1-D patches
      `[N, T] -> [N, num_patches, patch_len]`
- [x] `causal_temporal_conv_ptx` -- dilated causal 1-D conv for TCN residual blocks
- [x] `auto_correlation_ptx` -- FFT magnitude-squared step for Autoformer / TimesNet
- [x] `revin_normalize_ptx` -- RevIN normalise with per-`(n, c)` stats + learnable
      affine
- [x] `multirate_pool_ptx` -- average pool at variable stride for NHiTS multi-rate
      sampling
- [x] `period_detect_ptx` -- top-k FFT magnitude reduction for TimesNet period
      detection

#### Normalisation (`norm/`, 2 files + mod)
- [x] `norm/revin.rs` -- `RevIn`: reversible instance norm with forward + inverse,
      Bessel-corrected stats, learnable affine
- [x] `norm/instance_norm.rs` -- `InstanceNorm1d`: per-variate instance norm with
      optional affine

#### Decomposition (`decomp/`, 2 files + mod)
- [x] `decomp/moving_avg.rs` -- `MovingAvg`: centred moving-average filter with
      replicate-pad boundary
- [x] `decomp/series_decomp.rs` -- `SeriesDecomp`: trend + seasonal split matching
      Autoformer formulation

#### Patch Embedding (`patch/`, 1 file + mod)
- [x] `patch/patch_embed.rs` -- `PatchEmbed1d`: overlapping 1-D patches, Xavier
      init, univariate + multivariate `forward_mv`

#### TCN (`tcn/`, 2 files + mod)
- [x] `tcn/temporal_block.rs` -- `TcnBlock`: weight-normalised dilated causal conv,
      Kaiming He init, optional 1x1 residual projection
- [x] `tcn/tcn_encoder.rs` -- `TcnEncoder`: exponential dilation schedule `2^i`,
      `tiny` and `default` configs

#### NHiTS (`nhits/`, 3 files + mod)
- [x] `nhits/multi_rate_sampler.rs` -- `MultiRateSampler`: avg pool +
      nearest-neighbour upsample
- [x] `nhits/nhits_block.rs` -- `NHitsBlock`: pool -> MLP -> backcast + forecast
      heads
- [x] `nhits/nhits.rs` -- `NHits`: hierarchical residual stacks with default
      `pool_sizes = [1, 2, 4]`

#### PatchTST (`patchtst/`, 1 file + mod)
- [x] `patchtst/patch_transformer.rs` -- `PatchTst`: channel-independent patches ->
      sinusoidal PE -> N x pre-LN TransformerLayer -> per-variate linear head;
      `PatchTstConfig::tiny` / `base`

#### TimesNet (`timesnet/`, 2 files + mod)
- [x] `timesnet/times_block.rs` -- `TimesBlock`: O(T^2) DFT period detection ->
      top-k 2-D reshape -> depthwise 3x3 conv -> weighted sum -> residual + LN
- [x] `timesnet/timesnet.rs` -- `TimesNet`: input proj -> N blocks -> flatten ->
      linear head

#### iTransformer (`itransformer/`, 2 files + mod)
- [x] `itransformer/inverted_block.rs` -- `InvertedBlock`: attention over C variate
      tokens
- [x] `itransformer/itransformer.rs` -- `ITransformer`: variate embedding -> N
      blocks -> per-variate head; `ITransformerConfig::tiny` / `base`

#### Forecasting Heads (`head/`, 2 files + mod)
- [x] `head/linear_head.rs` -- `LinearHead`: in -> out, batch + per-variate ts
      variants
- [x] `head/mlp_head.rs` -- `MlpHead`: in -> hidden -> out with ReLU, Kaiming init
      for layer 1

#### Integration tests (`lib.rs::tests`)
- [x] 20 E2E tests covering RevIN forward + inverse round-trip, InstanceNorm,
      MovingAvg, SeriesDecomp, PatchEmbed1d, TCN block + encoder, NHiTS
      hierarchical residual, PatchTST forward, TimesNet block + forward,
      iTransformer block + forward, Linear / MLP heads, plus PTX generation across
      6 SM versions
- [x] `benches/ts_ops.rs` Criterion benches -- 7 PTX bench groups x 4 SM versions +
      5 architecture forward benches

### Future Enhancements [ ]

#### P0 -- Critical (Mainstream Forecasting Coverage)
- [x] Autoformer auto-correlation attention (currently `auto_correlation_ptx` is a
      magnitude-squared kernel only)
- [x] Informer ProbSparse self-attention
- [x] N-BEATS basis-expansion forecasting backbone
- [x] Crossformer cross-dimensional attention (crossformer.rs -- Zhang 2023 ICLR; Dimension-Segment-Wise embedding + Two-Stage Attention: cross-time MHSA + router-based cross-dimension attention)
- [x] DLinear / NLinear baseline forecasting models

#### P1 -- Important (Architecture and Feature Coverage)
- [x] FEDformer Fourier-enhanced attention (fedformer.rs -- Zhou 2022 ICML; series decomposition + Frequency Enhanced Block: DFT → select M low-freq modes → per-mode complex linear → iDFT)
- [x] Pyraformer pyramidal attention (pyraformer.rs -- Liu 2022 ICLR; pyramidal multi-scale graph (coarsen by factor c per scale) + PAM intra-scale window + parent/child cross-scale attention, O(L) complexity)
- [x] TimeMixer multi-scale mixing (timemixer.rs -- Wang 2024 ICLR; series decomp + multi-scale average-pooling downsample + Past-Decomposable-Mixing + Future-Multipredictor-Mixing ensemble pred)
- [ ] Conformer-TS (audio Conformer adapted for TS)
- [x] Probabilistic forecasting head (quantile regression / DeepAR-style)
- [x] N-BEATS (nbeats) basis-expansion backbone (nbeats/nbeats.rs -- Oreshkin 2020 ICLR; trend + seasonality stacks with doubly residual learning; generic / interpretable basis functions; `NBeatsForecast`)
- [x] SARIMA (sarima/sarima.rs) — Seasonal ARIMA: Box-Jenkins (p,d,q)×(P,D,Q,s) model; Yule-Walker AR initialisation; CSS-MLE parameter estimation; seasonal differencing + backshift operator; `SarimaModel { order, seasonal_order }`
- [ ] Multi-task forecasting (joint horizon + classification)

#### P2 -- Nice-to-Have (Research / Advanced)
- [x] Temporal Fusion Transformer (TFT) variable-selection + gated residual network
- [ ] PatchTST-Crossformer hybrid
- [ ] Foundation-model adapters (TimeGPT / Chronos / Moirai loading interfaces)
- [ ] Moirai universal forecasting model (`foundation/moirai.rs`) — Salesforce 2024: Masked Encoder with any-variate patching + patch-mixture decoder for zero-shot universal forecasting; `MoiraiForecaster`
- [ ] Chronos probabilistic foundation model (`foundation/chronos.rs`) — Amazon 2024: quantisation tokenisation of continuous time series → T5 seq2seq language model backbone for zero-shot probabilistic forecasting; `ChronosPredictor`
- [x] Anomaly Transformer (`anomaly/anomaly_transformer.rs`) — Xu 2022 ICLR: association discrepancy between prior-association (Gaussian kernel) and series-association (attention) for unsupervised anomaly detection; `AnomalyTransformer`
- [ ] Structural Time Series decomposition (`decomp/sts.rs`) — Harvey 1990: Kalman-filter-based trend + seasonality + irregular state-space decomposition with EM parameter estimation; `StsDecomposer`
- [ ] Hierarchical reconciliation (MinT-style) for grouped time-series
- [ ] Online / streaming forecasting helper
- [ ] Quantised TCN / PatchTST (INT8 / FP8) inference path

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader.

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 615 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

Time-series workloads are dominated by GEMM (delegated to `oxicuda-blas`) and
sequence-conv (custom PTX). Per-kernel targets:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `moving_average_ptx` | T 96 -- 720, C 7 -- 321 (ETT / Weather / ECL) | P0 |
| `patch_embed_1d_ptx` | T 96 -- 720, patch_len 16 -- 24 | P0 |
| `causal_temporal_conv_ptx` | T 96 -- 720, channels 64 -- 256, kernel 3, dil 1..32 | P0 |
| `revin_normalize_ptx` | N x T x C up to 32 x 720 x 321 | P0 |
| `auto_correlation_ptx` | T 96 -- 720 (FFT-friendly sizes) | P1 |
| `multirate_pool_ptx` | T / stride in {1, 2, 4} (NHiTS) | P1 |
| `period_detect_ptx` | T 96 -- 720, k in {3, 5} | P2 |

Target: bandwidth-bound kernels at >=85% peak DRAM throughput on sm_80+.

## Notes

- All tensors use time-major `[T, C]` layout; variates are channels-last
- `RevIn` forward / inverse round-trip exact to within 1e-6 with affine = identity
- Series decomposition follows the Autoformer formulation:
  trend = MA(x), seasonal = x - trend
- PatchTST is channel-independent: each variate is processed by the same shared
  Transformer encoder (no cross-variate attention)
- iTransformer inverts the convention: tokens are *variates*, not time steps
- TimesNet O(T^2) DFT is the reference path; FFT-based period detection is future
  work (link with `oxicuda-fft`)
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `revin_normalize_ptx` uses `rcp.approx.f32` for stats normalisation
- [x] `auto_correlation_ptx` uses `mul.rn.f32` for magnitude squared
- [x] `period_detect_ptx` uses warp-shuffle reduction for top-k
- [x] PTX × SM 80, 86 generation verified in integration tests
- [ ] `cp.async` 3-stage pipeline for long-horizon patch embedding
- [ ] FP16 Transformer MHSA path with Tensor Cores (PatchTST, iTransformer)

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for very long-horizon staging (T > 1024)
- [ ] `wgmma.mma_async` for PatchTST / iTransformer MHSA paths
- [ ] FFT-based period detection on Hopper (link with `oxicuda-fft`)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) PatchTST / iTransformer inference path
- [ ] Tensor-Memory (TMEM) staged variate-token loads (iTransformer)

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] RevIN forward / inverse exact round-trip (within 1e-6)
- [x] InstanceNorm1d zero-mean unit-variance per variate
- [x] MovingAvg replicate-pad boundary behaviour verified
- [x] SeriesDecomp `trend + seasonal == x` reconstruction
- [x] PatchEmbed1d shape: `T -> num_patches * patch_len` with stride
- [x] TCN block dilated causal-conv future-leak prevention test
- [x] NHiTS hierarchical residual: backcast subtracts from input
- [x] PatchTST channel-independent: each variate processed identically
- [x] TimesNet O(T^2) DFT period detection returns top-k integer periods
- [x] iTransformer variate-token shape `[N, C, D]` not `[N, T, D]`
- [x] LinearHead / MlpHead output dim equals horizon
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`)
- [ ] Numerical agreement with `Time-Series-Library` / `nixtla-statsforecast`
      reference within 1e-4 relative
- [ ] ETT / Weather / ECL MSE / MAE match for reference checkpoint forecasts
- [ ] Crossformer / TFT / FEDformer benchmark coverage once implemented

### Implementation Deepening
- [ ] Autoformer full auto-correlation attention (not just magnitude-squared kernel)
- [ ] Informer ProbSparse self-attention for very long horizons
- [ ] Probabilistic forecasting heads (quantile / DeepAR Gaussian / Student-t)
- [ ] Multi-task heads (jointly forecast horizon + classify regime / event)
- [ ] FFT-based period detection (link with `oxicuda-fft`) -- O(T log T) vs current
      O(T^2) DFT
- [ ] Online / streaming forecasting helper (sliding-window inference)
- [ ] Hierarchical reconciliation (MinT) for grouped time-series

### Benchmark Coverage
- [x] `benches/ts_ops.rs` Criterion harness wired: 7 PTX bench groups x 4 SM
      versions + 5 architecture forward benches (TCN, NHiTS, PatchTST, TimesNet,
      iTransformer)
- [ ] GPU-side throughput vs reference (`Time-Series-Library`, `darts`, `nixtla`)
      once Linux+NVIDIA harness is available
- [ ] Long-horizon sweep (T in 96 / 192 / 336 / 720) across all 5 architectures
