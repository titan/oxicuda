# oxicuda-ssl TODO

Pure-Rust self-supervised learning primitives for OxiCUDA, covering the four
canonical SSL families: contrastive (SimCLR, MoCo), non-contrastive (BYOL,
Barlow Twins, VICReg), masked (MAE), and clustering (SwAV, DINO). Plus shared
infrastructure for momentum encoders, projection / predictor heads, and SSL
data augmentation. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.26).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 4,262 (25 files)
- **PTX kernels:** 7 kernel generators emitted for 6 SM targets (sm_75 / 80 / 86 / 90 / 100 / 120)
- **Coverage:** CPU reference implementation + PTX string generation for GPU execution

### Completed

#### Core Infrastructure
- [x] `error.rs` (113 LoC) -- `SslError` (16 variants: DimensionMismatch, EmptyInput, InvalidTemperature, InvalidMomentum, InvalidMaskRatio, InvalidNumCrops, InvalidLossWeight, QueueCapacityTooSmall, QueueEmpty, NumPrototypesTooSmall, SinkhornDiverged, InvalidFeatureDim, BatchTooSmall, NanEncountered, InvalidProjectorDim, Internal) + `SslResult<T>`
- [x] `handle.rs` (264 LoC) -- `SmVersion`, `LcgRng` (Knuth MMIX core + Box-Muller normals + Fisher-Yates shuffle), `SslHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` (249 LoC) -- Module exports + `prelude` re-exports + 12 E2E integration tests

#### PTX Kernels (ptx_kernels.rs, 643 LoC)
- [x] `nt_xent_softmax_ptx` -- Per-row stable softmax over `2N x 2N` similarity matrix with `selp.f32` self-mask `-INF` on the diagonal
- [x] `momentum_update_ptx` -- `theta_target = m*theta_target + (1-m)*theta_online` with `fma.rn.f32` and grid-stride loop
- [x] `byol_cosine_loss_ptx` -- `2 - 2*cos(p, sg(z))` per-element accumulation via `atom.global.add.f32`
- [x] `barlow_cross_corr_ptx` -- Cross-correlation matrix `C[i,j] = sum_n Z_A[n,i] * Z_B[n,j]` with 2-D grid + `atom.global.add.f32`
- [x] `random_mask_ptx` -- Bernoulli mask via inline LCG `(rand < drop_ratio) ? 0 : 1` for MAE patch dropping
- [x] `cosine_similarity_ptx` -- Per-pair cosine similarity for memory-bank lookup with `atom.global.add.f32`
- [x] `gather_features_ptx` -- Memory-queue gather `out[k, d] = queue[idx[k], d]` for MoCo negatives

#### Contrastive (contrastive/)
- [x] `info_nce.rs` (259 LoC) -- Symmetric InfoNCE with stable log-sum-exp; returns `(loss, accuracy@1)`
- [x] `simclr.rs` (116 LoC) -- `simclr_loss` / `SimClrConfig` symmetric NT-Xent at default temperature tau = 0.1 (Chen 2020)
- [x] `moco.rs` (301 LoC) -- `MocoQueue` FIFO circular queue + `moco_loss` with positive vs queue negatives (He 2020)

#### Non-Contrastive (non_contrastive/)
- [x] `byol.rs` (156 LoC) -- `byol_loss` / `ByolPredictor` L2-normalised cosine `2 - 2*cos(p, sg(z))` (Grill 2020)
- [x] `barlow.rs` (215 LoC) -- `barlow_twins_loss` / `BarlowTwinsConfig` cross-correlation `sum(1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2` after column standardisation (Zbontar 2021)
- [x] `vicreg.rs` (230 LoC) -- `vicreg_loss` / `VicRegConfig` variance hinge + invariance MSE + off-diagonal covariance penalty (Bardes 2022)

#### Masked (masked/)
- [x] `mae.rs` (218 LoC) -- `random_patch_mask` Fisher-Yates patch selection (exact ratio) + `mae_reconstruction_loss` masked-patch-only MSE; default mask ratio 0.75 (He 2022)

#### Clustering (clustering/)
- [x] `swav.rs` (339 LoC) -- `swav_loss` + `sinkhorn_knopp` normalised codes + swapped CE (default 3 iters, epsilon = 0.05, tau = 0.1) (Caron 2020)
- [x] `dino.rs` (286 LoC) -- `dino_loss` + `update_dino_centre` centred + sharpened student-teacher CE (tau_s = 0.1, tau_t = 0.04) (Caron 2021)

#### Augment (augment/)
- [x] `color.rs` (188 LoC) -- `color_jitter` per-channel multiplicative jitter + `random_grayscale_chw` BT.601 grayscale conversion on `[3, H, W]` images
- [x] `multi_crop.rs` (145 LoC) -- `multi_crop` / `MultiCropConfig` / `CropSpec` DINO / SwAV global+local crop spec generation (default 2 globals @ 224 + 6 locals @ 96)

#### Momentum (momentum/)
- [x] `ema.rs` (165 LoC) -- `EmaUpdater` element-wise EMA target update + `cosine_momentum` half-cosine momentum schedule (BYOL: 0.996 -> 1.0)

#### Head (head/)
- [x] `projector.rs` (191 LoC) -- `MlpProjector` 2-layer Linear -> ReLU -> Linear projection head with Kaiming init; per-sample and batched forward
- [x] `predictor.rs` (145 LoC) -- `PredictorHead` identical architecture for BYOL / SimSiam online-branch predictor

#### Integration Tests (lib.rs)
- [x] 12 E2E tests: SimCLR aligned-pair drop, MoCo queue lifecycle, BYOL identity = 0, Barlow finite-loss, VICReg three-term combine, MAE mask ratio + perfect reconstruction, Sinkhorn-Knopp uniform row-sum, DINO centred CE finite, EMA monotone momentum, MLP projector shape, multi_crop count, PTX kernels x 6 SM versions

#### Benchmarks (benches/ssl_ops.rs)
- [x] 7 PTX kernel generator groups x 4 SM versions + 5 algorithm benches: `simclr_loss_b64_d128`, `moco_loss_b16_d64_q256`, `barlow_loss_b256_d64`, `mae_mask_p196_r075`, `dino_loss_b64_k128`

### Future Enhancements

#### P0 -- Critical (SSL Algorithm Coverage Gaps)
- [x] SimSiam stop-gradient baseline -- BYOL minus the momentum branch (Chen & He 2021); shares the existing `PredictorHead` architecture (`non_contrastive/simsiam.rs`)
- [x] MoCo v3 -- ViT-friendly projector + predictor combo; within-batch InfoNCE + cosine momentum schedule (`contrastive/moco_v3.rs`)
- [x] MSN -- masked siamese networks combining masked modelling with siamese contrastive loss (Assran 2022) (`non_contrastive/msn.rs`)
- [x] DenseCL / PixPro -- pixel-level dense contrastive losses for downstream-detection-friendly features (`non_contrastive/dense_cl.rs`)

#### P1 -- Important (Masked & Clustering Depth)
- [x] BEiT-style discrete tokeniser MAE -- predict VQ tokens instead of raw pixels (Bao 2021) (`masked/beit.rs`)
- [x] SimMIM rough random masking + L1 reconstruction -- complement to `random_patch_mask` (Xie 2022) (`masked/simmim.rs`)
- [x] data2vec joint-embedding masked prediction -- masked feature regression as a generic SSL recipe (Baevski 2022) (`masked/data2vec.rs`)
- [x] DeepCluster + DeeperCluster classical clustering baselines on top of `sinkhorn_knopp` (`clustering/deep_cluster.rs`)
- [x] iBOT / EsViT prototype-based clustering with online assignment (`clustering/ibot.rs`)

#### P2 -- Nice-to-Have (Augmentation, Probing, Diagnostics)
- [x] Solarization / Gaussian blur augmentations to round out SimCLR / BYOL recipes (`augment/solarize_blur.rs`)
- [x] AutoAugment / RandAugment policies wired into the augment pipeline (`augment/rand_augment.rs`)
- [x] Linear probing helper -- frozen-feature logistic regression evaluator with k-fold CV (`head/linear_probe.rs`)
- [x] kNN classifier eval -- standard cosine-NN evaluator on frozen features for monitoring training (`metrics/knn_eval.rs`)
- [x] Feature uniformity / alignment metrics (Wang & Isola 2020) -- diagnostic for collapse detection (`metrics/feature_metrics.rs`)
- [x] Representation rank / effective dimensionality diagnostics for non-contrastive methods (`metrics/feature_metrics.rs` effective_rank)

#### GPU Launcher Wiring
- [ ] Wire `ptx_kernels::*` strings through `oxicuda-launch::Kernel::from_module` for end-to-end GPU execution (PTX strings are emitted; CPU paths are the authoritative reference)
- [ ] GPU-resident `nt_xent_softmax_ptx` integrated with `oxicuda-blas` GEMM for the `Z @ Z^T` similarity step
- [ ] GPU-resident `momentum_update_ptx` parameter update fused with optimiser step

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
- Tests: 12 E2E in `lib.rs` + module unit tests (see root TODO.md Vol.26 reference for the workspace-wide count)
- `unwrap()` calls: 0 in library code
- macOS: compiles, returns `UnsupportedPlatform` from any actual GPU launch
- PTX targets covered: sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120

## Performance Targets

| Operation | Size | Target |
|-----------|------|--------|
| PTX kernel string generation | per call | < 100 us |
| `simclr_loss` (CPU, log-sum-exp stable) | N = 256, D = 128 | < 50 ms |
| `moco_loss` (queue Q = 65536, D = 128) | N = 256 | < 100 ms CPU |
| `barlow_twins_loss` (N = 512, D = 256) | -- | < 100 ms CPU |
| `vicreg_loss` (N = 512, D = 256) | -- | < 100 ms CPU |
| `mae_reconstruction_loss` (P = 196, D = 768) | -- | < 50 ms CPU |
| `sinkhorn_knopp` (N = 1024, K = 4096, iters = 3) | -- | < 50 ms CPU |
| `dino_loss` (N = 256, K = 65536) | -- | < 100 ms CPU |
| `EmaUpdater::update` (P = 25M params, BYOL ResNet50) | -- | < 50 ms CPU |
| `MlpProjector::forward` (single sample, 2048 -> 256) | -- | < 10 us |

Targets are CPU-reference budgets. Once GPU wiring lands, `nt_xent_softmax_ptx`
combined with a `oxicuda-blas` `Z @ Z^T` GEMM should approach Tensor-Core
throughput for the projection-similarity step.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/ssl_ops.rs`) -- 7 PTX kernel groups x 4 SM versions + 5 algorithm benches

## Notes

- All PTX kernels emit `.target sm_<version>` and use a grid-stride loop pattern.
- `nt_xent_softmax_ptx` self-mask is implemented with `selp.f32 ..., NEG_INF, ..., %p0` on the diagonal predicate -- callers should pre-normalise projections to unit norm.
- `MocoQueue` uses circular FIFO semantics: `enqueue` overwrites the oldest entry once `len() == capacity`; reads are valid only after the first full pass.
- `random_patch_mask` produces an *exact* mask ratio via Fisher-Yates -- not a Bernoulli draw -- so the test asserts `n_masked == floor(P * ratio)`.
- `sinkhorn_knopp` returns `SslError::SinkhornDiverged` after `max_iter` if row / column sums fail to converge -- check return values rather than `unwrap`.
- The PTX kernels target scalar f32 paths; no Tensor-Core (wgmma / mma.sync) usage -- projection-head and SimCLR similarity GEMMs delegate to `oxicuda-blas`.

---

## Architecture-Specific Deepening

### PTX Generation by SM Version

| SM Version | PTX Version | Notes |
|------------|-------------|-------|
| sm_75 (Turing) | 7.5 | Baseline; `selp.f32`, `fma.rn.f32`, `atom.global.add.f32` supported |
| sm_80 / sm_86 (Ampere) | 8.0 | Default target for `SslHandle::default_handle()` |
| sm_89 (Ada) | 8.0 | Treated as sm_80 by `ptx_version_str()` |
| sm_90 / sm_90a (Hopper) | 8.4 | No `wgmma` usage -- SSL kernels are reductions / blends |
| sm_100 / sm_120 (Blackwell) | 8.7 | Same scalar pattern |

The 7 generators all dispatch on the SM string and emit identical scalar PTX
modulo the `.target` directive. SSL losses are dominated by reductions and
masking, while the heavy `Z @ Z^T` work is intentionally delegated to
`oxicuda-blas`.

### Deepening Opportunities

- [ ] Hopper `barlow_cross_corr_ptx` rewrite using `wgmma.mma_async` for the `Z_A^T @ Z_B` outer product (currently scalar with atomic accumulate)
- [ ] Hopper `nt_xent_softmax_ptx` warp-level reduction with `redux.sync.max.f32` + `redux.sync.add.f32` to eliminate shared-memory traffic
- [ ] Blackwell (sm_100+) `cp.async.bulk.tensor` for the MoCo queue gather in `gather_features_ptx`
- [ ] FP16 / BF16 variants of `momentum_update_ptx` and `byol_cosine_loss_ptx` for mixed-precision SSL training

---

## Functional Quality Gates (Vol.26)

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| S1 | InfoNCE symmetric loss with stable log-sum-exp | P0 | [x] |
| S2 | SimCLR NT-Xent loss with diagonal self-mask | P0 | [x] |
| S3 | MoCo FIFO queue + queue-negatives InfoNCE | P0 | [x] |
| S4 | BYOL L2-normalised cosine loss | P0 | [x] |
| S5 | Barlow Twins cross-correlation loss | P0 | [x] |
| S6 | VICReg variance + invariance + covariance | P0 | [x] |
| S7 | MAE random patch mask + reconstruction MSE | P0 | [x] |
| S8 | SwAV Sinkhorn-Knopp + swapped-CE | P0 | [x] |
| S9 | DINO centred + sharpened student-teacher CE | P0 | [x] |
| S10 | EMA momentum encoder update | P0 | [x] |
| S11 | Cosine momentum schedule | P0 | [x] |
| S12 | MLP projector + predictor heads with Kaiming init | P0 | [x] |
| S13 | Multi-crop generator (global + local crops) | P0 | [x] |
| S14 | Color jitter + grayscale augmentations | P1 | [x] |
| S15 | PTX generators for 7 kernels x 6 SM versions | P0 | [x] |

## Performance Verification Harness Status

- All performance numbers above are CPU-side targets achievable on the build host.
- GPU end-to-end harnesses await the [ ] GPU launcher wiring item plus a
  Linux+NVIDIA test runner; the PTX strings themselves are covered by
  string-content unit tests inside `ptx_kernels.rs`.
