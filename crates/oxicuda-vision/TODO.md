# oxicuda-vision TODO

Pure-Rust Vision Transformer & CLIP primitives for OxiCUDA: ViT patch embedding,
multi-head self-attention, CLIP contrastive learning, image augmentation, FPN
multi-scale features, DETR decoder, and bipartite set matching. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.20).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~29,949 total lines across 67 `src/` files
- **Tests:** 840 `#[test]` in src/ (all passing under `--all-features`)
- **Crate:** `oxicuda-vision` -- Vol.20 Vision Transformer & CLIP Primitives

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `VisionError` (15 variants): `DimensionMismatch`, `ShapeMismatch`,
      `EmptyInput`, `InvalidImageSize`, `InvalidPatchSize`, `InvalidEmbedDim`,
      `InvalidNumHeads`, `HeadDimMismatch`, `InvalidNumClasses`, `InvalidProjDim`,
      `NonPositiveTemperature`, `InvalidRoiBox`, `WeightShapeMismatch`, `NonFinite`,
      `Internal`; `VisionResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle),
      `VisionHandle`
- [x] `lib.rs` -- crate root with `prelude` module and 19 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 7 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `patch_embed_ptx` -- strided Conv2D: `[C, H, W] -> [N_patches, embed_dim]`
      with `fma.rn.f32`
- [x] `bilinear_interp_ptx` -- sub-pixel 4-tap bilinear sampler with half-pixel
      convention
- [x] `contrastive_loss_ptx` -- InfoNCE: 3-pass numerically-stable row-softmax +
      diagonal cross-entropy
- [x] `roi_align_ptx` -- per-bin bilinear RoI feature extraction with
      `sampling_ratio^2` sample averaging
- [x] `image_normalize_ptx` -- channel-wise `(x - mean[c]) / std[c]` in-place
- [x] `adaptive_avg_pool_ptx` -- adaptive 2D average pool with integer window bounds
- [x] `focal_loss_ptx` -- focal loss `-alpha*(1-p)^gamma * log(p)` via stable sigmoid + log

#### Patch Embedding (`patch_embed/`, 2 files + mod)
- [x] `patch_embed/conv2d_patch.rs` -- `PatchEmbedConfig`, `PatchEmbedWeights`
      (Xavier init), `PatchEmbed::forward`, `prepend_cls`
- [x] `patch_embed/pos_embed.rs` -- `pos_2d_sincos` (4-band H/W sinusoidal),
      `LearnablePosEmbed`, `add_pos_embed`

#### ViT (`vit/`, 3 files + mod)
- [x] `vit/vit_block.rs` -- `ViTBlock`: pre-norm MHSA + GELU MLP + residuals;
      `layer_norm`, `gelu_exact` (tanh approx), `softmax_rows`, `mhsa` helpers
- [x] `vit/vit_encoder.rs` -- `ViTEncoder`: N stacked `ViTBlock` + final LayerNorm
- [x] `vit/vit_model.rs` -- `ViTModel`: PatchEmbed -> CLS-prepend -> PosEmbed ->
      Encoder -> head; `ViTConfig::tiny()` (img=32, p=4, D=64, depth=2, heads=4,
      classes=10)

#### CLIP (`clip/`, 3 files + mod)
- [x] `clip/vision_encoder.rs` -- `ClipVisionEncoder` wrapping `ViTEncoder`, CLS-pool
      to `[embed_dim]`
- [x] `clip/projection.rs` -- `ProjectionHead`: linear + L2-norm; `cosine_sim`
- [x] `clip/contrastive.rs` -- `info_nce_loss`: symmetric InfoNCE with numerically-stable
      log-sum-exp

#### Augmentation (`augment/`, 3 files + mod)
- [x] `augment/geometric.rs` -- `random_crop`, `center_crop`,
      `random_horizontal_flip`, `resize_bilinear` (half-pixel bilinear)
- [x] `augment/photometric.rs` -- `color_jitter` (brightness/contrast/saturation),
      `random_grayscale` (YIQ luminance)
- [x] `augment/normalize.rs` -- `normalize_chw`, `IMAGENET_MEAN`, `IMAGENET_STD`;
      `AugOp` enum + `Pipeline::push` builder

#### FPN (`fpn/`, 2 files + mod)
- [x] `fpn/lateral.rs` -- `LateralConv1x1`: 1x1 conv channel reduction (Xavier init)
- [x] `fpn/top_down.rs` -- `Fpn`: lateral -> top-down (nearest upsample + add) ->
      3x3 smooth conv; `FeatureMap {data, channels, h, w}`

#### Detection (`detection/`, 3 files + mod)
- [x] `detection/roi_align.rs` -- CPU reference RoI Align with `bilinear_sample_2d`;
      validates `x2 > x1, y2 > y1`
- [x] `detection/detr_decoder.rs` -- `DetrDecoderLayer`: self-attn + cross-attn + FFN
      (pre-norm); `DetrDecoder` depth stack; `DetrConfig::tiny()`
- [x] `detection/set_match.rs` -- `bipartite_match` (greedy + 2-opt);
      `build_cost_matrix` (class CE + L1 box + GIoU); `giou`

#### Integration tests (`lib.rs::tests`)
- [x] 19 E2E tests covering PatchEmbed, PosEmbed sincos, ViT block / encoder / model,
      CLIP projection / InfoNCE, geometric / photometric augmentation, normalize,
      FPN lateral + top-down, RoI Align, DETR decoder, bipartite set-match GIoU,
      plus PTX generation across 6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Mainstream Vision Coverage)
- [x] FlashAttention-2 fused MHSA CPU reference (vit/flash_attention.rs -- Dao 2023; online-softmax single-pass attention with running max/sum + per-tile rescale, causal-optional, cross-attention; block-size-invariant, validated bit-close to a 3-pass `reference_attention` oracle). NOTE: the on-device fused PTX/`oxicuda-dnn` Tensor-Core kernel remains hardware-gated.
- [x] Swin Transformer windowed + shifted-window attention (vit/swin.rs -- Liu 2021 ICCV; window partition/reverse, cyclic shift, SW-MSA attention mask, relative position bias table, W-MSA/SW-MSA pre-norm block)
- [x] ConvNeXt modern-CNN block (convnext/block.rs -- Liu 2022 CVPR; depthwise 7×7 same-pad conv + channel LayerNorm + 1×1 4× expansion + GELU + 1×1 projection + layer scale + residual)
- [x] EfficientNet-V2 fused-MBConv block
- [x] BatchNorm folding into Conv2d for inference

#### P1 -- Important (Architecture and Feature Coverage)
- [x] DeiT / BEiT / DINO ViT training-time tricks: DropPath / stochastic depth (vit/drop_path.rs -- Huang 2016; per-sample (row-wise) residual-branch drop with inverted `1/keep_prob` rescale, eval = identity, `residual_add_train`, linear `drop_path_schedule` across depth; expectation-preservation verified by Monte-Carlo)
- [x] CLIP text encoder (text/clip_text.rs -- token + positional embeddings, causal self-attention transformer blocks, EOS/largest-id pooling, joint-space projection; accepts pre-tokenised id sequences -- BPE byte-pair merge table is an external tokenizer-data concern)
- [x] OWL-ViT open-vocabulary detection head (detection/owl_vit.rs -- per-patch image embeddings + text-query cosine matching + box-regression head; `OwlVit::forward`, `score_queries`)
- [x] Mask R-CNN segmentation head (link with FPN + RoI Align) (detection/mask_head.rs -- He 2017; per-RoI FCN n_conv 3×3 + 2× deconv + 1×1 to n_classes → per-class sigmoid masks; reuses RoIAlign)
- [x] Anchor generator / NMS helper for two-stage detectors (detection/anchor_nms.rs -- multi-scale anchor grid (sizes×ratios×strides), IoU, greedy NMS + Soft-NMS linear/gaussian decay)
- [x] SAM (Segment Anything) image-encoder building blocks (segmentation/sam.rs -- ViT image encoder + `PositionEmbeddingRandom` + sparse/dense `PromptEncoder` + two-way transformer `MaskDecoder`)

#### P2 -- Nice-to-Have (Research / Advanced)
- [x] RTMDet real-time detection transformer (detection/rtmdet.rs -- Lyu 2022: CSPNeXt backbone + PAFPN neck + decoupled head + SimOTA-lite cost; `RtmDet`, `decode_level`, `simota_cost`)
- [x] SAM image-encoder + prompt encoder + mask decoder (segmentation/sam.rs -- Kirillov 2023 ICCV: ViT image encoder + positional + sparse/dense prompt encoders + two-way transformer mask decoder; `Sam`, `SamConfig`)
- [x] Point Transformer for 3D point-cloud classification (pointcloud/point_transformer.rs -- Zhao 2021 ICCV: vector self-attention with subtraction relation + position encoding; `PointTransformerLayer`, `PointAttention`)
- [x] DINOv2 self-supervised pre-training loss (ssl/dinov2.rs -- Oquab 2023: centred/sharpened `dino_loss`, patch-level `ibot_loss`, EMA-teacher head update, `CenteringBuffer`, and `koleo_loss` Kozachenko–Leonenko differential-entropy regulariser over L2-normalised embeddings)
- [x] MAE (Masked Autoencoder) random-mask + decoder (vit/mae.rs -- He 2022 CVPR; partial Fisher–Yates random mask, encoder over visible tokens only, decoder reconstructs full sequence with shared mask_token, MSE loss over masked positions only)
- [x] EVA / EVA-CLIP large-scale variant configurations (vit/eva.rs -- `EvaVariant` presets EVA-02-Ti/S/B/L + EVA-g/14 → validated `ViTConfig`; `EvaPoolHead` mean-pool + post-LayerNorm + joint-space projection)
- [x] Tokens-to-Token ViT, CaiT, XCiT variants (vit/t2t.rs -- `soft_split` overlapping unfold / re-tokenization; vit/cait.rs -- `ClassAttention` + `LayerScale`; vit/xcit.rs -- `cross_covariance_attention` channel-wise attention, linear in tokens)
- [x] Mixup / CutMix data-augmentation helpers
- [x] Quantised ViT (INT8) inference path (vit/quantize.rs -- symmetric & affine `QuantParams`, per-output-channel `QuantWeight`, dynamic-per-row-activation integer-domain `QuantLinear`, `fake_quantize_symmetric`; round-trip error ≤ half-step, INT8 GEMM within normalised-RMS tolerance of f32). NOTE: FP8 (E4M3) inference is hardware-gated and intentionally not implemented.

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader.

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 840 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All (largest is `ptx_kernels.rs` at ~1741 lines)
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

ViT and CLIP workloads are dominated by GEMM (delegated to `oxicuda-blas`) and
softmax (delegated to `oxicuda-blas` / `oxicuda-dnn`). This crate's PTX kernels target:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `patch_embed_ptx` | 224^2 image, patch in {14, 16, 32} | P0 |
| `bilinear_interp_ptx` | 1024x1024 -> arbitrary (FPN / resize) | P0 |
| `contrastive_loss_ptx` | batch 256 -- 65536 (CLIP) | P0 |
| `roi_align_ptx` | 1000 RoI, output 7x7 / 14x14 | P1 |
| `image_normalize_ptx` | 3 x 224^2 -- 3 x 1024^2 | P1 |
| `adaptive_avg_pool_ptx` | C x H x W -> C x 1 x 1 (global) | P1 |
| `focal_loss_ptx` | num_anchors x num_classes (e.g. 100K x 80) | P2 |

Target: bandwidth-bound kernels at >=90% peak DRAM throughput on sm_80+.

## Notes

- All image tensors use CHW layout (channels-first)
- `pos_2d_sincos` splits H and W into separate sin / cos bands (4 bands total) to
  preserve `sin^2 + cos^2 = 1` per axis
- `gelu_exact` uses the tanh approximation for compatibility with PyTorch's default
- `info_nce_loss` is symmetric (image-to-text + text-to-image) and uses log-sum-exp
  for numerical stability
- `bipartite_match` is a greedy + 2-opt heuristic (full Hungarian algorithm is
  future P1 work)
- `giou` returns `iou - (|enclosing - union|) / |enclosing|`
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `patch_embed_ptx` uses `fma.rn.f32` for accumulation
- [x] `contrastive_loss_ptx` uses warp-shuffle reduction for row-softmax
- [x] PTX × SM 80, 86 generation verified in integration tests
- [ ] `cp.async` 3-stage pipeline in patch-embed for large images (requires GPU hardware)
- [ ] FP16 MHSA path with FP32 softmax (link with Tensor Cores) (requires GPU hardware)

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for image-tile staging in patch-embed (requires GPU hardware)
- [ ] `wgmma.mma_async` for MHSA QK^T and PV paths (requires GPU hardware)
- [ ] FlashAttention-2 with Hopper-specific warp specialisation (requires GPU hardware; CPU online-softmax reference done in vit/flash_attention.rs)
- [ ] Cluster-launch contrastive loss for very large CLIP batch (>=65536) (requires GPU hardware)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) ViT inference path (requires GPU hardware; INT8 CPU path done in vit/quantize.rs)
- [ ] Tensor-Memory (TMEM) staging for attention KV (requires GPU hardware)
- [ ] FP4 MHSA experimental path (requires GPU hardware)

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] PatchEmbed shape and CLS-prepend correctness
- [x] 2D sincos positional embedding `sin^2 + cos^2 = 1` invariant
- [x] ViT block residual identity at zero-weight initialisation
- [x] ViT model end-to-end on 32x32 input (tiny config)
- [x] CLIP projection L2-norm unit-magnitude verified
- [x] InfoNCE loss symmetry (image-to-text vs text-to-image)
- [x] Geometric augmentations preserve shape; flip is involution
- [x] Color jitter clamps to `[0, 1]` range
- [x] FPN top-down upsample + add produces correct output spatial dims
- [x] RoI Align matches CPU reference within 1e-6
- [x] DETR decoder cross-attn / self-attn shape consistency
- [x] Bipartite match assigns N x M unique pairs; GIoU in `[-1, 1]`
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`) (requires GPU hardware)
- [ ] Numerical agreement with `torchvision` / `transformers` reference within 1e-4 (requires reference checkpoints)
- [ ] ImageNet-1K top-1 accuracy match for reference ViT-Tiny checkpoint (requires reference checkpoint + dataset)
- [ ] CLIP zero-shot accuracy match on small reference dataset (requires reference checkpoint + dataset)

### Implementation Deepening
- [x] CLIP text encoder (text/clip_text.rs -- Transformer encoder; consumes pre-tokenised id sequences -- BPE / SentencePiece merge-table adapter is external tokenizer data)
- [x] Full Hungarian algorithm in `bipartite_match` (currently greedy + 2-opt) (detection/hungarian.rs -- Kuhn 1955 / Munkres 1957; O(n^3) Kuhn-Munkres with potentials u/v + alternating-tree augmenting paths + slack updates, f64 internal accumulation, rectangular padding to max(n_workers, n_jobs); `exact_bipartite_match` wraps it in the existing greedy signature)
- [x] Anchor generator + NMS post-processing for two-stage detectors
- [x] Mask head (mask R-CNN style) for instance segmentation (detection/mask_head.rs -- He 2017; per-RoI FCN 3×3 + 2× deconv + 1×1 → per-class sigmoid masks)
- [x] DropPath / stochastic-depth helpers for training regularisation (vit/drop_path.rs -- per-sample residual drop + inverted rescale + linear depth schedule)
- [ ] Multi-GPU CLIP contrastive (all-gather batch across devices) — requires multi-device runtime (distributed all-gather)

### Benchmark Coverage
- [x] `benches/vision_ops.rs` Criterion harness wired (CPU-side PTX generation +
      patch-embed + ViT-tiny forward)
- [ ] GPU-side throughput vs reference (`torchvision`, OpenCLIP) once Linux+NVIDIA
      harness is available (requires GPU hardware)
- [ ] CLIP batch-size sweep (256 / 1024 / 4096 / 16384) (requires GPU hardware for throughput numbers)
