# oxicuda-multimodal TODO

Pure-Rust multi-modal learning primitives covering cross-modal attention,
compact bilinear fusion (MLB/MFB), contrastive alignment (CLIP bidirectional
InfoNCE, ImageBind triple alignment), ITM head, BERT text encoder, ViT image
encoder, Conformer audio encoder, temporal ViT video encoder, prefix-LM
captioning, and VQA head. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda)
(Vol.28).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 5,942 SLoC (22 files)** -- 156 unit tests + 12 E2E integration tests

The crate covers the complete vision-language-audio-video pipeline used by
modern multi-modal foundation models. Encoders are simulation-grade for CPU
unit testing; the seven PTX kernels target NVIDIA SM 7.5 through SM 12.0.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `MultiModalError` (14 variants: DimensionMismatch, EmptyInput,
      InvalidTemperature, InvalidHeads, InvalidBatchSize, MismatchedSeqLens,
      InvalidFeatureDim, NanEncountered, TokenOutOfRange, InvalidModalityCount,
      InvalidKFactor, InvalidPatchCount, InvalidLayerCount, Internal); `MmResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG),
      `MultiModalHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- module exports + prelude + 12 E2E integration tests

#### PTX Kernels (7 kernels x 6 SM versions = 42 generators)
- [x] `ptx_kernels.rs::cross_attn_score_ptx` -- QK^T/sqrt(d_k) scaled dot-product
      with `fma.rn.f32`
- [x] `ptx_kernels.rs::modal_align_loss_ptx` -- bidirectional InfoNCE over batch
      diagonal
- [x] `ptx_kernels.rs::bilinear_pool_ptx` -- tanh Hadamard product for MLB/MFB
      compact bilinear pooling
- [x] `ptx_kernels.rs::temporal_pool_ptx` -- mean-pool across frames,
      `atom.global.add.f32`
- [x] `ptx_kernels.rs::token_merge_ptx` -- concatenate and project tokens for
      prefix-LM
- [x] `ptx_kernels.rs::gate_fusion_ptx` -- softmax-gated attention fusion across
      modalities
- [x] `ptx_kernels.rs::itm_bce_ptx` -- numerically stable BCE for image-text
      matching
- [x] `ptx_kernels.rs::f32_hex` -- f32 to 0F-prefixed hex literal helper

#### Cross-Modal Attention (cross_attn/)
- [x] `cross_attention.rs::CrossAttention` -- multi-head attention with Q from
      modality A, K/V from modality B; scaled dot-product, softmax, output
      projection; `CrossAttnConfig`, `CrossAttnWeights::identity()`
- [x] `self_cross_block.rs::SelfCrossBlock` -- pre-norm residual block
      LN -> self-attn -> LN -> cross-attn -> LN -> FFN with skip connections;
      `LayerNorm`, `FeedForward`, `SelfCrossBlockWeights`

#### Fusion (fusion/)
- [x] `concat_fusion.rs::ConcatFusion` -- concatenate embeddings then project to
      joint space
- [x] `bilinear_fusion.rs::MlbFusion` -- compact bilinear pooling
      MLB = tanh(W_v * v) (.) tanh(W_q * q) -> linear projection
- [x] `bilinear_fusion.rs::MfbFusion` -- expand-to-k*d, sum-pool pairs, tanh
- [x] `attention_fusion.rs::AttentionFusion` -- softmax over modality-specific
      keys -> weighted sum of value embeddings

#### Alignment (alignment/)
- [x] `contrastive.rs::clip_loss` -- L2-normalised bidirectional InfoNCE
      (symmetric cross-entropy over similarity matrix)
- [x] `contrastive.rs::imagebind_loss` -- triple alignment loss over three
      modalities via pairwise InfoNCE average
- [x] `contrastive.rs::l2_normalise` -- row-wise L2 normalisation helper
- [x] `matching.rs::ItmHead` / `itm_loss` -- two-layer MLP binary classifier
      for image-text matching; numerically stable BCE

#### Encoders (encoder/)
- [x] `text_encoder.rs::BertEncoder` -- token + positional embedding ->
      N x (self-attention + FFN + LN) transformer blocks -> CLS-pool ->
      d_model output; `BertConfig::tiny()`, `BertWeights::zeros()`
- [x] `image_encoder.rs::ViTEncoder` -- flatten patches -> linear embed ->
      CLS prepend -> positional embed -> N transformer blocks -> CLS-pool;
      `ViTEncoderConfig`, `ViTEncoderWeights`
- [x] `audio_encoder.rs::AudioEncoder` -- linear mel projection ->
      N Conformer blocks (conv + attn + FFN) -> statistics pooling
      (mean || std) -> 2 * d_model
- [x] `video_encoder.rs::VideoEncoder` -- spatial ViT per frame -> temporal
      attention -> mean-pool -> d_model

#### Captioning / VQA (caption/)
- [x] `prefix_lm.rs::PrefixLm` -- greedy autoregressive decoding with visual
      prefix cross-attention; `PrefixLmConfig`, `PrefixLmWeights`
- [x] `vqa_head.rs::VqaHead` / `vqa_loss` -- two-layer MLP over fused features
      -> n_answers logits; cross-entropy loss; `softmax` helper

#### Integration Tests (lib.rs e2e_tests)
- [x] `e2e_cross_attention_shape` -- output shape == [q_len * d_model]
- [x] `e2e_self_cross_block_residual` -- block output shape + finiteness
- [x] `e2e_mlb_fusion_shape` -- MLB output shape == [batch * d_out]
- [x] `e2e_attention_fusion_weights_sum` -- attention weights sum to 1.0
- [x] `e2e_clip_loss_identical_gives_ln_n` -- identical features -> loss approx ln(N)
- [x] `e2e_itm_loss_perfect_prediction_near_zero` -- high logits + matched
      labels -> loss < 0.01
- [x] `e2e_bert_encoder_shape` -- BERT CLS output == [d_model]
- [x] `e2e_vit_encoder_shape` -- ViT CLS output == [d_model]
- [x] `e2e_audio_encoder_shape` -- audio output == [2 * d_model] (mean||std)
- [x] `e2e_video_encoder_shape` -- video output == [d_model]
- [x] `e2e_vqa_head_shape_and_loss` -- VQA logits shape + softmax sum + CE loss
- [x] `e2e_ptx_kernels_all_sm_versions` -- all 7 kernels x 6 SM versions
      contain `.version`, `.visible .entry`, `sm_X`, and kernel name

#### Benchmarks (benches/mm_ops.rs)
- [x] 7 PTX kernel groups x 4 SM versions (PTX generation throughput)
- [x] `clip_loss_b64_d256` -- bidirectional InfoNCE
- [x] `mlb_fusion_b32_d512` -- compact bilinear pooling
- [x] `cross_attn_heads8_d64_len32` -- multi-head cross-attention
- [x] `bert_tiny_forward` -- BERT-tiny end-to-end
- [x] `vit_tiny_forward` -- ViT-tiny end-to-end

### Future Enhancements [ ]

#### P0 -- Critical (Performance-Sensitive Paths)
- [ ] FlashAttention v2 cross-attention kernel -- block-sparse softmax with
      shared-memory tiling for d_k = 64 / 128 on SM 8.0+
- [ ] Tensor Core path for cross-attention QK^T -- `mma.sync.aligned.m16n16k16`
      for f16/bf16, dispatch from `CrossAttention::forward` when SM >= 75
- [ ] Fused MLB epilogue -- combine tanh + Hadamard + projection into a single
      kernel to halve global memory traffic
- [ ] PTX kernels for `BertEncoder::forward` / `ViTEncoder::forward` -- emit
      single end-to-end transformer kernel instead of CPU loop

#### P1 -- Important (Feature Completeness)
- [x] BLIP-2 Q-Former architecture -- learned query tokens cross-attending to
      frozen image features (new `caption/qformer.rs`)
      (encoder/qformer.rs -- Li 2023; learned query tokens, interleaved
      self-attn + cross-attn (reuses CrossAttention) + FFN, fixed n_query
      output from variable image tokens)
- [x] Flamingo-style gated cross-attention -- `tanh(alpha) * cross_attn(x) + x`
      with learnable `alpha` per layer
      (cross_attn/flamingo.rs -- Alayrac 2022; GATED XATTN-DENSE
      y=x+tanh(α_attn)·xattn(x,vis), z=y+tanh(α_ffn)·FFN(y), gates zero-init
      ⇒ identity at init)
- [x] LLaVA-style visual instruction tuning -- projector MLP from CLIP visual
      features to LLM token space
      (alignment/llava_projector.rs -- Liu 2023 LLaVA; mlp_depth-layer MLP
      with GELU mapping CLIP visual features → LLM text-embedding space;
      project_one and batched project_tokens)
- [x] CoCa contrastive + generative loss -- joint training of contrastive head
      and captioning decoder
      (encoder/coca.rs -- Yu 2022; attentional image pooling + InfoNCE
      contrastive + cross-attn captioning decoder logits + weighted coca_loss)
- [ ] AudioCLIP three-way (audio-image-text) alignment -- extension of
      `imagebind_loss` to AudioCLIP layout
- [x] PerceiverIO cross-attention -- iterative latent bottleneck for
      heterogeneous modality counts
      (encoder/perceiver_io.rs -- Jaegle 2021; encode cross-attn input→fixed
      n_latents (size-independent), latent self-attn, decode cross-attn
      latents→output queries)
- [x] Whisper-style log-mel front-end -- raw waveform -> mel spectrogram
      preprocessing in `encoder/audio_encoder.rs`
      (alignment/whisper_log_mel.rs -- Radford 2023; raw-waveform →
      Hann-windowed framed power spectrum → mel-filterbank → log10;
      per-frame and full-waveform forward)

#### P2 -- Nice-to-Have (Advanced Features)
- [ ] Mixture-of-Modality-Experts (MoME) router -- per-token modality routing
      across expert FFNs
- [ ] Token merging (ToMe) for ViT -- bipartite soft matching to prune
      redundant patch tokens at inference
- [ ] Beam search and top-k / nucleus sampling for `PrefixLm`
- [ ] Sparse contrastive negatives via hard-negative mining for `clip_loss`
- [ ] Multi-resolution ViT (NaViT) -- variable patch grids in a single batch
- [ ] FP8 (E4M3 / E5M2) inference path for the encoders (Hopper / Ada)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

No CUDA SDK, no C, no Fortran. The crate compiles standalone and produces PTX
strings that can be consumed by `oxicuda-driver` / `oxicuda-launch` at runtime.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 156 unit + 12 E2E = 168 passing
- unwrap() calls: 0 (production code)
- `#![allow(clippy::needless_range_loop)]` at crate root for kernel-style loops
- All public APIs return `MmResult<T>` or `Result<T, MultiModalError>`

## Performance Targets

Reference shapes (cross-attention is the hot path in vision-language inference):

| Kernel | Shape | Target |
|--------|-------|--------|
| cross_attn_score | heads=8, d_k=64, q_len=kv_len=512 | >= 90% of cuBLAS QK^T (sm_80) |
| modal_align_loss (CLIP) | batch=256, dim=512 | >= 85% of hand-tuned TC kernel |
| bilinear_pool (MLB) | batch=64, d=512 | bandwidth-limited (>= 80% of peak HBM) |
| temporal_pool | T=16, batch=32, d=768 | reduction-limited |
| BERT-tiny forward | seq=128, d=128, layers=2 | functional benchmark only |
| ViT-tiny forward | image=224^2, patch=16, d=192 | functional benchmark only |

## Notes

- All encoders use deterministic `LcgRng` seeded by `MultiModalHandle` so unit
  tests are reproducible without depending on `rand` or `getrandom`.
- `clip_loss` divides by `temperature` before softmax; identical L2-normalised
  features at T=1 yield loss == ln(N) (verified by E2E test).
- `MlbFusion::forward` returns shape [batch * d_out]; `MfbFusion` returns
  [batch * d_out / k_factor] after the sum-pool over k pairs.
- The Conformer audio encoder pools to mean || std so the output is
  `2 * d_model`, matching the reference x-vector layout.
- `PrefixLm::greedy_decode` always terminates: configurable `max_len` caps the
  generated sequence even when no EOS token is predicted.

---

## Architecture-Specific Deepening

### Hopper (sm_90 / sm_90a)
- [ ] `wgmma.mma_async` path for transformer block QK^T / AV in `BertEncoder`
      and `ViTEncoder`
- [ ] TMA (`cp.async.bulk`) loading of patch embeddings in `ViTEncoder`

### Ampere (sm_80 / sm_86) / Ada (sm_89)
- [ ] `cp.async` triple-buffered pipeline for cross-attention K/V tiles
- [ ] FP8 (E4M3 input, E5M2 accumulate) cross-attention on sm_89+

### Blackwell (sm_100 / sm_120)
- [ ] 5th-gen Tensor Core (`mma.sp` 2:4 sparse) for MLB / MFB projection
- [ ] Cluster-launch attention spanning multiple SMs for long-context cross-attn

---

## Deepening Opportunities

### Verification Gaps
- [x] All 7 PTX generators emit `.version`, `.target sm_X`, and named entry per
      SM version (verified by `e2e_ptx_kernels_all_sm_versions`)
- [ ] Cross-attention numerical parity vs. reference NumPy/PyTorch within 1e-4
- [ ] CLIP loss gradient correctness vs. autograd (finite-difference check)
- [ ] BERT / ViT / Audio / Video encoder shape contracts cross-checked against
      Hugging Face configurations

### Implementation Deepening
- [x] L2-normalisation reused across `clip_loss` and `imagebind_loss`
- [x] Numerically stable BCE in `itm_loss` (logsumexp-style)
- [ ] Mixed-precision (bf16 storage, fp32 accumulate) variants of all encoders
- [ ] Fused softmax + dropout for cross-attention (training-only path)
- [ ] Padding-mask support in `BertEncoder` (currently assumes no padding)
- [ ] Causal-mask helper for autoregressive `PrefixLm` decoding
