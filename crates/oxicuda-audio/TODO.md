# oxicuda-audio TODO

Pure-Rust Audio / Speech ML architectures for OxiCUDA: Conformer encoder, Wav2Vec2
CNN feature extractor, CTC forward + prefix beam search, WaveNet dilated stack,
SpecAugment augmentation, speaker embeddings (x-vector TDNN, attentive pooling).
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.21).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~24,900 (68 files)
- **Tests:** 845 passing (#[test] count in src/)
- **Crate:** `oxicuda-audio` -- Vol.21 Audio/Speech ML Architectures

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `AudioError` (17 variants): `DimensionMismatch`, `ShapeMismatch`,
      `EmptyInput`, `InvalidNumMels`, `InvalidSequenceLength`, `InvalidEmbedDim`,
      `InvalidNumHeads`, `HeadDimMismatch`, `InvalidVocabSize`, `InvalidBeamWidth`,
      `InvalidDilation`, `InvalidKernelSize`, `InvalidStride`, `BlankOutOfRange`,
      `WeightShapeMismatch`, `NonFinite`, `Internal`; `AudioResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle),
      `AudioHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- crate root with `prelude` module and 21 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 7 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `stride_conv1d_ptx` -- strided 1-D conv for Wav2Vec2 CNN feature extractor
- [x] `dilated_conv1d_ptx` -- causal dilated conv (WaveNet filter + gate, left-pad)
- [x] `ctc_alpha_ptx` -- log-domain CTC forward alpha recursion with `log_sum_exp`
      via `ex2` / `lg2`
- [x] `spec_augment_mask_ptx` -- in-place time + freq masking via `setp` / `selp.f32`
- [x] `depthwise_conv1d_ptx` -- causal depthwise conv for Conformer conv module
- [x] `rel_pos_bias_ptx` -- relative-position bias table lookup with `min` / `max.u32`
      clamping
- [x] `stats_pool_ptx` -- two-pass mean + std pooling with warp-shuffle reduction

#### Features (`features/`, 3 files + mod)
- [x] `features/log_mel_adapter.rs` -- `LogMelInput` validated `[T, F]` wrapper for
      `oxicuda-signal` output
- [x] `features/cmvn.rs` -- `CmvnConfig`, `compute_cmvn`, `apply_cmvn` (per-channel
      zero-mean unit-variance)
- [x] `features/delta.rs` -- `compute_delta`, `compute_delta_delta`,
      `stack_delta_features` (central-difference, edge-pad)

#### Encoder (`encoder/`, 3 files + mod)
- [x] `encoder/wav2vec_cnn.rs` -- `Wav2VecCnnEncoder`: 7-layer stride-conv1d +
      group-norm + GELU; `wav2vec2_base()` and `tiny()` configs
- [x] `encoder/conv_module.rs` -- `ConvModule`: LN -> PW-expand -> GLU ->
      depthwise-causal -> BN -> Swish -> PW-reduce
- [x] `encoder/conformer_block.rs` -- `ConformerBlock` (macaron: 1/2 FFN + MHSA
      (rel-pos) + ConvModule + 1/2 FFN + LN) + `ConformerEncoder`;
      `ConformerConfig::tiny()` (D=64, heads=4, depth=2, kernel=15)

#### Attention (`attention/`, 2 files + mod)
- [x] `attention/rel_pos_encoding.rs` -- `RelPosEncoding {table: [2*max_len-1]}` with
      seeded init, `bias(q, k)`, `bias_matrix(Q, K)`
- [x] `attention/rel_pos_attention.rs` -- `RelPosAttention`: multi-head SDPA +
      relative-position bias pre-softmax

#### CTC (`ctc/`, 2 files + mod)
- [x] `ctc/forward.rs` -- `ctc_forward_log`: log-domain alpha recursion, extended
      target `l' = [blank, l0, blank, l1, ...]`, `log_sum_exp2` stable
- [x] `ctc/beam_search.rs` -- `ctc_beam_search`: CTC prefix beam search with
      `HashMap<Vec<usize>, (p_blank, p_nb)>` merge and pruning

#### Vocoder (`vocoder/`, 2 files + mod)
- [x] `vocoder/wavenet_block.rs` -- `WaveNetBlock`: dilated-causal-conv ->
      `tanh(x) * sigmoid(g)` gated activation -> skip + residual pointwise convs
- [x] `vocoder/dilated_stack.rs` -- `WaveNetStack`: multi-cycle `[1, 2, 4, ..., 512]`
      dilation schedule + 2-layer ReLU head; `tiny()` and `default_config()`

#### Augmentation (`augment/`, 2 files + mod)
- [x] `augment/spec_augment.rs` -- `time_mask`, `freq_mask` (SpecAugment),
      enum-dispatched `SpecAugOp` + `SpecAugPipeline::push` builder
- [x] `augment/time_warp.rs` -- `time_warp`: single-anchor bilinear time-axis warping
      (no-op when `T <= 2 * max_w`)

#### Speaker (`speaker/`, 3 files + mod)
- [x] `speaker/stats_pool.rs` -- `stats_pool`: two-pass Bessel-corrected temporal
      mean + std pooling `[T, C] -> [2C]`
- [x] `speaker/attentive_pool.rs` -- `AttentivePool`: bottleneck `tanh`-attention
      softmax over time -> weighted mean + std `[2C]`
- [x] `speaker/x_vector.rs` -- `XVectorTdnn`: 5-layer dilated TDNN (Snyder 2018),
      stats pool, 512-d affine; `default_config()` + `tiny()`

#### Integration tests (`lib.rs::tests`)
- [x] 21 E2E tests covering CMVN, delta features, Wav2Vec2 CNN encoder, Conformer
      block / encoder, RelPos attention bias, CTC forward + beam search, WaveNet
      block / stack, SpecAugment, time-warp, stats pool, attentive pool, x-vector
      TDNN, plus PTX generation across 6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Mainstream ASR / Speech Coverage)
- [x] Mel-filterbank computation in this crate (features/mel_filterbank.rs -- triangular mel-scale filterbank from magnitude spectrum; mel↔Hz Slaney/HTK 2595·log10 convention)
- [ ] Connectionist Temporal Classification (CTC) GPU dispatch via `ctc_alpha_ptx`
      (currently CPU log-domain only) (requires GPU hardware)
- [x] Streaming Conformer (chunked attention + left-context cache)
      (encoder/streaming_conformer.rs -- block-wise causal-with-left-context MHSA;
      LeftContextCache + forward_chunk give incremental decode that is numerically
      equivalent to the full masked forward; Chen 2021 / Wu 2020)
- [x] RNN-T / Transducer loss (alternative to CTC for streaming ASR)
- [x] Beam-search lattice rescoring with shallow-fusion LM (rescoring.rs -- shallow-fusion total=acoustic+lm_weight·LM+wip·len, n-best re-ranking + prefix-beam lattice expansion; distinct from ctc/beam_search)

#### P1 -- Important (Architecture and Feature Coverage)
- [x] Whisper-style encoder (encoder/whisper.rs -- Radford 2023; conv stem (n_mels→d_model k=3 + d_model→d_model k=3 s=2) GELU + sinusoidal positional embedding + n_layers pre-norm transformer encoder blocks)
- [x] HuBERT / WavLM SSL pre-training pipeline (encoder/hubert_ssl.rs --
      KMeansQuantizer acoustic-unit discovery (Lloyd + k-means++), span masking,
      MaskedPredictionHead cosine-sim logits + masked-only cross-entropy loss;
      HubertPretrainer.step; Hsu 2021 / Chen 2022)
- [x] ECAPA-TDNN modern speaker embedding (multi-scale dilated TDNN + SE-block)
- [x] HiFi-GAN / BigVGAN GAN vocoder generator
- [x] DPRNN / Conv-TasNet source separation block
- [x] Voice-Activity Detection (Silero-style) lightweight inference path (vad.rs -- frame log-energy + spectral-flatness classifier with onset + hangover hysteresis smoothing, segment extraction)

#### P2 -- Nice-to-Have (Research / Advanced)
- [x] FastSpeech2 / VITS TTS acoustic model (synthesis/fastspeech2.rs -- variance
      adaptor: DurationPredictor (log-domain) + LengthRegulator (frame expansion) +
      pitch/energy VariancePredictors with bin embeddings; FFT encoder/decoder
      (conv-FFN transformer blocks) -> mel projection; forward_train (GT durations)
      + forward_infer; Ren 2021)
- [x] RVQ neural-codec core (codec/rvq.rs:ResidualVectorQuantizer -- round-trip/nestedness/NN verified)
      Residual Vector Quantization (SoundStream/EnCodec/Bark acoustic-token core,
      Zeghidour 2021 / Defossez 2022): `n_quantizers` independent codebooks; greedy
      stage-by-stage residual Euclidean-NN `encode` -> codes, `decode` = sum of
      chosen entries, `quantize` -> (x_hat, codes, residual_norm); reserved zero
      code (row 0) guarantees the monotone residual-descent property per input;
      stage-wise residual k-means `fit` (guarded, never increases batch error) +
      `from_codebooks`. codec/bark.rs:`BarkCodec`/`BarkAcousticTokens` is a thin
      coarse/fine acoustic-token *layout* wrapper over the RVQ stages (loss-free
      regroup). 9 CPU tests: monotone error sweep [3.013,2.789,2.285,2.101,1.844,
      1.844,1.600] (non-increasing as n_quantizers grows), exact-sum recovery
      (codes=[1,2,3] err=0), per-stage NN == brute-force argmin, fit improvement
      22.08->0.041 on clustered data (<= pre-fit error), determinism/shapes/finite,
      tier-split round-trip == flat RVQ, coarse-only error >= full error.
  - [ ] Bark *trained* semantic + acoustic token generation (the autoregressive
        transformers that generate semantic/coarse/fine tokens from text; Suno 2023)
        -- requires training-scale data, NOT CPU-unit-verifiable. The codec token
        *layout* (coarse/fine tiers) + exact round-trip are done above; the closest
        CPU primitive for semantic units is encoder/hubert_ssl.rs:KMeansQuantizer,
        but the trained semantic transformer itself is out of scope.
- [x] CTC + Attention joint-decoding helper (ESPnet style) (already in
      src/ctc/joint_ctc_attention.rs as `JointCtcAttention` -- Watanabe 2017
      λ·CTC + (1-λ)·attention joint_score + frame-synchronous greedy decode)
- [x] Quantised Conformer (INT8) inference path (encoder/quantized.rs --
      symmetric per-channel INT8 weight quant + per-tensor activation quant,
      integer-arithmetic-only GEMM (i32 accumulation), QuantizedLinear/QuantizedFfn;
      Jacob 2018 / Krishnamoorthi 2018). FP8 path remains (requires GPU hardware).
- [x] Streaming RNN-T greedy decoder (already in src/ctc/transducer_decode.rs as
      `TransducerGreedyDecoder` -- streaming greedy decode with caller-supplied
      joint closure; Graves 2012/2013)
- [ ] Multi-GPU SSL pre-training helper (DDP-style gradient sync) (requires multi-GPU hardware)
- [x] Whisper-CTC joint model — same construction as the Watanabe 2017 hybrid
      CTC/attention joint decode, already implemented in
      src/ctc/joint_ctc_attention.rs as `JointCtcAttention` (shared-encoder output
      scored by both branches; joint λ·CTC + (1-λ)·attention log-probs)
- [x] VITS2 TTS end-to-end acoustic model (synthesis/vits2/ submodule: common.rs,
      flow.rs, duration.rs, encoder.rs, mod.rs) — Kim 2021 / Kong 2023 -- conditional
      VAE + normalizing-flow core. `Vits2Flow` (ActNorm + affine coupling + channel
      flip, exact forward/inverse, analytic logdet); `StochasticDurationPredictor`
      (conditioned coupling flow over a 2-ch duration latent, sampling + stochastic
      change-of-variables ELBO); `PosteriorEncoder` q(z|x) + `PriorEncoder` p(z|c)
      (reuses FastSpeech2 FftBlock) + `reparameterize`; `gaussian_kl` (closed-form,
      >=0) and `flow_kl` (VITS MC ELBO term); `monotonic_alignment_search` (hard DP
      alignment); `Vits2` analysis (teacher) + inference (synthesis) passes returning
      a mel [t_mel,n_mels]. 51 CPU tests: flow + SDP-flow invertibility
      (inverse(forward(x))≈x, max err 4.8e-7 / 3.0e-8), logdet vs finite-difference
      Jacobian (dim 4/6, max err 3.5e-4 / 2.3e-5), KL>=0 & ==0 at posterior==prior,
      shape/finiteness/determinism; RQ-spline bijection (scalar fwd/inv 3.8e-6 /
      3.2e-6, coupling 1.2e-7), scalar logdet vs finite-diff (2.4e-4), coupling
      logdet vs numerical Jacobian (1.8e-5), C¹ tail continuity, strict monotonicity,
      spline-dequant ELBO finite/stochastic/deterministic.
  - [ ] VITS2 adversarial training loop (HiFi-GAN multi-period/multi-scale
        discriminators + feature-matching + mel reconstruction GAN objective) and
        the joint end-to-end waveform decoder — out of CPU/honest scope (needs
        training-scale GPU + a discriminator we cannot meaningfully verify on CPU);
        the HiFi-GAN *generator* already exists in vocoder/hifigan.rs.
  - [x] (synthesis/vits2/spline.rs:RationalQuadraticSpline -- monotone RQ spline,
        bijection+logdet verified) VITS SDP posterior spline-flow dequantization
        -- Durkan 2019 rational-quadratic neural spline. `RationalQuadraticSpline`
        (K-bin monotone RQ map on [-B,B], softmax widths/heights + softplus internal
        derivatives, δ₀=δ_K=1 identity tails; exact forward/inverse via per-bin
        quadratic + Newton polish, analytic logdet = Σ log(dy/dx)) and
        `RqSplineCoupling` (Durkan spline coupling: identity half + external cond
        emits per-element 3K-1 params for the transformed half, triangular Jacobian).
        Wired into the SDP as the auxiliary dequantiser: base noise u~N(0,1) ->
        e=T(u) conditioned on log d + text, with exact +logdet_dequant
        change-of-variables — a strictly tighter, still-valid ELBO (identity spline
        recovers the old fixed-N(0,1) augmented-flow bound).
- [x] Beat Tracker (already in src/rhythm/beat_tracker.rs as `BeatTracker` --
      dynamic-programming beat tracking over an onset-strength function with a
      tempo prior; Ellis 2007 / Böck 2011)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader.

Mel-filterbank / STFT upstream features are provided by `oxicuda-signal` and
consumed via the `LogMelInput` adapter.

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 845 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

ASR / speech workloads are dominated by GEMM (delegated to `oxicuda-blas`) and
sequence-conv (custom PTX). Per-kernel targets:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `stride_conv1d_ptx` | Wav2Vec2 base (T=16000, channels 512) | P0 |
| `dilated_conv1d_ptx` | WaveNet stack (dilation 1..512, channels 32 -- 256) | P0 |
| `ctc_alpha_ptx` | T x V = 1000 x 128 -- 10000 x 5000 | P0 |
| `depthwise_conv1d_ptx` | Conformer (T 1000, D 256 -- 512, kernel 15) | P0 |
| `spec_augment_mask_ptx` | mel 80 x T 1000 | P1 |
| `rel_pos_bias_ptx` | seq 1000, heads 4 -- 8 | P1 |
| `stats_pool_ptx` | T x C = 1000 x 512 -- 1500 x 1500 | P2 |

Target: bandwidth-bound kernels at >=85% peak DRAM throughput on sm_80+.

## Notes

- Audio tensors use `[T, F]` (time-major) or `[T, C]` layout throughout
- CTC blank index is configurable; default is `0` (`BlankOutOfRange` raised
  otherwise)
- Conformer macaron-FFN scaling = 0.5 (paper-faithful)
- SpecAugment time-mask + freq-mask are in-place; `time_warp` is a single-anchor
  bilinear approximation (no-op when `T <= 2 * max_w`)
- WaveNet uses left-pad causal convolution; receptive field per stack = sum of
  dilations
- Stats pool uses two-pass Welford / Bessel correction for numerical stability
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `ctc_alpha_ptx` uses `ex2.approx` / `lg2.approx` for stable log-sum-exp
- [x] `stats_pool_ptx` uses warp-shuffle reduction
- [x] `spec_augment_mask_ptx` uses `setp` / `selp.f32` predicate select
- [x] PTX × SM 80, 86 generation verified in integration tests
- [ ] `cp.async` 3-stage pipeline for stride-conv1d on long sequences (requires GPU hardware)
- [ ] FP16 Conformer MHSA path with Tensor Cores (requires GPU hardware)

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for very long audio sequence (T > 16K) (requires GPU hardware)
- [ ] `wgmma.mma_async` for Conformer MHSA QK^T / PV (requires GPU hardware)
- [ ] Cluster-launch CTC alpha for very large vocabulary (V >= 5000) (requires GPU hardware)
- [ ] Distributed-shared-memory for streaming Conformer left-context cache (requires GPU hardware)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) Conformer inference path (requires GPU hardware; INT8 CPU path done in encoder/quantized.rs)
- [ ] FP4 streaming RNN-T decoder experimental path (requires GPU hardware)
- [ ] Tensor-Memory (TMEM) staged audio-tile loads (requires GPU hardware)

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] CMVN per-channel zero-mean unit-variance correctness
- [x] Delta / delta-delta central-difference + edge-pad correctness
- [x] Wav2Vec2 CNN encoder downsampling factor (320x at base config)
- [x] Conformer block residual identity at zero-weight init
- [x] RelPos attention bias additivity pre-softmax
- [x] CTC forward log-domain probability normalisation
- [x] CTC beam-search returns deterministic top-k under fixed RNG seed
- [x] WaveNet block residual + skip-connection shape consistency
- [x] SpecAugment time + freq masks zero the masked region
- [x] Time-warp is no-op when `T <= 2 * max_w`
- [x] Stats pool Bessel-corrected std matches CPU reference within 1e-6
- [x] Attentive pool weights sum to 1 (softmax invariant)
- [x] X-vector TDNN output dim = 512 (default config)
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`) (requires GPU hardware)
- [ ] Numerical agreement with ESPnet / NeMo reference within 1e-4 relative (requires reference checkpoints / hardware)
- [ ] LibriSpeech CTC WER match for reference small-Conformer checkpoint (requires reference dataset / checkpoint)
- [ ] VoxCeleb speaker EER match for reference x-vector checkpoint (requires reference dataset / checkpoint)

### Implementation Deepening
- [x] End-to-end `LogMelExtractor` (STFT -> mel -> log) within this crate
      (features/log_mel_extractor.rs -- composes the crate's own stft_hann + power
      spectrum + MelFilterbank + log-floor; whisper_like()/tiny() presets;
      extract / extract_input)
- [ ] CTC GPU dispatch end-to-end via `ctc_alpha_ptx` (requires GPU hardware)
- [x] Streaming Conformer left-context cache + chunk-attention helper
      (encoder/streaming_conformer.rs -- StreamingConformerAttention + LeftContextCache)
- [x] RNN-T / Transducer loss with joint network (alternative to CTC)
      (already in src/ctc/rnnt.rs as `rnnt_loss` / `RnntConfig`; Graves 2012)
- [x] Beam-search shallow-fusion with external LM scorer interface
      (already in src/rescoring.rs as `LatticeRescorer` / `RescoreConfig`)
- [x] HiFi-GAN / BigVGAN GAN vocoder generator (already in src/vocoder/hifigan.rs
      as `HifiGanGenerator` -- transposed-conv upsampling + multi-receptive-field
      residual blocks; Kong 2020)

### Benchmark Coverage
- [x] `benches/audio_ops.rs` Criterion harness wired (CPU-side PTX generation +
      Conformer block / WaveNet stack forward)
- [ ] GPU-side throughput vs reference (ESPnet, NeMo) once Linux+NVIDIA harness is
      available (requires GPU hardware)
- [ ] Real-time-factor (RTF) measurement for streaming Conformer at various chunk
      sizes (requires GPU hardware for meaningful RTF)
