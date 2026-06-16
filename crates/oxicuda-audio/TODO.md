# oxicuda-audio TODO

Pure-Rust Audio / Speech ML architectures for OxiCUDA: Conformer encoder, Wav2Vec2
CNN feature extractor, CTC forward + prefix beam search, WaveNet dilated stack,
SpecAugment augmentation, speaker embeddings (x-vector TDNN, attentive pooling).
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.21).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 16,665 (54 files, Rust 16,665 code + 1,217 comments + 1,074 blanks)
- **Tests:** 669 passing (#[test] count in src/)
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
      (currently CPU log-domain only)
- [ ] Streaming Conformer (chunked attention + left-context cache)
- [x] RNN-T / Transducer loss (alternative to CTC for streaming ASR)
- [x] Beam-search lattice rescoring with shallow-fusion LM (rescoring.rs -- shallow-fusion total=acoustic+lm_weight·LM+wip·len, n-best re-ranking + prefix-beam lattice expansion; distinct from ctc/beam_search)

#### P1 -- Important (Architecture and Feature Coverage)
- [x] Whisper-style encoder (encoder/whisper.rs -- Radford 2023; conv stem (n_mels→d_model k=3 + d_model→d_model k=3 s=2) GELU + sinusoidal positional embedding + n_layers pre-norm transformer encoder blocks)
- [ ] HuBERT / WavLM SSL pre-training pipeline
- [x] ECAPA-TDNN modern speaker embedding (multi-scale dilated TDNN + SE-block)
- [x] HiFi-GAN / BigVGAN GAN vocoder generator
- [x] DPRNN / Conv-TasNet source separation block
- [x] Voice-Activity Detection (Silero-style) lightweight inference path (vad.rs -- frame log-energy + spectral-flatness classifier with onset + hangover hysteresis smoothing, segment extraction)

#### P2 -- Nice-to-Have (Research / Advanced)
- [ ] FastSpeech2 / VITS TTS acoustic model
- [ ] Bark-style hierarchical semantic + acoustic tokens
- [ ] CTC + Attention joint-decoding helper (ESPnet style)
- [ ] Quantised Conformer (INT8 / FP8) inference path
- [ ] Streaming RNN-T greedy decoder
- [ ] Multi-GPU SSL pre-training helper (DDP-style gradient sync)
- [ ] Whisper-CTC joint model (`decoder/whisper_ctc.rs`) — Watanabe 2017 Interspeech: shared encoder output fed to both attention-decoder and CTC branch; joint decoding with λ·CTC + (1-λ)·attention log-probs; `WhisperCtcDecoder`
- [ ] VITS2 TTS end-to-end acoustic model (`synthesis/vits2.rs`) — Kong 2023: improved normalizing-flow acoustic model with transformer-based duration predictor and adversarial training; `Vits2Synthesizer`
- [ ] Beat Tracker (`rhythm/beat_tracker.rs`) — Böck 2011 ISMIR: dynamic Bayesian network over onset strength function with tempo prior and inter-beat-interval Markov chain; `BeatTracker`

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
- Tests: 669 passing
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
- [ ] `cp.async` 3-stage pipeline for stride-conv1d on long sequences
- [ ] FP16 Conformer MHSA path with Tensor Cores

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for very long audio sequence (T > 16K)
- [ ] `wgmma.mma_async` for Conformer MHSA QK^T / PV
- [ ] Cluster-launch CTC alpha for very large vocabulary (V >= 5000)
- [ ] Distributed-shared-memory for streaming Conformer left-context cache

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) Conformer inference path
- [ ] FP4 streaming RNN-T decoder experimental path
- [ ] Tensor-Memory (TMEM) staged audio-tile loads

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
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`)
- [ ] Numerical agreement with ESPnet / NeMo reference within 1e-4 relative
- [ ] LibriSpeech CTC WER match for reference small-Conformer checkpoint
- [ ] VoxCeleb speaker EER match for reference x-vector checkpoint

### Implementation Deepening
- [ ] End-to-end `LogMelExtractor` (STFT -> mel -> log) within this crate (currently
      consumed from `oxicuda-signal`)
- [ ] CTC GPU dispatch end-to-end via `ctc_alpha_ptx`
- [ ] Streaming Conformer left-context cache + chunk-attention helper
- [ ] RNN-T / Transducer loss with joint network (alternative to CTC)
- [ ] Beam-search shallow-fusion with external LM scorer interface
- [ ] HiFi-GAN / BigVGAN GAN vocoder generator (link with `WaveNetStack`)

### Benchmark Coverage
- [x] `benches/audio_ops.rs` Criterion harness wired (CPU-side PTX generation +
      Conformer block / WaveNet stack forward)
- [ ] GPU-side throughput vs reference (ESPnet, NeMo) once Linux+NVIDIA harness is
      available
- [ ] Real-time-factor (RTF) measurement for streaming Conformer at various chunk
      sizes
