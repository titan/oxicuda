# oxicuda-audio

Audio and speech ML architectures for OxiCUDA — pure Rust, zero CUDA SDK dependency.

## What's inside

| Module | Contents |
|--------|----------|
| `features` | `LogMelInput` adapter for `oxicuda-signal` output; CMVN normalisation; delta/delta-delta features |
| `encoder` | Wav2Vec2 CNN feature extractor; Conformer conv module; full Conformer encoder with macaron FFN + rel-pos MHSA |
| `attention` | Relative-position encoding table; multi-head SDPA with rel-pos bias |
| `ctc` | Log-domain CTC forward algorithm; CTC prefix beam search decoder |
| `vocoder` | WaveNet dilated causal residual block; multi-cycle dilated stack with skip aggregation |
| `augment` | SpecAugment time/freq masking; bilinear time warping; enum-dispatched `SpecAugPipeline` |
| `speaker` | Temporal statistics pooling; attentive pooling; x-vector TDNN speaker embeddings (Snyder 2018) |
| `ptx_kernels` | 7 PTX kernel generators × 6 SM versions (75 / 80 / 86 / 90 / 100 / 120) |

## GPU PTX kernels

All 7 kernels are generated as PTX strings for JIT loading — no CUDA SDK at build time.

| Kernel | Purpose |
|--------|---------|
| `stride_conv1d_ptx` | Wav2Vec2 CNN strided feature extraction |
| `dilated_conv1d_ptx` | WaveNet causal dilated convolution (filter + gate) |
| `ctc_alpha_ptx` | CTC forward recursion in log domain |
| `spec_augment_mask_ptx` | SpecAugment time + frequency masking |
| `depthwise_conv1d_ptx` | Conformer causal depthwise convolution |
| `rel_pos_bias_ptx` | Relative-position bias matrix construction |
| `stats_pool_ptx` | Temporal mean + std pooling (warp-shuffle) |

## Usage

```rust
use oxicuda_audio::prelude::*;

// Build a Conformer-tiny encoder
let cfg = ConformerConfig::tiny();
let mut rng = LcgRng::new(42);
let encoder = ConformerEncoder::new(cfg.clone(), &mut rng)?;

// Forward pass: [T, D] → [T, D]
let t = 100;
let x = vec![0.0f32; t * cfg.embed_dim];
let out = encoder.forward(&x, t)?;

// CTC beam search decoding
let hyps = ctc_beam_search(&log_probs, t, vocab, blank, beam_width)?;
```

## Design

- Pure CPU-side reference implementations + GPU PTX string generators
- All sequences in time-major `[T, D]` layout; batch dim explicit at boundaries
- No cross-crate dependencies on `oxicuda-signal` or `oxicuda-dnn` — leaf crate
- `LogMelInput` documents the `oxicuda-signal` producer interface

## Relationship to other crates

- **`oxicuda-signal`** — produces log-mel spectrograms consumed via `LogMelInput`
- **`oxicuda-dnn`** — shares attention concepts; `oxicuda-audio` ships its own CPU reference
- **`oxicuda-infer`** — generic LM beam search; CTC beam search is a separate implementation
