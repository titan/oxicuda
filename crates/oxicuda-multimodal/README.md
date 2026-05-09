# oxicuda-multimodal

Multi-modal learning primitives for OxiCUDA: cross-modal attention, CLIP alignment, bilinear fusion, BERT/ViT/Conformer/temporal encoders — pure Rust, zero CUDA SDK dependency.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Cross-modal attention**: Multi-head cross-attention between query and key/value sequences from different modalities; self-cross blocks with residual connections and layer normalization
- **Contrastive alignment**: CLIP InfoNCE loss for vision-language alignment; ImageBind triple-modal loss; Image-Text Matching (ITM) BCE head
- **Fusion modules**: Concatenation fusion; MLB/MFB bilinear pooling; attention-gated fusion with learned modality weights
- **Encoders**: BERT text encoder with token and positional embeddings; ViT image encoder with patch projection and CLS token; Conformer audio encoder; Temporal ViT video encoder with frame-level processing
- **Generation heads**: Prefix-LM for captioning; VQA classification head with softmax scoring
- **PTX kernels**: 7 GPU kernels (cross-attention score, modal alignment loss, bilinear pool, temporal pool, token merge, gate fusion, ITM BCE) × 6 SM versions

## Usage

```rust
use oxicuda_multimodal::prelude::*;

let mut rng = LcgRng::new(42);

// Cross-modal attention between text query and image context
let cfg = CrossAttnConfig::tiny();
let d = cfg.d_model;
let weights = CrossAttnWeights::identity(&cfg);
let attn = CrossAttention::with_weights(cfg, weights);

let text_query = vec![0.0_f32; 5 * d];
let image_kv   = vec![0.0_f32; 49 * d];
let output = attn.forward(&text_query, &image_kv, &image_kv, 5, 49).unwrap();
println!("Cross-attention output length: {}", output.len());

// CLIP contrastive loss
let image_feats = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 2 × 4-dim
let text_feats  = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let loss = clip_loss(&image_feats, &text_feats, 2, 4, 0.07).unwrap();
println!("CLIP loss: {loss}");
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-multimodal)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
