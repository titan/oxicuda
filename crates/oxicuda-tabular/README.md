# oxicuda-tabular

Tabular deep learning primitives for OxiCUDA — TabNet, FT-Transformer, SAINT, NODE, sparsemax, quantile normalization.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Attention-based models**: TabNet with step-wise attentive feature selection (sparsemax masks, GLU blocks, batch normalization); SAINT inter-sample attention layer; FT-Transformer with feature tokenization, multi-head self-attention, and feed-forward layers
- **Tree-based differentiable model**: NODE (Neural Oblivious Decision Ensembles) with soft oblivious trees and response averaging
- **Sparse activations**: Sparsemax (Euclidean projection onto the probability simplex) and Entmax-1.5 as drop-in softmax replacements
- **Preprocessing**: QuantileNormalizer (rank-based transform to [0,1]); StandardNormalizer (z-score); MinMaxNormalizer; FeatureEmbedder for mixed continuous/categorical inputs
- **Evaluation metrics**: AUC-ROC (trapezoidal), AUC-PR, binary accuracy, multi-class accuracy, RMSE, MAE, F1-at-threshold
- **PTX kernels**: 7 GPU kernels (sparsemax, feature tokenize, TabNet step attention, inter-sample attention, NODE tree eval, quantile norm, AUC-ROC) × 6 SM versions

## Usage

```rust
use oxicuda_tabular::prelude::*;

let mut rng = LcgRng::new(42);

// TabNet forward pass
let cfg = TabNetConfig {
    n_features: 20,
    n_d: 16,
    n_a: 16,
    n_steps: 5,
    gamma: 1.5,
    n_classes: 2,
};
let layer = TabNetLayer::new(cfg, &mut rng).unwrap();
let x = vec![0.5_f32; 20];
let (logits, masks) = layer.forward(&x).unwrap();
println!("Logits: {logits:?}");

// Quantile normalization
let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
let (norm, _) = QuantileNormalizer::fit_transform(&data, 100, 1).unwrap();
let t = norm.transform(&[75.0_f32]).unwrap();
println!("Quantile-normalized 75: {}", t[0]); // ≈ 0.75
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-tabular)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
