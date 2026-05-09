# oxicuda-anomaly

Anomaly detection primitives for OxiCUDA — DeepSVDD, AE/VAE reconstruction, LOF, COPOD, isolation scoring, statistical methods, ensemble.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Deep learning detectors**: DeepSVDD (one-class deep SVDD with hypersphere center estimation); Autoencoder reconstruction-error scorer; VAE anomaly score via ELBO
- **Distance and density methods**: LOF (Local Outlier Factor) with k-nearest-neighbor reachability distances; COPOD (Copula-based Outlier Detection) via empirical CDFs; Mahalanobis distance detector
- **Tree-based isolation**: Random-projection isolation forest scoring with c-factor normalization
- **Statistical baselines**: MAD (Median Absolute Deviation) detector; Z-score detector; percentile threshold computation
- **Ensemble scoring**: Average, Maximum, and Weighted ensemble combination with min-max normalization across detectors
- **PTX kernels**: 7 GPU kernels (DeepSVDD loss, reconstruction score, LOF reachability distance, COPOD ECDF, Mahalanobis distance, isolation forest score, ensemble normalize) × 6 SM versions

## Usage

```rust
use oxicuda_anomaly::prelude::*;

let mut rng = LcgRng::new(42);

// Train and score with an Autoencoder
let cfg = AeConfig {
    encoder_dims: vec![16, 8, 4],
    decoder_dims: vec![4, 8, 16],
};
let ae = AutoencoderAnomaly::new(cfg, &mut rng).unwrap();
let score = ae.score(&[0.3_f32; 16]).unwrap();
println!("Reconstruction anomaly score: {score}");

// Statistical Z-score detector
let mut det = ZScoreDetector::new();
let train: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
det.fit(&train, 100, 1).unwrap();
let outlier_score = det.score(&[50.0_f32]).unwrap();
println!("Z-score anomaly score: {outlier_score}");
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-anomaly)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
