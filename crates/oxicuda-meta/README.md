# oxicuda-meta

Meta-learning algorithm primitives for OxiCUDA — MAML, FOMAML, ANIL, Reptile, ProtoNet, MatchingNet, RelationNet.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Gradient-based meta-learning**: MAML (Model-Agnostic Meta-Learning) with inner-loop SGD and meta-gradient computation; FOMAML (first-order approximation); ANIL (only head adapted in inner loop); Reptile (interpolation toward task minimum)
- **Metric-based meta-learning**: ProtoNet (prototype nearest-centroid classifier); MatchingNet (cosine attention over support set); RelationNet (learned relation score between query and support features)
- **Episode sampling**: Configurable N-way K-shot episode builder from class-labeled datasets
- **Evaluation metrics**: Per-episode accuracy, mean and 95% confidence interval over episodes
- **PTX kernels**: 7 GPU kernels (inner SGD, Reptile update, proto distance, cosine similarity, relation score, meta gradient accumulation, episode sample) × 6 SM versions

## Usage

```rust
use oxicuda_meta::prelude::*;

let n_classes = 3;
let feat_dim  = 64;

// ProtoNet: compute class prototypes and predict
let support_x: Vec<f32> = vec![/* n_classes * k_shot * feat_dim floats */];
let support_y: Vec<u32>  = vec![0, 0, 1, 1, 2, 2]; // 3-way 2-shot
let protos = compute_prototypes(&support_x, &support_y, n_classes, 2, feat_dim).unwrap();

let query_x: Vec<f32> = vec![/* query floats */];
let preds = proto_predict(&query_x, &protos, n_classes, feat_dim).unwrap();
println!("Predicted classes: {preds:?}");

// Reptile meta-update toward a task
let params = vec![0.0_f32; 10];
let cfg = ReptileConfig { inner_lr: 0.1, n_inner_steps: 5, step_size: 0.5 };
let task_data = vec![(vec![1.0_f32; feat_dim], vec![0_u32])];
let updated = reptile_update(&params, &task_data, n_classes, feat_dim, &cfg).unwrap();
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-meta)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
