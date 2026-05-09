# oxicuda-recsys

Recommender-system primitives for OxiCUDA — ALS/BPR/NMF, NCF, Two-Tower, DeepFM/AutoInt, SASRec/BERT4Rec, LightGCN/NGCF, MMoE/PLE/ESMM, negative sampling, ranking metrics.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Collaborative filtering**: ALS (Alternating Least Squares), BPR (Bayesian Personalized Ranking), NMF for matrix factorization
- **Deep models**: NCF (Neural Collaborative Filtering), Two-Tower, DeepFM, Wide & Deep architectures
- **Sequential models**: SASRec self-attentive recommendation
- **Graph-based**: LightGCN neighborhood propagation over user-item interaction graphs
- **Ranking metrics**: NDCG@K and Precision@K evaluation; uniform negative sampling
- **PTX kernels**: GPU kernel strings for ALS step, BPR gradient, embedding lookup, dot scoring, softmax top-K, negative sampling, and LightGCN propagation (7 kernels × 6 SM versions)

## Usage

```rust
use oxicuda_recsys::{
    factorization::als::Als,
    handle::LcgRng,
    metrics::recsys_metrics::{ndcg_at_k, precision_at_k},
};
use std::collections::HashSet;

let mut rng = LcgRng::new(42);
let mut model = Als::new(/*n_users=*/100, /*n_items=*/200, /*dim=*/16, /*lambda=*/0.01, &mut rng).unwrap();

let interactions = vec![(0usize, 0usize, 1.0_f32), (0, 5, 2.0), (1, 3, 1.0)];
model.fit(&interactions, /*n_iter=*/10).unwrap();

let score = model.score(0, 5).unwrap();
println!("User 0 → Item 5 score: {score}");

let recommended: Vec<usize> = vec![0, 5, 3, 8, 2];
let relevant: HashSet<usize> = vec![0, 5].into_iter().collect();
let ndcg = ndcg_at_k(&recommended, &relevant, 5);
println!("NDCG@5: {ndcg}");
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-recsys)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
