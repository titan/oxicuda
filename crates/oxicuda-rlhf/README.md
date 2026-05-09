# oxicuda-rlhf

RLHF and alignment algorithm primitives for OxiCUDA — DPO, IPO, KTO, ORPO, SimPO, reward modelling, PPO-RLHF.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Preference optimization losses**: DPO (Direct Preference Optimization), IPO (Identity Preference Optimization), KTO (Kahneman-Tversky Optimization), ORPO (Odds Ratio Preference Optimization), SimPO (length-normalized margin loss)
- **Reward modelling**: Bradley-Terry reward loss for pairwise preference learning; `RewardModel` with MLP head; online reward normalization (running mean/variance)
- **PPO-RLHF pipeline**: PPO surrogate loss with clipping; KL-divergence controller (adaptive β); rollout buffer for trajectories
- **SFT loss**: Masked cross-entropy for supervised fine-tuning; per-token masking support
- **Alignment metrics**: Win rate, reward gap, perplexity, KL from reference, per-batch alignment metric aggregation
- **PTX kernels**: 7 GPU kernels (BT reward loss, DPO loss, IPO loss, KTO loss, ORPO odds, RLHF KL, SFT mask) × 6 SM versions

## Usage

```rust
use oxicuda_rlhf::prelude::*;

// DPO loss for a preference batch
let batch = PairBatch::new(
    vec![-1.0_f32, -1.5],   // chosen log-probs (policy)
    vec![-2.5_f32, -3.0],   // rejected log-probs (policy)
    vec![-1.1_f32, -1.6],   // chosen log-probs (reference)
    vec![-2.6_f32, -3.1],   // rejected log-probs (reference)
).unwrap();

let cfg = DpoConfig { beta: 0.1 };
let loss = dpo_loss(&batch, &cfg).unwrap();
println!("DPO loss: {loss}");

// Reward normalization
let mut norm = RewardNormalizer::new();
for r in [1.0_f32, 2.0, 3.0, 4.0, 5.0] { norm.update(r); }
let normalized = norm.normalize(3.0).unwrap();
println!("Normalized reward: {normalized}"); // ≈ 0.0
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-rlhf)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
