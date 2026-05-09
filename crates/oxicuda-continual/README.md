# oxicuda-continual

Continual and lifelong learning primitives for OxiCUDA: EWC/SI/MAS regularization, PackNet/Piggyback/Progressive architectures, Experience Replay/GEM/A-GEM, DER++, forgetting metrics — pure Rust, zero CUDA SDK dependency.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Regularization methods**: EWC (Elastic Weight Consolidation) with empirical Fisher diagonal; Synaptic Intelligence (SI) with path-integral importance; MAS (Memory-Aware Synapses) with gradient-magnitude importance
- **Architecture-based methods**: PackNet L1 pruning with per-task binary masks; Piggyback real-valued masks binarized per task; Progressive Neural Networks with lateral connections between columns
- **Replay methods**: Experience Replay with reservoir sampling; GEM and A-GEM gradient projection to prevent forgetting; DER++ (Dark Experience Replay) with knowledge distillation on stored logits
- **Metrics**: Average Forgetting, Backward Transfer (BWT), Forward Transfer (FWT), Intransigence, Plasticity via accuracy matrices
- **PTX kernels**: 7 GPU kernels (EWC penalty, Fisher diagonal, gradient project, mask apply, SI omega update, logit distill, replay sample) × 6 SM versions

## Usage

```rust
use oxicuda_continual::prelude::*;

// EWC regularization
let anchor_params = vec![0.5_f32, -1.0, 2.0];
let fisher = FisherDiag { params: vec![1.0_f32, 2.0, 0.5] };
let mut reg = EwcRegularizer::new();
ewc_add_task(&mut reg, anchor_params.clone(), fisher);

let current_params = vec![0.6_f32, -0.9, 2.1]; // slight drift
let cfg = EwcConfig { lambda: 5000.0, n_tasks: 1 };
let penalty = ewc_loss(&current_params, &reg, &cfg).unwrap();
println!("EWC penalty: {penalty}");

// Experience Replay with reservoir sampling
let mut rng = LcgRng::new(0);
let mut buf = er_buffer_new(500).unwrap();
er_add(&mut buf, vec![0.1_f32; 32], 0_u32, &mut rng);
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-continual)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
