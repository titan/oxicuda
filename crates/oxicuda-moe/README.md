# oxicuda-moe

Mixture of Experts primitives for OxiCUDA — Switch Transformer, Top-K routing, Expert Choice, Soft MoE, load balancing.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Routing algorithms**: Top-K gating (with optional noise), Switch Transformer dispatch with capacity factor and token dropping, Expert Choice routing, differentiable Soft MoE dispatch
- **Expert networks**: Standard FFN experts with ReLU/GELU/SiLU activations; SwiGLU experts; grouped expert banks
- **Load balancing**: Load balance auxiliary loss (Switch Transformer), router Z-loss (ST-MoE), routing entropy tracking, per-expert utilization metrics
- **Full MoE layer**: `MoeLayer` composing routing, expert dispatch, FFN computation, and combine with auxiliary loss output
- **PTX kernels**: 7 GPU kernels (top-K gate, expert dispatch, expert FFN, expert combine, load balance loss, router Z-loss, soft MoE dispatch) × 6 SM versions

## Usage

```rust
use oxicuda_moe::prelude::*;

let mut rng = LcgRng::new(42);

// Build a full MoE layer
let cfg = MoeLayerConfig {
    input_dim: 512,
    ffn_dim: 2048,
    n_experts: 8,
    top_k: 2,
    capacity_factor: 1.25,
    load_balance_coef: 0.01,
    router_z_loss_coef: 0.001,
    activation: ExpertActivation::Gelu,
};
let layer = MoeLayer::new(cfg, &mut rng).unwrap();

let n_tokens = 32;
let x = vec![0.0_f32; n_tokens * 512];
let output = layer.forward(&x, n_tokens).unwrap();
println!("Hidden shape: {}, aux_loss: {}", output.hidden.len(), output.aux_loss);
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-moe)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
