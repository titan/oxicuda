# oxicuda-nerf

Neural Radiance Fields and neural rendering primitives for OxiCUDA — NeRF, Instant-NGP hash grid, Mip-NeRF, TensoRF, volume rendering.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Encodings**: Multi-frequency positional encoding, Instant-NGP multi-level hash grid (trilinear interpolation), Mip-NeRF integrated positional encoding
- **Fields**: TensoRF low-rank tensor radiance field, Instant-NGP hash field for density and color queries
- **Networks**: NeRF MLP and TinyNeRF with configurable depth and width
- **Rendering**: Pinhole camera ray generation, stratified and importance sampling, occupancy grid, full volume rendering pipeline with alpha compositing
- **Metrics**: PSNR and MSE image quality evaluation
- **PTX kernels**: 7 GPU kernels (positional encoding, volume render, hash grid lookup, ray march, SH eval, occupancy update, importance resample) × 6 SM versions

## Usage

```rust
use oxicuda_nerf::prelude::*;

let mut rng = LcgRng::new(42);

// Build a hash grid encoder
let cfg = HashGridConfig {
    n_levels: 16,
    n_features_per_level: 2,
    log2_hashmap_size: 19,
    base_resolution: 16,
    max_resolution: 2048,
};
let grid = HashGrid::new(cfg, &mut rng).unwrap();
let features = grid.query([0.3, 0.7, 0.5]).unwrap();

// Volume render a ray
let sigma = vec![0.1_f32; 64];
let color = vec![0.5_f32; 64 * 3];
let t: Vec<f32> = (0..64).map(|i| 0.1 + i as f32 * 0.05).collect();
let result = volume_render(&sigma, &color, &t).unwrap();
println!("RGB: {:?}, opacity: {}", result.rgb, result.opacity);
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-nerf)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
