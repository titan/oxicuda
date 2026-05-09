# oxicuda-geometry3d

3D geometry, point-cloud, mesh, and Gaussian-splatting primitives for OxiCUDA: FPS/kNN/KD-tree, PointNet/PointNet++/DGCNN/Point-Transformer, sparse 3D conv, chamfer/EMD, ICP, 3D Gaussian splatting — pure Rust, zero CUDA SDK dependency.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Point-cloud sampling and search**: Farthest Point Sampling (FPS), random sampling, voxel downsampling; kNN, ball query, KD-tree with exact nearest-neighbor queries
- **Deep architectures**: PointNet global max-pooling classifier; PointNet++ Set Abstraction and Feature Propagation with local grouping; DGCNN EdgeConv graph convolution; Point Transformer with vector self-attention
- **3D Gaussian Splatting**: `Gaussian3d` primitives with covariance representation; pinhole-camera projection; tile-based alpha-compositing rasterizer
- **Mesh and distance metrics**: Chamfer Distance (bidirectional, with gradient); Earth Mover's Distance via Sinkhorn; surface normal estimation from point neighborhoods
- **Registration and transforms**: ICP (Iterative Closest Point) with configurable convergence tolerance; rigid body transforms; quaternion operations
- **PTX kernels**: 7 GPU kernels (FPS, ball query, gather points, voxelize, Chamfer distance, Gaussian project, SH eval) × 6 SM versions

## Usage

```rust
use oxicuda_geometry3d::prelude::*;

let mut rng = LcgRng::new(42);

// Farthest Point Sampling: select 64 from 1024 points
let points: Vec<f32> = (0..1024 * 3).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
let selected = farthest_point_sample(&points, 1024, 64).unwrap();
println!("FPS selected {} point indices", selected.len());

// PointNet classification
let cfg = PointNetConfig { n_points: 64, n_classes: 10 };
let net = PointNet::new(cfg, &mut rng);
let logits = net.forward(&points[..64 * 3]).unwrap();
println!("Class logits: {logits:?}");

// Chamfer Distance between two identical clouds
let cd = chamfer_distance(&points, 1024, &points, 1024).unwrap();
println!("Chamfer(A,A) = {cd}"); // 0.0
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-geometry3d)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
