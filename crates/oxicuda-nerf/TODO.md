# oxicuda-nerf TODO

Neural Radiance Fields and neural rendering primitives (NeRF, Instant-NGP, Mip-NeRF, TensoRF) for OxiCUDA. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.34).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 3,898 SLoC (23 source files + 1 benches file) -- Coverage: NeRF / Instant-NGP / Mip-NeRF / TensoRF reference pipeline**

Current implementation covers NeRF positional encoding (sin/cos with L frequency levels, configurable include_input), Instant-NGP multi-resolution hash grid (L levels, T buckets, F features per entry, spatial hashing with primes pi2=2654435761, pi3=805459861, trilinear interpolation over 8 corners), Mip-NeRF integrated positional encoding (Gaussian attenuation `exp(-omega^2 * sigma^2 / 2)` for anti-aliasing), TensoRF CP decomposition (rank-R factored density and color field with 1D axis interpolation), volume rendering (alpha compositing `alpha_i = 1 - exp(-sigma_i * delta_i)`, transmittance, early termination at `T < 1e-4`), stratified sampling, importance resampling (inverse-CDF), pinhole camera ray generation (c2w 3x4 matrix), occupancy-grid acceleration, and PSNR/MSE image-quality metrics.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `NerfError` (16 variants: DimensionMismatch, EmptyInput, InvalidFreqLevels, InvalidHashConfig, NanEncountered, InvalidBounds, InvalidSampleCount, ZeroRayDirection, InvalidCameraIntrinsics, InvalidGridResolution, HashLevelOutOfRange, InvalidFeatureDim, TensorDecompError, VolumeRenderError, InvalidEncoding, Internal), `NerfResult<T>`
- [x] `handle.rs` -- `SmVersion` (Sm75/80/86/90/100/120), `LcgRng` (Knuth MMIX 64-bit LCG), `NerfHandle::default_handle()` (Sm80, device 0, seed 42)

#### Camera
- [x] `camera/pinhole.rs` -- re-export of `PinholeCamera`; intrinsics (fx, fy, cx, cy), image size, and `ray_through_pixel(u, v, c2w)`

#### Positional Encodings
- [x] `encoding/positional.rs` -- `positional_encode`, `PosEncConfig`; `gamma(p) = [sin(2^k * pi * p), cos(2^k * pi * p)]` for k = 0..L-1 per dimension; optional raw input concatenation
- [x] `encoding/hash_grid.rs` -- `HashGrid`, `HashGridConfig`; multi-resolution hash with trilinear lerp; spatial hashing with primes pi2 = 2654435761, pi3 = 805459861; `query()` returns `[n_levels * F]`, `query_batch()` for tensor inputs
- [x] `encoding/integrated_pe.rs` -- `integrated_pe`, `IpeConfig`; Mip-NeRF IPE: `sin(omega * mu) * exp(-omega^2 * sigma^2 / 2)`, `cos(omega * mu) * exp(-omega^2 * sigma^2 / 2)`

#### Fields
- [x] `field/tensorf.rs` -- `TensorRf`, `TensorRfConfig`; CP decomposition: rank-R factored density (ReLU) plus color; `query_density()`, `query_color()`
- [x] `field/hash_field.rs` -- `HashField`; Instant-NGP-style HashGrid + 2-layer MLP decoder to `(sigma, color_feat)`

#### Networks
- [x] `network/nerf_mlp.rs` -- `NerfMlp`, `NerfMlpConfig`; 8-layer ResNet MLP with skip connection at layer 4; sigma head (ReLU); color head (Sigmoid); batch forward
- [x] `network/tiny_nerf.rs` -- `TinyNerf`; compact 4-layer MLP suitable for tests and small-scene experiments

#### Rendering
- [x] `rendering/ray.rs` -- `Ray`, `PinholeCamera`; `Ray::at(t)`, `Ray::normalized()`; camera `ray_through_pixel(u, v, c2w)`, `generate_rays(c2w)`
- [x] `rendering/sampling.rs` -- `stratified_sample`, `importance_sample`, `merge_samples`; hierarchical NeRF coarse-to-fine sampling
- [x] `rendering/volume_render.rs` -- `volume_render`, `volume_render_batch`, `RenderResult`; alpha compositing with depth and opacity output, early termination
- [x] `rendering/occupancy.rs` -- `OccupancyGrid`; `resolution^3` boolean grid; `is_occupied_world()`, `update_from_density()`, `march_ray_occupied()`

#### Metrics
- [x] `metrics/image_quality.rs` -- `psnr()`, `mse_image()`, `compute_image_metrics()` -> `ImageMetrics`

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 7 GPU kernels x 6 SM versions (75/80/86/90/100/120):
  - [x] `pe_kernel` -- sin/cos frequency encoding of coordinate batch
  - [x] `volume_render_kernel` -- single-ray alpha compositing with transmittance cutoff
  - [x] `hash_grid_kernel` -- multi-resolution spatial hash + trilinear interpolation
  - [x] `ray_march_kernel` -- stratified sample generation along ray
  - [x] `sh_eval_nerf_kernel` -- spherical-harmonics basis evaluation for L = 0..3 (16 coefficients, view-dependent color)
  - [x] `occupancy_update_kernel` -- threshold density -> boolean occupancy grid
  - [x] `importance_resample_kernel` -- inverse-CDF resampling from coarse weight histogram

#### Integration Tests
- [x] 12 e2e tests (lib.rs): positional encoding shape, deterministic encoding, hash grid query shape, trilinear corner queries differ, volume render empty scene -> zero opacity, opaque first sample -> full color, stratified sample count, importance sample count, TensoRF density non-negative, TinyNerf forward finite, PSNR identity, PTX kernels x 6 SM versions

#### Benchmarks
- [x] `benches/nerf_ops.rs` -- 7 PTX kernel groups x 4 SM versions plus 6 algorithm benches: pos_enc_1024pts, hash_grid_batch_1024, volume_render_64rays, stratified_sample_128, tensorf_density_1024

### Future Enhancements

#### P0 -- Critical Algorithmic Coverage
- [ ] Trainable hash grid with backward pass -- gradient accumulation into hash table entries (currently inference only)
- [x] Proposal network (Mip-NeRF 360 style) -- learned density estimator for importance sampling
- [x] Contraction (Mip-NeRF 360) -- unbounded scene parametrization mapping infinity to unit sphere
- [x] Distortion loss (Mip-NeRF 360) -- regularises ray weight distribution

#### P1 -- Important Features
- [x] Spherical harmonics directional encoding (L = 0..4) for view-dependent color (currently L = 0..3 in PTX kernel)
- [x] Plenoxel grid (sparse voxel SH coefficients) field (field/plenoxel.rs -- Yu 2022; voxel grid with density + SH coeffs, trilinear interpolation, SH view-dir color eval, no MLP; reuses encoding::spherical_harmonics)
- [ ] DVGO / TensoRF-VM variant -- vector-matrix tensor decomposition
- [x] K-Planes (Fridovich-Keil et al. 2023) -- factorised hyperplane encoding for 4D dynamic scenes (field/kplanes.rs -- 3 factorized coordinate planes xy/xz/yz, bilinear interpolation + Hadamard combine -> density/color heads, SH view-dependent color)
- [ ] InstantAvatar / human-NeRF skeleton-driven canonical-space mapping
- [ ] LPIPS metric (perceptual image quality) alongside PSNR / MSE

#### P2 -- Advanced / Research
- [ ] 3D Gaussian Splatting differentiable rasterizer (Kerbl et al. 2023)
- [ ] BakedSDF / NeuS isosurface extraction for mesh export
- [ ] Generative NeRF (GIRAFFE / pi-GAN) latent-code conditioning
- [ ] Block-NeRF / Mega-NeRF scene partitioning for very large scenes
- [ ] Ref-NeRF reflective surface separation (specular + diffuse)
- [x] NeRF-W (in-the-wild) appearance embeddings for varying illumination (`network/nerf_w.rs` -- Martin-Brualla et al. 2021 CVPR; per-image appearance + transient embeddings, β-uncertainty head with softplus + beta_min, Gaussian NLL)
- [x] Sparse voxel octree acceleration (NSVF-style) (`rendering/sparse_voxel_octree.rs` -- Liu et al. 2020 NeurIPS; Aabb + slab-test, 8-octant recursive subdivision over Z·Y·X grid, front-to-back ray traversal, homogeneous-block pruning)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none) | Standalone primitives crate | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

## Quality Status

- Tests: 62+ passing (12 e2e in lib.rs + module unit tests)
- All production code uses `Result` / `Option` (no `unwrap()` outside tests)
- `clippy::all` warnings: 0
- `missing_docs` warnings: 0
- Files: 23 source `.rs` files, all under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns `UnsupportedPlatform` at runtime

## Performance Targets

Representative shapes for synthetic Blender / Tanks-and-Temples / Mip-NeRF 360 datasets.

| Operation | Configuration | Priority |
|-----------|---------------|----------|
| `pe_kernel` | 1024 points, L = 10, input_dim = 3 | P0 |
| `hash_grid_kernel` | 1024 points, 16 levels, F = 2, T = 2^19 | P0 |
| `volume_render_kernel` | 64 samples/ray, 4096-8192 rays/batch | P0 |
| `ray_march_kernel` | 128 stratified samples per ray | P0 |
| `sh_eval_nerf_kernel` | L = 0..3 (16 coefs), batch 4096 | P1 |
| `occupancy_update_kernel` | 128^3 grid | P1 |
| `importance_resample_kernel` | 64 -> 128 hierarchical samples | P1 |

Target: hash-grid + volume-render forward latency comparable to Instant-NGP CUDA reference on `sm_80+` for 800x800 NeRF synthetic scenes.

## Estimation vs Actual

| Metric | Description | Actual |
|--------|-------------|--------|
| Files | source `.rs` files under `src/` | 23 |
| SLoC | code lines (tokei) | ~3,898 |
| Tests | e2e + unit | 62+ |
| Coverage | PTX kernels x SM versions | 7 x 6 = 42 entry-point variants |

The current implementation provides a reference NeRF / Instant-NGP / Mip-NeRF / TensoRF inference pipeline. P0/P1 items extend toward Mip-NeRF 360, K-Planes, and full trainability with gradient kernels; P2 items cover Gaussian Splatting and generative variants.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX kernels generated for all 7 entry points on `sm_75`
- [ ] Tex-fetch fallback path for hash-grid trilinear interpolation verified on Turing

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels generated for `sm_80`, `sm_86`
- [ ] `cp.async` staging of hash-grid entries for high-resolution levels
- [ ] Tensor Core path for MLP decoder (8x16-dim hidden) on `sm_80+`

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [x] PTX kernels generated for `sm_90`, `sm_100`, `sm_120`
- [ ] TMA-based hash-table staging for very deep multi-resolution grids
- [ ] Distributed shared-memory cluster reduction for batch volume rendering

---

## Deepening Opportunities

> Items marked `[x]` in the Completed section represent API and CPU-simulation coverage. The opportunities below close gaps toward production neural-rendering deployment.

### Verification Gaps
- [x] PE shape, determinism, and round-trip frequency expansion verified
- [x] Volume render empty scene produces zero opacity (sanity test)
- [x] Volume render opaque first sample produces full opacity and pixel color (sanity test)
- [x] PSNR identity-image test (`psnr(x, x) -> inf`)
- [x] PTX entry points validated for `.version`, `.visible .entry`, kernel name, and SM target across all 6 SM versions
- [ ] End-to-end NeRF synthetic scene PSNR reproduction (Lego / Mic / Chair targets)
- [ ] Hash-grid GPU kernel correctness vs CPU simulation on `sm_80+`

### Implementation Deepening
- [x] TensoRF density non-negative for arbitrary 3D query points
- [x] TinyNerf forward returns finite sigma and `(0, 1)`-clipped RGB
- [x] Stratified and importance sampling produce correct `n_samples` and obey `[t_near, t_far]` bounds
- [ ] Trainable hash-grid backward pass (gradient accumulation into table entries)
- [ ] Proposal network + distortion loss for Mip-NeRF 360
- [ ] Multi-resolution unbounded-scene contraction mapping
- [ ] Multi-GPU NeRF training with ray sharding

## Notes

- All position queries normalise to `[0, 1]^3` for hash-grid lookups; world bounds are handled at the camera/sampling layer
- `volume_render` returns `RenderResult { rgb, depth, opacity }` per ray for downstream loss computation
- `OccupancyGrid::march_ray_occupied()` accelerates ray marching by skipping empty cells via bit-grid lookup
- All PTX kernels share a unified `.version` / `.target sm_X` / `.address_size 64` header consistent with the rest of the OxiCUDA ecosystem
- Volume rendering uses the canonical NeRF alpha compositing formula with early termination when `T < 1e-4`
