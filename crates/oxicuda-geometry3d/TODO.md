# oxicuda-geometry3d TODO

Pure-Rust 3D geometry and point-cloud deep-learning library covering sampling
(FPS / random / voxel-downsample), neighbourhood queries (kNN / ball-query /
KD-tree), point feature operations (gather / group / interpolate),
architectures (PointNet / PointNet++ / DGCNN / Point-Transformer), voxel ops
(voxelization / sparse 3D conv), mesh distances (Chamfer / EMD-Sinkhorn /
normal PCA), 3D Gaussian splatting primitives, and SE(3) / quaternion / ICP
transforms. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.30).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 14,490 SLoC (57 files)** -- 328 unit tests + 15 E2E integration tests

The crate covers the full point-cloud + 3D Gaussian splatting + classical
geometry pipeline. CPU paths are simulation-grade for unit testing; PTX
kernels target NVIDIA SM 7.5 through SM 12.0. The crate is `forbid(unsafe_code)`.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `Geom3dError` (15 variants: DimensionMismatch,
      EmptyPointCloud, InvalidPointDim, InvalidK, InvalidRadius,
      InvalidVoxelSize, InvalidSampleCount, InvalidShCoefficients,
      InvalidQuaternion, IcpDidNotConverge, EmdDidNotConverge, InvalidTopology,
      NanEncountered, BatchSizeMismatch, Internal); `Geom3dResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG,
      `next_usize`, `next_f32`), `Geom3dHandle::default_handle()` (SM 8.0,
      device 0, seed 42)
- [x] `lib.rs` -- module exports + prelude + 12 E2E integration tests;
      `#![forbid(unsafe_code)]`

#### PTX Kernels (7 kernels x 6 SM versions = 42 generators)
- [x] `ptx_kernels.rs::farthest_point_sample_ptx` -- per-point distance update
      + `atom.global.max.f32` argmax reduce
- [x] `ptx_kernels.rs::ball_query_ptx` -- radius test d^2 < r^2 + bounded
      atomic counter per query
- [x] `ptx_kernels.rs::gather_points_ptx` -- indexed feature gather with
      `mul.wide.u32` 64-bit offset
- [x] `ptx_kernels.rs::voxelize_ptx` -- voxel index from (p - o) / v,
      `atom.global.add.f32` per channel + count
- [x] `ptx_kernels.rs::chamfer_distance_ptx` -- tiled pairwise distance,
      warp-min reduce, `atom.global.min.f32`
- [x] `ptx_kernels.rs::gaussian_project_ptx` -- 3D -> 2D Jacobian
      J * Sigma * J^T via `fma.rn.f32`
- [x] `ptx_kernels.rs::sh_eval_ptx` -- spherical harmonic evaluation L = 0..2
      with precomputed constants as f32-hex literals
- [x] `ptx_kernels.rs::f32_hex` -- f32 to 0F-prefixed hex literal helper

#### Sampling (sampling/)
- [x] `farthest_point_sample.rs::farthest_point_sample` -- deterministic FPS
      with idx[0] = 0 initialisation; dist[i] = min(dist[i], d^2_to_last);
      argmax as next seed
- [x] `random_sample.rs::random_sample` -- partial Fisher-Yates without
      replacement via `LcgRng`
- [x] `voxel_downsample.rs::voxel_downsample` -- HashMap voxel grid, emit
      centroids + first-original-index per bucket; sort for determinism

#### Neighbourhood (neighborhood/)
- [x] `knn.rs::knn` -- brute-force k-NN per query; returns (indices, sq_dists)
      row-major [nq x k]
- [x] `ball_query.rs::ball_query` -- radius-limited search;
      `usize::MAX` sentinel for empty slots; returns (indices, counts)
      [nq x k_max]
- [x] `kd_tree.rs::KdTree` -- recursive median-split build; `nearest`, `knn`,
      `radius_search`; best-first traversal with AABB pruning

#### Point Feature Ops (pointops/)
- [x] `gather_points.rs::gather_points` -- [n x c] + [k] indices ->
      [k x c] with bounds check
- [x] `group_features.rs::group_features` -- [n x c] + [k x s] indices ->
      [k x s x c]
- [x] `interp_features.rs::interp_features` -- 3-NN
      inverse-distance-weighted feature interpolation (eps = 1e-10)

#### Architectures (arch/)
- [x] `pointnet.rs::PointNet` -- T-Net (3 x 3 transform, identity-init) +
      shared MLP [3 -> 64 -> 128 -> 1024] + global max-pool + FC head ->
      class logits; `PointNetConfig`
- [x] `pointnet_pp.rs::SetAbstraction` / `FeaturePropagation` -- PointNet++:
      FPS -> ball-query -> gather -> MLP -> max-pool; upsample via 3-NN
      interpolate + skip concat + MLP; `SetAbstractionConfig`
- [x] `dgcnn.rs::EdgeConv` -- DGCNN: dynamic kNN graph in feature space;
      edge feat = concat(x_i, x_j - x_i); MLP + max-pool; `EdgeConvConfig`
- [x] `point_transformer.rs::PointTransformerLayer` -- vector self-attention
      with relative-position MLP encoding delta_ij; element-wise attention
      weights; `PointTransformerConfig`

#### Voxel Ops (voxel/)
- [x] `voxelize.rs::VoxelGrid` / `VoxelPoolMode` -- scatter points into grid
      with Mean / Max / Sum pooling; `occupied_centroids()` emission
- [x] `sparse_conv3d.rs::SparseConv3d` / `SparseTensor` -- Minkowski-style
      sparse 3D convolution; HashMap output accumulation; configurable kernel
      size; `SparseConv3dConfig`

#### Mesh Distances (mesh/)
- [x] `chamfer_distance.rs::chamfer_distance` / `chamfer_distance_grad` --
      bidirectional CD with gradient 2 * (a - b_nearest) / |A|
- [x] `earth_movers.rs::earth_movers_distance` / `SinkhornConfig` --
      entropy-regularised OT via log-domain Sinkhorn (clamp +/-50,
      epsilon >= 1e-3)
- [x] `normal_estimate.rs::estimate_normals` -- per-point PCA normals via
      3 x 3 covariance smallest-eigenvector; +z orientation
- [x] `delaunay3d.rs::tetrahedralize` / `Delaunay3d` -- incremental
      Bowyer-Watson 3D Delaunay; f64 `orient3d` + lifted `in_sphere`
      predicates (relative scaled eps, co-spherical treated as outside);
      super-tetrahedron seeding; cavity re-triangulation; `circumcenter`
      (3x3 Cramer solve), `convex_hull_faces`, `tet_volume` / `total_volume`
- [x] `ray_triangle.rs` -- Moller-Trumbore `ray_triangle_intersect`,
      Ericson Voronoi-region `closest_point_on_triangle`, slab
      `ray_aabb_intersect`, and mesh reductions `ray_mesh_intersect` /
      `closest_point_on_mesh`
- [x] `curvature.rs::discrete_curvature` / `VertexCurvature` -- Meyer 2003
      discrete operators: angle-defect Gaussian, cotangent Laplace-Beltrami
      mean (mixed Voronoi/barycentric area), principal k1/k2; plus public
      `icosphere` test-oracle mesh generator

#### Gaussian Splatting (gaussian/)
- [x] `gaussian.rs::Gaussian3d` -- wxyz quaternion, log-scale, pre-sigmoid
      opacity, SH coefficients; `covariance3d()`, `sh_color()`,
      `Gaussian3d::new_unit()`
- [x] `project.rs::project_gaussian` / `ProjectedGaussian` /
      `CameraIntrinsics` -- view-space projection, 2 x 2 covariance via
      Jacobian, low-pass Sigma_2d += 0.3 * I
- [x] `rasterize.rs::rasterize_gaussians` / `RasterConfig` -- depth-sort,
      3-sigma AABB, alpha-composite front-to-back; T < 1e-4 early termination

#### Transforms (transform/)
- [x] `rigid.rs::RigidTransform` -- SE(3): rotation matrix + translation;
      Rodrigues axis-angle; `compose`, `inverse`, `apply`
- [x] `quaternion.rs::Quat` -- wxyz quaternion; `mul`, `conjugate`, to / from
      rotation matrix; slerp with shortest-path sign-flip and lerp fallback
- [x] `icp.rs::icp` / `IcpConfig` / `IcpResult` -- point-to-point ICP via
      3 x 3 Jacobi SVD, sign-correct det(V * U^T), KD-tree correspondences

#### Integration Tests (lib.rs e2e_tests)
- [x] `e2e_fps_selects_m_distinct_points` -- FPS returns m unique indices
- [x] `e2e_pointnet_forward_valid_logits` -- output shape == n_classes and
      finite
- [x] `e2e_set_abstraction_reduces_points` -- output point count == npoint
- [x] `e2e_dgcnn_output_shape` -- EdgeConv output == [n * c_out]
- [x] `e2e_chamfer_self_distance_zero` -- CD(A, A) < 1e-5
- [x] `e2e_icp_identity_convergence` -- ICP on identity converges with
      residual < 1e-3
- [x] `e2e_voxelize_roundtrip` -- scatter -> occupied_centroids round-trip
      coord and feature counts
- [x] `e2e_gaussian_project_valid_depth` -- Gaussian at z = 5 projects with
      depth == 5.0
- [x] `e2e_kdtree_nearest_correctness` -- KD-tree nearest to 9.9 returns
      index 10
- [x] `e2e_knn_vs_brute_force` -- kNN result agrees with sorted brute-force
- [x] `e2e_lcg_rng_determinism` -- two same-seed LCGs produce identical streams
- [x] `e2e_ptx_kernels_all_sm_versions` -- all 7 kernels x 6 SM versions
      contain `.version`, `sm_X`, and kernel name

#### Benchmarks (benches/geom3d_ops.rs)
- [x] 7 PTX kernel groups x 4 SM versions (PTX generation throughput)
- [x] `fps_n4096_m512` -- FPS down-sampling
- [x] `knn_n2048_k16` -- brute-force kNN
- [x] `chamfer_na1024_nb1024` -- Chamfer distance
- [x] `pointnet_forward_n512` -- PointNet end-to-end
- [x] `kdtree_build_n4096` -- KD-tree construction

### Future Enhancements [ ]

#### P0 -- Critical (Performance-Sensitive Paths)
- [x] Grid-based kNN for uniform-density clouds -- spatial hashing fallback
      to outperform brute force for n > 64k (neighborhood/grid_knn.rs -- uniform-grid
      spatial hash, expanding-ring exact kNN + radius search)
- [ ] FlashAttention-style block-sparse Point-Transformer attention with
      neighbour windows
- [ ] Tile-based Gaussian rasteriser -- 16 x 16 pixel tiles with sorted
      Gaussian lists per tile (Inria 3DGS layout)
- [ ] Fused FPS + ball-query + gather kernel for PointNet++ SetAbstraction

#### P1 -- Important (Feature Completeness)
- [x] PointNeXt training-time augmentations (random scale, jitter, drop)
      (sampling/pointnext_aug.rs -- Qian 2022; random scale + Gaussian jitter (clipped) + random-drop downsample + yaw rotation; ordered pipeline via apply())
- [x] KPConv (Kernel Point Convolution) layer
- [x] Octree-based hierarchical voxelisation for very large clouds (voxel/octree.rs
      -- hierarchical 8-way subdivision with max_depth/max_points_per_leaf, AABB-pruned
      radius + kNN queries)
- [x] Marching cubes mesh extraction from voxel SDF
- [x] Tetrahedral mesh ops (Delaunay, volume) for FEM applications
      (mesh/delaunay3d.rs -- Bowyer-Watson incremental tetrahedralization with
      robust f64 orient3d / in_sphere predicates, circumcenter + per-tet/total
      volume, convex-hull face extraction)
- [x] Ray/triangle + point/triangle distance queries (mesh/ray_triangle.rs --
      Moller-Trumbore intersection, Ericson closest-point, ray-AABB slab,
      mesh-level nearest-hit and nearest-surface-point)
- [x] Discrete differential-geometry curvature (mesh/curvature.rs -- Meyer 2003
      angle-defect Gaussian + cotangent-Laplacian mean + principal curvatures,
      mixed Voronoi area; icosphere oracle generator)
- [ ] Open3D-compatible PLY / PCD readers (Pure Rust)
- [x] Range-image projection helpers for LiDAR data
      (transform/range_image.rs -- LiDAR-style azimuth×elevation spherical projection with per-pixel min-range; unproject inverse for round-trip)

#### P2 -- Nice-to-Have (Advanced Features)
- [ ] 2D Gaussian splatting (2DGS) primitives
- [ ] Mip-Splatting anti-aliasing path
- [ ] Differentiable rasterisation gradients (de-rendering)
- [ ] Point-cloud generative diffusion models (PointFlow-style)
- [ ] SE(3)-equivariant network primitives (EGNN, e3nn-style)
- [ ] BVH-based ray-Gaussian intersection

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

No CUDA SDK, no C, no Fortran. The crate compiles standalone and produces PTX
strings that can be consumed by `oxicuda-driver` / `oxicuda-launch` at runtime.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 328 unit + 15 E2E = 343 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- All public APIs return `Geom3dResult<T>` or `Result<T, Geom3dError>`

## Performance Targets

Reference shapes (FPS and kNN dominate point-cloud pipelines; Gaussian project
+ rasterise dominate 3DGS):

| Kernel | Shape | Target |
|--------|-------|--------|
| farthest_point_sample | n = 16384, m = 1024 | bounded by min-reduce throughput |
| ball_query | n = 16384, nq = 4096, r = 0.1 | bandwidth-limited |
| knn | n = 16384, k = 32 | k * n^2 brute path; grid path target O(n) |
| voxelize | n = 1M points, grid = 256^3 | bandwidth-limited |
| chamfer_distance | |A| = |B| = 16384 | pairwise reduction |
| gaussian_project | n_gaussians = 1M | matrix-vector throughput |
| sh_eval (L = 0..2) | n_gaussians = 1M | arithmetic-bound |

## Notes

- All random sampling is deterministic via `LcgRng` seeded by `Geom3dHandle`;
  unit tests do not depend on `rand` or `getrandom`.
- `farthest_point_sample` initialises with idx[0] = 0 so output is fully
  deterministic given inputs (no random first-seed selection).
- `voxel_downsample` uses HashMap then sorts keys, so the emission order is
  deterministic and reproducible.
- KD-tree `nearest` returns the original index in the input array; node split
  selects on the longest-extent axis.
- `Gaussian3d` stores log-scale and pre-sigmoid opacity so unconstrained
  parameters can be passed directly from optimisers.
- ICP's SVD post-processing forces det(V * U^T) > 0 so the recovered rotation
  is a proper SO(3) element (no reflections).
- Sinkhorn EMD uses log-domain updates clamped to +/-50 to prevent overflow;
  epsilon defaults guard against zero-division.

---

## Architecture-Specific Deepening

### Hopper (sm_90 / sm_90a)
- [ ] `wgmma.mma_async` path for Point-Transformer attention QK^T
- [ ] TMA (`cp.async.bulk`) loading of point batches in PointNet shared MLP

### Ampere (sm_80 / sm_86) / Ada (sm_89)
- [ ] `cp.async` prefetch of neighbour features in `ball_query` + `gather`
- [ ] Cooperative groups for warp-wide min/max reduce in FPS

### Blackwell (sm_100 / sm_120)
- [ ] 5th-gen Tensor Core for ICP 3 x 3 SVD via warp-level GEMM
- [ ] Cluster launch for cross-CTA Gaussian sort in rasteriser

---

## Deepening Opportunities

### Verification Gaps
- [x] All 7 PTX generators emit `.version`, `.target sm_X`, and named entry per
      SM version (verified by `e2e_ptx_kernels_all_sm_versions`)
- [x] KD-tree `nearest` cross-checked against brute force (within crate tests)
- [x] kNN agreement with brute-force for k = 1..k_max (E2E)
- [x] ICP identity convergence (residual < 1e-3)
- [x] Chamfer self-distance == 0 (E2E)
- [ ] Gaussian rasteriser pixel-exact match vs. Inria 3DGS reference
- [ ] Sinkhorn EMD convergence vs. POT (Python OT library) on small problems

### Implementation Deepening
- [x] `forbid(unsafe_code)` enforced at crate level
- [x] All RNG operations deterministic given seed (E2E test)
- [x] KD-tree supports `nearest`, `knn`, and `radius_search` with AABB pruning
- [ ] Batched point-cloud forward (B x N x 3) variants of PointNet
      architectures
- [ ] Differentiable FPS gradient (straight-through estimator)
- [ ] Gradient through `project_gaussian` for differentiable rendering
- [ ] Stochastic ball query (random subsample when count > nsample)
- [ ] Mesh topology validation in `Geom3dError::InvalidTopology` path
