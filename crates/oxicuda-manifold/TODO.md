# oxicuda-manifold TODO

GPU-accelerated manifold learning, dimensionality reduction, and Riemannian geometry,
serving as a pure Rust equivalent to RAPIDS cuML's manifold module + Geomstats.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.53).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 29,018 (89 files, including 5,676 code + 205 comments + 339 blanks; markdown 415)
- **Tests:** 620 lib/e2e + 3 doctests passing (wired 3 orphan modules — config-struct Isomap, parametric t-SNE, SPD geodesic regression — reviving +55 tests)
- **Pure Rust:** Zero external linear-algebra dependencies; only `thiserror` runtime dep
- **PTX coverage:** 7 kernels x 6 SM versions = 42 PTX string generators

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `ManifoldError` enum (ShapeMismatch, NotConverged, EmptyInput, DimensionMismatch, InvalidParameter, EigenFailure, NumericalInstability, UnsupportedSmVersion, KNeighborsTooLarge, SingularMatrix, IndexOutOfBounds, ...) + `ManifoldResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 bool, Box-Muller normal), `ManifoldHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `pairwise_dist_sq`, `knn_topk`, `tsne_grad`, `umap_step`, `pca_center`, `mds_double_center`, `random_proj` (string concatenation only, no nvcc dependency)

#### Linear Methods
- [x] `linear/pca.rs` -- Center -> covariance Sigma = X^T X / (n - 1) -> Jacobi eigh -> sort descending -> project
- [x] `linear/kernel_pca.rs` -- Gaussian / Polynomial / Linear kernels -> centred Gram -> eigh
- [x] `linear/fast_ica.rs` -- Whitening + fixed-point iteration (tanh / gauss G), symmetric polar orthogonalisation

#### t-SNE Family
- [x] `tsne/perplexity.rs` -- Per-row binary search for sigma_i to match target perplexity
- [x] `tsne/tsne.rs` -- Full t-SNE: P->Q gradient with early-exaggeration + momentum; converges, separates clusters
- [x] `tsne/barnes_hut.rs` -- 2D quadtree approximate gradient for n >= 1000

#### UMAP
- [x] `umap/knn_graph.rs` -- kNN edges + smooth-kNN sigma/rho fit (binary search to log2(k))
- [x] `umap/fuzzy_simplicial.rs` -- Membership mu in (0, 1]; symmetrise via mu union nu = mu + nu - mu * nu
- [x] `umap/embedding.rs` -- a / b curve fit + SGD with negative sampling; cross-entropy on edges

#### Local / Spectral Methods
- [x] `local/lle.rs` -- Constrained-LS weights with sum w_ij = 1 over kNN; M = (I - W)^T (I - W); d + 1 smallest eigenvectors, drop first
- [x] `local/mlle.rs` -- Modified LLE with multi-weight basis (Zhang-Wang 2007)
- [x] `local/hessian_lle.rs` -- Hessian LLE (Donoho-Grimes 2003): per-point local tangent PCA + Hessian design [1 | linear | quadratic] -> orthonormalise -> Phi += H H^T -> smallest eigenvectors (drop constant), identity-covariance rescale
- [x] `local/ltsa.rs` -- Local Tangent Space Alignment (Zhang-Zha 2005): per-point tangent basis Q_i, G_i = [1/sqrt(k) | Q_i], B += I - G_i G_i^T -> smallest nonzero eigenvectors (drop constant)
- [x] `local/isomap.rs` -- kNN graph + Dijkstra all-pairs geodesic distance + classical MDS
- [x] `embed/isomap.rs:isomap` -- wired orphan: `IsomapConfig` + `isomap()` config-struct API over `local::isomap::isomap_fit` (kNN -> Dijkstra geodesics -> classical MDS); +11 revived tests
- [x] `local/laplacian_eigenmaps.rs` -- Gaussian-weight W -> normalised L_sym -> generalised eigh `L v = lambda D v` -> drop constant eigenvector

#### Diffusion
- [x] `diffusion/diffusion_map.rs` -- Coifman-Lafon: kernel + alpha density normalisation -> row-stochastic P -> eigh -> `Psi_i = lambda_i^t psi_i`

#### MDS
- [x] `mds/classical_mds.rs` -- Torgerson: B = -1/2 J D^2 J -> eigh -> U sqrt(Lambda)
- [x] `mds/smacof.rs` -- Iterative majorisation via Guttman transform
- [x] `mds/nonmetric_mds.rs` -- Non-metric / ordinal MDS (Kruskal 1964): classical-MDS init + PAVA isotonic regression of disparities + SMACOF Guttman step toward disparities; Stress-1 monotone non-increasing

#### Neighbour Search
- [x] `neighbor/knn_brute.rs` -- Brute force pairwise + partial sort
- [x] `neighbor/kd_tree.rs` -- Median-split KD-tree with backtracking pruning
- [x] `neighbor/ball_tree.rs` -- Centroid + radius Ball-tree neighbour search

#### Numerical Linear Algebra
- [x] `linalg/jacobi_eig.rs` -- Cyclic Jacobi eigh for symmetric matrices
- [x] `linalg/power_iter.rs` -- Deflated power iteration for dominant eigenpairs
- [x] `linalg/lanczos.rs` -- Lanczos with full reorthogonalisation
- [x] `linalg/householder_qr.rs` -- Householder QR + polar orthogonalisation

#### Riemannian Manifolds
- [x] `riemannian/stiefel.rs` -- St(n, p): QR retraction; tangent projection `X - Y * sym(Y^T X)`
- [x] `riemannian/grassmann.rs` -- Gr(n, p): principal-angle SVD geodesics
- [x] `riemannian/spd.rs` -- SPD affine-invariant: `exp_P(X) = P^{1/2} exp(P^{-1/2} X P^{-1/2}) P^{1/2}` via symmetric matrix square roots
- [x] `riemannian/hyperbolic_poincare.rs` -- Poincare ball: Mobius addition + `d(u, v) = arcosh(1 + 2 ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))`

#### Optimisation on Manifolds
- [x] `optim/riemannian_sgd.rs` -- Riemannian SGD on Stiefel and SPD
- [x] `optim/retraction.rs` -- QR, Cayley, Cholesky retractions

#### Metrics
- [x] `metrics/metrics.rs` -- Trustworthiness, continuity, KL(P || Q), neighbourhood preservation

#### Validation
- [x] `e2e_tests.rs` -- 24 cross-module tests covering PCA explained-variance, kernel-PCA class isolation, FastICA component recovery, t-SNE cluster separation, UMAP fuzzy roundtrip, LLE swiss-roll-like, Isomap geodesic, classical MDS distance preservation, SMACOF monotone-stress, KD-tree vs brute consistency, Jacobi eigh orthogonality, Stiefel retraction stays-on-manifold, SPD exp/log roundtrip, Poincare triangle inequality, PTX x 6 SM
- [x] `benches/manifold_ops.rs` -- Criterion: 7 PTX kernels x all SM + PCA, Jacobi-eigh, kNN algo benches

### Future Enhancements

#### P0 -- Critical
- [x] Sparse PCA via L1 penalty (Witten-Tibshirani-Hastie) for high-dimensional gene-expression / NLP data (`linear/sparse_pca.rs`)
- [x] Incremental PCA (Ross-Lim-Lin-Yang) for streaming data not fitting in memory (`linear/incremental_pca.rs`)
- [x] UMAP supervised / semi-supervised mode using labels in the fuzzy simplicial set merge step (`umap/supervised.rs`)

#### P1 -- Important
- [x] Approximate kNN via HNSW (orders-of-magnitude faster than KD-tree above d > 50) (`neighbor/hnsw.rs`)
- [x] Trimap / PaCMAP (modern alternatives to UMAP/t-SNE with better global structure preservation) (`reduction/trimap.rs`, `reduction/pacmap.rs`)
- [x] `reduction/parametric_tsne.rs:parametric_tsne_fit` -- wired orphan: MLP-encoder parametric t-SNE (van der Maaten 2009) with Kaiming init + Adam + early exaggeration and out-of-sample `transform`; +22 revived tests
- [x] PHATE diffusion potential for trajectory-preserving embeddings (`diffusion/phate.rs`)
- [x] Riemannian Adam on Stiefel and Grassmann (`optim/riemannian_adam.rs`)
- [x] Symmetric Stiefel (SO(n)) with skew-symmetric tangent and matrix-exponential retraction (`riemannian/so_n.rs`)
- [x] Wasserstein / Bures geometry on the SPD manifold (alternative to affine-invariant) (`riemannian/spd_bures.rs`)
- [x] Lorentz model of hyperbolic space as a numerically-stable alternative to the Poincare ball (`riemannian/hyperbolic_lorentz.rs`)
- [x] `riemannian/geodesic_regression.rs:geodesic_regression_fit` -- wired orphan: least-squares geodesic regression on SPD(d) (Fletcher 2013) -- affine-invariant Exp/Log + parallel transport, Fréchet-mean init, predict/SSE; +22 revived tests

#### P2 -- Nice-to-Have
- [x] Spectral clustering pipeline (Laplacian eigenmaps + k-means on embedding) (`clustering/spectral.rs`)
- [x] Self-organising maps (Kohonen SOM) with neighbourhood decay schedules (`clustering/kohonen_som.rs`)
- [x] Persistent homology / Mapper algorithm for topological data analysis (`topology/persistent_homology.rs`)
- [x] Riemannian k-means and Frechet mean on SPD/Grassmann (`riemannian/spd_kmeans.rs`)
- [x] Stochastic neighbour embedding variants: heavy-tailed t-SNE (`tsne/heavy_tsne.rs`), NeRV, JSE (`tsne/nerv_jse.rs`)
- [x] Auto-encoder-style manifold learning hooks (export embedding+gradient for use in `oxicuda-dnn`) (`autoencoder/manifold_hooks.rs`)
- [x] Cross-decomposition variants: CCA, PLS via the existing PCA backbone (`linear/cca_pls.rs`)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 620 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- Pure Rust: no C/C++/Fortran in default features

## Performance Targets

Representative algorithmic benchmarks (CPU-side reference + PTX generation timing):

| Routine | Problem size | Priority |
|---------|--------------|----------|
| PCA (Jacobi eigh) | 256 x 64, 1024 x 256 | High |
| kNN brute / KD-tree / Ball-tree | n in {1000, 10000}, d in {2, 16, 64} | High |
| t-SNE 1 step | n in {500, 2000}, perplexity = 30 | High |
| UMAP 1 SGD epoch | n in {1000, 10000}, k = 15 | High |
| LLE / Isomap / Laplacian Eigenmaps | n = 1000 | Mid |
| Jacobi eigh | 64 x 64, 256 x 256 | Mid |

Target for GPU execution path: match RAPIDS cuML manifold throughput within 15% on
representative dimensions once `oxicuda-launch` orchestrates the emitted PTX on Linux + NVIDIA.

## Notes

- All routines accept row-major `&[f32]` or `&[f64]` slices via `GpuFloat`-like trait bounds.
- Random sampling is deterministic given a `u64` seed (`LcgRng` is reproducible across runs).
- All eigh routines guarantee orthonormal eigenvectors (Jacobi rotates pairs until off-diagonal < tol).
- Stiefel / Grassmann / SPD / Poincare verified by manifold-residence invariants in `e2e_tests.rs`.

---

## Architecture-Specific Deepening

### PTX Coverage Matrix

| Kernel | sm_70 | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|--------|-------|-------|-------|-------|-------|-------|
| `pairwise_dist_sq` | [x] | [x] | [x] | [x] | [x] | [x] |
| `knn_topk` | [x] | [x] | [x] | [x] | [x] | [x] |
| `tsne_grad` | [x] | [x] | [x] | [x] | [x] | [x] |
| `umap_step` | [x] | [x] | [x] | [x] | [x] | [x] |
| `pca_center` | [x] | [x] | [x] | [x] | [x] | [x] |
| `mds_double_center` | [x] | [x] | [x] | [x] | [x] | [x] |
| `random_proj` | [x] | [x] | [x] | [x] | [x] | [x] |

All six SM versions produce non-empty PTX strings and pass content-substring checks in `e2e_tests.rs`.

### Per-Architecture Optimisation Hooks
- [ ] sm_80 (Ampere) -- emit `cp.async` for `pairwise_dist_sq` shared-memory tile loads (requires GPU hardware: cp.async timing only meaningful on-device)
- [ ] sm_89 (Ada) -- FP8 (e4m3) accumulation for `tsne_grad` (memory-bound on large n) (requires GPU hardware: FP8 tensor units)
- [ ] sm_90 (Hopper) -- `wgmma` + TMA for `pairwise_dist_sq` and `pca_center` covariance accumulation (requires GPU hardware)
- [ ] Verify `knn_topk` warp-level k-way selection beats per-row sort for k <= 32 (requires GPU hardware: warp-level timing)

---

## Deepening Opportunities

### Verification Gaps (require Linux + NVIDIA hardware)
- [ ] End-to-end GPU run of all 7 PTX kernels under `cargo nextest --features gpu-tests` on sm_80 / sm_89 / sm_90
- [ ] Numerical agreement between CPU reference (Jacobi eigh) and GPU `pca_center` + downstream eigh within FP32 tolerance (rel err < 1e-4)
- [ ] t-SNE / UMAP visual quality on canonical datasets (MNIST, Fashion-MNIST, Tabula Muris) -- ARI / NMI vs reference implementations

### Algorithmic Deepening
- [ ] Barnes-Hut t-SNE quadtree currently CPU-only; lift the tree-traversal kernel to PTX with warp-cooperative descent (requires GPU hardware: warp-cooperative descent runs on-device)
- [ ] UMAP fuzzy simplicial set construction parallelised at the per-vertex level (requires GPU hardware: per-vertex GPU parallelism)
- [x] Riemannian SGD with adaptive step size (Riemannian-Adam) for SPD / Stiefel (`optim/riemannian_adam.rs`)
- [x] Multi-scale t-SNE / UMAP for hierarchical embeddings (preserve both micro- and macro-structure) (`umap/multiscale.rs`)
- [x] `riemannian/hyperbolic_ball.rs` — Curvature-parametrised Poincaré ball (Ganea 2018 / Nickel-Kiela 2017): `PoincareBall { curvature, epsilon }` with curvature-`c` Möbius add/sub/scalar-mul, distance `d=(2/√c)·arctanh(√c‖⊖x⊕y‖)`, exp/log maps, gyration-based parallel transport (isometry-verified), egrad→rgrad, Riemannian SGD step (`exp_x(-lr·grad)`), and `poincare_frechet_mean` (Karcher mean). NOTE: distinct from the fixed-unit-curvature `riemannian/hyperbolic_poincare.rs`.
- [x] `riemannian/wrapped_normal.rs` — Wrapped Normal on hyperbolic space (Nagano 2019): push Euclidean Normal through exponential map at μ; RSVI for hierarchical VAE; `HyperbolicNormal { mu, sigma, manifold: PoincareBall }`
- [x] `embedding/umap_parametric.rs` — Parametric UMAP (Sainburg 2021): train a neural encoder to approximate UMAP embedding; new points via forward pass (no re-running umap); `ParametricUmap { encoder_dims: Vec<usize> }`
- [x] `geodesic/heat_method.rs` — Heat method for geodesic distances (Crane 2013): solve heat equation u_t=Δu for small t, normalise gradient, solve Poisson; O(n log n) via sparse Cholesky; output ≈ geodesic from source(s)

### API Polish
- [x] Builder-style configuration for t-SNE (perplexity, learning rate, early exaggeration, momentum schedule) (`TsneConfigBuilder` in `tsne/tsne.rs` — chained `with_*` setters + validating `build() -> ManifoldResult`)
- [x] Builder-style configuration for UMAP (n_neighbours, min_dist, spread, n_epochs) (`UmapConfigBuilder` in `umap/embedding.rs` — chained setters + validating `build()`; the `metric` knob is not part of `UmapConfig` so it is intentionally out of scope)
- [x] Cross-decomposition variants: CCA, PLS via the existing PCA backbone (`linear/cca_pls.rs`)
