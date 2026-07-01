# oxicuda-manifold

Manifold learning, dimensionality reduction, and Riemannian geometry in pure Rust, paired with PTX kernels emitted at runtime for OxiCUDA.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-manifold` covers a wide swath of non-linear dimensionality
reduction and manifold-learning techniques. Classical methods (PCA, Kernel
PCA, FastICA, classical MDS, SMACOF stress majorisation) sit alongside
neighbourhood embedding (t-SNE and its heavy-tailed / α / Cauchy / SSNE
variants, UMAP with multiscale and supervised variants, PaCMAP, TriMap,
PHATE), local methods (LLE, MLLE, Isomap, Laplacian Eigenmaps),
diffusion-based embeddings, and modern auto-encoder hooks.

The crate is also a substrate for Riemannian geometry: manifolds (Stiefel,
Grassmann, SPD with affine-invariant / Bures-Wasserstein metrics, Poincaré
ball, SO(n)), Riemannian k-means on SPD matrices, and Riemannian SGD with
retraction operators. Persistent homology (Vietoris-Rips) and the Mapper
algorithm provide a TDA layer on top.

All algorithms are implemented in safe Rust with no external linear-algebra
dependencies — the in-crate `linalg` module provides Jacobi eigen-decomp,
power iteration, Lanczos, Householder QR, and the other primitives used
internally. Random sampling uses the workspace `LcgRng`. PTX kernels for
the hot loops (t-SNE gradient, UMAP step, PCA centering, MDS double
centering) are emitted parametric in the device SM version.

## Modules

| Module | Description |
|--------|-------------|
| `linear` | Linear dimensionality reduction (PCA, Kernel PCA, FastICA, CCA, PLS) |
| `tsne` | t-SNE and heavy-tailed / α / Cauchy / SSNE / NeRV / JSE variants |
| `umap` | UMAP, multiscale UMAP, supervised UMAP |
| `local` | Local-neighbourhood methods: LLE, MLLE, Isomap, Laplacian Eigenmaps |
| `diffusion` | Diffusion Maps (Coifman-Lafon) and PHATE |
| `mds` | Classical MDS and SMACOF stress majorisation |
| `reduction` | Modern dimensionality reduction: PaCMAP, TriMap |
| `neighbor` | k-NN search structures (brute, KD-tree, ball tree, HNSW) |
| `linalg` | Jacobi eig, power iteration, Lanczos, Householder QR primitives |
| `riemannian` | Stiefel, Grassmann, SPD (affine-invariant + Bures), Poincaré ball, SO(n) |
| `optim` | Riemannian SGD with retractions |
| `clustering` | Kohonen SOM, spectral clustering, SPD k-means |
| `topology` | Persistent homology (Vietoris-Rips), Mapper graph |
| `autoencoder` | Manifold-learning hooks for autoencoder pipelines |
| `metrics` | Trustworthiness, continuity, KL, neighbourhood preservation |
| `handle` | `ManifoldHandle`, `SmVersion`, `LcgRng` |
| `error` | `ManifoldError` / `ManifoldResult` |
| `ptx_kernels` | Runtime PTX strings for t-SNE / UMAP / PCA / MDS per SM version |

## Quick Start

```rust,no_run
use oxicuda_manifold::handle::LcgRng;
use oxicuda_manifold::tsne::tsne::{tsne_fit, TsneConfig};
use oxicuda_manifold::ManifoldResult;

fn main() -> ManifoldResult<()> {
    // Input data: row-major (n_samples, dim).
    let n_samples: usize = unimplemented!();
    let dim: usize = unimplemented!();
    let x: Vec<f64> = unimplemented!();

    let cfg = TsneConfig::default();
    let mut rng = LcgRng::new(0xC001_CAFE);

    let out = tsne_fit(&x, n_samples, dim, &cfg, &mut rng)?;
    println!("final KL = {}", out.final_kl_divergence);
    println!("embedding len = {}", out.embedding.len());
    Ok(())
}
```

## Status

**Alpha** — 26,639 SLoC, 620 passing tests. API may evolve before v1.0.

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
