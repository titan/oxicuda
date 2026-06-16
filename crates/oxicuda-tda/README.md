# oxicuda-tda

Topological Data Analysis -- a pure Rust persistent-homology and Mapper
toolkit.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-tda` provides a pure-Rust topological data analysis stack covering
simplicial complex construction, persistent (co)homology, persistence-diagram
distances, and Mapper-style topological summaries. Filtrations include
Vietoris-Rips (from a distance matrix or raw point cloud) and Čech complexes
backed by a minimum-enclosing-ball solver, with sublevel-set filtrations
supported through generic `Filtration` construction.

Persistence is computed over Z/2 via standard column-reduction or the twist
algorithm (Chen-Kerber 2011) for accelerated descending-dimension passes,
with persistent cohomology and Euler-characteristic agreement checks.
Persistence diagrams expose 1-Wasserstein and bottleneck distances,
Wasserstein-p with Hungarian assignment, persistence images, persistence
silhouettes, and topological summary statistics (Betti numbers, persistent
entropy, total persistence). Mapper builds a topological graph from a cover
plus single-linkage clustering, and a lazy witness complex with maxmin
landmark selection handles large point clouds.

## Modules

| Module | Description |
|--------|-------------|
| `complex` | `Simplex`, `SimplicialComplex`, `Filtration`, Vietoris-Rips, Čech |
| `distance` | Pairwise distance matrix and k-NN graph |
| `homology` | `BoundaryMatrix`, column reduction (Z/2), twist reduction, persistence pairs, cohomology |
| `persistence` | `PersistenceDiagram`, barcode, bottleneck/Wasserstein distances, persistence images |
| `mapper` | Mapper algorithm with cover and single-linkage clustering |
| `witness` | Lazy witness complex and maxmin landmark selection |
| `metrics` | Betti numbers, persistent entropy, landscapes, total persistence |
| `handle` | `TdaHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernel strings for TDA algorithms |
| `error` | `TdaError` / `TdaResult` |

## Quick Start

```rust,no_run
use oxicuda_tda::complex::filtration::Filtration;
use oxicuda_tda::homology::boundary::BoundaryMatrix;
use oxicuda_tda::homology::persistent::extract_persistence_pairs;
use oxicuda_tda::persistence::diagram::PersistenceDiagram;
use oxicuda_tda::error::TdaResult;

fn main() -> TdaResult<()> {
    // Eight points around a unit circle (row-major, 8 x 2).
    let mut points: Vec<f64> = Vec::with_capacity(16);
    for k in 0..8 {
        let theta = (k as f64) * std::f64::consts::TAU / 8.0;
        points.push(theta.cos());
        points.push(theta.sin());
    }

    // Build a Vietoris-Rips filtration up to dim 1 with a generous radius.
    let filt = Filtration::vietoris_rips_from_points(&points, 2, 4.0, 1)?;

    // Reduce the boundary matrix and extract H_0 / H_1 persistence pairs.
    let bm = BoundaryMatrix::from_filtration(&filt)?;
    let pairs = extract_persistence_pairs(&bm, &filt)?;
    let diagrams = PersistenceDiagram::from_pairs_by_dim(&pairs, 1);
    for (d, diag) in diagrams.iter().enumerate() {
        println!("H_{d}: {} pairs", diag.pairs.len());
    }
    Ok(())
}
```

## Status

**Alpha** -- 12,009 SLoC, 379 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
