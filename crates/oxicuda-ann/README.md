# oxicuda-ann

Approximate Nearest Neighbor & vector-search primitives for OxiCUDA — HNSW, IVF, Product Quantization, IVFPQ, LSH, k-NN graph, top-K selection.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Graph-based search**: HNSW (Hierarchical Navigable Small World) with configurable M and ef parameters; layered graph construction and greedy beam search
- **Clustering-based search**: IVF (Inverted File Index) with k-means training, multi-probe search; IVFPQ combining IVF partitioning with product quantization
- **Product Quantization**: PQ codebook training via k-means on sub-vectors; asymmetric distance computation (ADC) table; vector encoding and decoding
- **Locality-sensitive hashing**: Random projection LSH for Euclidean space; MinHash for Jaccard similarity estimation
- **k-NN graph construction**: Brute-force and NN-Descent approximate graph build; flat L2 exact search index
- **PTX kernels**: 7 GPU kernels (L2 distance batch, inner-product distance batch, PQ ADC table, HNSW neighbor eval, IVF assign, LSH random projection, top-K select) × 6 SM versions

## Usage

```rust
use oxicuda_ann::{
    hnsw::{graph::HnswGraph, insert::hnsw_insert, search::hnsw_search},
    handle::LcgRng,
};

let mut rng = LcgRng::new(42);
let dim = 128;
let mut graph = HnswGraph::new(dim, /*M=*/16, /*ef_construction=*/200, /*max_elements=*/10_000);

// Index vectors
let data: Vec<f32> = (0..1000 * dim).map(|_| rng.next_f32()).collect();
for i in 0..1000 {
    hnsw_insert(&mut graph, &data[i * dim..(i + 1) * dim], &mut rng);
}

// Query top-10 neighbors
let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
let results = hnsw_search(&graph, &query, 10).unwrap();
for (id, dist) in &results {
    println!("id={id}, dist={dist:.4}");
}
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-ann)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
