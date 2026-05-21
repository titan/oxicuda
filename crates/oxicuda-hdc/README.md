# oxicuda-hdc

Hyperdimensional Computing / Vector Symbolic Architectures in pure Rust, with PTX kernels emitted at runtime for OxiCUDA.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-hdc` provides the building blocks of Hyperdimensional Computing
(HDC) and Vector Symbolic Architectures (VSA): random hypervectors over
several alphabets (binary {±1}, integer Z, complex unit / FHRR, real HRR,
HRR with FFT-accelerated binding, and Sparse Block-Codes), the standard
algebraic operations (binding, bundling, permutation, tensor-product
binding), associative memory structures, and a prototype-based HD
classifier with online error-corrective updates.

Encodings cover the typical sequence and structured-data use cases:
record-based encoding, n-grams, and spatial pattern encoders. Distance and
similarity functions for each hypervector type round out the API, alongside
capacity / dimensionality analysis utilities.

All algorithms are implemented in safe Rust with no external linear-algebra
dependencies. Random sampling uses the workspace `LcgRng`. PTX kernels for
the hot loops (XOR-bind, majority-bundle, cyclic shift, Hamming, cosine,
classifier vote) are emitted parametric in the device SM version.

## Modules

| Module | Description |
|--------|-------------|
| `vector` | Hypervector types: binary {±1}^D, integer Z^D, complex unit (FHRR), real HRR, HRR-FFT, Sparse Block-Codes |
| `ops` | Operations: binding (XOR / multiply / circular convolution), bundling, permutation, tensor-product |
| `memory` | Item memory (symbol→HV), associative / Hopfield memory, hetero-associative memory, VSA Resonator Networks |
| `classifier` | HD classifier with prototype-per-class and online error-corrective update |
| `encoding` | Record-based, n-gram, and spatial pattern encoders |
| `distance` | Hamming, cosine, Jaccard, MinHash similarity metrics |
| `metrics` | Capacity bounds, dimensionality analysis, classification accuracy |
| `handle` | `HdcHandle`, `SmVersion`, `LcgRng` |
| `error` | `HdcError` / `HdcResult` |
| `ptx_kernels` | Runtime PTX strings for binding, bundling, distances, classifier per SM version |

## Quick Start

```rust,no_run
use oxicuda_hdc::classifier::hd_classifier::HdClassifier;
use oxicuda_hdc::handle::LcgRng;
use oxicuda_hdc::vector::binary::random_binary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 1024;
    let mut rng = LcgRng::new(42);

    // 3-class HD classifier with 1024-D bipolar hypervectors.
    let mut clf = HdClassifier::new(3, dim)?;

    // Accumulate a few training examples per class.
    for class in 0..3 {
        for _ in 0..16 {
            let hv = random_binary(dim, &mut rng)?;
            clf.add_example(class, &hv)?;
        }
    }
    clf.build_prototypes(&mut rng)?;

    // Classify a query.
    let query = random_binary(dim, &mut rng)?;
    let predicted = clf.classify(&query)?;
    println!("predicted class = {predicted}");
    Ok(())
}
```

## Design Notes

- Pure Rust, no external linear-algebra or FFT dependencies. HRR binding
  with FFT acceleration uses an in-crate primitive.
- The crate exposes both functional-style operations on `&[i8]` / `&[i32]`
  / `&[f32]` buffers and stateful types (`HdClassifier`, `ItemMemory`,
  `AssocMemory`, `HeteroAssociativeMemory`, `ResonatorNetwork`).
- The CPU implementations are the reference oracle for the matching PTX
  kernels emitted by `ptx_kernels::*` — the GPU path is verified against
  the CPU path in the end-to-end test suite.

## Status

**Alpha** — 5,725 SLoC, 214 passing tests. API may evolve before v1.0.

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
