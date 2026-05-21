# oxicuda-sketch

Streaming sketches and sublinear algorithms -- HyperLogLog, Count-Min, Bloom, t-Digest, MinHash, and LSH in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-sketch` is the streaming / sublinear-algorithms volume of the
OxiCUDA stack. It collects the standard probabilistic data structures
used for cardinality estimation, frequency estimation, set membership,
quantile summarisation, top-k tracking, similarity search, sampling,
and norm / moment estimation over data streams.

All algorithms are implemented in pure Rust with no external dependencies
beyond `thiserror`. The crate also emits GPU PTX kernel strings through
`ptx_kernels`, parameterised on SM compute capability, for the inner
loops (register updates, hash table probes, reservoir sampling) that map
cleanly to GPU execution.

Hash families are kept first-class: Murmur3, FNV-1a, a slim xxHash3 variant,
a 2-universal family, and tabulation hashing are all available in the
`hash` module and reused throughout the sketch implementations.

## Modules

| Module | Description |
|--------|-------------|
| `hash` | Hash families: Murmur3, FNV-1a, xxHash3 (slim), 2-universal, tabulation |
| `cardinality` | HyperLogLog, HyperLogLog++, Linear Counting, Theta Sketch, sliding-window HLL |
| `frequency` | Count-Min Sketch, Count Sketch, Conservative-Update Count-Min |
| `membership` | Bloom filter, Counting Bloom, Cuckoo filter, Quotient filter |
| `quantile` | KLL, t-Digest, Greenwald-Khanna, P-square |
| `topk` | Misra-Gries, Space-Saving, Frequent, HeavyKeeper |
| `similarity` | MinHash (Jaccard), SimHash (cosine), Weighted MinHash, KMV |
| `lsh` | Cosine LSH (SimHash-based) and Jaccard LSH (banded MinHash) |
| `sampling` | Reservoir (Vitter), Weighted Reservoir (Efraimidis-Spirakis), Bernoulli, priority sampling |
| `moment` | AMS L2 sketch, Johnson-Lindenstrauss, Lp-norm via stable projections |
| `matrix` | Matrix sketching (Frequent Directions) |
| `stream` | Online mean / variance, exponential decay, sliding window counts, changepoint detection |
| `metrics` | Relative error, MAE, accuracy, recall-at-k |
| `handle` | `SketchHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernels for streaming sketch operations |
| `error` | `SketchError` / `SketchResult` |

## Method Coverage

### Cardinality (`cardinality`)
- HyperLogLog and HyperLogLog++
- Linear Counting (small-cardinality regime)
- Theta Sketch
- Sliding-window HLL for time-decayed distinct counts

### Frequency (`frequency`)
- Count-Min Sketch with point-query estimation
- Count Sketch (signed counters for unbiased estimates)
- Conservative-update Count-Min for skewed streams

### Membership (`membership`)
- Bloom filter and counting Bloom filter
- Cuckoo filter (supports deletes)
- Quotient filter (cache-friendly insertions)

### Quantiles (`quantile`)
- KLL sketch (mergeable, rank-error bounded)
- t-Digest (centroid-based quantile summary)
- Greenwald-Khanna (deterministic rank-error bounds)
- P-square (constant-space single quantile)

### Top-K (`topk`)
- Misra-Gries, Frequent
- Space-Saving
- HeavyKeeper

### Similarity and LSH (`similarity` + `lsh`)
- MinHash (Jaccard) and Weighted MinHash
- SimHash (cosine) and KMV (k minimum values)
- Cosine LSH and banded Jaccard LSH index

### Sampling and moments (`sampling` + `moment`)
- Reservoir sampling (Vitter), weighted reservoir (Efraimidis-Spirakis)
- Bernoulli and priority sampling
- AMS L2-norm sketch
- Johnson-Lindenstrauss random projection
- Lp-norm estimation via stable projections

## Quick Start

```rust,no_run
use oxicuda_sketch::cardinality::hll::HyperLogLog;
use oxicuda_sketch::SketchResult;

fn main() -> SketchResult<()> {
    // Allocate a HyperLogLog with precision p (m = 2^p registers).
    let mut hll = HyperLogLog::new(12, 0xC0FFEE)?;

    // Stream values into the sketch (raw u64 keys, or pre-hashed via `add_hash`).
    for x in 0u64..1_000_000 {
        hll.add_u64(x);
    }

    // Estimate the number of distinct elements seen so far.
    let _cardinality = hll.estimate();
    Ok(())
}
```

## Status

**Alpha** -- 8,533 SLoC, 332 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
