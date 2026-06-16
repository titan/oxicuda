# oxicuda-sketch TODO

GPU-accelerated streaming data sketches and sublinear algorithms,
serving as a pure Rust equivalent to Apache DataSketches + RAPIDS streaming primitives.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.55).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 11,692 (78 files, including 4,832 code + 142 comments + 554 blanks; markdown 543)
- **Tests:** 456 passing (lib + e2e_tests)
- **Pure Rust:** Zero external dependencies beyond `thiserror`
- **PTX coverage:** 7 kernels x 6 SM versions = 42 PTX string generators

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `SketchError` enum (InvalidParameter, EmptyStream, ShapeMismatch, DimensionMismatch, UnsupportedSmVersion, CapacityExceeded, IndexOutOfBounds, NumericalInstability, HashTableFull, DimensionMustBePowerOfTwo, ...) + `SketchResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `SketchHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `cm_update`, `cm_query`, `hll_register`, `bloom_insert`, `minhash_sketch`, `tdigest_merge`, `reservoir_sample` (string concatenation only, no nvcc dependency)

#### Hash Families
- [x] `hash/murmur3.rs` -- Murmur3-32
- [x] `hash/fnv64.rs` -- FNV-1a 64
- [x] `hash/xxh3_min.rs` -- Simplified xxH3 64-bit
- [x] `hash/universal.rs` -- 2-universal `((a * x + b) mod p) mod m` with `p = 2^61 - 1` Mersenne
- [x] `hash/twouniv.rs` -- Independent 2-universal family generator (seeded)
- [x] `hash/tabulation.rs` -- Tabulation hashing (per-byte 256-entry tables XOR)

#### Cardinality Estimators
- [x] `cardinality/hll.rs` -- Flajolet HyperLogLog (`m = 2^p`, `alpha_m * m^2 / sum 2^(-M_j)`, bias correction). Passes +/- 5% accuracy on 10000 distinct at p = 14
- [x] `cardinality/hll_plus.rs` -- HLL++ (Heule 2013: 6-bit registers + sparse representation + small-range bias-correction table)
- [x] `cardinality/linear_counting.rs` -- Linear counting for `n < 2.5 * m`

#### Frequency Sketches
- [x] `frequency/count_min.rs` -- Cormode-Muthukrishnan CM (`d x w` table + 2-universal hashes, min over rows; over-estimate guarantee)
- [x] `frequency/count_sketch.rs` -- Charikar Count Sketch (sign hashes + median)
- [x] `frequency/conservative_update.rs` -- Conservative-update CM (update only the minimum row)

#### Membership
- [x] `membership/bloom.rs` -- Classical Bloom filter (m-bit + k hashes, optimal `k = (m / n) * ln(2)`, FP-rate ~ `(1 - e^{-kn / m})^k`, never false negative)
- [x] `membership/counting_bloom.rs` -- 4-bit slots, supports deletion
- [x] `membership/cuckoo.rs` -- Cuckoo filter (Fan et al. 2014, fingerprint + cuckoo hashing)

#### Quantile Sketches
- [x] `quantile/kll.rs` -- Karnin-Lang-Liberty 2016 hierarchical compactors
- [x] `quantile/t_digest.rs` -- Dunning 2019 (`k(q, delta) = delta * arcsin(2q - 1) / (2 * pi)` scale, merge-and-resize)
- [x] `quantile/gk_quantile.rs` -- Greenwald-Khanna epsilon-approximate quantile (2001)
- [x] `quantile/p_square.rs` -- Jain-Chlamtac P^2 with 5 markers + parabolic prediction

#### Top-K / Heavy Hitters
- [x] `topk/misra_gries.rs` -- Misra-Gries (k slots, epsilon = 1 / k heavy-hitters)
- [x] `topk/space_saving.rs` -- Metwally Space-Saving (replace min counter slot)
- [x] `topk/frequent.rs` -- Frequent items above `n / (k + 1)`

#### Similarity Sketches
- [x] `similarity/minhash.rs` -- K independent hashes; Jaccard estimate converges to true value
- [x] `similarity/simhash.rs` -- Charikar SimHash (+/- w hyperplane votes -> bit signature; cosine ~ 1 - 2 * hamming / d)
- [x] `similarity/weighted_minhash.rs` -- Ioffe 2010 weighted MinHash (consistent weighted sampling)

#### Locality-Sensitive Hashing
- [x] `lsh/cosine_lsh.rs` -- K random hyperplanes -> K-bit signature with L bands
- [x] `lsh/jaccard_lsh.rs` -- r x b banded over MinHash signature
- [x] `lsh/lsh_index.rs` -- Generic LSH bucket-and-probe insert / query

#### Sampling Sketches
- [x] `sampling/reservoir.rs` -- Vitter Algorithm R (uniform sample test passes)
- [x] `sampling/weighted_reservoir.rs` -- Efraimidis-Spirakis (`key = u^(1 / w)`, keep top-k)
- [x] `sampling/bernoulli.rs` -- Bernoulli inclusion sampling
- [x] `sampling/priority.rs` -- Duffield priority sampling

#### Moment Sketches
- [x] `moment/ams_l2.rs` -- Alon-Matias-Szegedy L2 (Rademacher sketch + median-of-means)
- [x] `moment/johnson_lindenstrauss.rs` -- JL projection (Gaussian + Rademacher)
- [x] `moment/lp_norm.rs` -- Lp norm via stable random projections (Cauchy for L1, Gaussian for L2)

#### Streaming Aggregates
- [x] `stream/online_mean_var.rs` -- Welford online with Chan merge formula
- [x] `stream/exponential_decay.rs` -- EWMA-style exponentially decayed aggregates
- [x] `stream/sliding_window.rs` -- Sliding-window counts

#### Metrics
- [x] `metrics/metrics.rs` -- Relative error, MAE, accuracy, recall-at-k

#### Validation
- [x] `e2e_tests.rs` -- 22 cross-module tests: HLL accuracy 10000 distinct +/- 5% at p = 14, Count-Min over-estimate guarantee, Bloom false-negative-free, MinHash Jaccard convergence, t-Digest quantile within epsilon, KLL median accurate, Misra-Gries returns all heavy hitters, Space-Saving correct, reservoir uniform sample, weighted reservoir top-k, AMS L2-norm estimate, JL distance preservation, cosine LSH recall, Jaccard LSH, Welford online matches batch, PTX x 6 SM
- [x] `benches/sketch_ops.rs` -- Criterion: 7 PTX kernels x all SM + HLL / Count-Min / Bloom / MinHash / reservoir algo benches

### Future Enhancements

#### P0 -- Critical
- [ ] Mergeable sketch unions (HLL union for distributed cardinality [x]; KLL merge [x] -- `quantile/kll.rs::merge` / `merged`; CM column-wise sum [ ])
- [x] Streaming KMV (Bottom-K MinHash) for distinct count + similarity in one structure
- [ ] Apache DataSketches FI (frequent-items) byte-serialisation compatibility

#### P1 -- Important
- [x] Theta sketches (Apache DataSketches) for set operations (intersection, A \ B) on cardinalities
- [x] Quotient filter (alternative to Bloom with locality + delete)
- [x] HeavyKeeper (top-k with bias-correction superior to Space-Saving)
- [x] Misra-Gries on weighted streams (per-item value contribution) -- `topk/weighted_misra_gries.rs` (Berinde-Cormode-Indyk-Strauss 2010)
- [x] Streaming PCA via Frequent Directions (Liberty 2013)
- [x] Streaming SVD (low-rank approximation under bounded memory)
- [x] Reservoir sampling without replacement (Algorithm L for uniform sampling)
- [x] Tug-of-war / second-moment sketch with bounded error bounds -- `moment/ams_f2.rs` (AMS 1996) + 4-wise independent sign family `hash/fourwise.rs`

#### P2 -- Nice-to-Have
- [x] Bloomier filter (function-valued Bloom)
- [ ] Cuckoo filter with 4-byte fingerprints for very low FP rates
- [ ] Compressed sensing sketches (covered separately in `oxicuda-cs`)
- [x] Online change-point detection via PageHinkley / CUSUM (`src/stream/changepoint.rs`; Page 1954, Hinkley 1971)
- [x] Sliding-window HyperLogLog (Heule extensions) (`src/cardinality/sliding_window_hll.rs`; Chabchoub-Heroum 2010 / Heule 2013)
- [ ] Streaming graph sketches (graph sparsification via spectral sketch)
- [x] Differential privacy noise on top of CM and HLL queries
- [ ] Bloom-1 (Putze-Sanders-Singler) for very large filters
- [x] `moment/lp_stable.rs` — Lp-stable random projection (Indyk 2006): sketch each update (i, Δ) by S[j]+=Δ·g_ij where gᵢⱼ∼Stable(p); estimate ‖x‖_p from median of |S[j]|; `LpStableSketch { p: f32, width, depth }`
- [x] `frequency/ada_sketch.rs` — Ada-Sketch (Huang 2021): adaptive Count-Min that allocates extra counters to heavy hitters detected online; ε error guarantee with 2× fewer cells than vanilla CM for skewed distributions

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 456 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- Pure Rust: no C/C++/Fortran in default features

## Performance Targets

Representative algorithmic benchmarks (CPU-side reference + PTX generation timing):

| Routine | Problem size | Priority |
|---------|--------------|----------|
| HLL update (p = 14) | n in {1e4, 1e6} | High |
| Count-Min query | (d = 4, w = 1024), q in {1e4, 1e5} | High |
| Bloom insert / lookup | (m = 2^16, k = 7), n in {1e4, 1e5} | High |
| MinHash sketch | k = 128, |S| in {1e3, 1e4} | High |
| t-Digest merge | n in {1e4, 1e5} | High |
| Reservoir sample | n in {1e5, 1e6}, k = 1000 | Mid |
| KLL median | n in {1e4, 1e5} | Mid |
| Misra-Gries / Space-Saving | k = 100, n in {1e5, 1e6} | Mid |

Target for GPU execution path: process 10^8 updates / second for HLL and Count-Min
on sm_80, with overall accuracy bounds matching Apache DataSketches once
`oxicuda-launch` orchestrates the emitted PTX on Linux + NVIDIA.

## Notes

- HLL accuracy guarantee: standard error ~ 1.04 / sqrt(m) = 1.04 / sqrt(2^p).
- Count-Min over-estimation guarantee: never under-estimate; error bounded by `2 * total_count / w` with probability `1 - delta`.
- Bloom filter is "no false negatives"; counting variant supports decrement.
- MinHash similarity estimate has standard error ~ 1 / sqrt(k) for k hash signatures.
- All hash families seeded via `LcgRng` for reproducibility; production use should mix in an os-random seed.

---

## Architecture-Specific Deepening

### PTX Coverage Matrix

| Kernel | sm_70 | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|--------|-------|-------|-------|-------|-------|-------|
| `cm_update` | [x] | [x] | [x] | [x] | [x] | [x] |
| `cm_query` | [x] | [x] | [x] | [x] | [x] | [x] |
| `hll_register` | [x] | [x] | [x] | [x] | [x] | [x] |
| `bloom_insert` | [x] | [x] | [x] | [x] | [x] | [x] |
| `minhash_sketch` | [x] | [x] | [x] | [x] | [x] | [x] |
| `tdigest_merge` | [x] | [x] | [x] | [x] | [x] | [x] |
| `reservoir_sample` | [x] | [x] | [x] | [x] | [x] | [x] |

All six SM versions produce non-empty PTX strings and pass content-substring checks in `e2e_tests.rs`.

### Per-Architecture Optimisation Hooks
- [ ] sm_80 (Ampere) -- warp-cooperative `atom.global.add` for `cm_update` and `bloom_insert`
- [ ] sm_89 (Ada) -- shared-memory bucket-batching for `hll_register` writes
- [ ] sm_90 (Hopper) -- TMA + `cp.async.bulk` for streaming reservoir refills
- [ ] Verify `tdigest_merge` centroid coalescing is monotone on all SM versions

---

## Deepening Opportunities

### Verification Gaps (require Linux + NVIDIA hardware)
- [ ] GPU run of all 7 PTX kernels under `cargo nextest --features gpu-tests` on sm_80 / sm_89 / sm_90
- [ ] End-to-end throughput vs. Apache DataSketches CPU benchmark at 10^8 updates / second
- [ ] HLL distributed-merge semantic test: GPU streams produce same cardinality as CPU after union

### Algorithmic Deepening
- [x] Hierarchical HLL (HLL-TailCut) for very low cardinalities
- [ ] Sliding-window variants of CM / HLL / Bloom with time-decaying buckets
- [ ] Combined frequency + cardinality sketch (e.g., AMS over distinct elements)
- [ ] Locality-sensitive hashing for general Lp metrics (p-stable distributions)
- [ ] Bottom-K MinHash with weighted variant (consistent weighted sampling beyond Ioffe)

### API Polish
- [ ] Serialisation / deserialisation (`oxiarc`-compressed binary format) for sketch persistence
- [ ] Builder-style API for HLL (`HllBuilder::precision(p).build()`) and CM (`CmBuilder::epsilon(0.01).delta(0.001).build()`)
- [ ] Cross-link with `oxicuda-stats` for inference on sketch-based estimators
- [ ] Streaming interface trait (`StreamingSketch<T>`) with `update`, `merge`, `query`, `serialize`
