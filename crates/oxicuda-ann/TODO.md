# oxicuda-ann TODO

GPU-accelerated Approximate Nearest Neighbor & vector-search primitives, serving as
a pure-Rust complement to FAISS / RAFT / cuVS style libraries.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.39).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 3,094 total lines (2,624 code, 38 files)
- **Coverage:** brute-force flat index baseline, bounded top-K heap selection,
  mini-batch k-means++ trainer, Product Quantization (PQ) with asymmetric distance
  computation (ADC), IVF coarse quantizer, IVFPQ coarse-prune + re-rank, HNSW
  (Malkov & Yashunin 2018 with neighbor-selection heuristic), locality-sensitive
  hashing (random-projection cosine, MinHash Jaccard, SimHash), NN-Descent k-NN graph
  build, scalar quantizers (SQ8 / packed SQ4), and PTX kernel-string generation for
  6 SM tiers.

### Completed

#### Core Infrastructure
- [x] error.rs — `AnnError`, `AnnResult<T>`
- [x] handle.rs — `LcgRng` deterministic PRNG, `SmVersion` PTX target descriptor

#### Distance Metrics (distance/)
- [x] l2.rs — `l2_sq`, `l2`, `l2_sq_all` (batched query→corpus)
- [x] inner_product.rs — `ip`, `cosine_sim`
- [x] hamming.rs — `hamming_u32`, `hamming_f32_packed`

#### Flat (Brute-Force) Index (flat/)
- [x] flat.rs — `FlatIndex { dim, vectors }`, `add`, `search_l2`, `search_ip`
  (exact baseline, min-heap top-K)

#### Top-K Selection (topk/)
- [x] heap.rs — `BoundedMaxHeap` (k-smallest via max-heap eviction)
- [x] select.rs — `select_topk` utility selection helper

#### k-Means Clustering (kmeans/)
- [x] kmeans.rs — `KMeans`, `kmeans_pp_init`, mini-batch fit (25-epoch default)

#### Product Quantization (pq/)
- [x] codebook.rs — `PqCodebook { m, dsub, ksub, centroids }`
- [x] train.rs — `train_pq` per-subspace k-means
- [x] encode.rs — `encode_vector`, `encode_batch` to per-subspace codes
- [x] adc.rs — `build_adc_table`, `adc_distance` asymmetric distance computation

#### Inverted-File Index (ivf/, ivfpq/)
- [x] ivf/train.rs — IVF coarse quantizer training (k-means on the dataset)
- [x] ivf/ivf.rs — `IvfIndex` with posting lists, `search` top-nprobe lists
- [x] ivfpq/ivfpq.rs — `IvfPq` coarse-prune then ADC re-rank within top-nprobe lists

#### HNSW Graph (hnsw/)
- [x] graph.rs — `HnswGraph { dim, M, ef_construction, ef_search }`
- [x] insert.rs — `hnsw_insert` level draw + `select_neighbors_heuristic`
- [x] search.rs — `hnsw_search` greedy descent + ef-bounded BFS frontier

#### Locality-Sensitive Hashing (lsh/)
- [x] random_proj.rs — `RandomProjLsh` sign-bit cosine LSH
- [x] minhash.rs — `MinHash` Jaccard signature via LCG hash families
- [x] simhash.rs — `SimHash` cosine similarity via Gaussian projections

#### k-NN Graph (knn_graph/)
- [x] knn_graph.rs — `KnnGraph`, `build_brute` O(n²) baseline,
  `build_nn_descent` iteration with sampling

#### Scalar Quantizers (quantize/)
- [x] sq8.rs — `Sq8Quantizer` per-dim min/max → uint8
- [x] sq4.rs — `Sq4Quantizer` 4-bit nibble-packed

#### PTX Kernel Generation (ptx_kernels.rs)
- [x] 7 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `l2_distance_batch`, `ip_distance_batch`, `pq_adc_table`, `hnsw_neighbor_eval`,
  `ivf_assign`, `lsh_random_proj`, `topk_select`

#### Tests & Benchmarks
- [x] 12 end-to-end tests in `lib.rs::e2e_tests` (Flat exact-match, top-K count,
  k-means two-cluster, PQ training, PQ encode+ADC, IVF search, HNSW self-find,
  HNSW recall ≥ 80% vs brute-force, LSH determinism, MinHash Jaccard,
  NN-Descent quality, PTX non-empty × all SM versions)
- [x] Benchmarks (`benches/ann_ops.rs`) — 7 PTX kernel groups × 4 SM
  + 5 algorithm benches
- **Tests:** 32 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently PTX-string
  generation tested only)
- [ ] HNSW search benchmark measured on real GPU (CPU-side bench only today)
- [ ] PQ ADC throughput measured on GPU for batch query workloads

#### P1 — Algorithm Coverage Extensions
- [ ] OPQ (Optimized PQ) — learn a rotation R prior to PQ for lower distortion
- [ ] Anisotropic PQ — Guo et al. (ScaNN) score-aware quantization loss
- [x] DiskANN / SSD-resident graph index for billion-scale corpora (vamana.rs -- Subramanya 2019 NeurIPS Vamana graph core; α-relaxed RobustPrune (degree ≤ R) + greedy navigable search; in-memory, SSD residency remains for future)
- [ ] SPANN — partition + posting list with SSD-resident vectors
- [x] NGT (Neighborhood Graph and Tree) build / search (ngt/index.rs -- ANNG incremental approx-kNN-graph build + ε-relaxed greedy best-first graph search with deterministic seeds; reuses distance::l2)
- [x] HNSW-PQ — codes stored in the graph nodes (compressed HNSW) (hnsw_pq.rs -- PQ codes stored per HNSW node + Asymmetric Distance Computation (ADC) table lookup during search; reuses HnswGraph + PqCodebook)
- [ ] FreshDiskANN — incremental updates to an on-disk graph

#### P1 — Distance & Quantization
- [x] Mahalanobis distance with learned positive-definite metric (distance/mahalanobis.rs -- M=LLᵀ Cholesky-parameterized PSD metric; contrastive margin-loss gradient descent on similar/dissimilar pairs)
- [ ] Inner-product MIPS via L2 transformation (XBox / Bachrach)
- [ ] Additive quantization (AQ) / Composite quantization beyond PQ
- [x] Residual quantization for higher-bit-rate codes (pq/residual_quant.rs -- Chen 2010 / Liu 2019 multi-stage VQ; greedy stage-wise k-means on residuals + monotone-refinement MSE guarantee)
- [ ] Binary quantization (PQ-1bit / hyperplane sketches)

#### P1 — Approximate Search Strategy
- [x] Multi-probe LSH (Lv et al.) (lsh/multi_probe_lsh.rs -- Lv et al. VLDB 2007 E2LSH ⌊(a·x+b)/w⌋ keys + probe sequences ordered by expected perturbation distance s_j / w−s_j)
- [ ] PQFastScan (SIMD-friendly 4-bit PQ scan)
- [ ] Refined HNSW pruning policies (NSG, HCNNG, alpha-RNG)
- [ ] IVF residual coding (IVFADC) — per-list rotation + residual PQ

#### P2 — Tooling
- [ ] Recall-at-K / latency Pareto plot helpers
- [ ] Index-on-disk serializer (oxicode-based, no zip/bincode)
- [ ] GPU streaming construction for HNSW (out-of-core build)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA-SDK / nvcc / FAISS / cuVS dependency — PTX kernels are emitted as strings.
No oxicuda-driver / -memory / -launch dependency at this layer; higher-level
integrators perform the actual GPU launch.

## Quality Status

- Warnings: 0 (clippy clean, workspace lints inherited)
- Tests: 32 passing (Flat, k-means, PQ, IVF, HNSW recall, LSH, MinHash,
  NN-Descent, PTX × 6 SM)
- unwrap() calls: 0 in production code
- macOS: compiles but returns `UnsupportedPlatform` at runtime when actual launch
  is attempted (PTX emission still works on every host)
- Refactoring policy: every source file is well under 2,000 lines

## Performance Targets

| Workload | Target |
|----------|--------|
| L2 distance batch (n=1M, d=128) | ≥ 90% of cuVS / FAISS-GPU throughput |
| IVFPQ search (nprobe=16, top-100) | ≥ 85% of FAISS-GPU |
| HNSW search (M=32, ef_search=64) | ≥ 80% of CPU reference (HNSWLib) |
| PQ ADC table build | memory-bandwidth bound |

Performance harnesses are CPU-side today; GPU-side numbers will be filled in once
the Linux+NVIDIA verification run is executed.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/ann_ops.rs`) — 7 PTX kernel groups × 4 SM
  + algorithm benches

---

## Notes

- All vectors are FP32 today. Half-precision (FP16/BF16) is a future option.
- HNSW level draw uses `floor(-ln(unif(0,1)) * mL)` with `mL = 1/ln(M)`.
- `select_neighbors_heuristic` follows the Algorithm 4 of Malkov & Yashunin 2018
  (keep_pruned_connections variant).
- NN-Descent uses a simple sampling-based update; full ε-greedy reversed-list
  variant is a future option.
- `LcgRng` is reproducible but not cryptographic; used for k-means++ seeding,
  LSH random projections, MinHash hash families, and HNSW level draws.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX target string emitted for all 7 kernels
- [ ] WMMA m16n16k16 distance accumulation paths for FP16 corpora

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX target string emitted
- [ ] `cp.async` global→shared prefetch for distance-batch kernel
- [ ] Shared-memory bank-conflict-free PQ codebook tile layout
- [ ] Warp-shuffle top-K selection (block-level radix select)

### Hopper (sm_90)
- [x] PTX target string emitted
- [ ] TMA-based bulk corpus loading for billion-vector flat scans
- [ ] WGMMA-based fused distance + projection for binary / cosine indices

### Blackwell (sm_100)
- [x] PTX target string emitted
- [ ] Native FP4/FP6 codebook storage for ultra-compressed PQ

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage and PTX-string generation.
> These items represent the gap between current depth and full
> production-grade GPU ANN search.

### Verification Gaps
- [ ] HNSW recall measured on standard ANN benchmarks (SIFT-1M, GIST-1M, DEEP-1B)
- [ ] PQ distortion error vs. uncompressed L2 measured on standard datasets
- [ ] IVFPQ Pareto frontier (recall vs. queries-per-second) on real hardware
- [ ] LSH bucket-size distribution + collision-rate calibration

### Implementation Deepening
- [ ] OPQ rotation learning + fused encode kernel
- [ ] PQFastScan SIMD-friendly 4-bit table lookup
- [ ] On-disk graph index (DiskANN) with SSD prefetch
- [ ] Incremental HNSW updates with neighbor re-balancing

### Numerical Accuracy
- [ ] PQ encode reproducibility across SM versions
- [ ] LSH random-projection isotropy verified for d ≥ 128
- [ ] MinHash Jaccard estimator unbiasedness for small sketches (≤ 64 hashes)

## Performance Verification Harness Status (2026-05-16)

- **Distance & PQ kernels:** harnesses at `benches/ann_ops.rs`; CPU-side
  PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **HNSW / IVFPQ search throughput:** CPU-side recall tests pass; GPU-side
  queries-per-second numbers pending.
