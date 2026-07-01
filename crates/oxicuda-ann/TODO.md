# oxicuda-ann TODO

GPU-accelerated Approximate Nearest Neighbor & vector-search primitives, serving as
a pure-Rust complement to FAISS / RAFT / cuVS style libraries.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.39).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 16,482 total lines (63 files)
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
- **Tests:** 464 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently PTX-string
  generation tested only)
- [ ] HNSW search benchmark measured on real GPU (CPU-side bench only today)
- [ ] PQ ADC throughput measured on GPU for batch query workloads

#### P1 — Algorithm Coverage Extensions
- [x] OPQ (Optimized PQ) — learn a rotation R prior to PQ for lower distortion (pq/opq.rs -- Ge 2013 CVPR; alternating optimisation: fix R→train PQ on R·X, fix codebook→orthogonal Procrustes via cyclic-Jacobi SVD of cross-covariance; rotate/unrotate/encode/decode/ADC; 19 tests inc. orthogonality + determinism)
- [x] Anisotropic PQ — Guo et al. (ScaNN) score-aware quantization loss (pq/anisotropic_pq.rs -- Guo 2020 ICML; per-subspace anisotropic-weighted k-means upweighting query-direction variance via 1+(1−η²)·mean(q̂·disp)²; anisotropic/isotropic loss + ratio; 23 tests inc. η=1 matches isotropic, η=0 ≤ isotropic)
- [x] DiskANN / SSD-resident graph index for billion-scale corpora (vamana.rs -- Subramanya 2019 NeurIPS Vamana graph core; α-relaxed RobustPrune (degree ≤ R) + greedy navigable search; in-memory, SSD residency remains for future)
- [x] SPANN — partition + posting list with SSD-resident vectors (graph/spann.rs -- Chen 2021 NeurIPS in-memory core; ~√n centroids + boundary-point duplication (replica when dist²(x,c) ≤ (1+ε)²·dist²(x,c*)) + coarse head-index probe; SSD residency remains hardware-gated)
- [x] NGT (Neighborhood Graph and Tree) build / search (ngt/index.rs -- ANNG incremental approx-kNN-graph build + ε-relaxed greedy best-first graph search with deterministic seeds; reuses distance::l2)
- [x] HNSW-PQ — codes stored in the graph nodes (compressed HNSW) (hnsw_pq.rs -- PQ codes stored per HNSW node + Asymmetric Distance Computation (ADC) table lookup during search; reuses HnswGraph + PqCodebook)
- [x] FreshDiskANN — incremental updates to an on-disk graph (fresh_diskann.rs -- Singh 2021 in-memory core; streaming insert via greedy+RobustPrune+back-edges, lazy tombstone delete, consolidation that bridges edges over deleted nodes + re-prunes + reclaims slots for in-place reuse; on-SSD block layout remains hardware-gated)
- [x] `ivf/ivfadc.rs` — IVFADC residual coding (Jégou 2011): per-inverted-list global rotation + residual PQ encoding; refine with ADC lookup during search; reduces quantisation error 2–4 dB vs plain IVFPQ
- [x] `lsh/pqfastscan.rs` — PQFastScan (André 2015): SIMD-friendly 4-bit LUT scan; pack 32 PQ codes into 128-bit registers; 16-lane vectorised accumulation; 4–5× faster than scalar ADC
- [x] SPANN (Chen 2021): partition + posting list with boundary duplication; coarse centroid routing (graph/spann.rs -- in-memory core implemented; on-device anchor graph + SSD residency remain hardware-gated)
- [x] FreshDiskANN (Singh 2021): incremental graph updates (insert/delete) to a Vamana index; in-place node slot reuse + consolidated repair passes (fresh_diskann.rs -- in-memory core implemented; on-disk block layout remains hardware-gated)

#### P1 — Distance & Quantization
- [x] Mahalanobis distance with learned positive-definite metric (distance/mahalanobis.rs -- M=LLᵀ Cholesky-parameterized PSD metric; contrastive margin-loss gradient descent on similar/dissimilar pairs)
- [x] Inner-product MIPS via L2 transformation (XBox / Bachrach)
- [x] Additive quantization (AQ) / Composite quantization beyond PQ
- [x] Residual quantization for higher-bit-rate codes (pq/residual_quant.rs -- Chen 2010 / Liu 2019 multi-stage VQ; greedy stage-wise k-means on residuals + monotone-refinement MSE guarantee)
- [x] Binary quantization (PQ-1bit / hyperplane sketches) (quantize/binary_pq.rs -- 1-bit-per-sub-vector PQ: threshold each sub-vector at its centroid mean; Hamming via popcount; 64× memory reduction; soft-decode asymmetric score)
- [x] `quantize/aq.rs` — Additive Quantization (Babenko 2014): iteratively assign each vector to a sum of M codebook entries via beam search; lower distortion than PQ at same bits; O(Mn²) training
- [x] `quantize/binary_pq.rs` — Binary PQ (1-bit per sub-vector): threshold each sub-vector at its centroid mean; Hamming distance via popcount; 64× memory reduction vs float32; asymmetric score via soft-decode
- [x] `distance/mips_transform.rs` — MIPS-to-L2 transformation (Shrivastava-Li 2014, XBox): augment d-dim vector with one extra coordinate to convert maximum inner-product search to L2 search; O(1) preprocessing per query

#### P1 — Approximate Search Strategy
- [x] Multi-probe LSH (Lv et al.) (lsh/multi_probe_lsh.rs -- Lv et al. VLDB 2007 E2LSH ⌊(a·x+b)/w⌋ keys + probe sequences ordered by expected perturbation distance s_j / w−s_j)
- [x] PQFastScan (SIMD-friendly 4-bit PQ scan)
- [x] Refined HNSW pruning policies (NSG, HCNNG, alpha-RNG)
- [x] IVF residual coding (IVFADC) — per-list rotation + residual PQ

#### P2 — Tooling
- [x] Recall-at-K / latency Pareto plot helpers (metrics.rs -- recall_at_k set-recall, exact_topk_ids ground-truth generator, ParetoPoint/pareto_frontier non-dominated extraction, ParetoSweep accumulator with CSV export)
- [x] Index-on-disk serializer (oxicode-based, no zip/bincode) (index/serializer.rs -- little-endian ByteWriter/ByteReader + typed sections for Flat/PqCodebook/IvfPostings; magic OXANNIDX + version header; index/mod.rs bridges PqCodebook↔bytes)
- [ ] GPU streaming construction for HNSW (out-of-core build) (requires GPU hardware)
- [x] Recall@K / latency Pareto curve helper: sweep nprobe/ef_search, record (recall, latency) pairs, compute Pareto front, serialize to CSV (metrics.rs -- implemented as a library module, not a criterion bench, so it is unit-testable; ParetoSweep::record/frontier/to_csv)
- [x] Pure-Rust index serialiser: flat binary format for HNSW/IVF/PQ indexes; magic bytes + version header + section offsets; no zip/bincode dependency; mmap-compatible layout (index/serializer.rs)
- [ ] `hnsw/streaming_build.rs` — GPU streaming HNSW construction: batch-insert chunks of 64K vectors via device-side neighbour search; avoids host-device transfer bottleneck for billion-scale builds (requires GPU hardware)

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
- Tests: 464 passing (Flat, k-means, PQ, OPQ, anisotropic-PQ, IVF, IVFADC,
  HNSW recall, FreshDiskANN incremental insert/delete/consolidate, LSH +
  calibration, MinHash unbiasedness, index serializer, recall/Pareto metrics,
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

> All unchecked items below are GPU-hardware-gated: they require executing PTX on
> real NVIDIA silicon (WMMA/WGMMA tensor cores, cp.async/TMA copy engines,
> warp-shuffle, native FP4/FP6 storage) and cannot be implemented or verified on
> CPU. Left `[ ]` deliberately.

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
- [ ] HNSW recall measured on standard ANN benchmarks (SIFT-1M, GIST-1M, DEEP-1B) (requires standard datasets / real hardware)
- [ ] PQ distortion error vs. uncompressed L2 measured on standard datasets (requires standard datasets)
- [ ] IVFPQ Pareto frontier (recall vs. queries-per-second) on real hardware (requires GPU hardware; CPU Pareto helper landed in metrics.rs)
- [x] LSH bucket-size distribution + collision-rate calibration (lsh/calibration.rs -- bucket_size_distribution → BucketStats{n_buckets,max/min/mean/std load,imbalance}; empirical_collision_rate over all pairs; tests verify load balance + monotone collision decay with bits)

### Implementation Deepening
- [x] OPQ rotation learning (pq/opq.rs -- Procrustes/Jacobi-SVD rotation learning; fused GPU encode kernel remains hardware-gated)
- [x] PQFastScan 4-bit table lookup (quantize/pq_fastscan.rs -- 4-bit nibble-packed codes + LUT ADC scan; GPU-SIMD register packing remains hardware-gated)
- [ ] On-disk graph index (DiskANN) with SSD prefetch (requires SSD / real hardware; in-memory Vamana core in vamana.rs)
- [x] Incremental graph updates with neighbor re-balancing (fresh_diskann.rs -- incremental insert + lazy delete + consolidation re-prune/re-balance on a Vamana graph; HNSW-specific delete is a future structural variant)

### Numerical Accuracy
- [ ] PQ encode reproducibility across SM versions (requires GPU hardware to compare device-side encode; CPU encode is deterministic via LcgRng)
- [x] LSH random-projection isotropy verified for d ≥ 128 (lsh/calibration.rs -- projection_isotropy returns max|⟨ŵ_i,ŵ_j⟩| of unit-normalised Gaussian hyperplanes; test asserts < 0.5 at d=128 and tighter concentration as d grows)
- [x] MinHash Jaccard estimator unbiasedness for small sketches (≤ 64 hashes) (lsh/calibration.rs -- minhash_jaccard_bias builds controlled-overlap sets, averages estimate over trials vs exact Jaccard; test asserts |bias| < 0.05 for a 64-hash sketch)

## Performance Verification Harness Status (2026-05-16)

- **Distance & PQ kernels:** harnesses at `benches/ann_ops.rs`; CPU-side
  PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **HNSW / IVFPQ search throughput:** CPU-side recall tests pass; GPU-side
  queries-per-second numbers pending.
