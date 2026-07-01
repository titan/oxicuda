# oxicuda-hdc TODO

Pure Rust Hyperdimensional Computing (HDC) / Vector Symbolic Architectures primitives covering binary {±1}, integer (MAP), and complex (FHRR) hypervector models; binding, bundling, and permutation operators; item and associative memory; HD classifiers; record / n-gram / spatial-pattern encoders; Hamming / cosine / Jaccard distance metrics; and capacity-bound analyses, with PTX kernel templates for SM 7.5 through SM 10.0. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.47).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 16,640 SLoC (64 files)**

Current implementation covers the canonical HDC / VSA stack: three hypervector models (binary BSC / integer MAP / complex FHRR), binding / bundling / permutation operators, item memory and associative (Hopfield-style) memory, online error-corrective HD classifier, record / n-gram / spatial-pattern encoders, Hamming / cosine / Jaccard distance metrics, and capacity analyses (Hopfield capacity, bundle SNR, required-dimension birthday-paradox bound).

### Completed [x]

#### Core Infrastructure
- [x] `lib.rs` — Crate root, module declarations
- [x] `error.rs` — `HdcError` enum (16 variants: `ZeroDimension`, `DimensionMismatch`, `EmptyInput`, `ClassNotFound`, `ItemNotFound`, `InvalidNgramOrder`, `InvalidBinaryValue`, `FeatureIndexOutOfRange`, `EmptyItemMemory`, `AssocDimensionMismatch`, `CapacityExceeded`, …) + `HdcResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX LCG with bit-32 extraction for booleans to avoid period-2 low-bit defect), `HdcHandle` with `random_binary_hv` / `random_integer_hv` / `random_complex_hv`
- [x] `ptx_kernels.rs` — 7 GPU kernels × 6 SM versions (75 / 80 / 86 / 89 / 90 / 100); uses string concatenation (not `format!`) to avoid Rust 2024 conflict with PTX `%r` registers
- [x] `e2e_tests.rs` — 22 cross-module integration tests

#### Hypervector Models (vector/)
- [x] `vector/binary.rs` — `Vec<i8>` {±1} (BSC): `random_binary`, `validate_binary`, `binary_dot`, `bipolar_count`, `threshold_binary` (tie-breaking via `LcgRng`)
- [x] `vector/integer.rs` — `Vec<i32>` MAP model: `random_integer` (`rem_euclid(3) − 1` for uniform {−1, 0, +1}), `integer_bind` (element-wise multiplication), `integer_bundle`, `integer_to_binary`, `integer_norm`
- [x] `vector/complex.rs` — FHRR `Vec<f32>` length `2D` interleaved `[re₀, im₀, …]`: `random_complex` (uniform phases), `complex_bind` (element-wise complex multiplication), `complex_conjugate`, `complex_bundle` + normalisation, `complex_cosine`, `complex_normalize`

#### Operators (ops/)
- [x] `ops/binding.rs` — `binary_bind` (sign product `a · b`), `binary_unbind` (same operator: self-inverse), `integer_bind_op`, `circular_convolution` (O(n²)), `circular_correlation` (flipped-a convolution)
- [x] `ops/bundling.rs` — `bundle_binary` (majority vote with `LcgRng` tie-break), `bundle_integer` (element-wise sum), `bundle_complex` (complex sum + normalise), `weighted_bundle_binary`
- [x] `ops/permutation.rs` — `cyclic_shift` / `cyclic_shift_i32` / `cyclic_shift_f32` (left rotate by k), `cyclic_shift_right`, `random_permute`, `random_permutation` (Fisher-Yates), `inverse_permute`

#### Memory (memory/)
- [x] `memory/item_memory.rs` — `ItemMemory`: symbol → HV store, nearest-neighbour query by dot-product, `add_random`, `contains`, `len`
- [x] `memory/assoc_memory.rs` — `AssocMemory` bind-and-superpose Hopfield-style: `store(key, val)` accumulates, `finalize()` thresholds to `i8`, `retrieve(key)` unbinds, `capacity_estimate() = 0.138·D`

#### Classifier (classifier/)
- [x] `classifier/hd_classifier.rs` — `HdClassifier`: per-class `i32` accumulator → thresholded prototype, argmax-cosine classify, error-corrective `online_update`
- [x] `classifier/prototype.rs` — `Prototype`: incremental add / subtract on `i32` accumulator, `build()`, `cosine(query)`

#### Encoding (encoding/)
- [x] `encoding/record.rs` — `RecordEncoder`: `n_features × n_values_per_feature` random HVs; `encode(feature_values) = bundle(bind(feat_hv, val_hv))`
- [x] `encoding/ngram.rs` — `NgramEncoder`: vocabulary HVs + cyclic shift; `encode(tokens) = bundle` over n-gram windows with order-j shifts
- [x] `encoding/pattern.rs` — `PatternEncoder`: row / col HVs; `encode(pixels, threshold) = bundle` active pixel positions; `encode_multilevel` over threshold array

#### Distance Metrics (distance/)
- [x] `distance/hamming.rs` — `hamming_frac = (D − Σaᵢbᵢ) / (2D)`, `hamming_count`, `hamming_similarity_threshold` (`n_sigma / √D`)
- [x] `distance/cosine.rs` — `cosine_binary` via `dot / D`, `cosine_integer`, `cosine_real`, `cosine_complex` via `Re(a · conj(b)) / D`, `argmax_cosine_binary`
- [x] `distance/jaccard.rs` — `jaccard_binary` (set intersection / union), `minihash_similarity` (correlation estimate)

#### Diagnostics (metrics/)
- [x] `metrics/metrics.rs` — `hopfield_capacity = ⌊0.138·D⌋` (Amit-Gutfreund-Sompolinsky), `classification_accuracy`, `bundle_snr = √D / √k`, `required_dimension` (birthday-paradox bound), `average_pairwise_hamming`

#### Empirical Analysis (analysis/)
- [x] `analysis/capacity.rs` — MEASURED scaling-law characterisation on the crate's real primitives: `recall_accuracy` (bind-and-superpose `AssocMemory` + `ItemMemory::query` cleanup recall), `hopfield_capacity_curve`/`CapacityPoint` (capacity vs D — measured ~linear, ratio ≈ 0.105 at 80 % recall / K=10 cleanup, CV ≈ 0.03), `bundle_snr_point`/`bundle_snr_curve`/`SnrPoint` (SNR vs k for majority-vote `bundle_binary` — measured `snr·√k ≈ √(2D/π)`, CV ≈ 0.04, tracks the √(D/k) law)

#### GPU PTX Kernels
- [x] `xor_bind` — Bitwise XOR binding for binary HVs
- [x] `bundle_majority` — Majority-vote bundling
- [x] `cyclic_shift` — Cyclic permutation
- [x] `cosine_sim` — Cosine similarity
- [x] `hamming_dist` — Hamming distance count
- [x] `complex_bind` — FHRR element-wise complex multiplication
- [x] `hd_classify` — Class-prototype argmax-cosine classification

### Future Enhancements [ ]

#### P0 — Verification on GPU Hardware
- [ ] End-to-end GPU verification of all PTX kernels under Linux + NVIDIA driver 525+ (requires GPU hardware)
- [ ] Criterion benchmark suite executed on real hardware (requires GPU hardware)
- [ ] Bit-exact equivalence between CPU reference and GPU PTX path for binary operators (requires GPU hardware)
- [ ] Numerical equivalence within FP32 tolerance for complex / cosine kernels (requires GPU hardware)

#### P1 — Algorithm Coverage
- [x] Sparse binary HVs (k-of-D) with sparse-dot / sparse-bundle (vector/sparse_binary.rs -- Kanerva BSDR; sorted-index `k`-of-`D` codes; O(k) merge sparse-overlap/Jaccard, thinning-union sparse-bundle, modular-sumset sparse-bind/unbind; CPU reference, GPU kernels remain hardware-gated)
- [x] Sparse block-codes (SBC) — block-structured sparse VSA (vector/sparse_block_codes.rs -- Laiho 2015 / Frady 2020 SBC; n_blocks one-hot blocks; block-wise modular bind/unbind; argmax-resparsify bundle)
- [x] Holographic Reduced Representations (HRR) — circular convolution / correlation binding
- [x] HRR with FFT-accelerated binding (replaces O(D²) circular convolution with O(D log D)) (vector/hrr_fft.rs -- Plate 1995 + FFT convolution theorem; radix-2 Cooley-Tukey iterative FFT; O(D log D) circular convolve/correlate; verified against naive O(D²))
- [x] FHRR phasor-only model with explicit unit-magnitude constraint
- [x] Matrix Binding via Tensor Product Representation (ops/tensor_product.rs -- Smolensky 1990; role⊗filler outer-product bind, contract-unbind /‖role‖², bundle superposition, orthonormal-role recovery)
- [x] VSA-based Resonator Networks (decompose superposition into role-filler structure)
- [x] Resonator Network with attention-based unbinding (memory/resonator_attention.rs -- Kent/Frady-Kent-Olshausen-Sommer 2020 soft relaxation; temperature-scaled stable softmax over codebook similarities replacing hard argmax; attention-weighted superposition cleanup; temperature→0 recovers the hard resonator)
- [x] Vector Hetero-associative Memory (key ≠ value space) (memory/hetero_associative.rs -- correlation-matrix key→value memory, Hebbian outer-product + ridge pseudo-inverse exact recall + codebook cosine cleanup; distinct from auto-associative)
- [x] Cleanup memory with iterative refinement (multiple item-memory queries) (memory/cleanup_refine.rs -- VSA fixed-point cleanup over ItemMemory; Replace (hard projection) and Bundle (gradual majority) modes; converges when retrieved id is stable; reports final cosine to the matched prototype)
- [x] Hierarchical / tree-structured HD encoders (encoding/tree_hd.rs -- Kanerva/Plate/Gayler tree encoding; `TreeNode` Leaf/Internal; recursion-by-bundle with per-child-position permutation ρ^{c+1}, plus `encode_paths` composed-shift leaf-path view; structure/order sensitive)
- [x] Continuous-value encoders (thermometer code, fractional binding)
- [x] Spatial encoding for 2-D images via tensor binding of (row, col) positions (encoding/spatial_hd.rs -- Kleyko 2018/Hassan 2021; ρ^{r}(row)⊗ρ^{c}(col) position binding of cell values, majority bundle; shift/transpose sensitive)
- [x] Temporal encoding with continuous time (real-valued time embedding) (encoding/temporal_hd.rs -- Rahimi 2016 + Frady-Kleyko-Sommer fractional power encoding; phasor-only time-embedding θ=t·scale·φ from a shared FPE base, bind with FHRR symbol HVs, bundle events; continuous-time kernel)

#### P1 — Classifier Coverage
- [x] Multi-pass online classifier with explicit forgetting factor (learning/forgetting_hd.rs -- AdaptHD/OnlineHD + EWMA forgetting; real f32 recency-weighted centroids acc=λ·acc+(1−λ)·hv per updated class, multi-epoch fit, real-cosine classify; recency proven by drift test)
- [x] HD-classifier ensemble (bagging over independent random encodings) (classifier/hd_ensemble.rs -- Ho 1998 random-subspace/attribute-bagging; M members each on an independent seed-determined coordinate subspace, majority vote + summed-similarity tie-break)
- [x] HD-classifier with regularisation against rare classes (classifier/rare_class.rs -- Menon 2021 logit adjustment / Cui 2019 class-balanced; unit-L2 prototypes + inverse-frequency bias α·ln(N/N_c) added to cosine, unseen classes never predicted)
- [x] HD-classifier export / import (model persistence) (classifier/persistence.rs -- pure-Rust HdModel serialization; HDC1 magic + LE header, MSB-first bit-packed ±1 prototypes (ceil(D/8) B/class), bounds-safe from_bytes + +/- text form, exact round-trip)
- [x] Confidence-calibrated cosine output (Platt scaling) (classifier/platt.rs -- Platt 1999 + Lin-Lin-Weng 2007; numerically-stable Newton fit of P=1/(1+exp(A·s+B)) with safe targets + backtracking line search; predict_proba)

#### P2 — Optimisations and Tooling
- [x] Adaptive HD Learning (`learning/adaptive_hd.rs`) — Imani 2019: online retraining with per-class accumulated misclassification retraining for improved accuracy; `AdaptiveHdClassifier`
- [x] Graph HD Encoding (`encoding/graph_hd.rs`) — Poduval 2022 DAC: graph-structure-aware HD representation via vertex-HV binding with role-filler permuted edge encoding and graph-level bundle; `GraphHdEncoder`
- [x] Sequence HD Encoding (`encoding/sequence_hd.rs`) — Kanerva 2009: n-gram-permutation sequence encoding with position-dependent circular shift permutation for variable-length sequences; `SequenceHdEncoder` (position-bundle `encode` + bound-`k`-gram `encode_ngrams`, order sensitive)
- [x] HD Regression (`learning/hd_regression.rs`) — Hersche 2023 NeurIPS: least-squares regression in HD space via pseudoinverse of bundled prototype matrix; `HdRegressor`
- [ ] Fused bind + bundle kernel (saves intermediate HV materialisation) (requires GPU hardware)
- [ ] Fused bind + cosine kernel for one-shot similarity query (requires GPU hardware)
- [ ] Persistent CTA scheduling for very large item-memory NN search (requires GPU hardware)
- [ ] CUDA-graph capture for repeated record encoding inference (requires GPU hardware)
- [ ] Mixed-precision (FP16 storage / FP32 accumulate) complex FHRR (requires GPU hardware)
- [ ] Bit-packed binary HV storage (1 bit per element, AVX-like popcount on GPU) (requires GPU hardware; CPU bit-packing exists in classifier/persistence.rs)
- [ ] On-device random HV generation with Philox / ChaCha20 RNG (requires GPU hardware)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 609 unit + 1 doctest passing (incl. 22 e2e integration tests in `e2e_tests.rs`, 6 empirical scaling-law tests in `analysis/capacity.rs`)
- Warnings: 0 (`cargo clippy -p oxicuda-hdc --all-features --all-targets -- -D warnings` clean)
- `unwrap()` in production code: 0 (only `#[cfg(test)]` uses `.expect`)
- macOS: compiles, runtime returns `UnsupportedPlatform` for GPU launches
- All PTX kernels validated as non-empty strings for SM 75 / 80 / 86 / 89 / 90 / 100

## Performance Targets

HDC kernels are dominated by element-wise reductions over very large D (typically D = 1K to 100K). The XOR-bind kernel can use popcount intrinsics, and the cosine / bundle kernels are bandwidth-bound.

| Operation | Target Reference | Notes |
|-----------|------------------|-------|
| Binary bind (D=10K) | ≥ 95% of cuBLAS-equivalent element-wise mul | bandwidth-bound |
| Bundle 16× (D=10K) | ≥ 90% of cuBLAS-equivalent reduce-add | reduce-bound |
| Cosine binary (D=10K) | ≥ 95% of cuBLAS dot | reduction-bound |
| Hamming distance (D=10K) | ≥ 95% of bitwise XOR + popcount | popcount-bound |
| HD classify (k=10 classes, D=10K) | ≥ 90% of k-cuBLAS dot | k dot-products |
| Complex bind FHRR (D=10K) | ≥ 90% of cuBLAS-equivalent complex mul | 4-way complex mul |

## Notes

- All randomness flows through `LcgRng` for deterministic replay. Bit-32 extraction is used for boolean sampling because the low bits of an MMIX LCG have period 2.
- Hopfield capacity `0.138·D` matches Amit et al. (1985) for binary BSC; for FHRR the empirical capacity is closer to `0.5·D` (P2 documentation).
- The HD classifier uses error-corrective online update: on misclassification, subtract from the predicted prototype and add to the correct prototype.
- The associative memory uses bind-and-superpose: `M = Σᵢ bind(kᵢ, vᵢ)`; retrieval is `v̂ = unbind(M, k_query)` followed by item-memory cleanup.
- FHRR uses interleaved `[re, im]` storage; explicit phasor-only (unit-magnitude) renormalisation is P1.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [ ] Validate XOR-bind and cosine kernels on T4
- [ ] Popcount intrinsics for Hamming distance on Turing+

### Ampere (sm_80 / sm_86)
- [ ] `cp.async` staging of class prototypes for HD classify on A100
- [ ] Persistent CTA scheduling for repeated item-memory NN search
- [ ] Tensor-Core (mma.sync) acceleration of cosine-similarity batch query

### Ada (sm_89)
- [ ] FP8 (e4m3 / e5m2) storage for FHRR with FP32 accumulation
- [ ] Sparse Tensor-Core path for sparse-binary VSA

### Hopper (sm_90)
- [ ] TMA-based bulk class-prototype staging for very large class counts
- [ ] `wgmma.mma_async` for batched cosine-similarity computation
- [ ] Distributed shared memory across CTA cluster for distributed item-memory

### Blackwell (sm_100)
- [ ] `tcgen05` tensor memory layout for FP4 / FP6 FHRR
- [ ] 5th-generation Tensor Core for batched cosine similarity at low precision

---

## Deepening Opportunities

### Verification Gaps
- [ ] All 7 PTX kernels executed end-to-end on GPU hardware (currently only string-content verified)
- [ ] Bit-exact equivalence between CPU reference and GPU PTX path for binary operators
- [ ] Numerical equivalence within FP32 tolerance for complex / cosine kernels
- [ ] Benchmark numbers (binary_bind, cosine, bundle_16x on A100 / H100) recorded in `benches/hdc_ops.rs`
- [ ] Item-memory NN-query latency curve vs memory size documented (D, |memory|)

### Algorithmic Deepening
- [x] Hopfield-capacity empirical verification curve (capacity vs D) (analysis/capacity.rs:hopfield_capacity_curve,recall_accuracy -- measured bind-and-superpose memory + cleanup recall; capacity grows ~linearly in D, capacity/D ≈ 0.105 (CV ≈ 0.03) at 80 % item-recall / K=10 cleanup, inside the documented Hopfield 0.10–0.18 ball-park vs known law)
- [x] Bundle SNR empirical verification (SNR vs k for binary majority vote) (analysis/capacity.rs:bundle_snr_curve,bundle_snr_point -- measured majority-vote bundle; SNR decreases monotonically with k and snr·√k ≈ √(2D/π) (CV ≈ 0.04), tracking the √(D/k) law within tolerance)
- [ ] Online classifier convergence on a benchmark dataset (e.g., language ID, EMG gesture)
- [ ] n-gram encoder used for language classification with empirical accuracy report
- [ ] Pattern encoder used for MNIST classification with empirical accuracy report
- [x] Resonator network decomposition of structured superposition (memory/resonator.rs -- Frady-Kent-Olshausen-Sommer 2020 hard-argmax fixed-point resonator decomposing Σ bind(filler,role) into role-filler pairs; 1/2/3-role recovery tests; soft variant in memory/resonator_attention.rs)

### Coverage Gaps vs Literature
- [x] Tensor Product Representations (TPR) — Smolensky's original VSA (ops/tensor_product.rs -- role⊗filler outer-product bind, contract-unbind, orthonormal-role recovery)
- [x] Binary Sparse Distributed Representations (Kanerva BSDR) (vector/sparse_binary.rs -- `k`-of-`D` sorted-index codes; sparse overlap/Jaccard/bind/unbind/thinning-bundle)
- [x] Permutation-based binding (Plate's HRR with permutation matrices) (ops/permutation_bind.rs -- Gayler 2003/Kanerva 2009; `PermutationRole` stored permutation as binding ρ(f)=Pf, inverse-perm unbind Pᵀ, i8+i32, bind-superpose/recover over the orthogonal permutation matrices)
- [x] Vector Function Architecture (Frady-Sommer 2019) for continuous functions (vector/fpe.rs -- fractional power encoding base^v, bind(enc(a),enc(b))=enc(a+b), locality-preserving similarity kernel)
- [x] HDC for time-series classification (sliding-window n-gram encoders) (encoding/sequence_hd.rs `encode_ngrams` + encoding/ngram.rs -- sliding-window bound k-grams, order sensitive)
- [x] HDC for graph classification (graph-embedding via random walk bundling) (encoding/graph_hd.rs -- Poduval 2022 DAC; vertex-HV binding with role-filler permuted edge encoding and graph-level bundle; `GraphHdEncoder`)
- [x] HDC for genomics (k-mer encoding + bundling for sequence comparison) (encoding/kmer.rs -- GenieHD/Kim 2020; per-symbol HVs, order-sensitive ρ-shifted k-mer binding, sliding-window bundle, alignment-free cosine similarity; DNA convenience ctor + char parser)
- [x] Federated HD learning (gradient-free DP-compatible aggregation) (learning/federated_hd.rs -- FedHD/Imani; per-client i32 prototype accumulators, server element-wise sum (== centralized exactly), L∞ clip + bounded-uniform DP-style noise; honestly not a proven (ε,δ) guarantee)
- [x] HDC + neural-network hybrid (HD features as input to a small MLP) (learning/hd_mlp.rs -- HD ±1 features → 1-hidden-layer tanh MLP with hand-derived softmax-cross-entropy backprop, SGD, Box-Muller init; training loss decreases, separable accuracy 1.0)
