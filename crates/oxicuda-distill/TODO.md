# oxicuda-distill TODO

Pure Rust knowledge distillation primitives for teacher-student training, covering logit-level, feature-level, relation-based, attention-based, online, born-again, and data-free distillation, plus distillation metrics and PTX kernel templates. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.43).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 18,716 SLoC (70 files)**

Current implementation covers the canonical knowledge distillation taxonomy (logit / feature / relation / attention / online / born-again / data-free / metrics), with PTX kernel string templates emitted at runtime for SM 7.5 through SM 10.0.

### Completed [x]

#### Core Infrastructure
- [x] `lib.rs` — Crate root, module declarations, 12 e2e integration tests
- [x] `error.rs` — `DistillError` enum + `DistillResult<T>` alias
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit + Box-Muller), `DistillHandle`
- [x] `ptx_kernels.rs` — 7 GPU kernels × 6 SM versions (75 / 80 / 86 / 89 / 90 / 100)

#### Logit-Based Distillation (logit/)
- [x] `logit/hinton_kd.rs` — `HintonKdConfig {temperature, alpha}`, `softmax_with_temp`, `kl_divergence`, `cross_entropy`, `kd_loss = α·T²·KL + (1−α)·CE`, `kd_loss_batch`
- [x] `logit/dist_distill.rs` — `pearson_corr`, `inter_class_loss`, `intra_class_loss`, `dist_loss(β, γ)`
- [x] `logit/decoupled_kd.rs` — `tckd_loss`, `nckd_loss` (T²·KL on non-target classes), `dkd_loss = α·TCKD + β·NCKD`

#### Feature-Based Distillation (feature/)
- [x] `feature/fitnets.rs` — `FitNetsRegressor {w, b}` linear hint projector (He init), `hint_loss = MSE(proj(s), t)`, `mse`
- [x] `feature/at.rs` — `at_map = Σ_c |F_c|^p`, `l2_normalize`, `at_loss = ||q_s − q_t||²`, `at_loss_batch`
- [x] `feature/pkt.rs` — `cosine_similarity`, `build_affinity_matrix` (row-wise softmax cosine Gram), `pkt_loss = KL(K_t || K_s)`

#### Relation-Based Distillation (relation/)
- [x] `relation/rkd.rs` — Pairwise distances, μ-normalisation, `smooth_l1`, `distance_loss` (upper-triangle), `angle_loss` (500 random triplets), `rkd_loss = λ_d·dist + λ_a·angle`
- [x] `relation/crd.rs` — `CrdMemoryBank` with EMA momentum update, `crd_loss` (InfoNCE cosine pos/neg)
- [x] `relation/cc.rs` — `gram_matrix = F^T · F / n`, `frobenius_norm_sq`, `cc_loss`

#### Attention-Based Distillation (attention/)
- [x] `attention/attn_distill.rs` — `attn_loss` (MSE), `multi_head_attn_loss`, `multi_layer_attn_loss`
- [x] `attention/value_distill.rs` — `value_relation_matrix = softmax(VV^T)`, `value_relation_loss` (MSE) (MiniLM-style)
- [x] `attention/mha_distill.rs` — `head_attn_mse`, `wasserstein_1d` (CDF difference), `mha_distill_loss` (switchable)

#### Online / Mutual Distillation (online/)
- [x] `online/dml.rs` — `dml_peer_loss = CE + mean_peers KL(self || peer)`, `dml_all_losses` (all-pairs over N peers)
- [x] `online/byot.rs` — `BranchClassifier` linear head, `byot_loss` (branch vs deepest teacher KD), `byot_ensemble` (mean logits)
- [x] `online/sd_ema.rs` — `EmaTeacher {params, momentum}`, EMA update `θ_t ← m·θ_t + (1−m)·θ_s`, `ema_loss = α·T²·KL + (1−α)·CE`

#### Born-Again / Iterative Distillation (born_again/)
- [x] `born_again/ban.rs` — `BanGeneration {generation, params}`, `ban_loss` (KD from gen-k to gen-k+1), `ensemble_logits` (mean over generations)
- [x] `born_again/tas.rs` — `CapacityGap {ratio}`, `needs_assistant` (>10× gap), `optimal_assistant_size = √(teacher · student)`, `tas_loss`
- [x] `born_again/progressive.rs` — `ProgressiveConfig {initial_steps, current_steps}`, `next_generation` (halve steps), `consistency_loss` (MSE trajectory), `progressive_distill_step`

#### Data-Free Distillation (data_free/)
- [x] `data_free/dafl.rs` — `DaflGenerator {w1, b1, w2, b2}` (He init, ReLU), `dafl_teacher_loss`, `dafl_info_entropy_loss`, `dafl_activation_loss`, `dafl_total_generator_loss`
- [x] `data_free/zskd.rs` — `dirichlet_sample` (exponential approx + normalize), `class_impression_loss` (CE with Dirichlet target), `synthesize_impression`, `zskd_student_loss`

#### Evaluation Metrics (metrics/)
- [x] `metrics/agreement.rs` — `top_k_agreement`, `cohen_kappa`, `prediction_overlap`
- [x] `metrics/divergence.rs` — `kl_divergence`, `js_divergence = 0.5·KL(p||m) + 0.5·KL(q||m)`, `wasserstein_1d` (sorted L1)
- [x] `metrics/compression.rs` — `param_ratio`, `flops_ratio`, `latency_speedup`, `estimate_lora_flops`

#### GPU PTX Kernels
- [x] `kd_loss_ptx` — Hinton soft-label kernel
- [x] `mse_distill_ptx` — Generic feature-MSE kernel
- [x] `attn_distill_ptx` — Per-head attention MSE
- [x] `at_pool_ptx` — Spatial power-sum pooling
- [x] `dml_loss_ptx` — Mutual peer loss accumulation
- [x] `crd_score_ptx` — Contrastive InfoNCE scoring
- [x] `gram_matrix_ptx` — Correlation-congruence Gram

### Future Enhancements [ ]

#### P0 — Verification on GPU Hardware
- [ ] End-to-end GPU verification of all PTX kernels under Linux + NVIDIA driver 525+
- [ ] Criterion benchmark suite executed on real hardware (currently macOS = `UnsupportedPlatform`)
- [ ] Numerical accuracy harness comparing CPU reference vs GPU PTX path within FP32 tolerance

#### P1 — Algorithm Coverage
- [x] TinyBERT-style multi-stage embedding / attention / hidden-state / prediction distillation pipeline
- [x] CRD multi-positive InfoNCE variant (N positives per anchor)
- [x] Switchable knowledge distillation (SKD) with student-dependent gating
- [x] Self-knowledge distillation via cutmix / mixup consistency
- [x] Decoupled feature-projection-free distillation (RA-DKD)
- [x] Layer-wise adaptive temperature scheduling
- [x] LayerDrop / structured-pruning distillation
- [x] Quantisation-aware distillation (INT8 / FP8 student) (impl `src/losses/qat_distill.rs`; affine INT8 + FP8 e4m3/e5m2 fake-quant, straight-through estimator, KD loss on quantised logits)
- [x] DistWRD (Shen 2022): token-level distribution alignment via Wasserstein divergence; soft label matching beyond KL (impl `src/losses/distwrd.rs` — `DistWrd { lambda_wd, temperature }`, `wasserstein1_cdf`)
- [x] MiniLLM (Gu 2023): KD for LLMs via reverse KL + policy gradient; REINFORCE gradient estimator; avoids mean-seeking mode collapse of forward KL (impl `src/losses/minkd.rs` — `MinKd::{reverse_kl, policy_gradient, policy_gradient_sampled}`)
- [x] Progressive Knowledge Distillation (Wang 2021): curriculum of intermediate checkpoints as teachers; start with checkpoint closest to random init (impl `src/born_again/progressive_kd.rs` — `ProgressiveKdSchedule`, `TeacherCheckpoint`, soft cross-stage blending)
- [x] DFAD (Fang 2022): data-free adversarial distillation; generator + student trained adversarially against frozen teacher; no original training data required (impl `src/data_free/dfad.rs` — `Dfad`, `DfadConfig`, `DfadDims`)

#### P2 — Optimisations and Tooling
- [ ] Fused softmax + KL kernel for Hinton KD (reduce two passes to one)
- [ ] Fused L2-norm + AT-map kernel for AT distillation
- [ ] CUDA-graph capture for repeated distillation step
- [ ] Mixed-precision (FP16 / BF16) variants of every kernel
- [ ] Persistent CTA scheduling for very large class counts (n_classes > 32K) (requires GPU hardware)
- [ ] On-device generator (DAFL / ZSKD) sample synthesis pipeline (requires GPU hardware; CPU generator logic lives in `src/data_free/dafl.rs`, `dafl_deep.rs`, `zskd.rs`)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 523 passing (unit + 12 e2e integration tests in `lib.rs`)
- Warnings: 0 (clippy clean)
- `unwrap()` in production code: 0
- macOS: compiles, runtime returns `UnsupportedPlatform` for GPU launches
- All PTX kernels validated as non-empty strings for SM 75 / 80 / 86 / 89 / 90 / 100

## Performance Targets

Knowledge distillation kernels are bandwidth-limited (softmax / KL / MSE / Gram) and scale with class count, batch size, feature dimension, and number of attention heads.

| Operation | Target Reference | Notes |
|-----------|------------------|-------|
| Hinton KD (B=1024, C=1000, T=4) | ≥ 90% of cuDNN softmax+KL fused kernel | dominated by softmax |
| AT pooling (B=64, C=256, H=W=14) | ≥ 95% of equivalent reduction kernel | bandwidth-limited |
| Gram matrix (D=256, N=2048) | ≥ 90% of cuBLAS GEMM | reuse syrk path |
| CRD InfoNCE (B=256, K=4096) | ≥ 85% of cuBLAS GEMM-then-LSE pipeline | two-stage |
| DML N-peer aggregation (N=4, B=512) | ≥ 90% of N separate KL launches | benefit from fusion |

## Notes

- All algorithms are deterministic given an `LcgRng` seed (Box-Muller for normal sampling, integer LCG for InfoNCE negatives, Dirichlet via exponential approximation).
- The PTX kernel templates are plain `String` returns suitable for JIT compilation via the CUDA driver API; no `nvcc` is required to build or test this crate.
- This crate is **complementary** to `oxicuda-dnn` — high-level distillation glue uses cuDNN-equivalent ops (softmax / matmul / batchnorm) supplied by lower-level crates.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [ ] Validate `kd_loss_ptx(75)` and `crd_score_ptx(75)` against legacy warp-shuffle reductions
- [ ] Verify FP16 storage path (lacks BF16) for Hinton KD on T4

### Ampere (sm_80 / sm_86)
- [ ] `cp.async` staging of teacher logits for fused softmax + KL on A100
- [ ] Tensor-Core (mma.sync) path for Gram matrix on sm_80 (BF16 / TF32 inputs)
- [ ] Per-SM persistent CTA scheduling for very large class counts

### Ada (sm_89)
- [ ] FP8 (e4m3 / e5m2) student feature distillation path for inference distillation
- [ ] Sparse Tensor-Core (2:4) acceleration of Gram matrix when teacher activations are sparsified

### Hopper (sm_90)
- [ ] `wgmma.mma_async` + TMA-based teacher feature load for batched distillation
- [ ] Distributed-shared-memory CRD memory-bank access across CTA cluster
- [ ] Asynchronous transaction barrier for overlapping student forward with teacher prefetch

### Blackwell (sm_100)
- [ ] `tcgen05` tensor memory layout for FP4 / FP6 distillation kernels
- [ ] 5th-generation Tensor Core path for Gram matrix and value-relation distillation

---

## Deepening Opportunities

### Verification Gaps
- [ ] All 7 PTX kernels executed end-to-end on GPU hardware (currently only string-content verified)
- [ ] Numerical equivalence between CPU reference and GPU PTX path within FP32 tolerance for all 7 kernels
- [ ] Performance bench numbers (kd_loss, gram_matrix on A100 / H100) recorded in `benches/distill_ops.rs`

### Algorithmic Deepening
- [x] Born-again iteration up to generation ≥ 5 with empirical convergence study (BAN paper compares 1–6) (impl `src/born_again/ban_multigen.rs` — `BanMultiGen` multi-gen scheduler + `GenerationMetric`, BAN-k ensemble, `simulate_ban_trajectory` with monotone-decreasing inter-generation KL)
- [x] CRD with student-projection MLP head (the original paper uses a 2-layer projector) (impl `src/relation/crd_proj.rs` — `CrdProjectionHead` 2-layer Linear→ReLU→Linear + L2-norm, `crd_proj_loss` InfoNCE)
- [x] DAFL / ZSKD with deeper generator MLPs (current is 2-layer) and label-balanced sampling (impl `src/data_free/dafl_deep.rs` — `DeepGenerator` 3-layer class-conditional MLP, `label_balanced_classes` round-robin + Fisher-Yates, `conditional_one_hot_loss`)
- [x] Progressive distillation with non-uniform step halving schedule (impl `src/born_again/progressive.rs` — `ProgressiveConfig::non_uniform_schedule` geometric `initial→final` ratio)
- [x] RKD angle loss with full triplet enumeration (currently 500 random triplets) (impl `src/relation/rkd_full.rs` — deterministic `full_angle_loss` over all n·(n−1)·(n−2) triplets, guarded `full_rkd_loss`)

### Coverage Gaps vs Literature
- [x] MGD (Masked Generative Distillation) — feature masking + generator (impl `src/feature/mgd.rs`; Yang et al. 2022 ECCV "Masked Generative Distillation")
- [x] FCFD (Feature Compression by Frequency Decomposition) (impl `src/feature/fcfd.rs` — orthonormal 2-D DCT-II per channel, zig-zag low/high frequency banding with independent band weights, `low_band_energy_ratio` diagnostic)
- [x] WCoRD / VID variational mutual-information bounds
- [x] Relational graph distillation with edge-feature aggregation (impl `src/relation/graph_distill.rs`; Liu et al. 2019 CVPR "Knowledge Distillation via Instance Relationship Graph")
- [x] Cross-modal distillation (image teacher → text student, audio → vision) (impl `src/losses/cross_modal.rs` — `CrossModalProjector` shared-space heads, paired cosine/normalised-L2 alignment + CLIP-style cross-modal InfoNCE `cross_modal_loss`)
