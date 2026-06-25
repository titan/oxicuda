# oxicuda-recsys TODO

GPU-accelerated recommender-system primitives, covering classical matrix factorization,
neural collaborative filtering, two-tower retrieval, feature-crossing models,
sequence-aware recommenders, graph-based recommenders, multi-task learners, negative
sampling, and ranking metrics.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.40).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~17,400 total lines (64 source files)
- **Coverage:** ALS implicit-feedback, BPR pairwise ranking, NMF multiplicative
  updates; Neural CF (GMF ⊕ MLP); Two-Tower DSSM; DeepFM (linear + 2nd-order FM
  + Deep MLP); AutoInt multi-head self-attention over field embeddings;
  Wide & Deep; GRU4Rec full GRU cell; SASRec causal self-attention;
  BERT4Rec bidirectional MLM; LightGCN symmetric-normalized propagation;
  NGCF interaction-aware aggregation; MMoE / PLE / ESMM multi-task heads;
  uniform / popularity-biased / hard-negative / adaptive-importance samplers;
  sparse-gradient row-wise AdamW embedding optimizer; CL4SRec & DuoRec
  contrastive sequence models; content-based cold-start fallback + switching
  hybrid; ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K, MRR,
  HitRate@K, AUC); calibration metrics (ECE/MCE/Brier/log-loss + group
  calibration disparity); and PTX kernel-string generation for 6 SM tiers.

### Completed

#### Core Infrastructure
- [x] error.rs — `RecSysError`, `RecSysResult<T>`
- [x] handle.rs — `LcgRng` deterministic PRNG, `SmVersion` PTX target descriptor

#### Classical Matrix Factorization (factorization/)
- [x] als.rs — `Als` implicit-feedback factorization with Gauss-Jordan closed-form
  solve (c_ui = 1 + α · r_ui)
- [x] bpr.rs — `Bpr` triplet-loss SGD with σ(x_ui − x_uj) gradient
- [x] nmf.rs — `Nmf` multiplicative-update W / H rules

#### Neural Collaborative Filtering (ncf/)
- [x] ncf.rs — `Ncf` GMF element-wise product ⊕ MLP concat → sigmoid

#### Two-Tower Retrieval (two_tower/)
- [x] two_tower.rs — `TwoTower` dual MLP user / item encoders + dot-product score

#### Feature-Crossing Models (deepfm/)
- [x] deepfm.rs — `DeepFm` linear + FM second-order
  0.5 · ((Σe)² − Σe²) + Deep MLP
- [x] autoint.rs — `AutoInt` multi-head self-attention over field embeddings
  + residual
- [x] wide_deep.rs — `WideDeep` linear ⊕ MLP

#### Sequence-Aware Recommenders (sequential/)
- [x] gru4rec.rs — `Gru4Rec` full GRU cell (z / r / n gates)
- [x] sasrec.rs — `SasRec` causal self-attention + FFN + LayerNorm
- [x] bert4rec.rs — `Bert4Rec` bidirectional attention + MLM masking

#### Graph-Based Recommenders (graph_recsys/)
- [x] lightgcn.rs — `LightGcn` D⁻½ A D⁻½ propagation + layer-mean pooling
- [x] ngcf.rs — `Ngcf` LeakyReLU aggregation + concatenated layers

#### Multi-Task Learners (multitask/)
- [x] mmoe.rs — `Mmoe` per-task softmax gates over shared experts
- [x] ple.rs — `Ple` cascaded shared + task-specific expert layers
- [x] esmm.rs — `Esmm` pCTR × pCVR = pCTCVR product head

#### Negative Sampling (sampling/)
- [x] uniform_neg.rs — `UniformNegSampler` rejection (max 100 tries)
- [x] popularity_neg.rs — `PopularityNegSampler` CDF binary search
- [x] hard_neg.rs — `HardNegSampler` top-20% non-positive pool

#### Ranking Metrics (metrics/)
- [x] recsys_metrics.rs — `precision_at_k`, `recall_at_k`, `ndcg_at_k`
  (DCG / IDCG log2), `map_at_k`, `mrr`, `hit_rate_at_k`,
  `auc_recsys` (Wilcoxon-Mann-Whitney with tie handling)

#### PTX Kernel Generation (ptx_kernels.rs)
- [x] 7 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `als_step_ptx` (Cholesky-style), `bpr_grad_ptx`, `embedding_lookup_ptx`,
  `dot_score_ptx`, `softmax_topk_ptx`, `negsample_uniform_ptx`,
  `lightgcn_propagate_ptx`

#### Tests & Benchmarks
- [x] 12 end-to-end tests in `lib.rs::e2e_tests` (ALS score finite, BPR loss
  finite, NMF fit, NCF in [0,1], TwoTower score finite, DeepFM in [0,1],
  WideDeep in [0,1], SASRec logits all finite, LightGCN score finite,
  NDCG perfect ranking, uniform-neg never in positives, PTX non-empty +
  contains `.target sm_x` × all SM versions)
- [x] Benchmarks (`benches/recsys_ops.rs`) — PTX group (`als_step`,
  `dot_score` × 4 SM) + NDCG@10 bench + LCG RNG bench
- **Tests:** 461 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently
  PTX-string generation tested only)
- [ ] LightGCN propagation timed on real GPU for million-edge graphs
- [ ] ALS step throughput vs. CPU baseline on real GPU

#### P1 — Classical Algorithm Extensions
- [x] WARP / LambdaRank loss for learning-to-rank
- [x] SLIM (Sparse Linear Method) with elastic-net regularization
- [x] iALS Conjugate-Gradient solver (faster than Gauss-Jordan for high k)
- [x] FISM (Factored Item Similarity Models)
- [x] EASE / EASER closed-form item-item recommender

#### P1 — Neural Model Extensions
- [x] Transformer4Rec (full encoder block reuse)
- [x] xDeepFM CIN (Compressed Interaction Network) feature crossings
- [x] DIN / DIEN attention-over-history click-through rate models (sequential/din.rs -- Zhou 2018 KDD; local activation unit attention a(h_i, target) over user history → weighted-sum interest rep + concat target + MLP → CTR; DIEN remains for future)
- [x] DLRM (Deep Learning Recommendation Model) embedding + interaction tower (dlrm.rs -- Naumov 2019; per-field embedding tables + bottom MLP dense→embed_dim + upper-triangular pairwise dot-product interaction + top MLP → CTR)
- [x] FiBiNET bilinear feature interaction layer (fibinet.rs -- Huang 2019; SENET squeeze-excitation field reweighting + bilinear field-pair interaction p_i∘(W·p_j) over FieldAll/FieldEach/FieldInteraction + DNN)

#### P1 — Graph Recommenders
- [x] PinSAGE neighbor-sampling GraphSAGE variant
- [x] HGNN (heterogeneous graph neural network) for multi-relation graphs
- [x] KGAT (knowledge-graph attention) — fuse KG embeddings with user-item graph (graph_recsys/kgat.rs -- Wang 2019 KDD; relation-aware attention π(h,r,t)=tanh(W_r(e_h+e_r))ᵀtanh(W_r e_t), softmax-normalized propagation, n_layers concatenation, inner-product score)
- [x] UltraGCN — pre-computed weighted neighborhood replacing iterative GCN

#### P1 — Sequence Models
- [x] CL4SRec / DuoRec contrastive learning losses for sequential recsys
  (CL4SRec in sequential/cl4srec.rs -- Xie 2022 ICDE; crop/mask/reorder
  augmentations + InfoNCE/NT-Xent in-batch negatives. DuoRec in
  sequential/duorec.rs -- Qiu 2022 WSDM; dropout-based unsupervised views +
  same-target supervised positives + combined L_unsup + λ·L_sup objective)
- [x] STAMP short-term attention/memory priority model (sequential/stamp.rs -- Liu 2018 KDD; sigmoid local activation unit α_i = v_a · σ(W_a0·x_i + W_a1·x_t + W_a2·m_s + b_a), un-normalised gates, trilinear scoring e_j · (h_s ⊙ h_t))
- [x] FMLP-Rec frequency-domain filter-MLP (sequential/fmlp_rec.rs -- Zhou 2022 WWW; inline radix-2 Cooley-Tukey FFT + learnable complex filter (real=1, imag=0 init) + residual LayerNorm + position-wise GELU FFN)

#### P2 — Sampling & Training
- [x] Adaptive sampler with importance weighting (sampling/adaptive_neg.rs --
  Rendle-Freudenthaler 2014 WSDM / AdaSIR; uniform-proposal pool resampled
  ∝ exp(s/τ) with self-normalised importance weight ŵ returned for gradient
  debiasing of the uniform objective)
- [ ] Mixed-precision (FP16/BF16) embedding tables (requires GPU hardware /
  native half-precision storage)
- [ ] Embedding-table sharding across multiple GPUs (model parallelism)
  (requires multi-GPU hardware)
- [x] Sparse gradient AdamW updates for large embeddings (optim/sparse_adamw.rs
  -- Loshchilov-Hutter 2019 AdamW; per-row first/second moment + per-row step
  counter for lazy bias correction + decoupled weight decay; only touched rows
  time-stepped/updated)
- [ ] Distributed BPR / contrastive in-batch negatives (in-batch InfoNCE exists
  in sequential/cl4srec.rs; the *distributed* all-gather across GPUs requires
  multi-GPU hardware)

#### P2 — Evaluation & Tooling
- [x] Diversity / coverage / novelty metrics
- [x] Calibration & fairness-aware ranking metrics (metrics/calibration.rs --
  Naeini 2015 / Guo 2017; equal-width reliability bins, Expected & Maximum
  Calibration Error, Brier score, binary log loss, and group_calibration_disparity
  surfacing max−min per-group ECE for calibration fairness)
- [ ] LLM4Rec LLM-augmented recommendation (`llm/llm4rec.rs`) — Bao 2023: LLM-based item explanation + natural-language user profile construction with in-context learning for cold-start recommendation; `Llm4Rec` (requires an actual LLM backend; a pure-Rust stand-in cannot honestly provide LLM behaviour)
- [x] GraphRec interaction-aware graph recommender (`graph_recsys/graphrec.rs`) — Fan 2019 WWW: dual aggregation over item-space and social-space graphs with attention-weighted interactions; `GraphRec` (ALREADY IMPLEMENTED at graph_recsys/graphrec.rs)
- [x] Fairness-aware ranking exposure control (`ranking/fairness_ranking.rs`) — Singh-Joachims 2018 KDD: exposure-fairness constraint via deterministic ranking + constraint LP for proportional exposure across demographic groups; `FairnessRanker` (ALREADY IMPLEMENTED at ranking/fairness_ranking.rs)
- [x] MIND multi-interest network (`sequential/mind.rs`) — Li 2019 CIKM: capsule dynamic routing over user history to extract multiple interest vectors for diverse candidate retrieval; `MindNetwork`
- [x] Off-policy evaluation (IPS, SNIPS, doubly-robust estimators)
- [x] Cold-start handling (content-based fallback) (cold_start.rs -- Schein 2002
  SIGIR / Burke 2002; item-item content cosine KNN + interaction-weighted content
  user profile + switching/weighted hybrid that defers to content scoring when
  either user or item interaction count is below warm_threshold)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA-SDK / nvcc / Triton / TensorFlow dependency — PTX kernels are emitted
as strings. No oxicuda-driver / -memory / -launch dependency at this layer.

## Quality Status

- Warnings: 0 (clippy clean, workspace lints inherited)
- Tests: 461 passing (ALS, BPR, NMF, NCF, TwoTower, DeepFM, WideDeep, SASRec,
  LightGCN, NDCG, uniform-neg, PTX × 6 SM; + adaptive importance sampler,
  sparse-AdamW optimizer, calibration metrics, DuoRec, cold-start content
  fallback, AUC-tie / NDCG-IDCG / BPR finite-difference numerical checks)
- unwrap() calls: 0 in production code
- macOS: compiles but returns `UnsupportedPlatform` at runtime when actual launch
  is attempted (PTX emission still works on every host)
- Refactoring policy: every source file is well under 2,000 lines

## Performance Targets

| Workload | Target |
|----------|--------|
| ALS step (k=64, n_users=1M, n_items=1M) | ≥ 85% of cuBLAS-backed reference |
| Embedding lookup (batch 8192) | memory-bandwidth bound |
| Dot-score (batch 8192 × top-100) | ≥ 90% of cuBLAS GEMM |
| LightGCN propagation (1M edges, 2 layers) | ≥ 80% of cuSPARSE reference |

Performance harnesses are CPU-side today; GPU-side numbers will be filled in once
the Linux+NVIDIA verification run is executed.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/recsys_ops.rs`) — PTX group + NDCG bench
  + RNG bench

---

## Notes

- All embeddings and parameters are FP32 today. Mixed precision is a future option.
- The `LcgRng` is reproducible but not cryptographic; used for embedding init,
  negative sampling, and BPR triplet generation.
- DeepFM follows the Guo et al. 2017 formulation; AutoInt follows Song et al. 2019
  multi-head self-attention with residual.
- SASRec uses causal masking via additive `-inf` upper-triangular mask.
- BERT4Rec MLM uses uniform-mask sampling (no whole-word masking).
- LightGCN uses A_norm = D⁻½ A D⁻½ symmetric normalization on the
  user-item bipartite graph.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX target string emitted for all 7 kernels
- [ ] WMMA-based dense scoring path for two-tower retrieval

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX target string emitted
- [ ] `cp.async` global→shared prefetch for embedding tables
- [ ] Shared-memory tile reuse across batched dot-score
- [ ] Warp-shuffle softmax-top-K fusion

### Hopper (sm_90)
- [x] PTX target string emitted
- [ ] TMA-based batched embedding gather for very large tables
- [ ] WGMMA-based fused user×item dense matmul for two-tower

### Blackwell (sm_100)
- [x] PTX target string emitted
- [ ] Native FP4/FP6 embedding storage exploration for billion-scale tables

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage and PTX-string generation.
> These items represent the gap between current depth and full
> production-grade GPU recommender systems.

### Verification Gaps
- [ ] ALS convergence vs. CPU reference on MovieLens-25M
- [ ] LightGCN final embeddings reproducibility across SM versions
- [ ] BPR pairwise AUC tracking over training epochs

### Implementation Deepening
- [x] Sparse-gradient embedding optimizer (AdamW with row-wise state)
  (optim/sparse_adamw.rs)
- [ ] Distributed embedding-table sharding (requires multi-GPU hardware)
- [x] CIN / DIN / DLRM model coverage (CIN/xDeepFM in deepfm/cin.rs; DIN in
  sequential/din.rs; DLRM in dlrm.rs -- all ALREADY IMPLEMENTED)
- [x] PinSAGE / KGAT graph recommender extensions (graph_recsys/pinsage.rs and
  graph_recsys/kgat.rs -- both ALREADY IMPLEMENTED)

### Numerical Accuracy
- [x] AUC tie-handling unit-tested for synthetic pathological inputs
  (metrics/recsys_metrics.rs::tests -- all-ties=0.5, symmetric partial ties=0.5,
  one-tie-breaks-perfect=0.75, single-class=0.5)
- [x] NDCG IDCG denominator matches sklearn reference exactly
  (metrics/recsys_metrics.rs::tests::ndcg_idcg_matches_reference vs textbook
  1/log2(rank+1) binary-relevance reference over 4 cases)
- [x] BPR gradient direction verified via finite-difference probe
  (factorization/bpr.rs::tests::gradient_direction_matches_finite_difference --
  per-coordinate central difference of L=-ln σ(x_ui-x_uj), Δθ == -lr·∂L/∂θ)

## Performance Verification Harness Status (2026-05-16)

- **ALS / dot-score PTX kernels:** harnesses at `benches/recsys_ops.rs::recsys_ptx`;
  CPU-side PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **NDCG@10 bench:** CPU-side timing landed; not GPU-bound.
- **LCG RNG bench:** CPU-side baseline for sampling throughput.
