# oxicuda-meta TODO

Meta-learning algorithm primitives (MAML / FOMAML / ANIL / Reptile / Prototypical Networks / Matching Networks / Relation Networks) for OxiCUDA. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.33).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 17,594 SLoC (51 source files + 1 benches file) -- Coverage: full N-way K-shot meta-learning toolkit**

Current implementation covers MAML with second-order finite-difference outer gradients, FOMAML first-order approximation, ANIL head-only adaptation, Reptile first-order interpolation, three metric-learning few-shot heads (ProtoNet, MatchingNet, RelationNet), episode sampling, MLP backbone with Xavier init, and PTX kernels for inner SGD, Reptile interpolation, prototype distance, cosine similarity, relation score, meta-gradient accumulation, and episode sampling.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `MetaError` (15 variants: DimensionMismatch, EmptySupport, InvalidNWay, InvalidKShot, InvalidFeatDim, InsufficientClasses, InsufficientExamples, InvalidLr, NanEncountered, InvalidQuerySize, InvalidEpisodeConfig, GradientFailure, BackboneError, Internal, InvalidStepSize), `MetaResult<T>`
- [x] `handle.rs` -- `SmVersion` (Sm75/80/86/90/100/120), `LcgRng` (Knuth MMIX 64-bit LCG), `MetaHandle::default_handle()` (Sm80, device 0, seed 42)

#### Episode Utilities
- [x] `episode/types.rs` -- `FewShotEpisode` / `EpisodeConfig` with `support_for_class()` view helper
- [x] `episode/sampler.rs` -- `EpisodeSampler` with LCG Fisher-Yates sampling of N classes then K+Q examples per class

#### Network Backbones
- [x] `network/backbone.rs` -- `MlpBackbone` with ReLU (except final linear); Xavier init `U(-sqrt(6/(in+out)), sqrt(6/(in+out)))`; `to_params()` / `from_params()` for MAML flattening
- [x] `network/linear_head.rs` -- `LinearHead` linear probe with `to_params()` / `from_params()`

#### Gradient Utilities
- [x] `gradient/inner_loop.rs` -- `inner_sgd_step`, `multi_step_inner`, `cross_entropy_loss`
- [x] `gradient/finite_diff.rs` -- `fd_gradient` central finite differences `(f(theta + eps*e_i) - f(theta - eps*e_i)) / (2*eps)`

#### MAML Family
- [x] `maml/maml.rs` -- `maml_adapt`, `maml_meta_update`, `MamlConfig`; second-order finite-difference outer gradient
- [x] `maml/fomaml.rs` -- `fomaml_update`, `FoMamlConfig`; first-order approximation (gradient at theta' as constant)
- [x] `maml/anil.rs` -- `anil_adapt_head`, `anil_meta_update`, `AnilConfig`; only linear head updated in inner loop

#### Reptile
- [x] `reptile/reptile.rs` -- `reptile_update`, `ReptileConfig`; `theta <- theta + eps * (avg(theta'_i) - theta)` with k inner SGD steps per task

#### Metric Learning
- [x] `metric_learning/proto_net.rs` -- `compute_prototypes`, `proto_predict`, `proto_loss`; class prototype = mean(support), prediction via argmin L2
- [x] `metric_learning/matching_net.rs` -- `cosine_similarity`, `matching_net_attention`, `matching_net_predict`; softmax cosine attention over support with temperature
- [x] `metric_learning/relation_net.rs` -- `RelationNet { relation_score, predict_episode, relation_loss }`; 2-layer MLP concat(q,s) -> ReLU -> sigmoid; MSE loss on 0/1 targets

#### Metrics
- [x] `metrics/few_shot.rs` -- `episode_accuracy`, `mean_and_ci95` (95% confidence interval), `accuracy_at_k`

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 7 GPU kernels x 6 SM versions (75/80/86/90/100/120):
  - [x] `inner_sgd_kernel` -- elementwise `theta'[i] = theta[i] - alpha * g[i]` vector SGD step
  - [x] `reptile_update_kernel` -- `theta[i] += eps * (theta'[i] - theta[i])` interpolation step
  - [x] `proto_distance_kernel` -- squared L2 `d[q,k] = sum_j (q_j - proto_j)^2` for query x prototype pairs
  - [x] `cosine_sim_kernel` -- `cos(a,b) = a.b / (||a|| * ||b|| + eps)` for MatchingNet
  - [x] `relation_score_kernel` -- concat(q,s) + 2-layer ReLU MLP -> sigmoid for RelationNet
  - [x] `meta_grad_accum_kernel` -- elementwise sum of task gradients, divide by n_tasks
  - [x] `episode_sample_kernel` -- LCG-based class/example selection for N-way K-shot episodes

#### Integration Tests
- [x] 12 e2e tests (lib.rs): ProtoNet correct class, identity features -> correct label, MatchingNet attention sums to 1, same-class highest attention, RelationNet same > different, relation loss finite, MAML adapt changes params, Reptile moves toward task, inner SGD decreases loss, episode sampler correct shapes, 100% accuracy -> 1.0, PTX kernels x 6 SM versions

#### Benchmarks
- [x] `benches/meta_ops.rs` -- 7 PTX kernel groups x 4 SM versions plus 6 algorithm benches: proto_net_5way5shot_d64, matching_net_attention, maml_adapt_inner, reptile_update, episode_sampler

### Future Enhancements

#### P0 -- Critical Algorithmic Coverage
- [x] Second-order MAML with full Hessian-vector product (currently FD-based) -- explicit Hvp via dual-pass autodiff for better scaling at >10K params
- [x] Meta-SGD (Li et al. 2017) -- learnable per-parameter inner learning rates alongside meta-weights
- [x] Conditional Neural Processes (CNP / NP) -- amortised inference for few-shot via context aggregation
- [x] LEO (Latent Embedding Optimization, Rusu et al. 2019) -- meta-learn in low-dimensional latent space

#### P1 -- Important Features
- [x] Convolutional backbone (4-block conv-bn-relu) -- standard MiniImageNet / Omniglot convnet (network/conv4_backbone.rs -- standard 4-block conv-bn-relu-maxpool (Conv3x3 same-pad + BN + ReLU + MaxPool2x2 stride 2); halves H/W per block to H/16×W/16×width flattened)
- [x] ResNet-12 backbone -- canonical few-shot benchmark feature extractor (network/resnet12.rs:ResNet12 -- four residual stages, each 3×Conv3x3-BN(-ReLU) main path + 1×1 conv-BN shortcut, residual-add + ReLU + MaxPool2x2; He-uniform init, canonical widths [64,160,320,640]; now wired into network::mod + lib prelude with 19 deterministic tests)
- [x] Transductive batch norm (TBN) -- statistics across both support and query for better few-shot adaptation (network/tbn.rs -- transductive batch norm joint over support+query at meta-test, with γ/β affine; non-test-time path uses standard BN)
- [x] Cross-attention RelationNet (CAN) -- attention-weighted relation scoring across all support examples
- [x] FEAT (Few-shot Embedding Adaptation with Transformer) -- set-to-set transformer over support embeddings (metric_learning/feat.rs -- Ye 2020 CVPR; multi-head self-attention set-to-set adaptation of support prototypes + ProtoNet classification on adapted prototypes)

#### P2 -- Advanced / Research
- [x] MetaOptNet (Lee et al. 2019) -- closed-form differentiable convex solvers (SVM/ridge) as base learner
- [x] R2D2 (Bertinetto et al. 2019) -- differentiable ridge-regression base learner
- [x] DeepEMD -- Earth Mover's Distance over local features for fine-grained few-shot (metric_learning/deepemd.rs -- Zhang 2020 CVPR; cost=1-cosine over local features, Sinkhorn entropic OT plan, emd=<T,C>, EMD-to-prototype classification)
- [x] Continual Meta-Learning (OML / ANML) -- representation learning under online updates (online/oml.rs:Oml -- RLN/PLN factorisation, frozen encoder + online head SGD, forgetting-aware FOMAML meta-step; online/anml.rs:Anml -- adds learned neuromodulatory sigmoid gate z=ReLU(PN)⊙sigmoid(NM), analytic backprop; ANML now wired into online::mod + lib prelude with 13 deterministic tests)
- [x] Hyperparameter meta-learning (e.g. MAML-LR, ALFA) -- learn per-task inner-loop hyperparameters
- [x] Self-supervised pre-training hooks (S2M2, ProtoTransfer) -- contrastive backbone before meta-training (ssl/rotation.rs:RotationHead -- 4-way rotation-prediction pretext over Conv4Backbone; ssl/proto_transfer.rs:ProtoTransferHead -- ProtoCLR per-instance contrastive pretraining (softmax over -‖q-a‖²/τ, analytic gradient through L2-norm Jacobian) + transfer_classify to downstream ProtoNet; proto_transfer now wired into ssl::mod + lib prelude with 16 deterministic tests)
- [x] `meta/hyper_maml.rs` — HyperMAML (Przewięźlikowski 2022): hypernetwork generates fast-adapt weights rather than shared init; avoids inner-loop gradient computation; `HyperMaml { hyper_dims: Vec<usize> }`
- [x] `meta/meta_sgd.rs` — Meta-SGD (Li 2017): learn per-parameter learning rates along with init; α learned as parameter; inner update x←x-α⊙∇L; strictly more expressive than MAML
- [x] `metric/cross_attention_few.rs` — Cross-Attention Few-Shot (Ye 2021): cross-attend query to support set; task-specific attention produces class prototypes; `CafsFewShot { n_heads, n_layers }`
- [x] `meta/leap.rs` — LEAP (Flennerhag 2019): Meta-Learning with Warped Gradient Descent; learn a warp transformation of parameter space that makes inner MAML loop faster convergent; `LeapConfig { warp_dim: usize }`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none) | Standalone primitives crate | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

## Quality Status

- Tests: 530 passing (12 e2e in lib.rs + module unit tests; +67 over the prior 363: 15 new maml_conv_backbone + 4 new conv4 round-trip + 19 resnet12 + 13 anml + 16 proto_transfer modules brought online by wiring)
- All production code uses `Result` / `Option` (no `unwrap()` outside tests)
- `clippy::all` warnings: 0 (`cargo clippy -p oxicuda-meta --all-features --all-targets -- -D warnings` clean)
- `missing_docs` warnings: 0
- Files: 54 source `.rs` files, all under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns `UnsupportedPlatform` at runtime

## Performance Targets

Representative shapes for canonical few-shot benchmarks.

| Operation | Configuration | Priority |
|-----------|---------------|----------|
| `proto_distance_kernel` | 5-way 5-shot, feat_dim 64-512 | P0 |
| `cosine_sim_kernel` | 5-way 5-shot, feat_dim 64-512 | P0 |
| `relation_score_kernel` | 5-way 5-shot, hidden 128 | P0 |
| `inner_sgd_kernel` | 100K-1M params | P0 |
| `meta_grad_accum_kernel` | 16 tasks x 100K params | P1 |
| `episode_sample_kernel` | N=5, K=5, Q=15, 100 classes | P1 |

Target: episode forward latency comparable to PyTorch `torchmeta` reference on `sm_80+` for 5-way 5-shot N=5 K=5 with 64-dim features.

## Estimation vs Actual

| Metric | Description | Actual |
|--------|-------------|--------|
| Files | source `.rs` files under `src/` | 54 |
| SLoC | code lines (tokei) | 17,594 |
| Tests | e2e + unit | 530 |
| Coverage | algorithms with both CPU sim + PTX kernel | 7 (Proto/Matching/Relation + MAML/FOMAML/ANIL/Reptile) |

The current implementation provides a compact reference covering all canonical few-shot meta-learning algorithms used in the literature (MAML, FOMAML, ANIL, Reptile, ProtoNet, MatchingNet, RelationNet). The P0/P1/P2 future items cover more recent / specialised approaches and richer backbones.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX kernels generated for all 7 entry points on `sm_75`
- [ ] Cooperative groups for warp-level prototype reduction verified on Turing hardware

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels generated for `sm_80`, `sm_86`
- [ ] `cp.async` overlap for prototype/support feature staging
- [ ] Tensor Core relation-MLP path for 16x16x16 tile coverage on hidden_dim multiple of 16

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [x] PTX kernels generated for `sm_90`, `sm_100`, `sm_120`
- [ ] `wgmma`/TMA-based batched relation scoring for many-way large-feat episodes
- [ ] Distributed shared memory cluster reduction for very large episode batches

---

## Deepening Opportunities

> Items marked `[x]` in the Completed section represent API and CPU-simulation coverage. The opportunities below close gaps toward production few-shot deployment.

### Verification Gaps
- [x] Round-trip determinism verified for `LcgRng` (sample/shuffle reproducible across runs)
- [x] PTX entry points validated for `.version`, `.visible .entry`, kernel name, and SM target across all 6 SM versions
- [ ] End-to-end MiniImageNet / Omniglot accuracy reproduction against published baselines
- [ ] MAML / Reptile / ProtoNet GPU kernel correctness vs CPU simulation on `sm_80+`
- [ ] Cross-domain few-shot evaluation (Meta-Dataset protocol)

### Implementation Deepening
- [x] Episode sampler produces correct `(N*K*F, N*K, N*Q*F, N*Q)` shapes for any `(N, K, Q, F)`
- [x] ProtoNet 100% accuracy when query features equal prototypes (sanity-checked in `e2e_proto_net_correct_class`)
- [x] MatchingNet softmax attention sums to 1 within `1e-5` for any temperature
- [x] Convolutional backbone (4-block) integration with the MAML inner-loop closure pattern (maml/maml_conv_backbone.rs:Conv4MamlModel -- bundles Conv4Backbone + linear head as one flat param vector via to_params()/from_params() [backbone | head_w | head_b], with task_loss_at_params() forward closure and inner_adapt() finite-diff SGD; head slice is bit-compatible with the bare-linear maml_adapt; also added network/conv4_backbone.rs:Conv4Backbone::{to_params,from_params}; 15 + 4 deterministic tests incl. exact round-trip and support-loss-decrease)
- [ ] Distributed MAML across multiple GPUs (data-parallel task batching with NCCL-equivalent collective)
- [ ] Mixed-precision inner loop (FP16 forward + FP32 master parameters) for memory-bound large backbones

## Notes

- Algorithms operate on flat `Vec<f32>` parameter buffers (suitable for PTX kernel launch with raw device pointers)
- `MetaHandle` carries `(SmVersion, device, LcgRng)` so the same handle can deterministically drive both CPU simulation and PTX emission
- `to_params()` / `from_params()` on backbones enables the MAML inner loop to treat any module as a flat vector without bespoke serialisation
- All PTX kernels share a unified `.version` / `.target sm_X` / `.address_size 64` header consistent with the rest of the OxiCUDA ecosystem
