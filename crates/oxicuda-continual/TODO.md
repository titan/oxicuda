# oxicuda-continual TODO

Pure-Rust continual and lifelong learning library covering all major families
of catastrophic-forgetting mitigation: regularisation (EWC / SI / MAS),
architecture (PackNet / Piggyback / Progressive NN), and experience replay
(ER / GEM / A-GEM / DER++). Includes forgetting/plasticity metrics and
task-incremental / class-incremental data streams. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.29).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 21,327 SLoC (48 files)** -- 529 unit tests + 12 E2E integration tests

The crate spans the three canonical continual-learning families plus
metrics and data streams. All algorithms run pure-Rust on CPU for unit testing
and emit PTX strings for GPU acceleration on NVIDIA SM 7.5 through SM 12.0.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `ContinualError` (15 variants: DimensionMismatch, EmptyInput,
      InvalidLambda, InvalidBufferCapacity, InvalidTaskId, InsufficientData,
      InvalidThreshold, InvalidAlpha, InvalidBeta, GemProjectionFailed,
      NanEncountered, InvalidMaskSparsity, InvalidLateralDim, StreamExhausted,
      Internal); `ContinualResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG, reservoir
      sampling helpers, Box-Muller normals), `ContinualHandle::default_handle()`
      (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- module exports + prelude + 12 E2E integration tests

#### PTX Kernels (7 kernels x 6 SM versions = 42 generators)
- [x] `ptx_kernels.rs::ewc_penalty_ptx` -- `fma.rn.f32` for
      lambda/2 * sum F_i * (theta_i - theta*_i)^2; `atom.global.add.f32`
- [x] `ptx_kernels.rs::fisher_diag_ptx` -- element-wise g^2 accumulate for
      empirical Fisher diagonal
- [x] `ptx_kernels.rs::gradient_project_ptx` -- half-space projection
      g - (g . m / m . m) * m for GEM
- [x] `ptx_kernels.rs::mask_apply_ptx` -- `w *= mask` with `setp.ne.u32`
      predicate for PackNet / Piggyback
- [x] `ptx_kernels.rs::si_omega_update_ptx` -- `|delta_theta * grad L|` synaptic
      importance accumulate
- [x] `ptx_kernels.rs::logit_distill_ptx` -- KL divergence via `ex2 / lg2`
      approximations for DER++
- [x] `ptx_kernels.rs::replay_sample_ptx` -- reservoir-sampling conditional
      swap via LCG
- [x] `ptx_kernels.rs::f32_hex` -- f32 to 0F-prefixed hex literal helper

#### Regularisation (regularization/)
- [x] `ewc.rs::EwcRegularizer` / `ewc_loss` / `compute_fisher_empirical` --
      Elastic Weight Consolidation (Kirkpatrick 2017): empirical Fisher
      diagonal F_i = (1/N) sum g_i^2; penalty
      lambda/2 * sum_t sum_i F_i^t * (theta_i - theta_i*^t)^2;
      `add_task` anchors a new task; `EwcConfig`, `FisherDiag`
- [x] `si.rs::SiState` / `si_importance_update` / `si_penalty` -- Synaptic
      Intelligence (Zenke 2017): online importance
      Omega_i += |delta_theta_i * grad L_i|; SI penalty normalised by
      (delta_Theta_i^2 + xi); `SiConfig`
- [x] `mas.rs::MasImportance` / `mas_importance_update` / `mas_penalty` --
      Memory-Aware Synapses (Aljundi 2018): momentum-weighted importance
      Omega = alpha * Omega + (1 - alpha) * |grad L|; `MasConfig`

#### Architecture (architecture/)
- [x] `packnet.rs::PackNetMask` / `prune_weights_l1` / `apply_mask` /
      `freeze_task_weights` -- L1-magnitude pruning to sparsity fraction;
      task-specific binary masks; pruned weights frozen
- [x] `piggyback.rs::PiggybackMask` / `binarize_mask` / `piggyback_forward` --
      real-valued mask -> binary via threshold; effective weights
      w_eff = w_base (.) binarize(m); `PiggybackConfig`
- [x] `progressive.rs::ProgNnNetwork` / `ProgNnColumn` / `LateralConnection` /
      `add_column` / `prog_forward` -- Progressive Neural Networks (Rusu 2016):
      frozen previous columns with lateral connections
      h_k^l = relu(W * h + sum U * h_prev)

#### Replay (replay/)
- [x] `er.rs::ErBuffer` / `er_buffer_new` / `er_add` / `er_sample_batch` --
      Experience Replay with reservoir sampling (Vitter 1985): uniform buffer
      replacement with probability capacity / n_seen; Fisher-Yates batch
      sampling
- [x] `gem.rs::GemMemory` / `gem_project_gradient` -- Gradient Episodic Memory
      (Lopez-Paz 2017): iterative half-space projection onto
      g . g_k >= -margin constraints, most-violated-constraint first;
      `GemConfig`
- [x] `a_gem.rs::a_gem_project` / `average_gradients` -- Averaged GEM
      (Chaudhry 2018): single projection onto average reference gradient
      g_ref = (1/T) sum g_k; `AGemConfig`
- [x] `dark_exp.rs::DerBuffer` / `der_add` / `der_loss` -- Dark Experience
      Replay++ (Buzzega 2020): alpha * MSE(z, z_stored) + beta * CE(z, y);
      reservoir buffer with stored logits; `DerConfig`

#### Metrics (metrics/)
- [x] `forgetting.rs::AccuracyMatrix` / `TaskAccuracy` / `average_forgetting` /
      `backward_transfer` / `plasticity` -- standard CL metrics:
      BWT = (1/(T-1)) * sum_k (acc[T-1,k] - acc[k,k]),
      forgetting = max_j acc[j,k] - acc[T-1,k]
- [x] `intransigence.rs::forward_transfer` / `intransigence` /
      `per_task_intransigence` -- FWT = (1/(T-1)) * sum_k
      (acc[k-1,k] - acc_random[k]); intransigence = transfer gap to isolated
      task training

#### Data Streams (stream/)
- [x] `task_stream.rs::TaskStream` / `Task` / `task_stream_new` /
      `current_task` / `next_task` / `task_batch` -- task-incremental stream:
      ordered task sequence with batch sampler
- [x] `class_stream.rs::ClassIncStream` / `class_inc_new` / `init_class_inc` /
      `advance_class_inc` / `class_inc_batch` / `n_classes_seen` --
      class-incremental stream with disjoint label spaces; `n_classes_seen`
      grows monotonically

#### Integration Tests (lib.rs e2e_tests)
- [x] `e2e_ewc_loss_zero_at_anchor` -- EWC penalty = 0 at parameter anchor
- [x] `e2e_si_penalty_grows_with_displacement` -- SI penalty monotonic in
      |theta - theta_anchor|
- [x] `e2e_mas_importance_tracks_gradient` -- omega = |grad| with momentum = 0
- [x] `e2e_packnet_sparsity_respected` -- prune to sparsity fraction keeps
      floor((1 - sparsity) * N) weights
- [x] `e2e_piggyback_binarization` -- threshold = 0 -> sign(mask) binarisation
- [x] `e2e_progressive_multi_column_shape` -- multi-column forward shape
      preserved
- [x] `e2e_er_reservoir_bounded` -- buffer never exceeds capacity even with
      many adds
- [x] `e2e_gem_project_satisfies_constraint` -- projected gradient satisfies
      g . g_k >= -margin for all k
- [x] `e2e_a_gem_aligned_unchanged` -- gradient already aligned with reference
      remains unchanged
- [x] `e2e_der_loss_finite` -- DER++ alpha * MSE + beta * CE is finite and
      non-negative
- [x] `e2e_forgetting_zero_perfect_retention` -- perfect-retention matrix ->
      forgetting and BWT == 0
- [x] `e2e_ptx_kernels_all_sm_versions` -- all 7 kernels x 6 SM versions
      contain `.version`, `.visible .entry`, `sm_X`, and kernel name

#### Benchmarks (benches/continual_ops.rs)
- [x] 7 PTX kernel groups x 4 SM versions (PTX generation throughput)
- [x] `ewc_loss_d1024` -- EWC penalty on 1024 parameters
- [x] `fisher_diag_accumulate` -- empirical Fisher accumulation
- [x] `gem_project_d512` -- GEM iterative projection
- [x] `er_sample_b32` -- reservoir sampling batch of 32
- [x] `packnet_prune_d1024` -- L1 pruning + mask materialisation

### Future Enhancements [ ]

#### P0 -- Critical (Performance-Sensitive Paths)
- [x] Online EWC variant -- moving-average Fisher across tasks instead of
      additive sum (avoids unbounded penalty growth) (`regularization/online_ewc.rs`)
- [x] Vectorised GEM projection -- batch all memory constraints into single
      QP solve via projected coordinate descent on the dual (`replay/vectorised_gem.rs`)
- [x] Fused DER++ loss kernel -- combine logit distillation + label CE in a
      single epilogue pass
- [ ] Tensor-Core path for `prog_forward` lateral connections on SM 8.0+

#### P1 -- Important (Feature Completeness)
- [x] iCaRL -- nearest-mean-of-exemplars classifier + herding-based exemplar
      selection (`architecture/icarl.rs`)
- [x] LwF (Learning without Forgetting) -- knowledge-distillation regulariser
      between current and frozen previous model (`regularization/lwf.rs`)
- [x] BiC (Bias Correction) -- post-hoc logit bias layer for
      class-incremental learning (`architecture/bic.rs`)
- [x] MIR (Maximally Interfered Retrieval) -- replay sample selection by
      predicted loss increase (`regularization/mir.rs`)
- [x] CLEAR-style supervised contrastive replay (`regularization/clear_replay.rs`)
- [x] HAT (Hard Attention to the Task) -- task-conditional attention masks
      with cumulative gating (`architecture/hat.rs`)
- [x] Generative replay (VAE / GAN) -- substitute stored exemplars with
      sampled generations (`architecture/generative_replay.rs`)

#### P2 -- Nice-to-Have (Advanced Features)
- [x] Continual evaluation scenario harness (Permuted MNIST, Split-MNIST /
      Split-CIFAR, Rotated-MNIST / CORe50-style) -- deterministic synthetic
      `TaskStream` / `ClassIncStream` generators with the exact structural
      transforms (pixel permutation, disjoint class split, planar domain
      rotation) (`stream/scenario.rs`)
- [x] Online meta-learning (OML / ANML) primitives -- RLN+Gate+PLN, FOMAML (`regularization/meta_learning.rs`)
- [x] Domain-incremental scenario (`architecture/domain_incremental.rs`)
- [x] Memory-efficient replay via gradient compression -- GEM-style projection (`regularization/gradient_compression.rs`)
- [x] Multi-GPU sharded replay buffer -- CPU sharding algorithm: round-robin /
      hash-label routing, per-shard Vitter reservoir, balanced cross-shard
      retrieval, per-shard device-placement metadata (`replay/sharded_buffer.rs`).
      On-device kernels / NCCL all-gather remain GPU-gated.
- [x] `continual/l2p.rs` — Learning to Prompt (Wang 2022): prepend task-specific prompt tokens retrieved from prompt pool; key-query cosine matching; no EWC/replay needed; `L2pConfig { pool_size, prompt_len }`
- [x] `continual/dualprompt.rs` — DualPrompt (Wang 2022): G-Prompt (task-invariant) + E-Prompt (task-specific); orthogonal regularisation to separate general/specific knowledge; `DualPromptConfig { g_length, e_length }`
- [x] `continual/clser.rs` — CLSER (Arani 2022): complementary learning systems ER; slow + fast learners inspired by hippocampus/neocortex; EMA slow learner as stable knowledge base; `ClserConfig { alpha_ema: f32 }`
- [x] `continual/memo.rs` — MEMO (Zhou 2022): deep model expansion; new backbone layers per task with shared generalised layers; task-specific + generalised layer composition; `MemoConfig { expansion_rate: f32 }`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

No CUDA SDK, no C, no Fortran. The crate compiles standalone and produces PTX
strings that can be consumed by `oxicuda-driver` / `oxicuda-launch` at runtime.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 529 unit + 12 E2E = 541 passing
- unwrap() calls: 0 (production code)
- All public APIs return `ContinualResult<T>` or `Result<T, ContinualError>`

## Performance Targets

Reference shapes (EWC is the most parameter-heavy regulariser; GEM dominates
replay latency):

| Kernel | Shape | Target |
|--------|-------|--------|
| ewc_penalty | n_params = 1M, n_tasks = 10 | bandwidth-limited (>= 80% of peak HBM) |
| fisher_diag | n_params = 1M, batch = 64 | bandwidth-limited |
| gradient_project (GEM) | n_params = 1M, n_constraints = 10 | iterative, dot-product-bound |
| si_omega_update | n_params = 1M | bandwidth-limited |
| logit_distill | batch = 256, n_classes = 1000 | softmax-bound |
| mask_apply (PackNet) | n_params = 1M | bandwidth-limited |
| replay_sample | buffer = 65536, batch = 32 | latency-bound |

## Notes

- All randomness is deterministic via `LcgRng` seeded by `ContinualHandle`;
  unit tests do not depend on `rand` or `getrandom`.
- `EwcRegularizer` accumulates a vector of (anchor, Fisher) pairs; `ewc_loss`
  iterates over all anchored tasks (sum-of-penalties formulation).
- `ErBuffer` uses Vitter's algorithm R for reservoir sampling: each new item
  is admitted with probability capacity / n_seen.
- `gem_project_gradient` uses most-violated-constraint-first ordering; on
  convergence the result satisfies g . g_k >= -margin for every constraint.
- `PackNetMask::n_active()` returns floor((1 - sparsity) * total_weights);
  exact count verified by `e2e_packnet_sparsity_respected`.

---

## Architecture-Specific Deepening

### Hopper (sm_90 / sm_90a)
- [ ] `wgmma.mma_async` path for lateral-connection GEMM in `prog_forward`
- [ ] TMA (`cp.async.bulk`) loading of EWC Fisher tiles across tasks

### Ampere (sm_80 / sm_86) / Ada (sm_89)
- [ ] `cp.async` prefetch of replay batches into shared memory
- [ ] Atomics-coalescing for `fisher_diag` accumulate

### Blackwell (sm_100 / sm_120)
- [ ] 5th-gen Tensor Core path for projection inner products in GEM
- [ ] Cluster launch for cross-CTA buffer-wide reductions in `replay_sample`

---

## Deepening Opportunities

### Verification Gaps
- [x] All 7 PTX generators emit `.version`, `.target sm_X`, and named entry per
      SM version (verified by `e2e_ptx_kernels_all_sm_versions`)
- [x] EWC empirical Fisher vs. analytic Fisher on small Gaussian model --
      closed-form `F = 1/σ²` for the `N(θ,σ²)` mean model vs. the production
      `compute_fisher_empirical` estimator; converges to <2% rel-error at 200k
      samples (`metrics/verification.rs::gaussian_fisher_comparison`)
- [x] GEM convergence rate vs. number of memory constraints --
      feasibility / worst-constraint-dot / rotation-cosine profile across
      constraint counts (`metrics/verification.rs::gem_convergence_profile`)
- [x] DER++ alpha / beta sensitivity sweep on Split-MNIST -- `(α,β)` grid over
      the production `der_loss` (`metrics/verification.rs::der_sensitivity_grid`)

### Implementation Deepening
- [x] EWC supports multi-task anchor list (sum-of-tasks formulation)
- [x] SI normalises by (Delta_Theta^2 + xi) to prevent divide-by-zero
- [x] MAS supports momentum alpha in [0, 1] for online importance updates
- [x] PackNet `prune_weights_l1` is deterministic with respect to RNG seed
- [x] Sparse-gradient fast path for `mask_apply` when sparsity > 0.9 (`architecture/sparse_mask_apply.rs`)
- [x] Stochastic-binary forward for `PiggybackMask` (straight-through estimator) (`architecture/stochastic_binary.rs`)
- [x] Multi-head class-incremental classifier helper (`architecture/multihead.rs`)
- [x] Cross-task batch sampler for `TaskStream` -- Uniform/Proportional/Temperature (`stream/cross_task_sampler.rs`)
