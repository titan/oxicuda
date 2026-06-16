# oxicuda-peft TODO

GPU-accelerated Parameter-Efficient Fine-Tuning (PEFT) primitives, covering low-rank
adaptation, prompt-based methods, adapter modules, sparse fine-tuning, and model
merging utilities.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.42).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 19,975 (82 files)
- **Coverage:** LoRA (low-rank adaptation with configurable r / α,
  Kaiming-uniform A, zero B); QLoRA (NF4 dequantization with 16-bucket lookup
  table, double-quantization absmax); AdaLoRA (SVD-parameterized ΔW = P · diag(Λ) · Q
  with importance-score-based rank pruning); DoRA (weight-decomposed magnitude +
  direction fine-tuning); IA³ (learned per-position scale vectors for K / V / FFN
  placements); Prefix-Tuning (per-layer K / V prefix); P-Tuning v2 (independent
  prefix per transformer layer); Prompt-Tuning (soft prompt embeddings prepended
  to input); Houlsby adapters (dual placement post-attention + FFN, GELU, zero-init
  up); Pfeiffer adapters (post-FFN only, skip-init); Parallel adapters (FFN-parallel
  with summation); Compacter (PHM Kronecker low-rank); BitFit (bias-only training
  identification); Diff-Pruning (Hard-Concrete L0 relaxation); LoRA merging (linear,
  TIES sign consensus, DARE random pruning); efficiency / merge-quality metrics;
  and PTX kernel-string generation for 6 SM tiers.

### Completed

#### Core Infrastructure
- [x] error.rs — `PeftError`, `PeftResult<T>`
- [x] handle.rs — `LcgRng` deterministic PRNG, `SmVersion`, `PeftHandle`

#### Low-Rank Adaptation (lora/)
- [x] lora.rs — `LoraConfig { r, alpha, init_scale }`,
  `LoraLinear { W, A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, scale = α / r }`,
  `merge_into_w`, `unmerge_from_w`, `lora_delta`
- [x] qlora.rs — `NF4_TABLE: [f32; 16]` (Dettmers 2023 quantiles),
  `nf4_quantize`, `nf4_dequantize`, `quantize_block`, `dequantize_block`,
  `QloraLinear` with double-quant absmax
- [x] adalora.rs — `AdaloraLinear { P, Λ, Q }`,
  `importance_scores = |λ_i| · ||P_i|| · ||Q_i||`, `prune_to_target`
  (zero Λ below budget), `reconstruct_delta`
- [x] dora.rs — `DoraLinear { magnitude, direction_w, A, B }`, column-wise
  magnitude normalization + direction update

#### IA³ Scaling (ia3/)
- [x] ia3.rs — `Ia3Placement { Key, Value, FeedForward }`,
  `Ia3Vector { scale }`, element-wise apply `y = x ⊙ scale`

#### Prefix & Prompt (prefix/)
- [x] prefix_tuning.rs — `PrefixConfig { num_virtual_tokens, prefix_dim,
  num_layers, num_heads, head_dim }`, `PrefixModule { K_prefix, V_prefix }`
  per-layer ~N(0, 0.02)
- [x] p_tuning_v2.rs — `PTuningV2 { layers: Vec<PrefixModule> }` independent
  prefix per transformer layer
- [x] prompt_tuning.rs — `SoftPrompt { embeddings }`, `prepend_to_sequence`
  → output length = num_tokens + seq_len

#### Adapter Modules (adapter/)
- [x] houlsby.rs — `HoulsbyAdapter { down, up, layer_norm }`, GELU bottleneck,
  zero-init up, residual
- [x] pfeiffer.rs — post-FFN only, skip-init up projection
- [x] parallel_adapter.rs — `ParallelAdapter`, FFN-parallel branch summed
  at output
- [x] compacter.rs — PHM Kronecker decomposition: ΔW = Σ_i A_i ⊗ B_i

#### Sparse Fine-Tuning (bitfit/, diff_pruning/)
- [x] bitfit.rs — `BitFitLayerInfo`, `BitFitMask::for_transformer`,
  `total_trainable_params`, `is_bias_param`
- [x] diff_pruning.rs — `DiffPruner { log_alpha, delta }`, concrete distribution
  `s = σ((log_α − log(u / (1 − u))) / β)` with stretch [γ, ζ], L0 regularizer

#### Merging & Arithmetic (merge/)
- [x] merge.rs — `merge_loras` weighted Δ sum, `linear_merge`,
  `ties_merge` (magnitude prune + majority-vote sign)
- [x] arithmetic.rs — `dare_prune` random density pruning + (1 / density)
  rescale, `sign_consensus`, `weighted_sum`

#### Metrics (metrics/)
- [x] efficiency.rs — `param_efficiency_ratio`, `effective_rank` (energy-based),
  `lora_param_count`, `compression_ratio`
- [x] merge_test.rs — `output_mse`, `output_consistency`, `max_abs_diff`

#### PTX Kernel Generation (ptx_kernels.rs)
- [x] 7 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `lora_matmul_ptx`, `ia3_scale_ptx`, `prefix_expand_ptx`,
  `adapter_forward_ptx`, `nf4_dequant_ptx`, `lora_merge_ptx`,
  `prompt_concat_ptx`

#### Tests & Benchmarks
- [x] 12 end-to-end tests in `lib.rs::e2e_tests` (zero-B no-change,
  scale = α / r, merge-unmerge roundtrip, NF4 dequant range,
  AdaLoRA importance ≥ 0, AdaLoRA prune reduces rank, IA³ identity scale,
  prefix shape correctness, soft-prompt length, Houlsby residual-init equals
  input, BitFit trainable-param count, PTX non-empty × all SM versions)
- [x] Benchmarks (`benches/peft_ops.rs`) — PTX bench group (`lora_matmul`,
  `nf4_dequant` × 4 SM) + LoRA forward algorithm bench
- **Tests:** 643 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently
  PTX-string generation tested only)
- [ ] LoRA forward / merge throughput measured on real GPU
- [ ] NF4 dequantization timed end-to-end on real GPU

#### P1 — Low-Rank Family Extensions
- [x] VeRA (Vector-based Random Adaptation) — frozen-random A, B with trainable
  per-rank vectors d_d, d_b (`lora/vera.rs` -- Kopiczko-Blankevoort-Asano 2024)
- [x] LoHa (Low-rank Hadamard product) — element-wise product of two LoRA factors (`lora/loha.rs` -- Hyeon-Woo et al. 2022 / Kohaku-Blueleaf 2023)
- [x] LoKr (Low-rank Kronecker product) — Kronecker decomposition adapter (`lora/lokr.rs` -- Edalati et al. 2022 / Kohaku-Blueleaf 2023)
- [x] MoLoRA — mixture of LoRA experts routed per token (`lora/molora.rs` + `lora/molora_tests.rs` -- per-token softmax gating with top-k routing, per-expert (A_k, B_k) + W_g gating matrix, load-balance variance diagnostic)
- [x] LoRA-FA — frozen A matrix to halve trainable parameters (`lora/lora_fa.rs` -- Zhang et al. 2023)
- [x] LoRA+ — different learning rates for A and B (`lora/lora_plus.rs` -- Hayou-Ghosh-Yu 2024; separate η_A and η_B = λ·η_A learning rates with `apply_update` helper, recommended λ ≥ 16)
- [x] PiSSA (Principal Singular Adaptation) — SVD-initialized LoRA (`lora/pissa.rs` -- Meng-Wang-Zhang 2024; inline one-sided Jacobi SVD)
- [x] OLoRA — orthonormal initialization via Gram-Schmidt (`lora/olora.rs` -- Büyükakyüz 2024)

#### P1 — Quantization Extensions
- [x] HQQ (Half-Quadratic Quantization) — quantization-aware finetuning (`lora/hqq.rs` -- Badri-Shaji 2023; half-quadratic splitting with Lₚ proximal `z`-update, closed-form 2×2 `(scale, zero)` solve, β-annealed outer loop)
- [x] GPTQ-style activation-aware quantization (`lora/gptq.rs` + `lora/gptq_tests.rs` -- Frantar-Ashkboos-Hoefler-Alistarh 2023 ICLR; OBS-style sequential per-column quantization with Cholesky-of-inverse-Hessian error compensation, block-wise update, optional activation ordering, per-group scale/zero affine code with permutation tracking for act_order roundtrip)
- [x] AWQ (Activation-aware Weight Quantization) integration (`lora/awq.rs` + `lora/awq_tests.rs` -- Lin et al. 2024 MLSys; per-input-channel salience scale `s_i = (mean|x_i| + 1e-8)^α` normalised to unit geometric mean, grid search α ∈ {0/N, …, N/N} minimising activation-weighted MSE, per-group affine quantization along the output-channel axis, dequant rebuilds via `(scale·q + zero) / awq_scale[i]`, deterministic with no RNG)
- [x] FP4 / NF3 / NF2 lower-bit storage with double-quant absmax
- [x] QA-LoRA quantization-aware LoRA — lora/qa_lora.rs (group-wise NF4 quant + rank-split LoRA adapters per group)

#### P1 — Adapter Extensions
- [x] Hypercomplex / Quaternionic adapters (`adapter/hypercomplex.rs` + `adapter/hypercomplex_tests.rs` -- Hamilton 1843 / Parcollet-Morchid-Linares 2019 ICLR / Zhang-Wang 2022 arXiv; QuatMatrix with 4-component storage, Hamilton product matvec y_a=Σ_b W_ab⊗x_b, Kaiming-real init (wi=wj=wk=0), split-GELU on real part, zero-init up for residual identity start, 4× parameter reduction vs real adapter)
- [x] LST (Ladder Side-Tuning) (`adapter/lst.rs` + `adapter/lst_tests.rs` -- Sung-Cho-Bansal 2022 NeurIPS; per-layer side bottleneck `down→GELU→side_residual`, no-gradient through frozen trunk hidden states, gated final output `α·trunk + (1-α)·up(side)`, Kaiming-uniform down/side init, zero up init)
- [x] (IA)³ combined with adapter for hybrid scaling — ia3/ia3_adapter.rs (IA³ scale then bottleneck residual, He et al. 2022)
- [x] AdapterFusion — composition of multiple task adapters via attention (`adapter/adapter_fusion.rs` -- Pfeiffer-Kamath-Rücklé-Cho-Gurevych 2021 EACL; Q/K/V projection over K adapter outputs with temperature-scaled softmax attention, Kaiming-uniform init via LcgRng)

#### P1 — Prompt / Prefix Extensions
- [x] SPoT (Soft Prompt Transfer) (`prefix/spot.rs` -- Vu-Lester-Briskie-Liu-Chaturvedi-Iyyer 2022 ACL 2022:4643; cosine-similarity softmax-weighted retrieval of source-task soft prompts, temperature-scaled softmax, top-k restriction with re-normalization, returns `SoftPrompt` for use as a transfer-initialized prompt)
- [x] ATTEMPT mixture-of-soft-prompts with attention routing (`prefix/attempt.rs` -- Asai-Sadeghian-Hajishirzi 2022 EMNLP 2022:6655; attention routing q=W_query·input_repr, dot-product scores with per-source keys, temperature-scaled softmax weights, weighted sum of source prompts, route_top_k restriction + re-normalization)
- [x] APrompt / ProPrompt activation-prompt schemes (prefix/aprompt.rs -- Wang 2023; learnable prompt key/value pairs injected into multi-head attention, queries attend over prompt-augmented K/V)
- [x] Multi-task prompt pool (prefix/prompt_pool.rs -- Wang 2022 L2P; pool of M (key,prompt) pairs, top-N cosine selection, concatenated selected prompts, key-matching loss)

#### P1 — Merging Extensions
- [x] Task Arithmetic (Ilharco et al. 2023) — task-vector add / subtract (`merge/task_arithmetic.rs` -- ICLR 2023; `τ = θ_finetuned − θ_pretrained`, weighted-sum add, negate-to-forget, `a−b+c` analogy, cosine similarity)
- [x] AdaMerging unsupervised entropy-min merging (`merge/adamerging.rs` + `merge/adamerging_tests.rs` -- Yang et al. 2024 ICLR; per-task or per-layer-per-task coefficients λ in logit-space with softmax simplex projection, central finite-difference gradient on entropy of logit-proxy classifier, GD with re-projection after each step; layer-wise variant averages per-task softmax across layers preserving simplex property)
- [x] Model Soup (uniform / greedy soup) for weight averaging (`merge/model_soup.rs` -- Wortsman et al. 2022 ICML; uniform mean, normalised weighted, greedy add-if-validation-score-improves with configurable higher/lower-is-better direction)
- [x] Fisher-information-weighted merging (`merge/fisher_merging.rs` -- Matena-Raffel 2022 NeurIPS; diagonal Fisher proxy `F̂ᵢⱼ = mean(gⱼ²)` per task, closed-form coordinate-wise weighted average `θ̄ⱼ = Σ F̂ᵢⱼ·θᵢⱼ / (Σ F̂ᵢⱼ + ε)`)
- [x] RegMean closed-form regression-based merging (`merge/regmean.rs` + `merge/regmean_tests.rs` -- Jin et al. 2023 ICLR; closed-form per-layer least-squares merge `W_merged = (Σ G_i' + ε·I)⁻¹ · (Σ G_i' · W_i)` with off-diagonal mask `M(α) = α + (1−α)·δ_ab` scaling, in-place Gauss-Jordan with partial pivoting on augmented `[A|B]`, `compute_gram` helper)
- [x] Dare-TIES combined pruning + sign consensus pipeline (`merge/dare_ties.rs` -- Yu et al. 2024 ICML; sequential DARE per-task Bernoulli prune at `density` with 1/density rescale (LcgRng seeded with `seed + i`), TIES top-`trim_density` magnitude trim, sign-consensus across tasks zeroing minority-sign coords, disjoint-contributor mean output)

#### P2 — Training & Tooling
- [x] Gradient checkpointing for memory-efficient adapter training
- [ ] Quantized-optimizer state storage (bnb-style 8-bit Adam)
- [ ] Adapter-save / adapter-load oxicode serialization (no zip / bincode)
- [ ] LoRA hub / registry conventions for shared adapters
- [ ] Compression-ratio / effective-rank dashboards
- [ ] `peft/vera.rs` — VeRA (Kopiczko 2023): Very few trainable parameters; share random frozen A,B matrices across all layers; learn per-layer diagonal scaling vectors d,b; <1M trainable parameters for 7B model
- [x] `peft/flora.rs` — Flora (Han 2024): random projection of full gradient into low-rank update; maintain low-rank optimizer state; unbiased gradient estimator; theoretically equivalent to full Adam convergence
- [ ] `peft/lorafa.rs` — LoRA-FA (Zhang 2023): fix A random (frozen), only train B; halves LoRA memory; equivalent gradient direction as full LoRA when A∼N(0,1); drop-in replacement
- [ ] `peft/mosa.rs` — MoSA (Zeng 2024): Mixture of Sparse Adapters; sparse update mask per adapter; top-k weight selection + shared sparse structure; combine gating from MoE

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA-SDK / nvcc / bitsandbytes / PEFT-lib dependency — PTX kernels are
emitted as strings. No oxicuda-driver / -memory / -launch dependency at this
layer.

## Quality Status

- Warnings: 0 (clippy clean, workspace lints inherited)
- Tests: 643 passing (LoRA zero-B, scale, merge / unmerge, NF4 dequant,
  AdaLoRA importance + prune, IA³ identity, prefix shape, soft-prompt length,
  Houlsby residual-init, BitFit param count, PTX × 6 SM)
- unwrap() calls: 0 in production code
- macOS: compiles but returns `UnsupportedPlatform` at runtime when actual launch
  is attempted (PTX emission still works on every host)
- Refactoring policy: every source file is well under 2,000 lines

## Performance Targets

| Workload | Target |
|----------|--------|
| LoRA matmul (d=4096, k=4096, r=16, batch 32) | ≥ 90% of cuBLAS GEMM |
| NF4 dequantization (1 GiB block) | memory-bandwidth bound |
| LoRA merge / unmerge (d=k=4096) | memory-bandwidth bound |
| Adapter forward (Houlsby, d=768, bn=64) | ≥ 90% of cuBLAS GEMM |
| Prefix expand (4 layers × 8 heads × 64 head_dim × 10 vtok) | bandwidth bound |

Performance harnesses are CPU-side today; GPU-side numbers will be filled in once
the Linux+NVIDIA verification run is executed.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/peft_ops.rs`) — PTX bench group
  (`lora_matmul`, `nf4_dequant` × 4 SM) + LoRA forward algorithm bench

---

## Notes

- All parameters are FP32 today (except NF4-quantized weight blocks).
  BF16 / FP16 storage is a future option.
- LoRA B matrix is zero-initialized so the adapter contributes zero at training
  start — verified by the `lora_forward_no_change_with_zero_b` test.
- NF4 quantization follows Dettmers et al. 2023: 16-bucket normalized-float
  table chosen as percentiles of the standard normal, plus per-block absmax
  scale stored at 8-bit (double-quantization).
- AdaLoRA importance score is the product of |λ_i| with the row / column norm
  of P and Q; rank pruning zeros out Λ entries below the target budget.
- Houlsby adapter is dual-placement (post-attention + post-FFN); Pfeiffer is
  post-FFN only.
- Diff-Pruning uses the Hard-Concrete relaxation of L0 with stretch [γ, ζ] and
  temperature β; mask m = clamp(s · (ζ − γ) + γ, 0, 1).
- TIES merging follows magnitude pruning → sign consensus → mean of preserved
  values.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX target string emitted for all 7 kernels
- [ ] WMMA m16n16k16 LoRA matmul path for FP16 trunks

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX target string emitted
- [ ] `cp.async` global→shared prefetch for LoRA A / B factors
- [ ] Shared-memory bank-conflict-free LoRA tile layout
- [ ] Warp-shuffle prefix expand for very long virtual-token counts

### Hopper (sm_90)
- [x] PTX target string emitted
- [ ] TMA-based bulk loading of quantized weight blocks (NF4 / NF3)
- [ ] WGMMA-based fused LoRA + base-matmul kernel

### Blackwell (sm_100)
- [x] PTX target string emitted
- [ ] Native FP4 LoRA factor storage exploration
- [ ] FP6 trunk-weight + FP16 LoRA-factor mixed-precision path

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage and PTX-string generation.
> These items represent the gap between current depth and full
> production-grade GPU PEFT.

### Verification Gaps
- [ ] LoRA fine-tuning equivalence vs. PEFT reference on toy datasets
- [ ] NF4 dequantization bit-exact match to bitsandbytes reference
- [ ] AdaLoRA pruning-trajectory parameter count vs. paper formula
- [ ] TIES sign-consensus + DARE rescale aggregate against published tables

### Implementation Deepening
- [ ] Fused LoRA + base-matmul kernel (eliminate intermediate write-back)
- [ ] Block-wise NF4 with shared absmax stored on chip
- [ ] AdaLoRA with continuous importance EMA + scheduled pruning
- [ ] AdapterFusion composition with attention weights

### Numerical Accuracy
- [ ] LoRA merge-unmerge roundtrip max-abs-error bounded by ε_machine × ||W||
- [ ] NF4 quantization error vs. INT4 on standard transformer weight matrices
- [ ] Houlsby zero-init residual property unit-tested for varying bottleneck dims

## Performance Verification Harness Status (2026-05-16)

- **PTX kernels:** harnesses at `benches/peft_ops.rs::peft_ptx` (LoRA matmul
  and NF4 dequant × 4 SM); CPU-side PTX-emission timings landed,
  GPU launch path awaiting Linux+NVIDIA run.
- **LoRA forward algorithm bench:** CPU-side timing landed; GPU-side
  throughput pending.
