# oxicuda-train TODO

GPU-accelerated training engine providing fused optimizer update kernels, gradient checkpointing, LR schedulers, mixed precision (AMP), EMA, and ZeRO optimizer state sharding. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.8).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 11,901 SLoC across 32 files (includes Markdown doc-comments) / 5,984 pure Rust SLoC**

Production-grade GPU-accelerated training utilities implementing the v1.2 roadmap items:
fused optimizer kernels, gradient checkpointing, mixed-precision optimizer states, EMA,
and large-scale ZeRO-style distributed training.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `TrainError` (12 variants) with `TrainResult<T>` alias
- [x] `handle.rs` -- `TrainHandle` wraps `Arc<Context>` + `Arc<Stream>` + SM version metadata; `device_sm_version()` via CUdevice_attribute
- [x] `lib.rs` -- prelude module and 6 E2E integration tests

#### PTX Update Kernels
- [x] `ptx_kernels.rs` -- 31.6 KB of fused PTX generators
  - `adam_update_ptx` -- fused moment update + bias-corrected Adam step (`fma.rn.f32`, `sqrt.approx.f32`, `rcp.approx.f32`)
  - `adamw_update_ptx` -- decoupled weight decay: `p *= (1 - lr * wd)` before moment update
  - `sgd_update_ptx` -- Nesterov SGD with `setp.ne.f32` predicate for conditionality
  - `lion_update_ptx` -- sign via bit-mask `and.b32 sign_bit, c_bits, 0x80000000`
  - `came_row_factor_ptx` / `came_col_factor_ptx` -- per-row/col CAME factored second moment
  - `norm_sq_partial_ptx` -- block-level squared norm with warp butterfly `shfl.sync.bfly.b32` + smem merge
  - `scale_inplace_ptx` / `add_inplace_ptx` -- element-wise scale and gradient accumulation
  - Grid-stride `$LOOP`/`$DONE`, sm_80/sm_90 PTX header selection, `f32_hex()` IEEE literals

#### GPU Optimizers (`gpu_optimizer/`)
- [x] `mod.rs` -- `GpuOptimizer` trait: `step()`, `zero_grad()`, `lr()`, `set_lr()`, `name()`; `ParamTensor` flat buffer; `adam_bias_corrections()` helper
- [x] `adam.rs` -- `GpuAdam`: bias-corrected first+second moments, optional AMSGrad variant
- [x] `adamw.rs` -- `GpuAdamW`: decoupled weight decay (default wd=0.01)
- [x] `adagrad.rs` -- `GpuAdaGrad`: per-parameter adaptive learning rates with accumulated squared gradients
- [x] `rmsprop.rs` -- `GpuRMSProp`: EMA of squared gradients with optional momentum
- [x] `radam.rs` -- `GpuRAdam`: rectified Adam with variance-rectification term
- [x] `lion.rs` -- `GpuLion`: single moment buffer; sign update; 50% memory vs Adam
- [x] `came.rs` -- `GpuCame`: factored second moment `CameV::Matrix { row, col }` -- O(m+n) vs O(mn)
- [x] `muon.rs` -- `GpuMuon`: Nesterov + Newton-Schulz orthogonalisation; 5-iteration `X <- 1.5X - 0.5X * XtX`

#### Gradient Utilities
- [x] `grad_clip.rs` -- `GlobalNormClip` (joint, f64 accumulation), `PerLayerClip`, `ValueClip`; `clip_grad_norm` helper
- [x] `grad_accum.rs` -- `GradientAccumulator`: k micro-batch accumulation; `Average` and `Sum` reduction modes

#### Gradient Checkpointing
- [x] `checkpoint.rs` -- `CheckpointPolicy` (Uniform/Selective/Offload/None); `CheckpointManager` save/retrieve/recompute; `RecomputeFn` closures; `CheckpointOverflow` error

#### Learning Rate Schedulers
- [x] `lr_scheduler.rs` -- 11 variants via `LrScheduler` trait
  - ConstantLR, StepLR, MultiStepLR, ExponentialLR (with `base_lr()` getters)
  - CosineAnnealingLR, LinearWarmup, WarmupCosine, PolynomialDecayLR, OneCycleLR, CyclicLR
  - ReduceLROnPlateau -- metric-based reduction with patience and `min_lr` floor

#### ZeRO Distributed Optimizer
- [x] `zero.rs` -- `ZeroStage::{Stage1, Stage2, Stage3}`; `shard_range(n) = (rank*chunk, min(start+chunk, n))`
- [x] `ZeroOptimizer<O: GpuOptimizer>` -- wraps any optimizer; Stage2 zeros non-owned gradients; Stage3 operates only on owned parameter shard
- [x] `ZeroMemoryEstimate` -- `bytes_per_rank()` and `reduction_ratio()` capacity planning helpers

#### Mixed Precision (AMP)
- [x] `amp.rs` -- `GradScaler` with dynamic loss scaling
  - `GradScalerConfig` (init_scale, growth_factor, backoff_factor, growth_interval)
  - `unscale()` / `step()` / `update()` workflow; skips step on overflow, halves scale
  - `has_overflow()` helper detecting `inf`/`NaN` in gradients
  - `AmpState` snapshot type for checkpointing

#### Exponential Moving Average
- [x] `ema.rs` -- `ExponentialMovingAverage` of shadow parameters
  - `EmaDecayMode::{Fixed, BiasCorrected, Polynomial}` for ramp-up schedules
  - `LayerDecay` per-layer decay override
  - `update()` / `copy_to()` / `swap()` lifecycle for evaluation checkpoints

#### Integration Tests
- [x] 6 E2E tests in `lib.rs`: AdamW + WarmupCosine + clip, Lion + grad accumulation, CAME + CyclicLR, Muon + ReduceLROnPlateau, ZeRO-2 single-rank, checkpoint + recompute

### Future Enhancements

#### P0 -- Critical (Production-Sensitive)
- [x] Fused Adam/AdamW PTX kernels with bias correction (`ptx_kernels.rs`)
- [x] Global-norm gradient clipping with f64 accumulation (`grad_clip.rs`)
- [x] ZeRO-1/2/3 shard partitioning (`zero.rs`)
- [x] Dynamic loss scaling for FP16/BF16 training (`amp.rs`)

#### P1 -- Important (Memory / Throughput)
- [x] CAME factored second moment (memory-efficient for large LLMs) -- `gpu_optimizer/came.rs`
- [x] Lion sign-based optimizer (50% memory vs Adam) -- `gpu_optimizer/lion.rs`
- [x] Gradient checkpointing with uniform / selective / offload policies -- `checkpoint.rs`
- [x] EMA shadow parameters with bias correction -- `ema.rs`

#### P2 -- Nice-to-Have (Algorithmic Variety)
- [x] Muon optimizer with Newton-Schulz orthogonalisation -- `gpu_optimizer/muon.rs`
- [x] RAdam variance rectification -- `gpu_optimizer/radam.rs`
- [x] 11-variant LR scheduler family (including ReduceLROnPlateau, OneCycleLR, CyclicLR) -- `lr_scheduler.rs`
- [ ] (P2) Multi-rank ZeRO collective integration -- requires `oxicuda-driver` NCCL-equivalent bring-up on Linux+NVIDIA hardware (currently single-rank verified only)
- [x] `optimizer/adopt.rs` — ADOPT (Taniguchi 2024): convergence without bounded second-moment assumption; adaptive gradient with corrected denominators dₜ=|∇ₜ|/(1-β₁ᵗ), stable at high β₁ and sparse grads
- [x] `optimizer/muon.rs` — Muon (Kosson 2024): momentum + Nesterov + orthogonalisation via Newton-Schulz iteration; updates W←W-η·orth(M) where M is Nesterov momentum; empirically faster than Adam on LLMs
- [x] `scheduler/wsd.rs` — WSD (Warmup-Stable-Decay, Hu 2024): three-phase schedule; stable LR plateau for extended training + fast cooldown; recommended for continued training of Llama-style models
- [x] `regularization/sophia.rs` — Sophia (Liu 2023): second-order optimizer via Hutchinson estimator of Hessian diagonal hₜ = ∇f·v where v∼N(0,I); update = η·g/(max(ρ·h,ε)); 2× fewer steps vs Adam on GPT-2
- [x] `optimizer/lamb.rs` — LAMB (You 2019): Layer-wise Adaptive Moments with trust-ratio r = ||w|| / ||g + λw||; scaled Adam update η · r · (m / (√v + ε)); converges large-batch BERT training; `GpuLamb { beta1, beta2, eps, weight_decay }`
- [x] `grad_clip/grad_clip.rs` / `GradClip` — gradient clipping abstraction covering `GlobalNormClip`, `PerLayerClip`, `ValueClip`; unified `GradClip` trait with `apply()` and `last_scale()` inspector; gradient_clipping policy enforced across all optimisers
- [x] `optimizer/adafactor.rs` — Adafactor (Shazeer & Stern 2018): sublinear-memory adaptive optimizer; factored per-row/per-col second moment `V̂[i,j]=R[i]·C[j]/(1ᵀR)` → O(m+n) for 2-D params (dense fallback for vectors); β̂₂(t)=1−t⁻ᶜ decay, update clipping by RMS, relative step ρ(t)=min(eps_rel,1/√t) and AdamW-style decoupled decay; `Adafactor { dim, shape, AdafactorConfig }`
- [x] `optimizer/shampoo.rs` — Shampoo (Gupta et al. 2018): preconditioned tensor optimization; two-sided per-dimension preconditioners L=ΣGGᵀ, R=ΣGᵀG with Ĝ=L⁻¹ᐟ⁴·G·R⁻¹ᐟ⁴ via cyclic-Jacobi symmetric eigendecomposition + damped inverse-pth-root (diagonal/AdaGrad fallback for vectors); optional heavy-ball momentum, lazy `precond_interval` root refresh
- [x] `scheduler/cosine_restart.rs` — SGDR (Loshchilov & Hutter 2017): cosine annealing with warm restarts; lr(T_cur)=η_min+½(η_max−η_min)(1+cos(π·T_cur/T_i)), geometric cycle growth T_{i+1}=⌈t_mult·T_i⌉; implements `LrScheduler` (`CosineAnnealingWarmRestarts`)
- [x] `training/swa.rs` — Stochastic Weight Averaging (Izmailov et al. 2018): equal-weight running parameter average W_SWA←(n·W_SWA+W)/(n+1) (distinct from EMA's exponential weighting) + companion `SwaLr` linear/cosine anneal-then-hold schedule
- [x] `training/label_smoothing.rs` — label smoothing CE (Szegedy et al. 2016): q_k=(1−α)[k=y]+α/K soft targets, numerically-stable (log-sum-exp) loss with closed-form `∂L/∂z=p−q` gradient, per-example and batch APIs; α=0 reduces to plain cross-entropy
- [x] `training/early_stopping.rs` — early stopping: Min/Max-mode metric monitor with `patience`, `min_delta` noise threshold, best/best-epoch tracking and `should_stop()`
- [x] `training/curriculum.rs` — competence-based curriculum (Platanios et al. 2019): Linear/Sqrt/Geometric/Step pacing functions c(t)∈[c₀,1] gating the admissible easy-to-hard example window over a difficulty-sorted dataset
- [x] `training/sampler.rs` — data-loader index samplers: `SequentialSampler`, `RandomSampler` (Fisher–Yates permutation / with-replacement), `WeightedRandomSampler` (inverse-CDF binary search for class rebalancing), `SubsetRandomSampler`, `BatchSampler` (drop-last); all built on the deterministic crate `LcgRng`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device/Host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |
| num-traits | Numeric trait bounds | Yes |
| serde (optional) | Checkpoint serialization (`serialize` feature) | Yes |

## Quality Status

- Warnings: 0 (clippy clean, `#![warn(missing_docs)]`)
- Tests: 364 lib + 9 doc passing (`cargo test -p oxicuda-train --all-features`)
- unwrap() calls: 0 (production code; test code documented with `.expect()` messages)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, returns `UnsupportedPlatform` at runtime

## Performance Targets

| Kernel | Workload | Target |
|--------|----------|--------|
| `adam_update_ptx` | 100M parameters, single launch | >= 90% bandwidth-limited peak on sm_80+ |
| `adamw_update_ptx` | 100M parameters, fused weight decay | >= 90% bandwidth-limited peak |
| `lion_update_ptx` | 100M parameters, half memory traffic | >= 95% bandwidth-limited peak |
| `norm_sq_partial_ptx` | 1B element global-norm reduction | >= 85% bandwidth-limited peak |
| ZeRO-2 step (32 GPU) | 7B-param model, mid-stage shard | >= 80% scaling vs ZeRO-1 |

## Numerical Accuracy Requirements

| Operation | Tolerance vs FP64 reference |
|-----------|-----------------------------|
| Adam step (single update) | abs < 1e-5, rel < 1e-4 |
| AdamW step (with weight decay) | abs < 1e-5, rel < 1e-4 |
| Lion step (sign update) | exact match modulo +/-lr quantum |
| Global norm clip | abs < 1e-5 on >= 10K element tensor |
| EMA bias-corrected decay | abs < 1e-6 per step |

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86 / sm_89)
- [x] PTX header selection emits `.target sm_80` for cp.async-capable optimizer kernels
- [ ] `cp.async.shared` overlap for staged moment loads (deferred, requires hardware verification)

### Hopper (sm_90 / sm_90a)
- [x] PTX header selection emits `.target sm_90` for warp-grouped operations
- [ ] TMA-based moment streaming for FP16/BF16 optimizer states (deferred)

### Blackwell (sm_100 / sm_120)
- [x] PTX header generation supports sm_100/sm_120 (matches root SM table)
- [ ] FP4/FP6 optimizer state storage (waits on `oxicuda-blas` FP4/FP6 codecs)

## Deepening Opportunities

### Verification Gaps
- [x] Optimizer convergence smoke tests (AdamW, Lion, CAME, Muon, RAdam, RMSProp) integrated in `lib.rs::tests`
- [x] ZeRO sharding math validated single-rank in `e2e_zero_stage2_with_adamw`
- [ ] Multi-rank ZeRO correctness against a single-rank baseline (requires Linux+NVIDIA cluster)
- [ ] AMP overflow / underflow trace verified end-to-end on real FP16 GEMM gradient distributions

### Implementation Deepening
- [x] All 8 GPU optimizers expose the same `GpuOptimizer` trait surface
- [x] LR schedulers all expose `base_lr()` getters for resume / checkpoint
- [x] CheckpointManager supports `RecomputeFn` closures for arbitrary forward functions
- [x] EMA `LayerDecay` allows per-layer override (transformer block depth-decay schedules)
- [ ] Gradient communication backend pluggability (NCCL/UCX/MPI shim) -- pending Vol.12 integration

## Notes

- Several integration tests use `.expect()` messages in test code (not production); production code paths return `TrainResult<T>`.
- `serialize` feature gates `serde::Serialize`/`Deserialize` derives on `AmpState`, `ZeroConfig`, and checkpoint metadata.
- Benchmarks live in `benches/train_ops.rs` (Criterion harness) -- CPU-side dispatch heuristics only; GPU benchmarking awaits hardware.
