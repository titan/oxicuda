# oxicuda-gen TODO

Pure-Rust Generative AI primitives for OxiCUDA: diffusion schedulers
(DDPM/DDIM/DPM-Solver++/Flow Matching), classifier-free guidance, VQ-VAE codec,
LoRA adapters, and score-network building blocks. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.17).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 7,596 (25 files, Rust 5,685 code + 1,056 comments + 855 blanks)
- **Tests:** 221 passing (#[test] count in src/)
- **Crate:** `oxicuda-gen` -- Vol.17 Generative AI Primitives

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `GenError` (15 variants): `DimensionMismatch`, `InvalidBetaRange`,
      `InvalidGuidanceScale`, `UnsupportedDpmOrder`, `InvalidTimestep`,
      `InvalidCodebookSize`, `WeightShapeMismatch`, `InvalidLoraRank`, etc.; `GenResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (seed-based, Box-Muller normals), `GenHandle`
- [x] `lib.rs` -- crate root with `prelude` module and 11 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 6 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `ddpm_step_ptx` -- `x_{t-1} = (x_t - beta/sqrt(1-alpha_bar) * eps_hat) / sqrt(alpha) + sigma*z`
      via `sqrt.approx`, `rcp.approx`
- [x] `cfg_combine_ptx` -- `out = u + s*(c - u)` classifier-free guidance blend
- [x] `lora_apply_ptx` -- `y = x*W + (alpha/r)*x*B*A` low-rank update; grid-stride loop
- [x] `flow_velocity_ptx` -- Euler step `x_{t+delta} = x_t + delta*v(x_t, t)` for flow ODE
- [x] `vae_kl_loss_ptx` -- `0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))` latent KL divergence
- [x] `timestep_embed_ptx` -- sinusoidal timestep embedding via `sin/cos/lg2/ex2`
- [x] `f32_hex` -- shared float literal helper

#### Schedulers (`scheduler/`, 5 files)
- [x] `scheduler/beta_schedule.rs` -- `BetaSchedule`, `BetaScheduleType`: linear, cosine
      (Nichol & Dhariwal), scaled-cosine, sigmoid; `alphas_bar`, `sqrt_alphas_bar`,
      `sqrt_one_minus_alphas_bar`
- [x] `scheduler/ddpm.rs` -- `DdpmScheduler`: `add_noise()` with `q(x_t|x_0)`, `step()` reverse
      DDPM update with fixed sigma^2 = beta_t
- [x] `scheduler/ddim.rs` -- `DdimScheduler`: eta-parameterised deterministic/stochastic step;
      eta=0 deterministic verified in tests
- [x] `scheduler/dpm_solver.rs` -- `DpmSolverScheduler`: exponential integrator on
      `lambda_t = log(alpha_t/sigma_t)`; 1st/2nd-order multi-step (`DpmOrder`);
      `num_train_steps()` accessor
- [x] `scheduler/flow_matching.rs` -- `FlowMatchingScheduler`: linear OT path
      `x_t = (1-t)x_0 + t*x_1`; Euler and Heun ODE solvers; boundary conditions verified
- [x] `scheduler/mod.rs` -- module organisation + re-exports

#### Guidance (`guidance/`, 3 files)
- [x] `guidance/cfg.rs` -- `CfgConfig`, `CfgGuidance`: `eps_hat = uncond + s*(cond - uncond)`
      with scale-clipping and rescaling
- [x] `guidance/perp_neg.rs` -- `PerpNegGuidance`: perpendicular-negative prompt guidance
- [x] `guidance/adaptive.rs` -- `AdaptiveCfgPolicy`, `AdaptiveCfgScheduler`: constant,
      linear, cosine, stepwise dynamic-scale scheduling

#### VAE (`vae/`, 4 files)
- [x] `vae/kl.rs` -- `GaussianLatent`: reparameterised sampling `z = mu + eps*sigma`,
      `kl_loss()`, `standard_normal()`
- [x] `vae/quantize.rs` -- `VqCodebook`: EMA codebook update
      `e_k <- gamma*e_k + (1-gamma)*sum(x_j)`, nearest-entry lookup, commitment loss
- [x] `vae/encoder.rs` -- `Encoder`, `EncoderConfig`, `EncoderWeights`: ResNet down-blocks
      (GELU + GroupNorm); `EncoderWeights::zeros()`
- [x] `vae/decoder.rs` -- `Decoder`, `DecoderConfig`, `DecoderWeights`: mirrored up-sampling
      blocks; `DecoderWeights::zeros()`

#### LoRA (`lora/`, 2 files + mod)
- [x] `lora/adapter.rs` -- `LoraConfig`, `LoraLinear`, `LoraModel`:
      `W' = W + (alpha/r)*B*A`; B from Gaussian init, A = 0 init; `forward()` adds rank-r
      correction; named adapter collection with `add_adapter()`/`apply()`
- [x] `lora/merge.rs` -- `merge_lora`, `unmerge_lora`, `verify_merge_roundtrip`,
      `scale_adapter`, `compose_adapters`

#### Score networks (`score/`, 2 files + mod)
- [x] `score/timestep.rs` -- `SinusoidalEmbedding`, `FourierEmbedding`: sin+cos pair
      embedding with `sin^2 + cos^2 = 1` invariant verified
- [x] `score/unet_block.rs` -- `UNetResBlock`, `SelfAttentionBlock`, `CrossAttentionBlock`:
      SiLU activation + time-embedding injection + multi-head attention

#### Integration tests (`lib.rs::tests`)
- [x] 11 E2E tests: DDPM forward/reverse consistency, DDIM eta=0 determinism, DPM-Solver
      shape, Flow-Matching boundary, CFG combine, LoRA round-trip, VAE KL, VQ commit,
      sinusoidal embedding orthogonality, PTX generation x 6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Correctness / Mainstream Diffusion Coverage)
- [ ] DPM-Solver++ 3rd-order multi-step (currently `DpmOrder::First/Second`)
- [x] EDM (Karras et al. 2022) sigma-schedule + preconditioning helpers — scheduler/edm.rs (c_skip/c_out/c_in/c_noise, Heun's ODE, log-normal σ sampling, full trajectory sampler)
- [ ] DDPM/DDIM `step()` GPU dispatch via `ddpm_step_ptx` (currently CPU helper only)
- [ ] Classifier-Guidance (gradient-of-classifier) as alternative to CFG

#### P1 -- Important (Model-Architecture Coverage)
- [ ] Cross-attention KV-cache for fast sampling (1-step text-conditioning reuse)
- [ ] Rotary positional embedding (RoPE) variant of `SelfAttentionBlock`
- [ ] FlashAttention-style fused softmax block (link with oxicuda-dnn fused MHA)
- [ ] `VqCodebook::ema_decay` warm-up schedule + dead-code (unused entry) reinit
- [ ] `Decoder::forward` reference path + test against `Encoder` round-trip
- [ ] Mixed-rank LoRA: per-layer-different `r` selection

#### P2 -- Nice-to-Have (Advanced / Research)
- [x] Consistency Models (Song et al. 2023) `ConsistencyScheduler` — scheduler/consistency.rs (sigma schedule, c_skip/c_out preconditioning, one-step and multi-step sampling, consistency distillation loss)
- [x] Rectified Flow (Liu 2023) higher-order solver (scheduler/rectified_flow.rs -- linear interpolation path, constant target velocity x1−x0, Euler/Heun ODE sampling, reflow pair generation, straightness metric)
- [x] DoRA (weight-decomposed LoRA) adapter variant (lora/dora.rs -- Liu 2024 ICML; W'=m·(W0+BA)/‖W0+BA‖_row per-output-row, trainable magnitude m + LoRA direction update; decoupled magnitude/direction)
- [ ] QLoRA NF4 quantised base-weight path
- [x] Stochastic Interpolants generalisation of flow matching (scheduler/stochastic_interpolant.rs -- Albergo-Vanden-Eijnden 2023; X_t=α(t)x0+β(t)x1+σ(t)z unifying framework, target velocity α'x0+β'x1; LinearFlow/TrigInterpolant/NoisyLinear kinds; Euler ODE sample)
- [x] V-prediction parameterisation (scheduler/v_prediction.rs -- Salimans & Ho 2022; v=α_t ε−σ_t x orthonormal-rotation parameterization with exact predict_x0/predict_eps inverses, SNR, constant loss weight)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader (`libcuda.so` / `nvcuda.dll`).

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 221 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All (largest is `ptx_kernels.rs` at ~890 lines)
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

Generative-AI primitives are dominated by GEMM/attention (delegated to
`oxicuda-blas` / `oxicuda-dnn`). This crate's PTX kernels target:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `ddpm_step_ptx` | latent dim 4 x 64 x 64 (= 16,384) per timestep | P0 |
| `cfg_combine_ptx` | latent dim 4 x 64 x 64 | P0 |
| `lora_apply_ptx` | rank r in {4, 8, 16, 32}; d in {2048, 4096} | P1 |
| `flow_velocity_ptx` | same as `ddpm_step_ptx` | P1 |
| `vae_kl_loss_ptx` | latent dim 4 x 64 x 64 | P2 |
| `timestep_embed_ptx` | embed_dim 320..1280 | P2 |

Target: bandwidth-bound kernels at >=90% peak DRAM throughput on sm_80+.

## Notes

- All schedulers operate on flat `&[f32]` slices -- caller manages batch/spatial layout
- `LcgRng` is deterministic; tests pin `seed = 42` for reproducibility
- DPM-Solver works on the log-SNR domain `lambda_t = log(alpha_t/sigma_t)`
- Flow-matching boundary check enforces `x(0) = x_0`, `x(1) = x_1` within 1e-6
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `ddpm_step_ptx` uses `sqrt.approx.f32` / `rcp.approx.f32` (HW SFU)
- [x] `cfg_combine_ptx` issues coalesced `ld.global.f32` (vectorisable to v4)
- [ ] `lora_apply_ptx` upgraded to use Tensor Cores (mma.sync m16n8k16) for r >= 8
- [ ] `cp.async` double-buffer for `B*A` rank-update path
- [x] PTX × SM 80, 86 generation verified in integration tests

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 6 kernels
- [ ] `lora_apply_ptx` uses `wgmma.mma_async` for rank-update path
- [ ] TMA (`cp.async.bulk`) for latent-tensor staging in `ddpm_step_ptx`
- [ ] Cluster-launch variant for very-large batch sampling (>=4096)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) `lora_apply_ptx` path for low-rank inference on Blackwell
- [ ] Tensor-Memory (TMEM) optimised latent staging

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] All schedulers: shape + finiteness + numerical-stability tests
- [x] DDIM `eta=0` produces deterministic two-call result (within 1e-5)
- [x] Flow-matching boundary conditions `x(0)`, `x(1)` exact (within 1e-6)
- [x] LoRA `verify_merge_roundtrip()` checks `merge -> unmerge ≈ identity`
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 6 kernels (gated behind `gpu-tests`)
- [ ] Numerical agreement with reference PyTorch (`diffusers`) implementations within 1e-3
      relative for full 50-step DDIM sample
- [ ] LoRA inference accuracy vs HF `peft` reference within 1e-4 relative
- [ ] VQ-VAE codebook usage > 80% after training simulation

### Implementation Deepening
- [ ] `Encoder::forward` and `Decoder::forward` end-to-end with batched-tensor reshape
      (currently `EncoderWeights::zeros` constructors only)
- [ ] U-Net full forward (down/mid/up assembly) -- currently per-block primitives only
- [ ] Sampling loop helper (`sample_ddim`, `sample_dpm_solver`) for end-to-end inference
- [ ] LoRA save/load via `oxicuda-runtime` checkpoint format (round-trip verified)
- [ ] Adaptive CFG schedule curve plotting / fitting helpers

### Benchmark Coverage
- [x] `benches/gen_ops.rs` Criterion harness wired (CPU-side PTX generation + scheduler step)
- [ ] GPU-side throughput numbers vs reference (cuDNN attention, HF Diffusers) once
      Linux+NVIDIA harness is available
- [ ] LoRA rank-r ablation (r in 4/8/16/32) on representative GEMM sizes
