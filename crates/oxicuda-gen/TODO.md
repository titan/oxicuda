# oxicuda-gen TODO

Pure-Rust Generative AI primitives for OxiCUDA: diffusion schedulers
(DDPM/DDIM/DPM-Solver++/Flow Matching), classifier-free guidance, VQ-VAE codec,
LoRA adapters, and score-network building blocks. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.17).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Files:** 52 `.rs` files in src/
- **Tests:** 596 passing (#[test] count in src/) + 1 doctest
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
      blocks; `DecoderWeights::zeros()`; `Decoder::forward()` explicit-batch reconstruction
      (encode→decode spatial round-trip)

#### LoRA (`lora/`, 2 files + mod)
- [x] `lora/adapter.rs` -- `LoraConfig`, `LoraLinear`, `LoraModel`:
      `W' = W + (alpha/r)*B*A`; B from Gaussian init, A = 0 init; `forward()` adds rank-r
      correction; named adapter collection with `add_adapter()`/`apply()`
- [x] `lora/merge.rs` -- `merge_lora`, `unmerge_lora`, `verify_merge_roundtrip`,
      `scale_adapter`, `compose_adapters`
- [x] `lora/checkpoint.rs` -- `save`/`load` round-trip for `LoraModel` via a hand-rolled
      versioned little-endian byte format (`OXLORA01`); bit-for-bit A/B/rank/scaling/alpha
      identity; `LoraLinear::from_parts` exact-scaling constructor

#### Score networks (`score/`, 2 files + mod)
- [x] `score/timestep.rs` -- `SinusoidalEmbedding`, `FourierEmbedding`: sin+cos pair
      embedding with `sin^2 + cos^2 = 1` invariant verified
- [x] `score/unet_block.rs` -- `UNetResBlock`, `SelfAttentionBlock`, `CrossAttentionBlock`:
      SiLU activation + time-embedding injection + multi-head attention
- [x] `score/unet_full.rs` -- `UNet`, `UNetConfig`, `UNetWeights`, `ResBlockWeights`,
      `AttnWeights`: full down/mid/up assembly with skip connections, 2×2 avg-pool /
      nearest-neighbour up/downsampling, bottleneck self-attention, and broadcast
      timestep embedding; resolution + channel count preserved

#### Integration tests (`lib.rs::tests`)
- [x] 11 E2E tests: DDPM forward/reverse consistency, DDIM eta=0 determinism, DPM-Solver
      shape, Flow-Matching boundary, CFG combine, LoRA round-trip, VAE KL, VQ commit,
      sinusoidal embedding orthogonality, PTX generation x 6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Correctness / Mainstream Diffusion Coverage)
- [x] DPM-Solver++ 3rd-order multi-step (currently `DpmOrder::First/Second`)
- [x] EDM (Karras et al. 2022) sigma-schedule + preconditioning helpers — scheduler/edm.rs (c_skip/c_out/c_in/c_noise, Heun's ODE, log-normal σ sampling, full trajectory sampler)
- [x] DDPM/DDIM `step()` GPU dispatch via `ddpm_step_ptx` (currently CPU helper only)
- [x] Classifier-Guidance (gradient-of-classifier) as alternative to CFG

#### P1 -- Important (Model-Architecture Coverage)
- [x] Cross-attention KV-cache for fast sampling (1-step text-conditioning reuse) —
      score/kv_cache.rs (`CrossAttentionKvCache`: projects fixed context K/V once via
      `build()`, reuses across N denoising steps via `attend()`; bit-for-bit identical
      to `CrossAttentionBlock` verified in test)
- [x] Rotary positional embedding (RoPE) variant of `SelfAttentionBlock` —
      score/rope_attention.rs (`RotaryEmbedding`, `RopeSelfAttention`; relative-position
      invariance via `forward_with_offset`)
- [x] FlashAttention-style fused softmax block (link with oxicuda-dnn fused MHA) —
      score/flash_attention.rs (`FlashAttention`: tiled online-softmax recurrence
      running-max/running-sum/rescaled accumulator, never materialises S×S scores;
      causal masking; matches naive oracle and is block-size invariant)
- [x] `VqCodebook::ema_decay` warm-up schedule + dead-code (unused entry) reinit —
      vae/ema_codebook.rs (`current_decay()` warm-up ramp, `revive_dead_codes()` reinit
      of unused entries via `steps_since_used`)
- [x] `Decoder::forward` reference path + test against `Encoder` round-trip
- [x] Mixed-rank LoRA: per-layer-different `r` selection — lora/mixed_rank.rs
      (`LayerSpec`/`MixedRankLoraModel`: independent rank+alpha per named layer;
      `RankBudget` greedily allocates per-layer ranks under a global parameter budget,
      Uniform / WidthProportional strategies)

#### P2 -- Nice-to-Have (Advanced / Research)
- [x] Consistency Models (Song et al. 2023) `ConsistencyScheduler` — scheduler/consistency.rs (sigma schedule, c_skip/c_out preconditioning, one-step and multi-step sampling, consistency distillation loss)
- [x] Rectified Flow (Liu 2023) higher-order solver (scheduler/rectified_flow.rs -- linear interpolation path, constant target velocity x1−x0, Euler/Heun ODE sampling, reflow pair generation, straightness metric)
- [x] DoRA (weight-decomposed LoRA) adapter variant (lora/dora.rs -- Liu 2024 ICML; W'=m·(W0+BA)/‖W0+BA‖_row per-output-row, trainable magnitude m + LoRA direction update; decoupled magnitude/direction)
- [x] QLoRA NF4 quantised base-weight path — lora/qlora.rs (Dettmers 2023;
      `Nf4Tensor` block-wise 4-bit NormalFloat with 16-level normal-quantile codebook +
      per-block absmax scaling, two codes packed per byte; `QLoraLinear` forward
      = dequant(NF4 frozen base)·xᵀ + LoRA correction; `NF4_LEVELS` const)
- [x] Stochastic Interpolants generalisation of flow matching (scheduler/stochastic_interpolant.rs -- Albergo-Vanden-Eijnden 2023; X_t=α(t)x0+β(t)x1+σ(t)z unifying framework, target velocity α'x0+β'x1; LinearFlow/TrigInterpolant/NoisyLinear kinds; Euler ODE sample)
- [x] V-prediction parameterisation (scheduler/v_prediction.rs -- Salimans & Ho 2022; v=α_t ε−σ_t x orthonormal-rotation parameterization with exact predict_x0/predict_eps inverses, SNR, constant loss weight)
- [x] `diffusion/flow_matching.rs` — Conditional Flow Matching (Lipman 2022): simple velocity field u_t|x₁=x₁-x₀; Gaussian conditional probability path; simulation-free training; `CfmConfig { sigma_min: f32 }`
- [x] `diffusion/consistency.rs` — Consistency Models (Song 2023): learn self-consistency property f(x_t,t)=x₀; consistency distillation from diffusion teacher; one/two-step generation; `ConsistencyModel { steps: usize }` — IMPLEMENTED under sibling filename scheduler/consistency.rs (`ConsistencyScheduler`: sigma schedule, c_skip/c_out preconditioning, `single_step_sample`/`multi_step_sample`, `consistency_loss` distillation against EMA teacher)
- [x] `vae/vq_vae2.rs` — VQ-VAE-2 (Razavi 2019): hierarchical two-level discrete codes (top + bottom); PixelSnail prior on top codes; commitment loss + EMA codebook updates
- [x] `gan/stylegan3.rs` — StyleGAN3 alias-free operations (Karras 2021): equivariant generator with sinc-filtered up/downsampling; rotation/translation equivariance; `StyleGan3Config { c_dim, w_dim }` — IMPLEMENTED gan/stylegan3.rs (`AliasFreeOps` Kaiser low-pass FIR up/downsample + filtered non-linearity, `MappingNetwork`, `SynthesisLayer`, `StyleGan3Generator`)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader (`libcuda.so` / `nvcuda.dll`).

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 596 passing
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

> NOTE: The unchecked `[ ]` items below are **hardware-gated** — Tensor-Core
> `mma.sync`/`wgmma`, `cp.async`/TMA staging, cluster launch, FP8 (E4M3) and TMEM
> paths require a real NVIDIA device (sm_80+/sm_90+/sm_100+) to author and verify.
> They are intentionally left unchecked; PTX *emission* for all kernels across the
> six SM versions is already tested on CPU. (requires GPU hardware)

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
- [ ] GPU-hardware correctness for all 6 kernels (gated behind `gpu-tests`) (requires GPU hardware)
- [ ] Numerical agreement with reference PyTorch (`diffusers`) implementations within 1e-3
      relative for full 50-step DDIM sample (requires PyTorch/diffusers reference + GPU)
- [ ] LoRA inference accuracy vs HF `peft` reference within 1e-4 relative
      (requires HF peft reference)
- [x] VQ-VAE codebook usage > 80% after training simulation —
      vae/ema_codebook.rs::tests::codebook_usage_exceeds_80_percent_after_training
      (16-cluster grid data, 400 `train_step` iterations with EMA + dead-code revival;
      asserts > 80% of codes assigned)

### Implementation Deepening
- [x] `Encoder::forward` and `Decoder::forward` end-to-end with batched-tensor reshape
      (vae/decoder.rs -- `Decoder::forward(z, weights, batch)` adds the symmetric
      explicit-batch counterpart to `Encoder::encode`: validates the per-row
      `[batch × latent_dim]` reshape before running the shared residual stack and
      returns `[batch × out_channels]`; test `encode_decode_roundtrip_preserves_spatial_dims`
      threads encode→decode and asserts reconstruction matches the encoder's input
      spatial/channel dims, plus finite-output + bad-batch rejection tests)
- [x] U-Net full forward (down/mid/up assembly) -- score/unet_full.rs
      (`UNet`/`UNetConfig`/`UNetWeights`: wires `UNetResBlock` + `SelfAttentionBlock`
      into a real down/mid/up network over channel-last `[H × W × C]` tensors;
      down path runs per-level residual blocks then 2×2 avg-pool downsample saving
      skips, mid path is resblock + bottleneck self-attention over flattened tokens,
      up path 2×2 nearest-neighbour upsamples + adds skips + narrows channels, final
      block projects back to `in_channels`; timestep embedding broadcast to every
      token through every block; tests assert H×W resolution + channel count preserved
      under zero and non-trivial weights, timestep actually influences output, and
      indivisible-resolution rejection)
- [x] Sampling loop helper (`sample_ddim`, `sample_dpm_solver`) for end-to-end inference —
      closure-driven sampling loops exist under sibling names: scheduler/heun.rs
      (`sample_euler`/`sample_heun`/`sample_euler_ancestral`), solver/dpm_solver_pp.rs
      (`DpmSolverPp::sample`), solver/unipc.rs (`UniPc::sample`), solver/pndm.rs
      (`PndmSolver::sample`), scheduler/rectified_flow.rs (`RectifiedFlow::sample`),
      scheduler/stochastic_interpolant.rs (`StochasticInterpolant::sample_ode`)
- [x] LoRA save/load via checkpoint format (round-trip verified) -- lora/checkpoint.rs
      (`save`/`load` for `LoraModel` using a hand-rolled, versioned, length-prefixed
      little-endian byte layout -- magic `OXLORA01`, config rank/alpha/dropout/target
      modules, then per-adapter name/in/out/rank/scaling/A/B in sorted key order for
      determinism; serde deliberately avoided as it is not a dependency of this crate.
      Added `LoraLinear::from_parts` so the `scaling` field round-trips bit-for-bit;
      test `save_load_roundtrip_identity` asserts A/B/rank/scaling/alpha reproduced
      bit-for-bit, plus deterministic-output, forward-output-equivalence, empty-model,
      bad-magic/truncated/trailing-byte rejection tests)
- [x] Adaptive CFG schedule curve plotting / fitting helpers -- guidance/adaptive.rs
      (`AdaptiveCfgScheduler::sample_on_grid`/`sample_uniform_grid` sample the
      guidance-scale curve over arbitrary/uniform step grids; `fit_polynomial` plus a
      standalone `PolynomialFit` least-squares-fits a low-order polynomial to the curve
      by solving the Vandermonde normal equations via in-house f64 Gaussian elimination
      with partial pivoting (`eval`/`coeffs`/`rmse`); tests assert grid samples match
      the scheduler exactly, a known quadratic `1+2x+3x²` is recovered within 1e-3, the
      linear policy is reproduced by a degree-1 fit, the cosine policy gets <0.1 RMSE at
      degree 4, and under-determined / mismatched-length fits are rejected)

### Benchmark Coverage
- [x] `benches/gen_ops.rs` Criterion harness wired (CPU-side PTX generation + scheduler step)
- [ ] GPU-side throughput numbers vs reference (cuDNN attention, HF Diffusers) once
      Linux+NVIDIA harness is available (requires GPU hardware)
- [ ] LoRA rank-r ablation (r in 4/8/16/32) on representative GEMM sizes (requires GPU hardware)
