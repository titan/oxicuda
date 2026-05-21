# oxicuda-adversarial TODO

Pure-Rust adversarial robustness primitives for OxiCUDA, covering both the
attack side (FGSM, PGD L_inf / L2, MIM, CW, AutoPGD) and the defence side
(TRADES, MART, randomized smoothing, IBP / certified bounds). Includes Lp-ball
threat-model primitives, epsilon-budget tracking, and robustness evaluation
metrics. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.27).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 4,943 (21 files)
- **PTX kernels:** 7 kernel generators emitted for 6 SM targets (sm_75 / 80 / 86 / 90 / 100 / 120)
- **Coverage:** CPU reference implementation + PTX string generation for GPU execution

### Completed

#### Core Infrastructure
- [x] `error.rs` (110 LoC) -- `AdvError` (15 variants: DimensionMismatch, EmptyInput, InvalidEpsilon, InvalidAlpha, InvalidNumSteps, InvalidLpNorm, InvalidTemperature, InvalidNoiseSigma, InvalidConfidence, InsufficientCertSamples, InvalidLossWeight, BudgetExceeded, NanEncountered, OptimizationDiverged, AttackFailedAll, Internal) + `AdvResult<T>`
- [x] `handle.rs` (242 LoC) -- `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle, uniform `[0, 1)`, Knuth MMIX 64-bit LCG), `AdvHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` (255 LoC) -- Module exports + `prelude` re-exports + 13 E2E integration tests

#### PTX Kernels (ptx_kernels.rs, 672 LoC)
- [x] `fgsm_step_ptx` -- `x_adv[i] = clamp(x[i] + eps * sign(grad[i]), lo, hi)` with `fma.rn.f32` and grid-stride loop
- [x] `pgd_proj_l_inf_ptx` -- L_inf projection: `out[i] = clamp(clamp(x[i], x_orig[i] - eps, x_orig[i] + eps), lo, hi)`
- [x] `pgd_proj_l2_ptx` -- L2 projection: scale `delta = x - x_orig` so `||delta||_2 <= eps` via host-supplied norm + `div.rn.f32`
- [x] `smoothing_noise_ptx` -- Gaussian noise `z ~ N(0, sigma^2)` via inline LCG + Box-Muller using `lg2.approx.f32`
- [x] `grad_sign_ptx` -- `out[i] = sign(grad[i])` (`+1 / 0 / -1`) using `selp.f32` double-predicate
- [x] `certified_radius_reduce_ptx` -- Per-block argmax over `[K]` class-count vector for smoothed-predictor read-off (`selp.b32` swap pattern)
- [x] `attack_loss_grad_ptx` -- `out[i] = clamp(x[i] + alpha * dir[i], lo, hi)` inner step for MIM / PGD with momentum-accumulated gradient

#### Attacks (attacks/)
- [x] `fgsm.rs` (203 LoC) -- `fgsm_attack` single-step Fast Gradient Sign Method (Goodfellow 2014): `x_adv = clamp(x + eps * sign(grad_L), lo, hi)`
- [x] `pgd.rs` (364 LoC) -- `pgd_attack_l_inf` / `pgd_attack_l2` / `PgdConfig` Projected Gradient Descent with optional random restart and L_inf / L2 projections (Madry 2018)
- [x] `mim.rs` (254 LoC) -- `mim_attack` / `MimConfig` Momentum Iterative Method with exponential momentum accumulation (Dong 2018)
- [x] `cw.rs` (356 LoC) -- `cw_attack` / `CwConfig` Carlini-Wagner L2 attack with binary-search confidence parameter and change-of-variable `tanh` reparametrisation (Carlini 2017)
- [x] `auto_pgd.rs` (398 LoC) -- `auto_pgd_attack` / `AutoPgdConfig` AutoPGD with step-size schedule and checkpointing (Croce 2020)

#### Defenses (defenses/)
- [x] `trades.rs` (322 LoC) -- `trades_loss` / `TradesConfig` regulariser: `CE(clean) + beta * KL(clean || adv)` with KL computed from log-softmax pairs (Zhang 2019)
- [x] `mart.rs` (361 LoC) -- `mart_loss` / `MartConfig` misclassification-aware adversarial training: BCE on natural examples + weighted KL term (Wang 2020)
- [x] `randomized_smoothing.rs` (423 LoC) -- `smoothed_predict` / `certified_radius` / `RsConfig` Cohen (2019) randomized smoothing: Monte-Carlo majority vote + Binomial CI for certified L2 radius
- [x] `certified_bounds.rs` (402 LoC) -- `ibp_propagate` / `IntervalBound` Interval Bound Propagation through affine layers with per-bound `relu()`; `lipschitz_certified_radius` formula `m / (L * sqrt(2))`

#### Threat Model (threat_model/)
- [x] `lp_ball.rs` (190 LoC) -- `LpNorm` enum + `l_inf_norm` / `l1_norm` / `l2_norm` / `project_l_inf` / `project_l2` Lp-ball norm computations and projections
- [x] `budget.rs` (117 LoC) -- `EpsilonBudget::new(total)` -> `spend(amount)?` / `remaining()` epsilon-budget tracker with `AdvError::BudgetExceeded` on overdraft

#### Metrics (metrics/)
- [x] `robust_accuracy.rs` (166 LoC) -- `robust_accuracy` (fraction of adversarial examples predicted correctly) + `certified_accuracy` (correct AND certified at radius >= threshold)
- [x] `asr.rs` (77 LoC) -- `attack_success_rate` (fraction of adversarial examples on which the attack changed the prediction)

#### Integration Tests (lib.rs)
- [x] 13 E2E tests: FGSM pushes away from target, PGD L_inf / L2 respect epsilon-ball, MIM with zero momentum-decay matches PGD, TRADES collapses to CE when clean = adv, MART loss finite, RS constant classifier returns top class, IBP propagates through ReLU, Lipschitz radius formula, robust / certified accuracy, PTX kernels x 6 SM versions, epsilon-budget lifecycle

#### Benchmarks (benches/adv_ops.rs)
- [x] 7 PTX kernel generator groups x 4 SM versions + 4 algorithm benches: `fgsm_attack_d1024`, `pgd_l_inf_attack_d512_n10`, `trades_loss_b64_k10`, `ibp_propagate_64x32`

### Future Enhancements

#### P0 -- Critical (Attack & Defense Coverage Gaps)
- [x] AutoAttack ensemble -- standard robustness-evaluation suite combining APGD-CE + APGD-DLR + FAB + Square; we have APGD already (Croce 2020)
- [x] DeepFool minimum-perturbation attack -- linearised iterative attack producing tight L2 perturbations (Moosavi-Dezfooli 2016)
- [x] Square Attack -- black-box score-based attack with random search; reference baseline for AutoAttack-Square
- [x] Universal adversarial perturbation (UAP) -- batch-independent perturbation searched across a sample set (Moosavi-Dezfooli 2017)
- [x] Adversarial weight perturbation (AWP) defense -- weight-space smoothness regulariser layered on top of TRADES / MART (Wu 2020)

#### P1 -- Important (Certified-Robustness & Threat-Model Depth)
- [x] CROWN / alpha-CROWN tighter bound propagation -- improves over IBP especially through ReLU (Zhang 2018, Xu 2021)
- [x] Macer / SmoothAdv -- training procedures that maximise the smoothed-classifier certified radius (Salman 2019, Zhai 2020) (defenses/macer.rs -- Salman 2019 / Zhai 2020; certified L2 radius r=σ·Φ⁻¹(p̂_top) via Acklam-probit + hinge loss λ·max(0,γ−r) added to cls_loss; smoothed_predict via N(0,σ²I) averaging)
- [x] Randomized smoothing for L_inf -- replace Gaussian noise with Laplace / exponential noise for L1 / L_inf certificates (defenses/laplace_smoothing.rs -- Teng 2020; per-coordinate Laplace(0,b) noise via inverse-CDF; L1 certified radius r=(b/2)·ln(p̂_top/(1−p̂_top)); distinct from existing Gaussian randomized_smoothing.rs)
- [ ] LP-relaxation-based verification for small MLPs as a reference oracle for IBP tightness
- [x] Sparse / L0 attacks (JSMA, sparse-PGD) -- complement to the existing L_inf / L2 attacks (attacks/jsma.rs -- Papernot 2016; Jacobian saliency map, iterative most-salient-feature L0 perturbation toward target class)
- [x] Patch attack -- bounded-support adversarial sticker as a structured threat model (attacks/patch.rs -- Brown 2017; bounded-support rectangular patch, PGD-style ascent restricted to patch region, apply_patch + patch_mask)
- [ ] Targeted variant of every attack -- currently `fgsm_attack` / `pgd_attack_*` take a gradient closure but no explicit target-class API

#### P2 -- Nice-to-Have (Evaluation & Tooling)
- [ ] Stratified robust-accuracy reporter (per-class robust accuracy + worst-class accuracy)
- [ ] Gradient-masking diagnostics (Athalye 2018) -- BPDA / EOT helpers to detect obfuscated gradients in custom defences
- [ ] Loss-landscape probing utilities -- multi-restart PGD with random initialisation distance histograms
- [ ] Transferability matrix helper -- run attack from model A on model B and tabulate success rates
- [ ] CIFAR-10 / ImageNet-C corruption-robustness eval wrappers on top of `robust_accuracy`

#### GPU Launcher Wiring
- [ ] Wire `ptx_kernels::*` strings through `oxicuda-launch::Kernel::from_module` for end-to-end GPU execution (PTX strings are emitted but CPU reference paths are the authoritative driver in attacks / defenses)
- [ ] GPU-resident `pgd_attack_l_inf` inner loop using `fgsm_step_ptx` + `pgd_proj_l_inf_ptx` fused; gradient closure becomes a `oxicuda-blas` callout
- [ ] GPU-resident randomized smoothing using `smoothing_noise_ptx` for batched noise generation followed by `certified_radius_reduce_ptx` for majority-vote read-off

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device / Host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 13 E2E in `lib.rs` + module unit tests (see root TODO.md Vol.27 reference for the workspace-wide count)
- `unwrap()` calls: 0 in library code
- macOS: compiles, returns `UnsupportedPlatform` from any actual GPU launch
- PTX targets covered: sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120

## Performance Targets

| Operation | Size | Target |
|-----------|------|--------|
| PTX kernel string generation | per call | < 100 us |
| `fgsm_attack` (gradient closure dominates) | D = 224 * 224 * 3 | < 1 ms (closure-bound) |
| `pgd_attack_l_inf` (10 steps) | D = 224 * 224 * 3 | < 10 * gradient-closure-time |
| `cw_attack` (binary-search depth 9, 100 inner steps) | D = 224 * 224 * 3 | < 900 * gradient-closure-time |
| `auto_pgd_attack` (n_steps = 100) | D = 224 * 224 * 3 | < 100 * gradient-closure-time |
| `trades_loss` (B = 64, K = 1000) | -- | < 5 ms CPU |
| `mart_loss` (B = 64, K = 1000) | -- | < 5 ms CPU |
| `certified_radius` (RS, N = 100K samples) | -- | < 100K * gradient-closure-time |
| `ibp_propagate` (64 -> 32 affine layer) | -- | < 100 us CPU |
| `lipschitz_certified_radius` | per call | < 1 us |

Targets are CPU-reference budgets dominated by user-supplied gradient
closures. Once GPU wiring lands, `fgsm_step_ptx` + `pgd_proj_*_ptx` fused
inside the PGD inner loop should approach memory-bandwidth bound for the
elementwise projection step.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/adv_ops.rs`) -- 7 PTX kernel groups x 4 SM versions + 4 algorithm benches

## Notes

- All PTX kernels emit `.target sm_<version>` and use a grid-stride loop pattern.
- `fgsm_attack` / `pgd_attack_*` consume a user-supplied gradient closure (`Fn(&[f32]) -> AdvResult<Vec<f32>>`) -- the library does not attempt automatic differentiation. Wire your own backward pass.
- `pgd_attack_l2` uses the *host-computed* L2 norm to scale the perturbation (the projection PTX expects a precomputed norm constant); call `l2_norm(&delta)` before each projection.
- `EpsilonBudget::spend` returns `AdvError::BudgetExceeded` rather than panicking -- prefer `?` propagation in client code.
- `RsConfig::new(sigma, n_samples, alpha)` validates `sigma > 0`, `n_samples >= 1`, and `0 < alpha < 1`; downstream code can rely on the absence of NaN noise contributions.
- `IntervalBound::new(lo, hi)` enforces `lo <= hi` and returns `AdvError::Internal` if violated; `relu()` clamps `lo` to zero and propagates `hi`.
- The PTX kernels target scalar f32 paths; no Tensor-Core (wgmma / mma.sync) usage -- IBP / TRADES / MART GEMMs delegate to `oxicuda-blas`.

---

## Architecture-Specific Deepening

### PTX Generation by SM Version

| SM Version | PTX Version | Notes |
|------------|-------------|-------|
| sm_75 (Turing) | 7.5 | Baseline; `selp.f32`, `fma.rn.f32`, `div.rn.f32` supported |
| sm_80 / sm_86 (Ampere) | 8.0 | Default target for `AdvHandle::default_handle()` |
| sm_89 (Ada) | 8.0 | Treated as sm_80 by `ptx_version_str()` |
| sm_90 / sm_90a (Hopper) | 8.4 | No `wgmma` usage -- attack kernels are elementwise / reduction |
| sm_100 / sm_120 (Blackwell) | 8.7 | Same scalar pattern |

The 7 generators all dispatch on the SM string and emit identical scalar PTX
modulo the `.target` directive. Attack and projection kernels are
elementwise; randomized-smoothing is dominated by Box-Muller noise; the
heavy GEMM work (model forward / backward) is left to `oxicuda-blas`.

### Deepening Opportunities

- [ ] Hopper `certified_radius_reduce_ptx` rewrite using `redux.sync.max.u32` for warp-local argmax before the global atomic
- [ ] Blackwell (sm_100+) cluster launch for `smoothing_noise_ptx` to generate larger noise batches per kernel for randomized smoothing
- [ ] FP16 / BF16 variants of `fgsm_step_ptx` and `pgd_proj_l_inf_ptx` for low-precision robustness studies
- [ ] Tensor-Core path for `ibp_propagate` affine layer (currently delegates to `oxicuda-blas` GEMM, but a fused interval-aware GEMM would cut intermediate-bound rounding loss)

---

## Functional Quality Gates (Vol.27)

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| A1 | FGSM single-step attack with clamp | P0 | [x] |
| A2 | PGD L_inf with optional random restart | P0 | [x] |
| A3 | PGD L2 with norm-aware projection | P0 | [x] |
| A4 | MIM with momentum accumulation | P0 | [x] |
| A5 | CW L2 attack with binary search + tanh reparam | P1 | [x] |
| A6 | AutoPGD with step-size schedule + checkpointing | P1 | [x] |
| A7 | TRADES regulariser (CE + beta * KL) | P0 | [x] |
| A8 | MART misclassification-aware loss | P0 | [x] |
| A9 | Randomized Smoothing Monte-Carlo predict + radius | P0 | [x] |
| A10 | IBP propagation through affine layer with `IntervalBound::relu` | P0 | [x] |
| A11 | Lipschitz certified radius formula | P1 | [x] |
| A12 | Lp-ball norms + projections (L1 / L2 / L_inf) | P0 | [x] |
| A13 | Epsilon-budget tracker with overdraft error | P0 | [x] |
| A14 | Robust + certified accuracy + attack success rate metrics | P0 | [x] |
| A15 | PTX generators for 7 kernels x 6 SM versions | P0 | [x] |

## Performance Verification Harness Status

- All performance numbers above are CPU-side targets achievable on the build host.
- GPU end-to-end harnesses await the [ ] GPU launcher wiring item plus a
  Linux+NVIDIA test runner; the PTX strings themselves are covered by
  string-content unit tests inside `ptx_kernels.rs`.
