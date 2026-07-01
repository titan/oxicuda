# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-01

This release is an on-device validation pass: for the first time, hand-written PTX kernels across more than 60 crates were JIT-compiled and executed on real NVIDIA GPU hardware (an RTX A4000, sm_86, CUDA 12.4) rather than only checked for CPU-logic parity. This surfaced and fixed dozens of genuine bugs that no amount of CPU-side testing could have caught — kernels that never compiled (`ptxas` rejected them outright), kernels that compiled but computed the wrong thing (register shadowing, base-2/base-e mixups, races), and kernels that were still bare stubs behind a real, tested CPU reference implementation. Every fix was verified fail→revert→pass on the actual device. Alongside the validation sweep: several algorithms went from partial/proxy PTX implementations to full ones (P1 FEM assembly, NODE tree inference, soft-MoE dispatch, 3D Gaussian splatting projection/SH), a few new modules landed (variable-depth NODE trees, TabR-style retrieval, preconditioned CG, GPT-NeoX RoPE), and analytic test coverage was added for dozens of previously-untested modules.

### Added

- On-device GPU validation harness: a feature-gated `gpu-tests` Cargo feature plus a `src/gpu_tests.rs` module per crate, JIT-compiling each crate's hand-written PTX via `Module::from_ptx`, launching it on a live CUDA device, and asserting numerical equivalence to a CPU oracle. Rolled out workspace-wide (more than 60 crates); every test skips gracefully when no device is present. See Fixed below for what it caught.
- Crates that ran clean on the very first on-device pass (JIT-loaded and matched their CPU oracle with zero bugs found): `oxicuda-pinn` (7 kernels), `oxicuda-bayes` (7), `oxicuda-federated` (7), `oxicuda-continual` (7), `oxicuda-peft` (7), `oxicuda-meta` (7), `oxicuda-tn` (7), `oxicuda-sketch` (7), `oxicuda-graphalg` (7), `oxicuda-cvx` (7), `oxicuda-gen` (6), `oxicuda-adversarial` (7), `oxicuda-hdc` (7) — 42+ kernels across 13 crates with no defects on first real-hardware execution. `oxicuda-pde` (7) and `oxicuda-numeric` (7) likewise matched cleanly but each surfaced one honestly-documented (not fixed) caveat: `pde`'s `fem_assemble_kernel` (pre-completion, see below) was confirmed to do only a signed-triangle-area scatter; `numeric`'s `bessel_recurrence` aliases the next point's `J_0` when multiple points share a launch (documented, validation scoped to the single-point calling convention).
- `oxicuda-pde`: `fem_assemble_kernel` completed from a partial stub (per-element signed-area scatter into one matrix entry) to a full unconstrained dense P1 stiffness assembly — the 3×3 local `K_ij = (1/(4·Area))·(b_i·b_j + c_i·c_j)` per element, atomically scattered into the dense global matrix. Validated element-wise against the crate's own `p1_local_stiffness` to 1e-4 rel / 1e-5 abs.
- `oxicuda-tabular`: `sparsemax_kernel` replaced a dead threshold pass with the exact O(D²) Martins & Astudillo largest-support search; `quantile_norm_kernel` replaced a 2-bucket heuristic with true empirical-CDF linear-scan + interpolation; `node_tree_eval_kernel` (previously hardcoded to 2 leaves, ignoring `depth`) now runs the full multi-level NODE tree with per-level entmax-1.5 bisection and a `2^depth`-leaf mixture. Validated against `sparsemax`/`QuantileTransformer::transform`/`NodeTree::forward` at 1e-4–1e-5. Also new: `VarObliviousLayer` (variable-depth NODE oblivious trees via outer-product leaf gating) and `TabRecordLayer` (TabR-style retrieval: encode → scaled −L2 similarity → entmax attention → convex combination), reusing the crate's entmax/entmoid/sparsemax simplex code (19 tests).
- `oxicuda-moe`: `soft_moe_dispatch_kernel` completed from a first-slot-only proxy to the real 3-pass slot-softmax dispatch matrix `D[t,s] = softmax(x·Φ/√d)` over all slots. Validated against `SoftMoeRouter::dispatch_weights` to 5e-4, every output row confirmed to sum to 1.
- `oxicuda-geometry3d`: `project_kernel` now emits the full EWA 2D covariance `Σ_2d = J·R·Σ_3d·Rᵀ·Jᵀ + 0.3·I` (previously never written); `sh_eval_kernel` now evaluates all 9 L=0..2 spherical-harmonic terms per RGB channel (previously a reduced 5-term basis). Validated against `project_gaussian`/`Gaussian3d::sh_color` on the A4000.
- `oxicuda-recsys`: implemented 4 previously-empty-loop stub kernels — `embedding_lookup`, `dot_score`, `bpr_gradient`, `lightgcn_propagate` — now real, validated bit-exact / to ~1e-4 against `Bpr`/`LightGcn` CPU references. The remaining PTX surface (`softmax_topk`, `negsample_uniform`) is documented as still-stub with loud STUB/PTX-BUG doc comments designed to fail the day each is implemented for real.
- `oxicuda-webgpu`: `naga_tests.rs` — real WGSL parse+validate (`naga::front::wgsl::parse_str` + `valid::Validator`) across all 15 shader generators (31 tests), replacing prior substring-only shader checks.
- `oxicuda-pinn`: hand-written pure-Rust FFT (iterative radix-2 Cooley–Tukey + Bluestein chirp-z for arbitrary/prime N) replacing the FNO spectral path's O(N²) brute-force DFT; wired into 1D + separable-2D `spectral_conv`, zero rustfft/oxicuda-fft dependency (`fno_3d`'s DFT is a noted follow-up).
- `oxicuda-nas`: `LatencyLut::to_bytes`/`from_bytes` — a dependency-free little-endian persistence format (magic `LLUT` + version + stable `OpKind` discriminants), 7 tests including round-trip identity across all 8 `OpKind` variants.
- `oxicuda-cvx`: preconditioned conjugate gradient — `pcg_solve`/`pcg_solve_counted`/`cg_solve_counted` plus a `Preconditioner` trait with `IdentityPrecond`/`JacobiPrecond` (Jacobi cuts a κ=1e4 diagonal system from 6 CG iterations to 1).
- `oxicuda-blas`, `oxicuda-ptx`: new PTX kernel templates for broadcast bias-add and a numerically-stable causal (masked) softmax, F32/F64.
- `oxicuda-dnn`: GPT-NeoX half-split partial-rotary RoPE (`NeoXRopeConfig`, `apply_rope_neox_half_split`) alongside the existing GPT-J/RoFormer interleaved `Rope`, plus a RoPE-NeoX attention integration.
- `oxicuda-ptx`, `oxicuda-metal`, `oxicuda-memory`: new f64 math-intrinsic codegen module (`body_builder/math_f64.rs`), Metal backend function/type additions, and `device_buffer` helpers.
- Large-scale analytic test-coverage expansion for previously zero-coverage modules (property-based/closed-form assertions, not smoke tests): `oxicuda-rlhf` (kl_control, alignment metrics, PPO GAE rollout, SFT/reward/preference losses — 54 tests), `oxicuda-recsys` (popularity_neg + a 116-test suite across 16 models: BERT4Rec, SASRec, GRU4Rec, PLE, MMoE, ESMM, DeepFM, AutoInt, Wide&Deep, ALS, NMF, NGCF, LightGCN, NCF, Two-Tower, hard-neg sampling), `oxicuda-peft` (merge/p-tuning-v2 + a 67-test suite across 9 adapter variants), `oxicuda-meta` (few-shot/linear-head + a 69-test suite across the MAML family), `oxicuda-numeric` (8 Gauss-Patterson exactness tests to degree 46), `oxicuda-evol` (8 CMA-ES Jacobi-eigensolver spectral-identity tests), `oxicuda-solver` (16 PDE/ODE tests with measured O(h²) convergence), `oxicuda-ann` (43 tests across HNSW/kNN-graph/IVF/IVFPQ).
- Test suite expanded to 38,093 passing tests (workspace-wide, `--all-features`; 37,166 with default features), up from 36,984 at 0.3.0.

### Changed

- `oxicuda-solver`: `syevd` (symmetric eigensolver) and the blocked Householder QR / one-sided-Jacobi SVD device paths previously launched an incomplete GPU kernel and read back fabricated (never-computed) values. Replaced with an explicit, documented exact-CPU host fallback — no GPU acceleration yet, pending on-device follow-up — rather than silently returning wrong results.
- `oxicuda-ssl`: `barlow_cross_corr_wgmma`, `nt_xent_softmax_warp`, and `gather_features_bulk` are now documented as intentionally Hopper/Blackwell-only PTX (`wgmma`, `redux.sync`, TMA); each has an on-device-confirmed portable scalar fallback for Ampere and older.

### Fixed

- Register-shadowing of CUDA's built-in special registers (`.reg` declarations literally named `%tid`/`%ntid`/`%ctaid`/`%warpid`, clobbering the special registers like `%tid.x` actually read from) — the single most common defect class this pass found, affecting `oxicuda-primitives`, `oxicuda-train` (all 9 optimizer kernels), `oxicuda-ann` (`hnsw_neighbor_eval`/`ivf_assign`/`topk_select`), `oxicuda-rl` (all 5 kernels), `oxicuda-dist-infer` (all 5 kernels), and `oxicuda-timeseries` (all 7 kernels, plus a special register used directly as a `mad` operand). All renamed to non-colliding register names.
- Base-2 (`ex2.approx`/`lg2.approx`) used where the math needs base-e — silently plausible, genuinely wrong: `oxicuda-survival` (Cox risk/score/info, ~18–30% off), `oxicuda-seq` (HMM forward log-sum-exp, ~30% off), `oxicuda-ot` (Sinkhorn/unbalanced log-sum-exp, ~20% off), `oxicuda-rlhf` (BT/DPO/KTO losses), `oxicuda-nerf` (`volume_render`'s alpha compositing), `oxicuda-gnn` (`softmax_edge`, masked by softmax's own scale-invariance — only a base-e CPU oracle caught it), and `oxicuda-audio` (`ctc_alpha_kernel`, bundled with two other defects below). All fixed with the correct `log2(e)`/`ln(2)` scaling.
- Kernels that never compiled at all — invalid PTX rejected outright by `ptxas`: undeclared/out-of-range registers (`oxicuda-multimodal`'s `bilinear_pool`/`temporal_pool`, `oxicuda-audio`'s `ctc_alpha_kernel`, `oxicuda-quant`'s all 5 kernels, `oxicuda-moe`'s `expert_dispatch_kernel`), a duplicate register declaration (`oxicuda-geom2d`'s `point_in_aabb`), mid-function `.reg` declarations (`oxicuda-distill`'s `at_pool_kernel`/`gram_matrix_kernel`), a missing `.reg .pred` entirely (`oxicuda-recsys`'s `als_update_step`), illegal scaled-register shared-memory addressing (`oxicuda-ann`'s `topk_select`, `oxicuda-infer`'s `logits_softmax`, `oxicuda-causal`'s `expm_pade_kernel` across dozens of sites — the latter also stored immediate literals directly via `st.shared.f32`, which `st` cannot take as a source operand), the `[smem]` bracket form used as arithmetic instead of load/store (`oxicuda-lm`'s `rms_norm`/`causal_attn_softmax`), a non-existent `atom.exch.s32` (`oxicuda-tda`'s `boundary_reduce`, needs `.b32`), a non-existent `cos.approx.f64`/`lg2.approx.f64` (`oxicuda-fft`'s `precompute_window`, `oxicuda-evol`'s `gaussian_mutate_kernel`), an unsupported 4-byte `cp.async.cg` transaction (`oxicuda-cs`'s `iht_step_cp_async`, needs `.ca` for sub-16-byte transactions — a separate deadlock from out-of-range threads skipping `bar.sync` was fixed in the same kernel), a malformed branch label (`oxicuda-sparse`'s `spmv_bsr`), and invalid braced predication plus nonexistent f64 SFU forms (`oxicuda-privacy`, 6 of 7 differential-privacy kernels). All now compile and are ptxas-verified on sm_86.
- Kernels that compiled but computed the wrong thing: `oxicuda-manifold`'s `knn_topk` was correct only for k=1 (missing the ascending bubble-up pass for k>1); `oxicuda-vision`'s `bilinear_interp`/`roi_align` used the non-existent `floor.f32` (replaced with `cvt.rmi.f32.f32`); `oxicuda-stats`'s `mean_var`/`rank_assign` were missing the `.rn` rounding qualifier on `cvt.f32.u32`; `oxicuda-quantum`'s statevector simulator had 8 stacked defects (partial 4×4 gate matrix-vector products, wrong bit-insertion masks, an unguarded swap race, divergent-lane `shfl.sync`, wrong Taylor-series hex-float constants) — essentially every operation was wrong; `oxicuda-anomaly`'s `lof_reach_dist_kernel` had a loop-index register clobber that collapsed every `reach_dist` to `kd_j`; `oxicuda-mamba`'s `parallel_scan` used the wrong shuffle direction (100% wrong) and `wkv_forward` used the wrong softmax pivot (42.6% error); `oxicuda-audio`'s `rel_pos_bias_kernel` had an unsigned-underflow clamp bug and `stats_pool_kernel` had a 32-lane write race; `oxicuda-nas`'s `gumbel_softmax_kernel` used `log2(e)` where `ln(2)` was needed (its reciprocal, ~2.08× error); `oxicuda-geometry3d`'s `sh_eval_kernel` referenced a register past its declared bank and dropped a `dx` factor from every channel's `c1·Y11` term; `oxicuda-moe`'s `expert_ffn_kernel` GELU tanh approximation was missing a factor of 2 in its exponent; `oxicuda-infer`'s flagship `paged_attention` had 5 stacked defects (invalid 64-bit `mul.wide.u32`, a partial dot product instead of the full Σ_d, V read through the K pointer, base-2 softmax, wrong GQA head mapping) and `rope_apply` dropped a sign term plus used an imprecise `log2(10000)` constant.
- `oxicuda-solver`: `lu.rs`'s `launch_gemm_update` swapped the grid X/Y axes against the actual row/column mapping (dropping part of the GEMM update on non-square trailing tiles), and its Padé matrix-exponential kernel always loaded f64 coefficients even in the f32 variant — both fixed. Separately, the LU (`panel_lu`/`trsm_unit_lower`/`gemm_update`/`pivot_swap`) and Cholesky (`panel_cholesky`) kernels were literal `ret;` stub bodies performing no factorization at all — implemented with real `bar.sync`-staged panel/trailing updates, validated by two new on-device test suites including an independent splitmix64-seeded adversarial harness.
- `oxicuda-rand`: AES round-key words needed `.swap_bytes()` for the device's endianness; `mrg32k3a`/`philox`/`xorwow`'s Box-Muller kernels shared one f32/f64 register pool, corrupting Gaussian sampling on all three engines; 4 `philox_optimized` branch targets were missing the `$` label sigil.
- `oxicuda-signal`: `dct2_permute`/`dct3_pretwiddle`/`dct3_unpermute`/`fir_direct` used illegal brace predication; `fir_direct`'s bounds guard was an always-false `src > u64::MAX` (silently zeroing all output); `dct3_pretwiddle`/`dct4_postscale` emitted f32 immediates into f64 instructions.
- `oxicuda-ssl`: `random_mask_kernel` had a spurious `×0.5` roughly doubling the effective drop probability.
- `oxicuda-tabular`: `feature_tokenize_kernel` addressed its weight/bias rows with stride 1 instead of `feat*embed_dim` and never looped over the embedding dimension.
- `oxicuda-rlhf`: `dpo_loss_kernel`'s grid-stride accumulation clobbered the register holding `beta` with an atomic's return value; `ppo_rlhf/ppo_step.rs`'s per-step value-loss loop (a CPU-side bug, found by the new test suite) read `values[0]` for every step instead of the current step's value.
- `oxicuda-ann`: `ivf/ivf.rs` `search` (CPU-side, found by the new test suite) indexed insertion-ordered vectors with a list-traversal counter instead of per-list storage, scoring against the wrong stored vector whenever `add()` calls interleaved across coarse lists.
- `oxicuda-recsys`: (CPU-side, found by the new test suite) `multitask/ple.rs` fed shared experts at layer>0 the wrong-length input from the previous layer; `factorization/als.rs`'s `gauss_jordan` had a redundant row-swap that skipped exchanging column 0 whenever the pivot was off-diagonal.
- `oxicuda-blas` (GEMM), `oxicuda-dnn` (LayerNorm, and separately `implicit_gemm`/`conv1x1`/`depthwise` which were comment-only stubs with no arithmetic), `oxicuda-sparse` (f64 CSR SpMV, including an illegal 64-bit `shfl.sync.down.b64` split into `.b32` halves, and the mixed-precision SpMV FP64 path): a shared invalid-PTX bug class — an `.f32`-declared register bank and/or single-precision zero literal used in `.f64` instructions. Root-caused once in `oxicuda-ptx`'s `PtxType` (precision-correct zero-literal encoding + correctly-rounded `cvt` selection) and applied across all affected sites; LayerNorm's `.maxntid` directive was also misplaced inside the kernel body with a stray semicolon.
- `oxicuda-snn`: the `atan` surrogate-gradient kernel was off by π² (`α·π/(1+x²)` instead of `α/(π·(1+x²))`), affecting every SM target sm_75–sm_100.
- `oxicuda-webgpu`: `conv2d` WGSL codegen named a buffer `filter`, a reserved WGSL keyword, failing naga parsing outright — renamed to `kernel_w`.
- `oxicuda-metal`: cleared 2 clippy warnings in `memory.rs` (missing `dead_code` cfg-gate, a redundant explicit `drop`).

## [0.3.0] - 2026-06-25

This release adds no new crates (still 73). It is a depth pass: implementing genuine, CPU-verifiable algorithms across existing crates, reviving orphaned-but-real modules, wiring cross-crate paths, and fixing latent bugs surfaced along the way. Every algorithm was added with correctness tests (finite-difference-verified gradients, analytic-front residuals, bit-exact cross-path checks).

### Added

- Cross-crate integration: `oxicuda-gnn` `GcnLayer::forward_sparse` routes message passing through `oxicuda-sparse` HostCsr SpMM (sparse path bit-exact vs the dense path); `oxicuda-timeseries` `detect_period_fft` computes Wiener–Khinchin autocorrelation via `oxicuda-fft` rfft/irfft (matches the direct O(T²) result to 3.3e-12). Both dependencies are declared `{ workspace = true }` with no dependency cycle.
- `oxicuda-geometry3d`: PointFlow continuous-normalizing-flow core (reverse-time invertibility 1.1e-16, exact-trace logdet vs finite-difference 6.2e-11). Trained generation parts deferred.
- `oxicuda-audio`: residual-vector-quantization neural-codec core (Bark RVQ — monotone reconstruction error, exact index recovery, k-means fit non-increase). Trained generation parts deferred.
- `oxicuda-rlhf` is now fully gradient-capable: 20+ analytic, central-finite-difference-verified gradients across 17 loss modules — the closed-form preference family (DPO/IPO/KTO/SimPO/ORPO/BCO/DPOP/SLiC/Step-DPO/sDPO/RRHF/length-DPO/online-DPO), reward models (Bradley-Terry, soft-BT RLAIF, PRM), and RL estimators (PPO, GRPO clip + k3-KL, REBEL, RLOO, SAC-RLHF). Previously forward-value-only.
- `oxicuda-evol`: WFG1-9, ZDT4/6, and DTLZ3-7 multi-objective test problems (analytic-front residuals at machine epsilon).
- `oxicuda-seq`: Gaussian-HMM Baum-Welch EM (monotone log-likelihood); Kalman tracking and CRF chunker examples.
- `oxicuda-audio`: rational-quadratic spline flow (Durkan 2019) as a VITS stochastic-duration dequantizer.
- `oxicuda-hdc`: measured Hopfield-capacity and bundle-SNR scaling-law curves.
- `oxicuda-snn`: NARMA-10 reservoir benchmark and STDP sign/shape verification; sparse spike encoding and event-driven LIF.
- `oxicuda-ptx`: `CpAsyncGenerator` emitting `cp.async.cg/ca.global` PTX with multi-stage commit_group/wait_group pipelining and a pre-sm_80 fallback; `FusionCostModel` register-pressure + shared-memory + ILP heuristic wired into `kernel_fusion::plan_fusion`.
- `oxicuda-tabular`: analytic backward passes (FT-Transformer/TabNet/SAINT/NODE) with softmax/sparsemax/entmax Jacobians.
- `oxicuda-backend`: mixed-precision GEMM (binary16/bfloat16 round-to-nearest-even, FP32 accumulate) and conv2d backward.
- `oxicuda-quant`: GGUF v3 container read/write.
- `oxicuda-graph`: reduction-pattern fusion pass.
- `oxicuda-gnn`: edge-feature support in GAT.
- `oxicuda-runtime`: device-pointer cast / typed-slice helpers and stream-capture bookkeeping.
- `oxicuda-privacy`: Philox and ChaCha20 counter-based RNGs, a DP-Adam convergence harness, and PATE-GAN/DP-GAN.
- `oxicuda-nas`: Bayesian-optimization GP predictor and Once-for-All.
- `oxicuda-meta`: MAML inner-loop integration.
- `oxicuda-pinn`: PI-DeepONet forward-mode AD, a tree-GP symbolic regressor, and batched ODE solvers.
- `oxicuda-gen`: full U-Net assembly and LoRA checkpoint round-trip.
- `oxicuda-geometry3d`: straight-through FPS gradients.
- `oxicuda-pde`: convergence-verified Poisson, Crank-Nicolson, and multigrid solvers.
- `oxicuda-solver`: MINRES/QMR/LSQR Krylov solvers and Gilbert-Peierls sparse LU.
- `oxicuda-blas`: 2:4 structured-sparse SpGEMM with Ampere `mma.sp` codegen.
- `oxicuda-autotune`: persistent LRU tune-cache.
- `oxicuda-cvx`: fluent LP/QP/SOCP/SDP solver builder.
- `oxicuda-tda`: persistence and Mapper examples.
- `oxicuda-sketch`: `CuckooFilter32`.
- `oxicuda-causal`: discrete conditional-independence tests (chi-square / G-test) and the PC algorithm.
- `oxicuda-peft`: AdaLoRA, TIES, and DARE.
- `oxicuda-dist-infer`: autonomous `RebalanceMonitor` and `ElasticScaler`.
- Test suite expanded to 36,984 passing tests (workspace-wide, `--all-features`; 36,546 with default features), up from 32,320 at 0.2.0.

### Changed

- Wired 11 orphaned-but-real modules across 6 crates (180 previously-dead tests revived) — `oxicuda-evol` CMA-ME, `oxicuda-manifold` Isomap / parametric-tSNE / geodesic-regression, `oxicuda-rand` cuRAND-style host API, `oxicuda-rlhf` dpo/ppo loss + reward-norm, `oxicuda-stats` GMM + ARIMA, `oxicuda-timeseries` DTW; plus 4 more orphaned modules in `oxicuda-nas` (Once-for-All, NAS-Bench) and `oxicuda-geometry3d`. These were real, tested algorithm files never declared in `mod.rs`.
- `oxicuda-ptx`: kernel fusion is now cost-gated — `FusionCostModel` replaces the former unconditional acceptance of every structurally-legal fusion candidate with a register-spill / shared-memory / benefit-threshold fuse-or-refuse decision.

### Fixed

- `oxicuda-ot`: `network_simplex` `find_cycle` had an inverted closing-parity condition that failed 100% of n≥4 dense EMD instances — the exact optimal-transport solver was silently broken for all non-trivial problem sizes. Rewrote the alternating-axis cycle DFS; now a 100% solve rate for n=5..64, agreeing with Sinkhorn to relative gap < 8e-3 as ε→0.
- `oxicuda-peft`: corrected an NF4 codebook typo — `NF4_TABLE[3]` and `nf4_dequant_ptx` held `-0.3949468731880188`; the canonical QLoRA/bitsandbytes value (and the crate's own `nf4_quant.rs`) is `-0.39491748809814453`.
- `oxicuda-rand`: fixed an MRG32k3a `[0,1)` contract violation and scrambled-Sobol Inf/NaN (an errant `÷2^31` should have been `÷2^32`).
- `oxicuda-stats`: fixed a GMM kmeans++ degenerate fallback that could panic with a reversed range or produce wrong-length centers.
- `oxicuda-solver`: fixed a sparse-LU pivoting bug.
- `oxicuda-nas`: fixed 2 latent compile bugs (missing `PartialEq` derives) in the previously-never-compiled Once-for-All module.

## [0.2.0] - 2026-06-16

### Added

- Wave AAA+64 feature expansion: Extended Persistence and Discrete Morse theory (`oxicuda-tda`), Parametric UMAP (`oxicuda-manifold`), Fisher Information estimation (`oxicuda-bayes`), and adaptive RK45 integration with Richardson extrapolation for ODE/PDE solvers.
- Expanded CUDA kernel coverage across the driver, memory, launch, and backend layers.
- Test suite grew to 32,320 passing tests (up from 23,535 at 0.1.8).

### Changed

- Workspace-wide reliability pass: eliminated every `.unwrap()` from all `crates/*/src/` (production code and test modules now use descriptive `.expect(...)`), maintaining zero clippy warnings under `-D warnings`.

### Fixed

- `oxicuda-geometry3d`: corrected a sign error in the symmetric-3×3 Jacobi eigensolver (`crates/oxicuda-geometry3d/src/mesh/obb.rs`) whose rotation angle used `app - aqq` instead of `aqq - app`. The defect doubled the off-diagonal each sweep instead of annihilating it, so `Obb::fit_pca` returned eigenvectors tilted from the true principal axes and produced a non-tight oriented bounding box.

## [0.1.8] - 2026-05-21

### Changed

- Maintenance release: numerical-stability refinements in HMC variational sampler, stream-ordered allocator tuning, and TriMap reduction polish (`crates/oxicuda-bayes/src/variational/hmc.rs`, `crates/oxicuda-driver/src/stream_ordered_alloc.rs`, `crates/oxicuda-manifold/src/reduction/trimap.rs`)

## [0.1.7] - 2026-05-16

### Added

- `oxicuda-blas`: SYR2K Tensor Core kernel with two-operand cross-product variant — efficient symmetric rank-2k update using Tensor Core hardware units with fused A×Bᵀ + B×Aᵀ accumulation (`crates/oxicuda-blas/src/level3/syr2k.rs`)
- CUDA kernel enhancements across multiple subsystems (driver, memory, launch, blas, and backend layers)
- MOS (Multi-Operation Scheduling) improvements for GPU task orchestration

## [0.1.6] - 2026-05-08

### Added

- `oxicuda-blas`: Tensor Core fast path for SYRK — triangle-masked GEMM kernel that eliminates redundant symmetric writes while hitting Tensor Core hardware units (`crates/oxicuda-blas/src/level3/syrk.rs`, `syr2k.rs`)
- Vol.26 `oxicuda-adversarial` (Adversarial robustness: attack generation, adversarial training primitives)
- Vol.27 `oxicuda-ssl` (Self-Supervised Learning: contrastive, masked-autoencoder, and distillation scaffolding)
- Vol.28 `oxicuda-continual` (Continual Learning: PackNet architecture, task-incremental training, forgetting mitigation)
- Vol.29 `oxicuda-multimodal` (Multimodal Learning: cross-modal fusion, shared-encoder scaffolding)
- Vol.30 `oxicuda-geometry3d` (3-D Geometry: point-cloud ops, mesh primitives, spatial indexing)
- Vol.31 `oxicuda-pinn` (Physics-Informed Neural Networks: PDE loss terms, residual sampling)
- Vol.32 `oxicuda-ann` (Approximate Nearest Neighbour: flat / IVF / IVFPQ / HNSW / LSH / PQ / KNN-graph, Hamming / L2 / inner-product distances, SQ4/SQ8 quantizers, k-NN heap select)
- Vol.33 `oxicuda-anomaly` (Anomaly Detection: Mahalanobis / COPOD density estimators, kNN score, LOF)
- Vol.34 `oxicuda-causal` (Causal Inference: do-calculus primitives, causal graph scaffolding)
- Vol.35 `oxicuda-meta` (Meta-Learning: MAML / Prototypical-Network scaffolding)
- Vol.36 `oxicuda-moe` (Mixture-of-Experts: top-k routing, expert dispatch, load-balancing loss)
- Vol.37 `oxicuda-nerf` (Neural Radiance Fields: ray-marching primitives, positional encoding, volume rendering)
- Vol.38 `oxicuda-quantum` (Quantum-Classical Hybrid: qubit-state simulation primitives, variational circuit scaffolding)
- Vol.39 `oxicuda-recsys` (Recommender Systems: collaborative filtering, embedding lookup, ranking loss)
- Vol.40 `oxicuda-rlhf` (RLHF: reward-model scaffolding, PPO/DPO wrappers, KL-penalty helpers)
- Vol.41 `oxicuda-tabular` (Tabular ML: feature encoding, gradient-boosted tree scaffolding, TabNet blocks)

## [0.1.5] - 2026-05-03

### Added

- macOS stub integration test suite (`crates/oxicuda-driver/tests/macos_stub.rs`) — 9 tests asserting every `gpu-tests`-gated entrypoint returns `Err(UnsupportedPlatform)` or `Err(NotInitialized)` on macOS
- `[package.metadata.docs.rs]` configuration added to all 34 subcrate `Cargo.toml` files; `cargo doc --all-features` now builds cleanly workspace-wide
- Vol.17 `oxicuda-gen` (Generative AI: DDPM/DDIM/DPM-Solver++/Flow Matching schedulers, classifier-free guidance, VAE codec, LoRA adapters, score-network blocks)
- Vol.18 `oxicuda-gnn` (Graph Neural Networks: CSR/COO/Heterogeneous graphs, scatter / gather / aggregate primitives, GCN / GAT / GAT-v2 / GraphSAGE / GIN layers, global / Top-K / DiffPool pooling, Set2Set readout)
- Vol.19 `oxicuda-mamba` (State Space Models: HiPPO-NPLR initialization, S4D / S5 selective scan, Mamba SSM block, RWKV channel-mixing, gated SSM)
- Vol.20 `oxicuda-vision` (Vision Transformers & CLIP: patch embedding, ViT encoder blocks, learnable positional embeddings, CLS token, CLIP-style image / text tower scaffolding)
- Vol.21 `oxicuda-audio` (Audio / Speech ML: Conformer encoder, Wav2Vec2 feature extractor, CTC / RNN-T loss, WaveNet causal stack, SpecAugment, x-vector speaker embedding)
- Vol.22 `oxicuda-timeseries` (Time-Series Forecasting: TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN reversible normalization)
- Vol.23 `oxicuda-bayes` (Bayesian deep learning: variational inference, Bayesian linear / conv layers, Flipout, ELBO / IWAE, normalizing flows, MC Dropout, Deep Ensembles, SWAG, Laplace approximation, calibration / ECE)
- Vol.24 `oxicuda-federated` (Federated learning: FedAvg / FedProx / SCAFFOLD / FedAdam, PowerSGD / QSGD / Top-K / Random-K compression, Gaussian / Laplacian / Moments / RDP / PATE differential privacy, Shamir-based secure aggregation, random / stratified client selection)
- Vol.25 `oxicuda-nas` (Neural Architecture Search: DARTS bilevel optimizer with derived discrete cells, one-shot weight-shared Supernet with path sampling and Slimmable widths, evolutionary NSGA-II with non-dominated sort and crowding distance, hardware-aware FLOPs predictor)
- All three new leaf crates carry `[dependencies] thiserror.workspace = true` only — no internal `oxicuda-*` dependencies, fully standalone, 100% Pure Rust
- 8 missing per-crate `README.md` files created (`oxicuda-bayes`, `oxicuda-federated`, `oxicuda-gen`, `oxicuda-gnn`, `oxicuda-mamba`, `oxicuda-nas`, `oxicuda-timeseries`, `oxicuda-vision`) so `cargo publish` no longer errors on `readme = "README.md"`

### Changed

- Preemptive `splitrs` of 5 near-cap source files: `batched.rs` (1950→1288 LoC), `tensor_backend/ops.rs` (1986→1673 LoC), `fp4_fp6_ops.rs` (1955→1587 LoC), `ir/instruction.rs` (1973→1244 LoC), `tui_explorer.rs` (1931→1438 LoC); test blocks extracted to sibling `*/tests.rs` files
- `device_attrs.rs` integration test tightened to assert error variant (not just `is_err()`)
- `launch-overhead-driver-crate` TODO entry collapsed to canonical cross-reference
- All internal dependency versions bumped to 0.1.5
- Workspace test count: **9,568 passing**, 2 skipped (GPU-gated on macOS) — up from prior ~9,000-something
- Repaired 22 clippy warnings without introducing any `#[allow]` attributes — `needless_range_loop` ×15, `useless_vec` ×3, `manual_repeat_n` ×3, `ptr_arg` ×1, `nonminimal_bool` ×1
- Fixed 6 pre-existing compile errors — `unused-named-args` in `format!` PTX templates ×4, deprecated `std::f32::LN_2` reference, two `explicit-deref-pattern` lints
- Statistical test `compression::randomk::tests::random_sparsify_unbiased` retuned from `n_trials=500` (1.1σ) to `n_trials=5_000` (3.6σ) — eliminates the historical flake

## [0.1.4] - 2026-04-18

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.4

## [0.1.3] - 2026-04-17

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.3

## [0.1.2] - 2026-04-14

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.2

## [0.1.1] - 2026-04-14

### Added

- `oxicuda-blas`: New elementwise operations — `Ceil`, `Floor`, `HardSigmoid`, `HardSwish`, `Softplus`, and `LeakyRelu`

### Changed

- General enhancements across crates: improved robustness, performance, and internal code quality

## [0.1.0] - 2026-04-13

### Added

**Foundation (Vol.1 — 4 crates, 22,972 SLoC)**
- `oxicuda-driver` (11,548 SLoC, 333 tests): CUDA Driver API wrapper with dynamic loading via libloading, device/context/stream/event/module management, multi-GPU context pool, occupancy queries
- `oxicuda-memory` (4,178 SLoC, 204 tests): Type-safe GPU memory management — DeviceBuffer<T>, PinnedBuffer<T>, unified memory, async pool, virtual memory, 2D/3D copies, peer transfer
- `oxicuda-launch` (4,728 SLoC, 207 tests): Type-safe kernel launch — Dim3, LaunchParams, launch! macro, cooperative launch, cluster launch (Hopper+), graph-based launch
- `oxicuda-runtime` (2,518 SLoC, 46 tests): High-level CUDA runtime wrapper — streams, events, texture objects, surface objects

**PTX Codegen & Autotuner (Vol.2 — 2 crates, 43,122 SLoC)**
- `oxicuda-ptx` (29,206 SLoC, 873 tests): Full PTX IR type system, Rust DSL for SM 7.5–SM 10.0, Tensor Core support (WMMA/MMA/WGMMA), kernel templates (GEMM, elementwise, reduction, softmax, scan, transpose, attention, BN, MoE, convolution), register pressure analysis, dead code elimination, constant folding, strength reduction
- `oxicuda-autotune` (13,916 SLoC, 408 tests): Search space definition, GPU benchmarking with statistical analysis, Bayesian optimization, simulated annealing, genetic algorithm, result DB (JSON), problem size interpolation, early stopping

**Linear Algebra (Vol.3 — 1 crate, 21,845 SLoC)**
- `oxicuda-blas` (21,845 SLoC, 604 tests): Full cuBLAS equivalent — BLAS Level 1/2/3, GEMM (SIMT/Tensor Core/Split-K), batched GEMM (standard/strided/grouped), precision coverage (F16/BF16/TF32/F32/F64/FP8), elementwise ops, reductions, epilogue fusion

**Deep Learning (Vol.4 — 1 crate, 34,711 SLoC)**
- `oxicuda-dnn` (34,711 SLoC, 960 tests): Full cuDNN equivalent — convolution (implicit GEMM/im2col/Winograd/direct/fused), FlashAttention v2 (forward/backward), PagedAttention, MoE (top-k routing, permutation, fusion), normalization (BN/LN/RMSNorm/GroupNorm), pooling, resize, speculative decoding, linear layers

**Scientific Computing (Vol.5 — 4 crates, 47,946 SLoC)**
- `oxicuda-fft` (9,749 SLoC, 295 tests): Stockham FFT, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, pruned FFT, 1D/2D/3D
- `oxicuda-sparse` (12,278 SLoC, 320 tests): CSR/CSC/COO/BSR/ELL/HYB/CSR5 formats, SpMV/SpMM/SpGEMM/SDDMM, ILU(0)/IC(0), Krylov solvers, auto-dispatch
- `oxicuda-solver` (15,804 SLoC, 373 tests): Dense LU/QR/SVD/Cholesky/eigendecomp, CG/BiCGSTAB/GMRES, tensor decomposition, matrix functions (exp/log/sqrt)
- `oxicuda-rand` (10,115 SLoC, 264 tests): Philox/MRG32k3a/XORWOW/Sobol PRNGs, uniform/normal/Poisson/exponential/gamma distributions, NIST statistical tests

**Signal Processing (Vol.6 — 1 crate, 6,037 SLoC)**
- `oxicuda-signal` (6,037 SLoC, 231 tests): Audio (MFCC, STFT, Mel filterbank), image processing (Gaussian blur, Sobel, morphology), DCT (types I–IV), DWT (Haar, Daubechies), IIR/FIR filtering, correlation

**Computation Graph (Vol.7 — 1 crate, 4,784 SLoC)**
- `oxicuda-graph` (4,784 SLoC, 175 tests): CUDA Graph capture, execution plan with dependency sorting, event synchronization, sequential/parallel executors

**GPU Training (Vol.8 — 2 crates, 10,244 SLoC)**
- `oxicuda-train` (5,927 SLoC, 165 tests): Mixed precision AMP (FP16/BF16 + loss scaling), gradient accumulation/clipping, EMA, LR schedulers (cosine/warmup/cyclic/polynomial), GPU-fused optimizers (Adam/AdamW/SGD/RMSProp/LAMB), checkpointing
- `oxicuda-quant` (4,317 SLoC, 150 tests): INT8/INT4/FP8 weight quantization, block-scaled FP4, GPTQ-style post-training quantization

**Inference Engine (Vol.9 — 3 crates, 11,929 SLoC)**
- `oxicuda-infer` (4,256 SLoC, 137 tests): PagedKvCache, prefix caching, speculative decoding, continuous batching
- `oxicuda-dist-infer` (3,279 SLoC, 80 tests): Distributed inference with tensor/pipeline parallelism, all-reduce primitives
- `oxicuda-lm` (4,394 SLoC, 182 tests): BPE tokenizer, vocabulary management, sampling strategies (greedy/top-k/top-p/beam)

**Reinforcement Learning (Vol.10 — 1 crate, 4,234 SLoC)**
- `oxicuda-rl` (4,234 SLoC, 164 tests): Replay buffers (Uniform/PER/N-step), policy distributions (Categorical/Gaussian/Deterministic), advantage estimators (GAE/TD-λ/V-trace/Retrace-λ), loss functions (PPO/DQN/SAC/TD3), observation/reward normalization, Env/VecEnv abstractions

**Backends & Primitives (7 crates, 11,234 SLoC)**
- `oxicuda-backend` (271 SLoC, 7 tests): ComputeBackend trait definition
- `oxicuda-primitives` (4,372 SLoC, 142 tests): CUB-equivalent parallel primitives (block reduce/scan/sort, warp ops)
- `oxicuda-metal` (1,186 SLoC, 52 tests): Apple Metal GPU backend (macOS/iOS)
- `oxicuda-vulkan` (1,445 SLoC, 38 tests): Vulkan Compute backend (cross-platform)
- `oxicuda-webgpu` (1,108 SLoC, 42 tests): WebGPU backend (browser/WASM)
- `oxicuda-rocm` (1,087 SLoC, 36 tests): AMD ROCm/HIP backend
- `oxicuda-levelzero` (1,765 SLoC, 44 tests): Intel oneAPI Level Zero backend

**Umbrella (1 crate)**
- `oxicuda` (19,614 SLoC, 494 tests): Re-exports all sub-crates, ComputeBackend trait with CudaBackend, OxiONNX GPU inference backend, ToRSh tensor backend, TrustformeRS transformer backend, global init/device pool
