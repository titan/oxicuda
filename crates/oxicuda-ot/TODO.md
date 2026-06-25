# oxicuda-ot TODO

Pure Rust Optimal Transport primitives covering entropic, exact, Wasserstein, Gromov-Wasserstein, unbalanced, barycentric, JKO, Schrödinger-Bridge, multi-marginal, clustering, and domain-adaptation OT, with PTX kernel templates for SM 7.5 through SM 10.0. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.44).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 24,969 SLoC (74 files)**

Current implementation covers the canonical OT algorithm spectrum: entropic OT (Sinkhorn-Knopp), exact OT (network simplex / EMD-1D), Wasserstein-1/2 and Sliced / Max-Sliced approximations, Gromov-Wasserstein and Fused-GW for unaligned domains, KL-relaxed Unbalanced OT, Wasserstein barycenters (free and fixed support), JKO gradient flow, Schrödinger Bridge (IPF), multi-marginal OT, Wasserstein k-means, and OT-based domain adaptation, plus diagnostic metrics.

### Completed [x]

#### Core Infrastructure
- [x] `lib.rs` — Crate root, module declarations
- [x] `error.rs` — `OtError` enum + `OtResult<T>` alias
- [x] `handle.rs` — `SmVersion`, `LcgRng`, `OtHandle`
- [x] `ptx_kernels.rs` — 7 GPU kernels × 6 SM versions (75 / 80 / 86 / 89 / 90 / 100)
- [x] `e2e_tests.rs` — 19 cross-module integration tests

#### Entropic OT (sinkhorn/)
- [x] `sinkhorn/sinkhorn.rs` — `SinkhornConfig {eps, max_iter, tol}`, `SinkhornResult {plan, u, v, cost, iters}`; log-domain stabilised iterative Bregman projection with row-LSE / col-LSE updates and column-residual convergence
- [x] `sinkhorn/divergence.rs` — `sinkhorn_divergence` returns `OT_ε(a,b) − ½(OT_ε(a,a) + OT_ε(b,b))` (Feydy 2019)
- [x] `sinkhorn/log_sinkhorn.rs` — `log_sinkhorn_step_row`, `log_sinkhorn_step_col`, `log_to_plan` low-level half-iteration primitives

#### Exact OT (exact/)
- [x] `exact/network_simplex.rs` — `NsConfig {max_iter}`, Northwest-corner basis, dual potentials over spanning tree, Bland's-rule pivoting, DFS cycle detection, mass-shift along negative legs
- [x] `exact/emd.rs` — `emd_1d` via sorted breakpoint sweep computing `∫|F_a − F_b| dt`, `emd` generic dispatch to network-simplex

#### Wasserstein Distances (wasserstein/)
- [x] `wasserstein/w1.rs` — `w1_1d` (delegates to `emd_1d`), `w1` multi-dim with L₂ cost via simplex
- [x] `wasserstein/w2.rs` — `w2_1d` quantile sweep, `w2` multi-dim with `½‖·‖²` cost (returns `√(2·cost)`)
- [x] `wasserstein/sliced.rs` — `SlicedConfig {n_proj, p, seed}`, Box-Muller unit directions, equal-weight 1D `W_p^p` averaging
- [x] `wasserstein/max_sliced.rs` — `MaxSlicedConfig`, argmax-init + finite-difference gradient ascent with re-projection to unit sphere

#### Gromov-Wasserstein (gromov/)
- [x] `gromov/gromov_wasserstein.rs` — `GwConfig {eps, max_iter, inner_max_iter, tol}`, outer loop on gradient `G = −2·C₁·T·C₂^T` + inner Sinkhorn, Frobenius-norm convergence test
- [x] `gromov/fused.rs` — `FgwConfig {alpha, gw}`, cost `M = (1−α)·C_xy + α·∇_GW(T)` for cross-domain Wasserstein with intra-domain structural matching

#### Unbalanced OT (unbalanced/)
- [x] `unbalanced/unbalanced_ot.rs` — `UnbalancedConfig {eps, tau_a, tau_b, max_iter, tol}`, generalised log-domain Sinkhorn with `f_i ← (τ_a/(τ_a+ε))·(ε log a_i − ε·LSE)` (KL-relaxed marginals)

#### Wasserstein Barycenters (barycenter/)
- [x] `barycenter/free_support.rs` — `BaryConfig`, alternating Sinkhorn + barycentric support update from λ-weighted-mean initialisation
- [x] `barycenter/fixed_support.rs` — `FixedBaryConfig`, Cuturi-Doucet update `b ← Π_k (K_k^T · (a_k / (K_k · b)))^{λ_k}`

#### JKO Proximal Scheme (jko/)
- [x] `jko/jko.rs` — `JkoConfig {tau, eps, n_inner, tol}`, heat-equation prox step + closure-driven external-potential variant

#### Schrödinger Bridge (bridge/)
- [x] `bridge/schrodinger.rs` — `SchrodingerConfig`, log-domain IPF on `K = exp(−C/ε)` with marginal-violation convergence

#### Multi-Marginal OT (multi/)
- [x] `multi/multi_marginal.rs` — `MmConfig`, log-domain tensor scaling with alternating-axes update; `LSE_other(x)` excludes own potential; reduces to standard Sinkhorn for k=2

#### Wasserstein k-Means (clustering/)
- [x] `clustering/wasserstein_kmeans.rs` — `WkmConfig`, W2-distance assignment + free-support barycenter centroid refinement

#### Domain Adaptation (domain/)
- [x] `domain/mapping.rs` — `barycentric_map` (row-normalised plan applied to target supports), `ot_adapt` (Sinkhorn + barycentric map)

#### Diagnostics (metrics/)
- [x] `metrics/metrics.rs` — `marginal_violation`, `kl_divergence`, `js_divergence`, `transport_cost`, `entropy`

#### GPU PTX Kernels
- [x] `sinkhorn_step_ptx` — One Sinkhorn half-iteration (row or column LSE)
- [x] `cost_matrix_ptx` — Pairwise cost evaluation
- [x] `transport_apply_ptx` — Barycentric-map application
- [x] `sliced_proj_ptx` — Sliced-Wasserstein random projection
- [x] `gromov_grad_ptx` — GW gradient `−2·C₁·T·C₂^T`
- [x] `unbalanced_step_ptx` — Unbalanced-OT log-domain step
- [x] `barycenter_update_ptx` — Fixed-support barycenter potential update

### Future Enhancements [ ]

#### P0 — Verification on GPU Hardware
- [ ] End-to-end GPU verification of all PTX kernels under Linux + NVIDIA driver 525+ (requires GPU hardware)
- [ ] Criterion benchmark suite executed on real hardware (requires GPU hardware)
- [x] Numerical-stability harness for `eps → 0` regimes — epsilon-scaling (deterministic-annealing) Sinkhorn with warm-started dual potentials + `stability_sweep` diagnostic over decreasing ε (Schmitzer 2019 §3.2, Kosowsky-Yuille 1994) (`sinkhorn/epsilon_scaling.rs`)

#### P1 — Algorithm Coverage
- [x] Greenkhorn algorithm (greedy row/column update for faster sparse Sinkhorn) (`sinkhorn/greenkhorn.rs`)
- [x] Screened Sinkhorn (Alaya et al. 2019) for active-set acceleration (`sinkhorn/screened.rs`)
- [x] Conditional gradient Wasserstein (Frank-Wolfe for non-entropic OT) (`sinkhorn/cg_wasserstein.rs`)
- [x] Sinkhorn-EMA (averaged dual potentials for stabilised gradients) (`sinkhorn/ema_sinkhorn.rs`)
- [x] Low-rank Sinkhorn factorisations (Scetbon-Cuturi 2020) (`sinkhorn/low_rank.rs`)
- [x] Sliced-Wasserstein gradient flow (SWGF) for generative modelling (`wasserstein/sw_gradient_flow.rs`)
- [x] Entropic GW with linear-memory subroutine for very large n / m (`gromov/entropic_gw_fast.rs`)
- [x] Anchor-based partial OT (Chapel et al. 2020) (`sinkhorn/anchor_partial.rs`)
- [x] Knothe-Rosenblatt rearrangement transport (`wasserstein/knothe_rosenblatt.rs`)
- [x] OT-based mini-batch loss for deep generative models (Sinkhorn-GAN, WGAN-OT) (`wasserstein/minibatch_ot.rs`)
- [x] `wasserstein/neural_ot.rs` — Neural OT map (Makkuva 2020, Korotin 2021): input-convex neural network (ICNN) parameterisation of W2 Kantorovich potentials f,g; ∇f = optimal transport map T*; alternating min-max training
- [x] `bridge/flow_matching.rs` — Conditional Flow Matching (Lipman 2022, Liu 2023): simulation-free generative model; velocity field trained to CFM target u_t|x₀,x₁=x₁-x₀; continuous normalising flow T from x₀ to x₁
- [x] `domain/dro_wasserstein.rs` — Distributionally Robust Optimisation (Esfahani-Kuhn 2018): uncertainty set = Wasserstein ball B_ε(P̂); dual reformulation as regularised empirical risk + Lagrangian constraint; DRO-ERM-ε solver
- [x] `sinkhorn/stabilised_sinkhorn.rs` — Numerically stabilised Sinkhorn (Schmitzer 2019): log-domain LSE formulation with absorption of potentials into kernel; avoids NaN at small ε; O(n²) per iteration, identical convergence
- [x] `gromov/bregman_gw.rs` — Bregman-projected GW (Xu 2019): mirror descent on coupling Γ under entropic GW objective; Bregman proj. onto transport polytope; convergence guarantee for λ-strongly convex regulariser

#### P2 — Optimisations and Tooling
- [ ] Fused cost-matrix + Sinkhorn-step kernel (saves global-memory round trip) (requires GPU hardware)
- [ ] Mixed-precision (FP16 / BF16) Sinkhorn with FP32 LSE accumulator (requires GPU hardware)
- [ ] Block-LSE tile scheme for shared-memory cost matrices (requires GPU hardware)
- [ ] CUDA-graph capture for multi-iteration Sinkhorn outer loop (requires GPU hardware)
- [ ] Tensor-Core (mma.sync) path for cost matrix evaluation (requires GPU hardware)
- [ ] On-device random direction generation for SlicedW (requires GPU hardware)
- [x] `wasserstein/w2_interpolation.rs` — Displacement interpolation (McCann 1997): geodesic (1-t)ρ₀ + t ρ₁ in Wasserstein space via McCann interpolant; (push-forward of ρ₀ under (1-t)Id + t T*); barycentric projection formula
- [x] `domain/entropic_da.rs` — Entropic domain adaptation (Courty 2017): regularised joint OT plan with group lasso source-label prior; `sinkhorn_lpl1_mm` alternating MM optimisation; transport + classifier training
- [x] `exact/auction_alg.rs` — Auction algorithm for assignment (Bertsekas 1988): ε-scaling price iterations; O(n³/ε) convergence; complementary-slackness termination; alternative to network simplex for dense small-n problems

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 649 passing (unit + 19 e2e integration tests in `e2e_tests.rs`)
- Warnings: 0 (clippy clean)
- `unwrap()` in production code: 0
- macOS: compiles, runtime returns `UnsupportedPlatform` for GPU launches
- All PTX kernels validated as non-empty strings for SM 75 / 80 / 86 / 89 / 90 / 100

## Performance Targets

Optimal transport kernels are bandwidth-limited at small `n × m` and compute-limited at large `n × m`. Sinkhorn dominates total runtime for entropic OT and is the primary acceleration target.

| Operation | Target Reference | Notes |
|-----------|------------------|-------|
| Sinkhorn step (n=1024, m=1024) | ≥ 90% of cuBLAS gemv + LSE chain | bandwidth-bound LSE |
| Cost matrix (n=m=4096, d=128) | ≥ 95% of cuBLAS gemm | reuse syrk path |
| Sliced projection (N=10K, d=128, P=50) | ≥ 85% of cuBLAS gemm + sort | dominated by sort |
| Gromov gradient (n=m=512) | ≥ 90% of two cuBLAS gemms | two-step contraction |
| Barycentric map (n=m=2048, d=128) | ≥ 90% of cuBLAS gemv | scatter-add bound |

## Notes

- All algorithms are deterministic given an `LcgRng` seed (used for SlicedW direction sampling and stochastic ablations).
- The log-domain stabilisation (subtract-max log-sum-exp) is critical for very small `eps`; native-domain Sinkhorn diverges below `eps ≈ 1e-2`.
- The network-simplex solver uses Bland's anti-cycling rule for guaranteed termination; pivot enumeration is `O(n × m)` per iteration.
- This crate is **complementary** to `oxicuda-blas` (which provides the cuBLAS-equivalent linear algebra primitives) and does not duplicate `gemm` / `gemv` kernels.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [ ] Validate Sinkhorn step kernel on T4 (FP16 storage path)
- [ ] Block-size autotuning for cost matrix at small `n × m`

### Ampere (sm_80 / sm_86)
- [ ] `cp.async` 3-stage staging of cost rows for Sinkhorn LSE on A100
- [ ] Tensor-Core (mma.sync) acceleration of cost matrix and Gromov gradient
- [ ] Persistent CTA scheduling for repeated Sinkhorn outer iterations

### Ada (sm_89)
- [ ] FP8 (e4m3 / e5m2) input storage with FP32 LSE accumulator
- [ ] Sparse Tensor-Core path for cost matrix when active set is sparse

### Hopper (sm_90)
- [ ] TMA-based bulk cost-matrix staging for very large `n × m`
- [ ] `wgmma.mma_async` for Gromov gradient and cost matrix
- [ ] Distributed shared memory across CTA cluster for tiled Sinkhorn LSE

### Blackwell (sm_100)
- [ ] `tcgen05` tensor memory layout for FP4 / FP6 Sinkhorn
- [ ] 5th-generation Tensor Core for cost matrix at FP4 precision

---

## Deepening Opportunities

### Verification Gaps
- [ ] All 7 PTX kernels executed end-to-end on GPU hardware (currently only string-content verified)
- [ ] Numerical equivalence between CPU reference (Sinkhorn) and GPU PTX path within FP32 tolerance
- [ ] Benchmark numbers (sinkhorn_step, cost_matrix on A100 / H100) recorded in `benches/ot_ops.rs`
- [x] Sinkhorn ↔ network-simplex agreement verified for large `n × m` (`sinkhorn_agrees_with_network_simplex_on_large_problems` in `e2e_tests.rs` — seeded random 3-D Euclidean-cost instances at `n = m ∈ {16, 32, 64}`: exact network-simplex EMD cost and epsilon-scaled entropic-Sinkhorn cost (`ε` annealed `2.0 → 2e-3`) agree to relative gap `< 8e-3` (measured ≤ 2.6e-3 over 180 random instances), Sinkhorn plan marginals match targets `< 5e-3`, and both plans are feasible/non-negative/unit-mass. **Required first fixing a latent bug in `exact/network_simplex.rs::find_cycle`**: the stepping-stone cycle search used an inverted closing-parity condition and so failed with "could not close cycle" on *every* dense instance with `n ≥ 4` — it was rewritten as a correct iterative alternating-axis DFS (now 100 % solve rate at `n = 5…64`, regression-covered by `solves_generic_instances_above_n3`).)

### Algorithmic Deepening
- [x] Sinkhorn-divergence with debiased gradient backprop (Feydy 2020) for differentiable OT (`sinkhorn/debiased_divergence.rs`)
- [x] Gromov-Wasserstein with batched / mini-batched outer loop (`gromov/batched_gw.rs`)
- [x] Free-support barycenter with adaptive support refinement / pruning (`barycenter/free_support_adaptive.rs`)
- [x] Schrödinger Bridge over time-dependent reference measure (`bridge/tdsb.rs` -- piecewise-constant TDSB with marginal interpolation + transition-plan extraction)
- [x] Multi-marginal OT with structured cost (`multi/mmot_structured.rs` -- pairwise-separable + MMOT barycenter)
- [x] Wasserstein k-means with W2 + entropic regularisation (`clustering/sinkhorn_kmeans.rs`)

### Coverage Gaps vs Literature
- [x] Partial OT with TV and L2 relaxations (`unbalanced/partial_ot.rs`)
- [x] Sliced-Wasserstein on non-Euclidean manifolds (`wasserstein/spherical_sliced.rs` -- SSW + max-SSW)
- [x] Sinkhorn-Knopp with momentum / Nesterov / Anderson acceleration (`sinkhorn/momentum_sinkhorn.rs`)
- [x] Stochastic OT -- mini-batch EMA dual potentials (`wasserstein/stochastic_ot.rs`)
- [x] Gromov-Wasserstein with Wasserstein-Gromov hybrid for graph matching (`gromov/gw_graph_matching.rs`)
- [x] OT-based feature flow for domain generalisation (`domain/feature_flow.rs`)
