# oxicuda-cvx TODO

GPU-accelerated convex optimisation,
serving as a pure Rust equivalent to CVXPY / SCS / OSQP / Mosek-LP-QP.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.57).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 20,511 (106 files, including 5,420 code + 266 comments + 373 blanks; markdown 403)
- **Tests:** 659 passing (lib + e2e_tests) + 3 doc-tests
- **Pure Rust:** Zero external linear-algebra dependencies; only `thiserror` runtime dep
- **PTX coverage:** 7 kernels x 6 SM versions = 42 PTX string generators

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `CvxError` enum (14 variants: NotConverged, ShapeMismatch, Infeasible, Unbounded, InvalidParameter, NumericalInstability, UnsupportedSmVersion, SingularMatrix, IndexOutOfBounds, DimensionMismatch, EmptyInput, ConeViolation, ...) + `CvxResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `CvxHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `axpy`, `soft_threshold`, `simplex_proj`, `gradient_step`, `fista_extrapolate`, `admm_dual_update`, `proj_l2_ball` (string concatenation only, no nvcc dependency)

#### Linear Programming
- [x] `lp/revised_simplex.rs` -- Revised simplex with Bland's rule + LU-of-basis updates
- [x] `lp/primal_dual_lp.rs` -- Primal-dual interior-point LP solver
- [x] `lp/mehrotra.rs` -- Mehrotra predictor-corrector primal-dual IP with centring `sigma = (mu_aff / mu)^3` + step cap `alpha = 0.99`

#### Quadratic Programming
- [x] `qp/active_set_qp.rs` -- Active-set QP with Schur-complement KKT
- [x] `qp/primal_dual_qp.rs` -- Primal-dual interior-point QP
- [x] `qp/mehrotra_qp.rs` -- Mehrotra 1992 Predictor-Corrector QP: min ½xᵀPx+qᵀx s.t. Ax=b,x≥0; affine predictor (σ=0), centering σ=(μ_aff/μ)³, corrector with cross-term Δx·Δz, 0.99 fraction-to-boundary step; 9 unit tests + 2 e2e

#### Cone Programs
- [x] `socp/primal_dual_socp.rs` -- Alternating projection (cone + affine) with dual ascent over `(t, x): ||x||_2 <= t`
- [x] `sdp/sdp_interior_point.rs` -- Newton on `-log det X` with PSD projection
- [x] `sdp/log_det_barrier.rs` -- Self-concordant barrier evaluation and gradient / Hessian

#### Splitting Methods
- [x] `admm/admm.rs` -- Vanilla x / z / u updates with over-relaxation
- [x] `admm/consensus_admm.rs` -- Consensus ADMM for separable `f = sum f_i`
- [x] `admm/dual_decomp.rs` -- Dual Decomposition (Boyd §7.2): dual ascent for separable min Σfᵢ(xᵢ) s.t. ΣAᵢxᵢ=b; closure-based x-updates for maximum generality; configurable step_size/max_iter/tol; 9 unit tests
- [x] `proximal/prox_gradient.rs` -- Proximal gradient with backtracking
- [x] `proximal/fista.rs` -- FISTA momentum `t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2`
- [x] `proximal/accelerated.rs` -- Nesterov accelerated proximal gradient
- [x] `proximal/douglas_rachford.rs` -- `y <- prox_f(x), z <- prox_g(2 y - x), x <- x + z - y`
- [x] `primal_dual/chambolle_pock.rs` -- Primal-dual extrapolation for `min f(K x) + g(x)` with `tau * sigma * ||K||^2 < 1`

#### Proximal Operators (closed form)
- [x] `prox_ops/l1.rs` -- Soft-thresholding `sign(x) * max(|x| - lambda, 0)`
- [x] `prox_ops/l2.rs` -- Tikhonov / block-L2
- [x] `prox_ops/linf.rs` -- L-infinity via Moreau dual
- [x] `prox_ops/group_lasso.rs` -- Group-block soft-threshold
- [x] `prox_ops/elastic_net.rs` -- L1 + L2 combined
- [x] `prox_ops/nuclear.rs` -- Singular-value soft-threshold via Jacobi SVD
- [x] `prox_ops/total_variation_1d.rs` -- Condat O(n) 1D-TV
- [x] `prox_ops/indicator.rs` -- Indicator of convex set

#### Projection Operators (closed form)
- [x] `projection/simplex.rs` -- Wang-CP O(n log n) projection on probability simplex
- [x] `projection/l1_ball.rs` -- O(n log n) projection on L1 ball
- [x] `projection/l2_ball.rs` -- `x * min(1, r / ||x||)`
- [x] `projection/box_proj.rs` -- Coordinate-wise clamp
- [x] `projection/psd_cone.rs` -- Eigh + clip negative eigenvalues
- [x] `projection/soc_cone.rs` -- Second-order cone `(t, x): ||x||_2 <= t`
- [x] `projection/halfspace.rs` -- Affine halfspace `a^T x <= b`
- [x] `projection/dykstra_pocs.rs` -- Dykstra 1983 POCS: projection onto convex set intersections ∩Cᵢ with Dykstra increment corrections pᵢ; convergence to nearest point (not just any point in intersection); 9 unit tests + 4 e2e

#### Augmented Lagrangian
- [x] `augmented_lagrangian/alm.rs` -- Method of multipliers for equality-constrained problems

#### Gradient Methods
- [x] `gradient/projected_gradient.rs` -- Projected GD `x <- proj_C(x - alpha grad f)`
- [x] `gradient/accelerated_gd.rs` -- Nesterov accelerated gradient
- [x] `gradient/momentum_gd.rs` -- Polyak heavy-ball

#### Line Search
- [x] `linesearch/armijo.rs` -- Armijo `f(x + alpha d) <= f(x) + c_1 alpha grad f^T d`
- [x] `linesearch/wolfe.rs` -- Wolfe conditions
- [x] `linesearch/strong_wolfe.rs` -- Strong Wolfe with `|grad f(x + alpha d)^T d| <= c_2 |grad f^T d|`
- [x] `linesearch/backtracking.rs` -- Backtracking (Armijo-only)

#### Private Linear Algebra
- [x] `linalg/cg.rs` -- Dense conjugate-gradient solver
- [x] `linalg/matvec.rs` -- Generic matvec
- [x] `linalg/cholesky.rs` -- Cholesky factorisation
- [x] `linalg/qr.rs` -- Householder QR
- [x] `linalg/solve.rs` -- Triangular solves + dense LU

#### Diagnostics
- [x] `metrics/metrics.rs` -- Duality gap, primal / dual residual, KKT residual, convergence-rate estimator

#### Validation
- [x] `e2e_tests.rs` -- 39 cross-module tests: LP 2D recovers vertex with -1 objective; QP identity-constrained returns 1; L1 prox `[2, 0.5, -0.5, -2] -> [1, 0, 0, -1]`; simplex projection of `[1, 1, 1] = [1/3, 1/3, 1/3]`; PSD projection of `[-1, 0; 0, 1] -> [0, 0; 0, 1]`; TV-1D denoising reduces stair-step; FISTA L1-LS O(1/k^2) rate; ADMM-Lasso matches FISTA; Chambolle-Pock TV-L2 monotone primal energy; projected GD on box quadratic -> KKT; strong Wolfe satisfies both conditions; PTX x 6 SM
- [x] `benches/cvx_ops.rs` -- Criterion: 7 PTX kernels x all SM + LP / FISTA / ADMM / Chambolle-Pock algo benches

### Future Enhancements

#### P0 -- Critical
- [x] `scs/scs_solver.rs` — SCS-style unified conic solver (O'Donoghue 2021): operator splitting LP/QP/SOCP/SDP under one ADMM framework; K-cones = {non-negative, second-order, positive-semidefinite, exponential, power}; homogeneous self-dual embedding for unbounded/infeasible detection
- [x] `dcp/expr_tree.rs` — DCP expression tree (Grant-Boyd 2008): atom library (max,min,log,exp,norm,quad_form,huber,kl_div) with curvature propagation rules; reduce to standard conic form; dispatch to SCS or primal-dual solver
- [x] OSQP-equivalent: parametric QP with warm starts for model-predictive control (qp/osqp.rs -- Stellato 2020; ADMM on the KKT system, projection onto [l,u], over-relaxation, warm-start, primal/dual residual convergence)
- [x] Dual decomposition for separable problems with coupling constraints (`admm/dual_decomp.rs` -- Boyd §7.2; dual ascent for separable min Σfᵢ(xᵢ) s.t. ΣAᵢxᵢ=b; closure-based x-updates; 9 unit tests)

#### P1 -- Important
- [x] Trust-region methods (Steihaug-Toint, Newton-TR) for unconstrained non-quadratic objectives
- [x] Quasi-Newton: BFGS, L-BFGS (limited-memory) for large-scale smooth optimisation
- [x] Frank-Wolfe / conditional gradient for atomic-norm constrained problems
- [x] Bundle methods for non-smooth convex optimisation (proximal/bundle.rs -- Lemaréchal proximal-bundle; cutting-plane model + Wolfe dual simplex-projected QP master + serious/null step logic by actual-vs-predicted decrease ratio)
- [x] `lp/cut_plane.rs` — Cutting-plane method for semi-infinite programs: iteratively add violated constraints as cutting planes; column-generation outer loop; convergence O(1/ε²) for convex subproblems
- [x] Inexact prox via inner conjugate gradient (for non-closed-form prox) (proximal/inexact_prox.rs -- solve prox_g(v)=argmin g(x)+ρ/2‖x−v‖² for quadratic g via inner CG on (A+ρI)x=ρv+b SPD system)
- [x] Spectral projected gradient (SPG) with non-monotone line search (gradient/spg.rs -- Birgin 2000; Barzilai-Borwein spectral step + nonmonotone GLL line search + projection closure)
- [x] Interior-point method with Mehrotra correction for QP (`qp/mehrotra_qp.rs` -- Mehrotra 1992; predictor-corrector IPM for QP with σ=(μ_aff/μ)³ and cross-term correction; 9 unit tests)
- [x] `qp/mehrotra_socp.rs` — Mehrotra PC for SOCP: extend QP predictor-corrector to second-order cone; Nesterov-Todd scaling; iterate (x,s,λ) with cone-projection step-length guard; parity with existing LP path (ALREADY EXISTS as `socp/mehrotra_socp.rs` -- full NT scaling via Jordan-algebra quadratic-rep `Q_u=2uuᵀ−det(u)J`, normal equations `A W² Aᵀ`, predictor σ=(μ_aff/μ)³, corrector with cross-term `Δx̂∘Δŝ`, fraction-to-boundary cone-step guard; 11 unit tests; re-exported in `lib.rs`)
- [x] `riemannian/riemannian_cvx.rs` — Riemannian gradient descent + retraction (Absil 2008): Riemannian gradient via orthogonal projection of Euclidean gradient; retraction via QR/SVD for Stiefel, eigen for SPD, exp map for Grassmann; geodesic Armijo line search

#### P2 -- Nice-to-Have
- [x] Newton on the dual (for problems with cheap dual function) (gradient/dual_newton.rs -- Boyd-Vandenberghe §9.5; (H(λ)+μI)Δ=∇g, reuses linalg::solve_dense, Full or Armijo-backtracking step)
- [x] Stochastic / mini-batch versions: SGD, SVRG, SAGA with proximal step
- [x] Coordinate descent (cyclic, random, accelerated) for separable smooth objectives
- [x] Block-coordinate descent (BCD) for structured problems (gradient/block_coord_descent.rs -- Tseng 2001 / Beck-Tetruashvili 2013; cyclic/random sweep, exact per-block SPD solve via linalg::solve_dense or inner gradient descent, specialised quadratic API)
- [x] Mirror descent for non-Euclidean geometries
- [x] `differentiable/kkt_diff.rs` — Differentiating through KKT conditions (Amos-Kolter 2017 OptNet): implicit function theorem on KKT system; dL/dθ via solve of transposed KKT with adjoint; enables cvxpylayers-style end-to-end training
- [x] `admm/async_admm.rs` — Asynchronous ADMM (Zhang-Recht 2014): parallel block updates without global synchronisation barrier; bounded-delay convergence guarantee for separable problems
- [x] `gradient/polyak.rs` — Polyak step-size for subgradient (Polyak 1969): αₖ=(f(xₖ)-f*)/‖gₖ‖² with f* unknown (use moving estimate); geometric convergence for strongly convex + sharp subgradient problems

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 659 passing + 3 doc-tests
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- Pure Rust: no C/C++/Fortran in default features

## Performance Targets

Representative algorithmic benchmarks (CPU-side reference + PTX generation timing):

| Routine | Problem size | Priority |
|---------|--------------|----------|
| Revised simplex LP | (m, n) in {(50, 100), (200, 500)} | High |
| Mehrotra IPM LP | (m, n) in {(50, 100), (200, 500)} | High |
| Active-set QP | (n, m_ineq) in {(50, 20), (200, 100)} | High |
| FISTA on L1-LS | (m, n) in {(500, 1000), (2000, 5000)} | High |
| ADMM (Lasso) | (m, n) in {(500, 1000), (2000, 5000)} | High |
| Chambolle-Pock (TV-L2) | n in {1024, 4096} | High |
| Simplex / L1-ball projection | n in {1e3, 1e5} | Mid |
| PSD projection | dim in {32, 128} | Mid |
| SOCP primal-dual | (n, m_cone) in {(50, 2), (200, 5)} | Mid |

Target for GPU execution path: match SCS / OSQP convergence within 1.5x iterations
and outperform CPU SCS at n >= 1e4 once `oxicuda-launch` orchestrates the emitted
PTX on Linux + NVIDIA.

## Notes

- FISTA achieves O(1 / k^2) sublinear convergence; verified empirically against gradient descent's O(1 / k).
- ADMM penalty `rho` is fixed by the caller; adaptive `rho` (Boyd) is on the P1 roadmap.
- Chambolle-Pock step-size condition `tau * sigma * ||K||^2 < 1` is checked at problem setup.
- All projections are exact (closed form) and tested for idempotence and contraction.
- KKT residual computed once per iteration as a single convergence criterion.

---

## Architecture-Specific Deepening

### PTX Coverage Matrix

| Kernel | sm_70 | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|--------|-------|-------|-------|-------|-------|-------|
| `axpy` | [x] | [x] | [x] | [x] | [x] | [x] |
| `soft_threshold` | [x] | [x] | [x] | [x] | [x] | [x] |
| `simplex_proj` | [x] | [x] | [x] | [x] | [x] | [x] |
| `gradient_step` | [x] | [x] | [x] | [x] | [x] | [x] |
| `fista_extrapolate` | [x] | [x] | [x] | [x] | [x] | [x] |
| `admm_dual_update` | [x] | [x] | [x] | [x] | [x] | [x] |
| `proj_l2_ball` | [x] | [x] | [x] | [x] | [x] | [x] |

All six SM versions produce non-empty PTX strings and pass content-substring checks in `e2e_tests.rs`.

### Per-Architecture Optimisation Hooks
- [ ] sm_80 (Ampere) -- warp-cooperative reduction inside `axpy` and `gradient_step` for fused norm + step
- [ ] sm_89 (Ada) -- mixed-precision FP16 storage + FP32 accumulate for FISTA / ADMM iterates
- [ ] sm_90 (Hopper) -- TMA + warp-specialised pipelining for `fista_extrapolate` and `admm_dual_update`
- [ ] Verify `simplex_proj` and `proj_l2_ball` numerical stability under denormal inputs on all SM versions

---

## Deepening Opportunities

### Verification Gaps (require Linux + NVIDIA hardware)
- [ ] GPU run of all 7 PTX kernels under `cargo nextest --features gpu-tests` on sm_80 / sm_89 / sm_90
- [ ] End-to-end LP / QP solve agreement vs. SCS / OSQP on canonical benchmark suites (Netlib LP, Maros-Meszaros QP)
- [ ] FISTA / ADMM throughput vs. CPU `scs` and `osqp` at n = 1e5

### Algorithmic Deepening
- [x] Adaptive penalty `rho` for ADMM (Boyd 2011 residual-balancing) (`admm/adaptive_rho_admm.rs` -- Boyd §3.4.1; ρ⁺=τ⁺ρ when ‖r‖>μ‖s‖, ρ⁻=ρ/τ⁻ when ‖s‖>μ‖r‖, scaled-dual rescale u←(ρ_old/ρ_new)u to preserve y=ρu, throttled via `adapt_every`, Boyd √p·ε_abs+ε_rel·max(‖Ax‖,‖Bz‖,‖c‖) feasibility test; 6 unit tests incl. faster-than-fixed-bad-ρ)
- [x] Restart strategies for FISTA (gradient restart, function restart) (`proximal/fista_restart.rs` -- O'Donoghue-Candès 2015; `RestartRule::{Gradient,Function,None}`, gradient rule ⟨y−x⁺,x⁺−x⟩>0 (no extra objective eval), function rule on F(x⁺)>F(x) with monotone ISTA-from-x fallback, optional backtracking; 5 unit tests incl. beats-plain-FISTA on κ≈1000)
- [ ] Approximate projection / prox via inner iterations with bounded inexactness (NOTE: substantially covered by existing `proximal/inexact_prox.rs` -- inner-CG prox for quadratic g)
- [x] Higher-order primal-dual methods (golden-ratio Chambolle-Pock, GRPDA) (`primal_dual/grpda.rs` -- Chang-Yang 2021; convex-combination z_k=((ψ−1)/ψ)x_{k−1}+(1/ψ)z_{k−1} with ψ∈(1,φ], Gauss-Seidel prox steps, enlarged step region τσ‖K‖²<ψ≤φ vs PDHG's <1, `balanced()` step builder; 7 unit tests incl. matches Chambolle-Pock on least-squares + τσ‖K‖²≈1.5>1 still converges)
- [ ] Preconditioned conjugate gradient inside the linear-system solves of IPM (internal-refactor of existing dense-Cholesky IPM path; deferred)
- [ ] Sparse KKT factorisation (instead of dense Cholesky) for large LP / QP (requires sparse-matrix infrastructure; deferred)

### API Polish
- [x] Builder pattern `LpSolverBuilder::tolerance(1e-8).max_iter(100).method(Method::Mehrotra).solve(&problem)` (`builder.rs:LpSolverBuilder` -- fluent `.tolerance()/.max_iter()/.method()/.solve()`; `LpMethod::{Simplex,Mehrotra,PrimalDual}` dispatches to `revised_simplex`/`mehrotra_predictor_corrector`/`primal_dual_lp`; default Mehrotra, tol 1e-9, 200 iters; simplex falls back to trailing-m slack basis; unified `LpSolution`; 7 unit tests verify builder output matches each direct solver call)
- [x] Common `ProblemSpec` trait so the same problem can be dispatched to LP / QP / SOCP / SDP backends (`problem.rs:ProblemSpec` -- `form()` + `dispatch()`, generic free fn `solve(&impl ProblemSpec)`; self-contained data structs `LpProblem`→`mehrotra_predictor_corrector`, `QpProblem`→`mehrotra_qp`, `SocpProblem`→`mehrotra_socp`, `SdpProblem`→`sdp_interior_point`; uniform `ProblemSolution{x,objective,iter,form}`; 6 unit tests incl. generic-over-`P: ProblemSpec` dispatch and per-form match-vs-direct)
- [x] Re-export the most common prox / projection operators at the crate root for ergonomic use (`lib.rs` -- `pub use` of `soft_threshold`, `prox_l1/l2/linf/elastic_net/group_lasso/nuclear/tv_1d`, `prox_indicator_box/simplex/l1_ball/l2_ball`, `project_simplex/l1_ball/l2_ball/box/halfspace/psd_cone/soc`; crate-root doc-test exercises `oxicuda_cvx::soft_threshold` / `prox_l1` / `project_simplex`)
- [ ] Cross-link with `oxicuda-cs` (Vol.58) for sparse recovery via L1 / TV-L2 fronts and with
  `oxicuda-stats` for penalised regression objectives (cross-crate; out of scope for a single-crate change, no new deps)
