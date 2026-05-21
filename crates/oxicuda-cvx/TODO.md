# oxicuda-cvx TODO

GPU-accelerated convex optimisation,
serving as a pure Rust equivalent to CVXPY / SCS / OSQP / Mosek-LP-QP.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.57).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 6,462 (63 files, including 5,420 code + 266 comments + 373 blanks; markdown 403)
- **Tests:** 139 passing (lib + e2e_tests)
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

#### Cone Programs
- [x] `socp/primal_dual_socp.rs` -- Alternating projection (cone + affine) with dual ascent over `(t, x): ||x||_2 <= t`
- [x] `sdp/sdp_interior_point.rs` -- Newton on `-log det X` with PSD projection
- [x] `sdp/log_det_barrier.rs` -- Self-concordant barrier evaluation and gradient / Hessian

#### Splitting Methods
- [x] `admm/admm.rs` -- Vanilla x / z / u updates with over-relaxation
- [x] `admm/consensus_admm.rs` -- Consensus ADMM for separable `f = sum f_i`
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
- [ ] SCS-style operator splitting for general conic programs (LP / QP / SOCP / SDP under one solver)
- [x] OSQP-equivalent: parametric QP with warm starts for model-predictive control (qp/osqp.rs -- Stellato 2020; ADMM on the KKT system, projection onto [l,u], over-relaxation, warm-start, primal/dual residual convergence)
- [ ] Dual decomposition for separable problems with coupling constraints

#### P1 -- Important
- [x] Trust-region methods (Steihaug-Toint, Newton-TR) for unconstrained non-quadratic objectives
- [x] Quasi-Newton: BFGS, L-BFGS (limited-memory) for large-scale smooth optimisation
- [x] Frank-Wolfe / conditional gradient for atomic-norm constrained problems
- [x] Bundle methods for non-smooth convex optimisation (proximal/bundle.rs -- Lemaréchal proximal-bundle; cutting-plane model + Wolfe dual simplex-projected QP master + serious/null step logic by actual-vs-predicted decrease ratio)
- [ ] Cutting-plane methods for semi-infinite programs
- [x] Inexact prox via inner conjugate gradient (for non-closed-form prox) (proximal/inexact_prox.rs -- solve prox_g(v)=argmin g(x)+ρ/2‖x−v‖² for quadratic g via inner CG on (A+ρI)x=ρv+b SPD system)
- [x] Spectral projected gradient (SPG) with non-monotone line search (gradient/spg.rs -- Birgin 2000; Barzilai-Borwein spectral step + nonmonotone GLL line search + projection closure)
- [ ] Interior-point method with Mehrotra correction for QP and SOCP (parity with LP path)

#### P2 -- Nice-to-Have
- [x] Newton on the dual (for problems with cheap dual function) (gradient/dual_newton.rs -- Boyd-Vandenberghe §9.5; (H(λ)+μI)Δ=∇g, reuses linalg::solve_dense, Full or Armijo-backtracking step)
- [x] Stochastic / mini-batch versions: SGD, SVRG, SAGA with proximal step
- [x] Coordinate descent (cyclic, random, accelerated) for separable smooth objectives
- [x] Block-coordinate descent (BCD) for structured problems (gradient/block_coord_descent.rs -- Tseng 2001 / Beck-Tetruashvili 2013; cyclic/random sweep, exact per-block SPD solve via linalg::solve_dense or inner gradient descent, specialised quadratic API)
- [x] Mirror descent for non-Euclidean geometries
- [ ] Riemannian convex optimisation hooks (link with `oxicuda-manifold`)
- [ ] Disciplined Convex Programming (DCP) front-end with operator tree
- [ ] Differentiable convex layers (`cvxpylayers`-style differentiation through KKT)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 139 passing
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
- [ ] Adaptive penalty `rho` for ADMM (Boyd 2011 residual-balancing)
- [ ] Restart strategies for FISTA (gradient restart, function restart)
- [ ] Approximate projection / prox via inner iterations with bounded inexactness
- [ ] Higher-order primal-dual methods (golden-ratio Chambolle-Pock, GRPDA)
- [ ] Preconditioned conjugate gradient inside the linear-system solves of IPM
- [ ] Sparse KKT factorisation (instead of dense Cholesky) for large LP / QP

### API Polish
- [ ] Builder pattern `LpSolverBuilder::tolerance(1e-8).max_iter(100).method(Method::Mehrotra).solve(&problem)`
- [ ] Common `ProblemSpec` trait so the same problem can be dispatched to LP / QP / SOCP / SDP backends
- [ ] Re-export the most common prox / projection operators at the crate root for ergonomic use
- [ ] Cross-link with `oxicuda-cs` (Vol.58) for sparse recovery via L1 / TV-L2 fronts and with
  `oxicuda-stats` for penalised regression objectives
