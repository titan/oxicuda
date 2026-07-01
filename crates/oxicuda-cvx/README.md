# oxicuda-cvx

Convex optimization solvers -- LP, QP, SOCP, SDP, splitting, and proximal methods in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-cvx` is a self-contained convex optimization library covering the
canonical cones (linear, quadratic, second-order, semidefinite) and the major
modern first-order families (ADMM, proximal gradient, FISTA, accelerated
gradient, Douglas-Rachford, primal-dual). It is intended for use as a backend
to higher-level OxiCUDA crates that need to solve convex subproblems --
sparse recovery, robust regression, optimal control, conic relaxation -- as
well as for direct, embedded solver use.

GPU PTX kernels are generated and dispatched entirely from Rust via the
OxiCUDA driver stack. There is no C/CUDA toolchain at build time and no
external BLAS/LAPACK dependency: the internal `linalg` module supplies
conjugate gradient, Cholesky, Householder QR, and triangular solves in pure
Rust.

Coverage spans the revised simplex and Mehrotra predictor-corrector primal-
dual interior-point methods for LP; active-set and primal-dual IPM for QP;
primal-dual IPM for second-order cone programs; log-det barrier interior-
point for SDP; ADMM (vanilla and consensus); proximal gradient, FISTA,
accelerated, Douglas-Rachford splitting; Chambolle-Pock primal-dual; closed-
form proximal operators for the standard norms and indicator functions;
projections onto common convex sets; the augmented Lagrangian method; and
projected/Nesterov/Polyak gradient variants with Armijo/Wolfe line search.

## Modules

| Module | Description |
|--------|-------------|
| `lp` | Linear programming: revised simplex, Mehrotra primal-dual IPM |
| `qp` | Quadratic programming: active-set, primal-dual IPM, OSQP-style |
| `socp` | Second-order cone programming: primal-dual IPM |
| `sdp` | Semidefinite programming: interior point, log-det barrier |
| `admm` | Alternating Direction Method of Multipliers, consensus ADMM |
| `proximal` | Proximal gradient, FISTA, accelerated, Douglas-Rachford |
| `primal_dual` | Chambolle-Pock saddle-point algorithm |
| `prox_ops` | Closed-form prox: L1, L2, L-inf, group lasso, elastic net, nuclear, 1D-TV, indicator |
| `projection` | Projections onto simplex, L1/L2 balls, box, PSD cone, SOC, halfspace |
| `augmented_lagrangian` | Method of multipliers / ALM |
| `gradient` | Projected GD, Nesterov accelerated GD, Polyak heavy-ball |
| `linesearch` | Armijo, Wolfe, strong Wolfe, backtracking |
| `linalg` | CG, matvec, Cholesky, QR (Householder), triangular solves |
| `metrics` | Duality gap, primal/dual residual, KKT residual, convergence rates |
| `handle` | `CvxHandle`, `SmVersion`, `LcgRng` (MMIX LCG) |
| `error` | `CvxError` / `CvxResult` |
| `ptx_kernels` | GPU PTX kernel templates per SM target |

## Quick Start

```rust,no_run
use oxicuda_cvx::lp::revised_simplex;
use oxicuda_cvx::{CvxHandle, SmVersion};

fn main() -> oxicuda_cvx::CvxResult<()> {
    // Compute handle bound to an SM target and seeded RNG.
    let _handle = CvxHandle::new(SmVersion::SM_90, 0xC0FFEE);

    // Solve  min cᵀx  s.t.  A x = b,  x >= 0.
    let m = 2usize; // number of equality constraints
    let n = 4usize; // number of variables (including slacks)
    let a: Vec<f64> = unimplemented!();        // m * n, row-major
    let b: Vec<f64> = unimplemented!();        // length m, b >= 0
    let c: Vec<f64> = unimplemented!();        // length n
    let initial_basis: Vec<usize> = unimplemented!(); // length m, feasible basis indices

    let result = revised_simplex(&a, m, n, &b, &c, &initial_basis, 1_000)?;
    let _x = result.x;                 // optimal primal
    let _obj = result.objective;       // optimal value
    let _status = result.status;       // Optimal / Unbounded / Infeasible / IterationLimit
    Ok(())
}
```

## Status

**Alpha** -- 20,511 SLoC, 669 passing tests. API may evolve before v1.0.

## License

Apache-2.0
