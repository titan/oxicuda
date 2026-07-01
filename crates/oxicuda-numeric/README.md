# oxicuda-numeric

Numerical analysis primitives in pure Rust, paired with PTX kernels emitted at runtime for OxiCUDA.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-numeric` is a compact numerical-analysis toolkit covering the
operations that classical scientific-computing libraries provide on top of
BLAS / LAPACK: root finding, univariate quadrature, multi-D cubature,
ordinary-differential-equation integrators, polynomial root solvers, special
functions, numerical differentiation, and interpolation. It is the OxiCUDA
counterpart to the relevant subset of SciPy and the GNU Scientific Library.

The implementation philosophy is "from-scratch, safe Rust": no external
linear-algebra crates, no FFI, and the small linear-algebra primitives that
the higher-level routines need (Jacobi eigen-decomp, Givens QR, Householder
QR, LU with partial pivoting) live in a private `linalg` module. Random
sampling for the Monte Carlo / quasi-Monte Carlo cubature variants uses the
workspace `LcgRng`. The crate enables `#![forbid(unsafe_code)]`.

GPU acceleration for the elementwise / batched hot loops is provided via
PTX strings emitted at runtime, parametric in the device SM version.

## Modules

| Module | Description |
|--------|-------------|
| `root` | Root finding: bisection, Newton, secant, Brent, Halley, Aberth (complex polynomial all-roots) |
| `quadrature` | Romberg, Gauss-Legendre / Hermite / Laguerre / Chebyshev, Clenshaw-Curtis, adaptive Simpson, Gauss-Kronrod |
| `special` | Bessel J/Y/I/K, Airy, Lambert W, hypergeometric 2F1, elliptic K/E, zeta, dilogarithm, exponential integral, polygamma |
| `ode` | Euler, Heun, RK4, DOPRI5, BDF1/2, Rosenbrock-W, IMEX-Euler |
| `poly` | Polynomial roots: Durand-Kerner, Jenkins-Traub, companion matrix, Horner evaluation, polynomial deflation |
| `diff` | Numerical differentiation: central difference, Richardson extrapolation, complex-step |
| `interp` | Interpolation: linear, cubic spline, Akima, PCHIP, Lagrange, Hermite, barycentric Lagrange |
| `cubature` | Multi-D cubature: Monte Carlo, quasi-MC (Sobol), tensor-product Gauss, Genz-Malik adaptive |
| `linalg` | Private helpers: Jacobi eig, QR (Givens / Householder), LU |
| `metrics` | Relative / absolute error, condition-number diagnostics, residual norm |
| `handle` | `NumericHandle`, `SmVersion`, `LcgRng` |
| `error` | `NumericError` / `NumericResult` |
| `ptx_kernels` | Runtime PTX strings for elementwise / batched numerics per SM version |

## Quick Start

```rust,no_run
use oxicuda_numeric::root::brent::brent;
use oxicuda_numeric::NumericResult;

fn main() -> NumericResult<()> {
    // Solve x^3 - x - 2 = 0 on [1, 2] via Brent's method.
    let f = |x: f64| -> NumericResult<f64> { Ok(x * x * x - x - 2.0) };
    let root = brent(f, 1.0, 2.0, 1e-12, 100)?;
    println!("root = {root}");
    Ok(())
}
```

## Design Notes

- `#![forbid(unsafe_code)]` — every routine is implemented in safe Rust.
- Pure Rust: no `openblas` / `lapack` / `gsl` / FFI dependencies. Quadrature
  nodes are computed at runtime by an in-crate Golub-Welsch / Jacobi-eig
  solve rather than being shipped as a static table.
- The ODE solvers cover the standard explicit ladder (Euler → Heun → RK4 →
  embedded DOPRI5) plus the implicit / stiff side (BDF1/2, Rosenbrock-W,
  IMEX-Euler) for problems with stiff source terms.
- All public functions return `NumericResult<T>` and validate input shape /
  tolerance / iteration budget — no `panic!`, no `unwrap()` in production
  code.
- The `linalg` module is intentionally minimal and private — it exists to
  support the higher-level routines (companion-matrix root finding, BDF
  Jacobian solves, MDS-style decompositions). For full BLAS / LAPACK
  coverage, use `oxicuda-blas` and `oxicuda-solver`.

## Status

**Alpha** — 17,147 SLoC, 545 passing tests. API may evolve before v1.0.

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
