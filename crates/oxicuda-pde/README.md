# oxicuda-pde

Numerical PDE solvers -- finite differences, finite elements, spectral methods, multigrid, and iterative linear solvers in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-pde` is the numerical partial-differential-equation volume of the
OxiCUDA stack. It bundles the standard families of discretisation methods
needed to solve elliptic, parabolic, and hyperbolic PDEs in 1D / 2D / 3D,
together with the iterative linear solvers, time-stepping schemes, and
multigrid building blocks that those discretisations rely on.

All algorithms are implemented in pure Rust with no external linear-algebra
dependencies. The crate also ships GPU PTX kernel generators in
`ptx_kernels`, parameterised on SM compute capability (Turing through
Blackwell), for the operations that map directly to dense / sparse kernels.

Method coverage is intentionally broad rather than deep: the goal is to
provide a working reference implementation of each canonical scheme that
downstream OxiCUDA crates can call into without pulling in C/Fortran
libraries.

## Modules

| Module | Description |
|--------|-------------|
| `mesh` | Mesh data structures for finite-difference and finite-element solvers |
| `fdm` | Finite-difference methods (FDM) for Poisson, heat, wave, advection PDEs |
| `fem` | Finite-element method (P1 linear and P2 quadratic triangles) |
| `spectral` | Spectral methods: Chebyshev collocation and FFT-based pseudo-spectral |
| `time` | Time-stepping schemes for ODE / spatially-discretised PDE systems |
| `multigrid` | Geometric multigrid V-cycle (1D and 2D) |
| `bc` | Boundary condition types and helpers |
| `solver` | Iterative and direct linear solvers for sparse systems |
| `dg` | Discontinuous Galerkin (DG) methods |
| `metrics` | Convergence metrics: L2 norm, H1 seminorm, max-norm, convergence-order estimation |
| `handle` | `PdeHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernels for numerical PDE operations |
| `error` | `PdeError` / `PdeResult` |

## Supported Methods

### Finite differences (`fdm`)
- Poisson in 1D / 2D / 3D
- Heat equation in 1D / 2D
- Wave equation in 1D / 2D
- Linear advection 1D
- Viscous Burgers 1D

### Finite elements (`fem`)
- P1 linear triangles (2D) with mass / stiffness assembly
- P2 quadratic triangles
- P1 linear tetrahedra (3D)
- Dirichlet application on CSR systems

### Spectral methods (`spectral`)
- Chebyshev collocation differentiation
- Periodic pseudo-spectral via DFT
- 2D Fourier spectral operators

### Multigrid (`multigrid`)
- Geometric V-cycle in 1D and 2D
- Restriction / prolongation operators
- Weighted Jacobi and Gauss-Seidel smoothers

### Linear solvers (`solver`)
- Conjugate Gradient (CG)
- Preconditioned CG with Jacobi / ILU(0) / SSOR
- BiCGStab, GMRES
- Jacobi iteration
- Sparse CSR matvec primitives

### Time stepping (`time`)
- Forward / backward Euler
- Crank-Nicolson
- RK4, BDF2, IMEX

### Discontinuous Galerkin (`dg`)
- 1D DG with LGL nodal basis and upwind / Lax-Friedrichs fluxes

## Quick Start

```rust,no_run
use oxicuda_pde::bc::dirichlet::Dirichlet1d;
use oxicuda_pde::fdm::poisson_1d::solve_poisson_1d;
use oxicuda_pde::mesh::mesh1d::Mesh1d;
use oxicuda_pde::PdeResult;

fn main() -> PdeResult<()> {
    // Uniform mesh on [0, 1] with 65 nodes.
    let mesh = Mesh1d::uniform(0.0, 1.0, 65)?;

    // Right-hand side f(x) sampled at the mesh nodes (user supplies values).
    let f_vals: Vec<f64> = unimplemented!();

    // Dirichlet boundary conditions u(0) = 0, u(1) = 0.
    let bc = Dirichlet1d { ua: 0.0, ub: 0.0 };

    // Solve -u''(x) = f(x).
    let _u = solve_poisson_1d(&mesh, &f_vals, bc)?;
    Ok(())
}
```

## Status

**Alpha** -- 23,803 SLoC, 680 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
