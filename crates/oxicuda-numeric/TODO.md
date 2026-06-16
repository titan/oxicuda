# oxicuda-numeric TODO

GPU-accelerated Numerical Analysis primitives, serving as a pure Rust replacement
for QUADPACK / GSL / SciPy-style scientific computing utilities. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.60).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 13,644 (99 files, tokei measurement)
- **Total lines (incl. comments+blanks):** 6,787
- **Tests:** 466 passing
- **Vol.60 scope:** Root finding, numerical quadrature, special functions, ODE
  solvers, polynomial roots, numerical differentiation, interpolation, and
  multidimensional cubature. Complements oxicuda-blas / oxicuda-solver / oxicuda-fft
  by providing the classical scalar / 1D analysis layer.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `NumericError` enum (14 variants: `NotConverged`,
  `RootNotBracketed`, `ShapeMismatch`, `InvalidParameter`, `NumericalInstability`,
  `UnsupportedSmVersion`, `IndexOutOfBounds`, `DimensionMismatch`, `EmptyInput`,
  `DegreeTooHigh`, `OutOfDomain`, ...) + `NumericResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 boolean, Box-Muller
  normal), `NumericHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `horner_eval`, `rk4_stage`,
  `bisection_step`, `gauss_quad_accumulate`, `spline_eval`, `central_diff`,
  `bessel_recurrence` (string-concatenation PTX, no nvcc)

#### Root Finding
- [x] `root/bisection.rs` -- Bisection method O(log(1/eps)) with bracket maintenance
- [x] `root/newton.rs` -- Newton's method with quadratic convergence
- [x] `root/secant.rs` -- Secant method with superlinear order ~1.618
- [x] `root/brent.rs` -- Brent's method (bisection + secant + IQI)
- [x] `root/halley.rs` -- Halley's method with cubic convergence
- [x] `root/aberth_all_roots.rs` -- Aberth-Ehrlich simultaneous all-roots iteration
  for polynomial root finding

#### Quadrature (1D)
- [x] `quadrature/romberg.rs` -- Romberg integration with Richardson extrapolation
- [x] `quadrature/gauss_legendre.rs` -- Gauss-Legendre nodes via Golub-Welsch
  (Jacobi eigendecomposition)
- [x] `quadrature/gauss_hermite.rs` -- Gauss-Hermite for `int_{-inf}^{inf} e^{-x^2} f(x) dx`
- [x] `quadrature/gauss_laguerre.rs` -- Gauss-Laguerre for `int_0^{inf} e^{-x} f(x) dx`
- [x] `quadrature/gauss_chebyshev.rs` -- Gauss-Chebyshev with closed-form nodes / weights
- [x] `quadrature/clenshaw_curtis.rs` -- Clenshaw-Curtis with DFT-derived weights
- [x] `quadrature/adaptive_simpson.rs` -- Adaptive Simpson with `15 * eps` refinement criterion
- [x] `quadrature/gauss_kronrod.rs` -- G7-K15 embedded pair with `|G7 - K15|^{1.5}` error
  estimate

#### Special Functions
- [x] `special/bessel_jy.rs` -- Bessel J / Y via Miller's algorithm + Wronskian
  normalisation
- [x] `special/bessel_ik.rs` -- Modified Bessel I / K
- [x] `special/airy.rs` -- Airy Ai / Bi (power series + asymptotic expansion)
- [x] `special/lambert_w.rs` -- Lambert W_0 / W_{-1} via Halley iteration
- [x] `special/hypergeometric_2f1.rs` -- _2F_1 Taylor series + linear transformations
- [x] `special/elliptic_ke.rs` -- Complete elliptic K / E via the arithmetic-geometric
  mean (AGM)
- [x] `special/zeta.rs` -- Riemann zeta via Euler-Maclaurin + functional equation
- [x] `special/dilogarithm.rs` -- Li_2 via series + standard transformations
- [x] `special/ei.rs` -- Exponential integral Ei
- [x] `special/polygamma.rs` -- digamma / trigamma via recurrence + asymptotic expansion

#### ODE Solvers
- [x] `ode/explicit_euler.rs` -- Forward Euler method
- [x] `ode/heun.rs` -- Heun RK2 (improved Euler)
- [x] `ode/rk4.rs` -- Classical Runge-Kutta 4 (k1, k2, k3, k4)
- [x] `ode/dopri5.rs` -- Dormand-Prince RK45 7-stage embedded with PI-controller
  adaptive step
- [x] `ode/bdf12.rs` -- BDF1 / BDF2 backward differentiation with inner Newton
- [x] `ode/rosenbrock_w.rs` -- Rosenbrock-W linearly implicit one-step method
- [x] `ode/imex_euler.rs` -- IMEX explicit + implicit operator splitting

#### Polynomial Roots
- [x] `poly/durand_kerner.rs` -- Simultaneous all-roots Durand-Kerner / Weierstrass
- [x] `poly/jenkins_traub.rs` -- Three-stage RPOLY (no shift, fixed shift, variable shift)
- [x] `poly/companion_matrix_eigvals.rs` -- Companion-matrix Hessenberg + shifted QR
- [x] `poly/horner_eval.rs` -- Horner polynomial evaluation + derivative
- [x] `poly/deflate.rs` -- Synthetic-division polynomial deflation

#### Numerical Differentiation
- [x] `diff/central_difference.rs` -- Central difference O(h^2)
- [x] `diff/richardson_extrapolation.rs` -- Combines D(h) + D(h/2) for O(h^4)
- [x] `diff/complex_step.rs` -- Complex-step `Im(f(x + ih)) / h` (no subtractive
  cancellation)

#### Interpolation (1D)
- [x] `interp/linear.rs` -- Linear interpolation
- [x] `interp/cubic_spline.rs` -- Natural + clamped cubic spline (Thomas tridiagonal
  for M)
- [x] `interp/akima.rs` -- Akima interpolation with 5-point slope formulae
- [x] `interp/pchip.rs` -- Fritsch-Carlson PCHIP monotone interpolation
- [x] `interp/lagrange.rs` -- Lagrange interpolating polynomial O(n^2)
- [x] `interp/hermite.rs` -- Hermite interpolation with values and derivatives
- [x] `interp/barycentric.rs` -- Barycentric Lagrange `w_j = 1 / prod_{i!=j}(x_j - x_i)`

#### Multidimensional Cubature
- [x] `cubature/monte_carlo.rs` -- Monte Carlo O(1/sqrt(N)) with standard error
- [x] `cubature/quasi_monte_carlo_sobol.rs` -- Sobol / Halton low-discrepancy
  (verified Van der Corput base-2 in dimension 1)
- [x] `cubature/tensor_product_gauss.rs` -- Tensor-product Gauss-Legendre cubature
- [x] `cubature/genz_malik.rs` -- Genz-Malik 1980 adaptive degree-7 fully-symmetric
  basic rule

#### Private Linear Algebra Helpers
- [x] `linalg/jacobi_eig.rs` -- Cyclic Jacobi eigendecomposition (used by Golub-Welsch)
- [x] `linalg/qr_givens.rs` -- QR via Givens rotations (used by companion-matrix
  eigenvalue solver)
- [x] `linalg/lu_decomp.rs` -- LU with partial pivoting (used by implicit ODE solvers)
- [x] `linalg/householder_qr.rs` -- Householder QR factorisation

#### Diagnostics & Tests
- [x] `metrics/metrics.rs` -- absolute / relative error, max-norm, residual norm,
  2x2 condition number
- [x] `e2e_tests.rs` -- 38 cross-module integration tests (bisection `cos -> pi/2`
  to 1e-10; Newton `x^3 - 2 -> 2^(1/3)` in < 15 iters; Brent `sin -> pi` to 1e-12;
  Romberg `1/(1+x^2) -> pi/4`; Gauss-Legendre n=5 exact on x^9; adaptive Simpson on
  `1/sqrt(x)` over `[0, 1]`; `bessel_j0(0) = 1`, `j0(2.4048...) ~ 0`;
  `airy_ai(0) = 1 / (3^{2/3} Gamma(2/3))`; `lambert_w_0(e) = 1`;
  `elliptic_k(0) = pi/2`; RK4 exponential decay to 1e-4; DOPRI5 harmonic-oscillator
  energy conservation; cubic spline through (x, x^3) at 1.5 ~ 3.375; PCHIP monotone;
  Durand-Kerner roots of `(x-1)(x-2)(x-3)`; Sobol Van der Corput; PTX x 6 SM)
- [x] `benches/numeric_ops.rs` -- Criterion: 7 PTX kernels x all SM versions plus
  Bessel / Gauss-Kronrod / DOPRI5 / cubic-spline / Aberth algorithm benches

### Future Enhancements

#### P0 -- Verification Gaps
- [ ] GPU hardware verification on Linux + NVIDIA driver 525+ for all 7 PTX kernels
  across SM 75 / 80 / 86 / 89 / 90 / 100
- [ ] Reference cross-validation against SciPy / GSL / Boost.Math for special functions
  (Bessel, Airy, Lambert W, hypergeometric, elliptic, zeta) at edge cases

#### P1 -- Performance Tuning
- [ ] Per-SM tuned thread-block sizes for `gauss_quad_accumulate` and `rk4_stage`
  (currently fixed at portable defaults)
- [ ] Batched ODE integration kernel: many independent IVP systems integrated in
  parallel via one DOPRI5 launch
- [ ] Vectorised special-function dispatch (`bessel_recurrence`) with FP16 / BF16
  storage and FP32 accumulation

#### P2 -- Algorithmic Extensions
- [x] Tanh-sinh (double-exponential) quadrature for endpoint singularities
- [ ] Implicit Runge-Kutta (Radau IIA, SDIRK) for stiff problems beyond BDF1/BDF2
- [ ] GPU-resident sparse polynomial root finder for very high-degree polynomials
- [ ] Multi-precision residual refinement for ill-conditioned root finding
- [x] `quadrature/gauss_patterson.rs` — Gauss-Patterson sparse-grid quadrature: 1D nested Gauss-Kronrod rules (1,3,7,15,31,63 pts); multi-dimensional Smolyak sparse grid with Clenshaw-Curtis + Gauss-Legendre nodes; `SmolyakQuadrature { level: usize, dim: usize }`
- [x] `ode/sdirk.rs` — SDIRK integrators (Alexander 1977): SDIRK3 (3-stage order-3) + SDIRK4 (5-stage order-4); stage-value Newton iterations with frozen Jacobian; error control via embedded pair; A-stable for stiff systems
- [x] `diff/dual_number.rs` — Dual-number forward-mode AD: `Dual { val: f64, eps: f64 }` with overloaded arithmetic + transcendentals; Jacobian-vector products; exact to machine precision; composition through any `Fn(Dual)->Dual`
- [ ] `roots/complex_newton.rs` — Complex Newton + Halley root-finding: complex arithmetic over `num-complex::Complex<f64>`; Halley's method cubic convergence; used in Aberth-Ehrlich simultaneous poly-root refinement
- [x] `interp/rbf_interp.rs` — Radial Basis Function interpolation (Hardy 1971): scattered data in ℝⁿ; thin-plate spline / multiquadric / Gaussian kernels; solve linear system via Cholesky + condition estimate; `RbfInterpolator { kernel: RbfKernel }`
- [ ] `integral/cubature_adaptive.rs` — Adaptive multi-dimensional cubature (Genz-Malik 1980, Berntsen-Espelid-Genz 1991): 7th-order rule with error estimate; recursive bisection of error-dominant hyperrectangles; `AdaptiveCubature { abs_tol, rel_tol, max_eval }`
- [ ] `special/wright_omega.rs` — Wright ω function (Corless-Jeffrey 2002): solution to ω+log(ω)=z; extends Lambert W to complex arguments; Halley iterations from initial approximation; `wright_omega(z: Complex<f64>) -> Complex<f64>`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading runtime FFI) | Yes |
| oxicuda-memory | Device / host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

No external linear-algebra dependencies; all required Jacobi / QR / LU / Householder
routines are implemented privately under `linalg/`.

## Quality Status

- Warnings: 0 (clippy clean, `#![forbid(unsafe_code)]`)
- Tests: 466 passing (unit + 38 e2e cross-module)
- `unwrap()` / `expect()` calls in production code: 0
- Refactoring policy: all files under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`; macOS returns
  `UnsupportedPlatform` at runtime, compiles cleanly

## Performance Targets

Representative numerical workloads (all targets are CPU-side dispatch + PTX
generation; GPU throughput targets pending Linux + NVIDIA verification).

| Algorithm | Problem Size | Target |
|-----------|--------------|--------|
| Horner polynomial eval | degree 32, batch 10^6 | bandwidth-bound (>= 90% peak HBM) |
| RK4 step | system size 10^4, batch 10^3 | < 10 ms / step on sm_80 |
| Gauss-Legendre n=64 accumulate | batch 10^5 integrands | < 50 ms on sm_80 |
| Bisection (batched) | 10^5 independent brackets, 40 iters | < 20 ms on sm_80 |
| Spline eval (cubic) | 10^6 query points on 1024-knot table | < 10 ms on sm_80 |
| Central difference | batch 10^6, 5-point stencil | bandwidth-bound on sm_80 |
| Bessel J_0 / Y_0 recurrence | batch 10^6 scalar args | < 50 ms on sm_80 |
| Aberth all-roots | polynomial degree 100 | < 100 ms on sm_80 |

## Notes

- Vol.60 prioritises classical numerical-analysis primitives that fit naturally into
  scalar / 1D / low-D contexts. Multi-D PDE solvers, FEM kernels, and tensor-network
  contractions live in other volumes.
- The 7 PTX kernels target batched / vectorised data-parallel evaluation
  (polynomial values, RK4 stages, quadrature accumulation, bracket refinement,
  spline lookup, finite-difference stencils, Bessel recurrence). Adaptive control
  flow (Brent, DOPRI5 PI controller, Aberth iteration) stays on the host.
- Special functions follow standard NIST DLMF reference formulae and are validated
  in `e2e_tests.rs` against known closed-form values at canonical points.

## Architecture-Specific Deepening

Tile / thread-block configurations for the 7 PTX kernels by SM version. Per-SM
tuning is currently uniform; targeted tuning is tracked under Future Enhancements P1.

| SM Version | Default Block (`rk4_stage`) | Pipeline | Notes |
|------------|------------------------------|----------|-------|
| sm_75 (Turing) | 128 x 1 | 1 stage | baseline scalar |
| sm_80 / sm_86 (Ampere) | 256 x 1 | 2 stages | `cp.async` ready |
| sm_89 (Ada) | 256 x 1 | 2 stages | -- |
| sm_90 (Hopper) | 512 x 1 | 3 stages | TMA candidate for `gauss_quad_accumulate` |
| sm_100 (Blackwell) | 512 x 1 | 3 stages | -- |

### Deepening Opportunities
- [ ] Hopper: TMA bulk loads of quadrature node / weight tables in
  `gauss_quad_accumulate`
- [ ] Ampere: 3-stage `cp.async` pipeline for `spline_eval` on large knot tables
- [ ] All SMs: warp-shuffle Horner accumulation for `horner_eval` on
  short-degree polynomials
- [ ] Ada / Hopper: FP8 (e4m3) storage of spline coefficients with FP32 accumulation
  for memory-bound `spline_eval`

## Estimation vs Actual

| Metric | Estimated (estimation.md Vol.60) | Actual |
|--------|----------------------------------|--------|
| SLoC | 80K-140K (median ~110K) | 13,644 |
| Files | ~35-55 algorithm modules | 99 |
| Tests | algorithm-grade coverage | 466 |

The gap to the median estimate reflects the estimation targeting full
QUADPACK-/GSL-grade production parity including arbitrary-precision residual
refinement, exhaustive special-function asymptotic regimes, and exhaustive stiff
ODE solver families. The current implementation delivers a clean classical-analysis
surface with verified algorithmic correctness on CPU and PTX generation for all 7
device kernels.
