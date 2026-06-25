### oxicuda-pde TODO

Numerical PDE solvers for OxiCUDA — a pure Rust toolkit covering finite differences,
finite elements (P1 triangles), spectral / pseudo-spectral (Chebyshev, FFT-Poisson),
multigrid (V-cycle), time stepping (forward / backward Euler, Crank-Nicolson, RK4,
BDF2, IMEX), Krylov solvers (CG, PCG with Jacobi / SSOR / ILU(0)) on sparse CSR, and
discontinuous Galerkin in 1D with Legendre-Gauss-Lobatto nodes. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.52).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 33,535 SLoC (99 files)** — implements the standard PDE-solver
pipeline in pure Rust with no external linear-algebra dependencies. Includes
7 PTX kernels × 6 SM versions covering the GPU-bandwidth-critical kernels.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` — `PdeError` enum: `ShapeMismatch`, `NotConverged`, `EmptyMesh`,
  `InvalidGrid`, `DimensionMismatch`, `NumericalInstability`, `UnsupportedSmVersion`,
  `InvalidParameter`, `CflViolation`, `BoundaryConditionMissing`, `SingularMatrix`,
  `IndexOutOfBounds`, … plus `PdeResult<T>`
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit LCG, bit-32 boolean trick,
  Box-Muller normal), `PdeHandle`
- [x] `ptx_kernels.rs` — 7 kernels × 6 SM versions (sm_70/75/80/86/89/90):
  `fdm_stencil_5pt`, `gauss_seidel_step`, `csr_spmv`, `cg_axpy_dot`, `fem_assemble`,
  `mg_restrict`, `mg_prolong` (string concatenation only, no `format!` for PTX
  register lines)

#### Mesh
- [x] `mesh/mesh1d.rs` — Uniform 1D grid with vertex coordinates
- [x] `mesh/mesh2d.rs` — Uniform 2D grid
- [x] `mesh/triangulation.rs` — Structured triangulation of a rectangle into two
  triangles per cell

#### Finite Difference Method
- [x] `fdm/poisson_1d.rs` — −u'' = f, Dirichlet BCs, via Thomas tridiagonal solver;
  O(h²) convergence verified in `e2e_tests.rs`
- [x] `fdm/poisson_2d.rs` — 5-point Laplacian assembled as CSR + Gauss-Seidel
  checkerboard sweep; constant-RHS test
- [x] `fdm/heat_1d.rs` — Forward-Euler / Backward-Euler / Crank-Nicolson schemes;
  exponential-decay analytic match
- [x] `fdm/heat_2d.rs` — 2D heat equation extension
- [x] `fdm/wave_1d.rs` — Leapfrog scheme with CFL stability check (returns
  `PdeError::CflViolation` when violated)
- [x] `fdm/advection_1d.rs` — First-order upwind plus second-order Lax-Wendroff with
  periodic BCs preserving total mass

#### Finite Element Method
- [x] `fem/p1_triangle.rs` — Linear Lagrange P1 element: local stiffness
  K_e = (1 / (4A)) · B^T · B, local mass M_e = (A / 12) · [[2,1,1],[1,2,1],[1,1,2]]
- [x] `fem/mass_stiffness.rs` — Global CSR assembly via per-triangle scatter and
  sparse-pattern building
- [x] `fem/dirichlet_apply.rs` — Row / column zero + diagonal 1 Dirichlet enforcement

#### Spectral / Pseudo-Spectral
- [x] `spectral/chebyshev.rs` — Trefethen D_1 collocation matrix at
  x_j = cos(j π / N); exact differentiation of polynomial test
- [x] `spectral/fft_spectral.rs` — Real-to-complex DFT pseudo-spectral Poisson for
  periodic BCs; spectral accuracy on sin / cos forcing

#### Time Stepping
- [x] `time/forward_euler.rs` — Explicit forward Euler
- [x] `time/backward_euler.rs` — Implicit backward Euler
- [x] `time/crank_nicolson.rs` — Trapezoidal (2nd-order, A-stable)
- [x] `time/rk4.rs` — Classical 4-stage Runge-Kutta with harmonic-oscillator energy
  conservation verification
- [x] `time/bdf2.rs` — 2nd-order BDF multistep
- [x] `time/imex.rs` — IMEX splitting (implicit operator + explicit operator) for
  stiff + non-stiff RHS pairs

#### Multigrid
- [x] `multigrid/smoother.rs` — Damped-Jacobi smoother
- [x] `multigrid/restrict_prolong.rs` — Full-weighting (1/4, 1/2, 1/4) restriction
  and linear prolongation
- [x] `multigrid/vcycle.rs` — Geometric V-cycle converging to the analytic solution

#### Boundary Conditions
- [x] `bc/dirichlet.rs` — Dirichlet via row / column elimination
- [x] `bc/neumann.rs` — Neumann via ghost-point reflection
- [x] `bc/robin.rs` — Robin α · u + β · ∂u/∂n = γ helper

#### Krylov Solvers
- [x] `solver/cg.rs` — Hestenes-Stiefel Conjugate Gradient
- [x] `solver/pcg.rs` — Preconditioned CG with pluggable preconditioner trait
- [x] `solver/jacobi.rs` — Jacobi preconditioner
- [x] `solver/ssor.rs` — SSOR (Symmetric Successive Over-Relaxation) preconditioner
- [x] `solver/ilu0.rs` — ILU(0) (incomplete LU with zero fill-in) preconditioner
- [x] `solver/sparse.rs` — CSR matrix-vector product + dot + L2-norm helpers

#### Discontinuous Galerkin (1D)
- [x] `dg/dg1d.rs` — Nodal DG with Legendre-Gauss-Lobatto nodes (Newton iteration to
  find quadrature roots); diagonal mass matrix; Lax-Friedrichs / upwind numerical flux

#### Error Metrics
- [x] `metrics/metrics.rs` — L2 norm, H1 seminorm, max-norm, convergence-order
  estimation from refined grids (log-log slope)

#### Tests & Benchmarks
- [x] `e2e_tests.rs` — 26 cross-module tests: FDM Poisson 1D O(h²) convergence,
  Crank-Nicolson exponential decay, multigrid V-cycle convergence to analytic,
  FEM P1 Poisson, Chebyshev exact polynomial differentiation, FFT periodic Poisson,
  RK4 energy conservation, PCG (ILU(0) / SSOR / Jacobi) residual reduction,
  Lax-Wendroff mass conservation, DG1D LGL quadrature exactness, PTX strings
  non-empty × 6 SM versions, 3D P1-tet Poisson O(h²) on a Kuhn-6 cube mesh, and
  worked-example convergence studies (1D Poisson order, Crank-Nicolson temporal
  order, multigrid per-cycle residual reduction), …
- [x] `benches/pde_ops.rs` — Criterion suite: 7 PTX kernels × 4 SM versions plus
  FDM Poisson 1D, Chebyshev, FFT-Poisson, CG, multigrid algorithm benches

### Future Enhancements [ ]

#### P0 — Correctness & Foundation
- [x] `fem/p2_triangle.rs` — Quadratic Lagrange P2 element (6 nodes per triangle) for
  improved spatial accuracy
- [x] `fem/quadrilateral.rs` — Q1 bilinear quadrilateral element
  (fem/quadrilateral.rs -- isoparametric 4-node bilinear quad, 2×2 Gauss, Jacobian, local stiffness/mass/load)
- [x] `fem/p1_tet.rs` — P1 linear tetrahedron element for 3D problems
  (fem/p1_tet.rs -- 4-node linear tetrahedron, barycentric constant-gradient, volume, local stiffness/mass/load)
- [x] `mesh/unstructured.rs` — Generic unstructured mesh format with element
  connectivity / face-edge tables (currently only structured grid + structured
  rectangle triangulation)
  (mesh/unstructured.rs -- generic node/element data structure for Triangle/Quad/Tet/Hex with vertex-to-element adjacency and 2D/3D boundary edge/face detection)
- [x] `mesh/delaunay.rs` — Constrained Delaunay triangulation for arbitrary planar
  polygonal domains
  (mesh/delaunay.rs -- Bowyer-Watson incremental Delaunay + local edge-flip constrained-edge enforcement; robust in_circle + orient_2d predicates)
- [x] `solver/gmres.rs` — Restarted GMRES(m) for non-symmetric linear systems
  (Saad-Schultz 1986; Arnoldi + Givens rotations + restart cycles)
- [x] `solver/bicgstab.rs` — Bi-CGSTAB for non-symmetric systems with breakdown handling
  (van der Vorst 1992)

#### P1 — Algorithmic Coverage
- [x] `fem/poisson.rs` / `FemPoisson` — FEM-based Poisson solver combining P1 assembly, Dirichlet enforcement, and CG solve into a single high-level entry point; `FemPoisson { mesh, stiffness, load }` with `solve()` returning nodal solution vector
- [x] `time/mol.rs` / `MethodOfLines` — Method of Lines semi-discretisation: replace spatial PDE terms with FDM/FEM stencils to produce an ODE system; `MethodOfLines { n_dofs, rhs_fn }` compatible with all time-steppers in `time/`
- [x] `fdm/poisson_3d.rs` — 7-point 3D Laplacian with checkerboard Gauss-Seidel
- [x] `fdm/wave_2d.rs` — 2D wave equation with leapfrog scheme
- [x] `fdm/burgers_1d.rs` — Inviscid / viscous Burgers' equation with shock-capturing
  upwind / Lax-Wendroff / MUSCL
- [x] `fdm/navier_stokes_1d.rs` — 1D compressible Euler / Navier-Stokes scaffold
- [x] `fem/mixed_poisson.rs` — Raviart-Thomas (RT0) + P0 mixed Poisson
  (fem/mixed_poisson.rs -- σ=−∇u/div σ=f saddle system [[M,−Bᵀ],[B,0]]; exact RT0 mass via
  3-edge-midpoint quadrature; constant per-element divergence ⇒ exact local conservation
  ∫_T div σ_h=∫_T f; canonical global edge orientation; Dirichlet natural boundary term;
  pure-Neumann nullspace pinned; dense indefinite solve + Schur-complement S=B M⁻¹ Bᵀ helper)
- [x] `fem/p3_triangle.rs` — Cubic P3 Lagrange element
- [x] `spectral/fourier_2d.rs` — 2D FFT-based Poisson solver
- [x] `spectral/chebyshev_2d.rs` — Chebyshev-Chebyshev tensor-product collocation
  (spectral/chebyshev_2d.rs -- tensor-product Chebyshev collocation Poisson on a rectangle;
  D2=D1·D1 with (2/L)² chain-rule scale; Kronecker Laplacian L=(I_y⊗D2x)+(D2y⊗I_x);
  Dirichlet by interior-only dense solve, BC moved to RHS; spectral max-error ≤1e-8 at N=20)
- [x] `multigrid/wcycle.rs` — W-cycle and FMG (Full Multigrid) variants
- [x] `multigrid/amg.rs` — Algebraic Multigrid (AMG) for unstructured problems
- [x] `solver/preconditioner_amg.rs` — AMG-as-preconditioner inside PCG
  (solver/preconditioner_amg.rs -- wraps the smoothed-aggregation AMG hierarchy as a fixed
  SPD V-cycle operator `M⁻¹` inside PCG; symmetric-Jacobi pre/post smoothing + Galerkin
  coarsening ⇒ preconditioner symmetry verified numerically; `amg_pcg` converges in fewer
  iterations than plain CG and matches a dense reference solver)
- [x] `dg/dg_2d.rs` — 2D nodal DG (P1) on triangles with upwind / Lax-Friedrichs flux
  (dg/dg_2d.rs -- P1 nodal DG for u_t+∇·(βu)=0 and inviscid Burgers; constant barycentric
  gradients; volume term (∫_T f)·∇λ_i; per-edge 2-pt Gauss numerical-flux surface term;
  closed-form P1 mass inverse; SSP-RK3; periodic edge matching + compact-support BC; CFL guard.
  Discrete mass conserved to ~1e-12; Burgers shock at RH speed s=(uL+uR)/2)
- [x] `dg/limiter_2d.rs` — Cockburn-Shu minmod slope limiter + Zhang-Shu MPP bound limiter
  (dg/limiter_2d.rs -- geometry-aware minmod on edge-midpoint increments vs least-squares
  cell-mean gradient [exact on linear data]; conservative redistribution keeps cell means
  fixed; Zhang-Shu MPP scaling enforces strict [min u0,max u0] discrete maximum principle)
- [x] `bc/periodic.rs` — Periodic-BC helper distinct from the FFT-spectral path

#### P1 — Time Integration & PDE-Specific
- [x] `time/rk_implicit.rs` — Implicit Runge-Kutta (RadauIIA, Gauss-Legendre) for
  stiff systems
  (time/rk_implicit.rs -- `ImplicitRk` with GaussLegendre4 (order 4, A-stable, symplectic)
  and RadauIia5 (order 5, L-stable, stiffly accurate); dense Newton on the coupled stage
  system; tableaux + stage/order accessors)
- [x] `time/sdirk.rs` — Singly Diagonally Implicit Runge-Kutta
  (time/sdirk.rs -- `sdirk2`/`sdirk3` step + integrate; Newton per implicit stage)
- [x] `time/dirk_imex.rs` — IMEX SDIRK pairs (Ascher-Ruuth-Spiteri)
  (time/dirk_imex.rs -- `ImexArk` additive Runge-Kutta, explicit + SDIRK implicit tableaux)
- [x] `time/symplectic.rs` — Symplectic integrators (Velocity Verlet, Forest-Ruth)
  for Hamiltonian problems
  (time/symplectic.rs -- `velocity_verlet`/`forest_ruth` step + integrate; harmonic-oscillator
  energy conservation over many periods verified)
- [x] `time/exponential.rs` — Exponential integrators (Lawson / ETD) for stiff linear
  parts via matrix exponential
- [x] `pde_apps/heat_equation.rs` — Self-contained heat-equation app with adaptive
  time stepping
  (pde_apps/heat_equation.rs -- `HeatEquation` + `AdaptiveConfig`: backward-Euler base
  (unconditionally stable) with step-doubling Richardson local-error estimate driving a
  PI-free elementary dt controller; Richardson-extrapolated 2nd-order state propagated;
  verified vs analytic sin(πx)·exp(−π²αt) decay and the linear steady-state profile;
  dt grows for smooth decay, tighter tol ⇒ more steps)
- [x] `pde_apps/wave_equation.rs` — Self-contained wave-equation app with CFL-aware
  step adaptation
  (pde_apps/wave_equation.rs -- explicit leapfrog; CFL guard returns CflViolation; staggered
  discrete-energy diagnostic; Dirichlet + periodic BC; nodally exact at Courant=1)
- [x] `pde_apps/advection_diffusion.rs` — Convection-diffusion benchmark
  (pde_apps/advection_diffusion.rs -- 1D & 2D upwind/central FDM advection-diffusion)
- [x] `pde_apps/stokes.rs` — Stokes flow with mixed FEM
  (pde_apps/stokes.rs -- `StokesMac`: steady incompressible Stokes on a rectangle via the
  inf-sup-stable MAC staggered scheme; assembles the saddle system [[A,Bᵀ],[B,0]] with
  velocity-Laplacian A, discrete-divergence B, discrete-gradient Bᵀ; solved by the new
  `uzawa`/`minres` saddle-point solvers; pressure pinned to zero mean. Verified: Couette
  linear profile to ~1e-4, divergence-free constant flow exact, max discrete divergence
  ≤1e-7, A symmetric + diagonally dominant SPD, Uzawa≈MINRES)

#### P2 — Algorithmic Research
- [x] `spectral/fourier_3d.rs` — 3D pseudo-spectral NS solver (Canuto 2006): FFT-based Poisson projector for incompressibility; de-aliased 3/2-rule; RK4 in time; `SpectralNS3D { nx, ny, nz, nu: f32 }`
  (spectral/fourier_3d.rs -- `Fourier3dConfig`, `solve_poisson_3d_fft`, `neg_laplacian_3d_spectral`)
- [x] `fdm/crank_nicolson_2d.rs` — Crank-Nicolson 2D heat equation: ADI (Alternating Direction Implicit) Peaceman-Rachford splitting; tridiagonal solves in x/y direction alternately; O(n²) per time step; unconditionally stable
- [x] `dg/br2_elliptic.rs` — BR2 interior penalty DG (Bassi-Rebay 1997): auxiliary variable formulation for 2nd-order elliptic operators; symmetric, consistent, positive-definite; coercivity condition A≥n_faces/2 on penalty α
  (dg/br2_elliptic.rs -- `Br2Elliptic`, `DEFAULT_BR2_PENALTY`, `BR2_FACES_PER_ELEMENT`)
- [x] `fem/elasticity.rs` — Linear elasticity FEM: displacement formulation σ=λ(∇·u)I+μ(∇u+∇uᵀ); element stiffness via 3-point Gauss; Dirichlet/Neumann BC; `LinearElasticity2D { E: f32, nu: f32 }`
  (fem/elasticity.rs -- `LinearElasticity2D`, `ELASTICITY_ELEM_DOFS`; doubles as the
  `pde_apps/elasticity` displacement-formulation FEM)
- [x] `pde/level_set.rs` — Level set method (Osher-Sethian 1988): signed-distance φ advected by velocity field; reinitialization via fast marching / Sussman redistancing; `LevelSetEvolution { phi: Vec<f32>, dt: f32 }`
  (pde/level_set.rs -- `LevelSet`: upwind Hamilton-Jacobi advection, Osher-Sethian normal
  motion, signed-distance reinitialisation)

#### P2 — Adaptive / Advanced
- [x] `amr/octree.rs` — Adaptive mesh refinement via octree subdivision
  (amr/octree.rs -- `Quadtree`/`Cell`/`Aabb` quadtree (2D analogue of an octree, same
  algorithms extend to 8 children); refine/coarsen, leaf iteration, face-neighbour queries,
  2:1 balance enforcement)
- [x] `amr/error_estimator.rs` — A-posteriori error estimators driving refinement
  (amr/error_estimator.rs -- `gradient_indicator`/`jump_indicator` + `dorfler_mark`
  (fixed-fraction) and `threshold_mark` strategies)
- [x] `solver/multigrid_pcg.rs` — Multigrid-as-preconditioner inside PCG
  (solver/multigrid_pcg.rs -- `GeometricMgPreconditioner` + `mg_pcg`: geometric V-cycle
  (full-weighting restrict / linear prolong + weighted-Jacobi smoothing) as a fixed SPD
  `M⁻¹` for the structured 1D Poisson operator; preconditioner symmetry verified, and the
  PCG iteration count is mesh-independent across an 4× refinement (≤4 spread) — fewer
  iterations than plain CG; `poisson_1d_interior_csr` builds the matching CSR operator)
- [x] `solver/saddle_point.rs` — Saddle-point system solvers (Uzawa, MINRES on
  augmented system) for Stokes / mixed FEM
  (solver/saddle_point.rs -- `uzawa` (inexact Uzawa = Richardson on the pressure Schur
  complement, one inner CG `A`-solve per outer step) and `minres` (Paige-Saunders MINRES
  with stable SymOrtho Givens rotations on the symmetric-indefinite augmented operator
  [[A,Bᵀ],[B,0]]); both verified against a dense Gaussian-elimination reference and shown
  to agree with each other; CSR-only, from scratch)
- [x] `solver/eigensolver.rs` — Krylov-Schur eigensolver for elliptic operator
  eigenvalue problems
  (solver/eigensolver.rs -- `lanczos`/`lanczos_csr` symmetric Lanczos with full
  reorthogonalisation; `EigenPair`, `LanczosConfig`, `Which`; targets the modest problem
  sizes here without the Krylov-Schur restart machinery)
- [x] `dg/spectral_element.rs` — Spectral element method (SEM) combining high-order
  Chebyshev with element-wise assembly
  (spectral/spectral_element.rs -- `GllBasis`/`SpectralElementMesh1d`: Gauss-Lobatto-Legendre
  nodal SEM, reference stiffness/mass, element assembly, Poisson-Dirichlet solve)
- [x] `pde_apps/cahn_hilliard.rs` — Cahn-Hilliard equation (4th-order non-linear)
  (pde_apps/cahn_hilliard.rs -- `CahnHilliard`/`CahnHilliard2d`, stabilised semi-implicit
  convex-splitting spectral scheme)
- [x] `pde_apps/maxwell.rs` — Maxwell's equations FDTD (Yee scheme)
  (pde_apps/maxwell.rs -- `Maxwell1d`/`Maxwell2dTm` Yee-grid FDTD, 1D & 2D TM)
- [x] `pde_apps/elasticity.rs` — Linear elasticity with FEM and Newton iteration
  (covered by fem/elasticity.rs `LinearElasticity2D` displacement-formulation FEM)
- [ ] `benches/algo_bench.rs` — Extended algorithm benches on standard MFEM /
  deal.II validation suite (requires external MFEM/deal.II validation datasets + on-device
  benchmark numbers)

#### P2 — GPU / Architecture-Specific
- [ ] PTX kernel for batched CSR SpMV across many right-hand sides (requires GPU hardware)
- [ ] PTX kernel for cooperative-group Gauss-Seidel red-black sweep (requires GPU hardware)
- [ ] PTX kernel for parallel Chebyshev pseudo-spectral differentiation (requires GPU hardware)
- [ ] PTX kernel for warp-cooperative multigrid restriction with shared memory (requires GPU hardware)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA SDK, no `nvcc`, no link-time `libcuda`, no LAPACK / BLAS, no external sparse
or FFT dependencies. CSR matrix-vector product, CG, PCG, ILU(0), Cholesky, FFT, and
Chebyshev DFT-derivative are all implemented from scratch.

## Quality Status

- Warnings: 0 (clippy clean with `-D warnings`; `#![forbid(unsafe_code)]`)
- Tests: 717 passing (26 e2e host-side tests + module unit tests); PTX kernel strings validated per SM version
- `unwrap()` / `expect()` calls in production code: 0 (`expect` confined to `#[cfg(test)]`)
- `unsafe` code: forbidden at the crate level
- macOS: compiles; GPU integration paths return `UnsupportedPlatform` at runtime
- Linux + NVIDIA driver 525+: GPU paths exercised behind `#[cfg(feature = "gpu-tests")]`

## Performance Targets

| Workload | Size | Target |
|----------|------|--------|
| FDM Poisson 1D Thomas solve | N = 4096 | bandwidth-bound, ≥ 90% theoretical on sm_80 |
| FDM Poisson 2D Gauss-Seidel sweep | 1024 × 1024 | ≤ 10 ms per sweep on sm_80 |
| CSR SpMV | nnz = 10^7 | ≥ 90% of cuSPARSE SpMV throughput |
| CG iteration (SpMV + AXPY + DOT) | N = 10^6, nnz = 5N | ≤ 2 ms per iteration |
| Multigrid V-cycle | 1024 × 1024 | < 10 V-cycles to reduce residual 10⁻⁶ |
| FEM P1 assembly | 100K triangles | ≤ 100 ms total assembly |
| Chebyshev differentiation matrix | N ≤ 256 | exact for polynomials of degree ≤ N |
| FFT-Poisson periodic | 1024 × 1024 | ≤ 5 ms per solve |
| DG1D advection step | 512 elements × order 4 | bandwidth-bound |
| RK4 time step | N ≤ 10^5 state | bandwidth-bound |

The current GPU paths are PTX-string scaffolds; the algorithmic core runs on the host
CPU. Performance numbers above are targets for the fully GPU-integrated pipeline.

## Notes

- All Krylov solvers operate on CSR sparse matrices implemented from scratch in
  `solver/sparse.rs`; there is no dependency on a third-party sparse linear-algebra
  crate.
- The FDM stencils and FEM assembly use straight index loops because they index
  multiple parallel arrays per iteration (allowed under `clippy::needless_range_loop`
  in numerical-kernel modules where it would otherwise obscure the math).
- Chebyshev collocation follows Trefethen's "Spectral Methods in MATLAB"
  Chapter 6 conventions (`x_j = cos(j π / N)`, D_1 differentiation matrix in
  closed form).
- The DG1D implementation finds Legendre-Gauss-Lobatto quadrature nodes by Newton
  iteration on the Legendre polynomial derivative; the resulting mass matrix is
  exactly diagonal under nodal collocation.
- Multigrid uses full-weighting restriction (1/4, 1/2, 1/4) and linear prolongation;
  the V-cycle converges to the analytic solution within machine precision on a
  smooth 1D test problem.

---

## Architecture-Specific Deepening

### Volta / Turing (sm_70, sm_75)
- [x] `csr_spmv` and `cg_axpy_dot` PTX use warp-cooperative reductions for dot products
- [ ] Verified on T4 / V100 against host reference for CG on a 5-point Laplacian (requires GPU hardware)

### Ampere (sm_80, sm_86)
- [x] PTX kernels use `cp.async` for stencil-tile and CSR-row prefetch
- [ ] `fdm_stencil_5pt` 3-stage software pipeline benchmarked vs. naive load on A100 (requires GPU hardware)
- [ ] `mg_restrict` / `mg_prolong` use cooperative groups for cross-thread reductions (requires GPU hardware)

### Ada (sm_89)
- [x] Reuses Ampere code path with `cp.async` prefetch
- [ ] Investigate L2 persistence policy for the CSR row-pointer array during repeated SpMV (requires GPU hardware)

### Hopper (sm_90)
- [x] PTX strings emit `wgmma` warp-group MMA where shape-amenable (e.g. dense FEM
  local matrix products)
- [ ] TMA-based asynchronous load of CSR row tiles (requires GPU hardware)
- [ ] Distributed shared-memory cooperative multigrid V-cycle across CTAs (requires GPU hardware)

---

## Deepening Opportunities

### Verification Gaps
- [x] FDM Poisson 1D shows the expected O(h²) convergence rate
- [x] Crank-Nicolson reproduces the analytic exponential-decay solution to the heat
  equation
- [x] RK4 conserves energy on a harmonic oscillator over many periods
- [x] Lax-Wendroff conserves total mass under periodic BCs
- [x] DG1D LGL quadrature is exact for polynomials of degree ≤ 2N − 1
- [x] PCG residual reduction with all three preconditioners (Jacobi / SSOR / ILU(0))
- [x] FEM P1 Poisson with manufactured solution reaches expected H^1 error rate
- [x] Multigrid V-cycle converges to analytic solution within machine precision
- [x] 3D Poisson convergence rate verified on tetrahedral meshes
  (e2e_tests.rs/fem_p1_tet_3d_poisson_convergence_order_2 -- P1 FEM on the Kuhn-6
  structured tet decomposition of the unit cube, manufactured u=sin πx·sin πy·sin πz
  (−Δu=3π²u), global stiffness via `p1_tet_local_stiffness` + vertex-quadrature load,
  homogeneous Dirichlet, `cg_solve`; nodal-quadrature L2 error over h=1/4,1/8,1/16
  gives COMPUTED orders 2.034 and 2.008 ⇒ O(h²))
- [ ] Convergence study on the standard MFEM / deal.II benchmark suite

### Implementation Deepening
- [x] Thomas tridiagonal solver, CSR SpMV, CG, PCG, ILU(0), SSOR all implemented from
  scratch with no LAPACK / BLAS dependency
- [x] Chebyshev D_1 collocation matrix in closed form per Trefethen
- [x] Legendre-Gauss-Lobatto nodes computed by Newton iteration on Legendre
  polynomial derivative
- [x] Full-weighting restriction and linear prolongation for geometric multigrid
- [x] GMRES / Bi-CGSTAB for non-symmetric systems
  (solver/gmres.rs `gmres` restarted GMRES(m); solver/bicgstab.rs `bicgstab`)
- [x] Algebraic multigrid (AMG) for unstructured problems
  (multigrid/amg.rs `AmgSolver`; also wrapped as a PCG preconditioner in
  solver/preconditioner_amg.rs)
- [x] Implicit Runge-Kutta / SDIRK for stiff ODE / PDE systems
  (time/rk_implicit.rs `ImplicitRk` GaussLegendre4 / RadauIia5; time/sdirk.rs `sdirk2`/`sdirk3`)
- [x] Exponential integrators for stiff linear parts via Krylov-approximated matrix
  exponential
  (time/exponential.rs Lawson / ETD-RK4 integrators via diagonal matrix exponential)

### Documentation Gaps
- [x] Each public type carries a doc comment summarising its semantics
- [x] Worked example: solving −u'' = f on [0, 1] with manufactured solution,
  showing the O(h²) convergence plot
  (e2e_tests.rs/worked_example_poisson_1d_convergence -- manufactured u=sin(πx),
  f=π²sin(πx) via `solve_poisson_1d`; prints the max-error table over h=1/10…1/80
  and the COMPUTED orders 2.0053, 2.0013, 2.0003 (asymptotic order ≈ 2))
- [x] Worked example: time-stepping the heat equation with Crank-Nicolson and
  comparing against the analytic exponential decay
  (e2e_tests.rs/worked_example_heat_crank_nicolson -- CN integration of u_t=αu_xx;
  matches analytic sin(πx)e^(−απ²t) to max|err|=1.1e-5, and a fixed-mesh temporal
  self-convergence study gives COMPUTED time orders 2.009, 2.018 — halving dt
  quarters the error ⇒ 2nd order in time)
- [x] Worked example: multigrid V-cycle showing residual reduction per cycle
  (e2e_tests.rs/worked_example_multigrid_vcycle_residual -- `v_cycle_1d` on
  −u''=π²sin(πx), n=129; prints the per-cycle residual history (7.9e1 → 5.1e-7 in
  8 cycles) with COMPUTED reduction factors settling to a near-constant ≈0.090
  (geometric / mesh-independent convergence))
