# oxicuda-pinn TODO

Pure-Rust physics-informed scientific ML library covering: forward-mode autodiff
(dual numbers, MultiDual), tape-based reverse-mode AD (Wengert list), PINN
losses (residual / boundary / IC + NTK adaptive weighting), Neural ODEs (Euler
/ Heun / RK4 / Dopri45 + continuous adjoint method + CNF + latent-ODE), neural
operators (FNO 1D / 2D, DeepONet, MWT, GNO), PDE templates (heat / wave /
Burgers / Poisson / Navier-Stokes), coordinate-based MLP / SIREN networks, and
adaptive collocation sampling (residual-adaptive / LHS / Halton). Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.31).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 18,135 SLoC (59 files)** -- 624 unit tests + 12 E2E integration tests

The crate is the densest single PINN / scientific-ML library in the OxiCUDA
ecosystem: forward + reverse AD, four ODE solvers, four neural operator
families, five PDE templates, and three adaptive samplers. The crate is
`forbid(unsafe_code)`.

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `PinnError` (16 variants: DimensionMismatch, EmptyInput,
      InvalidStepSize, InvalidTimeInterval, NanEncountered,
      InvalidGridResolution, TooManyFourierModes, InvalidLayerWidth,
      InvalidNetworkDepth, InvalidWeight, InvalidActivation, SolverDivergence,
      EmptyCollocationSet, TapeIndexOutOfRange, InvalidPdeCoefficient,
      Internal); `PinnResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG, Box-Muller
      normals, `next_f32`, `next_usize`), `PinnHandle::default_handle()`
      (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- module exports + prelude + 12 E2E integration tests;
      `#![forbid(unsafe_code)]`

#### PTX Kernels (7 kernels x 6 SM versions = 42 generators)
- [x] `ptx_kernels.rs::pinn_residual_ptx` -- r^2 = r . r,
      `atom.global.add.f32` reduction for sum |F|^2
- [x] `ptx_kernels.rs::spectral_conv_ptx` -- complex multiply for FNO
      spectral convolution via `fma.rn.f32`
- [x] `ptx_kernels.rs::dual_op_ptx` -- dual-number multiply
      (a + eps * a') * (b + eps * b') = a*b + eps*(a'*b + a*b') via 4 x `fma.rn.f32`
- [x] `ptx_kernels.rs::adjoint_ode_ptx` -- reverse-time Euler step
      a[i] += h * dadt[i] for adjoint accumulation
- [x] `ptx_kernels.rs::branch_trunk_dot_ptx` -- DeepONet inner product with
      warp-shuffle `shfl.sync.bfly.b32` reduce
- [x] `ptx_kernels.rs::siren_forward_ptx` -- sin(omega_0 * (W*x + b)) SIREN
      layer via `sin.approx.f32`
- [x] `ptx_kernels.rs::lhs_sample_ptx` -- LCG per thread, cell-offset sample
      for Latin Hypercube Sampling
- [x] `ptx_kernels.rs::f32_hex` -- f32 to 0F-prefixed hex literal helper

#### Autodiff (autodiff/)
- [x] `dual.rs::Dual` -- forward-mode AD: dual numbers with all standard ops
      (sin / cos / exp / ln / sqrt / tanh / powi / abs + arithmetic) and
      chain rule
- [x] `tape.rs::Tape` / `Var` -- reverse-mode AD via index-based Wengert list;
      `gradient()` reverse pass; ops: add / sub / mul / div / sin / cos / exp
      / tanh / sq
- [x] `multidim.rs::MultiDual<N>` -- simultaneous N-variable partial
      derivatives; arithmetic and transcendental ops with product / chain rule
      on gradient arrays

#### PINN Losses (pinn_loss/)
- [x] `residual.rs::pde_residual_loss` / `compute_residuals` -- MSE over
      |F[u; theta](x_i)|^2; closure-based residual function
- [x] `boundary.rs::bc_loss` / `BcType` -- Dirichlet / NeumannX / NeumannY
      boundary condition loss
- [x] `initial.rs::ic_loss` -- initial condition MSE loss
- [x] `weighting.rs::AdaptiveWeights` -- NTK-style lambda update
      lambda_i <- alpha * lambda_i + (1 - alpha) / ||grad L_i||;
      `weighted_loss()` combiner

#### Neural ODE / SDE (neural_ode/)
- [x] `solvers.rs::euler_step` / `heun_step` / `rk4_step` / `dopri45_step` /
      `integrate_fixed` / `integrate_adaptive` -- Dormand-Prince RK4(5) with
      exact Butcher tableau coefficients; adaptive step control with
      0.9 * err^(-0.2) rescaling; `OdeRhsFn` type alias
- [x] `adjoint.rs::node_forward` / `node_adjoint_grad` -- continuous adjoint
      method: forward trajectory storage, reverse-time integration of
      a-dot = - a^T * df/dy, accumulation of dL/dtheta = - integral a^T * df/dtheta dt
- [x] `cnf.rs::cnf_forward` / `hutchinson_trace` / `dense_trace` -- Continuous
      Normalising Flow: log-density via Hutchinson Rademacher trace estimator
      or dense trace via dual numbers
- [x] `latent_ode.rs::LatentOde` / `LatentOdeConfig` -- encoder GRU ->
      reparametrise -> ODE on latent -> decoder MLP; Box-Muller normals via
      `LcgRng`

#### Neural Operators (neural_op/)
- [x] `fno.rs::Fno1d` / `Fno2d` / `dft_1d` / `idft_1d` -- Fourier Neural
      Operator: inline O(N^2) DFT (separable for 2D), spectral conv (complex
      multiply up to k_max modes), GeLU activation, lift / project layers;
      `Fno1dConfig`, `Fno2dConfig`
- [x] `deeponet.rs::DeepONet` / `DeepONetConfig` -- branch network (encodes
      function samples) x trunk network (encodes query coords) -> inner
      product output; batch forward
- [x] `mwt.rs::Mwt` / `MwtConfig` -- Multiwavelet Transform Operator via Haar
      wavelet decompose / reconstruct per-level with learnable kernel
- [x] `gno.rs::Gno` / `GnoConfig` -- Graph Neural Operator: radius-based
      neighbour aggregation with kernel MLP K(x_i - x_j; theta) * feat_j ->
      mean-pool

#### PDE Templates (pde/)
- [x] `heat.rs::heat_residual` / `heat_analytic` / `heat_residual_check` --
      1D heat: du/dt - alpha * d2u/dx2; analytic sin(pi*x) * exp(-alpha * pi^2 * t)
- [x] `wave.rs::wave_residual` / `wave_analytic` -- 1D wave:
      d2u/dt2 - c^2 * d2u/dx2; D'Alembert solution
- [x] `burgers.rs::burgers_residual` / `burgers_analytic` /
      `burgers_residual_check` -- 1D Burgers
      du/dt + u * du/dx - nu * d2u/dx2; viscous shock tanh solution
- [x] `poisson.rs::poisson_residual` / `poisson_analytic` -- 2D Poisson
      div(grad u) = f; f = -2 pi^2 sin(pi*x) sin(pi*y) -> u = sin(pi*x) sin(pi*y)
- [x] `navier_stokes.rs::ns_vorticity_residual` / `taylor_green_vortex` --
      2D NS vorticity form; Taylor-Green vortex
      omega = 2 cos(x) cos(y) exp(-2 nu t)

#### Networks (network/)
- [x] `mlp.rs::Mlp` / `Activation` / `MlpConfig` -- configurable MLP (tanh /
      sin / relu / gelu); SIREN initialisation (first layer U(-1/d, 1/d),
      hidden U(-sqrt(6/d) / omega_0, sqrt(6/d) / omega_0));
      `grad_input()` via Tape AD; gradient-descent `step()`
- [x] `coordinate_mlp.rs::FourierFeatureNetwork` / `FourierFeatureConfig` --
      sinusoidal positional encoding [sin(2*pi*B*x); cos(2*pi*B*x)] with
      Gaussian random B (Box-Muller), then MLP

#### Adaptive Sampling (sampling/)
- [x] `residual_adaptive.rs::residual_adaptive_sample` -- importance sampling
      p_i proportional to |R_i|^power via inverse-CDF with `LcgRng`
- [x] `latin_hypercube.rs::latin_hypercube_sample` -- LHS: each marginal cell
      hit exactly once via Fisher-Yates permutation per dimension
- [x] `quasi_random.rs::halton` / `halton_sequence` -- Halton radical-inverse
      sequence using first d primes for low-discrepancy sampling

#### Integration Tests (lib.rs e2e_tests)
- [x] `e2e_heat_pinn_loss_computable` -- heat PINN loss is finite and >= 0 on
      tiny MLP
- [x] `e2e_burgers_residual_near_zero_analytic` -- residual on travelling-wave
      solution is bounded
- [x] `e2e_neural_ode_rk4_exp_decay` -- RK4 integrates dy/dt = -y with error
      < 1e-4 at t = 1
- [x] `e2e_neural_ode_adjoint_gradient_sign` -- adjoint gradients are finite
      and well-defined
- [x] `e2e_fno1d_forward_shape` -- FNO1D forward preserves spatial size
- [x] `e2e_fno2d_forward_shape` -- FNO2D forward preserves H*W
- [x] `e2e_deeponet_scalar_output` -- DeepONet returns finite scalar
- [x] `e2e_cnf_log_det_finite` -- CNF log-det is finite for stable flow
- [x] `e2e_tape_gradient_xsquared` -- d/dx(x^2) at x = 3 == 6
- [x] `e2e_dual_sin_xsquared` -- d/dx(sin(x^2)) at x = 2 == cos(4) * 4 within
      1e-4
- [x] `e2e_lhs_marginal_coverage` -- each marginal cell is hit exactly once in
      every dimension
- [x] `e2e_ptx_kernels_all_sm_versions` -- all 7 kernels x 6 SM versions
      contain `.version`, `sm_X`, and kernel name

#### Benchmarks (benches/pinn_ops.rs)
- [x] 7 PTX kernel groups x 4 SM versions (PTX generation throughput)
- [x] `rk4_step_d64` -- single RK4 step on 64-dim state
- [x] `dopri45_step_d32` -- Dormand-Prince step on 32-dim state
- [x] `fno1d_forward_n32` -- FNO1D forward on N = 32 grid
- [x] `dft_n32` -- inline DFT used by FNO
- [x] `lhs_sample_d4_n256` -- Latin Hypercube sampling

### Future Enhancements [ ]

#### P0 -- Critical (Performance-Sensitive Paths)
- [ ] cuFFT-equivalent FFT path in FNO -- replace O(N^2) DFT with `oxicuda-fft`
      Stockham / Bluestein kernels for N >= 64
- [ ] Tensor-Core path for branch / trunk MLPs in DeepONet
- [ ] Fused Dopri45 step + error estimator + step controller in one PTX kernel
- [ ] PTX-side reverse-AD for `Mlp::grad_input` (currently CPU Tape replay)

#### P1 -- Important (Feature Completeness)
- [x] FNO 3D -- volumetric spectral convolution (e.g., for 3D incompressible
      Navier-Stokes) (neural_op/fno_3d.rs -- volumetric spectral conv: 3D DFT → keep top (mx,my,mz) modes → per-mode complex linear over channels → 3D iDFT + linear residual)
- [x] PINNs with hard boundary enforcement via output transform
      (e.g., u(x) = N(x) * x * (1 - x) for Dirichlet on [0, 1]) (network/hard_bc.rs -- Lagaris 1998 / Berg-Nyström 2018; û=g(x)+B(x)·N_θ with B=0 on ∂Ω (1D interval and 2D box); exact Dirichlet by construction — no boundary-loss term)
- [ ] Causal PINN training (time-marching loss weighting)
- [ ] Self-Adaptive PINN (SA-PINN) -- per-point trainable weights with maximin
      formulation
- [x] Neural SDE solver (Euler-Maruyama, Milstein, stochastic adjoint)
- [x] DeepRitz energy-functional variational PINN (pinn_loss/deep_ritz.rs -- E & Yu 2018; energy functional ∫½|∇u|²−fu dx + β∫(u−g)² ds, residual-block architecture with analytic ∇_x u, MC integration, finite-difference energy-descent train_step verified to decrease energy on 1D Poisson)
- [x] FBPINN (Finite Basis PINN) -- subdomain decomposition with partition of (network/fbpinn.rs -- Moseley et al. 2023; overlapping-subdomain Hann-window partition of unity Σω̂=1, local input normalization, per-subdomain MLPs, weighted-sum global forward)
      unity
- [x] Conservative PINN -- enforce conservation laws via integral form (pinn_loss/conservative.rs -- Liu 2020; enforce ∂_t u+∂_x F(u)=0 via INTEGRAL/flux form: residual = Δ∫u + ∫(F_R−F_L) dt over subdomain boxes; trapezoid quadrature)

#### P2 -- Nice-to-Have (Advanced Features)
- [x] hp-variational PINN (`pinn_loss/hp_variational.rs`) — Kharazmi 2021 CMAME: element-wise test functions from the hp-finite-element space; residual minimised in a Petrov-Galerkin variational formulation for improved convergence; `HpVariationalPinn`
- [x] X-PINN extended domain decomposition (`network/xpinn.rs`) — Jagtap 2021 JSSC: partition of domain into non-overlapping subdomains with interface residual conditions enforcing continuity and flux balance; `XPinn`
- [x] Wavelet Neural Operator (WNO) on Daubechies / biorthogonal bases (neural_op/wno.rs -- Tripura 2022; 1D Haar wavelet transform + per-level (in_channels×out_channels) channel-linear + inverse Haar reconstruction + linear residual)
- [ ] PointFNO / Graph FNO for unstructured meshes
- [ ] PI-DeepONet (physics-informed DeepONet) joint training
- [ ] Reservoir computing for chaotic dynamical systems
- [x] Hamiltonian / Lagrangian neural networks (HNN / LNN)
- [ ] Symbolic regression integration via PySR-style operator selection
- [ ] PDE-Net learned PDE discovery primitives

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

No CUDA SDK, no C, no Fortran. The crate compiles standalone and produces PTX
strings that can be consumed by `oxicuda-driver` / `oxicuda-launch` at runtime.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 624 unit + 12 E2E = 636 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- All public APIs return `PinnResult<T>` or `Result<T, PinnError>`

## Performance Targets

Reference shapes (FNO and ODE integration are the hot paths):

| Kernel | Shape | Target |
|--------|-------|--------|
| pinn_residual reduce | n_pts = 65536 | bandwidth + atomic-throughput |
| spectral_conv (FNO1D) | width = 64, k_max = 16, N = 256 | dominated by DFT |
| dual_op_ptx | tape length = 1024 | arithmetic-bound |
| adjoint_ode step | dim = 256, n_steps = 1024 | bandwidth-limited |
| branch_trunk_dot (DeepONet) | p = 128, batch = 1024 | warp-shuffle reduce |
| siren_forward | width = 256, n_pts = 4096 | bandwidth + sin throughput |
| lhs_sample | n = 4096, d = 8 | latency-bound |

## Notes

- All randomness is deterministic via `LcgRng` seeded by `PinnHandle`; unit
  tests do not depend on `rand` or `getrandom`.
- `Mlp` uses SIREN-style initialisation when `Activation::Sin` is selected;
  hidden-layer scale uses sqrt(6 / d) / omega_0 to maintain stable gradient
  magnitudes for sinusoidal activations.
- `Tape` is an index-based Wengert list (no `Rc`, no `RefCell`); reverse pass
  walks the tape in reverse order to accumulate gradients.
- `Dual` and `MultiDual` are independent of Tape; choose dual numbers for
  N <= O(10) inputs and Tape for everything else.
- `dopri45_step` returns (y_new, error_estimate); `integrate_adaptive` uses
  the standard PI step controller with safety factor 0.9 and order = 5.
- Both inline `dft_1d` / `idft_1d` are exposed for testing; production
  workloads should swap in `oxicuda-fft` for N >= 64.
- `latin_hypercube_sample` guarantees one sample per cell per dimension; the
  E2E test verifies this for n = 100, d = 2.

---

## Architecture-Specific Deepening

### Hopper (sm_90 / sm_90a)
- [ ] `wgmma.mma_async` path for FNO lift / project FC layers and DeepONet MLPs
- [ ] TMA (`cp.async.bulk`) loading of collocation point batches in
      `pinn_residual_ptx`

### Ampere (sm_80 / sm_86) / Ada (sm_89)
- [ ] `cp.async` prefetch of FNO spectral weights
- [ ] Cooperative groups for warp-wide dot-product reduction in DeepONet

### Blackwell (sm_100 / sm_120)
- [ ] 5th-gen Tensor Core path for MWT learnable kernel projection
- [ ] Cluster launch for cross-CTA Hutchinson trace estimator in CNF

---

## Deepening Opportunities

### Verification Gaps
- [x] All 7 PTX generators emit `.version`, `.target sm_X`, and named entry per
      SM version (verified by `e2e_ptx_kernels_all_sm_versions`)
- [x] RK4 verified against analytic exp-decay solution
- [x] Adjoint gradients verified finite vs. analytic linear ODE
- [x] Tape gradient verified on x^2 (analytic = 2x)
- [x] Dual gradient verified on sin(x^2) (analytic = cos(x^2) * 2x)
- [x] LHS marginal coverage verified (every cell hit once per dim)
- [ ] FNO spectral correctness vs. analytic 1D heat solution
- [ ] CNF log-det numerical parity vs. dense-Jacobian trace on small Gaussians
- [ ] Dopri45 step-controller stability on stiff problems

### Implementation Deepening
- [x] `forbid(unsafe_code)` enforced at crate level
- [x] All RNG operations deterministic given seed
- [x] Five PDE templates with analytic-solution checkers
- [x] Four ODE solvers (Euler, Heun, RK4, Dopri45) + adaptive controller
- [x] Four neural-operator families (FNO 1D / 2D, DeepONet, MWT, GNO)
- [ ] Mixed-precision (bf16 storage, fp32 accumulate) variants for FNO
      and DeepONet
- [ ] Stiff-ODE solvers (Rosenbrock, BDF) for chemistry / circuits
- [x] Symplectic integrators (leapfrog, Stormer-Verlet) for Hamiltonian systems
- [ ] Automatic differentiation of PDE residual w.r.t. inputs via `MultiDual`
      (currently CPU-only via Tape)
- [x] Periodic boundary helper in `BcType`
