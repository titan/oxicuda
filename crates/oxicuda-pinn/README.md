# oxicuda-pinn

Physics-Informed Neural Networks for OxiCUDA: PINN losses, Neural ODEs/SDEs (adjoint method), Neural Operators (FNO/DeepONet/MWT/GNO), PDE templates, autodiff (dual numbers + tape), adaptive collocation sampling — pure Rust, zero CUDA SDK dependency.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **Autodiff**: Forward-mode dual numbers and reverse-mode tape (scalar and multi-dimensional), supporting `sin`, `cos`, `exp`, `sq`, and arithmetic operations
- **Neural ODEs**: Euler, Heun, RK4, and DOPRI4/5 adaptive solvers; adjoint gradient method; continuous normalizing flows (CNF); Latent ODE
- **Neural Operators**: FNO 1D/2D (Fourier Neural Operator with DFT/iDFT), DeepONet (branch-trunk), MWT, GNO
- **PDE templates**: Heat, Wave, Burgers, Poisson, and Navier-Stokes residual and analytic-solution helpers
- **Collocation sampling**: Latin Hypercube Sampling (LHS), Halton quasi-random sequences, residual-adaptive refinement
- **PTX kernels**: 7 GPU kernels (PINN residual, spectral conv, dual mul, adjoint ODE step, branch-trunk dot, SIREN forward, LHS sample) × 6 SM versions

## Usage

```rust
use oxicuda_pinn::prelude::*;

// Forward-mode autodiff: d/dx sin(x²) at x=2
let x = Dual::variable(2.0_f32);
let f = (x * x).sin();
println!("sin(x²)' at x=2: {}", f.dvalue); // ≈ cos(4) * 4

// Solve dy/dt = -y with RK4 for 100 steps
fn exp_decay(_t: f32, y: &[f32], dy: &mut [f32]) { dy[0] = -y[0]; }
let mut y = vec![1.0_f32];
for step in 0..100 {
    y = rk4_step(&exp_decay, step as f32 * 0.01, &y, 0.01);
}
println!("y(1.0) ≈ {} (exact: {})", y[0], (-1.0_f32).exp());

// PINN residual loss
let residuals = vec![0.1_f32, -0.05, 0.08];
let loss = pde_residual_loss(&residuals).unwrap();
println!("PDE residual loss: {loss}");
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-pinn)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
