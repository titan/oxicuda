//! `oxicuda-pinn` — Physics-Informed Neural Networks for OxiCUDA.
//!
//! Pure-Rust implementation of PINN algorithms, suitable for CPU simulation
//! and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-pinn
//! ├── autodiff/      — Dual numbers (forward AD) + Tape (reverse AD) + MultiDual
//! ├── pinn_loss/     — PDE residual, BC, IC losses + adaptive NTK weighting
//! ├── neural_ode/    — Euler/Heun/RK4/DOPRI45 solvers, adjoint method, CNF, LatentODE
//! ├── neural_op/     — FNO (1D/2D), DeepONet, MWT, GNO
//! ├── pde/           — Heat, Wave, Burgers, Poisson, Navier-Stokes templates
//! ├── network/       — MLP (SIREN init) + Fourier Feature Network
//! ├── sampling/      — Residual-adaptive, LHS, Halton quasi-random
//! ├── error          — PinnError / PinnResult
//! ├── handle         — PinnHandle (SmVersion + LcgRng)
//! └── ptx_kernels    — 7 GPU PTX kernel strings × 6 SM versions
//! ```

#![forbid(unsafe_code)]

pub mod autodiff;
pub mod error;
pub mod features;
pub mod handle;
pub mod network;
pub mod neural_ode;
pub mod neural_op;
pub mod pde;
pub mod pinn_loss;
pub mod ptx_kernels;
pub mod sampling;
pub mod symbolic;
pub mod variants;

/// Convenience re-exports for common PINN types.
pub mod prelude {
    // Error handling
    pub use crate::error::{PinnError, PinnResult};

    // Handle
    pub use crate::handle::{LcgRng, PinnHandle, SmVersion};

    // PTX kernels
    pub use crate::ptx_kernels::{
        adjoint_ode_ptx, branch_trunk_dot_ptx, dual_op_ptx, f32_hex, lhs_sample_ptx,
        pinn_residual_ptx, siren_forward_ptx, spectral_conv_ptx,
    };

    // Autodiff
    pub use crate::autodiff::dual::Dual;
    pub use crate::autodiff::multidim::MultiDual;
    pub use crate::autodiff::pde_residual::{
        HyperDual, burgers_residual_ad, heat_residual_ad, linear_2nd_order_residual_ad,
        poisson_residual_ad,
    };
    pub use crate::autodiff::tape::{Tape, Var};

    // PINN losses
    pub use crate::pinn_loss::boundary::{BcType, bc_loss};
    pub use crate::pinn_loss::causal::{CausalPinnConfig, CausalPinnLoss};
    pub use crate::pinn_loss::conservative::{ConservativeConfig, ConservativeLoss, SubdomainBox};
    pub use crate::pinn_loss::deep_ritz::{DeepRitz, DeepRitzConfig, DeepRitzEnergy, DeepRitzNet};
    pub use crate::pinn_loss::hp_variational::{
        HpVariationalConfig, HpVariationalPinn, gauss_legendre, legendre_basis,
    };
    pub use crate::pinn_loss::initial::ic_loss;
    pub use crate::pinn_loss::periodic::{
        PeriodicEmbedding, periodic_bc_loss, periodic_bc_loss_value,
    };
    pub use crate::pinn_loss::relobralo::{ReloBraLo, ReloBraLoConfig};
    pub use crate::pinn_loss::residual::{compute_residuals, pde_residual_loss};
    pub use crate::pinn_loss::sa_pinn::{SaPinn, SaPinnConfig};
    pub use crate::pinn_loss::weighting::AdaptiveWeights;

    // Feature embeddings
    pub use crate::features::fourier_features::{FourierFeatureEmbeddingConfig, FourierFeatures};

    // PINN variants
    pub use crate::variants::gpinn::{GPinnConfig, GPinnLoss, GPinnLossTerms};
    pub use crate::variants::pde_discovery::{
        LibraryConfig, PdeNetCell, SindyConfig, SindyModel, build_library, fit_sindy,
    };

    // Neural ODE
    pub use crate::neural_ode::adjoint::{node_adjoint_grad, node_forward};
    pub use crate::neural_ode::cnf::{cnf_forward, dense_trace, hutchinson_trace};
    pub use crate::neural_ode::hamiltonian::{
        HamiltonianNn, HnnConfig, HnnTrajectory, HnnWeights, LagrangianNn, LnnConfig, LnnTrajectory,
    };
    pub use crate::neural_ode::latent_ode::{LatentOde, LatentOdeConfig};
    pub use crate::neural_ode::neural_sde::{
        NeuralSde, NeuralSdeConfig, NeuralSdeWeights, NoiseType, SdeMethod, SdePath,
    };
    pub use crate::neural_ode::solvers::{
        OdeRhsFn, dopri45_step, euler_step, heun_step, integrate_adaptive, integrate_fixed,
        rk4_step,
    };
    pub use crate::neural_ode::solvers_batch::{
        OdeRhsFnBatch, euler_step_batch, heun_step_batch, integrate_batch, rk4_step_batch,
    };
    pub use crate::neural_ode::stiff::{
        StiffConfig, StiffRhsFn, backward_euler_step, integrate_backward_euler, integrate_bdf,
        integrate_rosenbrock2, rosenbrock2_step,
    };
    pub use crate::neural_ode::symplectic::{
        ForceFn, SymplecticMethod, hamiltonian_energy, integrate_symplectic, leapfrog_step,
        stormer_verlet_step, symplectic_euler_step, velocity_verlet_step,
    };

    // Neural operators
    pub use crate::neural_op::deeponet::{DeepONet, DeepONetConfig};
    pub use crate::neural_op::fno::{Fno1d, Fno1dConfig, Fno2d, Fno2dConfig, dft_1d, idft_1d};
    pub use crate::neural_op::fno_3d::{Fno3d, Fno3dConfig};
    pub use crate::neural_op::gno::{Gno, GnoConfig};
    pub use crate::neural_op::mwt::{Mwt, MwtConfig};
    pub use crate::neural_op::pi_deeponet::{PiDeepONet, PiDeepONetConfig};
    pub use crate::neural_op::point_fno::{PointFno, PointFnoConfig};
    pub use crate::neural_op::wno::{Wno, WnoConfig};

    // PDE templates
    pub use crate::pde::burgers::{burgers_analytic, burgers_residual};
    pub use crate::pde::heat::{heat_analytic, heat_residual, heat_residual_check};
    pub use crate::pde::navier_stokes::{ns_vorticity_residual, taylor_green_vortex};
    pub use crate::pde::poisson::{poisson_analytic, poisson_residual};
    pub use crate::pde::wave::{wave_analytic, wave_residual};

    // Networks
    pub use crate::network::coordinate_mlp::{FourierFeatureConfig, FourierFeatureNetwork};
    pub use crate::network::fbpinn::{Fbpinn, FbpinnConfig, Subdomain};
    pub use crate::network::hard_bc::{BoundaryDomain, HardBc, HardBcConfig};
    pub use crate::network::mlp::{Activation, Mlp, MlpConfig};
    pub use crate::network::rbf_features::{
        RbfFeatureConfig, RbfFeatureNetwork, RbfFeatures, RbfKind,
    };
    pub use crate::network::reservoir_computing::{EchoStateNetwork, EsnConfig, spectral_radius};

    // Sampling
    pub use crate::sampling::latin_hypercube::latin_hypercube_sample;
    pub use crate::sampling::quasi_random::{halton, halton_sequence};
    pub use crate::sampling::residual_adaptive::residual_adaptive_sample;

    // Symbolic regression
    pub use crate::symbolic::regression::{Expr, Individual, SymbolicConfig, SymbolicRegressor};
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

#[cfg(test)]
mod e2e_tests {
    use super::prelude::*;

    #[test]
    fn e2e_heat_pinn_loss_computable() {
        // Build small MLP, compute residual at 16 heat pts, verify loss is finite and >= 0
        let mut rng = LcgRng::new(1);
        let cfg = MlpConfig {
            layer_widths: vec![2, 16, 1],
            activation: Activation::Tanh,
            omega_0: 1.0,
        };
        let mlp =
            Mlp::new(cfg, &mut rng).expect("MLP construction with valid config should succeed");
        let pts: Vec<f32> = (0..16)
            .flat_map(|i| {
                let x = (i % 4) as f32 * 0.25;
                let t = (i / 4) as f32 * 0.25;
                vec![x, t]
            })
            .collect();
        let residuals: Vec<f32> = (0..16)
            .map(|i| {
                let out = mlp.forward(&pts[i * 2..i * 2 + 2]).unwrap_or(vec![0.0]);
                out[0]
            })
            .collect();
        let loss =
            pde_residual_loss(&residuals).expect("PDE residual loss computation should succeed");
        assert!(loss.is_finite(), "Heat PINN loss should be finite: {loss}");
        assert!(loss >= 0.0, "Heat PINN loss should be >= 0: {loss}");
    }

    #[test]
    fn e2e_burgers_residual_near_zero_analytic() {
        // Traveling wave solution is approximate; verify residual is bounded
        let nu = 0.1_f32;
        let x = 0.5_f32;
        let t = 0.3_f32;
        let ok = crate::pde::burgers::burgers_residual_check(x, t, nu, 0.5)
            .expect("Burgers residual check on traveling wave analytic solution should succeed");
        assert!(ok, "Burgers residual on analytic solution should be small");
    }

    #[test]
    fn e2e_neural_ode_rk4_exp_decay() {
        fn exp_decay(_t: f32, y: &[f32], dy: &mut [f32]) {
            dy[0] = -y[0];
        }
        let mut y = vec![1.0_f32];
        for step in 0..100 {
            let t = step as f32 * 0.01;
            y = rk4_step(&exp_decay, t, &y, 0.01);
        }
        let expected = (-1.0_f32).exp();
        assert!(
            (y[0] - expected).abs() < 1e-4,
            "RK4: y(1)={} expected {}",
            y[0],
            expected
        );
    }

    #[test]
    fn e2e_neural_ode_adjoint_gradient_sign() {
        fn exp_decay(_t: f32, y: &[f32], dy: &mut [f32]) {
            dy[0] = -y[0];
        }
        let (_, traj) = node_forward(&exp_decay, 0.0, 0.5, &[1.0], 0.05).expect(
            "Neural ODE forward integration of exponential decay from t=0 to t=0.5 should succeed",
        );
        let dfdy = |_t: f32, _y: &[f32]| vec![-1.0_f32];
        let dfdth = |_t: f32, _y: &[f32]| vec![1.0_f32];
        let dl_dth = node_adjoint_grad(&exp_decay, &dfdy, &dfdth, &traj, &[1.0], 0.05)
            .expect("Adjoint gradient computation for exponential decay Neural ODE should succeed");
        assert!(
            dl_dth.iter().all(|v| v.is_finite()),
            "Adjoint grads not finite: {:?}",
            dl_dth
        );
    }

    #[test]
    fn e2e_fno1d_forward_shape() {
        let mut rng = LcgRng::new(2);
        let cfg = Fno1dConfig {
            d_in: 1,
            d_out: 1,
            width: 8,
            k_max: 4,
            n_blocks: 2,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let input = vec![0.5_f32; 16];
        let output = fno
            .forward(&input, 16)
            .expect("FNO1d forward pass on 16-point uniform input should succeed");
        assert_eq!(output.len(), 16);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "FNO1d output not all finite"
        );
    }

    #[test]
    fn e2e_fno2d_forward_shape() {
        let mut rng = LcgRng::new(3);
        let cfg = Fno2dConfig {
            d_in: 1,
            d_out: 1,
            width: 4,
            k_max: 2,
            n_blocks: 1,
        };
        let fno = Fno2d::new(cfg, &mut rng);
        let input = vec![0.3_f32; 8 * 8];
        let output = fno
            .forward(&input, 8, 8)
            .expect("FNO2d forward pass on 8×8 uniform grid input should succeed");
        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_deeponet_scalar_output() {
        let mut rng = LcgRng::new(4);
        let cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 8,
            d_query: 1,
            p: 16,
            branch_hidden: vec![32],
            trunk_hidden: vec![32],
        };
        let model = DeepONet::new(cfg, &mut rng);
        let fs = vec![0.5_f32; 8];
        let q = vec![0.3_f32];
        let out = model.forward(&fs, &q).expect(
            "DeepONet forward pass with 8 sensor values and scalar query point should succeed",
        );
        assert!(out.is_finite(), "DeepONet output not finite: {out}");
    }

    #[test]
    fn e2e_cnf_log_det_finite() {
        fn scale_flow(_t: f32, z: &[f32], dz: &mut [f32]) {
            for (dzi, &zi) in dz.iter_mut().zip(z.iter()) {
                *dzi = 0.1 * zi;
            }
        }
        let z0 = vec![1.0_f32, 0.5];
        let (z1, dlp) = cnf_forward(&scale_flow, &z0, 0.0, 0.5, 0.05)
            .expect("CNF forward pass with linear scale flow from t=0 to t=0.5 should succeed");
        assert!(z1.iter().all(|v| v.is_finite()));
        assert!(dlp.is_finite(), "log-det not finite: {dlp}");
    }

    #[test]
    fn e2e_tape_gradient_xsquared() {
        let mut tape = Tape::new();
        let x = tape.variable(3.0);
        let f = tape.sq(x);
        let grads = tape.gradient(f).expect(
            "Reverse-mode gradient of x² on the tape should succeed for scalar variable x=3",
        );
        assert!(
            (grads[x.idx] - 6.0).abs() < 1e-6,
            "grad x² at 3 = 6, got {}",
            grads[x.idx]
        );
    }

    #[test]
    fn e2e_dual_sin_xsquared() {
        let x_val = 2.0_f32;
        let x = Dual::variable(x_val);
        let f = (x * x).sin();
        let expected = (x_val * x_val).cos() * 2.0 * x_val;
        assert!((f.dvalue - expected).abs() < 1e-4);
    }

    #[test]
    fn e2e_lhs_marginal_coverage() {
        let mut rng = LcgRng::new(5);
        let n = 100;
        let d = 2;
        let samples = latin_hypercube_sample(n, d, &mut rng);
        for j in 0..d {
            let mut bins = vec![0_usize; n];
            for i in 0..n {
                let v = samples[i * d + j];
                let bin = (v * n as f32).floor() as usize;
                let bin = bin.min(n - 1);
                bins[bin] += 1;
            }
            assert!(
                bins.iter().all(|&c| c == 1),
                "Not all LHS bins hit exactly once in dim {j}"
            );
        }
    }

    #[test]
    fn e2e_fno_spectral_matches_analytic_heat() {
        // Verification gap: FNO spectral correctness vs the analytic 1-D heat
        // solution. The heat equation u_t = α u_xx is *diagonal* in the Fourier
        // basis: a mode of wavenumber κ decays as exp(-α κ² t). The FNO spectral
        // path is exactly "forward DFT → per-mode complex multiply → inverse DFT",
        // so applying the analytic heat propagator as the per-mode multiplier must
        // reproduce the analytic solution evolved by Δt.
        //
        // Work on a periodic grid with u₀(x) = sin(2πx) (a single Fourier mode on
        // [0,1)), for which u(x,t) = sin(2πx)·exp(-α(2π)² t) exactly.
        let n = 32usize;
        let alpha = 0.05_f32;
        let dt = 0.1_f32;
        let two_pi = 2.0 * std::f32::consts::PI;

        // Sample the initial condition (one period over the grid).
        let u0: Vec<f32> = (0..n)
            .map(|i| (two_pi * i as f32 / n as f32).sin())
            .collect();

        // Forward DFT (the FNO spectral entry point).
        let (mut re, mut im) = dft_1d(&u0);

        // Per-mode multiply by the heat propagator exp(-α κ_k² Δt), where the
        // grid wavenumber for bin k on [0,1) is κ_k = 2π·k_signed (k_signed folds
        // bins above n/2 to negative frequencies, matching the real-DFT symmetry).
        for k in 0..n {
            let k_signed = if k <= n / 2 {
                k as f32
            } else {
                k as f32 - n as f32
            };
            let kappa = two_pi * k_signed;
            let decay = (-alpha * kappa * kappa * dt).exp();
            re[k] *= decay;
            im[k] *= decay;
        }

        // Inverse DFT back to physical space (the FNO spectral exit point).
        let u_spectral = idft_1d(&re, &im);

        // Analytic reference at t = dt.
        let kappa1 = two_pi; // first mode
        let decay1 = (-alpha * kappa1 * kappa1 * dt).exp();
        let u_exact: Vec<f32> = (0..n)
            .map(|i| (two_pi * i as f32 / n as f32).sin() * decay1)
            .collect();

        let max_err = u_spectral
            .iter()
            .zip(u_exact.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-4,
            "FNO spectral heat propagation should match analytic solution, max_err = {max_err}"
        );

        // Sanity: the analytic single-mode solution agrees with heat_analytic on
        // the [0,1] Dirichlet problem for the κ = π mode (independent cross-check
        // that the propagator form exp(-α κ² t) is the correct one).
        let u_half = heat_analytic(0.5, dt, alpha); // sin(π·0.5)·exp(-απ²·dt)
        let expect_half = (std::f32::consts::PI * 0.5).sin()
            * (-alpha * std::f32::consts::PI * std::f32::consts::PI * dt).exp();
        assert!((u_half - expect_half).abs() < 1e-6);
    }

    #[test]
    fn e2e_cnf_log_det_parity_dense_trace() {
        // Verification gap: CNF log-det numerical parity vs a dense-Jacobian trace
        // on a small Gaussian. For a linear flow f(z) = A·z the Jacobian is the
        // constant matrix A, so the exact log-density change over [t0, t1] is
        //   Δlog p = -∫ tr(∂f/∂z) dt = -tr(A)·(t1 - t0).
        // Use a 2×2 diagonal "Gaussian-shaping" flow with tr(A) = 0.1 + 0.2 = 0.3.
        fn linear_flow(_t: f32, z: &[f32], dz: &mut [f32]) {
            dz[0] = 0.1 * z[0];
            dz[1] = 0.2 * z[1];
        }
        let trace_a = 0.3_f32;
        let t0 = 0.0_f32;
        let t1 = 0.75_f32;

        // dense_trace must recover tr(A) exactly (finite-difference of a linear
        // map is exact up to rounding).
        let z = vec![0.7_f32, -1.3];
        let tr = dense_trace(&linear_flow, 0.3, &z);
        assert!(
            (tr - trace_a).abs() < 1e-3,
            "dense_trace should equal tr(A) = 0.3, got {tr}"
        );

        // cnf_forward integrates -tr over the trajectory; for a constant Jacobian
        // it must match the closed form -tr(A)·(t1 - t0).
        let z0 = vec![0.5_f32, -0.2];
        let (_z1, delta_log_p) = cnf_forward(&linear_flow, &z0, t0, t1, 0.005)
            .expect("CNF forward with linear Gaussian-shaping flow should succeed");
        let expected = -trace_a * (t1 - t0);
        assert!(
            (delta_log_p - expected).abs() < 5e-3,
            "CNF Δlog p = {delta_log_p} should match -tr(A)·T = {expected}"
        );
    }

    #[test]
    fn e2e_dopri45_step_controller_stiff_stability() {
        // Verification gap: Dopri45 step-controller stability on a stiff problem.
        // The scalar test equation y' = -1000(y - cos t) is stiff: its fast
        // timescale is 1/1000 while the solution relaxes onto the slow manifold
        // y ≈ cos t. A fixed-step *explicit Euler* with h on the order of the slow
        // scale (h = 0.05 ⇒ h·1000 = 50 ≫ 2) is unconditionally unstable and
        // blows up, whereas the adaptive Dopri45 PI controller must shrink h to
        // stay in the stability region and track cos t.
        fn stiff(t: f32, y: &[f32], dy: &mut [f32]) {
            dy[0] = -1000.0 * (y[0] - t.cos());
        }

        // (a) Fixed large-step explicit Euler diverges.
        let mut y_euler = vec![0.0_f32];
        for step in 0..40 {
            let t = step as f32 * 0.05;
            y_euler = euler_step(&stiff, t, &y_euler, 0.05);
        }
        assert!(
            !y_euler[0].is_finite() || y_euler[0].abs() > 1e3,
            "fixed-step explicit Euler should be unstable on the stiff problem, got {}",
            y_euler[0]
        );

        // (b) Adaptive Dopri45 stays bounded and tracks the slow manifold cos t.
        let (times, states) = integrate_adaptive(&stiff, 0.0, 2.0, &[0.0], 1e-6, 1e-5, 0.05)
            .expect("adaptive Dopri45 integration of the stiff problem should succeed");
        assert!(
            states.iter().all(|s| s[0].is_finite()),
            "adaptive Dopri45 must keep the stiff solution finite"
        );
        let t_final = *times
            .last()
            .expect("adaptive integration produced no times");
        let y_final = states
            .last()
            .expect("adaptive integration produced no states")[0];
        // After the fast initial transient the solution lies on y ≈ cos t.
        assert!(
            (y_final - t_final.cos()).abs() < 1e-2,
            "Dopri45 should track the slow manifold cos t: y({t_final}) = {y_final}, cos = {}",
            t_final.cos()
        );
        // The controller must have refined the step well below the unstable 0.05.
        assert!(
            times.len() > 50,
            "adaptive controller should take many small steps on the stiff problem, took {}",
            times.len() - 1
        );
    }

    #[test]
    fn e2e_symbolic_regression_recovers_quadratic() {
        // End-to-end: genetic-programming symbolic regression recovers x²+1.
        let xs: Vec<f32> = (0..21).map(|i| -2.0 + i as f32 * 0.2).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| x * x + 1.0).collect();
        let signal_var = {
            let mean = ys.iter().sum::<f32>() / ys.len() as f32;
            ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f32>() / ys.len() as f32
        };
        let mut cfg = SymbolicConfig::new();
        cfg.population = 500;
        cfg.generations = 80;
        let mut rng = LcgRng::new(99);
        let mut reg = SymbolicRegressor::new(cfg);
        let best = reg
            .fit(&xs, &ys, &mut rng)
            .expect("symbolic regression should recover the quadratic target");
        assert!(
            best.mse < 0.05 * signal_var,
            "recovered MSE {} should be well below signal variance {}",
            best.mse,
            signal_var
        );
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("pinn_residual_kernel", pinn_residual_ptx),
            ("spectral_conv_kernel", spectral_conv_ptx),
            ("dual_mul_kernel", dual_op_ptx),
            ("adjoint_step_kernel", adjoint_ode_ptx),
            ("branch_trunk_dot_kernel", branch_trunk_dot_ptx),
            ("siren_forward_kernel", siren_forward_ptx),
            ("lhs_sample_kernel", lhs_sample_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(!ptx.is_empty(), "PTX for {kernel_name} sm={sm} is empty");
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} sm={sm} missing .version"
                );
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing target"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} sm={sm} missing kernel name"
                );
            }
        }
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
