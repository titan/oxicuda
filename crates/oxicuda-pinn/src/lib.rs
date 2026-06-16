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
}

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
