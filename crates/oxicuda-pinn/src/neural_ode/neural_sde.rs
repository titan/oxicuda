//! Neural Stochastic Differential Equations (Neural SDEs).
//!
//! Models stochastic dynamics:
//!   dx = f(x, t) dt + g(x, t) dW
//! where f (drift) and g (diffusion) are neural networks, and W is
//! a standard Brownian motion of dimension `noise_dim`.
//!
//! **Methods**:
//! - Euler-Maruyama (EM): first-order strong convergence.
//! - Milstein (diagonal noise): second-order strong convergence via Itô
//!   correction for diagonal diffusion coefficients.
//!
//! **References**:
//! - Chen et al. "Neural Ordinary Differential Equations" NeurIPS 2018.
//! - Liu "Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise"
//!   arXiv 2019.
//! - Kloeden & Platen "Numerical Solution of Stochastic Differential Equations"
//!   Ch. 9 (Milstein scheme).

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

// ─── Enums ───────────────────────────────────────────────────────────────────

/// Numerical integration scheme for the SDE.
#[derive(Debug, Clone, PartialEq)]
pub enum SdeMethod {
    /// Euler-Maruyama: x_{t+h} = x_t + f(x_t,t)h + g(x_t,t)√h dW.
    EulerMaruyama,
    /// Milstein (diagonal noise only): adds Itô correction term.
    Milstein,
}

/// Noise structure for the diffusion network.
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseType {
    /// Diagonal: g maps x → ℝ^{noise_dim}; applied element-wise to dW.
    Diagonal,
    /// Scalar: g maps x → ℝ^1; all state dimensions share one noise coefficient.
    Scalar,
}

// ─── Config & Weights ────────────────────────────────────────────────────────

/// Configuration for a Neural SDE.
#[derive(Debug, Clone)]
pub struct NeuralSdeConfig {
    /// State-space dimension.
    pub state_dim: usize,
    /// Brownian motion dimension (noise sources).
    pub noise_dim: usize,
    /// Width of each hidden layer for drift and diffusion networks.
    pub hidden_width: usize,
    /// Number of hidden layers for drift and diffusion networks.
    pub n_layers: usize,
    /// Start time.
    pub t0: f32,
    /// End time.
    pub t1: f32,
    /// Number of integration steps.
    pub n_steps: usize,
    /// Integration method.
    pub method: SdeMethod,
    /// Noise structure.
    pub noise_type: NoiseType,
}

/// Weight storage for drift and diffusion networks.
#[derive(Debug, Clone)]
pub struct NeuralSdeWeights {
    /// Drift network weight matrices (flat row-major).
    pub drift_layers: Vec<Vec<f32>>,
    /// Drift network bias vectors.
    pub drift_biases: Vec<Vec<f32>>,
    /// Diffusion network weight matrices.
    pub diffusion_layers: Vec<Vec<f32>>,
    /// Diffusion network bias vectors.
    pub diffusion_biases: Vec<Vec<f32>>,
}

/// A Neural SDE with learned drift f(x,t) and diffusion g(x,t).
pub struct NeuralSde {
    pub cfg: NeuralSdeConfig,
    pub weights: NeuralSdeWeights,
}

/// A single sample path of the Neural SDE.
pub struct SdePath {
    /// Time grid: `n_steps + 1` points in `[t0, t1]`.
    pub times: Vec<f32>,
    /// State trajectory: `(n_steps + 1) × state_dim`, row-major.
    pub states: Vec<f32>,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Apply a single dense layer: output = W x + b.
///
/// `w`: row-major `[d_out × d_in]`, `b`: `[d_out]`, `x`: `[d_in]`.
fn linear_layer(w: &[f32], b: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
    let d_in = x.len();
    (0..d_out)
        .map(|i| {
            let dot: f32 = (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum();
            dot + b[i]
        })
        .collect()
}

/// Run an MLP defined by `(layers, biases)` on `input`.
/// All hidden layers use tanh; output layer is linear.
fn run_mlp(layers: &[Vec<f32>], biases: &[Vec<f32>], input: &[f32]) -> PinnResult<Vec<f32>> {
    if layers.is_empty() {
        return Err(PinnError::InvalidNetworkDepth { depth: 0 });
    }
    let n_layers = layers.len();
    let mut act: Vec<f32> = input.to_vec();
    for (idx, (w, b)) in layers.iter().zip(biases.iter()).enumerate() {
        let d_out = b.len();
        act = linear_layer(w, b, &act, d_out);
        if idx + 1 < n_layers {
            for v in &mut act {
                *v = v.tanh();
            }
        }
    }
    if !act.iter().all(|v| v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "run_mlp output",
        });
    }
    Ok(act)
}

/// Kaiming uniform: U(-√(6/fan_in), +√(6/fan_in)).
fn kaiming_uniform(fan_in: usize, n: usize, rng: &mut LcgRng) -> Vec<f32> {
    let bound = (6.0_f32 / fan_in as f32).sqrt();
    (0..n)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// Build weight/bias vectors for an MLP: input_dim → [hidden_width × n_layers] → output_dim.
fn build_mlp(
    input_dim: usize,
    hidden_width: usize,
    n_layers: usize,
    output_dim: usize,
    rng: &mut LcgRng,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut layers = Vec::new();
    let mut biases = Vec::new();

    // Input → first hidden
    layers.push(kaiming_uniform(input_dim, hidden_width * input_dim, rng));
    biases.push(vec![0.0_f32; hidden_width]);

    // Hidden → hidden (n_layers - 1 additional)
    for _ in 1..n_layers {
        layers.push(kaiming_uniform(
            hidden_width,
            hidden_width * hidden_width,
            rng,
        ));
        biases.push(vec![0.0_f32; hidden_width]);
    }

    // Last hidden → output
    layers.push(kaiming_uniform(
        hidden_width,
        output_dim * hidden_width,
        rng,
    ));
    biases.push(vec![0.0_f32; output_dim]);

    (layers, biases)
}

// ─── NeuralSde ───────────────────────────────────────────────────────────────

impl NeuralSde {
    /// Construct a randomly initialised Neural SDE.
    pub fn new(cfg: NeuralSdeConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if cfg.state_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.noise_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.hidden_width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.n_layers == 0 {
            return Err(PinnError::InvalidNetworkDepth { depth: 0 });
        }
        if cfg.n_steps == 0 {
            return Err(PinnError::InvalidStepSize { h: 0.0 });
        }
        if cfg.t0 >= cfg.t1 {
            return Err(PinnError::InvalidTimeInterval {
                t0: cfg.t0,
                t1: cfg.t1,
            });
        }

        // Drift input: [x (state_dim), t_norm (1)] → state_dim output.
        let drift_input_dim = cfg.state_dim + 1;
        let (drift_layers, drift_biases) = build_mlp(
            drift_input_dim,
            cfg.hidden_width,
            cfg.n_layers,
            cfg.state_dim,
            rng,
        );

        // Diffusion same architecture; output = noise_dim (Diagonal) or 1 (Scalar).
        let diff_output_dim = match cfg.noise_type {
            NoiseType::Diagonal => cfg.noise_dim,
            NoiseType::Scalar => 1,
        };
        let diff_input_dim = cfg.state_dim + 1;
        let (diffusion_layers, diffusion_biases) = build_mlp(
            diff_input_dim,
            cfg.hidden_width,
            cfg.n_layers,
            diff_output_dim,
            rng,
        );

        Ok(Self {
            cfg,
            weights: NeuralSdeWeights {
                drift_layers,
                drift_biases,
                diffusion_layers,
                diffusion_biases,
            },
        })
    }

    /// Normalised time coordinate: t_norm = (t − t0) / (t1 − t0) ∈ [0, 1].
    #[inline]
    fn t_norm(&self, t: f32) -> f32 {
        (t - self.cfg.t0) / (self.cfg.t1 - self.cfg.t0)
    }

    /// Build network input: \[x..., t_norm\].
    fn build_input(&self, x: &[f32], t: f32) -> Vec<f32> {
        let mut inp = x.to_vec();
        inp.push(self.t_norm(t));
        inp
    }

    /// Drift f(x, t) → `state_dim` vector.
    pub fn drift(&self, x: &[f32], t: f32) -> PinnResult<Vec<f32>> {
        let d = self.cfg.state_dim;
        if x.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let inp = self.build_input(x, t);
        let out = run_mlp(&self.weights.drift_layers, &self.weights.drift_biases, &inp)?;
        if out.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: out.len(),
            });
        }
        Ok(out)
    }

    /// Diffusion g(x, t).
    ///
    /// For `Diagonal` noise: returns `noise_dim` values.
    /// For `Scalar` noise: returns a `Vec` of length 1.
    pub fn diffusion(&self, x: &[f32], t: f32) -> PinnResult<Vec<f32>> {
        let d = self.cfg.state_dim;
        if x.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let inp = self.build_input(x, t);
        run_mlp(
            &self.weights.diffusion_layers,
            &self.weights.diffusion_biases,
            &inp,
        )
    }

    /// Sample a `noise_dim`-dimensional standard normal increment vector.
    fn sample_noise(&self, rng: &mut LcgRng) -> Vec<f32> {
        let nd = self.cfg.noise_dim;
        let mut z = vec![0.0_f32; nd];
        rng.fill_normal(&mut z);
        z
    }

    /// Euler-Maruyama step: x_{t+h} = x_t + f(x_t,t)·h + g(x_t,t)·√h·dW.
    ///
    /// For `Diagonal` noise (noise_dim == state_dim):
    ///   x_i += g_i(x,t) · √h · z_i
    ///
    /// For `Scalar` noise (noise_dim can differ from state_dim):
    ///   `x_i += g[0](x,t) · √h · z_i`   where z ∈ ℝ^{state_dim}
    pub fn euler_maruyama_step(
        &self,
        x: &[f32],
        t: f32,
        dt: f32,
        rng: &mut LcgRng,
    ) -> PinnResult<Vec<f32>> {
        let d = self.cfg.state_dim;
        if x.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        if dt <= 0.0 || !dt.is_finite() {
            return Err(PinnError::InvalidStepSize { h: dt });
        }

        let f_val = self.drift(x, t)?;
        let g_val = self.diffusion(x, t)?;
        let sqrt_dt = dt.sqrt();
        let z = self.sample_noise(rng);

        let mut x_next = vec![0.0_f32; d];
        match self.cfg.noise_type {
            NoiseType::Diagonal => {
                for i in 0..d {
                    x_next[i] = x[i] + f_val[i] * dt + g_val[i] * sqrt_dt * z[i];
                }
            }
            NoiseType::Scalar => {
                // g_val has length 1; z is state_dim-length
                let g0 = g_val[0];
                // Need state_dim-dimensional noise for scalar case.
                let mut z_state = vec![0.0_f32; d];
                rng.fill_normal(&mut z_state);
                for i in 0..d {
                    x_next[i] = x[i] + f_val[i] * dt + g0 * sqrt_dt * z_state[i];
                }
            }
        }

        Ok(x_next)
    }

    /// Milstein step for diagonal noise.
    ///
    /// Adds the Itô correction: for each i,
    ///   x_i += g_i·dW_i + 0.5·g_i·(∂g_i/∂x_i)·(dW_i² − dt)
    ///
    /// ∂g_i/∂x_i is estimated via central finite differences with ε = 1e-5.
    ///
    /// Falls back to Euler-Maruyama for `Scalar` noise type.
    pub fn milstein_step(
        &self,
        x: &[f32],
        t: f32,
        dt: f32,
        rng: &mut LcgRng,
    ) -> PinnResult<Vec<f32>> {
        let d = self.cfg.state_dim;
        if x.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        if dt <= 0.0 || !dt.is_finite() {
            return Err(PinnError::InvalidStepSize { h: dt });
        }

        // For Scalar noise, Milstein correction is scalar * identity → same as EM.
        if self.cfg.noise_type == NoiseType::Scalar {
            return self.euler_maruyama_step(x, t, dt, rng);
        }

        let f_val = self.drift(x, t)?;
        let g_val = self.diffusion(x, t)?; // length noise_dim
        let sqrt_dt = dt.sqrt();
        let z = self.sample_noise(rng); // length noise_dim

        // FD epsilon for ∂g_i/∂x_i.
        let fd_eps = 1e-5_f32;
        let two_fd_eps = 2.0 * fd_eps;

        let mut x_next = vec![0.0_f32; d];
        let mut x_fwd = x.to_vec();
        let mut x_bwd = x.to_vec();

        for i in 0..d {
            let dw_i = z[i] * sqrt_dt;
            let g_i = g_val[i];

            // ∂g_i/∂x_i via central FD.
            x_fwd[i] = x[i] + fd_eps;
            x_bwd[i] = x[i] - fd_eps;
            let g_fwd = self.diffusion(&x_fwd, t)?;
            let g_bwd = self.diffusion(&x_bwd, t)?;
            let dg_dxi = (g_fwd[i] - g_bwd[i]) / two_fd_eps;
            x_fwd[i] = x[i];
            x_bwd[i] = x[i];

            // Milstein: x_i += f_i*dt + g_i*dW_i + 0.5*g_i*(∂g_i/∂x_i)*(dW_i² - dt)
            x_next[i] = x[i] + f_val[i] * dt + g_i * dw_i + 0.5 * g_i * dg_dxi * (dw_i * dw_i - dt);
        }

        Ok(x_next)
    }

    /// Sample a full SDE trajectory from t0 to t1.
    ///
    /// Records `n_steps + 1` states (including the initial condition).
    pub fn sample_path(&self, x0: &[f32], rng: &mut LcgRng) -> PinnResult<SdePath> {
        let d = self.cfg.state_dim;
        let n = self.cfg.n_steps;

        if x0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x0.len(),
            });
        }

        let dt = (self.cfg.t1 - self.cfg.t0) / n as f32;
        let n_pts = n + 1;

        let mut times = Vec::with_capacity(n_pts);
        let mut states = Vec::with_capacity(n_pts * d);

        // Record initial condition.
        times.push(self.cfg.t0);
        states.extend_from_slice(x0);

        let mut x_cur: Vec<f32> = x0.to_vec();

        for step in 0..n {
            let t = self.cfg.t0 + step as f32 * dt;
            let x_next = match self.cfg.method {
                SdeMethod::EulerMaruyama => self.euler_maruyama_step(&x_cur, t, dt, rng)?,
                SdeMethod::Milstein => self.milstein_step(&x_cur, t, dt, rng)?,
            };

            let t_next = self.cfg.t0 + (step + 1) as f32 * dt;
            times.push(t_next);
            states.extend_from_slice(&x_next);
            x_cur = x_next;
        }

        Ok(SdePath { times, states })
    }

    /// Sample `n_paths` independent trajectories and compute Welford running
    /// mean and standard deviation of the **final** state.
    ///
    /// Returns `(mean_final, std_final)`, each of length `state_dim`.
    pub fn sample_statistics(
        &self,
        x0: &[f32],
        n_paths: usize,
        rng: &mut LcgRng,
    ) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        let d = self.cfg.state_dim;
        if n_paths == 0 {
            return Err(PinnError::EmptyInput);
        }
        if x0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: x0.len(),
            });
        }

        let n = self.cfg.n_steps;

        // Welford online algorithm for mean and M2 (sum of squared deviations).
        let mut mean = vec![0.0_f32; d];
        let mut m2 = vec![0.0_f32; d];

        for path_idx in 0..n_paths {
            let path = self.sample_path(x0, rng)?;
            // Final state: last d elements.
            let final_state = &path.states[n * d..(n + 1) * d];

            // Welford update.
            let count = (path_idx + 1) as f32;
            for i in 0..d {
                let delta = final_state[i] - mean[i];
                mean[i] += delta / count;
                let delta2 = final_state[i] - mean[i];
                m2[i] += delta * delta2;
            }
        }

        // Population std (biased) for n_paths >= 1.
        let std_dev: Vec<f32> = m2
            .iter()
            .map(|&s| {
                let var = s / n_paths as f32;
                if var > 0.0 { var.sqrt() } else { 0.0 }
            })
            .collect();

        Ok((mean, std_dev))
    }

    /// Approximate ELBO estimate for the SDE path measure.
    ///
    /// Uses the Girsanov / path-measure KL divergence under unit diffusion:
    ///   ELBO ≈ −Σ_t (Δt / 2) · ||f(x_t, t)||²
    ///
    /// This is the negative KL divergence from the reference Brownian path
    /// measure, serving as a lower bound on the log path-space likelihood.
    /// A value closer to 0 (less negative) indicates the drift is closer to 0,
    /// which means less deviation from the reference measure.
    pub fn elbo_estimate(&self, path: &SdePath, _rng: &mut LcgRng) -> PinnResult<f32> {
        let d = self.cfg.state_dim;
        let n = self.cfg.n_steps;

        if path.times.len() < 2 {
            return Err(PinnError::EmptyInput);
        }
        if path.states.len() != (n + 1) * d {
            return Err(PinnError::DimensionMismatch {
                expected: (n + 1) * d,
                got: path.states.len(),
            });
        }

        let mut elbo = 0.0_f32;

        // Sum over all time steps using the stored states.
        for step in 0..n {
            let t = path.times[step];
            let dt = path.times[step + 1] - t;
            let x_t = &path.states[step * d..(step + 1) * d];

            let f_val = self.drift(x_t, t)?;
            // KL contribution: 0.5 * ||f||² * dt (for unit-diffusion reference)
            let f_sq: f32 = f_val.iter().map(|&fi| fi * fi).sum();
            elbo -= 0.5 * f_sq * dt;
        }

        if !elbo.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "elbo_estimate",
            });
        }
        Ok(elbo)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sde_diagonal(state_dim: usize, n_steps: usize) -> NeuralSde {
        let mut rng = LcgRng::new(42);
        let cfg = NeuralSdeConfig {
            state_dim,
            noise_dim: state_dim,
            hidden_width: 8,
            n_layers: 2,
            t0: 0.0,
            t1: 1.0,
            n_steps,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        NeuralSde::new(cfg, &mut rng)
            .expect("NeuralSde construction with valid diagonal config should succeed")
    }

    fn make_sde_scalar(state_dim: usize, noise_dim: usize, n_steps: usize) -> NeuralSde {
        let mut rng = LcgRng::new(7);
        let cfg = NeuralSdeConfig {
            state_dim,
            noise_dim,
            hidden_width: 8,
            n_layers: 1,
            t0: 0.0,
            t1: 1.0,
            n_steps,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Scalar,
        };
        NeuralSde::new(cfg, &mut rng)
            .expect("NeuralSde construction with valid scalar config should succeed")
    }

    fn make_sde_milstein(state_dim: usize, n_steps: usize) -> NeuralSde {
        let mut rng = LcgRng::new(13);
        let cfg = NeuralSdeConfig {
            state_dim,
            noise_dim: state_dim,
            hidden_width: 8,
            n_layers: 2,
            t0: 0.0,
            t1: 1.0,
            n_steps,
            method: SdeMethod::Milstein,
            noise_type: NoiseType::Diagonal,
        };
        NeuralSde::new(cfg, &mut rng)
            .expect("NeuralSde construction with valid Milstein config should succeed")
    }

    // ── Drift / Diffusion shape ──

    #[test]
    fn drift_output_shape() {
        let sde = make_sde_diagonal(3, 10);
        let x = vec![0.1_f32, 0.2, 0.3];
        let f = sde
            .drift(&x, 0.5)
            .expect("drift evaluation with valid state should succeed");
        assert_eq!(f.len(), 3, "drift output should be state_dim");
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diffusion_output_shape_diagonal() {
        let sde = make_sde_diagonal(4, 10);
        let x = vec![0.0_f32; 4];
        let g = sde
            .diffusion(&x, 0.3)
            .expect("diagonal diffusion evaluation with valid state should succeed");
        assert_eq!(
            g.len(),
            4,
            "diagonal diffusion should return noise_dim values"
        );
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn diffusion_output_shape_scalar() {
        let sde = make_sde_scalar(3, 2, 10);
        let x = vec![0.0_f32; 3];
        let g = sde
            .diffusion(&x, 0.5)
            .expect("scalar diffusion evaluation with valid state should succeed");
        assert_eq!(g.len(), 1, "scalar diffusion should return length-1 vec");
        assert!(g[0].is_finite());
    }

    // ── Euler-Maruyama ──

    #[test]
    fn em_step_shape() {
        let sde = make_sde_diagonal(3, 10);
        let mut rng = LcgRng::new(99);
        let x = vec![0.5_f32; 3];
        let x_next = sde
            .euler_maruyama_step(&x, 0.0, 0.1, &mut rng)
            .expect("Euler-Maruyama step with valid state and dt should succeed");
        assert_eq!(x_next.len(), 3, "EM step output should be state_dim");
    }

    #[test]
    fn em_step_finite() {
        let sde = make_sde_diagonal(2, 10);
        let mut rng = LcgRng::new(55);
        let x = vec![1.0_f32, -0.5];
        let x_next = sde
            .euler_maruyama_step(&x, 0.0, 0.01, &mut rng)
            .expect("Euler-Maruyama step with small dt should produce finite output");
        assert!(
            x_next.iter().all(|v| v.is_finite()),
            "EM step must produce finite output"
        );
    }

    // ── Milstein ──

    #[test]
    fn milstein_step_shape() {
        let sde = make_sde_milstein(3, 10);
        let mut rng = LcgRng::new(88);
        let x = vec![0.3_f32; 3];
        let x_next = sde
            .milstein_step(&x, 0.0, 0.1, &mut rng)
            .expect("Milstein step with valid state and dt should succeed");
        assert_eq!(x_next.len(), 3, "Milstein step output should be state_dim");
    }

    #[test]
    fn milstein_step_finite() {
        let sde = make_sde_milstein(2, 10);
        let mut rng = LcgRng::new(77);
        let x = vec![0.0_f32, 1.0];
        let x_next = sde
            .milstein_step(&x, 0.0, 0.05, &mut rng)
            .expect("Milstein step with small dt should produce finite output");
        assert!(
            x_next.iter().all(|v| v.is_finite()),
            "Milstein step must produce finite output"
        );
    }

    // ── sample_path ──

    #[test]
    fn sample_path_shape() {
        let n_steps = 20;
        let state_dim = 3;
        let sde = make_sde_diagonal(state_dim, n_steps);
        let mut rng = LcgRng::new(11);
        let x0 = vec![1.0_f32; state_dim];
        let path = sde
            .sample_path(&x0, &mut rng)
            .expect("sample_path with valid initial condition should succeed");

        assert_eq!(
            path.times.len(),
            n_steps + 1,
            "times should have n_steps+1 entries"
        );
        assert_eq!(
            path.states.len(),
            (n_steps + 1) * state_dim,
            "states should have (n_steps+1)*state_dim entries"
        );
    }

    #[test]
    fn sample_path_starts_at_x0() {
        let sde = make_sde_diagonal(2, 10);
        let mut rng = LcgRng::new(22);
        let x0 = vec![1.5_f32, -0.7];
        let path = sde
            .sample_path(&x0, &mut rng)
            .expect("sample_path for initial state check should succeed");
        // First state should match x0 exactly.
        assert!((path.states[0] - x0[0]).abs() < 1e-7, "states[0] ≠ x0[0]");
        assert!((path.states[1] - x0[1]).abs() < 1e-7, "states[1] ≠ x0[1]");
    }

    #[test]
    fn sample_path_times_correct() {
        let n_steps = 5;
        let sde = make_sde_diagonal(1, n_steps);
        let mut rng = LcgRng::new(33);
        let x0 = vec![0.0_f32];
        let path = sde
            .sample_path(&x0, &mut rng)
            .expect("sample_path for time grid check should succeed");

        assert!(
            (path.times[0] - 0.0_f32).abs() < 1e-6,
            "times[0] should be t0"
        );
        assert!(
            (path.times[n_steps] - 1.0_f32).abs() < 1e-5,
            "times[n_steps] should be t1, got {}",
            path.times[n_steps]
        );
    }

    // ── sample_statistics ──

    #[test]
    fn sample_statistics_shape() {
        let sde = make_sde_diagonal(3, 5);
        let mut rng = LcgRng::new(44);
        let x0 = vec![0.5_f32; 3];
        let (mean, std) = sde
            .sample_statistics(&x0, 10, &mut rng)
            .expect("sample_statistics with 10 paths should succeed");
        assert_eq!(mean.len(), 3, "mean should have state_dim entries");
        assert_eq!(std.len(), 3, "std should have state_dim entries");
    }

    #[test]
    fn sample_statistics_std_non_negative() {
        let sde = make_sde_diagonal(2, 5);
        let mut rng = LcgRng::new(55);
        let x0 = vec![1.0_f32, -1.0];
        let (_, std) = sde
            .sample_statistics(&x0, 20, &mut rng)
            .expect("sample_statistics with 20 paths should succeed");
        assert!(
            std.iter().all(|&s| s >= 0.0),
            "all std values must be non-negative: {:?}",
            std
        );
    }

    // ── ELBO ──

    #[test]
    fn elbo_finite() {
        let sde = make_sde_diagonal(2, 10);
        let mut rng = LcgRng::new(66);
        let x0 = vec![0.5_f32, -0.5];
        let path = sde
            .sample_path(&x0, &mut rng)
            .expect("sample_path for ELBO test should succeed");
        let elbo = sde
            .elbo_estimate(&path, &mut rng)
            .expect("ELBO estimate on valid path should return finite value");
        assert!(elbo.is_finite(), "ELBO estimate must be finite, got {elbo}");
    }

    // ── Error cases ──

    #[test]
    fn err_state_dim_zero() {
        let mut rng = LcgRng::new(1);
        let cfg = NeuralSdeConfig {
            state_dim: 0,
            noise_dim: 2,
            hidden_width: 8,
            n_layers: 1,
            t0: 0.0,
            t1: 1.0,
            n_steps: 10,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        assert!(NeuralSde::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_noise_dim_zero() {
        let mut rng = LcgRng::new(2);
        let cfg = NeuralSdeConfig {
            state_dim: 2,
            noise_dim: 0,
            hidden_width: 8,
            n_layers: 1,
            t0: 0.0,
            t1: 1.0,
            n_steps: 10,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        assert!(NeuralSde::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_n_steps_zero() {
        let mut rng = LcgRng::new(3);
        let cfg = NeuralSdeConfig {
            state_dim: 2,
            noise_dim: 2,
            hidden_width: 8,
            n_layers: 1,
            t0: 0.0,
            t1: 1.0,
            n_steps: 0,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        assert!(NeuralSde::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_invalid_time() {
        let mut rng = LcgRng::new(4);
        // t0 == t1
        let cfg = NeuralSdeConfig {
            state_dim: 2,
            noise_dim: 2,
            hidden_width: 8,
            n_layers: 1,
            t0: 1.0,
            t1: 0.5,
            n_steps: 10,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        assert!(NeuralSde::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_equal_times() {
        let mut rng = LcgRng::new(5);
        let cfg = NeuralSdeConfig {
            state_dim: 1,
            noise_dim: 1,
            hidden_width: 4,
            n_layers: 1,
            t0: 0.5,
            t1: 0.5,
            n_steps: 5,
            method: SdeMethod::EulerMaruyama,
            noise_type: NoiseType::Diagonal,
        };
        assert!(NeuralSde::new(cfg, &mut rng).is_err());
    }
}
