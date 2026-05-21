//! Hamiltonian Neural Networks (HNN) and Lagrangian Neural Networks (LNN).
//!
//! - **HNN** (Greydanus et al. NeurIPS 2019): learns H(q,p) such that
//!   Hamilton's equations dq/dt = ∂H/∂p, dp/dt = -∂H/∂q automatically hold.
//! - **LNN** (Cranmer et al. NeurIPS 2020): learns L(q, q̇); Euler-Lagrange
//!   equations M(q)q̈ = ∇_q L - (∂²L/∂q∂q̇)q̇ give the equations of motion.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

// ─── Shared weight container ─────────────────────────────────────────────────

/// Weight storage shared by both HNN and LNN.
///
/// Stores layer weight matrices and bias vectors as flat row-major `Vec<f32>`.
/// For a layer mapping `d_in → d_out`: `layers[i]` has `d_out * d_in` entries,
/// `biases[i]` has `d_out` entries.
#[derive(Debug, Clone)]
pub struct HnnWeights {
    pub layers: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
}

// ─── Internal MLP helpers ────────────────────────────────────────────────────

/// Apply one linear layer: y = W x + b.
///
/// `w`: row-major `[d_out × d_in]`; `b`: `[d_out]`; `x`: `[d_in]`.
fn linear_fwd(w: &[f32], b: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
    let d_in = x.len();
    (0..d_out)
        .map(|i| {
            let sum: f32 = (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum();
            sum + b[i]
        })
        .collect()
}

/// Run the MLP defined by `weights` on `input`.
/// Hidden layers use tanh; output layer is linear (returns scalar).
fn mlp_forward(weights: &HnnWeights, input: &[f32]) -> PinnResult<f32> {
    let n_layers = weights.layers.len();
    if n_layers == 0 {
        return Err(PinnError::InvalidNetworkDepth { depth: 0 });
    }

    // Infer d_out for each layer from bias vector length.
    let mut act: Vec<f32> = input.to_vec();
    for (layer_idx, (w, b)) in weights.layers.iter().zip(weights.biases.iter()).enumerate() {
        let d_out = b.len();
        act = linear_fwd(w, b, &act, d_out);
        // Apply tanh to all hidden layers; leave output layer (last) linear.
        if layer_idx + 1 < n_layers {
            for v in &mut act {
                *v = v.tanh();
            }
        }
    }
    // Output is 1-dimensional scalar.
    if act.len() != 1 {
        return Err(PinnError::DimensionMismatch {
            expected: 1,
            got: act.len(),
        });
    }
    let out = act[0];
    if !out.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "mlp_forward output",
        });
    }
    Ok(out)
}

/// Kaiming uniform initialisation: U(-√(6/fan_in), +√(6/fan_in)).
fn kaiming_uniform(fan_in: usize, n: usize, rng: &mut LcgRng) -> Vec<f32> {
    let bound = (6.0_f32 / fan_in as f32).sqrt();
    (0..n)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// Build MLP weight/bias vectors for the network:
///   input_dim → [hidden_width × n_layers] → 1 (scalar output)
fn build_mlp_weights(
    input_dim: usize,
    hidden_width: usize,
    n_layers: usize,
    rng: &mut LcgRng,
) -> HnnWeights {
    let mut layers = Vec::new();
    let mut biases = Vec::new();

    // First layer: input_dim → hidden_width
    layers.push(kaiming_uniform(input_dim, hidden_width * input_dim, rng));
    biases.push(vec![0.0_f32; hidden_width]);

    // Intermediate hidden layers
    for _ in 1..n_layers {
        layers.push(kaiming_uniform(
            hidden_width,
            hidden_width * hidden_width,
            rng,
        ));
        biases.push(vec![0.0_f32; hidden_width]);
    }

    // Output layer: hidden_width → 1
    layers.push(kaiming_uniform(hidden_width, hidden_width, rng));
    biases.push(vec![0.0_f32]);

    HnnWeights { layers, biases }
}

// ─── HNN ─────────────────────────────────────────────────────────────────────

/// Configuration for a Hamiltonian Neural Network.
#[derive(Debug, Clone)]
pub struct HnnConfig {
    /// Dimension of q (position); also dimension of p (momentum).
    /// Total MLP input is `2 * state_dim`.
    pub state_dim: usize,
    /// Width of each hidden layer.
    pub hidden_width: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Finite-difference step for gradient computation (e.g. 1e-4).
    pub fd_epsilon: f32,
}

/// A Hamiltonian Neural Network that learns H(q, p).
pub struct HamiltonianNn {
    pub cfg: HnnConfig,
    pub weights: HnnWeights,
}

/// Trajectory produced by leapfrog integration.
pub struct HnnTrajectory {
    /// Time at each step (length `n_steps`).
    pub times: Vec<f32>,
    /// Phase-space positions q — row-major `[n_steps × state_dim]`.
    pub q: Vec<f32>,
    /// Phase-space momenta p — row-major `[n_steps × state_dim]`.
    pub p: Vec<f32>,
    /// Hamiltonian evaluated at each step (length `n_steps`).
    pub energy: Vec<f32>,
}

impl HamiltonianNn {
    /// Construct a randomly initialised HNN.
    pub fn new(cfg: HnnConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if cfg.state_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.hidden_width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.n_layers == 0 {
            return Err(PinnError::InvalidNetworkDepth { depth: 0 });
        }
        if cfg.fd_epsilon <= 0.0 || !cfg.fd_epsilon.is_finite() {
            return Err(PinnError::InvalidStepSize { h: cfg.fd_epsilon });
        }

        let input_dim = 2 * cfg.state_dim;
        let weights = build_mlp_weights(input_dim, cfg.hidden_width, cfg.n_layers, rng);

        Ok(Self { cfg, weights })
    }

    /// Forward pass through the Hamiltonian MLP.
    ///
    /// Concatenates `q` and `p` into a `2 * state_dim` vector and returns H(q, p).
    pub fn forward_mlp(&self, input: &[f32]) -> PinnResult<f32> {
        let expected = 2 * self.cfg.state_dim;
        if input.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: input.len(),
            });
        }
        mlp_forward(&self.weights, input)
    }

    /// Evaluate H(q, p) — scalar Hamiltonian.
    pub fn hamiltonian(&self, q: &[f32], p: &[f32]) -> PinnResult<f32> {
        let d = self.cfg.state_dim;
        if q.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q.len(),
            });
        }
        if p.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: p.len(),
            });
        }
        let input: Vec<f32> = q.iter().chain(p.iter()).copied().collect();
        mlp_forward(&self.weights, &input)
    }

    /// Central-difference gradients: ∂H/∂q and ∂H/∂p.
    ///
    /// Uses `fd_epsilon` for the perturbation step.
    pub fn hamiltonian_grad(&self, q: &[f32], p: &[f32]) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        let d = self.cfg.state_dim;
        if q.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q.len(),
            });
        }
        if p.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: p.len(),
            });
        }

        let eps = self.cfg.fd_epsilon;
        let two_eps = 2.0 * eps;

        let mut grad_q = vec![0.0_f32; d];
        let mut grad_p = vec![0.0_f32; d];

        // ∂H/∂q_i
        let mut q_fwd = q.to_vec();
        let mut q_bwd = q.to_vec();
        for i in 0..d {
            q_fwd[i] = q[i] + eps;
            q_bwd[i] = q[i] - eps;
            let h_fwd = self.hamiltonian(&q_fwd, p)?;
            let h_bwd = self.hamiltonian(&q_bwd, p)?;
            grad_q[i] = (h_fwd - h_bwd) / two_eps;
            q_fwd[i] = q[i];
            q_bwd[i] = q[i];
        }

        // ∂H/∂p_i
        let mut p_fwd = p.to_vec();
        let mut p_bwd = p.to_vec();
        for i in 0..d {
            p_fwd[i] = p[i] + eps;
            p_bwd[i] = p[i] - eps;
            let h_fwd = self.hamiltonian(q, &p_fwd)?;
            let h_bwd = self.hamiltonian(q, &p_bwd)?;
            grad_p[i] = (h_fwd - h_bwd) / two_eps;
            p_fwd[i] = p[i];
            p_bwd[i] = p[i];
        }

        Ok((grad_q, grad_p))
    }

    /// Hamilton's equations of motion.
    ///
    /// Returns `(dq/dt, dp/dt) = (∂H/∂p, -∂H/∂q)`.
    pub fn time_derivative(&self, q: &[f32], p: &[f32]) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        let (grad_q, grad_p) = self.hamiltonian_grad(q, p)?;
        let dq_dt = grad_p;
        let dp_dt: Vec<f32> = grad_q.iter().map(|&v| -v).collect();
        Ok((dq_dt, dp_dt))
    }

    /// Störmer-Verlet (leapfrog) symplectic integrator.
    ///
    /// The leapfrog scheme exactly conserves a modified Hamiltonian, giving
    /// far better long-time energy behaviour than RK4 for Hamiltonian systems.
    ///
    /// Each step:
    /// 1. p_{½} = p_n − (h/2) ∂H/∂q(q_n, p_n)
    /// 2. q_{n+1} = q_n + h ∂H/∂p(q_n, p_{½})
    /// 3. p_{n+1} = p_{½} − (h/2) ∂H/∂q(q_{n+1}, p_{½})
    pub fn integrate_leapfrog(
        &self,
        q0: &[f32],
        p0: &[f32],
        t_span: (f32, f32),
        n_steps: usize,
    ) -> PinnResult<HnnTrajectory> {
        let d = self.cfg.state_dim;
        if q0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q0.len(),
            });
        }
        if p0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: p0.len(),
            });
        }
        if t_span.0 >= t_span.1 {
            return Err(PinnError::InvalidTimeInterval {
                t0: t_span.0,
                t1: t_span.1,
            });
        }
        if n_steps == 0 {
            return Err(PinnError::InvalidStepSize { h: 0.0 });
        }

        let dt = (t_span.1 - t_span.0) / n_steps as f32;
        let half_dt = 0.5 * dt;

        let mut times = Vec::with_capacity(n_steps);
        let mut q_traj = Vec::with_capacity(n_steps * d);
        let mut p_traj = Vec::with_capacity(n_steps * d);
        let mut energy = Vec::with_capacity(n_steps);

        let mut q_cur: Vec<f32> = q0.to_vec();
        let mut p_cur: Vec<f32> = p0.to_vec();

        for step in 0..n_steps {
            let t = t_span.0 + step as f32 * dt;

            // Record current state.
            times.push(t);
            q_traj.extend_from_slice(&q_cur);
            p_traj.extend_from_slice(&p_cur);
            let h_val = self.hamiltonian(&q_cur, &p_cur)?;
            energy.push(h_val);

            // Step 1: half-step momentum using ∂H/∂q at (q_n, p_n).
            let (grad_q_n, _) = self.hamiltonian_grad(&q_cur, &p_cur)?;
            let p_half: Vec<f32> = p_cur
                .iter()
                .zip(grad_q_n.iter())
                .map(|(&pi, &dh_qi)| pi - half_dt * dh_qi)
                .collect();

            // Step 2: full-step position using ∂H/∂p at (q_n, p_{½}).
            let (_, grad_p_half) = self.hamiltonian_grad(&q_cur, &p_half)?;
            let q_next: Vec<f32> = q_cur
                .iter()
                .zip(grad_p_half.iter())
                .map(|(&qi, &dh_pi)| qi + dt * dh_pi)
                .collect();

            // Step 3: half-step momentum using ∂H/∂q at (q_{n+1}, p_{½}).
            let (grad_q_next, _) = self.hamiltonian_grad(&q_next, &p_half)?;
            let p_next: Vec<f32> = p_half
                .iter()
                .zip(grad_q_next.iter())
                .map(|(&pi, &dh_qi)| pi - half_dt * dh_qi)
                .collect();

            q_cur = q_next;
            p_cur = p_next;
        }

        Ok(HnnTrajectory {
            times,
            q: q_traj,
            p: p_traj,
            energy,
        })
    }

    /// HNN training loss: mean squared error between predicted and observed derivatives.
    ///
    /// Loss = (1/n) Σ_i ( ||dq/dt_i − ∂H/∂p_i||² + ||dp/dt_i − (-∂H/∂q_i)||² )
    pub fn hnn_loss(
        &self,
        q: &[f32],
        p: &[f32],
        dq_dt: &[f32],
        dp_dt: &[f32],
        n_points: usize,
    ) -> PinnResult<f32> {
        let d = self.cfg.state_dim;

        if n_points == 0 {
            return Err(PinnError::EmptyInput);
        }
        if q.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: q.len(),
            });
        }
        if p.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: p.len(),
            });
        }
        if dq_dt.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: dq_dt.len(),
            });
        }
        if dp_dt.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: dp_dt.len(),
            });
        }

        let mut loss_sum = 0.0_f32;
        for pt in 0..n_points {
            let qi = &q[pt * d..(pt + 1) * d];
            let pi = &p[pt * d..(pt + 1) * d];
            let dq_true = &dq_dt[pt * d..(pt + 1) * d];
            let dp_true = &dp_dt[pt * d..(pt + 1) * d];

            let (grad_qi, grad_pi) = self.hamiltonian_grad(qi, pi)?;
            // Predicted: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            let sq_err: f32 = grad_pi
                .iter()
                .zip(dq_true.iter())
                .map(|(&pred, &true_v)| (pred - true_v).powi(2))
                .sum::<f32>()
                + grad_qi
                    .iter()
                    .zip(dp_true.iter())
                    .map(|(&dh_q, &dp_true_v)| (-dh_q - dp_true_v).powi(2))
                    .sum::<f32>();

            loss_sum += sq_err;
        }

        let loss = loss_sum / (n_points as f32 * d as f32 * 2.0);
        if !loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "hnn_loss",
            });
        }
        Ok(loss)
    }
}

// ─── LNN ─────────────────────────────────────────────────────────────────────

/// Configuration for a Lagrangian Neural Network.
#[derive(Debug, Clone)]
pub struct LnnConfig {
    /// Configuration-space dimension (dimension of q and q̇).
    pub q_dim: usize,
    /// Width of each hidden layer.
    pub hidden_width: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Finite-difference step for gradient/Hessian computation.
    pub fd_epsilon: f32,
}

/// A Lagrangian Neural Network that learns L(q, q̇).
pub struct LagrangianNn {
    pub cfg: LnnConfig,
    pub weights: HnnWeights,
}

/// Trajectory produced by RK4 integration of the Euler-Lagrange equations.
pub struct LnnTrajectory {
    /// Time at each step (length `n_steps`).
    pub times: Vec<f32>,
    /// Configuration — row-major `[n_steps × q_dim]`.
    pub q: Vec<f32>,
    /// Velocity — row-major `[n_steps × q_dim]`.
    pub q_dot: Vec<f32>,
    /// Lagrangian evaluated at each step (length `n_steps`).
    pub lagrangian: Vec<f32>,
}

impl LagrangianNn {
    /// Construct a randomly initialised LNN.
    pub fn new(cfg: LnnConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if cfg.q_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.hidden_width == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.n_layers == 0 {
            return Err(PinnError::InvalidNetworkDepth { depth: 0 });
        }
        if cfg.fd_epsilon <= 0.0 || !cfg.fd_epsilon.is_finite() {
            return Err(PinnError::InvalidStepSize { h: cfg.fd_epsilon });
        }

        let input_dim = 2 * cfg.q_dim;
        let weights = build_mlp_weights(input_dim, cfg.hidden_width, cfg.n_layers, rng);

        Ok(Self { cfg, weights })
    }

    /// Evaluate L(q, q̇) — scalar Lagrangian.
    pub fn lagrangian(&self, q: &[f32], q_dot: &[f32]) -> PinnResult<f32> {
        let d = self.cfg.q_dim;
        if q.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q.len(),
            });
        }
        if q_dot.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q_dot.len(),
            });
        }
        let input: Vec<f32> = q.iter().chain(q_dot.iter()).copied().collect();
        mlp_forward(&self.weights, &input)
    }

    /// Diagonal mass matrix M(q) = diag(∂²L/∂q̇²) via second-order FD.
    ///
    /// M_ii ≈ (L(q, q̇ + ε e_i) − 2L(q, q̇) + L(q, q̇ − ε e_i)) / ε²
    ///
    /// Returns a `Vec<f32>` of length `q_dim` (diagonal entries only).
    pub fn mass_matrix(&self, q: &[f32], q_dot: &[f32]) -> PinnResult<Vec<f32>> {
        let d = self.cfg.q_dim;
        if q.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q.len(),
            });
        }
        if q_dot.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q_dot.len(),
            });
        }

        let eps = self.cfg.fd_epsilon;
        let eps_sq = eps * eps;
        let l0 = self.lagrangian(q, q_dot)?;
        let mut mass_diag = vec![0.0_f32; d];

        let mut qd_fwd = q_dot.to_vec();
        let mut qd_bwd = q_dot.to_vec();

        for i in 0..d {
            qd_fwd[i] = q_dot[i] + eps;
            qd_bwd[i] = q_dot[i] - eps;
            let l_fwd = self.lagrangian(q, &qd_fwd)?;
            let l_bwd = self.lagrangian(q, &qd_bwd)?;
            mass_diag[i] = (l_fwd - 2.0 * l0 + l_bwd) / eps_sq;
            qd_fwd[i] = q_dot[i];
            qd_bwd[i] = q_dot[i];
        }

        Ok(mass_diag)
    }

    /// Euler-Lagrange equations of motion.
    ///
    /// Computes q̈ via the diagonal-mass-matrix approximation:
    ///   M(q)q̈ = ∇_q L − (∂²L/∂q∂q̇) q̇
    /// where M is diagonal (from `mass_matrix`).
    ///
    /// Gradients are obtained by central finite differences.
    pub fn equations_of_motion(&self, q: &[f32], q_dot: &[f32]) -> PinnResult<Vec<f32>> {
        let d = self.cfg.q_dim;
        if q.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q.len(),
            });
        }
        if q_dot.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q_dot.len(),
            });
        }

        let eps = self.cfg.fd_epsilon;
        let two_eps = 2.0 * eps;

        // ∇_q L: ∂L/∂q_i via central FD.
        let mut grad_q_l = vec![0.0_f32; d];
        let mut q_fwd = q.to_vec();
        let mut q_bwd = q.to_vec();
        for i in 0..d {
            q_fwd[i] = q[i] + eps;
            q_bwd[i] = q[i] - eps;
            let l_fwd = self.lagrangian(&q_fwd, q_dot)?;
            let l_bwd = self.lagrangian(&q_bwd, q_dot)?;
            grad_q_l[i] = (l_fwd - l_bwd) / two_eps;
            q_fwd[i] = q[i];
            q_bwd[i] = q[i];
        }

        // Coriolis / coupling term: Σ_j (∂²L/∂q_i ∂q̇_j) q̇_j
        // Mixed partial: central FD: ∂²L/(∂q_i ∂q̇_j) ≈
        //   [L(q+ε e_i, q̇+ε e_j) - L(q-ε e_i, q̇+ε e_j)
        //    - L(q+ε e_i, q̇-ε e_j) + L(q-ε e_i, q̇-ε e_j)] / (4 ε²)
        let mut coupling = vec![0.0_f32; d];
        let four_eps_sq = 4.0 * eps * eps;

        let mut q_pp = q.to_vec();
        let mut q_pm = q.to_vec();
        let mut q_mp = q.to_vec();
        let mut q_mm = q.to_vec();
        let mut qd_pp = q_dot.to_vec();
        let mut qd_pm = q_dot.to_vec();
        let mut qd_mp = q_dot.to_vec();
        let mut qd_mm = q_dot.to_vec();

        for i in 0..d {
            let mut sum_j = 0.0_f32;
            q_pp[i] = q[i] + eps;
            q_pm[i] = q[i] + eps;
            q_mp[i] = q[i] - eps;
            q_mm[i] = q[i] - eps;

            for j in 0..d {
                qd_pp[j] = q_dot[j] + eps;
                qd_pm[j] = q_dot[j] - eps;
                qd_mp[j] = q_dot[j] + eps;
                qd_mm[j] = q_dot[j] - eps;

                let l_pp = self.lagrangian(&q_pp, &qd_pp)?;
                let l_pm = self.lagrangian(&q_pm, &qd_pm)?;
                let l_mp = self.lagrangian(&q_mp, &qd_mp)?;
                let l_mm = self.lagrangian(&q_mm, &qd_mm)?;

                let mixed = (l_pp - l_pm - l_mp + l_mm) / four_eps_sq;
                sum_j += mixed * q_dot[j];

                qd_pp[j] = q_dot[j];
                qd_pm[j] = q_dot[j];
                qd_mp[j] = q_dot[j];
                qd_mm[j] = q_dot[j];
            }
            coupling[i] = sum_j;

            q_pp[i] = q[i];
            q_pm[i] = q[i];
            q_mp[i] = q[i];
            q_mm[i] = q[i];
        }

        // M(q) diagonal
        let mass_diag = self.mass_matrix(q, q_dot)?;

        // q̈_i = (∇_q L_i - coupling_i) / M_ii
        let mut q_ddot = vec![0.0_f32; d];
        for i in 0..d {
            let m = mass_diag[i];
            // If mass is very small, regularise to avoid div-by-zero.
            let m_reg = if m.abs() < 1e-8 {
                // preserve sign if available, else use positive regulariser
                if m >= 0.0 { 1e-8 } else { -1e-8 }
            } else {
                m
            };
            q_ddot[i] = (grad_q_l[i] - coupling[i]) / m_reg;
            if !q_ddot[i].is_finite() {
                q_ddot[i] = 0.0;
            }
        }

        Ok(q_ddot)
    }

    /// RK4 integration of the Euler-Lagrange equations.
    ///
    /// State = (q, q̇); derivative = (q̇, q̈) where q̈ comes from `equations_of_motion`.
    pub fn integrate_rk4(
        &self,
        q0: &[f32],
        q_dot0: &[f32],
        t_span: (f32, f32),
        n_steps: usize,
    ) -> PinnResult<LnnTrajectory> {
        let d = self.cfg.q_dim;
        if q0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q0.len(),
            });
        }
        if q_dot0.len() != d {
            return Err(PinnError::DimensionMismatch {
                expected: d,
                got: q_dot0.len(),
            });
        }
        if t_span.0 >= t_span.1 {
            return Err(PinnError::InvalidTimeInterval {
                t0: t_span.0,
                t1: t_span.1,
            });
        }
        if n_steps == 0 {
            return Err(PinnError::InvalidStepSize { h: 0.0 });
        }

        let dt = (t_span.1 - t_span.0) / n_steps as f32;

        let mut times = Vec::with_capacity(n_steps);
        let mut q_traj = Vec::with_capacity(n_steps * d);
        let mut qd_traj = Vec::with_capacity(n_steps * d);
        let mut lag_traj = Vec::with_capacity(n_steps);

        let mut q_cur: Vec<f32> = q0.to_vec();
        let mut qd_cur: Vec<f32> = q_dot0.to_vec();

        /// Compute the RHS of (q, q̇) system for RK4.
        fn rhs(lnn: &LagrangianNn, q: &[f32], qd: &[f32]) -> PinnResult<(Vec<f32>, Vec<f32>)> {
            let q_ddot = lnn.equations_of_motion(q, qd)?;
            Ok((qd.to_vec(), q_ddot))
        }

        for step in 0..n_steps {
            let t = t_span.0 + step as f32 * dt;

            times.push(t);
            q_traj.extend_from_slice(&q_cur);
            qd_traj.extend_from_slice(&qd_cur);
            let l_val = self.lagrangian(&q_cur, &qd_cur)?;
            lag_traj.push(l_val);

            // k1
            let (dq1, dqd1) = rhs(self, &q_cur, &qd_cur)?;

            // k2
            let q_mid1: Vec<f32> = q_cur
                .iter()
                .zip(dq1.iter())
                .map(|(&qi, &k)| qi + 0.5 * dt * k)
                .collect();
            let qd_mid1: Vec<f32> = qd_cur
                .iter()
                .zip(dqd1.iter())
                .map(|(&qdi, &k)| qdi + 0.5 * dt * k)
                .collect();
            let (dq2, dqd2) = rhs(self, &q_mid1, &qd_mid1)?;

            // k3
            let q_mid2: Vec<f32> = q_cur
                .iter()
                .zip(dq2.iter())
                .map(|(&qi, &k)| qi + 0.5 * dt * k)
                .collect();
            let qd_mid2: Vec<f32> = qd_cur
                .iter()
                .zip(dqd2.iter())
                .map(|(&qdi, &k)| qdi + 0.5 * dt * k)
                .collect();
            let (dq3, dqd3) = rhs(self, &q_mid2, &qd_mid2)?;

            // k4
            let q_end: Vec<f32> = q_cur
                .iter()
                .zip(dq3.iter())
                .map(|(&qi, &k)| qi + dt * k)
                .collect();
            let qd_end: Vec<f32> = qd_cur
                .iter()
                .zip(dqd3.iter())
                .map(|(&qdi, &k)| qdi + dt * k)
                .collect();
            let (dq4, dqd4) = rhs(self, &q_end, &qd_end)?;

            // Combine
            q_cur = q_cur
                .iter()
                .enumerate()
                .map(|(i, &qi)| qi + dt / 6.0 * (dq1[i] + 2.0 * dq2[i] + 2.0 * dq3[i] + dq4[i]))
                .collect();
            qd_cur = qd_cur
                .iter()
                .enumerate()
                .map(|(i, &qdi)| {
                    qdi + dt / 6.0 * (dqd1[i] + 2.0 * dqd2[i] + 2.0 * dqd3[i] + dqd4[i])
                })
                .collect();
        }

        Ok(LnnTrajectory {
            times,
            q: q_traj,
            q_dot: qd_traj,
            lagrangian: lag_traj,
        })
    }

    /// LNN training loss: MSE between predicted and true q̈.
    ///
    /// Loss = (1/n) Σ_i ||q̈_predicted_i - q̈_true_i||²
    pub fn lnn_loss(
        &self,
        q: &[f32],
        q_dot: &[f32],
        q_ddot: &[f32],
        n_points: usize,
    ) -> PinnResult<f32> {
        let d = self.cfg.q_dim;

        if n_points == 0 {
            return Err(PinnError::EmptyInput);
        }
        if q.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: q.len(),
            });
        }
        if q_dot.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: q_dot.len(),
            });
        }
        if q_ddot.len() != n_points * d {
            return Err(PinnError::DimensionMismatch {
                expected: n_points * d,
                got: q_ddot.len(),
            });
        }

        let mut loss_sum = 0.0_f32;
        for pt in 0..n_points {
            let qi = &q[pt * d..(pt + 1) * d];
            let qdi = &q_dot[pt * d..(pt + 1) * d];
            let qdd_true = &q_ddot[pt * d..(pt + 1) * d];

            let qdd_pred = self.equations_of_motion(qi, qdi)?;
            let sq_err: f32 = qdd_pred
                .iter()
                .zip(qdd_true.iter())
                .map(|(&pred, &true_v)| (pred - true_v).powi(2))
                .sum();
            loss_sum += sq_err;
        }

        let loss = loss_sum / (n_points as f32 * d as f32);
        if !loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "lnn_loss",
            });
        }
        Ok(loss)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hnn(state_dim: usize) -> HamiltonianNn {
        let mut rng = LcgRng::new(42);
        let cfg = HnnConfig {
            state_dim,
            hidden_width: 16,
            n_layers: 2,
            fd_epsilon: 1e-4,
        };
        HamiltonianNn::new(cfg, &mut rng).unwrap()
    }

    fn make_lnn(q_dim: usize) -> LagrangianNn {
        let mut rng = LcgRng::new(42);
        let cfg = LnnConfig {
            q_dim,
            hidden_width: 16,
            n_layers: 2,
            fd_epsilon: 1e-4,
        };
        LagrangianNn::new(cfg, &mut rng).unwrap()
    }

    // ── HNN tests ──

    #[test]
    fn hamiltonian_output_scalar() {
        let hnn = make_hnn(2);
        let q = vec![0.5_f32, -0.3];
        let p = vec![0.1_f32, 0.7];
        let h = hnn.hamiltonian(&q, &p).unwrap();
        assert!(h.is_finite(), "H should be finite, got {h}");
    }

    #[test]
    fn hamiltonian_grad_shape() {
        let hnn = make_hnn(3);
        let q = vec![0.1_f32; 3];
        let p = vec![0.2_f32; 3];
        let (grad_q, grad_p) = hnn.hamiltonian_grad(&q, &p).unwrap();
        assert_eq!(grad_q.len(), 3, "grad_q wrong length");
        assert_eq!(grad_p.len(), 3, "grad_p wrong length");
        assert!(grad_q.iter().all(|v| v.is_finite()));
        assert!(grad_p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn time_derivative_shape() {
        let hnn = make_hnn(4);
        let q = vec![0.0_f32; 4];
        let p = vec![0.5_f32; 4];
        let (dq, dp) = hnn.time_derivative(&q, &p).unwrap();
        assert_eq!(dq.len(), 4);
        assert_eq!(dp.len(), 4);
    }

    #[test]
    fn leapfrog_shape() {
        let hnn = make_hnn(2);
        let q0 = vec![1.0_f32, 0.0];
        let p0 = vec![0.0_f32, 1.0];
        let n_steps = 10;
        let traj = hnn
            .integrate_leapfrog(&q0, &p0, (0.0, 1.0), n_steps)
            .unwrap();
        assert_eq!(traj.times.len(), n_steps);
        assert_eq!(traj.q.len(), n_steps * 2);
        assert_eq!(traj.p.len(), n_steps * 2);
        assert_eq!(traj.energy.len(), n_steps);
    }

    #[test]
    fn leapfrog_energy_approximately_conserved() {
        // Use a simple harmonic oscillator:
        // H_exact(q,p) = 0.5*(q^2 + p^2).
        // With a randomly init'd NN, H won't match, but we test
        // that leapfrog itself doesn't blow up (bounded energy drift).
        let hnn = make_hnn(1);
        let q0 = vec![1.0_f32];
        let p0 = vec![0.0_f32];
        let n_steps = 100;
        let traj = hnn
            .integrate_leapfrog(&q0, &p0, (0.0, 1.0), n_steps)
            .unwrap();

        let e0 = traj.energy[0];
        let e_final = traj.energy[n_steps - 1];
        // Energy should stay finite and not drift more than 10× the initial.
        assert!(e0.is_finite(), "Initial energy not finite");
        assert!(e_final.is_finite(), "Final energy not finite");
        // With leapfrog the modified energy is conserved; absolute drift should be bounded.
        let drift = (e_final - e0).abs();
        assert!(
            drift < 5.0 * e0.abs().max(1.0),
            "Energy drift too large: e0={e0}, e_final={e_final}, drift={drift}"
        );
    }

    #[test]
    fn hnn_loss_zero_when_correct() {
        // Provide predicted derivatives as the ground truth → loss should be ~0.
        let hnn = make_hnn(2);
        let q = vec![0.3_f32, -0.2, 0.7, 0.1];
        let p = vec![0.1_f32, 0.4, -0.3, 0.5];

        // Compute what the HNN thinks the derivatives are.
        let mut dq_true = vec![0.0_f32; 4];
        let mut dp_true = vec![0.0_f32; 4];
        for pt in 0..2 {
            let qi = &q[pt * 2..(pt + 1) * 2];
            let pi = &p[pt * 2..(pt + 1) * 2];
            let (dq, dp) = hnn.time_derivative(qi, pi).unwrap();
            dq_true[pt * 2..pt * 2 + 2].copy_from_slice(&dq);
            dp_true[pt * 2..pt * 2 + 2].copy_from_slice(&dp);
        }

        let loss = hnn.hnn_loss(&q, &p, &dq_true, &dp_true, 2).unwrap();
        assert!(
            loss < 1e-6,
            "HNN loss should be ~0 when using exact derivatives, got {loss}"
        );
    }

    #[test]
    fn forward_mlp_shape() {
        let hnn = make_hnn(2);
        let input = vec![0.1_f32, 0.2, 0.3, 0.4]; // 2*state_dim
        let out = hnn.forward_mlp(&input).unwrap();
        assert!(out.is_finite(), "forward_mlp should return finite f32");
    }

    // ── LNN tests ──

    #[test]
    fn lagrangian_output_scalar() {
        let lnn = make_lnn(3);
        let q = vec![0.5_f32, -0.1, 0.3];
        let q_dot = vec![1.0_f32, -0.5, 0.2];
        let l = lnn.lagrangian(&q, &q_dot).unwrap();
        assert!(l.is_finite(), "L should be finite, got {l}");
    }

    #[test]
    fn mass_matrix_shape() {
        let lnn = make_lnn(3);
        let q = vec![0.1_f32; 3];
        let q_dot = vec![0.2_f32; 3];
        let m = lnn.mass_matrix(&q, &q_dot).unwrap();
        assert_eq!(
            m.len(),
            3,
            "mass_matrix should return q_dim entries (diagonal)"
        );
        assert!(m.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn equations_of_motion_shape() {
        let lnn = make_lnn(2);
        let q = vec![0.4_f32, -0.2];
        let q_dot = vec![0.1_f32, 0.3];
        let qdd = lnn.equations_of_motion(&q, &q_dot).unwrap();
        assert_eq!(
            qdd.len(),
            2,
            "equations_of_motion should return q_dim entries"
        );
    }

    #[test]
    fn integrate_rk4_shape() {
        let lnn = make_lnn(2);
        let q0 = vec![1.0_f32, 0.0];
        let qd0 = vec![0.0_f32, 0.5];
        let n_steps = 8;
        let traj = lnn.integrate_rk4(&q0, &qd0, (0.0, 1.0), n_steps).unwrap();
        assert_eq!(traj.times.len(), n_steps);
        assert_eq!(traj.q.len(), n_steps * 2);
        assert_eq!(traj.q_dot.len(), n_steps * 2);
        assert_eq!(traj.lagrangian.len(), n_steps);
    }

    #[test]
    fn lnn_loss_non_negative() {
        let lnn = make_lnn(2);
        let q = vec![0.3_f32, -0.2, 0.7, 0.1];
        let q_dot = vec![0.1_f32, 0.4, -0.3, 0.5];
        let q_ddot = vec![0.0_f32; 4];
        let loss = lnn.lnn_loss(&q, &q_dot, &q_ddot, 2).unwrap();
        assert!(loss >= 0.0, "LNN loss must be non-negative, got {loss}");
    }

    // ── Error cases ──

    #[test]
    fn err_state_dim_zero() {
        let mut rng = LcgRng::new(1);
        let cfg = HnnConfig {
            state_dim: 0,
            hidden_width: 8,
            n_layers: 1,
            fd_epsilon: 1e-4,
        };
        assert!(HamiltonianNn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_hidden_width_zero() {
        let mut rng = LcgRng::new(1);
        let cfg = HnnConfig {
            state_dim: 2,
            hidden_width: 0,
            n_layers: 1,
            fd_epsilon: 1e-4,
        };
        assert!(HamiltonianNn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_fd_epsilon_negative() {
        let mut rng = LcgRng::new(1);
        let cfg = HnnConfig {
            state_dim: 2,
            hidden_width: 8,
            n_layers: 1,
            fd_epsilon: -1e-4,
        };
        assert!(HamiltonianNn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_fd_epsilon_zero() {
        let mut rng = LcgRng::new(1);
        let cfg = HnnConfig {
            state_dim: 2,
            hidden_width: 8,
            n_layers: 1,
            fd_epsilon: 0.0,
        };
        assert!(HamiltonianNn::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_invalid_t_span() {
        let hnn = make_hnn(1);
        let q0 = vec![0.0_f32];
        let p0 = vec![1.0_f32];
        // t0 == t1
        assert!(hnn.integrate_leapfrog(&q0, &p0, (1.0, 0.5), 10).is_err());
    }

    #[test]
    fn err_n_steps_zero() {
        let hnn = make_hnn(1);
        let q0 = vec![0.0_f32];
        let p0 = vec![1.0_f32];
        assert!(hnn.integrate_leapfrog(&q0, &p0, (0.0, 1.0), 0).is_err());
    }

    #[test]
    fn lnn_err_n_steps_zero() {
        let lnn = make_lnn(1);
        let q0 = vec![1.0_f32];
        let qd0 = vec![0.0_f32];
        assert!(lnn.integrate_rk4(&q0, &qd0, (0.0, 1.0), 0).is_err());
    }

    #[test]
    fn lnn_err_invalid_t_span() {
        let lnn = make_lnn(1);
        let q0 = vec![1.0_f32];
        let qd0 = vec![0.0_f32];
        assert!(lnn.integrate_rk4(&q0, &qd0, (2.0, 1.0), 10).is_err());
    }
}
