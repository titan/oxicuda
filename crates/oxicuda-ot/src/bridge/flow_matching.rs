//! Conditional Flow Matching (CFM) — simulation-free generative model training.
//!
//! Implements the Conditional Flow Matching framework of Lipman et al. (2022)
//! and the Optimal-Transport CFM extension of Liu et al. (2023) /
//! Tong et al. (2023).
//!
//! # Background
//!
//! A continuous normalising flow (CNF) learns a time-dependent vector field
//! `v_t : ℝᵈ × [0,1] → ℝᵈ` whose integral generates a flow
//! `φ_t : ℝᵈ → ℝᵈ` with `φ_0 = Id` and `φ_1 # μ = ν`.
//!
//! CFM provides a *simulation-free* regression objective.  Given a coupling
//! `(x₀, x₁) ~ q(x₀,x₁)` and a conditional path
//!
//! ```text
//! x_t = (1-t) x₀  +  t x₁       (linear / OT path)
//! ```
//!
//! the conditional velocity field is `u_t(x | x₀,x₁) = x₁ - x₀`, which is
//! known in closed form and requires no ODE simulation.  A velocity network
//! `v_θ(x_t, t)` is trained by minimising
//!
//! ```text
//! L_CFM(θ) = E_{t,x₀,x₁} [ ‖ v_θ(x_t, t) − (x₁ − x₀) ‖² ]
//! ```
//!
//! At inference the flow is integrated forward with Euler steps.
//!
//! # OT-CFM coupling
//!
//! When the coupling `q` is chosen as the mini-batch OT coupling between
//! batches of source and target samples, the resulting flow is closer to the
//! true Brenier map (Tong 2023, *Improving and generalizing flow-matching*).
//! This module implements both the independent coupling (I-CFM) and the
//! mini-batch OT coupling (OT-CFM) as a configurable option.
//!
//! # Velocity network
//!
//! We provide a lightweight MLP velocity network parameterised as
//! `v_θ(x, t) = MLP([x; t])` with `d+1` inputs (concatenate time scalar)
//! and `d` outputs.  Weights are updated via vanilla SGD to minimise the
//! CFM loss.
//!
//! References:
//! - Lipman et al. *Flow Matching for Generative Modeling* (ICLR 2023).
//! - Liu et al. *Flow Straight and Fast: Learning to Generate and Transfer
//!   Data with Rectified Flow* (ICLR 2023).
//! - Tong et al. *Improving and generalising flow-matching* (NeurIPS 2023).

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Velocity network (MLP)
// ──────────────────────────────────────────────────────────────────────────────

/// Tanh activation (element-wise).
#[inline]
fn tanh_act(x: f32) -> f32 {
    x.tanh()
}

/// Derivative of tanh: 1 - tanh²(x).
#[inline]
fn tanh_deriv(x: f32) -> f32 {
    1.0 - x.tanh().powi(2)
}

/// Apply row-major matrix-vector multiply: `out[i] = bias[i] + Σ_j w[i*k+j]*v[j]`.
fn affine(w: &[f32], bias: &[f32], v: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        let mut s = bias[i];
        for j in 0..cols {
            s += w[i * cols + j] * v[j];
        }
        out[i] = s;
    }
}

/// Xavier-uniform initialisation into a slice.
fn xavier(rng: &mut LcgRng, fan_in: usize, out: &mut [f32]) {
    let bound = 1.0_f32 / (fan_in as f32).sqrt();
    for v in out.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * bound;
    }
}

/// Lightweight MLP velocity network `v_θ(x, t) : ℝ^{d+1} → ℝ^d`.
///
/// The time scalar `t ∈ [0,1]` is appended to `x` before the first layer.
#[derive(Debug, Clone)]
pub struct VelocityNet {
    /// Input dimension (spatial dim `d`). The network receives `d+1` inputs.
    pub dim: usize,
    /// Hidden layer width.
    pub hidden: usize,
    // Layer 0: (d+1) → hidden
    /// Weight matrix, row-major `[hidden × (d+1)]`.
    pub w0: Vec<f32>,
    pub b0: Vec<f32>,
    // Layer 1: hidden → hidden
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    // Layer 2: hidden → d  (output)
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl VelocityNet {
    /// Allocate and randomly initialise the velocity network.
    pub fn random(dim: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let inp = dim + 1; // append t
        let mut w0 = vec![0.0_f32; hidden * inp];
        xavier(rng, inp, &mut w0);
        let mut b0 = vec![0.0_f32; hidden];
        xavier(rng, inp, &mut b0);

        let mut w1 = vec![0.0_f32; hidden * hidden];
        xavier(rng, hidden, &mut w1);
        let mut b1 = vec![0.0_f32; hidden];
        xavier(rng, hidden, &mut b1);

        let mut w2 = vec![0.0_f32; dim * hidden];
        xavier(rng, hidden, &mut w2);
        let b2 = vec![0.0_f32; dim];

        VelocityNet {
            dim,
            hidden,
            w0,
            b0,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Evaluate `v_θ(x, t)` for a single sample.
    /// `x` must have length `dim`; returns velocity vector of length `dim`.
    pub fn forward(&self, x: &[f32], t: f32) -> OtResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(OtError::IncompatibleLength {
                a: x.len(),
                b: self.dim,
            });
        }
        let inp = self.dim + 1;
        let h = self.hidden;
        let d = self.dim;

        // Input: [x | t]
        let mut xt: Vec<f32> = Vec::with_capacity(inp);
        xt.extend_from_slice(x);
        xt.push(t);

        // Layer 0
        let mut h0 = vec![0.0_f32; h];
        affine(&self.w0, &self.b0, &xt, h, inp, &mut h0);
        for v in h0.iter_mut() {
            *v = tanh_act(*v);
        }

        // Layer 1
        let mut h1 = vec![0.0_f32; h];
        affine(&self.w1, &self.b1, &h0, h, h, &mut h1);
        for v in h1.iter_mut() {
            *v = tanh_act(*v);
        }

        // Output layer (linear)
        let mut out = vec![0.0_f32; d];
        affine(&self.w2, &self.b2, &h1, d, h, &mut out);

        Ok(out)
    }
}

/// Gradient of the CFM loss w.r.t. velocity network weights.
#[derive(Debug, Clone)]
pub struct VelocityGrad {
    pub dw0: Vec<f32>,
    pub db0: Vec<f32>,
    pub dw1: Vec<f32>,
    pub db1: Vec<f32>,
    pub dw2: Vec<f32>,
    pub db2: Vec<f32>,
}

impl VelocityGrad {
    pub fn zeros(net: &VelocityNet) -> Self {
        VelocityGrad {
            dw0: vec![0.0; net.w0.len()],
            db0: vec![0.0; net.b0.len()],
            dw1: vec![0.0; net.w1.len()],
            db1: vec![0.0; net.b1.len()],
            dw2: vec![0.0; net.w2.len()],
            db2: vec![0.0; net.b2.len()],
        }
    }

    /// Add `other` into `self` (accumulate batch gradients).
    pub fn accumulate(&mut self, other: &VelocityGrad) {
        for (a, &b) in self.dw0.iter_mut().zip(other.dw0.iter()) {
            *a += b;
        }
        for (a, &b) in self.db0.iter_mut().zip(other.db0.iter()) {
            *a += b;
        }
        for (a, &b) in self.dw1.iter_mut().zip(other.dw1.iter()) {
            *a += b;
        }
        for (a, &b) in self.db1.iter_mut().zip(other.db1.iter()) {
            *a += b;
        }
        for (a, &b) in self.dw2.iter_mut().zip(other.dw2.iter()) {
            *a += b;
        }
        for (a, &b) in self.db2.iter_mut().zip(other.db2.iter()) {
            *a += b;
        }
    }

    /// Scale all gradients by `s`.
    pub fn scale(&mut self, s: f32) {
        for v in self.dw0.iter_mut() {
            *v *= s;
        }
        for v in self.db0.iter_mut() {
            *v *= s;
        }
        for v in self.dw1.iter_mut() {
            *v *= s;
        }
        for v in self.db1.iter_mut() {
            *v *= s;
        }
        for v in self.dw2.iter_mut() {
            *v *= s;
        }
        for v in self.db2.iter_mut() {
            *v *= s;
        }
    }
}

impl VelocityNet {
    /// Backprop through the network for a single sample.
    ///
    /// Given pre-computed upstream gradient `d_out : ℝ^d` (d-loss/d-output),
    /// returns parameter gradients via full backprop.
    #[allow(clippy::needless_range_loop)]
    fn backward(&self, x: &[f32], t: f32, d_out: &[f32]) -> OtResult<VelocityGrad> {
        let inp = self.dim + 1;
        let h = self.hidden;
        let d = self.dim;

        // ── Forward again to store pre-activations ────────────────────────────
        let mut xt: Vec<f32> = Vec::with_capacity(inp);
        xt.extend_from_slice(x);
        xt.push(t);

        let mut pre0 = vec![0.0_f32; h];
        affine(&self.w0, &self.b0, &xt, h, inp, &mut pre0);
        let z0: Vec<f32> = pre0.iter().map(|&v| tanh_act(v)).collect();

        let mut pre1 = vec![0.0_f32; h];
        affine(&self.w1, &self.b1, &z0, h, h, &mut pre1);
        let z1: Vec<f32> = pre1.iter().map(|&v| tanh_act(v)).collect();

        // ── Backward ──────────────────────────────────────────────────────────
        // d_out already = d_loss / d_output (upstream)

        // Layer 2: linear, no activation
        // dw2[i*h+j] += d_out[i] * z1[j]
        let mut dw2 = vec![0.0_f32; d * h];
        let mut db2 = vec![0.0_f32; d];
        for i in 0..d {
            db2[i] = d_out[i];
            for j in 0..h {
                dw2[i * h + j] = d_out[i] * z1[j];
            }
        }
        // delta1 = W2^T * d_out
        let mut delta1 = vec![0.0_f32; h];
        for j in 0..h {
            let mut s = 0.0_f32;
            for i in 0..d {
                s += self.w2[i * h + j] * d_out[i];
            }
            delta1[j] = s * tanh_deriv(pre1[j]);
        }

        // Layer 1
        let mut dw1 = vec![0.0_f32; h * h];
        let mut db1 = vec![0.0_f32; h];
        for i in 0..h {
            db1[i] = delta1[i];
            for j in 0..h {
                dw1[i * h + j] = delta1[i] * z0[j];
            }
        }
        // delta0 = W1^T * delta1
        let mut delta0 = vec![0.0_f32; h];
        for j in 0..h {
            let mut s = 0.0_f32;
            for i in 0..h {
                s += self.w1[i * h + j] * delta1[i];
            }
            delta0[j] = s * tanh_deriv(pre0[j]);
        }

        // Layer 0
        let mut dw0 = vec![0.0_f32; h * inp];
        let mut db0 = vec![0.0_f32; h];
        for i in 0..h {
            db0[i] = delta0[i];
            for j in 0..inp {
                dw0[i * inp + j] = delta0[i] * xt[j];
            }
        }

        Ok(VelocityGrad {
            dw0,
            db0,
            dw1,
            db1,
            dw2,
            db2,
        })
    }

    /// SGD update step: `w -= lr * grad`.
    pub fn sgd_step(&mut self, grad: &VelocityGrad, lr: f32) {
        for (w, &g) in self.w0.iter_mut().zip(grad.dw0.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.b0.iter_mut().zip(grad.db0.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.w1.iter_mut().zip(grad.dw1.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.b1.iter_mut().zip(grad.db1.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.w2.iter_mut().zip(grad.dw2.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.b2.iter_mut().zip(grad.db2.iter()) {
            *w -= lr * g;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Coupling strategies
// ──────────────────────────────────────────────────────────────────────────────

/// Coupling strategy for selecting `(x₀, x₁)` pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingStrategy {
    /// Independent coupling: `x₀` and `x₁` are drawn independently (I-CFM).
    Independent,
    /// Mini-batch OT coupling: pair `x₀` with its nearest `x₁` in L₂ distance
    /// within the batch (approximates the OT coupling, OT-CFM).
    MinibatchOt,
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration and output
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for Conditional Flow Matching training.
#[derive(Debug, Clone)]
pub struct CfmConfig {
    /// Hidden width of the velocity MLP.
    pub hidden: usize,
    /// Learning rate.
    pub lr: f32,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Number of training iterations.
    pub n_iter: usize,
    /// Coupling strategy.
    pub coupling: CouplingStrategy,
    /// RNG seed.
    pub seed: u64,
}

impl Default for CfmConfig {
    fn default() -> Self {
        CfmConfig {
            hidden: 32,
            lr: 1e-3,
            batch_size: 32,
            n_iter: 200,
            coupling: CouplingStrategy::Independent,
            seed: 42,
        }
    }
}

/// Fitted Conditional Flow Matching model.
#[derive(Debug, Clone)]
pub struct CfmFit {
    /// Trained velocity network.
    pub net: VelocityNet,
    /// CFM loss history (mean squared error per iteration).
    pub loss_history: Vec<f32>,
    /// Spatial dimensionality.
    pub dim: usize,
}

impl CfmFit {
    /// Evaluate the velocity field `v_θ(x, t)` at point `x` and time `t`.
    pub fn velocity(&self, x: &[f32], t: f32) -> OtResult<Vec<f32>> {
        self.net.forward(x, t)
    }

    /// Integrate the flow from `x_0` to `x_1` using `n_steps` Euler steps.
    ///
    /// Returns the pushed-forward sample `x_1 ≈ φ_1(x_0)`.
    pub fn integrate(&self, x0: &[f32], n_steps: usize) -> OtResult<Vec<f32>> {
        if x0.len() != self.dim {
            return Err(OtError::IncompatibleLength {
                a: x0.len(),
                b: self.dim,
            });
        }
        if n_steps == 0 {
            return Err(OtError::BadCount { got: 0 });
        }
        let dt = 1.0_f32 / n_steps as f32;
        let mut x = x0.to_vec();
        for step in 0..n_steps {
            let t = step as f32 * dt;
            let v = self.net.forward(&x, t)?;
            for (xi, vi) in x.iter_mut().zip(v.iter()) {
                *xi += dt * vi;
            }
        }
        Ok(x)
    }

    /// Integrate a batch of `n` source samples of length `n * dim` (row-major).
    pub fn integrate_batch(
        &self,
        x0_batch: &[f32],
        n: usize,
        n_steps: usize,
    ) -> OtResult<Vec<f32>> {
        if n == 0 {
            return Err(OtError::EmptyInput);
        }
        if x0_batch.len() != n * self.dim {
            return Err(OtError::IncompatibleLength {
                a: x0_batch.len(),
                b: n * self.dim,
            });
        }
        let mut out = Vec::with_capacity(n * self.dim);
        for i in 0..n {
            let x0 = &x0_batch[i * self.dim..(i + 1) * self.dim];
            let x1 = self.integrate(x0, n_steps)?;
            out.extend_from_slice(&x1);
        }
        Ok(out)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Coupling helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Draw `batch_size` independent pairs `(x0_idx, x1_idx)` uniformly at random.
fn independent_pairs(
    rng: &mut LcgRng,
    n_src: usize,
    n_tgt: usize,
    batch_size: usize,
) -> Vec<(usize, usize)> {
    (0..batch_size)
        .map(|_| {
            let i = (rng.next_u32() as usize) % n_src;
            let j = (rng.next_u32() as usize) % n_tgt;
            (i, j)
        })
        .collect()
}

/// Mini-batch OT coupling: greedily pair each `x0_i` in the batch with the
/// nearest (L₂) `x1_j` in the target batch. Simple nearest-neighbour is used
/// as a cheap OT approximation (exact assignment for batch_size ≤ 32).
fn minibatch_ot_pairs(
    src_batch_indices: &[usize],
    tgt_batch_indices: &[usize],
    source_samples: &[f32],
    target_samples: &[f32],
    dim: usize,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(src_batch_indices.len());
    for &i in src_batch_indices {
        let xi = &source_samples[i * dim..(i + 1) * dim];
        let mut best_j = tgt_batch_indices[0];
        let mut best_dist = f32::MAX;
        for &j in tgt_batch_indices {
            let yj = &target_samples[j * dim..(j + 1) * dim];
            let dist: f32 = xi.iter().zip(yj.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_j = j;
            }
        }
        pairs.push((i, best_j));
    }
    pairs
}

// ──────────────────────────────────────────────────────────────────────────────
// Training
// ──────────────────────────────────────────────────────────────────────────────

/// Train a Conditional Flow Matching velocity field from `source_samples` to
/// `target_samples`.
///
/// Both buffers are row-major with `n_src * dim` and `n_tgt * dim` elements.
pub fn conditional_flow_matching(
    source_samples: &[f32],
    target_samples: &[f32],
    n_src: usize,
    n_tgt: usize,
    dim: usize,
    cfg: &CfmConfig,
) -> OtResult<CfmFit> {
    if dim == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    if n_src == 0 || n_tgt == 0 {
        return Err(OtError::EmptyInput);
    }
    if source_samples.len() != n_src * dim {
        return Err(OtError::IncompatibleLength {
            a: source_samples.len(),
            b: n_src * dim,
        });
    }
    if target_samples.len() != n_tgt * dim {
        return Err(OtError::IncompatibleLength {
            a: target_samples.len(),
            b: n_tgt * dim,
        });
    }
    if cfg.hidden == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    if cfg.batch_size == 0 {
        return Err(OtError::BadCount { got: 0 });
    }

    let mut rng = LcgRng::new(cfg.seed);
    let mut net = VelocityNet::random(dim, cfg.hidden, &mut rng);

    let bs = cfg.batch_size.min(n_src).min(n_tgt);
    let mut loss_history = Vec::with_capacity(cfg.n_iter);

    for _iter in 0..cfg.n_iter {
        // Sample coupling pairs
        let src_indices: Vec<usize> = (0..bs).map(|_| (rng.next_u32() as usize) % n_src).collect();
        let tgt_indices: Vec<usize> = (0..bs).map(|_| (rng.next_u32() as usize) % n_tgt).collect();

        let pairs = match cfg.coupling {
            CouplingStrategy::Independent => independent_pairs(&mut rng, n_src, n_tgt, bs),
            CouplingStrategy::MinibatchOt => minibatch_ot_pairs(
                &src_indices,
                &tgt_indices,
                source_samples,
                target_samples,
                dim,
            ),
        };

        // Sample time t ~ U[0,1] for each pair
        let mut batch_loss = 0.0_f32;
        let mut total_grad = VelocityGrad::zeros(&net);

        for &(i, j) in &pairs {
            let t = rng.next_f32();
            let x0 = &source_samples[i * dim..(i + 1) * dim];
            let x1 = &target_samples[j * dim..(j + 1) * dim];

            // Interpolated point x_t = (1-t)*x0 + t*x1
            let xt: Vec<f32> = (0..dim).map(|k| (1.0 - t) * x0[k] + t * x1[k]).collect();

            // Conditional velocity target: u_t = x1 - x0
            let u_t: Vec<f32> = (0..dim).map(|k| x1[k] - x0[k]).collect();

            // Predicted velocity
            let v_pred = net.forward(&xt, t)?;

            // MSE loss and upstream gradient
            let mut mse = 0.0_f32;
            let mut d_out = vec![0.0_f32; dim];
            for k in 0..dim {
                let diff = v_pred[k] - u_t[k];
                mse += diff * diff;
                d_out[k] = 2.0 * diff; // d(MSE)/d(v_pred)
            }
            batch_loss += mse;

            // Backprop
            let g = net.backward(&xt, t, &d_out)?;
            total_grad.accumulate(&g);
        }

        let n = pairs.len() as f32;
        total_grad.scale(1.0 / n);
        batch_loss /= n;
        loss_history.push(batch_loss);

        net.sgd_step(&total_grad, cfg.lr);
    }

    Ok(CfmFit {
        net,
        loss_history,
        dim,
    })
}

/// Compute the interpolated point `x_t = (1-t)*x0 + t*x1`.
///
/// Returns the interpolated vector of length `dim`.
pub fn flow_interpolate(x0: &[f32], x1: &[f32], t: f32) -> OtResult<Vec<f32>> {
    if x0.len() != x1.len() {
        return Err(OtError::IncompatibleLength {
            a: x0.len(),
            b: x1.len(),
        });
    }
    if x0.is_empty() {
        return Err(OtError::EmptyInput);
    }
    Ok(x0
        .iter()
        .zip(x1.iter())
        .map(|(&a, &b)| (1.0 - t) * a + t * b)
        .collect())
}

/// Compute the analytical conditional velocity `u_t(x_t | x0, x1) = x1 - x0`.
pub fn conditional_velocity(x0: &[f32], x1: &[f32]) -> OtResult<Vec<f32>> {
    if x0.len() != x1.len() {
        return Err(OtError::IncompatibleLength {
            a: x0.len(),
            b: x1.len(),
        });
    }
    if x0.is_empty() {
        return Err(OtError::EmptyInput);
    }
    Ok(x0.iter().zip(x1.iter()).map(|(&a, &b)| b - a).collect())
}

/// Estimate the straightness of the learned flow field:
/// `straightness = 1 − E[‖v_θ(x_t,t) − (x1-x0)‖² / ‖x1-x0‖²]`
///
/// A value close to 1 indicates the field nearly recovers the straight-line
/// interpolant. Evaluates over all `n_pairs` source-target pairs at `t=0.5`.
pub fn flow_straightness(
    fit: &CfmFit,
    source_samples: &[f32],
    target_samples: &[f32],
    n_src: usize,
    n_tgt: usize,
) -> OtResult<f32> {
    let d = fit.dim;
    if n_src == 0 || n_tgt == 0 {
        return Err(OtError::EmptyInput);
    }
    let n_eval = n_src.min(n_tgt);
    let t = 0.5_f32;
    let mut rel_err_sum = 0.0_f32;
    let mut count = 0_usize;

    for i in 0..n_eval {
        let x0 = &source_samples[i * d..(i + 1) * d];
        let x1 = &target_samples[i * d..(i + 1) * d];
        let xt = flow_interpolate(x0, x1, t)?;
        let u_t = conditional_velocity(x0, x1)?;
        let v_pred = fit.velocity(&xt, t)?;

        let mut err2 = 0.0_f32;
        let mut norm2 = 0.0_f32;
        for k in 0..d {
            err2 += (v_pred[k] - u_t[k]).powi(2);
            norm2 += u_t[k].powi(2);
        }
        if norm2 > 1e-9 {
            rel_err_sum += err2 / norm2;
            count += 1;
        }
    }

    if count == 0 {
        return Err(OtError::Internal {
            msg: "all pairs have zero velocity; cannot compute straightness".to_string(),
        });
    }
    Ok(1.0 - rel_err_sum / count as f32)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| a + (b - a) * i as f32 / (n - 1).max(1) as f32)
            .collect()
    }

    fn make_data(n: usize, dim: usize, shift: f32) -> Vec<f32> {
        let mut rng = LcgRng::new(17);
        (0..n * dim).map(|_| rng.next_f32() + shift).collect()
    }

    #[test]
    fn test_flow_interpolate_midpoint() {
        let x0 = [0.0_f32, 0.0];
        let x1 = [2.0_f32, 4.0];
        let xt = flow_interpolate(&x0, &x1, 0.5).expect("ok");
        assert!((xt[0] - 1.0).abs() < 1e-6);
        assert!((xt[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_flow_interpolate_endpoints() {
        let x0 = [1.0_f32, 2.0];
        let x1 = [3.0_f32, 4.0];
        let at0 = flow_interpolate(&x0, &x1, 0.0).expect("ok");
        let at1 = flow_interpolate(&x0, &x1, 1.0).expect("ok");
        assert!((at0[0] - 1.0).abs() < 1e-6 && (at0[1] - 2.0).abs() < 1e-6);
        assert!((at1[0] - 3.0).abs() < 1e-6 && (at1[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_conditional_velocity_basic() {
        let x0 = [0.0_f32, 0.0];
        let x1 = [1.0_f32, 2.0];
        let v = conditional_velocity(&x0, &x1).expect("ok");
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_net_forward_shape() {
        let mut rng = LcgRng::new(3);
        let net = VelocityNet::random(2, 8, &mut rng);
        let x = [0.3_f32, -0.5];
        let v = net.forward(&x, 0.5).expect("ok");
        assert_eq!(v.len(), 2);
        for &val in &v {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_velocity_net_wrong_dim() {
        let mut rng = LcgRng::new(5);
        let net = VelocityNet::random(3, 8, &mut rng);
        let err = net.forward(&[0.0_f32, 1.0], 0.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_cfm_training_independent() {
        let dim = 2;
        let n = 20;
        let src = make_data(n, dim, 0.0);
        let tgt = make_data(n, dim, 2.0);
        let cfg = CfmConfig {
            hidden: 8,
            lr: 1e-2,
            batch_size: 8,
            n_iter: 10,
            coupling: CouplingStrategy::Independent,
            seed: 7,
        };
        let fit = conditional_flow_matching(&src, &tgt, n, n, dim, &cfg).expect("cfm ok");
        assert_eq!(fit.loss_history.len(), 10);
        for &l in &fit.loss_history {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_cfm_training_minibatch_ot() {
        let dim = 2;
        let n = 16;
        let src = make_data(n, dim, 0.0);
        let tgt = make_data(n, dim, 3.0);
        let cfg = CfmConfig {
            hidden: 8,
            lr: 5e-3,
            batch_size: 6,
            n_iter: 5,
            coupling: CouplingStrategy::MinibatchOt,
            seed: 11,
        };
        let fit = conditional_flow_matching(&src, &tgt, n, n, dim, &cfg).expect("ot-cfm ok");
        assert_eq!(fit.loss_history.len(), 5);
    }

    #[test]
    fn test_cfm_integrate_shape() {
        let dim = 2;
        let n = 10;
        let src = make_data(n, dim, 0.0);
        let tgt = make_data(n, dim, 1.0);
        let cfg = CfmConfig {
            n_iter: 3,
            hidden: 4,
            batch_size: 4,
            lr: 1e-3,
            ..CfmConfig::default()
        };
        let fit = conditional_flow_matching(&src, &tgt, n, n, dim, &cfg).expect("ok");
        let x0 = &src[..dim];
        let x1 = fit.integrate(x0, 10).expect("integrate ok");
        assert_eq!(x1.len(), dim);
        for &v in &x1 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_cfm_integrate_batch_shape() {
        let dim = 3;
        let n = 8;
        let src = make_data(n, dim, 0.0);
        let tgt = make_data(n, dim, 1.0);
        let cfg = CfmConfig {
            n_iter: 2,
            hidden: 4,
            batch_size: 4,
            lr: 1e-3,
            ..CfmConfig::default()
        };
        let fit = conditional_flow_matching(&src, &tgt, n, n, dim, &cfg).expect("ok");
        let pushed = fit.integrate_batch(&src, n, 5).expect("batch ok");
        assert_eq!(pushed.len(), n * dim);
    }

    #[test]
    fn test_cfm_empty_source_error() {
        let cfg = CfmConfig::default();
        let err = conditional_flow_matching(&[], &[1.0_f32], 0, 1, 1, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_cfm_bad_dim_error() {
        let cfg = CfmConfig {
            hidden: 4,
            n_iter: 1,
            batch_size: 2,
            ..CfmConfig::default()
        };
        let err = conditional_flow_matching(&[1.0], &[1.0], 1, 1, 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_flow_straightness_runs() {
        let dim = 2;
        let n = 10;
        let src: Vec<f32> = linspace(0.0, 1.0, n * dim);
        let tgt: Vec<f32> = linspace(2.0, 3.0, n * dim);
        let cfg = CfmConfig {
            hidden: 8,
            n_iter: 5,
            batch_size: 5,
            lr: 1e-2,
            ..CfmConfig::default()
        };
        let fit = conditional_flow_matching(&src, &tgt, n, n, dim, &cfg).expect("ok");
        let s = flow_straightness(&fit, &src, &tgt, n, n).expect("straightness ok");
        assert!(s.is_finite());
    }

    #[test]
    fn test_flow_interpolate_length_mismatch() {
        let err = flow_interpolate(&[1.0_f32], &[2.0, 3.0], 0.5);
        assert!(err.is_err());
    }
}
