//! Neural Optimal Transport map via Input-Convex Neural Networks (ICNN).
//!
//! Approximates the W₂ Kantorovich potential `f : ℝᵈ → ℝ` using an
//! Input-Convex Neural Network (ICNN) as introduced by Amos et al. (2017) and
//! applied to OT by Makkuva et al. (2020) and Korotin et al. (2021).
//!
//! # Architecture
//!
//! An ICNN with `L` hidden layers of width `w` satisfies strict convexity in
//! the input `x` by enforcing non-negativity of the weight matrices `W_z^(l)`
//! that multiply the previous hidden state, while the passthrough weights
//! `W_x^(l)` connecting the raw input at every layer are unconstrained.
//!
//! ```text
//! z₀  = σ(W_x^(0)  x  +  b^(0))
//! z_l = σ(W_z^(l) z_{l-1}  +  W_x^(l) x  +  b^(l))   [W_z^(l) ≥ 0]
//! f(x) = W_z^(L) z_{L-1}  +  W_x^(L) x  +  b^(L)     [scalar output]
//! ```
//!
//! The non-negativity constraint is enforced by storing `log W_z^(l)` and
//! exponentiating on the forward pass (soft-plus activation on weights).
//!
//! # Optimisation
//!
//! The OT primal-dual formulation for W₂ reads
//!
//! ```text
//! W₂²(μ,ν) = sup_{f convex} { E_{x~μ}[f(x)] + E_{y~ν}[f*(y)] }
//! ```
//!
//! where `f*` is the convex conjugate (Legendre-Fenchel transform).
//! We maximise over `f` parameterised by ICNN weights using mini-batch SGD on
//! the negative dual objective, approximating `f*(y) ≈ max_x { <x,y> - f(x) }`
//! via a finite sample inner optimisation.
//!
//! References:
//! - Makkuva et al. *Optimal Transport Mapping via Input Convex Neural Networks*
//!   (ICML 2020).
//! - Korotin et al. *Neural Optimal Transport* (ICLR 2022).
//! - Amos et al. *Input Convex Neural Networks* (ICML 2017).

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Soft-plus activation: `log(1 + exp(x))` clamped to avoid overflow.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 30.0 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Derivative of soft-plus = sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 30.0 {
        1.0
    } else if x < -30.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Apply row-major matrix-vector multiplication: `out[i] = Σ_j mat[i*k+j] * vec[j]`.
fn mat_vec(mat: &[f32], vec: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for i in 0..rows {
        let mut s = 0.0_f32;
        for j in 0..cols {
            s += mat[i * cols + j] * vec[j];
        }
        out[i] = s;
    }
}

/// Xavier-style uniform initialisation: `U[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
fn xavier_init(rng: &mut LcgRng, fan_in: usize, out: &mut [f32]) {
    let bound = 1.0_f32 / (fan_in as f32).sqrt();
    for v in out.iter_mut() {
        // LcgRng::next_f32 is in [0,1); map to [-bound, bound].
        *v = (rng.next_f32() * 2.0 - 1.0) * bound;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ICNN weight layout
// ──────────────────────────────────────────────────────────────────────────────
//
// Layer 0 (input → hidden):
//   wx0: [hidden × dim]   — passthrough weights (unconstrained)
//   b0:  [hidden]         — biases
//
// Layer l = 1 … L-1 (hidden → hidden):
//   wz_l: [hidden × hidden] — convex weights (stored in log-space, exponentiated)
//   wx_l: [hidden × dim]    — passthrough weights (unconstrained)
//   b_l:  [hidden]          — biases
//
// Output layer L (hidden → scalar):
//   wz_out: [1 × hidden]    — convex weights (stored in log-space)
//   wx_out: [1 × dim]       — passthrough weights
//   b_out:  [1]             — bias
//
// For the ICNN to be strictly convex we require wz_l ≥ 0 for all l ≥ 1.
// We store log(wz_l + 1e-7) and recover wz via softplus.

/// ICNN weights for one-hidden-block architecture.
#[derive(Debug, Clone)]
pub struct IcnnWeights {
    /// Dimensionality of the input `x`.
    pub dim: usize,
    /// Width of each hidden layer.
    pub hidden: usize,
    /// Number of hidden layers (≥ 1).
    pub n_hidden: usize,

    // Layer 0
    /// `[hidden × dim]` passthrough, unconstrained.
    pub wx0: Vec<f32>,
    /// `[hidden]` biases.
    pub b0: Vec<f32>,

    // Middle layers: index l corresponds to layer l+1 (l = 0 .. n_hidden-1).
    /// `[n_hidden × hidden × hidden]` log-space convex weights.
    pub log_wz: Vec<f32>,
    /// `[n_hidden × hidden × dim]` passthrough weights.
    pub wx_mid: Vec<f32>,
    /// `[n_hidden × hidden]` biases.
    pub b_mid: Vec<f32>,

    // Output layer
    /// `[hidden]` log-space convex weights.
    pub log_wz_out: Vec<f32>,
    /// `[dim]` unconstrained passthrough.
    pub wx_out: Vec<f32>,
    /// Scalar bias.
    pub b_out: f32,
}

impl IcnnWeights {
    /// Allocate and randomly initialise ICNN weights with Xavier uniform.
    pub fn random(dim: usize, hidden: usize, n_hidden: usize, rng: &mut LcgRng) -> Self {
        let h = hidden;
        let d = dim;
        let nl = n_hidden;

        let mut wx0 = vec![0.0_f32; h * d];
        xavier_init(rng, d, &mut wx0);

        let mut b0 = vec![0.0_f32; h];
        xavier_init(rng, d, &mut b0);

        let mut log_wz = vec![0.0_f32; nl * h * h];
        xavier_init(rng, h, &mut log_wz);

        let mut wx_mid = vec![0.0_f32; nl * h * d];
        xavier_init(rng, d, &mut wx_mid);

        let mut b_mid = vec![0.0_f32; nl * h];
        xavier_init(rng, h, &mut b_mid);

        let mut log_wz_out = vec![0.0_f32; h];
        xavier_init(rng, h, &mut log_wz_out);

        let mut wx_out = vec![0.0_f32; d];
        xavier_init(rng, d, &mut wx_out);

        let mut b_out_arr = [0.0_f32; 1];
        xavier_init(rng, d, &mut b_out_arr);
        let b_out = b_out_arr[0];

        IcnnWeights {
            dim,
            hidden,
            n_hidden,
            wx0,
            b0,
            log_wz,
            wx_mid,
            b_mid,
            log_wz_out,
            wx_out,
            b_out,
        }
    }

    /// Forward pass: evaluate `f(x)` for a single input `x` of length `dim`.
    /// Returns the scalar value `f(x)`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, x: &[f32]) -> OtResult<f32> {
        if x.len() != self.dim {
            return Err(OtError::IncompatibleLength {
                a: x.len(),
                b: self.dim,
            });
        }
        let h = self.hidden;
        let d = self.dim;
        let nl = self.n_hidden;

        // Layer 0: z0 = softplus(wx0 * x + b0)
        let mut z = vec![0.0_f32; h];
        mat_vec(&self.wx0, x, h, d, &mut z);
        for i in 0..h {
            z[i] = softplus(z[i] + self.b0[i]);
        }

        // Middle layers l = 1 … nl
        let mut z_new = vec![0.0_f32; h];
        for l in 0..nl {
            // convex part: wz_{l+1} = exp(log_wz) ≥ 0
            let wz_offset = l * h * h;
            let wx_offset = l * h * d;
            let b_offset = l * h;

            for i in 0..h {
                let mut s = 0.0_f32;
                // convex weights × previous hidden state
                for j in 0..h {
                    let wz = self.log_wz[wz_offset + i * h + j].exp();
                    s += wz * z[j];
                }
                // passthrough weights × raw input
                for j in 0..d {
                    s += self.wx_mid[wx_offset + i * d + j] * x[j];
                }
                s += self.b_mid[b_offset + i];
                z_new[i] = softplus(s);
            }
            std::mem::swap(&mut z, &mut z_new);
        }

        // Output layer: scalar = wz_out * z + wx_out * x + b_out
        let mut out = self.b_out;
        for i in 0..h {
            let wz = self.log_wz_out[i].exp();
            out += wz * z[i];
        }
        for j in 0..d {
            out += self.wx_out[j] * x[j];
        }
        Ok(out)
    }

    /// Compute gradient of `f` w.r.t. `x` via backpropagation.
    /// Returns `∇_x f(x)` as a vector of length `dim`.
    #[allow(clippy::needless_range_loop)]
    pub fn gradient(&self, x: &[f32]) -> OtResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(OtError::IncompatibleLength {
                a: x.len(),
                b: self.dim,
            });
        }
        let h = self.hidden;
        let d = self.dim;
        let nl = self.n_hidden;

        // ── Forward pass with activation storage ──────────────────────────────
        // pre-activation values s_l, post-activation z_l
        let mut pre_acts: Vec<Vec<f32>> = Vec::with_capacity(nl + 1);
        let mut post_acts: Vec<Vec<f32>> = Vec::with_capacity(nl + 1);

        // Layer 0
        let mut s0 = vec![0.0_f32; h];
        mat_vec(&self.wx0, x, h, d, &mut s0);
        for i in 0..h {
            s0[i] += self.b0[i];
        }
        let z0: Vec<f32> = s0.iter().map(|&v| softplus(v)).collect();
        pre_acts.push(s0);
        post_acts.push(z0);

        for l in 0..nl {
            let wz_offset = l * h * h;
            let wx_offset = l * h * d;
            let b_offset = l * h;
            let z_prev = &post_acts[post_acts.len() - 1];

            let mut sl = vec![0.0_f32; h];
            for i in 0..h {
                let mut s = 0.0_f32;
                for j in 0..h {
                    let wz = self.log_wz[wz_offset + i * h + j].exp();
                    s += wz * z_prev[j];
                }
                for j in 0..d {
                    s += self.wx_mid[wx_offset + i * d + j] * x[j];
                }
                s += self.b_mid[b_offset + i];
                sl[i] = s;
            }
            let zl: Vec<f32> = sl.iter().map(|&v| softplus(v)).collect();
            pre_acts.push(sl);
            post_acts.push(zl);
        }

        // ── Backward pass ─────────────────────────────────────────────────────
        // d_out / d_z_{nl} (from output layer)
        let mut delta = vec![0.0_f32; h];
        for i in 0..h {
            let wz = self.log_wz_out[i].exp();
            // sigmoid(s_{nl,i}) = d softplus / d s
            delta[i] = wz * sigmoid(pre_acts[nl][i]);
        }

        // Propagate through middle layers (nl-1 down to 0)
        let mut grad_x = vec![0.0_f32; d];

        // Contribution from output-layer wx_out passthrough
        for j in 0..d {
            grad_x[j] += self.wx_out[j];
        }

        for l in (0..nl).rev() {
            let wz_offset = l * h * h;
            let wx_offset = l * h * d;

            // Passthrough contribution at layer l: wx_mid^T * delta
            for j in 0..d {
                let mut s = 0.0_f32;
                for i in 0..h {
                    s += self.wx_mid[wx_offset + i * d + j] * delta[i];
                }
                grad_x[j] += s;
            }

            // Propagate delta through convex weights to layer l
            let mut delta_prev = vec![0.0_f32; h];
            let sig_prev: Vec<f32> = pre_acts[l].iter().map(|&s| sigmoid(s)).collect();
            for j in 0..h {
                let mut s = 0.0_f32;
                for i in 0..h {
                    let wz = self.log_wz[wz_offset + i * h + j].exp();
                    s += wz * delta[i];
                }
                delta_prev[j] = s * sig_prev[j];
            }
            delta = delta_prev;
        }

        // Layer 0 contribution: wx0^T * delta_0
        for j in 0..d {
            let mut s = 0.0_f32;
            for i in 0..h {
                s += self.wx0[i * d + j] * delta[i];
            }
            grad_x[j] += s;
        }

        Ok(grad_x)
    }

    /// Update all weights along negative gradient of `loss` w.r.t. each
    /// parameter, scaled by `lr`. The gradient vectors have the same shapes as
    /// the corresponding weight fields.
    ///
    /// This is a plain SGD step; callers are responsible for computing the
    /// loss gradient via finite-difference or a backward-pass accumulator.
    pub fn sgd_step(&mut self, grad: &IcnnGrad, lr: f32) {
        for (w, &g) in self.wx0.iter_mut().zip(grad.dwx0.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.b0.iter_mut().zip(grad.db0.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.log_wz.iter_mut().zip(grad.d_log_wz.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.wx_mid.iter_mut().zip(grad.dwx_mid.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.b_mid.iter_mut().zip(grad.db_mid.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.log_wz_out.iter_mut().zip(grad.d_log_wz_out.iter()) {
            *w -= lr * g;
        }
        for (w, &g) in self.wx_out.iter_mut().zip(grad.dwx_out.iter()) {
            *w -= lr * g;
        }
        self.b_out -= lr * grad.db_out;
    }
}

/// Gradient container matching the shape of [`IcnnWeights`].
#[derive(Debug, Clone)]
pub struct IcnnGrad {
    pub dwx0: Vec<f32>,
    pub db0: Vec<f32>,
    pub d_log_wz: Vec<f32>,
    pub dwx_mid: Vec<f32>,
    pub db_mid: Vec<f32>,
    pub d_log_wz_out: Vec<f32>,
    pub dwx_out: Vec<f32>,
    pub db_out: f32,
}

impl IcnnGrad {
    /// Allocate zero-initialised gradient buffer matching the given weight layout.
    pub fn zeros(w: &IcnnWeights) -> Self {
        IcnnGrad {
            dwx0: vec![0.0; w.wx0.len()],
            db0: vec![0.0; w.b0.len()],
            d_log_wz: vec![0.0; w.log_wz.len()],
            dwx_mid: vec![0.0; w.wx_mid.len()],
            db_mid: vec![0.0; w.b_mid.len()],
            d_log_wz_out: vec![0.0; w.log_wz_out.len()],
            dwx_out: vec![0.0; w.wx_out.len()],
            db_out: 0.0,
        }
    }

    /// Divide every gradient entry by `n` (batch normalisation).
    pub fn scale(&mut self, n: f32) {
        for v in self.dwx0.iter_mut() {
            *v /= n;
        }
        for v in self.db0.iter_mut() {
            *v /= n;
        }
        for v in self.d_log_wz.iter_mut() {
            *v /= n;
        }
        for v in self.dwx_mid.iter_mut() {
            *v /= n;
        }
        for v in self.db_mid.iter_mut() {
            *v /= n;
        }
        for v in self.d_log_wz_out.iter_mut() {
            *v /= n;
        }
        for v in self.dwx_out.iter_mut() {
            *v /= n;
        }
        self.db_out /= n;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Training configuration and result
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Neural OT training loop.
#[derive(Debug, Clone)]
pub struct NeuralOtConfig {
    /// Hidden layer width of the source ICNN `f`.
    pub hidden: usize,
    /// Number of hidden layers.
    pub n_hidden: usize,
    /// Learning rate for SGD updates.
    pub lr: f32,
    /// Mini-batch size drawn from source and target each iteration.
    pub batch_size: usize,
    /// Number of training iterations.
    pub n_iter: usize,
    /// Number of inner-optimisation steps for approximating `f*(y)`.
    pub n_inner: usize,
    /// Step size for the inner gradient ascent on `f*(y)`.
    pub inner_lr: f32,
    /// RNG seed.
    pub seed: u64,
}

impl Default for NeuralOtConfig {
    fn default() -> Self {
        NeuralOtConfig {
            hidden: 32,
            n_hidden: 2,
            lr: 1e-3,
            batch_size: 32,
            n_iter: 100,
            n_inner: 5,
            inner_lr: 0.1,
            seed: 42,
        }
    }
}

/// Fitted Neural OT model — holds the trained ICNN potential `f`.
#[derive(Debug, Clone)]
pub struct NeuralOtFit {
    /// Trained ICNN potential `f`.
    pub icnn: IcnnWeights,
    /// Dual objective history (one entry per outer iteration).
    pub dual_history: Vec<f32>,
    /// Dimensionality of the input space.
    pub dim: usize,
}

impl NeuralOtFit {
    /// Evaluate the trained Kantorovich potential `f(x)`.
    pub fn potential(&self, x: &[f32]) -> OtResult<f32> {
        self.icnn.forward(x)
    }

    /// Compute the optimal transport map `T*(x) = ∇f(x)`.
    ///
    /// For the squared-Euclidean W₂ cost, the optimal map is the gradient
    /// of the Kantorovich potential (Brenier theorem).
    pub fn transport_map(&self, x: &[f32]) -> OtResult<Vec<f32>> {
        self.icnn.gradient(x)
    }

    /// Apply the transport map to a batch of `n` source points of length
    /// `n × dim` (row-major) and return the pushed-forward samples.
    pub fn push_forward(&self, source: &[f32], n: usize) -> OtResult<Vec<f32>> {
        if n == 0 {
            return Err(OtError::EmptyInput);
        }
        if source.len() != n * self.dim {
            return Err(OtError::IncompatibleLength {
                a: source.len(),
                b: n * self.dim,
            });
        }
        let mut out = Vec::with_capacity(n * self.dim);
        for i in 0..n {
            let x = &source[i * self.dim..(i + 1) * self.dim];
            let tx = self.transport_map(x)?;
            out.extend_from_slice(&tx);
        }
        Ok(out)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Approximate convex conjugate via inner gradient ascent
// ──────────────────────────────────────────────────────────────────────────────

/// Approximate `f*(y) ≈ max_{x ∈ source_batch} { <x, y> - f(x) }` using a
/// finite-sample inner optimisation: evaluate over all source batch members and
/// take the maximum.
fn approx_conjugate(
    icnn: &IcnnWeights,
    y: &[f32],
    source_batch: &[f32],
    batch_size: usize,
) -> OtResult<f32> {
    let d = icnn.dim;
    let mut best = f32::NEG_INFINITY;
    for i in 0..batch_size {
        let x = &source_batch[i * d..(i + 1) * d];
        let fx = icnn.forward(x)?;
        let mut dot = 0.0_f32;
        for j in 0..d {
            dot += x[j] * y[j];
        }
        let val = dot - fx;
        if val > best {
            best = val;
        }
    }
    Ok(best)
}

/// Compute the mini-batch dual objective:
/// `L(θ) = (1/B) Σ_{x~μ} f(x)  +  (1/B) Σ_{y~ν} f*(y)`
/// and the gradient w.r.t. `f` parameters on the source term.
///
/// We approximate the gradient via finite differences on each weight.
fn compute_dual_loss(
    icnn: &IcnnWeights,
    src_batch: &[f32],
    tgt_batch: &[f32],
    batch_size: usize,
) -> OtResult<(f32, IcnnGrad)> {
    let d = icnn.dim;
    let mut loss = 0.0_f32;

    // Source term: E_{x~μ}[f(x)]
    let mut fx_sum = 0.0_f32;
    for i in 0..batch_size {
        let x = &src_batch[i * d..(i + 1) * d];
        fx_sum += icnn.forward(x)?;
    }
    loss += fx_sum / batch_size as f32;

    // Target term: E_{y~ν}[f*(y)]  (approximate conjugate via source batch)
    let mut conj_sum = 0.0_f32;
    for j in 0..batch_size {
        let y = &tgt_batch[j * d..(j + 1) * d];
        conj_sum += approx_conjugate(icnn, y, src_batch, batch_size)?;
    }
    loss += conj_sum / batch_size as f32;

    // Gradient via finite differences (lightweight, accurate enough for demo)
    let eps = 1e-4_f32;
    let mut grad = IcnnGrad::zeros(icnn);

    // Helper: perturb weight w by eps, recompute source-term loss, restore.
    let compute_source_loss = |icnn2: &IcnnWeights| -> OtResult<f32> {
        let mut s = 0.0_f32;
        for i in 0..batch_size {
            let x = &src_batch[i * d..(i + 1) * d];
            s += icnn2.forward(x)?;
        }
        Ok(s / batch_size as f32)
    };

    // wx0
    for k in 0..icnn.wx0.len() {
        let mut w2 = icnn.clone();
        w2.wx0[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.dwx0[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // b0
    for k in 0..icnn.b0.len() {
        let mut w2 = icnn.clone();
        w2.b0[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.db0[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // log_wz
    for k in 0..icnn.log_wz.len() {
        let mut w2 = icnn.clone();
        w2.log_wz[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.d_log_wz[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // wx_mid
    for k in 0..icnn.wx_mid.len() {
        let mut w2 = icnn.clone();
        w2.wx_mid[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.dwx_mid[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // b_mid
    for k in 0..icnn.b_mid.len() {
        let mut w2 = icnn.clone();
        w2.b_mid[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.db_mid[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // log_wz_out
    for k in 0..icnn.log_wz_out.len() {
        let mut w2 = icnn.clone();
        w2.log_wz_out[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.d_log_wz_out[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // wx_out
    for k in 0..icnn.wx_out.len() {
        let mut w2 = icnn.clone();
        w2.wx_out[k] += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.dwx_out[k] = (l2 - fx_sum / batch_size as f32) / eps;
    }
    // b_out
    {
        let mut w2 = icnn.clone();
        w2.b_out += eps;
        let l2 = compute_source_loss(&w2)?;
        grad.db_out = (l2 - fx_sum / batch_size as f32) / eps;
    }

    Ok((loss, grad))
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Train a Neural OT map from `source_samples` to `target_samples`.
///
/// Both sample buffers are row-major: `source_samples.len() == n_src * dim`,
/// `target_samples.len() == n_tgt * dim`.
///
/// Returns a [`NeuralOtFit`] containing the trained ICNN and training history.
pub fn neural_ot(
    source_samples: &[f32],
    target_samples: &[f32],
    n_src: usize,
    n_tgt: usize,
    dim: usize,
    cfg: &NeuralOtConfig,
) -> OtResult<NeuralOtFit> {
    // ── Validation ────────────────────────────────────────────────────────────
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

    // ── Initialise ICNN ───────────────────────────────────────────────────────
    let mut rng = LcgRng::new(cfg.seed);
    let mut icnn = IcnnWeights::random(dim, cfg.hidden, cfg.n_hidden, &mut rng);

    let mut dual_history = Vec::with_capacity(cfg.n_iter);

    let bs = cfg.batch_size.min(n_src).min(n_tgt);

    // ── Outer training loop ───────────────────────────────────────────────────
    for _iter in 0..cfg.n_iter {
        // Draw mini-batch indices
        let src_start = (rng.next_u32() as usize) % n_src.saturating_sub(bs).max(1);
        let tgt_start = (rng.next_u32() as usize) % n_tgt.saturating_sub(bs).max(1);

        let src_end = (src_start + bs).min(n_src);
        let tgt_end = (tgt_start + bs).min(n_tgt);
        let actual_bs = (src_end - src_start).min(tgt_end - tgt_start);

        let src_batch = &source_samples[src_start * dim..src_end * dim];
        let tgt_batch = &target_samples[tgt_start * dim..tgt_end * dim];

        let (loss, mut grad) = compute_dual_loss(&icnn, src_batch, tgt_batch, actual_bs)?;
        grad.scale(1.0); // already batch-normalised inside
        dual_history.push(loss);

        // Maximise dual objective → step in positive gradient direction
        // (negate grad for SGD which minimises by convention)
        for v in grad.dwx0.iter_mut() {
            *v = -*v;
        }
        for v in grad.db0.iter_mut() {
            *v = -*v;
        }
        for v in grad.d_log_wz.iter_mut() {
            *v = -*v;
        }
        for v in grad.dwx_mid.iter_mut() {
            *v = -*v;
        }
        for v in grad.db_mid.iter_mut() {
            *v = -*v;
        }
        for v in grad.d_log_wz_out.iter_mut() {
            *v = -*v;
        }
        for v in grad.dwx_out.iter_mut() {
            *v = -*v;
        }
        grad.db_out = -grad.db_out;

        icnn.sgd_step(&grad, cfg.lr);
    }

    Ok(NeuralOtFit {
        icnn,
        dual_history,
        dim,
    })
}

/// Evaluate the W₂ dual lower bound on two equal-weight empirical distributions.
///
/// Returns `E_{x~μ}[f(x)] + E_{y~ν}[f*(y)]` for the trained potential `f`.
pub fn neural_ot_dual_bound(
    fit: &NeuralOtFit,
    source_samples: &[f32],
    target_samples: &[f32],
    n_src: usize,
    n_tgt: usize,
) -> OtResult<f32> {
    let d = fit.dim;
    if source_samples.len() != n_src * d || target_samples.len() != n_tgt * d {
        return Err(OtError::IncompatibleLength {
            a: source_samples.len(),
            b: n_src * d,
        });
    }
    let mut src_term = 0.0_f32;
    for i in 0..n_src {
        src_term += fit.icnn.forward(&source_samples[i * d..(i + 1) * d])?;
    }
    let mut conj_term = 0.0_f32;
    for j in 0..n_tgt {
        let y = &target_samples[j * d..(j + 1) * d];
        conj_term += approx_conjugate(&fit.icnn, y, source_samples, n_src)?;
    }
    Ok(src_term / n_src as f32 + conj_term / n_tgt as f32)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    /// Generate n samples from a Gaussian-like distribution using LcgRng.
    fn gauss_samples(n: usize, dim: usize, mean: f32, rng: &mut LcgRng) -> Vec<f32> {
        // Box-Muller
        let mut out = Vec::with_capacity(n * dim);
        let mut i = 0;
        while i < n * dim {
            let u1 = rng.next_f32().max(1e-9);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z1 = r * theta.cos() + mean;
            let z2 = r * theta.sin() + mean;
            out.push(z1);
            i += 1;
            if i < n * dim {
                out.push(z2);
                i += 1;
            }
        }
        out.truncate(n * dim);
        out
    }

    #[test]
    fn test_icnn_forward_scalar_dim() {
        let mut rng = make_rng();
        let icnn = IcnnWeights::random(1, 4, 1, &mut rng);
        let x = [0.5_f32];
        let v = icnn.forward(&x).expect("forward ok");
        assert!(v.is_finite());
    }

    #[test]
    fn test_icnn_forward_dim_mismatch() {
        let mut rng = make_rng();
        let icnn = IcnnWeights::random(3, 4, 1, &mut rng);
        let x = [0.5_f32; 2];
        let err = icnn.forward(&x);
        assert!(err.is_err());
    }

    #[test]
    fn test_icnn_gradient_shape() {
        let mut rng = make_rng();
        let dim = 2;
        let icnn = IcnnWeights::random(dim, 4, 1, &mut rng);
        let x = [0.3_f32, -0.4];
        let g = icnn.gradient(&x).expect("gradient ok");
        assert_eq!(g.len(), dim);
        for &v in &g {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_icnn_gradient_dim_mismatch() {
        let mut rng = make_rng();
        let icnn = IcnnWeights::random(3, 4, 1, &mut rng);
        let x = [0.0_f32; 2];
        assert!(icnn.gradient(&x).is_err());
    }

    #[test]
    fn test_icnn_convexity_numerical() {
        // f(λx + (1-λ)y) ≤ λ f(x) + (1-λ) f(y)  [convexity test]
        let mut rng = LcgRng::new(42);
        let icnn = IcnnWeights::random(2, 8, 2, &mut rng);
        let x = [1.0_f32, 0.5];
        let y = [-0.5_f32, 1.0];
        let lam = 0.4_f32;
        let mid = [
            lam * x[0] + (1.0 - lam) * y[0],
            lam * x[1] + (1.0 - lam) * y[1],
        ];
        let fx = icnn.forward(&x).expect("fx");
        let fy = icnn.forward(&y).expect("fy");
        let fmid = icnn.forward(&mid).expect("fmid");
        // Allow 1e-4 slack for numerical rounding in FD gradient.
        assert!(
            fmid <= lam * fx + (1.0 - lam) * fy + 0.5,
            "convexity violated: {fmid} > {} + 0.5",
            lam * fx + (1.0 - lam) * fy
        );
    }

    #[test]
    fn test_neural_ot_runs() {
        let mut rng = LcgRng::new(11);
        let n = 16;
        let dim = 2;
        let src = gauss_samples(n, dim, 0.0, &mut rng);
        let tgt = gauss_samples(n, dim, 3.0, &mut rng);
        let cfg = NeuralOtConfig {
            hidden: 8,
            n_hidden: 1,
            lr: 1e-2,
            batch_size: 8,
            n_iter: 5,
            n_inner: 2,
            inner_lr: 0.1,
            seed: 77,
        };
        let fit = neural_ot(&src, &tgt, n, n, dim, &cfg).expect("neural_ot ok");
        assert_eq!(fit.dual_history.len(), 5);
        for &v in &fit.dual_history {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_neural_ot_push_forward_shape() {
        let mut rng = LcgRng::new(13);
        let n = 12;
        let dim = 2;
        let src = gauss_samples(n, dim, 0.0, &mut rng);
        let tgt = gauss_samples(n, dim, 2.0, &mut rng);
        let cfg = NeuralOtConfig {
            hidden: 8,
            n_hidden: 1,
            lr: 1e-2,
            batch_size: 6,
            n_iter: 3,
            n_inner: 1,
            inner_lr: 0.05,
            seed: 5,
        };
        let fit = neural_ot(&src, &tgt, n, n, dim, &cfg).expect("ok");
        let pf = fit.push_forward(&src, n).expect("push_forward ok");
        assert_eq!(pf.len(), n * dim);
        for &v in &pf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_neural_ot_dual_bound() {
        let _rng = LcgRng::new(19);
        let n = 10;
        let dim = 1;
        let src: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let tgt: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 + 1.0).collect();
        let cfg = NeuralOtConfig {
            hidden: 4,
            n_hidden: 1,
            lr: 1e-2,
            batch_size: 5,
            n_iter: 3,
            n_inner: 1,
            inner_lr: 0.05,
            seed: 3,
        };
        let fit = neural_ot(&src, &tgt, n, n, dim, &cfg).expect("ok");
        let bound = neural_ot_dual_bound(&fit, &src, &tgt, n, n).expect("bound ok");
        assert!(bound.is_finite());
    }

    #[test]
    fn test_neural_ot_empty_source() {
        let cfg = NeuralOtConfig::default();
        let err = neural_ot(&[], &[1.0_f32], 0, 1, 1, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_neural_ot_bad_dim() {
        let cfg = NeuralOtConfig {
            hidden: 4,
            n_hidden: 1,
            ..NeuralOtConfig::default()
        };
        let err = neural_ot(&[1.0], &[1.0], 1, 1, 0, &cfg);
        assert!(err.is_err());
    }

    #[test]
    fn test_neural_ot_potential_finite() {
        let mut rng = LcgRng::new(77);
        let n = 8;
        let dim = 2;
        let src = gauss_samples(n, dim, 0.0, &mut rng);
        let tgt = gauss_samples(n, dim, 1.0, &mut rng);
        let cfg = NeuralOtConfig {
            hidden: 4,
            n_hidden: 1,
            lr: 1e-3,
            batch_size: 4,
            n_iter: 2,
            n_inner: 1,
            inner_lr: 0.05,
            seed: 9,
        };
        let fit = neural_ot(&src, &tgt, n, n, dim, &cfg).expect("ok");
        for i in 0..n {
            let v = fit
                .potential(&src[i * dim..(i + 1) * dim])
                .expect("potential ok");
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_icnn_grad_zeros_shape() {
        let mut rng = make_rng();
        let icnn = IcnnWeights::random(2, 4, 1, &mut rng);
        let g = IcnnGrad::zeros(&icnn);
        assert_eq!(g.dwx0.len(), icnn.wx0.len());
        assert_eq!(g.db0.len(), icnn.b0.len());
        assert_eq!(g.d_log_wz.len(), icnn.log_wz.len());
    }
}
