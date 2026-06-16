//! GANITE — Generative Adversarial Nets for Individualized Treatment Effects.
//!
//! Reference: Yoon, J., Jordon, J. & van der Schaar, M. (2018). "GANITE:
//! Estimation of Individualized Treatment Effects using Generative Adversarial
//! Nets." *International Conference on Learning Representations* (ICLR 2018).
//!
//! # Overview
//!
//! GANITE casts counterfactual inference as a *missing-data imputation* problem
//! solved with two adversarial games over single-hidden-layer perceptrons.
//!
//! 1. **Counterfactual block** `(G, D_cf)`.
//!    The *counterfactual generator* `G(x, t, y_f, z_G)` maps a unit's
//!    covariates `x`, its observed (factual) treatment `t` and outcome `y_f`,
//!    plus noise `z_G`, to a complete pair of potential outcomes
//!    `ỹ = (ỹ_0, ỹ_1)`. The factual coordinate is then *overwritten* with the
//!    true `y_f` to form the imputed complete vector `ȳ`. A *counterfactual
//!    discriminator* `D_cf(x, ȳ)` is trained to recover **which** coordinate
//!    was the factual one; `G` is trained to fool it (minimax), so that the
//!    imputed counterfactuals become indistinguishable from factuals.
//!
//! 2. **ITE block** `(I, D_ite)` — *implemented here as a supervised
//!    regressor* `I(x)` trained on the completed dataset `{x, ȳ}` produced by
//!    `G`. `I` outputs both potential outcomes `(ŷ_0, ŷ_1)`; the predicted
//!    individualized treatment effect is `τ̂(x) = ŷ_1 − ŷ_0`. (The original
//!    paper wraps `I` in a second GAN against `D_ite`; we use the supervised
//!    surrogate, which is the standard CPU-friendly reduction and yields the
//!    same population CATE under a correctly trained `G`.)
//!
//! All nets are single-hidden-layer ReLU MLPs in FP32. Optimisation is plain
//! SGD with analytic backpropagation; randomness (weight init + GAN noise +
//! minibatch shuffling) is fully deterministic via [`LcgRng`].
//!
//! ## Losses
//!
//! - Discriminator (binary cross-entropy, label = factual treatment `t`):
//!   `L_D = −[ t·log σ(d) + (1−t)·log(1−σ(d)) ]` where `d = D_cf(x, ȳ)`.
//! - Generator: supervised factual reconstruction `(ỹ_t − y_f)²` plus the
//!   adversarial term `−L_D` (push `D_cf` toward chance).
//! - Inference net `I`: squared error against the *completed* targets `ȳ`.

use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

// =====================================================================
// small dense-layer helpers (single hidden layer perceptron)
// =====================================================================

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn drelu(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable logistic.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// A single fully-connected layer `y = W x + b` with row-major `W` of shape
/// `(out, in)`.
#[derive(Debug, Clone)]
struct Dense {
    w: Vec<f32>,
    b: Vec<f32>,
    fan_in: usize,
}

impl Dense {
    fn new(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0_f32 / fan_in.max(1) as f32).sqrt();
        let w = (0..fan_in * fan_out)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self {
            w,
            b: vec![0.0_f32; fan_out],
            fan_in,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.b.clone();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.w[o * self.fan_in..(o + 1) * self.fan_in];
            let dot: f32 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
            *slot += dot;
        }
        out
    }
}

/// A two-layer ReLU MLP: `out = W2 · relu(W1 x + b1) + b2`.
#[derive(Debug, Clone)]
struct Mlp {
    l1: Dense,
    l2: Dense,
    hidden: usize,
}

impl Mlp {
    fn new(input: usize, hidden: usize, output: usize, rng: &mut LcgRng) -> Self {
        Self {
            l1: Dense::new(input, hidden, rng),
            l2: Dense::new(hidden, output, rng),
            hidden,
        }
    }

    /// Forward pass returning `(pre_activation_hidden, hidden, output)` so the
    /// backward pass can reuse the cached intermediates.
    fn forward_cache(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let z1 = self.l1.forward(x);
        let h: Vec<f32> = z1.iter().map(|&v| relu(v)).collect();
        let out = self.l2.forward(&h);
        (z1, h, out)
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (_, _, o) = self.forward_cache(x);
        o
    }

    /// Backpropagate an upstream gradient `d_out` (length `output`) through the
    /// MLP, applying an in-place SGD update with learning rate `lr`. Returns the
    /// gradient w.r.t. the input `x` (length `input`), enabling chained
    /// generator→discriminator adversarial gradients.
    fn backward_sgd(
        &mut self,
        x: &[f32],
        z1: &[f32],
        h: &[f32],
        d_out: &[f32],
        lr: f32,
    ) -> Vec<f32> {
        let in_dim = self.l1.fan_in;

        // Gradient into hidden activations: dh = W2^T d_out.
        let mut dh = vec![0.0_f32; self.hidden];
        for (o, &g) in d_out.iter().enumerate() {
            let row = &self.l2.w[o * self.hidden..(o + 1) * self.hidden];
            for (acc, &w) in dh.iter_mut().zip(row.iter()) {
                *acc += w * g;
            }
        }
        // Through ReLU.
        let mut dz1 = vec![0.0_f32; self.hidden];
        for ((slot, &dhk), &z) in dz1.iter_mut().zip(dh.iter()).zip(z1.iter()) {
            *slot = dhk * drelu(z);
        }
        // Gradient into input: dx = W1^T dz1 (computed before W1 update).
        let mut dx = vec![0.0_f32; in_dim];
        for (k, &g) in dz1.iter().enumerate() {
            let row = &self.l1.w[k * in_dim..(k + 1) * in_dim];
            for (acc, &w) in dx.iter_mut().zip(row.iter()) {
                *acc += w * g;
            }
        }

        // --- parameter updates ---
        // Layer 2: dW2 = d_out ⊗ h, db2 = d_out.
        for (o, &g) in d_out.iter().enumerate() {
            let row = &mut self.l2.w[o * self.hidden..(o + 1) * self.hidden];
            for (w, &hk) in row.iter_mut().zip(h.iter()) {
                *w -= lr * g * hk;
            }
            self.l2.b[o] -= lr * g;
        }
        // Layer 1: dW1 = dz1 ⊗ x, db1 = dz1.
        for (k, &g) in dz1.iter().enumerate() {
            let row = &mut self.l1.w[k * in_dim..(k + 1) * in_dim];
            for (w, &xi) in row.iter_mut().zip(x.iter()) {
                *w -= lr * g * xi;
            }
            self.l1.b[k] -= lr * g;
        }
        dx
    }
}

// =====================================================================
// configuration
// =====================================================================

/// Configuration for [`Ganite`].
#[derive(Debug, Clone)]
pub struct GaniteConfig {
    /// Hidden width of every MLP.
    pub hidden_dim: usize,
    /// Dimension of the generator noise vector `z_G`.
    pub noise_dim: usize,
    /// Training epochs (full passes over the dataset).
    pub epochs: usize,
    /// SGD learning rate.
    pub lr: f32,
    /// Number of discriminator steps per generator step (`k` in the GAN
    /// literature). Must be `≥ 1`.
    pub disc_steps: usize,
    /// Weight on the generator's supervised factual-reconstruction loss
    /// relative to the adversarial term (`α` in the paper).
    pub alpha: f32,
}

impl Default for GaniteConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 16,
            noise_dim: 4,
            epochs: 200,
            lr: 0.01,
            disc_steps: 1,
            alpha: 1.0,
        }
    }
}

impl GaniteConfig {
    fn validate(&self) -> CausalResult<()> {
        if self.hidden_dim == 0 || self.disc_steps == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "hidden_dim and disc_steps must be >= 1".to_string(),
            });
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("lr must be finite and > 0, got {}", self.lr),
            });
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(CausalError::InvalidParameter {
                reason: format!("alpha must be finite and >= 0, got {}", self.alpha),
            });
        }
        Ok(())
    }
}

// =====================================================================
// GANITE model
// =====================================================================

/// A trained (or untrained) GANITE estimator.
///
/// - The **generator** consumes `[x, t, y_f, z_G]` and emits two potential
///   outcomes `(ỹ_0, ỹ_1)`.
/// - The **discriminator** consumes `[x, ȳ_0, ȳ_1]` and emits a single logit;
///   `σ(logit)` is its estimate of `P(T = 1)` (i.e. which coordinate is
///   factual).
/// - The **inference net** consumes `x` and emits `(ŷ_0, ŷ_1)`; the ITE is
///   `ŷ_1 − ŷ_0`.
#[derive(Debug, Clone)]
pub struct Ganite {
    generator: Mlp,
    discriminator: Mlp,
    inference: Mlp,
    input_dim: usize,
    noise_dim: usize,
}

impl Ganite {
    /// Construct a GANITE model with freshly initialised weights.
    ///
    /// # Errors
    /// - [`CausalError::InvalidParameter`] if `input_dim == 0` or the config is
    ///   invalid.
    pub fn new(input_dim: usize, cfg: &GaniteConfig, rng: &mut LcgRng) -> CausalResult<Self> {
        cfg.validate()?;
        if input_dim == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "input_dim must be >= 1".to_string(),
            });
        }
        // Generator input: x (input_dim) + t (1) + y_f (1) + noise (noise_dim).
        let g_in = input_dim + 2 + cfg.noise_dim;
        // Discriminator input: x + two potential outcomes.
        let d_in = input_dim + 2;
        Ok(Self {
            generator: Mlp::new(g_in, cfg.hidden_dim, 2, rng),
            discriminator: Mlp::new(d_in, cfg.hidden_dim, 1, rng),
            inference: Mlp::new(input_dim, cfg.hidden_dim, 2, rng),
            input_dim,
            noise_dim: cfg.noise_dim,
        })
    }

    /// Generate the complete imputed potential-outcome pair `ȳ = (ȳ_0, ȳ_1)`
    /// for one unit, with the factual coordinate overwritten by `y_f`.
    fn generate_complete(&self, x: &[f32], t: f32, y_f: f32, z: &[f32]) -> [f32; 2] {
        let mut g_input = Vec::with_capacity(self.input_dim + 2 + self.noise_dim);
        g_input.extend_from_slice(x);
        g_input.push(t);
        g_input.push(y_f);
        g_input.extend_from_slice(z);
        let out = self.generator.forward(&g_input);
        let mut y = [out[0], out[1]];
        // Overwrite factual coordinate.
        if t >= 0.5 {
            y[1] = y_f;
        } else {
            y[0] = y_f;
        }
        y
    }

    /// Train the GANITE model on observed data.
    ///
    /// # Parameters
    /// - `x`: row-major `n × input_dim` covariates.
    /// - `t`: length `n` binary treatment in `{0.0, 1.0}`.
    /// - `y`: length `n` factual outcomes.
    /// - `n`: number of samples (`> 0`).
    /// - `cfg`: training hyper-parameters.
    ///
    /// # Errors
    /// - [`CausalError::EmptyInput`] if `n == 0` or any slice empty.
    /// - [`CausalError::DimensionMismatch`] if slice lengths disagree with `n`
    ///   / `input_dim`.
    /// - [`CausalError::InvalidParameter`] if any `t[i] ∉ {0,1}` or config
    ///   invalid.
    pub fn fit(
        &mut self,
        x: &[f32],
        t: &[f32],
        y: &[f32],
        n: usize,
        cfg: &GaniteConfig,
        rng: &mut LcgRng,
    ) -> CausalResult<()> {
        cfg.validate()?;
        let d = self.input_dim;
        if n == 0 || x.is_empty() || t.is_empty() || y.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: x.len(),
            });
        }
        if t.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: t.len(),
            });
        }
        if y.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }
        for &ti in t {
            if !(ti == 0.0 || ti == 1.0) {
                return Err(CausalError::InvalidParameter {
                    reason: "treatment must be binary {0,1}".to_string(),
                });
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        for _ in 0..cfg.epochs {
            rng.shuffle_indices(&mut order);
            for &i in &order {
                let xi = &x[i * d..(i + 1) * d];
                let ti = t[i];
                let yi = y[i];

                // === counterfactual-block GAN step(s) ===
                for _ in 0..cfg.disc_steps {
                    self.disc_step(xi, ti, yi, cfg, rng);
                }
                self.gen_step(xi, ti, yi, cfg, rng);

                // === inference-net supervised step on completed data ===
                self.inference_step(xi, ti, yi, cfg, rng);
            }
        }
        Ok(())
    }

    /// One discriminator update (maximise correct factual classification).
    fn disc_step(&mut self, xi: &[f32], ti: f32, yi: f32, cfg: &GaniteConfig, rng: &mut LcgRng) {
        let z = sample_noise(self.noise_dim, rng);
        let ybar = self.generate_complete(xi, ti, yi, &z);
        let mut d_input = Vec::with_capacity(self.input_dim + 2);
        d_input.extend_from_slice(xi);
        d_input.push(ybar[0]);
        d_input.push(ybar[1]);
        let (z1, h, out) = self.discriminator.forward_cache(&d_input);
        let p = sigmoid(out[0]);
        // BCE with label = ti (1 ⇒ coordinate 1 is factual). dL/d_logit = p − t.
        let d_logit = p - ti;
        self.discriminator
            .backward_sgd(&d_input, &z1, &h, &[d_logit], cfg.lr);
    }

    /// One generator update: supervised factual reconstruction + adversarial.
    fn gen_step(&mut self, xi: &[f32], ti: f32, yi: f32, cfg: &GaniteConfig, rng: &mut LcgRng) {
        let z = sample_noise(self.noise_dim, rng);
        let mut g_input = Vec::with_capacity(self.input_dim + 2 + self.noise_dim);
        g_input.extend_from_slice(xi);
        g_input.push(ti);
        g_input.push(yi);
        g_input.extend_from_slice(&z);
        let (gz1, gh, gout) = self.generator.forward_cache(&g_input);

        // Completed vector with factual overwrite.
        let mut ybar = [gout[0], gout[1]];
        let fac = if ti >= 0.5 { 1usize } else { 0usize };
        let cf = 1 - fac;
        ybar[fac] = yi;

        // Discriminator forward on completed vector (no D update here).
        let mut d_input = Vec::with_capacity(self.input_dim + 2);
        d_input.extend_from_slice(xi);
        d_input.push(ybar[0]);
        d_input.push(ybar[1]);
        let (dz1, dh, dout) = self.discriminator.forward_cache(&d_input);
        let p = sigmoid(dout[0]);

        // Adversarial gradient: generator wants D to predict factual=0.5, i.e.
        // pushes the discriminator logit toward the *wrong* label (1 − ti).
        // Treating D as fixed, dL_adv/d_logit = p − (1 − ti).
        let d_logit_adv = p - (1.0 - ti);
        // Backprop through D *without* updating its parameters to obtain the
        // gradient w.r.t. d_input = [x, ybar0, ybar1].
        let d_dinput = backprop_input_only(&self.discriminator, &dz1, &dh, &[d_logit_adv]);
        // Gradient w.r.t. the generator's two outputs from the adversarial path
        // (only the counterfactual coordinate flows back; factual is overwritten
        // by the constant y_f and carries no generator gradient).
        let mut d_gout = [0.0_f32; 2];
        // d_dinput layout: [d/dx (input_dim), d/dybar0, d/dybar1].
        d_gout[0] = d_dinput[self.input_dim];
        d_gout[1] = d_dinput[self.input_dim + 1];
        // Kill factual coordinate (constant), keep counterfactual.
        d_gout[fac] = 0.0;

        // Supervised factual reconstruction: 0.5·(gout[fac] − y_f)² ⇒
        // d/dgout[fac] = (gout[fac] − y_f) · alpha.
        d_gout[fac] += cfg.alpha * (gout[fac] - yi);
        // (counterfactual coordinate gets only the adversarial signal)
        let _ = cf;

        self.generator
            .backward_sgd(&g_input, &gz1, &gh, &d_gout, cfg.lr);
    }

    /// One supervised inference-net step against the completed targets.
    fn inference_step(
        &mut self,
        xi: &[f32],
        ti: f32,
        yi: f32,
        cfg: &GaniteConfig,
        rng: &mut LcgRng,
    ) {
        let z = sample_noise(self.noise_dim, rng);
        let ybar = self.generate_complete(xi, ti, yi, &z);
        let (z1, h, out) = self.inference.forward_cache(xi);
        // MSE against completed potential outcomes.
        let d_out = [out[0] - ybar[0], out[1] - ybar[1]];
        self.inference.backward_sgd(xi, &z1, &h, &d_out, cfg.lr);
    }

    /// Predict the pair of potential outcomes `(ŷ_0, ŷ_1)` for a single unit.
    ///
    /// # Errors
    /// - [`CausalError::DimensionMismatch`] if `x.len() != input_dim`.
    pub fn predict_outcomes(&self, x: &[f32]) -> CausalResult<(f32, f32)> {
        if x.len() != self.input_dim {
            return Err(CausalError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let out = self.inference.forward(x);
        Ok((out[0], out[1]))
    }

    /// Predict the individualized treatment effect `τ̂(x) = ŷ_1 − ŷ_0`.
    ///
    /// # Errors
    /// See [`Self::predict_outcomes`].
    pub fn predict_ite(&self, x: &[f32]) -> CausalResult<f32> {
        let (y0, y1) = self.predict_outcomes(x)?;
        Ok(y1 - y0)
    }

    /// Predict ITEs for a row-major `n × input_dim` batch.
    ///
    /// # Errors
    /// - [`CausalError::EmptyInput`] if `n == 0`.
    /// - [`CausalError::DimensionMismatch`] if `x.len() != n · input_dim`.
    pub fn predict_ite_batch(&self, x: &[f32], n: usize) -> CausalResult<Vec<f32>> {
        if n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * self.input_dim {
            return Err(CausalError::DimensionMismatch {
                expected: n * self.input_dim,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.predict_ite(&x[i * self.input_dim..(i + 1) * self.input_dim])?);
        }
        Ok(out)
    }

    /// Average treatment effect estimate over a batch: `mean_i τ̂(x_i)`.
    ///
    /// # Errors
    /// See [`Self::predict_ite_batch`].
    pub fn estimate_ate(&self, x: &[f32], n: usize) -> CausalResult<f32> {
        let ites = self.predict_ite_batch(x, n)?;
        Ok(ites.iter().sum::<f32>() / n as f32)
    }

    /// Input covariate dimension.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

/// Backprop an upstream gradient through an MLP returning *only* the gradient
/// w.r.t. its input, leaving parameters untouched (used for adversarial flow).
fn backprop_input_only(mlp: &Mlp, z1: &[f32], _h: &[f32], d_out: &[f32]) -> Vec<f32> {
    let in_dim = mlp.l1.fan_in;
    let mut dh = vec![0.0_f32; mlp.hidden];
    for (o, &g) in d_out.iter().enumerate() {
        let row = &mlp.l2.w[o * mlp.hidden..(o + 1) * mlp.hidden];
        for (acc, &w) in dh.iter_mut().zip(row.iter()) {
            *acc += w * g;
        }
    }
    let mut dz1 = vec![0.0_f32; mlp.hidden];
    for ((slot, &dhk), &z) in dz1.iter_mut().zip(dh.iter()).zip(z1.iter()) {
        *slot = dhk * drelu(z);
    }
    let mut dx = vec![0.0_f32; in_dim];
    for (k, &g) in dz1.iter().enumerate() {
        let row = &mlp.l1.w[k * in_dim..(k + 1) * in_dim];
        for (acc, &w) in dx.iter_mut().zip(row.iter()) {
            *acc += w * g;
        }
    }
    dx
}

/// Sample a uniform noise vector in `[−1, 1]^noise_dim`.
fn sample_noise(noise_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    (0..noise_dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

// LcgRng helper: Fisher-Yates shuffle of an index vector. Implemented as a free
// function so we don't depend on a particular handle method name.
trait ShuffleIndices {
    fn shuffle_indices(&mut self, idx: &mut [usize]);
}

impl ShuffleIndices for LcgRng {
    fn shuffle_indices(&mut self, idx: &mut [usize]) {
        let n = idx.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            idx.swap(i, j);
        }
    }
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> GaniteConfig {
        GaniteConfig {
            hidden_dim: 8,
            noise_dim: 2,
            epochs: 60,
            lr: 0.02,
            disc_steps: 1,
            alpha: 1.0,
        }
    }

    /// Synthetic data: Y0 = 0.5·x0, Y1 = 0.5·x0 + tau (constant effect),
    /// treatment assigned at random. Returns (x, t, y, true_ite).
    fn make_data(n: usize, d: usize, tau: f32, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let mut rng = LcgRng::new(seed);
        let mut x = vec![0.0_f32; n * d];
        for v in x.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let mut t = vec![0.0_f32; n];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            t[i] = if rng.next_f32() > 0.5 { 1.0 } else { 0.0 };
            let base = 0.5 * x[i * d];
            y[i] = base + tau * t[i];
        }
        (x, t, y, tau)
    }

    // -------------------- construction validation --------------------------

    #[test]
    fn new_input_dim_0_error() {
        let mut rng = LcgRng::new(1);
        let r = Ganite::new(0, &small_cfg(), &mut rng);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn new_bad_lr_error() {
        let mut rng = LcgRng::new(1);
        let cfg = GaniteConfig {
            lr: 0.0,
            ..small_cfg()
        };
        let r = Ganite::new(3, &cfg, &mut rng);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn new_zero_hidden_error() {
        let mut rng = LcgRng::new(1);
        let cfg = GaniteConfig {
            hidden_dim: 0,
            ..small_cfg()
        };
        let r = Ganite::new(3, &cfg, &mut rng);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    // -------------------- fit validation -----------------------------------

    #[test]
    fn fit_dim_mismatch_x_error() {
        let mut rng = LcgRng::new(2);
        let cfg = small_cfg();
        let mut g = Ganite::new(2, &cfg, &mut rng)
            .expect("Ganite::new with valid 2-feature config should succeed");
        // x has wrong length (5 instead of n*d = 6).
        let r = g.fit(
            &[0.0; 5],
            &[1.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0],
            3,
            &cfg,
            &mut rng,
        );
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn fit_non_binary_treatment_error() {
        let mut rng = LcgRng::new(3);
        let cfg = small_cfg();
        let mut g = Ganite::new(1, &cfg, &mut rng)
            .expect("Ganite::new with valid 1-feature config should succeed");
        let r = g.fit(&[0.0, 1.0], &[0.5, 0.0], &[1.0, 2.0], 2, &cfg, &mut rng);
        assert!(matches!(r, Err(CausalError::InvalidParameter { .. })));
    }

    #[test]
    fn fit_empty_error() {
        let mut rng = LcgRng::new(3);
        let cfg = small_cfg();
        let mut g = Ganite::new(1, &cfg, &mut rng)
            .expect("Ganite::new with valid 1-feature config should succeed");
        let r = g.fit(&[], &[], &[], 0, &cfg, &mut rng);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    // -------------------- forward / shape ----------------------------------

    #[test]
    fn predict_outcomes_finite() {
        let mut rng = LcgRng::new(7);
        let cfg = small_cfg();
        let g = Ganite::new(3, &cfg, &mut rng)
            .expect("Ganite::new with valid 3-feature config should succeed");
        let (y0, y1) = g
            .predict_outcomes(&[0.1, -0.2, 0.3])
            .expect("predict_outcomes with correct input dim should succeed");
        assert!(y0.is_finite() && y1.is_finite());
    }

    #[test]
    fn predict_outcomes_wrong_dim_error() {
        let mut rng = LcgRng::new(7);
        let cfg = small_cfg();
        let g = Ganite::new(3, &cfg, &mut rng)
            .expect("Ganite::new with valid 3-feature config should succeed");
        let r = g.predict_outcomes(&[0.1, -0.2]); // dim 2 != 3
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn ite_batch_shape() {
        let mut rng = LcgRng::new(9);
        let cfg = small_cfg();
        let g = Ganite::new(2, &cfg, &mut rng)
            .expect("Ganite::new with valid 2-feature config should succeed");
        let n = 5;
        let x: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.1).collect();
        let ites = g
            .predict_ite_batch(&x, n)
            .expect("predict_ite_batch should succeed");
        assert_eq!(ites.len(), n);
        assert!(ites.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ite_batch_dim_mismatch_error() {
        let mut rng = LcgRng::new(9);
        let cfg = small_cfg();
        let g = Ganite::new(2, &cfg, &mut rng).expect("new should succeed");
        let r = g.predict_ite_batch(&[0.0; 7], 5); // 7 != 5*2
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    // -------------------- training behaviour -------------------------------

    #[test]
    fn fit_runs_and_predicts_finite() {
        let mut rng = LcgRng::new(123);
        let cfg = small_cfg();
        let (x, t, y, _) = make_data(60, 2, 2.0, 42);
        let mut g = Ganite::new(2, &cfg, &mut rng).expect("new should succeed");
        g.fit(&x, &t, &y, 60, &cfg, &mut rng)
            .expect("fit should succeed");
        let ites = g
            .predict_ite_batch(&x, 60)
            .expect("predict_ite_batch should succeed");
        assert!(ites.iter().all(|v| v.is_finite()));
        let ate = g.estimate_ate(&x, 60).expect("estimate_ate should succeed");
        assert!(ate.is_finite());
    }

    /// After training on a constant-effect dataset, the estimated ATE should
    /// move toward the true tau (sign-correct and within a generous tolerance,
    /// since this is a small SGD-trained GAN surrogate).
    #[test]
    fn fit_recovers_positive_ate_sign() {
        let mut rng = LcgRng::new(2024);
        let cfg = GaniteConfig {
            hidden_dim: 16,
            noise_dim: 2,
            epochs: 300,
            lr: 0.02,
            disc_steps: 1,
            alpha: 2.0,
        };
        let tau = 3.0;
        let (x, t, y, _) = make_data(120, 2, tau, 7);
        let mut g = Ganite::new(2, &cfg, &mut rng).expect("new should succeed");
        g.fit(&x, &t, &y, 120, &cfg, &mut rng)
            .expect("fit should succeed");
        let ate = g
            .estimate_ate(&x, 120)
            .expect("estimate_ate should succeed");
        // Sign-correct and not collapsed to ~0.
        assert!(ate > 0.5, "expected positive ATE near {tau}, got {ate}");
    }

    /// Deterministic: identical seeds produce identical predictions.
    #[test]
    fn deterministic_training() {
        let cfg = small_cfg();
        let (x, t, y, _) = make_data(40, 2, 1.5, 11);

        let mut rng_a = LcgRng::new(555);
        let mut ga = Ganite::new(2, &cfg, &mut rng_a).expect("new should succeed");
        ga.fit(&x, &t, &y, 40, &cfg, &mut rng_a)
            .expect("fit should succeed");
        let ites_a = ga
            .predict_ite_batch(&x, 40)
            .expect("predict_ite_batch should succeed");

        let mut rng_b = LcgRng::new(555);
        let mut gb = Ganite::new(2, &cfg, &mut rng_b).expect("new should succeed");
        gb.fit(&x, &t, &y, 40, &cfg, &mut rng_b)
            .expect("fit should succeed");
        let ites_b = gb
            .predict_ite_batch(&x, 40)
            .expect("predict_ite_batch should succeed");

        assert_eq!(ites_a, ites_b);
    }

    #[test]
    fn config_default_is_sane() {
        let cfg = GaniteConfig::default();
        assert!(cfg.validate().is_ok());
        assert!(cfg.hidden_dim >= 1);
    }
}
