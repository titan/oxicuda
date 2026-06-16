//! JEM — Joint Energy Model (Grathwohl et al. 2019).
//!
//! Interprets a classifier `f(x)` as an energy model:
//! ```text
//!   p(x) ∝ exp(logsumexp(f(x)))
//!   p(y|x) = softmax(f(x))[y]
//!   E(x) = -logsumexp(f(x))
//! ```
//!
//! Training uses contrastive divergence: maximise `log p(x_data) - log p(x_mcmc)`
//! where `x_mcmc` samples are generated via SGLD (Stochastic Gradient Langevin
//! Dynamics).
//!
//! ## SGLD update rule
//! ```text
//!   x_{t+1} = x_t - (step_size/2) * ∇_x E(x_t) + noise * N(0,1)
//! ```
//! The energy gradient is approximated with central finite differences:
//! ```text
//!   ∂E/∂x_i ≈ (E(x + ε·eᵢ) − E(x − ε·eᵢ)) / (2ε)
//! ```
//!
//! A persistent replay buffer stores MCMC chains between training steps so that
//! the distribution slowly converges to `p_θ(x)`.
//!
//! Reference: "Your Classifier is Secretly an Energy Based Model and You Should
//! Treat it Like One", Grathwohl et al., ICLR 2020.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for the [`Jem`] joint energy model.
#[derive(Debug, Clone)]
pub struct JemConfig {
    /// Input dimension (feature / latent vector length).
    pub d_input: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Hidden layer dimension (single hidden layer MLP).
    pub n_hidden: usize,
    /// Number of SGLD steps per [`Jem::sgld_step`] call.
    pub sgld_steps: usize,
    /// SGLD gradient step size `α`.
    pub sgld_step_size: f32,
    /// SGLD noise scale `σ` applied to the Langevin noise term.
    pub sgld_noise: f32,
    /// Number of persistent MCMC chains stored in the replay buffer.
    pub buffer_size: usize,
}

impl Default for JemConfig {
    fn default() -> Self {
        Self {
            d_input: 4,
            n_classes: 2,
            n_hidden: 16,
            sgld_steps: 20,
            sgld_step_size: 0.01,
            sgld_noise: 0.005,
            buffer_size: 64,
        }
    }
}

// ─── JEM model ────────────────────────────────────────────────────────────────

/// Joint Energy Model that combines a discriminative classifier with an
/// energy-based generative model via a shared 2-layer MLP.
///
/// Architecture: `d_input → (Linear + ReLU) → n_hidden → Linear → n_classes`
#[derive(Debug, Clone)]
pub struct Jem {
    /// First layer weights `[n_hidden × d_input]`.
    w1: Vec<f32>,
    /// First layer bias `[n_hidden]`.
    b1: Vec<f32>,
    /// Second layer weights `[n_classes × n_hidden]`.
    w2: Vec<f32>,
    /// Second layer bias `[n_classes]`.
    b2: Vec<f32>,
    /// Persistent MCMC replay buffer; each entry is a `[d_input]` vector.
    replay_buffer: Vec<Vec<f32>>,
    /// Model configuration.
    config: JemConfig,
}

impl Jem {
    /// Create a new [`Jem`] model with Kaiming-initialised weights and a
    /// small-noise replay buffer.
    ///
    /// # Errors
    /// [`SslError::InvalidParameter`] if `d_input`, `n_classes`, or `n_hidden` is zero.
    pub fn new(config: JemConfig, rng: &mut LcgRng) -> SslResult<Self> {
        if config.d_input == 0 {
            return Err(SslError::InvalidParameter {
                name: "d_input".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.n_classes == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_classes".into(),
                reason: "must be > 0".into(),
            });
        }
        if config.n_hidden == 0 {
            return Err(SslError::InvalidParameter {
                name: "n_hidden".into(),
                reason: "must be > 0".into(),
            });
        }

        let w1 = kaiming_init(config.n_hidden, config.d_input, rng);
        let b1 = vec![0.0_f32; config.n_hidden];
        let w2 = kaiming_init(config.n_classes, config.n_hidden, rng);
        let b2 = vec![0.0_f32; config.n_classes];

        // Initialise replay buffer with small normal noise in [-0.01, 0.01] range.
        let buf_size = config.buffer_size.max(1);
        let mut replay_buffer = Vec::with_capacity(buf_size);
        for _ in 0..buf_size {
            let mut entry = vec![0.0_f32; config.d_input];
            rng.fill_normal(&mut entry);
            for v in entry.iter_mut() {
                *v *= 0.01;
            }
            replay_buffer.push(entry);
        }

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            replay_buffer,
            config,
        })
    }

    /// Compute the logit vector `f(x)` of shape `[n_classes]`.
    ///
    /// Performs `x → W1·x + b1 → ReLU → W2·h + b2 → logits`.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] when `x.len() != d_input`.
    pub fn logits(&self, x: &[f32]) -> SslResult<Vec<f32>> {
        let d = self.config.d_input;
        if x.len() != d {
            return Err(SslError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let h = linear_relu(&self.w1, &self.b1, x, d, self.config.n_hidden);
        Ok(linear(
            &self.w2,
            &self.b2,
            &h,
            self.config.n_hidden,
            self.config.n_classes,
        ))
    }

    /// Compute the scalar energy `E(x) = -logsumexp(f(x))`.
    ///
    /// Uses the numerically stable max-shift trick for `logsumexp`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::logits`].
    pub fn energy(&self, x: &[f32]) -> SslResult<f32> {
        let logits = self.logits(x)?;
        Ok(-logsumexp(&logits))
    }

    /// Cross-entropy classification loss for a single sample.
    ///
    /// `CE(x, y) = -log softmax(f(x))[y] = logsumexp(f(x)) - f(x)[y]`
    ///
    /// # Errors
    /// - Propagates errors from [`Self::logits`].
    /// - [`SslError::InvalidParameter`] when `y >= n_classes`.
    pub fn classify_loss(&self, x: &[f32], y: usize) -> SslResult<f32> {
        if y >= self.config.n_classes {
            return Err(SslError::InvalidParameter {
                name: "y".into(),
                reason: "class index must be < n_classes".into(),
            });
        }
        let logits = self.logits(x)?;
        let lse = logsumexp(&logits);
        Ok(lse - logits[y])
    }

    /// Approximate `∇_x E(x)` via central finite differences.
    ///
    /// `grad[i] = (E(x + ε·eᵢ) − E(x − ε·eᵢ)) / (2ε)`
    ///
    /// # Arguments
    /// * `x`   — input `[d_input]`.
    /// * `eps` — finite-difference step size (e.g. `1e-3`).
    ///
    /// # Errors
    /// Propagates errors from [`Self::energy`].
    pub fn energy_grad(&self, x: &[f32], eps: f32) -> SslResult<Vec<f32>> {
        let d = self.config.d_input;
        if x.len() != d {
            return Err(SslError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        let two_eps = 2.0 * eps;
        let mut grad = vec![0.0_f32; d];
        let mut x_pos = x.to_vec();
        let mut x_neg = x.to_vec();
        for i in 0..d {
            x_pos[i] = x[i] + eps;
            x_neg[i] = x[i] - eps;
            let e_pos = self.energy(&x_pos)?;
            let e_neg = self.energy(&x_neg)?;
            grad[i] = (e_pos - e_neg) / two_eps;
            x_pos[i] = x[i];
            x_neg[i] = x[i];
        }
        Ok(grad)
    }

    /// Run SGLD for `config.sgld_steps` steps starting from `x_init`.
    ///
    /// ```text
    /// x_{t+1} = x_t − (step_size / 2) · ∇_x E(x_t) + sgld_noise · N(0, 1)
    /// ```
    ///
    /// Finite-difference gradient approximation is used with `eps = 1e-3`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::energy_grad`].
    pub fn sgld_step(&self, x_init: &[f32], rng: &mut LcgRng) -> SslResult<Vec<f32>> {
        let d = self.config.d_input;
        if x_init.len() != d {
            return Err(SslError::DimensionMismatch {
                expected: d,
                got: x_init.len(),
            });
        }
        let half_step = self.config.sgld_step_size * 0.5;
        let noise_scale = self.config.sgld_noise;
        let fd_eps = 1e-3_f32;

        let mut x = x_init.to_vec();
        for _ in 0..self.config.sgld_steps {
            let grad = self.energy_grad(&x, fd_eps)?;
            let mut noise = vec![0.0_f32; d];
            rng.fill_normal(&mut noise);
            for i in 0..d {
                x[i] -= half_step * grad[i];
                x[i] += noise_scale * noise[i];
            }
        }
        Ok(x)
    }

    /// Compute the contrastive divergence loss `E(x_mcmc) − E(x_data)`.
    ///
    /// 1. Randomly selects an entry from the replay buffer as the MCMC seed.
    /// 2. Runs [`Self::sgld_step`] to advance the chain.
    /// 3. Replaces the selected buffer entry with the updated MCMC sample.
    /// 4. Returns `E(x_mcmc_updated) − E(x_data)`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::sgld_step`] and [`Self::energy`].
    pub fn cd_loss(&mut self, x_data: &[f32], rng: &mut LcgRng) -> SslResult<f32> {
        let buf_len = self.replay_buffer.len();
        // Pick a random replay buffer index.
        let idx = rng.next_usize(buf_len);
        let x_mcmc_init = self.replay_buffer[idx].clone();
        // Run SGLD.
        let x_mcmc = self.sgld_step(&x_mcmc_init, rng)?;
        // Update the replay buffer slot.
        self.replay_buffer[idx] = x_mcmc.clone();
        // Compute energies.
        let e_mcmc = self.energy(&x_mcmc)?;
        let e_data = self.energy(x_data)?;
        Ok(e_mcmc - e_data)
    }

    /// Return the input dimension.
    #[inline]
    #[must_use]
    pub fn d_input(&self) -> usize {
        self.config.d_input
    }

    /// Return the number of output classes.
    #[inline]
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.config.n_classes
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Kaiming (He) normal weight init: `scale = sqrt(2 / fan_in)`.
fn kaiming_init(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (2.0_f32 / in_dim as f32).sqrt();
    let mut w = vec![0.0_f32; out_dim * in_dim];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= scale;
    }
    w
}

/// Row-major matrix-vector multiply: `out[i] = b[i] + Σ_j w[i·in_dim + j] * x[j]`.
fn linear(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        let row = i * in_dim;
        for j in 0..in_dim {
            acc += w[row + j] * x[j];
        }
        out[i] = acc;
    }
    out
}

/// `linear` followed by element-wise ReLU.
fn linear_relu(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = linear(w, b, x, in_dim, out_dim);
    for v in out.iter_mut() {
        *v = v.max(0.0);
    }
    out
}

/// Numerically stable `logsumexp` using the max-shift trick.
fn logsumexp(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return max;
    }
    let sum_exp: f32 = v.iter().map(|&x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_jem(seed: u64) -> Jem {
        let mut rng = LcgRng::new(seed);
        Jem::new(JemConfig::default(), &mut rng).expect("value should be present")
    }

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn logits_shape() {
        let j = make_jem(1);
        let x = random_vec(4, 2);
        let logits = j.logits(&x).expect("logits should succeed");
        assert_eq!(
            logits.len(),
            j.n_classes(),
            "logits len must equal n_classes"
        );
    }

    #[test]
    fn energy_finite() {
        let j = make_jem(3);
        let x = random_vec(4, 4);
        let e = j.energy(&x).expect("energy should succeed");
        assert!(e.is_finite(), "energy must be finite, got {e}");
    }

    #[test]
    fn classify_loss_finite() {
        let j = make_jem(5);
        let x = random_vec(4, 6);
        let ce = j
            .classify_loss(&x, 0)
            .expect("classify_loss should succeed");
        assert!(ce.is_finite(), "classify_loss must be finite, got {ce}");
    }

    #[test]
    fn classify_loss_nonneg() {
        let j = make_jem(7);
        let x = random_vec(4, 8);
        let ce = j
            .classify_loss(&x, 1)
            .expect("classify_loss should succeed");
        assert!(ce >= 0.0, "cross-entropy must be >= 0, got {ce}");
    }

    #[test]
    fn cd_loss_finite() {
        let mut rng = LcgRng::new(9);
        let mut j = Jem::new(JemConfig::default(), &mut rng).expect("value should be present");
        let x = random_vec(4, 10);
        let cd = j.cd_loss(&x, &mut rng).expect("cd_loss should succeed");
        assert!(cd.is_finite(), "cd_loss must be finite, got {cd}");
    }

    #[test]
    fn sgld_moves_from_init() {
        let mut rng = LcgRng::new(11);
        let j = Jem::new(JemConfig::default(), &mut rng).expect("value should be present");
        let x_init = random_vec(4, 12);
        let x_out = j
            .sgld_step(&x_init, &mut rng)
            .expect("sgld_step should succeed");
        let diff: f32 = x_init
            .iter()
            .zip(x_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-8, "SGLD must move from init, diff={diff}");
    }

    #[test]
    fn energy_grad_finite() {
        let j = make_jem(13);
        let x = random_vec(4, 14);
        let g = j.energy_grad(&x, 1e-3).expect("energy_grad should succeed");
        assert_eq!(g.len(), 4, "gradient must have len == d_input");
        assert!(
            g.iter().all(|v| v.is_finite()),
            "gradient must be all-finite"
        );
    }

    #[test]
    fn d_input_0_error() {
        let mut rng = LcgRng::new(15);
        let result = Jem::new(
            JemConfig {
                d_input: 0,
                ..JemConfig::default()
            },
            &mut rng,
        );
        assert!(result.is_err(), "d_input=0 must return Err");
    }

    #[test]
    fn n_classes_0_error() {
        let mut rng = LcgRng::new(16);
        let result = Jem::new(
            JemConfig {
                n_classes: 0,
                ..JemConfig::default()
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_classes=0 must return Err");
    }

    #[test]
    fn n_hidden_0_error() {
        let mut rng = LcgRng::new(17);
        let result = Jem::new(
            JemConfig {
                n_hidden: 0,
                ..JemConfig::default()
            },
            &mut rng,
        );
        assert!(result.is_err(), "n_hidden=0 must return Err");
    }

    #[test]
    fn classify_loss_invalid_class_error() {
        let j = make_jem(18);
        let x = random_vec(4, 19);
        // n_classes = 2, so y=2 is out of range
        let r = j.classify_loss(&x, 2);
        assert!(r.is_err(), "y >= n_classes must return Err");
    }

    #[test]
    fn d_input_n_classes_accessors() {
        let j = make_jem(20);
        assert_eq!(j.d_input(), 4);
        assert_eq!(j.n_classes(), 2);
    }
}
