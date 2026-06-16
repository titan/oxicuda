//! TabDDPM: Gaussian DDPM for tabular data.
//!
//! Reference: Kotelnikov, Baranchuk, Rubachev & Babenko (2023),
//! "TabDDPM: Modelling Tabular Data with Diffusion Models", ICML 2023.
//!
//! This implementation covers the noise schedule, forward diffusion process,
//! a randomly-initialised denoising MLP (not trained — serves as a structural
//! scaffold for training loops), reverse (ancestral) sampling, and loss
//! computation (MSE between actual and predicted noise).

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`TabDdpm`].
#[derive(Debug, Clone)]
pub struct TabDdpmConfig {
    /// Number of continuous features in each tabular row.
    pub n_features: usize,
    /// Total diffusion timesteps `T`.
    pub n_timesteps: usize,
    /// Starting value of the linear noise schedule `β₁`.
    pub beta_start: f32,
    /// Ending value of the linear noise schedule `β_T`.
    pub beta_end: f32,
    /// Width of each hidden layer in the denoising MLP.
    pub hidden_dim: usize,
    /// Number of hidden layers in the MLP (not counting the final output layer).
    pub n_layers: usize,
    /// Dimension of the sinusoidal time embedding (must be even).
    pub time_emb_dim: usize,
    /// RNG seed used to initialise the MLP weights.
    pub seed: u64,
}

// ─── DenoisingMlp ─────────────────────────────────────────────────────────────

/// A simple MLP with SiLU activations used as the denoising network.
///
/// Architecture: `input_dim → hidden_dim × n_layers → output_dim`.
/// The input concatenates the noisy sample with a sinusoidal time embedding,
/// so `input_dim = n_features + time_emb_dim`. `output_dim = n_features`.
pub struct DenoisingMlp {
    /// Flattened weight matrices per layer, row-major `[out_dim × in_dim]`.
    weights: Vec<Vec<f32>>,
    /// Bias vectors per layer, length `out_dim`.
    biases: Vec<Vec<f32>>,
    /// `(in_dim, out_dim)` per layer.
    layer_dims: Vec<(usize, usize)>,
}

impl DenoisingMlp {
    /// Construct an MLP with Xavier-uniform initialisation.
    ///
    /// # Panics (internal)
    /// None — all dimensions are validated by the caller ([`TabDdpm::new`]).
    pub(crate) fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> Self {
        // Build layer specification:
        //   input_dim → hidden_dim (×n_layers hidden) → output_dim
        let mut dims: Vec<(usize, usize)> = Vec::with_capacity(n_layers + 1);
        if n_layers == 0 {
            // Degenerate case: single linear layer.
            dims.push((input_dim, output_dim));
        } else {
            dims.push((input_dim, hidden_dim));
            for _ in 1..n_layers {
                dims.push((hidden_dim, hidden_dim));
            }
            dims.push((hidden_dim, output_dim));
        }

        let mut weights = Vec::with_capacity(dims.len());
        let mut biases = Vec::with_capacity(dims.len());

        for &(fan_in, fan_out) in &dims {
            // Xavier uniform: range [-limit, limit], limit = sqrt(6/(fan_in+fan_out))
            let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            let size = fan_in * fan_out;
            let mut w = Vec::with_capacity(size);
            for _ in 0..size {
                // Scale next_f32 ∈ [0,1) to [-limit, limit].
                let u = (rng.next_u32() as f32) / (u32::MAX as f32); // ∈ [0,1)
                w.push(2.0 * limit * u - limit);
            }
            weights.push(w);

            let b = vec![0.0_f32; fan_out];
            biases.push(b);
        }

        Self {
            weights,
            biases,
            layer_dims: dims,
        }
    }

    /// Forward pass: `x [input_dim] → output [output_dim]`.
    ///
    /// Hidden layers use SiLU activation; the final layer is linear.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n_layers = self.layer_dims.len();
        let mut current: Vec<f32> = x.to_vec();

        for (layer_idx, &(in_dim, out_dim)) in self.layer_dims.iter().enumerate() {
            let w = &self.weights[layer_idx];
            let b = &self.biases[layer_idx];
            let mut next = Vec::with_capacity(out_dim);

            // Matrix-vector multiply: y = W * current + b
            // w is stored row-major: row j has entries w[j*in_dim .. (j+1)*in_dim]
            for j in 0..out_dim {
                let mut acc = b[j];
                for k in 0..in_dim {
                    // Index is safe because w has size in_dim*out_dim.
                    acc += w[j * in_dim + k] * current.get(k).copied().unwrap_or(0.0);
                }
                next.push(acc);
            }

            // Apply SiLU to all layers except the last (output) layer.
            if layer_idx < n_layers - 1 {
                for v in next.iter_mut() {
                    *v = silu(*v);
                }
            }
            current = next;
        }
        current
    }
}

// ─── TabDdpm ─────────────────────────────────────────────────────────────────

/// Gaussian DDPM model for tabular data.
///
/// Holds the linear noise schedule (`betas`, `alphas`, `alphas_cumprod`) and a
/// randomly-initialised denoising MLP. Supports forward diffusion, reverse
/// (ancestral) sampling, and MSE loss computation.
pub struct TabDdpm {
    /// The resolved configuration.
    pub config: TabDdpmConfig,
    /// β_t for t = 0..T.
    pub betas: Vec<f32>,
    /// α_t = 1 − β_t.
    pub alphas: Vec<f32>,
    /// ᾱ_t = ∏_{s=0}^{t} α_s.
    pub alphas_cumprod: Vec<f32>,
    /// The denoising network ε_θ.
    denoiser: DenoisingMlp,
}

impl TabDdpm {
    /// Construct a new TabDDPM model, computing the noise schedule and
    /// initialising the denoising MLP.
    ///
    /// # Errors
    /// - [`TabularError::InvalidFeatureCount`] when `n_features == 0`.
    /// - [`TabularError::InvalidStepCount`] when `n_timesteps == 0`.
    /// - [`TabularError::InvalidEmbedDim`] when `time_emb_dim` is zero or odd.
    pub fn new(config: TabDdpmConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if config.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount {
                n: config.n_features,
            });
        }
        if config.n_timesteps == 0 {
            return Err(TabularError::InvalidStepCount {
                steps: config.n_timesteps,
            });
        }
        if config.time_emb_dim == 0 || !config.time_emb_dim.is_multiple_of(2) {
            return Err(TabularError::InvalidEmbedDim {
                dim: config.time_emb_dim,
            });
        }

        let t = config.n_timesteps;

        // Linear noise schedule.
        let mut betas = Vec::with_capacity(t);
        let mut alphas = Vec::with_capacity(t);
        let mut alphas_cumprod = Vec::with_capacity(t);

        for step in 0..t {
            let beta = if t == 1 {
                config.beta_start
            } else {
                config.beta_start
                    + (config.beta_end - config.beta_start) * step as f32 / (t - 1) as f32
            };
            let alpha = 1.0 - beta;
            betas.push(beta);
            alphas.push(alpha);

            let abar = if step == 0 {
                alpha
            } else {
                alphas_cumprod[step - 1] * alpha
            };
            alphas_cumprod.push(abar);
        }

        // Denoising MLP: input_dim = n_features + time_emb_dim.
        let input_dim = config.n_features + config.time_emb_dim;
        let denoiser = DenoisingMlp::new(
            input_dim,
            config.hidden_dim,
            config.n_features,
            config.n_layers,
            rng,
        );

        Ok(Self {
            config,
            betas,
            alphas,
            alphas_cumprod,
            denoiser,
        })
    }

    /// Sinusoidal time embedding of dimension `time_emb_dim` for timestep `t`.
    ///
    /// `emb[2i] = sin(t / 10000^(2i / D))`, `emb[2i+1] = cos(t / 10000^(2i / D))`.
    #[must_use]
    pub fn sinusoidal_embedding(&self, t: usize) -> Vec<f32> {
        let d = self.config.time_emb_dim;
        let mut emb = Vec::with_capacity(d);
        let half = d / 2;
        for i in 0..half {
            let freq = 1.0_f32 / (10000.0_f32.powf(2.0 * i as f32 / d as f32));
            let angle = t as f32 * freq;
            emb.push(angle.sin());
            emb.push(angle.cos());
        }
        emb
    }

    /// Forward diffusion: sample `x_t` given `x₀` at timestep `t`.
    ///
    /// `x_t = √(ᾱ_t) * x₀ + √(1 − ᾱ_t) * ε` where `ε ~ N(0, I)`.
    ///
    /// Returns `(x_t, eps)` so callers can compute the loss without re-sampling.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] when `x0.len() != n_features`.
    /// - [`TabularError::InvalidParameter`] when `t >= n_timesteps`.
    pub fn forward_sample(
        &self,
        x0: &[f32],
        t: usize,
        rng: &mut LcgRng,
    ) -> TabularResult<(Vec<f32>, Vec<f32>)> {
        if x0.len() != self.config.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.n_features,
                got: x0.len(),
            });
        }
        if t >= self.config.n_timesteps {
            return Err(TabularError::InvalidParameter {
                name: "t".into(),
                msg: format!("timestep {t} out of range [0, {})", self.config.n_timesteps),
            });
        }

        let alpha_bar = self.alphas_cumprod[t];
        let sqrt_alpha_bar = alpha_bar.sqrt();
        let sqrt_one_minus = (1.0 - alpha_bar).sqrt();

        let n = self.config.n_features;
        let mut eps = Vec::with_capacity(n);
        let mut idx = 0;
        while idx + 1 < n {
            let (a, b) = rng.next_normal_pair();
            eps.push(a);
            eps.push(b);
            idx += 2;
        }
        if idx < n {
            let (a, _) = rng.next_normal_pair();
            eps.push(a);
        }

        let x_t: Vec<f32> = x0
            .iter()
            .zip(eps.iter())
            .map(|(&x0_i, &eps_i)| sqrt_alpha_bar * x0_i + sqrt_one_minus * eps_i)
            .collect();

        Ok((x_t, eps))
    }

    /// Predict noise `ε̂` from noisy sample `x_t` at timestep `t`.
    ///
    /// Concatenates `x_t` with the sinusoidal time embedding and runs the MLP.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] when `x_t.len() != n_features`.
    /// - [`TabularError::InvalidParameter`] when `t >= n_timesteps`.
    pub fn denoise(&self, x_t: &[f32], t: usize) -> TabularResult<Vec<f32>> {
        if x_t.len() != self.config.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.config.n_features,
                got: x_t.len(),
            });
        }
        if t >= self.config.n_timesteps {
            return Err(TabularError::InvalidParameter {
                name: "t".into(),
                msg: format!("timestep {t} out of range [0, {})", self.config.n_timesteps),
            });
        }

        let time_emb = self.sinusoidal_embedding(t);
        let mut inp: Vec<f32> = Vec::with_capacity(x_t.len() + time_emb.len());
        inp.extend_from_slice(x_t);
        inp.extend_from_slice(&time_emb);

        let eps_hat = self.denoiser.forward(&inp);
        Ok(eps_hat)
    }

    /// DDPM ancestral sampling: one reverse step from `x_t` at timestep `t`.
    ///
    /// Computes the posterior mean and, for `t > 0`, adds Gaussian noise
    /// scaled by the posterior variance.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] / [`TabularError::InvalidParameter`]
    ///   propagated from [`denoise`](Self::denoise).
    pub fn reverse_step(&self, x_t: &[f32], t: usize, rng: &mut LcgRng) -> TabularResult<Vec<f32>> {
        let eps_hat = self.denoise(x_t, t)?;
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_prev = if t > 0 {
            self.alphas_cumprod[t - 1]
        } else {
            1.0_f32
        };
        let beta_t = self.betas[t];
        let alpha_t = self.alphas[t];

        let sqrt_alpha_bar_t = alpha_bar_t.sqrt().max(1e-8);
        let denom = 1.0 - alpha_bar_t;

        let n = self.config.n_features;

        // x₀ prediction from current noisy sample and predicted noise.
        let x0_pred: Vec<f32> = x_t
            .iter()
            .zip(eps_hat.iter())
            .map(|(&xt_i, &eps_i)| (xt_i - (1.0 - alpha_bar_t).sqrt() * eps_i) / sqrt_alpha_bar_t)
            .collect();

        // Posterior mean:
        //   μ = √ᾱ_{t-1} * β_t / (1 - ᾱ_t) * x₀_pred
        //     + √α_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * x_t
        let coeff_x0 = alpha_bar_prev.sqrt() * beta_t / denom.max(1e-8);
        let coeff_xt = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / denom.max(1e-8);

        let mean: Vec<f32> = x0_pred
            .iter()
            .zip(x_t.iter())
            .map(|(&x0_i, &xt_i)| coeff_x0 * x0_i + coeff_xt * xt_i)
            .collect();

        if t == 0 {
            return Ok(mean);
        }

        // Posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        let variance = (1.0 - alpha_bar_prev) / denom.max(1e-8) * beta_t;
        let std_dev = variance.max(0.0).sqrt();

        // Sample noise and combine.
        let mut noise = Vec::with_capacity(n);
        let mut idx = 0;
        while idx + 1 < n {
            let (a, b) = rng.next_normal_pair();
            noise.push(a);
            noise.push(b);
            idx += 2;
        }
        if idx < n {
            let (a, _) = rng.next_normal_pair();
            noise.push(a);
        }

        let x_prev: Vec<f32> = mean
            .iter()
            .zip(noise.iter())
            .map(|(&m, &z)| m + std_dev * z)
            .collect();

        Ok(x_prev)
    }

    /// Full ancestral sampling: start from `x_T ~ N(0, I)` and run all `T`
    /// reverse steps.
    ///
    /// Returns a flat row-major `[n_samples × n_features]` buffer.
    ///
    /// # Errors
    /// Propagates errors from [`reverse_step`](Self::reverse_step).
    pub fn sample(&self, n_samples: usize, rng: &mut LcgRng) -> TabularResult<Vec<f32>> {
        let n_feat = self.config.n_features;
        let t_max = self.config.n_timesteps;
        let mut out = Vec::with_capacity(n_samples * n_feat);

        for _ in 0..n_samples {
            // Start from pure Gaussian noise x_T.
            let mut x = Vec::with_capacity(n_feat);
            let mut idx = 0;
            while idx + 1 < n_feat {
                let (a, b) = rng.next_normal_pair();
                x.push(a);
                x.push(b);
                idx += 2;
            }
            if idx < n_feat {
                let (a, _) = rng.next_normal_pair();
                x.push(a);
            }

            // Reverse from t = T-1 down to t = 0.
            for t in (0..t_max).rev() {
                x = self.reverse_step(&x, t, rng)?;
            }
            out.extend_from_slice(&x);
        }
        Ok(out)
    }

    /// MSE loss between actual noise `ε` and predicted noise `ε̂` at timestep `t`.
    ///
    /// Internally samples `ε ~ N(0, I)`, computes `x_t`, then predicts `ε̂`.
    /// Returns `(1 / n_features) * Σ (ε_i − ε̂_i)²`.
    ///
    /// # Errors
    /// Propagated from [`forward_sample`](Self::forward_sample) and
    /// [`denoise`](Self::denoise).
    pub fn compute_loss(&self, x0: &[f32], t: usize, rng: &mut LcgRng) -> TabularResult<f32> {
        let (x_t, eps) = self.forward_sample(x0, t, rng)?;
        let eps_hat = self.denoise(&x_t, t)?;

        let n = self.config.n_features as f32;
        let mse = eps
            .iter()
            .zip(eps_hat.iter())
            .map(|(&e, &eh)| (e - eh) * (e - eh))
            .sum::<f32>()
            / n;
        Ok(mse)
    }

    /// Number of diffusion timesteps `T`.
    #[must_use]
    pub fn n_timesteps(&self) -> usize {
        self.config.n_timesteps
    }

    /// Number of tabular features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.config.n_features
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit): `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TabDdpmConfig {
        TabDdpmConfig {
            n_features: 8,
            n_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
            hidden_dim: 32,
            n_layers: 2,
            time_emb_dim: 16,
            seed: 42,
        }
    }

    fn make_model() -> TabDdpm {
        let cfg = default_config();
        let mut rng = LcgRng::new(cfg.seed);
        TabDdpm::new(cfg, &mut rng).expect("new should succeed")
    }

    // ── 1. Noise schedule is strictly decreasing ─────────────────────────────
    #[test]
    fn noise_schedule_monotone() {
        let model = make_model();
        let abar = &model.alphas_cumprod;
        for i in 1..abar.len() {
            assert!(
                abar[i] < abar[i - 1],
                "alphas_cumprod not monotone at i={i}: {} >= {}",
                abar[i],
                abar[i - 1]
            );
        }
    }

    // ── 2. alphas_cumprod[0] ≈ 1 - beta_start ───────────────────────────────
    #[test]
    fn alphas_cumprod_start_near_one() {
        let model = make_model();
        let expected = 1.0 - model.config.beta_start;
        let got = model.alphas_cumprod[0];
        assert!(
            (got - expected).abs() < 1e-5,
            "alphas_cumprod[0] expected {expected}, got {got}"
        );
    }

    // ── 3. alphas_cumprod[T-1] < 0.01 for T=1000, beta_end=0.02 ────────────
    #[test]
    fn alphas_cumprod_end_near_zero() {
        let cfg = TabDdpmConfig {
            n_features: 4,
            n_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
            hidden_dim: 16,
            n_layers: 1,
            time_emb_dim: 8,
            seed: 0,
        };
        let mut rng = LcgRng::new(0);
        let model = TabDdpm::new(cfg, &mut rng).expect("new should succeed");
        let abar_last = model.alphas_cumprod[999];
        assert!(
            abar_last < 0.01,
            "alphas_cumprod[T-1] should be near zero for T=1000, got {abar_last}"
        );
    }

    // ── 4. forward_sample at t=0: x_t ≈ x0 (noise weight ≈ 0) ─────────────
    #[test]
    fn forward_sample_at_t0_near_x0() {
        let model = make_model();
        let mut rng = LcgRng::new(7);
        let x0: Vec<f32> = (0..model.config.n_features)
            .map(|i| i as f32 * 0.1)
            .collect();
        let (x_t, _eps) = model
            .forward_sample(&x0, 0, &mut rng)
            .expect("forward_sample should succeed");
        // At t=0, sqrt(alphas_cumprod[0]) ≈ 1 and sqrt(1-alphas_cumprod[0]) ≈ 0.01
        let abar0 = model.alphas_cumprod[0];
        let noise_weight = (1.0 - abar0).sqrt();
        for (&xt_i, &x0_i) in x_t.iter().zip(x0.iter()) {
            let signal_part = abar0.sqrt() * x0_i;
            let diff = (xt_i - signal_part).abs();
            // Noise contribution is very small (< 3σ, where σ ≈ noise_weight).
            assert!(
                diff < 10.0 * noise_weight + 1e-4,
                "at t=0, x_t should be close to sqrt(abar)*x0 + small_noise"
            );
        }
    }

    // ── 5. forward_sample at t=T-1: output magnitude ~ 1 ────────────────────
    #[test]
    fn forward_sample_at_tmax_near_noise() {
        let cfg = TabDdpmConfig {
            n_features: 16,
            n_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
            hidden_dim: 16,
            n_layers: 1,
            time_emb_dim: 8,
            seed: 1,
        };
        let mut rng = LcgRng::new(1);
        let model = TabDdpm::new(cfg, &mut rng).expect("new should succeed");
        let mut rng2 = LcgRng::new(2);
        let x0 = vec![0.0_f32; model.config.n_features];
        let t_last = model.config.n_timesteps - 1;
        let (x_t, _) = model
            .forward_sample(&x0, t_last, &mut rng2)
            .expect("forward_sample should succeed");
        // x0=0 so x_t = sqrt(1-abar_last) * eps; abar_last≈0 so x_t ≈ eps ~ N(0,1).
        // Variance should be approximately 1: empirical std across dims is O(1).
        let mean: f32 = x_t.iter().sum::<f32>() / x_t.len() as f32;
        let var: f32 = x_t.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / x_t.len() as f32;
        let std = var.sqrt();
        // With only 16 samples the estimate is noisy; just check it's non-negligible.
        assert!(
            std > 0.05,
            "at t=T-1, x_t should have non-trivial variance, std={std}"
        );
    }

    // ── 6. forward_sample correct shape ─────────────────────────────────────
    #[test]
    fn forward_sample_correct_shape() {
        let model = make_model();
        let mut rng = LcgRng::new(42);
        let x0 = vec![0.5_f32; model.config.n_features];
        let (x_t, eps) = model
            .forward_sample(&x0, 5, &mut rng)
            .expect("forward_sample should succeed");
        assert_eq!(x_t.len(), model.config.n_features);
        assert_eq!(eps.len(), model.config.n_features);
    }

    // ── 7. sinusoidal_embedding has correct shape ────────────────────────────
    #[test]
    fn sinusoidal_embedding_shape() {
        let model = make_model();
        let emb = model.sinusoidal_embedding(42);
        assert_eq!(
            emb.len(),
            model.config.time_emb_dim,
            "embedding length mismatch"
        );
    }

    // ── 8. sinusoidal_embedding values in [-1, 1] ────────────────────────────
    #[test]
    fn sinusoidal_embedding_range() {
        let model = make_model();
        for t in [0, 1, 50, 99] {
            let emb = model.sinusoidal_embedding(t);
            for (i, &v) in emb.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "emb[{i}]={v} out of [-1,1] at t={t}"
                );
            }
        }
    }

    // ── 9. denoise output shape and all finite ───────────────────────────────
    #[test]
    fn denoise_output_shape() {
        let model = make_model();
        let x_t = vec![0.1_f32; model.config.n_features];
        let eps_hat = model.denoise(&x_t, 5).expect("denoise should succeed");
        assert_eq!(eps_hat.len(), model.config.n_features, "denoise shape");
        assert!(
            eps_hat.iter().all(|v| v.is_finite()),
            "denoiser output must be finite"
        );
    }

    // ── 10. reverse_step shape and finite ────────────────────────────────────
    #[test]
    fn reverse_step_shape() {
        let model = make_model();
        let mut rng = LcgRng::new(99);
        let x_t = vec![0.1_f32; model.config.n_features];
        let x_prev = model
            .reverse_step(&x_t, 50, &mut rng)
            .expect("reverse_step should succeed");
        assert_eq!(x_prev.len(), model.config.n_features, "reverse_step shape");
        assert!(
            x_prev.iter().all(|v| v.is_finite()),
            "reverse_step output must be finite"
        );
    }

    // ── 11. compute_loss ≥ 0.0 and finite ───────────────────────────────────
    #[test]
    fn compute_loss_nonneg() {
        let model = make_model();
        let mut rng = LcgRng::new(123);
        let x0 = vec![1.0_f32; model.config.n_features];
        let loss = model
            .compute_loss(&x0, 10, &mut rng)
            .expect("compute_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    // ── 12. sample shape and finite ──────────────────────────────────────────
    #[test]
    fn sample_shape() {
        let model = make_model();
        let mut rng = LcgRng::new(55);
        let n_samples = 5;
        let out = model
            .sample(n_samples, &mut rng)
            .expect("sample should succeed");
        assert_eq!(
            out.len(),
            n_samples * model.config.n_features,
            "sample shape mismatch"
        );
        assert!(out.iter().all(|v| v.is_finite()), "samples must be finite");
    }

    // ── 13. n_timesteps = 0 → error ──────────────────────────────────────────
    #[test]
    fn n_timesteps_zero_error() {
        let mut cfg = default_config();
        cfg.n_timesteps = 0;
        let mut rng = LcgRng::new(0);
        assert!(
            matches!(
                TabDdpm::new(cfg, &mut rng),
                Err(TabularError::InvalidStepCount { .. })
            ),
            "expected InvalidStepCount"
        );
    }

    // ── 14. n_features = 0 → error ───────────────────────────────────────────
    #[test]
    fn n_features_zero_error() {
        let mut cfg = default_config();
        cfg.n_features = 0;
        let mut rng = LcgRng::new(0);
        assert!(
            matches!(
                TabDdpm::new(cfg, &mut rng),
                Err(TabularError::InvalidFeatureCount { .. })
            ),
            "expected InvalidFeatureCount"
        );
    }
}
