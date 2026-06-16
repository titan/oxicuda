//! MLP-based β-VAE encoder for continuous latent representations.
//!
//! Implements a symmetric MLP encoder-decoder following the β-VAE formulation
//! of Rombach et al. (2022) "High-Resolution Image Synthesis with Latent
//! Diffusion Models".  The encoder maps an input `x ∈ ℝ^{d_input}` to a
//! Gaussian distribution `q(z|x) = N(μ, diag(exp(log_var)))`, samples
//! `z ∈ ℝ^{d_latent}` via the reparameterisation trick, and the decoder maps
//! `z` back to `x̂ ∈ ℝ^{d_input}`.
//!
//! The ELBO loss is:
//! ```text
//! L = ‖x - x̂‖² + β · KL(q(z|x) ‖ N(0, I))
//! ```
//!
//! where the KL term is computed analytically:
//! ```text
//! KL = -½ · Σ_i (1 + log_var_i - μ_i² - exp(log_var_i)) / d_latent
//! ```

use crate::error::{GenError, GenResult};

/// Type alias for the crate-level LCG random number generator.
pub type GenRng = crate::handle::LcgRng;

// ─── Box-Muller helper ───────────────────────────────────────────────────────

/// Sample a single standard-normal deviate using the Box-Muller transform.
///
/// Uses two uniform samples from `rng.next_f32()` and operates entirely in
/// `f64` for numerical stability.
fn sample_normal_f64(rng: &mut GenRng) -> f64 {
    let u1 = (rng.next_f32() as f64 + 1e-10_f64).min(1.0 - 1e-10_f64);
    let u2 = rng.next_f32() as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

// ─── MLP helpers ─────────────────────────────────────────────────────────────

/// Compute `y = W · x + b` where `W` is stored row-major as a flat slice.
///
/// `w` has length `out_dim * in_dim`; `x` has length `in_dim`.
fn matmul_f64(w: &[f64], b: &[f64], x: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = x.len();
    let mut y = vec![0.0_f64; out_dim];
    for i in 0..out_dim {
        let mut acc = b[i];
        for j in 0..in_dim {
            acc += w[i * in_dim + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Apply element-wise ReLU in-place.
fn relu_f64(v: &mut [f64]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

// ─── VaeEncoderConfig ────────────────────────────────────────────────────────

/// Configuration for the MLP β-VAE encoder/decoder.
#[derive(Debug, Clone)]
pub struct VaeEncoderConfig {
    /// Dimensionality of the input space `x`.
    pub d_input: usize,
    /// Dimensionality of the latent space `z`.
    pub d_latent: usize,
    /// Number of hidden layers in both the encoder and decoder.
    ///
    /// `0` means a single linear layer (no hidden layers).
    pub n_layers: usize,
    /// Weight for the KL divergence term in the ELBO loss (`β` in β-VAE).
    pub kl_weight: f64,
}

// ─── VaeEncoder ──────────────────────────────────────────────────────────────

/// MLP-based β-VAE encoder/decoder pair.
///
/// ## Architecture
///
/// ### Encoder
/// ```text
/// d_input  →  [d_hidden]*n_layers  →  2·d_latent   (outputs μ ‖ log_var)
/// ```
///
/// ### Decoder
/// ```text
/// d_latent  →  [d_hidden]*n_layers  →  d_input
/// ```
///
/// where `d_hidden = (d_input + d_latent) / 2 + 1`.
///
/// All hidden activations use ReLU; the final layer in both encoder and
/// decoder has no activation.
///
/// Weights are He-initialised: `N(0, sqrt(2 / fan_in))`.
#[derive(Debug, Clone)]
pub struct VaeEncoder {
    /// Encoder weight matrices, one per layer (row-major flat).
    enc_w: Vec<Vec<f64>>,
    /// Encoder bias vectors, one per layer.
    enc_b: Vec<Vec<f64>>,
    /// Decoder weight matrices, one per layer (row-major flat).
    dec_w: Vec<Vec<f64>>,
    /// Decoder bias vectors, one per layer.
    dec_b: Vec<Vec<f64>>,
    /// Configuration this encoder was built from.
    config: VaeEncoderConfig,
    /// Layer sizes for the encoder (including input and output dimensions).
    layer_sizes_enc: Vec<usize>,
    /// Layer sizes for the decoder (including input and output dimensions).
    layer_sizes_dec: Vec<usize>,
}

impl VaeEncoder {
    /// Construct a new `VaeEncoder` with He-initialised weights.
    ///
    /// # Errors
    ///
    /// - [`GenError::EmptyInput`] if `d_input == 0`.
    /// - [`GenError::EmptyInput`] if `d_latent == 0`.
    pub fn new(config: VaeEncoderConfig, rng: &mut GenRng) -> GenResult<Self> {
        if config.d_input == 0 {
            return Err(GenError::EmptyInput("d_input must be > 0"));
        }
        if config.d_latent == 0 {
            return Err(GenError::EmptyInput("d_latent must be > 0"));
        }

        let d_input = config.d_input;
        let d_latent = config.d_latent;
        let d_hidden = (d_input + d_latent) / 2 + 1;

        // Build layer size vectors
        let layer_sizes_enc: Vec<usize> = {
            let mut v = Vec::with_capacity(config.n_layers + 2);
            v.push(d_input);
            for _ in 0..config.n_layers {
                v.push(d_hidden);
            }
            v.push(2 * d_latent);
            v
        };

        let layer_sizes_dec: Vec<usize> = {
            let mut v = Vec::with_capacity(config.n_layers + 2);
            v.push(d_latent);
            for _ in 0..config.n_layers {
                v.push(d_hidden);
            }
            v.push(d_input);
            v
        };

        // Initialise encoder weights
        let (enc_w, enc_b) = init_mlp_weights(&layer_sizes_enc, rng);
        // Initialise decoder weights
        let (dec_w, dec_b) = init_mlp_weights(&layer_sizes_dec, rng);

        Ok(Self {
            enc_w,
            enc_b,
            dec_w,
            dec_b,
            config,
            layer_sizes_enc,
            layer_sizes_dec,
        })
    }

    /// Encode an input `x` into `(μ, log_var)` pairs in latent space.
    ///
    /// Runs the encoder MLP with ReLU activations on all hidden layers.
    /// The final layer is linear, outputting the concatenated `[μ ‖ log_var]`.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `x.len() != d_input`.
    pub fn encode(&self, x: &[f64]) -> GenResult<(Vec<f64>, Vec<f64>)> {
        if x.len() != self.config.d_input {
            return Err(GenError::DimensionMismatch {
                expected: self.config.d_input,
                got: x.len(),
            });
        }
        let n_layers = self.layer_sizes_enc.len() - 1;
        let mut h: Vec<f64> = x.to_vec();
        for l in 0..n_layers {
            let out_dim = self.layer_sizes_enc[l + 1];
            let mut out = matmul_f64(&self.enc_w[l], &self.enc_b[l], &h, out_dim);
            // ReLU on all layers except the final one
            if l + 1 < n_layers {
                relu_f64(&mut out);
            }
            h = out;
        }
        // Split output into (mu, log_var)
        let d_latent = self.config.d_latent;
        let mu = h[..d_latent].to_vec();
        let log_var = h[d_latent..].to_vec();
        Ok((mu, log_var))
    }

    /// Sample `z` from `q(z|x)` using the reparameterisation trick.
    ///
    /// `z = μ + exp(0.5 · log_var) ⊙ ε`,  where  `ε ~ N(0, I)`.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `mu.len() != log_var.len()`.
    pub fn reparameterize(
        &self,
        mu: &[f64],
        log_var: &[f64],
        rng: &mut GenRng,
    ) -> GenResult<Vec<f64>> {
        if mu.len() != log_var.len() {
            return Err(GenError::DimensionMismatch {
                expected: mu.len(),
                got: log_var.len(),
            });
        }
        let z = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let eps = sample_normal_f64(rng);
                m + (0.5 * lv).exp() * eps
            })
            .collect();
        Ok(z)
    }

    /// Decode a latent vector `z` back to the input space.
    ///
    /// Runs the decoder MLP with ReLU activations on all hidden layers.
    /// The final layer is linear.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `z.len() != d_latent`.
    pub fn decode(&self, z: &[f64]) -> GenResult<Vec<f64>> {
        if z.len() != self.config.d_latent {
            return Err(GenError::DimensionMismatch {
                expected: self.config.d_latent,
                got: z.len(),
            });
        }
        let n_layers = self.layer_sizes_dec.len() - 1;
        let mut h: Vec<f64> = z.to_vec();
        for l in 0..n_layers {
            let out_dim = self.layer_sizes_dec[l + 1];
            let mut out = matmul_f64(&self.dec_w[l], &self.dec_b[l], &h, out_dim);
            if l + 1 < n_layers {
                relu_f64(&mut out);
            }
            h = out;
        }
        Ok(h)
    }

    /// Compute the ELBO loss for a single input `x`.
    ///
    /// ```text
    /// ELBO = ‖x - decode(z)‖² + kl_weight · KL(q(z|x) ‖ N(0,I))
    /// ```
    ///
    /// where `KL = -½ Σ_i (1 + log_var_i - μ_i² - exp(log_var_i)) / d_latent`.
    ///
    /// # Errors
    ///
    /// - Propagates errors from `encode`, `reparameterize`, `decode`.
    pub fn elbo(&self, x: &[f64], rng: &mut GenRng) -> GenResult<f64> {
        let (mu, log_var) = self.encode(x)?;
        let z = self.reparameterize(&mu, &log_var, rng)?;
        let x_hat = self.decode(&z)?;

        // Reconstruction loss: MSE
        let recon_loss = x
            .iter()
            .zip(x_hat.iter())
            .map(|(&xi, &xi_hat)| {
                let diff = xi - xi_hat;
                diff * diff
            })
            .sum::<f64>()
            / (x.len() as f64);

        // KL divergence: -½ Σ (1 + log_var - mu² - exp(log_var)) / d_latent
        let kl = self.kl_divergence(&mu, &log_var);

        Ok(recon_loss + self.config.kl_weight * kl)
    }

    /// Compute the KL divergence `KL(q(z|x) ‖ N(0,I))` analytically.
    ///
    /// This is always non-negative by Gibbs' inequality.
    fn kl_divergence(&self, mu: &[f64], log_var: &[f64]) -> f64 {
        let d = mu.len() as f64;
        if d == 0.0 {
            return 0.0;
        }
        let sum: f64 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| 1.0 + lv - m * m - lv.exp())
            .sum();
        // KL = -½ * sum / d_latent  (note: the formula gives a positive KL)
        -0.5 * sum / d
    }

    /// Return the dimensionality of the latent space.
    #[must_use]
    #[inline]
    pub fn d_latent(&self) -> usize {
        self.config.d_latent
    }

    /// Return the dimensionality of the input space.
    #[must_use]
    #[inline]
    pub fn d_input(&self) -> usize {
        self.config.d_input
    }
}

// ─── Weight initialisation ───────────────────────────────────────────────────

/// He-initialise weights for a multi-layer perceptron.
///
/// For each layer `l` with `fan_in` input neurons, weights are drawn from
/// `N(0, sqrt(2 / fan_in))` and biases are initialised to zero.
fn init_mlp_weights(layer_sizes: &[usize], rng: &mut GenRng) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_layers = layer_sizes.len() - 1;
    let mut ws = Vec::with_capacity(n_layers);
    let mut bs = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let fan_in = layer_sizes[l];
        let fan_out = layer_sizes[l + 1];
        let std_dev = (2.0_f64 / (fan_in as f64)).sqrt();
        let n_params = fan_in * fan_out;
        let w: Vec<f64> = (0..n_params)
            .map(|_| sample_normal_f64(rng) * std_dev)
            .collect();
        let b = vec![0.0_f64; fan_out];
        ws.push(w);
        bs.push(b);
    }
    (ws, bs)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(d_input: usize, d_latent: usize, n_layers: usize) -> VaeEncoder {
        let config = VaeEncoderConfig {
            d_input,
            d_latent,
            n_layers,
            kl_weight: 1.0,
        };
        let mut rng = GenRng::new(42);
        VaeEncoder::new(config, &mut rng).expect("new should succeed")
    }

    fn make_rng() -> GenRng {
        GenRng::new(7)
    }

    #[test]
    fn encode_shape() {
        let enc = make_encoder(16, 4, 2);
        let x = vec![0.5_f64; 16];
        let (mu, log_var) = enc.encode(&x).expect("encode should succeed");
        assert_eq!(mu.len(), 4, "mu should have d_latent=4 elements");
        assert_eq!(log_var.len(), 4, "log_var should have d_latent=4 elements");
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(log_var.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn reparameterize_shape() {
        let enc = make_encoder(8, 3, 1);
        let mu = vec![0.0_f64; 3];
        let log_var = vec![0.0_f64; 3];
        let mut rng = make_rng();
        let z = enc
            .reparameterize(&mu, &log_var, &mut rng)
            .expect("reparameterize should succeed");
        assert_eq!(z.len(), 3);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_shape() {
        let enc = make_encoder(16, 4, 2);
        let z = vec![0.1_f64; 4];
        let x_hat = enc.decode(&z).expect("decode should succeed");
        assert_eq!(
            x_hat.len(),
            16,
            "decoded output should have d_input=16 elements"
        );
        assert!(x_hat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn elbo_finite() {
        let enc = make_encoder(8, 2, 1);
        let x = vec![0.3_f64; 8];
        let mut rng = make_rng();
        let loss = enc.elbo(&x, &mut rng).expect("elbo should succeed");
        assert!(loss.is_finite(), "ELBO must be finite: {loss}");
    }

    #[test]
    fn kl_nonneg() {
        // KL should always be non-negative
        let enc = make_encoder(8, 4, 1);
        let x = vec![0.5_f64; 8];
        let (mu, log_var) = enc.encode(&x).expect("encode should succeed");
        let kl = enc.kl_divergence(&mu, &log_var);
        assert!(kl >= 0.0, "KL divergence must be non-negative: {kl}");
    }

    #[test]
    fn kl_zero_for_prior_posterior() {
        // When mu=0 and log_var=0 (std=1), KL(N(0,1) || N(0,1)) = 0
        let enc = make_encoder(4, 2, 0);
        let mu = vec![0.0_f64; 2];
        let log_var = vec![0.0_f64; 2]; // exp(0) = 1 => std = 1
        let kl = enc.kl_divergence(&mu, &log_var);
        assert!(kl.abs() < 1e-10, "KL(N(0,1)||N(0,1)) should be 0: {kl}");
    }

    #[test]
    fn different_x_different_z() {
        let enc = make_encoder(8, 4, 1);
        let x1 = vec![1.0_f64; 8];
        let x2 = vec![-1.0_f64; 8];
        let (mu1, _) = enc.encode(&x1).expect("encode should succeed");
        let (mu2, _) = enc.encode(&x2).expect("encode should succeed");
        let diff: f64 = mu1
            .iter()
            .zip(&mu2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff > 1e-6,
            "Different inputs should yield different mu: diff={diff}"
        );
    }

    #[test]
    fn reparameterize_stochastic() {
        let enc = make_encoder(8, 4, 1);
        let mu = vec![0.0_f64; 4];
        let log_var = vec![1.0_f64; 4]; // std = sqrt(e) ≈ 1.65, samples should differ
        let mut rng1 = GenRng::new(1);
        let mut rng2 = GenRng::new(999);
        let z1 = enc
            .reparameterize(&mu, &log_var, &mut rng1)
            .expect("reparameterize should succeed");
        let z2 = enc
            .reparameterize(&mu, &log_var, &mut rng2)
            .expect("reparameterize should succeed");
        let diff: f64 = z1
            .iter()
            .zip(&z2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff > 1e-6,
            "Reparameterisation with different seeds should give different z: diff={diff}"
        );
    }

    #[test]
    fn d_latent_0_error() {
        let config = VaeEncoderConfig {
            d_input: 8,
            d_latent: 0,
            n_layers: 1,
            kl_weight: 1.0,
        };
        let mut rng = make_rng();
        let err = VaeEncoder::new(config, &mut rng);
        assert!(
            matches!(err, Err(GenError::EmptyInput(_))),
            "Expected EmptyInput for d_latent=0, got: {err:?}"
        );
    }

    #[test]
    fn n_layers_0_works() {
        // Single-layer encoder/decoder (no hidden layers)
        let enc = make_encoder(4, 2, 0);
        let x = vec![0.1_f64; 4];
        let (mu, log_var) = enc.encode(&x).expect("encode should succeed");
        assert_eq!(mu.len(), 2);
        assert_eq!(log_var.len(), 2);
        let z = vec![0.5_f64; 2];
        let x_hat = enc.decode(&z).expect("decode should succeed");
        assert_eq!(x_hat.len(), 4);
        assert!(x_hat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn d_input_accessor() {
        let enc = make_encoder(12, 3, 1);
        assert_eq!(enc.d_input(), 12);
        assert_eq!(enc.d_latent(), 3);
    }
}
