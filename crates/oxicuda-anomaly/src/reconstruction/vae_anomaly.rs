//! VAE-based anomaly detection.
//!
//! Anomaly score = `-ELBO = -(recon_loss + β · KL)`.
//!
//! * High ELBO → normal
//! * Low ELBO  → anomalous
//!
//! KL divergence: `−0.5 · Σ (1 + log_var − μ² − exp(log_var))`
//! Reparametrisation: `z = μ + ε · exp(0.5 · log_var)`, `ε ~ N(0,1)`.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32();
            u * 2.0 * limit - limit
        })
        .collect()
}

fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; fan_out];
    for o in 0..fan_out {
        let mut acc = b[o];
        for i in 0..fan_in {
            acc += w[o * fan_in + i] * x[i];
        }
        out[o] = acc;
    }
    out
}

fn relu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x.max(0.0)).collect()
}

fn sigmoid(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
}

// ─── VaeAnomaly ───────────────────────────────────────────────────────────────

/// VAE anomaly detector.
pub struct VaeAnomaly {
    /// Shared encoder body: intermediate dense layers.
    encoder_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Encoder body dimension spec (including input and final hidden).
    enc_body_dims: Vec<usize>,
    /// μ head: `(weight, bias)` mapping last encoder hidden → latent.
    mu_layer: (Vec<f32>, Vec<f32>),
    /// log-variance head: `(weight, bias)` mapping last encoder hidden → latent.
    logvar_layer: (Vec<f32>, Vec<f32>),
    /// Decoder layers: `(weight, bias)`.
    decoder_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Decoder dimension spec (from latent to output).
    dec_dims: Vec<usize>,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality.
    pub latent_dim: usize,
    /// β weighting for the KL term (β-VAE; 1.0 = standard VAE).
    pub beta: f32,
}

impl VaeAnomaly {
    /// Build a VAE.
    ///
    /// * `encoder_dims`: `[input, h1, ..., last_hidden]`
    /// * `latent_dim`: size of the latent space
    /// * `decoder_dims`: `[latent, h1, ..., output]` (typically mirroring encoder)
    pub fn new(
        encoder_dims: &[usize],
        latent_dim: usize,
        decoder_dims: &[usize],
        rng: &mut LcgRng,
    ) -> AnomalyResult<Self> {
        if encoder_dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "encoder_dims must have at least [input, hidden]".into(),
            });
        }
        if decoder_dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "decoder_dims must have at least [latent, output]".into(),
            });
        }
        if latent_dim == 0 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "latent_dim must be > 0".into(),
            });
        }
        let input_dim = encoder_dims[0];
        let last_hidden = *encoder_dims.last().unwrap_or(&1);

        // Encoder body layers
        let mut encoder_layers = Vec::with_capacity(encoder_dims.len() - 1);
        for i in 0..encoder_dims.len() - 1 {
            let fan_in = encoder_dims[i];
            let fan_out = encoder_dims[i + 1];
            encoder_layers.push((xavier_init(fan_in, fan_out, rng), vec![0.0_f32; fan_out]));
        }

        // μ and log_var heads
        let mu_layer = (
            xavier_init(last_hidden, latent_dim, rng),
            vec![0.0_f32; latent_dim],
        );
        let logvar_layer = (
            xavier_init(last_hidden, latent_dim, rng),
            vec![0.0_f32; latent_dim],
        );

        // Decoder layers
        let mut decoder_layers = Vec::with_capacity(decoder_dims.len() - 1);
        for i in 0..decoder_dims.len() - 1 {
            let fan_in = decoder_dims[i];
            let fan_out = decoder_dims[i + 1];
            decoder_layers.push((xavier_init(fan_in, fan_out, rng), vec![0.0_f32; fan_out]));
        }

        Ok(Self {
            encoder_layers,
            enc_body_dims: encoder_dims.to_vec(),
            mu_layer,
            logvar_layer,
            decoder_layers,
            dec_dims: decoder_dims.to_vec(),
            input_dim,
            latent_dim,
            beta: 1.0,
        })
    }

    /// Encode `x` → `(μ, log_var)`.
    pub fn encode(&self, x: &[f32]) -> AnomalyResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let n_enc = self.encoder_layers.len();
        let mut act: Vec<f32> = x.to_vec();
        for (idx, (w, b)) in self.encoder_layers.iter().enumerate() {
            let fan_in = self.enc_body_dims[idx];
            let fan_out = self.enc_body_dims[idx + 1];
            let out = dense(&act, w, b, fan_in, fan_out);
            act = if idx < n_enc - 1 { relu(&out) } else { out };
        }
        let last_hidden = *self.enc_body_dims.last().unwrap_or(&1);
        let mu = dense(
            &act,
            &self.mu_layer.0,
            &self.mu_layer.1,
            last_hidden,
            self.latent_dim,
        );
        let log_var = dense(
            &act,
            &self.logvar_layer.0,
            &self.logvar_layer.1,
            last_hidden,
            self.latent_dim,
        );
        Ok((mu, log_var))
    }

    /// Reparametrisation trick: `z = μ + ε · exp(0.5 · log_var)`, `ε ~ N(0,1)`.
    pub fn reparametrize(
        &self,
        mu: &[f32],
        log_var: &[f32],
        rng: &mut LcgRng,
    ) -> AnomalyResult<Vec<f32>> {
        if mu.len() != self.latent_dim || log_var.len() != self.latent_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.latent_dim,
                got: mu.len(),
            });
        }
        let z: Vec<f32> = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let eps = rng.next_normal();
                m + eps * (0.5 * lv).exp()
            })
            .collect();
        Ok(z)
    }

    /// Decode `z` → `[input_dim]` (sigmoid output).
    pub fn decode(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        if z.len() != self.latent_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.latent_dim,
                got: z.len(),
            });
        }
        let n_dec = self.decoder_layers.len();
        let mut act: Vec<f32> = z.to_vec();
        for (idx, (w, b)) in self.decoder_layers.iter().enumerate() {
            let fan_in = self.dec_dims[idx];
            let fan_out = self.dec_dims[idx + 1];
            let out = dense(&act, w, b, fan_in, fan_out);
            act = if idx < n_dec - 1 {
                relu(&out)
            } else {
                sigmoid(&out)
            };
        }
        Ok(act)
    }

    /// KL divergence: `−0.5 · Σ (1 + log_var − μ² − exp(log_var))`.
    pub fn kl_divergence(mu: &[f32], log_var: &[f32]) -> AnomalyResult<f32> {
        if mu.len() != log_var.len() {
            return Err(AnomalyError::DimensionMismatch {
                expected: mu.len(),
                got: log_var.len(),
            });
        }
        let kl: f32 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let clamped_lv = lv.clamp(-20.0, 20.0);
                -0.5 * (1.0 + clamped_lv - m * m - clamped_lv.exp())
            })
            .sum();
        Ok(kl)
    }

    /// Anomaly score per sample: `MSE(x, decode(μ)) + β · KL(μ, log_var)`.
    ///
    /// Uses the deterministic `μ` for scoring (no sampling noise).
    pub fn anomaly_score(&self, x: &[f32], _rng: &mut LcgRng) -> AnomalyResult<f32> {
        let (mu, log_var) = self.encode(x)?;
        let x_hat = self.decode(&mu)?;
        let mse = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / x.len() as f32;
        let kl = Self::kl_divergence(&mu, &log_var)?;
        Ok(mse + self.beta * kl)
    }

    /// Batch scoring; `x` is `[n * input_dim]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize, rng: &mut LcgRng) -> AnomalyResult<Vec<f32>> {
        if x.len() != n * self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.input_dim,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.input_dim..(i + 1) * self.input_dim];
            scores.push(self.anomaly_score(sample, rng)?);
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_encode_decode_shape() {
        let mut rng = LcgRng::new(7);
        let vae = VaeAnomaly::new(&[8, 4], 2, &[2, 4, 8], &mut rng).expect("VAE should initialize");
        let x = vec![0.5_f32; 8];
        let (mu, lv) = vae.encode(&x).expect("VAE encode should succeed");
        assert_eq!(mu.len(), 2);
        assert_eq!(lv.len(), 2);
        let z = vae
            .reparametrize(&mu, &lv, &mut rng)
            .expect("VAE reparametrize should succeed");
        let xr = vae.decode(&z).expect("VAE decode should succeed");
        assert_eq!(xr.len(), 8);
        assert!(xr.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn vae_kl_zero_at_standard_normal() {
        let mu = vec![0.0_f32; 4];
        let lv = vec![0.0_f32; 4]; // log_var=0 → var=1
        let kl = VaeAnomaly::kl_divergence(&mu, &lv)
            .expect("KL divergence of standard normal should succeed");
        assert!(kl.abs() < 1e-5, "kl={kl}");
    }

    #[test]
    fn vae_score_finite_nonneg() {
        let mut rng = LcgRng::new(42);
        let vae = VaeAnomaly::new(&[8, 4], 2, &[2, 4, 8], &mut rng)
            .expect("VAE should initialize with valid dimensions");
        let s = vae
            .anomaly_score(&[0.2_f32; 8], &mut rng)
            .expect("anomaly score computation should succeed");
        assert!(s.is_finite(), "score={s}");
    }
}
