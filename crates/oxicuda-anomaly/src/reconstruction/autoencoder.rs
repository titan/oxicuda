//! Autoencoder-based anomaly detection.
//!
//! **Score** = MSE reconstruction error `(1/d) Σ (x_j - x̂_j)²`.
//! Normal data has low reconstruction error; anomalies have high error.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Xavier initialisation ───────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u = rng.next_f32();
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─── Layer helpers ────────────────────────────────────────────────────────────

/// Apply a single dense layer: `y = W x + b`.
fn dense_layer(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
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

fn relu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

fn sigmoid_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

// ─── AeConfig ────────────────────────────────────────────────────────────────

/// Autoencoder architecture specification.
pub struct AeConfig {
    /// Encoder layer widths including input: `[input, h1, ..., latent]`.
    pub encoder_dims: Vec<usize>,
    /// Decoder layer widths: `[latent, h1, ..., input]`.
    pub decoder_dims: Vec<usize>,
}

// ─── AutoencoderAnomaly ───────────────────────────────────────────────────────

/// Autoencoder anomaly detector.
pub struct AutoencoderAnomaly {
    /// Encoder layers: `(weight [out*in], bias [out])`.
    encoder_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Decoder layers: `(weight [out*in], bias [out])`.
    decoder_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Encoder layer dimensions (including input dim).
    enc_dims: Vec<usize>,
    /// Decoder layer dimensions (including latent dim).
    dec_dims: Vec<usize>,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality.
    pub latent_dim: usize,
}

impl AutoencoderAnomaly {
    /// Construct an autoencoder from `AeConfig`.
    ///
    /// ReLU activations on all encoder/decoder intermediate layers;
    /// sigmoid on the final decoder output.
    pub fn new(cfg: AeConfig, rng: &mut LcgRng) -> AnomalyResult<Self> {
        if cfg.encoder_dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "encoder_dims needs at least [input, latent]".into(),
            });
        }
        if cfg.decoder_dims.len() < 2 {
            return Err(AnomalyError::InvalidLayerDims {
                msg: "decoder_dims needs at least [latent, output]".into(),
            });
        }
        let input_dim = cfg.encoder_dims[0];
        let latent_dim = *cfg.encoder_dims.last().unwrap_or(&1);

        let mut encoder_layers = Vec::with_capacity(cfg.encoder_dims.len() - 1);
        for i in 0..cfg.encoder_dims.len() - 1 {
            let fan_in = cfg.encoder_dims[i];
            let fan_out = cfg.encoder_dims[i + 1];
            let w = xavier_init(fan_in, fan_out, rng);
            let b = vec![0.0_f32; fan_out];
            encoder_layers.push((w, b));
        }

        let mut decoder_layers = Vec::with_capacity(cfg.decoder_dims.len() - 1);
        for i in 0..cfg.decoder_dims.len() - 1 {
            let fan_in = cfg.decoder_dims[i];
            let fan_out = cfg.decoder_dims[i + 1];
            let w = xavier_init(fan_in, fan_out, rng);
            let b = vec![0.0_f32; fan_out];
            decoder_layers.push((w, b));
        }

        Ok(Self {
            encoder_layers,
            decoder_layers,
            enc_dims: cfg.encoder_dims,
            dec_dims: cfg.decoder_dims,
            input_dim,
            latent_dim,
        })
    }

    /// Encode `x` → `[latent_dim]`.
    pub fn encode(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let n_enc = self.encoder_layers.len();
        let mut act: Vec<f32> = x.to_vec();
        for (layer_idx, (w, b)) in self.encoder_layers.iter().enumerate() {
            let fan_in = self.enc_dims[layer_idx];
            let fan_out = self.enc_dims[layer_idx + 1];
            let mut out = dense_layer(&act, w, b, fan_in, fan_out);
            if layer_idx < n_enc - 1 {
                relu_inplace(&mut out);
            }
            act = out;
        }
        Ok(act)
    }

    /// Decode `z` → `[input_dim]` with sigmoid output.
    pub fn decode(&self, z: &[f32]) -> AnomalyResult<Vec<f32>> {
        if z.len() != self.latent_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.latent_dim,
                got: z.len(),
            });
        }
        let n_dec = self.decoder_layers.len();
        let mut act: Vec<f32> = z.to_vec();
        for (layer_idx, (w, b)) in self.decoder_layers.iter().enumerate() {
            let fan_in = self.dec_dims[layer_idx];
            let fan_out = self.dec_dims[layer_idx + 1];
            let mut out = dense_layer(&act, w, b, fan_in, fan_out);
            if layer_idx < n_dec - 1 {
                relu_inplace(&mut out);
            } else {
                sigmoid_inplace(&mut out);
            }
            act = out;
        }
        Ok(act)
    }

    /// Encode then decode.
    pub fn reconstruct(&self, x: &[f32]) -> AnomalyResult<Vec<f32>> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    /// Anomaly score: `MSE(x, x̂) = (1/d) Σ (x_j - x̂_j)²`.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        let x_hat = self.reconstruct(x)?;
        let mse = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / x.len() as f32;
        Ok(mse)
    }

    /// Batch scoring; `x` is `[n * input_dim]`; returns `[n]`.
    pub fn score_batch(&self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if x.len() != n * self.input_dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.input_dim,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = &x[i * self.input_dim..(i + 1) * self.input_dim];
            scores.push(self.score(sample)?);
        }
        Ok(scores)
    }

    /// Total reconstruction loss over a batch (sum of MSEs).
    pub fn recon_loss(&self, x: &[f32], n: usize) -> AnomalyResult<f32> {
        let scores = self.score_batch(x, n)?;
        Ok(scores.iter().sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ae_encode_decode_shapes() {
        let cfg = AeConfig {
            encoder_dims: vec![8, 4, 2],
            decoder_dims: vec![2, 4, 8],
        };
        let mut rng = LcgRng::new(42);
        let ae = AutoencoderAnomaly::new(cfg, &mut rng).expect("autoencoder should initialize");
        let x = vec![0.5_f32; 8];
        let z = ae.encode(&x).expect("autoencoder encode should succeed");
        assert_eq!(z.len(), 2);
        let xr = ae.decode(&z).expect("autoencoder decode should succeed");
        assert_eq!(xr.len(), 8);
    }

    #[test]
    fn ae_score_finite() {
        let cfg = AeConfig {
            encoder_dims: vec![8, 4, 2],
            decoder_dims: vec![2, 4, 8],
        };
        let mut rng = LcgRng::new(99);
        let ae = AutoencoderAnomaly::new(cfg, &mut rng)
            .expect("autoencoder should initialize with valid config");
        let s = ae
            .score(&[0.1_f32; 8])
            .expect("score computation should succeed");
        assert!(s.is_finite(), "score={s}");
        assert!(s >= 0.0, "score={s}");
    }
}
