use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    (0..fan_out)
        .map(|o| {
            b[o] + w[o * fan_in..(o + 1) * fan_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

fn init_layer(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let scale = (2.0_f32 / fan_in as f32).sqrt();
    let w = (0..fan_in * fan_out)
        .map(|_| rng.next_normal() * scale)
        .collect();
    let b = vec![0.0_f32; fan_out];
    (w, b)
}

fn mlp_forward(
    layers: &[(Vec<f32>, Vec<f32>)],
    x: &[f32],
    input_dim: usize,
    hidden_dim: usize,
) -> f32 {
    let mut h = x.to_vec();
    let n = layers.len();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let fan_in = if idx == 0 { input_dim } else { hidden_dim };
        let fan_out = if idx == n - 1 { 1 } else { hidden_dim };
        let out = dense(&h, w, b, fan_in, fan_out);
        h = if idx < n - 1 {
            out.iter().map(|&v| relu(v)).collect()
        } else {
            out
        };
    }
    h[0]
}

/// Twin Network for counterfactual inference.
/// Shares encoder; uses separate decoders for T=0 and T=1.
pub struct TwinNetwork {
    encoder: Vec<(Vec<f32>, Vec<f32>)>,
    decoder0: Vec<(Vec<f32>, Vec<f32>)>,
    decoder1: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
    pub latent_dim: usize,
}

impl TwinNetwork {
    pub fn new(input_dim: usize, latent_dim: usize, n_layers: usize, rng: &mut LcgRng) -> Self {
        let mut encoder = Vec::with_capacity(n_layers);
        let mut prev = input_dim;
        for _ in 0..n_layers {
            encoder.push(init_layer(prev, latent_dim, rng));
            prev = latent_dim;
        }

        let mut decoder0 = Vec::with_capacity(n_layers);
        let mut decoder1 = Vec::with_capacity(n_layers);
        prev = latent_dim;
        for layer_idx in 0..n_layers {
            let fan_in = prev;
            let fan_out = if layer_idx == n_layers - 1 {
                1
            } else {
                latent_dim
            };
            decoder0.push(init_layer(fan_in, fan_out, rng));
            decoder1.push(init_layer(fan_in, fan_out, rng));
            prev = fan_out;
        }

        Self {
            encoder,
            decoder0,
            decoder1,
            input_dim,
            latent_dim,
        }
    }

    fn encode(&self, x: &[f32]) -> CausalResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(CausalError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let mut h = x.to_vec();
        let mut prev = self.input_dim;
        for (w, b) in self.encoder.iter() {
            let fan_out = self.latent_dim;
            let out = dense(&h, w, b, prev, fan_out);
            h = out.iter().map(|&v| relu(v)).collect();
            prev = fan_out;
        }
        Ok(h)
    }

    fn decode(&self, latent: &[f32], treatment: f32) -> f32 {
        let decoder = if treatment >= 0.5 {
            &self.decoder1
        } else {
            &self.decoder0
        };
        mlp_forward(decoder, latent, self.latent_dim, self.latent_dim)
    }

    /// Predict factual outcome for given x and treatment t.
    pub fn forward_factual(&self, x: &[f32], t: f32) -> CausalResult<f32> {
        let latent = self.encode(x)?;
        Ok(self.decode(&latent, t))
    }

    /// Predict counterfactual outcome by flipping treatment.
    pub fn forward_counterfactual(&self, x: &[f32], t: f32) -> CausalResult<f32> {
        let latent = self.encode(x)?;
        Ok(self.decode(&latent, 1.0 - t))
    }

    /// Individual treatment effect: E[Y(1)] - E[Y(0)].
    pub fn ite(&self, x: &[f32]) -> CausalResult<f32> {
        let latent = self.encode(x)?;
        let y1 = self.decode(&latent, 1.0);
        let y0 = self.decode(&latent, 0.0);
        Ok(y1 - y0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twin_network_ite_finite() {
        let mut rng = LcgRng::new(55);
        let net = TwinNetwork::new(4, 8, 2, &mut rng);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let ite = net
            .ite(&x)
            .expect("TwinNetwork::ite should succeed for valid input");
        assert!(ite.is_finite());
    }
}
