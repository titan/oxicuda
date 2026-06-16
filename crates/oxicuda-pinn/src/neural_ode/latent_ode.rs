//! Latent ODE with GRU encoder, reparameterization trick, and MLP decoder.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for the Latent ODE model.
pub struct LatentOdeConfig {
    /// Observation dimensionality.
    pub d_obs: usize,
    /// Latent dimensionality.
    pub d_latent: usize,
    /// Hidden layer width for encoder GRU and decoder MLP.
    pub d_hidden: usize,
}

// ─── GRU Encoder ─────────────────────────────────────────────────────────────

/// Minimal GRU cell: hidden state update via gating.
struct GruEncoder {
    d_hidden: usize,
    // Weights for gates: [d_hidden × (d_hidden + d_in)]
    w_z: Vec<f32>,
    b_z: Vec<f32>,
    w_r: Vec<f32>,
    b_r: Vec<f32>,
    w_h: Vec<f32>,
    b_h: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn linear_layer(w: &[f32], b: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
    let d_in = x.len();
    (0..d_out)
        .map(|i| {
            let dot: f32 = (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum();
            dot + b[i]
        })
        .collect()
}

impl GruEncoder {
    fn new(d_obs: usize, d_hidden: usize, rng: &mut LcgRng) -> Self {
        let d_in = d_hidden + d_obs; // [h, x] concatenated
        let scale = (2.0_f32 / d_in as f32).sqrt();
        let n = d_hidden * d_in;

        let mut init = |n: usize| -> (Vec<f32>, Vec<f32>) {
            let mut w = vec![0.0_f32; n];
            for v in &mut w {
                *v = (rng.next_f32() * 2.0 - 1.0) * scale;
            }
            (w, vec![0.0_f32; d_hidden])
        };

        let (w_z, b_z) = init(n);
        let (w_r, b_r) = init(n);
        let (w_h, b_h) = init(n);

        let _ = d_in; // d_in is captured by scale/n above, not stored
        Self {
            d_hidden,
            w_z,
            b_z,
            w_r,
            b_r,
            w_h,
            b_h,
        }
    }

    /// One GRU step: (h, x) → h_new.
    fn step(&self, h: &[f32], x: &[f32]) -> Vec<f32> {
        // Concatenate [h, x]
        let hx: Vec<f32> = h.iter().chain(x.iter()).copied().collect();

        let z_gate: Vec<f32> = linear_layer(&self.w_z, &self.b_z, &hx, self.d_hidden)
            .into_iter()
            .map(sigmoid)
            .collect();
        let r_gate: Vec<f32> = linear_layer(&self.w_r, &self.b_r, &hx, self.d_hidden)
            .into_iter()
            .map(sigmoid)
            .collect();

        // r ⊙ h
        let rh: Vec<f32> = r_gate
            .iter()
            .zip(h.iter())
            .map(|(&r, &hi)| r * hi)
            .collect();
        // [r⊙h, x]
        let rhx: Vec<f32> = rh.iter().chain(x.iter()).copied().collect();
        let h_tilde: Vec<f32> = linear_layer(&self.w_h, &self.b_h, &rhx, self.d_hidden)
            .into_iter()
            .map(|v| v.tanh())
            .collect();

        // h_new = (1 - z) ⊙ h + z ⊙ h_tilde
        z_gate
            .iter()
            .zip(h.iter())
            .zip(h_tilde.iter())
            .map(|((&zi, &hi), &hti)| (1.0 - zi) * hi + zi * hti)
            .collect()
    }

    /// Encode a sequence of observations [T × d_obs] → final hidden state.
    fn encode_sequence(&self, obs: &[f32], t_steps: usize, d_obs: usize) -> PinnResult<Vec<f32>> {
        if obs.len() != t_steps * d_obs {
            return Err(PinnError::DimensionMismatch {
                expected: t_steps * d_obs,
                got: obs.len(),
            });
        }
        let mut h = vec![0.0_f32; self.d_hidden];
        // Run in reverse for better encoding (like LSTM encoder in Latent ODE paper)
        for t in (0..t_steps).rev() {
            let x = &obs[t * d_obs..(t + 1) * d_obs];
            h = self.step(&h, x);
        }
        Ok(h)
    }
}

// ─── LatentOde ───────────────────────────────────────────────────────────────

/// Latent ODE model: VAE with ODE prior.
pub struct LatentOde {
    config: LatentOdeConfig,
    encoder_gru: GruEncoder,
    // Linear layers from d_hidden → d_latent for μ and log σ
    mu_w: Vec<f32>,
    mu_b: Vec<f32>,
    sigma_w: Vec<f32>,
    sigma_b: Vec<f32>,
    // Decoder: [d_latent → d_hidden → d_obs] (2-layer MLP)
    decoder_w: Vec<Vec<f32>>,
    decoder_b: Vec<Vec<f32>>,
}

impl LatentOde {
    /// Construct a new Latent ODE model with randomly initialized weights.
    pub fn new(config: LatentOdeConfig, rng: &mut LcgRng) -> Self {
        let d_h = config.d_hidden;
        let d_l = config.d_latent;
        let d_o = config.d_obs;

        let encoder_gru = GruEncoder::new(d_o, d_h, rng);

        let scale_h = (2.0 / d_h as f32).sqrt();
        let mut init_lin = |d_in: usize, d_out: usize, s: f32| -> (Vec<f32>, Vec<f32>) {
            let n = d_out * d_in;
            let w: Vec<f32> = (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * s).collect();
            (w, vec![0.0_f32; d_out])
        };

        let (mu_w, mu_b) = init_lin(d_h, d_l, scale_h);
        let (sigma_w, sigma_b) = init_lin(d_h, d_l, scale_h);

        let scale_l = (2.0 / d_l as f32).sqrt();
        let scale_o = (2.0 / d_h as f32).sqrt();
        let (dw0, db0) = init_lin(d_l, d_h, scale_l);
        let (dw1, db1) = init_lin(d_h, d_o, scale_o);

        Self {
            config,
            encoder_gru,
            mu_w,
            mu_b,
            sigma_w,
            sigma_b,
            decoder_w: vec![dw0, dw1],
            decoder_b: vec![db0, db1],
        }
    }

    /// Encode observation sequence \[T × d_obs\] → (mu, log_sigma) both \[d_latent\].
    pub fn encode(&self, obs: &[f32], t_steps: usize) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        let h = self
            .encoder_gru
            .encode_sequence(obs, t_steps, self.config.d_obs)?;
        let d_l = self.config.d_latent;
        let mu = linear_layer(&self.mu_w, &self.mu_b, &h, d_l);
        let log_sigma = linear_layer(&self.sigma_w, &self.sigma_b, &h, d_l);
        Ok((mu, log_sigma))
    }

    /// Reparameterization: `z = mu + exp(log_sigma) * ε`, ε ~ N(0, I) via Box-Muller.
    pub fn reparam(&self, mu: &[f32], log_sigma: &[f32], rng: &mut LcgRng) -> Vec<f32> {
        let d_l = mu.len();
        let mut eps = vec![0.0_f32; d_l];
        rng.fill_normal(&mut eps);
        mu.iter()
            .zip(log_sigma.iter())
            .zip(eps.iter())
            .map(|((&m, &ls), &e)| m + ls.exp() * e)
            .collect()
    }

    /// Decode latent trajectory [T × d_latent] → observation predictions [T × d_obs].
    pub fn decode(&self, latent_traj: &[f32], t_steps: usize) -> PinnResult<Vec<f32>> {
        let d_l = self.config.d_latent;
        let d_o = self.config.d_obs;
        let d_h = self.config.d_hidden;

        if latent_traj.len() != t_steps * d_l {
            return Err(PinnError::DimensionMismatch {
                expected: t_steps * d_l,
                got: latent_traj.len(),
            });
        }

        let mut output = vec![0.0_f32; t_steps * d_o];
        for t in 0..t_steps {
            let z = &latent_traj[t * d_l..(t + 1) * d_l];
            // Layer 0: d_l → d_h, tanh
            let h: Vec<f32> = linear_layer(&self.decoder_w[0], &self.decoder_b[0], z, d_h)
                .into_iter()
                .map(|v| v.tanh())
                .collect();
            // Layer 1: d_h → d_o, linear
            let obs_pred: Vec<f32> = linear_layer(&self.decoder_w[1], &self.decoder_b[1], &h, d_o);
            output[t * d_o..(t + 1) * d_o].copy_from_slice(&obs_pred);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> LatentOdeConfig {
        LatentOdeConfig {
            d_obs: 3,
            d_latent: 4,
            d_hidden: 8,
        }
    }

    #[test]
    fn latent_ode_construct_no_panic() {
        let mut rng = LcgRng::new(42);
        let _model = LatentOde::new(make_config(), &mut rng);
    }

    #[test]
    fn encode_output_shape() {
        let mut rng = LcgRng::new(1);
        let model = LatentOde::new(make_config(), &mut rng);
        let obs = vec![0.0_f32; 5 * 3]; // T=5, d_obs=3
        let (mu, log_sigma) = model
            .encode(&obs, 5)
            .expect("encode should succeed with valid observation sequence");
        assert_eq!(mu.len(), 4);
        assert_eq!(log_sigma.len(), 4);
    }

    #[test]
    fn reparam_shape() {
        let mut rng = LcgRng::new(2);
        let model = LatentOde::new(make_config(), &mut rng);
        let mu = vec![0.0_f32; 4];
        let log_sigma = vec![0.0_f32; 4];
        let z = model.reparam(&mu, &log_sigma, &mut rng);
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn decode_output_shape() {
        let mut rng = LcgRng::new(3);
        let model = LatentOde::new(make_config(), &mut rng);
        let traj = vec![0.0_f32; 5 * 4]; // T=5, d_latent=4
        let obs = model
            .decode(&traj, 5)
            .expect("decode with valid latent trajectory of 5 steps should succeed");
        assert_eq!(obs.len(), 5 * 3);
    }

    #[test]
    fn deterministic_with_seed() {
        let mut rng1 = LcgRng::new(100);
        let m1 = LatentOde::new(make_config(), &mut rng1);
        let mut rng2 = LcgRng::new(100);
        let m2 = LatentOde::new(make_config(), &mut rng2);
        // Same seed → same weights
        assert_eq!(m1.mu_w, m2.mu_w);
    }

    #[test]
    fn zero_log_sigma_deterministic_decode() {
        // σ = exp(log_σ = -inf) → σ ≈ 0 → z = mu (deterministic)
        let mut rng = LcgRng::new(42);
        let model = LatentOde::new(make_config(), &mut rng);
        let mu = vec![0.5_f32; 4];
        let log_sigma_neg_inf = vec![-20.0_f32; 4]; // exp(-20) ≈ 0
        let z = model.reparam(&mu, &log_sigma_neg_inf, &mut rng);
        // z should be very close to mu
        for (&zi, &mi) in z.iter().zip(mu.iter()) {
            assert!((zi - mi).abs() < 0.01, "z={zi} should be near mu={mi}");
        }
    }

    #[test]
    fn decode_finite_outputs() {
        let mut rng = LcgRng::new(7);
        let model = LatentOde::new(make_config(), &mut rng);
        let traj = vec![0.3_f32; 10 * 4];
        let obs = model
            .decode(&traj, 10)
            .expect("decode with valid latent trajectory of 10 steps should succeed");
        assert!(
            obs.iter().all(|v| v.is_finite()),
            "Decoded outputs not finite"
        );
    }
}
