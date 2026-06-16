use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
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

/// DragonNet (Shi et al. 2019): shared representation + 3 heads + targeted regularization.
pub struct DragonNet {
    shared_w: Vec<(Vec<f32>, Vec<f32>)>,
    mu0_w: Vec<f32>,
    mu0_b: f32,
    mu1_w: Vec<f32>,
    mu1_b: f32,
    pi_w: Vec<f32>,
    pi_b: f32,
    eps: f32,
    pub input_dim: usize,
    hidden_dim: usize,
}

impl DragonNet {
    pub fn new(input_dim: usize, hidden_dim: usize, n_hidden: usize, rng: &mut LcgRng) -> Self {
        let mut shared_w = Vec::with_capacity(n_hidden);
        let mut prev = input_dim;
        for _ in 0..n_hidden {
            let (w, b) = init_layer(prev, hidden_dim, rng);
            shared_w.push((w, b));
            prev = hidden_dim;
        }

        let (mu0_w, mu0_b_vec) = init_layer(hidden_dim, 1, rng);
        let (mu1_w, mu1_b_vec) = init_layer(hidden_dim, 1, rng);
        let (pi_w, pi_b_vec) = init_layer(hidden_dim, 1, rng);

        Self {
            shared_w,
            mu0_w,
            mu0_b: mu0_b_vec[0],
            mu1_w,
            mu1_b: mu1_b_vec[0],
            pi_w,
            pi_b: pi_b_vec[0],
            eps: 0.01,
            input_dim,
            hidden_dim,
        }
    }

    fn shared_forward(&self, x: &[f32]) -> CausalResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(CausalError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let mut h = x.to_vec();
        let mut prev = self.input_dim;
        for (w, b) in &self.shared_w {
            let out = dense(&h, w, b, prev, self.hidden_dim);
            h = out.iter().map(|&v| relu(v)).collect();
            prev = self.hidden_dim;
        }
        Ok(h)
    }

    /// Returns (mu0, mu1, pi) for a single sample.
    pub fn forward(&self, x: &[f32]) -> CausalResult<(f32, f32, f32)> {
        let h = self.shared_forward(x)?;
        let mu0 = dense(&h, &self.mu0_w, &[self.mu0_b], self.hidden_dim, 1)[0];
        let mu1 = dense(&h, &self.mu1_w, &[self.mu1_b], self.hidden_dim, 1)[0];
        let pi_logit = dense(&h, &self.pi_w, &[self.pi_b], self.hidden_dim, 1)[0];
        let pi = sigmoid(pi_logit).clamp(0.01, 0.99);
        Ok((mu0, mu1, pi))
    }

    /// Targeted regularization loss: MSE + alpha*CE + beta*eps*(Y-mu_T)*(T-pi).
    pub fn targeted_loss(&self, x: &[f32], t: f32, y: f32) -> CausalResult<f32> {
        let (mu0, mu1, pi) = self.forward(x)?;
        let mu_t = if t >= 0.5 { mu1 } else { mu0 };
        let mse = (y - mu_t).powi(2);
        let ce = -(t * pi.ln() + (1.0 - t) * (1.0 - pi).ln());
        let targeted = self.eps * (y - mu_t) * (t - pi);
        Ok(mse + ce + targeted)
    }

    /// CATE estimate: mu1 - mu0.
    pub fn cate(&self, x: &[f32]) -> CausalResult<f32> {
        let (mu0, mu1, _pi) = self.forward(x)?;
        Ok(mu1 - mu0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragonnet_forward() {
        let mut rng = LcgRng::new(99);
        let net = DragonNet::new(4, 8, 2, &mut rng);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let (mu0, mu1, pi) = net
            .forward(&x)
            .expect("DragonNet::forward should succeed for valid input");
        assert!(mu0.is_finite());
        assert!(mu1.is_finite());
        assert!(pi > 0.0 && pi < 1.0);
    }
}
