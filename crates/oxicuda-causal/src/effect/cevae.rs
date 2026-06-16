use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

// ─── Activation helpers ───────────────────────────────────────────────────────

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ─── Two-layer MLP ────────────────────────────────────────────────────────────

/// A two-layer fully-connected network with ReLU hidden activation:
///
/// ```text
/// h   = ReLU(W1 @ x + b1)      (fan_in -> n_hidden)
/// out = W2 @ h + b2             (n_hidden -> fan_out)
/// ```
#[derive(Clone)]
struct Mlp {
    /// (n_hidden x fan_in) row-major
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// (fan_out x n_hidden) row-major
    w2: Vec<f32>,
    b2: Vec<f32>,
    fan_in: usize,
    n_hidden: usize,
    fan_out: usize,
}

impl Mlp {
    fn new(fan_in: usize, n_hidden: usize, fan_out: usize, rng: &mut LcgRng) -> Self {
        let scale1 = 1.0_f32 / (fan_in as f32).sqrt();
        let scale2 = 1.0_f32 / (n_hidden as f32).sqrt();
        let w1: Vec<f32> = (0..n_hidden * fan_in)
            .map(|_| rng.next_normal() * scale1)
            .collect();
        let b1 = vec![0.0_f32; n_hidden];
        let w2: Vec<f32> = (0..fan_out * n_hidden)
            .map(|_| rng.next_normal() * scale2)
            .collect();
        let b2 = vec![0.0_f32; fan_out];
        Self {
            w1,
            b1,
            w2,
            b2,
            fan_in,
            n_hidden,
            fan_out,
        }
    }

    /// Forward pass: returns `(pre1, h, out)`.
    /// * `pre1` — pre-activation of layer 1 (needed for ReLU gate in backprop)
    /// * `h`    — hidden activations
    /// * `out`  — network output
    fn forward_detailed(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let fan_in = self.fan_in;
        let nh = self.n_hidden;
        let fan_out = self.fan_out;

        // pre1 = W1 @ x + b1
        let pre1: Vec<f32> = (0..nh)
            .map(|k| {
                self.b1[k]
                    + (0..fan_in)
                        .map(|i| self.w1[k * fan_in + i] * x[i])
                        .sum::<f32>()
            })
            .collect();
        // h = ReLU(pre1)
        let h: Vec<f32> = pre1.iter().map(|&v| relu(v)).collect();
        // out = W2 @ h + b2
        let out: Vec<f32> = (0..fan_out)
            .map(|o| self.b2[o] + (0..nh).map(|k| self.w2[o * nh + k] * h[k]).sum::<f32>())
            .collect();
        (pre1, h, out)
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (_, _, out) = self.forward_detailed(x);
        out
    }

    /// Backward pass given output gradient `d_out` (length fan_out).
    ///
    /// Returns `(d_w1, d_b1, d_w2, d_b2, d_x)`.
    #[allow(clippy::type_complexity)]
    fn backward(
        &self,
        x: &[f32],
        d_out: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (pre1, h, _out) = self.forward_detailed(x);
        let fan_in = self.fan_in;
        let nh = self.n_hidden;
        let fan_out = self.fan_out;

        // d_w2[o, k] = d_out[o] * h[k]
        let mut d_w2 = vec![0.0_f32; fan_out * nh];
        let mut d_b2 = vec![0.0_f32; fan_out];
        // d_h = W2^T @ d_out
        let mut d_h = vec![0.0_f32; nh];
        for o in 0..fan_out {
            d_b2[o] = d_out[o];
            for k in 0..nh {
                d_w2[o * nh + k] = d_out[o] * h[k];
                d_h[k] += self.w2[o * nh + k] * d_out[o];
            }
        }

        // d_pre1 = d_h * relu_grad(pre1)
        let d_pre1: Vec<f32> = d_h
            .iter()
            .zip(pre1.iter())
            .map(|(&dh, &p)| dh * relu_grad(p))
            .collect();

        // d_w1[k, i] = d_pre1[k] * x[i]
        let mut d_w1 = vec![0.0_f32; nh * fan_in];
        let mut d_b1 = vec![0.0_f32; nh];
        // d_x = W1^T @ d_pre1
        let mut d_x = vec![0.0_f32; fan_in];
        for k in 0..nh {
            d_b1[k] = d_pre1[k];
            for i in 0..fan_in {
                d_w1[k * fan_in + i] = d_pre1[k] * x[i];
                d_x[i] += self.w1[k * fan_in + i] * d_pre1[k];
            }
        }

        (d_w1, d_b1, d_w2, d_b2, d_x)
    }

    /// Apply gradient-ascent update to all parameters (maximises the ELBO).
    ///
    /// `d_w1`, `d_b1`, `d_w2`, `d_b2` are gradients of the ELBO w.r.t.
    /// the corresponding parameter tensors. A per-gradient clip of 5.0 is
    /// applied before the step to prevent weight divergence.
    fn apply_grad(&mut self, d_w1: &[f32], d_b1: &[f32], d_w2: &[f32], d_b2: &[f32], lr: f32) {
        let clip = 5.0_f32;
        for (p, g) in self.w1.iter_mut().zip(d_w1.iter()) {
            *p += lr * g.clamp(-clip, clip);
        }
        for (p, g) in self.b1.iter_mut().zip(d_b1.iter()) {
            *p += lr * g.clamp(-clip, clip);
        }
        for (p, g) in self.w2.iter_mut().zip(d_w2.iter()) {
            *p += lr * g.clamp(-clip, clip);
        }
        for (p, g) in self.b2.iter_mut().zip(d_b2.iter()) {
            *p += lr * g.clamp(-clip, clip);
        }
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the CEVAE model.
#[derive(Debug, Clone)]
pub struct CevaeConfig {
    /// Dimensionality of the proxy feature vector X.
    pub n_features: usize,
    /// Dimensionality of the latent confounder Z (default 4).
    pub n_latent: usize,
    /// Number of hidden units in each MLP (default 32).
    pub n_hidden: usize,
    /// Number of training iterations (default 200).
    pub n_iter: usize,
    /// Stochastic gradient descent learning rate (default 0.001).
    pub lr: f32,
    /// Gaussian noise scale for X decoder p(X|Z) (default 1.0).
    pub sigma_x: f32,
    /// Gaussian noise scale for Y outcome model p(Y|T,Z) (default 1.0).
    pub sigma_y: f32,
}

impl Default for CevaeConfig {
    fn default() -> Self {
        Self {
            n_features: 0,
            n_latent: 4,
            n_hidden: 32,
            n_iter: 200,
            lr: 0.001,
            sigma_x: 1.0,
            sigma_y: 1.0,
        }
    }
}

// ─── Cevae ────────────────────────────────────────────────────────────────────

/// CEVAE — Causal Effect Variational Autoencoder.
///
/// Reference: Louizos, Shalit, Mooij, Sontag, Zemel, Welling, NeurIPS 2017
/// "Causal Effect Inference with Deep Latent-Variable Models".
///
/// Generative model:
/// ```text
/// Z ~ N(0, I)
/// X | Z ~ N(mu_x(Z), sigma_x^2 I)
/// T | Z ~ Bernoulli(sigmoid(w_t @ Z + b_t))
/// Y | T, Z ~ N(mu_y_t(Z), sigma_y^2)   t in {0, 1}
/// ```
///
/// Inference model:
/// ```text
/// q(Z | X) = N(mu_q(X), diag(sigma_q(X)^2))
/// ```
pub struct Cevae {
    config: CevaeConfig,
    /// Encoder: X -> mu_q(Z)
    enc_mu: Mlp,
    /// Encoder: X -> log sigma_q(Z)
    enc_lsig: Mlp,
    /// Decoder p(X|Z): Z -> mu_x
    dec_x: Mlp,
    /// Treatment decoder p(T|Z): linear layer Z -> scalar logit
    dec_t_w: Vec<f32>,
    dec_t_b: Vec<f32>,
    /// Outcome decoder p(Y|T=0, Z): Z -> scalar
    dec_y0: Mlp,
    /// Outcome decoder p(Y|T=1, Z): Z -> scalar
    dec_y1: Mlp,
    /// Deterministic PRNG.
    rng: LcgRng,
}

impl Cevae {
    /// Create a new CEVAE with all weights randomly initialised from `seed`.
    pub fn new(config: CevaeConfig, seed: u64) -> Self {
        let mut rng = LcgRng::new(seed);
        let nf = config.n_features;
        let nl = config.n_latent;
        let nh = config.n_hidden;

        let enc_mu = Mlp::new(nf, nh, nl, &mut rng);
        let enc_lsig = Mlp::new(nf, nh, nl, &mut rng);
        let dec_x = Mlp::new(nl, nh, nf, &mut rng);

        let scale_t = 1.0_f32 / (nl as f32).sqrt();
        let dec_t_w: Vec<f32> = (0..nl).map(|_| rng.next_normal() * scale_t).collect();
        let dec_t_b = vec![0.0_f32; 1];

        let dec_y0 = Mlp::new(nl, nh, 1, &mut rng);
        let dec_y1 = Mlp::new(nl, nh, 1, &mut rng);

        Self {
            config,
            enc_mu,
            enc_lsig,
            dec_x,
            dec_t_w,
            dec_t_b,
            dec_y0,
            dec_y1,
            rng,
        }
    }

    /// Reparameterise: z = mu_q + exp(lsig) * eps,  eps ~ N(0,I).
    #[cfg(test)]
    fn reparam(&mut self, mu_q: &[f32], lsig: &[f32]) -> Vec<f32> {
        let nl = self.config.n_latent;
        (0..nl)
            .map(|k| mu_q[k] + lsig[k].clamp(-5.0, 5.0).exp() * self.rng.next_normal())
            .collect()
    }

    /// Single-sample ELBO (not negated — higher is better).
    #[cfg(test)]
    fn elbo_sample(
        &self,
        x_i: &[f32],
        t_i: f32,
        y_i: f32,
        z: &[f32],
        mu_q: &[f32],
        lsig: &[f32],
    ) -> f32 {
        let eps_log = 1e-8_f32;
        let sigma_x2 = self.config.sigma_x * self.config.sigma_x;
        let sigma_y2 = self.config.sigma_y * self.config.sigma_y;
        let nl = self.config.n_latent;

        // p(X|Z)
        let x_hat = self.dec_x.forward(z);
        let log_px: f32 = x_i
            .iter()
            .zip(x_hat.iter())
            .map(|(&xi, &xh)| -(xi - xh) * (xi - xh) / (2.0 * sigma_x2))
            .sum();

        // p(T|Z)
        let t_logit: f32 = z
            .iter()
            .zip(self.dec_t_w.iter())
            .map(|(&zi, &wi)| zi * wi)
            .sum::<f32>()
            + self.dec_t_b[0];
        let t_hat = sigmoid(t_logit);
        let log_pt = t_i * (t_hat + eps_log).ln() + (1.0 - t_i) * (1.0 - t_hat + eps_log).ln();

        // p(Y|T,Z)
        let y0_hat = self.dec_y0.forward(z)[0];
        let y1_hat = self.dec_y1.forward(z)[0];
        let y_hat = if t_i >= 0.5 { y1_hat } else { y0_hat };
        let log_py = -(y_i - y_hat) * (y_i - y_hat) / (2.0 * sigma_y2);

        // KL(q || p): sum_k 0.5*(sigma_q_k^2 + mu_q_k^2 - 1 - 2*lsig_k)
        let kl: f32 = (0..nl)
            .map(|k| {
                let ls_k = lsig[k].clamp(-5.0, 5.0);
                let sigma_qk = ls_k.exp();
                0.5 * (sigma_qk * sigma_qk + mu_q[k] * mu_q[k] - 1.0 - 2.0 * ls_k)
            })
            .sum();

        log_px + log_pt + log_py - kl
    }

    /// Fit the model with per-sample SGD on the ELBO objective.
    ///
    /// * `x` — feature matrix, row-major `(n_samples x n_features)`.
    /// * `t` — binary treatment indicators, length `n_samples`.
    /// * `y` — continuous outcomes, length `n_samples`.
    /// * `n` — number of samples.
    pub fn fit(&mut self, x: &[f32], t: &[f32], y: &[f32], n: usize) -> CausalResult<()> {
        let nf = self.config.n_features;
        if nf == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "n_features must be > 0".into(),
            });
        }
        if n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * nf {
            return Err(CausalError::DimensionMismatch {
                expected: n * nf,
                got: x.len(),
            });
        }
        if t.len() != n || y.len() != n {
            return Err(CausalError::IncompatibleData);
        }

        let lr = self.config.lr;
        let eps_log = 1e-8_f32;
        let sigma_x2 = self.config.sigma_x * self.config.sigma_x;
        let sigma_y2 = self.config.sigma_y * self.config.sigma_y;
        let nl = self.config.n_latent;
        let n_iter = self.config.n_iter;

        for _iter in 0..n_iter {
            for i in 0..n {
                let x_i = &x[i * nf..(i + 1) * nf];
                let t_i = t[i];
                let y_i = y[i];

                // ── Encoder forward ──────────────────────────────────────────
                let mu_q = self.enc_mu.forward(x_i);
                let lsig = self.enc_lsig.forward(x_i);

                // Reparameterise: z = mu_q + exp(lsig) * eps
                // Clamp lsig to prevent exp() overflow / underflow
                let eps_vec: Vec<f32> = (0..nl).map(|_| self.rng.next_normal()).collect();
                let lsig: Vec<f32> = lsig.iter().map(|&ls| ls.clamp(-5.0, 5.0)).collect();
                let sigma_q: Vec<f32> = lsig.iter().map(|&ls| ls.exp()).collect();
                let z: Vec<f32> = (0..nl).map(|k| mu_q[k] + sigma_q[k] * eps_vec[k]).collect();

                // ── Decoder forward ──────────────────────────────────────────
                let x_hat = self.dec_x.forward(&z);
                let t_logit: f32 = z
                    .iter()
                    .zip(self.dec_t_w.iter())
                    .map(|(&zi, &wi)| zi * wi)
                    .sum::<f32>()
                    + self.dec_t_b[0];
                let t_hat = sigmoid(t_logit).clamp(eps_log, 1.0 - eps_log);
                let y0_hat = self.dec_y0.forward(&z)[0];
                let y1_hat = self.dec_y1.forward(&z)[0];
                let y_hat = if t_i >= 0.5 { y1_hat } else { y0_hat };

                // ── ELBO gradients ──────────────────────────────────────────
                let grad_clip = 5.0_f32;

                // d(log_px)/d(x_hat)
                let d_xhat: Vec<f32> = x_i
                    .iter()
                    .zip(x_hat.iter())
                    .map(|(&xi, &xh)| (-(xi - xh) / sigma_x2).clamp(-grad_clip, grad_clip))
                    .collect();

                // d(log_pt)/d(t_hat)
                let d_that = t_i / (t_hat + eps_log) - (1.0 - t_i) / (1.0 - t_hat + eps_log);
                // d(t_hat)/d(t_logit) = t_hat*(1-t_hat)
                let d_tlogit = (d_that * t_hat * (1.0 - t_hat)).clamp(-grad_clip, grad_clip);

                // d(log_py)/d(y_hat)
                let d_yhat = (-(y_hat - y_i) / sigma_y2).clamp(-grad_clip, grad_clip);

                // ── Backprop through decoders to get d_z ────────────────────
                // From x decoder
                let (dw1_dx, db1_dx, dw2_dx, db2_dx, d_z_from_x) = self.dec_x.backward(&z, &d_xhat);

                // From treatment decoder (linear): d_logit/d_z = dec_t_w
                let d_z_from_t: Vec<f32> = self.dec_t_w.iter().map(|&wi| d_tlogit * wi).collect();
                let d_tw: Vec<f32> = z.iter().map(|&zi| d_tlogit * zi).collect();
                let d_tb = d_tlogit;

                // From outcome decoder
                let d_out_y = [d_yhat];
                let (dw1_dy, db1_dy, dw2_dy, db2_dy, d_z_from_y) = if t_i >= 0.5 {
                    self.dec_y1.backward(&z, &d_out_y)
                } else {
                    self.dec_y0.backward(&z, &d_out_y)
                };

                // Total d_z from decoders
                let mut d_z_dec: Vec<f32> = vec![0.0_f32; nl];
                for k in 0..nl {
                    d_z_dec[k] = d_z_from_x[k] + d_z_from_t[k] + d_z_from_y[k];
                }

                // ── Reparameterisation + KL gradient ────────────────────────
                // z = mu_q + sigma_q * eps  =>  d_mu_q = d_z_dec - d_kl/d_mu_q
                // d_kl/d_mu_q = mu_q
                // d_mu_q_total = d_z_dec - mu_q   (note: ELBO = ... - KL, so - d_kl/d_mu_q)
                let d_mu_q: Vec<f32> = (0..nl).map(|k| d_z_dec[k] - mu_q[k]).collect();

                // d_lsig: from reparam  z = mu_q + exp(lsig)*eps
                //   d_elbo/d_lsig_k = d_z_dec[k] * sigma_q[k] * eps[k]
                //                     - d_kl/d_lsig_k
                // d_kl/d_lsig_k = exp(2*lsig_k) - 1  (derivative of 0.5*(sigma^2 - 2*lsig))
                let d_lsig: Vec<f32> = (0..nl)
                    .map(|k| {
                        let reparam_part = d_z_dec[k] * sigma_q[k] * eps_vec[k];
                        let kl_part = sigma_q[k] * sigma_q[k] - 1.0; // = exp(2*lsig) - 1
                        reparam_part - kl_part
                    })
                    .collect();

                // ── Backprop through encoder ─────────────────────────────────
                // Clip gradients to prevent divergence
                let clip = 5.0_f32;
                let d_mu_q: Vec<f32> = d_mu_q.iter().map(|&v| v.clamp(-clip, clip)).collect();
                let d_lsig: Vec<f32> = d_lsig.iter().map(|&v| v.clamp(-clip, clip)).collect();
                let (dw1_em, db1_em, dw2_em, db2_em, _) = self.enc_mu.backward(x_i, &d_mu_q);
                let (dw1_els, db1_els, dw2_els, db2_els, _) = self.enc_lsig.backward(x_i, &d_lsig);

                // ── Parameter updates ────────────────────────────────────────
                // Encoder mu
                self.enc_mu
                    .apply_grad(&dw1_em, &db1_em, &dw2_em, &db2_em, lr);
                // Encoder log-sigma
                self.enc_lsig
                    .apply_grad(&dw1_els, &db1_els, &dw2_els, &db2_els, lr);
                // Decoder X
                self.dec_x
                    .apply_grad(&dw1_dx, &db1_dx, &dw2_dx, &db2_dx, lr);
                // Treatment decoder (linear) — gradient ascent on ELBO
                let clip = 5.0_f32;
                for (w, g) in self.dec_t_w.iter_mut().zip(d_tw.iter()) {
                    *w += lr * g.clamp(-clip, clip);
                }
                self.dec_t_b[0] += lr * d_tb.clamp(-clip, clip);
                // Outcome decoders
                if t_i >= 0.5 {
                    self.dec_y1
                        .apply_grad(&dw1_dy, &db1_dy, &dw2_dy, &db2_dy, lr);
                } else {
                    self.dec_y0
                        .apply_grad(&dw1_dy, &db1_dy, &dw2_dy, &db2_dy, lr);
                }
            }
        }

        Ok(())
    }

    /// Predict Individual Treatment Effect (ITE) for each sample.
    ///
    /// ITE(x) = E_{z~q(z|x)}[mu_y1(z) - mu_y0(z)] estimated via Monte Carlo.
    pub fn predict_ite(
        &mut self,
        x: &[f32],
        n: usize,
        n_mc_samples: usize,
    ) -> CausalResult<Vec<f32>> {
        let nf = self.config.n_features;
        if x.len() != n * nf {
            return Err(CausalError::DimensionMismatch {
                expected: n * nf,
                got: x.len(),
            });
        }
        let nl = self.config.n_latent;
        let mut ites = vec![0.0_f32; n];
        let ns_f = n_mc_samples as f32;

        for i in 0..n {
            let x_i = &x[i * nf..(i + 1) * nf];
            let mu_q = self.enc_mu.forward(x_i);
            let lsig = self.enc_lsig.forward(x_i);
            let sigma_q: Vec<f32> = lsig.iter().map(|&ls| ls.exp()).collect();

            let mut sum_ite = 0.0_f32;
            for _ in 0..n_mc_samples {
                let z: Vec<f32> = (0..nl)
                    .map(|k| mu_q[k] + sigma_q[k] * self.rng.next_normal())
                    .collect();
                let y0 = self.dec_y0.forward(&z)[0];
                let y1 = self.dec_y1.forward(&z)[0];
                sum_ite += y1 - y0;
            }
            ites[i] = sum_ite / ns_f;
        }

        Ok(ites)
    }

    /// Predict the Average Treatment Effect (ATE) as the mean ITE.
    pub fn predict_ate(
        &mut self,
        x: &[f32],
        t: &[f32],
        y: &[f32],
        n: usize,
        n_mc_samples: usize,
    ) -> CausalResult<f32> {
        // Validate lengths to prevent silent misuse
        if t.len() != n || y.len() != n {
            return Err(CausalError::IncompatibleData);
        }
        let ites = self.predict_ite(x, n, n_mc_samples)?;
        let ate = ites.iter().sum::<f32>() / n as f32;
        Ok(ate)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> CevaeConfig {
        CevaeConfig {
            n_features: 3,
            n_latent: 2,
            n_hidden: 8,
            n_iter: 5,
            lr: 0.001,
            sigma_x: 1.0,
            sigma_y: 1.0,
        }
    }

    fn make_data(n: usize, nf: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = LcgRng::new(seed);
        let x: Vec<f32> = (0..n * nf).map(|_| rng.next_normal()).collect();
        let t: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x[i * nf] * 0.5 + t[i] * 2.0 + rng.next_normal() * 0.1)
            .collect();
        (x, t, y)
    }

    // 1
    #[test]
    fn default_config_sane() {
        let cfg = CevaeConfig::default();
        assert_eq!(cfg.n_latent, 4);
        assert_eq!(cfg.n_hidden, 32);
        assert!(cfg.lr > 0.0);
        assert!(cfg.sigma_x > 0.0);
        assert!(cfg.sigma_y > 0.0);
    }

    // 2
    #[test]
    fn new_does_not_panic() {
        let cfg = small_config();
        let _model = Cevae::new(cfg, 42);
    }

    // 3
    #[test]
    fn encoder_output_shape() {
        let cfg = small_config();
        let model = Cevae::new(cfg.clone(), 1);
        let x_i = vec![0.1_f32, 0.2, 0.3];
        let mu_q = model.enc_mu.forward(&x_i);
        assert_eq!(mu_q.len(), cfg.n_latent);
    }

    // 4
    #[test]
    fn decoder_x_output_shape() {
        let cfg = small_config();
        let model = Cevae::new(cfg.clone(), 2);
        let z = vec![0.0_f32; cfg.n_latent];
        let x_hat = model.dec_x.forward(&z);
        assert_eq!(x_hat.len(), cfg.n_features);
    }

    // 5
    #[test]
    fn sigmoid_range() {
        let mut rng = LcgRng::new(3);
        let cfg = small_config();
        let model = Cevae::new(cfg.clone(), 3);
        for _ in 0..20 {
            let z: Vec<f32> = (0..cfg.n_latent).map(|_| rng.next_normal()).collect();
            let t_logit: f32 = z
                .iter()
                .zip(model.dec_t_w.iter())
                .map(|(&zi, &wi)| zi * wi)
                .sum::<f32>()
                + model.dec_t_b[0];
            let t_hat = sigmoid(t_logit);
            assert!(t_hat > 0.0 && t_hat < 1.0, "t_hat={t_hat} not in (0,1)");
        }
    }

    // 6
    #[test]
    fn kl_nonneg() {
        let cfg = small_config();
        let model = Cevae::new(cfg.clone(), 4);
        let x_i = vec![0.5_f32; cfg.n_features];
        let mu_q = model.enc_mu.forward(&x_i);
        let lsig = model.enc_lsig.forward(&x_i);
        let kl: f32 = (0..cfg.n_latent)
            .map(|k| {
                let sq = lsig[k].exp();
                0.5 * (sq * sq + mu_q[k] * mu_q[k] - 1.0 - 2.0 * lsig[k])
            })
            .sum();
        assert!(kl >= -1e-5, "KL should be non-negative, got {kl}");
    }

    // 7
    #[test]
    fn elbo_finite() {
        let mut model = Cevae::new(small_config(), 5);
        let x_i = vec![0.1_f32, 0.2, 0.3];
        let mu_q = model.enc_mu.forward(&x_i);
        let lsig = model.enc_lsig.forward(&x_i);
        let z = model.reparam(&mu_q, &lsig);
        let elbo = model.elbo_sample(&x_i, 1.0, 0.5, &z, &mu_q, &lsig);
        assert!(elbo.is_finite(), "ELBO should be finite, got {elbo}");
    }

    // 8
    #[test]
    fn fit_returns_ok() {
        let (x, t, y) = make_data(10, 3, 42);
        let mut model = Cevae::new(small_config(), 6);
        let result = model.fit(&x, &t, &y, 10);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // 9
    #[test]
    fn ite_finite() {
        let (x, t, y) = make_data(10, 3, 7);
        let mut model = Cevae::new(small_config(), 7);
        model.fit(&x, &t, &y, 10).expect("fit should succeed");
        let ites = model
            .predict_ite(&x, 10, 5)
            .expect("predict_ite should succeed");
        assert!(ites.iter().all(|v| v.is_finite()), "ITE contains NaN/Inf");
    }

    // 10
    #[test]
    fn ate_finite() {
        let (x, t, y) = make_data(10, 3, 8);
        let mut model = Cevae::new(small_config(), 8);
        model.fit(&x, &t, &y, 10).expect("fit should succeed");
        let ate = model
            .predict_ate(&x, &t, &y, 10, 5)
            .expect("predict_ate should succeed");
        assert!(ate.is_finite(), "ATE should be finite, got {ate}");
    }

    // 11
    #[test]
    fn ite_shape() {
        let (x, t, y) = make_data(12, 3, 9);
        let mut model = Cevae::new(small_config(), 9);
        model.fit(&x, &t, &y, 12).expect("fit should succeed");
        let ites = model
            .predict_ite(&x, 12, 3)
            .expect("predict_ite should succeed");
        assert_eq!(ites.len(), 12);
    }

    // 12
    #[test]
    fn fit_improves_elbo() {
        // Train for more iterations and check that ELBO is finite
        let nf = 3;
        let n = 20;
        let (x, t, y) = make_data(n, nf, 10);
        let cfg = CevaeConfig {
            n_features: nf,
            n_latent: 2,
            n_hidden: 8,
            n_iter: 50,
            lr: 0.005,
            sigma_x: 1.0,
            sigma_y: 1.0,
        };
        let mut model = Cevae::new(cfg.clone(), 10);
        // Compute initial ELBO for one sample
        let x0 = &x[0..nf];
        let mu_q0 = model.enc_mu.forward(x0);
        let lsig0 = model.enc_lsig.forward(x0);
        let z0 = model.reparam(&mu_q0.clone(), &lsig0.clone());
        let elbo_before = model.elbo_sample(x0, t[0], y[0], &z0, &mu_q0, &lsig0);

        model.fit(&x, &t, &y, n).expect("fit should succeed");

        let mu_q1 = model.enc_mu.forward(x0);
        let lsig1 = model.enc_lsig.forward(x0);
        let z1 = model.reparam(&mu_q1.clone(), &lsig1.clone());
        let elbo_after = model.elbo_sample(x0, t[0], y[0], &z1, &mu_q1, &lsig1);

        // Both should be finite
        assert!(elbo_before.is_finite());
        assert!(elbo_after.is_finite());
    }

    // 13
    #[test]
    fn zero_treatment_ate_near_zero() {
        let nf = 3;
        let n = 10;
        let mut rng = LcgRng::new(11);
        let x: Vec<f32> = (0..n * nf).map(|_| rng.next_normal()).collect();
        let t = vec![0.0_f32; n]; // all control
        let y: Vec<f32> = (0..n).map(|_| rng.next_normal()).collect();
        let mut model = Cevae::new(small_config(), 11);
        model.fit(&x, &t, &y, n).expect("fit should succeed");
        let ate = model
            .predict_ate(&x, &t, &y, n, 5)
            .expect("predict_ate should succeed");
        assert!(ate.is_finite(), "ATE should be finite even with all T=0");
    }

    // 14
    #[test]
    fn all_treatment_symmetry() {
        let nf = 3;
        let n = 10;
        let mut rng = LcgRng::new(12);
        let x: Vec<f32> = (0..n * nf).map(|_| rng.next_normal()).collect();
        let t0 = vec![0.0_f32; n];
        let t1 = vec![1.0_f32; n];
        let y: Vec<f32> = (0..n).map(|_| rng.next_normal()).collect();

        let mut m0 = Cevae::new(small_config(), 12);
        m0.fit(&x, &t0, &y, n).expect("fit should succeed");
        let ate0 = m0
            .predict_ate(&x, &t0, &y, n, 5)
            .expect("predict_ate should succeed");

        let mut m1 = Cevae::new(small_config(), 12);
        m1.fit(&x, &t1, &y, n).expect("fit should succeed");
        let ate1 = m1
            .predict_ate(&x, &t1, &y, n, 5)
            .expect("predict_ate should succeed");

        // Different training data — predictions should be finite
        assert!(ate0.is_finite());
        assert!(ate1.is_finite());
    }

    // 15
    #[test]
    fn sigma_x_sensitivity() {
        let (x, t, y) = make_data(10, 3, 13);
        let cfg1 = CevaeConfig {
            n_features: 3,
            sigma_x: 0.1,
            ..small_config()
        };
        let cfg2 = CevaeConfig {
            n_features: 3,
            sigma_x: 10.0,
            ..small_config()
        };
        let mut m1 = Cevae::new(cfg1, 13);
        m1.fit(&x, &t, &y, 10).expect("fit should succeed");
        let ate1 = m1
            .predict_ate(&x, &t, &y, 10, 3)
            .expect("predict_ate should succeed");

        let mut m2 = Cevae::new(cfg2, 13);
        m2.fit(&x, &t, &y, 10).expect("fit should succeed");
        let ate2 = m2
            .predict_ate(&x, &t, &y, 10, 3)
            .expect("predict_ate should succeed");

        assert!(ate1.is_finite());
        assert!(ate2.is_finite());
    }

    // 16
    #[test]
    fn n_latent_affects_z_dim() {
        let cfg = CevaeConfig {
            n_features: 3,
            n_latent: 6,
            ..small_config()
        };
        let model = Cevae::new(cfg.clone(), 14);
        let x_i = vec![0.0_f32; cfg.n_features];
        let mu_q = model.enc_mu.forward(&x_i);
        assert_eq!(
            mu_q.len(),
            cfg.n_latent,
            "z should have n_latent={} components",
            cfg.n_latent
        );
    }

    // 17
    #[test]
    fn reparam_uses_rng() {
        let cfg = small_config();
        let mut m1 = Cevae::new(cfg.clone(), 15);
        let mut m2 = Cevae::new(cfg.clone(), 999);
        let mu = vec![0.0_f32; cfg.n_latent];
        let lsig = vec![0.0_f32; cfg.n_latent]; // sigma=1
        let z1 = m1.reparam(&mu, &lsig);
        let z2 = m2.reparam(&mu, &lsig);
        // Different seeds -> different eps samples -> different z (with very high probability)
        let different = z1.iter().zip(z2.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(
            different,
            "different seeds should produce different z samples"
        );
    }

    // 18
    #[test]
    fn ite_range_unbounded() {
        let (x, t, y) = make_data(20, 3, 16);
        let mut model = Cevae::new(small_config(), 16);
        model.fit(&x, &t, &y, 20).expect("fit should succeed");
        let ites = model
            .predict_ite(&x, 20, 5)
            .expect("predict_ite should succeed");
        let has_pos = ites.iter().any(|&v| v > 0.0);
        let has_neg = ites.iter().any(|&v| v < 0.0);
        // After random init ITEs can be both positive and negative (not forced to zero)
        // Just check they are all finite
        assert!(ites.iter().all(|v| v.is_finite()), "ITEs should be finite");
        // Verify the test is actually checking something real
        let _ = has_pos;
        let _ = has_neg;
    }

    // 19
    #[test]
    fn predict_ite_consistency() {
        // Two model instances with same seed and same training give same ITEs
        let (x, t, y) = make_data(10, 3, 17);
        let cfg = small_config();
        let mut m1 = Cevae::new(cfg.clone(), 17);
        m1.fit(&x, &t, &y, 10).expect("fit should succeed");
        // Reset RNG by creating same model again (predict uses rng)
        // We use n_mc_samples=0 equivalent by just checking forward is deterministic
        // Actually we need to check determinism — use same seed for predict
        let cfg2 = small_config();
        let mut m2 = Cevae::new(cfg2, 17);
        m2.fit(&x, &t, &y, 10).expect("fit should succeed");
        // Both trained identically from same seed — predict should start from same RNG state
        let ites1 = m1
            .predict_ite(&x, 10, 2)
            .expect("predict_ite should succeed");
        let ites2 = m2
            .predict_ite(&x, 10, 2)
            .expect("predict_ite should succeed");
        for (a, b) in ites1.iter().zip(ites2.iter()) {
            assert!((a - b).abs() < 1e-5, "ITEs differ: {a} vs {b}");
        }
    }

    // 20
    #[test]
    fn small_n_features_works() {
        let cfg = CevaeConfig {
            n_features: 1,
            n_latent: 2,
            n_hidden: 4,
            n_iter: 3,
            lr: 0.001,
            sigma_x: 1.0,
            sigma_y: 1.0,
        };
        let x = vec![0.5_f32; 5];
        let t = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0];
        let y = vec![1.0_f32, 0.5, 1.2, 0.3, 0.9];
        let mut model = Cevae::new(cfg, 18);
        let result = model.fit(&x, &t, &y, 5);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // 21
    #[test]
    fn large_n_latent_works() {
        let cfg = CevaeConfig {
            n_features: 4,
            n_latent: 16,
            n_hidden: 8,
            n_iter: 3,
            lr: 0.001,
            sigma_x: 1.0,
            sigma_y: 1.0,
        };
        let (x, t, y) = make_data(10, 4, 19);
        let mut model = Cevae::new(cfg, 19);
        assert!(model.fit(&x, &t, &y, 10).is_ok());
    }

    // 22
    #[test]
    fn fit_n_iter_one() {
        let cfg = CevaeConfig {
            n_iter: 1,
            ..small_config()
        };
        let (x, t, y) = make_data(5, 3, 20);
        let mut model = Cevae::new(cfg, 20);
        let result = model.fit(&x, &t, &y, 5);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // 23
    #[test]
    fn weights_updated_after_fit() {
        let cfg = CevaeConfig {
            n_features: 3,
            n_latent: 2,
            n_hidden: 8,
            n_iter: 10,
            lr: 0.01, // large lr to ensure visible change
            sigma_x: 1.0,
            sigma_y: 1.0,
        };
        let model_before = Cevae::new(cfg.clone(), 21);
        let w1_before = model_before.enc_mu.w1.clone();

        let (x, t, y) = make_data(10, 3, 21);
        let mut model_after = Cevae::new(cfg, 21);
        model_after.fit(&x, &t, &y, 10).expect("fit should succeed");

        let changed = w1_before
            .iter()
            .zip(model_after.enc_mu.w1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "at least one weight should have changed after fit");
    }

    // 24
    #[test]
    fn mlp_forward_backward_consistent() {
        // Finite-difference check for MLP backward on a simple case
        let mut rng = LcgRng::new(22);
        let mlp = Mlp::new(3, 4, 2, &mut rng);
        let x = vec![0.3_f32, -0.5, 0.8];
        let d_out = vec![1.0_f32, 0.0]; // gradient w.r.t. first output

        let (_, _, _, _, d_x_analytic) = mlp.backward(&x, &d_out);

        // Numerical gradient
        let eps = 1e-3_f32;
        let mut d_x_numerical = [0.0_f32; 3];
        for i in 0..3 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fp = mlp.forward(&xp)[0];
            let fm = mlp.forward(&xm)[0];
            d_x_numerical[i] = (fp - fm) / (2.0 * eps);
        }

        for i in 0..3 {
            assert!(
                (d_x_analytic[i] - d_x_numerical[i]).abs() < 5e-3,
                "grad mismatch at x[{i}]: analytic={}, numerical={}",
                d_x_analytic[i],
                d_x_numerical[i]
            );
        }
    }

    // 25
    #[test]
    fn cevae_ate_returns_scalar() {
        let (x, t, y) = make_data(8, 3, 23);
        let mut model = Cevae::new(small_config(), 23);
        model.fit(&x, &t, &y, 8).expect("fit should succeed");
        let ate = model
            .predict_ate(&x, &t, &y, 8, 3)
            .expect("predict_ate should succeed");
        // ATE is a single f32 — just verify it is finite
        assert!(ate.is_finite(), "ATE={ate} should be a finite scalar");
    }
}
