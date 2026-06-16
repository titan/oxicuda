//! Nouveau VAE (NVAE, Vahdat & Kautz 2020) — a compact hierarchical VAE with
//! free-bits KL balancing.
//!
//! The latent space is split into `L` ordered groups `z_1, …, z_L`.  A
//! bottom-up encoder produces a deterministic feature `h = tanh(W_enc·x + b)`;
//! a top-down generative path defines a conditional prior `p(z_l | z_{<l})` and
//! the approximate posterior `q(z_l | x, z_{<l})` for every group:
//!
//! ```text
//! q(z_l | x, z_{<l}) = N(μ_q^l([h; z_{l-1}]),  diag σ_q^l²)
//! p(z_l | z_{<l})    = N(μ_p^l(z_{l-1}),       diag σ_p^l²)        (l ≥ 2)
//! p(z_1)             = N(0, I).
//! ```
//!
//! Samples are drawn with the reparameterisation trick and propagated forward;
//! the decoder maps the concatenated latents to a Gaussian reconstruction.  The
//! evidence lower bound is
//!
//! ```text
//! ELBO = E_q[log p(x | z)] − Σ_l max(λ, KL(q(z_l) ‖ p(z_l))),
//! ```
//!
//! where the per-group KL is clamped from below by the **free-bits** threshold
//! `λ` (Kingma et al. 2016).  Free bits prevent posterior collapse by removing
//! the optimisation pressure to drive a group's KL below `λ` nats; this is the
//! KL-balancing heuristic NVAE applies per group.
//!
//! This is a CPU reference implementation: it is intentionally compact (no
//! residual flows, single-sample Monte-Carlo ELBO) but keeps the genuine
//! hierarchical top-down prior and the free-bits objective.
//!
//! **References:**
//! - Vahdat, A., & Kautz, J. (2020). NVAE: A Deep Hierarchical Variational
//!   Autoencoder. *NeurIPS*.
//! - Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., &
//!   Welling, M. (2016). Improved Variational Inference with Inverse
//!   Autoregressive Flow. *NeurIPS* (free-bits objective).

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Constructor configuration for [`NVae`].
#[derive(Debug, Clone)]
pub struct NVaeConfig {
    /// Observation dimensionality.
    pub input_dim: usize,
    /// Latent dimensionality of each hierarchical group (ordered top-down).
    pub group_dims: Vec<usize>,
    /// Width of the deterministic encoder feature `h`.
    pub hidden_dim: usize,
    /// Free-bits threshold `λ` (nats) applied per group in the ELBO.
    pub free_bits: f32,
}

// ─── Forward output ──────────────────────────────────────────────────────────

/// Result of a single stochastic forward pass through [`NVae`].
#[derive(Debug, Clone)]
pub struct NVaeOutput {
    /// Reparameterised latent sample for each group.
    pub z: Vec<Vec<f32>>,
    /// Gaussian reconstruction mean, length `input_dim`.
    pub x_hat: Vec<f32>,
    /// Raw per-group KL divergence `KL(q(z_l) ‖ p(z_l))`.
    pub kl_per_group: Vec<f32>,
    /// Free-bits-balanced per-group KL `max(λ, KL_l)`.
    pub kl_balanced_per_group: Vec<f32>,
    /// Reconstruction log-likelihood `log p(x | z)` (unit-variance Gaussian).
    pub recon_log_likelihood: f32,
    /// Evidence lower bound `recon − Σ_l max(λ, KL_l)`.
    pub elbo: f32,
}

// ─── Main struct ─────────────────────────────────────────────────────────────

/// Compact hierarchical variational autoencoder with free-bits KL balancing.
///
/// All weight tensors are row-major `[out × in]`.  The per-group head weights
/// are indexed by group; entry `0` of the prior heads is empty (the first
/// group's prior is the fixed standard normal).
#[derive(Debug, Clone)]
pub struct NVae {
    /// Configuration (dimensions and free-bits threshold).
    pub cfg: NVaeConfig,
    /// Encoder weight `[hidden × input]`.
    pub enc_w: Vec<f32>,
    /// Encoder bias `[hidden]`.
    pub enc_b: Vec<f32>,
    /// Posterior-mean head weights, one `[d_l × ctx_l]` matrix per group.
    pub q_mu_w: Vec<Vec<f32>>,
    /// Posterior-mean head biases, one `[d_l]` vector per group.
    pub q_mu_b: Vec<Vec<f32>>,
    /// Posterior log-σ head weights, one `[d_l × ctx_l]` matrix per group.
    pub q_ls_w: Vec<Vec<f32>>,
    /// Posterior log-σ head biases, one `[d_l]` vector per group.
    pub q_ls_b: Vec<Vec<f32>>,
    /// Prior-mean head weights, one `[d_l × d_{l-1}]` matrix per group (empty for group 0).
    pub p_mu_w: Vec<Vec<f32>>,
    /// Prior-mean head biases, one `[d_l]` vector per group (empty for group 0).
    pub p_mu_b: Vec<Vec<f32>>,
    /// Prior log-σ head weights, one `[d_l × d_{l-1}]` matrix per group (empty for group 0).
    pub p_ls_w: Vec<Vec<f32>>,
    /// Prior log-σ head biases, one `[d_l]` vector per group (empty for group 0).
    pub p_ls_b: Vec<Vec<f32>>,
    /// Decoder weight `[input × Σ d_l]`.
    pub dec_w: Vec<f32>,
    /// Decoder bias `[input]`.
    pub dec_b: Vec<f32>,
}

// ─── Free helpers ────────────────────────────────────────────────────────────

/// KL divergence between two diagonal-Gaussian coordinates (in nats):
/// `KL(N(μ_q, σ_q²) ‖ N(μ_p, σ_p²))` with `σ = exp(log_sigma)`.
///
/// ```text
/// = (log σ_p − log σ_q) + (σ_q² + (μ_q − μ_p)²) / (2 σ_p²) − ½.
/// ```
#[must_use]
pub fn kl_gaussian_diag(mu_q: f32, log_sigma_q: f32, mu_p: f32, log_sigma_p: f32) -> f32 {
    let var_q = (2.0 * log_sigma_q).exp();
    let var_p = (2.0 * log_sigma_p).exp();
    let dmu = mu_q - mu_p;
    (log_sigma_p - log_sigma_q) + (var_q + dmu * dmu) / (2.0 * var_p) - 0.5
}

/// Apply the free-bits floor to a group KL: `max(λ, kl)`.
#[must_use]
pub fn apply_free_bits(kl: f32, free_bits: f32) -> f32 {
    kl.max(free_bits)
}

/// Affine map `out = W·v + b` for a row-major `[out_dim × in_dim]` matrix.
fn affine(w: &[f32], b: &[f32], v: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|r| {
            let off = r * in_dim;
            let acc: f32 = w[off..off + in_dim]
                .iter()
                .zip(v.iter().take(in_dim))
                .map(|(&wi, &vi)| wi * vi)
                .sum();
            acc + b[r]
        })
        .collect()
}

/// Clamp every log-σ into `[-8, 8]` for numerical safety.
fn clamp_log_sigma(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x.clamp(-8.0, 8.0)).collect()
}

impl NVae {
    /// Construct a new `NVae` with small-variance weight initialisation.
    ///
    /// Mean heads / encoder / decoder use `N(0, 0.1 / sqrt(fan_in + fan_out))`;
    /// every log-σ head starts at zero (so the initial posterior and prior are
    /// both `≈ N(0, 1)`).
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] — `input_dim == 0`.
    /// - [`BayesError::InsufficientSamples`] — `hidden_dim == 0`.
    /// - [`BayesError::InvalidConfig`] — empty `group_dims`, a zero group
    ///   dimension, or a negative / non-finite `free_bits`.
    pub fn new(cfg: NVaeConfig, rng: &mut LcgRng) -> BayesResult<Self> {
        if cfg.input_dim == 0 {
            return Err(BayesError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.hidden_dim == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        if cfg.group_dims.is_empty() {
            return Err(BayesError::InvalidConfig(
                "NVae requires at least one latent group".into(),
            ));
        }
        if cfg.group_dims.contains(&0) {
            return Err(BayesError::InvalidConfig(
                "every NVae latent group must have dimension >= 1".into(),
            ));
        }
        if cfg.free_bits < 0.0 || !cfg.free_bits.is_finite() {
            return Err(BayesError::InvalidConfig(
                "free_bits must be finite and non-negative".into(),
            ));
        }

        let hidden = cfg.hidden_dim;
        let input = cfg.input_dim;
        let groups = &cfg.group_dims;
        let n_groups = groups.len();

        // Random matrix with N(0, scale) entries.
        let make_mat = |rows: usize, cols: usize, rng: &mut LcgRng| -> Vec<f32> {
            if rows * cols == 0 {
                return Vec::new();
            }
            let scale = 0.1_f32 / ((rows + cols) as f32).sqrt();
            let mut v = vec![0.0_f32; rows * cols];
            rng.fill_normal(&mut v);
            for x in v.iter_mut() {
                *x *= scale;
            }
            v
        };
        let zeros = |len: usize| vec![0.0_f32; len];

        let enc_w = make_mat(hidden, input, rng);
        let enc_b = zeros(hidden);

        let mut q_mu_w = Vec::with_capacity(n_groups);
        let mut q_mu_b = Vec::with_capacity(n_groups);
        let mut q_ls_w = Vec::with_capacity(n_groups);
        let mut q_ls_b = Vec::with_capacity(n_groups);
        let mut p_mu_w = Vec::with_capacity(n_groups);
        let mut p_mu_b = Vec::with_capacity(n_groups);
        let mut p_ls_w = Vec::with_capacity(n_groups);
        let mut p_ls_b = Vec::with_capacity(n_groups);

        for (l, &d) in groups.iter().enumerate() {
            let ctx_dim = hidden + if l == 0 { 0 } else { groups[l - 1] };
            // Posterior heads (mean random, log-σ zero-initialised).
            q_mu_w.push(make_mat(d, ctx_dim, rng));
            q_mu_b.push(zeros(d));
            q_ls_w.push(zeros(d * ctx_dim));
            q_ls_b.push(zeros(d));
            // Prior heads: only groups l >= 1 have a learned conditional prior.
            if l == 0 {
                p_mu_w.push(Vec::new());
                p_mu_b.push(Vec::new());
                p_ls_w.push(Vec::new());
                p_ls_b.push(Vec::new());
            } else {
                let prev = groups[l - 1];
                p_mu_w.push(make_mat(d, prev, rng));
                p_mu_b.push(zeros(d));
                p_ls_w.push(zeros(d * prev));
                p_ls_b.push(zeros(d));
            }
        }

        let latent_total: usize = groups.iter().sum();
        let dec_w = make_mat(input, latent_total, rng);
        let dec_b = zeros(input);

        Ok(Self {
            cfg,
            enc_w,
            enc_b,
            q_mu_w,
            q_mu_b,
            q_ls_w,
            q_ls_b,
            p_mu_w,
            p_mu_b,
            p_ls_w,
            p_ls_b,
            dec_w,
            dec_b,
        })
    }

    /// Total latent dimensionality `Σ_l d_l`.
    #[must_use]
    pub fn latent_total(&self) -> usize {
        self.cfg.group_dims.iter().sum()
    }

    /// Encoder feature `h = tanh(W_enc·x + b_enc)`.
    fn encode_feature(&self, x: &[f32]) -> Vec<f32> {
        let h = affine(
            &self.enc_w,
            &self.enc_b,
            x,
            self.cfg.hidden_dim,
            self.cfg.input_dim,
        );
        h.into_iter().map(|v| v.tanh()).collect()
    }

    /// Posterior parameters `(μ_q, log σ_q)` for group `l` given the context.
    fn posterior_params(&self, l: usize, ctx: &[f32], ctx_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.group_dims[l];
        let mu = affine(&self.q_mu_w[l], &self.q_mu_b[l], ctx, d, ctx_dim);
        let ls = clamp_log_sigma(affine(&self.q_ls_w[l], &self.q_ls_b[l], ctx, d, ctx_dim));
        (mu, ls)
    }

    /// Prior parameters `(μ_p, log σ_p)` for group `l` given the previous
    /// group's sample (`prev_z` is ignored for `l == 0`, whose prior is `N(0, I)`).
    fn prior_params(&self, l: usize, prev_z: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.group_dims[l];
        if l == 0 {
            (vec![0.0_f32; d], vec![0.0_f32; d])
        } else {
            let prev = self.cfg.group_dims[l - 1];
            let mu = affine(&self.p_mu_w[l], &self.p_mu_b[l], prev_z, d, prev);
            let ls = clamp_log_sigma(affine(&self.p_ls_w[l], &self.p_ls_b[l], prev_z, d, prev));
            (mu, ls)
        }
    }

    /// Decode the concatenated latents into the reconstruction mean.
    fn decode(&self, z_all: &[f32]) -> Vec<f32> {
        affine(
            &self.dec_w,
            &self.dec_b,
            z_all,
            self.cfg.input_dim,
            self.latent_total(),
        )
    }

    /// Reconstruction log-likelihood under a unit-variance Gaussian decoder.
    fn recon_log_likelihood(&self, x: &[f32], x_hat: &[f32]) -> f32 {
        let sse: f32 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(&xi, &xh)| (xi - xh) * (xi - xh))
            .sum();
        let log2pi = (2.0 * std::f32::consts::PI).ln();
        -0.5 * sse - 0.5 * self.cfg.input_dim as f32 * log2pi
    }

    /// Stochastic forward pass: sample every latent group top-down and decode.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `x.len() != input_dim`.
    pub fn forward(&self, x: &[f32], rng: &mut LcgRng) -> BayesResult<NVaeOutput> {
        if x.len() != self.cfg.input_dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.cfg.input_dim,
                got: x.len(),
            });
        }

        let h = self.encode_feature(x);
        let n_groups = self.cfg.group_dims.len();

        let mut z_groups: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
        let mut kl_raw: Vec<f32> = Vec::with_capacity(n_groups);
        let mut prev_z: Vec<f32> = Vec::new();

        for l in 0..n_groups {
            let ctx_dim = self.cfg.hidden_dim
                + if l == 0 {
                    0
                } else {
                    self.cfg.group_dims[l - 1]
                };
            let ctx: Vec<f32> = if l == 0 {
                h.clone()
            } else {
                let mut c = h.clone();
                c.extend_from_slice(&prev_z);
                c
            };

            let (q_mu, q_ls) = self.posterior_params(l, &ctx, ctx_dim);
            let (p_mu, p_ls) = self.prior_params(l, &prev_z);

            // Reparameterised sample z = μ_q + σ_q · ε.
            let z: Vec<f32> = q_mu
                .iter()
                .zip(q_ls.iter())
                .map(|(&mu, &ls)| {
                    let (eps, _) = rng.next_normal_pair();
                    mu + ls.exp() * eps
                })
                .collect();

            // Analytic per-group KL.
            let kl_l: f32 = q_mu
                .iter()
                .zip(q_ls.iter())
                .zip(p_mu.iter().zip(p_ls.iter()))
                .map(|((&mq, &lq), (&mp, &lp))| kl_gaussian_diag(mq, lq, mp, lp))
                .sum();

            kl_raw.push(kl_l);
            prev_z = z.clone();
            z_groups.push(z);
        }

        let z_all: Vec<f32> = z_groups.concat();
        let x_hat = self.decode(&z_all);
        let recon = self.recon_log_likelihood(x, &x_hat);

        let kl_balanced: Vec<f32> = kl_raw
            .iter()
            .map(|&k| apply_free_bits(k, self.cfg.free_bits))
            .collect();
        let elbo = recon - kl_balanced.iter().sum::<f32>();

        Ok(NVaeOutput {
            z: z_groups,
            x_hat,
            kl_per_group: kl_raw,
            kl_balanced_per_group: kl_balanced,
            recon_log_likelihood: recon,
            elbo,
        })
    }

    /// Deterministic reconstruction: use the posterior means (no sampling).
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `x.len() != input_dim`.
    pub fn reconstruct(&self, x: &[f32]) -> BayesResult<Vec<f32>> {
        if x.len() != self.cfg.input_dim {
            return Err(BayesError::DimensionMismatch {
                expected: self.cfg.input_dim,
                got: x.len(),
            });
        }
        let h = self.encode_feature(x);
        let n_groups = self.cfg.group_dims.len();
        let mut z_groups: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
        let mut prev_z: Vec<f32> = Vec::new();

        for l in 0..n_groups {
            let ctx_dim = self.cfg.hidden_dim
                + if l == 0 {
                    0
                } else {
                    self.cfg.group_dims[l - 1]
                };
            let ctx: Vec<f32> = if l == 0 {
                h.clone()
            } else {
                let mut c = h.clone();
                c.extend_from_slice(&prev_z);
                c
            };
            let (q_mu, _q_ls) = self.posterior_params(l, &ctx, ctx_dim);
            prev_z = q_mu.clone();
            z_groups.push(q_mu);
        }

        let z_all: Vec<f32> = z_groups.concat();
        Ok(self.decode(&z_all))
    }

    /// Convenience wrapper returning only the ELBO of a stochastic pass.
    ///
    /// # Errors
    /// [`BayesError::DimensionMismatch`] — `x.len() != input_dim`.
    pub fn elbo(&self, x: &[f32], rng: &mut LcgRng) -> BayesResult<f32> {
        Ok(self.forward(x, rng)?.elbo)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> NVaeConfig {
        NVaeConfig {
            input_dim: 6,
            group_dims: vec![3, 2],
            hidden_dim: 5,
            free_bits: 0.1,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Free helper analytics ────────────────────────────────────────────────

    #[test]
    fn kl_gaussian_diag_zero_when_equal() {
        // q == p ⇒ KL = 0.
        assert!(kl_gaussian_diag(0.0, 0.0, 0.0, 0.0).abs() < 1e-6);
        assert!(kl_gaussian_diag(1.5, -0.3, 1.5, -0.3).abs() < 1e-6);
    }

    #[test]
    fn kl_gaussian_diag_positive_when_different() {
        // Standard normal posterior vs shifted prior ⇒ KL = ½ Δμ² > 0.
        let kl = kl_gaussian_diag(2.0, 0.0, 0.0, 0.0);
        assert!((kl - 2.0).abs() < 1e-5, "expected ½·2² = 2, got {kl}");
        assert!(kl > 0.0);
    }

    #[test]
    fn apply_free_bits_floor() {
        assert!((apply_free_bits(0.0, 0.5) - 0.5).abs() < 1e-7);
        assert!((apply_free_bits(1.2, 0.5) - 1.2).abs() < 1e-7);
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_succeeds() {
        let mut rng = make_rng();
        assert!(NVae::new(small_cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_fails_zero_input_dim() {
        let mut rng = make_rng();
        let cfg = NVaeConfig {
            input_dim: 0,
            ..small_cfg()
        };
        assert!(matches!(
            NVae::new(cfg, &mut rng),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn new_fails_empty_groups() {
        let mut rng = make_rng();
        let cfg = NVaeConfig {
            group_dims: vec![],
            ..small_cfg()
        };
        assert!(matches!(
            NVae::new(cfg, &mut rng),
            Err(BayesError::InvalidConfig(_))
        ));
    }

    #[test]
    fn new_fails_negative_free_bits() {
        let mut rng = make_rng();
        let cfg = NVaeConfig {
            free_bits: -1.0,
            ..small_cfg()
        };
        assert!(matches!(
            NVae::new(cfg, &mut rng),
            Err(BayesError::InvalidConfig(_))
        ));
    }

    // ── (a) forward shapes & finiteness ──────────────────────────────────────

    #[test]
    fn forward_shapes_and_finite() {
        let mut rng = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng).expect("new");
        let x = vec![0.3_f32; small_cfg().input_dim];
        let out = vae.forward(&x, &mut rng).expect("forward");
        assert_eq!(out.z.len(), small_cfg().group_dims.len());
        for (l, zg) in out.z.iter().enumerate() {
            assert_eq!(zg.len(), small_cfg().group_dims[l]);
        }
        assert_eq!(out.x_hat.len(), small_cfg().input_dim);
        assert_eq!(out.kl_per_group.len(), small_cfg().group_dims.len());
        assert!(out.x_hat.iter().all(|v| v.is_finite()));
        assert!(out.recon_log_likelihood.is_finite());
    }

    // ── (b) KL >= 0 and balanced >= free_bits ────────────────────────────────

    #[test]
    fn kl_non_negative_and_balanced_at_least_free_bits() {
        let mut rng = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng).expect("new");
        let x = vec![0.5_f32; small_cfg().input_dim];
        let out = vae.forward(&x, &mut rng).expect("forward");
        for &kl in &out.kl_per_group {
            assert!(kl >= -1e-6, "raw KL must be >= 0, got {kl}");
        }
        for &kb in &out.kl_balanced_per_group {
            assert!(
                kb >= small_cfg().free_bits - 1e-6,
                "balanced KL must be >= free_bits, got {kb}"
            );
        }
    }

    // ── (c) ELBO finite ──────────────────────────────────────────────────────

    #[test]
    fn elbo_is_finite() {
        let mut rng = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng).expect("new");
        let x = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6];
        let elbo = vae.elbo(&x, &mut rng).expect("elbo");
        assert!(elbo.is_finite(), "ELBO must be finite, got {elbo}");
    }

    // ── (d) determinism / stochasticity ──────────────────────────────────────

    #[test]
    fn forward_deterministic_same_seed() {
        let mut rng_build = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng_build).expect("new");
        let x = vec![0.2_f32; small_cfg().input_dim];
        let a = vae.forward(&x, &mut LcgRng::new(5)).expect("fwd");
        let b = vae.forward(&x, &mut LcgRng::new(5)).expect("fwd");
        for (za, zb) in a.z.concat().iter().zip(b.z.concat().iter()) {
            assert!((za - zb).abs() < 1e-9);
        }
        assert!((a.elbo - b.elbo).abs() < 1e-5);
    }

    #[test]
    fn forward_stochastic_different_seed() {
        let mut rng_build = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng_build).expect("new");
        let x = vec![0.2_f32; small_cfg().input_dim];
        let a = vae.forward(&x, &mut LcgRng::new(1)).expect("fwd");
        let b = vae.forward(&x, &mut LcgRng::new(123_456)).expect("fwd");
        let za = a.z.concat();
        let zb = b.z.concat();
        assert!(za.iter().zip(zb.iter()).any(|(x, y)| (x - y).abs() > 1e-9));
    }

    // ── (e) free-bits clamps a collapsed group's KL ──────────────────────────

    #[test]
    fn free_bits_clamps_collapsed_group() {
        let mut rng = make_rng();
        let mut vae = NVae::new(small_cfg(), &mut rng).expect("new");
        // Force group 0's posterior to N(0, 1) == its prior ⇒ raw KL_0 = 0.
        for v in vae.q_mu_w[0].iter_mut() {
            *v = 0.0;
        }
        for v in vae.q_mu_b[0].iter_mut() {
            *v = 0.0;
        }
        for v in vae.q_ls_w[0].iter_mut() {
            *v = 0.0;
        }
        for v in vae.q_ls_b[0].iter_mut() {
            *v = 0.0;
        }
        let x = vec![0.7_f32; small_cfg().input_dim];
        let out = vae.forward(&x, &mut rng).expect("forward");
        assert!(out.kl_per_group[0].abs() < 1e-5, "raw KL_0 should be ~0");
        assert!(
            (out.kl_balanced_per_group[0] - small_cfg().free_bits).abs() < 1e-5,
            "balanced KL_0 should equal free_bits"
        );
    }

    // ── (f) dim mismatch ─────────────────────────────────────────────────────

    #[test]
    fn forward_dim_mismatch() {
        let mut rng = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng).expect("new");
        let x = vec![0.0_f32; small_cfg().input_dim + 1];
        assert!(matches!(
            vae.forward(&x, &mut rng),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn reconstruct_shape_and_deterministic() {
        let mut rng = make_rng();
        let vae = NVae::new(small_cfg(), &mut rng).expect("new");
        let x = vec![0.4_f32; small_cfg().input_dim];
        let r1 = vae.reconstruct(&x).expect("recon");
        let r2 = vae.reconstruct(&x).expect("recon");
        assert_eq!(r1.len(), small_cfg().input_dim);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-9, "reconstruct must be deterministic");
        }
    }

    #[test]
    fn single_group_works() {
        // L = 1: only the standard-normal prior group; still valid.
        let mut rng = make_rng();
        let cfg = NVaeConfig {
            input_dim: 4,
            group_dims: vec![3],
            hidden_dim: 4,
            free_bits: 0.0,
        };
        let vae = NVae::new(cfg, &mut rng).expect("new");
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let out = vae.forward(&x, &mut rng).expect("forward");
        assert_eq!(out.z.len(), 1);
        assert!(out.elbo.is_finite());
    }
}
