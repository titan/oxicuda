#![allow(clippy::needless_range_loop)]
//! Fully-connected Spiking Variational Autoencoder (FSVAE-style).
//!
//! Kamata, Mukuta & Harada 2022 (AAAI, "Fully Spiking Variational Autoencoder").
//! This module implements a simplified, fully-connected spiking VAE that keeps
//! the variational structure of a classic Gaussian VAE while running its
//! encoder and decoder as leaky-integrate-and-fire spiking networks.
//!
//! Pipeline:
//!
//! ```text
//! encoder:   x_spikes  ──SN──▶ h           (rate-coded over T steps)
//!            μ      = W_μ · r(h)            r(h) = mean spike rate of h
//!            logσ²  = W_σ · r(h)
//! latent:    z      = μ + σ · ε,           ε ~ N(0, 1),   σ = exp(½·logσ²)
//! decoder:   z ──W_dec──▶ SN ──W_out──▶ SN  ▶  x̂          (rate-coded output)
//! ```
//!
//! The encoder hidden layer is a spiking-linear layer (`current = W · x`, then a
//! LIF neuron emits `0/1` spikes); its spike train is averaged over the
//! `n_timesteps` to a real-valued rate vector `r(h) ∈ [0, 1]^hidden`. The latent
//! mean and log-variance are linear read-outs of `r(h)`. The reparameterisation
//! `z = μ + σ · ε` (Kingma & Welling 2014) makes the sampling differentiable in
//! `μ` and `logσ²`. The decoder injects the (real) latent `z` as a constant input
//! current across the `n_timesteps`, integrates it through two spiking layers,
//! and reports the **mean output spike rate** as the reconstruction `x̂ ∈ [0, 1]`.
//!
//! Regulariser. The analytic KL divergence between the approximate posterior
//! `q(z|x) = N(μ, σ²)` and the standard normal prior `p(z) = N(0, I)` is
//!
//! ```text
//! KL = −½ · Σ_k (1 + logσ²_k − μ_k² − σ²_k)
//! ```
//!
//! supplied by the free function [`crate::layer::spiking_vae::kl_divergence`].
//!
//! Layout. The encoder input is a flat row-major `[n_timesteps, in_dim]` spike
//! train of `0/1` values (same convention as the rate encoder). `μ`, `logσ²`,
//! `z` are length-`latent` real vectors and `x̂` is a length-`in_dim` real vector
//! of rates in `[0, 1]`.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, ResetMode, lif_step};

/// Configuration for a [`SpikingVae`].
#[derive(Debug, Clone, Copy)]
pub struct SpikingVaeConfig {
    /// Input / reconstruction dimension `in_dim`.
    pub in_dim: usize,
    /// Hidden width of the encoder and decoder spiking layers.
    pub hidden: usize,
    /// Latent dimension `latent`.
    pub latent: usize,
    /// Number of discrete timesteps `T` the encoder/decoder integrate over.
    pub n_timesteps: usize,
    /// Spike threshold shared by every LIF neuron.
    pub threshold: f32,
    /// Membrane time constant `τ_m` for every LIF neuron.
    pub tau: f32,
}

impl Default for SpikingVaeConfig {
    fn default() -> Self {
        Self {
            in_dim: 64,
            hidden: 32,
            latent: 8,
            n_timesteps: 8,
            threshold: 1.0,
            tau: 2.0,
        }
    }
}

/// Fully-connected spiking variational autoencoder.
///
/// Holds the encoder spiking layer (`enc1`), the two latent read-out matrices
/// (`enc_mu`, `enc_logvar`), and the two decoder spiking layers (`dec1`,
/// `dec_out`), together with the shared LIF configuration. All forward methods
/// rebuild their LIF membrane states, so the result depends only on the inputs
/// and the (fixed) weights — apart from the explicit stochastic
/// reparameterisation, which draws from the supplied RNG.
#[derive(Debug, Clone)]
pub struct SpikingVae {
    /// Encoder hidden weights `W_h`, row-major `[hidden, in_dim]`.
    pub enc1: Vec<f32>,
    /// Latent-mean read-out `W_μ`, row-major `[latent, hidden]`.
    pub enc_mu: Vec<f32>,
    /// Latent log-variance read-out `W_σ`, row-major `[latent, hidden]`.
    pub enc_logvar: Vec<f32>,
    /// Decoder hidden weights `W_dec`, row-major `[hidden, latent]`.
    pub dec1: Vec<f32>,
    /// Decoder output weights `W_out`, row-major `[in_dim, hidden]`.
    pub dec_out: Vec<f32>,
    /// LIF configuration shared by every spiking neuron.
    pub lif: LifConfig,
    /// Layer configuration.
    pub cfg: SpikingVaeConfig,
}

/// Build the LIF config used by the VAE neurons (Hard reset, `dt = 1`).
fn lif_cfg_from(cfg: &SpikingVaeConfig) -> LifConfig {
    LifConfig {
        tau_m: cfg.tau,
        v_th: cfg.threshold,
        v_rest: 0.0,
        dt: 1.0,
        reset: ResetMode::Hard,
    }
}

/// Validate the structural and numeric invariants of a [`SpikingVaeConfig`].
fn validate_cfg(cfg: &SpikingVaeConfig) -> SnnResult<()> {
    if cfg.in_dim == 0 {
        return Err(SnnError::BadDim { got: cfg.in_dim });
    }
    if cfg.hidden == 0 {
        return Err(SnnError::BadDim { got: cfg.hidden });
    }
    if cfg.latent == 0 {
        return Err(SnnError::BadDim { got: cfg.latent });
    }
    if cfg.n_timesteps == 0 {
        return Err(SnnError::BadTimesteps {
            got: cfg.n_timesteps,
        });
    }
    if !cfg.threshold.is_finite() || cfg.threshold <= 0.0 {
        return Err(SnnError::BadThreshold {
            v_th: cfg.threshold,
        });
    }
    if !cfg.tau.is_finite() || cfg.tau <= 0.0 {
        return Err(SnnError::BadTau { tau: cfg.tau });
    }
    Ok(())
}

/// Kaiming-normal weight matrix `[out_dim, in_dim]`, scaled by `sqrt(2/in_dim)`.
fn init_weights(out_dim: usize, in_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let scale = (2.0_f32 / in_dim.max(1) as f32).sqrt();
    let mut w = vec![0.0_f32; out_dim * in_dim];
    rng.fill_normal(&mut w);
    for v in &mut w {
        *v *= scale;
    }
    w
}

/// Dense matrix-vector product `out = W · x`, with `W` row-major `[out_dim, in_dim]`.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> SnnResult<Vec<f32>> {
    if x.len() != in_dim {
        return Err(SnnError::BadShape {
            expected: in_dim,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; out_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let row_off = i * in_dim;
        let row = w.get(row_off..row_off + in_dim).ok_or(SnnError::Internal {
            msg: "weight row out of range".into(),
        })?;
        let mut acc = 0.0_f32;
        for (&wij, &xj) in row.iter().zip(x.iter()) {
            acc += wij * xj;
        }
        *o = acc;
    }
    Ok(out)
}

impl SpikingVae {
    /// Allocate a new spiking VAE with Kaiming-normal weights.
    ///
    /// Returns an error when any dimension is zero or any hyper-parameter is out
    /// of range.
    pub fn new(cfg: SpikingVaeConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        validate_cfg(&cfg)?;
        let enc1 = init_weights(cfg.hidden, cfg.in_dim, rng);
        let enc_mu = init_weights(cfg.latent, cfg.hidden, rng);
        let enc_logvar = init_weights(cfg.latent, cfg.hidden, rng);
        let dec1 = init_weights(cfg.hidden, cfg.latent, rng);
        let dec_out = init_weights(cfg.in_dim, cfg.hidden, rng);
        Ok(Self {
            enc1,
            enc_mu,
            enc_logvar,
            dec1,
            dec_out,
            lif: lif_cfg_from(&cfg),
            cfg,
        })
    }

    /// Expected flat length of an encoder input spike train.
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.cfg.n_timesteps * self.cfg.in_dim
    }

    /// Encode an input spike train into the latent mean `μ` and log-variance
    /// `logσ²`.
    ///
    /// `input_spikes` is `[n_timesteps, in_dim]` flat row-major (`0/1`). The
    /// encoder hidden layer is integrated across the `n_timesteps`; its spike
    /// train is averaged to a hidden rate vector `r(h) ∈ [0, 1]^hidden`, from
    /// which `μ = W_μ · r(h)` and `logσ² = W_σ · r(h)` are read out.
    ///
    /// Returns [`SnnError::EmptyInput`] / [`SnnError::BadShape`] on a malformed
    /// input.
    pub fn encode(&self, input_spikes: &[f32]) -> SnnResult<(Vec<f32>, Vec<f32>)> {
        if input_spikes.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        let expected = self.input_len();
        if input_spikes.len() != expected {
            return Err(SnnError::BadShape {
                expected,
                got: input_spikes.len(),
            });
        }
        let d = self.cfg.in_dim;
        let h = self.cfg.hidden;
        let t = self.cfg.n_timesteps;

        let mut state = LifState::new(h);
        let mut spikes = vec![0.0_f32; h];
        let mut rate = vec![0.0_f32; h];
        for ti in 0..t {
            let off = ti * d;
            let x_row = input_spikes.get(off..off + d).ok_or(SnnError::Internal {
                msg: "encoder input row out of range".into(),
            })?;
            let current = matvec(&self.enc1, x_row, h, d)?;
            lif_step(&mut state, &current, &self.lif, &mut spikes)?;
            for (acc, &s) in rate.iter_mut().zip(spikes.iter()) {
                *acc += s;
            }
        }
        let inv_t = 1.0_f32 / t as f32;
        for v in &mut rate {
            *v *= inv_t;
        }
        let mu = matvec(&self.enc_mu, &rate, self.cfg.latent, h)?;
        let logvar = matvec(&self.enc_logvar, &rate, self.cfg.latent, h)?;
        Ok((mu, logvar))
    }

    /// Reparameterise: draw `z = μ + σ · ε` with `σ = exp(½·logσ²)` and
    /// `ε ~ N(0, 1)` from `rng`.
    ///
    /// Deterministic for a fixed RNG seed. As `logσ² → −∞` (`σ → 0`) the result
    /// converges to `μ`. Returns [`SnnError::IncompatibleLength`] if `mu` and
    /// `logvar` differ in length.
    pub fn reparameterize(
        &self,
        mu: &[f32],
        logvar: &[f32],
        rng: &mut LcgRng,
    ) -> SnnResult<Vec<f32>> {
        if mu.len() != logvar.len() {
            return Err(SnnError::IncompatibleLength {
                a: mu.len(),
                b: logvar.len(),
            });
        }
        let mut z = vec![0.0_f32; mu.len()];
        for (zi, (&m, &lv)) in z.iter_mut().zip(mu.iter().zip(logvar.iter())) {
            let sigma = (0.5_f32 * lv).exp();
            let eps = rng.next_normal();
            *zi = m + sigma * eps;
        }
        Ok(z)
    }

    /// Decode a latent vector `z` into a rate-coded reconstruction `x̂ ∈ [0, 1]`.
    ///
    /// `z` (length `latent`) is injected as a constant input current across the
    /// `n_timesteps`, integrated through the two decoder spiking layers, and the
    /// mean output spike rate is returned. Returns [`SnnError::BadShape`] if
    /// `z.len() != latent`.
    pub fn decode(&self, z: &[f32]) -> SnnResult<Vec<f32>> {
        if z.len() != self.cfg.latent {
            return Err(SnnError::BadShape {
                expected: self.cfg.latent,
                got: z.len(),
            });
        }
        let d = self.cfg.in_dim;
        let h = self.cfg.hidden;
        let t = self.cfg.n_timesteps;

        // Constant decoder hidden current `W_dec · z` injected every step.
        let dec_current = matvec(&self.dec1, z, h, self.cfg.latent)?;

        let mut state_h = LifState::new(h);
        let mut state_o = LifState::new(d);
        let mut hid_spikes = vec![0.0_f32; h];
        let mut out_spikes = vec![0.0_f32; d];
        let mut recon = vec![0.0_f32; d];
        for _ in 0..t {
            lif_step(&mut state_h, &dec_current, &self.lif, &mut hid_spikes)?;
            let out_current = matvec(&self.dec_out, &hid_spikes, d, h)?;
            lif_step(&mut state_o, &out_current, &self.lif, &mut out_spikes)?;
            for (acc, &s) in recon.iter_mut().zip(out_spikes.iter()) {
                *acc += s;
            }
        }
        let inv_t = 1.0_f32 / t as f32;
        for v in &mut recon {
            *v *= inv_t;
        }
        Ok(recon)
    }

    /// Full forward pass: encode, reparameterise, decode.
    ///
    /// Returns `(x̂, μ, logσ²)` so the caller can form both the reconstruction
    /// loss (on `x̂`) and the KL regulariser (on `μ`, `logσ²` via
    /// [`crate::layer::spiking_vae::kl_divergence`]).
    pub fn forward(
        &self,
        input: &[f32],
        rng: &mut LcgRng,
    ) -> SnnResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (mu, logvar) = self.encode(input)?;
        let z = self.reparameterize(&mu, &logvar, rng)?;
        let recon = self.decode(&z)?;
        Ok((recon, mu, logvar))
    }
}

/// Analytic Gaussian KL divergence regulariser of a VAE,
/// `KL[N(μ, σ²) ‖ N(0, I)] = −½ · Σ_k (1 + logσ²_k − μ_k² − σ²_k)`
/// with `σ²_k = exp(logσ²_k)`.
///
/// This is always non-negative and equals zero exactly when `μ = 0` and
/// `logσ² = 0` (the posterior matches the standard-normal prior). `mu` and
/// `logvar` must have the same length.
#[must_use]
pub fn kl_divergence(mu: &[f32], logvar: &[f32]) -> f32 {
    let n = mu.len().min(logvar.len());
    let mut acc = 0.0_f32;
    for k in 0..n {
        let m = mu[k];
        let lv = logvar[k];
        let var = lv.exp();
        acc += 1.0 + lv - m * m - var;
    }
    -0.5 * acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> SpikingVaeConfig {
        SpikingVaeConfig {
            in_dim: 6,
            hidden: 5,
            latent: 3,
            n_timesteps: 8,
            threshold: 0.5,
            tau: 2.0,
        }
    }

    fn random_input(cfg: &SpikingVaeConfig, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let len = cfg.n_timesteps * cfg.in_dim;
        let mut v = vec![0.0_f32; len];
        for x in &mut v {
            *x = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
        }
        v
    }

    #[test]
    fn encode_shapes_correct() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(1);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let input = random_input(&cfg, 7);
        let (mu, logvar) = vae.encode(&input).expect("encode");
        assert_eq!(mu.len(), cfg.latent);
        assert_eq!(logvar.len(), cfg.latent);
    }

    #[test]
    fn decode_shape_correct_and_rates_in_unit() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(2);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let z = vec![0.5_f32; cfg.latent];
        let recon = vae.decode(&z).expect("decode");
        assert_eq!(recon.len(), cfg.in_dim);
        for &r in &recon {
            assert!((0.0..=1.0).contains(&r), "rate out of [0,1]: {r}");
        }
    }

    #[test]
    fn kl_zero_for_standard_normal_params() {
        let mu = vec![0.0_f32; 4];
        let logvar = vec![0.0_f32; 4];
        let kl = kl_divergence(&mu, &logvar);
        assert!(kl.abs() < 1e-6, "KL(0,0) should be 0, got {kl}");
    }

    #[test]
    fn kl_non_negative_for_random_params() {
        let mut rng = LcgRng::new(3);
        for _ in 0..50 {
            let mut mu = vec![0.0_f32; 5];
            let mut logvar = vec![0.0_f32; 5];
            for m in &mut mu {
                *m = rng.next_normal();
            }
            for lv in &mut logvar {
                // Keep logvar in a sane band so exp() does not overflow f32.
                *lv = rng.next_normal();
            }
            let kl = kl_divergence(&mu, &logvar);
            assert!(kl >= -1e-4, "KL should be >= 0, got {kl}");
        }
    }

    #[test]
    fn kl_known_value() {
        // Single dim, mu=1, logvar=0 (sigma^2=1):
        // KL = -0.5*(1 + 0 - 1 - 1) = -0.5*(-1) = 0.5.
        let kl = kl_divergence(&[1.0], &[0.0]);
        assert!((kl - 0.5).abs() < 1e-6, "got {kl}");
    }

    #[test]
    fn reparameterize_deterministic_given_seed() {
        let cfg = small_cfg();
        let mut rng_ctor = LcgRng::new(4);
        let vae = SpikingVae::new(cfg, &mut rng_ctor).expect("ctor");
        let mu = vec![0.2_f32, -0.5, 1.0];
        let logvar = vec![0.0_f32, 0.1, -0.2];
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let z1 = vae.reparameterize(&mu, &logvar, &mut r1).expect("z1");
        let z2 = vae.reparameterize(&mu, &logvar, &mut r2).expect("z2");
        assert_eq!(z1, z2, "reparameterize must be deterministic per seed");
    }

    #[test]
    fn reparameterize_equals_mu_when_logvar_minus_inf() {
        let cfg = small_cfg();
        let mut rng_ctor = LcgRng::new(5);
        let vae = SpikingVae::new(cfg, &mut rng_ctor).expect("ctor");
        let mu = vec![0.2_f32, -0.5, 1.0];
        // logvar -> -inf  => sigma = exp(0.5 * -inf) = 0 => z = mu exactly.
        let logvar = vec![f32::NEG_INFINITY; 3];
        let mut rng = LcgRng::new(7);
        let z = vae.reparameterize(&mu, &logvar, &mut rng).expect("z");
        for (&zi, &mi) in z.iter().zip(mu.iter()) {
            assert!((zi - mi).abs() < 1e-6, "z={zi} should equal mu={mi}");
        }
    }

    #[test]
    fn reparameterize_very_negative_logvar_close_to_mu() {
        let cfg = small_cfg();
        let mut rng_ctor = LcgRng::new(6);
        let vae = SpikingVae::new(cfg, &mut rng_ctor).expect("ctor");
        let mu = vec![0.3_f32, -0.7, 2.0];
        let logvar = vec![-60.0_f32; 3]; // sigma ~ exp(-30) ~ 1e-13
        let mut rng = LcgRng::new(11);
        let z = vae.reparameterize(&mu, &logvar, &mut rng).expect("z");
        for (&zi, &mi) in z.iter().zip(mu.iter()) {
            assert!((zi - mi).abs() < 1e-4, "z={zi} vs mu={mi}");
        }
    }

    #[test]
    fn forward_runs_end_to_end() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(8);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let input = random_input(&cfg, 13);
        let mut sample_rng = LcgRng::new(99);
        let (recon, mu, logvar) = vae.forward(&input, &mut sample_rng).expect("forward");
        assert_eq!(recon.len(), cfg.in_dim);
        assert_eq!(mu.len(), cfg.latent);
        assert_eq!(logvar.len(), cfg.latent);
        for &r in &recon {
            assert!((0.0..=1.0).contains(&r), "recon rate out of [0,1]: {r}");
        }
    }

    #[test]
    fn encode_is_deterministic() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(9);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let input = random_input(&cfg, 17);
        let (mu1, lv1) = vae.encode(&input).expect("e1");
        let (mu2, lv2) = vae.encode(&input).expect("e2");
        assert_eq!(mu1, mu2);
        assert_eq!(lv1, lv2);
    }

    #[test]
    fn decode_is_deterministic() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(10);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let z = vec![0.4_f32, -0.3, 0.9];
        let d1 = vae.decode(&z).expect("d1");
        let d2 = vae.decode(&z).expect("d2");
        assert_eq!(d1, d2);
    }

    #[test]
    fn empty_encode_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(11);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        assert!(matches!(vae.encode(&[]), Err(SnnError::EmptyInput)));
    }

    #[test]
    fn encode_bad_length_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(12);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let bad = vec![0.0_f32; vae.input_len() + 1];
        assert!(matches!(vae.encode(&bad), Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn decode_bad_length_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(13);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let bad = vec![0.0_f32; cfg.latent + 2];
        assert!(matches!(vae.decode(&bad), Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn reparameterize_length_mismatch_errors() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(14);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let mut srng = LcgRng::new(1);
        let err = vae.reparameterize(&[0.0, 0.0], &[0.0], &mut srng);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn larger_sigma_widens_samples() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(15);
        let vae = SpikingVae::new(cfg, &mut rng).expect("ctor");
        let mu = vec![0.0_f32; 1];
        // Draw many samples at small vs large logvar; the spread must grow.
        let spread = |lv: f32, seed: u64| -> f32 {
            let mut r = LcgRng::new(seed);
            let logvar = vec![lv; 1];
            let mut sumsq = 0.0_f32;
            let n = 2000;
            for _ in 0..n {
                let z = vae.reparameterize(&mu, &logvar, &mut r).expect("z");
                sumsq += z[0] * z[0];
            }
            sumsq / n as f32
        };
        let var_small = spread(-4.0, 123);
        let var_large = spread(2.0, 123);
        assert!(
            var_large > var_small,
            "larger logvar should widen samples: small={var_small} large={var_large}"
        );
    }

    #[test]
    fn bad_config_dims_error() {
        let mut rng = LcgRng::new(16);
        assert!(matches!(
            SpikingVae::new(
                SpikingVaeConfig {
                    in_dim: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            SpikingVae::new(
                SpikingVaeConfig {
                    latent: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            SpikingVae::new(
                SpikingVaeConfig {
                    n_timesteps: 0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadTimesteps { .. })
        ));
        assert!(matches!(
            SpikingVae::new(
                SpikingVaeConfig {
                    threshold: 0.0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadThreshold { .. })
        ));
        assert!(matches!(
            SpikingVae::new(
                SpikingVaeConfig {
                    tau: -1.0,
                    ..small_cfg()
                },
                &mut rng
            ),
            Err(SnnError::BadTau { .. })
        ));
    }
}
