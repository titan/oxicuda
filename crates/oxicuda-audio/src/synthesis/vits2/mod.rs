//! VITS2 conditional-VAE-with-normalising-flow acoustic model.
//!
//! VITS ([Kim et al. 2021](https://arxiv.org/abs/2106.06103)) and VITS2
//! ([Kong et al. 2023](https://arxiv.org/abs/2307.16430)) cast text-to-speech as
//! a conditional variational auto-encoder whose prior is sharpened by a
//! normalising flow and whose alignment is learned by monotonic alignment
//! search, with a stochastic (flow-based) duration predictor for natural rhythm.
//!
//! ```text
//!  feature x ─► PosteriorEncoder q(z|x) ─► (m_q, logs_q) ─► z = m_q + ε·exp(logs_q)
//!                                                              │
//!  phonemes ─► PriorEncoder p(z|c) ─► (m_p, logs_p), h_text    │ flow f
//!                  │                         │                 ▼
//!                  │  StochasticDuration     │            z_p = f(z) ─► KL(q‖p)
//!                  │  Predictor (sample d)   │ MAS(align) ─► durations
//!                  ▼                         ▼
//!            inference: d ─► expand (m_p, logs_p) ─► z_p~N ─► f⁻¹ ─► decoder ─► mel
//! ```
//!
//! This module implements the **generator / flow / ELBO core** exactly and on
//! CPU: the flow is a true bijection with analytic log-determinant, the duration
//! predictor is a real conditional flow, the KL terms follow the VITS ELBO, and
//! both analysis (teacher) and inference (synthesis) passes return a correctly
//! shaped acoustic-feature (mel) tensor. The HiFi-GAN waveform decoder and the
//! adversarial (discriminator) training loop are **out of CPU scope** and are
//! recorded as deferred in `TODO.md`; the mel decoder here reuses the
//! feed-forward Transformer blocks of [`crate::synthesis::fastspeech2`].

pub mod common;
pub mod duration;
pub mod encoder;
pub mod flow;
pub mod spline;

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::synthesis::fastspeech2::{FftBlock, length_regulate};
use crate::synthesis::vits2::common::DenseLayer;

pub use duration::StochasticDurationPredictor;
pub use encoder::{
    PosteriorEncoder, PriorEncoder, flow_kl, gaussian_kl, monotonic_alignment_search,
    reparameterize,
};
pub use flow::{ActNorm, AffineCoupling, Vits2Flow};
pub use spline::{RationalQuadraticSpline, RqSplineCoupling};

// ─── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a [`Vits2`] acoustic model.
#[derive(Debug, Clone)]
pub struct Vits2Config {
    /// Phoneme-embedding / prior-hidden dimension `D`.
    pub embed_dim: usize,
    /// Number of self-attention heads (must divide `embed_dim` and `latent_dim`).
    pub n_heads: usize,
    /// Conv-FFN hidden channel count inside each Transformer block.
    pub conv_dim: usize,
    /// Transformer conv-FFN kernel size (odd).
    pub ffn_kernel: usize,
    /// Depth of the prior encoder and (separately) the mel decoder.
    pub depth: usize,
    /// Latent (flow channel) dimension `C` (`>= 2`, divisible by `n_heads`).
    pub latent_dim: usize,
    /// Posterior-encoder hidden channel count.
    pub post_hidden: usize,
    /// Posterior-encoder conv kernel size (odd).
    pub post_kernel: usize,
    /// Acoustic-feature dimension fed to the posterior encoder (e.g. spec bins).
    pub n_feat: usize,
    /// Output mel-spectrogram channel count.
    pub n_mels: usize,
    /// Hidden width of the prior flow's coupling conditioner.
    pub flow_hidden: usize,
    /// Number of Glow steps in the prior flow.
    pub flow_layers: usize,
    /// Stochastic-duration-predictor condition dimension.
    pub sdp_cond: usize,
    /// Stochastic-duration-predictor coupling hidden width.
    pub sdp_hidden: usize,
    /// Number of coupling layers in the stochastic duration predictor.
    pub sdp_flows: usize,
}

impl Vits2Config {
    /// Tiny preset suitable for unit tests.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            embed_dim: 16,
            n_heads: 2,
            conv_dim: 32,
            ffn_kernel: 9,
            depth: 2,
            latent_dim: 8,
            post_hidden: 24,
            post_kernel: 5,
            n_feat: 16,
            n_mels: 16,
            flow_hidden: 24,
            flow_layers: 3,
            sdp_cond: 6,
            sdp_hidden: 16,
            sdp_flows: 3,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `embed_dim` or `latent_dim` is `0`.
    /// - [`AudioError::InvalidNumHeads`] when `n_heads == 0`.
    /// - [`AudioError::HeadDimMismatch`] when `embed_dim` or `latent_dim` is not
    ///   divisible by `n_heads`.
    /// - [`AudioError::InvalidKernelSize`] when `ffn_kernel` or `post_kernel` is
    ///   `0` or even.
    /// - [`AudioError::InvalidNumMels`] when `n_mels == 0`.
    /// - [`AudioError::InvalidSequenceLength`] when `latent_dim < 2`.
    /// - [`AudioError::Internal`] when any remaining size is `0`.
    pub fn validate(&self) -> AudioResult<()> {
        if self.embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.latent_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.latent_dim < 2 {
            return Err(AudioError::InvalidSequenceLength(self.latent_dim));
        }
        if self.n_heads == 0 {
            return Err(AudioError::InvalidNumHeads(0));
        }
        if self.embed_dim % self.n_heads != 0 {
            return Err(AudioError::HeadDimMismatch {
                embed_dim: self.embed_dim,
                n_heads: self.n_heads,
            });
        }
        if self.latent_dim % self.n_heads != 0 {
            return Err(AudioError::HeadDimMismatch {
                embed_dim: self.latent_dim,
                n_heads: self.n_heads,
            });
        }
        if self.ffn_kernel == 0 || self.ffn_kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(self.ffn_kernel));
        }
        if self.post_kernel == 0 || self.post_kernel % 2 == 0 {
            return Err(AudioError::InvalidKernelSize(self.post_kernel));
        }
        if self.n_mels == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        for (v, name) in [
            (self.conv_dim, "conv_dim"),
            (self.depth, "depth"),
            (self.post_hidden, "post_hidden"),
            (self.n_feat, "n_feat"),
            (self.flow_hidden, "flow_hidden"),
            (self.flow_layers, "flow_layers"),
            (self.sdp_cond, "sdp_cond"),
            (self.sdp_hidden, "sdp_hidden"),
            (self.sdp_flows, "sdp_flows"),
        ] {
            if v == 0 {
                return Err(AudioError::Internal(format!("Vits2Config: {name} == 0")));
            }
        }
        Ok(())
    }
}

// ─── Analysis output ───────────────────────────────────────────────────────────

/// Outputs of the VITS2 analysis (teacher / training-style) pass.
#[derive(Debug, Clone)]
pub struct Vits2Analysis {
    /// Reconstructed mel/acoustic feature `[t_mel, n_mels]`.
    pub mel: Vec<f32>,
    /// Posterior latent sample `z` `[t_mel, latent_dim]`.
    pub z: Vec<f32>,
    /// Flow-transformed latent `z_p = f(z)` `[t_mel, latent_dim]`.
    pub z_p: Vec<f32>,
    /// Posterior mean `m_q` `[t_mel, latent_dim]`.
    pub m_q: Vec<f32>,
    /// Posterior log-sigma `logs_q` `[t_mel, latent_dim]`.
    pub logs_q: Vec<f32>,
    /// Per-phoneme hard-alignment durations from MAS (`sum == t_mel`).
    pub durations: Vec<usize>,
    /// Flow log-determinant accumulated in the forward pass.
    pub flow_logdet: f32,
    /// Closed-form diagonal-Gaussian KL `KL(q ‖ prior-base)` (`>= 0`).
    pub kl: f32,
    /// VITS Monte-Carlo KL term incorporating the flow (`z_p`, `flow_logdet`).
    pub kl_flow: f32,
    /// Stochastic-duration-predictor log-likelihood of the MAS durations.
    pub duration_log_likelihood: f32,
}

// ─── Model ─────────────────────────────────────────────────────────────────────

/// VITS2 acoustic model: posterior encoder, prior (text) encoder, prior flow,
/// stochastic duration predictor and a mel decoder.
///
/// All parameters are initialised deterministically from a seeded [`LcgRng`].
pub struct Vits2 {
    /// Prior / text encoder `p(z | c)`.
    pub prior: PriorEncoder,
    /// Posterior encoder `q(z | x)`.
    pub posterior: PosteriorEncoder,
    /// Prior normalising flow `f`.
    pub flow: Vits2Flow,
    /// Stochastic (flow-based) duration predictor.
    pub sdp: StochasticDurationPredictor,
    /// Mel decoder Transformer blocks (operating at `latent_dim`).
    pub decoder: Vec<FftBlock>,
    /// Latent → mel projection `[n_mels, latent_dim]` (internal helper type).
    mel_proj: DenseLayer,
    /// Configuration this model was built from.
    pub config: Vits2Config,
}

impl Vits2 {
    /// Construct a VITS2 model with deterministic initialisation.
    ///
    /// # Errors
    ///
    /// Returns any error from [`Vits2Config::validate`] or the sub-module
    /// constructors.
    pub fn new(config: Vits2Config, rng: &mut LcgRng) -> AudioResult<Self> {
        config.validate()?;
        let prior = PriorEncoder::new(
            config.embed_dim,
            config.n_heads,
            config.conv_dim,
            config.ffn_kernel,
            config.depth,
            config.latent_dim,
            rng,
        )?;
        let posterior = PosteriorEncoder::new(
            config.n_feat,
            config.post_hidden,
            config.latent_dim,
            config.post_kernel,
            rng,
        )?;
        let flow = Vits2Flow::new(
            config.latent_dim,
            config.flow_hidden,
            config.flow_layers,
            rng,
        )?;
        let sdp = StochasticDurationPredictor::new(
            config.embed_dim,
            config.sdp_cond,
            config.sdp_hidden,
            config.sdp_flows,
            rng,
        )?;
        let mut decoder = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            decoder.push(FftBlock::new(
                config.latent_dim,
                config.n_heads,
                config.conv_dim,
                config.ffn_kernel,
                rng,
            )?);
        }
        let mel_proj = DenseLayer::new(
            config.latent_dim,
            config.n_mels,
            1.0 / (config.latent_dim as f32).sqrt(),
            rng,
        );
        Ok(Self {
            prior,
            posterior,
            flow,
            sdp,
            decoder,
            mel_proj,
            config,
        })
    }

    /// Decode a latent `z` of `[t_mel, latent_dim]` to a mel `[t_mel, n_mels]`.
    fn decode(&self, z: &[f32], t_mel: usize) -> AudioResult<Vec<f32>> {
        let mut h = z.to_vec();
        for block in &self.decoder {
            h = block.forward(&h, t_mel)?;
        }
        Ok(self.mel_proj.forward(&h, t_mel))
    }

    /// Expand per-token prior statistics to frame level via length regulation.
    fn expand_stats(
        &self,
        m_p: &[f32],
        logs_p: &[f32],
        t_text: usize,
        durations: &[usize],
    ) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        let c = self.config.latent_dim;
        let m_exp = length_regulate(m_p, t_text, c, durations)?;
        let logs_exp = length_regulate(logs_p, t_text, c, durations)?;
        Ok((m_exp, logs_exp))
    }

    /// Analysis (teacher) pass over aligned phonemes and acoustic features.
    ///
    /// `phon` is `[t_text, embed_dim]`; `feat` is `[t_mel, n_feat]`. Encodes the
    /// posterior, samples `z`, applies the flow to obtain `z_p`, runs monotonic
    /// alignment search to recover hard durations, expands the prior to frame
    /// level, computes the ELBO KL terms and reconstructs the mel from `z`.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shapes.
    /// - [`AudioError::InvalidSequenceLength`] when `t_mel < t_text` (MAS).
    /// - Propagates errors from the sub-modules.
    pub fn analysis(
        &self,
        phon: &[f32],
        t_text: usize,
        feat: &[f32],
        t_mel: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Vits2Analysis> {
        let c = self.config.latent_dim;
        let (h_text, m_p_tok, logs_p_tok) = self.prior.forward(phon, t_text)?;
        let (m_q, logs_q) = self.posterior.forward(feat, t_mel)?;
        let z = reparameterize(&m_q, &logs_q, rng);
        let (z_p, flow_logdet) = self.flow.forward(&z, t_mel)?;

        let durations = monotonic_alignment_search(&m_p_tok, &logs_p_tok, &z_p, t_text, t_mel, c)?;
        let (m_p_exp, logs_p_exp) = self.expand_stats(&m_p_tok, &logs_p_tok, t_text, &durations)?;

        let kl = gaussian_kl(&m_q, &logs_q, &m_p_exp, &logs_p_exp)?;
        let kl_flow = flow_kl(&z_p, &logs_q, &m_p_exp, &logs_p_exp, flow_logdet)?;

        let durations_f: Vec<f32> = durations.iter().map(|&d| d as f32).collect();
        let duration_log_likelihood =
            self.sdp
                .log_likelihood(&h_text, &durations_f, t_text, rng)?;

        let mel = self.decode(&z, t_mel)?;

        Ok(Vits2Analysis {
            mel,
            z,
            z_p,
            m_q,
            logs_q,
            durations,
            flow_logdet,
            kl,
            kl_flow,
            duration_log_likelihood,
        })
    }

    /// Inference (synthesis) pass: phonemes → mel spectrogram.
    ///
    /// Encodes the text prior, samples per-phoneme durations from the stochastic
    /// duration predictor, expands the prior to frame level, samples `z_p` from
    /// the (expanded) prior Gaussian, maps it back through the flow inverse to
    /// the latent `z`, and decodes the mel.
    ///
    /// `noise_scale` scales both the duration noise and the prior sample;
    /// `length_scale` multiplies the predicted durations (`> 1` slower / longer).
    ///
    /// Returns the mel `[sum(durations), n_mels]` and the per-phoneme durations.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shape.
    /// - [`AudioError::Internal`] when `noise_scale` or `length_scale` is not a
    ///   finite, positive number.
    /// - Propagates errors from the sub-modules.
    pub fn inference(
        &self,
        phon: &[f32],
        t_text: usize,
        rng: &mut LcgRng,
        noise_scale: f32,
        length_scale: f32,
    ) -> AudioResult<(Vec<f32>, Vec<usize>)> {
        if !noise_scale.is_finite() || noise_scale <= 0.0 {
            return Err(AudioError::Internal(format!(
                "Vits2::inference: invalid noise_scale {noise_scale}"
            )));
        }
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(AudioError::Internal(format!(
                "Vits2::inference: invalid length_scale {length_scale}"
            )));
        }
        let c = self.config.latent_dim;
        let (h_text, m_p_tok, logs_p_tok) = self.prior.forward(phon, t_text)?;

        let dur_f = self.sdp.sample(&h_text, t_text, rng, noise_scale)?;
        let mut durations = Vec::with_capacity(t_text);
        for &d in &dur_f {
            let scaled = (d * length_scale).round();
            let v = if !scaled.is_finite() || scaled <= 0.0 {
                0
            } else if scaled >= 1000.0 {
                1000
            } else {
                scaled as usize
            };
            durations.push(v);
        }
        if durations.iter().all(|&d| d == 0) {
            for slot in durations.iter_mut() {
                *slot = 1;
            }
        }
        let t_mel: usize = durations.iter().sum();

        let (m_p_exp, logs_p_exp) = self.expand_stats(&m_p_tok, &logs_p_tok, t_text, &durations)?;

        // Sample z_p ~ N(m_p, (noise_scale·s_p)²) at frame level.
        let mut eps = vec![0.0_f32; t_mel * c];
        rng.fill_normal(&mut eps);
        let mut z_p = vec![0.0_f32; t_mel * c];
        for (i, ((&mp, &lp), &e)) in m_p_exp
            .iter()
            .zip(logs_p_exp.iter())
            .zip(eps.iter())
            .enumerate()
        {
            z_p[i] = mp + e * lp.exp() * noise_scale;
        }

        let z = self.flow.inverse(&z_p, t_mel)?;
        let mel = self.decode(&z, t_mel)?;
        Ok((mel, durations))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_tiny_is_valid() {
        assert!(Vits2Config::tiny().validate().is_ok());
    }

    #[test]
    fn config_rejects_bad_heads_and_latent() {
        let mut cfg = Vits2Config::tiny();
        cfg.n_heads = 3; // 16 % 3 != 0
        assert!(cfg.validate().is_err());

        let mut cfg = Vits2Config::tiny();
        cfg.latent_dim = 1; // < 2
        assert!(matches!(
            cfg.validate(),
            Err(AudioError::InvalidSequenceLength(1))
        ));

        let mut cfg = Vits2Config::tiny();
        cfg.latent_dim = 6; // 6 % 2 == 0 ok, but check non-divisible:
        cfg.n_heads = 4; // 16 % 4 == 0, 6 % 4 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn build_ok() {
        let mut rng = LcgRng::new(100);
        let model = Vits2::new(Vits2Config::tiny(), &mut rng);
        assert!(model.is_ok(), "build failed: {:?}", model.err());
    }

    #[test]
    fn analysis_shapes_finite_and_kl_nonneg() {
        // TEST 3 (analysis) + TEST 4 (kl >= 0).
        let cfg = Vits2Config::tiny();
        let mut rng = LcgRng::new(101);
        let model = Vits2::new(cfg.clone(), &mut rng).expect("new");
        let t_text = 5usize;
        let t_mel = 17usize;
        let mut phon = vec![0.0_f32; t_text * cfg.embed_dim];
        let mut feat = vec![0.0_f32; t_mel * cfg.n_feat];
        let mut data = LcgRng::new(202);
        data.fill_normal(&mut phon);
        data.fill_normal(&mut feat);

        let out = model
            .analysis(&phon, t_text, &feat, t_mel, &mut LcgRng::new(303))
            .expect("analysis");
        assert_eq!(out.mel.len(), t_mel * cfg.n_mels);
        assert_eq!(out.z.len(), t_mel * cfg.latent_dim);
        assert_eq!(out.z_p.len(), t_mel * cfg.latent_dim);
        assert_eq!(out.durations.len(), t_text);
        assert_eq!(out.durations.iter().sum::<usize>(), t_mel);
        assert!(out.mel.iter().all(|v| v.is_finite()));
        assert!(out.z_p.iter().all(|v| v.is_finite()));
        assert!(out.flow_logdet.is_finite());
        assert!(out.kl.is_finite() && out.kl_flow.is_finite());
        assert!(out.duration_log_likelihood.is_finite());
        // Closed-form VAE KL is non-negative.
        assert!(out.kl >= -1e-3, "analysis kl negative: {}", out.kl);
    }

    #[test]
    fn analysis_is_deterministic_under_seed() {
        // TEST 3: determinism.
        let cfg = Vits2Config::tiny();
        let model_a = Vits2::new(cfg.clone(), &mut LcgRng::new(7)).expect("a");
        let model_b = Vits2::new(cfg.clone(), &mut LcgRng::new(7)).expect("b");
        let t_text = 4usize;
        let t_mel = 13usize;
        let mut phon = vec![0.0_f32; t_text * cfg.embed_dim];
        let mut feat = vec![0.0_f32; t_mel * cfg.n_feat];
        let mut data = LcgRng::new(11);
        data.fill_normal(&mut phon);
        data.fill_normal(&mut feat);
        let a = model_a
            .analysis(&phon, t_text, &feat, t_mel, &mut LcgRng::new(5))
            .expect("a");
        let b = model_b
            .analysis(&phon, t_text, &feat, t_mel, &mut LcgRng::new(5))
            .expect("b");
        assert_eq!(a.mel, b.mel);
        assert_eq!(a.durations, b.durations);
        assert_eq!(a.z_p, b.z_p);
    }

    #[test]
    fn inference_shapes_finite_and_deterministic() {
        // TEST 3: inference pass shape / finiteness / determinism.
        let cfg = Vits2Config::tiny();
        let model_a = Vits2::new(cfg.clone(), &mut LcgRng::new(2024)).expect("a");
        let model_b = Vits2::new(cfg.clone(), &mut LcgRng::new(2024)).expect("b");
        let t_text = 6usize;
        let mut phon = vec![0.0_f32; t_text * cfg.embed_dim];
        LcgRng::new(555).fill_normal(&mut phon);

        let (mel_a, dur_a) = model_a
            .inference(&phon, t_text, &mut LcgRng::new(42), 0.8, 1.0)
            .expect("a");
        let (mel_b, dur_b) = model_b
            .inference(&phon, t_text, &mut LcgRng::new(42), 0.8, 1.0)
            .expect("b");

        assert_eq!(dur_a.len(), t_text);
        let total: usize = dur_a.iter().sum();
        assert!(total >= 1);
        assert_eq!(mel_a.len(), total * cfg.n_mels);
        assert!(mel_a.iter().all(|v| v.is_finite()));
        // Deterministic under identical seeds.
        assert_eq!(dur_a, dur_b);
        assert_eq!(mel_a, mel_b);
    }

    #[test]
    fn inference_length_scale_changes_total() {
        let cfg = Vits2Config::tiny();
        let model = Vits2::new(cfg.clone(), &mut LcgRng::new(9)).expect("m");
        let t_text = 8usize;
        let mut phon = vec![0.0_f32; t_text * cfg.embed_dim];
        LcgRng::new(90).fill_normal(&mut phon);
        let (_mel, dur_long) = model
            .inference(&phon, t_text, &mut LcgRng::new(1), 0.6, 2.0)
            .expect("long");
        // A 2× length scale must not produce fewer frames than 1× on the same
        // duration draw (monotone in length_scale by construction).
        let (_mel2, dur_unit) = model
            .inference(&phon, t_text, &mut LcgRng::new(1), 0.6, 1.0)
            .expect("unit");
        assert!(dur_long.iter().sum::<usize>() >= dur_unit.iter().sum::<usize>());
    }

    #[test]
    fn inference_rejects_bad_scales() {
        let cfg = Vits2Config::tiny();
        let model = Vits2::new(cfg.clone(), &mut LcgRng::new(3)).expect("m");
        let t_text = 3usize;
        let phon = vec![0.1_f32; t_text * cfg.embed_dim];
        assert!(
            model
                .inference(&phon, t_text, &mut LcgRng::new(1), 0.0, 1.0)
                .is_err()
        );
        assert!(
            model
                .inference(&phon, t_text, &mut LcgRng::new(1), 1.0, -1.0)
                .is_err()
        );
        assert!(
            model
                .inference(&phon, t_text, &mut LcgRng::new(1), f32::NAN, 1.0)
                .is_err()
        );
    }

    #[test]
    fn analysis_rejects_too_few_frames() {
        let cfg = Vits2Config::tiny();
        let model = Vits2::new(cfg.clone(), &mut LcgRng::new(4)).expect("m");
        let t_text = 5usize;
        let t_mel = 3usize; // < t_text → MAS infeasible
        let phon = vec![0.1_f32; t_text * cfg.embed_dim];
        let feat = vec![0.1_f32; t_mel * cfg.n_feat];
        assert!(
            model
                .analysis(&phon, t_text, &feat, t_mel, &mut LcgRng::new(1))
                .is_err()
        );
    }

    #[test]
    fn end_to_end_flow_consistency_in_inference() {
        // The flow used in inference must be the exact inverse used in analysis:
        // round-trip a frame-level latent through forward∘inverse.
        let cfg = Vits2Config::tiny();
        let model = Vits2::new(cfg.clone(), &mut LcgRng::new(8)).expect("m");
        let t_mel = 9usize;
        let mut z = vec![0.0_f32; t_mel * cfg.latent_dim];
        LcgRng::new(80).fill_normal(&mut z);
        let (z_p, _ld) = model.flow.forward(&z, t_mel).expect("fwd");
        let back = model.flow.inverse(&z_p, t_mel).expect("inv");
        let err = z
            .iter()
            .zip(back.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(err < 1e-4, "flow inconsistency {err}");
    }
}
