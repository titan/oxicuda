//! AudioCLIP-style three-way (audio · image · text) contrastive alignment.
//!
//! Guzhov et al. "AudioCLIP: Extending CLIP to Image, Text and Audio." (2021).
//!
//! AudioCLIP extends the bimodal CLIP objective to three modalities by training
//! all three pairwise InfoNCE terms jointly, each with its own learnable
//! logit-scale (temperature). Where [`crate::alignment::contrastive::imagebind_loss`]
//! averages three CLIP losses at a *single shared* temperature, AudioCLIP keeps
//! three independent logit-scales — one per modality pair — exactly mirroring the
//! reference layout in which `logit_scale_ai`, `logit_scale_at`, `logit_scale_it`
//! are separate parameters.
//!
//! Given L2-normalised features for audio `A`, image `I` and text `T` (all
//! `[batch × dim]`), the loss is
//!
//! ```text
//! L = w_ai · clip(A, I; τ_ai) + w_at · clip(A, T; τ_at) + w_it · clip(I, T; τ_it)
//! ```
//!
//! where each `clip(·, ·; τ)` is the symmetric bidirectional InfoNCE already
//! implemented in [`crate::alignment::contrastive::clip_loss`] and the `w_*`
//! are non-negative pair weights (default `1/3` each so the loss reduces to
//! `imagebind_loss` when all three temperatures coincide).

use crate::alignment::contrastive::clip_loss;
use crate::error::{MmResult, MultiModalError};

/// Per-pair temperatures and weights for the AudioCLIP objective.
#[derive(Debug, Clone)]
pub struct AudioClipConfig {
    /// Feature dimension shared by all three modalities.
    pub dim: usize,
    /// Temperature for the audio↔image term.
    pub tau_ai: f32,
    /// Temperature for the audio↔text term.
    pub tau_at: f32,
    /// Temperature for the image↔text term.
    pub tau_it: f32,
    /// Non-negative weight of the audio↔image term.
    pub w_ai: f32,
    /// Non-negative weight of the audio↔text term.
    pub w_at: f32,
    /// Non-negative weight of the image↔text term.
    pub w_it: f32,
}

impl AudioClipConfig {
    /// Symmetric preset: all temperatures `temperature`, equal `1/3` weights.
    ///
    /// With this preset the loss equals
    /// [`crate::alignment::contrastive::imagebind_loss`] for the same batch.
    #[must_use]
    pub fn symmetric(dim: usize, temperature: f32) -> Self {
        Self {
            dim,
            tau_ai: temperature,
            tau_at: temperature,
            tau_it: temperature,
            w_ai: 1.0 / 3.0,
            w_at: 1.0 / 3.0,
            w_it: 1.0 / 3.0,
        }
    }

    /// Validate dimensions, temperatures and weights.
    ///
    /// # Errors
    /// - [`MultiModalError::InvalidFeatureDim`] when `dim == 0`.
    /// - [`MultiModalError::InvalidTemperature`] when any temperature is not a
    ///   finite positive number.
    /// - [`MultiModalError::Internal`] when any weight is negative or non-finite,
    ///   or when all three weights are zero.
    pub fn validate(&self) -> MmResult<()> {
        if self.dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        for &t in &[self.tau_ai, self.tau_at, self.tau_it] {
            if t <= 0.0 || !t.is_finite() {
                return Err(MultiModalError::InvalidTemperature { temp: t });
            }
        }
        for &w in &[self.w_ai, self.w_at, self.w_it] {
            if w < 0.0 || !w.is_finite() {
                return Err(MultiModalError::Internal(
                    "AudioCLIP pair weights must be non-negative and finite".to_string(),
                ));
            }
        }
        if self.w_ai + self.w_at + self.w_it <= 0.0 {
            return Err(MultiModalError::Internal(
                "AudioCLIP pair weights must not all be zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decomposition of the AudioCLIP loss into its three pairwise terms.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClipLoss {
    /// Weighted total loss.
    pub total: f32,
    /// Unweighted audio↔image InfoNCE.
    pub ai: f32,
    /// Unweighted audio↔text InfoNCE.
    pub at: f32,
    /// Unweighted image↔text InfoNCE.
    pub it: f32,
}

/// Compute the AudioCLIP three-way contrastive loss.
///
/// `audio`, `image`, `text` are each `[batch × dim]` row-major. They are
/// L2-normalised internally (inside [`clip_loss`]), so callers may pass raw
/// encoder outputs.
///
/// # Errors
/// - [`MultiModalError::InvalidBatchSize`] when `batch == 0`.
/// - Any error surfaced by [`AudioClipConfig::validate`] or [`clip_loss`].
pub fn audio_clip_loss(
    audio: &[f32],
    image: &[f32],
    text: &[f32],
    batch: usize,
    cfg: &AudioClipConfig,
) -> MmResult<AudioClipLoss> {
    cfg.validate()?;
    if batch == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    let d = cfg.dim;
    let ai = clip_loss(audio, image, batch, d, cfg.tau_ai)?;
    let at = clip_loss(audio, text, batch, d, cfg.tau_at)?;
    let it = clip_loss(image, text, batch, d, cfg.tau_it)?;
    let total = cfg.w_ai * ai + cfg.w_at * at + cfg.w_it * it;
    if !total.is_finite() {
        return Err(MultiModalError::NanEncountered {
            location: "audio_clip_loss",
        });
    }
    Ok(AudioClipLoss { total, ai, at, it })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::contrastive::imagebind_loss;
    use crate::handle::LcgRng;

    fn random_feats(batch: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; batch * dim];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn symmetric_preset_matches_imagebind() {
        // With equal 1/3 weights and a single shared temperature, AudioCLIP must
        // reproduce the ImageBind triple loss exactly (same three clip_loss calls,
        // same averaging).
        let (batch, dim) = (6, 16);
        let a = random_feats(batch, dim, 1);
        let i = random_feats(batch, dim, 2);
        let t = random_feats(batch, dim, 3);
        let cfg = AudioClipConfig::symmetric(dim, 0.07);
        let loss = audio_clip_loss(&a, &i, &t, batch, &cfg).expect("loss should succeed");
        let ib = imagebind_loss(&a, &i, &t, batch, dim, 0.07).expect("imagebind should succeed");
        assert!(
            (loss.total - ib).abs() < 1e-5,
            "audioclip {} vs imagebind {}",
            loss.total,
            ib
        );
    }

    #[test]
    fn total_is_weighted_sum_of_terms() {
        let (batch, dim) = (4, 8);
        let a = random_feats(batch, dim, 10);
        let i = random_feats(batch, dim, 11);
        let t = random_feats(batch, dim, 12);
        let cfg = AudioClipConfig {
            dim,
            tau_ai: 0.05,
            tau_at: 0.1,
            tau_it: 0.2,
            w_ai: 0.5,
            w_at: 0.3,
            w_it: 0.2,
        };
        let l = audio_clip_loss(&a, &i, &t, batch, &cfg).expect("loss should succeed");
        let recomputed = cfg.w_ai * l.ai + cfg.w_at * l.at + cfg.w_it * l.it;
        assert!((l.total - recomputed).abs() < 1e-5);
    }

    #[test]
    fn aligned_triple_lower_than_shuffled() {
        // Build a batch where all three modalities are identical (perfect tri-modal
        // alignment) and compare against a version where the text rows are rotated
        // (so the diagonal pairing is broken). Misalignment must raise the loss.
        let (batch, dim) = (5, 12);
        let base = random_feats(batch, dim, 99);
        let cfg = AudioClipConfig::symmetric(dim, 0.07);

        let aligned =
            audio_clip_loss(&base, &base, &base, batch, &cfg).expect("aligned should succeed");

        // Rotate the text rows by one position.
        let mut shuffled_text = vec![0.0_f32; batch * dim];
        for b in 0..batch {
            let src = (b + 1) % batch;
            shuffled_text[b * dim..(b + 1) * dim]
                .copy_from_slice(&base[src * dim..(src + 1) * dim]);
        }
        let shuffled = audio_clip_loss(&base, &base, &shuffled_text, batch, &cfg)
            .expect("shuffled should succeed");

        assert!(
            shuffled.total > aligned.total,
            "aligned {} should be < shuffled {}",
            aligned.total,
            shuffled.total
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let (batch, dim) = (4, 8);
        let a = random_feats(batch, dim, 7);
        let i = random_feats(batch, dim, 8);
        let t = random_feats(batch, dim, 9);
        let cfg = AudioClipConfig::symmetric(dim, 0.07);
        let l1 = audio_clip_loss(&a, &i, &t, batch, &cfg).expect("loss");
        let l2 = audio_clip_loss(&a, &i, &t, batch, &cfg).expect("loss");
        assert_eq!(l1, l2);
    }

    #[test]
    fn zero_batch_errors() {
        let cfg = AudioClipConfig::symmetric(8, 0.07);
        let err = audio_clip_loss(&[], &[], &[], 0, &cfg).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidBatchSize));
    }

    #[test]
    fn invalid_temperature_errors() {
        let cfg = AudioClipConfig {
            dim: 8,
            tau_ai: 0.0,
            tau_at: 0.1,
            tau_it: 0.1,
            w_ai: 1.0,
            w_at: 1.0,
            w_it: 1.0,
        };
        let f = vec![0.0_f32; 2 * 8];
        let err = audio_clip_loss(&f, &f, &f, 2, &cfg).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidTemperature { .. }));
    }

    #[test]
    fn negative_weight_errors() {
        let cfg = AudioClipConfig {
            dim: 8,
            tau_ai: 0.1,
            tau_at: 0.1,
            tau_it: 0.1,
            w_ai: -1.0,
            w_at: 1.0,
            w_it: 1.0,
        };
        assert!(matches!(cfg.validate(), Err(MultiModalError::Internal(_))));
    }

    #[test]
    fn all_zero_weights_error() {
        let cfg = AudioClipConfig {
            dim: 8,
            tau_ai: 0.1,
            tau_at: 0.1,
            tau_it: 0.1,
            w_ai: 0.0,
            w_at: 0.0,
            w_it: 0.0,
        };
        assert!(matches!(cfg.validate(), Err(MultiModalError::Internal(_))));
    }

    #[test]
    fn disabling_a_pair_drops_its_contribution() {
        // Zero-weighting the image↔text term must make the total independent of
        // whatever the text/image pairing looks like in that channel.
        let (batch, dim) = (4, 8);
        let a = random_feats(batch, dim, 21);
        let i = random_feats(batch, dim, 22);
        let t = random_feats(batch, dim, 23);
        let cfg = AudioClipConfig {
            dim,
            tau_ai: 0.07,
            tau_at: 0.07,
            tau_it: 0.07,
            w_ai: 0.5,
            w_at: 0.5,
            w_it: 0.0,
        };
        let l = audio_clip_loss(&a, &i, &t, batch, &cfg).expect("loss");
        let expected = 0.5 * l.ai + 0.5 * l.at;
        assert!((l.total - expected).abs() < 1e-5);
    }
}
