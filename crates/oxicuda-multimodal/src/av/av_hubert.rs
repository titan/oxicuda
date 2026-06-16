//! AV-HuBERT — audio-visual self-supervised speech (Shi et al., 2022).
//!
//! AV-HuBERT learns audio-visual speech representations by predicting
//! frame-level k-means cluster assignments (HuBERT-style pseudo-labels) at
//! *masked* positions, from a sequence that fuses an acoustic stream and a
//! lip-region video stream. The architecture implemented here mirrors the
//! reference design:
//!
//! 1. **Modality-specific frontends.** A small linear acoustic frontend maps
//!    per-frame mel feature vectors to the model width; a linear video frontend
//!    maps per-frame flattened lip-ROI features to the same width. (The paper
//!    uses a stack of convolutions / a ResNet for the video frontend and a
//!    linear projection for audio; a per-frame linear stack is the standard
//!    lightweight stand-in and keeps the feature semantics identical.)
//! 2. **Modality fusion.** The two per-frame feature streams are *concatenated*
//!    along the channel axis and passed through a fusion linear back to the
//!    model width — exactly the concatenation fusion used by AV-HuBERT (cross
//!    attention is an alternative; concat is the canonical default).
//! 3. **Transformer encoder.** A pre-norm Transformer encoder (reusing the
//!    crate's [`CrossAttention`], [`LayerNorm`] and [`FeedForward`]) contextualises
//!    the fused sequence.
//! 4. **Masked-prediction head.** A linear projection produces logits over `k`
//!    discrete cluster units; the masked-prediction cross-entropy is computed
//!    **only at masked positions**.
//! 5. **Modality dropout.** During a forward pass the *entire* audio or video
//!    stream can be zeroed (modality dropout), forcing the model to use the
//!    surviving modality. Audio-only and video-only inference are the limiting
//!    cases.
//!
//! All tensors are flat row-major `Vec<f32>`; randomness uses the crate's
//! deterministic [`LcgRng`].

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Modality-dropout selection ─────────────────────────────────────────────────

/// Which streams survive a forward pass.
///
/// Modality dropout zeroes a whole stream's *pre-fusion* contribution. The two
/// single-modality variants ([`ModalityDrop::DropAudio`], [`ModalityDrop::DropVideo`])
/// are also exactly what an audio-only / video-only inference path uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityDrop {
    /// Keep both audio and video.
    Both,
    /// Zero the audio stream (video-only forward).
    DropAudio,
    /// Zero the video stream (audio-only forward).
    DropVideo,
}

impl ModalityDrop {
    /// `true` if the audio stream survives.
    #[must_use]
    pub fn audio_kept(self) -> bool {
        !matches!(self, ModalityDrop::DropAudio)
    }

    /// `true` if the video stream survives.
    #[must_use]
    pub fn video_kept(self) -> bool {
        !matches!(self, ModalityDrop::DropVideo)
    }

    /// Sample a dropout decision from the crate RNG.
    ///
    /// With probability `p_drop` *one* modality is dropped; when a drop happens
    /// audio vs. video is chosen with probability `p_audio` of dropping audio.
    /// This matches the AV-HuBERT modality-dropout schedule (a single stream is
    /// dropped, never both, so the model always sees some signal).
    #[must_use]
    pub fn sample(rng: &mut LcgRng, p_drop: f32, p_audio: f32) -> Self {
        // next_f32() spans [0, 0.5) for this crate's LcgRng (top 31 bits / 2^32);
        // rescale next_u32() over [0, 2^31) to obtain a full [0, 1) uniform.
        let u_drop = rng.next_u32() as f32 / 4_294_967_296.0;
        if u_drop >= p_drop {
            return ModalityDrop::Both;
        }
        let u_which = rng.next_u32() as f32 / 4_294_967_296.0;
        if u_which < p_audio {
            ModalityDrop::DropAudio
        } else {
            ModalityDrop::DropVideo
        }
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────────

/// AV-HuBERT model configuration.
#[derive(Debug, Clone)]
pub struct AvHubertConfig {
    /// Dimension of the per-frame acoustic feature vector (e.g. mel bins).
    pub audio_feat_dim: usize,
    /// Dimension of the per-frame flattened lip-ROI feature vector.
    pub video_feat_dim: usize,
    /// Model hidden width `d_model`.
    pub d_model: usize,
    /// Number of attention heads (must divide `d_model`).
    pub n_heads: usize,
    /// Number of Transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward intermediate width.
    pub d_ff: usize,
    /// Number of discrete cluster units `K` (the k-means codebook size).
    pub n_clusters: usize,
}

impl AvHubertConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            audio_feat_dim: 13,
            video_feat_dim: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
            n_clusters: 20,
        }
    }

    /// Channel width of the concatenated `[audio ‖ video]` feature vector.
    #[must_use]
    pub fn fused_dim(&self) -> usize {
        self.d_model
    }

    /// Validate the configuration.
    pub fn validate(&self) -> MmResult<()> {
        if self.audio_feat_dim == 0 || self.video_feat_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.d_model == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.n_layers == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        if self.n_clusters == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Encoder-layer weights ──────────────────────────────────────────────────────

/// Weights for one pre-norm Transformer encoder layer.
#[derive(Debug, Clone)]
pub struct AvEncoderLayer {
    /// Self-attention projections.
    pub attn: CrossAttnWeights,
    /// Feed-forward network.
    pub ffn: FeedForward,
    /// LayerNorm before attention.
    pub ln1: LayerNorm,
    /// LayerNorm before the feed-forward.
    pub ln2: LayerNorm,
}

impl AvEncoderLayer {
    #[must_use]
    fn zeros(cfg: &AvHubertConfig) -> Self {
        let d = cfg.d_model;
        let attn_cfg = CrossAttnConfig {
            n_heads: cfg.n_heads,
            d_model: d,
            d_k: d / cfg.n_heads,
            d_v: d / cfg.n_heads,
            dropout_rate: 0.0,
        };
        Self {
            attn: CrossAttnWeights::zeros(&attn_cfg),
            ffn: FeedForward::zeros(d, cfg.d_ff),
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
        }
    }
}

// ─── Model weights ──────────────────────────────────────────────────────────────

/// All learnable weights of the AV-HuBERT model.
#[derive(Debug, Clone)]
pub struct AvHubertWeights {
    /// Audio frontend weight `[audio_feat_dim × d_model]`.
    pub audio_w: Vec<f32>,
    /// Audio frontend bias `[d_model]`.
    pub audio_b: Vec<f32>,
    /// Video frontend weight `[video_feat_dim × d_model]`.
    pub video_w: Vec<f32>,
    /// Video frontend bias `[d_model]`.
    pub video_b: Vec<f32>,
    /// Fusion weight over the concatenated `[2·d_model × d_model]` features.
    pub fusion_w: Vec<f32>,
    /// Fusion bias `[d_model]`.
    pub fusion_b: Vec<f32>,
    /// Learnable mask-embedding substituted at masked positions `[d_model]`.
    pub mask_embed: Vec<f32>,
    /// Transformer encoder layers.
    pub layers: Vec<AvEncoderLayer>,
    /// Final encoder LayerNorm.
    pub final_ln: LayerNorm,
    /// Cluster-prediction head weight `[d_model × n_clusters]`.
    pub head_w: Vec<f32>,
    /// Cluster-prediction head bias `[n_clusters]`.
    pub head_b: Vec<f32>,
}

impl AvHubertWeights {
    /// Zero-initialised weights (LayerNorm scales = 1).
    #[must_use]
    pub fn zeros(cfg: &AvHubertConfig) -> Self {
        let d = cfg.d_model;
        Self {
            audio_w: vec![0.0_f32; cfg.audio_feat_dim * d],
            audio_b: vec![0.0_f32; d],
            video_w: vec![0.0_f32; cfg.video_feat_dim * d],
            video_b: vec![0.0_f32; d],
            fusion_w: vec![0.0_f32; 2 * d * d],
            fusion_b: vec![0.0_f32; d],
            mask_embed: vec![0.0_f32; d],
            layers: (0..cfg.n_layers)
                .map(|_| AvEncoderLayer::zeros(cfg))
                .collect(),
            final_ln: LayerNorm::ones(d),
            head_w: vec![0.0_f32; d * cfg.n_clusters],
            head_b: vec![0.0_f32; cfg.n_clusters],
        }
    }

    /// Deterministic Gaussian initialisation (transformer `1/√d` scale) so the
    /// model actually propagates inputs to non-trivial outputs.
    #[must_use]
    pub fn random(cfg: &AvHubertConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.d_model;
        let mut w = Self::zeros(cfg);
        let s_in_a = 1.0_f32 / (cfg.audio_feat_dim as f32).sqrt();
        let s_in_v = 1.0_f32 / (cfg.video_feat_dim as f32).sqrt();
        let s_d = 1.0_f32 / (d as f32).sqrt();
        let s_ff = 1.0_f32 / (cfg.d_ff as f32).sqrt();

        fill_scaled(&mut w.audio_w, s_in_a, rng);
        fill_scaled(&mut w.video_w, s_in_v, rng);
        fill_scaled(&mut w.fusion_w, 1.0 / ((2 * d) as f32).sqrt(), rng);
        fill_scaled(&mut w.mask_embed, s_d, rng);
        for layer in &mut w.layers {
            fill_scaled(&mut layer.attn.w_q, s_d, rng);
            fill_scaled(&mut layer.attn.w_k, s_d, rng);
            fill_scaled(&mut layer.attn.w_v, s_d, rng);
            fill_scaled(&mut layer.attn.w_o, s_d, rng);
            fill_scaled(&mut layer.ffn.w1, s_d, rng);
            fill_scaled(&mut layer.ffn.w2, s_ff, rng);
        }
        fill_scaled(&mut w.head_w, s_d, rng);
        w
    }
}

/// Fill `buf` with N(0,1) samples scaled by `scale`.
fn fill_scaled(buf: &mut [f32], scale: f32, rng: &mut LcgRng) {
    rng.fill_normal(buf);
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

// ─── Fused-feature bundle ───────────────────────────────────────────────────────

/// Output of the frontends + fusion stage: the per-frame fused sequence plus the
/// two pre-fusion modality streams (so callers / tests can inspect exactly what
/// each modality contributed, e.g. to assert a dropped stream is all-zero).
#[derive(Debug, Clone)]
pub struct FusedFeatures {
    /// Fused sequence `[seq_len × d_model]`.
    pub fused: Vec<f32>,
    /// Pre-fusion audio features `[seq_len × d_model]` (all-zero if audio dropped).
    pub audio_pre: Vec<f32>,
    /// Pre-fusion video features `[seq_len × d_model]` (all-zero if video dropped).
    pub video_pre: Vec<f32>,
    /// Number of frames.
    pub seq_len: usize,
    /// Model width.
    pub d_model: usize,
}

// ─── Model ──────────────────────────────────────────────────────────────────────

/// AV-HuBERT model.
pub struct AvHubert {
    cfg: AvHubertConfig,
    weights: AvHubertWeights,
}

impl AvHubert {
    /// Construct from a config and explicit weights.
    #[must_use]
    pub fn new(cfg: AvHubertConfig, weights: AvHubertWeights) -> Self {
        Self { cfg, weights }
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &AvHubertConfig {
        &self.cfg
    }

    /// Run the modality frontends and fusion.
    ///
    /// - `audio`: `[seq_len × audio_feat_dim]` per-frame acoustic features.
    /// - `video`: `[seq_len × video_feat_dim]` per-frame flattened lip-ROI features.
    /// - `drop`: which streams survive (modality dropout).
    ///
    /// A dropped stream's pre-fusion tensor is set to exactly zero before the
    /// concatenation, so it contributes nothing to the fused features.
    pub fn fuse(
        &self,
        audio: &[f32],
        video: &[f32],
        seq_len: usize,
        drop: ModalityDrop,
    ) -> MmResult<FusedFeatures> {
        self.cfg.validate()?;
        let d = self.cfg.d_model;
        let a_dim = self.cfg.audio_feat_dim;
        let v_dim = self.cfg.video_feat_dim;

        if seq_len == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if audio.len() != seq_len * a_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq_len * a_dim,
                got: audio.len(),
            });
        }
        if video.len() != seq_len * v_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq_len * v_dim,
                got: video.len(),
            });
        }

        // Per-frame linear frontends → [seq_len × d_model].
        let mut audio_pre = linear(
            audio,
            &self.weights.audio_w,
            &self.weights.audio_b,
            seq_len,
            a_dim,
            d,
        );
        let mut video_pre = linear(
            video,
            &self.weights.video_w,
            &self.weights.video_b,
            seq_len,
            v_dim,
            d,
        );

        // Modality dropout: zero the dropped stream's contribution entirely.
        if !drop.audio_kept() {
            for v in audio_pre.iter_mut() {
                *v = 0.0;
            }
        }
        if !drop.video_kept() {
            for v in video_pre.iter_mut() {
                *v = 0.0;
            }
        }

        // Concatenate [audio ‖ video] → [seq_len × 2d] then fuse → [seq_len × d].
        let mut concat = vec![0.0_f32; seq_len * 2 * d];
        for t in 0..seq_len {
            concat[t * 2 * d..t * 2 * d + d].copy_from_slice(&audio_pre[t * d..(t + 1) * d]);
            concat[t * 2 * d + d..(t + 1) * 2 * d].copy_from_slice(&video_pre[t * d..(t + 1) * d]);
        }
        let fused = linear(
            &concat,
            &self.weights.fusion_w,
            &self.weights.fusion_b,
            seq_len,
            2 * d,
            d,
        );

        Ok(FusedFeatures {
            fused,
            audio_pre,
            video_pre,
            seq_len,
            d_model: d,
        })
    }

    /// Full forward pass: frontends → fusion → mask substitution → Transformer →
    /// cluster-prediction head.
    ///
    /// `mask` is a per-frame boolean: at masked positions the fused feature is
    /// replaced by the learnable mask embedding (HuBERT masking). Returns the
    /// per-frame cluster logits `[seq_len × n_clusters]`.
    pub fn forward(
        &self,
        audio: &[f32],
        video: &[f32],
        seq_len: usize,
        mask: &[bool],
        drop: ModalityDrop,
    ) -> MmResult<Vec<f32>> {
        if mask.len() != seq_len {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq_len,
                got: mask.len(),
            });
        }
        let d = self.cfg.d_model;
        let fused = self.fuse(audio, video, seq_len, drop)?;

        // Substitute the learnable mask embedding at masked frames.
        let mut hidden = fused.fused;
        for (t, &m) in mask.iter().enumerate() {
            if m {
                hidden[t * d..(t + 1) * d].copy_from_slice(&self.weights.mask_embed);
            }
        }

        // Transformer encoder.
        for layer in &self.weights.layers {
            hidden = encoder_layer_forward(&hidden, seq_len, &self.cfg, layer)?;
        }
        hidden = self.weights.final_ln.forward(&hidden, seq_len)?;

        // Cluster-prediction head → [seq_len × n_clusters].
        let logits = linear(
            &hidden,
            &self.weights.head_w,
            &self.weights.head_b,
            seq_len,
            d,
            self.cfg.n_clusters,
        );
        Ok(logits)
    }

    /// Masked-prediction cross-entropy loss.
    ///
    /// `logits`: `[seq_len × n_clusters]` from [`AvHubert::forward`].
    /// `targets`: per-frame k-means cluster id in `[0, n_clusters)`.
    /// `mask`: per-frame boolean — the loss averages cross-entropy **only over
    /// masked frames** (HuBERT predicts masked positions). Unmasked frames do
    /// not contribute, exactly as in AV-HuBERT.
    ///
    /// Returns the mean masked cross-entropy (`≥ 0`); if no frame is masked the
    /// loss is `0`.
    pub fn masked_prediction_loss(
        &self,
        logits: &[f32],
        targets: &[usize],
        mask: &[bool],
        seq_len: usize,
    ) -> MmResult<f32> {
        let k = self.cfg.n_clusters;
        if logits.len() != seq_len * k {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq_len * k,
                got: logits.len(),
            });
        }
        if targets.len() != seq_len || mask.len() != seq_len {
            return Err(MultiModalError::DimensionMismatch {
                expected: seq_len,
                got: targets.len().min(mask.len()),
            });
        }

        let mut total = 0.0_f32;
        let mut n_masked = 0usize;
        for t in 0..seq_len {
            if !mask[t] {
                continue;
            }
            let tgt = targets[t];
            if tgt >= k {
                return Err(MultiModalError::TokenOutOfRange {
                    token_id: tgt as u32,
                    vocab_size: k,
                });
            }
            let row = &logits[t * k..(t + 1) * k];
            // Stable cross-entropy: −log_softmax(row)[tgt].
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0_f32;
            for &v in row {
                sum_exp += (v - max).exp();
            }
            let log_sum = max + sum_exp.ln();
            total += log_sum - row[tgt];
            n_masked += 1;
        }
        if n_masked == 0 {
            return Ok(0.0);
        }
        let loss = total / n_masked as f32;
        if !loss.is_finite() {
            return Err(MultiModalError::NanEncountered {
                location: "av_hubert::masked_prediction_loss",
            });
        }
        Ok(loss)
    }
}

// ─── Helper layers ──────────────────────────────────────────────────────────────

/// Affine map `out = x · W + b` with `x [rows × in_dim]`, `W [in_dim × out_dim]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = b[o];
            for i in 0..in_dim {
                acc += x[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    out
}

/// One pre-norm Transformer encoder layer (self-attention + FFN, both residual).
fn encoder_layer_forward(
    input: &[f32],
    seq: usize,
    cfg: &AvHubertConfig,
    w: &AvEncoderLayer,
) -> MmResult<Vec<f32>> {
    let d = cfg.d_model;
    let h = cfg.n_heads;
    let d_k = d / h;

    let normed1 = w.ln1.forward(input, seq)?;
    let attn_cfg = CrossAttnConfig {
        n_heads: h,
        d_model: d,
        d_k,
        d_v: d_k,
        dropout_rate: 0.0,
    };
    let attn = CrossAttention::with_weights(attn_cfg, w.attn.clone());
    let sa = attn.forward(&normed1, &normed1, &normed1, seq, seq)?;
    let mut x: Vec<f32> = input.iter().zip(sa.iter()).map(|(a, b)| a + b).collect();

    let normed2 = w.ln2.forward(&x, seq)?;
    let ffn_out = w.ffn.forward(&normed2, seq)?;
    for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
        *xi += fi;
    }
    Ok(x)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> AvHubert {
        let cfg = AvHubertConfig::tiny();
        let mut rng = LcgRng::new(seed);
        let weights = AvHubertWeights::random(&cfg, &mut rng);
        AvHubert::new(cfg, weights)
    }

    fn make_inputs(cfg: &AvHubertConfig, seq_len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = LcgRng::new(seed);
        let mut audio = vec![0.0_f32; seq_len * cfg.audio_feat_dim];
        let mut video = vec![0.0_f32; seq_len * cfg.video_feat_dim];
        rng.fill_normal(&mut audio);
        rng.fill_normal(&mut video);
        (audio, video)
    }

    // (a) Fused sequence shape == [seq_len, fused_dim].
    #[test]
    fn fused_sequence_shape() {
        let m = model(1);
        let seq_len = 7;
        let (audio, video) = make_inputs(m.config(), seq_len, 100);
        let f = m
            .fuse(&audio, &video, seq_len, ModalityDrop::Both)
            .expect("fuse should succeed");
        assert_eq!(f.seq_len, seq_len);
        assert_eq!(f.d_model, m.config().fused_dim());
        assert_eq!(f.fused.len(), seq_len * m.config().fused_dim());
        assert!(f.fused.iter().all(|v| v.is_finite()));
    }

    // (b) Masked-position cluster-prediction cross-entropy loss ≥ 0.
    #[test]
    fn masked_prediction_loss_non_negative() {
        let m = model(2);
        let seq_len = 8;
        let (audio, video) = make_inputs(m.config(), seq_len, 200);
        let mask: Vec<bool> = (0..seq_len).map(|t| t % 2 == 0).collect();
        let targets: Vec<usize> = (0..seq_len).map(|t| t % m.config().n_clusters).collect();
        let logits = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::Both)
            .expect("value should be present");
        let loss = m
            .masked_prediction_loss(&logits, &targets, &mask, seq_len)
            .expect("value should be present");
        assert!(loss >= 0.0, "masked CE must be non-negative, got {loss}");
        assert!(loss.is_finite());
    }

    // (c) Modality dropout — dropped stream's pre-fusion tensor is exactly zero,
    //     and the forward still produces finite output.
    #[test]
    fn modality_dropout_zeros_dropped_stream() {
        let m = model(3);
        let seq_len = 6;
        let (audio, video) = make_inputs(m.config(), seq_len, 300);

        // Drop audio → audio_pre all zeros, video_pre non-trivial.
        let fa = m
            .fuse(&audio, &video, seq_len, ModalityDrop::DropAudio)
            .expect("value should be present");
        assert!(
            fa.audio_pre.iter().all(|&v| v == 0.0),
            "dropped audio must be exactly zero"
        );
        assert!(
            fa.video_pre.iter().any(|&v| v != 0.0),
            "kept video must be non-zero"
        );
        assert!(fa.fused.iter().all(|v| v.is_finite()));

        // Drop video → video_pre all zeros.
        let fv = m
            .fuse(&audio, &video, seq_len, ModalityDrop::DropVideo)
            .expect("value should be present");
        assert!(
            fv.video_pre.iter().all(|&v| v == 0.0),
            "dropped video must be exactly zero"
        );
        assert!(
            fv.audio_pre.iter().any(|&v| v != 0.0),
            "kept audio must be non-zero"
        );

        // Forward with a dropped modality still yields finite logits.
        let mask = vec![true; seq_len];
        let logits = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::DropAudio)
            .expect("value should be present");
        assert_eq!(logits.len(), seq_len * m.config().n_clusters);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // (d) Audio-only and video-only forward passes give finite output of correct shape.
    #[test]
    fn single_modality_forward_correct_shape() {
        let m = model(4);
        let seq_len = 5;
        let (audio, video) = make_inputs(m.config(), seq_len, 400);
        let mask = vec![true; seq_len];
        let k = m.config().n_clusters;

        // Audio-only = DropVideo.
        let logits_a = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::DropVideo)
            .expect("value should be present");
        assert_eq!(logits_a.len(), seq_len * k);
        assert!(logits_a.iter().all(|v| v.is_finite()));

        // Video-only = DropAudio.
        let logits_v = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::DropAudio)
            .expect("value should be present");
        assert_eq!(logits_v.len(), seq_len * k);
        assert!(logits_v.iter().all(|v| v.is_finite()));
    }

    // (e) Deterministic given a fixed RNG seed.
    #[test]
    fn deterministic_under_fixed_seed() {
        let seq_len = 6;
        let m1 = model(7);
        let m2 = model(7);
        let (audio, video) = make_inputs(m1.config(), seq_len, 500);
        let mask: Vec<bool> = (0..seq_len).map(|t| t % 3 == 0).collect();
        let l1 = m1
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::Both)
            .expect("value should be present");
        let l2 = m2
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::Both)
            .expect("value should be present");
        assert_eq!(l1, l2, "identical seed must give identical logits");
    }

    // (f) Masked positions differ from unmasked in the loss: only masked contribute.
    #[test]
    fn only_masked_positions_contribute_to_loss() {
        let m = model(9);
        let seq_len = 8;
        let (audio, video) = make_inputs(m.config(), seq_len, 600);
        let mask_full = vec![true; seq_len];
        let logits = m
            .forward(&audio, &video, seq_len, &mask_full, ModalityDrop::Both)
            .expect("value should be present");
        let targets: Vec<usize> = (0..seq_len)
            .map(|t| (t * 3) % m.config().n_clusters)
            .collect();

        // Loss over only the first half vs. only the second half should differ,
        // proving the mask selects which positions contribute.
        let mut mask_a = vec![false; seq_len];
        let mut mask_b = vec![false; seq_len];
        for t in 0..seq_len / 2 {
            mask_a[t] = true;
        }
        for t in seq_len / 2..seq_len {
            mask_b[t] = true;
        }
        let loss_a = m
            .masked_prediction_loss(&logits, &targets, &mask_a, seq_len)
            .expect("value should be present");
        let loss_b = m
            .masked_prediction_loss(&logits, &targets, &mask_b, seq_len)
            .expect("value should be present");
        assert!(loss_a.is_finite() && loss_b.is_finite());
        assert!(
            (loss_a - loss_b).abs() > 1e-6,
            "different masked subsets must give different losses ({loss_a} vs {loss_b})"
        );

        // No mask → zero loss (no masked positions).
        let none = vec![false; seq_len];
        let loss_none = m
            .masked_prediction_loss(&logits, &targets, &none, seq_len)
            .expect("value should be present");
        assert_eq!(
            loss_none, 0.0,
            "an all-unmasked sequence has zero masked loss"
        );
    }

    #[test]
    fn modality_drop_helpers() {
        assert!(ModalityDrop::Both.audio_kept() && ModalityDrop::Both.video_kept());
        assert!(!ModalityDrop::DropAudio.audio_kept());
        assert!(ModalityDrop::DropAudio.video_kept());
        assert!(!ModalityDrop::DropVideo.video_kept());
        assert!(ModalityDrop::DropVideo.audio_kept());
    }

    #[test]
    fn modality_drop_sample_never_drops_both_and_is_deterministic() {
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        for _ in 0..200 {
            let da = ModalityDrop::sample(&mut rng_a, 0.5, 0.5);
            let db = ModalityDrop::sample(&mut rng_b, 0.5, 0.5);
            assert_eq!(da, db, "sampling must be deterministic for a fixed seed");
            // never drops both modalities (audio_kept || video_kept always true).
            assert!(da.audio_kept() || da.video_kept());
        }
    }

    #[test]
    fn modality_drop_sample_respects_probabilities() {
        // p_drop = 0 → always Both; p_drop = 1, p_audio = 1 → always DropAudio.
        let mut rng = LcgRng::new(55);
        for _ in 0..50 {
            assert_eq!(ModalityDrop::sample(&mut rng, 0.0, 0.5), ModalityDrop::Both);
        }
        let mut rng2 = LcgRng::new(56);
        for _ in 0..50 {
            assert_eq!(
                ModalityDrop::sample(&mut rng2, 1.0, 1.0),
                ModalityDrop::DropAudio
            );
        }
        let mut rng3 = LcgRng::new(57);
        for _ in 0..50 {
            assert_eq!(
                ModalityDrop::sample(&mut rng3, 1.0, 0.0),
                ModalityDrop::DropVideo
            );
        }
    }

    #[test]
    fn config_validation_rejects_bad_shapes() {
        let mut cfg = AvHubertConfig::tiny();
        cfg.n_heads = 3; // 16 % 3 != 0
        assert!(matches!(
            cfg.validate(),
            Err(MultiModalError::InvalidHeads { .. })
        ));

        let mut cfg2 = AvHubertConfig::tiny();
        cfg2.audio_feat_dim = 0;
        assert!(matches!(
            cfg2.validate(),
            Err(MultiModalError::InvalidFeatureDim)
        ));
    }

    #[test]
    fn fuse_rejects_wrong_input_length() {
        let m = model(11);
        let seq_len = 4;
        let audio = vec![0.0_f32; seq_len * m.config().audio_feat_dim + 1]; // wrong
        let video = vec![0.0_f32; seq_len * m.config().video_feat_dim];
        let err = m
            .fuse(&audio, &video, seq_len, ModalityDrop::Both)
            .unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn loss_rejects_out_of_range_target() {
        let m = model(13);
        let seq_len = 3;
        let (audio, video) = make_inputs(m.config(), seq_len, 700);
        let mask = vec![true; seq_len];
        let logits = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::Both)
            .expect("value should be present");
        let mut targets: Vec<usize> = vec![0; seq_len];
        targets[0] = m.config().n_clusters + 5; // out of range
        let err = m
            .masked_prediction_loss(&logits, &targets, &mask, seq_len)
            .unwrap_err();
        assert!(matches!(err, MultiModalError::TokenOutOfRange { .. }));
    }

    #[test]
    fn zero_weights_give_uniform_logits() {
        // With zero head weights and zero bias, every cluster logit is 0, so the
        // masked CE equals ln(n_clusters) at each masked frame.
        let cfg = AvHubertConfig::tiny();
        let weights = AvHubertWeights::zeros(&cfg);
        let k = cfg.n_clusters;
        let m = AvHubert::new(cfg, weights);
        let seq_len = 4;
        let (audio, video) = make_inputs(m.config(), seq_len, 800);
        let mask = vec![true; seq_len];
        let logits = m
            .forward(&audio, &video, seq_len, &mask, ModalityDrop::Both)
            .expect("value should be present");
        assert!(
            logits.iter().all(|&v| v.abs() < 1e-6),
            "zero weights → zero logits"
        );
        let targets = vec![0usize; seq_len];
        let loss = m
            .masked_prediction_loss(&logits, &targets, &mask, seq_len)
            .expect("value should be present");
        let expected = (k as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "uniform CE should be ln(K): {loss} vs {expected}"
        );
    }
}
