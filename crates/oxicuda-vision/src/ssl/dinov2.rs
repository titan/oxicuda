//! DINOv2 — a faithful CPU reference of the self-supervised distillation
//! recipe from Oquab et al. 2023, *"DINOv2: Learning Robust Visual Features
//! without Supervision"*, which combines the image-level **DINO** objective
//! (Caron et al. 2021, *"Emerging Properties in Self-Supervised Vision
//! Transformers"*) with the patch-level **iBOT** masked-image-modelling
//! objective (Zhou et al. 2022, *"iBOT: Image BERT Pre-Training with Online
//! Tokenizer"*).
//!
//! ## The pieces (all real computation, no stubs)
//!
//! 1. **ViT backbone** — patch-embed → prepend CLS → positional embed →
//!    transformer encoder, returning the `[CLS]` embedding *and* the patch
//!    tokens. We reuse [`crate::patch_embed::PatchEmbed`] and
//!    [`crate::vit::ViTEncoder`].
//! 2. **DINO projection head** — a 3-layer MLP (GELU) → L2-normalise →
//!    **weight-normalised prototype layer** producing `K` prototype logits.
//!    The prototype layer's rows are L2-normalised (weight norm with unit
//!    magnitude), so a logit is the cosine similarity between the projected
//!    feature and a learned prototype.
//! 3. **DINO loss** — cross-entropy between a *sharpened, centred* teacher
//!    distribution and a *softer* student distribution:
//!    `H = −Σ_k p_t(k) · log p_s(k)`, with
//!    `p_t = softmax((g_t − c) / τ_t)`, `p_s = softmax(g_s / τ_s)`, and the
//!    teacher temperature `τ_t < τ_s` (sharper teacher).
//! 4. **EMA teacher** — `θ_t ← m·θ_t + (1−m)·θ_s`.
//! 5. **Centering** — running buffer `c ← λ·c + (1−λ)·mean_batch(g_t)`,
//!    subtracted from teacher logits before the softmax to prevent collapse.
//! 6. **iBOT masked-patch term** — at student patch positions that are masked,
//!    predict the *teacher's* (unmasked) patch-prototype distribution via the
//!    same cross-entropy, giving a dense per-patch self-distillation signal.
//!
//! All parameters are flat row-major `Vec<f32>`; no `unsafe`, no external RNG.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
    patch_embed::{PatchEmbed, PatchEmbedConfig, prepend_cls},
    vit::vit_block::{gelu_exact, linear},
    vit::{ViTConfig, ViTEncoder, ViTEncoderConfig},
};

// ─── Backbone output ────────────────────────────────────────────────────────────

/// The two outputs of a DINOv2 ViT backbone forward pass.
#[derive(Debug, Clone)]
pub struct BackboneOutput {
    /// The `[CLS]` embedding: flat `[embed_dim]`.
    pub cls: Vec<f32>,
    /// The patch tokens: flat `[n_patches · embed_dim]`.
    pub patches: Vec<f32>,
    /// Number of patch tokens.
    pub n_patches: usize,
}

// ─── ViT backbone ───────────────────────────────────────────────────────────────

/// A ViT backbone that returns both the `[CLS]` embedding and the patch tokens.
pub struct DinoBackbone {
    /// ViT hyper-parameters.
    pub config: ViTConfig,
    patch_embed: PatchEmbed,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>, // [(n_patches+1) · embed_dim]
    encoder: ViTEncoder,
}

impl DinoBackbone {
    /// Construct a backbone with Gaussian-initialised weights.
    ///
    /// # Errors
    /// Propagates patch / encoder validation errors.
    pub fn new(cfg: ViTConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let e = cfg.embed_dim;
        let pe_cfg = PatchEmbedConfig::new(cfg.img_size, cfg.patch_size, cfg.in_chans, e)?;
        let patch_embed = PatchEmbed::new(pe_cfg, rng);

        let mut cls_token = vec![0.0f32; e];
        rng.fill_normal(&mut cls_token);
        for v in &mut cls_token {
            *v *= 0.02;
        }

        let seq_len = cfg.n_patches() + 1;
        let mut pos_embed = vec![0.0f32; seq_len * e];
        rng.fill_normal(&mut pos_embed);
        for v in &mut pos_embed {
            *v *= 0.02;
        }

        let enc_cfg = ViTEncoderConfig::new(e, cfg.n_heads, cfg.mlp_ratio, cfg.depth)?;
        let encoder = ViTEncoder::new(enc_cfg, rng)?;

        Ok(Self {
            config: cfg,
            patch_embed,
            cls_token,
            pos_embed,
            encoder,
        })
    }

    /// Forward pass returning the `[CLS]` embedding and the patch tokens.
    ///
    /// # Errors
    /// Propagates dimension / backbone errors.
    pub fn forward(&self, image: &[f32]) -> VisionResult<BackboneOutput> {
        let e = self.config.embed_dim;
        let n_patches = self.config.n_patches();

        let patch_tokens = self.patch_embed.forward(image)?;
        let mut tokens = prepend_cls(&patch_tokens, &self.cls_token, e)?;
        // Add positional embedding over CLS + patches.
        for (t, p) in tokens.iter_mut().zip(self.pos_embed.iter()) {
            *t += p;
        }
        let seq_len = n_patches + 1;
        let encoded = self.encoder.forward(&tokens, seq_len)?;

        let cls = encoded[..e].to_vec();
        let patches = encoded[e..].to_vec();
        Ok(BackboneOutput {
            cls,
            patches,
            n_patches,
        })
    }
}

// ─── DINO projection head ───────────────────────────────────────────────────────

/// The DINO projection head: 3-layer MLP (GELU) → L2-normalise →
/// weight-normalised prototype layer producing `n_prototypes` logits.
///
/// The prototype layer is *weight-normalised*: its rows are L2-normalised so
/// that each logit is the cosine similarity of the bottleneck feature with a
/// learned unit prototype, scaled by a learned per-layer gain `g`.
#[derive(Clone)]
pub struct DinoHead {
    in_dim: usize,
    hidden_dim: usize,
    bottleneck_dim: usize,
    n_prototypes: usize,
    // MLP: in → hidden → hidden → bottleneck.
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
    /// Prototype directions `[n_prototypes · bottleneck_dim]` (normalised at use).
    prototypes: Vec<f32>,
    /// Weight-norm gain (scalar magnitude `g`), as in `weight_norm`.
    gain: f32,
}

impl DinoHead {
    /// Construct a head with Gaussian-initialised weights.
    ///
    /// # Errors
    /// - [`VisionError::InvalidEmbedDim`] if `in_dim`, `hidden_dim`, or
    ///   `bottleneck_dim` is 0.
    /// - [`VisionError::InvalidProjDim`] if `n_prototypes == 0`.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        bottleneck_dim: usize,
        n_prototypes: usize,
        rng: &mut LcgRng,
    ) -> VisionResult<Self> {
        if in_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(in_dim));
        }
        if hidden_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(hidden_dim));
        }
        if bottleneck_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(bottleneck_dim));
        }
        if n_prototypes == 0 {
            return Err(VisionError::InvalidProjDim(n_prototypes));
        }

        let fill = |rng: &mut LcgRng, n: usize, sc: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            rng.fill_normal(&mut v);
            for x in &mut v {
                *x *= sc;
            }
            v
        };

        let w1 = fill(rng, hidden_dim * in_dim, 1.0 / (in_dim as f32).sqrt());
        let b1 = vec![0.0f32; hidden_dim];
        let w2 = fill(
            rng,
            hidden_dim * hidden_dim,
            1.0 / (hidden_dim as f32).sqrt(),
        );
        let b2 = vec![0.0f32; hidden_dim];
        let w3 = fill(
            rng,
            bottleneck_dim * hidden_dim,
            1.0 / (hidden_dim as f32).sqrt(),
        );
        let b3 = vec![0.0f32; bottleneck_dim];
        // Prototype directions — random, normalised on the fly.
        let prototypes = fill(
            rng,
            n_prototypes * bottleneck_dim,
            1.0 / (bottleneck_dim as f32).sqrt(),
        );

        Ok(Self {
            in_dim,
            hidden_dim,
            bottleneck_dim,
            n_prototypes,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            prototypes,
            gain: 1.0,
        })
    }

    /// Number of prototype logits produced by this head.
    #[must_use]
    pub fn n_prototypes(&self) -> usize {
        self.n_prototypes
    }

    /// Apply the head to a single feature vector `[in_dim]`, returning the
    /// `[n_prototypes]` prototype logits.
    ///
    /// Pipeline: `MLP → L2-normalise (bottleneck) → cosine vs each prototype × gain`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn forward(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(VisionError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }

        // 3-layer MLP with GELU between layers (no activation after the last).
        let h1 = linear(x, &self.w1, &self.b1, self.in_dim, self.hidden_dim);
        let h1: Vec<f32> = h1.into_iter().map(gelu_exact).collect();
        let h2 = linear(&h1, &self.w2, &self.b2, self.hidden_dim, self.hidden_dim);
        let h2: Vec<f32> = h2.into_iter().map(gelu_exact).collect();
        let mut z = linear(
            &h2,
            &self.w3,
            &self.b3,
            self.hidden_dim,
            self.bottleneck_dim,
        );

        // L2-normalise the bottleneck feature.
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv = 1.0 / norm.max(1e-12);
        for v in &mut z {
            *v *= inv;
        }

        // Weight-normalised prototype layer: logit_k = gain · ⟨z, p̂_k⟩.
        let bd = self.bottleneck_dim;
        let mut logits = vec![0.0f32; self.n_prototypes];
        for (k, lk) in logits.iter_mut().enumerate() {
            let proto = &self.prototypes[k * bd..(k + 1) * bd];
            let pnorm: f32 = proto.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let pinv = 1.0 / pnorm.max(1e-12);
            let dot: f32 = z.iter().zip(proto.iter()).map(|(&a, &b)| a * b).sum();
            *lk = self.gain * dot * pinv;
        }
        Ok(logits)
    }

    /// Apply the head to a batch of features `[batch · in_dim]`, returning
    /// `[batch · n_prototypes]`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] on a length not divisible by `in_dim`.
    pub fn forward_batch(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        if x.is_empty() || x.len() % self.in_dim != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len() % self.in_dim,
            });
        }
        let batch = x.len() / self.in_dim;
        let mut out = vec![0.0f32; batch * self.n_prototypes];
        for b in 0..batch {
            let row = self.forward(&x[b * self.in_dim..(b + 1) * self.in_dim])?;
            out[b * self.n_prototypes..(b + 1) * self.n_prototypes].copy_from_slice(&row);
        }
        Ok(out)
    }

    /// Total number of learnable scalars (used by the EMA update).
    fn num_params(&self) -> usize {
        self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.w3.len()
            + self.b3.len()
            + self.prototypes.len()
            + 1 // gain
    }

    /// Flatten all parameters into a single vector (for distance computations).
    #[cfg(test)]
    fn flatten(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.num_params());
        v.extend_from_slice(&self.w1);
        v.extend_from_slice(&self.b1);
        v.extend_from_slice(&self.w2);
        v.extend_from_slice(&self.b2);
        v.extend_from_slice(&self.w3);
        v.extend_from_slice(&self.b3);
        v.extend_from_slice(&self.prototypes);
        v.push(self.gain);
        v
    }

    /// EMA update **of this (teacher) head toward** the `student` head:
    /// `θ_t ← m·θ_t + (1−m)·θ_s` applied parameter-wise.
    ///
    /// # Errors
    /// - [`VisionError::Internal`] if the heads have mismatched parameter shapes.
    pub fn ema_update(&mut self, student: &DinoHead, momentum: f32) -> VisionResult<()> {
        if self.num_params() != student.num_params()
            || self.w1.len() != student.w1.len()
            || self.prototypes.len() != student.prototypes.len()
        {
            return Err(VisionError::Internal(
                "ema_update: teacher/student head shape mismatch".into(),
            ));
        }
        let m = momentum;
        let lerp = |dst: &mut [f32], src: &[f32]| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = m * *d + (1.0 - m) * s;
            }
        };
        lerp(&mut self.w1, &student.w1);
        lerp(&mut self.b1, &student.b1);
        lerp(&mut self.w2, &student.w2);
        lerp(&mut self.b2, &student.b2);
        lerp(&mut self.w3, &student.w3);
        lerp(&mut self.b3, &student.b3);
        lerp(&mut self.prototypes, &student.prototypes);
        self.gain = m * self.gain + (1.0 - m) * student.gain;
        Ok(())
    }
}

// ─── Distributions, centering, and the DINO loss ────────────────────────────────

/// Numerically-stable softmax of `logits / temperature` (after subtracting an
/// optional per-element `center`).
///
/// `center` may be empty (no centering) or the same length as `logits`.
fn softmax_temp(logits: &[f32], center: &[f32], temperature: f32) -> Vec<f32> {
    let n = logits.len();
    let mut scaled = vec![0.0f32; n];
    if center.is_empty() {
        for i in 0..n {
            scaled[i] = logits[i] / temperature;
        }
    } else {
        for i in 0..n {
            scaled[i] = (logits[i] - center[i]) / temperature;
        }
    }
    let mx = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in &mut scaled {
        *s = (*s - mx).exp();
        sum += *s;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for s in &mut scaled {
        *s *= inv;
    }
    scaled
}

/// Softmax of `logits/τ_s` (student branch — never centred).
///
/// # Errors
/// - [`VisionError::NonPositiveTemperature`] if `tau <= 0`.
pub fn student_softmax(logits: &[f32], tau: f32) -> VisionResult<Vec<f32>> {
    if tau <= 0.0 {
        return Err(VisionError::NonPositiveTemperature(tau));
    }
    Ok(softmax_temp(logits, &[], tau))
}

/// Sharpened, centred teacher distribution `softmax((g_t − c)/τ_t)`.
///
/// `center` must be empty or have the same length as `logits`.
///
/// # Errors
/// - [`VisionError::NonPositiveTemperature`] if `tau <= 0`.
/// - [`VisionError::DimensionMismatch`] if `center` is non-empty and mismatched.
pub fn teacher_softmax(logits: &[f32], center: &[f32], tau: f32) -> VisionResult<Vec<f32>> {
    if tau <= 0.0 {
        return Err(VisionError::NonPositiveTemperature(tau));
    }
    if !center.is_empty() && center.len() != logits.len() {
        return Err(VisionError::DimensionMismatch {
            expected: logits.len(),
            got: center.len(),
        });
    }
    Ok(softmax_temp(logits, center, tau))
}

/// Cross-entropy `H(p_t, p_s) = −Σ_k p_t(k) · log p_s(k)`.
///
/// The teacher distribution `p_t` is the *target*; the student distribution
/// `p_s` is the *prediction*. Returns a value `≥ 0`, equal to the entropy of
/// `p_t` when `p_s == p_t` (its minimum over `p_s`), and `0` exactly when both
/// are the same one-hot distribution.
///
/// # Errors
/// - [`VisionError::DimensionMismatch`] if the two distributions differ in length.
pub fn cross_entropy(p_teacher: &[f32], p_student: &[f32]) -> VisionResult<f32> {
    if p_teacher.len() != p_student.len() {
        return Err(VisionError::DimensionMismatch {
            expected: p_teacher.len(),
            got: p_student.len(),
        });
    }
    let mut h = 0.0f32;
    for (&pt, &ps) in p_teacher.iter().zip(p_student.iter()) {
        if pt > 0.0 {
            // Guard log(0); ps ∈ (0, 1] for a softmax, but clamp defensively.
            h -= pt * ps.max(1e-12).ln();
        }
    }
    Ok(h)
}

/// The full DINO loss between teacher logits and student logits.
///
/// Computes `H(softmax((g_t − c)/τ_t), softmax(g_s/τ_s))`. The teacher branch
/// is centred (with running buffer `c`) and sharpened (`τ_t < τ_s`).
///
/// # Errors
/// - Non-positive temperatures, or mismatched lengths.
pub fn dino_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    center: &[f32],
    tau_student: f32,
    tau_teacher: f32,
) -> VisionResult<f32> {
    if student_logits.len() != teacher_logits.len() {
        return Err(VisionError::DimensionMismatch {
            expected: teacher_logits.len(),
            got: student_logits.len(),
        });
    }
    let p_t = teacher_softmax(teacher_logits, center, tau_teacher)?;
    let p_s = student_softmax(student_logits, tau_student)?;
    cross_entropy(&p_t, &p_s)
}

// ─── Centering buffer ───────────────────────────────────────────────────────────

/// Running centre buffer for the teacher outputs.
///
/// Updated by `c ← λ·c + (1−λ)·mean_batch(g_t)` and subtracted from teacher
/// logits before the softmax. This prevents the trivial collapse where the
/// teacher always predicts the same prototype.
#[derive(Debug, Clone)]
pub struct CenteringBuffer {
    /// The centre vector `[n_prototypes]`.
    pub center: Vec<f32>,
    /// EMA decay `λ ∈ [0, 1)`.
    pub momentum: f32,
}

impl CenteringBuffer {
    /// New zero-initialised buffer of dimension `dim`.
    #[must_use]
    pub fn new(dim: usize, momentum: f32) -> Self {
        Self {
            center: vec![0.0f32; dim],
            momentum,
        }
    }

    /// Update the centre from a batch of teacher logits `[batch · dim]`.
    ///
    /// `c ← λ·c + (1−λ)·mean_batch(g_t)`.
    ///
    /// # Errors
    /// - [`VisionError::DimensionMismatch`] if `batch_logits` is not a multiple
    ///   of `dim` or is empty.
    pub fn update(&mut self, batch_logits: &[f32]) -> VisionResult<()> {
        let dim = self.center.len();
        if dim == 0 || batch_logits.is_empty() || batch_logits.len() % dim != 0 {
            return Err(VisionError::DimensionMismatch {
                expected: dim,
                got: batch_logits.len(),
            });
        }
        let batch = batch_logits.len() / dim;
        let mut mean = vec![0.0f32; dim];
        for b in 0..batch {
            for k in 0..dim {
                mean[k] += batch_logits[b * dim + k];
            }
        }
        let inv_b = 1.0 / batch as f32;
        let lam = self.momentum;
        for (c, m) in self.center.iter_mut().zip(mean.iter()) {
            let batch_mean = m * inv_b;
            *c = lam * *c + (1.0 - lam) * batch_mean;
        }
        Ok(())
    }
}

// ─── iBOT masked-patch term ─────────────────────────────────────────────────────

/// The iBOT masked-image-modelling loss term.
///
/// For each *masked* student patch position, the student must predict the
/// teacher's (unmasked) patch-prototype distribution. The loss is the mean
/// cross-entropy over the masked positions; positions that are not masked are
/// ignored.
///
/// - `student_patch_logits` / `teacher_patch_logits`: `[n_patches · n_proto]`.
/// - `mask`: `[n_patches]` booleans; `true` ⇒ that patch is masked for the
///   student and contributes to the loss.
/// - `patch_center`: optional `[n_proto]` centre for the teacher patch head.
///
/// Returns `0.0` if no patch is masked.
///
/// # Errors
/// - Mismatched shapes or non-positive temperatures.
pub fn ibot_loss(
    student_patch_logits: &[f32],
    teacher_patch_logits: &[f32],
    mask: &[bool],
    patch_center: &[f32],
    n_proto: usize,
    tau_student: f32,
    tau_teacher: f32,
) -> VisionResult<f32> {
    if n_proto == 0 {
        return Err(VisionError::InvalidProjDim(n_proto));
    }
    let n_patches = mask.len();
    if student_patch_logits.len() != n_patches * n_proto
        || teacher_patch_logits.len() != n_patches * n_proto
    {
        return Err(VisionError::DimensionMismatch {
            expected: n_patches * n_proto,
            got: student_patch_logits.len(),
        });
    }

    let mut total = 0.0f32;
    let mut count = 0usize;
    for p in 0..n_patches {
        if !mask[p] {
            continue;
        }
        let s = &student_patch_logits[p * n_proto..(p + 1) * n_proto];
        let t = &teacher_patch_logits[p * n_proto..(p + 1) * n_proto];
        let l = dino_loss(s, t, patch_center, tau_student, tau_teacher)?;
        total += l;
        count += 1;
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(total / count as f32)
}

// ─── KoLeo regulariser ────────────────────────────────────────────────────────────

/// KoLeo (Kozachenko–Leonenko) differential-entropy regulariser.
///
/// Introduced for DINOv2 (Oquab 2023), the KoLeo term encourages a *uniform*
/// span of the embedding space by maximising the Kozachenko–Leonenko estimate of
/// the differential entropy of the batch. Concretely, for L2-normalised
/// embeddings `z_0 … z_{N−1}` it penalises the negative log of each point's
/// nearest-neighbour distance:
///
/// ```text
/// d_i      = min_{j ≠ i} ‖ẑ_i − ẑ_j‖₂
/// L_koleo  = (1/N) Σ_i −log(d_i + ε)
/// ```
///
/// Minimising `L_koleo` pushes the closest pair of embeddings apart, which
/// spreads the batch over the hypersphere and combats representation collapse.
/// Embeddings are L2-normalised internally (the DINOv2 recipe normalises before
/// computing distances); a near-zero embedding is renormalised defensively.
///
/// - `embeddings`: flat `[batch · dim]` row-major.
/// - `eps`: small positive stabiliser added inside the log (e.g. `1e-8`).
///
/// Returns `0.0` for a single embedding (no neighbour to repel).
///
/// # Errors
/// - [`VisionError::EmptyInput`] if `embeddings` is empty or `dim == 0`.
/// - [`VisionError::DimensionMismatch`] if `embeddings.len()` is not a multiple
///   of `dim`.
/// - [`VisionError::NonFinite`] if an input or `eps` is non-finite.
pub fn koleo_loss(embeddings: &[f32], dim: usize, eps: f32) -> VisionResult<f32> {
    if embeddings.is_empty() || dim == 0 {
        return Err(VisionError::EmptyInput("koleo embeddings"));
    }
    if embeddings.len() % dim != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: dim,
            got: embeddings.len(),
        });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(VisionError::NonFinite("koleo eps"));
    }
    if embeddings.iter().any(|v| !v.is_finite()) {
        return Err(VisionError::NonFinite("koleo embeddings"));
    }
    let batch = embeddings.len() / dim;
    if batch < 2 {
        return Ok(0.0);
    }

    // L2-normalise each embedding into a contiguous buffer.
    let mut z = vec![0.0f32; batch * dim];
    for i in 0..batch {
        let src = &embeddings[i * dim..(i + 1) * dim];
        let norm = src.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
        let dst = &mut z[i * dim..(i + 1) * dim];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s / norm;
        }
    }

    // Nearest-neighbour distance per point (brute-force O(N²·dim)).
    let mut acc = 0.0f64;
    for i in 0..batch {
        let zi = &z[i * dim..(i + 1) * dim];
        let mut best = f32::INFINITY;
        for j in 0..batch {
            if i == j {
                continue;
            }
            let zj = &z[j * dim..(j + 1) * dim];
            let mut d2 = 0.0f32;
            for k in 0..dim {
                let diff = zi[k] - zj[k];
                d2 += diff * diff;
            }
            if d2 < best {
                best = d2;
            }
        }
        let dist = best.sqrt();
        acc += -((dist + eps) as f64).ln();
    }
    Ok((acc / batch as f64) as f32)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn entropy(p: &[f32]) -> f32 {
        let mut h = 0.0f32;
        for &v in p {
            if v > 0.0 {
                h -= v * v.ln();
            }
        }
        h
    }

    fn make_head(seed: u64, k: usize) -> DinoHead {
        let mut rng = LcgRng::new(seed);
        DinoHead::new(32, 64, 16, k, &mut rng).expect("head ok")
    }

    // ── Backbone ──────────────────────────────────────────────────────────────────

    #[test]
    fn backbone_returns_cls_and_patches() {
        let mut rng = LcgRng::new(1);
        let cfg = ViTConfig::tiny();
        let e = cfg.embed_dim;
        let n_patches = cfg.n_patches();
        let bb = DinoBackbone::new(cfg, &mut rng).expect("backbone ok");
        let img = vec![0.3f32; 3 * 32 * 32];
        let out = bb.forward(&img).expect("forward ok");
        assert_eq!(out.cls.len(), e, "CLS must be [embed_dim]");
        assert_eq!(
            out.patches.len(),
            n_patches * e,
            "patches must be [n_patches, e]"
        );
        assert_eq!(out.n_patches, n_patches);
        assert!(out.cls.iter().all(|v| v.is_finite()));
        assert!(out.patches.iter().all(|v| v.is_finite()));
    }

    // ── Head: (f) prototype logits shape + softmax sums to 1 ──────────────────────

    #[test]
    fn head_prototype_logits_shape_and_softmax() {
        let head = make_head(2, 128);
        let mut rng = LcgRng::new(3);
        let mut x = vec![0.0f32; 32];
        rng.fill_normal(&mut x);
        let logits = head.forward(&x).expect("ok");
        assert_eq!(logits.len(), 128, "prototype logits must be [n_prototypes]");
        let p = student_softmax(&logits, 0.1).expect("ok");
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1; got {sum}");
        // Logits are cosine·gain, so each is within [-gain, gain] = [-1, 1].
        for &l in &logits {
            assert!(
                (-1.0 - 1e-4..=1.0 + 1e-4).contains(&l),
                "logit out of cosine range: {l}"
            );
        }
    }

    // ── (a) EMA update moves teacher strictly toward student ──────────────────────

    #[test]
    fn ema_update_moves_teacher_toward_student() {
        let mut teacher = make_head(10, 64);
        let student = make_head(20, 64); // different seed ⇒ different params
        let before = l2(&teacher.flatten(), &student.flatten());
        assert!(before > 0.0, "teacher and student must start apart");
        teacher.ema_update(&student, 0.9).expect("ema ok");
        let after = l2(&teacher.flatten(), &student.flatten());
        assert!(
            after < before,
            "EMA must reduce ‖θ_t − θ_s‖: before={before}, after={after}"
        );
        // For momentum m, the distance scales by exactly m.
        assert!(
            (after - 0.9 * before).abs() < 1e-3 * before.max(1.0),
            "EMA distance should scale by m=0.9: after={after}, 0.9·before={}",
            0.9 * before
        );
    }

    #[test]
    fn ema_update_shape_mismatch_errors() {
        let mut teacher = make_head(10, 64);
        let other = make_head(11, 32); // different n_prototypes
        let r = teacher.ema_update(&other, 0.9);
        assert!(matches!(r, Err(VisionError::Internal(_))));
    }

    // ── (b) DINO loss ≥ 0, = 0 only when distributions match ──────────────────────

    #[test]
    fn dino_loss_nonnegative() {
        let mut rng = LcgRng::new(30);
        for _ in 0..20 {
            let mut sl = vec![0.0f32; 16];
            let mut tl = vec![0.0f32; 16];
            rng.fill_normal(&mut sl);
            rng.fill_normal(&mut tl);
            let l = dino_loss(&sl, &tl, &[], 0.1, 0.04).expect("ok");
            assert!(l >= -1e-6, "DINO loss must be ≥ 0; got {l}");
        }
    }

    #[test]
    fn dino_loss_minimised_when_student_matches_teacher() {
        // When the student distribution equals the teacher distribution, the
        // cross-entropy equals the teacher entropy (its minimum over p_s).
        // A one-hot teacher (entropy 0) ⇒ loss → 0 as the student concentrates.
        let teacher_logits = vec![20.0f32, -20.0, -20.0, -20.0]; // ~one-hot at 0
        let student_logits = vec![20.0f32, -20.0, -20.0, -20.0];
        // Same temperature so distributions coincide.
        let p_t = teacher_softmax(&teacher_logits, &[], 0.1).expect("ok");
        let p_s = student_softmax(&student_logits, 0.1).expect("ok");
        let h_self = cross_entropy(&p_t, &p_s).expect("ok");
        assert!(
            h_self < 1e-3,
            "matched ~one-hot dists give ≈0 loss; got {h_self}"
        );

        // A mismatched student must give a strictly larger loss.
        let student_bad = vec![-20.0f32, 20.0, -20.0, -20.0]; // peaks elsewhere
        let p_bad = student_softmax(&student_bad, 0.1).expect("ok");
        let h_bad = cross_entropy(&p_t, &p_bad).expect("ok");
        assert!(
            h_bad > h_self + 1.0,
            "mismatched student must raise the loss: self={h_self}, bad={h_bad}"
        );
    }

    #[test]
    fn cross_entropy_equals_entropy_at_self() {
        // H(p, p) == entropy(p) for any distribution p.
        let logits = vec![1.0f32, 0.3, -0.5, 2.0, -1.0];
        let p = student_softmax(&logits, 1.0).expect("ok");
        let ce = cross_entropy(&p, &p).expect("ok");
        let ent = entropy(&p);
        assert!(
            (ce - ent).abs() < 1e-5,
            "H(p,p) must equal entropy(p): {ce} vs {ent}"
        );
    }

    // ── (c) Centering keeps running teacher-output mean near 0 ────────────────────

    #[test]
    fn centering_drives_mean_near_zero() {
        // Repeatedly feed the same biased batch; the centred logits' mean must
        // shrink toward 0 as the centre converges to the batch mean.
        let dim = 8;
        let mut buf = CenteringBuffer::new(dim, 0.9);
        // Biased teacher logits: every sample equals the same vector with a
        // strong offset on dim 0.
        let base: Vec<f32> = (0..dim).map(|k| if k == 0 { 5.0 } else { 0.1 }).collect();
        let batch = 4;
        let mut flat = Vec::new();
        for _ in 0..batch {
            flat.extend_from_slice(&base);
        }

        // Many updates ⇒ centre → batch mean (== base here).
        for _ in 0..400 {
            buf.update(&flat).expect("ok");
        }
        // Centred logits: base − centre ≈ 0.
        let centred_mean: f32 = base
            .iter()
            .zip(buf.center.iter())
            .map(|(&g, &c)| (g - c).abs())
            .sum::<f32>()
            / dim as f32;
        assert!(
            centred_mean < 1e-2,
            "centering should drive (g − c) mean ≈ 0; got {centred_mean}"
        );
    }

    #[test]
    fn centering_update_bad_shape_errors() {
        let mut buf = CenteringBuffer::new(8, 0.9);
        let r = buf.update(&[0.0f32; 7]); // 7 not a multiple of 8
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── (d) Lower teacher temperature sharpens (lowers entropy) ───────────────────

    #[test]
    fn lower_teacher_temperature_sharpens_distribution() {
        let logits = vec![2.0f32, 1.0, 0.5, -0.5, -1.0, 0.2];
        let p_hot = teacher_softmax(&logits, &[], 0.04).expect("ok"); // sharp
        let p_soft = teacher_softmax(&logits, &[], 0.5).expect("ok"); // soft
        let h_hot = entropy(&p_hot);
        let h_soft = entropy(&p_soft);
        assert!(
            h_hot < h_soft,
            "lower τ_t must lower entropy (sharper): H(0.04)={h_hot} vs H(0.5)={h_soft}"
        );
        // And the sharper distribution must have a larger peak probability.
        let max_hot = p_hot.iter().cloned().fold(0.0f32, f32::max);
        let max_soft = p_soft.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_hot > max_soft, "sharper dist must have a higher peak");
    }

    // ── (e) Nudging student toward teacher drives the loss DOWN ───────────────────

    #[test]
    fn nudging_student_toward_teacher_lowers_loss() {
        // Simulate one gradient-free optimisation step: move student logits a
        // fraction of the way toward the teacher logits and check the DINO loss
        // (with matched temperatures so the target is well-defined) decreases.
        let teacher_logits = vec![1.5f32, -0.5, 0.7, -1.2, 0.3, 0.9];
        let student_before = vec![-1.0f32, 0.8, -0.3, 1.1, -0.6, 0.0];
        let tau = 0.1;

        let loss_before = dino_loss(&student_before, &teacher_logits, &[], tau, tau).expect("ok");

        // Nudge 60% toward the teacher.
        let alpha = 0.6f32;
        let student_after: Vec<f32> = student_before
            .iter()
            .zip(teacher_logits.iter())
            .map(|(&s, &t)| s + alpha * (t - s))
            .collect();
        let loss_after = dino_loss(&student_after, &teacher_logits, &[], tau, tau).expect("ok");

        assert!(
            loss_after < loss_before,
            "moving the student toward the teacher must lower the loss: before={loss_before}, after={loss_after}"
        );
    }

    #[test]
    fn two_views_loss_decreases_when_student_aligns() {
        // Two augmented "views": teacher sees view A, student sees view B. We
        // emulate the head outputs as logits and verify that aligning the
        // student logits toward the teacher target (one gradient-free step)
        // reduces the cross-view DINO loss.
        let head = make_head(40, 32);
        let mut rng = LcgRng::new(41);
        let mut view_a = vec![0.0f32; 32];
        let mut view_b = vec![0.0f32; 32];
        rng.fill_normal(&mut view_a);
        rng.fill_normal(&mut view_b);

        let teacher_logits = head.forward(&view_a).expect("ok");
        let student_logits = head.forward(&view_b).expect("ok");
        let tau = 0.1;
        let loss_before = dino_loss(&student_logits, &teacher_logits, &[], tau, tau).expect("ok");

        let nudged: Vec<f32> = student_logits
            .iter()
            .zip(teacher_logits.iter())
            .map(|(&s, &t)| s + 0.5 * (t - s))
            .collect();
        let loss_after = dino_loss(&nudged, &teacher_logits, &[], tau, tau).expect("ok");
        assert!(
            loss_after < loss_before,
            "aligning student to teacher across views must lower loss: {loss_before} → {loss_after}"
        );
    }

    // ── Student softmax temperature guard ─────────────────────────────────────────

    #[test]
    fn nonpositive_temperature_errors() {
        let r = student_softmax(&[1.0, 2.0], 0.0);
        assert!(matches!(r, Err(VisionError::NonPositiveTemperature(_))));
        let r2 = teacher_softmax(&[1.0, 2.0], &[], -0.1);
        assert!(matches!(r2, Err(VisionError::NonPositiveTemperature(_))));
    }

    // ── iBOT masked-patch term ────────────────────────────────────────────────────

    #[test]
    fn ibot_loss_only_counts_masked_patches() {
        let n_patches = 4;
        let n_proto = 6;
        let mut rng = LcgRng::new(50);
        let mut s = vec![0.0f32; n_patches * n_proto];
        let mut t = vec![0.0f32; n_patches * n_proto];
        rng.fill_normal(&mut s);
        rng.fill_normal(&mut t);

        // No patch masked ⇒ loss is exactly 0.
        let none = vec![false; n_patches];
        let l0 = ibot_loss(&s, &t, &none, &[], n_proto, 0.1, 0.04).expect("ok");
        assert_eq!(l0, 0.0, "no masked patches ⇒ zero iBOT loss");

        // Mask patches 0 and 2 ⇒ loss equals the mean of their per-patch losses.
        let mut mask = vec![false; n_patches];
        mask[0] = true;
        mask[2] = true;
        let l = ibot_loss(&s, &t, &mask, &[], n_proto, 0.1, 0.04).expect("ok");
        let l_p0 = dino_loss(&s[0..n_proto], &t[0..n_proto], &[], 0.1, 0.04).expect("ok");
        let l_p2 = dino_loss(
            &s[2 * n_proto..3 * n_proto],
            &t[2 * n_proto..3 * n_proto],
            &[],
            0.1,
            0.04,
        )
        .expect("ok");
        let expected = 0.5 * (l_p0 + l_p2);
        assert!(
            (l - expected).abs() < 1e-5,
            "iBOT loss must average masked-patch losses: {l} vs {expected}"
        );
        assert!(l >= 0.0, "iBOT loss must be ≥ 0");
    }

    #[test]
    fn ibot_loss_nudging_masked_student_lowers_loss() {
        // Aligning the masked student patch toward the teacher patch reduces it.
        let n_proto = 5;
        let teacher = vec![
            // patch 0 (masked target)
            1.2f32, -0.4, 0.6, -1.0, 0.2, // patch 1
            0.1, 0.1, 0.1, 0.1, 0.1,
        ];
        let student = vec![
            -1.0f32, 0.7, -0.2, 1.0, -0.5, // patch 1
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mask = vec![true, false];
        let tau = 0.1;
        let before = ibot_loss(&student, &teacher, &mask, &[], n_proto, tau, tau).expect("ok");

        let mut nudged = student.clone();
        for k in 0..n_proto {
            nudged[k] += 0.6 * (teacher[k] - student[k]);
        }
        let after = ibot_loss(&nudged, &teacher, &mask, &[], n_proto, tau, tau).expect("ok");
        assert!(
            after < before,
            "nudging masked student patch toward teacher must lower iBOT loss: {before} → {after}"
        );
    }

    #[test]
    fn ibot_loss_bad_shape_errors() {
        let mask = vec![true, false];
        let r = ibot_loss(&[0.0f32; 5], &[0.0f32; 10], &mask, &[], 5, 0.1, 0.04);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── Head batch ────────────────────────────────────────────────────────────────

    #[test]
    fn head_forward_batch_matches_single() {
        let head = make_head(60, 32);
        let mut rng = LcgRng::new(61);
        let batch = 3;
        let mut x = vec![0.0f32; batch * 32];
        rng.fill_normal(&mut x);
        let all = head.forward_batch(&x).expect("ok");
        let k = head.n_prototypes();
        for b in 0..batch {
            let single = head.forward(&x[b * 32..(b + 1) * 32]).expect("ok");
            for (j, &v) in single.iter().enumerate() {
                assert!(
                    (all[b * k + j] - v).abs() < 1e-6,
                    "batch vs single mismatch at b={b}, j={j}"
                );
            }
        }
    }

    #[test]
    fn head_dimension_mismatch_errors() {
        let head = make_head(70, 32);
        let r = head.forward(&[0.0f32; 31]);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn head_zero_prototypes_errors() {
        let mut rng = LcgRng::new(80);
        let r = DinoHead::new(32, 64, 16, 0, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidProjDim(0))));
    }

    // ── KoLeo regulariser ─────────────────────────────────────────────────────────

    #[test]
    fn koleo_single_embedding_is_zero() {
        let z = vec![1.0f32, 0.0, 0.0, 0.0];
        let v = koleo_loss(&z, 4, 1e-8).expect("ok");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn koleo_validation_errors() {
        assert!(koleo_loss(&[], 4, 1e-8).is_err());
        assert!(koleo_loss(&[1.0, 2.0, 3.0], 0, 1e-8).is_err());
        assert!(koleo_loss(&[1.0, 2.0, 3.0], 2, 1e-8).is_err()); // not multiple of dim
        assert!(koleo_loss(&[1.0, 2.0, 3.0, 4.0], 2, 0.0).is_err()); // eps <= 0
        assert!(koleo_loss(&[1.0, f32::NAN, 3.0, 4.0], 2, 1e-8).is_err());
    }

    #[test]
    fn koleo_spread_embeddings_lower_than_clustered() {
        // Two well-separated antipodal points (after normalisation) have a large
        // NN distance → smaller (more negative-log-of-large) KoLeo than two nearly
        // identical points whose tiny NN distance blows up −log.
        let dim = 2;
        // Spread: +x and −x axis → distance 2.
        let spread = vec![1.0f32, 0.0, -1.0, 0.0];
        // Clustered: two almost-identical vectors → distance ≈ 0.
        let clustered = vec![1.0f32, 0.0, 1.0, 0.001];
        let l_spread = koleo_loss(&spread, dim, 1e-8).expect("ok");
        let l_clustered = koleo_loss(&clustered, dim, 1e-8).expect("ok");
        assert!(
            l_clustered > l_spread,
            "clustered KoLeo {l_clustered} should exceed spread {l_spread}"
        );
    }

    #[test]
    fn koleo_known_value_orthogonal_pair() {
        // Two orthonormal vectors: NN distance = sqrt(2). With ε≈0,
        // L = -log(sqrt(2)) = -0.5·ln(2).
        let dim = 2;
        let z = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = koleo_loss(&z, dim, 1e-9).expect("ok");
        let expected = -0.5f32 * 2.0f32.ln();
        assert!((v - expected).abs() < 1e-4, "got {v}, expected {expected}");
    }

    #[test]
    fn koleo_normalisation_invariant_to_scaling() {
        // Scaling every embedding by a constant must not change KoLeo (it
        // normalises internally).
        let dim = 3;
        let mut rng = LcgRng::new(123);
        let mut base = vec![0.0f32; 6 * dim];
        rng.fill_normal(&mut base);
        let scaled: Vec<f32> = base.iter().map(|&v| v * 7.5).collect();
        let a = koleo_loss(&base, dim, 1e-8).expect("ok");
        let b = koleo_loss(&scaled, dim, 1e-8).expect("ok");
        assert!((a - b).abs() < 1e-4, "scale-invariance broken: {a} vs {b}");
    }

    #[test]
    fn koleo_finite_for_random_batch() {
        let dim = 16;
        let batch = 32;
        let mut rng = LcgRng::new(321);
        let mut z = vec![0.0f32; batch * dim];
        rng.fill_normal(&mut z);
        let v = koleo_loss(&z, dim, 1e-8).expect("ok");
        assert!(v.is_finite());
    }
}
