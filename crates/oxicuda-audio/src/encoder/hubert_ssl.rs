//! HuBERT / WavLM-style self-supervised masked-prediction pre-training (CPU).
//!
//! This module implements the three numerical pillars of HuBERT pre-training as
//! a faithful, dependency-free CPU scaffold:
//!
//! 1. **Acoustic-unit discovery via k-means** ([`KMeansQuantizer`]).  The
//!    offline clustering step that converts continuous encoder/MFCC features
//!    into discrete *hidden-unit* pseudo-labels.  HuBERT runs Lloyd's algorithm
//!    over (intermediate) features to obtain the prediction targets.
//! 2. **Span masking** ([`SpanMaskConfig`], [`compute_mask_indices`],
//!    [`apply_span_mask`]).  The BERT-style masking applied to the input feature
//!    sequence: random starting positions each expand into a span of
//!    `mask_span` frames whose feature vectors are replaced by a single learned
//!    mask embedding.  HuBERT's defaults are `p = 0.08`, `span = 10`.
//! 3. **Masked-prediction loss** ([`MaskedPredictionHead`]).  The actual SSL
//!    objective: a projection followed by a cosine-similarity classifier over a
//!    codebook of cluster embeddings, with cross-entropy computed **only over
//!    masked frames** (the defining property of HuBERT).
//!
//! [`HubertPretrainer`] ties the three pillars together into a single
//! [`HubertPretrainer::step`] that, given a feature matrix `[T, D]` and the
//! pre-computed k-means target ids, samples a mask, applies it to a copy of the
//! features, and returns the masked-only cross-entropy loss.
//!
//! All tensors are flat row-major `Vec<f32>`; feature frames are laid out as
//! `[T, D]` (`T` frames, `D`-dimensional each).  Randomness uses the crate's
//! deterministic [`LcgRng`]; there is no dependency on `rand`/`ndarray`.
//!
//! The encoder forward pass that produces contextualised representations is out
//! of scope here (it is provided by sibling modules such as
//! [`crate::encoder::conformer_block`] / [`crate::encoder::wav2vec_cnn`]); the
//! `features` argument therefore stands in for encoder output / projected
//! features when computing the SSL loss.
//!
//! # References
//!
//! - Hsu, Bolte, Tsai, Lakhotia, Salakhutdinov, Mohamed (2021).
//!   *HuBERT: Self-Supervised Speech Representation Learning by Masked
//!   Prediction of Hidden Units.* IEEE/ACM TASLP.
//! - Chen, Wang, Chen, Wu, Liu, Chen, Li, Kanda, Yoshioka, Xiao, Wu, Zhou, Ren,
//!   Qian, Qian, Wu, Zeng, Yu, Wei (2022).
//!   *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech
//!   Processing.* IEEE JSTSP.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Numerically stable log-sum-exp over a logits slice.
///
/// Returns `0.0` for an empty slice (no contribution).
#[inline]
fn log_sum_exp(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All `-inf` (or NaN); fall back to the raw max to avoid `NaN` blow-up.
        return max;
    }
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    max + sum.ln()
}

/// L2 norm of a slice, floored to `eps` to avoid division by zero.
#[inline]
fn l2_norm(v: &[f32], eps: f32) -> f32 {
    let sq: f32 = v.iter().map(|&x| x * x).sum();
    sq.sqrt().max(eps)
}

// ─── KMeansQuantizer ─────────────────────────────────────────────────────────

/// Offline k-means quantiser used for HuBERT acoustic-unit discovery.
///
/// Holds `k` centroids of dimension `dim` (row-major `[k, dim]`).  Fitting runs
/// Lloyd's algorithm with k-means++ seeding; [`KMeansQuantizer::assign`]
/// produces the discrete *hidden-unit* pseudo-labels used as masked-prediction
/// targets.
#[derive(Debug, Clone)]
pub struct KMeansQuantizer {
    /// Centroid table, row-major `[k, dim]`.
    centroids: Vec<f32>,
    /// Number of clusters.
    k: usize,
    /// Feature dimensionality.
    dim: usize,
}

impl KMeansQuantizer {
    /// Fit `k` cluster centroids over `n_frames` feature rows via Lloyd's
    /// algorithm.
    ///
    /// `features` is a flat `[n_frames, dim]` row-major matrix.  Initialisation
    /// uses k-means++ (probability proportional to squared distance from the
    /// nearest already-chosen centre); subsequent `n_iter` iterations alternate
    /// hard assignment and centroid recomputation under squared Euclidean
    /// distance.  Empty clusters are re-seeded from a uniformly random frame so
    /// that all `k` centroids stay populated.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidVocabSize`] if `k == 0`.
    /// - [`AudioError::InvalidEmbedDim`] if `dim == 0`.
    /// - [`AudioError::EmptyInput`] if `n_frames == 0`.
    /// - [`AudioError::InvalidSequenceLength`] if `n_frames < k` (cannot seed
    ///   `k` distinct centres).
    /// - [`AudioError::ShapeMismatch`] if `features.len() != n_frames * dim`.
    pub fn fit(
        features: &[f32],
        n_frames: usize,
        dim: usize,
        k: usize,
        n_iter: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if k == 0 {
            return Err(AudioError::InvalidVocabSize(0));
        }
        if dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if n_frames == 0 {
            return Err(AudioError::EmptyInput {
                msg: "k-means fit: no frames".into(),
            });
        }
        if n_frames < k {
            return Err(AudioError::InvalidSequenceLength(n_frames));
        }
        if features.len() != n_frames * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "k-means fit: features.len()={} != n_frames*dim={}",
                    features.len(),
                    n_frames * dim
                ),
            });
        }

        let frame = |i: usize| &features[i * dim..(i + 1) * dim];

        // ── k-means++ initialisation ──────────────────────────────────────
        let mut centroids = vec![0.0_f32; k * dim];
        // First centre: uniformly random frame.
        let first = rng.next_usize(n_frames);
        centroids[..dim].copy_from_slice(frame(first));

        // `d2[i]` = squared distance from frame `i` to the nearest chosen centre.
        let mut d2 = vec![0.0_f32; n_frames];
        for (i, slot) in d2.iter_mut().enumerate() {
            *slot = squared_distance(frame(i), &centroids[..dim]);
        }
        for c in 1..k {
            // Sample the next centre with probability proportional to `d2`.
            let total: f32 = d2.iter().sum();
            let chosen = if total <= 0.0 {
                // Degenerate (all frames coincide with chosen centres):
                // fall back to a uniformly random frame.
                rng.next_usize(n_frames)
            } else {
                let target = rng.next_f32() * total;
                let mut acc = 0.0_f32;
                let mut idx = n_frames - 1;
                for (i, &w) in d2.iter().enumerate() {
                    acc += w;
                    if acc >= target {
                        idx = i;
                        break;
                    }
                }
                idx
            };
            centroids[c * dim..(c + 1) * dim].copy_from_slice(frame(chosen));
            // Update nearest-centre distances with the newly added centre.
            let new_centre = &centroids[c * dim..(c + 1) * dim];
            for (i, slot) in d2.iter_mut().enumerate() {
                let dist = squared_distance(frame(i), new_centre);
                if dist < *slot {
                    *slot = dist;
                }
            }
        }

        let mut quantizer = Self { centroids, k, dim };

        // ── Lloyd iterations ──────────────────────────────────────────────
        let mut sums = vec![0.0_f32; k * dim];
        let mut counts = vec![0_usize; k];
        for _ in 0..n_iter {
            sums.iter_mut().for_each(|s| *s = 0.0);
            counts.iter_mut().for_each(|c| *c = 0);

            // Assignment + accumulation.
            for i in 0..n_frames {
                let f = frame(i);
                let cid = quantizer.nearest(f);
                counts[cid] += 1;
                let dst = &mut sums[cid * dim..(cid + 1) * dim];
                for (s, &v) in dst.iter_mut().zip(f.iter()) {
                    *s += v;
                }
            }

            // Update: mean of assigned frames, or re-seed empty clusters.
            for c in 0..k {
                if counts[c] == 0 {
                    // Re-seed from a random frame to keep all centres live.
                    let r = rng.next_usize(n_frames);
                    quantizer.centroids[c * dim..(c + 1) * dim].copy_from_slice(frame(r));
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    let src = &sums[c * dim..(c + 1) * dim];
                    let dst = &mut quantizer.centroids[c * dim..(c + 1) * dim];
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d = s * inv;
                    }
                }
            }
        }

        Ok(quantizer)
    }

    /// Index of the centroid nearest to feature row `f` (length `dim`).
    #[inline]
    fn nearest(&self, f: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let centre = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let d = squared_distance(f, centre);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Assign each of `n_frames` rows to its nearest centroid id.
    ///
    /// Returns a `Vec<usize>` of length `n_frames` holding cluster ids in
    /// `0..k` — the discrete hidden-unit pseudo-labels.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `dim` does not match the fitted
    ///   dimensionality.
    /// - [`AudioError::ShapeMismatch`] if `features.len() != n_frames * dim`.
    pub fn assign(&self, features: &[f32], n_frames: usize, dim: usize) -> AudioResult<Vec<usize>> {
        if dim != self.dim {
            return Err(AudioError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if features.len() != n_frames * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "k-means assign: features.len()={} != n_frames*dim={}",
                    features.len(),
                    n_frames * dim
                ),
            });
        }
        let mut ids = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let f = &features[i * dim..(i + 1) * dim];
            ids.push(self.nearest(f));
        }
        Ok(ids)
    }

    /// Total within-cluster sum of squares (inertia) of `features` under the
    /// current centroids.
    ///
    /// Lower is tighter; Lloyd iterations are guaranteed to be non-increasing
    /// in this quantity (the tests rely on it).
    ///
    /// # Errors
    ///
    /// Same validation as [`KMeansQuantizer::assign`].
    pub fn inertia(&self, features: &[f32], n_frames: usize, dim: usize) -> AudioResult<f32> {
        if dim != self.dim {
            return Err(AudioError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if features.len() != n_frames * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "k-means inertia: features.len()={} != n_frames*dim={}",
                    features.len(),
                    n_frames * dim
                ),
            });
        }
        let mut total = 0.0_f32;
        for i in 0..n_frames {
            let f = &features[i * dim..(i + 1) * dim];
            let cid = self.nearest(f);
            let centre = &self.centroids[cid * dim..(cid + 1) * dim];
            total += squared_distance(f, centre);
        }
        Ok(total)
    }

    /// Number of clusters `k`.
    #[must_use]
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Feature dimensionality `dim`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Read-only view of the centroid table, row-major `[k, dim]`.
    #[must_use]
    #[inline]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
}

// ─── Span masking ────────────────────────────────────────────────────────────

/// Configuration for BERT-style span masking of the input feature sequence.
///
/// HuBERT samples a fraction `mask_prob` of frames as span *start* positions,
/// each expanding into a contiguous span of `mask_span` frames; spans may
/// overlap.  The published defaults are `mask_prob = 0.08`, `mask_span = 10`.
#[derive(Debug, Clone)]
pub struct SpanMaskConfig {
    /// Fraction of frames selected as span starts (`0.0..=1.0`).
    pub mask_prob: f32,
    /// Length of each contiguous masked span (in frames, `> 0`).
    pub mask_span: usize,
}

impl SpanMaskConfig {
    /// Construct and validate a span-masking configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::NonFinite`] if `mask_prob` is non-finite or outside
    ///   `[0, 1]`.
    /// - [`AudioError::InvalidSequenceLength`] if `mask_span == 0`.
    pub fn new(mask_prob: f32, mask_span: usize) -> AudioResult<Self> {
        if !mask_prob.is_finite() || !(0.0..=1.0).contains(&mask_prob) {
            return Err(AudioError::NonFinite {
                msg: format!("mask_prob={mask_prob} must be finite in [0, 1]"),
            });
        }
        if mask_span == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }
        Ok(Self {
            mask_prob,
            mask_span,
        })
    }

    /// HuBERT default preset (`mask_prob = 0.08`, `mask_span = 10`).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            mask_prob: 0.08,
            mask_span: 10,
        }
    }
}

/// Compute the boolean mask over `n_frames` frames under `cfg`.
///
/// Each frame is independently selected as a span *start* with probability
/// `cfg.mask_prob`; a selected start marks the next `cfg.mask_span` frames
/// (clamped to the sequence end) as masked.  Spans may overlap, so the realised
/// masked fraction is approximately `mask_prob * mask_span` (saturating below
/// `1.0`).  The returned vector has length `n_frames`.
///
/// # Errors
///
/// [`AudioError::InvalidSequenceLength`] if `n_frames == 0`.
pub fn compute_mask_indices(
    n_frames: usize,
    cfg: &SpanMaskConfig,
    rng: &mut LcgRng,
) -> AudioResult<Vec<bool>> {
    if n_frames == 0 {
        return Err(AudioError::InvalidSequenceLength(0));
    }
    let mut mask = vec![false; n_frames];
    if cfg.mask_prob == 0.0 {
        return Ok(mask);
    }
    for start in 0..n_frames {
        if rng.next_f32() < cfg.mask_prob {
            let end = (start + cfg.mask_span).min(n_frames);
            for m in mask.iter_mut().take(end).skip(start) {
                *m = true;
            }
        }
    }
    Ok(mask)
}

/// Replace masked frames' feature vectors with a learned mask embedding.
///
/// `features` is a flat `[n_frames, dim]` matrix modified in place: every frame
/// `i` with `mask[i] == true` has its `dim`-length row overwritten by
/// `mask_embedding` (length `dim`); unmasked frames are left untouched.
///
/// # Errors
///
/// - [`AudioError::ShapeMismatch`] if `features.len() != n_frames * dim`.
/// - [`AudioError::DimensionMismatch`] if `mask.len() != n_frames` or
///   `mask_embedding.len() != dim`.
pub fn apply_span_mask(
    features: &mut [f32],
    n_frames: usize,
    dim: usize,
    mask: &[bool],
    mask_embedding: &[f32],
) -> AudioResult<()> {
    if features.len() != n_frames * dim {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "apply_span_mask: features.len()={} != n_frames*dim={}",
                features.len(),
                n_frames * dim
            ),
        });
    }
    if mask.len() != n_frames {
        return Err(AudioError::DimensionMismatch {
            expected: n_frames,
            got: mask.len(),
        });
    }
    if mask_embedding.len() != dim {
        return Err(AudioError::DimensionMismatch {
            expected: dim,
            got: mask_embedding.len(),
        });
    }
    for (i, &m) in mask.iter().enumerate() {
        if m {
            features[i * dim..(i + 1) * dim].copy_from_slice(mask_embedding);
        }
    }
    Ok(())
}

// ─── MaskedPredictionHead ────────────────────────────────────────────────────

/// Masked-prediction classifier head for the HuBERT SSL objective.
///
/// Projects each hidden frame from `dim` to `code_dim`, L2-normalises, and
/// scores it against a codebook of `k` cluster embeddings (also `code_dim`)
/// using cosine similarity scaled by `1 / temperature`.  HuBERT uses cosine
/// similarity with `temperature = 0.1`.
#[derive(Debug, Clone)]
pub struct MaskedPredictionHead {
    /// Projection matrix, row-major `[dim, code_dim]` (applied as `xᵀ W`).
    proj: Vec<f32>,
    /// Codebook of cluster embeddings, row-major `[k, code_dim]`.
    codebook: Vec<f32>,
    /// Input hidden dimensionality.
    dim: usize,
    /// Projected / codebook dimensionality.
    code_dim: usize,
    /// Number of target clusters `k`.
    k: usize,
    /// Softmax temperature (cosine logits are divided by this).
    temperature: f32,
}

impl MaskedPredictionHead {
    /// Construct a randomly initialised prediction head.
    ///
    /// The projection and codebook are filled with N(0, 1) samples scaled by
    /// `1 / sqrt(code_dim)` (a standard fan-out-aware initialisation); the
    /// codebook rows are the prototype embeddings the projected hidden states
    /// are matched against.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] if `dim == 0` or `code_dim == 0`.
    /// - [`AudioError::InvalidVocabSize`] if `k == 0`.
    /// - [`AudioError::NonFinite`] if `temperature` is non-finite or `<= 0`.
    pub fn new(
        dim: usize,
        code_dim: usize,
        k: usize,
        temperature: f32,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if dim == 0 || code_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(if dim == 0 {
                0
            } else {
                code_dim
            }));
        }
        if k == 0 {
            return Err(AudioError::InvalidVocabSize(0));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(AudioError::NonFinite {
                msg: format!("temperature={temperature} must be finite and > 0"),
            });
        }
        let scale = 1.0 / (code_dim as f32).sqrt();
        let mut proj = vec![0.0_f32; dim * code_dim];
        rng.fill_normal(&mut proj);
        proj.iter_mut().for_each(|w| *w *= scale);
        let mut codebook = vec![0.0_f32; k * code_dim];
        rng.fill_normal(&mut codebook);
        codebook.iter_mut().for_each(|w| *w *= scale);
        Ok(Self {
            proj,
            codebook,
            dim,
            code_dim,
            k,
            temperature,
        })
    }

    /// Project one hidden frame (length `dim`) into `code_dim` via `xᵀ W`.
    fn project(&self, x: &[f32], out: &mut [f32]) {
        for (j, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (d, &xv) in x.iter().enumerate() {
                acc += xv * self.proj[d * self.code_dim + j];
            }
            *slot = acc;
        }
    }

    /// Compute cosine-similarity logits for every frame against every code.
    ///
    /// `hidden` is a flat `[n_frames, dim]` matrix.  Each frame is projected to
    /// `code_dim`, L2-normalised, and dotted with each L2-normalised code
    /// embedding; the resulting cosine similarities are divided by
    /// `temperature`.  Returns a flat `[n_frames, k]` logits matrix.
    ///
    /// # Errors
    ///
    /// - [`AudioError::DimensionMismatch`] if `dim` mismatches the head.
    /// - [`AudioError::ShapeMismatch`] if `hidden.len() != n_frames * dim`.
    pub fn forward_logits(
        &self,
        hidden: &[f32],
        n_frames: usize,
        dim: usize,
    ) -> AudioResult<Vec<f32>> {
        if dim != self.dim {
            return Err(AudioError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if hidden.len() != n_frames * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "forward_logits: hidden.len()={} != n_frames*dim={}",
                    hidden.len(),
                    n_frames * dim
                ),
            });
        }

        const EPS: f32 = 1e-8;
        let inv_temp = 1.0 / self.temperature;

        // Pre-compute the L2 norm of each code embedding once.
        let mut code_norms = vec![0.0_f32; self.k];
        for (c, slot) in code_norms.iter_mut().enumerate() {
            let code = &self.codebook[c * self.code_dim..(c + 1) * self.code_dim];
            *slot = l2_norm(code, EPS);
        }

        let mut logits = vec![0.0_f32; n_frames * self.k];
        let mut proj = vec![0.0_f32; self.code_dim];
        for t in 0..n_frames {
            let x = &hidden[t * dim..(t + 1) * dim];
            self.project(x, &mut proj);
            let proj_norm = l2_norm(&proj, EPS);
            for c in 0..self.k {
                let code = &self.codebook[c * self.code_dim..(c + 1) * self.code_dim];
                let dot: f32 = proj.iter().zip(code.iter()).map(|(a, b)| a * b).sum();
                let cos = dot / (proj_norm * code_norms[c]);
                logits[t * self.k + c] = cos * inv_temp;
            }
        }
        Ok(logits)
    }

    /// Cross-entropy of the per-frame softmax against `targets`, averaged over
    /// **masked frames only**.
    ///
    /// This is the defining HuBERT objective: the loss is the mean negative
    /// log-likelihood `-log p(target_t | hidden_t)` taken over exactly the
    /// frames with `mask[t] == true`; unmasked frames contribute nothing.  The
    /// log-softmax is computed in a numerically stable way via log-sum-exp.
    ///
    /// If no frame is masked the loss is defined as `0.0`.
    ///
    /// # Errors
    ///
    /// - validation from [`MaskedPredictionHead::forward_logits`];
    /// - [`AudioError::DimensionMismatch`] if `targets.len() != n_frames` or
    ///   `mask.len() != n_frames`;
    /// - [`AudioError::InvalidVocabSize`] if any target id is `>= k`;
    /// - [`AudioError::NonFinite`] if the computed loss is non-finite.
    pub fn masked_ce_loss(
        &self,
        hidden: &[f32],
        n_frames: usize,
        dim: usize,
        targets: &[usize],
        mask: &[bool],
    ) -> AudioResult<f32> {
        self.ce_loss_selected(hidden, n_frames, dim, targets, mask, true)
    }

    /// Cross-entropy averaged over **unmasked frames only**.
    ///
    /// The complement of [`MaskedPredictionHead::masked_ce_loss`]; WavLM and the
    /// ablations in HuBERT combine the two via a mixing weight `alpha`.  If no
    /// frame is unmasked the loss is defined as `0.0`.
    ///
    /// # Errors
    ///
    /// Same as [`MaskedPredictionHead::masked_ce_loss`].
    pub fn unmasked_ce_loss(
        &self,
        hidden: &[f32],
        n_frames: usize,
        dim: usize,
        targets: &[usize],
        mask: &[bool],
    ) -> AudioResult<f32> {
        self.ce_loss_selected(hidden, n_frames, dim, targets, mask, false)
    }

    /// Combined HuBERT loss `alpha * masked + (1 - alpha) * unmasked`.
    ///
    /// With `alpha = 1.0` this reduces to the masked-only objective (HuBERT's
    /// default); `alpha = 0.0` yields the unmasked-only objective.
    ///
    /// # Errors
    ///
    /// - [`AudioError::NonFinite`] if `alpha` is non-finite or outside `[0, 1]`;
    /// - otherwise as [`MaskedPredictionHead::masked_ce_loss`].
    pub fn combined_loss(
        &self,
        hidden: &[f32],
        n_frames: usize,
        dim: usize,
        targets: &[usize],
        mask: &[bool],
        alpha: f32,
    ) -> AudioResult<f32> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(AudioError::NonFinite {
                msg: format!("alpha={alpha} must be finite in [0, 1]"),
            });
        }
        let masked = if alpha > 0.0 {
            self.masked_ce_loss(hidden, n_frames, dim, targets, mask)?
        } else {
            0.0
        };
        let unmasked = if alpha < 1.0 {
            self.unmasked_ce_loss(hidden, n_frames, dim, targets, mask)?
        } else {
            0.0
        };
        Ok(alpha * masked + (1.0 - alpha) * unmasked)
    }

    /// Shared cross-entropy core selecting masked (`want == true`) or unmasked
    /// (`want == false`) frames.
    fn ce_loss_selected(
        &self,
        hidden: &[f32],
        n_frames: usize,
        dim: usize,
        targets: &[usize],
        mask: &[bool],
        want: bool,
    ) -> AudioResult<f32> {
        if targets.len() != n_frames {
            return Err(AudioError::DimensionMismatch {
                expected: n_frames,
                got: targets.len(),
            });
        }
        if mask.len() != n_frames {
            return Err(AudioError::DimensionMismatch {
                expected: n_frames,
                got: mask.len(),
            });
        }
        if let Some(&bad) = targets.iter().find(|&&t| t >= self.k) {
            return Err(AudioError::InvalidVocabSize(bad));
        }

        // `forward_logits` validates `dim` / `hidden.len()`.
        let logits = self.forward_logits(hidden, n_frames, dim)?;

        let mut loss_sum = 0.0_f32;
        let mut count = 0usize;
        for t in 0..n_frames {
            if mask[t] != want {
                continue;
            }
            let row = &logits[t * self.k..(t + 1) * self.k];
            let lse = log_sum_exp(row);
            let target_logit = row[targets[t]];
            // -log softmax(target) = log_sum_exp - target_logit.
            loss_sum += lse - target_logit;
            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }
        let loss = loss_sum / count as f32;
        if !loss.is_finite() {
            return Err(AudioError::NonFinite {
                msg: format!("masked CE loss is non-finite: {loss}"),
            });
        }
        // Cross-entropy is non-negative; clamp away tiny negative round-off.
        Ok(loss.max(0.0))
    }

    /// Number of target clusters `k`.
    #[must_use]
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Projected / codebook dimensionality.
    #[must_use]
    #[inline]
    pub fn code_dim(&self) -> usize {
        self.code_dim
    }

    /// Input hidden dimensionality.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ─── HubertPretrainer ────────────────────────────────────────────────────────

/// Configuration bundle for a [`HubertPretrainer`].
#[derive(Debug, Clone)]
pub struct HubertPretrainConfig {
    /// Number of acoustic-unit clusters (`k`).
    pub k: usize,
    /// Input feature dimensionality `D`.
    pub dim: usize,
    /// Projected / codebook dimensionality of the prediction head.
    pub code_dim: usize,
    /// Softmax temperature for the cosine classifier.
    pub temperature: f32,
    /// Masked/unmasked mixing weight (`1.0` = masked-only, HuBERT default).
    pub alpha: f32,
    /// Span-masking configuration.
    pub mask: SpanMaskConfig,
}

impl HubertPretrainConfig {
    /// Construct and validate a pre-training configuration.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidVocabSize`] if `k == 0`.
    /// - [`AudioError::InvalidEmbedDim`] if `dim == 0` or `code_dim == 0`.
    /// - [`AudioError::NonFinite`] if `temperature` is non-finite / `<= 0`, or
    ///   `alpha` is non-finite / outside `[0, 1]`.
    pub fn new(
        k: usize,
        dim: usize,
        code_dim: usize,
        temperature: f32,
        alpha: f32,
        mask: SpanMaskConfig,
    ) -> AudioResult<Self> {
        if k == 0 {
            return Err(AudioError::InvalidVocabSize(0));
        }
        if dim == 0 || code_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(if dim == 0 {
                0
            } else {
                code_dim
            }));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(AudioError::NonFinite {
                msg: format!("temperature={temperature} must be finite and > 0"),
            });
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(AudioError::NonFinite {
                msg: format!("alpha={alpha} must be finite in [0, 1]"),
            });
        }
        Ok(Self {
            k,
            dim,
            code_dim,
            temperature,
            alpha,
            mask,
        })
    }

    /// Small preset for tests / examples
    /// (`k = 8`, `dim = 16`, `code_dim = 16`, `temperature = 0.1`,
    /// `alpha = 1.0`, [`SpanMaskConfig::tiny`]).
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            k: 8,
            dim: 16,
            code_dim: 16,
            temperature: 0.1,
            alpha: 1.0,
            mask: SpanMaskConfig::tiny(),
        }
    }
}

/// End-to-end HuBERT masked-prediction pre-training scaffold.
///
/// Bundles the [`SpanMaskConfig`], the learned mask embedding, and the
/// [`MaskedPredictionHead`].  [`HubertPretrainer::step`] performs one forward
/// SSL step: sample a span mask, apply it to a copy of the input features, and
/// return the (masked/unmasked-mixed) cross-entropy loss against the supplied
/// k-means target ids.
///
/// The `features` passed to [`HubertPretrainer::step`] stand in for the encoder
/// output / projected features; the contextual encoder forward pass itself
/// lives in sibling modules and is out of scope here.
#[derive(Debug, Clone)]
pub struct HubertPretrainer {
    cfg: HubertPretrainConfig,
    head: MaskedPredictionHead,
    /// Learned mask embedding, length `dim`.
    mask_embedding: Vec<f32>,
}

impl HubertPretrainer {
    /// Construct a pre-trainer with a randomly initialised prediction head and
    /// mask embedding.
    ///
    /// The mask embedding is drawn from N(0, 1) and scaled by `1 / sqrt(dim)`,
    /// matching the head's initialisation scale.
    ///
    /// # Errors
    ///
    /// Propagates [`MaskedPredictionHead::new`] validation
    /// ([`AudioError::InvalidEmbedDim`], [`AudioError::InvalidVocabSize`],
    /// [`AudioError::NonFinite`]).
    pub fn new(cfg: HubertPretrainConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        let head = MaskedPredictionHead::new(cfg.dim, cfg.code_dim, cfg.k, cfg.temperature, rng)?;
        let scale = 1.0 / (cfg.dim as f32).sqrt();
        let mut mask_embedding = vec![0.0_f32; cfg.dim];
        rng.fill_normal(&mut mask_embedding);
        mask_embedding.iter_mut().for_each(|w| *w *= scale);
        Ok(Self {
            cfg,
            head,
            mask_embedding,
        })
    }

    /// Run one masked-prediction step and return the scalar loss.
    ///
    /// Given `features` (`[n_frames, dim]`) and the pre-computed k-means
    /// `targets` (length `n_frames`), this samples a span mask, applies the
    /// learned mask embedding to a **copy** of the features (the original is not
    /// modified), runs the prediction head over the masked copy, and returns the
    /// combined cross-entropy loss `alpha * masked + (1 - alpha) * unmasked`.
    ///
    /// With the default `alpha = 1.0` the loss is computed only over masked
    /// frames — the canonical HuBERT objective.
    ///
    /// # Errors
    ///
    /// - [`AudioError::ShapeMismatch`] if `features.len() != n_frames * dim`;
    /// - [`AudioError::DimensionMismatch`] if `targets.len() != n_frames`;
    /// - propagated masking / loss validation.
    pub fn step(
        &self,
        features: &[f32],
        n_frames: usize,
        dim: usize,
        targets: &[usize],
        rng: &mut LcgRng,
    ) -> AudioResult<f32> {
        if dim != self.cfg.dim {
            return Err(AudioError::DimensionMismatch {
                expected: self.cfg.dim,
                got: dim,
            });
        }
        if features.len() != n_frames * dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "hubert step: features.len()={} != n_frames*dim={}",
                    features.len(),
                    n_frames * dim
                ),
            });
        }
        if targets.len() != n_frames {
            return Err(AudioError::DimensionMismatch {
                expected: n_frames,
                got: targets.len(),
            });
        }

        let mask = compute_mask_indices(n_frames, &self.cfg.mask, rng)?;
        let mut masked = features.to_vec();
        apply_span_mask(&mut masked, n_frames, dim, &mask, &self.mask_embedding)?;
        self.head
            .combined_loss(&masked, n_frames, dim, targets, &mask, self.cfg.alpha)
    }

    /// Read-only view of the prediction head.
    #[must_use]
    #[inline]
    pub fn head(&self) -> &MaskedPredictionHead {
        &self.head
    }

    /// Read-only view of the learned mask embedding (length `dim`).
    #[must_use]
    #[inline]
    pub fn mask_embedding(&self) -> &[f32] {
        &self.mask_embedding
    }

    /// Read-only view of the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &HubertPretrainConfig {
        &self.cfg
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `n_per` frames per cluster around each `center` (length `dim`),
    /// with small N(0, σ) jitter, laid out `[n_clusters * n_per, dim]`.
    fn make_blobs(
        centers: &[Vec<f32>],
        n_per: usize,
        sigma: f32,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, usize, usize) {
        let dim = centers[0].len();
        let n_frames = centers.len() * n_per;
        let mut feats = vec![0.0_f32; n_frames * dim];
        let mut idx = 0usize;
        for center in centers {
            for _ in 0..n_per {
                let mut jitter = vec![0.0_f32; dim];
                rng.fill_normal(&mut jitter);
                for d in 0..dim {
                    feats[idx * dim + d] = center[d] + sigma * jitter[d];
                }
                idx += 1;
            }
        }
        (feats, n_frames, dim)
    }

    #[test]
    fn kmeans_recovers_separated_clusters() {
        let mut rng = LcgRng::new(2024);
        // Three well-separated blobs in 4-D.
        let centers = vec![
            vec![10.0, 10.0, 0.0, 0.0],
            vec![-10.0, -10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, -10.0],
        ];
        let n_per = 40;
        let (feats, n_frames, dim) = make_blobs(&centers, n_per, 0.3, &mut rng);

        let km = KMeansQuantizer::fit(&feats, n_frames, dim, 3, 25, &mut rng).unwrap();
        let ids = km.assign(&feats, n_frames, dim).unwrap();
        assert_eq!(ids.len(), n_frames);

        // Within each true blob, every frame must share one cluster id…
        for c in 0..centers.len() {
            let start = c * n_per;
            let first = ids[start];
            for (i, &id) in ids.iter().enumerate().skip(start).take(n_per) {
                assert_eq!(id, first, "blob {c} not homogeneous at frame {i}");
            }
        }
        // …and distinct blobs must map to distinct cluster ids (a true partition).
        let id0 = ids[0];
        let id1 = ids[n_per];
        let id2 = ids[2 * n_per];
        assert_ne!(id0, id1);
        assert_ne!(id0, id2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn kmeans_inertia_non_increasing() {
        let mut rng = LcgRng::new(7);
        let centers = vec![
            vec![5.0, 0.0, 0.0],
            vec![-5.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0],
            vec![0.0, -5.0, 0.0],
        ];
        let (feats, n_frames, dim) = make_blobs(&centers, 30, 1.0, &mut rng);

        // Identical seed for both runs so initialisation is the same; only the
        // number of Lloyd iterations differs.
        let mut rng_a = LcgRng::new(123);
        let km_few = KMeansQuantizer::fit(&feats, n_frames, dim, 4, 1, &mut rng_a).unwrap();
        let inertia_few = km_few.inertia(&feats, n_frames, dim).unwrap();

        let mut rng_b = LcgRng::new(123);
        let km_many = KMeansQuantizer::fit(&feats, n_frames, dim, 4, 20, &mut rng_b).unwrap();
        let inertia_many = km_many.inertia(&feats, n_frames, dim).unwrap();

        assert!(
            inertia_many <= inertia_few + 1e-3,
            "inertia increased: few(1 iter)={inertia_few}, many(20 iter)={inertia_many}"
        );
    }

    #[test]
    fn kmeans_assign_is_deterministic() {
        let mut rng = LcgRng::new(55);
        let centers = vec![vec![3.0, 3.0], vec![-3.0, -3.0]];
        let (feats, n_frames, dim) = make_blobs(&centers, 20, 0.4, &mut rng);
        let km = KMeansQuantizer::fit(&feats, n_frames, dim, 2, 15, &mut rng).unwrap();
        let a = km.assign(&feats, n_frames, dim).unwrap();
        let b = km.assign(&feats, n_frames, dim).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn kmeans_validation_errors() {
        let mut rng = LcgRng::new(1);
        let feats = vec![0.0_f32; 4 * 2];
        // k == 0.
        assert_eq!(
            KMeansQuantizer::fit(&feats, 4, 2, 0, 5, &mut rng).unwrap_err(),
            AudioError::InvalidVocabSize(0)
        );
        // dim == 0.
        assert_eq!(
            KMeansQuantizer::fit(&feats, 4, 0, 2, 5, &mut rng).unwrap_err(),
            AudioError::InvalidEmbedDim(0)
        );
        // n_frames < k.
        assert!(matches!(
            KMeansQuantizer::fit(&feats, 4, 2, 8, 5, &mut rng).unwrap_err(),
            AudioError::InvalidSequenceLength(4)
        ));
        // empty input.
        assert!(matches!(
            KMeansQuantizer::fit(&[], 0, 2, 1, 5, &mut rng).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
        // shape mismatch on assign.
        let km = KMeansQuantizer::fit(&feats, 4, 2, 2, 1, &mut rng).unwrap();
        assert!(matches!(
            km.assign(&feats, 5, 2).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            km.assign(&feats, 4, 3).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn mask_fraction_within_tolerance() {
        let mut rng = LcgRng::new(99);
        let cfg = SpanMaskConfig::new(0.05, 8).unwrap();
        let n = 5000;
        let mask = compute_mask_indices(n, &cfg, &mut rng).unwrap();
        assert_eq!(mask.len(), n);
        let frac = mask.iter().filter(|&&m| m).count() as f32 / n as f32;
        // Expected coverage ≈ 1 - (1 - p)^span ≈ 0.337 for p=0.05, span=8.
        let expected = 1.0 - (1.0 - cfg.mask_prob).powi(cfg.mask_span as i32);
        assert!(
            (frac - expected).abs() < 0.05,
            "masked fraction {frac} far from expected {expected}"
        );
    }

    #[test]
    fn mask_edge_cases() {
        let mut rng = LcgRng::new(3);
        // mask_prob = 0 -> no masks.
        let cfg0 = SpanMaskConfig::new(0.0, 10).unwrap();
        let m0 = compute_mask_indices(200, &cfg0, &mut rng).unwrap();
        assert!(m0.iter().all(|&m| !m));
        assert_eq!(m0.len(), 200);

        // Coverage never exceeds n_frames (vector length invariant).
        let cfg = SpanMaskConfig::new(0.5, 30).unwrap();
        let m = compute_mask_indices(50, &cfg, &mut rng).unwrap();
        assert_eq!(m.len(), 50);
        assert!(m.iter().filter(|&&x| x).count() <= 50);

        // n_frames == 0 errors.
        assert!(matches!(
            compute_mask_indices(0, &cfg, &mut rng).unwrap_err(),
            AudioError::InvalidSequenceLength(0)
        ));

        // Config validation.
        assert!(matches!(
            SpanMaskConfig::new(1.5, 4).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
        assert!(matches!(
            SpanMaskConfig::new(0.1, 0).unwrap_err(),
            AudioError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn apply_span_mask_overwrites_only_masked() {
        let n_frames = 6;
        let dim = 3;
        let mut feats: Vec<f32> = (0..(n_frames * dim)).map(|x| x as f32).collect();
        let original = feats.clone();
        let mask = vec![false, true, false, true, false, false];
        let embed = vec![-1.0, -2.0, -3.0];
        apply_span_mask(&mut feats, n_frames, dim, &mask, &embed).unwrap();
        for i in 0..n_frames {
            let row = &feats[i * dim..(i + 1) * dim];
            if mask[i] {
                assert_eq!(row, embed.as_slice(), "masked frame {i} not overwritten");
            } else {
                assert_eq!(
                    row,
                    &original[i * dim..(i + 1) * dim],
                    "unmasked frame {i} was modified"
                );
            }
        }
    }

    #[test]
    fn apply_span_mask_validation() {
        let mut feats = vec![0.0_f32; 6];
        let embed = vec![1.0, 1.0];
        // bad feature length.
        assert!(matches!(
            apply_span_mask(&mut feats, 4, 2, &[false; 4], &embed).unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        // bad mask length.
        assert!(matches!(
            apply_span_mask(&mut feats, 3, 2, &[false; 2], &embed).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
        // bad embedding length.
        assert!(matches!(
            apply_span_mask(&mut feats, 3, 2, &[false; 3], &[1.0, 1.0, 1.0]).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn masked_loss_near_zero_for_perfect_prediction() {
        // Use an identity-like setup: dim == code_dim == k, projection ~ identity
        // achieved by directly constructing a head whose codebook is the standard
        // basis and whose projection passes hidden through unchanged.
        let mut rng = LcgRng::new(11);
        let dim = 4;
        let k = 4;
        let mut head = MaskedPredictionHead::new(dim, dim, k, 0.05, &mut rng).unwrap();
        // Overwrite proj = identity, codebook = identity (one-hot codes).
        head.proj = vec![0.0; dim * dim];
        for d in 0..dim {
            head.proj[d * dim + d] = 1.0;
        }
        head.codebook = vec![0.0; k * dim];
        for c in 0..k {
            head.codebook[c * dim + c] = 1.0;
        }

        let n_frames = k;
        // Frame t is a strong one-hot on dimension t -> code t wins by a margin.
        let mut hidden = vec![0.0_f32; n_frames * dim];
        for t in 0..n_frames {
            hidden[t * dim + t] = 5.0;
        }
        let targets: Vec<usize> = (0..n_frames).collect();
        let mask = vec![true; n_frames];

        let loss = head
            .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();
        assert!(loss >= 0.0);
        assert!(loss < 0.05, "perfect-prediction loss too high: {loss}");
    }

    #[test]
    fn masked_loss_near_ln_k_for_random_hidden() {
        let mut rng = LcgRng::new(404);
        let dim = 8;
        let k = 6;
        // Temperature 1.0 keeps cosine logits in [-1, 1] so a random head gives a
        // near-uniform softmax -> loss ≈ ln(k).
        let head = MaskedPredictionHead::new(dim, dim, k, 1.0, &mut rng).unwrap();
        let n_frames = 400;
        let mut hidden = vec![0.0_f32; n_frames * dim];
        rng.fill_normal(&mut hidden);
        let targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();
        let mask = vec![true; n_frames];
        let loss = head
            .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();
        let ln_k = (k as f32).ln();
        assert!(
            (loss - ln_k).abs() < 0.25,
            "random-hidden loss {loss} not near ln(k)={ln_k}"
        );
    }

    #[test]
    fn masked_loss_depends_only_on_masked_frames() {
        // The crucial HuBERT property: changing an UNMASKED frame's target must
        // not change the masked-only loss.
        let mut rng = LcgRng::new(2025);
        let dim = 5;
        let k = 4;
        let head = MaskedPredictionHead::new(dim, dim, k, 0.1, &mut rng).unwrap();
        let n_frames = 10;
        let mut hidden = vec![0.0_f32; n_frames * dim];
        rng.fill_normal(&mut hidden);

        let mask = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let mut targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();

        let loss_before = head
            .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();

        // Flip every UNMASKED frame's target to a different id.
        for t in 0..n_frames {
            if !mask[t] {
                targets[t] = (targets[t] + 1) % k;
            }
        }
        let loss_after = head
            .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();

        assert!(
            (loss_before - loss_after).abs() < 1e-6,
            "masked loss changed when unmasked targets changed: {loss_before} vs {loss_after}"
        );

        // Sanity: the *unmasked* loss should have changed.
        let mut targets2: Vec<usize> = (0..n_frames).map(|i| i % k).collect();
        let unmasked_before = head
            .unmasked_ce_loss(&hidden, n_frames, dim, &targets2, &mask)
            .unwrap();
        for t in 0..n_frames {
            if !mask[t] {
                targets2[t] = (targets2[t] + 2) % k;
            }
        }
        let unmasked_after = head
            .unmasked_ce_loss(&hidden, n_frames, dim, &targets2, &mask)
            .unwrap();
        // With a non-degenerate random head these differ (not asserting magnitude
        // to stay robust, just that both are finite and non-negative).
        assert!(unmasked_before >= 0.0 && unmasked_before.is_finite());
        assert!(unmasked_after >= 0.0 && unmasked_after.is_finite());
    }

    #[test]
    fn loss_always_finite_and_non_negative() {
        let mut rng = LcgRng::new(321);
        let dim = 6;
        let k = 5;
        let head = MaskedPredictionHead::new(dim, dim, k, 0.1, &mut rng).unwrap();
        let n_frames = 32;
        let mut hidden = vec![0.0_f32; n_frames * dim];
        rng.fill_normal(&mut hidden);
        let targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();

        // Various masks, including all-false and all-true.
        for variant in 0..4 {
            let mask: Vec<bool> = (0..n_frames)
                .map(|i| match variant {
                    0 => false,
                    1 => true,
                    2 => i % 2 == 0,
                    _ => i % 3 == 0,
                })
                .collect();
            let loss = head
                .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
                .unwrap();
            assert!(
                loss.is_finite() && loss >= 0.0,
                "bad loss {loss} variant {variant}"
            );
        }
    }

    #[test]
    fn masked_ce_loss_validation() {
        let mut rng = LcgRng::new(5);
        let dim = 3;
        let k = 3;
        let head = MaskedPredictionHead::new(dim, dim, k, 0.1, &mut rng).unwrap();
        let n_frames = 4;
        let hidden = vec![0.0_f32; n_frames * dim];
        let mask = vec![true; n_frames];

        // wrong targets length.
        assert!(matches!(
            head.masked_ce_loss(&hidden, n_frames, dim, &[0, 1], &mask)
                .unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
        // wrong mask length.
        assert!(matches!(
            head.masked_ce_loss(&hidden, n_frames, dim, &[0; 4], &[true; 2])
                .unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
        // target id out of range.
        assert!(matches!(
            head.masked_ce_loss(&hidden, n_frames, dim, &[0, 1, 2, 9], &mask)
                .unwrap_err(),
            AudioError::InvalidVocabSize(9)
        ));
        // dim mismatch.
        assert!(matches!(
            head.forward_logits(&hidden, n_frames, 7).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn head_new_validation() {
        let mut rng = LcgRng::new(1);
        assert_eq!(
            MaskedPredictionHead::new(0, 4, 4, 0.1, &mut rng).unwrap_err(),
            AudioError::InvalidEmbedDim(0)
        );
        assert!(matches!(
            MaskedPredictionHead::new(4, 0, 4, 0.1, &mut rng).unwrap_err(),
            AudioError::InvalidEmbedDim(0)
        ));
        assert_eq!(
            MaskedPredictionHead::new(4, 4, 0, 0.1, &mut rng).unwrap_err(),
            AudioError::InvalidVocabSize(0)
        );
        assert!(matches!(
            MaskedPredictionHead::new(4, 4, 4, 0.0, &mut rng).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
    }

    #[test]
    fn combined_loss_alpha_extremes() {
        let mut rng = LcgRng::new(64);
        let dim = 4;
        let k = 4;
        let head = MaskedPredictionHead::new(dim, dim, k, 0.1, &mut rng).unwrap();
        let n_frames = 12;
        let mut hidden = vec![0.0_f32; n_frames * dim];
        rng.fill_normal(&mut hidden);
        let targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();
        let mask: Vec<bool> = (0..n_frames).map(|i| i % 2 == 0).collect();

        let masked = head
            .masked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();
        let unmasked = head
            .unmasked_ce_loss(&hidden, n_frames, dim, &targets, &mask)
            .unwrap();

        let c1 = head
            .combined_loss(&hidden, n_frames, dim, &targets, &mask, 1.0)
            .unwrap();
        let c0 = head
            .combined_loss(&hidden, n_frames, dim, &targets, &mask, 0.0)
            .unwrap();
        assert!((c1 - masked).abs() < 1e-6);
        assert!((c0 - unmasked).abs() < 1e-6);

        // alpha out of range errors.
        assert!(matches!(
            head.combined_loss(&hidden, n_frames, dim, &targets, &mask, 1.5)
                .unwrap_err(),
            AudioError::NonFinite { .. }
        ));
    }

    #[test]
    fn pretrainer_step_runs_and_is_deterministic() {
        let cfg = HubertPretrainConfig::tiny();
        let dim = cfg.dim;
        let k = cfg.k;
        let mut init_rng = LcgRng::new(2026);
        let trainer = HubertPretrainer::new(cfg, &mut init_rng).unwrap();

        let n_frames = 100;
        let mut feats = vec![0.0_f32; n_frames * dim];
        let mut frng = LcgRng::new(8);
        frng.fill_normal(&mut feats);
        let targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();
        let original = feats.clone();

        let mut step_rng_a = LcgRng::new(500);
        let loss_a = trainer
            .step(&feats, n_frames, dim, &targets, &mut step_rng_a)
            .unwrap();
        // step must not mutate the caller's features.
        assert_eq!(feats, original);

        let mut step_rng_b = LcgRng::new(500);
        let loss_b = trainer
            .step(&feats, n_frames, dim, &targets, &mut step_rng_b)
            .unwrap();
        assert_eq!(loss_a, loss_b, "step not deterministic for equal rng seeds");
        assert!(loss_a.is_finite() && loss_a >= 0.0);
    }

    #[test]
    fn pretrainer_step_validation() {
        let cfg = HubertPretrainConfig::tiny();
        let dim = cfg.dim;
        let k = cfg.k;
        let mut init_rng = LcgRng::new(1);
        let trainer = HubertPretrainer::new(cfg, &mut init_rng).unwrap();
        let n_frames = 10;
        let feats = vec![0.0_f32; n_frames * dim];
        let targets: Vec<usize> = (0..n_frames).map(|i| i % k).collect();
        let mut rng = LcgRng::new(2);

        // wrong dim.
        assert!(matches!(
            trainer
                .step(&feats, n_frames, dim + 1, &targets, &mut rng)
                .unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
        // wrong feature length.
        assert!(matches!(
            trainer
                .step(&feats, n_frames + 1, dim, &targets, &mut rng)
                .unwrap_err(),
            AudioError::ShapeMismatch { .. }
        ));
        // wrong targets length.
        assert!(matches!(
            trainer
                .step(&feats, n_frames, dim, &[0, 1], &mut rng)
                .unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn pretrain_config_validation() {
        let mask = SpanMaskConfig::tiny();
        assert_eq!(
            HubertPretrainConfig::new(0, 4, 4, 0.1, 1.0, mask.clone()).unwrap_err(),
            AudioError::InvalidVocabSize(0)
        );
        assert_eq!(
            HubertPretrainConfig::new(4, 0, 4, 0.1, 1.0, mask.clone()).unwrap_err(),
            AudioError::InvalidEmbedDim(0)
        );
        assert!(matches!(
            HubertPretrainConfig::new(4, 4, 4, -1.0, 1.0, mask.clone()).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
        assert!(matches!(
            HubertPretrainConfig::new(4, 4, 4, 0.1, 2.0, mask).unwrap_err(),
            AudioError::NonFinite { .. }
        ));
    }

    #[test]
    fn end_to_end_kmeans_then_pretrain() {
        // Discover units with k-means, then run a pretrain step on the same
        // features against those targets — the full HuBERT data flow.
        let mut rng = LcgRng::new(909);
        let centers = vec![
            vec![
                6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let (feats, n_frames, dim) = make_blobs(&centers, 30, 0.5, &mut rng);
        let k = 4;
        let km = KMeansQuantizer::fit(&feats, n_frames, dim, k, 20, &mut rng).unwrap();
        let targets = km.assign(&feats, n_frames, dim).unwrap();

        let cfg = HubertPretrainConfig::new(k, dim, 16, 0.1, 1.0, SpanMaskConfig::tiny()).unwrap();
        let trainer = HubertPretrainer::new(cfg, &mut rng).unwrap();
        let mut step_rng = LcgRng::new(77);
        let loss = trainer
            .step(&feats, n_frames, dim, &targets, &mut step_rng)
            .unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }
}
