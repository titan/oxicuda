//! Cross-Modal Distillation.
//!
//! Transfers knowledge between networks operating on *different* input modalities — for
//! example an image teacher supervising a text student, or an audio teacher supervising a
//! vision student. Because the two modalities live in incompatible feature spaces (different
//! dimensionalities and semantics), distillation cannot match raw features directly. Instead
//! each modality is projected into a **shared embedding space** via a learnable linear head,
//! after which alignment objectives become well defined:
//!
//! * **Paired alignment** — for a batch of co-occurring (teacher, student) pairs (e.g. an
//!   image and its caption), pull the projected embeddings together with a cosine or squared-
//!   L2 distance.
//! * **Cross-modal contrastive** — an InfoNCE objective over the batch that pulls each
//!   student embedding towards its paired teacher embedding while pushing it away from the
//!   other (mismatched) teacher embeddings. This is the symmetric CLIP-style loss restricted
//!   to teacher→student transfer and is what gives cross-modal distillation its semantic
//!   grounding when explicit pixel/word correspondence is unavailable.
//!
//! References: Gupta et al. 2016 ("Cross Modal Distillation for Supervision Transfer");
//! Radford et al. 2021 (CLIP contrastive alignment).

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-8;

/// Distance used for the paired-alignment term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignDistance {
    /// `1 − cos(s, t)` per pair.
    Cosine,
    /// Squared L2 between L2-normalised embeddings.
    NormalizedL2,
}

/// Configuration for cross-modal distillation.
#[derive(Debug, Clone)]
pub struct CrossModalConfig {
    /// Teacher feature dimension.
    pub teacher_dim: usize,
    /// Student feature dimension.
    pub student_dim: usize,
    /// Shared embedding dimension both modalities project into.
    pub shared_dim: usize,
    /// Contrastive temperature `tau > 0`.
    pub tau: f32,
    /// Weight on the paired-alignment term.
    pub align_weight: f32,
    /// Weight on the contrastive term.
    pub contrast_weight: f32,
    /// Distance metric for paired alignment.
    pub distance: AlignDistance,
}

impl CrossModalConfig {
    /// Validate and construct a configuration.
    pub fn new(
        teacher_dim: usize,
        student_dim: usize,
        shared_dim: usize,
        tau: f32,
        align_weight: f32,
        contrast_weight: f32,
        distance: AlignDistance,
    ) -> DistillResult<Self> {
        if teacher_dim == 0 || student_dim == 0 || shared_dim == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "teacher_dim, student_dim and shared_dim must be non-zero".into(),
            });
        }
        if tau <= 0.0 || !tau.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("tau must be finite and > 0, got {tau}"),
            });
        }
        if align_weight < 0.0 || contrast_weight < 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: "weights must be non-negative".into(),
            });
        }
        Ok(Self {
            teacher_dim,
            student_dim,
            shared_dim,
            tau,
            align_weight,
            contrast_weight,
            distance,
        })
    }
}

fn linear_project(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            x.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum()
        })
        .collect()
}

fn l2_normalize(x: &[f32]) -> Vec<f32> {
    let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt().max(EPS);
    x.iter().map(|&v| v / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

/// Two linear projection heads mapping each modality into the shared space.
#[derive(Debug, Clone)]
pub struct CrossModalProjector {
    /// Teacher head weights `[shared_dim × teacher_dim]`, row-major.
    pub w_teacher: Vec<f32>,
    /// Student head weights `[shared_dim × student_dim]`, row-major.
    pub w_student: Vec<f32>,
    /// Teacher input dimension.
    pub teacher_dim: usize,
    /// Student input dimension.
    pub student_dim: usize,
    /// Shared output dimension.
    pub shared_dim: usize,
}

impl CrossModalProjector {
    /// Construct projection heads with He-style normal initialisation.
    #[must_use]
    pub fn new(
        teacher_dim: usize,
        student_dim: usize,
        shared_dim: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let st = if teacher_dim == 0 {
            1.0
        } else {
            (2.0_f32 / teacher_dim as f32).sqrt()
        };
        let ss = if student_dim == 0 {
            1.0
        } else {
            (2.0_f32 / student_dim as f32).sqrt()
        };
        let mut w_teacher = vec![0.0_f32; shared_dim * teacher_dim];
        for w in w_teacher.iter_mut() {
            *w = rng.next_normal() * st;
        }
        let mut w_student = vec![0.0_f32; shared_dim * student_dim];
        for w in w_student.iter_mut() {
            *w = rng.next_normal() * ss;
        }
        Self {
            w_teacher,
            w_student,
            teacher_dim,
            student_dim,
            shared_dim,
        }
    }

    /// Project a teacher feature vector into the shared space.
    pub fn project_teacher(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        if x.len() != self.teacher_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.teacher_dim,
                got: x.len(),
            });
        }
        Ok(linear_project(
            x,
            &self.w_teacher,
            self.teacher_dim,
            self.shared_dim,
        ))
    }

    /// Project a student feature vector into the shared space.
    pub fn project_student(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        if x.len() != self.student_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.student_dim,
                got: x.len(),
            });
        }
        Ok(linear_project(
            x,
            &self.w_student,
            self.student_dim,
            self.shared_dim,
        ))
    }
}

/// Paired-alignment loss between two already-projected batches.
///
/// `s_emb` / `t_emb` are `[batch × shared_dim]` flat row-major. Returns the mean per-pair
/// distance under `cfg.distance`.
pub fn paired_alignment_loss(
    s_emb: &[f32],
    t_emb: &[f32],
    batch: usize,
    cfg: &CrossModalConfig,
) -> DistillResult<f32> {
    if s_emb.is_empty() || t_emb.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let expected = batch * cfg.shared_dim;
    if s_emb.len() != expected || t_emb.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: s_emb.len().min(t_emb.len()),
        });
    }
    let mut total = 0.0_f32;
    for b in 0..batch {
        let s = &s_emb[b * cfg.shared_dim..(b + 1) * cfg.shared_dim];
        let t = &t_emb[b * cfg.shared_dim..(b + 1) * cfg.shared_dim];
        total += match cfg.distance {
            AlignDistance::Cosine => 1.0 - cosine(s, t),
            AlignDistance::NormalizedL2 => {
                let sn = l2_normalize(s);
                let tn = l2_normalize(t);
                sn.iter()
                    .zip(tn.iter())
                    .map(|(&a, &c)| (a - c).powi(2))
                    .sum::<f32>()
            }
        };
    }
    Ok(total / batch as f32)
}

/// Cross-modal contrastive (InfoNCE) loss over a batch of projected embeddings.
///
/// For each student embedding `i`, the positive is the paired teacher embedding `i` and the
/// negatives are all other teacher embeddings in the batch. Similarities are cosine,
/// temperature-scaled by `cfg.tau`. Returns the mean cross-entropy over the batch.
pub fn cross_modal_contrastive_loss(
    s_emb: &[f32],
    t_emb: &[f32],
    batch: usize,
    cfg: &CrossModalConfig,
) -> DistillResult<f32> {
    if s_emb.is_empty() || t_emb.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if batch < 1 {
        return Err(DistillError::InvalidConfig {
            msg: "batch must be >= 1".into(),
        });
    }
    let expected = batch * cfg.shared_dim;
    if s_emb.len() != expected || t_emb.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: s_emb.len().min(t_emb.len()),
        });
    }
    let d = cfg.shared_dim;
    let tau = cfg.tau.max(EPS);
    let mut total = 0.0_f32;
    for i in 0..batch {
        let s = &s_emb[i * d..(i + 1) * d];
        // Numerically stable log-softmax over teacher columns for row i.
        let mut logits = vec![0.0_f32; batch];
        let mut max_logit = f32::NEG_INFINITY;
        for j in 0..batch {
            let t = &t_emb[j * d..(j + 1) * d];
            let l = cosine(s, t) / tau;
            logits[j] = l;
            if l > max_logit {
                max_logit = l;
            }
        }
        let denom: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>();
        let log_denom = max_logit + denom.max(EPS).ln();
        // −log p(positive = i).
        total += log_denom - logits[i];
    }
    Ok(total / batch as f32)
}

/// Full cross-modal distillation loss: projects both batches then combines the
/// paired-alignment and contrastive terms with the configured weights.
///
/// `s_feat` / `t_feat` are raw modality features `[batch × student_dim]` /
/// `[batch × teacher_dim]`, flat row-major.
pub fn cross_modal_loss(
    s_feat: &[f32],
    t_feat: &[f32],
    batch: usize,
    projector: &CrossModalProjector,
    cfg: &CrossModalConfig,
) -> DistillResult<f32> {
    if s_feat.is_empty() || t_feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if batch == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "batch must be non-zero".into(),
        });
    }
    if projector.shared_dim != cfg.shared_dim
        || projector.teacher_dim != cfg.teacher_dim
        || projector.student_dim != cfg.student_dim
    {
        return Err(DistillError::InvalidConfig {
            msg: "projector dimensions do not match config".into(),
        });
    }
    if s_feat.len() != batch * cfg.student_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * cfg.student_dim,
            got: s_feat.len(),
        });
    }
    if t_feat.len() != batch * cfg.teacher_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * cfg.teacher_dim,
            got: t_feat.len(),
        });
    }
    let mut s_emb = vec![0.0_f32; batch * cfg.shared_dim];
    let mut t_emb = vec![0.0_f32; batch * cfg.shared_dim];
    for b in 0..batch {
        let s_in = &s_feat[b * cfg.student_dim..(b + 1) * cfg.student_dim];
        let t_in = &t_feat[b * cfg.teacher_dim..(b + 1) * cfg.teacher_dim];
        let sp = projector.project_student(s_in)?;
        let tp = projector.project_teacher(t_in)?;
        s_emb[b * cfg.shared_dim..(b + 1) * cfg.shared_dim].copy_from_slice(&sp);
        t_emb[b * cfg.shared_dim..(b + 1) * cfg.shared_dim].copy_from_slice(&tp);
    }
    let align = paired_alignment_loss(&s_emb, &t_emb, batch, cfg)?;
    let contrast = cross_modal_contrastive_loss(&s_emb, &t_emb, batch, cfg)?;
    Ok(cfg.align_weight * align + cfg.contrast_weight * contrast)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(distance: AlignDistance) -> CrossModalConfig {
        CrossModalConfig::new(6, 4, 5, 0.1, 1.0, 1.0, distance).expect("cfg")
    }

    #[test]
    fn projector_output_dim() {
        let mut rng = LcgRng::new(1);
        let p = CrossModalProjector::new(6, 4, 5, &mut rng);
        let t = vec![0.1_f32; 6];
        let s = vec![0.2_f32; 4];
        assert_eq!(p.project_teacher(&t).expect("t").len(), 5);
        assert_eq!(p.project_student(&s).expect("s").len(), 5);
    }

    #[test]
    fn projector_dim_mismatch_errors() {
        let mut rng = LcgRng::new(2);
        let p = CrossModalProjector::new(6, 4, 5, &mut rng);
        assert!(p.project_teacher(&[0.0_f32; 3]).is_err());
        assert!(p.project_student(&[0.0_f32; 9]).is_err());
    }

    #[test]
    fn paired_alignment_identical_cosine_zero() {
        let c = cfg(AlignDistance::Cosine);
        let emb = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let loss = paired_alignment_loss(&emb, &emb, 2, &c).expect("loss");
        assert!(loss < 1e-5, "loss {loss}");
    }

    #[test]
    fn paired_alignment_identical_l2_zero() {
        let c = cfg(AlignDistance::NormalizedL2);
        let emb = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let loss = paired_alignment_loss(&emb, &emb, 2, &c).expect("loss");
        assert!(loss < 1e-5, "loss {loss}");
    }

    #[test]
    fn contrastive_perfect_alignment_is_minimal() {
        // When student == teacher embeddings and the batch is well separated, the
        // contrastive loss should be small (positive dominates the softmax).
        let c = cfg(AlignDistance::Cosine);
        let d = 5;
        let mut emb = vec![0.0_f32; 3 * d];
        // Three near-orthogonal embeddings.
        emb[0] = 1.0;
        emb[d + 1] = 1.0;
        emb[2 * d + 2] = 1.0;
        let loss = cross_modal_contrastive_loss(&emb, &emb, 3, &c).expect("loss");
        assert!(loss >= 0.0 && loss.is_finite());
        // Scramble the pairing so positives are no longer aligned → loss must increase.
        let mut scrambled = vec![0.0_f32; 3 * d];
        scrambled[1] = 1.0;
        scrambled[d + 2] = 1.0;
        scrambled[2 * d] = 1.0;
        let bad = cross_modal_contrastive_loss(&scrambled, &emb, 3, &c).expect("bad");
        assert!(
            bad > loss,
            "misaligned loss {bad} should exceed aligned {loss}"
        );
    }

    #[test]
    fn contrastive_batch_one_is_zero() {
        // With a single example there are no negatives, so −log(1) = 0.
        let c = cfg(AlignDistance::Cosine);
        let emb = vec![0.3_f32, -0.2, 0.5, 0.1, 0.0];
        let loss = cross_modal_contrastive_loss(&emb, &emb, 1, &c).expect("loss");
        assert!(loss.abs() < 1e-5, "loss {loss}");
    }

    #[test]
    fn full_loss_deterministic_and_finite() {
        let c = cfg(AlignDistance::Cosine);
        let mut rng = LcgRng::new(2024);
        let p = CrossModalProjector::new(6, 4, 5, &mut rng);
        let batch = 4;
        let mut r2 = LcgRng::new(7);
        let s_feat: Vec<f32> = (0..batch * 4).map(|_| r2.next_normal()).collect();
        let t_feat: Vec<f32> = (0..batch * 6).map(|_| r2.next_normal()).collect();
        let l1 = cross_modal_loss(&s_feat, &t_feat, batch, &p, &c).expect("l1");
        let l2 = cross_modal_loss(&s_feat, &t_feat, batch, &p, &c).expect("l2");
        assert!(l1.is_finite() && l1 >= 0.0);
        assert!((l1 - l2).abs() < 1e-7, "non-deterministic: {l1} vs {l2}");
    }

    #[test]
    fn full_loss_dim_mismatch_errors() {
        let c = cfg(AlignDistance::Cosine);
        let mut rng = LcgRng::new(3);
        let p = CrossModalProjector::new(6, 4, 5, &mut rng);
        let s_feat = vec![0.0_f32; 4 * 4];
        let t_feat = vec![0.0_f32; 3 * 6]; // wrong batch
        assert!(cross_modal_loss(&s_feat, &t_feat, 4, &p, &c).is_err());
    }

    #[test]
    fn config_rejects_bad_params() {
        assert!(CrossModalConfig::new(0, 4, 5, 0.1, 1.0, 1.0, AlignDistance::Cosine).is_err());
        assert!(CrossModalConfig::new(6, 4, 5, 0.0, 1.0, 1.0, AlignDistance::Cosine).is_err());
        assert!(CrossModalConfig::new(6, 4, 5, 0.1, -1.0, 1.0, AlignDistance::Cosine).is_err());
    }
}
