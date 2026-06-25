//! CRD with a learnable 2-layer projection-MLP head.
//!
//! The baseline [`crate::relation::crd`] computes its InfoNCE loss directly on raw features.
//! The original Contrastive Representation Distillation paper (Tian et al. 2020), however,
//! routes both student and teacher features through a small **2-layer projection head**
//! (`Linear → ReLU → Linear`) followed by L2-normalisation onto the unit sphere, mapping the
//! two networks' differently-sized embeddings into a shared contrastive space. The contrast
//! is then taken between these projected, normalised embeddings. This module supplies that
//! projector and a matching InfoNCE loss.
//!
//! For each anchor the positive is the teacher embedding of the *same* instance and the
//! negatives are teacher embeddings of other instances in the batch. Cosine similarities are
//! temperature-scaled and the loss is the mean InfoNCE cross-entropy over the batch.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-8;

/// Configuration for the CRD projection-head loss.
#[derive(Debug, Clone)]
pub struct CrdProjConfig {
    /// Student feature dimension.
    pub student_dim: usize,
    /// Teacher feature dimension.
    pub teacher_dim: usize,
    /// Projection-head hidden dimension.
    pub hidden_dim: usize,
    /// Shared contrastive embedding dimension.
    pub embed_dim: usize,
    /// InfoNCE temperature `tau > 0`.
    pub tau: f32,
}

impl CrdProjConfig {
    /// Validate and construct a configuration.
    pub fn new(
        student_dim: usize,
        teacher_dim: usize,
        hidden_dim: usize,
        embed_dim: usize,
        tau: f32,
    ) -> DistillResult<Self> {
        if student_dim == 0 || teacher_dim == 0 || hidden_dim == 0 || embed_dim == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "all dimensions must be non-zero".into(),
            });
        }
        if tau <= 0.0 || !tau.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("tau must be finite and > 0, got {tau}"),
            });
        }
        Ok(Self {
            student_dim,
            teacher_dim,
            hidden_dim,
            embed_dim,
            tau,
        })
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn l2_normalize(x: &[f32]) -> Vec<f32> {
    let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt().max(EPS);
    x.iter().map(|&v| v / norm).collect()
}

/// One 2-layer projection MLP: `in_dim → hidden (ReLU) → embed`, output L2-normalised.
#[derive(Debug, Clone)]
struct Mlp2 {
    in_dim: usize,
    hidden_dim: usize,
    embed_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Mlp2 {
    fn new(in_dim: usize, hidden_dim: usize, embed_dim: usize, rng: &mut LcgRng) -> Self {
        let s1 = (2.0_f32 / in_dim as f32).sqrt();
        let s2 = (2.0_f32 / hidden_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        for w in w1.iter_mut() {
            *w = rng.next_normal() * s1;
        }
        let mut w2 = vec![0.0_f32; embed_dim * hidden_dim];
        for w in w2.iter_mut() {
            *w = rng.next_normal() * s2;
        }
        Self {
            in_dim,
            hidden_dim,
            embed_dim,
            w1,
            b1: vec![0.0_f32; hidden_dim],
            w2,
            b2: vec![0.0_f32; embed_dim],
        }
    }

    fn forward(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let h: Vec<f32> = (0..self.hidden_dim)
            .map(|j| {
                let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
                let dot: f32 = x.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                relu(dot + self.b1[j])
            })
            .collect();
        let e: Vec<f32> = (0..self.embed_dim)
            .map(|k| {
                let row = &self.w2[k * self.hidden_dim..(k + 1) * self.hidden_dim];
                let dot: f32 = h.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                dot + self.b2[k]
            })
            .collect();
        Ok(l2_normalize(&e))
    }
}

/// Paired student / teacher projection heads producing unit embeddings in a shared space.
#[derive(Debug, Clone)]
pub struct CrdProjectionHead {
    student: Mlp2,
    teacher: Mlp2,
    cfg: CrdProjConfig,
}

impl CrdProjectionHead {
    /// Construct both projection heads with He-style initialisation.
    #[must_use]
    pub fn new(cfg: CrdProjConfig, rng: &mut LcgRng) -> Self {
        let student = Mlp2::new(cfg.student_dim, cfg.hidden_dim, cfg.embed_dim, rng);
        let teacher = Mlp2::new(cfg.teacher_dim, cfg.hidden_dim, cfg.embed_dim, rng);
        Self {
            student,
            teacher,
            cfg,
        }
    }

    /// Project a student feature vector to a unit embedding.
    pub fn project_student(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        self.student.forward(x)
    }

    /// Project a teacher feature vector to a unit embedding.
    pub fn project_teacher(&self, x: &[f32]) -> DistillResult<Vec<f32>> {
        self.teacher.forward(x)
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &CrdProjConfig {
        &self.cfg
    }
}

/// CRD InfoNCE loss over projected embeddings.
///
/// `s_feats` / `t_feats` are `[batch × student_dim]` / `[batch × teacher_dim]` flat
/// row-major. For anchor `i` the positive is teacher `i`; negatives are all other teachers.
pub fn crd_proj_loss(
    s_feats: &[f32],
    t_feats: &[f32],
    batch: usize,
    head: &CrdProjectionHead,
) -> DistillResult<f32> {
    if s_feats.is_empty() || t_feats.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if batch == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "batch must be non-zero".into(),
        });
    }
    let cfg = head.config();
    if s_feats.len() != batch * cfg.student_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * cfg.student_dim,
            got: s_feats.len(),
        });
    }
    if t_feats.len() != batch * cfg.teacher_dim {
        return Err(DistillError::DimensionMismatch {
            expected: batch * cfg.teacher_dim,
            got: t_feats.len(),
        });
    }
    // Project all embeddings up front.
    let mut s_emb = Vec::with_capacity(batch);
    let mut t_emb = Vec::with_capacity(batch);
    for b in 0..batch {
        let sp = head.project_student(&s_feats[b * cfg.student_dim..(b + 1) * cfg.student_dim])?;
        let tp = head.project_teacher(&t_feats[b * cfg.teacher_dim..(b + 1) * cfg.teacher_dim])?;
        s_emb.push(sp);
        t_emb.push(tp);
    }
    let tau = cfg.tau.max(EPS);
    let mut total = 0.0_f32;
    for i in 0..batch {
        let s = &s_emb[i];
        // Embeddings are unit-norm, so the dot product is the cosine similarity.
        let mut logits = vec![0.0_f32; batch];
        let mut max_logit = f32::NEG_INFINITY;
        for (j, t) in t_emb.iter().enumerate() {
            let l: f32 = s.iter().zip(t.iter()).map(|(&a, &b)| a * b).sum::<f32>() / tau;
            logits[j] = l;
            if l > max_logit {
                max_logit = l;
            }
        }
        let denom: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>();
        let log_denom = max_logit + denom.max(EPS).ln();
        total += log_denom - logits[i];
    }
    Ok(total / batch as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CrdProjConfig {
        CrdProjConfig::new(8, 6, 16, 4, 0.1).expect("cfg")
    }

    #[test]
    fn projection_is_unit_norm() {
        let mut rng = LcgRng::new(1);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        let x = vec![0.5_f32; 8];
        let e = head.project_student(&x).expect("proj");
        let norm: f32 = e.iter().map(|&v| v * v).sum::<f32>().sqrt();
        // Unit norm unless the pre-normalisation vector was exactly zero.
        assert!((norm - 1.0).abs() < 1e-4 || norm < EPS, "norm {norm}");
        assert_eq!(e.len(), 4);
    }

    #[test]
    fn teacher_projection_dim() {
        let mut rng = LcgRng::new(2);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        let e = head.project_teacher(&[0.3_f32; 6]).expect("proj");
        assert_eq!(e.len(), 4);
    }

    #[test]
    fn projection_dim_mismatch_errors() {
        let mut rng = LcgRng::new(3);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        assert!(head.project_student(&[0.0_f32; 3]).is_err());
        assert!(head.project_teacher(&[0.0_f32; 10]).is_err());
    }

    #[test]
    fn loss_finite_and_deterministic() {
        let mut rng = LcgRng::new(2024);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        let batch = 5;
        let mut r = LcgRng::new(7);
        let s: Vec<f32> = (0..batch * 8).map(|_| r.next_normal()).collect();
        let t: Vec<f32> = (0..batch * 6).map(|_| r.next_normal()).collect();
        let a = crd_proj_loss(&s, &t, batch, &head).expect("a");
        let b = crd_proj_loss(&s, &t, batch, &head).expect("b");
        assert!(a.is_finite() && a >= 0.0, "loss {a}");
        assert!((a - b).abs() < 1e-7, "non-deterministic {a} vs {b}");
    }

    #[test]
    fn batch_one_loss_is_zero() {
        // Single instance: no negatives, InfoNCE = −log(1) = 0.
        let mut rng = LcgRng::new(9);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        let s = vec![0.2_f32; 8];
        let t = vec![0.4_f32; 6];
        let loss = crd_proj_loss(&s, &t, 1, &head).expect("loss");
        assert!(loss.abs() < 1e-5, "loss {loss}");
    }

    #[test]
    fn loss_dim_mismatch_errors() {
        let mut rng = LcgRng::new(4);
        let head = CrdProjectionHead::new(cfg(), &mut rng);
        let s = vec![0.0_f32; 5 * 8];
        let t = vec![0.0_f32; 4 * 6]; // wrong batch
        assert!(crd_proj_loss(&s, &t, 5, &head).is_err());
    }

    #[test]
    fn config_rejects_bad_params() {
        assert!(CrdProjConfig::new(0, 6, 16, 4, 0.1).is_err());
        assert!(CrdProjConfig::new(8, 6, 16, 4, 0.0).is_err());
    }
}
