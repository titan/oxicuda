//! SimKD — Knowledge Distillation with the Reused Teacher Classifier
//! (Chen et al. 2022).
//!
//! Reference: Chen, D., Mei, J.-P., Zhang, H., Wang, C., Feng, Y., & Chen, C.
//! (2022). *Knowledge Distillation with the Reused Teacher Classifier*. CVPR 2022.
//! <https://arxiv.org/abs/2203.14001>
//!
//! # Idea
//!
//! Classic feature distillation aligns intermediate features and *also* trains a
//! fresh student classifier head. SimKD observes that the teacher's final
//! classifier already encodes a strong decision boundary, so it **freezes and
//! reuses the teacher classifier directly on the student**. Only a lightweight
//! **projector** has to be learned: it maps the student's penultimate feature
//! `f_s ∈ ℝ^{d_s}` into the teacher's feature space `ℝ^{d_t}`, after which the
//! frozen teacher classifier `W_t · · + b_t` produces the logits.
//!
//! ```text
//!   p = Proj(f_s)                         (d_s → d_t, optional bottleneck)
//!   L_simkd = ‖ p − f_t ‖²₂ / d_t         (feature-alignment distillation loss)
//!   logits  = W_t · p + b_t               (frozen teacher classifier reused)
//! ```
//!
//! The projector is a `1×1`-convolution-equivalent linear map (optionally with a
//! small bottleneck `d_s → bottleneck → d_t`, mirroring the paper's
//! bottleneck-style projector). The teacher classifier weights are **never**
//! modified by the distillation loss or by a projector update — they stay frozen.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// The frozen teacher classifier head reused by SimKD.
///
/// Stores `weight ∈ ℝ^{n_classes × feat_dim}` (row-major) and `bias ∈ ℝ^{n_classes}`.
/// SimKD never updates these — they are loaded from the (already-trained) teacher
/// and applied to the projected student features.
#[derive(Debug, Clone)]
pub struct TeacherClassifier {
    /// Teacher feature dimension (classifier input size).
    pub feat_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
    /// Weight matrix `[n_classes × feat_dim]`, row-major.
    pub weight: Vec<f32>,
    /// Bias vector `[n_classes]`.
    pub bias: Vec<f32>,
}

impl TeacherClassifier {
    /// Construct a teacher classifier from explicit (frozen) weights and bias.
    ///
    /// # Errors
    ///
    /// - [`DistillError::EmptyInput`] if `feat_dim == 0` or `n_classes == 0`.
    /// - [`DistillError::DimensionMismatch`] if `weight.len() != n_classes *
    ///   feat_dim` or `bias.len() != n_classes`.
    pub fn new(
        feat_dim: usize,
        n_classes: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> DistillResult<Self> {
        if feat_dim == 0 || n_classes == 0 {
            return Err(DistillError::EmptyInput);
        }
        if weight.len() != n_classes * feat_dim {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes * feat_dim,
                got: weight.len(),
            });
        }
        if bias.len() != n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes,
                got: bias.len(),
            });
        }
        Ok(Self {
            feat_dim,
            n_classes,
            weight,
            bias,
        })
    }

    /// Apply the frozen classifier to a single `feat_dim`-length feature vector,
    /// producing `n_classes` logits.
    ///
    /// # Errors
    ///
    /// [`DistillError::DimensionMismatch`] if `feat.len() != feat_dim`.
    pub fn forward(&self, feat: &[f32]) -> DistillResult<Vec<f32>> {
        if feat.len() != self.feat_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.feat_dim,
                got: feat.len(),
            });
        }
        let mut logits = vec![0.0_f32; self.n_classes];
        for (c, lv) in logits.iter_mut().enumerate() {
            let w_row = &self.weight[c * self.feat_dim..(c + 1) * self.feat_dim];
            let dot: f32 = feat.iter().zip(w_row.iter()).map(|(&x, &w)| x * w).sum();
            *lv = dot + self.bias[c];
        }
        Ok(logits)
    }
}

/// SimKD projector mapping student penultimate features to the teacher space.
///
/// Two linear layers form an optional bottleneck `d_s → bottleneck → d_t`. When
/// `bottleneck == d_t` and the second layer is identity-like this reduces to a
/// single `d_s → d_t` `1×1` projection, matching the paper's lightweight design.
/// All weights are learnable; only the projector is trained — the teacher
/// classifier stays frozen.
#[derive(Debug, Clone)]
pub struct SimKdProjector {
    /// Student (input) feature dimension.
    pub in_dim: usize,
    /// Bottleneck width.
    pub bottleneck: usize,
    /// Teacher (output) feature dimension.
    pub out_dim: usize,
    /// First layer weights `[bottleneck × in_dim]`, row-major.
    pub w1: Vec<f32>,
    /// Second layer weights `[out_dim × bottleneck]`, row-major.
    pub w2: Vec<f32>,
}

impl SimKdProjector {
    /// Construct a projector with He-style normal initialisation.
    ///
    /// # Errors
    ///
    /// [`DistillError::EmptyInput`] if any dimension is zero.
    pub fn new(
        in_dim: usize,
        bottleneck: usize,
        out_dim: usize,
        rng: &mut LcgRng,
    ) -> DistillResult<Self> {
        if in_dim == 0 || bottleneck == 0 || out_dim == 0 {
            return Err(DistillError::EmptyInput);
        }
        let s1 = 1.0 / (in_dim as f32).sqrt();
        let s2 = 1.0 / (bottleneck as f32).sqrt();
        let mut w1 = vec![0.0_f32; bottleneck * in_dim];
        for v in w1.iter_mut() {
            *v = rng.next_normal() * s1;
        }
        let mut w2 = vec![0.0_f32; out_dim * bottleneck];
        for v in w2.iter_mut() {
            *v = rng.next_normal() * s2;
        }
        Ok(Self {
            in_dim,
            bottleneck,
            out_dim,
            w1,
            w2,
        })
    }

    /// Construct an *exactly-aligning* projector for a no-bottleneck `d → d` case
    /// where the desired output equals the input (identity). Useful for testing
    /// the perfect-alignment limit. Both layers are identity matrices, so
    /// `forward(x) == x`.
    ///
    /// # Errors
    ///
    /// [`DistillError::EmptyInput`] if `dim == 0`.
    pub fn identity(dim: usize) -> DistillResult<Self> {
        if dim == 0 {
            return Err(DistillError::EmptyInput);
        }
        let mut w1 = vec![0.0_f32; dim * dim];
        let mut w2 = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            w1[i * dim + i] = 1.0;
            w2[i * dim + i] = 1.0;
        }
        Ok(Self {
            in_dim: dim,
            bottleneck: dim,
            out_dim: dim,
            w1,
            w2,
        })
    }

    /// Project a single `in_dim`-length student feature into the teacher space,
    /// returning an `out_dim`-length vector.
    ///
    /// # Errors
    ///
    /// [`DistillError::DimensionMismatch`] if `feat.len() != in_dim`.
    pub fn forward(&self, feat: &[f32]) -> DistillResult<Vec<f32>> {
        if feat.len() != self.in_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.in_dim,
                got: feat.len(),
            });
        }
        // First layer: in_dim → bottleneck.
        let mut hidden = vec![0.0_f32; self.bottleneck];
        for (h, hv) in hidden.iter_mut().enumerate() {
            let w_row = &self.w1[h * self.in_dim..(h + 1) * self.in_dim];
            *hv = feat.iter().zip(w_row.iter()).map(|(&x, &w)| x * w).sum();
        }
        // Second layer: bottleneck → out_dim.
        let mut out = vec![0.0_f32; self.out_dim];
        for (o, ov) in out.iter_mut().enumerate() {
            let w_row = &self.w2[o * self.bottleneck..(o + 1) * self.bottleneck];
            *ov = hidden.iter().zip(w_row.iter()).map(|(&x, &w)| x * w).sum();
        }
        Ok(out)
    }

    /// Number of learnable parameters in the projector.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.w2.len()
    }
}

/// Squared-L2 SimKD alignment loss between a projected student feature and the
/// teacher feature, normalised by the teacher feature dimension.
///
/// `L = ‖ projected − teacher ‖²₂ / d_t`.
///
/// # Errors
///
/// - [`DistillError::EmptyInput`] if either input is empty.
/// - [`DistillError::DimensionMismatch`] if the lengths differ.
/// - [`DistillError::NumericalError`] if the result is non-finite.
pub fn simkd_loss(projected: &[f32], teacher: &[f32]) -> DistillResult<f32> {
    if projected.is_empty() || teacher.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if projected.len() != teacher.len() {
        return Err(DistillError::DimensionMismatch {
            expected: projected.len(),
            got: teacher.len(),
        });
    }
    let sq: f32 = projected
        .iter()
        .zip(teacher.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum();
    let loss = sq / projected.len() as f32;
    if !loss.is_finite() {
        return Err(DistillError::NumericalError {
            msg: "SimKD loss produced a non-finite value".into(),
        });
    }
    Ok(loss)
}

/// Full SimKD forward pass: project the student feature, compute the alignment
/// loss against the teacher feature, and produce the reused-classifier logits.
///
/// Returns `(loss, logits)` where `logits` come from the **frozen** teacher
/// classifier applied to the *projected* student feature.
///
/// # Errors
///
/// - [`DistillError::DimensionMismatch`] if the projector output dimension does
///   not match the classifier feature dimension, or the teacher feature length
///   does not match the projector output.
pub fn simkd_forward(
    projector: &SimKdProjector,
    classifier: &TeacherClassifier,
    student_feat: &[f32],
    teacher_feat: &[f32],
) -> DistillResult<(f32, Vec<f32>)> {
    if projector.out_dim != classifier.feat_dim {
        return Err(DistillError::DimensionMismatch {
            expected: classifier.feat_dim,
            got: projector.out_dim,
        });
    }
    let projected = projector.forward(student_feat)?;
    if projected.len() != teacher_feat.len() {
        return Err(DistillError::DimensionMismatch {
            expected: projected.len(),
            got: teacher_feat.len(),
        });
    }
    let loss = simkd_loss(&projected, teacher_feat)?;
    let logits = classifier.forward(&projected)?;
    Ok((loss, logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classifier(feat_dim: usize, n_classes: usize) -> TeacherClassifier {
        let mut rng = LcgRng::new(123);
        let mut w = vec![0.0_f32; n_classes * feat_dim];
        for v in w.iter_mut() {
            *v = rng.next_normal() * 0.5;
        }
        let mut b = vec![0.0_f32; n_classes];
        for v in b.iter_mut() {
            *v = rng.next_normal() * 0.1;
        }
        TeacherClassifier::new(feat_dim, n_classes, w, b).expect("valid classifier")
    }

    // (a) Projector maps student-dim → teacher-dim (shape).
    #[test]
    fn projector_maps_student_to_teacher_dim() {
        let mut rng = LcgRng::new(7);
        let d_s = 6;
        let d_t = 10;
        let proj = SimKdProjector::new(d_s, 4, d_t, &mut rng).expect("ok");
        let f_s: Vec<f32> = (0..d_s).map(|i| i as f32 * 0.3 - 0.5).collect();
        let out = proj.forward(&f_s).expect("forward ok");
        assert_eq!(out.len(), d_t, "projected feature must have teacher dim");
    }

    // (b) Frozen classifier on projected features yields valid logits (= n_classes).
    #[test]
    fn reused_classifier_logits_shape() {
        let mut rng = LcgRng::new(8);
        let d_s = 5;
        let d_t = 8;
        let n_classes = 7;
        let proj = SimKdProjector::new(d_s, d_t, d_t, &mut rng).expect("ok");
        let clf = make_classifier(d_t, n_classes);
        let f_s: Vec<f32> = (0..d_s).map(|i| (i as f32).sin()).collect();
        let projected = proj.forward(&f_s).expect("ok");
        let logits = clf.forward(&projected).expect("ok");
        assert_eq!(
            logits.len(),
            n_classes,
            "logits length must equal n_classes"
        );
        assert!(logits.iter().all(|v| v.is_finite()), "logits finite");
    }

    // (c) L2 projection loss ≥ 0 and DECREASES as the projector aligns better.
    #[test]
    fn loss_nonneg_and_decreases_with_alignment() {
        let d_t = 4;
        let teacher: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        // A poorly-aligned projection (far from teacher).
        let proj_far: Vec<f32> = vec![-3.0, 4.0, -1.0, 0.0];
        // A better-aligned projection (closer to teacher).
        let proj_near: Vec<f32> = vec![1.2, -1.8, 0.4, 2.7];
        let l_far = simkd_loss(&proj_far, &teacher).expect("ok");
        let l_near = simkd_loss(&proj_near, &teacher).expect("ok");
        assert!(l_far >= 0.0 && l_near >= 0.0, "losses must be non-negative");
        assert!(
            l_near < l_far,
            "closer alignment must reduce loss: near={l_near} far={l_far}"
        );
        let _ = d_t;
    }

    // (d) Perfect projector (projected == teacher) → ~0 loss.
    #[test]
    fn perfect_projection_zero_loss() {
        let teacher: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let perfect = teacher.clone();
        let loss = simkd_loss(&perfect, &teacher).expect("ok");
        assert!(
            loss < 1e-12,
            "perfect projection must give ~0 loss, got {loss}"
        );
    }

    // (d') Identity projector on a feature equal to teacher → ~0 SimKD loss end-to-end.
    #[test]
    fn identity_projector_perfect_alignment() {
        let dim = 4;
        let proj = SimKdProjector::identity(dim).expect("ok");
        let clf = make_classifier(dim, 3);
        let feat: Vec<f32> = vec![0.7, -0.3, 1.1, 0.05];
        // Teacher feature equals the (identity-)projected student feature.
        let teacher = proj.forward(&feat).expect("ok");
        let (loss, logits) = simkd_forward(&proj, &clf, &feat, &teacher).expect("ok");
        assert!(
            loss < 1e-10,
            "identity perfect alignment loss ~0, got {loss}"
        );
        assert_eq!(logits.len(), 3);
    }

    // (e) Teacher classifier weights are NOT modified by the loss / forward pass (frozen).
    #[test]
    fn teacher_classifier_frozen() {
        let mut rng = LcgRng::new(9);
        let d_s = 4;
        let d_t = 6;
        let n_classes = 5;
        let proj = SimKdProjector::new(d_s, d_t, d_t, &mut rng).expect("ok");
        let clf = make_classifier(d_t, n_classes);
        let weight_before = clf.weight.clone();
        let bias_before = clf.bias.clone();
        let f_s: Vec<f32> = (0..d_s).map(|i| i as f32 + 0.1).collect();
        let teacher: Vec<f32> = (0..d_t).map(|i| (i as f32) * 0.2).collect();
        // Running the full pass (loss + reused-classifier logits) repeatedly must
        // leave the teacher classifier completely unchanged.
        for _ in 0..5 {
            let _ = simkd_forward(&proj, &clf, &f_s, &teacher).expect("ok");
        }
        assert_eq!(
            clf.weight, weight_before,
            "classifier weights must be frozen"
        );
        assert_eq!(clf.bias, bias_before, "classifier bias must be frozen");
    }

    // (f) Gradient / finiteness sanity: a finite-difference step on the projector
    //     output reduces the loss in the descent direction, and all values finite.
    #[test]
    fn loss_descent_direction_finite() {
        let teacher: Vec<f32> = vec![2.0, -1.0, 0.0, 1.5];
        let mut p: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        // Analytic gradient of ‖p−t‖²/d wrt p is 2(p−t)/d. Step opposite the
        // gradient and confirm the loss drops monotonically.
        let mut prev = simkd_loss(&p, &teacher).expect("ok");
        // Gradient of ‖p−t‖²/d wrt p is 2(p−t)/d. With lr = d/2 the per-step
        // contraction factor is (1 − lr·2/d·½) = 0 in one step for the isolated
        // coordinate; use a safe lr that contracts quickly and stays finite.
        let lr = (p.len() as f32) / 2.0;
        for _ in 0..40 {
            let grad: Vec<f32> = p
                .iter()
                .zip(teacher.iter())
                .map(|(&pi, &ti)| 2.0 * (pi - ti) / p.len() as f32)
                .collect();
            for (pi, gi) in p.iter_mut().zip(grad.iter()) {
                *pi -= lr * gi;
            }
            let cur = simkd_loss(&p, &teacher).expect("ok");
            assert!(cur.is_finite(), "loss must stay finite");
            assert!(
                cur <= prev + 1e-7,
                "gradient descent must not increase loss"
            );
            prev = cur;
        }
        assert!(
            prev < 1e-3,
            "after descent the loss should be small, got {prev}"
        );
    }

    // Construction validation.
    #[test]
    fn classifier_rejects_bad_shapes() {
        assert!(matches!(
            TeacherClassifier::new(0, 3, vec![], vec![]),
            Err(DistillError::EmptyInput)
        ));
        assert!(matches!(
            TeacherClassifier::new(2, 3, vec![1.0; 5], vec![0.0; 3]),
            Err(DistillError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            TeacherClassifier::new(2, 3, vec![1.0; 6], vec![0.0; 2]),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn projector_num_params() {
        let mut rng = LcgRng::new(3);
        let proj = SimKdProjector::new(4, 2, 6, &mut rng).expect("ok");
        // w1: 2*4 = 8, w2: 6*2 = 12 → 20.
        assert_eq!(proj.num_params(), 20);
    }

    #[test]
    fn forward_dim_mismatch_errors() {
        let mut rng = LcgRng::new(4);
        let proj = SimKdProjector::new(4, 2, 6, &mut rng).expect("ok");
        // Classifier feat_dim (5) != projector out_dim (6).
        let clf = make_classifier(5, 3);
        let f_s = vec![0.0_f32; 4];
        let teacher = vec![0.0_f32; 6];
        assert!(matches!(
            simkd_forward(&proj, &clf, &f_s, &teacher),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }
}
