//! DINOv2 — Oquab et al. 2023 — Self-supervised vision transformers with curated
//! data + KoLeo regularisation + iBOT patch-level loss + center-momentum
//! normalization.
//!
//! Key improvements over DINO:
//! * **Curated data** — automatic deduplication & quality filtering (handled
//!   upstream; not modelled here).
//! * **KoLeo regularisation** — encourages feature diversity by pushing each
//!   sample's nearest-neighbour farther away in cosine space.
//! * **iBOT patch loss** — masked image modelling on top of the DINO CLS loss
//!   (iBOT integration is available in `crate::clustering::ibot`).
//! * **Centre momentum** — EMA update to the teacher prototype assignments,
//!   preventing mode collapse without stop-gradient tricks.
//!
//! ```text
//!   scores = features @ prototypes^T          [N × K]
//!   p_s = softmax(scores / τ_s)               [N × K]
//!   p_t = softmax((scores − center) / τ_t)    [N × K]
//!   CE  = −mean_i Σ_k p_t[i,k] · log(p_s[i,k] + ε)
//!   KoLeo = −mean_i log(min_{j≠i} cosine_sim(f_i, f_j) + ε)
//!   Loss  = CE + λ_koleo · KoLeo
//! ```
//!
//! Reference: "DINOv2: Learning Robust Visual Features without Supervision",
//! Oquab et al., TMLR 2024.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// Convenience alias — same concrete type used throughout this module.
pub type SslRng = LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Hyper-parameters for DINOv2.
#[derive(Debug, Clone)]
pub struct DinoV2Config {
    /// Number of extra register / memory tokens appended to the sequence
    /// (Darcet et al. 2023; stored for bookkeeping — not used in CPU MLP path).
    pub n_register_tokens: usize,
    /// Feature dimension output by the backbone.
    pub d_model: usize,
    /// Number of prototype vectors used for the soft assignment.
    /// Must be ≥ 2.
    pub n_prototypes: usize,
    /// Student softmax temperature (default 0.1).
    pub temp_student: f32,
    /// Teacher softmax temperature (default 0.04, sharper than student).
    pub temp_teacher: f32,
    /// Weight λ applied to the KoLeo regularisation term (default 0.1).
    pub koleo_weight: f32,
    /// EMA momentum for the center vector (default 0.9).
    pub center_momentum: f32,
}

impl Default for DinoV2Config {
    fn default() -> Self {
        Self {
            n_register_tokens: 4,
            d_model: 64,
            n_prototypes: 8,
            temp_student: 0.1,
            temp_teacher: 0.04,
            koleo_weight: 0.1,
            center_momentum: 0.9,
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// DINOv2 model: prototype matrix + running teacher center.
#[derive(Debug)]
pub struct DinoV2 {
    /// Prototype weight matrix `[n_prototypes × d_model]`.
    prototypes: Vec<f32>,
    /// Running teacher center `[n_prototypes]` — updated via EMA.
    center: Vec<f32>,
    config: DinoV2Config,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Numerically stable row-wise softmax of `[n × k]` matrix at temperature `t`.
fn row_softmax_temp(scores: &[f32], n: usize, k: usize, t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        let row = &scores[i * k..(i + 1) * k];
        // Find row maximum for numerical stability
        let mut row_max = f32::NEG_INFINITY;
        for &v in row {
            let scaled = v / t;
            if scaled > row_max {
                row_max = scaled;
            }
        }
        // Compute exp(x/t - max) and their sum using f64 for precision
        let mut exps = Vec::with_capacity(k);
        let mut s = 0.0_f64;
        for &v in row {
            let e = ((v / t - row_max) as f64).exp();
            exps.push(e);
            s += e;
        }
        let inv = 1.0_f64 / s.max(1e-30_f64);
        for e in exps {
            out.push((e * inv) as f32);
        }
    }
    out
}

/// Compute the L2 norm of a slice, returning 0.0 if the slice is empty.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    let sq: f32 = v.iter().map(|x| x * x).sum();
    sq.sqrt()
}

/// Cosine similarity between two equal-length slices.
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    let denom = na * nb;
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

impl DinoV2 {
    // ─── Constructor ─────────────────────────────────────────────────────────

    /// Create a new [`DinoV2`] with randomly initialised prototypes and a zero
    /// center.
    ///
    /// # Errors
    /// - [`SslError::InvalidFeatureDim`] — `d_model == 0`.
    /// - [`SslError::NumPrototypesTooSmall`] — `n_prototypes < 2`.
    /// - [`SslError::InvalidTemperature`] — `temp_student <= 0` or not finite.
    /// - [`SslError::InvalidMomentum`] — `center_momentum` outside `[0, 1]`.
    pub fn new(config: DinoV2Config, rng: &mut SslRng) -> SslResult<Self> {
        if config.d_model == 0 {
            return Err(SslError::InvalidFeatureDim);
        }
        if config.n_prototypes < 2 {
            return Err(SslError::NumPrototypesTooSmall);
        }
        if !(config.temp_student.is_finite() && config.temp_student > 0.0) {
            return Err(SslError::InvalidTemperature {
                temp: config.temp_student,
            });
        }
        // Teacher temperature also validated for completeness
        if !(config.temp_teacher.is_finite() && config.temp_teacher > 0.0) {
            return Err(SslError::InvalidTemperature {
                temp: config.temp_teacher,
            });
        }
        if !(config.center_momentum.is_finite()
            && (0.0_f32..=1.0_f32).contains(&config.center_momentum))
        {
            return Err(SslError::InvalidMomentum {
                momentum: config.center_momentum,
            });
        }

        let p = config.n_prototypes;
        let d = config.d_model;

        // Xavier-uniform initialisation for prototype rows
        let scale = 1.0_f32 / (d as f32).sqrt();
        let mut prototypes = Vec::with_capacity(p * d);
        for _ in 0..p * d {
            prototypes.push((rng.next_f32() * 2.0 - 1.0) * scale);
        }

        let center = vec![0.0_f32; p];

        Ok(Self {
            prototypes,
            center,
            config,
        })
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Return the feature dimension.
    #[must_use]
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Return a shared reference to the current center vector.
    #[must_use]
    #[inline]
    pub fn center(&self) -> &[f32] {
        &self.center
    }

    // ─── Core operations ─────────────────────────────────────────────────────

    /// Compute prototype logit scores: `features @ prototypes^T`.
    ///
    /// `features` — `[n_samples × d_model]` row-major.
    ///
    /// Returns `[n_samples × n_prototypes]`.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when `features.len() != n_samples *
    ///   d_model`.
    /// - [`SslError::EmptyInput`] when `n_samples == 0`.
    pub fn compute_scores(&self, features: &[f32], n_samples: usize) -> SslResult<Vec<f32>> {
        let d = self.config.d_model;
        let p = self.config.n_prototypes;

        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * d;
        if features.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        // scores[i, k] = features[i, :] · prototypes[k, :]
        let mut scores = vec![0.0_f32; n_samples * p];
        for i in 0..n_samples {
            for k in 0..p {
                let mut dot = 0.0_f32;
                for j in 0..d {
                    dot += features[i * d + j] * self.prototypes[k * d + j];
                }
                scores[i * p + k] = dot;
            }
        }
        Ok(scores)
    }

    /// Compute student soft assignments via `softmax(scores / τ_student)`.
    ///
    /// `scores` — `[n_samples × n_prototypes]` row-major.
    ///
    /// Returns `[n_samples × n_prototypes]` — each row sums to 1.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when `scores.len() != n_samples *
    ///   n_prototypes`.
    pub fn student_probs(&self, scores: &[f32], n_samples: usize) -> SslResult<Vec<f32>> {
        let p = self.config.n_prototypes;
        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * p;
        if scores.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: scores.len(),
            });
        }
        Ok(row_softmax_temp(
            scores,
            n_samples,
            p,
            self.config.temp_student,
        ))
    }

    /// Compute teacher soft assignments:
    /// `softmax((scores − center) / τ_teacher)`.
    ///
    /// `scores` — `[n_samples × n_prototypes]` row-major.
    ///
    /// Returns `[n_samples × n_prototypes]` — each row sums to 1.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when shapes mismatch.
    pub fn teacher_probs(&self, scores: &[f32], n_samples: usize) -> SslResult<Vec<f32>> {
        let p = self.config.n_prototypes;
        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * p;
        if scores.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: scores.len(),
            });
        }
        // Subtract center before softmax
        let mut centred = scores.to_vec();
        for i in 0..n_samples {
            for k in 0..p {
                centred[i * p + k] -= self.center[k];
            }
        }
        Ok(row_softmax_temp(
            &centred,
            n_samples,
            p,
            self.config.temp_teacher,
        ))
    }

    /// Cross-entropy loss between student and teacher soft assignments.
    ///
    /// `CE = −mean_i Σ_k p_t[i,k] · log(p_s[i,k] + ε)`
    ///
    /// `student_scores` and `teacher_scores` are raw score matrices
    /// `[n_samples × n_prototypes]`; probabilities are computed internally.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when shapes mismatch.
    pub fn dino_v2_loss(
        &self,
        student_scores: &[f32],
        teacher_scores: &[f32],
        n_samples: usize,
    ) -> SslResult<f32> {
        let p = self.config.n_prototypes;
        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * p;
        if student_scores.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: student_scores.len(),
            });
        }
        if teacher_scores.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: teacher_scores.len(),
            });
        }

        let p_s = self.student_probs(student_scores, n_samples)?;
        let p_t = self.teacher_probs(teacher_scores, n_samples)?;

        let mut total = 0.0_f64;
        for i in 0..n_samples {
            for k in 0..p {
                let log_ps = ((p_s[i * p + k] as f64) + 1e-8_f64).ln();
                total -= (p_t[i * p + k] as f64) * log_ps;
            }
        }
        Ok((total / n_samples as f64) as f32)
    }

    /// EMA update of the center: `center = m * center + (1-m) * mean(p_t)`.
    ///
    /// `teacher_scores` — raw prototype logit scores `[n_samples × n_prototypes]`.
    /// The mean is computed over teacher soft assignments (i.e. after centring +
    /// teacher softmax), so the updated center tracks the expected prototype
    /// occupancy under the teacher distribution.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when `teacher_scores.len() != n_samples *
    ///   n_prototypes`.
    pub fn update_center(&mut self, teacher_scores: &[f32], n_samples: usize) -> SslResult<()> {
        let p = self.config.n_prototypes;
        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * p;
        if teacher_scores.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: teacher_scores.len(),
            });
        }

        let p_t = self.teacher_probs(teacher_scores, n_samples)?;
        let m = self.config.center_momentum;
        let inv_n = 1.0_f32 / n_samples as f32;

        for k in 0..p {
            let mut mean_k = 0.0_f32;
            for i in 0..n_samples {
                mean_k += p_t[i * p + k];
            }
            mean_k *= inv_n;
            self.center[k] = m * self.center[k] + (1.0 - m) * mean_k;
        }
        Ok(())
    }

    /// KoLeo regularisation loss.
    ///
    /// For each sample `i`, find the minimum cosine similarity to any other
    /// sample `j ≠ i`, then return `−mean_i log(min_sim_i + ε)`.
    ///
    /// A higher loss means samples are clustered (low diversity); pushing it
    /// down spreads features uniformly on the hypersphere.
    ///
    /// If `n_samples < 2`, returns `0.0` (no pairs to evaluate).
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when `features.len() != n_samples *
    ///   d_model`.
    pub fn koleo_loss(&self, features: &[f32], n_samples: usize) -> SslResult<f32> {
        let d = self.config.d_model;
        if n_samples == 0 {
            return Err(SslError::EmptyInput);
        }
        let expected = n_samples * d;
        if features.len() != expected {
            return Err(SslError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        if n_samples < 2 {
            return Ok(0.0);
        }

        let mut total = 0.0_f64;
        for i in 0..n_samples {
            let fi = &features[i * d..(i + 1) * d];
            let mut min_sim = f32::INFINITY;
            for j in 0..n_samples {
                if j == i {
                    continue;
                }
                let fj = &features[j * d..(j + 1) * d];
                let s = cosine_sim(fi, fj);
                if s < min_sim {
                    min_sim = s;
                }
            }
            // Shift to [0, 2] range by adding 1 before log to keep argument > 0
            // Standard KoLeo uses min cosine distance = 1 - sim; we use sim + ε.
            let sim_shifted = (min_sim + 1.0).max(0.0);
            total -= (sim_shifted as f64 + 1e-8_f64).ln();
        }
        Ok((total / n_samples as f64) as f32)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DinoV2Config {
        DinoV2Config {
            n_register_tokens: 4,
            d_model: 8,
            n_prototypes: 4,
            temp_student: 0.1,
            temp_teacher: 0.04,
            koleo_weight: 0.1,
            center_momentum: 0.9,
        }
    }

    fn make_rng() -> SslRng {
        LcgRng::new(99)
    }

    // ── 1. compute_scores_shape ───────────────────────────────────────────────
    #[test]
    fn compute_scores_shape() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n = 6_usize;
        let feats = vec![0.3_f32; n * d];
        let scores = model
            .compute_scores(&feats, n)
            .expect("compute_scores should succeed");
        assert_eq!(scores.len(), n * default_config().n_prototypes);
    }

    // ── 2. student_probs_sum_to_one ───────────────────────────────────────────
    #[test]
    fn student_probs_sum_to_one() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let p = default_config().n_prototypes;
        let n = 5_usize;
        let scores: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.17).sin()).collect();
        let probs = model
            .student_probs(&scores, n)
            .expect("student_probs should succeed");
        for i in 0..n {
            let row_sum: f32 = probs[i * p..(i + 1) * p].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sum = {row_sum}");
        }
    }

    // ── 3. teacher_probs_sum_to_one ───────────────────────────────────────────
    #[test]
    fn teacher_probs_sum_to_one() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let p = default_config().n_prototypes;
        let n = 5_usize;
        let scores: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.23).cos()).collect();
        let probs = model
            .teacher_probs(&scores, n)
            .expect("teacher_probs should succeed");
        for i in 0..n {
            let row_sum: f32 = probs[i * p..(i + 1) * p].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sum = {row_sum}");
        }
    }

    // ── 4. dino_v2_loss_finite ────────────────────────────────────────────────
    #[test]
    fn dino_v2_loss_finite() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let p = default_config().n_prototypes;
        let n = 8_usize;
        let student_scores: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.09).sin()).collect();
        let teacher_scores: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.11).cos()).collect();
        let loss = model
            .dino_v2_loss(&student_scores, &teacher_scores, n)
            .expect("value should be present");
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(loss > 0.0, "cross-entropy must be positive, got {loss}");
    }

    // ── 5. update_center_changes_center ──────────────────────────────────────
    #[test]
    fn update_center_changes_center() {
        let mut rng = make_rng();
        let mut model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let p = default_config().n_prototypes;
        let n = 4_usize;
        let scores: Vec<f32> = (0..n * p).map(|i| (i as f32 * 0.31).sin() + 1.0).collect();
        let center_before = model.center().to_vec();
        model
            .update_center(&scores, n)
            .expect("update_center should succeed");
        let center_after = model.center().to_vec();
        // At least one element should differ
        let changed = center_before
            .iter()
            .zip(center_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(changed, "center should change after update");
    }

    // ── 6. d_model_zero_error ────────────────────────────────────────────────
    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.d_model = 0;
        let result = DinoV2::new(cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SslError::InvalidFeatureDim));
    }

    // ── 7. n_prototypes_too_small_error ──────────────────────────────────────
    #[test]
    fn n_prototypes_too_small_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.n_prototypes = 1;
        let result = DinoV2::new(cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SslError::NumPrototypesTooSmall
        ));
    }

    // ── 8. koleo_loss_nonneg ──────────────────────────────────────────────────
    #[test]
    fn koleo_loss_nonneg() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let n = 6_usize;
        let feats: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.19).sin()).collect();
        let loss = model
            .koleo_loss(&feats, n)
            .expect("koleo_loss should succeed");
        assert!(loss >= 0.0, "KoLeo loss must be non-negative, got {loss}");
        assert!(loss.is_finite(), "KoLeo loss must be finite, got {loss}");
    }

    // ── 9. invalid_temperature_error ─────────────────────────────────────────
    #[test]
    fn invalid_temperature_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.temp_student = 0.0;
        let result = DinoV2::new(cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SslError::InvalidTemperature { .. }
        ));
    }

    // ── 10. invalid_momentum_error ────────────────────────────────────────────
    #[test]
    fn invalid_momentum_error() {
        let mut rng = make_rng();
        let mut cfg = default_config();
        cfg.center_momentum = 1.5;
        let result = DinoV2::new(cfg, &mut rng);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SslError::InvalidMomentum { .. }
        ));
    }

    // ── 11. koleo_single_sample_returns_zero ─────────────────────────────────
    #[test]
    fn koleo_single_sample_returns_zero() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        let d = model.d_model();
        let feats = vec![0.5_f32; d];
        let loss = model
            .koleo_loss(&feats, 1)
            .expect("koleo_loss should succeed");
        assert_eq!(loss, 0.0);
    }

    // ── 12. compute_scores_dim_mismatch_error ─────────────────────────────────
    #[test]
    fn compute_scores_dim_mismatch_error() {
        let mut rng = make_rng();
        let model = DinoV2::new(default_config(), &mut rng).expect("value should be present");
        // Supply wrong number of features
        let feats = vec![0.1_f32; 3]; // should be n_samples * d_model
        let result = model.compute_scores(&feats, 2);
        assert!(matches!(
            result.unwrap_err(),
            SslError::DimensionMismatch { .. }
        ));
    }
}
