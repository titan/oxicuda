//! DyLoRA — Dynamic Low-Rank Adaptation with nested, search-free rank training.
//!
//! Reference: Valipour, M., Rezagholizadeh, M., Kobyzev, I., & Ghodsi, A. (2023).
//! *DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free
//! Low-Rank Adaptation*. EACL 2023. <https://arxiv.org/abs/2210.07558>
//!
//! A vanilla LoRA layer fixes the rank `r` up front, so finding the best rank means training
//! several models. DyLoRA instead trains a *single* adapter that is robust across a whole
//! range of ranks `[r_min, r_max]`. The factors are organised so that the first `b` rows of
//! `A` and the first `b` columns of `B` form a self-contained rank-`b` adapter (a *nested*
//! decomposition). At each training step a rank `b` is sampled uniformly from the range and
//! only that truncation is exercised, so after training the layer can be deployed at any rank
//! `b ∈ [r_min, r_max]` without retraining:
//!
//! ```text
//!   forward_b(x) = W₀·x + (α / b) · B[:, :b] · A[:b, :] · x
//! ```
//!
//! The active-rank scaling `α / b` matches the standard LoRA scale at every truncation, so at
//! `b = r_max` the layer is identical to a plain LoRA adapter of rank `r_max`.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;
use crate::lora::lora::mat_vec_mul;

/// Hyper-parameter bundle for a [`DyLoraLinear`].
#[derive(Debug, Clone)]
pub struct DyLoraConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Maximum (allocated) rank `r_max`.
    pub r_max: usize,
    /// Minimum deployable rank `r_min` (`1 ≤ r_min ≤ r_max`).
    pub r_min: usize,
    /// LoRA scaling factor `α`; the effective scale at rank `b` is `α / b`.
    pub alpha: f32,
    /// Standard deviation used to initialise the `A` factor.
    pub init_scale: f32,
}

impl DyLoraConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] when a dimension is zero.
    /// - [`PeftError::RankTooLarge`] when `r_max > min(in_features, out_features)`.
    /// - [`PeftError::InvalidTargetRank`] when `r_min` is zero or exceeds `r_max`.
    pub fn validate(&self) -> PeftResult<()> {
        if self.in_features == 0 || self.out_features == 0 || self.r_max == 0 {
            return Err(PeftError::EmptyInput);
        }
        let dim = self.in_features.min(self.out_features);
        if self.r_max > dim {
            return Err(PeftError::RankTooLarge {
                rank: self.r_max,
                dim,
            });
        }
        if self.r_min == 0 || self.r_min > self.r_max {
            return Err(PeftError::InvalidTargetRank {
                target_r: self.r_min,
                r: self.r_max,
            });
        }
        Ok(())
    }
}

/// DyLoRA adapter holding rank-`r_max` factors that can be evaluated at any nested rank.
///
/// `A` shape: `[r_max × in_features]`; `B` shape: `[out_features × r_max]`.
#[derive(Debug, Clone)]
pub struct DyLoraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Maximum (allocated) rank.
    pub r_max: usize,
    /// Minimum deployable rank.
    pub r_min: usize,
    /// LoRA scaling factor `α`.
    pub alpha: f32,
    /// Frozen base weight, row-major `[out_features × in_features]`.
    pub w: Vec<f32>,
    /// Down-projection `A`, row-major `[r_max × in_features]`.
    pub a: Vec<f32>,
    /// Up-projection `B`, row-major `[out_features × r_max]`.
    pub b: Vec<f32>,
}

impl DyLoraLinear {
    /// Build a fresh DyLoRA adapter.
    ///
    /// `W₀` is zero-initialised; `A ~ N(0, init_scale²)`; `B` is zero (so the initial delta
    /// is zero, matching the LoRA convention).
    ///
    /// # Errors
    ///
    /// Forwards [`DyLoraConfig::validate`].
    pub fn new(cfg: DyLoraConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        cfg.validate()?;
        let mut a = vec![0.0_f32; cfg.r_max * cfg.in_features];
        rng.fill_normal(&mut a);
        for v in a.iter_mut() {
            *v *= cfg.init_scale;
        }
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            r_max: cfg.r_max,
            r_min: cfg.r_min,
            alpha: cfg.alpha,
            w: vec![0.0_f32; cfg.out_features * cfg.in_features],
            a,
            b: vec![0.0_f32; cfg.out_features * cfg.r_max],
        })
    }

    /// Effective scale `α / b` at active rank `b` (returns `0.0` for `b == 0`).
    #[must_use]
    pub fn scale_at(&self, b: usize) -> f32 {
        if b == 0 { 0.0 } else { self.alpha / b as f32 }
    }

    /// Forward pass at the full rank `r_max`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        self.forward_at_rank(x, self.r_max)
    }

    /// Forward pass at the nested rank `b`: `W₀·x + (α/b)·B[:, :b]·A[:b, :]·x`.
    ///
    /// # Errors
    ///
    /// - [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    /// - [`PeftError::InvalidTargetRank`] when `b == 0` or `b > r_max`.
    pub fn forward_at_rank(&self, x: &[f32], b: usize) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        self.check_rank(b)?;
        let in_f = self.in_features;
        // Base projection W₀·x.
        let mut y = mat_vec_mul(&self.w, x, self.out_features, in_f);
        // t = A[:b, :] · x  (first b rows of A are contiguous in row-major storage).
        let t = mat_vec_mul(&self.a[0..b * in_f], x, b, in_f);
        let scale = self.scale_at(b);
        // y += scale · B[:, :b] · t  (first b columns of each B row).
        for (i, y_i) in y.iter_mut().enumerate() {
            let row = i * self.r_max;
            let mut acc = 0.0_f32;
            for (k, t_k) in t.iter().enumerate() {
                acc += self.b[row + k] * t_k;
            }
            *y_i += scale * acc;
        }
        Ok(y)
    }

    /// Sample a deployment rank uniformly from `[r_min, r_max]` (the DyLoRA per-step draw).
    pub fn sample_rank(&self, rng: &mut LcgRng) -> usize {
        let span = self.r_max - self.r_min + 1;
        self.r_min + rng.next_usize(span)
    }

    /// Effective delta at nested rank `b`: `(α/b)·B[:, :b]·A[:b, :]`,
    /// row-major `[out_features × in_features]`.
    ///
    /// # Errors
    ///
    /// [`PeftError::InvalidTargetRank`] when `b == 0` or `b > r_max`.
    pub fn truncated_delta(&self, b: usize) -> PeftResult<Vec<f32>> {
        self.check_rank(b)?;
        let in_f = self.in_features;
        let out_f = self.out_features;
        let scale = self.scale_at(b);
        let mut delta = vec![0.0_f32; out_f * in_f];
        for i in 0..out_f {
            let brow = i * self.r_max;
            for k in 0..b {
                let b_ik = scale * self.b[brow + k];
                if b_ik == 0.0 {
                    continue;
                }
                let arow = k * in_f;
                for j in 0..in_f {
                    delta[i * in_f + j] += b_ik * self.a[arow + j];
                }
            }
        }
        Ok(delta)
    }

    /// Merge the rank-`b` adapter into the base weight: `W₀ += (α/b)·B[:, :b]·A[:b, :]`.
    ///
    /// # Errors
    ///
    /// [`PeftError::InvalidTargetRank`] when `b == 0` or `b > r_max`.
    pub fn merge_at_rank(&mut self, b: usize) -> PeftResult<()> {
        let delta = self.truncated_delta(b)?;
        for (w, d) in self.w.iter_mut().zip(delta.iter()) {
            *w += d;
        }
        Ok(())
    }

    fn check_rank(&self, b: usize) -> PeftResult<()> {
        if b == 0 || b > self.r_max {
            return Err(PeftError::InvalidTargetRank {
                target_r: b,
                r: self.r_max,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::lora::LoraLinear;

    fn cfg() -> DyLoraConfig {
        DyLoraConfig {
            in_features: 6,
            out_features: 5,
            r_max: 4,
            r_min: 1,
            alpha: 8.0,
            init_scale: 0.05,
        }
    }

    fn filled(seed: u64) -> DyLoraLinear {
        let mut rng = LcgRng::new(seed);
        let mut d = DyLoraLinear::new(cfg(), &mut rng)
            .expect("DyLoraLinear::new should succeed with valid config");
        // Give W and B non-trivial values so forwards are meaningful.
        for (i, w) in d.w.iter_mut().enumerate() {
            *w = (i as f32 % 7.0) * 0.02 - 0.05;
        }
        rng.fill_normal(&mut d.b);
        for v in d.b.iter_mut() {
            *v *= 0.1;
        }
        d
    }

    #[test]
    fn nested_truncation_ignores_higher_slices() {
        let d = filled(1);
        let x: Vec<f32> = (0..d.in_features).map(|i| (i as f32 - 3.0) * 0.3).collect();
        let b = 2;
        let y_ref = d
            .forward_at_rank(&x, b)
            .expect("forward_at_rank should succeed with valid rank and input");
        // Zero out the A rows and B columns of rank ≥ b; the rank-b forward must not change.
        let mut d2 = d.clone();
        for k in b..d2.r_max {
            for j in 0..d2.in_features {
                d2.a[k * d2.in_features + j] = 0.0;
            }
            for i in 0..d2.out_features {
                d2.b[i * d2.r_max + k] = 0.0;
            }
        }
        let y_trunc = d2
            .forward_at_rank(&x, b)
            .expect("forward_at_rank should succeed on the zeroed-out copy");
        for (lhs, rhs) in y_ref.iter().zip(y_trunc.iter()) {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "rank-{b} forward must depend only on the first {b} ranks: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn full_rank_matches_plain_lora() {
        let d = filled(2);
        let lora = LoraLinear {
            in_features: d.in_features,
            out_features: d.out_features,
            rank: d.r_max,
            scale: d.alpha / d.r_max as f32,
            w: d.w.clone(),
            a: d.a.clone(),
            b: d.b.clone(),
        };
        let x: Vec<f32> = (0..d.in_features).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let y_dy = d
            .forward(&x)
            .expect("DyLoRA forward pass should succeed with valid input");
        let y_lora = lora.forward(&x);
        for (a, b) in y_dy.iter().zip(y_lora.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "DyLoRA at full rank must equal plain LoRA: {a} vs {b}"
            );
        }
    }

    #[test]
    fn sampled_ranks_in_range() {
        let d = filled(3);
        let mut rng = LcgRng::new(99);
        for _ in 0..1000 {
            let b = d.sample_rank(&mut rng);
            assert!(
                b >= d.r_min && b <= d.r_max,
                "sampled rank {b} out of range"
            );
        }
    }

    #[test]
    fn output_shape_correct() {
        let d = filled(4);
        let x = vec![0.1_f32; d.in_features];
        for b in 1..=d.r_max {
            let y = d
                .forward_at_rank(&x, b)
                .expect("forward_at_rank should succeed for each valid nested rank");
            assert_eq!(y.len(), d.out_features);
        }
    }

    #[test]
    fn merge_adds_truncated_delta() {
        let mut d = filled(5);
        let b = 3;
        let delta = d
            .truncated_delta(b)
            .expect("truncated_delta should succeed with valid rank");
        let w_before = d.w.clone();
        d.merge_at_rank(b)
            .expect("merge_at_rank should succeed with valid rank");
        for (i, (before, after)) in w_before.iter().zip(d.w.iter()).enumerate() {
            assert!(
                (after - (before + delta[i])).abs() < 1e-5,
                "merge mismatch at {i}"
            );
        }
    }

    #[test]
    fn scale_uses_active_rank() {
        let d = filled(6);
        assert!((d.scale_at(d.r_max) - d.alpha / d.r_max as f32).abs() < 1e-7);
        assert!((d.scale_at(1) - d.alpha).abs() < 1e-7);
    }

    #[test]
    fn outputs_are_finite_and_deterministic() {
        let d1 = filled(7);
        let d2 = filled(7);
        let x: Vec<f32> = (0..d1.in_features)
            .map(|i| (i as f32 - 2.0) * 1.3)
            .collect();
        let y1 = d1
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let y2 = d2
            .forward(&x)
            .expect("forward pass should succeed with same valid input");
        assert!(y1.iter().all(|v| v.is_finite()));
        assert_eq!(y1, y2);
    }

    #[test]
    fn invalid_rank_rejected() {
        let d = filled(8);
        let x = vec![0.0_f32; d.in_features];
        assert!(d.forward_at_rank(&x, 0).is_err());
        assert!(d.forward_at_rank(&x, d.r_max + 1).is_err());
        assert!(d.truncated_delta(d.r_max + 1).is_err());
    }
}
