//! DistWRD — Wasserstein-based distribution alignment (Shen et al. 2022).
//!
//! Classical soft-label distillation matches the teacher and student output
//! distributions with a *pointwise* Kullback-Leibler term, which treats the
//! class probabilities as an unordered bag and ignores any geometry over the
//! label space. DistWRD ("Distillation with Wasserstein Regularised
//! Distance") instead aligns the two temperature-softmax distributions with a
//! 1-D **optimal-transport** divergence along the class ordering: a small
//! amount of probability that the student misplaces onto a *nearby* class is
//! penalised less than the same mass placed on a *distant* class.
//!
//! For two discrete distributions `p` (teacher) and `q` (student) over the
//! ordered support `0..C` with the unit ground cost `c(i, j) = |i − j|`, the
//! Wasserstein-1 (Earth-Mover) distance has the closed form
//!
//! ```text
//! W₁(p, q) = Σ_{k=0}^{C-1} | CDF_p(k) − CDF_q(k) |
//! ```
//!
//! i.e. the L1 distance between the two cumulative distribution functions. The
//! DistWRD objective is the batch mean of this distance, scaled by `lambda_wd`,
//! and may optionally be blended with the standard forward-KL term to recover a
//! pointwise component (see [`DistWrd::blended_loss`]).
//!
//! The temperature-softmax and KL primitives are shared with
//! [`crate::logit::hinton_kd`].

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

/// Sign of `x` as a float (`+1`, `−1`, or `0` exactly at zero).
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// 1-D Wasserstein-1 distance between two discrete distributions `p` and `q`
/// over the shared, ordered support `0..p.len()` with the unit ground cost
/// `|i − j|`.
///
/// Equals the L1 distance between the cumulative distribution functions,
/// `Σ_k |CDF_p(k) − CDF_q(k)|`. Inputs are assumed to be (approximately)
/// normalised probability vectors of equal length; the shorter length governs
/// the sum.
#[must_use]
pub fn wasserstein1_cdf(p: &[f32], q: &[f32]) -> f32 {
    let mut cdf_p = 0.0_f32;
    let mut cdf_q = 0.0_f32;
    let mut acc = 0.0_f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cdf_p += pi;
        cdf_q += qi;
        acc += (cdf_p - cdf_q).abs();
    }
    acc
}

/// DistWRD distribution-alignment loss.
///
/// Holds the transport weight `lambda_wd` and the softmax `temperature` shared
/// by the teacher and student distributions.
#[derive(Debug, Clone, PartialEq)]
pub struct DistWrd {
    /// Non-negative scale applied to the mean Wasserstein-1 term.
    pub lambda_wd: f32,
    /// Softmax temperature `T > 0` used to soften both distributions.
    pub temperature: f32,
}

impl DistWrd {
    /// Construct a new [`DistWrd`], validating the configuration.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `lambda_wd` is negative or
    ///   non-finite, or if `temperature ≤ 0` or non-finite.
    pub fn new(lambda_wd: f32, temperature: f32) -> DistillResult<Self> {
        if !lambda_wd.is_finite() || lambda_wd < 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("lambda_wd must be finite and >= 0, got {lambda_wd}"),
            });
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be finite and > 0, got {temperature}"),
            });
        }
        Ok(Self {
            lambda_wd,
            temperature,
        })
    }

    /// Validate the flattened logit layout for a `n_samples × n_classes` batch.
    fn check(
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_samples: usize,
        n_classes: usize,
    ) -> DistillResult<()> {
        if n_samples == 0
            || n_classes == 0
            || student_logits.is_empty()
            || teacher_logits.is_empty()
        {
            return Err(DistillError::EmptyInput);
        }
        let expected = n_samples * n_classes;
        if student_logits.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: student_logits.len(),
            });
        }
        if teacher_logits.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: teacher_logits.len(),
            });
        }
        Ok(())
    }

    /// DistWRD loss: `lambda_wd · mean_s W₁(p_s, q_s)` over the batch.
    ///
    /// `p_s` and `q_s` are the temperature-softmax distributions of the teacher
    /// and student logits of sample `s`; `W₁` is [`wasserstein1_cdf`].
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `n_samples == 0`, `n_classes == 0`, or
    ///   either slice is empty.
    /// - [`DistillError::DimensionMismatch`] if either slice length is not
    ///   `n_samples · n_classes`.
    pub fn loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_samples: usize,
        n_classes: usize,
    ) -> DistillResult<f32> {
        Self::check(student_logits, teacher_logits, n_samples, n_classes)?;
        let mut total = 0.0_f32;
        for (s_row, t_row) in student_logits
            .chunks_exact(n_classes)
            .zip(teacher_logits.chunks_exact(n_classes))
        {
            let q = softmax_with_temp(s_row, self.temperature);
            let p = softmax_with_temp(t_row, self.temperature);
            total += wasserstein1_cdf(&p, &q);
        }
        Ok(self.lambda_wd * total / n_samples as f32)
    }

    /// DistWRD loss blended with the forward-KL soft-target term:
    /// `lambda_wd · mean W₁(p, q) + kl_weight · mean KL(p ‖ q)`.
    ///
    /// Recovers a pointwise component alongside the transport term; with
    /// `kl_weight == 0` this is exactly [`DistWrd::loss`].
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `kl_weight` is negative or
    ///   non-finite.
    /// - the error conditions of [`DistWrd::loss`].
    pub fn blended_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_samples: usize,
        n_classes: usize,
        kl_weight: f32,
    ) -> DistillResult<f32> {
        Self::check(student_logits, teacher_logits, n_samples, n_classes)?;
        if !kl_weight.is_finite() || kl_weight < 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("kl_weight must be finite and >= 0, got {kl_weight}"),
            });
        }
        let mut total_wd = 0.0_f32;
        let mut total_kl = 0.0_f32;
        for (s_row, t_row) in student_logits
            .chunks_exact(n_classes)
            .zip(teacher_logits.chunks_exact(n_classes))
        {
            let q = softmax_with_temp(s_row, self.temperature);
            let p = softmax_with_temp(t_row, self.temperature);
            total_wd += wasserstein1_cdf(&p, &q);
            total_kl += kl_divergence(&p, &q);
        }
        let n = n_samples as f32;
        Ok(self.lambda_wd * total_wd / n + kl_weight * total_kl / n)
    }

    /// Analytic per-logit gradient of the (`lambda_wd`-scaled) single-sample
    /// Wasserstein-1 term with respect to the student logits.
    ///
    /// Let `D_k = CDF_q(k) − CDF_p(k)` and `s_k = sign(D_k)`. With the suffix
    /// sums `g_j = Σ_{k≥j} s_k = ∂W₁/∂q_j` and the temperature-softmax Jacobian
    /// `∂q_j/∂z_l = (1/T) q_j (δ_{jl} − q_l)`, the gradient collapses to
    ///
    /// ```text
    /// ∂(λ·W₁)/∂z_l = λ · (1/T) · q_l · (g_l − Σ_j q_j g_j).
    /// ```
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `n_classes == 0` or either slice is
    ///   empty.
    /// - [`DistillError::DimensionMismatch`] if either slice length is not
    ///   `n_classes`.
    pub fn sample_gradient(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n_classes: usize,
    ) -> DistillResult<Vec<f32>> {
        if n_classes == 0 || student_logits.is_empty() || teacher_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        if student_logits.len() != n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes,
                got: student_logits.len(),
            });
        }
        if teacher_logits.len() != n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_classes,
                got: teacher_logits.len(),
            });
        }
        let q = softmax_with_temp(student_logits, self.temperature);
        let p = softmax_with_temp(teacher_logits, self.temperature);

        // s_k = sign(CDF_q(k) − CDF_p(k)).
        let mut cdf_q = 0.0_f32;
        let mut cdf_p = 0.0_f32;
        let s_k: Vec<f32> = q
            .iter()
            .zip(p.iter())
            .map(|(&qi, &pi)| {
                cdf_q += qi;
                cdf_p += pi;
                sign(cdf_q - cdf_p)
            })
            .collect();

        // g_j = Σ_{k≥j} s_k (suffix sums).
        let mut suffix = 0.0_f32;
        let mut g: Vec<f32> = s_k
            .iter()
            .rev()
            .map(|&sk| {
                suffix += sk;
                suffix
            })
            .collect();
        g.reverse();

        let gbar: f32 = q.iter().zip(g.iter()).map(|(&qj, &gj)| qj * gj).sum();
        let inv_t = 1.0 / self.temperature;
        let grad = q
            .iter()
            .zip(g.iter())
            .map(|(&ql, &gl)| self.lambda_wd * inv_t * ql * (gl - gbar))
            .collect();
        Ok(grad)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_zero_loss() {
        let d = DistWrd::new(1.0, 2.0).expect("valid");
        let logits = vec![0.3_f32, 1.2, -0.5, 2.0, 0.1, -1.0];
        let v = d.loss(&logits, &logits, 2, 3).expect("ok");
        assert!(
            v.abs() < 1e-6,
            "W1 of identical distributions must be 0, got {v}"
        );
    }

    #[test]
    fn loss_nonnegative() {
        let d = DistWrd::new(0.7, 3.0).expect("valid");
        let s = vec![1.0_f32, 0.0, -1.0, 2.0, 0.5, 0.1];
        let t = vec![-1.0_f32, 0.5, 2.0, 0.0, 0.0, 1.0];
        let v = d.loss(&s, &t, 2, 3).expect("ok");
        assert!(v >= 0.0 && v.is_finite(), "loss = {v}");
    }

    #[test]
    fn w1_increases_with_shift_distance() {
        // Teacher places almost all mass on class 0. Students place almost all
        // mass on class 1, 2, 3 respectively; W1 must grow with the index gap.
        let d = DistWrd::new(1.0, 1.0).expect("valid");
        let n_classes = 5;
        let peak = |idx: usize| -> Vec<f32> {
            let mut v = vec![0.0_f32; n_classes];
            v[idx] = 10.0; // large logit ⇒ near-one softmax mass at idx
            v
        };
        let teacher = peak(0);
        let w_near = d.loss(&peak(1), &teacher, 1, n_classes).expect("ok");
        let w_mid = d.loss(&peak(2), &teacher, 1, n_classes).expect("ok");
        let w_far = d.loss(&peak(3), &teacher, 1, n_classes).expect("ok");
        assert!(
            w_near < w_mid && w_mid < w_far,
            "W1 must grow with shift: {w_near} < {w_mid} < {w_far}"
        );
    }

    #[test]
    fn lambda_scales_loss_linearly() {
        let s = vec![1.0_f32, 0.0, -1.0, 0.5];
        let t = vec![-1.0_f32, 0.0, 1.0, 0.2];
        let d1 = DistWrd::new(1.0, 2.0).expect("valid");
        let d3 = DistWrd::new(3.0, 2.0).expect("valid");
        let l1 = d1.loss(&s, &t, 2, 2).expect("ok");
        let l3 = d3.loss(&s, &t, 2, 2).expect("ok");
        assert!(l1 > 1e-6, "base loss must be non-trivial, got {l1}");
        assert!(
            (l3 - 3.0 * l1).abs() < 1e-5,
            "lambda must scale linearly: 3*{l1} vs {l3}"
        );
    }

    #[test]
    fn blended_recovers_loss_at_zero_kl_weight() {
        let s = vec![1.0_f32, 0.0, -1.0, 0.5, 0.2, -0.3];
        let t = vec![-1.0_f32, 0.5, 1.0, 0.0, 0.1, 0.4];
        let d = DistWrd::new(0.8, 2.5).expect("valid");
        let plain = d.loss(&s, &t, 2, 3).expect("ok");
        let blended = d.blended_loss(&s, &t, 2, 3, 0.0).expect("ok");
        assert!((plain - blended).abs() < 1e-6, "{plain} vs {blended}");
        let with_kl = d.blended_loss(&s, &t, 2, 3, 1.0).expect("ok");
        assert!(with_kl >= blended, "adding KL must not reduce the loss");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let d = DistWrd::new(1.3, 2.0).expect("valid");
        let s = vec![0.7_f32, -0.4, 1.1, 0.2];
        let t = vec![-0.3_f32, 0.9, 0.1, -0.6];
        let n_classes = 4;
        let grad = d.sample_gradient(&s, &t, n_classes).expect("ok");
        let eps = 1e-3_f32;
        let single_w1 = |logits: &[f32]| -> f32 {
            let q = softmax_with_temp(logits, d.temperature);
            let p = softmax_with_temp(&t, d.temperature);
            d.lambda_wd * wasserstein1_cdf(&p, &q)
        };
        for l in 0..n_classes {
            let mut sp = s.clone();
            let mut sm = s.clone();
            sp[l] += eps;
            sm[l] -= eps;
            let fd = (single_w1(&sp) - single_w1(&sm)) / (2.0 * eps);
            assert!((grad[l] - fd).abs() < 2e-2, "grad[{l}]={} fd={fd}", grad[l]);
        }
    }

    #[test]
    fn dim_mismatch_error() {
        let d = DistWrd::new(1.0, 1.0).expect("valid");
        let s = vec![1.0_f32, 2.0, 3.0]; // 3 != 2*2
        let t = vec![0.0_f32, 1.0, 2.0, 3.0];
        let r = d.loss(&s, &t, 2, 2);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_input_error() {
        let d = DistWrd::new(1.0, 1.0).expect("valid");
        let s = vec![1.0_f32, 2.0];
        let t = vec![0.0_f32, 1.0];
        assert!(matches!(
            d.loss(&s, &t, 1, 0),
            Err(DistillError::EmptyInput)
        ));
        assert!(matches!(
            d.loss(&[], &[], 0, 0),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_config_error() {
        assert!(matches!(
            DistWrd::new(-1.0, 1.0),
            Err(DistillError::InvalidConfig { .. })
        ));
        assert!(matches!(
            DistWrd::new(1.0, 0.0),
            Err(DistillError::InvalidConfig { .. })
        ));
    }
}
