//! MiniLLM — reverse-KL distillation with a policy-gradient estimator (Gu et al. 2023).
//!
//! "Knowledge Distillation of Large Language Models" (MiniLLM, Gu et al. 2023)
//! replaces the standard *forward* KL `KL(p_teacher ‖ q_student)` with the
//! *reverse* KL
//!
//! ```text
//! KL(q_student ‖ p_teacher) = E_{i∼q}[ log q_i − log p_i ] = Σ_i q_i (log q_i − log p_i).
//! ```
//!
//! Forward KL is *mass-covering*: it forces the student to spread probability
//! over every class on which the teacher places mass, blurring multi-modal
//! teachers. Reverse KL is *mode-seeking* (zero-forcing): the `q_i log(q_i/p_i)`
//! term diverges wherever the student places mass on a class the teacher deems
//! unlikely (`p_i ≈ 0`), so the student is driven to avoid the teacher's
//! low-probability regions and concentrate on a high-probability mode rather
//! than the inter-mode mean.
//!
//! Because the expectation is taken under the *student's own* distribution, the
//! gradient is estimated with the REINFORCE / score-function rule. With the
//! reward `r_i = log q_i − log p_i` and the variance-reducing baseline
//! `b = E_q[r] = KL(q ‖ p)`, the single-sample estimator
//! `(r_{i*} − b) ∇_z log q_{i*}` is unbiased, and its closed-form expectation
//! over the categorical sampling distribution (the Rao-Blackwellised estimator
//! returned by [`MinKd::policy_gradient`]) is the exact gradient
//!
//! ```text
//! ∂KL(q‖p)/∂z_l = (1/T) · q_l · ( (log q_l − log p_l) − KL(q‖p) ).
//! ```
//!
//! The temperature-softmax and KL primitives are shared with
//! [`crate::logit::hinton_kd`].

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-10;

/// MiniLLM reverse-KL distillation loss.
#[derive(Debug, Clone, PartialEq)]
pub struct MinKd {
    /// Softmax temperature `T > 0` applied to both teacher and student logits.
    pub temperature: f32,
}

impl MinKd {
    /// Construct a new [`MinKd`], validating the temperature.
    ///
    /// # Errors
    /// - [`DistillError::InvalidConfig`] if `temperature ≤ 0` or non-finite.
    pub fn new(temperature: f32) -> DistillResult<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be finite and > 0, got {temperature}"),
            });
        }
        Ok(Self { temperature })
    }

    /// Validate the flattened logit layout for a `n × c` batch.
    fn check(
        student_logits: &[f32],
        teacher_logits: &[f32],
        n: usize,
        c: usize,
    ) -> DistillResult<()> {
        if n == 0 || c == 0 || student_logits.is_empty() || teacher_logits.is_empty() {
            return Err(DistillError::EmptyInput);
        }
        let expected = n * c;
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

    /// Mean reverse KL `KL(q_student ‖ p_teacher)` over the batch.
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `n == 0`, `c == 0`, or either slice is
    ///   empty.
    /// - [`DistillError::DimensionMismatch`] if either slice length is not
    ///   `n · c`.
    pub fn reverse_kl(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n: usize,
        c: usize,
    ) -> DistillResult<f32> {
        Self::check(student_logits, teacher_logits, n, c)?;
        let mut total = 0.0_f32;
        for (s_row, t_row) in student_logits
            .chunks_exact(c)
            .zip(teacher_logits.chunks_exact(c))
        {
            let q = softmax_with_temp(s_row, self.temperature);
            let p = softmax_with_temp(t_row, self.temperature);
            // Reverse KL is KL(q ‖ p) — student first.
            total += kl_divergence(&q, &p);
        }
        Ok(total / n as f32)
    }

    /// Closed-form score-function (REINFORCE) gradient of the reverse KL with
    /// respect to the student logits, returned flat as `[n · c]`.
    ///
    /// For each sample, with `a_l = log q_l − log p_l` and
    /// `j = Σ_l q_l a_l = KL(q ‖ p)`, the per-logit gradient is
    /// `(1/T) q_l (a_l − j)`. This equals the expectation of the single-sample
    /// estimator [`MinKd::policy_gradient_sampled`] and so matches a
    /// finite-difference of [`MinKd::reverse_kl`].
    ///
    /// # Errors
    /// - the error conditions of [`MinKd::reverse_kl`].
    pub fn policy_gradient(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        n: usize,
        c: usize,
    ) -> DistillResult<Vec<f32>> {
        Self::check(student_logits, teacher_logits, n, c)?;
        let inv_t = 1.0 / self.temperature;
        let mut grad = vec![0.0_f32; student_logits.len()];
        for ((g_row, s_row), t_row) in grad
            .chunks_exact_mut(c)
            .zip(student_logits.chunks_exact(c))
            .zip(teacher_logits.chunks_exact(c))
        {
            let q = softmax_with_temp(s_row, self.temperature);
            let p = softmax_with_temp(t_row, self.temperature);
            let a: Vec<f32> = q
                .iter()
                .zip(p.iter())
                .map(|(&qi, &pi)| (qi + EPS).ln() - (pi + EPS).ln())
                .collect();
            let j: f32 = q.iter().zip(a.iter()).map(|(&qi, &ai)| qi * ai).sum();
            for ((gl, &ql), &al) in g_row.iter_mut().zip(q.iter()).zip(a.iter()) {
                *gl = inv_t * ql * (al - j);
            }
        }
        Ok(grad)
    }

    /// Single-sample REINFORCE estimate of the reverse-KL gradient for one
    /// sample (`c` logits), drawing the action `i* ∼ q` from `rng`.
    ///
    /// Uses the score function `∇_z log q_{i*} = (1/T)(e_{i*} − q)` weighted by
    /// the advantage `r_{i*} − b`, with reward `r_i = log q_i − log p_i` and
    /// baseline `b = KL(q ‖ p)`. Unbiased but high-variance; its expectation is
    /// [`MinKd::policy_gradient`].
    ///
    /// # Errors
    /// - [`DistillError::EmptyInput`] if `c == 0` or either slice is empty.
    /// - [`DistillError::DimensionMismatch`] if either slice length is not `c`.
    pub fn policy_gradient_sampled(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        c: usize,
        rng: &mut LcgRng,
    ) -> DistillResult<Vec<f32>> {
        Self::check(student_logits, teacher_logits, 1, c)?;
        let q = softmax_with_temp(student_logits, self.temperature);
        let p = softmax_with_temp(teacher_logits, self.temperature);
        let a: Vec<f32> = q
            .iter()
            .zip(p.iter())
            .map(|(&qi, &pi)| (qi + EPS).ln() - (pi + EPS).ln())
            .collect();
        let baseline: f32 = q.iter().zip(a.iter()).map(|(&qi, &ai)| qi * ai).sum();

        // Draw i* ∼ q via inverse-CDF sampling. Use the full 32-bit mantissa for
        // a uniform in [0, 1) (next_u32 / u32::MAX), then locate the bucket.
        let u = (rng.next_u32() as f32) / (u32::MAX as f32);
        let mut cum = 0.0_f32;
        let mut chosen = c - 1;
        for (idx, &qi) in q.iter().enumerate() {
            cum += qi;
            if u < cum {
                chosen = idx;
                break;
            }
        }
        let advantage = a[chosen] - baseline;
        let inv_t = 1.0 / self.temperature;
        let grad = q
            .iter()
            .enumerate()
            .map(|(l, &ql)| {
                let indicator = if l == chosen { 1.0 } else { 0.0 };
                advantage * inv_t * (indicator - ql)
            })
            .collect();
        Ok(grad)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(logits: &[f32]) -> Vec<f32> {
        softmax_with_temp(logits, 1.0)
    }

    #[test]
    fn identical_logits_zero_reverse_kl() {
        let m = MinKd::new(2.0).expect("valid");
        let logits = vec![0.5_f32, 1.5, -0.3, 2.0, 0.1, -1.0];
        let v = m.reverse_kl(&logits, &logits, 2, 3).expect("ok");
        assert!(
            v.abs() < 1e-5,
            "reverse KL of identical logits must be 0, got {v}"
        );
    }

    #[test]
    fn reverse_kl_nonnegative() {
        let m = MinKd::new(1.5).expect("valid");
        let s = vec![1.0_f32, 0.0, -1.0, 2.0, 0.5, 0.1];
        let t = vec![-1.0_f32, 0.5, 2.0, 0.0, 0.0, 1.0];
        let v = m.reverse_kl(&s, &t, 2, 3).expect("ok");
        assert!(v >= -1e-6 && v.is_finite(), "reverse KL = {v}");
    }

    #[test]
    fn gradient_sums_to_zero() {
        // Softmax logit gradients always sum to zero.
        let m = MinKd::new(1.0).expect("valid");
        let s = vec![0.7_f32, -0.4, 1.1];
        let t = vec![-0.3_f32, 0.9, 0.1];
        let g = m.policy_gradient(&s, &t, 1, 3).expect("ok");
        let sum: f32 = g.iter().sum();
        assert!(sum.abs() < 1e-5, "gradient must sum to 0, got {sum}");
    }

    #[test]
    fn mode_seeking_gradient_direction() {
        // Strongly bimodal teacher: modes at classes 0 and 2, deep valley at 1.
        let m = MinKd::new(1.0).expect("valid");
        let teacher = vec![6.0_f32, 0.0, 6.0];
        // Student leans toward mode 0 (and currently over-covers the valley).
        let student = vec![0.5_f32, 0.0, 0.0];
        let grad = m.policy_gradient(&student, &teacher, 1, 3).expect("ok");
        // Descent direction (minimising reverse KL).
        let d: Vec<f32> = grad.iter().map(|&g| -g).collect();
        // (i) Mass is driven OUT of the central / "mean" valley class — the
        //     zero-forcing signature, NOT mean-seeking.
        assert!(d[1] < 0.0, "valley logit must decrease, d={d:?}");
        // (ii) Both modes attract mass...
        assert!(d[0] > 0.0 && d[2] > 0.0, "modes must attract mass, d={d:?}");
        // (iii) ...but the already-favoured mode is reinforced MORE than the
        //       other, breaking symmetry toward a single mode.
        assert!(
            d[0] > d[2],
            "favoured mode 0 must be pushed harder than mode 2: d={d:?}"
        );
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let m = MinKd::new(2.0).expect("valid");
        let s = vec![0.7_f32, -0.4, 1.1, 0.2];
        let t = vec![-0.3_f32, 0.9, 0.1, -0.6];
        let c = 4;
        let grad = m.policy_gradient(&s, &t, 1, c).expect("ok");
        let eps = 1e-3_f32;
        for l in 0..c {
            let mut sp = s.clone();
            let mut sm = s.clone();
            sp[l] += eps;
            sm[l] -= eps;
            let fp = m.reverse_kl(&sp, &t, 1, c).expect("ok");
            let fm = m.reverse_kl(&sm, &t, 1, c).expect("ok");
            let fd = (fp - fm) / (2.0 * eps);
            assert!((grad[l] - fd).abs() < 1e-2, "grad[{l}]={} fd={fd}", grad[l]);
        }
    }

    #[test]
    fn sampled_gradient_finite_and_zero_sum() {
        let m = MinKd::new(1.0).expect("valid");
        let s = vec![0.3_f32, 0.8, -0.2];
        let t = vec![0.1_f32, -0.5, 1.0];
        let mut rng = LcgRng::new(7);
        for _ in 0..20 {
            let g = m.policy_gradient_sampled(&s, &t, 3, &mut rng).expect("ok");
            assert_eq!(g.len(), 3);
            let sum: f32 = g.iter().sum();
            assert!(
                sum.abs() < 1e-5 && g.iter().all(|v| v.is_finite()),
                "g={g:?}"
            );
        }
    }

    #[test]
    fn sampled_gradient_mean_approaches_analytic() {
        // The single-sample estimator is unbiased; its average over many draws
        // should approach the closed-form gradient.
        let m = MinKd::new(1.0).expect("valid");
        let s = vec![0.3_f32, 0.8, -0.2];
        let t = vec![0.1_f32, -0.5, 1.0];
        let analytic = m.policy_gradient(&s, &t, 1, 3).expect("ok");
        let mut rng = LcgRng::new(123);
        let mut acc = [0.0_f32; 3];
        let n = 4000;
        for _ in 0..n {
            let g = m.policy_gradient_sampled(&s, &t, 3, &mut rng).expect("ok");
            for (a, &gi) in acc.iter_mut().zip(g.iter()) {
                *a += gi;
            }
        }
        for (a, &an) in acc.iter_mut().zip(analytic.iter()) {
            *a /= n as f32;
            assert!((*a - an).abs() < 5e-2, "mean {a} vs analytic {an}");
        }
    }

    #[test]
    fn mode_seeking_distinct_from_softmax_mean() {
        // Sanity: the student here does NOT sit at the teacher mean; verify the
        // teacher truly is bimodal (valley probability below either mode).
        let p = softmax(&[6.0_f32, 0.0, 6.0]);
        assert!(p[1] < p[0] && p[1] < p[2], "teacher must be bimodal: {p:?}");
    }

    #[test]
    fn dim_mismatch_error() {
        let m = MinKd::new(1.0).expect("valid");
        let s = vec![1.0_f32, 2.0, 3.0];
        let t = vec![0.0_f32, 1.0, 2.0, 3.0];
        assert!(matches!(
            m.reverse_kl(&s, &t, 1, 3),
            Err(DistillError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_input_error() {
        let m = MinKd::new(1.0).expect("valid");
        assert!(matches!(
            m.reverse_kl(&[], &[], 0, 3),
            Err(DistillError::EmptyInput)
        ));
        let s = vec![1.0_f32, 2.0];
        assert!(matches!(
            m.reverse_kl(&s, &s, 1, 0),
            Err(DistillError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_temperature_error() {
        assert!(matches!(
            MinKd::new(0.0),
            Err(DistillError::InvalidConfig { .. })
        ));
        assert!(matches!(
            MinKd::new(-1.0),
            Err(DistillError::InvalidConfig { .. })
        ));
    }
}
