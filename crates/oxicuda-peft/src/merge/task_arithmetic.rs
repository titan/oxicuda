//! Task Arithmetic — editing models via task vectors.
//!
//! Reference: Ilharco G, Ribeiro MT, Wortsman M, Schmidt L, Hajishirzi H,
//! Farhadi A (2023) "Editing Models with Task Arithmetic", ICLR.
//! <https://arxiv.org/abs/2212.04089>
//!
//! A *task vector* `τ = θ_finetuned − θ_pretrained` captures the directional
//! displacement learnt by fine-tuning on a downstream task. Such vectors can
//! be:
//!
//! * **applied** with a scalar weight: `θ_new = θ_pretrained + α · τ`,
//! * **summed** to combine multiple tasks: `Σ_j w_j · τ_j`,
//! * **negated** to *unlearn* a task: `−τ`,
//! * combined via **analogy**: `τ_A − τ_B + τ_C`.
//!
//! All operations are elementwise and require matching dimensions. This
//! module shares conventions with [`crate::merge::arithmetic`] for empty/length
//! handling and reuses [`crate::error::PeftError`] variants where appropriate.

use crate::error::{PeftError, PeftResult};

/// A task vector — the per-parameter delta `θ_finetuned − θ_pretrained`.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskVector {
    /// Elementwise delta between the fine-tuned and pretrained checkpoints.
    pub delta: Vec<f32>,
}

impl TaskVector {
    /// Construct a task vector directly from its `delta` slice.
    #[must_use]
    pub fn from_delta(delta: Vec<f32>) -> Self {
        Self { delta }
    }

    /// Return the dimensionality of the task vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.delta.len()
    }

    /// Return whether the task vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.delta.is_empty()
    }
}

/// Task arithmetic algorithm namespace.
pub struct TaskArithmetic;

impl TaskArithmetic {
    /// Compute the task vector `θ_finetuned − θ_pretrained`.
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] when either slice is empty, or
    /// [`PeftError::DimensionMismatch`] when the slices differ in length.
    pub fn task_vector(
        theta_finetuned: &[f32],
        theta_pretrained: &[f32],
    ) -> PeftResult<TaskVector> {
        if theta_finetuned.is_empty() || theta_pretrained.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        if theta_finetuned.len() != theta_pretrained.len() {
            return Err(PeftError::DimensionMismatch {
                expected: theta_pretrained.len(),
                got: theta_finetuned.len(),
            });
        }
        let delta = theta_finetuned
            .iter()
            .zip(theta_pretrained.iter())
            .map(|(f, p)| f - p)
            .collect();
        Ok(TaskVector { delta })
    }

    /// Apply a task vector to the pretrained parameters: `θ_pretrained + α · τ`.
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] when `theta_pretrained` is empty, or
    /// [`PeftError::DimensionMismatch`] when the lengths disagree.
    pub fn apply(theta_pretrained: &[f32], tau: &TaskVector, alpha: f32) -> PeftResult<Vec<f32>> {
        if theta_pretrained.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        if theta_pretrained.len() != tau.delta.len() {
            return Err(PeftError::DimensionMismatch {
                expected: theta_pretrained.len(),
                got: tau.delta.len(),
            });
        }
        Ok(theta_pretrained
            .iter()
            .zip(tau.delta.iter())
            .map(|(p, d)| p + alpha * d)
            .collect())
    }

    /// Weighted sum of task vectors: `Σⱼ wⱼ · τⱼ`.
    ///
    /// Returns a fresh task vector with the same dimensionality as the input
    /// vectors. The list must be non-empty and every member must share the
    /// dimensionality of the first.
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] when `taus` is empty or the first
    /// vector is empty, or [`PeftError::DimensionMismatch`] when any vector
    /// disagrees in length with the first.
    pub fn add(taus: &[(f32, &TaskVector)]) -> PeftResult<TaskVector> {
        if taus.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = taus[0].1.delta.len();
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        let mut delta = vec![0.0_f32; n];
        for &(weight, tv) in taus.iter() {
            if tv.delta.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: tv.delta.len(),
                });
            }
            for (d_slot, &v) in delta.iter_mut().zip(tv.delta.iter()) {
                *d_slot += weight * v;
            }
        }
        Ok(TaskVector { delta })
    }

    /// Negate a task vector elementwise: returns `−τ`.
    #[must_use]
    pub fn negate(tau: &TaskVector) -> TaskVector {
        TaskVector {
            delta: tau.delta.iter().map(|v| -v).collect(),
        }
    }

    /// Analogy: `τ_a − τ_b + τ_c`, modelling "is to" relations between tasks.
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] when any vector is empty, or
    /// [`PeftError::DimensionMismatch`] when the lengths disagree.
    pub fn analogy(
        tau_a: &TaskVector,
        tau_b: &TaskVector,
        tau_c: &TaskVector,
    ) -> PeftResult<TaskVector> {
        if tau_a.delta.is_empty() || tau_b.delta.is_empty() || tau_c.delta.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = tau_a.delta.len();
        if tau_b.delta.len() != n {
            return Err(PeftError::DimensionMismatch {
                expected: n,
                got: tau_b.delta.len(),
            });
        }
        if tau_c.delta.len() != n {
            return Err(PeftError::DimensionMismatch {
                expected: n,
                got: tau_c.delta.len(),
            });
        }
        let delta = tau_a
            .delta
            .iter()
            .zip(tau_b.delta.iter())
            .zip(tau_c.delta.iter())
            .map(|((a, b), c)| a - b + c)
            .collect();
        Ok(TaskVector { delta })
    }

    /// Cosine similarity between two task vectors: `⟨a, b⟩ / (‖a‖₂ · ‖b‖₂)`.
    ///
    /// Returns `0.0` if either vector has zero L2 norm to avoid division by
    /// zero (so that callers can treat undefined as "no similarity").
    ///
    /// # Errors
    /// Returns [`PeftError::EmptyInput`] when either vector is empty, or
    /// [`PeftError::DimensionMismatch`] when the lengths disagree.
    pub fn cosine_similarity(tau_a: &TaskVector, tau_b: &TaskVector) -> PeftResult<f32> {
        if tau_a.delta.is_empty() || tau_b.delta.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        if tau_a.delta.len() != tau_b.delta.len() {
            return Err(PeftError::DimensionMismatch {
                expected: tau_a.delta.len(),
                got: tau_b.delta.len(),
            });
        }
        let mut dot = 0.0_f64;
        let mut norm_a_sq = 0.0_f64;
        let mut norm_b_sq = 0.0_f64;
        for (&a, &b) in tau_a.delta.iter().zip(tau_b.delta.iter()) {
            let af = a as f64;
            let bf = b as f64;
            dot += af * bf;
            norm_a_sq += af * af;
            norm_b_sq += bf * bf;
        }
        let denom = (norm_a_sq.sqrt()) * (norm_b_sq.sqrt());
        if denom <= f64::EPSILON {
            return Ok(0.0);
        }
        let cos = dot / denom;
        // Clamp away tiny floating-point excursions outside [-1, 1].
        Ok(cos.clamp(-1.0, 1.0) as f32)
    }

    /// L2 norm of a task vector.
    #[must_use]
    pub fn norm(tau: &TaskVector) -> f32 {
        let sum_sq: f64 = tau.delta.iter().map(|&v| (v as f64) * (v as f64)).sum();
        sum_sq.sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn task_vector_dimension_mismatch() {
        let res = TaskArithmetic::task_vector(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn task_vector_empty_input_errors() {
        let res = TaskArithmetic::task_vector(&[], &[1.0]);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
        let res2 = TaskArithmetic::task_vector(&[1.0], &[]);
        assert!(matches!(res2, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn task_vector_subtraction_correctness() {
        let fine = vec![3.0_f32, 5.0, 7.0, 11.0];
        let pre = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tau = TaskArithmetic::task_vector(&fine, &pre).expect("delta");
        assert!(approx_eq_slice(&tau.delta, &[2.0, 3.0, 4.0, 7.0], 1e-7));
    }

    #[test]
    fn apply_alpha_one_recovers_finetuned() {
        let fine = vec![2.0_f32, -1.0, 4.5];
        let pre = vec![0.5_f32, 0.0, 1.0];
        let tau = TaskArithmetic::task_vector(&fine, &pre).expect("delta");
        let recovered = TaskArithmetic::apply(&pre, &tau, 1.0).expect("apply");
        assert!(approx_eq_slice(&recovered, &fine, 1e-6));
    }

    #[test]
    fn apply_negated_with_minus_one_recovers_finetuned() {
        let fine = vec![2.0_f32, -1.0, 4.5];
        let pre = vec![0.5_f32, 0.0, 1.0];
        let tau = TaskArithmetic::task_vector(&fine, &pre).expect("delta");
        let neg = TaskArithmetic::negate(&tau);
        let recovered = TaskArithmetic::apply(&pre, &neg, -1.0).expect("apply");
        assert!(approx_eq_slice(&recovered, &fine, 1e-6));
    }

    #[test]
    fn weighted_sum_matches_manual_loop() {
        let a = TaskVector::from_delta(vec![1.0, 2.0, 3.0, 4.0]);
        let b = TaskVector::from_delta(vec![10.0, 20.0, 30.0, 40.0]);
        let c = TaskVector::from_delta(vec![-1.0, -2.0, -3.0, -4.0]);
        let weights = [0.5_f32, 0.25, 2.0];
        let combined = TaskArithmetic::add(&[(weights[0], &a), (weights[1], &b), (weights[2], &c)])
            .expect("add");
        let mut expected = vec![0.0_f32; 4];
        for (i, exp_slot) in expected.iter_mut().enumerate() {
            *exp_slot = weights[0] * a.delta[i] + weights[1] * b.delta[i] + weights[2] * c.delta[i];
        }
        assert!(approx_eq_slice(&combined.delta, &expected, 1e-6));
    }

    #[test]
    fn add_empty_errors() {
        let res = TaskArithmetic::add(&[]);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn add_single_weighted_identity() {
        let a = TaskVector::from_delta(vec![1.0, -2.0, 3.0]);
        let only = TaskArithmetic::add(&[(1.0, &a)]).expect("add single");
        assert!(approx_eq_slice(&only.delta, &a.delta, 1e-7));
    }

    #[test]
    fn add_dimension_mismatch() {
        let a = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let b = TaskVector::from_delta(vec![1.0, 2.0]);
        let res = TaskArithmetic::add(&[(1.0, &a), (1.0, &b)]);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn analogy_property() {
        let a = TaskVector::from_delta(vec![5.0, 7.0, 9.0]);
        let b = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let c = TaskVector::from_delta(vec![10.0, 20.0, 30.0]);
        let result = TaskArithmetic::analogy(&a, &b, &c).expect("analogy");
        let expected = [
            a.delta[0] - b.delta[0] + c.delta[0],
            a.delta[1] - b.delta[1] + c.delta[1],
            a.delta[2] - b.delta[2] + c.delta[2],
        ];
        assert!(approx_eq_slice(&result.delta, &expected, 1e-7));
    }

    #[test]
    fn analogy_dimension_mismatch_errors() {
        let a = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let b = TaskVector::from_delta(vec![1.0, 2.0]);
        let c = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let res = TaskArithmetic::analogy(&a, &b, &c);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = TaskVector::from_delta(vec![1.0, 2.0, 3.0, 4.0]);
        let cos = TaskArithmetic::cosine_similarity(&a, &a).expect("cos");
        assert!(approx_eq(cos, 1.0, 1e-6));
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = TaskVector::from_delta(vec![1.0, 0.0, 0.0, 0.0]);
        let b = TaskVector::from_delta(vec![0.0, 1.0, 0.0, 0.0]);
        let cos = TaskArithmetic::cosine_similarity(&a, &b).expect("cos");
        assert!(approx_eq(cos, 0.0, 1e-6));
    }

    #[test]
    fn cosine_antiparallel_is_minus_one() {
        let a = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let b = TaskVector::from_delta(vec![-1.0, -2.0, -3.0]);
        let cos = TaskArithmetic::cosine_similarity(&a, &b).expect("cos");
        assert!(approx_eq(cos, -1.0, 1e-6));
    }

    #[test]
    fn cosine_zero_norm_guard() {
        let a = TaskVector::from_delta(vec![0.0, 0.0, 0.0]);
        let b = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let cos = TaskArithmetic::cosine_similarity(&a, &b).expect("cos");
        assert!(approx_eq(cos, 0.0, 1e-7));
    }

    #[test]
    fn cosine_dimension_mismatch() {
        let a = TaskVector::from_delta(vec![1.0, 2.0]);
        let b = TaskVector::from_delta(vec![1.0, 2.0, 3.0]);
        let res = TaskArithmetic::cosine_similarity(&a, &b);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn apply_alpha_zero_returns_pretrained() {
        let pre = vec![0.1_f32, 0.2, 0.3, 0.4];
        let tau = TaskVector::from_delta(vec![1.0, 2.0, 3.0, 4.0]);
        let out = TaskArithmetic::apply(&pre, &tau, 0.0).expect("apply");
        assert!(approx_eq_slice(&out, &pre, 1e-7));
    }

    #[test]
    fn multi_task_weighted_sum_equivalent_to_sequential_apply() {
        let pre = vec![0.0_f32, 0.0, 0.0, 0.0];
        let tau_a = TaskVector::from_delta(vec![1.0, 0.0, -1.0, 2.0]);
        let tau_b = TaskVector::from_delta(vec![0.5, -0.5, 1.5, 1.0]);
        let alpha = 0.7_f32;
        let beta = 0.3_f32;

        let merged = TaskArithmetic::add(&[(alpha, &tau_a), (beta, &tau_b)]).expect("add");
        let one_shot = TaskArithmetic::apply(&pre, &merged, 1.0).expect("apply");

        let step_one = TaskArithmetic::apply(&pre, &tau_a, alpha).expect("apply a");
        let sequential = TaskArithmetic::apply(&step_one, &tau_b, beta).expect("apply b");

        assert!(
            approx_eq_slice(&one_shot, &sequential, 1e-6),
            "merge+apply not equivalent to sequential apply"
        );
    }

    #[test]
    fn norm_matches_naive_sum_of_squares() {
        let v = vec![3.0_f32, 4.0, 0.0, 12.0];
        let tau = TaskVector::from_delta(v.clone());
        let n = TaskArithmetic::norm(&tau);
        let expected = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert!(
            approx_eq(n, expected, 1e-5),
            "norm {n} != expected {expected}"
        );
        // 3-4-?-12 → 3² + 4² + 12² = 9 + 16 + 144 = 169 → √169 = 13.
        assert!(approx_eq(n, 13.0, 1e-5));
    }

    #[test]
    fn apply_dimension_mismatch_errors() {
        let pre = vec![0.0_f32; 3];
        let tau = TaskVector::from_delta(vec![1.0, 2.0]);
        let res = TaskArithmetic::apply(&pre, &tau, 1.0);
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn apply_empty_pretrained_errors() {
        let pre: Vec<f32> = Vec::new();
        let tau = TaskVector::from_delta(vec![]);
        let res = TaskArithmetic::apply(&pre, &tau, 1.0);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn task_vector_helpers_len_and_is_empty() {
        let tv = TaskVector::from_delta(vec![1.0, 2.0]);
        assert_eq!(tv.len(), 2);
        assert!(!tv.is_empty());
        let empty = TaskVector::from_delta(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
