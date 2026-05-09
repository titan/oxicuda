//! Forward transfer and intransigence metrics.
//!
//! - **Forward Transfer (FWT)**: how much learning on task k influences future
//!   performance on task k+1 (positive = beneficial).
//! - **Intransigence**: the inability to learn a new task due to constraints
//!   imposed by previous tasks.

use crate::error::{ContinualError, ContinualResult};
use crate::metrics::forgetting::AccuracyMatrix;

/// Compute forward transfer (FWT) across tasks.
///
/// `FWT = (1 / (T-1)) · Σ_{k=1}^{T-1} [acc[k-1, k] - baseline[k]]`
///
/// where `acc[k-1, k]` is the accuracy on task k *before* training on it
/// (immediately before), and `baseline[k]` is the random/chance accuracy.
///
/// Positive FWT indicates beneficial knowledge transfer between tasks.
pub fn forward_transfer(
    acc_matrix: &AccuracyMatrix,
    random_baseline: &[f32],
) -> ContinualResult<f32> {
    let t = acc_matrix.n_tasks;
    if t < 2 {
        return Ok(0.0);
    }
    if random_baseline.len() < t {
        return Err(ContinualError::DimensionMismatch {
            expected: t,
            got: random_baseline.len(),
        });
    }
    let mut total = 0.0_f32;
    // For task k (k=1..T-1): acc[k-1, k] is the performance on task k
    // after training on task k-1 (i.e., zero-shot transfer).
    for (k, baseline) in random_baseline.iter().enumerate().take(t).skip(1) {
        let transfer_acc = acc_matrix.data[k - 1][k];
        total += transfer_acc - baseline;
    }
    let fwt = total / (t - 1) as f32;
    if !fwt.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "forward_transfer",
        });
    }
    Ok(fwt)
}

/// Compute intransigence for each task.
///
/// Intransigence of task k is defined as:
/// `I_k = max_acc_k - acc[k, k]`
///
/// where `max_acc_k = max_{j ≥ k} acc[j, k]` (best accuracy ever on task k)
/// represents an approximation of the jointly trained reference.
///
/// A positive intransigence means the model could not learn task k as well
/// due to constraints from previous tasks.
///
/// Returns the average intransigence across all tasks.
pub fn intransigence(acc_matrix: &AccuracyMatrix) -> ContinualResult<f32> {
    let t = acc_matrix.n_tasks;
    if t == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let mut total = 0.0_f32;
    for k in 0..t {
        // Best accuracy ever achieved on task k (approximation of joint training)
        let max_acc = (k..t)
            .map(|j| acc_matrix.data[j][k])
            .fold(f32::NEG_INFINITY, f32::max);
        let on_task_acc = acc_matrix.data[k][k];
        // Intransigence = max - on_task (can be 0 if on_task = max)
        total += (max_acc - on_task_acc).max(0.0);
    }
    let avg_int = total / t as f32;
    if !avg_int.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "intransigence",
        });
    }
    Ok(avg_int)
}

/// Compute per-task intransigence values.
pub fn per_task_intransigence(acc_matrix: &AccuracyMatrix) -> ContinualResult<Vec<f32>> {
    let t = acc_matrix.n_tasks;
    if t == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let mut result = Vec::with_capacity(t);
    for k in 0..t {
        let max_acc = (k..t)
            .map(|j| acc_matrix.data[j][k])
            .fold(f32::NEG_INFINITY, f32::max);
        let on_task_acc = acc_matrix.data[k][k];
        result.push((max_acc - on_task_acc).max(0.0));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_transfer_positive_for_beneficial_transfer() {
        // Task 1 benefits from task 0 training (zero-shot acc > baseline)
        let mut mat = AccuracyMatrix::new(3);
        mat.data[0][0] = 0.9;
        mat.data[0][1] = 0.4; // acc on task 1 before training on it
        mat.data[0][2] = 0.3; // acc on task 2 before training
        mat.data[1][1] = 0.85;
        mat.data[1][2] = 0.35;
        mat.data[2][2] = 0.88;
        // Random baseline for a 10-class problem: 0.1
        let baseline = vec![0.1_f32; 3];
        let fwt = forward_transfer(&mat, &baseline).unwrap();
        // FWT = ((0.4 - 0.1) + (0.35 - 0.1)) / 2 = (0.3 + 0.25) / 2 = 0.275
        assert!(
            fwt > 0.0,
            "FWT should be positive for beneficial transfer, got {fwt}"
        );
        assert!(
            (fwt - 0.275).abs() < 1e-5,
            "Expected FWT = 0.275, got {fwt}"
        );
    }

    #[test]
    fn forward_transfer_single_task_is_zero() {
        let mat = AccuracyMatrix::new(1);
        let baseline = vec![0.1_f32];
        let fwt = forward_transfer(&mat, &baseline).unwrap();
        assert_eq!(fwt, 0.0);
    }

    #[test]
    fn intransigence_zero_when_on_task_is_max() {
        // If the on-task accuracy equals the best ever, intransigence = 0
        let mut mat = AccuracyMatrix::new(3);
        // Make diagonal the maximum for each task
        mat.data[0][0] = 0.9;
        mat.data[1][0] = 0.5;
        mat.data[2][0] = 0.3;
        mat.data[1][1] = 0.85;
        mat.data[2][1] = 0.4;
        mat.data[2][2] = 0.88;
        let int_val = intransigence(&mat).unwrap();
        // For task 0: max=0.9 (at j=0), on-task=0.9 → 0
        // For task 1: max=0.85 (at j=1), on-task=0.85 → 0
        // For task 2: max=0.88 (at j=2), on-task=0.88 → 0
        assert!(
            int_val.abs() < 1e-6,
            "Intransigence should be 0 when on-task = max, got {int_val}"
        );
    }

    #[test]
    fn intransigence_positive_when_later_accuracy_better() {
        let mut mat = AccuracyMatrix::new(2);
        // Task 0: on-task = 0.5, but later acc = 0.8 (better after more training?)
        mat.data[0][0] = 0.5;
        mat.data[1][0] = 0.8; // better later (positive transfer)
        mat.data[1][1] = 0.9;
        let int_val = intransigence(&mat).unwrap();
        // Task 0: max=0.8, on-task=0.5 → intransigence = 0.3
        // Task 1: max=0.9, on-task=0.9 → 0
        // avg = 0.15
        assert!(int_val > 0.0, "Intransigence should be positive");
    }

    #[test]
    fn per_task_intransigence_length_correct() {
        let mat = AccuracyMatrix::new(4);
        let per_task = per_task_intransigence(&mat).unwrap();
        assert_eq!(per_task.len(), 4);
    }

    #[test]
    fn forward_transfer_dimension_mismatch() {
        let mat = AccuracyMatrix::new(3);
        let baseline = vec![0.1_f32; 2]; // too short
        assert!(forward_transfer(&mat, &baseline).is_err());
    }

    #[test]
    fn intransigence_empty_matrix() {
        let mat = AccuracyMatrix::new(0);
        assert!(intransigence(&mat).is_err());
    }
}
