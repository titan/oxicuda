//! Forgetting metrics for continual learning evaluation.
//!
//! Computes standard metrics from the accuracy matrix:
//! - **Average forgetting**: how much the model forgets previous tasks.
//! - **Backward transfer (BWT)**: signed measure of performance change on past tasks.
//! - **Plasticity**: ability to learn new tasks (last-task accuracy).

use crate::error::{ContinualError, ContinualResult};

/// Accuracy on a specific task at a specific training point.
#[derive(Debug, Clone)]
pub struct TaskAccuracy {
    /// Task identifier.
    pub task_id: usize,
    /// Accuracy value in [0, 1].
    pub accuracy: f32,
}

/// Accuracy matrix: `data[t][k]` = test accuracy on task `k` after training on task `t`.
///
/// Shape: `n_tasks × n_tasks` (upper triangle filled as tasks complete).
#[derive(Debug, Clone)]
pub struct AccuracyMatrix {
    /// Row-major accuracy data. `data[t][k]` = acc on task k after task t training.
    pub data: Vec<Vec<f32>>,
    /// Number of tasks.
    pub n_tasks: usize,
}

impl AccuracyMatrix {
    /// Construct a new accuracy matrix filled with zeros.
    #[must_use]
    pub fn new(n_tasks: usize) -> Self {
        Self {
            data: vec![vec![0.0_f32; n_tasks]; n_tasks],
            n_tasks,
        }
    }

    /// Set `data[t][k]` = `acc`.
    pub fn set(&mut self, t: usize, k: usize, acc: f32) -> ContinualResult<()> {
        if t >= self.n_tasks || k >= self.n_tasks {
            return Err(ContinualError::TaskIndexOutOfRange {
                index: t.max(k),
                n_tasks: self.n_tasks,
            });
        }
        self.data[t][k] = acc;
        Ok(())
    }
}

/// Compute average forgetting after training on all tasks.
///
/// `AF = (1 / (T-1)) · Σ_{k=0}^{T-2} [max_{j: 0≤j≤T-2} acc[j,k]] - acc[T-1,k]`
///
/// Measures the average drop from the best performance on each past task
/// to the final performance on that task.
///
/// Returns 0.0 for single-task matrices (no forgetting possible).
pub fn average_forgetting(acc_matrix: &AccuracyMatrix) -> ContinualResult<f32> {
    let t = acc_matrix.n_tasks;
    if t < 2 {
        return Ok(0.0);
    }
    let last = t - 1;
    let mut total = 0.0_f32;
    for k in 0..last {
        // Best accuracy on task k among steps 0..last (exclusive of final step)
        let max_acc = (0..last)
            .map(|j| acc_matrix.data[j][k])
            .fold(f32::NEG_INFINITY, f32::max);
        let final_acc = acc_matrix.data[last][k];
        total += max_acc - final_acc;
    }
    let af = total / last as f32;
    if !af.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "average_forgetting",
        });
    }
    Ok(af)
}

/// Compute backward transfer (BWT) after training on all T tasks.
///
/// `BWT = (1 / (T-1)) · Σ_{k=0}^{T-2} acc[T-1, k] - acc[k, k]`
///
/// Negative BWT indicates forgetting; positive BWT indicates positive transfer.
pub fn backward_transfer(acc_matrix: &AccuracyMatrix) -> ContinualResult<f32> {
    let t = acc_matrix.n_tasks;
    if t < 2 {
        return Ok(0.0);
    }
    let last = t - 1;
    let mut total = 0.0_f32;
    for k in 0..last {
        let final_acc = acc_matrix.data[last][k];
        let on_task_acc = acc_matrix.data[k][k];
        total += final_acc - on_task_acc;
    }
    let bwt = total / last as f32;
    if !bwt.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "backward_transfer",
        });
    }
    Ok(bwt)
}

/// Compute the average plasticity (average on-task accuracy at training time).
///
/// `Plasticity = (1/T) · Σ_{k=0}^{T-1} acc[k, k]`
pub fn plasticity(acc_matrix: &AccuracyMatrix) -> ContinualResult<f32> {
    let t = acc_matrix.n_tasks;
    if t == 0 {
        return Err(ContinualError::EmptyInput);
    }
    let total: f32 = (0..t).map(|k| acc_matrix.data[k][k]).sum();
    let p = total / t as f32;
    if !p.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "plasticity",
        });
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_retention_matrix(n: usize, acc: f32) -> AccuracyMatrix {
        let mut mat = AccuracyMatrix::new(n);
        for t in 0..n {
            for k in 0..=t {
                mat.data[t][k] = acc;
            }
        }
        mat
    }

    #[test]
    fn average_forgetting_zero_perfect_retention() {
        let mat = perfect_retention_matrix(4, 0.9);
        let af =
            average_forgetting(&mat).expect("average forgetting should compute on valid matrix");
        assert!(
            af.abs() < 1e-6,
            "Avg forgetting should be 0 for perfect retention, got {af}"
        );
    }

    #[test]
    fn average_forgetting_positive_for_degradation() {
        let mut mat = AccuracyMatrix::new(3);
        // After task 0: 0.9 on task 0
        mat.data[0][0] = 0.9;
        // After task 1: 0.5 on task 0, 0.9 on task 1
        mat.data[1][0] = 0.5;
        mat.data[1][1] = 0.9;
        // After task 2: 0.2 on task 0, 0.4 on task 1, 0.9 on task 2
        mat.data[2][0] = 0.2;
        mat.data[2][1] = 0.4;
        mat.data[2][2] = 0.9;
        let af =
            average_forgetting(&mat).expect("average forgetting should compute on valid matrix");
        // For task 0: max(0.9, 0.5) - 0.2 = 0.7
        // For task 1: max(0.0, 0.9) - 0.4 = 0.5
        // AF = (0.7 + 0.5) / 2 = 0.6
        assert!((af - 0.6).abs() < 1e-5, "Expected AF = 0.6, got {af}");
    }

    #[test]
    fn backward_transfer_negative_for_forgetting() {
        let mut mat = AccuracyMatrix::new(3);
        mat.data[0][0] = 0.9;
        mat.data[1][0] = 0.5;
        mat.data[1][1] = 0.85;
        mat.data[2][0] = 0.3;
        mat.data[2][1] = 0.4;
        mat.data[2][2] = 0.88;
        let bwt =
            backward_transfer(&mat).expect("backward transfer should compute on valid matrix");
        // BWT = ((0.3 - 0.9) + (0.4 - 0.85)) / 2 = (-0.6 - 0.45) / 2 = -0.525
        assert!(
            bwt < 0.0,
            "BWT should be negative for forgetting, got {bwt}"
        );
    }

    #[test]
    fn backward_transfer_zero_perfect_retention() {
        let mat = perfect_retention_matrix(4, 0.95);
        let bwt =
            backward_transfer(&mat).expect("backward transfer should compute on valid matrix");
        assert!(
            bwt.abs() < 1e-6,
            "BWT should be 0 for perfect retention, got {bwt}"
        );
    }

    #[test]
    fn single_task_forgetting_is_zero() {
        let mut mat = AccuracyMatrix::new(1);
        mat.data[0][0] = 0.9;
        let af =
            average_forgetting(&mat).expect("average forgetting should compute on valid matrix");
        assert_eq!(af, 0.0);
    }

    #[test]
    fn plasticity_averages_diagonal() {
        let mut mat = AccuracyMatrix::new(3);
        mat.data[0][0] = 0.8;
        mat.data[1][1] = 0.9;
        mat.data[2][2] = 0.7;
        let p = plasticity(&mat).expect("plasticity should compute on valid matrix");
        assert!(
            (p - 0.8).abs() < 1e-6,
            "Plasticity should be avg of diagonal"
        );
    }

    #[test]
    fn accuracy_matrix_set_out_of_range() {
        let mut mat = AccuracyMatrix::new(3);
        assert!(mat.set(5, 0, 0.5).is_err());
    }
}
