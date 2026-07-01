/// Compute the mean squared error between two output vectors.
///
/// Returns `Σ(a_i - b_i)² / n`. Returns `0.0` if `a` is empty.
/// Both slices must have equal length.
#[must_use]
pub fn output_mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum();
    sum_sq / n as f32
}

/// Check whether two output vectors are consistent within a tolerance.
///
/// Returns `true` iff `output_mse(merged_out, adapter_out) < tol`.
#[must_use]
pub fn output_consistency(merged_out: &[f32], adapter_out: &[f32], tol: f32) -> bool {
    output_mse(merged_out, adapter_out) < tol
}

/// Compute the maximum absolute element-wise difference between two vectors.
///
/// Returns `0.0` if `a` is empty.
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f32, |acc, (&ai, &bi)| acc.max((ai - bi).abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──── output_mse ──────────────────────────────────────────────────────────

    #[test]
    fn mse_equal_vecs_is_zero() {
        let v = [1.0_f32, 2.0, -3.0];
        assert_eq!(
            output_mse(&v, &v),
            0.0,
            "MSE of identical vectors must be 0.0"
        );
    }

    #[test]
    fn mse_empty_is_zero() {
        assert_eq!(output_mse(&[], &[]), 0.0, "MSE of empty slices must be 0.0");
    }

    #[test]
    fn mse_single_element_known_value() {
        // ([0] vs [2])^2 / 1 = 4.0
        let got = output_mse(&[0.0_f32], &[2.0_f32]);
        assert!((got - 4.0).abs() < 1e-6, "expected 4.0, got {got}");
    }

    #[test]
    fn mse_multi_element_analytic() {
        // a=[1,3], b=[0,1]: ((1-0)^2 + (3-1)^2) / 2 = (1+4)/2 = 2.5
        let a = [1.0_f32, 3.0];
        let b = [0.0_f32, 1.0];
        let got = output_mse(&a, &b);
        assert!((got - 2.5).abs() < 1e-6, "expected 2.5, got {got}");
    }

    // ──── output_consistency ────────────────────────────────────────────────

    #[test]
    fn consistency_equal_vecs_returns_true() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!(
            output_consistency(&v, &v, 0.01),
            "identical vectors must be consistent for any positive tolerance"
        );
    }

    #[test]
    fn consistency_false_when_mse_exceeds_tol() {
        // MSE([0],[2]) = 4.0; tol = 1.0 → MSE >= tol → false
        assert!(
            !output_consistency(&[0.0_f32], &[2.0_f32], 1.0),
            "MSE 4.0 must not be consistent within tol 1.0"
        );
    }

    #[test]
    fn consistency_true_when_mse_just_below_tol() {
        // MSE([0],[2]) = 4.0; tol = 5.0 → MSE < tol → true
        assert!(
            output_consistency(&[0.0_f32], &[2.0_f32], 5.0),
            "MSE 4.0 must be consistent within tol 5.0"
        );
    }

    // ──── max_abs_diff ────────────────────────────────────────────────────────

    #[test]
    fn max_abs_diff_empty_is_zero() {
        assert_eq!(
            max_abs_diff(&[], &[]),
            0.0,
            "max_abs_diff of empty slices must be 0.0"
        );
    }

    #[test]
    fn max_abs_diff_equal_vecs_is_zero() {
        let v = [3.0_f32, -1.0, 2.0];
        assert_eq!(
            max_abs_diff(&v, &v),
            0.0,
            "identical vectors: max_abs_diff must be 0.0"
        );
    }

    #[test]
    fn max_abs_diff_analytic() {
        // |1-2|=1, |5-2|=3, |3-4|=1 → max = 3.0
        let a = [1.0_f32, 5.0, 3.0];
        let b = [2.0_f32, 2.0, 4.0];
        let got = max_abs_diff(&a, &b);
        assert!((got - 3.0).abs() < 1e-6, "expected 3.0, got {got}");
    }
}
