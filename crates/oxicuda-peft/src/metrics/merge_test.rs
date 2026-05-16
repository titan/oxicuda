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
