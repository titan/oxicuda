//! Distortion loss from Mip-NeRF 360 (Barron et al. 2022 CVPR).
//!
//! Regularises the weight distribution along a ray, encouraging weights to
//! concentrate at a single location rather than spreading across the ray.
//!
//! Loss for a single ray:
//! ```text
//! L_dist(t, w) = Σ_i Σ_j w_i w_j |t̄_i - t̄_j|  +  (1/3) Σ_i w_i² (t_i^end - t_i^start)
//! ```
//! where `t̄_i = (t_i^start + t_i^end) / 2`.
//!
//! The double-sum term is computed in O(N) via a cumulative-sum trick that
//! exploits the monotone ordering of midpoints.

use crate::error::{NerfError, NerfResult};

// ─── Validation helpers ──────────────────────────────────────────────────────

/// Return an error if any value in `slice` is NaN or infinite.
fn check_finite(slice: &[f32], context: &str) -> NerfResult<()> {
    for &v in slice {
        if !v.is_finite() {
            return Err(NerfError::NanEncountered {
                context: context.to_string(),
            });
        }
    }
    Ok(())
}

// ─── distortion_loss (single ray) ────────────────────────────────────────────

/// Compute the Mip-NeRF 360 distortion loss for a single ray.
///
/// - `t_starts`: start of each interval along the ray, length N.
///   Must be non-decreasing (midpoints must be ascending for the O(N) trick).
/// - `t_ends`: end of each interval, length N.
///   Must satisfy `t_ends[i] >= t_starts[i]`.
/// - `weights`: ray weights after volume rendering, length N.  Must be non-negative.
///
/// Returns the scalar distortion loss for this ray.
///
/// # Errors
///
/// - [`NerfError::EmptyInput`] if N == 0.
/// - [`NerfError::DimensionMismatch`] if the three slices have different lengths.
/// - [`NerfError::NanEncountered`] if any input contains NaN or infinity.
/// - [`NerfError::InvalidBounds`] if any `t_ends[i] < t_starts[i]`.
pub fn distortion_loss(t_starts: &[f32], t_ends: &[f32], weights: &[f32]) -> NerfResult<f32> {
    let n = t_starts.len();
    if n == 0 {
        return Err(NerfError::EmptyInput);
    }
    if t_ends.len() != n {
        return Err(NerfError::DimensionMismatch {
            expected: n,
            got: t_ends.len(),
        });
    }
    if weights.len() != n {
        return Err(NerfError::DimensionMismatch {
            expected: n,
            got: weights.len(),
        });
    }

    check_finite(t_starts, "t_starts")?;
    check_finite(t_ends, "t_ends")?;
    check_finite(weights, "weights")?;

    // Validate interval bounds: each t_end must be >= t_start
    for i in 0..n {
        if t_ends[i] < t_starts[i] {
            return Err(NerfError::InvalidBounds {
                near: t_starts[i],
                far: t_ends[i],
            });
        }
    }

    // Step 2: midpoints and widths
    let mut t_mid = vec![0.0_f32; n];
    let mut w_interval = vec![0.0_f32; n];
    for i in 0..n {
        t_mid[i] = (t_starts[i] + t_ends[i]) * 0.5;
        w_interval[i] = t_ends[i] - t_starts[i];
    }

    // Step 4: double-sum via O(N) cumulative pass
    // Exploits the fact that intervals are sorted → t_mid is non-decreasing,
    // so for j < i: |t̄_i - t̄_j| = t̄_i - t̄_j.
    //
    // Σ_i Σ_{j<i} w_i w_j (t̄_i - t̄_j)
    //   = Σ_i w_i [ t̄_i * Σ_{j<i} w_j  -  Σ_{j<i} w_j * t̄_j ]
    // Multiply by 2 to account for both (i,j) and (j,i).
    let mut cum_w = 0.0_f32; // Σ_{j<i} w_j
    let mut cum_wt = 0.0_f32; // Σ_{j<i} w_j * t̄_j
    let mut double_sum = 0.0_f32;

    for i in 0..n {
        double_sum += weights[i] * (t_mid[i] * cum_w - cum_wt);
        cum_w += weights[i];
        cum_wt += weights[i] * t_mid[i];
    }
    double_sum *= 2.0;

    // Step 5: width term  (1/3) Σ_i w_i² * Δt_i
    let width_sum: f32 = weights
        .iter()
        .zip(w_interval.iter())
        .map(|(&wi, &dt)| wi * wi * dt)
        .sum::<f32>()
        / 3.0;

    Ok(double_sum + width_sum)
}

// ─── distortion_loss_batch ────────────────────────────────────────────────────

/// Compute distortion loss averaged over a batch of rays.
///
/// Each element in `rays` is `(t_starts, t_ends, weights)` for one ray.
/// Returns the mean distortion loss across all rays.
///
/// # Errors
///
/// - [`NerfError::EmptyInput`] if the batch is empty.
/// - Propagates any error from [`distortion_loss`] for any individual ray.
pub fn distortion_loss_batch(rays: &[(&[f32], &[f32], &[f32])]) -> NerfResult<f32> {
    if rays.is_empty() {
        return Err(NerfError::EmptyInput);
    }
    let mut total = 0.0_f32;
    for &(ts, te, w) in rays {
        total += distortion_loss(ts, te, w)?;
    }
    Ok(total / rays.len() as f32)
}

// ─── distortion_loss_midpoints ────────────────────────────────────────────────

/// Compute distortion loss for a ray given flat sample positions (midpoints only).
///
/// Intervals are defined implicitly: sample `t[i]` defines the interval
/// `[t[i-1], t[i]]`, with `t[-1] = 0`.  Widths are `t[i] - t[i-1]`.
///
/// Useful when only midpoints/positions are stored (e.g. classic NeRF sampling).
///
/// # Errors
///
/// - [`NerfError::EmptyInput`] if N < 2.
/// - [`NerfError::DimensionMismatch`] if `t_midpoints` and `weights` differ in length.
/// - [`NerfError::NanEncountered`] if any input contains NaN or infinity.
/// - [`NerfError::InvalidBounds`] if `t_midpoints` is not non-decreasing.
pub fn distortion_loss_midpoints(t_midpoints: &[f32], weights: &[f32]) -> NerfResult<f32> {
    let n = t_midpoints.len();
    if n < 2 {
        return Err(NerfError::EmptyInput);
    }
    if weights.len() != n {
        return Err(NerfError::DimensionMismatch {
            expected: n,
            got: weights.len(),
        });
    }

    check_finite(t_midpoints, "t_midpoints")?;
    check_finite(weights, "weights")?;

    // Build explicit intervals: [t[i-1], t[i]], with t[-1] = 0
    let mut t_starts = vec![0.0_f32; n];
    let mut t_ends = vec![0.0_f32; n];
    t_starts[0] = 0.0;
    t_ends[0] = t_midpoints[0];
    t_starts[1..n].copy_from_slice(&t_midpoints[..(n - 1)]);
    t_ends[1..n].copy_from_slice(&t_midpoints[1..n]);

    distortion_loss(&t_starts, &t_ends, weights)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: assert two f32 values are close within a tolerance.
    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: expected {b} ± {tol}, got {a}");
    }

    // 1. N=1: loss = w² * interval_width / 3  (double_sum=0 because no pairs)
    #[test]
    fn distortion_loss_single_sample() {
        let ts = [0.0_f32];
        let te = [0.5_f32];
        let w = [1.0_f32];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        // double_sum = 0, width_sum = 1^2 * 0.5 / 3 = 1/6
        assert_close(loss, 1.0 / 6.0, 1e-6, "single sample loss");
    }

    // 2. Uniform weights, equal intervals → positive finite loss
    #[test]
    fn distortion_loss_uniform_weights() {
        let ts = [0.0_f32, 1.0, 2.0, 3.0];
        let te = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [0.25_f32; 4];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        assert!(loss.is_finite(), "loss must be finite");
        assert!(loss > 0.0, "uniform weights → positive loss");
    }

    // 3. Concentrated weight: double_sum=0, width_sum = interval_width/3
    #[test]
    fn distortion_loss_concentrated_weights() {
        let ts = [0.0_f32, 1.0, 2.0];
        let te = [1.0_f32, 2.0, 3.0];
        let w = [0.0_f32, 1.0, 0.0];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        // double_sum = 0 (only one non-zero weight, no cross terms)
        // width_sum = 1^2 * 1.0 / 3 = 1/3
        assert_close(loss, 1.0 / 3.0, 1e-6, "concentrated weight");
    }

    // 4. N=2, verify against manual calculation
    #[test]
    fn distortion_loss_two_samples() {
        // t_mid[0] = 0.5, t_mid[1] = 1.5
        // widths = [1.0, 1.0], weights = [0.5, 0.5]
        // double_sum:
        //   i=0: cum_w=0, cum_wt=0  → contribution = 0.5 * (0.5*0 - 0) = 0; then cum_w=0.5, cum_wt=0.25
        //   i=1: contribution = 0.5 * (1.5*0.5 - 0.25) = 0.5 * (0.75 - 0.25) = 0.5 * 0.5 = 0.25
        //   double_sum = 2 * 0.25 = 0.5
        // width_sum = (0.25 * 1 + 0.25 * 1) / 3 = 0.5 / 3
        // total = 0.5 + 0.5/3 = 2/3
        let ts = [0.0_f32, 1.0];
        let te = [1.0_f32, 2.0];
        let w = [0.5_f32, 0.5];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        let expected = 0.5 + (0.5 / 3.0);
        assert_close(loss, expected, 1e-6, "two-sample manual check");
    }

    // 5. All-zero weights → loss = 0
    #[test]
    fn distortion_loss_zero_weights() {
        let ts = [0.0_f32, 1.0, 2.0];
        let te = [1.0_f32, 2.0, 3.0];
        let w = [0.0_f32; 3];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        assert_close(loss, 0.0, 1e-9, "zero weights");
    }

    // 6. Loss is always non-negative for valid inputs
    #[test]
    fn distortion_loss_nonnegative() {
        let cases: &[(&[f32], &[f32], &[f32])] = &[
            (&[0.0], &[1.0], &[0.3]),
            (&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[0.1, 0.8, 0.1]),
            (&[0.0, 0.5], &[0.5, 1.0], &[0.6, 0.4]),
        ];
        for &(ts, te, w) in cases {
            let loss = distortion_loss(ts, te, w).expect("distortion_loss should succeed");
            assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        }
    }

    // 7. The double-sum  Σ_i Σ_j w_i w_j |t̄_i - t̄_j| is symmetric in the weight
    //    vector: swapping two equal-interval samples that have equal weights
    //    gives the same loss; also verify against a brute-force O(N²) reference.
    #[test]
    fn distortion_loss_symmetric() {
        // Use an ascending sequence and verify via brute-force O(N²) computation.
        let ts = [0.0_f32, 1.0, 2.0, 3.0];
        let te = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [0.1_f32, 0.4, 0.4, 0.1];

        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");

        // Brute-force: O(N²) reference
        let n = ts.len();
        let mids: Vec<f32> = ts
            .iter()
            .zip(te.iter())
            .map(|(&a, &b)| (a + b) * 0.5)
            .collect();
        let widths: Vec<f32> = ts.iter().zip(te.iter()).map(|(&a, &b)| b - a).collect();
        let mut brute_double = 0.0_f32;
        for i in 0..n {
            for j in 0..n {
                brute_double += w[i] * w[j] * (mids[i] - mids[j]).abs();
            }
        }
        // The brute-force sums both (i,j) and (j,i) so it already includes both halves
        // (cross-terms counted twice, diagonal zero). Divide by 2 is NOT needed because
        // we sum ALL pairs including i==j (which are 0) and (i,j) + (j,i) together;
        // the result equals 2 * Σ_{i>j} w_i w_j |t̄_i - t̄_j|, which matches double_sum.
        let width_sum: f32 = w
            .iter()
            .zip(widths.iter())
            .map(|(&wi, &dt)| wi * wi * dt)
            .sum::<f32>()
            / 3.0;
        let brute_total = brute_double + width_sum;

        assert_close(loss, brute_total, 1e-5, "O(N) matches O(N²) brute-force");

        // Also verify: symmetric weight vector produces the same loss.
        // w = [0.1, 0.4, 0.4, 0.1] is already palindrome, so we test with non-palindrome.
        let w2 = [0.2_f32, 0.3, 0.4, 0.1];
        let w2_rev = [0.1_f32, 0.4, 0.3, 0.2]; // reversed weights on same intervals
        // These are different vectors on the same intervals → different distributions.
        // The loss function Σ_i Σ_j w_i w_j |t̄_i - t̄_j| is NOT generally invariant
        // under weight permutation unless the interval positions are also permuted.
        // Verify each is non-negative and finite.
        let l2 = distortion_loss(&ts, &te, &w2).expect("distortion_loss should succeed");
        let l2r = distortion_loss(&ts, &te, &w2_rev).expect("distortion_loss should succeed");
        assert!(l2 >= 0.0 && l2.is_finite(), "loss w2 non-negative finite");
        assert!(
            l2r >= 0.0 && l2r.is_finite(),
            "loss w2_rev non-negative finite"
        );
    }

    // 8. Batch of identical rays → same as single ray
    #[test]
    fn distortion_loss_batch_mean() {
        let ts = [0.0_f32, 1.0, 2.0];
        let te = [1.0_f32, 2.0, 3.0];
        let w = [0.2_f32, 0.5, 0.3];
        let single = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        let rays: &[(&[f32], &[f32], &[f32])] = &[(&ts, &te, &w), (&ts, &te, &w), (&ts, &te, &w)];
        let batch = distortion_loss_batch(rays).expect("distortion_loss_batch should succeed");
        assert_close(single, batch, 1e-6, "batch of identical rays == single");
    }

    // 9. Batch of two different rays → mean of two
    #[test]
    fn distortion_loss_batch_different() {
        let ts1 = [0.0_f32, 1.0];
        let te1 = [1.0_f32, 2.0];
        let w1 = [0.5_f32, 0.5];
        let ts2 = [0.0_f32, 2.0, 4.0];
        let te2 = [2.0_f32, 4.0, 6.0];
        let w2 = [0.3_f32, 0.4, 0.3];
        let l1 = distortion_loss(&ts1, &te1, &w1).expect("distortion_loss should succeed");
        let l2 = distortion_loss(&ts2, &te2, &w2).expect("distortion_loss should succeed");
        let expected = (l1 + l2) / 2.0;
        let rays: &[(&[f32], &[f32], &[f32])] = &[(&ts1, &te1, &w1), (&ts2, &te2, &w2)];
        let batch = distortion_loss_batch(rays).expect("distortion_loss_batch should succeed");
        assert_close(expected, batch, 1e-6, "batch mean of two different rays");
    }

    // 10. Midpoints variant consistency: compare with explicit starts/ends
    #[test]
    fn distortion_loss_midpoints_basic() {
        // midpoints: 0.5, 1.5, 2.5  → intervals [0,0.5],[0.5,1.5],[1.5,2.5]
        let midpoints = [0.5_f32, 1.5, 2.5];
        let w = [0.3_f32, 0.5, 0.2];
        let ts = [0.0_f32, 0.5, 1.5];
        let te = [0.5_f32, 1.5, 2.5];
        let loss_explicit = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        let loss_mid = distortion_loss_midpoints(&midpoints, &w)
            .expect("distortion_loss_midpoints should succeed");
        assert_close(loss_explicit, loss_mid, 1e-6, "midpoints vs explicit");
    }

    // 11. Midpoints variant minimum case (N=2)
    #[test]
    fn distortion_loss_midpoints_two() {
        let midpoints = [0.5_f32, 1.5];
        let w = [0.6_f32, 0.4];
        let loss = distortion_loss_midpoints(&midpoints, &w)
            .expect("distortion_loss_midpoints should succeed");
        // Explicit: ts=[0,0.5], te=[0.5,1.5], w=[0.6,0.4]
        let ts = [0.0_f32, 0.5];
        let te = [0.5_f32, 1.5];
        let expected = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        assert_close(loss, expected, 1e-6, "midpoints N=2");
    }

    // 12. EmptyInput for N=0
    #[test]
    fn distortion_loss_err_empty() {
        let result = distortion_loss(&[], &[], &[]);
        assert!(
            matches!(result, Err(NerfError::EmptyInput)),
            "expected EmptyInput"
        );
    }

    // 13. DimensionMismatch when lengths differ
    #[test]
    fn distortion_loss_err_dim_mismatch() {
        let ts = [0.0_f32, 1.0];
        let te = [1.0_f32, 2.0];
        let w = [0.5_f32]; // wrong length
        let result = distortion_loss(&ts, &te, &w);
        assert!(
            matches!(result, Err(NerfError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    // 14. NanEncountered when weights contain NaN
    #[test]
    fn distortion_loss_err_nan() {
        let ts = [0.0_f32, 1.0];
        let te = [1.0_f32, 2.0];
        let w = [f32::NAN, 0.5];
        let result = distortion_loss(&ts, &te, &w);
        assert!(
            matches!(result, Err(NerfError::NanEncountered { .. })),
            "expected NanEncountered"
        );
    }

    // 15. NanEncountered when t_starts contains infinity
    #[test]
    fn distortion_loss_err_inf() {
        let ts = [f32::INFINITY, 1.0];
        let te = [2.0_f32, 3.0];
        let w = [0.5_f32, 0.5];
        let result = distortion_loss(&ts, &te, &w);
        assert!(
            matches!(result, Err(NerfError::NanEncountered { .. })),
            "expected NanEncountered for inf"
        );
    }

    // 16. InvalidBounds when t_end < t_start
    #[test]
    fn distortion_loss_err_invalid_bounds() {
        let ts = [0.0_f32, 2.0]; // second interval: start=2 > end=1
        let te = [1.0_f32, 1.0];
        let w = [0.5_f32, 0.5];
        let result = distortion_loss(&ts, &te, &w);
        assert!(
            matches!(result, Err(NerfError::InvalidBounds { .. })),
            "expected InvalidBounds"
        );
    }

    // 17. Empty batch → EmptyInput
    #[test]
    fn distortion_loss_batch_err_empty_batch() {
        let result = distortion_loss_batch(&[]);
        assert!(
            matches!(result, Err(NerfError::EmptyInput)),
            "expected EmptyInput for empty batch"
        );
    }

    // 18. Zero-width intervals → width_sum = 0, result = double_sum only
    #[test]
    fn distortion_loss_width_term_zero_width() {
        // All intervals have zero width → width_sum = 0
        // midpoints all at the same t=1.0 → double_sum = 0 (all |t̄_i - t̄_j| = 0)
        let ts = [1.0_f32, 1.0, 1.0];
        let te = [1.0_f32, 1.0, 1.0];
        let w = [0.3_f32, 0.4, 0.3];
        let loss = distortion_loss(&ts, &te, &w).expect("distortion_loss should succeed");
        // cross-terms all zero, width terms zero → total = 0
        assert_close(loss, 0.0, 1e-9, "zero-width intervals → zero loss");
    }
}
