//! Sparsemax and Entmax-1.5 probability simplex projections.
//!
//! - `sparsemax`: Martins & Astudillo (2016) — projection onto probability simplex.
//! - `entmax15`: α=1.5 entmax, sparser than softmax, used in NODE feature selection.

use crate::error::{TabularError, TabularResult};

/// Sparsemax: project `z` onto the probability simplex.
///
/// `sparsemax(z)_i = max(0, z_i - τ(z))` where τ is chosen so that the output sums to 1.
///
/// Algorithm (Martins & Astudillo 2016):
/// 1. Sort z descending.
/// 2. Find k*: largest j such that `1 + (j+1)*z_{j} - cumsum_j > 0`.
/// 3. Compute τ = (cumsum_{k*+1} - 1) / (k*+1).
/// 4. Output: `max(0, z_i - τ)`.
pub fn sparsemax(z: &[f32]) -> TabularResult<Vec<f32>> {
    if z.is_empty() {
        return Err(TabularError::EmptyInput);
    }

    // Sort descending
    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find k*
    let mut cumsum = 0.0_f32;
    let mut k_star = 0usize;
    for (j, &z_j) in sorted.iter().enumerate() {
        cumsum += z_j;
        if 1.0 + (j as f32 + 1.0) * z_j - cumsum > 0.0 {
            k_star = j;
        }
    }

    let tau = (sorted.iter().take(k_star + 1).sum::<f32>() - 1.0) / (k_star as f32 + 1.0);

    let out: Vec<f32> = z.iter().map(|&zi| (zi - tau).max(0.0)).collect();
    Ok(out)
}

/// Batch sparsemax: applies sparsemax row-wise to a flat `[batch_size * d]` buffer.
pub fn sparsemax_batch(z: &[f32], batch_size: usize, d: usize) -> TabularResult<Vec<f32>> {
    if z.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if z.len() != batch_size * d {
        return Err(TabularError::DimensionMismatch {
            expected: batch_size * d,
            got: z.len(),
        });
    }
    let mut out = Vec::with_capacity(z.len());
    for b in 0..batch_size {
        let row = &z[b * d..(b + 1) * d];
        let row_out = sparsemax(row)?;
        out.extend_from_slice(&row_out);
    }
    Ok(out)
}

/// Entmax-1.5: a sparser-than-softmax transform.
///
/// For α=1.5: `p_i = max(0, z_i - τ)²`
/// where τ is found via bisection such that `Σ max(0, z_i - τ)² = 1`.
///
/// Note: bisection runs up to 64 iterations; returns error if not converged.
pub fn entmax15(z: &[f32]) -> TabularResult<Vec<f32>> {
    if z.is_empty() {
        return Err(TabularError::EmptyInput);
    }

    let z_max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z_min = z.iter().cloned().fold(f32::INFINITY, f32::min);

    // Bracket: at tau = z_max, sum = 0 (too low); at tau = z_min - 2, sum is very large.
    let mut lo = z_min - 2.0;
    let mut hi = z_max;

    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let sum: f32 = z.iter().map(|&zi| (zi - mid).max(0.0).powi(2)).sum();
        if sum > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let tau = 0.5 * (lo + hi);
    let out: Vec<f32> = z.iter().map(|&zi| (zi - tau).max(0.0).powi(2)).collect();

    // Verify convergence
    let total: f32 = out.iter().sum();
    if (total - 1.0).abs() > 1e-3 {
        return Err(TabularError::NormalizationFailed {
            msg: format!("entmax15 bisection did not converge: sum={total}"),
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparsemax_uniform_sums_to_one() {
        let z = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sparsemax_sparse_result() {
        // Large gap → only max survives
        let z = vec![10.0_f32, 0.0, 0.0, 0.0];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!(out[1..].iter().all(|&v| v < 1e-6));
    }

    #[test]
    fn entmax15_sums_to_one() {
        let z = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = entmax15(&z).expect("entmax15 should succeed");
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-3);
    }

    #[test]
    fn sparsemax_batch_shape() {
        let z = vec![1.0_f32; 6];
        let out = sparsemax_batch(&z, 2, 3).expect("sparsemax_batch should succeed");
        assert_eq!(out.len(), 6);
    }
}
