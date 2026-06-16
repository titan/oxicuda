//! InfoNCE / NT-Xent contrastive loss helper used by SimCLR and MoCo.
//!
//! Given two `[N, D]` projection matrices `z_a` and `z_b` representing two
//! augmented views of the same `N` items, the symmetric InfoNCE loss is
//!
//! ```text
//!   L = (L_{a → b} + L_{b → a}) / 2
//!   L_{a → b} = - (1/N) Σ_i log( exp(s_{i,i}/τ) / Σ_j exp(s_{i,j}/τ) )
//! ```
//!
//! where `s_{i,j} = (z_a[i] · z_b[j]) / (‖z_a[i]‖·‖z_b[j]‖)` is the cosine
//! similarity between the i-th `a` view and the j-th `b` view.
//!
//! This implementation is numerically stable (log-sum-exp), pure-CPU, and
//! returns the loss together with the per-row top-1 accuracy
//! `frac_i { argmax_j s_{i,j} = i }`.

use crate::error::{SslError, SslResult};

/// Per-row L2 normalisation of a `[N, D]` row-major matrix.
fn l2_normalize_rows(z: &mut [f32], n: usize, d: usize) -> SslResult<()> {
    if z.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z.len(),
        });
    }
    for row in z.chunks_mut(d) {
        let s: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if s > 1e-12 { 1.0 / s } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    Ok(())
}

/// Compute symmetric InfoNCE loss between `[N, D]` projections `z_a` and `z_b`.
///
/// Returns `(loss, accuracy@1)`. Both are `f32`. Inputs are not modified
/// (internal copies are taken for normalisation).
///
/// # Errors
/// - [`SslError::EmptyInput`] when `z_a.is_empty()` or `n == 0`.
/// - [`SslError::DimensionMismatch`] when `z_a.len() != n*d` or `z_b.len() != n*d`.
/// - [`SslError::InvalidTemperature`] if `temperature <= 0` or non-finite.
/// - [`SslError::BatchTooSmall`] if `n < 2`.
pub fn info_nce_loss(
    z_a: &[f32],
    z_b: &[f32],
    n: usize,
    d: usize,
    temperature: f32,
) -> SslResult<(f32, f32)> {
    if z_a.is_empty() || n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(SslError::InvalidTemperature { temp: temperature });
    }
    if n < 2 {
        return Err(SslError::BatchTooSmall);
    }
    if z_a.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_a.len(),
        });
    }
    if z_b.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_b.len(),
        });
    }

    let mut a = z_a.to_vec();
    let mut b = z_b.to_vec();
    l2_normalize_rows(&mut a, n, d)?;
    l2_normalize_rows(&mut b, n, d)?;

    // Similarity matrix `s[i, j] = a[i] · b[j] / temperature`.
    let inv_t = 1.0 / temperature;
    let mut sim = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0_f32;
            let row_a = &a[i * d..(i + 1) * d];
            let row_b = &b[j * d..(j + 1) * d];
            for (x, y) in row_a.iter().zip(row_b.iter()) {
                dot += x * y;
            }
            sim[i * n + j] = dot * inv_t;
        }
    }

    // L_{a → b}: softmax cross-entropy with diagonal labels.
    let l_ab = softmax_cross_entropy_diag(&sim, n);
    // L_{b → a}: same matrix but transposed.
    let mut sim_t = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            sim_t[i * n + j] = sim[j * n + i];
        }
    }
    let l_ba = softmax_cross_entropy_diag(&sim_t, n);
    let loss = 0.5 * (l_ab + l_ba);

    // Top-1 accuracy averaged across both directions.
    let acc_ab = top1_diag_accuracy(&sim, n);
    let acc_ba = top1_diag_accuracy(&sim_t, n);
    let acc = 0.5 * (acc_ab + acc_ba);

    Ok((loss, acc))
}

/// `-1/N · Σ_i (s_{i,i} - logsumexp(s_i,*))`, numerically stable.
fn softmax_cross_entropy_diag(sim: &[f32], n: usize) -> f32 {
    let mut total = 0.0_f64;
    for i in 0..n {
        let row = &sim[i * n..(i + 1) * n];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            if v > max_v {
                max_v = v;
            }
        }
        let mut s = 0.0_f64;
        for &v in row {
            s += ((v - max_v) as f64).exp();
        }
        let log_z = (max_v as f64) + s.ln();
        let target = row[i] as f64;
        total += -(target - log_z);
    }
    (total / n as f64) as f32
}

/// Fraction of rows where `argmax_j sim[i, j] == i`.
fn top1_diag_accuracy(sim: &[f32], n: usize) -> f32 {
    let mut hits = 0usize;
    for i in 0..n {
        let row = &sim[i * n..(i + 1) * n];
        let mut best = 0usize;
        let mut best_v = row[0];
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = j;
            }
        }
        if best == i {
            hits += 1;
        }
    }
    hits as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_nce_distinct_paired_inputs_low_loss() {
        // Distinct rows so off-diagonal cosine similarity is small; identical
        // pair (z_a == z_b) so diagonal is 1.
        let n = 4;
        let d = 8;
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            z[i * d + i] = 1.0; // basis vector — orthogonal between rows.
        }
        let (loss, acc) = info_nce_loss(&z, &z, n, d, 0.1).expect("info_nce_loss should succeed");
        assert!(loss < 0.5, "loss = {loss}");
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn info_nce_random_inputs_high_loss() {
        let n = 16;
        let d = 32;
        let mut z_a = vec![0.0_f32; n * d];
        let mut z_b = vec![0.0_f32; n * d];
        let mut rng_state = 1u64;
        for v in z_a.iter_mut().chain(z_b.iter_mut()) {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((rng_state >> 33) as f32 / (u32::MAX as f32 + 1.0)) - 0.5;
        }
        let (loss, _acc) =
            info_nce_loss(&z_a, &z_b, n, d, 0.1).expect("info_nce_loss should succeed");
        // ln(N) is the loss for completely random inputs.
        let upper_bound = (n as f32).ln() + 1.0;
        assert!(loss > 0.0 && loss < upper_bound, "loss = {loss}");
    }

    #[test]
    fn info_nce_temperature_must_be_positive() {
        let z = vec![0.0_f32; 4];
        assert!(info_nce_loss(&z, &z, 2, 2, 0.0).is_err());
        assert!(info_nce_loss(&z, &z, 2, 2, -1.0).is_err());
        assert!(info_nce_loss(&z, &z, 2, 2, f32::NAN).is_err());
    }

    #[test]
    fn info_nce_rejects_n_lt_2() {
        let z = vec![1.0_f32, 2.0, 3.0];
        assert!(info_nce_loss(&z, &z, 1, 3, 0.1).is_err());
    }

    #[test]
    fn info_nce_rejects_dim_mismatch() {
        let a = vec![1.0_f32; 8];
        let b = vec![1.0_f32; 6];
        assert!(info_nce_loss(&a, &b, 2, 4, 0.1).is_err());
    }

    #[test]
    fn info_nce_rejects_empty() {
        let r = info_nce_loss(&[], &[], 0, 0, 0.1);
        assert!(r.is_err());
    }

    #[test]
    fn l2_normalize_rows_unit_norm() {
        let mut z = vec![1.0_f32, 2.0, 3.0, 4.0];
        l2_normalize_rows(&mut z, 1, 4).expect("l2_normalize_rows should succeed");
        let s: f32 = z.iter().map(|v| v * v).sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_rows_zero_row_safe() {
        let mut z = vec![0.0_f32; 4];
        l2_normalize_rows(&mut z, 1, 4).expect("l2_normalize_rows should succeed");
        // zero row stays zero (no NaN)
        for v in &z {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn info_nce_top1_accuracy_at_least_correct() {
        // Construct a strongly aligned dataset: z_b[i] = z_a[i] for all i.
        let n = 8;
        let d = 4;
        let mut rng = 7u64;
        let mut z_a = vec![0.0_f32; n * d];
        for v in z_a.iter_mut() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = (rng >> 33) as f32 / (u32::MAX as f32 + 1.0);
        }
        let z_b = z_a.clone();
        let (_loss, acc) =
            info_nce_loss(&z_a, &z_b, n, d, 0.1).expect("info_nce_loss should succeed");
        assert!(acc > 0.9, "acc = {acc}");
    }
}
