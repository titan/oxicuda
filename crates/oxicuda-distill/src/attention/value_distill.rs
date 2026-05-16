//! Value-vector VVᵀ distillation (MiniLM — Wang et al. 2020).

use crate::error::{DistillError, DistillResult};

fn softmax_row_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let sum_safe = sum.max(1e-30_f32);
    for v in row.iter_mut() {
        *v /= sum_safe;
    }
}

/// Compute the value relation matrix VVᵀ ∈ ℝ^{seq × seq} and apply row-wise softmax.
///
/// `v` is `[seq_len × head_dim]` flat row-major.
pub fn value_relation_matrix(
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> DistillResult<Vec<f32>> {
    let expected = seq_len * head_dim;
    if v.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: v.len(),
        });
    }
    let mut mat = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let dot: f32 = (0..head_dim)
                .map(|d| v[i * head_dim + d] * v[j * head_dim + d])
                .sum();
            mat[i * seq_len + j] = dot;
        }
        softmax_row_inplace(&mut mat[i * seq_len..(i + 1) * seq_len]);
    }
    Ok(mat)
}

/// MSE between student and teacher value relation matrices.
pub fn value_relation_loss(
    s_v: &[f32],
    t_v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> DistillResult<f32> {
    if s_v.is_empty() || t_v.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let r_s = value_relation_matrix(s_v, seq_len, head_dim)?;
    let r_t = value_relation_matrix(t_v, seq_len, head_dim)?;
    if r_s.len() != r_t.len() {
        return Err(DistillError::DimensionMismatch {
            expected: r_s.len(),
            got: r_t.len(),
        });
    }
    let n = r_s.len() as f32;
    let mse = r_s
        .iter()
        .zip(r_t.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n;
    Ok(mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_relation_shape() {
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 3 × 4
        let mat = value_relation_matrix(&v, 3, 4).unwrap();
        assert_eq!(mat.len(), 9);
    }

    #[test]
    fn value_relation_loss_finite() {
        let s: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let t: Vec<f32> = (0..12).map(|i| i as f32 * 0.11).collect();
        let loss = value_relation_loss(&s, &t, 3, 4).unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }
}
