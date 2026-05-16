//! Cosine similarity for all hypervector types.

use crate::error::{HdcError, HdcResult};

/// Cosine similarity for binary HVs (via dot / D).
/// For ±1 binary HVs, ||a|| = ||b|| = sqrt(D), so cosine = dot / D.
pub fn cosine_binary(a: &[i8], b: &[i8]) -> HdcResult<f32> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = a.len();
    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai as i64) * (bi as i64))
        .sum();
    Ok(dot as f32 / dim as f32)
}

/// Cosine similarity for integer HVs (via dot / (||a|| * ||b||)).
pub fn cosine_integer(a: &[i32], b: &[i32]) -> HdcResult<f32> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai as i64) * (bi as i64))
        .sum();
    let norm_a: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
        return Err(HdcError::DivisionByZero);
    }
    Ok((dot as f64 / denom) as f32)
}

/// Cosine similarity for real-valued HVs.
pub fn cosine_real(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai as f64) * (bi as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON {
        return Err(HdcError::DivisionByZero);
    }
    Ok((dot / denom) as f32)
}

/// Cosine similarity for complex HVs (Re(a·conj(b)) / D).
/// Stored as interleaved [re_0, im_0, re_1, im_1, ...], length = 2*dim.
pub fn cosine_complex(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if !a.len().is_multiple_of(2) {
        return Err(HdcError::DimensionMismatch {
            expected: a.len() + 1,
            got: a.len(),
        });
    }
    let dim = a.len() / 2;
    let mut re_sum = 0f32;
    let mut i = 0;
    while i < dim {
        // Re(a * conj(b)) = a_re*b_re + a_im*b_im
        re_sum += a[2 * i] * b[2 * i] + a[2 * i + 1] * b[2 * i + 1];
        i += 1;
    }
    Ok(re_sum / dim as f32)
}

/// Argmax cosine: find index in matrix (slice of HVs) most similar to query.
pub fn argmax_cosine_binary(query: &[i8], matrix: &[Vec<i8>]) -> HdcResult<usize> {
    if matrix.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = query.len();
    if dim == 0 {
        return Err(HdcError::EmptyInput);
    }
    let mut best_idx = 0usize;
    let mut best_sim = f32::NEG_INFINITY;
    for (idx, row) in matrix.iter().enumerate() {
        let sim = cosine_binary(query, row)?;
        if sim > best_sim {
            best_sim = sim;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_binary_self_one() {
        let a: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let sim = cosine_binary(&a, &a).expect("cosine");
        assert!((sim - 1.0_f32).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn cosine_binary_opposite_neg_one() {
        let a: Vec<i8> = vec![1, 1, 1, 1];
        let b: Vec<i8> = vec![-1, -1, -1, -1];
        let sim = cosine_binary(&a, &b).expect("cosine");
        assert!((sim + 1.0_f32).abs() < 1e-6, "sim={sim}");
    }

    #[test]
    fn argmax_cosine_binary_finds_match() {
        let q: Vec<i8> = vec![1, -1, 1, -1];
        let matrix = vec![
            vec![-1i8, 1, -1, 1],
            vec![1i8, -1, 1, -1],
            vec![1i8, 1, -1, -1],
        ];
        let best = argmax_cosine_binary(&q, &matrix).expect("argmax");
        assert_eq!(best, 1); // exact match at index 1
    }
}
