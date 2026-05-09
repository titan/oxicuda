use crate::error::{AnnError, AnnResult};

/// Inner product of two equal-length slices.
pub fn ip(a: &[f32], b: &[f32]) -> AnnResult<f32> {
    if a.len() != b.len() {
        return Err(AnnError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Negated inner product (for max-IP as min problem).
pub fn neg_ip(a: &[f32], b: &[f32]) -> AnnResult<f32> {
    ip(a, b).map(|v| -v)
}

/// Cosine similarity: IP after L2-normalising both vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> AnnResult<f32> {
    if a.len() != b.len() {
        return Err(AnnError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return Ok(0.0);
    }
    let na: Vec<f32> = a.iter().map(|x| x / norm_a).collect();
    let nb: Vec<f32> = b.iter().map(|x| x / norm_b).collect();
    ip(&na, &nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ip_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_eq!(ip(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn cosine_parallel() {
        let a = vec![1.0_f32, 1.0];
        let b = vec![2.0_f32, 2.0];
        let c = cosine_sim(&a, &b).unwrap();
        assert!((c - 1.0).abs() < 1e-6, "cosine={c}");
    }
}
