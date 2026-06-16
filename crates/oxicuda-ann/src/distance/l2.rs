use crate::error::{AnnError, AnnResult};

/// Squared Euclidean distance between two equal-length slices.
pub fn l2_sq(a: &[f32], b: &[f32]) -> AnnResult<f32> {
    if a.len() != b.len() {
        return Err(AnnError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum())
}

/// Euclidean distance between two equal-length slices.
pub fn l2(a: &[f32], b: &[f32]) -> AnnResult<f32> {
    l2_sq(a, b).map(f32::sqrt)
}

/// Compute all B×N pairwise L2² distances.
/// `queries` is row-major `[n_q, dim]`, `db` is row-major `[n_db, dim]`.
/// Returns `[n_q, n_db]` row-major.
pub fn l2_sq_all(
    queries: &[f32],
    db: &[f32],
    n_q: usize,
    n_db: usize,
    dim: usize,
) -> AnnResult<Vec<f32>> {
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim: 0 });
    }
    if queries.len() != n_q * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n_q * dim,
            got: queries.len(),
        });
    }
    if db.len() != n_db * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n_db * dim,
            got: db.len(),
        });
    }

    let mut out = vec![0.0_f32; n_q * n_db];
    for qi in 0..n_q {
        let q = &queries[qi * dim..(qi + 1) * dim];
        for ni in 0..n_db {
            let x = &db[ni * dim..(ni + 1) * dim];
            let d: f32 = q.iter().zip(x.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            out[qi * n_db + ni] = d;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_zero_self() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(l2_sq(&v, &v).expect("test invariant: should succeed"), 0.0);
    }

    #[test]
    fn l2_known() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        assert!((l2(&a, &b).expect("test invariant: should succeed") - 5.0).abs() < 1e-6);
    }
}
