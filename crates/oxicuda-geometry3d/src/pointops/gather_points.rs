//! Indexed feature gather for point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Gather features by indices: `in [n×c]`, `idx [k]` → `out [k×c]`.
///
/// Bounds-checks each index against `n`.
pub fn gather_points(
    features: &[f32],
    n: usize,
    c: usize,
    indices: &[usize],
) -> Geom3dResult<Vec<f32>> {
    if features.len() != n * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * c,
            got: features.len(),
        });
    }
    if n == 0 {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        return Err(Geom3dError::EmptyPointCloud);
    }

    let k = indices.len();
    let mut out = vec![0.0_f32; k * c];

    for (j, &idx) in indices.iter().enumerate() {
        if idx >= n {
            return Err(Geom3dError::InvalidK { k: idx, n });
        }
        out[j * c..(j + 1) * c].copy_from_slice(&features[idx * c..(idx + 1) * c]);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_basic() {
        let feat = vec![
            1.0_f32, 2.0, // point 0
            3.0, 4.0, // point 1
            5.0, 6.0, // point 2
        ];
        let idx = vec![2usize, 0, 1];
        let out = gather_points(&feat, 3, 2, &idx).unwrap();
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gather_empty_indices() {
        let feat = vec![1.0_f32, 2.0];
        let out = gather_points(&feat, 1, 2, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn gather_empty_features_empty_indices() {
        let out = gather_points(&[], 0, 3, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn gather_out_of_bounds_error() {
        let feat = vec![1.0_f32, 2.0];
        let idx = vec![5usize];
        assert!(gather_points(&feat, 1, 2, &idx).is_err());
    }

    #[test]
    fn gather_single_channel() {
        let feat = vec![10.0_f32, 20.0, 30.0];
        let idx = vec![1usize, 2, 0];
        let out = gather_points(&feat, 3, 1, &idx).unwrap();
        assert_eq!(out, vec![20.0, 30.0, 10.0]);
    }
}
