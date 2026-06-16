//! Group features for neighborhood operations.

use crate::error::{Geom3dError, Geom3dResult};

/// Group features for neighborhoods: `features [n×c]`, `idx [k×s]` → `out [k×s×c]`.
///
/// For each of `k` centers and `s` neighbors per center, gathers the features
/// into a contiguous output tensor.
pub fn group_features(
    features: &[f32],
    n: usize,
    c: usize,
    indices: &[usize],
    k: usize,
    s: usize,
) -> Geom3dResult<Vec<f32>> {
    if features.len() != n * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * c,
            got: features.len(),
        });
    }
    if indices.len() != k * s {
        return Err(Geom3dError::DimensionMismatch {
            expected: k * s,
            got: indices.len(),
        });
    }
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }

    let mut out = vec![0.0_f32; k * s * c];

    for ki in 0..k {
        for si in 0..s {
            let pt_idx = indices[ki * s + si];
            if pt_idx == usize::MAX {
                // sentinel: leave as zeros
                continue;
            }
            if pt_idx >= n {
                return Err(Geom3dError::InvalidK { k: pt_idx, n });
            }
            let src = &features[pt_idx * c..(pt_idx + 1) * c];
            let dst_start = (ki * s + si) * c;
            out[dst_start..dst_start + c].copy_from_slice(src);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_features_basic() {
        // 3 points, 2 channels; k=2 centers, s=2 neighbors
        let feat = vec![
            1.0_f32, 2.0, // pt 0
            3.0, 4.0, // pt 1
            5.0, 6.0, // pt 2
        ];
        let idx = vec![0usize, 1, 1, 2];
        let out = group_features(&feat, 3, 2, &idx, 2, 2).expect("group_features should succeed");
        assert_eq!(out.len(), 2 * 2 * 2);
        // center 0: pts[0]=[1,2], pts[1]=[3,4]
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // center 1: pts[1]=[3,4], pts[2]=[5,6]
        assert_eq!(&out[4..8], &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn group_features_sentinel() {
        let feat = vec![1.0_f32, 2.0];
        let idx = vec![0usize, usize::MAX];
        let out = group_features(&feat, 1, 2, &idx, 1, 2).expect("group_features should succeed");
        // sentinel slot should be zeros
        assert_eq!(&out[2..4], &[0.0, 0.0]);
    }

    #[test]
    fn group_features_dim_mismatch() {
        let feat = vec![1.0_f32, 2.0, 3.0];
        let idx = vec![0usize, 1];
        assert!(group_features(&feat, 1, 2, &idx, 1, 2).is_err());
    }

    #[test]
    fn group_features_oob_error() {
        let feat = vec![1.0_f32, 2.0];
        let idx = vec![5usize];
        assert!(group_features(&feat, 1, 2, &idx, 1, 1).is_err());
    }
}
