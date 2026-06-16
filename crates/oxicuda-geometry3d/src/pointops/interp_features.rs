//! 3-NN inverse-distance-weighted feature interpolation.

use crate::error::{Geom3dError, Geom3dResult};

/// 3-NN inverse-distance-weighted interpolation.
///
/// `src_xyz [ns×3]`, `src_feat [ns×c]`, `tgt_xyz [nt×3]` → `out [nt×c]`.
///
/// For each target point: find 3 nearest in `src_xyz`, compute
/// `w_j = 1/(d_j² + 1e-10)`, normalize, weighted sum of features.
pub fn interp_features(
    src_xyz: &[f32],
    src_feat: &[f32],
    ns: usize,
    tgt_xyz: &[f32],
    nt: usize,
    c: usize,
) -> Geom3dResult<Vec<f32>> {
    if ns == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if nt == 0 {
        return Ok(Vec::new());
    }
    if src_xyz.len() != ns * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: ns * 3,
            got: src_xyz.len(),
        });
    }
    if src_feat.len() != ns * c {
        return Err(Geom3dError::DimensionMismatch {
            expected: ns * c,
            got: src_feat.len(),
        });
    }
    if tgt_xyz.len() != nt * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nt * 3,
            got: tgt_xyz.len(),
        });
    }

    let k = 3.min(ns);
    let mut out = vec![0.0_f32; nt * c];

    for ti in 0..nt {
        let tx = tgt_xyz[ti * 3];
        let ty = tgt_xyz[ti * 3 + 1];
        let tz = tgt_xyz[ti * 3 + 2];

        // Find k nearest in src
        let mut dists: Vec<(f32, usize)> = (0..ns)
            .map(|si| {
                let dx = src_xyz[si * 3] - tx;
                let dy = src_xyz[si * 3 + 1] - ty;
                let dz = src_xyz[si * 3 + 2] - tz;
                (dx * dx + dy * dy + dz * dz, si)
            })
            .collect();

        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k_nearest = &dists[..k];
        let weights: Vec<f32> = k_nearest.iter().map(|&(d, _)| 1.0 / (d + 1e-10)).collect();
        let weight_sum: f32 = weights.iter().sum();

        for ch in 0..c {
            let mut val = 0.0_f32;
            for (j, &(_, si)) in k_nearest.iter().enumerate() {
                val += (weights[j] / weight_sum) * src_feat[si * c + ch];
            }
            out[ti * c + ch] = val;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_exact_match() {
        // Target at exact source location → should return that feature exactly
        let src_xyz = vec![0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let src_feat = vec![10.0_f32, 20.0, 30.0];
        let tgt_xyz = vec![0.0_f32, 0.0, 0.0]; // matches src[0]
        let out = interp_features(&src_xyz, &src_feat, 3, &tgt_xyz, 1, 1)
            .expect("interp_features should succeed");
        // Should be very close to 10.0 (dominated by nearest neighbor)
        assert!(
            (out[0] - 10.0).abs() < 0.1,
            "Expected ~10.0, got {}",
            out[0]
        );
    }

    #[test]
    fn interp_midpoint() {
        // Query at midpoint between two equal-distance sources
        let src_xyz = vec![
            -1.0_f32, 0.0, 0.0, // pt 0, feat=0
            1.0_f32, 0.0, 0.0, // pt 1, feat=10
            0.0_f32, 10.0, 0.0, // pt 2 (far)
        ];
        let src_feat = vec![0.0_f32, 10.0, 5.0];
        let tgt_xyz = vec![0.0_f32, 0.0, 0.0];
        let out = interp_features(&src_xyz, &src_feat, 3, &tgt_xyz, 1, 1)
            .expect("interp_features should succeed");
        // Should be symmetric around 5.0
        assert!((out[0] - 5.0).abs() < 0.5, "Expected ~5.0, got {}", out[0]);
    }

    #[test]
    fn interp_empty_src_error() {
        assert!(interp_features(&[], &[], 0, &[1.0, 0.0, 0.0], 1, 1).is_err());
    }

    #[test]
    fn interp_empty_tgt_ok() {
        let src_xyz = vec![0.0_f32, 0.0, 0.0];
        let src_feat = vec![1.0_f32];
        let out = interp_features(&src_xyz, &src_feat, 1, &[], 0, 1)
            .expect("interp_features should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn interp_output_shape() {
        let n_src = 5;
        let n_tgt = 10;
        let c = 4;
        let src_xyz: Vec<f32> = (0..n_src).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let src_feat: Vec<f32> = vec![1.0; n_src * c];
        let tgt_xyz: Vec<f32> = (0..n_tgt)
            .flat_map(|i| vec![i as f32 * 0.5, 0.0, 0.0])
            .collect();
        let out = interp_features(&src_xyz, &src_feat, n_src, &tgt_xyz, n_tgt, c)
            .expect("interp_features should succeed");
        assert_eq!(out.len(), n_tgt * c);
    }
}
