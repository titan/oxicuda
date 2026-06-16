//! Voxel grid downsampling for point clouds.

use std::collections::HashMap;

use crate::error::{Geom3dError, Geom3dResult};

/// Voxel accumulator: (sum_x, sum_y, sum_z, count, first_idx).
type VoxelAcc = (f64, f64, f64, u32, usize);

/// Voxel grid downsampling.
///
/// Returns `(centroids: Vec<f32> flat n'×3, first_idx: Vec<usize>)`.
///
/// Uses a `HashMap<(i32,i32,i32), (sum_x, sum_y, sum_z, count, first_idx)>`.
/// Key = `(floor(x/v) as i32, ...)`. Emits centroids as `(sum/count) as f32`.
/// Output is sorted by first_idx for determinism.
pub fn voxel_downsample(
    points: &[f32],
    n: usize,
    voxel_size: f32,
) -> Geom3dResult<(Vec<f32>, Vec<usize>)> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: points.len(),
        });
    }
    if voxel_size <= 0.0 || !voxel_size.is_finite() {
        return Err(Geom3dError::InvalidVoxelSize { voxel_size });
    }

    let mut voxel_map: HashMap<(i32, i32, i32), VoxelAcc> = HashMap::new();

    for i in 0..n {
        let x = points[i * 3];
        let y = points[i * 3 + 1];
        let z = points[i * 3 + 2];

        let ix = (x / voxel_size).floor() as i32;
        let iy = (y / voxel_size).floor() as i32;
        let iz = (z / voxel_size).floor() as i32;

        let key = (ix, iy, iz);
        let entry = voxel_map.entry(key).or_insert((0.0, 0.0, 0.0, 0, i));
        entry.0 += x as f64;
        entry.1 += y as f64;
        entry.2 += z as f64;
        entry.3 += 1;
    }

    // Sort by first_idx for determinism
    let mut entries: Vec<((i32, i32, i32), VoxelAcc)> = voxel_map.into_iter().collect();
    entries.sort_unstable_by_key(|(_, v)| v.4);

    let m = entries.len();
    let mut centroids = Vec::with_capacity(m * 3);
    let mut first_indices = Vec::with_capacity(m);

    for (_, (sx, sy, sz, cnt, fi)) in entries {
        let c = cnt as f64;
        centroids.push((sx / c) as f32);
        centroids.push((sy / c) as f32);
        centroids.push((sz / c) as f32);
        first_indices.push(fi);
    }

    Ok((centroids, first_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_downsample_empty_error() {
        assert_eq!(
            voxel_downsample(&[], 0, 1.0),
            Err(Geom3dError::EmptyPointCloud)
        );
    }

    #[test]
    fn voxel_downsample_invalid_voxel_size() {
        let pts = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(
            voxel_downsample(&pts, 1, 0.0),
            Err(Geom3dError::InvalidVoxelSize { voxel_size: 0.0 })
        );
        assert_eq!(
            voxel_downsample(&pts, 1, -1.0),
            Err(Geom3dError::InvalidVoxelSize { voxel_size: -1.0 })
        );
    }

    #[test]
    fn voxel_downsample_single_voxel() {
        // All points within same voxel cell
        let pts = vec![0.1_f32, 0.0, 0.0, 0.2_f32, 0.0, 0.0, 0.3_f32, 0.0, 0.0];
        let (centroids, first_idx) =
            voxel_downsample(&pts, 3, 1.0).expect("voxel_downsample should succeed");
        assert_eq!(centroids.len(), 3); // one voxel => 1 centroid => 3 floats
        assert_eq!(first_idx.len(), 1);
        assert_eq!(first_idx[0], 0);
        assert!((centroids[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn voxel_downsample_multiple_voxels() {
        // 3 distinct voxels
        let pts = vec![
            0.5_f32, 0.0, 0.0, // voxel (0,0,0)
            1.5_f32, 0.0, 0.0, // voxel (1,0,0)
            2.5_f32, 0.0, 0.0, // voxel (2,0,0)
        ];
        let (centroids, first_idx) =
            voxel_downsample(&pts, 3, 1.0).expect("voxel_downsample should succeed");
        assert_eq!(centroids.len(), 9);
        assert_eq!(first_idx.len(), 3);
    }

    #[test]
    fn voxel_downsample_sorted_by_first_idx() {
        let pts = vec![
            2.5_f32, 0.0, 0.0, // voxel (2,0,0), first_idx=0
            0.5_f32, 0.0, 0.0, // voxel (0,0,0), first_idx=1
            1.5_f32, 0.0, 0.0, // voxel (1,0,0), first_idx=2
        ];
        let (_, first_idx) =
            voxel_downsample(&pts, 3, 1.0).expect("voxel_downsample should succeed");
        assert_eq!(first_idx, vec![0, 1, 2], "Must be sorted by first_idx");
    }
}
