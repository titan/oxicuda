//! Voxel grid scatter for 3D point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Voxel pooling mode.
#[derive(Debug, Clone, PartialEq)]
pub enum VoxelPoolMode {
    Mean,
    Max,
    Sum,
}

/// Dense voxel grid with feature storage.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub dims: [u32; 3],
    pub channels: usize,
    pub data: Vec<f32>,   // [dims[0]*dims[1]*dims[2] * channels]
    pub counts: Vec<u32>, // [dims[0]*dims[1]*dims[2]]
}

impl VoxelGrid {
    /// Create a new voxel grid, initialized to zeros.
    pub fn new(origin: [f32; 3], voxel_size: f32, dims: [u32; 3], channels: usize) -> Self {
        let total = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        Self {
            origin,
            voxel_size,
            dims,
            channels,
            data: vec![0.0_f32; total * channels],
            counts: vec![0u32; total],
        }
    }

    /// Scatter points + features into grid.
    ///
    /// `points [n×3]`, `features [n×c]`. Mode controls aggregation.
    pub fn scatter(
        &mut self,
        points: &[f32],
        n: usize,
        features: &[f32],
        mode: VoxelPoolMode,
    ) -> Geom3dResult<()> {
        if n == 0 {
            return Ok(());
        }
        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }
        if features.len() != n * self.channels {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * self.channels,
                got: features.len(),
            });
        }

        let vx = self.voxel_size;
        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;
        let dz = self.dims[2] as usize;
        let c = self.channels;

        // For Max mode: initialize data to -inf for occupied voxels
        if mode == VoxelPoolMode::Max {
            // Reset data to -inf before scatter
            for v in &mut self.data {
                *v = f32::NEG_INFINITY;
            }
        }

        for i in 0..n {
            let px = points[i * 3] - self.origin[0];
            let py = points[i * 3 + 1] - self.origin[1];
            let pz = points[i * 3 + 2] - self.origin[2];

            let ix = (px / vx).floor() as i32;
            let iy = (py / vx).floor() as i32;
            let iz = (pz / vx).floor() as i32;

            if ix < 0 || iy < 0 || iz < 0 {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            let iz = iz as usize;
            if ix >= dx || iy >= dy || iz >= dz {
                continue;
            }

            let vox_idx = ix * dy * dz + iy * dz + iz;
            self.counts[vox_idx] += 1;

            let feat = &features[i * c..(i + 1) * c];
            match mode {
                VoxelPoolMode::Mean | VoxelPoolMode::Sum => {
                    for (ch, &fv) in feat.iter().enumerate() {
                        self.data[vox_idx * c + ch] += fv;
                    }
                }
                VoxelPoolMode::Max => {
                    for (ch, &fv) in feat.iter().enumerate() {
                        if fv > self.data[vox_idx * c + ch] {
                            self.data[vox_idx * c + ch] = fv;
                        }
                    }
                }
            }
        }

        // For Mean: divide by count
        if mode == VoxelPoolMode::Mean {
            let total = dx * dy * dz;
            for vox_idx in 0..total {
                let cnt = self.counts[vox_idx];
                if cnt > 0 {
                    for ch in 0..c {
                        self.data[vox_idx * c + ch] /= cnt as f32;
                    }
                }
            }
        }

        // For Max: replace -inf with 0 in empty voxels
        if mode == VoxelPoolMode::Max {
            let total = dx * dy * dz;
            for vox_idx in 0..total {
                if self.counts[vox_idx] == 0 {
                    for ch in 0..c {
                        self.data[vox_idx * c + ch] = 0.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Return `(occupied_coords: [m×3 as f32], features: [m×c])` for occupied voxels.
    ///
    /// Coords are voxel center positions.
    pub fn occupied_centroids(&self) -> Geom3dResult<(Vec<f32>, Vec<f32>)> {
        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;
        let dz = self.dims[2] as usize;
        let c = self.channels;
        let half = self.voxel_size * 0.5;

        let mut coords = Vec::new();
        let mut feats = Vec::new();

        for ix in 0..dx {
            for iy in 0..dy {
                for iz in 0..dz {
                    let vox_idx = ix * dy * dz + iy * dz + iz;
                    if self.counts[vox_idx] == 0 {
                        continue;
                    }
                    coords.push(self.origin[0] + ix as f32 * self.voxel_size + half);
                    coords.push(self.origin[1] + iy as f32 * self.voxel_size + half);
                    coords.push(self.origin[2] + iz as f32 * self.voxel_size + half);
                    feats.extend_from_slice(&self.data[vox_idx * c..(vox_idx + 1) * c]);
                }
            }
        }

        Ok((coords, feats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxelgrid_scatter_mean() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 1.0, [4, 4, 4], 1);
        let pts = vec![0.5_f32, 0.5, 0.5, 0.3_f32, 0.3, 0.3];
        let feats = vec![2.0_f32, 4.0_f32];
        grid.scatter(&pts, 2, &feats, VoxelPoolMode::Mean).unwrap();
        let (_, out_feats) = grid.occupied_centroids().unwrap();
        // Both points fall in same voxel → mean = 3.0
        assert_eq!(out_feats.len(), 1);
        assert!((out_feats[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn voxelgrid_scatter_multiple_voxels() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 1.0, [4, 4, 4], 1);
        let pts = vec![
            0.5_f32, 0.5, 0.5, // voxel (0,0,0)
            1.5_f32, 0.5, 0.5, // voxel (1,0,0)
        ];
        let feats = vec![1.0_f32, 2.0_f32];
        grid.scatter(&pts, 2, &feats, VoxelPoolMode::Sum).unwrap();
        let (_, out_feats) = grid.occupied_centroids().unwrap();
        assert_eq!(out_feats.len(), 2);
    }

    #[test]
    fn voxelgrid_out_of_bounds_skipped() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 1.0, [2, 2, 2], 1);
        let pts = vec![100.0_f32, 100.0, 100.0]; // out of bounds
        let feats = vec![1.0_f32];
        grid.scatter(&pts, 1, &feats, VoxelPoolMode::Sum).unwrap();
        let (coords, _) = grid.occupied_centroids().unwrap();
        assert!(coords.is_empty(), "Out-of-bounds point should be skipped");
    }

    #[test]
    fn voxelgrid_max_pool() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 1.0, [2, 2, 2], 1);
        let pts = vec![0.1_f32, 0.1, 0.1, 0.2_f32, 0.2, 0.2];
        let feats = vec![3.0_f32, 7.0_f32];
        grid.scatter(&pts, 2, &feats, VoxelPoolMode::Max).unwrap();
        let (_, out_feats) = grid.occupied_centroids().unwrap();
        assert_eq!(out_feats.len(), 1);
        assert!((out_feats[0] - 7.0).abs() < 1e-5);
    }
}
