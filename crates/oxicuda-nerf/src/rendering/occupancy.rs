//! Occupancy grid for accelerating NeRF rendering by skipping empty space.

use crate::error::{NerfError, NerfResult};
use crate::rendering::ray::Ray;

/// 3D occupancy grid for fast ray marching.
///
/// The scene is assumed to span `[-scene_bound, scene_bound]^3`.
/// The grid has `resolution^3` voxels.
#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    /// Flat bool occupancy: `[resolution * resolution * resolution]`.
    pub data: Vec<bool>,
    /// Grid resolution per axis.
    pub resolution: usize,
    /// Scene half-extent: scene spans `[-bound, bound]^3`.
    pub scene_bound: f32,
}

impl OccupancyGrid {
    /// Create a new occupancy grid, all empty.
    ///
    /// # Errors
    ///
    /// Returns `InvalidGridResolution` if `resolution == 0`.
    pub fn new(resolution: usize, scene_bound: f32) -> NerfResult<Self> {
        if resolution == 0 {
            return Err(NerfError::InvalidGridResolution { res: 0 });
        }
        let total = resolution * resolution * resolution;
        Ok(Self {
            data: vec![false; total],
            resolution,
            scene_bound,
        })
    }

    #[inline]
    fn voxel_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.resolution * self.resolution + iy * self.resolution + iz
    }

    /// Mark a voxel as occupied/empty.
    ///
    /// # Errors
    ///
    /// Returns `HashLevelOutOfRange` if any index exceeds the resolution.
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, occupied: bool) -> NerfResult<()> {
        if ix >= self.resolution || iy >= self.resolution || iz >= self.resolution {
            return Err(NerfError::HashLevelOutOfRange {
                level: ix.max(iy).max(iz),
            });
        }
        let idx = self.voxel_index(ix, iy, iz);
        self.data[idx] = occupied;
        Ok(())
    }

    /// Query occupancy of a voxel.
    ///
    /// # Errors
    ///
    /// Returns `HashLevelOutOfRange` if any index exceeds the resolution.
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> NerfResult<bool> {
        if ix >= self.resolution || iy >= self.resolution || iz >= self.resolution {
            return Err(NerfError::HashLevelOutOfRange {
                level: ix.max(iy).max(iz),
            });
        }
        Ok(self.data[self.voxel_index(ix, iy, iz)])
    }

    /// Query if a world-space point lies in an occupied voxel.
    ///
    /// Points outside `[-scene_bound, scene_bound]^3` are considered empty.
    #[must_use]
    pub fn is_occupied_world(&self, xyz: [f32; 3]) -> bool {
        let bound = self.scene_bound;
        // Check bounds
        if xyz[0] < -bound
            || xyz[0] > bound
            || xyz[1] < -bound
            || xyz[1] > bound
            || xyz[2] < -bound
            || xyz[2] > bound
        {
            return false;
        }
        let res = self.resolution as f32;
        let to_idx = |v: f32| -> usize {
            let norm = (v + bound) / (2.0 * bound);
            (norm * res).floor().clamp(0.0, res - 1.0) as usize
        };
        let ix = to_idx(xyz[0]);
        let iy = to_idx(xyz[1]);
        let iz = to_idx(xyz[2]);
        self.data[self.voxel_index(ix, iy, iz)]
    }

    /// Update occupancy from density values using a threshold.
    ///
    /// `density` must have exactly `resolution^3` elements.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if sizes don't match.
    pub fn update_from_density(&mut self, density: &[f32], threshold: f32) -> NerfResult<()> {
        let expected = self.resolution * self.resolution * self.resolution;
        if density.len() != expected {
            return Err(NerfError::DimensionMismatch {
                expected,
                got: density.len(),
            });
        }
        for (occ, &den) in self.data.iter_mut().zip(density.iter()) {
            *occ = den > threshold;
        }
        Ok(())
    }

    /// March a ray and return t values where the grid is occupied.
    ///
    /// Steps along the ray with `step_size` and records t values for occupied voxels.
    #[must_use]
    pub fn march_ray_occupied(
        &self,
        ray: &Ray,
        t_near: f32,
        t_far: f32,
        step_size: f32,
    ) -> Vec<f32> {
        if step_size <= 0.0 || t_far <= t_near {
            return Vec::new();
        }
        let mut t = t_near;
        let mut occupied_t = Vec::new();
        while t <= t_far {
            let pt = ray.at(t);
            if self.is_occupied_world(pt) {
                occupied_t.push(t);
            }
            t += step_size;
        }
        occupied_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_set_get() {
        let mut grid = OccupancyGrid::new(8, 1.0).unwrap();
        grid.set(2, 3, 4, true).unwrap();
        assert!(grid.get(2, 3, 4).unwrap());
        assert!(!grid.get(0, 0, 0).unwrap());
    }

    #[test]
    fn world_query_inside_bound() {
        let mut grid = OccupancyGrid::new(4, 1.0).unwrap();
        // Mark center voxel
        grid.set(2, 2, 2, true).unwrap();
        // World point that maps to (2,2,2) in a 4-res grid spanning [-1,1]
        // cell width = 2/4 = 0.5, center of voxel 2 = -1 + 2.5*0.5 = 0.25
        assert!(grid.is_occupied_world([0.25, 0.25, 0.25]));
    }

    #[test]
    fn update_from_density() {
        let mut grid = OccupancyGrid::new(2, 1.0).unwrap();
        let density = vec![0.1, 0.2, 0.5, 0.8, 0.0, 0.9, 0.3, 0.7];
        grid.update_from_density(&density, 0.4).unwrap();
        // voxels 0,1,2,4,6 below threshold → empty; 3,5,7 above → occupied
        assert!(!grid.data[0]);
        assert!(grid.data[3]);
        assert!(grid.data[5]);
        assert!(grid.data[7]);
    }
}
