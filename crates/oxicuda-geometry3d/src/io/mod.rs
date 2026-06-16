//! Point-cloud I/O — pure-`std` ASCII readers for PLY and PCD files.
//!
//! These readers parse the common Open3D / PCL ASCII layouts and return an
//! in-memory [`PointCloud`]. Only the standard library is used (no extra
//! dependencies). Binary PLY/PCD payloads are intentionally rejected with an
//! error rather than silently mis-parsed.

pub mod pcd;
pub mod ply;

pub use pcd::{parse_pcd_str, read_pcd};
pub use ply::{parse_ply_str, read_ply};

/// An in-memory point cloud.
///
/// Coordinates and the optional per-point attributes are stored flattened in
/// `xyz` / `nx ny nz` / `rgb` order. When present, each optional buffer has the
/// same point count as `points` (i.e. length `3 · N`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    /// Point positions, flattened `[N × 3]` as `x0 y0 z0 x1 y1 z1 …`.
    pub points: Vec<f32>,
    /// Optional per-point normals, flattened `[N × 3]`.
    pub normals: Option<Vec<f32>>,
    /// Optional per-point RGB colors, flattened `[N × 3]`. `uchar` color
    /// channels are normalised to `[0, 1]`; float channels are kept as-is.
    pub colors: Option<Vec<f32>>,
}

impl PointCloud {
    /// Number of points in the cloud.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len() / 3
    }

    /// `true` if the cloud has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Borrow the `i`-th point as `[x, y, z]`, if it exists.
    #[must_use]
    pub fn point(&self, i: usize) -> Option<[f32; 3]> {
        let base = i * 3;
        Some([
            *self.points.get(base)?,
            *self.points.get(base + 1)?,
            *self.points.get(base + 2)?,
        ])
    }
}
