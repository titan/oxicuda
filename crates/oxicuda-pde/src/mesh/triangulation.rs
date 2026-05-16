//! 2D triangular mesh for the FEM P1 solver.

use crate::error::{PdeError, PdeResult};

/// 2D triangular mesh.
///
/// `coords`: flat `[x0, y0, x1, y1, ...]` of length `2 * n_nodes`.
/// `triangles`: flat `[a, b, c, ...]` of node indices, length `3 * n_tri`.
/// `boundary_nodes`: indices of nodes that lie on a Dirichlet boundary.
#[derive(Debug, Clone)]
pub struct TriMesh2d {
    pub coords: Vec<f64>,
    pub triangles: Vec<usize>,
    pub boundary_nodes: Vec<usize>,
}

impl TriMesh2d {
    /// Number of nodes (= `coords.len() / 2`).
    pub fn n_nodes(&self) -> usize {
        self.coords.len() / 2
    }

    /// Number of triangles (= `triangles.len() / 3`).
    pub fn n_tri(&self) -> usize {
        self.triangles.len() / 3
    }

    /// Construct a regular rectangular triangulation: `nx` × `ny` grid split into
    /// 2*(nx-1)*(ny-1) right-triangles.
    pub fn rect_grid(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> PdeResult<Self> {
        if nx < 2 || ny < 2 {
            return Err(PdeError::InvalidGrid(format!(
                "rect_grid requires nx,ny>=2, got nx={nx} ny={ny}"
            )));
        }
        let hx = (x1 - x0) / (nx - 1) as f64;
        let hy = (y1 - y0) / (ny - 1) as f64;
        let mut coords = Vec::with_capacity(2 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                coords.push(x0 + hx * i as f64);
                coords.push(y0 + hy * j as f64);
            }
        }
        let mut triangles = Vec::with_capacity(6 * (nx - 1) * (ny - 1));
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let a = j * nx + i;
                let b = a + 1;
                let c = a + nx;
                let d = c + 1;
                // first triangle: (a, b, d)
                triangles.push(a);
                triangles.push(b);
                triangles.push(d);
                // second triangle: (a, d, c)
                triangles.push(a);
                triangles.push(d);
                triangles.push(c);
            }
        }
        let mut boundary_nodes = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if i == 0 || j == 0 || i + 1 == nx || j + 1 == ny {
                    boundary_nodes.push(j * nx + i);
                }
            }
        }
        Ok(Self {
            coords,
            triangles,
            boundary_nodes,
        })
    }

    /// Get the `(x,y)` coordinates of node `k`.
    pub fn node(&self, k: usize) -> PdeResult<(f64, f64)> {
        if k >= self.n_nodes() {
            return Err(PdeError::IndexOutOfBounds {
                index: k,
                len: self.n_nodes(),
            });
        }
        Ok((self.coords[2 * k], self.coords[2 * k + 1]))
    }

    /// Get the 3 node indices of triangle `e`.
    pub fn tri(&self, e: usize) -> PdeResult<(usize, usize, usize)> {
        if e >= self.n_tri() {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_tri(),
            });
        }
        Ok((
            self.triangles[3 * e],
            self.triangles[3 * e + 1],
            self.triangles[3 * e + 2],
        ))
    }

    /// Compute the (signed) area of triangle `e`.
    pub fn area(&self, e: usize) -> PdeResult<f64> {
        let (i, j, k) = self.tri(e)?;
        let (x0, y0) = self.node(i)?;
        let (x1, y1) = self.node(j)?;
        let (x2, y2) = self.node(k)?;
        Ok(0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_grid_basic_counts() {
        let m = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        assert_eq!(m.n_nodes(), 9);
        assert_eq!(m.n_tri(), 8);
    }

    #[test]
    fn rect_grid_areas_positive() {
        let m = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        for e in 0..m.n_tri() {
            let a = m.area(e).expect("ok");
            assert!(a > 0.0);
        }
    }

    #[test]
    fn rect_grid_boundary_count() {
        let m = TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, 4, 4).expect("ok");
        // 4*4 = 16 nodes, boundary = 12
        assert_eq!(m.boundary_nodes.len(), 12);
    }

    #[test]
    fn rect_grid_total_area() {
        let m = TriMesh2d::rect_grid(0.0, 2.0, 0.0, 1.0, 4, 3).expect("ok");
        let total: f64 = (0..m.n_tri()).map(|e| m.area(e).unwrap_or(0.0)).sum();
        assert!((total - 2.0).abs() < 1.0e-12);
    }
}
