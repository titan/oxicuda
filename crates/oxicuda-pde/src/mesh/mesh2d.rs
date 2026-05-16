//! 2D rectangular mesh with optional stretching.

use crate::error::{PdeError, PdeResult};

/// 2D uniform rectangular mesh `[x0,x1] x [y0,y1]` with `nx*ny` nodes.
#[derive(Debug, Clone)]
pub struct Mesh2d {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
    pub nx: usize,
    pub ny: usize,
    pub x_nodes: Vec<f64>,
    pub y_nodes: Vec<f64>,
}

impl Mesh2d {
    /// Build a uniform 2D mesh.
    pub fn uniform(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> PdeResult<Self> {
        if nx < 2 || ny < 2 {
            return Err(PdeError::InvalidGrid(format!(
                "uniform 2d mesh requires nx>=2 ny>=2, got nx={nx} ny={ny}"
            )));
        }
        if x1 <= x0 || y1 <= y0 {
            return Err(PdeError::InvalidGrid(format!(
                "uniform 2d mesh: bad bounds x0={x0} x1={x1} y0={y0} y1={y1}"
            )));
        }
        let hx = (x1 - x0) / (nx - 1) as f64;
        let hy = (y1 - y0) / (ny - 1) as f64;
        let x_nodes = (0..nx).map(|i| x0 + hx * i as f64).collect();
        let y_nodes = (0..ny).map(|j| y0 + hy * j as f64).collect();
        Ok(Self {
            x0,
            x1,
            y0,
            y1,
            nx,
            ny,
            x_nodes,
            y_nodes,
        })
    }

    /// hx (uniform spacing in x).
    pub fn hx(&self) -> f64 {
        if self.nx < 2 {
            0.0
        } else {
            (self.x1 - self.x0) / (self.nx - 1) as f64
        }
    }

    /// hy (uniform spacing in y).
    pub fn hy(&self) -> f64 {
        if self.ny < 2 {
            0.0
        } else {
            (self.y1 - self.y0) / (self.ny - 1) as f64
        }
    }

    /// Total number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nx * self.ny
    }

    /// Linear index `i*ny + j`.
    pub fn idx(&self, i: usize, j: usize) -> PdeResult<usize> {
        if i >= self.nx || j >= self.ny {
            return Err(PdeError::IndexOutOfBounds {
                index: i * self.ny + j,
                len: self.n_nodes(),
            });
        }
        Ok(i * self.ny + j)
    }

    /// Returns true if `(i,j)` is on the rectangle boundary.
    pub fn is_boundary(&self, i: usize, j: usize) -> bool {
        i == 0 || j == 0 || i + 1 == self.nx || j + 1 == self.ny
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_2d_basic() {
        let m = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 3, 4).expect("ok");
        assert_eq!(m.n_nodes(), 12);
        assert!((m.hx() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn mesh_2d_idx() {
        let m = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 3, 4).expect("ok");
        assert_eq!(m.idx(1, 2).expect("ok"), 6);
        assert!(m.idx(3, 0).is_err());
    }

    #[test]
    fn mesh_2d_boundary() {
        let m = Mesh2d::uniform(0.0, 1.0, 0.0, 1.0, 3, 3).expect("ok");
        assert!(m.is_boundary(0, 1));
        assert!(m.is_boundary(2, 1));
        assert!(m.is_boundary(1, 0));
        assert!(m.is_boundary(1, 2));
        assert!(!m.is_boundary(1, 1));
    }
}
