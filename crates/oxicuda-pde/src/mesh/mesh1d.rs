//! 1D uniform/non-uniform meshes.

use crate::error::{PdeError, PdeResult};

/// 1D mesh over `[x0, x1]` with `n` nodes (so `n-1` intervals).
#[derive(Debug, Clone)]
pub struct Mesh1d {
    pub x0: f64,
    pub x1: f64,
    pub n: usize,
    pub nodes: Vec<f64>,
}

impl Mesh1d {
    /// Build a uniform 1D mesh with `n` nodes on `[x0, x1]`.
    pub fn uniform(x0: f64, x1: f64, n: usize) -> PdeResult<Self> {
        if n < 2 {
            return Err(PdeError::InvalidGrid(format!(
                "uniform mesh requires n >= 2, got {n}"
            )));
        }
        if x1 <= x0 {
            return Err(PdeError::InvalidGrid(format!(
                "uniform mesh requires x1 > x0, got x0={x0}, x1={x1}"
            )));
        }
        let h = (x1 - x0) / (n - 1) as f64;
        let nodes = (0..n).map(|i| x0 + h * i as f64).collect();
        Ok(Self { x0, x1, n, nodes })
    }

    /// Build a stretched (Chebyshev-Lobatto) mesh: `x_j = cos(j*pi/(n-1))` mapped to `[x0,x1]`.
    pub fn chebyshev_lobatto(x0: f64, x1: f64, n: usize) -> PdeResult<Self> {
        if n < 2 {
            return Err(PdeError::InvalidGrid(format!(
                "chebyshev mesh requires n >= 2, got {n}"
            )));
        }
        let mid = 0.5 * (x0 + x1);
        let half = 0.5 * (x1 - x0);
        // Cluster towards both endpoints. Reverse so nodes[0] = x0.
        let nodes: Vec<f64> = (0..n)
            .map(|j| {
                let theta = std::f64::consts::PI * j as f64 / (n - 1) as f64;
                // cos goes from +1 (j=0) to -1 (j=n-1); map (-cos) so j=0 -> x0
                mid - half * theta.cos()
            })
            .collect();
        Ok(Self { x0, x1, n, nodes })
    }

    /// Uniform spacing `h` (only valid for uniform meshes).
    pub fn h(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.x1 - self.x0) / (self.n - 1) as f64
        }
    }

    /// Local spacing `h_i = x_{i+1} - x_i` for `i in 0..n-1`.
    pub fn local_h(&self, i: usize) -> PdeResult<f64> {
        if i + 1 >= self.n {
            return Err(PdeError::IndexOutOfBounds {
                index: i + 1,
                len: self.n,
            });
        }
        Ok(self.nodes[i + 1] - self.nodes[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_mesh_endpoints() {
        let m = Mesh1d::uniform(0.0, 1.0, 5).expect("ok");
        assert_eq!(m.nodes[0], 0.0);
        assert!((m.nodes[4] - 1.0).abs() < 1.0e-12);
        assert_eq!(m.n, 5);
    }

    #[test]
    fn uniform_mesh_spacing() {
        let m = Mesh1d::uniform(0.0, 2.0, 5).expect("ok");
        assert!((m.h() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_uniform_too_few() {
        assert!(Mesh1d::uniform(0.0, 1.0, 1).is_err());
    }

    #[test]
    fn chebyshev_endpoints() {
        let m = Mesh1d::chebyshev_lobatto(-1.0, 1.0, 9).expect("ok");
        assert!((m.nodes[0] + 1.0).abs() < 1.0e-12);
        assert!((m.nodes[8] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn local_h_correct() {
        let m = Mesh1d::uniform(0.0, 1.0, 5).expect("ok");
        let h = m.local_h(2).expect("ok");
        assert!((h - 0.25).abs() < 1.0e-12);
    }
}
