//! PEPS data structure: a 2D grid of rank-5 tensors.

use crate::handle::LcgRng;
use crate::{TnError, TnResult};

/// Single PEPS tensor with shape `(D_l, D_r, D_u, D_d, d)` in row-major order:
/// element `[l, r, u, d, p]` lives at index `(((l*D_r + r)*D_u + u)*D_d + d)*d_p + p`.
#[derive(Debug, Clone)]
pub struct PepsTensor {
    pub d_l: usize,
    pub d_r: usize,
    pub d_u: usize,
    pub d_d: usize,
    pub d_p: usize,
    pub data: Vec<f64>,
}

impl PepsTensor {
    pub fn new(
        d_l: usize,
        d_r: usize,
        d_u: usize,
        d_d: usize,
        d_p: usize,
        data: Vec<f64>,
    ) -> TnResult<Self> {
        if d_l == 0 || d_r == 0 || d_u == 0 || d_d == 0 || d_p == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        let expected = d_l * d_r * d_u * d_d * d_p;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l, d_r, d_u, d_d, d_p],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_l,
            d_r,
            d_u,
            d_d,
            d_p,
            data,
        })
    }

    pub fn zeros(d_l: usize, d_r: usize, d_u: usize, d_d: usize, d_p: usize) -> TnResult<Self> {
        let n = d_l * d_r * d_u * d_d * d_p;
        Self::new(d_l, d_r, d_u, d_d, d_p, vec![0.0; n])
    }

    pub fn shape(&self) -> (usize, usize, usize, usize, usize) {
        (self.d_l, self.d_r, self.d_u, self.d_d, self.d_p)
    }

    /// Row-major access.
    pub fn get(&self, l: usize, r: usize, u: usize, d: usize, p: usize) -> TnResult<f64> {
        if l >= self.d_l || r >= self.d_r || u >= self.d_u || d >= self.d_d || p >= self.d_p {
            return Err(TnError::IndexOutOfBounds {
                index: l,
                len: self.d_l,
            });
        }
        let idx = (((l * self.d_r + r) * self.d_u + u) * self.d_d + d) * self.d_p + p;
        Ok(self.data[idx])
    }
}

/// 2D PEPS container.
#[derive(Debug, Clone)]
pub struct Peps {
    pub rows: usize,
    pub cols: usize,
    /// Tensors indexed `[row * cols + col]`.
    pub tensors: Vec<PepsTensor>,
}

impl Peps {
    /// Build a random PEPS with bond dimension `chi` and physical dim `d`.
    ///
    /// Boundary virtual bonds equal 1 along the relevant edge.
    pub fn random(
        rows: usize,
        cols: usize,
        d: usize,
        chi: usize,
        rng: &mut LcgRng,
    ) -> TnResult<Self> {
        if rows == 0 || cols == 0 || d == 0 || chi == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut tensors = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let d_l = if c == 0 { 1 } else { chi };
                let d_r = if c + 1 == cols { 1 } else { chi };
                let d_u = if r == 0 { 1 } else { chi };
                let d_d = if r + 1 == rows { 1 } else { chi };
                let n = d_l * d_r * d_u * d_d * d;
                let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                tensors.push(PepsTensor::new(d_l, d_r, d_u, d_d, d, data)?);
            }
        }
        Ok(Self {
            rows,
            cols,
            tensors,
        })
    }

    pub fn n_sites(&self) -> usize {
        self.rows * self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_peps_shape() {
        let mut rng = LcgRng::new(7);
        let p = Peps::random(2, 3, 2, 2, &mut rng).expect("ok");
        assert_eq!(p.n_sites(), 6);
    }

    #[test]
    fn random_peps_corners() {
        let mut rng = LcgRng::new(7);
        let p = Peps::random(2, 2, 2, 3, &mut rng).expect("ok");
        let top_left = &p.tensors[0];
        // top-left: d_l=1, d_u=1
        assert_eq!(top_left.d_l, 1);
        assert_eq!(top_left.d_u, 1);
        let bot_right = &p.tensors[3];
        assert_eq!(bot_right.d_r, 1);
        assert_eq!(bot_right.d_d, 1);
    }
}
