//! TT-tensor data structure.
//!
//! A TT-tensor stores a `d`-dimensional array `A[i_0, ..., i_{d-1}]` of total size
//! `n_0 * n_1 * ... * n_{d-1}` as a chain of `d` rank-3 cores:
//! `A[i_0, ..., i_{d-1}] = G_0[1, i_0, :] @ G_1[:, i_1, :] @ ... @ G_{d-1}[:, i_{d-1}, 1]`.
//! Each core `G_k` has shape `(r_k, n_k, r_{k+1})` with `r_0 = r_d = 1`.

use crate::{TnError, TnResult};

/// A single TT core of shape `(r_l, n, r_r)` row-major.
#[derive(Debug, Clone)]
pub struct TtCore {
    pub r_l: usize,
    pub n: usize,
    pub r_r: usize,
    pub data: Vec<f64>,
}

impl TtCore {
    pub fn new(r_l: usize, n: usize, r_r: usize, data: Vec<f64>) -> TnResult<Self> {
        if r_l == 0 || n == 0 || r_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        if data.len() != r_l * n * r_r {
            return Err(TnError::ShapeMismatch {
                expected: vec![r_l, n, r_r],
                got: vec![data.len()],
            });
        }
        Ok(Self { r_l, n, r_r, data })
    }

    pub fn get(&self, l: usize, i: usize, r: usize) -> TnResult<f64> {
        if l >= self.r_l || i >= self.n || r >= self.r_r {
            return Err(TnError::IndexOutOfBounds {
                index: l,
                len: self.r_l,
            });
        }
        Ok(self.data[(l * self.n + i) * self.r_r + r])
    }
}

/// A TT-tensor: a chain of cores.
#[derive(Debug, Clone)]
pub struct TtTensor {
    pub cores: Vec<TtCore>,
}

impl TtTensor {
    pub fn new(cores: Vec<TtCore>) -> TnResult<Self> {
        if cores.is_empty() {
            return Err(TnError::EmptyInput);
        }
        let last_rr = cores.last().ok_or(TnError::EmptyInput)?.r_r;
        if cores[0].r_l != 1 || last_rr != 1 {
            return Err(TnError::InvalidBondDimension(0));
        }
        for w in cores.windows(2) {
            if w[0].r_r != w[1].r_l {
                return Err(TnError::DimensionMismatch {
                    a: w[0].r_r,
                    b: w[1].r_l,
                });
            }
        }
        Ok(Self { cores })
    }

    /// Reconstruct the full tensor by contracting the cores along the bond dimensions.
    /// Result is a flat `Vec<f64>` of length `prod(n_k)` in C-order
    /// (slowest index = `i_0`, fastest = `i_{d-1}`).
    pub fn reconstruct(&self) -> TnResult<Vec<f64>> {
        let dims: Vec<usize> = self.cores.iter().map(|c| c.n).collect();
        let total: usize = dims.iter().product();
        if self.cores.len() == 1 {
            return Ok(self.cores[0].data.clone());
        }
        // Sequential reconstruction by reshaping to (current_size, r_k) and matmul with
        // (r_k, n_k * r_{k+1}) of the next core.
        let first = &self.cores[0];
        let mut current = vec![0.0; first.n * first.r_r];
        for i in 0..first.n {
            for r in 0..first.r_r {
                current[i * first.r_r + r] = first.data[i * first.r_r + r];
            }
        }
        let mut current_rows = first.n;
        let mut current_cols = first.r_r;
        for k in 1..self.cores.len() {
            let core = &self.cores[k];
            // matmul: (current_rows, current_cols) * (current_cols, n_k * r_{k+1})
            let nk = core.n;
            let rr = core.r_r;
            let new_cols = nk * rr;
            let mut new_mat = vec![0.0; current_rows * new_cols];
            for i in 0..current_rows {
                for j in 0..new_cols {
                    let mut acc = 0.0;
                    for c in 0..current_cols {
                        // core data stored as (r_l, n, r_r); index (c, nk_idx, rr_idx) → c*(n*r_r) + nk_idx*r_r + rr_idx
                        acc += current[i * current_cols + c] * core.data[c * nk * rr + j];
                    }
                    new_mat[i * new_cols + j] = acc;
                }
            }
            current = new_mat;
            current_rows *= nk;
            current_cols = rr;
        }
        // current_cols should be 1 at the end
        if current.len() != total {
            return Err(TnError::ShapeMismatch {
                expected: vec![total],
                got: vec![current.len()],
            });
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tt_reconstruct_constant() {
        // 1×n×1 single core: reconstructs to itself
        let core = TtCore::new(1, 3, 1, vec![1.0, 2.0, 3.0]).expect("ok");
        let tt = TtTensor::new(vec![core]).expect("ok");
        let full = tt.reconstruct().expect("ok");
        assert_eq!(full, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn tt_two_cores_outer_product() {
        // G0: 1×2×1 = [a, b], G1: 1×3×1 = [c, d, e]
        let g0 = TtCore::new(1, 2, 1, vec![2.0, 3.0]).expect("ok");
        let g1 = TtCore::new(1, 3, 1, vec![5.0, 7.0, 11.0]).expect("ok");
        let tt = TtTensor::new(vec![g0, g1]).expect("ok");
        let full = tt.reconstruct().expect("ok");
        let expect = vec![10.0, 14.0, 22.0, 15.0, 21.0, 33.0];
        assert_eq!(full, expect);
    }
}
