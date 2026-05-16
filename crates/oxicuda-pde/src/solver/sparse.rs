//! Compressed-sparse-row (CSR) sparse matrix utilities.

use crate::error::{PdeError, PdeResult};

/// Compressed Sparse Row matrix.
#[derive(Debug, Clone)]
pub struct SparseCsr {
    pub n_rows: usize,
    pub n_cols: usize,
    pub row_ptr: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
}

impl SparseCsr {
    /// Construct from raw arrays with sanity validation.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<f64>,
    ) -> PdeResult<Self> {
        if row_ptr.len() != n_rows + 1 {
            return Err(PdeError::DimensionMismatch {
                a: row_ptr.len(),
                b: n_rows + 1,
            });
        }
        if cols.len() != vals.len() {
            return Err(PdeError::DimensionMismatch {
                a: cols.len(),
                b: vals.len(),
            });
        }
        if let Some(&last) = row_ptr.last()
            && last != cols.len()
        {
            return Err(PdeError::InvalidParameter {
                name: "row_ptr".into(),
                reason: format!("last entry {} != cols.len() {}", last, cols.len()),
            });
        }
        for &c in &cols {
            if c >= n_cols {
                return Err(PdeError::IndexOutOfBounds {
                    index: c,
                    len: n_cols,
                });
            }
        }
        Ok(Self {
            n_rows,
            n_cols,
            row_ptr,
            cols,
            vals,
        })
    }

    /// Matrix-vector product `y = A * x`.
    pub fn matvec(&self, x: &[f64]) -> PdeResult<Vec<f64>> {
        if x.len() != self.n_cols {
            return Err(PdeError::DimensionMismatch {
                a: x.len(),
                b: self.n_cols,
            });
        }
        let mut y = vec![0.0; self.n_rows];
        for (i, yi) in y.iter_mut().enumerate().take(self.n_rows) {
            let row_lo = self.row_ptr[i];
            let row_hi = self.row_ptr[i + 1];
            let mut s = 0.0;
            for k in row_lo..row_hi {
                s += self.vals[k] * x[self.cols[k]];
            }
            *yi = s;
        }
        Ok(y)
    }

    /// Diagonal as a vector.
    pub fn diagonal(&self) -> PdeResult<Vec<f64>> {
        let n = self.n_rows.min(self.n_cols);
        let mut d = vec![0.0; n];
        for (i, di) in d.iter_mut().enumerate().take(n) {
            let row_lo = self.row_ptr[i];
            let row_hi = self.row_ptr[i + 1];
            for k in row_lo..row_hi {
                if self.cols[k] == i {
                    *di = self.vals[k];
                    break;
                }
            }
        }
        Ok(d)
    }

    /// Number of stored non-zeros.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }
}

/// Inner product of two equal-length vectors.
pub fn dot(a: &[f64], b: &[f64]) -> PdeResult<f64> {
    if a.len() != b.len() {
        return Err(PdeError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Euclidean norm of a vector.
pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_matvec_identity() {
        let a =
            SparseCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).expect("ok");
        let x = vec![1.0, 2.0, 3.0];
        let y = a.matvec(&x).expect("ok");
        assert_eq!(y, x);
    }

    #[test]
    fn csr_matvec_tridiag() {
        // [[2,-1,0],[-1,2,-1],[0,-1,2]] * [1,1,1] = [1,0,1]
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .expect("ok");
        let y = a.matvec(&[1.0, 1.0, 1.0]).expect("ok");
        assert!((y[0] - 1.0).abs() < 1.0e-12);
        assert!((y[1] - 0.0).abs() < 1.0e-12);
        assert!((y[2] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn csr_diagonal_extract() {
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![3.0, 1.0, 4.0, 2.0, 5.0, 6.0],
        )
        .expect("ok");
        let d = a.diagonal().expect("ok");
        assert!((d[0] - 3.0).abs() < 1.0e-12);
        assert!((d[1] - 4.0).abs() < 1.0e-12);
        assert!((d[2] - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn dot_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b).expect("ok") - 32.0).abs() < 1.0e-12);
    }

    #[test]
    fn norm2_simple() {
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_row_ptr_length() {
        let r = SparseCsr::new(3, 3, vec![0, 1], vec![0], vec![1.0]);
        assert!(r.is_err());
    }
}
