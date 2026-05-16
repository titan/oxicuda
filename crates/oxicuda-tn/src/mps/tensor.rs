//! Single-site MPS rank-3 tensor `M[a, p, b]` of shape `(D_l, d, D_r)` in row-major layout.

use crate::{TnError, TnResult};

/// A single MPS site tensor with shape `(D_l, d, D_r)`.
///
/// The storage is row-major, i.e. element `[a, p, b]` lives at index `(a*d + p)*D_r + b`.
#[derive(Debug, Clone)]
pub struct MpsTensor {
    /// Left virtual bond dimension.
    pub d_l: usize,
    /// Physical dimension.
    pub d_p: usize,
    /// Right virtual bond dimension.
    pub d_r: usize,
    /// Row-major data of length `d_l * d_p * d_r`.
    pub data: Vec<f64>,
}

impl MpsTensor {
    /// Construct an MPS site tensor from raw data. Returns an error on shape mismatch.
    pub fn new(d_l: usize, d_p: usize, d_r: usize, data: Vec<f64>) -> TnResult<Self> {
        if d_l == 0 || d_p == 0 || d_r == 0 {
            return Err(TnError::InvalidBondDimension(0));
        }
        if data.len() != d_l * d_p * d_r {
            return Err(TnError::ShapeMismatch {
                expected: vec![d_l, d_p, d_r],
                got: vec![data.len()],
            });
        }
        Ok(Self {
            d_l,
            d_p,
            d_r,
            data,
        })
    }

    /// Zero tensor of given shape.
    pub fn zeros(d_l: usize, d_p: usize, d_r: usize) -> TnResult<Self> {
        Self::new(d_l, d_p, d_r, vec![0.0; d_l * d_p * d_r])
    }

    /// Get the element at `[a, p, b]`.
    pub fn get(&self, a: usize, p: usize, b: usize) -> TnResult<f64> {
        if a >= self.d_l || p >= self.d_p || b >= self.d_r {
            return Err(TnError::IndexOutOfBounds {
                index: a * self.d_p * self.d_r + p * self.d_r + b,
                len: self.data.len(),
            });
        }
        Ok(self.data[(a * self.d_p + p) * self.d_r + b])
    }

    /// Set the element at `[a, p, b]`.
    pub fn set(&mut self, a: usize, p: usize, b: usize, v: f64) -> TnResult<()> {
        if a >= self.d_l || p >= self.d_p || b >= self.d_r {
            return Err(TnError::IndexOutOfBounds {
                index: a * self.d_p * self.d_r + p * self.d_r + b,
                len: self.data.len(),
            });
        }
        self.data[(a * self.d_p + p) * self.d_r + b] = v;
        Ok(())
    }

    /// View as `(d_l*d_p, d_r)` matrix (left-grouped reshape).
    pub fn as_left_matrix(&self) -> (usize, usize, &[f64]) {
        (self.d_l * self.d_p, self.d_r, &self.data)
    }

    /// View as `(d_l, d_p*d_r)` matrix (right-grouped reshape).
    pub fn as_right_matrix(&self) -> (usize, usize, &[f64]) {
        (self.d_l, self.d_p * self.d_r, &self.data)
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if zero-element.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shape as a 3-tuple.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.d_l, self.d_p, self.d_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mps_tensor_construction() {
        let t = MpsTensor::zeros(2, 2, 3).expect("ok");
        assert_eq!(t.shape(), (2, 2, 3));
        assert_eq!(t.data.len(), 12);
    }

    #[test]
    fn mps_tensor_set_get() {
        let mut t = MpsTensor::zeros(2, 2, 3).expect("ok");
        t.set(1, 1, 2, 7.5).expect("ok");
        assert!((t.get(1, 1, 2).expect("ok") - 7.5).abs() < 1e-15);
        assert!(t.get(0, 0, 0).expect("ok").abs() < 1e-15);
    }

    #[test]
    fn mps_tensor_shape_mismatch() {
        assert!(MpsTensor::new(2, 2, 2, vec![0.0; 7]).is_err());
    }

    #[test]
    fn mps_tensor_oob() {
        let t = MpsTensor::zeros(2, 2, 2).expect("ok");
        assert!(t.get(2, 0, 0).is_err());
        assert!(t.get(0, 2, 0).is_err());
    }
}
