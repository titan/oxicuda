//! TV denoising public façade: chooses 1D or 2D variant based on input shape.

use crate::error::{CsError, CsResult};
use crate::tv::tv_1d_chambolle::tv_1d_chambolle;
use crate::tv::tv_2d_chambolle::{TvVariant, tv_2d_chambolle};

/// Spatial dimension of the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvDim {
    OneD,
    TwoD { h: usize, w: usize },
}

/// Denoise input with TV regularisation. For 2D pass `TvDim::TwoD { h, w }` and `h * w = y.len()`.
pub fn total_variation_denoise(
    y: &[f64],
    dim: TvDim,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<Vec<f64>> {
    match dim {
        TvDim::OneD => tv_1d_chambolle(y, lambda, max_iter, tol),
        TvDim::TwoD { h, w } => {
            if h * w != y.len() {
                return Err(CsError::ShapeMismatch {
                    expected: vec![h, w],
                    got: vec![y.len()],
                });
            }
            tv_2d_chambolle(y, h, w, lambda, TvVariant::Anisotropic, max_iter, tol)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_variation_dispatches() {
        let y = vec![1.0, 1.0, 2.0, 2.0];
        let x = total_variation_denoise(&y, TvDim::OneD, 0.1, 100, 1.0e-9).expect("ok");
        assert_eq!(x.len(), 4);
        let img = vec![1.0; 4];
        let x2 = total_variation_denoise(&img, TvDim::TwoD { h: 2, w: 2 }, 0.1, 100, 1.0e-9)
            .expect("ok");
        assert_eq!(x2.len(), 4);
    }
}
