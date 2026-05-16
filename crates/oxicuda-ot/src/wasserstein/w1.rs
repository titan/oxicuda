//! Wasserstein-1 distance.
//!
//! In 1D, `W_1(a, b) = ∫ |F_a(t) − F_b(t)| dt`, which is identical to EMD-1D
//! and is computed via a sort + cumulative-difference sweep. In higher
//! dimensions we build the L2 cost matrix and dispatch to the
//! network-simplex.

use crate::error::{OtError, OtResult};
use crate::exact::emd::emd_1d;
use crate::exact::network_simplex::{NsConfig, network_simplex};

/// 1D Wasserstein-1 between weighted samples (sort + cumulative difference).
pub fn w1_1d(x: &[f32], y: &[f32], a: &[f32], b: &[f32]) -> OtResult<f32> {
    emd_1d(x, y, a, b)
}

/// Validate sample-buffer shapes for the multi-dimensional W1.
fn validate_md(
    samples_x: &[f32],
    samples_y: &[f32],
    a: &[f32],
    b: &[f32],
    dim: usize,
) -> OtResult<(usize, usize)> {
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if samples_x.is_empty() || samples_y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if !samples_x.len().is_multiple_of(dim) || !samples_y.len().is_multiple_of(dim) {
        return Err(OtError::IncompatibleLength {
            a: samples_x.len(),
            b: samples_y.len(),
        });
    }
    let nx = samples_x.len() / dim;
    let ny = samples_y.len() / dim;
    if a.len() != nx {
        return Err(OtError::IncompatibleLength { a: a.len(), b: nx });
    }
    if b.len() != ny {
        return Err(OtError::IncompatibleLength { a: b.len(), b: ny });
    }
    Ok((nx, ny))
}

/// Multi-dimensional Wasserstein-1 with cost `C_ij = ‖x_i − y_j‖_2`.
pub fn w1(samples_x: &[f32], samples_y: &[f32], a: &[f32], b: &[f32], dim: usize) -> OtResult<f32> {
    let (nx, ny) = validate_md(samples_x, samples_y, a, b, dim)?;
    let mut c = vec![0.0_f32; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            let mut sq = 0.0_f32;
            for d in 0..dim {
                let diff = samples_x[i * dim + d] - samples_y[j * dim + d];
                sq += diff * diff;
            }
            c[i * ny + j] = sq.sqrt();
        }
    }
    let res = network_simplex(&c, a, b, nx, ny, &NsConfig::default())?;
    Ok(res.cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_invariance_1d() {
        let x = vec![0.0_f32];
        let y = vec![3.5_f32];
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        let d = w1_1d(&x, &y, &a, &b).expect("ok");
        assert!((d - 3.5).abs() < 1e-4);
    }

    #[test]
    fn zero_on_equal() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let d = w1_1d(&x, &x, &a, &a).expect("ok");
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn multi_dim_translation() {
        let x = vec![0.0_f32, 0.0];
        let y = vec![3.0_f32, 4.0]; // distance = 5
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        let d = w1(&x, &y, &a, &b, 2).expect("ok");
        assert!((d - 5.0).abs() < 1e-3, "d={} expected 5", d);
    }

    #[test]
    fn bad_dim_rejected() {
        let res = w1(&[0.0_f32], &[0.0_f32], &[1.0_f32], &[1.0_f32], 0);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }
}
