//! 2D TV denoising via Chambolle's projection algorithm (Chambolle 2004).
//!
//! Anisotropic TV: `TV(X) = Σ |X_{i+1,j} − X_{i,j}| + |X_{i,j+1} − X_{i,j}|`.
//! Isotropic TV: `TV(X) = Σ √((X_{i+1,j}−X_{i,j})² + (X_{i,j+1}−X_{i,j})²)`.

use crate::error::{CsError, CsResult};

/// Whether the 2D TV penalty is isotropic (√sum of squares of gradients) or anisotropic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvVariant {
    Isotropic,
    Anisotropic,
}

/// 2D TV denoising of an image `y` (row-major `h × w`) with regularisation `lambda`.
pub fn tv_2d_chambolle(
    y: &[f64],
    h: usize,
    w: usize,
    lambda: f64,
    variant: TvVariant,
    max_iter: usize,
    tol: f64,
) -> CsResult<Vec<f64>> {
    if y.len() != h * w {
        return Err(CsError::ShapeMismatch {
            expected: vec![h, w],
            got: vec![y.len()],
        });
    }
    if lambda < 0.0 {
        return Err(CsError::InvalidParameter("lambda must be ≥ 0".into()));
    }
    if lambda == 0.0 {
        return Ok(y.to_vec());
    }
    let mut p_x = vec![0.0_f64; h * w];
    let mut p_y = vec![0.0_f64; h * w];
    let tau = 0.124_f64; // 1/(2*||D||) ≤ 1/8 conservative
    let mut x = y.to_vec();
    for _ in 0..max_iter {
        // x = y - div(p)
        // div(p)_{i,j} = (p_x_{i,j} - p_x_{i-1,j}) + (p_y_{i,j} - p_y_{i,j-1})
        let mut div = vec![0.0_f64; h * w];
        for i in 0..h {
            for j in 0..w {
                let mut d = p_x[i * w + j];
                if i > 0 {
                    d -= p_x[(i - 1) * w + j];
                }
                d += p_y[i * w + j];
                if j > 0 {
                    d -= p_y[i * w + (j - 1)];
                }
                div[i * w + j] = d;
            }
        }
        for k in 0..(h * w) {
            x[k] = y[k] - div[k];
        }
        // Compute gradient.
        let mut max_change = 0.0_f64;
        for i in 0..h {
            for j in 0..w {
                let idx = i * w + j;
                let gx = if i + 1 < h {
                    x[(i + 1) * w + j] - x[idx]
                } else {
                    0.0
                };
                let gy = if j + 1 < w {
                    x[i * w + (j + 1)] - x[idx]
                } else {
                    0.0
                };
                let mut p_x_new = p_x[idx] + tau * gx;
                let mut p_y_new = p_y[idx] + tau * gy;
                match variant {
                    TvVariant::Anisotropic => {
                        p_x_new = p_x_new.clamp(-lambda, lambda);
                        p_y_new = p_y_new.clamp(-lambda, lambda);
                    }
                    TvVariant::Isotropic => {
                        let nrm = (p_x_new * p_x_new + p_y_new * p_y_new).sqrt();
                        if nrm > lambda {
                            let scale = lambda / nrm;
                            p_x_new *= scale;
                            p_y_new *= scale;
                        }
                    }
                }
                let dp = (p_x_new - p_x[idx]).abs().max((p_y_new - p_y[idx]).abs());
                if dp > max_change {
                    max_change = dp;
                }
                p_x[idx] = p_x_new;
                p_y[idx] = p_y_new;
            }
        }
        if max_change < tol {
            break;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tv_2d_no_noise() {
        let y = vec![1.0; 16];
        let x = tv_2d_chambolle(&y, 4, 4, 0.5, TvVariant::Anisotropic, 200, 1.0e-9).expect("ok");
        for &v in &x {
            assert!((v - 1.0).abs() < 1.0e-3);
        }
    }
}
