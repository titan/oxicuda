//! Chambolle-Pock primal-dual algorithm.
//!
//! Solves saddle-point form: `min_x max_y  <Kx, y> + g(x) − f*(y)`,
//! equivalent to `min_x  f(Kx) + g(x)`.
//!
//! Updates (with `K` linear operator, `K^T` adjoint):
//!   y_{k+1} = prox_{σ f*}(y_k + σ K x̄_k)
//!   x_{k+1} = prox_{τ g} (x_k − τ K^T y_{k+1})
//!   x̄_{k+1} = x_{k+1} + θ (x_{k+1} − x_k)
//!
//! Convergence requires `τ σ ||K||² < 1`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Chambolle-Pock primal-dual algorithm.
///
/// - `k_op(x)` returns `K x`.
/// - `kt_op(y)` returns `K^T y`.
/// - `prox_f_star(y, sigma)` returns `prox_{σ f*}(y)`.
/// - `prox_g(x, tau)` returns `prox_{τ g}(x)`.
#[allow(clippy::too_many_arguments)]
pub fn chambolle_pock<K, KT, Pf, Pg>(
    x0: &[f64],
    y0: &[f64],
    k_op: K,
    kt_op: KT,
    prox_f_star: Pf,
    prox_g: Pg,
    tau: f64,
    sigma: f64,
    theta: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<(Vec<f64>, Vec<f64>)>
where
    K: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    KT: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    Pf: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    Pg: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if tau <= 0.0 || sigma <= 0.0 || !(0.0..=1.0).contains(&theta) {
        return Err(CvxError::InvalidParameter(format!(
            "Chambolle-Pock requires tau>0, sigma>0, theta∈[0,1]; got tau={tau}, sigma={sigma}, theta={theta}"
        )));
    }
    if x0.is_empty() || y0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let mut x = x0.to_vec();
    let mut y = y0.to_vec();
    let mut xbar = x0.to_vec();
    for it in 0..max_iter {
        // Dual update.
        let kx = k_op(&xbar)?;
        if kx.len() != y.len() {
            return Err(CvxError::DimensionMismatch {
                a: kx.len(),
                b: y.len(),
            });
        }
        let y_arg: Vec<f64> = y
            .iter()
            .zip(kx.iter())
            .map(|(yi, ki)| yi + sigma * ki)
            .collect();
        let y_new = prox_f_star(&y_arg, sigma)?;
        if y_new.len() != y.len() {
            return Err(CvxError::DimensionMismatch {
                a: y_new.len(),
                b: y.len(),
            });
        }
        // Primal update.
        let kt_y = kt_op(&y_new)?;
        if kt_y.len() != x.len() {
            return Err(CvxError::DimensionMismatch {
                a: kt_y.len(),
                b: x.len(),
            });
        }
        let x_arg: Vec<f64> = x
            .iter()
            .zip(kt_y.iter())
            .map(|(xi, ki)| xi - tau * ki)
            .collect();
        let x_new = prox_g(&x_arg, tau)?;
        if x_new.len() != x.len() {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: x.len(),
            });
        }
        // Extrapolation.
        let xbar_new: Vec<f64> = x_new
            .iter()
            .zip(x.iter())
            .map(|(xn, xo)| xn + theta * (xn - xo))
            .collect();
        // Stop test.
        let dx: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let dy: Vec<f64> = y_new.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&dx) + norm2(&dy);
        x = x_new;
        y = y_new;
        xbar = xbar_new;
        if d_nrm < tol {
            return Ok((x, y));
        }
        let _ = it;
    }
    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l2::prox_l2;

    #[test]
    fn cp_simple_separable() {
        // min ½||x||² + ½||y||²  with K = I → Kx = x.
        // f(z) = ½||z||² so f* = ½||y||², prox_{σ f*}(y) = y/(1+σ).
        // g(x) = 0 so prox_g(x, τ) = x.
        let k = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(x.to_vec()) };
        let kt = |y: &[f64]| -> CvxResult<Vec<f64>> { Ok(y.to_vec()) };
        let pf_star = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l2(y, s) };
        let pg = |x: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(x.to_vec()) };
        let (x, _y) = chambolle_pock(
            &[1.0, 2.0],
            &[0.0, 0.0],
            &k,
            &kt,
            &pf_star,
            &pg,
            0.5,
            0.5,
            1.0,
            500,
            1.0e-9,
        )
        .expect("ok");
        // Saddle of <x, y> - ½||y||² is x=0 (then y=0).
        for &xi in &x {
            assert!(xi.abs() < 1.0e-3);
        }
    }
}
