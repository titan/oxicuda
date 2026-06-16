//! Peaceman-Rachford splitting for `min f(x) + g(x)`.
//!
//! Peaceman-Rachford (PR) iterates the composition of the two *reflection*
//! operators `R_{γh} = 2·prox_{γh} − I`:
//!
//! ```text
//!   x_k     = prox_{γf}(z_k)
//!   u_k     = 2 x_k − z_k                 (= R_{γf} z_k)
//!   w_k     = prox_{γg}(u_k)
//!   z_{k+1} = z_k + 2 (w_k − x_k)         (= R_{γg} R_{γf} z_k)
//! ```
//!
//! At a fixed point `z*` the iterates satisfy `x* = w* = argmin f + g`.
//!
//! PR differs from Douglas-Rachford (DR), whose update averages the reflected
//! point with the current one (`z_{k+1} = ½ z_k + ½ R_{γg} R_{γf} z_k`).  The
//! un-averaged PR map `R_{γg} R_{γf}` is only firmly nonexpansive when one of
//! the operators is *strongly* monotone, so PR converges (faster than DR when it
//! does) for problems where `f` or `g` is strongly convex; otherwise it may
//! merely cycle.  Pass a strongly convex piece as `f` for guaranteed
//! convergence.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Peaceman-Rachford splitting.
///
/// `prox_f` and `prox_g` each receive `(point, gamma)` and return
/// `prox_{γ·}(point)`.
///
/// # Parameters
/// * `x0`       – starting reflected variable `z_0` (length ≥ 1).
/// * `prox_f`   – proximal operator of `f` (supply the strongly convex piece).
/// * `prox_g`   – proximal operator of `g`.
/// * `gamma`    – positive proximal step.
/// * `max_iter` – iteration cap.
/// * `tol`      – stop when `‖z_{k+1} − z_k‖ < tol`.
///
/// Returns `prox_{γf}(z)` at the final `z` (the splitting's primal estimate).
///
/// # Errors
/// * [`CvxError::InvalidParameter`] if `gamma ≤ 0` or non-finite.
/// * [`CvxError::EmptyInput`] if `x0` is empty.
/// * [`CvxError::DimensionMismatch`] if a prox returns a wrong-length vector.
pub fn peaceman_rachford<F, G>(
    x0: &[f64],
    prox_f: F,
    prox_g: G,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    F: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    G: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if gamma <= 0.0 || !gamma.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "PR gamma must be > 0, got {gamma}"
        )));
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let n = x0.len();
    let mut z = x0.to_vec();
    for _ in 0..max_iter {
        // x = prox_f(z); reflect: u = 2x − z.
        let x = prox_f(&z, gamma)?;
        if x.len() != n {
            return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
        }
        let u: Vec<f64> = x
            .iter()
            .zip(z.iter())
            .map(|(xi, zi)| 2.0 * xi - zi)
            .collect();
        // w = prox_g(u); full reflection update z_new = z + 2(w − x).
        let w = prox_g(&u, gamma)?;
        if w.len() != n {
            return Err(CvxError::DimensionMismatch { a: w.len(), b: n });
        }
        let mut z_new = vec![0.0_f64; n];
        for i in 0..n {
            z_new[i] = z[i] + 2.0 * (w[i] - x[i]);
        }
        let diff: Vec<f64> = z_new.iter().zip(z.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        z = z_new;
        if d_nrm < tol {
            return prox_f(&z, gamma);
        }
    }
    prox_f(&z, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::prox_l1;

    /// prox of `(a/2)‖x − p‖²`: argmin (γa/2)‖x−p‖² + ½‖x−v‖² = (γa·p + v)/(1+γa).
    fn prox_quad(v: &[f64], gamma: f64, a: f64, p: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(v.iter()
            .zip(p.iter())
            .map(|(vi, pi)| (gamma * a * pi + vi) / (1.0 + gamma * a))
            .collect())
    }

    #[test]
    fn pr_two_strongly_convex_quadratics() {
        // f = ½‖x − p‖², g = ½‖x − q‖² ⇒ min at (p + q)/2.
        let p = vec![4.0_f64, 0.0];
        let q = vec![0.0_f64, 2.0];
        let pf = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &p);
        let pg = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &q);
        // gamma = 0.5 keeps both reflections genuine contractions (non-degenerate).
        let x = peaceman_rachford(&[0.0, 0.0], &pf, &pg, 0.5, 500, 1e-12).expect("ok");
        assert!((x[0] - 2.0).abs() < 1e-6, "x0={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "x1={}", x[1]);
    }

    #[test]
    fn pr_lasso_scalar() {
        // f = ½(x − b)² (strongly convex), g = |x|.  min ½(x−3)² + |x| ⇒ x = 2.
        let b = vec![3.0_f64];
        let pf = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &b);
        let pg = |v: &[f64], g: f64| prox_l1(v, g);
        let x = peaceman_rachford(&[0.0], &pf, &pg, 0.5, 1000, 1e-12).expect("ok");
        assert!((x[0] - 2.0).abs() < 1e-5, "x={}", x[0]);
    }

    #[test]
    fn pr_weighted_quadratics() {
        // f = ½‖x−p‖², g = (3/2)‖x−q‖² ⇒ min at (p + 3q)/4.
        let p = vec![2.0_f64];
        let q = vec![6.0_f64];
        let pf = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &p);
        let pg = |v: &[f64], g: f64| prox_quad(v, g, 3.0, &q);
        let x = peaceman_rachford(&[0.0], &pf, &pg, 0.4, 500, 1e-12).expect("ok");
        // (2 + 18)/4 = 5.
        assert!((x[0] - 5.0).abs() < 1e-6, "x={}", x[0]);
    }

    #[test]
    fn pr_bad_gamma_errors() {
        let pf = |v: &[f64], _g: f64| Ok(v.to_vec());
        let pg = |v: &[f64], _g: f64| Ok(v.to_vec());
        assert!(matches!(
            peaceman_rachford(&[1.0], &pf, &pg, 0.0, 10, 1e-8),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn pr_empty_errors() {
        let pf = |v: &[f64], _g: f64| Ok(v.to_vec());
        let pg = |v: &[f64], _g: f64| Ok(v.to_vec());
        assert!(matches!(
            peaceman_rachford(&[], &pf, &pg, 0.5, 10, 1e-8),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn pr_dimension_mismatch_errors() {
        // prox_f returns wrong length.
        let pf = |_v: &[f64], _g: f64| Ok(vec![0.0, 0.0]);
        let pg = |v: &[f64], _g: f64| Ok(v.to_vec());
        assert!(matches!(
            peaceman_rachford(&[1.0], &pf, &pg, 0.5, 10, 1e-8),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }
}
