//! Projection onto the second-order cone `K = {(t, x) ∈ R × R^n : ||x||_2 ≤ t}`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Project `(t, x)` onto SOC.  Returns `(t_proj, x_proj)`.
///
/// Three cases:
/// - if `||x|| ≤ t`: identity (already inside).
/// - if `||x|| ≤ -t`: project to origin.
/// - else: `α = (t + ||x||)/2`, project to `(α, α · x/||x||)`.
pub fn project_soc(t: f64, x: &[f64]) -> CvxResult<(f64, Vec<f64>)> {
    if !t.is_finite() {
        return Err(CvxError::InvalidParameter(format!("non-finite t = {t}")));
    }
    if x.is_empty() {
        // SOC in 1D — t ≥ 0.
        return Ok((t.max(0.0), Vec::new()));
    }
    let nx = norm2(x);
    if nx <= t {
        return Ok((t, x.to_vec()));
    }
    if nx <= -t {
        return Ok((0.0, vec![0.0; x.len()]));
    }
    let alpha = 0.5 * (t + nx);
    let factor = alpha / nx;
    let new_x: Vec<f64> = x.iter().map(|v| factor * v).collect();
    Ok((alpha, new_x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soc_inside_unchanged() {
        let (t, x) = project_soc(2.0, &[1.0, 0.0]).expect("ok");
        assert!((t - 2.0).abs() < 1.0e-12);
        assert_eq!(x, vec![1.0, 0.0]);
    }

    #[test]
    fn soc_strict_dual_to_origin() {
        let (t, x) = project_soc(-2.0, &[1.0, 0.0]).expect("ok");
        assert!(t.abs() < 1.0e-12);
        assert_eq!(x, vec![0.0, 0.0]);
    }

    #[test]
    fn soc_mid_case_projects() {
        // (t, x) = (0, (3, 4)), ||x||=5, so alpha = 5/2.
        let (t, x) = project_soc(0.0, &[3.0, 4.0]).expect("ok");
        assert!((t - 2.5).abs() < 1.0e-10);
        // new x should be alpha * x / ||x|| = 2.5 * (3,4)/5 = (1.5, 2).
        assert!((x[0] - 1.5).abs() < 1.0e-10);
        assert!((x[1] - 2.0).abs() < 1.0e-10);
    }
}
