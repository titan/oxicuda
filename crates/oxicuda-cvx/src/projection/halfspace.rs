//! Projection onto the halfspace `{x : a^T x ≤ b}`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::dot;

/// Project `v` onto `{x : a^T x ≤ b}`.
///
/// If `a^T v ≤ b`, returns `v` (already inside).  Otherwise:
/// `x = v - (a^T v - b)/||a||^2 · a`.
pub fn project_halfspace(v: &[f64], a: &[f64], b: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if v.len() != a.len() {
        return Err(CvxError::DimensionMismatch {
            a: v.len(),
            b: a.len(),
        });
    }
    let av = dot(a, v)?;
    if av <= b {
        return Ok(v.to_vec());
    }
    let a2 = dot(a, a)?;
    if a2 < 1.0e-300 {
        return Err(CvxError::NumericalInstability(
            "halfspace projection: ||a||^2 ≈ 0".into(),
        ));
    }
    let factor = (av - b) / a2;
    Ok(v.iter()
        .zip(a.iter())
        .map(|(vi, ai)| vi - factor * ai)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halfspace_inside_unchanged() {
        let v = vec![0.0, 0.0];
        let p = project_halfspace(&v, &[1.0, 1.0], 1.0).expect("ok");
        assert_eq!(p, v);
    }

    #[test]
    fn halfspace_outside_projects() {
        let v = vec![1.0, 1.0];
        let p = project_halfspace(&v, &[1.0, 1.0], 1.0).expect("ok");
        // a^T p should equal b.
        let av: f64 = p.iter().sum();
        assert!((av - 1.0).abs() < 1.0e-10);
        // Symmetric.
        assert!((p[0] - p[1]).abs() < 1.0e-12);
    }
}
