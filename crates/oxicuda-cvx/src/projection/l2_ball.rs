//! Projection onto the L2 ball `{x : ||x||_2 ≤ r}`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Project `v` onto `{x : ||x||_2 ≤ r}` using scaling `x = v · min(1, r/||v||_2)`.
pub fn project_l2_ball(v: &[f64], r: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !r.is_finite() || r < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "l2 ball radius must be ≥ 0, got {r}"
        )));
    }
    let nrm = norm2(v);
    if nrm <= r {
        return Ok(v.to_vec());
    }
    let s = r / nrm;
    Ok(v.iter().map(|x| s * x).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_inside() {
        let v = vec![0.5, 0.5];
        let p = project_l2_ball(&v, 1.0).expect("ok");
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - vi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn l2_clip() {
        let v = vec![3.0, 4.0];
        let p = project_l2_ball(&v, 1.0).expect("ok");
        let nrm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1.0e-10);
        assert!((p[0] - 0.6).abs() < 1.0e-10);
        assert!((p[1] - 0.8).abs() < 1.0e-10);
    }
}
