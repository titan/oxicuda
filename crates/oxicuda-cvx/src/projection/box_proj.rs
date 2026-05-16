//! Element-wise box projection `[lo, hi]`.

use crate::error::{CvxError, CvxResult};

/// Project `v` element-wise onto `[lo, hi]` (scalar bounds).
pub fn project_box(v: &[f64], lo: f64, hi: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(CvxError::InvalidParameter(format!(
            "box bounds invalid: lo={lo}, hi={hi}"
        )));
    }
    Ok(v.iter().map(|x| x.clamp(lo, hi)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_clamps() {
        let v = vec![-2.0, 0.5, 5.0];
        let p = project_box(&v, -1.0, 1.0).expect("ok");
        assert_eq!(p, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn box_rejects_invalid_bounds() {
        assert!(project_box(&[0.0], 1.0, 0.0).is_err());
    }
}
