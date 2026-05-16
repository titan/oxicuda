//! Projection onto the L1 ball `{x : ||x||_1 ≤ r}`.

use crate::error::{CvxError, CvxResult};

/// Project `v` onto `{x : ||x||_1 ≤ r}`.  If already inside, returns a copy.
/// Otherwise applies sign-and-simplex trick: project |v| onto simplex of radius r, recombine sign.
pub fn project_l1_ball(v: &[f64], r: f64) -> CvxResult<Vec<f64>> {
    if v.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !r.is_finite() || r < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "l1 ball radius must be ≥ 0, got {r}"
        )));
    }
    let s_norm: f64 = v.iter().map(|x| x.abs()).sum();
    if s_norm <= r {
        return Ok(v.to_vec());
    }
    // Degenerate radius 0 → projection is origin.
    if r == 0.0 {
        return Ok(vec![0.0_f64; v.len()]);
    }
    // Sort |v| descending.
    let n = v.len();
    let mut u: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum_sum = 0.0_f64;
    let mut tau = 0.0_f64;
    let mut found = false;
    for (k, &uk) in u.iter().enumerate().take(n) {
        cum_sum += uk;
        let candidate = (cum_sum - r) / (k as f64 + 1.0);
        if uk - candidate > 0.0 {
            tau = candidate;
            found = true;
        } else {
            break;
        }
    }
    if !found {
        return Err(CvxError::NumericalInstability(
            "l1 ball projection threshold search failed".into(),
        ));
    }
    Ok(v.iter()
        .map(|x| {
            let mag = (x.abs() - tau).max(0.0);
            if *x >= 0.0 { mag } else { -mag }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_inside_passthrough() {
        let v = vec![0.3, -0.2, 0.1];
        let p = project_l1_ball(&v, 1.0).expect("ok");
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!((pi - vi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn l1_clips_to_r() {
        let v = vec![1.0, 1.0, -1.0];
        let p = project_l1_ball(&v, 1.0).expect("ok");
        let s: f64 = p.iter().map(|x| x.abs()).sum();
        assert!((s - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn l1_preserves_signs() {
        let v = vec![3.0, -2.0, 1.0];
        let p = project_l1_ball(&v, 2.0).expect("ok");
        assert!(p[0] >= 0.0);
        assert!(p[1] <= 0.0);
        assert!(p[2] >= 0.0);
    }
}
