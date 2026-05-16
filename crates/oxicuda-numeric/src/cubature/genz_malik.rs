//! Genz-Malik adaptive multi-dimensional cubature (1980).
//!
//! Uses a fully-symmetric basic rule of degree 7 (with embedded degree 5 estimate) on the
//! d-dimensional hyperrectangle, and subdivides along the axis with maximum estimated
//! 4th-derivative error.

use crate::error::{NumericError, NumericResult};

#[derive(Debug, Clone)]
struct Region {
    lo: Vec<f64>,
    hi: Vec<f64>,
    val: f64,
    err: f64,
    split_dim: usize,
}

fn evaluate<F>(f: &F, lo: &[f64], hi: &[f64]) -> NumericResult<(f64, f64, usize)>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    let d = lo.len();
    let center: Vec<f64> = (0..d).map(|i| 0.5 * (lo[i] + hi[i])).collect();
    let half: Vec<f64> = (0..d).map(|i| 0.5 * (hi[i] - lo[i])).collect();
    let vol: f64 = half.iter().map(|h| 2.0 * h).product();
    // weights (Genz-Malik 1980, eq. 6)
    let w1: f64 = (12_824.0 - 9120.0 * d as f64 + 400.0 * (d as f64).powi(2)) / 19_683.0;
    let w2: f64 = 980.0 / 6561.0;
    let w3: f64 = (1820.0 - 400.0 * d as f64) / 19_683.0;
    let w4: f64 = 200.0 / 19_683.0;
    let w5: f64 = 6859.0 / 19_683.0 / 2_f64.powi(d as i32);
    let w_p1: f64 = (729.0 - 950.0 * d as f64 + 50.0 * (d as f64).powi(2)) / 729.0;
    let w_p2: f64 = 245.0 / 486.0;
    let w_p3: f64 = (265.0 - 100.0 * d as f64) / 1458.0;
    let w_p4: f64 = 25.0 / 729.0;
    let lambda2 = (9.0_f64 / 70.0).sqrt();
    let lambda3 = (9.0_f64 / 10.0).sqrt();
    let lambda4 = lambda3;
    let lambda5 = (9.0_f64 / 19.0).sqrt();
    // base evaluation at the center
    let f_c = f(&center)?;
    let mut sum1 = w1 * f_c;
    let mut sum2 = w_p1 * f_c;
    let mut sum_v_axis = vec![0.0_f64; d];
    for i in 0..d {
        let mut p = center.clone();
        p[i] = center[i] + lambda2 * half[i];
        let fp = f(&p)?;
        p[i] = center[i] - lambda2 * half[i];
        let fm = f(&p)?;
        sum1 += w2 * (fp + fm);
        sum2 += w_p2 * (fp + fm);
        let mut p2 = center.clone();
        p2[i] = center[i] + lambda3 * half[i];
        let fp2 = f(&p2)?;
        p2[i] = center[i] - lambda3 * half[i];
        let fm2 = f(&p2)?;
        sum1 += w3 * (fp2 + fm2);
        sum2 += w_p3 * (fp2 + fm2);
        // capture per-axis variation (degree-5 vs degree-7 difference)
        sum_v_axis[i] = (fp + fm - 2.0 * f_c).abs();
    }
    // pairs (i, j) with lambda4
    for i in 0..d {
        for j in (i + 1)..d {
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    let mut p = center.clone();
                    p[i] += sx * lambda4 * half[i];
                    p[j] += sy * lambda4 * half[j];
                    let fp = f(&p)?;
                    sum1 += w4 * fp;
                    sum2 += w_p4 * fp;
                }
            }
        }
    }
    // full vertices with lambda5
    let total_vertices = 1_u32 << d;
    for vi in 0..total_vertices {
        let mut p = center.clone();
        for k in 0..d {
            let s = if (vi >> k) & 1 == 1 { 1.0 } else { -1.0 };
            p[k] += s * lambda5 * half[k];
        }
        let fp = f(&p)?;
        sum1 += w5 * fp;
        // degree-5 rule contributes 0 at these vertices (only sum1 receives w5 here)
    }
    let val = vol * sum1;
    let val_lower = vol * sum2;
    let err = (val - val_lower).abs();
    // pick split axis = argmax sum_v_axis
    let mut split_dim = 0;
    let mut best = sum_v_axis[0];
    for (i, &v) in sum_v_axis.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            split_dim = i;
        }
    }
    Ok((val, err, split_dim))
}

/// Adaptive Genz-Malik cubature.
pub fn genz_malik_cubature<F>(
    f: F,
    lo: &[f64],
    hi: &[f64],
    tol: f64,
    max_regions: usize,
) -> NumericResult<f64>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    if lo.len() != hi.len() {
        return Err(NumericError::DimensionMismatch {
            a: lo.len(),
            b: hi.len(),
        });
    }
    if lo.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let (val0, err0, dim0) = evaluate(&f, lo, hi)?;
    let mut regions = vec![Region {
        lo: lo.to_vec(),
        hi: hi.to_vec(),
        val: val0,
        err: err0,
        split_dim: dim0,
    }];
    let mut total_val = val0;
    let mut total_err = err0;
    for _ in 0..max_regions {
        if total_err < tol {
            return Ok(total_val);
        }
        // pick region with largest error
        let mut worst_idx = 0;
        let mut worst_err = regions[0].err;
        for (i, r) in regions.iter().enumerate().skip(1) {
            if r.err > worst_err {
                worst_err = r.err;
                worst_idx = i;
            }
        }
        let r = regions.swap_remove(worst_idx);
        let mid = 0.5 * (r.lo[r.split_dim] + r.hi[r.split_dim]);
        let mut hi1 = r.hi.clone();
        hi1[r.split_dim] = mid;
        let mut lo2 = r.lo.clone();
        lo2[r.split_dim] = mid;
        let (v1, e1, d1) = evaluate(&f, &r.lo, &hi1)?;
        let (v2, e2, d2) = evaluate(&f, &lo2, &r.hi)?;
        total_val += (v1 + v2) - r.val;
        total_err += (e1 + e2) - r.err;
        regions.push(Region {
            lo: r.lo,
            hi: hi1,
            val: v1,
            err: e1,
            split_dim: d1,
        });
        regions.push(Region {
            lo: lo2,
            hi: r.hi,
            val: v2,
            err: e2,
            split_dim: d2,
        });
    }
    Ok(total_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gm_constant_2d() {
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(1.0) };
        let v = genz_malik_cubature(f, &[0.0, 0.0], &[2.0, 3.0], 1.0e-8, 200).expect("ok");
        assert!((v - 6.0).abs() < 1.0e-8);
    }

    #[test]
    fn gm_polynomial_3d() {
        // ∫_{[0,1]³} (x + y + z) dx dy dz = 1.5
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] + x[1] + x[2]) };
        let v =
            genz_malik_cubature(f, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], 1.0e-6, 200).expect("ok");
        assert!((v - 1.5).abs() < 1.0e-4);
    }
}
