//! Numerical helpers shared by the FCI algorithm: f64 Pearson/partial
//! correlation, OLS residual computation via Gauss-Jordan inversion, the
//! Fisher-Z conditional-independence test, and Acklam's rational
//! approximation to the inverse standard normal CDF.

pub(super) fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

pub(super) fn pearson_corr_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean_f64(&x[..n]);
    let my = mean_f64(&y[..n]);
    let mut num = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    if sx < 1e-18 || sy < 1e-18 {
        return 0.0;
    }
    (num / (sx.sqrt() * sy.sqrt())).clamp(-1.0, 1.0)
}

/// OLS residuals of `y` on the columns of `z_mat` (row-major, `n × d`).
/// A tiny ridge is added to keep `Z^T Z` invertible at small `n`.
pub(super) fn regress_residuals_f64(z_mat: &[f64], y: &[f64], n: usize, d: usize) -> Vec<f64> {
    if d == 0 {
        return y.to_vec();
    }
    let mut xtx = vec![0.0_f64; d * d];
    let mut xty = vec![0.0_f64; d];
    for row in 0..n {
        for i in 0..d {
            for j in 0..d {
                xtx[i * d + j] += z_mat[row * d + i] * z_mat[row * d + j];
            }
            xty[i] += z_mat[row * d + i] * y[row];
        }
    }
    for i in 0..d {
        xtx[i * d + i] += 1e-8;
    }
    let inv = match gauss_jordan_inv_f64(&xtx, d) {
        Some(m) => m,
        None => return y.to_vec(),
    };
    let beta: Vec<f64> = (0..d)
        .map(|i| (0..d).map(|j| inv[i * d + j] * xty[j]).sum())
        .collect();
    let mut residuals = vec![0.0_f64; n];
    for row in 0..n {
        let pred: f64 = (0..d).map(|j| z_mat[row * d + j] * beta[j]).sum();
        residuals[row] = y[row] - pred;
    }
    residuals
}

pub(super) fn gauss_jordan_inv_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[row * 2 * n + col].abs() > m[pivot * 2 * n + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * 2 * n + col].abs() < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..(2 * n) {
                m.swap(col * 2 * n + k, pivot * 2 * n + k);
            }
        }
        let div = m[col * 2 * n + col];
        for k in 0..(2 * n) {
            m[col * 2 * n + k] /= div;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row * 2 * n + col];
            if factor.abs() < 1e-18 {
                continue;
            }
            for k in 0..(2 * n) {
                let v = m[col * 2 * n + k] * factor;
                m[row * 2 * n + k] -= v;
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

pub(super) fn partial_corr_f64(x: &[f64], y: &[f64], z: &[&Vec<f64>], n: usize) -> f64 {
    let dz = z.len();
    if dz == 0 {
        return pearson_corr_f64(x, y);
    }
    let mut z_mat = vec![0.0_f64; n * dz];
    for (col, zv) in z.iter().enumerate() {
        for row in 0..n.min(zv.len()) {
            z_mat[row * dz + col] = zv[row];
        }
    }
    let rx = regress_residuals_f64(&z_mat, x, n, dz);
    let ry = regress_residuals_f64(&z_mat, y, n, dz);
    pearson_corr_f64(&rx, &ry)
}

/// Fisher-Z conditional-independence test.
/// Returns `true` when the null of zero partial correlation is *rejected* at
/// level `alpha` (two-sided).
pub(super) fn fisher_z_dependent(r: f64, n: usize, cond_set_size: usize, alpha: f64) -> bool {
    let r_clamped = r.clamp(-0.999_999, 0.999_999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let df = (n as f64 - cond_set_size as f64 - 3.0).max(1.0);
    let stat = z.abs() * df.sqrt();
    let z_alpha = normal_quantile_two_sided(alpha);
    stat > z_alpha
}

pub(super) fn normal_quantile_two_sided(alpha: f64) -> f64 {
    inverse_normal_cdf(1.0 - alpha / 2.0)
}

/// Acklam's rational approximation to the inverse standard normal CDF,
/// canonical coefficients per Peter Acklam (max abs err ~1.15e-9 on (0,1)).
pub(super) fn inverse_normal_cdf(p: f64) -> f64 {
    let p_low = 0.02425_f64;
    let p_high = 1.0 - p_low;
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_153_56,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d_ = [
        7.784_695_709_41_e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if p < p_low {
        let q = (-2.0_f64 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1.0);
    }
    if p > p_high {
        let q = (-2.0_f64 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    let num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5];
    let den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0;
    q * num / den
}

pub(super) fn subsets_of_size(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let rest = subsets_of_size(&items[i + 1..], k - 1);
        for mut sub in rest {
            sub.insert(0, items[i]);
            result.push(sub);
        }
    }
    result
}

pub(super) fn extract_columns(data: &[f64], n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..d)
        .map(|j| (0..n).map(|i| data[i * d + j]).collect())
        .collect()
}
