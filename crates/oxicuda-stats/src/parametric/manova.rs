//! MANOVA: Wilks' lambda, Pillai's trace, Hotelling-Lawley trace.

use crate::distributions::f_dist::FDist;
use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{matrix_inverse_lu, matrix_mul, matrix_transpose};

/// Result of a MANOVA computation.
#[derive(Debug, Clone)]
pub struct ManovaResult {
    pub wilks_lambda: f64,
    pub pillai_trace: f64,
    pub hotelling_lawley: f64,
    pub f_approx: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
}

/// Compute one-way MANOVA via Wilks' lambda with Bartlett-Lawley F approximation.
///
/// `groups[k][i][j]` = observation j of subject i in group k. `p` = number of dependent variables.
pub fn manova_wilks(groups: &[Vec<Vec<f64>>], p: usize) -> StatsResult<ManovaResult> {
    let g = groups.len();
    if g < 2 {
        return Err(StatsError::InsufficientSampleSize { got: g, need: 2 });
    }
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "p".into(),
            reason: "must be > 0".into(),
        });
    }
    // Validate dimensions
    let mut n_total = 0usize;
    for grp in groups {
        if grp.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        for obs in grp {
            if obs.len() != p {
                return Err(StatsError::ShapeMismatch {
                    expected: vec![p],
                    got: vec![obs.len()],
                });
            }
        }
        n_total += grp.len();
    }
    // Compute grand mean
    let mut grand = vec![0.0; p];
    for grp in groups {
        for obs in grp {
            for j in 0..p {
                grand[j] += obs[j];
            }
        }
    }
    for v in &mut grand {
        *v /= n_total as f64;
    }
    // Compute group means
    let mut group_means = Vec::with_capacity(g);
    for grp in groups {
        let mut m = vec![0.0; p];
        for obs in grp {
            for j in 0..p {
                m[j] += obs[j];
            }
        }
        for v in &mut m {
            *v /= grp.len() as f64;
        }
        group_means.push(m);
    }
    // H = sum_k n_k (mean_k - grand) (mean_k - grand)^T  (p x p matrix)
    let mut h = vec![0.0; p * p];
    for (k, grp) in groups.iter().enumerate() {
        let nk = grp.len() as f64;
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] +=
                    nk * (group_means[k][i] - grand[i]) * (group_means[k][j] - grand[j]);
            }
        }
    }
    // E = sum_k sum_i (x_ki - mean_k) (x_ki - mean_k)^T  (p x p)
    let mut e = vec![0.0; p * p];
    for (k, grp) in groups.iter().enumerate() {
        for obs in grp {
            for i in 0..p {
                for j in 0..p {
                    e[i * p + j] += (obs[i] - group_means[k][i]) * (obs[j] - group_means[k][j]);
                }
            }
        }
    }
    // Wilks' lambda = det(E) / det(E + H)
    // Pillai = trace(H * (H + E)^{-1})
    // Hotelling-Lawley = trace(H * E^{-1})
    let mut eh = vec![0.0; p * p];
    for i in 0..p * p {
        eh[i] = e[i] + h[i];
    }
    let det_e = det_via_lu(&e, p)?;
    let det_eh = det_via_lu(&eh, p)?;
    if det_eh.abs() < 1e-300 {
        return Err(StatsError::SingularMatrix("E+H".into()));
    }
    let wilks_lambda = det_e / det_eh;
    let eh_inv = matrix_inverse_lu(&eh, p)?;
    let h_eh_inv = matrix_mul(&h, &eh_inv, p, p, p)?;
    let pillai_trace: f64 = (0..p).map(|i| h_eh_inv[i * p + i]).sum();
    let e_inv = matrix_inverse_lu(&e, p)?;
    let h_e_inv = matrix_mul(&h, &e_inv, p, p, p)?;
    let hotelling: f64 = (0..p).map(|i| h_e_inv[i * p + i]).sum();
    // F approx for Wilks (Rao):
    let q = (g - 1) as f64;
    let pp = p as f64;
    let n = n_total as f64;
    let s_val = if pp * pp + q * q == 5.0 {
        1.0
    } else {
        ((pp * pp * q * q - 4.0) / (pp * pp + q * q - 5.0))
            .max(1.0)
            .sqrt()
    };
    let m_val = n - 1.0 - (pp + q) / 2.0;
    let df1 = pp * q;
    let df2 = m_val * s_val - pp * q / 2.0 + 1.0;
    let lambda_root = wilks_lambda.powf(1.0 / s_val);
    let f = (1.0 - lambda_root) / lambda_root * df2 / df1;
    let p_value = if df1 > 0.0 && df2 > 0.0 && f.is_finite() && f > 0.0 {
        let fd = FDist::new(df1, df2)?;
        1.0 - fd.cdf(f)?
    } else {
        1.0
    };
    // Avoid useless warning on transpose
    let _ = matrix_transpose(&h, p, p);
    Ok(ManovaResult {
        wilks_lambda,
        pillai_trace,
        hotelling_lawley: hotelling,
        f_approx: f,
        df1,
        df2,
        p_value,
    })
}

fn det_via_lu(mat: &[f64], n: usize) -> StatsResult<f64> {
    let mut a = mat.to_vec();
    let mut det = 1.0;
    let mut sign = 1.0;
    for k in 0..n {
        let mut pivot_row = k;
        let mut max_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                pivot_row = i;
            }
        }
        if max_val < 1e-300 {
            return Ok(0.0);
        }
        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }
        let pivot = a[k * n + k];
        det *= pivot;
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }
    Ok(det * sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manova_two_groups_runs() {
        let g1 = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![1.2, 1.9]];
        let g2 = vec![vec![2.0, 3.0], vec![2.2, 3.1], vec![2.1, 2.9]];
        let r = manova_wilks(&[g1, g2], 2).expect("ok");
        assert!(r.wilks_lambda > 0.0 && r.wilks_lambda < 1.0);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn manova_rejects_one_group() {
        let g1 = vec![vec![1.0, 2.0]];
        assert!(manova_wilks(&[g1], 2).is_err());
    }
}
