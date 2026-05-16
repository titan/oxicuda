//! Wasserstein-2 distance.
//!
//! 1D closed-form: `W_2² = ∫ (F_a^{-1}(u) − F_b^{-1}(u))² du`. We compute
//! this via a quantile-merging sweep over sorted weighted samples. In higher
//! dimensions we set the cost to `½ ‖x − y‖²` and use the network-simplex,
//! taking `√(2 · cost)` as the Wasserstein-2 distance.

use crate::error::{OtError, OtResult};
use crate::exact::network_simplex::{NsConfig, network_simplex};

/// Validate balanced 1D inputs.
fn validate_1d(x: &[f32], y: &[f32], a: &[f32], b: &[f32]) -> OtResult<()> {
    if x.is_empty() || y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if x.len() != a.len() {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: a.len(),
        });
    }
    if y.len() != b.len() {
        return Err(OtError::IncompatibleLength {
            a: y.len(),
            b: b.len(),
        });
    }
    let mut sa = 0.0_f32;
    for &ai in a {
        if ai < 0.0 || !ai.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sa += ai;
    }
    let mut sb = 0.0_f32;
    for &bj in b {
        if bj < 0.0 || !bj.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        sb += bj;
    }
    if (sa - sb).abs() > 1e-4 {
        return Err(OtError::MassImbalance {
            sum_a: sa,
            sum_b: sb,
        });
    }
    Ok(())
}

/// Sort samples by position and return the sorted (positions, weights).
fn sort_samples(s: &[f32], w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap_or(std::cmp::Ordering::Equal));
    let pos: Vec<f32> = idx.iter().map(|&i| s[i]).collect();
    let wts: Vec<f32> = idx.iter().map(|&i| w[i]).collect();
    (pos, wts)
}

/// 1D Wasserstein-2 between weighted samples.
///
/// Runs a quantile sweep: walk the sorted samples of both distributions,
/// at each step `u ∈ [u_lo, u_hi]` accumulate `(F_a^{-1}(u) − F_b^{-1}(u))² · Δu`.
pub fn w2_1d(x: &[f32], y: &[f32], a: &[f32], b: &[f32]) -> OtResult<f32> {
    validate_1d(x, y, a, b)?;
    let (sx, wx) = sort_samples(x, a);
    let (sy, wy) = sort_samples(y, b);

    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut cum_x = 0.0_f32;
    let mut cum_y = 0.0_f32;
    let mut total = 0.0_f32;
    while i < sx.len() && j < sy.len() {
        let nx = cum_x + wx[i];
        let ny = cum_y + wy[j];
        let upper = nx.min(ny);
        let segment = upper - cum_x.max(cum_y);
        if segment > 0.0 {
            let diff = sx[i] - sy[j];
            total += segment * diff * diff;
        }
        if nx <= ny {
            cum_x = nx;
            i += 1;
        } else {
            cum_y = ny;
            j += 1;
        }
    }
    Ok(total.sqrt())
}

/// Validate inputs for higher-dimensional W2.
fn validate_md(
    samples_x: &[f32],
    samples_y: &[f32],
    a: &[f32],
    b: &[f32],
    dim: usize,
) -> OtResult<(usize, usize)> {
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if samples_x.is_empty() || samples_y.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if !samples_x.len().is_multiple_of(dim) || !samples_y.len().is_multiple_of(dim) {
        return Err(OtError::IncompatibleLength {
            a: samples_x.len(),
            b: samples_y.len(),
        });
    }
    let nx = samples_x.len() / dim;
    let ny = samples_y.len() / dim;
    if a.len() != nx || b.len() != ny {
        return Err(OtError::IncompatibleLength {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok((nx, ny))
}

/// Multi-dimensional W2 via the network-simplex with `½ ‖·‖²` cost.
pub fn w2(samples_x: &[f32], samples_y: &[f32], a: &[f32], b: &[f32], dim: usize) -> OtResult<f32> {
    let (nx, ny) = validate_md(samples_x, samples_y, a, b, dim)?;
    let mut c = vec![0.0_f32; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            let mut sq = 0.0_f32;
            for d in 0..dim {
                let diff = samples_x[i * dim + d] - samples_y[j * dim + d];
                sq += diff * diff;
            }
            c[i * ny + j] = 0.5 * sq;
        }
    }
    let res = network_simplex(&c, a, b, nx, ny, &NsConfig::default())?;
    Ok((2.0 * res.cost).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diracs_distance_equals_euclidean() {
        let x = vec![0.0_f32, 0.0];
        let y = vec![3.0_f32, 4.0];
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        let d = w2(&x, &y, &a, &b, 2).expect("ok");
        assert!((d - 5.0).abs() < 1e-3, "d={} expected 5", d);
    }

    #[test]
    fn zero_on_equal_distributions() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let a = vec![1.0_f32 / 3.0; 3];
        let d = w2_1d(&x, &x, &a, &a).expect("ok");
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn translation_scales_linearly_1d() {
        let x = vec![0.0_f32, 1.0];
        let a = vec![0.5_f32, 0.5];
        let t = 2.0_f32;
        let y: Vec<f32> = x.iter().map(|v| v + t).collect();
        let d = w2_1d(&x, &y, &a, &a).expect("ok");
        // Both diracs translated by t → W2 = t.
        assert!((d - t).abs() < 1e-3, "d={} expected {}", d, t);
    }

    #[test]
    fn empty_rejected_1d() {
        let res = w2_1d(&[], &[], &[], &[]);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }
}
