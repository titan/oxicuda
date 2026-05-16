//! Sliced Wasserstein — Monte-Carlo over random projection directions.
//!
//! For each random unit vector `θ ∈ S^{d−1}` we compute the 1D Wasserstein-p
//! distance between the projected samples; the sliced distance is then
//!
//! ```text
//! SW_p(μ, ν) = ( E_θ [ W_p^p(P_θ μ, P_θ ν) ] )^{1/p}
//! ```
//!
//! For equal-weight empirical samples, the 1D `W_p^p` is the L^p difference
//! between the sorted projection vectors.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for the sliced Wasserstein estimator.
#[derive(Debug, Clone)]
pub struct SlicedConfig {
    /// Number of random projections (Monte-Carlo samples).
    pub n_proj: usize,
    /// Wasserstein exponent `p`.
    pub p: u32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SlicedConfig {
    fn default() -> Self {
        Self {
            n_proj: 50,
            p: 2,
            seed: 42,
        }
    }
}

/// Validate flat sample buffers and matching sample counts.
fn validate(
    samples_x: &[f32],
    samples_y: &[f32],
    dim: usize,
    n_x: usize,
    n_y: usize,
    cfg: &SlicedConfig,
) -> OtResult<()> {
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if n_x == 0 || n_y == 0 {
        return Err(OtError::EmptyInput);
    }
    if samples_x.len() != n_x * dim || samples_y.len() != n_y * dim {
        return Err(OtError::IncompatibleLength {
            a: samples_x.len(),
            b: samples_y.len(),
        });
    }
    if cfg.n_proj == 0 {
        return Err(OtError::BadCount { got: cfg.n_proj });
    }
    if cfg.p == 0 {
        return Err(OtError::BadCount {
            got: cfg.p as usize,
        });
    }
    Ok(())
}

/// Draw a unit vector in `R^d` via Box-Muller and L2 normalisation.
fn unit_vector(rng: &mut LcgRng, dim: usize, out: &mut [f32]) {
    rng.fill_normal(out);
    let mut nrm = 0.0_f32;
    for &v in out.iter() {
        nrm += v * v;
    }
    let nrm = nrm.sqrt().max(1e-12);
    for v in out.iter_mut() {
        *v /= nrm;
    }
    // Suppress unused dim warning when out.len()==dim is known to caller.
    let _ = dim;
}

/// Sort a buffer in place (ascending) using f32-safe comparator.
fn sort_f32(v: &mut [f32]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Project all samples onto direction `theta` and write to `out`.
fn project(samples: &[f32], theta: &[f32], dim: usize, out: &mut [f32]) {
    let n = samples.len() / dim;
    for i in 0..n {
        let mut s = 0.0_f32;
        for d in 0..dim {
            s += samples[i * dim + d] * theta[d];
        }
        out[i] = s;
    }
}

/// Sliced Wasserstein-`p` estimator.
///
/// The two sample sets are assumed to have equal weights (uniform).
/// `n_x = n_y` is **not** required; the closed-form 1D `W_p^p` for
/// possibly-mismatched sample sizes is computed by linear interpolation
/// across uniform quantiles.
pub fn sliced_w(
    samples_x: &[f32],
    samples_y: &[f32],
    dim: usize,
    n_x: usize,
    n_y: usize,
    cfg: &SlicedConfig,
) -> OtResult<f32> {
    validate(samples_x, samples_y, dim, n_x, n_y, cfg)?;

    let mut theta = vec![0.0_f32; dim];
    let mut proj_x = vec![0.0_f32; n_x];
    let mut proj_y = vec![0.0_f32; n_y];

    let mut sum_pp = 0.0_f32;
    for k in 0..cfg.n_proj {
        let sub_seed = cfg
            .seed
            .wrapping_add((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = LcgRng::new(sub_seed);
        unit_vector(&mut rng, dim, &mut theta);
        project(samples_x, &theta, dim, &mut proj_x);
        project(samples_y, &theta, dim, &mut proj_y);
        sort_f32(&mut proj_x);
        sort_f32(&mut proj_y);
        sum_pp += w_pp_uniform(&proj_x, &proj_y, cfg.p);
    }
    let mean = sum_pp / cfg.n_proj as f32;
    Ok(mean.powf(1.0 / cfg.p as f32))
}

/// W_p^p between uniform-weight sorted 1D samples (possibly different sizes).
///
/// Interpolates the inverse CDFs on a common uniform-quantile grid of size
/// `lcm(n_x, n_y)` (capped to `n_x · n_y`). For the typical equal-size case
/// this reduces to a direct L^p difference of sorted vectors.
fn w_pp_uniform(sx: &[f32], sy: &[f32], p: u32) -> f32 {
    let nx = sx.len();
    let ny = sy.len();
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    if nx == ny {
        let mut s = 0.0_f32;
        for (a, b) in sx.iter().zip(sy.iter()) {
            let d = (a - b).abs();
            s += pow_u32(d, p);
        }
        return s / nx as f32;
    }
    // Generic merge via cumulative quantiles.
    let mut total = 0.0_f32;
    let inv_nx = 1.0_f32 / nx as f32;
    let inv_ny = 1.0_f32 / ny as f32;
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut cum_x = 0.0_f32;
    let mut cum_y = 0.0_f32;
    while i < nx && j < ny {
        let nxv = cum_x + inv_nx;
        let nyv = cum_y + inv_ny;
        let upper = nxv.min(nyv);
        let segment = upper - cum_x.max(cum_y);
        if segment > 0.0 {
            let d = (sx[i] - sy[j]).abs();
            total += segment * pow_u32(d, p);
        }
        if nxv <= nyv {
            cum_x = nxv;
            i += 1;
        } else {
            cum_y = nyv;
            j += 1;
        }
    }
    total
}

/// `x^p` for non-negative integer exponent.
fn pow_u32(x: f32, p: u32) -> f32 {
    let mut acc = 1.0_f32;
    for _ in 0..p {
        acc *= x;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_on_equal_samples() {
        let x = vec![0.0_f32, 1.0, 2.0, 3.0];
        let cfg = SlicedConfig {
            n_proj: 16,
            p: 2,
            seed: 1,
        };
        let d = sliced_w(&x, &x, 1, 4, 4, &cfg).expect("ok");
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn symmetry_mc() {
        let x = vec![0.0_f32, 0.0, 1.0, 0.0];
        let y = vec![0.5_f32, 1.0, 1.5, 1.0];
        let cfg = SlicedConfig {
            n_proj: 32,
            p: 2,
            seed: 7,
        };
        let dab = sliced_w(&x, &y, 2, 2, 2, &cfg).expect("ok");
        let dba = sliced_w(&y, &x, 2, 2, 2, &cfg).expect("ok");
        assert!((dab - dba).abs() < 1e-4);
    }

    #[test]
    fn finite_and_non_negative() {
        let x = vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let y = vec![1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let cfg = SlicedConfig::default();
        let d = sliced_w(&x, &y, 2, 3, 3, &cfg).expect("ok");
        assert!(d.is_finite() && d >= 0.0);
    }

    #[test]
    fn bad_dim_rejected() {
        let cfg = SlicedConfig::default();
        let res = sliced_w(&[0.0_f32], &[0.0_f32], 0, 1, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }
}
