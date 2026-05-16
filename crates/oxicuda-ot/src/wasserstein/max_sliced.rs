//! Max-sliced Wasserstein — supremum of 1D projection distances.
//!
//! ```text
//! MSW_p(μ, ν) = sup_{θ∈S^{d−1}} W_p(P_θ μ, P_θ ν)
//! ```
//!
//! We initialise `θ` as the argmax direction over `n_proj` random samples,
//! then refine it by `n_iter` finite-difference gradient ascent steps,
//! re-projecting onto the unit sphere after each update.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for the max-sliced Wasserstein estimator.
#[derive(Debug, Clone)]
pub struct MaxSlicedConfig {
    /// Random initialisation budget.
    pub n_proj: usize,
    /// Refinement iterations.
    pub n_iter: usize,
    /// Step size for finite-difference gradient ascent.
    pub lr: f32,
    /// Wasserstein exponent.
    pub p: u32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for MaxSlicedConfig {
    fn default() -> Self {
        Self {
            n_proj: 50,
            n_iter: 20,
            lr: 0.05,
            p: 2,
            seed: 42,
        }
    }
}

fn validate(
    samples_x: &[f32],
    samples_y: &[f32],
    dim: usize,
    n_x: usize,
    n_y: usize,
    cfg: &MaxSlicedConfig,
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

fn unit_vector(rng: &mut LcgRng, out: &mut [f32]) {
    rng.fill_normal(out);
    let mut nrm = 0.0_f32;
    for &v in out.iter() {
        nrm += v * v;
    }
    let nrm = nrm.sqrt().max(1e-12);
    for v in out.iter_mut() {
        *v /= nrm;
    }
}

fn project_unit(v: &mut [f32]) {
    let mut nrm = 0.0_f32;
    for &x in v.iter() {
        nrm += x * x;
    }
    let nrm = nrm.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= nrm;
    }
}

fn pow_u32(x: f32, p: u32) -> f32 {
    let mut acc = 1.0_f32;
    for _ in 0..p {
        acc *= x;
    }
    acc
}

fn sort_f32(v: &mut [f32]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

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

/// Compute `W_p^p` between equal-weight 1D samples (possibly different sizes).
fn w_pp(sx: &[f32], sy: &[f32], p: u32) -> f32 {
    let mut x = sx.to_vec();
    let mut y = sy.to_vec();
    sort_f32(&mut x);
    sort_f32(&mut y);
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    if nx == ny {
        let mut s = 0.0_f32;
        for (a, b) in x.iter().zip(y.iter()) {
            let d = (a - b).abs();
            s += pow_u32(d, p);
        }
        return s / nx as f32;
    }
    let inv_nx = 1.0_f32 / nx as f32;
    let inv_ny = 1.0_f32 / ny as f32;
    let mut total = 0.0_f32;
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
            let d = (x[i] - y[j]).abs();
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

fn objective(
    samples_x: &[f32],
    samples_y: &[f32],
    theta: &[f32],
    dim: usize,
    proj_x: &mut [f32],
    proj_y: &mut [f32],
    p: u32,
) -> f32 {
    project(samples_x, theta, dim, proj_x);
    project(samples_y, theta, dim, proj_y);
    w_pp(proj_x, proj_y, p)
}

/// Max-sliced Wasserstein-`p` with finite-difference refinement.
pub fn max_sliced_w(
    samples_x: &[f32],
    samples_y: &[f32],
    dim: usize,
    n_x: usize,
    n_y: usize,
    cfg: &MaxSlicedConfig,
) -> OtResult<f32> {
    validate(samples_x, samples_y, dim, n_x, n_y, cfg)?;
    let mut rng = LcgRng::new(cfg.seed);
    let mut theta = vec![0.0_f32; dim];
    let mut best_theta = vec![0.0_f32; dim];
    let mut proj_x = vec![0.0_f32; n_x];
    let mut proj_y = vec![0.0_f32; n_y];

    let mut best_val = f32::NEG_INFINITY;
    for _ in 0..cfg.n_proj {
        unit_vector(&mut rng, &mut theta);
        let val = objective(
            samples_x,
            samples_y,
            &theta,
            dim,
            &mut proj_x,
            &mut proj_y,
            cfg.p,
        );
        if val > best_val {
            best_val = val;
            best_theta.copy_from_slice(&theta);
        }
    }
    if !best_val.is_finite() {
        best_val = 0.0;
    }
    theta.copy_from_slice(&best_theta);

    // Finite-difference gradient ascent.
    let h = 1e-3_f32;
    let mut grad = vec![0.0_f32; dim];
    for _ in 0..cfg.n_iter {
        let f0 = objective(
            samples_x,
            samples_y,
            &theta,
            dim,
            &mut proj_x,
            &mut proj_y,
            cfg.p,
        );
        for d in 0..dim {
            theta[d] += h;
            let fp = objective(
                samples_x,
                samples_y,
                &theta,
                dim,
                &mut proj_x,
                &mut proj_y,
                cfg.p,
            );
            theta[d] -= h;
            grad[d] = (fp - f0) / h;
        }
        for d in 0..dim {
            theta[d] += cfg.lr * grad[d];
        }
        project_unit(&mut theta);
        let f1 = objective(
            samples_x,
            samples_y,
            &theta,
            dim,
            &mut proj_x,
            &mut proj_y,
            cfg.p,
        );
        if f1 > best_val {
            best_val = f1;
            best_theta.copy_from_slice(&theta);
        }
    }
    Ok(best_val.max(0.0).powf(1.0 / cfg.p as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_and_non_negative() {
        let x = vec![0.0_f32, 0.0, 1.0, 0.0];
        let y = vec![1.0_f32, 1.0, 2.0, 1.0];
        let cfg = MaxSlicedConfig {
            n_proj: 20,
            n_iter: 5,
            lr: 0.05,
            p: 2,
            seed: 11,
        };
        let d = max_sliced_w(&x, &y, 2, 2, 2, &cfg).expect("ok");
        assert!(d.is_finite() && d >= 0.0);
    }

    #[test]
    fn at_least_as_large_as_random_sliced() {
        // For the same draws, max-slice ≥ averaged sliced (loose check).
        let x = vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 0.0];
        let y = vec![1.0_f32, 1.0, 2.0, 1.0, 3.0, 0.0];
        let max_cfg = MaxSlicedConfig {
            n_proj: 32,
            n_iter: 8,
            lr: 0.05,
            p: 2,
            seed: 5,
        };
        let max_d = max_sliced_w(&x, &y, 2, 3, 3, &max_cfg).expect("ok");
        let sliced_cfg = crate::wasserstein::sliced::SlicedConfig {
            n_proj: 32,
            p: 2,
            seed: 5,
        };
        let sliced =
            crate::wasserstein::sliced::sliced_w(&x, &y, 2, 3, 3, &sliced_cfg).expect("ok");
        assert!(max_d + 1e-3 >= sliced, "max_d={} sliced={}", max_d, sliced);
    }

    #[test]
    fn bad_dim_rejected() {
        let cfg = MaxSlicedConfig::default();
        let res = max_sliced_w(&[0.0_f32], &[0.0_f32], 0, 1, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }
}
