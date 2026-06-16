//! Sliced Wasserstein distance — f64 API, Rabin (2012).
//!
//! For each random unit vector `θ ∈ S^{d−1}` we compute the 1D Wasserstein-p
//! distance between the projected samples; the sliced distance is then
//!
//! ```text
//! SW_p(μ, ν) = ( E_θ [ W_p^p(P_θ μ, P_θ ν) ] )^{1/p}
//! ```
//!
//! For equal-weight empirical samples, the 1D `W_p^p` is the L^p difference
//! between the sorted projection vectors, with quantile interpolation when the
//! two sample sets differ in size.
//!
//! Reference: Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012).
//! *Wasserstein barycenter and its application to texture mixing.*
//! Scale Space and Variational Methods in Computer Vision.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for the f64 sliced Wasserstein estimator.
#[derive(Debug, Clone)]
pub struct SlicedWassersteinConfig {
    /// Number of random projections (Monte-Carlo samples).
    pub n_projections: usize,
    /// Wasserstein exponent `p`.
    pub p: u32,
}

impl Default for SlicedWassersteinConfig {
    fn default() -> Self {
        Self {
            n_projections: 50,
            p: 2,
        }
    }
}

// ─── internal helpers ──────────────────────────────────────────────────────

/// Validate inputs and configuration; return an error on any violation.
fn validate(
    x: &[f64],
    n: usize,
    y: &[f64],
    m: usize,
    dim: usize,
    cfg: &SlicedWassersteinConfig,
) -> OtResult<()> {
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if x.len() != n * dim {
        return Err(OtError::IncompatibleLength {
            a: x.len(),
            b: n * dim,
        });
    }
    if y.len() != m * dim {
        return Err(OtError::IncompatibleLength {
            a: y.len(),
            b: m * dim,
        });
    }
    if cfg.n_projections == 0 {
        return Err(OtError::BadCount {
            got: cfg.n_projections,
        });
    }
    if cfg.p == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    Ok(())
}

/// Project all samples (row-major, `count × dim`) onto unit vector `theta`.
fn project_samples(samples: &[f64], count: usize, dim: usize, theta: &[f64], out: &mut [f64]) {
    for i in 0..count {
        let row = &samples[i * dim..(i + 1) * dim];
        let mut dot = 0.0_f64;
        for (&r, &t) in row.iter().zip(theta.iter()) {
            dot += r * t;
        }
        out[i] = dot;
    }
}

/// Sort a `f64` buffer in ascending order (NaN-safe).
fn sort_f64(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Raise `x` to the integer power `p`.
#[inline]
fn pow_u32_f64(x: f64, p: u32) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..p {
        acc *= x;
    }
    acc
}

/// 1D W_p^p between sorted, uniform-weight empirical samples.
///
/// When `nx == ny` this reduces to the direct sorted L^p average.
/// Otherwise we integrate over a common quantile grid via a merge-scan.
fn w_pp_1d(sx: &[f64], sy: &[f64], p: u32) -> f64 {
    let nx = sx.len();
    let ny = sy.len();
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    if nx == ny {
        let mut s = 0.0_f64;
        for (a, b) in sx.iter().zip(sy.iter()) {
            let d = (a - b).abs();
            s += pow_u32_f64(d, p);
        }
        return s / nx as f64;
    }
    // Generic: scan through merged CDF breakpoints.
    let inv_nx = 1.0_f64 / nx as f64;
    let inv_ny = 1.0_f64 / ny as f64;
    let mut total = 0.0_f64;
    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut cum_x = 0.0_f64;
    let mut cum_y = 0.0_f64;
    while i < nx && j < ny {
        let nxv = cum_x + inv_nx;
        let nyv = cum_y + inv_ny;
        let upper = nxv.min(nyv);
        let segment = upper - cum_x.max(cum_y);
        if segment > 0.0 {
            let d = (sx[i] - sy[j]).abs();
            total += segment * pow_u32_f64(d, p);
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

// ─── public API ────────────────────────────────────────────────────────────

/// Draw a random unit vector in `R^dim` using `rng`.
///
/// Components are sampled from N(0,1) (via Box-Muller inside `LcgRng`) then
/// L2-normalised; the result lives on the unit sphere `S^{dim-1}`.
pub fn random_unit_vector(dim: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut v = Vec::with_capacity(dim);
    // Box-Muller pairs; fill_normal works on f32 so we draw component by component.
    let mut i = 0_usize;
    while i + 1 < dim {
        // Exploit paired draw to avoid discarding a sample.
        let (a, b) = {
            let u1 = (rng.next_f32() as f64).max(1e-15);
            let u2 = rng.next_f32() as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            (r * theta.cos(), r * theta.sin())
        };
        v.push(a);
        v.push(b);
        i += 2;
    }
    if i < dim {
        v.push(rng.next_normal() as f64);
    }
    // L2-normalise.
    let mut norm_sq = 0.0_f64;
    for &c in &v {
        norm_sq += c * c;
    }
    let norm = norm_sq.sqrt().max(1e-15);
    for c in v.iter_mut() {
        *c /= norm;
    }
    v
}

/// Sliced Wasserstein-`p` estimator (f64 API, Rabin 2012).
///
/// # Arguments
///
/// * `x`   – flat row-major sample array, shape `[n × dim]`.
/// * `n`   – number of samples in `x`.
/// * `y`   – flat row-major sample array, shape `[m × dim]`.
/// * `m`   – number of samples in `y`.
/// * `dim` – ambient dimension.
/// * `cfg` – estimator configuration.
/// * `rng` – mutable reference to the caller-owned `LcgRng`.
///
/// # Returns
///
/// The Monte-Carlo estimate of SW_p(μ, ν), i.e. `(mean W_p^p)^(1/p)`.
pub fn sliced_wasserstein(
    x: &[f64],
    n: usize,
    y: &[f64],
    m: usize,
    dim: usize,
    cfg: &SlicedWassersteinConfig,
    rng: &mut LcgRng,
) -> OtResult<f64> {
    validate(x, n, y, m, dim, cfg)?;

    let mut proj_x = vec![0.0_f64; n];
    let mut proj_y = vec![0.0_f64; m];
    let mut sum_pp = 0.0_f64;

    for _ in 0..cfg.n_projections {
        let theta = random_unit_vector(dim, rng);
        project_samples(x, n, dim, &theta, &mut proj_x);
        project_samples(y, m, dim, &theta, &mut proj_y);
        sort_f64(&mut proj_x);
        sort_f64(&mut proj_y);
        sum_pp += w_pp_1d(&proj_x, &proj_y, cfg.p);
    }

    let mean_pp = sum_pp / cfg.n_projections as f64;
    let result = mean_pp.powf(1.0 / cfg.p as f64);
    Ok(result)
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(0xDEAD_BEEF_CAFE_1234)
    }

    // ── Test 1: identical point clouds → distance ≈ 0 ──
    #[test]
    fn equal_distributions_near_zero() {
        let pts: Vec<f64> = (0..30)
            .flat_map(|i| [i as f64 * 0.1, i as f64 * 0.05])
            .collect();
        let n = 30;
        let cfg = SlicedWassersteinConfig {
            n_projections: 100,
            p: 2,
        };
        let mut rng = make_rng();
        let d = sliced_wasserstein(&pts, n, &pts, n, 2, &cfg, &mut rng).expect("ok");
        assert!(
            d.abs() < 1e-10,
            "identical clouds should yield distance ≈ 0, got {d}"
        );
    }

    // ── Test 2: result is non-negative ──
    #[test]
    fn output_nonneg() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = (5..25).map(|i| i as f64).collect();
        let cfg = SlicedWassersteinConfig::default();
        let mut rng = make_rng();
        let d = sliced_wasserstein(&x, 20, &y, 20, 1, &cfg, &mut rng).expect("ok");
        assert!(d >= 0.0, "distance must be non-negative, got {d}");
    }

    // ── Test 3: SW is symmetric ──
    #[test]
    fn symmetry() {
        let x: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0];
        let y: Vec<f64> = vec![2.0, 0.0, 3.0, 0.0, 2.5, 1.0];
        let n = 3;
        let m = 3;
        let cfg = SlicedWassersteinConfig {
            n_projections: 200,
            p: 2,
        };
        // Both calls must use identical RNG states to test mathematical symmetry
        // while controlling for MC variance — we check they are close, not equal.
        let mut rng_ab = LcgRng::new(1111);
        let mut rng_ba = LcgRng::new(1111);
        let dab = sliced_wasserstein(&x, n, &y, m, 2, &cfg, &mut rng_ab).expect("ok");
        let dba = sliced_wasserstein(&y, m, &x, n, 2, &cfg, &mut rng_ba).expect("ok");
        assert!(
            (dab - dba).abs() < 1e-8,
            "SW should be symmetric: d(x,y)={dab}, d(y,x)={dba}"
        );
    }

    // ── Test 4: translated cloud has known approximate distance ──
    #[test]
    fn shape_preserving_translation() {
        // 1-D cloud translated by delta: SW_2 should equal |delta|.
        let n = 50_usize;
        let delta = 3.0_f64;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v + delta).collect();
        let cfg = SlicedWassersteinConfig {
            n_projections: 500,
            p: 2,
        };
        let mut rng = make_rng();
        let d = sliced_wasserstein(&x, n, &y, n, 1, &cfg, &mut rng).expect("ok");
        // In 1-D with a single projection dimension the estimate is exact.
        assert!(
            (d - delta).abs() < 0.05,
            "1-D translation by {delta}: expected SW≈{delta}, got {d}"
        );
    }

    // ── Test 5: n_projections = 1 succeeds ──
    #[test]
    fn n_projections_1_ok() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![1.0_f64, 2.0, 3.0];
        let cfg = SlicedWassersteinConfig {
            n_projections: 1,
            p: 2,
        };
        let mut rng = make_rng();
        let d = sliced_wasserstein(&x, 3, &y, 3, 1, &cfg, &mut rng).expect("should succeed");
        assert!(d.is_finite() && d >= 0.0);
    }

    // ── Test 6: more projections → lower variance of repeated estimates ──
    #[test]
    fn n_projections_convergence() {
        // For higher-dimensional data, more projections should yield a more
        // stable (lower-variance) estimate. We measure variance over independent
        // estimates and confirm it is strictly smaller at high projection count.
        let dim = 4_usize;
        let n = 40_usize;
        let m = 40_usize;
        let x: Vec<f64> = (0..n * dim).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..m * dim).map(|i| (i as f64) * 0.01 + 0.5).collect();

        let repeats = 20_usize;

        let variance_of = |n_proj: usize| -> f64 {
            let mut vals = Vec::with_capacity(repeats);
            for seed in 0..repeats as u64 {
                let mut rng = LcgRng::new(seed * 7919 + 3);
                let cfg = SlicedWassersteinConfig {
                    n_projections: n_proj,
                    p: 2,
                };
                let v = sliced_wasserstein(&x, n, &y, m, dim, &cfg, &mut rng).expect("ok");
                vals.push(v);
            }
            let mean = vals.iter().sum::<f64>() / repeats as f64;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / repeats as f64
        };

        let var_low = variance_of(5);
        let var_high = variance_of(500);
        assert!(
            var_high < var_low,
            "higher projection count should reduce variance: var(5)={var_low:.6}, var(500)={var_high:.6}"
        );
    }

    // ── Test 7: output is always finite ──
    #[test]
    fn finite_output() {
        let x: Vec<f64> = vec![0.0, 1.0, 0.0, -1.0];
        let y: Vec<f64> = vec![10.0, 20.0, -5.0, 7.0];
        let cfg = SlicedWassersteinConfig {
            n_projections: 50,
            p: 3,
        };
        let mut rng = make_rng();
        let d = sliced_wasserstein(&x, 2, &y, 2, 2, &cfg, &mut rng).expect("ok");
        assert!(d.is_finite(), "result must be finite, got {d}");
    }

    // ── Test 8: dimension mismatch yields an error ──
    #[test]
    fn dim_mismatch_error() {
        // x claims n=3 samples in dim=2 but only has 5 elements (not 6).
        let x = vec![0.0_f64; 5];
        let y = vec![0.0_f64; 6];
        let cfg = SlicedWassersteinConfig::default();
        let mut rng = make_rng();
        let res = sliced_wasserstein(&x, 3, &y, 3, 2, &cfg, &mut rng);
        assert!(
            matches!(res, Err(OtError::IncompatibleLength { .. })),
            "expected IncompatibleLength error, got {res:?}"
        );
    }

    // ── Test 9: random_unit_vector has unit norm ──
    #[test]
    fn unit_vector_normalized() {
        let mut rng = make_rng();
        for dim in [1_usize, 2, 3, 5, 10, 100] {
            let v = random_unit_vector(dim, &mut rng);
            assert_eq!(v.len(), dim);
            let norm_sq: f64 = v.iter().map(|&c| c * c).sum();
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 1e-12,
                "dim={dim}: expected unit norm, got {norm_sq:.15}"
            );
        }
    }
}
