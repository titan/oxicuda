//! Spherical Sliced-Wasserstein (SSW) for distributions on S^{d−1}.
//!
//! ## Overview
//!
//! The Spherical Sliced-Wasserstein distance (Bonet et al. 2022) extends the
//! classical Sliced-Wasserstein to distributions on the unit sphere. Instead of
//! projecting onto Euclidean lines, SSW projects onto great-circle geodesics and
//! uses the arc-length metric for the 1-D comparison.
//!
//! For a random unit vector `v ∈ S^{d−1}` the projection of a point `x ∈ S^{d-1}`
//! onto the great-circle with pole `v` is the signed geodesic coordinate
//!
//! ```text
//! t = arcsin(clamp(x^T v, -1, 1))  ∈ [-π/2, π/2]
//! ```
//!
//! The SSW distance is then the expectation (over uniform `v`) of the 1-D
//! W_p distance between the projected distributions, raised to the power `1/p`.
//!
//! ## Max-SSW
//!
//! The Max-SSW variant optimises over the projection direction via Riemannian
//! gradient ascent on `S^{d-1}`, using a warm-start from a pool of random
//! directions.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

// ─────────────────────────────── config ─────────────────────────────────────

/// Configuration for the Spherical Sliced-Wasserstein estimator.
#[derive(Debug, Clone)]
pub struct SphericalSlicedConfig {
    /// Number of random great-circle projections (Monte-Carlo budget).
    pub n_proj: usize,
    /// Wasserstein order `p` (1 or 2, higher values also supported).
    pub p: u32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SphericalSlicedConfig {
    fn default() -> Self {
        Self {
            n_proj: 200,
            p: 2,
            seed: 42,
        }
    }
}

/// Configuration for the Max-Spherical-Sliced-Wasserstein estimator.
#[derive(Debug, Clone)]
pub struct MaxSSWConfig {
    /// Number of random projections used for warm-start initialisation.
    pub n_init_proj: usize,
    /// Number of Riemannian gradient ascent steps.
    pub n_grad_steps: usize,
    /// Learning rate (step size on the sphere manifold).
    pub lr: f64,
    /// Wasserstein order.
    pub p: u32,
    /// RNG seed.
    pub seed: u64,
}

impl Default for MaxSSWConfig {
    fn default() -> Self {
        Self {
            n_init_proj: 50,
            n_grad_steps: 30,
            lr: 0.05,
            p: 2,
            seed: 42,
        }
    }
}

// ─────────────────────────────── geometry helpers ───────────────────────────

/// Sample a uniformly random unit vector on `S^{d-1}` by normalising i.i.d.
/// standard-normal coordinates (Box-Muller via LcgRng).
pub fn sample_uniform_sphere(d: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..d).map(|_| rng.next_normal() as f64).collect();
    normalise_to_sphere_in_place(&mut v);
    v
}

/// Divide a vector by its L2 norm; returns the input unchanged if norm ≤ 1e-15.
pub fn normalise_to_sphere(v: &[f64]) -> Vec<f64> {
    let mut out = v.to_vec();
    normalise_to_sphere_in_place(&mut out);
    out
}

fn normalise_to_sphere_in_place(v: &mut [f64]) {
    let nrm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if nrm > 1e-15 {
        let inv = 1.0 / nrm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Clamp to `[-1, 1]` to guard against floating-point errors outside this range.
#[inline]
fn clamp_unit(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Project a point on the sphere onto the great-circle with pole `v`.
/// Returns the signed arc-length coordinate `arcsin(x^T v)` in `[-π/2, π/2]`.
#[inline]
fn sphere_project(x: &[f64], v: &[f64]) -> f64 {
    clamp_unit(dot(x, v)).asin()
}

// ─────────────────────────────── 1-D W_p via quantile matching ──────────────

/// Compute W_p^p between two sorted 1-D empirical distributions with weights.
///
/// The quantile functions are linearly interpolated on a common uniform grid.
/// For equal-weight uniform distributions this reduces to direct L^p comparison
/// of sorted values.
///
/// `sorted_a` and `weights_a`: support and weights of distribution A (sorted
/// by support value ascending). Same for `sorted_b`.
pub fn w_p_1d(
    sorted_a: &[f64],
    weights_a: &[f64],
    sorted_b: &[f64],
    weights_b: &[f64],
    p: u32,
) -> f64 {
    let na = sorted_a.len();
    let nb = sorted_b.len();
    if na == 0 || nb == 0 {
        return 0.0;
    }
    debug_assert_eq!(na, weights_a.len());
    debug_assert_eq!(nb, weights_b.len());

    // Build CDFs: CDF_a[i] = Σ_{k≤i} weights_a[k].
    let mut cdf_a = vec![0.0_f64; na + 1];
    let mut cdf_b = vec![0.0_f64; nb + 1];
    for i in 0..na {
        cdf_a[i + 1] = cdf_a[i] + weights_a[i];
    }
    for j in 0..nb {
        cdf_b[j + 1] = cdf_b[j] + weights_b[j];
    }

    // Merge CDF breakpoints and integrate |Q_a(u) - Q_b(u)|^p du.
    // Each segment lies in interval [cum_lo, cum_hi].
    let mut total = 0.0_f64;
    let mut ia = 0_usize; // current atom of A
    let mut ib = 0_usize; // current atom of B
    let mut cum = 0.0_f64; // current lower CDF boundary

    while ia < na && ib < nb {
        let next_a = cdf_a[ia + 1];
        let next_b = cdf_b[ib + 1];
        let next_boundary = next_a.min(next_b);
        let segment_mass = next_boundary - cum;
        if segment_mass > 0.0 {
            let diff = (sorted_a[ia] - sorted_b[ib]).abs();
            total += segment_mass * pow_u32_f64(diff, p);
        }
        cum = next_boundary;
        if next_a <= next_b {
            ia += 1;
        }
        if next_b <= next_a {
            ib += 1;
        }
    }
    total
}

/// Fast integer power for f64.
#[inline]
fn pow_u32_f64(x: f64, p: u32) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..p {
        acc *= x;
    }
    acc
}

// ─────────────────────────────── SSW ────────────────────────────────────────

/// Validate common SSW inputs.
fn validate_ssw(
    x_pts: &[Vec<f64>],
    x_weights: &[f64],
    y_pts: &[Vec<f64>],
    y_weights: &[f64],
    n_proj: usize,
    p: u32,
) -> OtResult<usize> {
    if x_pts.is_empty() || y_pts.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if x_pts.len() != x_weights.len() {
        return Err(OtError::IncompatibleLength {
            a: x_pts.len(),
            b: x_weights.len(),
        });
    }
    if y_pts.len() != y_weights.len() {
        return Err(OtError::IncompatibleLength {
            a: y_pts.len(),
            b: y_weights.len(),
        });
    }
    if n_proj == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    if p == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    let d = x_pts[0].len();
    if d == 0 {
        return Err(OtError::BadDim { got: 0 });
    }
    for pt in x_pts.iter() {
        if pt.len() != d {
            return Err(OtError::IncompatibleLength { a: d, b: pt.len() });
        }
    }
    for pt in y_pts.iter() {
        if pt.len() != d {
            return Err(OtError::IncompatibleLength { a: d, b: pt.len() });
        }
    }
    for &w in x_weights.iter().chain(y_weights.iter()) {
        if w < 0.0 || !w.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(d)
}

/// Normalise a weight vector to sum to 1.
fn normalise_weights(w: &[f64]) -> Vec<f64> {
    let total: f64 = w.iter().sum();
    if total <= 1e-15 {
        let n = w.len();
        return vec![1.0 / n as f64; n];
    }
    w.iter().map(|&v| v / total).collect()
}

/// Normalise all points to lie on the unit sphere.
fn normalise_points(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter().map(|p| normalise_to_sphere(p)).collect()
}

/// Sort projected values together with weights; returns sorted (proj, weight) pairs.
fn sort_proj_with_weights(proj: &[f64], weights: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = proj.len();
    let mut indexed: Vec<(f64, f64)> = proj
        .iter()
        .zip(weights.iter())
        .map(|(&p, &w)| (p, w))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_proj: Vec<f64> = indexed.iter().map(|(p, _)| *p).collect();
    let sorted_w: Vec<f64> = indexed.iter().map(|(_, w)| *w).collect();
    let _ = n;
    (sorted_proj, sorted_w)
}

/// Compute SSW for a single projection direction `v`.
fn ssw_single_proj(
    x_pts_norm: &[Vec<f64>],
    x_weights_norm: &[f64],
    y_pts_norm: &[Vec<f64>],
    y_weights_norm: &[f64],
    v: &[f64],
    p: u32,
) -> f64 {
    let proj_x: Vec<f64> = x_pts_norm.iter().map(|pt| sphere_project(pt, v)).collect();
    let proj_y: Vec<f64> = y_pts_norm.iter().map(|pt| sphere_project(pt, v)).collect();

    let (sx, wx) = sort_proj_with_weights(&proj_x, x_weights_norm);
    let (sy, wy) = sort_proj_with_weights(&proj_y, y_weights_norm);

    w_p_1d(&sx, &wx, &sy, &wy, p)
}

/// Spherical Sliced-Wasserstein distance between two discrete distributions on S^{d−1}.
///
/// Points are normalised to the sphere internally; weights are normalised to
/// form probability vectors. The returned distance is averaged over `n_proj`
/// random great-circle projections.
pub fn spherical_sliced_wasserstein(
    x_pts: &[Vec<f64>],
    x_weights: &[f64],
    y_pts: &[Vec<f64>],
    y_weights: &[f64],
    config: &SphericalSlicedConfig,
    rng: &mut LcgRng,
) -> OtResult<f64> {
    let d = validate_ssw(x_pts, x_weights, y_pts, y_weights, config.n_proj, config.p)?;

    let x_norm = normalise_points(x_pts);
    let y_norm = normalise_points(y_pts);
    let xw_norm = normalise_weights(x_weights);
    let yw_norm = normalise_weights(y_weights);

    let mut sum_pp = 0.0_f64;
    for _ in 0..config.n_proj {
        let v = sample_uniform_sphere(d, rng);
        sum_pp += ssw_single_proj(&x_norm, &xw_norm, &y_norm, &yw_norm, &v, config.p);
    }

    let mean_pp = sum_pp / config.n_proj as f64;
    Ok(mean_pp.max(0.0).powf(1.0 / config.p as f64))
}

// ─────────────────────────────── Max-SSW ────────────────────────────────────

/// Compute the Riemannian gradient of `SSW_proj(v)` w.r.t. `v` at a given direction.
///
/// Uses finite differences on the sphere (perturbation in each coordinate direction,
/// followed by retraction via normalisation).
fn riemannian_gradient(
    x_pts_norm: &[Vec<f64>],
    x_weights_norm: &[f64],
    y_pts_norm: &[Vec<f64>],
    y_weights_norm: &[f64],
    v: &[f64],
    p: u32,
    h: f64,
) -> Vec<f64> {
    let d = v.len();
    let f0 = ssw_single_proj(x_pts_norm, x_weights_norm, y_pts_norm, y_weights_norm, v, p);
    let mut grad = vec![0.0_f64; d];

    for i in 0..d {
        let mut v_perturbed = v.to_vec();
        v_perturbed[i] += h;
        normalise_to_sphere_in_place(&mut v_perturbed);
        let fp = ssw_single_proj(
            x_pts_norm,
            x_weights_norm,
            y_pts_norm,
            y_weights_norm,
            &v_perturbed,
            p,
        );
        grad[i] = (fp - f0) / h;
    }

    // Riemannian gradient = Euclidean gradient projected onto tangent space T_v S^{d-1}:
    // Riem_grad = grad - (grad · v) v
    let dot_gv = dot(&grad, v);
    for i in 0..d {
        grad[i] -= dot_gv * v[i];
    }
    grad
}

/// Max-Spherical-Sliced-Wasserstein: find the projection direction maximising SSW.
///
/// Initialises with `n_init_proj` random directions, picks the best, then refines
/// it with `n_grad_steps` Riemannian gradient ascent steps on `S^{d-1}`.
pub fn max_spherical_sliced_wasserstein(
    x_pts: &[Vec<f64>],
    x_weights: &[f64],
    y_pts: &[Vec<f64>],
    y_weights: &[f64],
    config: &MaxSSWConfig,
    rng: &mut LcgRng,
) -> OtResult<f64> {
    let d = validate_ssw(
        x_pts,
        x_weights,
        y_pts,
        y_weights,
        config.n_init_proj.max(1),
        config.p,
    )?;

    let x_norm = normalise_points(x_pts);
    let y_norm = normalise_points(y_pts);
    let xw_norm = normalise_weights(x_weights);
    let yw_norm = normalise_weights(y_weights);

    // Warm-start: sample n_init_proj random directions and keep the best.
    let mut best_val = f64::NEG_INFINITY;
    let mut best_v = sample_uniform_sphere(d, rng);

    for _ in 0..config.n_init_proj {
        let v = sample_uniform_sphere(d, rng);
        let val = ssw_single_proj(&x_norm, &xw_norm, &y_norm, &yw_norm, &v, config.p);
        if val > best_val {
            best_val = val;
            best_v = v;
        }
    }

    // Riemannian gradient ascent from best_v.
    let h = 1e-4_f64;
    let mut v = best_v;

    for _ in 0..config.n_grad_steps {
        let grad = riemannian_gradient(&x_norm, &xw_norm, &y_norm, &yw_norm, &v, config.p, h);
        // Retracted gradient step: v ← norm(v + lr * grad).
        for i in 0..d {
            v[i] += config.lr * grad[i];
        }
        normalise_to_sphere_in_place(&mut v);
        let val = ssw_single_proj(&x_norm, &xw_norm, &y_norm, &yw_norm, &v, config.p);
        if val > best_val {
            best_val = val;
        }
    }

    Ok(best_val.max(0.0).powf(1.0 / config.p as f64))
}

// ─────────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_pts_2d(angles_deg: &[f64]) -> Vec<Vec<f64>> {
        angles_deg
            .iter()
            .map(|&a| {
                let r = a.to_radians();
                vec![r.cos(), r.sin()]
            })
            .collect()
    }

    fn uniform_w(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    // ─── geometry helpers ────────────────────────────────────────────────────

    #[test]
    fn uniform_sphere_sample_has_unit_norm() {
        let mut rng = LcgRng::new(1);
        for d in [2, 3, 5, 10] {
            let v = sample_uniform_sphere(d, &mut rng);
            let nrm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-12, "d={d}: norm={nrm}");
        }
    }

    #[test]
    fn normalise_to_sphere_is_idempotent() {
        let v = vec![3.0_f64, 4.0];
        let once = normalise_to_sphere(&v);
        let twice = normalise_to_sphere(&once);
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        let nrm: f64 = once.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_project_maps_to_valid_range() {
        let mut rng = LcgRng::new(2);
        let d = 5;
        for _ in 0..50 {
            let x = sample_uniform_sphere(d, &mut rng);
            let v = sample_uniform_sphere(d, &mut rng);
            let t = sphere_project(&x, &v);
            let lo = -std::f64::consts::FRAC_PI_2 - 1e-10;
            let hi = std::f64::consts::FRAC_PI_2 + 1e-10;
            assert!((lo..=hi).contains(&t), "t={t} out of [-π/2, π/2]");
        }
    }

    // ─── SSW = 0 for identical distributions ─────────────────────────────────

    #[test]
    fn ssw_zero_for_identical_distributions() {
        let pts = unit_pts_2d(&[0.0, 90.0, 180.0, 270.0]);
        let w = uniform_w(4);
        let mut rng = LcgRng::new(42);
        let cfg = SphericalSlicedConfig {
            n_proj: 50,
            p: 2,
            seed: 42,
        };
        let d = spherical_sliced_wasserstein(&pts, &w, &pts, &w, &cfg, &mut rng).expect("ok");
        assert!(d.abs() < 1e-10, "SSW(P,P)={d}");
    }

    // ─── SSW > 0 for well-separated distributions ────────────────────────────

    #[test]
    fn ssw_positive_for_separated_distributions() {
        // North pole vs south pole on S^2.
        let x_pts = vec![vec![0.0, 0.0, 1.0]]; // north
        let y_pts = vec![vec![0.0, 0.0, -1.0]]; // south
        let w = vec![1.0];
        let mut rng = LcgRng::new(7);
        let cfg = SphericalSlicedConfig {
            n_proj: 100,
            p: 2,
            seed: 7,
        };
        let d = spherical_sliced_wasserstein(&x_pts, &w, &y_pts, &w, &cfg, &mut rng).expect("ok");
        assert!(d > 0.1, "SSW should be positive, got {d}");
    }

    // ─── SSW is symmetric ────────────────────────────────────────────────────

    #[test]
    fn ssw_is_symmetric() {
        let pts_x = unit_pts_2d(&[0.0, 45.0]);
        let pts_y = unit_pts_2d(&[90.0, 135.0]);
        let wx = uniform_w(2);
        let wy = uniform_w(2);
        let cfg = SphericalSlicedConfig {
            n_proj: 80,
            p: 2,
            seed: 5,
        };
        let mut rng1 = LcgRng::new(5);
        let mut rng2 = LcgRng::new(5);
        let dxy =
            spherical_sliced_wasserstein(&pts_x, &wx, &pts_y, &wy, &cfg, &mut rng1).expect("ok");
        let dyx =
            spherical_sliced_wasserstein(&pts_y, &wy, &pts_x, &wx, &cfg, &mut rng2).expect("ok");
        // With the same seed, directions are the same so should be exactly equal.
        assert!(
            (dxy - dyx).abs() < 1e-12,
            "SSW not symmetric: {dxy} vs {dyx}"
        );
    }

    // ─── SSW with p=1 vs p=2 ─────────────────────────────────────────────────

    #[test]
    fn ssw_p1_vs_p2_both_finite_positive() {
        let pts_x = unit_pts_2d(&[0.0, 60.0]);
        let pts_y = unit_pts_2d(&[120.0, 180.0]);
        let w = uniform_w(2);
        let mut rng = LcgRng::new(17);
        let cfg1 = SphericalSlicedConfig {
            n_proj: 40,
            p: 1,
            seed: 17,
        };
        let cfg2 = SphericalSlicedConfig {
            n_proj: 40,
            p: 2,
            seed: 17,
        };
        let mut rng2 = LcgRng::new(17);
        let d1 = spherical_sliced_wasserstein(&pts_x, &w, &pts_y, &w, &cfg1, &mut rng).expect("ok");
        let d2 =
            spherical_sliced_wasserstein(&pts_x, &w, &pts_y, &w, &cfg2, &mut rng2).expect("ok");
        assert!(d1.is_finite() && d1 > 0.0, "p=1 distance={d1}");
        assert!(d2.is_finite() && d2 > 0.0, "p=2 distance={d2}");
    }

    // ─── SSW decreases as distributions converge ─────────────────────────────

    #[test]
    fn ssw_decreases_as_distributions_converge() {
        // y_pts shifts gradually toward x_pts.
        let pts_x = unit_pts_2d(&[0.0]);
        let wx = vec![1.0];
        let pts_y_far = unit_pts_2d(&[180.0]);
        let pts_y_close = unit_pts_2d(&[10.0]);
        let wy = vec![1.0];
        let cfg = SphericalSlicedConfig {
            n_proj: 100,
            p: 2,
            seed: 3,
        };
        let mut rng1 = LcgRng::new(3);
        let mut rng2 = LcgRng::new(3);
        let d_far = spherical_sliced_wasserstein(&pts_x, &wx, &pts_y_far, &wy, &cfg, &mut rng1)
            .expect("ok");
        let d_close = spherical_sliced_wasserstein(&pts_x, &wx, &pts_y_close, &wy, &cfg, &mut rng2)
            .expect("ok");
        assert!(
            d_close < d_far,
            "distance should decrease as y approaches x: far={d_far}, close={d_close}"
        );
    }

    // ─── SphericalSlicedConfig defaults ──────────────────────────────────────

    #[test]
    fn spherical_sliced_config_defaults() {
        let cfg = SphericalSlicedConfig::default();
        assert_eq!(cfg.n_proj, 200);
        assert_eq!(cfg.p, 2);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn max_ssw_config_defaults() {
        let cfg = MaxSSWConfig::default();
        assert_eq!(cfg.n_init_proj, 50);
        assert_eq!(cfg.n_grad_steps, 30);
        assert!((cfg.lr - 0.05).abs() < 1e-15);
        assert_eq!(cfg.p, 2);
        assert_eq!(cfg.seed, 42);
    }

    // ─── max-SSW ≥ SSW ───────────────────────────────────────────────────────

    #[test]
    fn max_ssw_at_least_ssw() {
        let pts_x = unit_pts_2d(&[0.0, 90.0]);
        let pts_y = unit_pts_2d(&[45.0, 135.0]);
        let w = uniform_w(2);
        let mut rng_ssw = LcgRng::new(22);
        let mut rng_max = LcgRng::new(22);

        let ssw_cfg = SphericalSlicedConfig {
            n_proj: 100,
            p: 2,
            seed: 22,
        };
        let max_cfg = MaxSSWConfig {
            n_init_proj: 100,
            n_grad_steps: 20,
            lr: 0.05,
            p: 2,
            seed: 22,
        };

        let ssw_val = spherical_sliced_wasserstein(&pts_x, &w, &pts_y, &w, &ssw_cfg, &mut rng_ssw)
            .expect("ok");
        let max_val =
            max_spherical_sliced_wasserstein(&pts_x, &w, &pts_y, &w, &max_cfg, &mut rng_max)
                .expect("ok");

        // Max-SSW should be ≥ SSW (max ≥ mean), allowing small numerical slack.
        assert!(
            max_val + 1e-6 >= ssw_val,
            "max_ssw={max_val} should be >= ssw={ssw_val}"
        );
    }

    // ─── w_p_1d on Dirac masses ───────────────────────────────────────────────

    #[test]
    fn w_p_1d_dirac_masses_exact_distance() {
        // W_2(δ_0, δ_1) = 1, W_2^2 = 1.
        let sa = [0.0_f64];
        let wa = [1.0_f64];
        let sb = [1.0_f64];
        let wb = [1.0_f64];
        let wp2 = w_p_1d(&sa, &wa, &sb, &wb, 2);
        assert!((wp2 - 1.0).abs() < 1e-12, "W_2^2(δ_0,δ_1)={wp2}");

        // W_1(δ_0, δ_3) = 3, W_1^1 = 3.
        let sc = [3.0_f64];
        let wp1 = w_p_1d(&sa, &wa, &sc, &wb, 1);
        assert!((wp1 - 3.0).abs() < 1e-12, "W_1(δ_0,δ_3)={wp1}");
    }

    #[test]
    fn w_p_1d_same_distribution_zero() {
        let s = [0.0_f64, 0.5, 1.0];
        let w = [1.0 / 3.0; 3];
        let wp = w_p_1d(&s, &w, &s, &w, 2);
        assert!(wp.abs() < 1e-12, "W_2^2(P,P)={wp}");
    }

    // ─── error cases ─────────────────────────────────────────────────────────

    #[test]
    fn rejects_empty_points() {
        let cfg = SphericalSlicedConfig::default();
        let mut rng = LcgRng::new(0);
        let res = spherical_sliced_wasserstein(&[], &[], &[vec![1.0]], &[1.0], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn rejects_zero_projections() {
        let pts = vec![vec![1.0_f64, 0.0]];
        let w = vec![1.0];
        let cfg = SphericalSlicedConfig {
            n_proj: 0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(0);
        let res = spherical_sliced_wasserstein(&pts, &w, &pts, &w, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }
}
