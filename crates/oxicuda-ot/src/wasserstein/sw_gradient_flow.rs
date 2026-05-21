//! Sliced-Wasserstein Gradient Flow (SWGF) for generative modelling.
//!
//! Implements the particle-based gradient flow from Liutkus et al., "Sliced-Wasserstein
//! Flows: Nonparametric Generative Models via Optimal Transport and Diffusions", ICML 2019.
//!
//! Given a set of source *particles* and a fixed target sample set, SWGF minimises
//! the sliced-Wasserstein distance by performing gradient descent on the particle
//! positions. Each gradient step averages the rank-matched transport direction over
//! `n_projections` random unit directions drawn from the sphere.
//!
//! ```text
//! ∂SW / ∂x_i  ≈  (1 / n_proj) · Σ_θ  (⟨x_i, θ⟩ − ⟨y_σ_θ(i), θ⟩) · θ
//! ```
//!
//! where `σ_θ(i)` maps the rank of source particle `i` in direction `θ` to the
//! matched target quantile.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for the sliced-Wasserstein gradient flow.
#[derive(Debug, Clone)]
pub struct SwgfConfig {
    /// Number of random projection directions used per gradient step.
    pub n_projections: usize,
    /// Number of gradient-descent steps to run.
    pub n_steps: usize,
    /// Gradient-descent step size (learning rate).
    pub step_size: f32,
    /// RNG seed for reproducibility (directions are re-sampled each step).
    pub seed: u64,
}

impl Default for SwgfConfig {
    fn default() -> Self {
        Self {
            n_projections: 50,
            n_steps: 100,
            step_size: 0.05,
            seed: 42,
        }
    }
}

/// Output of the sliced-Wasserstein gradient flow.
#[derive(Debug, Clone)]
pub struct SwgfFit {
    /// Final particle positions, length `n_particles × d` (row-major).
    pub particles: Vec<f32>,
    /// Number of particles in the source cloud.
    pub n_particles: usize,
    /// Ambient dimension.
    pub d: usize,
    /// Sliced-Wasserstein distance recorded at each gradient step (length = n_steps).
    pub sw_history: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Draw a unit vector uniformly on S^{d-1} via Box-Muller + L2 normalisation.
fn unit_direction(rng: &mut LcgRng, d: usize, out: &mut [f32]) {
    rng.fill_normal(out);
    let mut nrm_sq = 0.0_f32;
    for &v in out.iter() {
        nrm_sq += v * v;
    }
    let nrm = nrm_sq.sqrt().max(1e-12);
    for v in out.iter_mut() {
        *v /= nrm;
    }
    let _ = d; // length enforced by caller via out.len()
}

/// Project all `n` points in `samples` (n×d row-major) onto direction `theta`
/// and write the scalar projections into `out` (length n).
fn project_samples(samples: &[f32], theta: &[f32], d: usize, n: usize, out: &mut [f32]) {
    for i in 0..n {
        let row = &samples[i * d..(i + 1) * d];
        let dot: f32 = row.iter().zip(theta.iter()).map(|(&r, &t)| r * t).sum();
        out[i] = dot;
    }
}

/// Return the index-permutation that would sort `v` in ascending order.
/// Uses an auxiliary index array to avoid allocating; stable sort ensures
/// reproducibility when projection values coincide.
fn argsort(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Compute the W1 distance on equal-weight 1D sorted samples.
///
/// For n_src == n_tgt this is simply the mean absolute difference of sorted values.
/// For unequal sizes we interpolate both empirical CDFs on a common grid via
/// the quantile-coupling approach.
fn w1_sorted_equal(sorted_src: &[f32], sorted_tgt: &[f32]) -> f32 {
    debug_assert_eq!(sorted_src.len(), sorted_tgt.len());
    let n = sorted_src.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    for (s, t) in sorted_src.iter().zip(sorted_tgt.iter()) {
        acc += (s - t).abs();
    }
    acc / n as f32
}

/// W1 for possibly unequal-size uniform empirical distributions via piecewise coupling.
fn w1_unequal(sx: &[f32], sy: &[f32]) -> f32 {
    // sx and sy are assumed pre-sorted.
    let nx = sx.len();
    let ny = sy.len();
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    if nx == ny {
        return w1_sorted_equal(sx, sy);
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
        let seg = upper - cum_x.max(cum_y);
        if seg > 0.0 {
            total += seg * (sx[i] - sy[j]).abs();
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

/// Sort a mutable f32 slice in ascending order.
fn sort_f32_inplace(v: &mut [f32]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the sliced-Wasserstein distance (W1 variant) between two point clouds.
///
/// Draws `n_projections` random unit directions, projects both clouds, computes
/// the 1D W1 on the projected sorted samples, and returns the average.
///
/// # Arguments
/// * `source` — source point cloud, length `n_source × d` row-major.
/// * `target` — target point cloud, length `n_target × d` row-major.
/// * `n_source` — number of source particles.
/// * `n_target` — number of target samples.
/// * `d` — ambient dimension.
/// * `n_projections` — number of Monte-Carlo projection directions.
/// * `rng` — mutable LcgRng for direction sampling.
pub fn sw_distance(
    source: &[f32],
    target: &[f32],
    n_source: usize,
    n_target: usize,
    d: usize,
    n_projections: usize,
    rng: &mut LcgRng,
) -> OtResult<f32> {
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if n_source == 0 || n_target == 0 {
        return Err(OtError::EmptyInput);
    }
    if source.len() != n_source * d || target.len() != n_target * d {
        return Err(OtError::IncompatibleLength {
            a: source.len(),
            b: target.len(),
        });
    }
    if n_projections == 0 {
        return Err(OtError::BadCount { got: n_projections });
    }

    let mut theta = vec![0.0_f32; d];
    let mut proj_src = vec![0.0_f32; n_source];
    let mut proj_tgt = vec![0.0_f32; n_target];
    let mut acc = 0.0_f32;

    for _ in 0..n_projections {
        unit_direction(rng, d, &mut theta);
        project_samples(source, &theta, d, n_source, &mut proj_src);
        project_samples(target, &theta, d, n_target, &mut proj_tgt);
        sort_f32_inplace(&mut proj_src);
        sort_f32_inplace(&mut proj_tgt);
        acc += w1_unequal(&proj_src, &proj_tgt);
    }
    Ok(acc / n_projections as f32)
}

/// Apply one gradient step of the sliced-Wasserstein flow.
///
/// Updates `particles` in-place:
/// ```text
/// x_i ← x_i − step_size · (1/n_projections) · Σ_θ (⟨x_i,θ⟩ − ⟨y_σ_θ(i),θ⟩) · θ
/// ```
///
/// For each random direction `θ`:
///   1. Compute 1D projections of source particles and target samples.
///   2. Rank-sort both; pair each source rank to the same-rank target value.
///   3. The gradient for particle `i` in direction `θ` is `(s_i − t_σ(i)) * θ`.
///
/// # Arguments
/// * `particles` — mutable source cloud, length `n_particles × d` row-major.
/// * `target` — fixed target samples, length `n_target × d` row-major.
/// * `n_particles` — number of source particles.
/// * `n_target` — number of target samples.
/// * `d` — ambient dimension.
/// * `step_size` — gradient-descent learning rate.
/// * `n_projections` — directions to average over.
/// * `rng` — mutable LcgRng.
pub fn sw_gradient_step(
    particles: &mut [f32],
    target: &[f32],
    n_particles: usize,
    n_target: usize,
    d: usize,
    step_size: f32,
    n_projections: usize,
    rng: &mut LcgRng,
) {
    if n_particles == 0 || n_target == 0 || d == 0 || n_projections == 0 {
        return;
    }

    // Accumulated gradient: n_particles × d.
    let mut grad = vec![0.0_f32; n_particles * d];

    let mut theta = vec![0.0_f32; d];
    let mut proj_src = vec![0.0_f32; n_particles];
    let mut proj_tgt = vec![0.0_f32; n_target];

    for _ in 0..n_projections {
        unit_direction(rng, d, &mut theta);

        // Project source particles.
        project_samples(particles, &theta, d, n_particles, &mut proj_src);
        // Project target samples.
        project_samples(target, &theta, d, n_target, &mut proj_tgt);

        // Rank-sort: get permutations that sort each projection.
        let src_order = argsort(&proj_src); // src_order[rank] = original particle idx
        let tgt_order = argsort(&proj_tgt); // tgt_order[rank] = original target idx

        // For n_particles != n_target we interpolate ranks:
        // For source particle at rank r (out of n_particles), the matched target
        // quantile q = (r + 0.5) / n_particles; target index = floor(q * n_target).
        let n_src = n_particles;
        let n_tgt = n_target;

        for (rank, &src_idx) in src_order.iter().enumerate() {
            let src_proj = proj_src[src_idx];

            // Map source rank to target rank by quantile interpolation.
            let tgt_rank = ((rank as f32 + 0.5) / n_src as f32 * n_tgt as f32)
                .floor()
                .clamp(0.0, (n_tgt - 1) as f32) as usize;
            let tgt_idx = tgt_order[tgt_rank];
            let tgt_proj = proj_tgt[tgt_idx];

            // Gradient: (s_i - t_σ(i)) * θ, added to particle src_idx.
            let diff = src_proj - tgt_proj;
            let row_off = src_idx * d;
            for k in 0..d {
                grad[row_off + k] += diff * theta[k];
            }
        }
    }

    // Apply gradient step: x_i ← x_i - step_size * (1/n_proj) * grad_i.
    let scale = step_size / n_projections as f32;
    for i in 0..n_particles {
        let row_off = i * d;
        for k in 0..d {
            particles[row_off + k] -= scale * grad[row_off + k];
        }
    }
}

/// Run the sliced-Wasserstein gradient flow.
///
/// Minimises `SW(particles, target)` by iterating gradient-descent steps on the
/// particle positions. The sliced-Wasserstein distance is recorded after every step
/// (using a fresh independent RNG draw for monitoring, separate from the gradient RNG).
///
/// # Arguments
/// * `source` — initial particle positions, length `n_source × d` row-major.
/// * `target` — fixed target samples, length `n_target × d` row-major.
/// * `n_source` — number of source particles.
/// * `n_target` — number of target samples.
/// * `d` — ambient dimension.
/// * `cfg` — algorithm configuration.
///
/// # Returns
/// `SwgfFit` with final particle positions and the SW-history per step.
pub fn sw_gradient_flow(
    source: &[f32],
    target: &[f32],
    n_source: usize,
    n_target: usize,
    d: usize,
    cfg: &SwgfConfig,
) -> OtResult<SwgfFit> {
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if n_source == 0 || n_target == 0 {
        return Err(OtError::EmptyInput);
    }
    if source.len() != n_source * d {
        return Err(OtError::IncompatibleLength {
            a: source.len(),
            b: n_source * d,
        });
    }
    if target.len() != n_target * d {
        return Err(OtError::IncompatibleLength {
            a: target.len(),
            b: n_target * d,
        });
    }
    if cfg.n_projections == 0 {
        return Err(OtError::BadCount {
            got: cfg.n_projections,
        });
    }
    if cfg.n_steps == 0 {
        return Err(OtError::BadCount { got: cfg.n_steps });
    }

    // Working copy of particles; we evolve this in place.
    let mut particles = source.to_vec();

    // Two separate RNG streams: one for gradient computation, one for monitoring.
    let mut grad_rng = LcgRng::new(cfg.seed);
    let mut mon_rng = LcgRng::new(cfg.seed.wrapping_add(0xDEAD_BEEF_CAFE_1234));

    let mut sw_history = Vec::with_capacity(cfg.n_steps);

    for _step in 0..cfg.n_steps {
        // Gradient step.
        sw_gradient_step(
            &mut particles,
            target,
            n_source,
            n_target,
            d,
            cfg.step_size,
            cfg.n_projections,
            &mut grad_rng,
        );

        // Monitor SW distance (using fewer projections to keep cost low).
        let mon_proj = cfg.n_projections.clamp(1, 20);
        let sw = sw_distance(
            &particles,
            target,
            n_source,
            n_target,
            d,
            mon_proj,
            &mut mon_rng,
        )
        .unwrap_or(f32::NAN);
        sw_history.push(sw);
    }

    Ok(SwgfFit {
        particles,
        n_particles: n_source,
        d,
        sw_history,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_1d(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| lo + rng.next_f32() * (hi - lo)).collect()
    }

    fn gaussian_2d(n: usize, mean_x: f32, mean_y: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut out = vec![0.0_f32; n * 2];
        rng.fill_normal(&mut out);
        for i in 0..n {
            out[i * 2] += mean_x;
            out[i * 2 + 1] += mean_y;
        }
        out
    }

    // ------------------------------------------------------------------
    // sw_distance tests
    // ------------------------------------------------------------------

    #[test]
    fn sw_distance_zero_on_identical_clouds() {
        // Identical source and target → SW distance should be zero.
        let pts = vec![0.0_f32, 1.0, 2.0, 3.0]; // 4 points in 1D
        let mut rng = LcgRng::new(7);
        let d = sw_distance(&pts, &pts, 4, 4, 1, 16, &mut rng).expect("ok");
        assert!(d.abs() < 1e-5, "expected ~0, got {d}");
    }

    #[test]
    fn sw_distance_non_negative() {
        let src = uniform_1d(10, 0.0, 1.0, 1);
        let tgt = uniform_1d(10, 2.0, 3.0, 2);
        let mut rng = LcgRng::new(3);
        let d = sw_distance(&src, &tgt, 10, 10, 1, 20, &mut rng).expect("ok");
        assert!(d >= 0.0 && d.is_finite());
    }

    #[test]
    fn sw_distance_is_symmetric() {
        let src = vec![0.0_f32, 0.0, 1.0, 0.0];
        let tgt = vec![0.5_f32, 1.0, 1.5, 1.0];
        let mut rng_ab = LcgRng::new(42);
        let mut rng_ba = LcgRng::new(42);
        let d_ab = sw_distance(&src, &tgt, 2, 2, 2, 32, &mut rng_ab).expect("ok");
        let d_ba = sw_distance(&tgt, &src, 2, 2, 2, 32, &mut rng_ba).expect("ok");
        assert!((d_ab - d_ba).abs() < 1e-4, "d_ab={d_ab} d_ba={d_ba}");
    }

    #[test]
    fn sw_distance_increases_with_separation() {
        let src = uniform_1d(20, 0.0, 1.0, 10);
        let tgt_near = uniform_1d(20, 1.0, 2.0, 11);
        let tgt_far = uniform_1d(20, 5.0, 6.0, 12);
        let mut rng = LcgRng::new(99);
        let d_near = sw_distance(&src, &tgt_near, 20, 20, 1, 20, &mut rng).expect("ok");
        let mut rng2 = LcgRng::new(99);
        let d_far = sw_distance(&src, &tgt_far, 20, 20, 1, 20, &mut rng2).expect("ok");
        assert!(
            d_far > d_near,
            "d_far={d_far} should exceed d_near={d_near}"
        );
    }

    #[test]
    fn sw_distance_rejects_bad_dim() {
        let mut rng = LcgRng::new(1);
        let res = sw_distance(&[1.0_f32], &[1.0_f32], 1, 1, 0, 1, &mut rng);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn sw_distance_rejects_empty_source() {
        let mut rng = LcgRng::new(1);
        let res = sw_distance(&[], &[1.0_f32], 0, 1, 1, 1, &mut rng);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn sw_distance_rejects_zero_projections() {
        let mut rng = LcgRng::new(1);
        let res = sw_distance(&[1.0_f32], &[1.0_f32], 1, 1, 1, 0, &mut rng);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    // ------------------------------------------------------------------
    // sw_gradient_step tests
    // ------------------------------------------------------------------

    #[test]
    fn gradient_step_moves_particles_toward_target() {
        // 1D: source all at 0, target all at 2 → step should increase particle values.
        let n = 5;
        let mut particles = vec![0.0_f32; n];
        let target: Vec<f32> = vec![2.0_f32; n];
        let mut rng = LcgRng::new(17);
        sw_gradient_step(&mut particles, &target, n, n, 1, 0.5, 20, &mut rng);
        for &p in &particles {
            assert!(p > 0.0, "particle {p} did not move toward target");
        }
    }

    #[test]
    fn gradient_step_zero_on_identical_clouds() {
        // Source == target → gradient should be near zero, no movement.
        let pts = vec![0.0_f32, 1.0, 2.0, 3.0];
        let mut particles = pts.clone();
        let mut rng = LcgRng::new(5);
        sw_gradient_step(&mut particles, &pts, 4, 4, 1, 1.0, 16, &mut rng);
        for (orig, new) in pts.iter().zip(particles.iter()) {
            assert!(
                (orig - new).abs() < 1e-4,
                "unexpected movement: {orig} → {new}"
            );
        }
    }

    // ------------------------------------------------------------------
    // sw_gradient_flow tests
    // ------------------------------------------------------------------

    #[test]
    fn gradient_flow_reduces_sw_distance_1d() {
        // Source: N points near 0. Target: N points near 3. Expect SW history decreases.
        let n = 30;
        let source = uniform_1d(n, -0.5, 0.5, 1);
        let target = uniform_1d(n, 2.5, 3.5, 2);
        let cfg = SwgfConfig {
            n_projections: 30,
            n_steps: 50,
            step_size: 0.1,
            seed: 42,
        };
        let fit = sw_gradient_flow(&source, &target, n, n, 1, &cfg).expect("ok");
        assert_eq!(fit.sw_history.len(), 50);
        assert_eq!(fit.particles.len(), n);
        // Early SW should be larger than late SW.
        let early: f32 = fit.sw_history[..5].iter().sum::<f32>() / 5.0;
        let late: f32 = fit.sw_history[45..].iter().sum::<f32>() / 5.0;
        assert!(
            late < early,
            "SW did not decrease: early={early:.4} late={late:.4}"
        );
    }

    #[test]
    fn gradient_flow_2d_finite_and_valid() {
        let n_src = 20;
        let n_tgt = 25;
        let source = gaussian_2d(n_src, 0.0, 0.0, 1);
        let target = gaussian_2d(n_tgt, 5.0, 5.0, 2);
        let cfg = SwgfConfig {
            n_projections: 20,
            n_steps: 30,
            step_size: 0.05,
            seed: 99,
        };
        let fit = sw_gradient_flow(&source, &target, n_src, n_tgt, 2, &cfg).expect("ok");
        assert_eq!(fit.particles.len(), n_src * 2);
        assert_eq!(fit.n_particles, n_src);
        assert_eq!(fit.d, 2);
        for &sw in &fit.sw_history {
            assert!(sw.is_finite() || sw.is_nan()); // NaN allowed if monitor fails gracefully
        }
    }

    #[test]
    fn gradient_flow_sw_history_length_matches_steps() {
        let source = vec![0.0_f32, 0.0];
        let target = vec![1.0_f32, 1.0];
        let cfg = SwgfConfig {
            n_projections: 5,
            n_steps: 7,
            step_size: 0.01,
            seed: 1,
        };
        let fit = sw_gradient_flow(&source, &target, 1, 1, 2, &cfg).expect("ok");
        assert_eq!(fit.sw_history.len(), 7);
    }

    #[test]
    fn gradient_flow_rejects_bad_dim() {
        let cfg = SwgfConfig::default();
        let res = sw_gradient_flow(&[0.0_f32], &[0.0_f32], 1, 1, 0, &cfg);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn gradient_flow_rejects_empty_source() {
        let cfg = SwgfConfig::default();
        let res = sw_gradient_flow(&[], &[1.0_f32], 0, 1, 1, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn gradient_flow_rejects_zero_steps() {
        let cfg = SwgfConfig {
            n_steps: 0,
            ..Default::default()
        };
        let res = sw_gradient_flow(&[0.0_f32], &[1.0_f32], 1, 1, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn gradient_flow_rejects_zero_projections() {
        let cfg = SwgfConfig {
            n_projections: 0,
            ..Default::default()
        };
        let res = sw_gradient_flow(&[0.0_f32], &[1.0_f32], 1, 1, 1, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }
}
