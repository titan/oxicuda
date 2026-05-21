//! JKO proximal gradient flow via particle approximation (Blob method).
//!
//! Implements the Jordan-Kinderlehrer-Otto (JKO 1998) Wasserstein gradient
//! flow as a *particle method* following the Blob approach of Carrillo et al.
//! (2019). Instead of discretising the density on a fixed grid, we represent
//! the evolving measure as a cloud of equally-weighted particles
//! `{x_1, …, x_n} ⊂ ℝ^d`. The Wasserstein gradient flow of a functional `F`
//! is approximated by the particle ODE:
//!
//! ```text
//! ẋ_i = −∇V(x_i) − ∇U(x_i; {x_j})
//! ```
//!
//! where
//! - `∇V(x_i)` is the external potential gradient (supplied by the caller), and
//! - `∇U(x_i; {x_j}) = Σ_j w_j ∇K(x_i − x_j)` with a Gaussian interaction kernel
//!   `K(r) = −σ exp(−‖r‖² / h)` (repulsive: pushes particles apart like entropy).
//!
//! The bandwidth `h` is set adaptively as the median pairwise distance squared
//! (Silverman's rule) when `n > 1`.
//!
//! Each JKO step advances the particles by one explicit Euler step:
//! ```text
//! x_i ← x_i − τ · ∇V(x_i) − τ · ∇U(x_i; μ)
//! ```
//!
//! The `jko_wasserstein_distance` function uses the sliced-Wasserstein-1 proxy
//! to give a cheap estimate of how far the current particle cloud is from a
//! target distribution.

use crate::error::{OtError, OtResult};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the JKO particle gradient flow.
#[derive(Debug, Clone)]
pub struct JkoStepConfig {
    /// JKO time-step τ > 0.
    pub tau: f64,
    /// Number of equally-weighted particles.
    pub n_particles: usize,
    /// Total number of JKO steps to perform in `jko_run`.
    pub n_jko_steps: usize,
    /// Sinkhorn regularisation (reserved for future W2-coupling schemes).
    pub reg: f64,
    /// Explicit Euler sub-step size (usually equals τ; allows smaller sub-steps).
    pub step_size: f64,
}

impl Default for JkoStepConfig {
    fn default() -> Self {
        Self {
            tau: 0.05,
            n_particles: 50,
            n_jko_steps: 20,
            reg: 0.1,
            step_size: 0.05,
        }
    }
}

/// State of the JKO particle system.
#[derive(Debug, Clone)]
pub struct JkoState {
    /// Particle positions, shape `[n_particles × d]` row-major.
    pub particles: Vec<f64>,
    /// Particle weights (uniform: all equal to `1/n_particles`).
    pub weights: Vec<f64>,
    /// Number of particles.
    pub n_particles: usize,
    /// Ambient dimension.
    pub d: usize,
    /// Current simulation time `t = n_steps_completed * tau`.
    pub time: f64,
    /// Per-step energy values (total potential + repulsion), length = steps done.
    pub energy_history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_jko_cfg(n: usize, d: usize, cfg: &JkoStepConfig) -> OtResult<()> {
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if cfg.tau <= 0.0 {
        return Err(OtError::BadTau {
            tau: cfg.tau as f32,
        });
    }
    if cfg.step_size <= 0.0 {
        return Err(OtError::BadTau {
            tau: cfg.step_size as f32,
        });
    }
    if cfg.n_jko_steps == 0 {
        return Err(OtError::BadCount {
            got: cfg.n_jko_steps,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared L2 distance between two d-dimensional points.
#[inline]
fn sq_dist(xi: &[f64], xj: &[f64]) -> f64 {
    xi.iter()
        .zip(xj.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Adaptively estimate the kernel bandwidth `h` from particle positions.
///
/// Uses median pairwise squared distance (Silverman-style); falls back to 1.0
/// when `n <= 1` or when all particles are identical.
fn adaptive_bandwidth(particles: &[f64], n: usize, d: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    // Collect a subsample of squared pairwise distances (at most 500 pairs)
    // to keep O(n²) work manageable.
    let max_pairs = 500_usize;
    let mut dists: Vec<f64> = Vec::with_capacity(max_pairs);
    let step = (n * (n - 1) / 2).max(1);
    let skip = (step / max_pairs).max(1);
    let mut pair_idx = 0_usize;
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            if pair_idx.is_multiple_of(skip) {
                let xi = &particles[i * d..(i + 1) * d];
                let xj = &particles[j * d..(j + 1) * d];
                dists.push(sq_dist(xi, xj));
                if dists.len() >= max_pairs {
                    break 'outer;
                }
            }
            pair_idx += 1;
        }
    }
    if dists.is_empty() {
        return 1.0;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_sq = dists[dists.len() / 2];
    if median_sq < 1e-30 { 1.0 } else { median_sq }
}

/// Compute the repulsion gradient for particle `i` from all other particles,
/// using Gaussian kernel `K(r) = -σ · exp(-‖r‖² / h)` (negative = repulsive).
///
/// The gradient of `-exp(-‖r‖²/h)` w.r.t. `x_i` is `(2/h)(x_i - x_j) exp(-‖r‖²/h)`.
///
/// So `∇_{x_i} U = (1/n) Σ_{j≠i} w_j · (2/h)(x_i - x_j) exp(-‖x_i-x_j‖²/h)`.
fn repulsion_grad(particles: &[f64], i: usize, n: usize, d: usize, h: f64, sigma: f64) -> Vec<f64> {
    let mut grad = vec![0.0_f64; d];
    let xi = &particles[i * d..(i + 1) * d];
    let w = 1.0 / n as f64;
    let two_over_h = 2.0 / h;

    for j in 0..n {
        if j == i {
            continue;
        }
        let xj = &particles[j * d..(j + 1) * d];
        let r2 = sq_dist(xi, xj);
        // Clamp exponent to avoid overflow
        let exp_val = if -r2 / h < -500.0 {
            0.0
        } else {
            (-r2 / h).exp()
        };
        let coeff = w * sigma * two_over_h * exp_val;
        for dim in 0..d {
            grad[dim] += coeff * (xi[dim] - xj[dim]);
        }
    }
    grad
}

/// Compute the total instantaneous energy (external potential + repulsion self-energy).
fn compute_energy<F>(
    particles: &[f64],
    n: usize,
    d: usize,
    h: f64,
    sigma: f64,
    potential_grad: &F,
) -> f64
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let w = 1.0 / n as f64;
    let mut energy = 0.0_f64;

    // External potential energy: Σ_i w_i ⟨∇V(x_i), x_i⟩ ≈ Σ_i V(x_i)
    // We approximate using a finite-difference magnitude of ∇V
    for i in 0..n {
        let xi = &particles[i * d..(i + 1) * d];
        let gv = potential_grad(xi);
        let gv_norm_sq: f64 = gv.iter().map(|&g| g * g).sum();
        energy += w * gv_norm_sq.sqrt();
    }

    // Repulsion self-energy: (1/2) Σ_{i≠j} w² K(x_i - x_j)
    for i in 0..n {
        for j in (i + 1)..n {
            let xi = &particles[i * d..(i + 1) * d];
            let xj = &particles[j * d..(j + 1) * d];
            let r2 = sq_dist(xi, xj);
            let k_val = if -r2 / h < -500.0 {
                0.0
            } else {
                -sigma * (-r2 / h).exp()
            };
            energy -= w * w * k_val; // positive because K is negative
        }
    }

    energy
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise a [`JkoState`] from a flat `n × d` particle array.
///
/// # Errors
///
/// Returns an error if `x0.len() != n * d` or if `n == 0` or `d == 0`.
pub fn jko_init(x0: &[f64], n: usize, d: usize) -> OtResult<JkoState> {
    if n == 0 || d == 0 {
        return Err(OtError::EmptyInput);
    }
    if x0.len() != n * d {
        return Err(OtError::IncompatibleLength {
            a: x0.len(),
            b: n * d,
        });
    }
    let w = 1.0 / n as f64;
    Ok(JkoState {
        particles: x0.to_vec(),
        weights: vec![w; n],
        n_particles: n,
        d,
        time: 0.0,
        energy_history: Vec::new(),
    })
}

/// Advance one JKO step using the Blob particle method.
///
/// Moves particles by the explicit Euler update:
/// ```text
/// x_i ← x_i − step_size · (∇V(x_i) + ∇U(x_i; μ))
/// ```
///
/// The repulsion kernel bandwidth `h` is computed adaptively from the current
/// particle cloud each step.
///
/// # Parameters
///
/// - `state`: mutable reference to the current particle state (updated in-place).
/// - `potential_grad`: closure taking a d-slice `x` and returning `∇V(x)` as a
///   `Vec<f64>` of length `d`.
/// - `cfg`: solver configuration (only `step_size` and `tau` are used here).
///
/// # Errors
///
/// Returns an error if the state has zero particles or if the potential gradient
/// returns a slice of the wrong length.
pub fn jko_step<F>(state: &mut JkoState, potential_grad: F, cfg: &JkoStepConfig) -> OtResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = state.n_particles;
    let d = state.d;
    if n == 0 || d == 0 {
        return Err(OtError::EmptyInput);
    }

    let h = adaptive_bandwidth(&state.particles, n, d);
    // Repulsion amplitude: σ = reg / (n * bandwidth) — scales with regularisation
    let sigma = cfg.reg / (n as f64 * h.sqrt().max(1e-12));

    // Pre-compute the full gradient for each particle before any update
    // (synchronous Euler step: all positions read from old state)
    let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &state.particles[i * d..(i + 1) * d];
        let gv = potential_grad(xi);
        if gv.len() != d {
            return Err(OtError::Internal {
                msg: format!("potential_grad returned length {}, expected {d}", gv.len()),
            });
        }
        let gu = repulsion_grad(&state.particles, i, n, d, h, sigma);
        let mut g_total = vec![0.0_f64; d];
        for dim in 0..d {
            g_total[dim] = gv[dim] + gu[dim];
        }
        grads.push(g_total);
    }

    // Record energy before the step
    let energy = compute_energy(&state.particles, n, d, h, sigma, &potential_grad);
    state.energy_history.push(energy);

    // Apply the Euler update
    for (i, grad_i) in grads.iter().enumerate() {
        for (dim, &gi) in grad_i.iter().enumerate() {
            state.particles[i * d + dim] -= cfg.step_size * gi;
        }
    }

    state.time += cfg.tau;
    Ok(())
}

/// Run `cfg.n_jko_steps` JKO particle steps from initial positions `x0`.
///
/// # Parameters
///
/// - `x0`: flat `n × d` row-major array of initial particle positions.
/// - `n`: number of particles.
/// - `d`: ambient dimension.
/// - `potential_grad`: closure `∇V(x) → ℝ^d` (same interface as in `jko_step`).
/// - `cfg`: solver configuration.
///
/// # Returns
///
/// The final [`JkoState`] after all steps.
///
/// # Errors
///
/// Returns errors from `jko_init` or any `jko_step` call.
pub fn jko_run<F>(
    x0: &[f64],
    n: usize,
    d: usize,
    potential_grad: F,
    cfg: &JkoStepConfig,
) -> OtResult<JkoState>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_jko_cfg(n, d, cfg)?;
    let mut state = jko_init(x0, n, d)?;
    for _ in 0..cfg.n_jko_steps {
        jko_step(&mut state, &potential_grad, cfg)?;
    }
    Ok(state)
}

/// Estimate the Wasserstein-1-like distance between the current particle cloud
/// and a target point set using a sliced approach.
///
/// We project both sets onto `n_proj = 20` random unit directions drawn
/// deterministically from a fixed Sobol-like pattern, sort the projected
/// coordinates, and compute the mean L1 distance between the sorted projections.
/// This gives a biased but cheap proxy for W1.
///
/// # Parameters
///
/// - `state`: current JKO state with particles.
/// - `target`: flat `n_target × d` row-major array of target positions.
/// - `n_target`: number of target points.
///
/// # Returns
///
/// A non-negative real number approximating W1(current, target).
///
/// # Errors
///
/// Returns an error if dimensions are inconsistent.
pub fn jko_wasserstein_distance(
    state: &JkoState,
    target: &[f64],
    n_target: usize,
) -> OtResult<f64> {
    let n = state.n_particles;
    let d = state.d;
    if n == 0 || n_target == 0 {
        return Err(OtError::EmptyInput);
    }
    if target.len() != n_target * d {
        return Err(OtError::IncompatibleLength {
            a: target.len(),
            b: n_target * d,
        });
    }

    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }

    let n_proj = 20_usize;
    // Generate pseudo-random unit directions using a deterministic LCG-like sequence
    // seeded to 0 for reproducibility (no external RNG required).
    let directions = gen_unit_directions(d, n_proj);

    let mut total_w1 = 0.0_f64;

    for dir in &directions {
        // Project source particles
        let mut src_proj: Vec<f64> = (0..n)
            .map(|i| {
                dir.iter()
                    .zip(state.particles[i * d..(i + 1) * d].iter())
                    .map(|(dj, xij)| dj * xij)
                    .sum::<f64>()
            })
            .collect();

        // Project target particles
        let mut tgt_proj: Vec<f64> = (0..n_target)
            .map(|i| {
                dir.iter()
                    .zip(target[i * d..(i + 1) * d].iter())
                    .map(|(dj, xij)| dj * xij)
                    .sum::<f64>()
            })
            .collect();

        src_proj.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tgt_proj.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Interpolate to the same length for the 1-D W1 comparison
        let w1 = sliced_w1_sorted(&src_proj, &tgt_proj);
        total_w1 += w1;
    }

    Ok(total_w1 / n_proj as f64)
}

/// Compute 1-D Wasserstein-1 between two sorted empirical CDFs of possibly
/// different lengths by linearly interpolating the quantile functions.
fn sliced_w1_sorted(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return 0.0;
    }

    // Evaluate the quantile functions at n_eval common quantile points
    let n_eval = na.max(nb).min(200);
    let mut dist = 0.0_f64;
    for k in 0..n_eval {
        let q = k as f64 / (n_eval - 1).max(1) as f64;
        let qa = quantile_sorted(a, q);
        let qb = quantile_sorted(b, q);
        dist += (qa - qb).abs();
    }
    dist / n_eval as f64
}

/// Linear-interpolation quantile for a sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    if hi >= n {
        return sorted[n - 1];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Generate `n_proj` pseudo-random unit directions in `ℝ^d` deterministically.
/// Uses a simple van-der-Corput-like sequence seeded at 0 for reproducibility.
fn gen_unit_directions(d: usize, n_proj: usize) -> Vec<Vec<f64>> {
    if d == 0 {
        return vec![];
    }
    // Use a simple LCG to generate coordinates, then normalise
    let mut state: u64 = 6_364_136_223_846_793_005_u64;
    let mut dirs = Vec::with_capacity(n_proj);
    for _ in 0..n_proj {
        let mut v = vec![0.0_f64; d];
        for x in v.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((state >> 33) as f64) / (u32::MAX as f64 + 1.0);
            // Box-Muller for standard normal
            let u2_raw = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state = u2_raw;
            let u2 = ((u2_raw >> 33) as f64) / (u32::MAX as f64 + 1.0);
            let u1 = (u * 0.9999 + 0.00005).max(1e-12);
            *x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        } else {
            v[0] = 1.0;
        }
        dirs.push(v);
    }
    dirs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_particles(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
            .collect()
    }

    #[test]
    fn jko_init_correct_shape() {
        let n = 10;
        let d = 2;
        let x0: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.1).collect();
        let state = jko_init(&x0, n, d).expect("ok");
        assert_eq!(state.n_particles, n);
        assert_eq!(state.d, d);
        assert_eq!(state.particles.len(), n * d);
        assert_eq!(state.weights.len(), n);
        assert!((state.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert_eq!(state.time, 0.0);
    }

    #[test]
    fn jko_init_error_on_empty() {
        let res = jko_init(&[], 0, 1);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn jko_init_error_on_shape_mismatch() {
        let x0 = vec![0.0_f64; 5];
        let res = jko_init(&x0, 3, 2); // 3*2=6 != 5
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn jko_step_advances_time() {
        let n = 8;
        let d = 1;
        let x0 = line_particles(n, -1.0, 1.0);
        let cfg = JkoStepConfig {
            tau: 0.01,
            n_particles: n,
            n_jko_steps: 5,
            reg: 0.1,
            step_size: 0.01,
        };
        let mut state = jko_init(&x0, n, d).expect("ok");
        // Identity potential: zero gradient everywhere
        jko_step(&mut state, |_x| vec![0.0; d], &cfg).expect("ok");
        assert!((state.time - cfg.tau).abs() < 1e-12, "time={}", state.time);
        assert_eq!(state.energy_history.len(), 1);
    }

    #[test]
    fn jko_step_moves_particles() {
        // Potential V(x) = x² → ∇V(x) = 2x pulls particles toward 0.
        let n = 6;
        let d = 1;
        let x0 = line_particles(n, 2.0, 3.0); // all particles far from 0
        let x0_clone = x0.clone();
        let cfg = JkoStepConfig {
            tau: 0.05,
            step_size: 0.05,
            n_particles: n,
            n_jko_steps: 10,
            reg: 0.1,
        };
        let mut state = jko_init(&x0, n, d).expect("ok");
        for _ in 0..10 {
            jko_step(&mut state, |x| vec![2.0 * x[0]], &cfg).expect("ok");
        }
        // Particles should have moved toward 0 (decreased in absolute value)
        let mean_old: f64 = x0_clone.iter().sum::<f64>() / n as f64;
        let mean_new: f64 = state.particles.iter().sum::<f64>() / n as f64;
        assert!(
            mean_new.abs() < mean_old.abs(),
            "mean_old={mean_old}, mean_new={mean_new}: particles didn't move toward 0"
        );
    }

    #[test]
    fn jko_run_completes_all_steps() {
        let n = 8;
        let d = 1;
        let x0 = line_particles(n, -1.0, 1.0);
        let cfg = JkoStepConfig {
            tau: 0.01,
            n_particles: n,
            n_jko_steps: 15,
            reg: 0.05,
            step_size: 0.01,
        };
        let state = jko_run(&x0, n, d, |_x| vec![0.0; d], &cfg).expect("ok");
        assert_eq!(state.energy_history.len(), 15);
        assert!((state.time - 15.0 * cfg.tau).abs() < 1e-10);
    }

    #[test]
    fn jko_run_particles_finite() {
        let n = 10;
        let d = 2;
        let x0: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.1).collect();
        let cfg = JkoStepConfig {
            tau: 0.02,
            n_particles: n,
            n_jko_steps: 5,
            reg: 0.1,
            step_size: 0.02,
        };
        let state = jko_run(&x0, n, d, |x| vec![x[0] * 0.1, x[1] * 0.1], &cfg).expect("ok");
        for &p in &state.particles {
            assert!(p.is_finite(), "particle coord is not finite: {p}");
        }
    }

    #[test]
    fn jko_run_error_on_zero_tau() {
        let x0 = vec![0.0_f64; 4];
        let cfg = JkoStepConfig {
            tau: 0.0,
            n_particles: 4,
            n_jko_steps: 5,
            reg: 0.1,
            step_size: 0.01,
        };
        let res = jko_run(&x0, 4, 1, |_x| vec![0.0], &cfg);
        assert!(matches!(res, Err(OtError::BadTau { .. })));
    }

    #[test]
    fn jko_run_error_on_zero_steps() {
        let x0 = vec![0.0_f64; 4];
        let cfg = JkoStepConfig {
            tau: 0.1,
            n_particles: 4,
            n_jko_steps: 0,
            reg: 0.1,
            step_size: 0.1,
        };
        let res = jko_run(&x0, 4, 1, |_x| vec![0.0], &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn jko_wasserstein_distance_self_is_small() {
        let n = 12;
        let d = 2;
        let x0: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.05).collect();
        let state = jko_init(&x0, n, d).expect("ok");
        let dist = jko_wasserstein_distance(&state, &x0, n).expect("ok");
        assert!(dist < 1e-6, "self-distance should be ~0, got {dist}");
    }

    #[test]
    fn jko_wasserstein_distance_increases_with_separation() {
        let n = 10;
        let d = 1;
        let x0 = line_particles(n, 0.0, 1.0);
        let target_close: Vec<f64> = line_particles(n, 0.0, 1.0)
            .iter()
            .map(|&x| x + 0.01)
            .collect();
        let target_far: Vec<f64> = line_particles(n, 0.0, 1.0)
            .iter()
            .map(|&x| x + 5.0)
            .collect();
        let state = jko_init(&x0, n, d).expect("ok");
        let dist_close = jko_wasserstein_distance(&state, &target_close, n).expect("ok");
        let dist_far = jko_wasserstein_distance(&state, &target_far, n).expect("ok");
        assert!(
            dist_far > dist_close,
            "far dist {dist_far} should exceed close dist {dist_close}"
        );
    }

    #[test]
    fn jko_wasserstein_distance_error_on_shape_mismatch() {
        let n = 5;
        let d = 2;
        let x0: Vec<f64> = vec![0.0; n * d];
        let state = jko_init(&x0, n, d).expect("ok");
        // Wrong target length
        let target = vec![0.0_f64; 7]; // 7 != n_target * d
        let res = jko_wasserstein_distance(&state, &target, 4);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn energy_history_all_finite() {
        let n = 8;
        let d = 1;
        let x0 = line_particles(n, -1.0, 1.0);
        let cfg = JkoStepConfig {
            tau: 0.05,
            n_particles: n,
            n_jko_steps: 10,
            reg: 0.1,
            step_size: 0.05,
        };
        // Harmonic potential
        let state = jko_run(&x0, n, d, |x| vec![x[0]], &cfg).expect("ok");
        for (t, &e) in state.energy_history.iter().enumerate() {
            assert!(e.is_finite(), "energy[{t}] = {e} is not finite");
        }
    }

    #[test]
    fn quadratic_potential_contracts_particles() {
        // V(x,y) = x²+y² → should contract the particle cloud toward origin
        let n = 12;
        let d = 2;
        let x0: Vec<f64> = (0..n)
            .flat_map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                vec![3.0 * theta.cos(), 3.0 * theta.sin()]
            })
            .collect();
        let cfg = JkoStepConfig {
            tau: 0.05,
            step_size: 0.05,
            n_particles: n,
            n_jko_steps: 20,
            reg: 0.5,
        };
        let mean_r_before: f64 = (0..n)
            .map(|i| {
                let x = x0[2 * i];
                let y = x0[2 * i + 1];
                (x * x + y * y).sqrt()
            })
            .sum::<f64>()
            / n as f64;

        let state = jko_run(&x0, n, d, |x| vec![2.0 * x[0], 2.0 * x[1]], &cfg).expect("ok");

        let mean_r_after: f64 = (0..n)
            .map(|i| {
                let x = state.particles[2 * i];
                let y = state.particles[2 * i + 1];
                (x * x + y * y).sqrt()
            })
            .sum::<f64>()
            / n as f64;

        assert!(
            mean_r_after < mean_r_before,
            "radius didn't shrink: before={mean_r_before}, after={mean_r_after}"
        );
    }
}
