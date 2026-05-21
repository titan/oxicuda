//! Stein Variational Gradient Descent (SVGD).
//!
//! Particle-based variational inference via the Stein operator:
//! iteratively transports a set of particles {xᵢ} toward a target distribution
//! p(x) by combining a score-function descent term with a kernel-induced
//! repulsion term that prevents particle collapse (Liu & Wang, NeurIPS 2016).

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Config / Result ──────────────────────────────────────────────────────────

/// Configuration for Stein Variational Gradient Descent.
#[derive(Debug, Clone)]
pub struct SvgdConfig {
    /// Number of particles (≥ 2).
    pub n_particles: usize,
    /// Dimension of the parameter space (≥ 1).
    pub dim: usize,
    /// Step size ε > 0.
    pub step_size: f32,
    /// Bandwidth h for the RBF kernel.  `None` → median heuristic each iteration.
    pub bandwidth: Option<f32>,
    /// Number of iterations (≥ 1).
    pub n_iter: usize,
}

/// Output of a completed SVGD run.
#[derive(Debug, Clone)]
pub struct SvgdResult {
    /// Final particle positions, `n_particles × dim`, row-major.
    pub particles: Vec<f32>,
    /// Bandwidth used in the last iteration.
    pub final_bandwidth: f32,
    /// Number of iterations that were run.
    pub n_iter: usize,
}

// ─── Svgd ────────────────────────────────────────────────────────────────────

/// Stein Variational Gradient Descent runner.
///
/// All computation is deterministic given a fixed `init_particles` slice and
/// a fixed `cfg.bandwidth`.  The `rng` parameter is accepted for future
/// extensions (e.g. stochastic mini-batch particle sub-sampling) but is not
/// consumed when the bandwidth is fixed.
pub struct Svgd;

impl Svgd {
    // ── Public kernel helpers ─────────────────────────────────────────────

    /// RBF kernel: k(x, y) = exp(−‖x − y‖² / (2 h²)).
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != y.len()`.
    /// - `NonPositiveSigma` if `h ≤ 0`.
    pub fn rbf_kernel(x: &[f32], y: &[f32], h: f32) -> BayesResult<f32> {
        if x.len() != y.len() {
            return Err(BayesError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        if h <= 0.0 {
            return Err(BayesError::NonPositiveSigma);
        }
        let sq_dist: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi) * (xi - yi))
            .sum();
        Ok((-sq_dist / (2.0 * h * h)).exp())
    }

    /// Gradient of the RBF kernel with respect to x:
    /// ∇_x k(x, y) = −(x − y) / h² · k(x, y).
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != y.len()`.
    /// - `NonPositiveSigma` if `h ≤ 0`.
    pub fn rbf_kernel_grad(x: &[f32], y: &[f32], h: f32) -> BayesResult<Vec<f32>> {
        if x.len() != y.len() {
            return Err(BayesError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        if h <= 0.0 {
            return Err(BayesError::NonPositiveSigma);
        }
        let k_val = Self::rbf_kernel(x, y, h)?;
        let h_sq = h * h;
        let grad: Vec<f32> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| -(xi - yi) / h_sq * k_val)
            .collect();
        Ok(grad)
    }

    /// Median-heuristic bandwidth from the current particle positions.
    ///
    /// Computes all n*(n−1)/2 pairwise squared distances, takes their median,
    /// then sets h = sqrt(median / (2 ln n)).
    /// Falls back to h = 1.0 when n ≤ 2 or the median distance is ≤ 0.
    ///
    /// # Errors
    /// - `EmptyInputs` if `n == 0` or `dim == 0`.
    /// - `DimensionMismatch` if `particles.len() != n * dim`.
    pub fn median_bandwidth(particles: &[f32], n: usize, dim: usize) -> BayesResult<f32> {
        if n == 0 || dim == 0 {
            return Err(BayesError::EmptyInputs);
        }
        if particles.len() != n * dim {
            return Err(BayesError::DimensionMismatch {
                expected: n * dim,
                got: particles.len(),
            });
        }
        // Fallback: too few particles to compute a meaningful median
        if n <= 2 {
            return Ok(1.0);
        }
        // Collect all pairwise squared distances (i < j)
        let n_pairs = n * (n - 1) / 2;
        let mut sq_dists = Vec::with_capacity(n_pairs);
        for i in 0..n {
            let xi = &particles[i * dim..(i + 1) * dim];
            for j in (i + 1)..n {
                let xj = &particles[j * dim..(j + 1) * dim];
                let d2: f32 = xi
                    .iter()
                    .zip(xj.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                sq_dists.push(d2);
            }
        }
        // Sort and take the median
        sq_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sq_dists.len();
        let median_dist = if len % 2 == 0 {
            (sq_dists[len / 2 - 1] + sq_dists[len / 2]) * 0.5
        } else {
            sq_dists[len / 2]
        };
        if median_dist <= 0.0 {
            return Ok(1.0);
        }
        let h = (median_dist / (2.0 * (n as f32).ln())).sqrt();
        Ok(h)
    }

    // ── Main algorithm ────────────────────────────────────────────────────

    /// Run SVGD for the given configuration, returning the final particle positions.
    ///
    /// `log_prob_grad` receives a particle slice of length `dim` and returns the
    /// gradient of log p(x) with respect to x as a `Vec<f32>` of length `dim`.
    ///
    /// # Errors
    /// - `InvalidPriorVariance` if `n_particles < 2`, `dim == 0`, `step_size ≤ 0`,
    ///   or `n_iter == 0`.
    /// - `DimensionMismatch` if `init_particles.len() != n_particles * dim`.
    pub fn run<F>(
        cfg: &SvgdConfig,
        init_particles: &[f32],
        log_prob_grad: F,
        _rng: &mut LcgRng,
    ) -> BayesResult<SvgdResult>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        // ── Validation ────────────────────────────────────────────────────
        if cfg.n_particles < 2 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if cfg.dim == 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if cfg.step_size <= 0.0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if cfg.n_iter == 0 {
            return Err(BayesError::InvalidPriorVariance);
        }
        if init_particles.len() != cfg.n_particles * cfg.dim {
            return Err(BayesError::DimensionMismatch {
                expected: cfg.n_particles * cfg.dim,
                got: init_particles.len(),
            });
        }

        let n = cfg.n_particles;
        let d = cfg.dim;

        // Working copy of particles (n × d, row-major)
        let mut particles = init_particles.to_vec();

        // Temporary flat buffer for all Φ*(xⱼ) updates before applying them
        let mut phi_buf = vec![0.0_f32; n * d];

        let mut final_bandwidth = cfg.bandwidth.unwrap_or(1.0);

        for _ in 0..cfg.n_iter {
            // ── Step (a): compute bandwidth ───────────────────────────────
            let h = match cfg.bandwidth {
                Some(bw) => bw,
                None => Self::median_bandwidth(&particles, n, d)?,
            };
            final_bandwidth = h;

            // ── Step (b): compute Φ*(xⱼ) for each particle j ─────────────
            // Φ*(xⱼ) = (1/n) Σᵢ [ k(xᵢ, xⱼ) · ∇log p(xᵢ) + ∇_{xᵢ} k(xᵢ, xⱼ) ]
            for j in 0..n {
                let phi_j = &mut phi_buf[j * d..(j + 1) * d];
                for elem in phi_j.iter_mut() {
                    *elem = 0.0;
                }
                for i in 0..n {
                    let xi = &particles[i * d..(i + 1) * d];
                    let xj = &particles[j * d..(j + 1) * d];

                    let k_val = Self::rbf_kernel(xi, xj, h)?;
                    let score_i = log_prob_grad(xi); // ∇log p(xᵢ), length d
                    let grad_k_i = Self::rbf_kernel_grad(xi, xj, h)?;

                    for elem_d in 0..d {
                        phi_buf[j * d + elem_d] += k_val * score_i[elem_d] + grad_k_i[elem_d];
                    }
                }
                // Divide by n
                for elem_d in 0..d {
                    phi_buf[j * d + elem_d] /= n as f32;
                }
            }

            // ── Step (c): update all particles simultaneously ─────────────
            for j in 0..n {
                for elem_d in 0..d {
                    particles[j * d + elem_d] += cfg.step_size * phi_buf[j * d + elem_d];
                }
            }
        }

        Ok(SvgdResult {
            particles,
            final_bandwidth,
            n_iter: cfg.n_iter,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RBF kernel ─────────────────────────────────────────────────────────

    #[test]
    fn rbf_kernel_same_point() {
        // k(x, x) must equal 1.0 for any x and h
        let x = vec![1.0_f32, -2.0, 3.5];
        let k = Svgd::rbf_kernel(&x, &x, 1.5).expect("rbf_kernel must succeed");
        assert!((k - 1.0).abs() < 1e-6, "expected 1.0, got {k}");
    }

    #[test]
    fn rbf_kernel_far_points() {
        // When ||x - y|| >> h, k → 0
        let x = vec![0.0_f32; 4];
        let y = vec![100.0_f32; 4];
        let k = Svgd::rbf_kernel(&x, &y, 1.0).expect("rbf_kernel must succeed");
        assert!(k < 1e-30, "expected near-zero kernel, got {k}");
    }

    #[test]
    fn rbf_kernel_grad_same_point() {
        // ∇_x k(x, x) = 0 since (x − y) = 0
        let x = vec![1.0_f32, 2.0, -1.5];
        let grad = Svgd::rbf_kernel_grad(&x, &x, 2.0).expect("rbf_kernel_grad must succeed");
        for &g in &grad {
            assert!(g.abs() < 1e-7, "expected zero gradient, got {g}");
        }
    }

    #[test]
    fn rbf_kernel_grad_len() {
        let x = vec![1.0_f32; 5];
        let y = vec![0.0_f32; 5];
        let grad = Svgd::rbf_kernel_grad(&x, &y, 1.0).expect("rbf_kernel_grad must succeed");
        assert_eq!(grad.len(), 5);
    }

    #[test]
    fn rbf_kernel_grad_sign() {
        // When x[d] > y[d], ∇_x k points from y toward x → gradient < 0
        // because ∇_x k = -(x-y)/h^2 * k(x,y), and (x-y) > 0 → negative gradient
        let x = vec![2.0_f32];
        let y = vec![0.0_f32];
        let grad = Svgd::rbf_kernel_grad(&x, &y, 1.0).expect("rbf_kernel_grad must succeed");
        assert!(
            grad[0] < 0.0,
            "expected negative gradient component when x > y, got {}",
            grad[0]
        );
    }

    // ── Median bandwidth ───────────────────────────────────────────────────

    #[test]
    fn median_bandwidth_two_points() {
        // n ≤ 2 → fallback = 1.0
        let particles = vec![0.0_f32, 1.0, 5.0, 6.0];
        let h = Svgd::median_bandwidth(&particles, 2, 2).expect("median_bandwidth must succeed");
        assert!((h - 1.0).abs() < 1e-6, "expected fallback 1.0, got {h}");
    }

    #[test]
    fn median_bandwidth_positive() {
        // n ≥ 3 and particles spread out → h > 0
        let particles = vec![
            0.0_f32, 0.0, // particle 0
            1.0, 0.0, // particle 1
            0.0, 1.0, // particle 2
        ];
        let h = Svgd::median_bandwidth(&particles, 3, 2).expect("median_bandwidth must succeed");
        assert!(h > 0.0, "expected positive bandwidth, got {h}");
    }

    // ── Run ────────────────────────────────────────────────────────────────

    #[test]
    fn particles_move() {
        let mut rng = LcgRng::new(42);
        let cfg = SvgdConfig {
            n_particles: 3,
            dim: 2,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        let init = vec![1.0_f32, 2.0, -1.0, 0.5, 3.0, -2.0];
        // Non-zero score so particles are pushed
        let result =
            Svgd::run(&cfg, &init, |_x| vec![1.0_f32, 1.0], &mut rng).expect("run must succeed");
        // Particles must have changed
        let changed = result
            .particles
            .iter()
            .zip(init.iter())
            .any(|(&p, &q)| (p - q).abs() > 1e-9);
        assert!(changed, "particles must move after one iteration");
    }

    #[test]
    fn zero_score_particles_still_repel() {
        // With zero score gradient, only the kernel-gradient (repulsion) term drives updates.
        // Two particles that start slightly apart should move further apart.
        let mut rng = LcgRng::new(7);
        let cfg = SvgdConfig {
            n_particles: 2,
            dim: 1,
            step_size: 0.1,
            bandwidth: Some(1.0),
            n_iter: 5,
        };
        // Particles start 0.1 apart — kernel grad is non-zero, so repulsion applies
        let init = vec![-0.05_f32, 0.05_f32];
        let result =
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32], &mut rng).expect("run must succeed");
        // After repulsion, the gap should have grown
        let p0 = result.particles[0];
        let p1 = result.particles[1];
        let final_gap = (p0 - p1).abs();
        let init_gap = 0.1_f32;
        assert!(
            final_gap > init_gap,
            "repulsion should widen the gap: initial={init_gap}, final={final_gap}"
        );
    }

    #[test]
    fn gaussian_target_particles_spread() {
        // Target = N(0, 1); score = -x.
        // Start particles at slightly different positions near the origin;
        // repulsion + score gradient should spread them out over time.
        let mut rng = LcgRng::new(13);
        let n = 5;
        let cfg = SvgdConfig {
            n_particles: n,
            dim: 1,
            step_size: 0.05,
            bandwidth: Some(0.5),
            n_iter: 50,
        };
        // Slightly jittered starting positions so kernel gradients are non-zero
        let init = vec![-0.04_f32, -0.02, 0.0, 0.02, 0.04];
        let result = Svgd::run(&cfg, &init, |x| vec![-x[0]], &mut rng).expect("run must succeed");
        // Compute variance of final particle positions
        let mean: f32 = result.particles.iter().sum::<f32>() / n as f32;
        let var: f32 = result
            .particles
            .iter()
            .map(|&p| (p - mean) * (p - mean))
            .sum::<f32>()
            / n as f32;
        assert!(
            var > 1e-4,
            "particles should spread from near the origin, variance = {var}"
        );
    }

    #[test]
    fn bandwidth_override() {
        let mut rng = LcgRng::new(99);
        let cfg = SvgdConfig {
            n_particles: 3,
            dim: 1,
            step_size: 0.01,
            bandwidth: Some(2.0),
            n_iter: 2,
        };
        let init = vec![0.0_f32, 1.0, -1.0];
        let result =
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32], &mut rng).expect("run must succeed");
        assert!(
            (result.final_bandwidth - 2.0).abs() < 1e-6,
            "final_bandwidth must equal cfg.bandwidth = 2.0, got {}",
            result.final_bandwidth
        );
    }

    #[test]
    fn output_shape() {
        let mut rng = LcgRng::new(1);
        let n_particles = 4;
        let dim = 3;
        let cfg = SvgdConfig {
            n_particles,
            dim,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        let init = vec![0.0_f32; n_particles * dim];
        let result =
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32; dim], &mut rng).expect("run must succeed");
        assert_eq!(
            result.particles.len(),
            n_particles * dim,
            "output shape must be n_particles × dim"
        );
    }

    #[test]
    fn n_iter_preserved() {
        let mut rng = LcgRng::new(2);
        let cfg = SvgdConfig {
            n_particles: 2,
            dim: 1,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 7,
        };
        let init = vec![0.0_f32, 1.0];
        let result =
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32], &mut rng).expect("run must succeed");
        assert_eq!(result.n_iter, 7);
    }

    // ── Error cases ────────────────────────────────────────────────────────

    #[test]
    fn err_n_particles_lt_2() {
        let mut rng = LcgRng::new(0);
        let cfg = SvgdConfig {
            n_particles: 1,
            dim: 2,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        let init = vec![0.0_f32, 0.0];
        assert!(
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32; 2], &mut rng).is_err(),
            "n_particles < 2 must return Err"
        );
    }

    #[test]
    fn err_dim_zero() {
        let mut rng = LcgRng::new(0);
        let cfg = SvgdConfig {
            n_particles: 3,
            dim: 0,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        let init: Vec<f32> = vec![];
        assert!(
            Svgd::run(&cfg, &init, |_x| vec![], &mut rng).is_err(),
            "dim == 0 must return Err"
        );
    }

    #[test]
    fn err_step_size_zero() {
        let mut rng = LcgRng::new(0);
        let cfg = SvgdConfig {
            n_particles: 2,
            dim: 1,
            step_size: 0.0,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        let init = vec![0.0_f32, 1.0];
        assert!(
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32], &mut rng).is_err(),
            "step_size == 0 must return Err"
        );
    }

    #[test]
    fn err_init_len_mismatch() {
        let mut rng = LcgRng::new(0);
        let cfg = SvgdConfig {
            n_particles: 3,
            dim: 2,
            step_size: 0.01,
            bandwidth: Some(1.0),
            n_iter: 1,
        };
        // Correct length would be 6; pass 4 instead
        let init = vec![0.0_f32; 4];
        assert!(
            Svgd::run(&cfg, &init, |_x| vec![0.0_f32; 2], &mut rng).is_err(),
            "init_particles length mismatch must return Err"
        );
    }

    #[test]
    fn deterministic() {
        // Same config + fixed bandwidth → same result regardless of rng state
        let make_result = || {
            let mut rng = LcgRng::new(42);
            let cfg = SvgdConfig {
                n_particles: 3,
                dim: 2,
                step_size: 0.05,
                bandwidth: Some(1.0),
                n_iter: 10,
            };
            let init = vec![1.0_f32, 0.0, -1.0, 0.5, 0.0, -0.5];
            Svgd::run(&cfg, &init, |x| vec![-x[0], -x[1]], &mut rng).expect("run must succeed")
        };
        let r1 = make_result();
        let r2 = make_result();
        for (&a, &b) in r1.particles.iter().zip(r2.particles.iter()) {
            assert!((a - b).abs() < 1e-9, "determinism violated: {a} != {b}");
        }
    }
}
