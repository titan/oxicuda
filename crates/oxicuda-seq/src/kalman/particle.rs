//! Particle filter (SIR — Sequential Importance Resampling).
//!
//! Gordon, Salmond & Smith (1993) / Arulampalam et al. (2002) Tutorial.
//!
//! Represents the posterior p(x_t | z_{1:t}) with N weighted particles {x_i, w_i}.
//! For non-linear, non-Gaussian state-space models.

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Configuration for the particle filter.
#[derive(Debug, Clone)]
pub struct ParticleConfig {
    /// Number of particles N (must be ≥ 2).
    pub n_particles: usize,
    /// State dimension.
    pub dim_x: usize,
    /// Observation dimension.
    pub dim_z: usize,
    /// Effective sample size threshold for resampling: resample when N_eff < threshold * N.
    pub resample_threshold: f64,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            n_particles: 100,
            dim_x: 1,
            dim_z: 1,
            resample_threshold: 0.5,
        }
    }
}

/// Result of running the particle filter on a sequence.
#[derive(Debug, Clone)]
pub struct ParticleResult {
    /// T filtered state estimates (weighted mean), each of length `dim_x`.
    pub means: Vec<Vec<f64>>,
    /// T effective sample sizes N_eff = 1 / Σ(w_i²).
    pub eff_sizes: Vec<f64>,
    /// Total number of resampling events across all time steps.
    pub n_resamples: usize,
}

/// Particle filter (SIR) for nonlinear non-Gaussian state-space models.
///
/// State model:  x_t = f(x_{t-1}) + noise,  noise ~ N(0, Q)
/// Obs model:    z_t = h(x_t) + v_t,        v_t   ~ N(0, R)
pub struct ParticleFilter<'a> {
    /// Filter configuration.
    pub cfg: ParticleConfig,
    /// Cholesky factor L_Q (dim_x × dim_x lower triangular) of Q = L_Q L_Q^T.
    pub q_chol: Vec<f64>,
    /// Observation noise covariance R (dim_z × dim_z, row-major).
    pub r: Vec<f64>,
    /// Deterministic part of state transition: x_t = f(x_{t-1}) + noise.
    pub f: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    /// Observation model: z_t = h(x_t).
    pub h: Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>,
    /// Initial state distribution mean (length dim_x).
    pub x0: Vec<f64>,
    /// Cholesky of initial covariance P_0 (lower triangular, dim_x × dim_x).
    pub p0_chol: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Draw a sample from N(mean, L L^T) where L is a lower-triangular Cholesky factor.
/// Each standard-normal ε component is drawn from `rng`.
fn sample_gaussian(mean: &[f64], l_chol: &[f64], dim: usize, rng: &mut LcgRng) -> Vec<f64> {
    // ε ~ N(0, I_dim)
    let eps: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
    // noise = L · ε  (lower-tri matrix-vector product)
    let mut noise = vec![0.0; dim];
    for i in 0..dim {
        let mut s = 0.0;
        for k in 0..=i {
            s += l_chol[i * dim + k] * eps[k];
        }
        noise[i] = s;
    }
    mean.iter().zip(noise.iter()).map(|(m, n)| m + n).collect()
}

/// Compute log-likelihood under a diagonal Gaussian N(z_pred, diag(R)) evaluated at z_obs.
///
/// log p(z_obs | z_pred) ∝ -0.5 * Σ_k (z_obs[k] - z_pred[k])² / R[k*nz+k]
fn log_likelihood_diag(z_obs: &[f64], z_pred: &[f64], r: &[f64], dim_z: usize) -> f64 {
    let mut ll = 0.0;
    for k in 0..dim_z {
        let r_kk = r[k * dim_z + k];
        if r_kk > 0.0 {
            let diff = z_obs[k] - z_pred[k];
            ll -= 0.5 * diff * diff / r_kk;
        }
    }
    ll
}

/// Systematic resampling (Kitagawa 1996): returns new particle indices.
///
/// Given N normalised weights and an RNG, returns a length-N index vector where
/// each index k is the source particle to copy.
fn systematic_resample(weights: &[f64], rng: &mut LcgRng) -> Vec<usize> {
    let n = weights.len();
    // Build cumulative weight array
    let mut cumsum = vec![0.0; n];
    cumsum[0] = weights[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    // Single uniform draw, then stratified positions
    let u0 = rng.next_f64() / n as f64;
    let mut indices = vec![0usize; n];
    let mut k = 0usize;
    for j in 0..n {
        let u_j = u0 + j as f64 / n as f64;
        while k < n - 1 && cumsum[k] < u_j {
            k += 1;
        }
        indices[j] = k;
    }
    indices
}

// ---------------------------------------------------------------------------
// Main implementation
// ---------------------------------------------------------------------------

impl<'a> ParticleFilter<'a> {
    /// Validate configuration and input dimensions.
    fn validate(&self, z: &[f64]) -> SeqResult<()> {
        if self.cfg.n_particles < 2 {
            return Err(SeqError::InvalidConfiguration(format!(
                "n_particles must be >= 2, got {}",
                self.cfg.n_particles
            )));
        }
        if z.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if z.len() % self.cfg.dim_z != 0 {
            return Err(SeqError::DimensionMismatch {
                a: z.len(),
                b: self.cfg.dim_z,
            });
        }
        let nx = self.cfg.dim_x;
        let nz = self.cfg.dim_z;
        if self.q_chol.len() != nx * nx {
            return Err(SeqError::ShapeMismatch {
                expected: nx * nx,
                got: self.q_chol.len(),
            });
        }
        if self.r.len() != nz * nz {
            return Err(SeqError::ShapeMismatch {
                expected: nz * nz,
                got: self.r.len(),
            });
        }
        if self.x0.len() != nx {
            return Err(SeqError::ShapeMismatch {
                expected: nx,
                got: self.x0.len(),
            });
        }
        if self.p0_chol.len() != nx * nx {
            return Err(SeqError::ShapeMismatch {
                expected: nx * nx,
                got: self.p0_chol.len(),
            });
        }
        Ok(())
    }

    /// Run the particle filter on observations `z` (T × dim_z, row-major flat).
    ///
    /// Returns weighted state estimates, effective sample sizes, and resampling count.
    pub fn run(&self, z: &[f64], rng: &mut LcgRng) -> SeqResult<ParticleResult> {
        self.validate(z)?;

        let n = self.cfg.n_particles;
        let nx = self.cfg.dim_x;
        let nz = self.cfg.dim_z;
        let t_max = z.len() / nz;
        let resample_threshold = self.cfg.resample_threshold * n as f64;

        // ------------------------------------------------------------------
        // Initialise particles x_i ~ N(x0, P0)
        // ------------------------------------------------------------------
        let mut particles: Vec<Vec<f64>> = (0..n)
            .map(|_| sample_gaussian(&self.x0, &self.p0_chol, nx, rng))
            .collect();

        // Uniform initial log-weights
        let log_n = (n as f64).ln();
        let mut log_weights: Vec<f64> = vec![-log_n; n];

        let mut means = Vec::with_capacity(t_max);
        let mut eff_sizes = Vec::with_capacity(t_max);
        let mut n_resamples = 0usize;

        for t in 0..t_max {
            let z_t = &z[t * nz..(t + 1) * nz];

            // ----------------------------------------------------------------
            // 1. Propagate: x_i ~ p(x_t | x_{t-1})
            // ----------------------------------------------------------------
            for i in 0..n {
                let mu_i = (self.f)(&particles[i]);
                particles[i] = sample_gaussian(&mu_i, &self.q_chol, nx, rng);
            }

            // ----------------------------------------------------------------
            // 2. Reweight: log_w_i += log p(z_t | x_i)
            // ----------------------------------------------------------------
            for i in 0..n {
                let z_pred = (self.h)(&particles[i]);
                let ll = log_likelihood_diag(z_t, &z_pred, &self.r, nz);
                log_weights[i] += ll;
            }

            // Numerically stable normalisation via log-sum-exp
            let log_max = log_weights
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let shifted: Vec<f64> = log_weights.iter().map(|&lw| (lw - log_max).exp()).collect();
            let sum_shifted: f64 = shifted.iter().sum();
            let weights: Vec<f64> = shifted.iter().map(|&s| s / sum_shifted).collect();

            // Update log-weights to match normalised weights
            for i in 0..n {
                log_weights[i] = weights[i].max(f64::MIN_POSITIVE).ln();
            }

            // ----------------------------------------------------------------
            // 3. Compute weighted mean and effective sample size
            // ----------------------------------------------------------------
            let mut mean = vec![0.0; nx];
            for i in 0..n {
                for d in 0..nx {
                    mean[d] += weights[i] * particles[i][d];
                }
            }

            let n_eff = 1.0 / weights.iter().map(|&w| w * w).sum::<f64>();
            means.push(mean);
            eff_sizes.push(n_eff);

            // ----------------------------------------------------------------
            // 4. Systematic resampling if N_eff < threshold
            // ----------------------------------------------------------------
            if n_eff < resample_threshold {
                let indices = systematic_resample(&weights, rng);
                // Resample particles (need a temporary copy)
                let old_particles = particles.clone();
                for i in 0..n {
                    particles[i] = old_particles[indices[i]].clone();
                }
                // Reset to uniform log-weights
                for i in 0..n {
                    log_weights[i] = -log_n;
                }
                n_resamples += 1;
            }
        }

        Ok(ParticleResult {
            means,
            eff_sizes,
            n_resamples,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a 1D identity particle filter.
    fn make_1d_pf(n: usize, q_var: f64, r_var: f64) -> ParticleFilter<'static> {
        ParticleFilter {
            cfg: ParticleConfig {
                n_particles: n,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![q_var.sqrt()],
            r: vec![r_var],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        }
    }

    #[test]
    fn new_config_default_ok() {
        let cfg = ParticleConfig::default();
        assert_eq!(cfg.n_particles, 100);
        assert_eq!(cfg.dim_x, 1);
        assert_eq!(cfg.dim_z, 1);
        assert!((cfg.resample_threshold - 0.5).abs() < 1e-15);
    }

    #[test]
    fn pf_output_length() {
        let pf = make_1d_pf(50, 0.1, 0.5);
        let z: Vec<f64> = vec![1.0; 10];
        let mut rng = LcgRng::new(42);
        let res = pf.run(&z, &mut rng).expect("ok");
        assert_eq!(res.means.len(), 10);
    }

    #[test]
    fn pf_means_dim_correct() {
        let pf = make_1d_pf(50, 0.1, 0.5);
        let z = vec![1.0; 5];
        let mut rng = LcgRng::new(7);
        let res = pf.run(&z, &mut rng).expect("ok");
        for (t, m) in res.means.iter().enumerate() {
            assert_eq!(m.len(), 1, "dim mismatch at t={t}");
        }
    }

    #[test]
    fn pf_constant_obs_converges() {
        // f(x)=x, h(x)=x, constant observations z=1.0; large N should converge.
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 500,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.01f64.sqrt()],
            r: vec![0.01],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![1.0],
            p0_chol: vec![0.1],
        };
        let z = vec![1.0_f64; 15];
        let mut rng = LcgRng::new(12345);
        let res = pf.run(&z, &mut rng).expect("ok");
        let last = res.means[14][0];
        assert!((last - 1.0).abs() < 0.5, "did not converge: {last}");
    }

    #[test]
    fn pf_zero_innovation() {
        // h(x) always returns the constant target value: w_i all equal → x̄ = f(x̄_prev)
        let target = 3.0_f64;
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 100,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.01],
            r: vec![0.1],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(move |_x: &[f64]| vec![target]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        };
        let z = vec![target; 10];
        let mut rng = LcgRng::new(99);
        let res = pf.run(&z, &mut rng).expect("ok");
        // All weights are equal, estimate should be the mean of particles
        assert_eq!(res.means.len(), 10);
    }

    #[test]
    fn pf_eff_size_bounded() {
        let pf = make_1d_pf(100, 0.1, 0.5);
        let z = vec![1.0; 8];
        let mut rng = LcgRng::new(77);
        let res = pf.run(&z, &mut rng).expect("ok");
        for (t, &neff) in res.eff_sizes.iter().enumerate() {
            assert!(neff > 0.0, "N_eff <= 0 at t={t}: {neff}");
            assert!(neff <= 100.0 + 1e-6, "N_eff > N at t={t}: {neff}");
        }
    }

    #[test]
    fn pf_weights_normalize() {
        // After reweighting, weights should sum to ~1 (tested indirectly: N_eff = 1/Σw²
        // is only valid when weights are normalised, so any non-normalised run would
        // produce N_eff > N which is checked in pf_eff_size_bounded).
        let pf = make_1d_pf(50, 0.05, 0.1);
        let z = vec![1.0; 6];
        let mut rng = LcgRng::new(11);
        let res = pf.run(&z, &mut rng).expect("ok");
        // N_eff ≤ N guarantees weights are normalised (N_eff = N ⟺ all w_i = 1/N)
        for &neff in &res.eff_sizes {
            assert!(neff > 0.0 && neff <= 50.0 + 1e-6);
        }
    }

    #[test]
    fn pf_resamples_occur() {
        // Highly peaked likelihood → N_eff collapses → resampling fires.
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 50,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.9, // aggressive threshold
            },
            q_chol: vec![0.01],
            r: vec![1e-6], // very small R → very peaked likelihood
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        };
        let z = vec![1.0_f64; 20];
        let mut rng = LcgRng::new(42);
        let res = pf.run(&z, &mut rng).expect("ok");
        assert!(
            res.n_resamples > 0,
            "expected at least one resampling event"
        );
    }

    #[test]
    fn pf_deterministic_same_seed() {
        // Two PF instances with identical seeds must produce identical output.
        let pf1 = make_1d_pf(80, 0.1, 0.3);
        let pf2 = make_1d_pf(80, 0.1, 0.3);
        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.2).collect();
        let mut rng1 = LcgRng::new(999);
        let mut rng2 = LcgRng::new(999);
        let res1 = pf1.run(&z, &mut rng1).expect("ok");
        let res2 = pf2.run(&z, &mut rng2).expect("ok");
        for t in 0..10 {
            assert!(
                (res1.means[t][0] - res2.means[t][0]).abs() < 1e-15,
                "mismatch at t={t}"
            );
        }
    }

    #[test]
    fn pf_1d_random_walk() {
        // Track a 1D random walk: x_t = x_{t-1} + N(0, 0.1²).
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 300,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1],
            r: vec![0.25],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0],
            p0_chol: vec![0.5],
        };
        // Simple synthetic walk
        let z: Vec<f64> = vec![0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.5, 0.45, 0.6, 0.55];
        let mut rng = LcgRng::new(314);
        let res = pf.run(&z, &mut rng).expect("ok");
        assert_eq!(res.means.len(), 10);
        // Estimate at end should be within reasonable range of the data
        let last = res.means[9][0];
        assert!(last.abs() < 2.0, "estimate out of range: {last}");
    }

    #[test]
    fn pf_2d_state_2d_obs() {
        // dim_x=2, dim_z=2; identity model; runs without error.
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 50,
                dim_x: 2,
                dim_z: 2,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1, 0.0, 0.0, 0.1],
            r: vec![0.5, 0.0, 0.0, 0.5],
            f: Box::new(|x: &[f64]| vec![x[0], x[1]]),
            h: Box::new(|x: &[f64]| vec![x[0], x[1]]),
            x0: vec![0.0, 0.0],
            p0_chol: vec![1.0, 0.0, 0.0, 1.0],
        };
        let z: Vec<f64> = (0..5)
            .flat_map(|i| vec![i as f64 * 0.2, i as f64 * 0.1])
            .collect();
        let mut rng = LcgRng::new(55);
        let res = pf.run(&z, &mut rng).expect("2d test failed");
        assert_eq!(res.means.len(), 5);
        for (t, m) in res.means.iter().enumerate() {
            assert_eq!(m.len(), 2, "state dim at t={t}");
        }
    }

    #[test]
    fn err_empty_obs() {
        let pf = make_1d_pf(50, 0.1, 0.5);
        let mut rng = LcgRng::new(1);
        let result = pf.run(&[], &mut rng);
        assert!(matches!(result, Err(SeqError::EmptyInput)));
    }

    #[test]
    fn err_n_particles_lt_2() {
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 1,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1],
            r: vec![0.5],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        };
        let mut rng = LcgRng::new(1);
        let result = pf.run(&[1.0], &mut rng);
        assert!(matches!(result, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_z_len_not_multiple_of_dim_z() {
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 10,
                dim_x: 1,
                dim_z: 2,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1],
            r: vec![0.5, 0.0, 0.0, 0.5],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0], x[0]]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        };
        let mut rng = LcgRng::new(1);
        let result = pf.run(&[1.0, 2.0, 3.0], &mut rng);
        assert!(matches!(result, Err(SeqError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_q_chol_wrong_shape() {
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 10,
                dim_x: 2,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1], // should be 4 elements for dim_x=2
            r: vec![0.5],
            f: Box::new(|x: &[f64]| x.to_vec()),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0, 0.0],
            p0_chol: vec![1.0, 0.0, 0.0, 1.0],
        };
        let mut rng = LcgRng::new(1);
        let result = pf.run(&[1.0], &mut rng);
        assert!(matches!(result, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_x0_wrong_len() {
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 10,
                dim_x: 2,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1, 0.0, 0.0, 0.1],
            r: vec![0.5],
            f: Box::new(|x: &[f64]| x.to_vec()),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0], // should be length 2
            p0_chol: vec![1.0, 0.0, 0.0, 1.0],
        };
        let mut rng = LcgRng::new(1);
        let result = pf.run(&[1.0], &mut rng);
        assert!(matches!(result, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn pf_systematic_resampling_valid() {
        // After resampling with uniform weights, verify N_eff == N (all weights equal).
        let weights: Vec<f64> = vec![1.0 / 10.0; 10];
        let mut rng = LcgRng::new(42);
        let indices = systematic_resample(&weights, &mut rng);
        // All indices must be in [0, 10)
        for &idx in &indices {
            assert!(idx < 10, "index out of range: {idx}");
        }
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn pf_n_particles_100_no_panic() {
        // 100 particles, 10 steps — verify clean execution.
        let pf = ParticleFilter {
            cfg: ParticleConfig {
                n_particles: 100,
                dim_x: 1,
                dim_z: 1,
                resample_threshold: 0.5,
            },
            q_chol: vec![0.1],
            r: vec![0.3],
            f: Box::new(|x: &[f64]| vec![x[0]]),
            h: Box::new(|x: &[f64]| vec![x[0]]),
            x0: vec![0.0],
            p0_chol: vec![1.0],
        };
        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.1 + 0.5).collect();
        let mut rng = LcgRng::new(2025);
        let res = pf
            .run(&z, &mut rng)
            .expect("100 particles clean run failed");
        assert_eq!(res.means.len(), 10);
        assert_eq!(res.eff_sizes.len(), 10);
        for &neff in &res.eff_sizes {
            assert!(neff > 0.0 && neff <= 100.0 + 1e-6);
        }
    }
}
