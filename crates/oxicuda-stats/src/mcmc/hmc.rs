//! Hamiltonian Monte Carlo (HMC) sampler.
//!
//! HMC augments the target distribution π(q) ∝ exp(−U(q)) with an auxiliary
//! momentum variable p ~ N(0, M) and simulates Hamiltonian dynamics on the joint
//! density to propose distant, high-acceptance moves. With a unit mass matrix
//! (M = I) the Hamiltonian is
//!
//! ```text
//! H(q, p) = U(q) + ½ pᵀp
//! ```
//!
//! where `U(q) = −log π(q)` is the *potential energy*. Trajectories are
//! integrated with the leapfrog (Störmer–Verlet) scheme, which is symplectic and
//! time-reversible, then accepted/rejected with a Metropolis correction that
//! exactly cancels the integrator's volume-preserving bias.
//!
//! # References
//! - Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of Markov
//!   Chain Monte Carlo*, Chapman & Hall/CRC, Ch. 5.
//! - Duane, Kennedy, Pendleton & Roweth (1987). "Hybrid Monte Carlo."
//!   *Physics Letters B* 195(2):216-222.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

/// Boxed potential-energy closure `U(q) = −log π(q)`.
type PotentialFn<'a> = Box<dyn Fn(&[f64]) -> f64 + 'a>;

/// Boxed gradient closure `∇U(q)`.
type GradientFn<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>;

/// A target distribution for HMC/NUTS specified through its potential energy.
///
/// The potential energy is `U(q) = −log π(q)` (up to an additive constant). The
/// gradient `∇U(q)` may be supplied analytically; if omitted, a central
/// finite-difference approximation is used.
pub struct PotentialTarget<'a> {
    /// Dimension of the parameter space.
    pub dim: usize,
    /// Potential energy `U(q) = −log π(q)`.
    potential: PotentialFn<'a>,
    /// Optional analytic gradient `∇U(q)`.
    gradient: Option<GradientFn<'a>>,
    /// Step size for the finite-difference gradient fallback.
    fd_eps: f64,
}

impl<'a> PotentialTarget<'a> {
    /// Build a target from a potential-energy closure with a finite-difference
    /// gradient fallback.
    pub fn new<F>(dim: usize, potential: F) -> StatsResult<Self>
    where
        F: Fn(&[f64]) -> f64 + 'a,
    {
        if dim == 0 {
            return Err(StatsError::InvalidParameter {
                name: "dim".to_string(),
                reason: "dimension must be ≥ 1".to_string(),
            });
        }
        Ok(Self {
            dim,
            potential: Box::new(potential),
            gradient: None,
            fd_eps: 1e-5,
        })
    }

    /// Attach an analytic gradient `∇U(q)`.
    #[must_use]
    pub fn with_gradient<G>(mut self, gradient: G) -> Self
    where
        G: Fn(&[f64]) -> Vec<f64> + 'a,
    {
        self.gradient = Some(Box::new(gradient));
        self
    }

    /// Override the finite-difference step size used by the gradient fallback.
    #[must_use]
    pub fn with_fd_eps(mut self, eps: f64) -> Self {
        self.fd_eps = eps;
        self
    }

    /// Evaluate the potential energy `U(q)`.
    #[must_use]
    pub fn potential(&self, q: &[f64]) -> f64 {
        (self.potential)(q)
    }

    /// Evaluate the gradient `∇U(q)`, analytic if available, otherwise via a
    /// central finite difference.
    #[must_use]
    pub fn grad(&self, q: &[f64]) -> Vec<f64> {
        if let Some(g) = &self.gradient {
            g(q)
        } else {
            self.finite_difference(q)
        }
    }

    /// Central finite-difference gradient of the potential.
    fn finite_difference(&self, q: &[f64]) -> Vec<f64> {
        let mut grad = vec![0.0_f64; self.dim];
        let mut qp = q.to_vec();
        for i in 0..self.dim {
            let h = self.fd_eps * (1.0 + q[i].abs());
            let orig = qp[i];
            qp[i] = orig + h;
            let f_plus = (self.potential)(&qp);
            qp[i] = orig - h;
            let f_minus = (self.potential)(&qp);
            qp[i] = orig;
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        grad
    }
}

/// One step of the leapfrog (Störmer–Verlet) integrator.
///
/// Updates `(q, p)` in place by a half-step momentum kick, a full-step position
/// drift, and a closing half-step momentum kick, using a unit mass matrix:
///
/// ```text
/// p ← p − (ε/2) ∇U(q)
/// q ← q + ε p
/// p ← p − (ε/2) ∇U(q)
/// ```
pub fn leapfrog_step(target: &PotentialTarget<'_>, q: &mut [f64], p: &mut [f64], step_size: f64) {
    let dim = target.dim;
    let half = 0.5 * step_size;
    let grad0 = target.grad(q);
    for i in 0..dim {
        p[i] -= half * grad0[i];
    }
    for i in 0..dim {
        q[i] += step_size * p[i];
    }
    let grad1 = target.grad(q);
    for i in 0..dim {
        p[i] -= half * grad1[i];
    }
}

/// Integrate `n_steps` leapfrog steps, mutating `(q, p)` in place.
pub fn leapfrog(
    target: &PotentialTarget<'_>,
    q: &mut [f64],
    p: &mut [f64],
    step_size: f64,
    n_steps: usize,
) {
    for _ in 0..n_steps {
        leapfrog_step(target, q, p, step_size);
    }
}

/// Total Hamiltonian energy `H(q, p) = U(q) + ½ pᵀp` (unit mass matrix).
#[must_use]
pub fn hamiltonian(target: &PotentialTarget<'_>, q: &[f64], p: &[f64]) -> f64 {
    let kinetic: f64 = 0.5 * p.iter().map(|&pi| pi * pi).sum::<f64>();
    target.potential(q) + kinetic
}

/// Configuration for the HMC sampler.
#[derive(Debug, Clone)]
pub struct HmcConfig {
    /// Leapfrog step size ε.
    pub step_size: f64,
    /// Number of leapfrog steps L per proposal.
    pub n_leapfrog: usize,
    /// Number of post-warmup samples to retain.
    pub n_samples: usize,
    /// Number of warmup (burn-in) iterations discarded before sampling.
    pub n_warmup: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            n_leapfrog: 20,
            n_samples: 1000,
            n_warmup: 500,
            seed: 0,
        }
    }
}

/// Output of an HMC run.
#[derive(Debug, Clone)]
pub struct HmcSamples {
    /// Retained samples, row-major `n_samples × dim`.
    pub samples: Vec<f64>,
    /// Parameter dimension.
    pub dim: usize,
    /// Number of retained samples.
    pub n_samples: usize,
    /// Fraction of proposals accepted (over warmup + sampling).
    pub accept_rate: f64,
}

impl HmcSamples {
    /// Borrow sample `i` as a slice of length `dim`.
    #[must_use]
    pub fn sample(&self, i: usize) -> &[f64] {
        &self.samples[i * self.dim..(i + 1) * self.dim]
    }

    /// Per-dimension sample mean.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let mut m = vec![0.0_f64; self.dim];
        for i in 0..self.n_samples {
            let s = self.sample(i);
            for d in 0..self.dim {
                m[d] += s[d];
            }
        }
        let inv = 1.0 / self.n_samples.max(1) as f64;
        for v in &mut m {
            *v *= inv;
        }
        m
    }

    /// Population covariance matrix of the samples (row-major `dim × dim`).
    #[must_use]
    pub fn covariance(&self) -> Vec<f64> {
        let mean = self.mean();
        let d = self.dim;
        let mut cov = vec![0.0_f64; d * d];
        for i in 0..self.n_samples {
            let s = self.sample(i);
            for a in 0..d {
                let da = s[a] - mean[a];
                for b in 0..d {
                    cov[a * d + b] += da * (s[b] - mean[b]);
                }
            }
        }
        let inv = 1.0 / self.n_samples.max(1) as f64;
        for v in &mut cov {
            *v *= inv;
        }
        cov
    }
}

/// Draw a standard-normal momentum vector of length `dim`.
fn sample_momentum(rng: &mut LcgRng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rng.next_normal()).collect()
}

/// Run Hamiltonian Monte Carlo starting from `q_init`.
///
/// Each iteration resamples the momentum from N(0, I), integrates a leapfrog
/// trajectory of `n_leapfrog` steps, and applies a Metropolis acceptance test on
/// the change in Hamiltonian energy.
pub fn hmc_sample(
    target: &PotentialTarget<'_>,
    q_init: &[f64],
    config: &HmcConfig,
) -> StatsResult<HmcSamples> {
    let dim = target.dim;
    if q_init.len() != dim {
        return Err(StatsError::DimensionMismatch {
            a: q_init.len(),
            b: dim,
        });
    }
    if !(config.step_size > 0.0 && config.step_size.is_finite()) {
        return Err(StatsError::InvalidParameter {
            name: "step_size".to_string(),
            reason: format!("must be > 0 and finite; got {}", config.step_size),
        });
    }
    if config.n_leapfrog == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_leapfrog".to_string(),
            reason: "must be ≥ 1".to_string(),
        });
    }

    let mut rng = LcgRng::new(config.seed);
    let mut q_current = q_init.to_vec();
    let total = config.n_warmup + config.n_samples;
    let mut samples = Vec::with_capacity(config.n_samples * dim);
    let mut n_accept = 0usize;

    for iter in 0..total {
        // Resample momentum p ~ N(0, I).
        let p_current = sample_momentum(&mut rng, dim);
        let h_current = hamiltonian(target, &q_current, &p_current);

        // Integrate a leapfrog trajectory from a copy of the current state.
        let mut q_prop = q_current.clone();
        let mut p_prop = p_current.clone();
        leapfrog(
            target,
            &mut q_prop,
            &mut p_prop,
            config.step_size,
            config.n_leapfrog,
        );
        // Negating momentum makes the proposal symmetric; it does not affect the
        // kinetic energy, so it is folded into the acceptance test directly.
        let h_prop = hamiltonian(target, &q_prop, &p_prop);

        let log_accept = h_current - h_prop;
        let accept = log_accept >= 0.0 || rng.next_f64() < log_accept.exp();
        if accept && q_prop.iter().all(|v| v.is_finite()) {
            q_current = q_prop;
            n_accept += 1;
        }

        if iter >= config.n_warmup {
            samples.extend_from_slice(&q_current);
        }
    }

    Ok(HmcSamples {
        samples,
        dim,
        n_samples: config.n_samples,
        accept_rate: n_accept as f64 / total.max(1) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard isotropic Gaussian target: U(q) = ½ qᵀq, ∇U(q) = q.
    fn std_gaussian(dim: usize) -> PotentialTarget<'static> {
        PotentialTarget::new(dim, |q: &[f64]| 0.5 * q.iter().map(|&x| x * x).sum::<f64>())
            .expect("dim ≥ 1")
            .with_gradient(|q: &[f64]| q.to_vec())
    }

    /// Correlated 2-D Gaussian with covariance Σ; U(q) = ½ qᵀ Σ⁻¹ q.
    fn correlated_gaussian(rho: f64) -> PotentialTarget<'static> {
        // Σ = [[1, ρ], [ρ, 1]]; Σ⁻¹ = 1/(1−ρ²) [[1, −ρ], [−ρ, 1]].
        let det = 1.0 - rho * rho;
        PotentialTarget::new(2, move |q: &[f64]| {
            let (a, b) = (q[0], q[1]);
            0.5 * (a * a - 2.0 * rho * a * b + b * b) / det
        })
        .expect("dim ≥ 1")
        .with_gradient(move |q: &[f64]| {
            let (a, b) = (q[0], q[1]);
            vec![(a - rho * b) / det, (b - rho * a) / det]
        })
    }

    #[test]
    fn leapfrog_is_reversible() {
        // Integrating forward L steps, negating p, then integrating L more steps
        // must return to the starting (q, p) up to round-off.
        let target = std_gaussian(2);
        let q0 = vec![0.7, -1.3];
        let p0 = vec![0.2, 0.9];
        let mut q = q0.clone();
        let mut p = p0.clone();
        let step = 0.05;
        let l = 40;
        leapfrog(&target, &mut q, &mut p, step, l);
        // Negate momentum and integrate the same number of steps.
        for pi in &mut p {
            *pi = -*pi;
        }
        leapfrog(&target, &mut q, &mut p, step, l);
        // Negate again to compare with the original momentum.
        for pi in &mut p {
            *pi = -*pi;
        }
        for i in 0..2 {
            assert!(
                (q[i] - q0[i]).abs() < 1e-8,
                "q[{i}] = {} vs {}",
                q[i],
                q0[i]
            );
            assert!(
                (p[i] - p0[i]).abs() < 1e-8,
                "p[{i}] = {} vs {}",
                p[i],
                p0[i]
            );
        }
    }

    #[test]
    fn leapfrog_conserves_energy_approximately() {
        // Over a short trajectory the symplectic integrator keeps |ΔH| small.
        let target = std_gaussian(3);
        let q0 = vec![1.0, -0.5, 0.3];
        let p0 = vec![0.4, 0.8, -0.2];
        let h0 = hamiltonian(&target, &q0, &p0);
        let mut q = q0;
        let mut p = p0;
        leapfrog(&target, &mut q, &mut p, 0.02, 50);
        let h1 = hamiltonian(&target, &q, &p);
        assert!((h1 - h0).abs() < 1e-3, "ΔH = {}", (h1 - h0).abs());
    }

    #[test]
    fn samples_standard_gaussian_moments() {
        let target = std_gaussian(1);
        let config = HmcConfig {
            step_size: 0.25,
            n_leapfrog: 12,
            n_samples: 4000,
            n_warmup: 1000,
            seed: 2024,
        };
        let out = hmc_sample(&target, &[0.0], &config).expect("hmc ok");
        let mean = out.mean()[0];
        let cov = out.covariance()[0];
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!((cov - 1.0).abs() < 0.15, "variance = {cov}");
        assert!(out.accept_rate > 0.6, "accept = {}", out.accept_rate);
    }

    #[test]
    fn samples_correlated_gaussian_covariance() {
        let rho = 0.7;
        let target = correlated_gaussian(rho);
        let config = HmcConfig {
            step_size: 0.18,
            n_leapfrog: 18,
            n_samples: 6000,
            n_warmup: 1500,
            seed: 7,
        };
        let out = hmc_sample(&target, &[0.0, 0.0], &config).expect("hmc ok");
        let cov = out.covariance();
        // Off-diagonal must recover the sign and rough magnitude of ρ.
        assert!(cov[1] > 0.4 && cov[1] < 1.0, "cov01 = {}", cov[1]);
        assert!(cov[0] > 0.6 && cov[0] < 1.4, "var0 = {}", cov[0]);
        assert!(cov[3] > 0.6 && cov[3] < 1.4, "var1 = {}", cov[3]);
    }

    #[test]
    fn finite_difference_gradient_matches_analytic() {
        // Without an analytic gradient the central FD must match ∇U(q) = Σ⁻¹ q.
        let det = 1.0 - 0.5 * 0.5;
        let target = PotentialTarget::new(2, |q: &[f64]| {
            let (a, b) = (q[0], q[1]);
            0.5 * (a * a - 2.0 * 0.5 * a * b + b * b) / det
        })
        .expect("ok");
        let q = vec![0.9, -0.4];
        let g = target.grad(&q);
        let expected = [(q[0] - 0.5 * q[1]) / det, (q[1] - 0.5 * q[0]) / det];
        for i in 0..2 {
            assert!((g[i] - expected[i]).abs() < 1e-5, "g[{i}] = {}", g[i]);
        }
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let target = std_gaussian(2);
        let config = HmcConfig {
            step_size: 0.2,
            n_leapfrog: 10,
            n_samples: 200,
            n_warmup: 100,
            seed: 42,
        };
        let a = hmc_sample(&target, &[0.0, 0.0], &config).expect("ok");
        let b = hmc_sample(&target, &[0.0, 0.0], &config).expect("ok");
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn rejects_bad_dimension() {
        let target = std_gaussian(2);
        let config = HmcConfig::default();
        assert!(hmc_sample(&target, &[0.0], &config).is_err());
    }
}
