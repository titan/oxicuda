//! Hamiltonian Monte Carlo (HMC) with an explicit leapfrog integrator.
//!
//! Neal, *"MCMC using Hamiltonian dynamics"*, Handbook of Markov Chain Monte
//! Carlo, 2011. HMC augments the target parameter `θ` with an auxiliary
//! momentum `p ~ N(0, I)` and simulates Hamiltonian dynamics of the joint
//! energy
//!
//! ```text
//! H(θ, p) = -log p(θ) + ½ pᵀp
//! ```
//!
//! using the leapfrog (Störmer–Verlet) integrator, then accepts/rejects the
//! end-of-trajectory proposal with a Metropolis correction that exactly
//! compensates for the integrator's energy error. This `f64` implementation is
//! a full-precision reference distinct from the `f32` sampler in
//! [`crate::variational::hmc`].

use crate::error::{BayesError, BayesResult};
use crate::mcmc::BayesRng;
use std::cell::Cell;

/// Configuration for the [`HmcSampler`].
#[derive(Debug, Clone)]
pub struct HmcConfig {
    /// Leapfrog step size `ε > 0`.
    pub step_size: f64,
    /// Number of leapfrog steps `L ≥ 1` per proposal.
    pub n_leapfrog: usize,
    /// Number of samples to collect (= length of the returned chain).
    pub n_samples: usize,
}

/// Hamiltonian Monte Carlo sampler with leapfrog dynamics.
#[derive(Debug)]
pub struct HmcSampler {
    config: HmcConfig,
    /// Acceptance rate from the most recent [`HmcSampler::sample`] call.
    last_accept_rate: Cell<f64>,
}

impl HmcSampler {
    /// Create a new sampler.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `step_size` is not strictly
    /// positive and finite, if `n_leapfrog == 0`, or if `n_samples == 0`.
    pub fn new(config: HmcConfig) -> BayesResult<Self> {
        if !config.step_size.is_finite() || config.step_size <= 0.0 {
            return Err(BayesError::InvalidConfig(format!(
                "HMC step_size must be positive and finite, got {}",
                config.step_size
            )));
        }
        if config.n_leapfrog == 0 {
            return Err(BayesError::InvalidConfig(
                "HMC n_leapfrog must be >= 1".to_string(),
            ));
        }
        if config.n_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "HMC n_samples must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            config,
            last_accept_rate: Cell::new(0.0),
        })
    }

    /// Leapfrog (Störmer–Verlet) integration of `(θ, p)` for `n_steps` steps.
    ///
    /// Uses the standard half-step momentum scheme with a unit mass matrix:
    /// ```text
    /// p ← p + (ε/2)·∇log p(θ)
    /// repeat n_steps times:
    ///     θ ← θ + ε·p
    ///     (full ε momentum update, except the last which is a half step)
    /// p ← p + (ε/2)·∇log p(θ)
    /// ```
    /// Returns the new `(θ, p)`. The momentum potential gradient is the
    /// log-posterior gradient because `H = -log p(θ) + ½pᵀp`.
    pub fn leapfrog(
        theta: &[f64],
        momentum: &[f64],
        grad_log_post: impl Fn(&[f64]) -> Vec<f64>,
        step: f64,
        n_steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = theta.len();
        let mut q = theta.to_vec();
        let mut p = momentum.to_vec();
        if n_steps == 0 {
            return (q, p);
        }
        let half = 0.5 * step;

        // Initial half step for momentum.
        let mut grad = grad_log_post(&q);
        for d in 0..dim {
            p[d] += half * grad[d];
        }

        for l in 0..n_steps {
            // Full position step.
            for d in 0..dim {
                q[d] += step * p[d];
            }
            // Full momentum step, except the final iteration which is a half step.
            grad = grad_log_post(&q);
            let scale = if l == n_steps - 1 { half } else { step };
            for d in 0..dim {
                p[d] += scale * grad[d];
            }
        }

        (q, p)
    }

    /// Run the HMC chain, returning `n_samples` posterior draws.
    ///
    /// `log_post(θ)` returns the unnormalised log-posterior and
    /// `grad_log_post(θ)` its gradient. Both closures must accept a slice of
    /// length `theta_init.len()`; the gradient closure must return a vector of
    /// the same length.
    ///
    /// The acceptance rate of the run is recorded and retrievable via
    /// [`HmcSampler::acceptance_rate`].
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `theta_init` is empty,
    /// [`BayesError::DimensionMismatch`] if a gradient of the wrong length is
    /// produced, and [`BayesError::NanEncountered`] if the initial
    /// log-posterior is non-finite.
    pub fn sample(
        &self,
        theta_init: &[f64],
        log_post: impl Fn(&[f64]) -> f64,
        grad_log_post: impl Fn(&[f64]) -> Vec<f64>,
        rng: &mut BayesRng,
    ) -> BayesResult<Vec<Vec<f64>>> {
        if theta_init.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let dim = theta_init.len();
        let step = self.config.step_size;
        let n_leap = self.config.n_leapfrog;

        // Validate the gradient closure once up front.
        let g0 = grad_log_post(theta_init);
        if g0.len() != dim {
            return Err(BayesError::DimensionMismatch {
                expected: dim,
                got: g0.len(),
            });
        }

        let mut current = theta_init.to_vec();
        let mut current_logp = log_post(&current);
        if !current_logp.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "hmc_initial_log_post",
            });
        }

        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        let mut n_accept: usize = 0;

        for _ in 0..self.config.n_samples {
            // Sample a fresh momentum p ~ N(0, I).
            let mut momentum = vec![0.0_f64; dim];
            rng.fill_normal(&mut momentum);

            // Kinetic energy at the start (½ pᵀp).
            let kinetic_start: f64 = 0.5 * momentum.iter().map(|&v| v * v).sum::<f64>();

            // Simulate trajectory.
            let (proposal, mut p_end) =
                Self::leapfrog(&current, &momentum, &grad_log_post, step, n_leap);
            // Negate momentum to make the proposal symmetric (Neal 2011);
            // does not affect the kinetic energy but keeps the kernel reversible.
            for v in &mut p_end {
                *v = -*v;
            }

            let proposal_logp = log_post(&proposal);
            let kinetic_end: f64 = 0.5 * p_end.iter().map(|&v| v * v).sum::<f64>();

            // Metropolis acceptance in terms of the (negative) Hamiltonian.
            // log α = (log p(θ*) − ½p*ᵀp*) − (log p(θ) − ½pᵀp).
            let accepted = if proposal_logp.is_finite() {
                let log_accept = (proposal_logp - kinetic_end) - (current_logp - kinetic_start);
                log_accept >= 0.0 || rng.next_f64() < log_accept.exp()
            } else {
                false
            };

            if accepted {
                current = proposal;
                current_logp = proposal_logp;
                n_accept += 1;
            }
            samples.push(current.clone());
        }

        let rate = n_accept as f64 / self.config.n_samples as f64;
        self.last_accept_rate.set(rate);
        Ok(samples)
    }

    /// Return the acceptance rate of the most recent [`HmcSampler::sample`]
    /// call. Returns `0.0` if no run has completed yet.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        self.last_accept_rate.get()
    }

    /// Return the configured leapfrog step size.
    #[must_use]
    pub fn step_size(&self) -> f64 {
        self.config.step_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard isotropic Gaussian target centred at `mu`:
    /// `log p(θ) = -½‖θ - μ‖²`, `∇ log p(θ) = -(θ - μ)`.
    fn gaussian_logp(mu: &[f64]) -> impl Fn(&[f64]) -> f64 + '_ {
        move |theta: &[f64]| {
            -0.5 * theta
                .iter()
                .zip(mu)
                .map(|(&t, &m)| (t - m) * (t - m))
                .sum::<f64>()
        }
    }

    fn gaussian_grad(mu: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
        move |theta: &[f64]| theta.iter().zip(mu).map(|(&t, &m)| -(t - m)).collect()
    }

    #[test]
    fn leapfrog_reversible() {
        // Integrating forward then flipping momentum and integrating again
        // returns to the start (up to floating-point error).
        let mu = [0.0, 0.0];
        let grad = gaussian_grad(&mu);
        let theta0 = [1.0, -0.5];
        let p0 = [0.3, 0.7];
        let step = 0.05;
        let n = 20;
        let (theta1, p1) = HmcSampler::leapfrog(&theta0, &p0, &grad, step, n);
        // Reverse: negate momentum, integrate same number of steps.
        let p1_neg: Vec<f64> = p1.iter().map(|&v| -v).collect();
        let (theta2, p2) = HmcSampler::leapfrog(&theta1, &p1_neg, &grad, step, n);
        let p2_neg: Vec<f64> = p2.iter().map(|&v| -v).collect();
        for d in 0..2 {
            assert!((theta2[d] - theta0[d]).abs() < 1e-9, "theta not reversible");
            assert!((p2_neg[d] - p0[d]).abs() < 1e-9, "momentum not reversible");
        }
    }

    #[test]
    fn leapfrog_energy_conserved_approx() {
        // The leapfrog integrator approximately conserves H = -log p + ½pᵀp.
        let mu = [0.0];
        let logp = gaussian_logp(&mu);
        let grad = gaussian_grad(&mu);
        let theta0 = [2.0];
        let p0 = [0.5];
        let step = 0.01;
        let n = 50;
        let h_start = -logp(&theta0) + 0.5 * p0[0] * p0[0];
        let (theta1, p1) = HmcSampler::leapfrog(&theta0, &p0, &grad, step, n);
        let h_end = -logp(&theta1) + 0.5 * p1[0] * p1[0];
        assert!(
            (h_end - h_start).abs() < 1e-2,
            "energy drift {}",
            h_end - h_start
        );
    }

    #[test]
    fn samples_len() {
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 10,
            n_samples: 100,
        })
        .expect("value should be present");
        let mu = [0.0, 0.0];
        let mut rng = BayesRng::new(1);
        let out = sampler
            .sample(
                &[0.0, 0.0],
                gaussian_logp(&mu),
                gaussian_grad(&mu),
                &mut rng,
            )
            .expect("value should be present");
        assert_eq!(out.len(), 100);
        assert!(out.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn samples_finite() {
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 15,
            n_samples: 200,
        })
        .expect("value should be present");
        let mu = [1.0, -1.0];
        let mut rng = BayesRng::new(2);
        let out = sampler
            .sample(
                &[0.0, 0.0],
                gaussian_logp(&mu),
                gaussian_grad(&mu),
                &mut rng,
            )
            .expect("value should be present");
        assert!(out.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn gaussian_target_mean() {
        // HMC on N(μ, I) should recover μ in its sample mean.
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.15,
            n_leapfrog: 20,
            n_samples: 4_000,
        })
        .expect("value should be present");
        let mu = [2.0, -3.0];
        let mut rng = BayesRng::new(123);
        let out = sampler
            .sample(
                &[0.0, 0.0],
                gaussian_logp(&mu),
                gaussian_grad(&mu),
                &mut rng,
            )
            .expect("value should be present");
        let burn = 500;
        let kept = &out[burn..];
        let mut mean = [0.0_f64; 2];
        for s in kept {
            mean[0] += s[0];
            mean[1] += s[1];
        }
        mean[0] /= kept.len() as f64;
        mean[1] /= kept.len() as f64;
        assert!((mean[0] - mu[0]).abs() < 0.2, "mean[0]={}", mean[0]);
        assert!((mean[1] - mu[1]).abs() < 0.2, "mean[1]={}", mean[1]);
    }

    #[test]
    fn n_leapfrog_0_error() {
        let res = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 0,
            n_samples: 10,
        });
        assert!(matches!(res, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn step_size_0_error() {
        let res = HmcSampler::new(HmcConfig {
            step_size: 0.0,
            n_leapfrog: 10,
            n_samples: 10,
        });
        assert!(matches!(res, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn n_samples_0_error() {
        let res = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 10,
            n_samples: 0,
        });
        assert!(matches!(res, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn acceptance_rate_in_range() {
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 15,
            n_samples: 500,
        })
        .expect("value should be present");
        let mu = [0.0, 0.0];
        let mut rng = BayesRng::new(9);
        let _ = sampler
            .sample(
                &[0.0, 0.0],
                gaussian_logp(&mu),
                gaussian_grad(&mu),
                &mut rng,
            )
            .expect("value should be present");
        let rate = sampler.acceptance_rate();
        assert!((0.0..=1.0).contains(&rate), "rate={rate}");
        // A well-tuned Gaussian HMC should accept the vast majority of proposals.
        assert!(rate > 0.6, "acceptance unexpectedly low: {rate}");
    }

    #[test]
    fn theta_init_respected() {
        // First sample is either the (accepted) proposal or the init point;
        // with a far-from-mode init and tiny step the chain stays near init.
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 1e-6,
            n_leapfrog: 1,
            n_samples: 1,
        })
        .expect("value should be present");
        let mu = [0.0, 0.0];
        let init = [5.0, -4.0];
        let mut rng = BayesRng::new(3);
        let out = sampler
            .sample(&init, gaussian_logp(&mu), gaussian_grad(&mu), &mut rng)
            .expect("value should be present");
        assert!((out[0][0] - init[0]).abs() < 1e-2);
        assert!((out[0][1] - init[1]).abs() < 1e-2);
    }

    #[test]
    fn empty_init_error() {
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 5,
            n_samples: 5,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(1);
        let res = sampler.sample(&[], |_t: &[f64]| 0.0, |_t: &[f64]| Vec::new(), &mut rng);
        assert!(matches!(res, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn dimension_mismatch_error() {
        let sampler = HmcSampler::new(HmcConfig {
            step_size: 0.1,
            n_leapfrog: 5,
            n_samples: 5,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(1);
        let res = sampler.sample(
            &[0.0, 0.0],
            |_t: &[f64]| 0.0,
            |_t: &[f64]| vec![0.0, 0.0, 0.0],
            &mut rng,
        );
        assert!(matches!(res, Err(BayesError::DimensionMismatch { .. })));
    }
}
