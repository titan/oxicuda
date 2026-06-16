//! Stochastic Gradient Langevin Dynamics (SGLD).
//!
//! Welling & Teh, *"Bayesian Learning via Stochastic Gradient Langevin
//! Dynamics"*, ICML 2011. SGLD bridges stochastic optimisation and Langevin
//! MCMC: each update takes a gradient-ascent step on the (unnormalised)
//! log-posterior and adds isotropic Gaussian noise scaled so that, as the step
//! size decreases, the iterates converge in distribution to the posterior:
//!
//! ```text
//! θ_{t+1} = θ_t + (ε / 2) · ∇ log p(θ_t) + N(0, ε · I)
//! ```
//!
//! where `∇ log p` is the gradient of the log-posterior (which may itself be a
//! stochastic mini-batch estimate). The Gaussian increment has standard
//! deviation `√ε` per coordinate, i.e. `√ε · z` with `z ~ N(0, I)`.

use crate::error::{BayesError, BayesResult};
use crate::mcmc::BayesRng;

/// Configuration for the [`SgldSampler`].
#[derive(Debug, Clone)]
pub struct SgldConfig {
    /// Langevin step size `ε > 0`.
    pub step_size: f64,
    /// Number of SGLD iterations (= number of returned samples).
    pub n_iter: usize,
}

/// Stochastic Gradient Langevin Dynamics sampler.
#[derive(Debug, Clone)]
pub struct SgldSampler {
    config: SgldConfig,
}

impl SgldSampler {
    /// Create a new sampler.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `step_size` is not strictly
    /// positive and finite, or if `n_iter == 0`.
    pub fn new(config: SgldConfig) -> BayesResult<Self> {
        if !config.step_size.is_finite() || config.step_size <= 0.0 {
            return Err(BayesError::InvalidConfig(format!(
                "SGLD step_size must be positive and finite, got {}",
                config.step_size
            )));
        }
        if config.n_iter == 0 {
            return Err(BayesError::InvalidConfig(
                "SGLD n_iter must be >= 1".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Run the SGLD chain.
    ///
    /// Starting from `theta_init`, performs `n_iter` Langevin updates and
    /// returns the resulting samples (the post-update state at each iteration).
    /// The returned vector has length `n_iter`; each inner vector matches the
    /// dimension of `theta_init`.
    ///
    /// `grad_log_post(θ)` must return the gradient of the log-posterior at `θ`;
    /// its length must equal `theta_init.len()`.
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `theta_init` is empty,
    /// [`BayesError::DimensionMismatch`] if the gradient closure returns a
    /// vector of the wrong length, and [`BayesError::NanEncountered`] if a
    /// non-finite gradient component or iterate is produced.
    pub fn sample(
        &self,
        theta_init: &[f64],
        grad_log_post: impl Fn(&[f64]) -> Vec<f64>,
        rng: &mut BayesRng,
    ) -> BayesResult<Vec<Vec<f64>>> {
        if theta_init.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let dim = theta_init.len();
        let step = self.config.step_size;
        let half_step = 0.5 * step;
        let noise_scale = step.sqrt();

        let mut theta = theta_init.to_vec();
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_iter);

        for _ in 0..self.config.n_iter {
            let grad = grad_log_post(&theta);
            if grad.len() != dim {
                return Err(BayesError::DimensionMismatch {
                    expected: dim,
                    got: grad.len(),
                });
            }
            for d in 0..dim {
                if !grad[d].is_finite() {
                    return Err(BayesError::NanEncountered {
                        location: "sgld_gradient",
                    });
                }
                let noise = noise_scale * rng.next_normal();
                theta[d] += half_step * grad[d] + noise;
                if !theta[d].is_finite() {
                    return Err(BayesError::NanEncountered {
                        location: "sgld_iterate",
                    });
                }
            }
            samples.push(theta.clone());
        }

        Ok(samples)
    }

    /// Return the configured Langevin step size `ε`.
    #[must_use]
    pub fn step_size(&self) -> f64 {
        self.config.step_size
    }

    /// Return the configured number of iterations.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.config.n_iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradient of an isotropic Gaussian log-posterior centred at `mu`:
    /// `log p(θ) = -½‖θ - μ‖²`  ⇒  `∇ log p(θ) = -(θ - μ)`.
    fn grad_gaussian(mu: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
        move |theta: &[f64]| theta.iter().zip(mu).map(|(&t, &m)| -(t - m)).collect()
    }

    #[test]
    fn samples_len() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.01,
            n_iter: 50,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(1);
        let mu = [0.0, 0.0];
        let out = sampler
            .sample(&[1.0, -1.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn samples_shape() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.01,
            n_iter: 10,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(3);
        let mu = [0.0; 3];
        let out = sampler
            .sample(&[1.0, 2.0, 3.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        assert!(out.iter().all(|s| s.len() == 3));
    }

    #[test]
    fn samples_finite() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.02,
            n_iter: 200,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(7);
        let mu = [2.0, -3.0];
        let out = sampler
            .sample(&[0.0, 0.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        assert!(out.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn converges_to_gaussian_mean() {
        // With grad = -(θ - μ) the stationary distribution is N(μ, I); the
        // running average of the chain should approach μ.
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.05,
            n_iter: 20_000,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(42);
        let mu = [3.0, -2.0];
        let out = sampler
            .sample(&[0.0, 0.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        // Discard a burn-in segment, then average.
        let burn = 2_000;
        let kept = &out[burn..];
        let mut mean = [0.0_f64; 2];
        for s in kept {
            mean[0] += s[0];
            mean[1] += s[1];
        }
        mean[0] /= kept.len() as f64;
        mean[1] /= kept.len() as f64;
        assert!((mean[0] - mu[0]).abs() < 0.25, "mean[0]={}", mean[0]);
        assert!((mean[1] - mu[1]).abs() < 0.25, "mean[1]={}", mean[1]);
    }

    #[test]
    fn step_size_0_error() {
        let err = SgldSampler::new(SgldConfig {
            step_size: 0.0,
            n_iter: 10,
        });
        assert!(matches!(err, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn n_iter_0_error() {
        let err = SgldSampler::new(SgldConfig {
            step_size: 0.01,
            n_iter: 0,
        });
        assert!(matches!(err, Err(BayesError::InvalidConfig(_))));
    }

    #[test]
    fn different_seeds_different_chains() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.05,
            n_iter: 100,
        })
        .expect("value should be present");
        let mu = [0.0, 0.0];
        let mut rng_a = BayesRng::new(11);
        let mut rng_b = BayesRng::new(22);
        let a = sampler
            .sample(&[1.0, 1.0], grad_gaussian(&mu), &mut rng_a)
            .expect("value should be present");
        let b = sampler
            .sample(&[1.0, 1.0], grad_gaussian(&mu), &mut rng_b)
            .expect("value should be present");
        assert_ne!(
            a.last().expect("last should succeed"),
            b.last().expect("last should succeed")
        );
    }

    #[test]
    fn theta_init_respected() {
        // With a tiny step the very first sample should sit close to θ_init.
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 1e-8,
            n_iter: 1,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(5);
        let mu = [0.0, 0.0];
        let init = [4.0, -7.0];
        let out = sampler
            .sample(&init, grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        assert!((out[0][0] - init[0]).abs() < 1e-3);
        assert!((out[0][1] - init[1]).abs() < 1e-3);
    }

    #[test]
    fn decreasing_step_works() {
        // Two samplers with different step sizes should both run and produce
        // finite chains; the smaller step produces a tighter spread.
        let mu = [0.0];
        let big = SgldSampler::new(SgldConfig {
            step_size: 0.1,
            n_iter: 500,
        })
        .expect("value should be present");
        let small = SgldSampler::new(SgldConfig {
            step_size: 0.001,
            n_iter: 500,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(8);
        let ob = big
            .sample(&[0.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        let os = small
            .sample(&[0.0], grad_gaussian(&mu), &mut rng)
            .expect("value should be present");
        assert!(ob.iter().flatten().all(|v| v.is_finite()));
        assert!(os.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn dimension_mismatch_error() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.01,
            n_iter: 5,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(1);
        // Closure returns wrong-length gradient.
        let bad = |_theta: &[f64]| vec![0.0, 0.0, 0.0];
        let res = sampler.sample(&[1.0, 2.0], bad, &mut rng);
        assert!(matches!(res, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_init_error() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.01,
            n_iter: 5,
        })
        .expect("value should be present");
        let mut rng = BayesRng::new(1);
        let res = sampler.sample(&[], |_t: &[f64]| Vec::new(), &mut rng);
        assert!(matches!(res, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn step_size_accessor() {
        let sampler = SgldSampler::new(SgldConfig {
            step_size: 0.037,
            n_iter: 5,
        })
        .expect("value should be present");
        assert!((sampler.step_size() - 0.037).abs() < 1e-12);
        assert_eq!(sampler.n_iter(), 5);
    }
}
