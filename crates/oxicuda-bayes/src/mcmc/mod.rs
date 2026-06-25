//! Gradient-based Markov-chain Monte Carlo samplers operating in `f64`.
//!
//! This module provides full-precision (`f64`) reference samplers for Bayesian
//! posterior inference that are kept deliberately separate from the `f32`
//! variational samplers in [`crate::variational::hmc`]:
//!
//! * [`sgld`] — Stochastic Gradient Langevin Dynamics (Welling & Teh 2011),
//!   a scalable MCMC method that injects calibrated Gaussian noise into a
//!   (stochastic) gradient-ascent step on the log-posterior.
//! * [`hmc`] — Hamiltonian Monte Carlo (Neal 2011) with an explicit leapfrog
//!   integrator and Metropolis acceptance correction.
//!
//! Both samplers draw their randomness from [`BayesRng`], a 64-bit MMIX
//! linear-congruential generator that yields `f64` uniforms and standard
//! normals via the Box-Muller transform. The generator is fully deterministic
//! given its seed, which makes the chains reproducible for testing.

pub mod hmc;
pub mod metropolis;
pub mod sgld;

pub use hmc::{HmcConfig as McmcHmcConfig, HmcSampler};
pub use metropolis::{
    MetropolisConfig, MetropolisResult, MetropolisSampler, SliceConfig, SliceSampler, sample_mean,
    sample_variance,
};
pub use sgld::{SgldConfig, SgldSampler};

/// Deterministic 64-bit random number generator for `f64` MCMC sampling.
///
/// Uses the Knuth MMIX 64-bit linear-congruential recurrence
/// `x_{n+1} = 6364136223846793005 · x_n + 1442695040888963407 (mod 2⁶⁴)`
/// and derives `f64` uniforms from the high 53 bits of the state. Standard
/// normal variates are produced with the Box-Muller transform; one of each
/// generated pair is cached so successive `next_normal` calls cost one LCG
/// step on average.
#[derive(Debug, Clone)]
pub struct BayesRng {
    state: u64,
    cached_normal: Option<f64>,
}

impl BayesRng {
    /// Create a new generator from the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            // Offsetting by the increment avoids a degenerate first draw at seed 0.
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            cached_normal: None,
        }
    }

    /// Advance the generator one step and return the raw 64-bit state.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return an `f64` uniformly distributed in `[0, 1)` using 53 random bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Take the top 53 bits to fill the f64 mantissa exactly.
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / 9_007_199_254_740_992.0) // 1 / 2^53
    }

    /// Sample a single standard normal `N(0, 1)` variate via Box-Muller.
    ///
    /// Successive calls alternate between the two outputs of one Box-Muller
    /// evaluation, so the generator advances on every other call.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        // Guard the logarithm against an exact zero draw.
        let u1 = (self.next_f64() + 1e-300).min(1.0 - 1e-16);
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.cached_normal = Some(radius * theta.sin());
        radius * theta.cos()
    }

    /// Fill `buf` with independent standard-normal samples.
    pub fn fill_normal(&mut self, buf: &mut [f64]) {
        for slot in buf.iter_mut() {
            *slot = self.next_normal();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayes_rng_deterministic() {
        let mut a = BayesRng::new(123);
        let mut b = BayesRng::new(123);
        for _ in 0..256 {
            assert_eq!(a.next_f64().to_bits(), b.next_f64().to_bits());
        }
    }

    #[test]
    fn bayes_rng_uniform_in_range() {
        let mut rng = BayesRng::new(7);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
        }
    }

    #[test]
    fn bayes_rng_normal_finite_and_centered() {
        let mut rng = BayesRng::new(99);
        let n = 50_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let z = rng.next_normal();
            assert!(z.is_finite());
            sum += z;
        }
        let mean = sum / n as f64;
        // Sample mean of N(0,1) over 50k draws should be well within 0.05.
        assert!(mean.abs() < 0.05, "normal mean drifted: {mean}");
    }

    #[test]
    fn bayes_rng_normal_variance_near_one() {
        let mut rng = BayesRng::new(2024);
        let n = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let z = rng.next_normal();
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((var - 1.0).abs() < 0.05, "normal variance off: {var}");
    }

    #[test]
    fn bayes_rng_different_seeds_differ() {
        let mut a = BayesRng::new(1);
        let mut b = BayesRng::new(2);
        let mut any_diff = false;
        for _ in 0..32 {
            if a.next_f64() != b.next_f64() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "distinct seeds produced identical streams");
    }
}
