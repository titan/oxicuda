//! Bayesian SNN weights via Bayes-by-Backprop (variational inference).
//!
//! Implements the "Weight Uncertainty in Neural Networks" / Bayes-by-Backprop
//! scheme of Blundell, Cornebise, Kavukcuoglu & Wierstra (*ICML* 2015), here
//! applied to the synaptic weights of a spiking layer. Instead of a single
//! point estimate, every weight `w` carries an independent variational Gaussian
//! posterior
//!
//! ```text
//! q(w | θ) = N(w ; μ, σ²),   σ = softplus(ρ) = ln(1 + e^ρ)  > 0
//! ```
//!
//! parameterised by the unconstrained pair `θ = (μ, ρ)`. The softplus
//! re-parameterisation of `ρ` guarantees a strictly positive standard deviation
//! for any real `ρ`.
//!
//! # Reparameterisation trick
//!
//! Weights are sampled differentiably as
//!
//! ```text
//! w = μ + softplus(ρ) · ε,    ε ~ N(0, 1)
//! ```
//!
//! so that gradients of a downstream loss flow into `μ` and `ρ` through the
//! deterministic transform.
//!
//! # Evidence Lower BOund (ELBO)
//!
//! Training maximises the ELBO, equivalently minimises the variational free
//! energy
//!
//! ```text
//! F(θ) = β · KL[ q(w|θ) ‖ p(w) ]  −  E_{q}[ log p(D | w) ]
//!      = β · KL                    +  NLL,
//! ```
//!
//! where `β` (`kl_weight`) anneals/scales the complexity cost (often `1/n_batches`).
//! For a Gaussian prior `p(w) = N(0, σ_prior²)` the per-weight KL has the closed
//! form
//!
//! ```text
//! KL[ N(μ,σ²) ‖ N(0,σ_p²) ] = ln(σ_p/σ) + (σ² + μ²) / (2 σ_p²) − ½.
//! ```
//!
//! Analytic gradients of this KL term with respect to `μ` and `ρ` are provided
//! by [`crate::training::bayesian_snn::grad_mu`] and [`crate::training::bayesian_snn::grad_rho`] so a caller can combine them with the NLL
//! gradient from the reparameterised forward pass.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;

/// Numerically-stable softplus `ln(1 + e^x)`.
///
/// For large `x` returns `x` (the `e^x` term dominates and `ln(1+e^x) ≈ x`),
/// avoiding overflow; for very negative `x` the value approaches `e^x`. The
/// result is floored at [`f32::MIN_POSITIVE`] so it is *always* strictly
/// positive (mathematically `ln(1+e^x) > 0` for every finite `x`), even where
/// `e^x` underflows to `0` in `f32`. This guarantees a usable `σ > 0` for the
/// downstream KL and reparameterisation terms.
#[must_use]
#[inline]
pub fn softplus(x: f32) -> f32 {
    let raw = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    };
    raw.max(f32::MIN_POSITIVE)
}

/// Logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`, the derivative of [`softplus`].
#[must_use]
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-weight KL divergence `KL[ N(μ, σ²) ‖ N(0, σ_prior²) ]`.
///
/// Closed form `ln(σ_p/σ) + (σ² + μ²)/(2 σ_p²) − ½`, which is `≥ 0` and equals
/// `0` exactly when `μ = 0` and `σ = σ_prior`.
///
/// # Errors
///
/// Returns [`SnnError::OutOfRange`] when `sigma ≤ 0` or `sigma_prior ≤ 0` (or
/// either is non-finite).
pub fn kl_gaussian(mu: f32, sigma: f32, sigma_prior: f32) -> SnnResult<f32> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "sigma".into(),
            val: sigma,
        });
    }
    if !sigma_prior.is_finite() || sigma_prior <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "sigma_prior".into(),
            val: sigma_prior,
        });
    }
    let sp2 = sigma_prior * sigma_prior;
    Ok((sigma_prior / sigma).ln() + (sigma * sigma + mu * mu) / (2.0 * sp2) - 0.5)
}

/// Analytic gradient of the per-weight KL term with respect to `μ`:
/// `∂KL/∂μ = μ / σ_prior²`.
///
/// # Errors
///
/// Returns [`SnnError::OutOfRange`] when `sigma_prior ≤ 0` or non-finite.
pub fn grad_mu(mu: f32, sigma_prior: f32) -> SnnResult<f32> {
    if !sigma_prior.is_finite() || sigma_prior <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "sigma_prior".into(),
            val: sigma_prior,
        });
    }
    Ok(mu / (sigma_prior * sigma_prior))
}

/// Analytic gradient of the per-weight KL term with respect to `ρ`.
///
/// With `σ = softplus(ρ)` and `dσ/dρ = sigmoid(ρ)`, the chain rule gives
///
/// ```text
/// ∂KL/∂σ = −1/σ + σ/σ_prior²
/// ∂KL/∂ρ = (∂KL/∂σ) · sigmoid(ρ) = (σ/σ_prior² − 1/σ) · sigmoid(ρ).
/// ```
///
/// # Errors
///
/// Returns [`SnnError::OutOfRange`] when `sigma_prior ≤ 0` or non-finite.
pub fn grad_rho(rho: f32, sigma_prior: f32) -> SnnResult<f32> {
    if !sigma_prior.is_finite() || sigma_prior <= 0.0 {
        return Err(SnnError::OutOfRange {
            name: "sigma_prior".into(),
            val: sigma_prior,
        });
    }
    let sigma = softplus(rho).max(1e-12);
    let dkl_dsigma = sigma / (sigma_prior * sigma_prior) - 1.0 / sigma;
    Ok(dkl_dsigma * sigmoid(rho))
}

/// Evidence-lower-bound helper expressed as the (minimised) variational free
/// energy `F = kl_weight · KL + NLL`.
///
/// Maximising the ELBO is equivalent to minimising this quantity; `kl_weight`
/// (`β`) scales the complexity cost, e.g. `1/num_minibatches`.
///
/// # Errors
///
/// Returns [`SnnError::OutOfRange`] when `kl_weight < 0` or non-finite.
pub fn elbo(nll: f32, kl: f32, kl_weight: f32) -> SnnResult<f32> {
    if !kl_weight.is_finite() || kl_weight < 0.0 {
        return Err(SnnError::OutOfRange {
            name: "kl_weight".into(),
            val: kl_weight,
        });
    }
    Ok(kl_weight * kl + nll)
}

/// A Bayesian linear (dense) layer with an independent Gaussian variational
/// posterior over every weight.
///
/// Weights are laid out row-major `[out_dim × in_dim]`, matching the rest of the
/// crate. Each weight `w[k]` has variational parameters `mu[k]` and `rho[k]`
/// with `σ[k] = softplus(rho[k])`.
#[derive(Debug, Clone)]
pub struct BayesianLinear {
    /// Posterior means `μ`, length `out_dim * in_dim`, row-major.
    pub mu: Vec<f32>,
    /// Unconstrained posterior scales `ρ` (so `σ = softplus(ρ)`),
    /// length `out_dim * in_dim`, row-major.
    pub rho: Vec<f32>,
    /// Number of input features.
    pub in_dim: usize,
    /// Number of output features.
    pub out_dim: usize,
    /// Prior standard deviation `σ_prior` of the zero-mean Gaussian prior.
    pub sigma_prior: f32,
}

impl BayesianLinear {
    /// Construct a Bayesian linear layer with random initial means and a fixed
    /// small initial scale.
    ///
    /// Means are drawn from `N(0, (1/√in_dim)²)` (LeCun-style fan-in) and `ρ` is
    /// initialised so that `σ ≈ 0.05` for every weight (a common small-variance
    /// start). The prior is `N(0, σ_prior²)` with `σ_prior = 1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::BadDim`] when `in_dim` or `out_dim` is zero.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> SnnResult<Self> {
        if in_dim == 0 {
            return Err(SnnError::BadDim { got: in_dim });
        }
        if out_dim == 0 {
            return Err(SnnError::BadDim { got: out_dim });
        }
        let n = in_dim * out_dim;
        let mu_scale = 1.0 / (in_dim as f32).sqrt();
        let mu: Vec<f32> = (0..n).map(|_| rng.next_normal() * mu_scale).collect();
        // Solve softplus(ρ) = 0.05 ⇒ ρ = ln(e^{0.05} − 1).
        let rho_init = (0.05_f32.exp() - 1.0).ln();
        let rho = vec![rho_init; n];
        Ok(Self {
            mu,
            rho,
            in_dim,
            out_dim,
            sigma_prior: 1.0,
        })
    }

    /// Number of variational weights, `out_dim * in_dim`.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.mu.len()
    }

    /// Whether the layer has zero weights (always `false` after a successful
    /// [`BayesianLinear::new`]).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mu.is_empty()
    }

    /// Per-weight posterior standard deviations `σ = softplus(ρ)`, length
    /// `out_dim * in_dim`. Every entry is strictly positive.
    #[must_use]
    pub fn posterior_sigma(&self) -> Vec<f32> {
        self.rho.iter().map(|&r| softplus(r)).collect()
    }

    /// Draw one set of concrete weights via the reparameterisation trick,
    /// `w = μ + softplus(ρ) · ε` with `ε ~ N(0, 1)` from `rng`. Returns a
    /// row-major `[out_dim × in_dim]` weight vector. Deterministic given a
    /// seeded `rng`.
    #[must_use]
    pub fn sample(&self, rng: &mut LcgRng) -> Vec<f32> {
        let mut w = vec![0.0_f32; self.mu.len()];
        for (k, w_k) in w.iter_mut().enumerate() {
            let eps = rng.next_normal();
            *w_k = self.mu[k] + softplus(self.rho[k]) * eps;
        }
        w
    }

    /// Alias of [`BayesianLinear::sample`] matching the task's `sample_weights`
    /// naming.
    #[must_use]
    #[inline]
    pub fn sample_weights(&self, rng: &mut LcgRng) -> Vec<f32> {
        self.sample(rng)
    }

    /// Total KL divergence of the layer posterior to the prior, summed over all
    /// weights (`Σ_k KL[ N(μ_k, σ_k²) ‖ N(0, σ_prior²) ]`).
    ///
    /// The result is always `≥ 0`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::OutOfRange`] when `sigma_prior` is invalid (the
    /// per-weight `σ` are guaranteed positive by softplus, so only the prior is
    /// validated, inside [`kl_gaussian`]).
    pub fn kl(&self) -> SnnResult<f32> {
        let mut total = 0.0_f32;
        for k in 0..self.mu.len() {
            let sigma = softplus(self.rho[k]).max(1e-12);
            total += kl_gaussian(self.mu[k], sigma, self.sigma_prior)?;
        }
        Ok(total)
    }

    /// Per-weight analytic KL gradients `(∂KL/∂μ_k, ∂KL/∂ρ_k)` summed for the KL
    /// term, returned as two row-major vectors of length `out_dim * in_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::OutOfRange`] when `sigma_prior` is invalid.
    pub fn kl_grads(&self) -> SnnResult<(Vec<f32>, Vec<f32>)> {
        let n = self.mu.len();
        let mut g_mu = vec![0.0_f32; n];
        let mut g_rho = vec![0.0_f32; n];
        for k in 0..n {
            g_mu[k] = grad_mu(self.mu[k], self.sigma_prior)?;
            g_rho[k] = grad_rho(self.rho[k], self.sigma_prior)?;
        }
        Ok((g_mu, g_rho))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. softplus is strictly positive everywhere, including extremes.
    #[test]
    fn softplus_positive() {
        for &x in &[-1000.0_f32, -25.0, -1.0, 0.0, 1.0, 25.0, 1000.0] {
            let s = softplus(x);
            assert!(s > 0.0, "softplus({x})={s} must be > 0");
            assert!(s.is_finite(), "softplus({x}) must be finite");
        }
        // softplus(0) = ln 2
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
    }

    // 2. sigmoid is the derivative of softplus (finite-difference check).
    #[test]
    fn sigmoid_is_softplus_derivative() {
        let eps = 1e-3_f32;
        for &x in &[-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            let fd = (softplus(x + eps) - softplus(x - eps)) / (2.0 * eps);
            assert!(
                (fd - sigmoid(x)).abs() < 1e-2,
                "x={x}: fd={fd}, σ={}",
                sigmoid(x)
            );
        }
    }

    // 3. KL ≥ 0 for arbitrary parameters.
    #[test]
    fn kl_non_negative() {
        let cases = [
            (0.0_f32, 1.0_f32, 1.0_f32),
            (2.0, 0.1, 1.0),
            (-1.5, 3.0, 0.5),
            (0.3, 0.7, 2.0),
        ];
        for (mu, sigma, sp) in cases {
            let kl = kl_gaussian(mu, sigma, sp).expect("kl");
            assert!(kl >= -1e-6, "KL({mu},{sigma},{sp})={kl} must be ≥ 0");
        }
    }

    // 4. KL = 0 when posterior == prior (μ=0, σ=σ_prior).
    #[test]
    fn kl_zero_at_prior() {
        let kl = kl_gaussian(0.0, 1.3, 1.3).expect("kl");
        assert!(
            kl.abs() < 1e-6,
            "KL should be 0 when posterior==prior, got {kl}"
        );
    }

    // 5. KL rejects non-positive sigma / sigma_prior.
    #[test]
    fn kl_rejects_bad_sigma() {
        assert!(matches!(
            kl_gaussian(0.0, 0.0, 1.0),
            Err(SnnError::OutOfRange { .. })
        ));
        assert!(matches!(
            kl_gaussian(0.0, 1.0, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    // 6. grad_mu = μ/σ_p²; finite-difference check against kl_gaussian.
    #[test]
    fn grad_mu_matches_fd() {
        let sp = 1.7_f32;
        let mu = 0.6_f32;
        let sigma = 0.9_f32;
        let eps = 1e-3_f32;
        let kp = kl_gaussian(mu + eps, sigma, sp).expect("kl");
        let km = kl_gaussian(mu - eps, sigma, sp).expect("kl");
        let fd = (kp - km) / (2.0 * eps);
        let g = grad_mu(mu, sp).expect("grad");
        assert!((fd - g).abs() < 1e-2, "fd={fd}, analytic={g}");
        // closed form μ/σ_p²
        assert!((g - mu / (sp * sp)).abs() < 1e-6);
    }

    // 7. grad_rho matches a finite-difference of KL through σ=softplus(ρ).
    #[test]
    fn grad_rho_matches_fd() {
        let sp = 1.2_f32;
        let mu = -0.4_f32;
        let rho = 0.3_f32;
        let kl_of_rho = |r: f32| -> f32 {
            let sigma = softplus(r);
            kl_gaussian(mu, sigma, sp).expect("kl")
        };
        let eps = 1e-3_f32;
        let fd = (kl_of_rho(rho + eps) - kl_of_rho(rho - eps)) / (2.0 * eps);
        let g = grad_rho(rho, sp).expect("grad");
        assert!((fd - g).abs() < 1e-2, "fd={fd}, analytic={g}");
    }

    // 8. ELBO arithmetic: F = β·KL + NLL.
    #[test]
    fn elbo_arithmetic() {
        let f = elbo(2.0, 5.0, 0.5).expect("elbo");
        assert!((f - (0.5 * 5.0 + 2.0)).abs() < 1e-6);
        // β = 0 ⇒ free energy is just the NLL.
        let f0 = elbo(3.0, 100.0, 0.0).expect("elbo");
        assert!((f0 - 3.0).abs() < 1e-6);
        // negative β rejected
        assert!(matches!(
            elbo(0.0, 0.0, -0.1),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    // 9. Constructor shapes + rejects zero dims.
    #[test]
    fn constructor_shapes() {
        let mut rng = LcgRng::new(1);
        let layer = BayesianLinear::new(3, 2, &mut rng).expect("new");
        assert_eq!(layer.mu.len(), 6);
        assert_eq!(layer.rho.len(), 6);
        assert_eq!(layer.in_dim, 3);
        assert_eq!(layer.out_dim, 2);
        assert_eq!(layer.len(), 6);
        assert!(!layer.is_empty());

        assert!(matches!(
            BayesianLinear::new(0, 2, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            BayesianLinear::new(2, 0, &mut rng),
            Err(SnnError::BadDim { .. })
        ));
    }

    // 10. posterior_sigma is always strictly positive.
    #[test]
    fn posterior_sigma_positive() {
        let mut rng = LcgRng::new(2);
        let mut layer = BayesianLinear::new(4, 3, &mut rng).expect("new");
        // Push rho to extreme negatives — softplus must still be > 0.
        for r in layer.rho.iter_mut() {
            *r = -50.0;
        }
        for &s in &layer.posterior_sigma() {
            assert!(s > 0.0, "sigma={s} must be > 0");
        }
    }

    // 11. KL = 0 when the whole layer is at the prior (μ=0, σ=σ_prior).
    #[test]
    fn layer_kl_zero_at_prior() {
        let mut rng = LcgRng::new(3);
        let mut layer = BayesianLinear::new(3, 3, &mut rng).expect("new");
        for m in layer.mu.iter_mut() {
            *m = 0.0;
        }
        // Solve softplus(ρ) = σ_prior.
        let target_sigma = layer.sigma_prior;
        let rho = (target_sigma.exp() - 1.0).ln();
        for r in layer.rho.iter_mut() {
            *r = rho;
        }
        let kl = layer.kl().expect("kl");
        assert!(kl.abs() < 1e-3, "layer KL at prior should be ≈0, got {kl}");
    }

    // 12. KL of layer is non-negative and increases when μ moves off zero.
    #[test]
    fn layer_kl_increases_away_from_prior() {
        let mut rng = LcgRng::new(4);
        let mut layer = BayesianLinear::new(2, 2, &mut rng).expect("new");
        for m in layer.mu.iter_mut() {
            *m = 0.0;
        }
        let target_sigma = layer.sigma_prior;
        let rho = (target_sigma.exp() - 1.0).ln();
        for r in layer.rho.iter_mut() {
            *r = rho;
        }
        let kl0 = layer.kl().expect("kl");
        for m in layer.mu.iter_mut() {
            *m = 1.0;
        }
        let kl1 = layer.kl().expect("kl");
        assert!(
            kl1 > kl0,
            "moving μ off the prior must raise KL: {kl0} -> {kl1}"
        );
        assert!(kl0 >= -1e-6 && kl1 >= -1e-6);
    }

    // 13. Sampling is deterministic given a seeded RNG.
    #[test]
    fn sampling_deterministic() {
        let mut rng_init = LcgRng::new(5);
        let layer = BayesianLinear::new(3, 4, &mut rng_init).expect("new");
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let wa = layer.sample(&mut rng_a);
        let wb = layer.sample_weights(&mut rng_b);
        assert_eq!(wa, wb, "same seed must give identical samples");
    }

    // 14. Sample mean over many draws ≈ μ.
    #[test]
    fn sample_mean_approaches_mu() {
        let mut rng_init = LcgRng::new(6);
        let mut layer = BayesianLinear::new(2, 2, &mut rng_init).expect("new");
        // Set known μ and a moderate σ.
        layer.mu = vec![0.5, -0.3, 1.0, 0.0];
        let sigma = 0.2_f32;
        let rho = (sigma.exp() - 1.0).ln();
        for r in layer.rho.iter_mut() {
            *r = rho;
        }
        let n_draws = 20_000;
        let mut rng = LcgRng::new(7);
        let mut accum = vec![0.0_f32; layer.len()];
        for _ in 0..n_draws {
            let w = layer.sample(&mut rng);
            for (a, &x) in accum.iter_mut().zip(w.iter()) {
                *a += x;
            }
        }
        for (a, &mu) in accum.iter().zip(layer.mu.iter()) {
            let mean = *a / n_draws as f32;
            assert!((mean - mu).abs() < 0.02, "empirical mean {mean} vs μ {mu}");
        }
    }

    // 15. kl_grads has correct shapes and matches per-element helpers.
    #[test]
    fn kl_grads_shapes_and_values() {
        let mut rng = LcgRng::new(8);
        let layer = BayesianLinear::new(3, 2, &mut rng).expect("new");
        let (g_mu, g_rho) = layer.kl_grads().expect("grads");
        assert_eq!(g_mu.len(), layer.len());
        assert_eq!(g_rho.len(), layer.len());
        for k in 0..layer.len() {
            let em = grad_mu(layer.mu[k], layer.sigma_prior).expect("gm");
            let er = grad_rho(layer.rho[k], layer.sigma_prior).expect("gr");
            assert!((g_mu[k] - em).abs() < 1e-6);
            assert!((g_rho[k] - er).abs() < 1e-6);
        }
    }
}
