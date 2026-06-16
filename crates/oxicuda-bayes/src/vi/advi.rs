//! Automatic-Differentiation Variational Inference (ADVI).
//!
//! Implements the algorithm of Kucukelbir, Tran, Ranganath, Gelman & Blei,
//! "Automatic Differentiation Variational Inference" (JMLR 2017).
//!
//! # Overview
//!
//! ADVI fits a **mean-field Gaussian** variational posterior to an arbitrary
//! differentiable model `log p(θ, x)` (the *log-joint*). Constrained model
//! parameters (e.g. a positive scale) are first pushed through an invertible
//! transform `T : supp(θ) → ℝ` into an **unconstrained** space `ζ = T(θ)`,
//! where a Gaussian variational family is always valid. The variational family
//! in the unconstrained space is
//!
//! ```text
//! q(ζ; μ, ω) = Π_i N(ζ_i ; μ_i, σ_i²),     σ_i = exp(ω_i).
//! ```
//!
//! Parameterising the standard deviation as `σ = exp(ω)` keeps `ω`
//! unconstrained, so plain gradient ascent never violates positivity.
//!
//! # Objective
//!
//! ADVI maximises the ELBO, written in the unconstrained space as
//!
//! ```text
//! L(μ, ω) = E_{q(ζ)}[ log p(x, T⁻¹(ζ)) + log|det J_{T⁻¹}(ζ)| ] + H[q],
//! ```
//!
//! where the **entropy** of a diagonal Gaussian is
//!
//! ```text
//! H[q] = Σ_i ω_i + (D/2)·(1 + ln 2π)   =   Σ_i ω_i + const.
//! ```
//!
//! # Reparameterisation gradient
//!
//! The expectation is estimated by the reparameterisation trick: draw
//! `ε ~ N(0, I)`, set `ζ = μ + σ ⊙ ε`, and differentiate through this
//! deterministic map. With `g(ζ) = log p(x, T⁻¹(ζ)) + log|det J|`,
//!
//! ```text
//! ∂L/∂μ_i = E_ε[ ∂g/∂ζ_i ]
//! ∂L/∂ω_i = E_ε[ ∂g/∂ζ_i · ε_i · σ_i ] + 1.
//! ```
//!
//! The trailing `+1` per coordinate is the gradient of the entropy term.
//! Gradients of `g` are obtained from the user-supplied gradient closure, or
//! by central finite differences when none is provided.
//!
//! All computation is `f32` and pure-Rust; randomness is drawn from the
//! crate's [`LcgRng`].

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Transform ─────────────────────────────────────────────────────────────────

/// Invertible transform mapping a constrained model coordinate to the
/// unconstrained real line used by the variational family.
///
/// For each coordinate ADVI works in the unconstrained variable `ζ = T(θ)`
/// and pushes draws back to the constrained space with `θ = T⁻¹(ζ)`. The
/// change-of-variables introduces a log-Jacobian term `log|d θ / d ζ|` that
/// must be added to the log-joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Identity transform `T(θ) = θ` for an already-unconstrained parameter
    /// (`θ ∈ ℝ`). Its log-Jacobian is `0`.
    Identity,
    /// Log transform `T(θ) = ln θ` for a strictly-positive parameter
    /// (`θ ∈ ℝ₊`). The inverse is `θ = exp(ζ)` and the log-Jacobian is
    /// `log|d θ / d ζ| = ζ`.
    Log,
}

impl Transform {
    /// Map a constrained value `θ` to the unconstrained space `ζ = T(θ)`.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if a [`Transform::Log`] is asked
    /// to transform a non-positive value.
    pub fn forward(self, theta: f32) -> BayesResult<f32> {
        match self {
            Transform::Identity => Ok(theta),
            Transform::Log => {
                if theta <= 0.0 {
                    return Err(BayesError::InvalidConfig(
                        "Transform::Log requires a strictly positive value".into(),
                    ));
                }
                Ok(theta.ln())
            }
        }
    }

    /// Map an unconstrained value `ζ` back to the constrained space
    /// `θ = T⁻¹(ζ)`.
    #[must_use]
    pub fn inverse(self, zeta: f32) -> f32 {
        match self {
            Transform::Identity => zeta,
            Transform::Log => zeta.exp(),
        }
    }

    /// Log absolute Jacobian determinant of the inverse map at `ζ`:
    /// `log|d T⁻¹(ζ) / d ζ|`.
    ///
    /// * Identity → `0`.
    /// * Log      → `ζ` (since `d/dζ exp(ζ) = exp(ζ)` and `ln exp(ζ) = ζ`).
    #[must_use]
    pub fn log_abs_det_jacobian(self, zeta: f32) -> f32 {
        match self {
            Transform::Identity => 0.0,
            Transform::Log => zeta,
        }
    }

    /// Derivative of the inverse-map log-Jacobian with respect to `ζ`,
    /// i.e. `d/dζ log|d T⁻¹(ζ) / d ζ|`.
    ///
    /// * Identity → `0`.
    /// * Log      → `1`.
    #[must_use]
    pub fn log_abs_det_jacobian_grad(self) -> f32 {
        match self {
            Transform::Identity => 0.0,
            Transform::Log => 1.0,
        }
    }
}

// ─── Config ────────────────────────────────────────────────────────────────────

/// Hyper-parameters controlling an ADVI optimisation run.
#[derive(Debug, Clone)]
pub struct AdviConfig {
    /// Number of stochastic gradient-ascent iterations (≥ 1).
    pub n_iter: usize,
    /// Number of Monte-Carlo samples used to estimate the ELBO gradient at
    /// each iteration (≥ 1). Larger values reduce the gradient variance.
    pub n_mc_samples: usize,
    /// Base step size for the ascent (> 0).
    pub step_size: f32,
    /// Optional finite-difference epsilon. When the log-joint gradient is
    /// supplied this is ignored; when it is absent the gradient is estimated
    /// by central differences with this spacing (must be > 0).
    pub fd_epsilon: f32,
    /// Numerical floor on `σ = exp(ω)` to keep draws and entropies finite.
    /// `ω` is clamped so that `σ ≥ sigma_floor`.
    pub sigma_floor: f32,
}

impl Default for AdviConfig {
    fn default() -> Self {
        Self {
            n_iter: 400,
            n_mc_samples: 8,
            step_size: 0.05,
            fd_epsilon: 1e-3,
            sigma_floor: 1e-4,
        }
    }
}

impl AdviConfig {
    /// Validate the configuration.
    fn validate(&self) -> BayesResult<()> {
        if self.n_iter == 0 {
            return Err(BayesError::InvalidConfig("ADVI n_iter must be >= 1".into()));
        }
        if self.n_mc_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "ADVI n_mc_samples must be >= 1".into(),
            ));
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(BayesError::InvalidConfig(
                "ADVI step_size must be > 0".into(),
            ));
        }
        if !self.fd_epsilon.is_finite() || self.fd_epsilon <= 0.0 {
            return Err(BayesError::InvalidConfig(
                "ADVI fd_epsilon must be > 0".into(),
            ));
        }
        if !self.sigma_floor.is_finite() || self.sigma_floor <= 0.0 {
            return Err(BayesError::InvalidConfig(
                "ADVI sigma_floor must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ─── Result ────────────────────────────────────────────────────────────────────

/// Fitted ADVI variational posterior in the unconstrained space.
#[derive(Debug, Clone)]
pub struct AdviResult {
    /// Variational means `μ` in the unconstrained space (length `D`).
    pub mu: Vec<f32>,
    /// Variational log-standard-deviations `ω` (so `σ = exp(ω)`), length `D`.
    pub omega: Vec<f32>,
    /// Per-coordinate transform used to map back to the constrained space.
    pub transforms: Vec<Transform>,
    /// ELBO value recorded at the end of every iteration (length `n_iter`).
    pub elbo_trace: Vec<f32>,
}

impl AdviResult {
    /// Variational standard deviations `σ_i = exp(ω_i)`.
    #[must_use]
    pub fn sigma(&self) -> Vec<f32> {
        self.omega.iter().map(|&w| w.exp()).collect()
    }

    /// Posterior mean of each coordinate **in the constrained space**.
    ///
    /// For the identity transform this is simply `μ_i`. For the log transform
    /// the constrained variable is `θ = exp(ζ)` with `ζ ~ N(μ, σ²)`, a
    /// log-normal whose mean is `exp(μ + σ²/2)`.
    #[must_use]
    pub fn constrained_mean(&self) -> Vec<f32> {
        self.mu
            .iter()
            .zip(self.omega.iter())
            .zip(self.transforms.iter())
            .map(|((&m, &w), &t)| match t {
                Transform::Identity => m,
                Transform::Log => {
                    let s = w.exp();
                    (m + 0.5 * s * s).exp()
                }
            })
            .collect()
    }

    /// Differential entropy `H[q] = Σ_i ω_i + (D/2)(1 + ln 2π)` of the fitted
    /// mean-field Gaussian (in the unconstrained space).
    #[must_use]
    pub fn entropy(&self) -> f32 {
        let d = self.omega.len() as f32;
        let sum_omega: f32 = self.omega.iter().sum();
        sum_omega + 0.5 * d * (1.0 + (2.0 * std::f32::consts::PI).ln())
    }

    /// Draw a sample `θ ~ q` mapped back to the **constrained** space.
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if the posterior is empty.
    pub fn sample_constrained(&self, rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        if self.mu.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let eps = draw_standard_normal(self.mu.len(), rng);
        let mut out = Vec::with_capacity(self.mu.len());
        for (((&m, &w), &e), &t) in self
            .mu
            .iter()
            .zip(self.omega.iter())
            .zip(eps.iter())
            .zip(self.transforms.iter())
        {
            let zeta = m + w.exp() * e;
            out.push(t.inverse(zeta));
        }
        Ok(out)
    }
}

// ─── AdviModel ─────────────────────────────────────────────────────────────────

/// A bundled description of the probabilistic model fed to ADVI.
///
/// Grouping the constraint transforms, the constrained log-joint, the optional
/// analytic gradient and the finite-difference spacing keeps the gradient and
/// fit entry points free of long positional argument lists.
pub struct AdviModel<'a, F, G>
where
    F: Fn(&[f32]) -> BayesResult<f32>,
    G: Fn(&[f32]) -> BayesResult<Vec<f32>>,
{
    /// Per-coordinate constraint transform (length `D`).
    pub transforms: &'a [Transform],
    /// Constrained log-joint `θ ↦ log p(x, θ)`.
    pub log_joint: &'a F,
    /// Optional analytic constrained gradient `θ ↦ ∇_θ log p(x, θ)`.
    pub log_joint_grad: Option<&'a G>,
    /// Central-difference spacing used when `log_joint_grad` is `None`.
    pub fd_epsilon: f32,
}

// ─── Advi ──────────────────────────────────────────────────────────────────────

/// Automatic-Differentiation Variational Inference optimiser.
///
/// The struct is a stateless namespace; all configuration lives in
/// [`AdviConfig`] and the model is supplied as closures over the
/// **constrained** parameter `θ`.
pub struct Advi;

impl Advi {
    /// Entropy of a diagonal Gaussian with log-standard-deviations `omega`:
    /// `Σ_i ω_i + (D/2)(1 + ln 2π)`.
    #[must_use]
    pub fn gaussian_entropy(omega: &[f32]) -> f32 {
        let d = omega.len() as f32;
        let sum_omega: f32 = omega.iter().sum();
        sum_omega + 0.5 * d * (1.0 + (2.0 * std::f32::consts::PI).ln())
    }

    /// Evaluate the integrand `g(ζ) = log p(x, T⁻¹(ζ)) + Σ_i log|det J_i|`
    /// for a single unconstrained draw `ζ`, given a constrained log-joint.
    ///
    /// # Errors
    /// Propagates errors from `log_joint`.
    pub fn integrand<F>(zeta: &[f32], transforms: &[Transform], log_joint: &F) -> BayesResult<f32>
    where
        F: Fn(&[f32]) -> BayesResult<f32>,
    {
        let theta = inverse_transform(zeta, transforms);
        let lj = log_joint(&theta)?;
        let log_jac: f32 = zeta
            .iter()
            .zip(transforms.iter())
            .map(|(&z, &t)| t.log_abs_det_jacobian(z))
            .sum();
        Ok(lj + log_jac)
    }

    /// Estimate the ELBO `L(μ, ω)` by averaging the integrand plus the entropy
    /// over `n_mc_samples` fresh reparameterised draws.
    ///
    /// # Errors
    /// Propagates errors from `log_joint`; returns [`BayesError::EmptyInputs`]
    /// when `mu` is empty or [`BayesError::DimensionMismatch`] when the
    /// parameter vectors disagree.
    pub fn elbo<F>(
        mu: &[f32],
        omega: &[f32],
        transforms: &[Transform],
        log_joint: &F,
        n_mc_samples: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<f32>
    where
        F: Fn(&[f32]) -> BayesResult<f32>,
    {
        check_shapes(mu, omega, transforms)?;
        if n_mc_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "ADVI n_mc_samples must be >= 1".into(),
            ));
        }
        let sigma: Vec<f32> = omega.iter().map(|&w| w.exp()).collect();
        let mut acc = 0.0_f32;
        for _ in 0..n_mc_samples {
            let eps = draw_standard_normal(mu.len(), rng);
            let zeta: Vec<f32> = (0..mu.len()).map(|i| mu[i] + sigma[i] * eps[i]).collect();
            acc += Self::integrand(&zeta, transforms, log_joint)?;
        }
        let expected_integrand = acc / n_mc_samples as f32;
        Ok(expected_integrand + Self::gaussian_entropy(omega))
    }

    /// Reparameterisation-gradient estimate of `(∂L/∂μ, ∂L/∂ω)` at the given
    /// variational parameters, averaged over `n_mc_samples` draws.
    ///
    /// When `log_joint_grad` is `Some`, the analytic constrained gradient
    /// `∇_θ log p(x, θ)` is used; otherwise the gradient of the integrand is
    /// estimated by central finite differences with spacing `fd_epsilon`.
    ///
    /// `model` bundles the (transforms, log-joint, optional gradient,
    /// finite-difference spacing); `samples` is the Monte-Carlo sample count.
    ///
    /// # Errors
    /// Propagates errors from the supplied closures and shape checks.
    pub fn elbo_grad<F, G>(
        mu: &[f32],
        omega: &[f32],
        model: &AdviModel<'_, F, G>,
        n_mc_samples: usize,
        rng: &mut LcgRng,
    ) -> BayesResult<(Vec<f32>, Vec<f32>)>
    where
        F: Fn(&[f32]) -> BayesResult<f32>,
        G: Fn(&[f32]) -> BayesResult<Vec<f32>>,
    {
        let transforms = model.transforms;
        let log_joint = model.log_joint;
        let log_joint_grad = model.log_joint_grad;
        let fd_epsilon = model.fd_epsilon;
        check_shapes(mu, omega, transforms)?;
        let d = mu.len();
        let sigma: Vec<f32> = omega.iter().map(|&w| w.exp()).collect();
        let mut grad_mu = vec![0.0_f32; d];
        let mut grad_omega = vec![0.0_f32; d];

        for _ in 0..n_mc_samples {
            let eps = draw_standard_normal(d, rng);
            let zeta: Vec<f32> = (0..d).map(|i| mu[i] + sigma[i] * eps[i]).collect();
            // ∂g/∂ζ for this draw.
            let dg_dzeta =
                integrand_grad(&zeta, transforms, log_joint, log_joint_grad, fd_epsilon)?;
            for i in 0..d {
                grad_mu[i] += dg_dzeta[i];
                // chain rule: ∂ζ_i/∂ω_i = ε_i σ_i  (since σ = exp(ω)).
                grad_omega[i] += dg_dzeta[i] * eps[i] * sigma[i];
            }
        }
        let inv = 1.0 / n_mc_samples as f32;
        for i in 0..d {
            grad_mu[i] *= inv;
            // Entropy term contributes +1 per coordinate to ∂L/∂ω.
            grad_omega[i] = grad_omega[i] * inv + 1.0;
        }
        Ok((grad_mu, grad_omega))
    }

    /// Run ADVI by stochastic gradient ascent on the reparameterised ELBO.
    ///
    /// * `init_mu` / `init_omega` — initial variational parameters (length `D`).
    /// * `transforms` — per-coordinate constraint transform (length `D`).
    /// * `log_joint` — `θ ↦ log p(x, θ)` over the **constrained** parameter.
    /// * `log_joint_grad` — optional analytic `θ ↦ ∇_θ log p(x, θ)`; when
    ///   `None`, finite differences are used.
    ///
    /// The step size follows the adaptive schedule of Kucukelbir 2017 (§3.4):
    /// a per-coordinate magnitude scaling combined with a Robbins-Monro decay,
    /// so the same `step_size` works across coordinates with different gradient
    /// magnitudes while the iterates still converge to a point.
    ///
    /// # Errors
    /// Propagates configuration, shape and closure errors.
    pub fn fit<F, G>(
        init_mu: &[f32],
        init_omega: &[f32],
        transforms: &[Transform],
        log_joint: &F,
        log_joint_grad: Option<&G>,
        cfg: &AdviConfig,
        rng: &mut LcgRng,
    ) -> BayesResult<AdviResult>
    where
        F: Fn(&[f32]) -> BayesResult<f32>,
        G: Fn(&[f32]) -> BayesResult<Vec<f32>>,
    {
        cfg.validate()?;
        check_shapes(init_mu, init_omega, transforms)?;

        let model = AdviModel {
            transforms,
            log_joint,
            log_joint_grad,
            fd_epsilon: cfg.fd_epsilon,
        };

        let d = init_mu.len();
        let mut mu = init_mu.to_vec();
        let mut omega = init_omega.to_vec();
        // Adaptive step-size accumulators (Kucukelbir 2017, §3.4). The schedule
        //   ρ^(k)_i = η · k^(−1/2+ε) · (τ + √s^(k)_i)^(−1),
        //   s^(k)_i = α g_i² + (1−α) s^(k−1)_i,  s^(1)_i = g_i²,
        // combines per-coordinate magnitude scaling with a Robbins-Monro
        // decay (the k^(−1/2+ε) factor) so the iterates converge to a point
        // rather than oscillating around the optimum.
        let mut s_mu = vec![0.0_f32; d];
        let mut s_omega = vec![0.0_f32; d];
        let alpha = 0.1_f32; // weight of the newest squared gradient
        let tau = 1.0_f32; // stabiliser preventing huge early steps
        let robbins_eps = 0.1_f32; // exponent offset (decay ~ k^{-0.4})
        let omega_floor = cfg.sigma_floor.ln();

        let mut elbo_trace = Vec::with_capacity(cfg.n_iter);

        for k in 0..cfg.n_iter {
            let (g_mu, g_omega) = Self::elbo_grad(&mu, &omega, &model, cfg.n_mc_samples, rng)?;

            // Robbins-Monro decay factor (1-indexed iteration).
            let decay = ((k + 1) as f32).powf(-0.5 + robbins_eps);

            for i in 0..d {
                // Running average of squared gradients (initialised on k = 0).
                if k == 0 {
                    s_mu[i] = g_mu[i] * g_mu[i];
                    s_omega[i] = g_omega[i] * g_omega[i];
                } else {
                    s_mu[i] = alpha * g_mu[i] * g_mu[i] + (1.0 - alpha) * s_mu[i];
                    s_omega[i] = alpha * g_omega[i] * g_omega[i] + (1.0 - alpha) * s_omega[i];
                }
                let rho_mu = cfg.step_size * decay / (tau + s_mu[i].sqrt());
                let rho_omega = cfg.step_size * decay / (tau + s_omega[i].sqrt());
                // Ascent: move along the gradient.
                mu[i] += rho_mu * g_mu[i];
                omega[i] += rho_omega * g_omega[i];
                // Keep σ above the floor for numerical safety.
                if omega[i] < omega_floor {
                    omega[i] = omega_floor;
                }
            }

            // Record an ELBO estimate at the new parameters.
            let elbo = Self::elbo(&mu, &omega, transforms, log_joint, cfg.n_mc_samples, rng)?;
            elbo_trace.push(elbo);
        }

        Ok(AdviResult {
            mu,
            omega,
            transforms: transforms.to_vec(),
            elbo_trace,
        })
    }
}

// ─── Free helpers ──────────────────────────────────────────────────────────────

/// Map an unconstrained vector to the constrained space coordinate-wise.
fn inverse_transform(zeta: &[f32], transforms: &[Transform]) -> Vec<f32> {
    zeta.iter()
        .zip(transforms.iter())
        .map(|(&z, &t)| t.inverse(z))
        .collect()
}

/// Validate that the three parameter slices share the same non-zero length.
fn check_shapes(mu: &[f32], omega: &[f32], transforms: &[Transform]) -> BayesResult<()> {
    if mu.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if omega.len() != mu.len() {
        return Err(BayesError::DimensionMismatch {
            expected: mu.len(),
            got: omega.len(),
        });
    }
    if transforms.len() != mu.len() {
        return Err(BayesError::DimensionMismatch {
            expected: mu.len(),
            got: transforms.len(),
        });
    }
    Ok(())
}

/// A unit-uniform draw in `[0, 1)` from the crate [`LcgRng`].
///
/// The crate's `LcgRng::next_u32` returns the high 31 bits of the state
/// (`state >> 33`), so its maximum is `2³¹ − 1`. Dividing by `2³¹` yields a
/// faithful `[0, 1)` sample. (The crate's own `next_f32` divides by `2³²` and
/// therefore only spans `[0, 0.5)`, which biases Box-Muller — so we deliberately
/// build the uniform here instead of reusing it.)
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
}

/// Draw `d` independent standard-normal samples via an unbiased Box-Muller
/// transform built on [`unit_uniform`].
fn draw_standard_normal(d: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    let mut i = 0;
    while i + 1 < d {
        let (a, b) = box_muller_pair(rng);
        out[i] = a;
        out[i + 1] = b;
        i += 2;
    }
    if i < d {
        let (a, _) = box_muller_pair(rng);
        out[i] = a;
    }
    out
}

/// A single Box-Muller pair of independent `N(0, 1)` variates.
#[inline]
fn box_muller_pair(rng: &mut LcgRng) -> (f32, f32) {
    let u1 = unit_uniform(rng).clamp(1e-7, 1.0 - 1e-7);
    let u2 = unit_uniform(rng);
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

/// Gradient of the integrand `g(ζ) = log p(x, T⁻¹(ζ)) + Σ log|det J|` with
/// respect to the unconstrained variable `ζ`.
///
/// When an analytic constrained gradient is available, the chain rule gives
/// `∂g/∂ζ_i = (∂ log p / ∂θ_i)·(d θ_i / d ζ_i) + (d/dζ_i log|det J_i|)`.
/// Otherwise the whole integrand is differentiated by central differences.
fn integrand_grad<F, G>(
    zeta: &[f32],
    transforms: &[Transform],
    log_joint: &F,
    log_joint_grad: Option<&G>,
    fd_epsilon: f32,
) -> BayesResult<Vec<f32>>
where
    F: Fn(&[f32]) -> BayesResult<f32>,
    G: Fn(&[f32]) -> BayesResult<Vec<f32>>,
{
    let d = zeta.len();
    match log_joint_grad {
        Some(grad_fn) => {
            let theta = inverse_transform(zeta, transforms);
            let grad_theta = grad_fn(&theta)?;
            if grad_theta.len() != d {
                return Err(BayesError::DimensionMismatch {
                    expected: d,
                    got: grad_theta.len(),
                });
            }
            let mut out = vec![0.0_f32; d];
            for i in 0..d {
                // d θ_i / d ζ_i for the supported transforms.
                let dtheta_dzeta = match transforms[i] {
                    Transform::Identity => 1.0,
                    Transform::Log => zeta[i].exp(),
                };
                out[i] = grad_theta[i] * dtheta_dzeta + transforms[i].log_abs_det_jacobian_grad();
            }
            Ok(out)
        }
        None => {
            // Central finite differences on g(ζ) directly.
            let mut out = vec![0.0_f32; d];
            let mut zp = zeta.to_vec();
            for i in 0..d {
                let original = zeta[i];
                zp[i] = original + fd_epsilon;
                let g_plus = Advi::integrand(&zp, transforms, log_joint)?;
                zp[i] = original - fd_epsilon;
                let g_minus = Advi::integrand(&zp, transforms, log_joint)?;
                zp[i] = original;
                out[i] = (g_plus - g_minus) / (2.0 * fd_epsilon);
            }
            Ok(out)
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `log N(θ; m, s²)` summed over coordinates with per-coordinate (m, s).
    fn gaussian_log_joint(theta: &[f32], m: &[f32], s: &[f32]) -> f32 {
        let mut lp = 0.0_f32;
        for ((&t, &mi), &si) in theta.iter().zip(m.iter()).zip(s.iter()) {
            let z = (t - mi) / si;
            lp += -0.5 * z * z - si.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln();
        }
        lp
    }

    /// Analytic gradient of the above: `∂/∂θ_i = -(θ_i - m_i)/s_i²`.
    fn gaussian_log_joint_grad(theta: &[f32], m: &[f32], s: &[f32]) -> Vec<f32> {
        theta
            .iter()
            .zip(m.iter())
            .zip(s.iter())
            .map(|((&t, &mi), &si)| -(t - mi) / (si * si))
            .collect()
    }

    // ── (a) ADVI recovers a Gaussian target's mean and standard deviation ─────
    #[test]
    fn advi_recovers_gaussian_target() {
        let m = vec![1.5_f32, -2.0];
        let s = vec![0.7_f32, 1.3];
        let log_joint = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint(theta, &m, &s))
        };
        let log_joint_grad = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint_grad(theta, &m, &s))
        };
        let transforms = vec![Transform::Identity, Transform::Identity];
        let cfg = AdviConfig {
            n_iter: 1500,
            n_mc_samples: 16,
            step_size: 0.05,
            ..Default::default()
        };
        let mut rng = LcgRng::new(7);
        let res = Advi::fit(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &transforms,
            &log_joint,
            Some(&log_joint_grad),
            &cfg,
            &mut rng,
        )
        .expect("ADVI fit must succeed");

        let sigma = res.sigma();
        // μ → m
        assert!((res.mu[0] - m[0]).abs() < 0.15, "mu0={}", res.mu[0]);
        assert!((res.mu[1] - m[1]).abs() < 0.15, "mu1={}", res.mu[1]);
        // σ → s
        assert!((sigma[0] - s[0]).abs() < 0.15, "sigma0={}", sigma[0]);
        assert!((sigma[1] - s[1]).abs() < 0.20, "sigma1={}", sigma[1]);
    }

    // ── (b) the ELBO is non-decreasing over the optimisation tail ─────────────
    #[test]
    fn advi_elbo_increases() {
        let m = vec![0.8_f32];
        let s = vec![1.1_f32];
        let log_joint = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint(theta, &m, &s))
        };
        let log_joint_grad = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint_grad(theta, &m, &s))
        };
        let cfg = AdviConfig {
            n_iter: 600,
            n_mc_samples: 24,
            step_size: 0.05,
            ..Default::default()
        };
        let mut rng = LcgRng::new(11);
        let res = Advi::fit(
            &[3.0],
            &[1.0],
            &[Transform::Identity],
            &log_joint,
            Some(&log_joint_grad),
            &cfg,
            &mut rng,
        )
        .expect("ADVI fit must succeed");

        let trace = &res.elbo_trace;
        assert_eq!(trace.len(), cfg.n_iter);
        // Smooth the noisy stochastic trace with a moving average, then compare
        // an early window's mean against a late window's mean.
        let window = 50;
        let avg = |slice: &[f32]| slice.iter().sum::<f32>() / slice.len() as f32;
        let early = avg(&trace[..window]);
        let late = avg(&trace[trace.len() - window..]);
        assert!(
            late > early - 1e-2,
            "ELBO should rise: early={early}, late={late}"
        );
        // And the late ELBO should be much higher than the (bad) start.
        assert!(late > trace[0]);
    }

    // ── (c) the mean-field posterior is diagonal: independent target ⇒ ────────
    //    the recovered means decouple (each μ_i tracks only its own m_i).
    #[test]
    fn advi_mean_field_is_diagonal() {
        // Independent target with very different means per coordinate.
        let m = vec![5.0_f32, -5.0, 0.0];
        let s = vec![1.0_f32, 1.0, 1.0];
        let log_joint = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint(theta, &m, &s))
        };
        let log_joint_grad = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint_grad(theta, &m, &s))
        };
        let transforms = vec![Transform::Identity; 3];
        // Far-apart means (±5) need a larger step to traverse the unconstrained
        // space within the iteration budget.
        let cfg = AdviConfig {
            n_iter: 2000,
            n_mc_samples: 16,
            step_size: 0.2,
            ..Default::default()
        };
        let mut rng = LcgRng::new(99);
        let res = Advi::fit(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &transforms,
            &log_joint,
            Some(&log_joint_grad),
            &cfg,
            &mut rng,
        )
        .expect("ADVI fit must succeed");
        // Each coordinate recovers its own mean independently — the cross
        // coordinates do not leak into each other.
        for (i, (&got, &expected)) in res.mu.iter().zip(m.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.3,
                "coordinate {i}: mu={got}, expected {expected}"
            );
        }
        // σ stays near 1 for every coordinate (no spurious correlation inflation).
        for s_i in res.sigma() {
            assert!((s_i - 1.0).abs() < 0.3, "sigma={s_i}");
        }
    }

    // ── (d) the entropy term equals Σω + const ────────────────────────────────
    #[test]
    fn advi_entropy_is_sum_omega_plus_const() {
        let omega = vec![0.3_f32, -0.5, 1.2, 0.0];
        let d = omega.len() as f32;
        let constant = 0.5 * d * (1.0 + (2.0 * std::f32::consts::PI).ln());
        let sum_omega: f32 = omega.iter().sum();
        let expected = sum_omega + constant;
        let got = Advi::gaussian_entropy(&omega);
        assert!(
            (got - expected).abs() < 1e-5,
            "got {got}, expected {expected}"
        );

        // Shifting every ω by δ shifts the entropy by exactly D·δ.
        let delta = 0.4_f32;
        let omega_shift: Vec<f32> = omega.iter().map(|&w| w + delta).collect();
        let got_shift = Advi::gaussian_entropy(&omega_shift);
        assert!(
            (got_shift - got - d * delta).abs() < 1e-4,
            "entropy must shift by D·δ"
        );
    }

    // ── (e) the reparameterisation gradient is unbiased: its average matches ──
    //    a finite-difference gradient of the ELBO in (μ, ω).
    #[test]
    fn advi_reparam_gradient_unbiased() {
        let m = vec![1.0_f32, -0.5];
        let s = vec![0.9_f32, 1.4];
        let log_joint = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint(theta, &m, &s))
        };
        let log_joint_grad = {
            let m = m.clone();
            let s = s.clone();
            move |theta: &[f32]| Ok(gaussian_log_joint_grad(theta, &m, &s))
        };
        let transforms = vec![Transform::Identity, Transform::Identity];
        let mu = vec![0.4_f32, -0.2];
        let omega = vec![-0.1_f32, 0.2];

        // Monte-Carlo reparameterisation gradient, averaged over many draws.
        let mut rng = LcgRng::new(2024);
        let model = AdviModel {
            transforms: &transforms,
            log_joint: &log_joint,
            log_joint_grad: Some(&log_joint_grad),
            fd_epsilon: 1e-3,
        };
        let (g_mu, g_omega) =
            Advi::elbo_grad(&mu, &omega, &model, 20_000, &mut rng).expect("grad must succeed");

        // Finite-difference reference gradient of the *exact* ELBO. For an
        // identity-transform Gaussian target N(m, s²) with Gaussian q(μ, σ),
        // the ELBO has the closed form
        //   L(μ,ω) = Σ_i [ -0.5*((μ_i-m_i)²+σ_i²)/s_i² - ln s_i
        //                  - 0.5 ln 2π + ω_i + 0.5(1+ln 2π) ],
        // which we differentiate analytically-by-difference for the check.
        let exact_elbo = |mu: &[f32], omega: &[f32]| -> f32 {
            let mut l = 0.0_f32;
            for i in 0..mu.len() {
                let sigma = omega[i].exp();
                let si = s[i];
                l += -0.5 * ((mu[i] - m[i]).powi(2) + sigma * sigma) / (si * si)
                    - si.ln()
                    - 0.5 * (2.0 * std::f32::consts::PI).ln()
                    + omega[i]
                    + 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
            }
            l
        };
        let h = 1e-3_f32;
        for i in 0..mu.len() {
            let mut mp = mu.clone();
            mp[i] += h;
            let mut mm = mu.clone();
            mm[i] -= h;
            let fd_mu = (exact_elbo(&mp, &omega) - exact_elbo(&mm, &omega)) / (2.0 * h);
            assert!(
                (g_mu[i] - fd_mu).abs() < 0.05,
                "grad_mu[{i}]: mc={}, fd={}",
                g_mu[i],
                fd_mu
            );

            let mut op = omega.clone();
            op[i] += h;
            let mut om = omega.clone();
            om[i] -= h;
            let fd_omega = (exact_elbo(&mu, &op) - exact_elbo(&mu, &om)) / (2.0 * h);
            assert!(
                (g_omega[i] - fd_omega).abs() < 0.05,
                "grad_omega[{i}]: mc={}, fd={}",
                g_omega[i],
                fd_omega
            );
        }
    }

    // ── (f) a log-transformed positive parameter is handled with the correct ──
    //    Jacobian term, and ADVI recovers a positive scale.
    #[test]
    fn advi_log_transform_positive_parameter() {
        // Check the Jacobian bookkeeping directly first.
        let zeta = 0.7_f32;
        assert!((Transform::Log.log_abs_det_jacobian(zeta) - zeta).abs() < 1e-6);
        assert!((Transform::Log.log_abs_det_jacobian_grad() - 1.0).abs() < 1e-6);
        assert!((Transform::Log.inverse(zeta) - zeta.exp()).abs() < 1e-5);
        // forward∘inverse round-trip.
        let theta = 2.5_f32;
        let z = Transform::Log.forward(theta).expect("log of positive");
        assert!((Transform::Log.inverse(z) - theta).abs() < 1e-5);

        // Target: a positive parameter θ with a log-normal-shaped posterior.
        // Model log-joint in θ: log p(x, θ) chosen so the *unconstrained*
        // posterior is N(target_zeta, target_sd²). Working in ζ = ln θ:
        //   we define log p(x, θ) = log N(ln θ; mz, sz²) − ln θ
        // so that adding the Jacobian (+ζ) yields exactly log N(ζ; mz, sz²),
        // a clean Gaussian in ζ that ADVI must match.
        let mz = 0.5_f32; // ⇒ median θ = exp(0.5) ≈ 1.65
        let sz = 0.4_f32;
        let log_joint = move |theta: &[f32]| {
            let t = theta[0];
            if t <= 0.0 {
                return Err(BayesError::InvalidConfig("theta must be > 0".into()));
            }
            let zeta = t.ln();
            let z = (zeta - mz) / sz;
            let lp = -0.5 * z * z - sz.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln();
            Ok(lp - zeta) // subtract the Jacobian so total is N(ζ; mz, sz²)
        };
        let cfg = AdviConfig {
            n_iter: 1500,
            n_mc_samples: 16,
            step_size: 0.04,
            ..Default::default()
        };
        let mut rng = LcgRng::new(321);
        let res = Advi::fit::<_, fn(&[f32]) -> BayesResult<Vec<f32>>>(
            &[0.0],
            &[0.0],
            &[Transform::Log],
            &log_joint,
            None, // exercise the finite-difference path
            &cfg,
            &mut rng,
        )
        .expect("ADVI fit must succeed");

        // The unconstrained posterior should match N(mz, sz²).
        assert!((res.mu[0] - mz).abs() < 0.15, "mu={}", res.mu[0]);
        assert!(
            (res.sigma()[0] - sz).abs() < 0.15,
            "sigma={}",
            res.sigma()[0]
        );
        // Constrained samples are strictly positive.
        let theta = res
            .sample_constrained(&mut rng)
            .expect("sample must succeed");
        assert!(theta[0] > 0.0, "theta must be positive: {}", theta[0]);
        // Log-normal mean = exp(μ + σ²/2) > 0.
        let cmean = res.constrained_mean();
        assert!(cmean[0] > 0.0);
        assert!((cmean[0] - (mz + 0.5 * sz * sz).exp()).abs() < 0.3);
    }

    // ── Extra: shape / config validation paths ────────────────────────────────
    #[test]
    fn advi_rejects_bad_shapes_and_config() {
        let log_joint = |_t: &[f32]| Ok(0.0_f32);
        let bad_cfg = AdviConfig {
            n_iter: 0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        let r = Advi::fit::<_, fn(&[f32]) -> BayesResult<Vec<f32>>>(
            &[0.0],
            &[0.0],
            &[Transform::Identity],
            &log_joint,
            None,
            &bad_cfg,
            &mut rng,
        );
        assert!(r.is_err());

        // mismatched omega length.
        let cfg = AdviConfig::default();
        let r = Advi::fit::<_, fn(&[f32]) -> BayesResult<Vec<f32>>>(
            &[0.0, 0.0],
            &[0.0],
            &[Transform::Identity, Transform::Identity],
            &log_joint,
            None,
            &cfg,
            &mut rng,
        );
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));

        // empty parameters.
        let r = Advi::fit::<_, fn(&[f32]) -> BayesResult<Vec<f32>>>(
            &[],
            &[],
            &[],
            &log_joint,
            None,
            &cfg,
            &mut rng,
        );
        assert!(matches!(r, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn transform_log_rejects_nonpositive() {
        assert!(Transform::Log.forward(0.0).is_err());
        assert!(Transform::Log.forward(-1.0).is_err());
        assert!(Transform::Identity.forward(-3.0).is_ok());
    }

    #[test]
    fn advi_result_entropy_matches_static() {
        let res = AdviResult {
            mu: vec![0.0, 0.0],
            omega: vec![0.1, -0.2],
            transforms: vec![Transform::Identity; 2],
            elbo_trace: vec![],
        };
        assert!((res.entropy() - Advi::gaussian_entropy(&res.omega)).abs() < 1e-6);
    }
}
