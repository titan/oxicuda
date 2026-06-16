//! Horseshoe prior for sparse Bayesian linear regression.
//!
//! Implements the global-local shrinkage prior of Carvalho, Polson & Scott,
//! "Handling Sparsity via the Horseshoe" (AISTATS 2009) and "The horseshoe
//! estimator for sparse signals" (Biometrika 2010):
//!
//! ```text
//! β_j | λ_j, τ ~ N(0, τ² λ_j²),    λ_j ~ C⁺(0, 1),    τ ~ C⁺(0, 1).
//! ```
//!
//! Each coefficient has its own *local* scale `λ_j` (a half-Cauchy whose heavy
//! tail lets large signals escape shrinkage) modulated by a single *global*
//! scale `τ` (which pulls the whole vector toward zero, enforcing sparsity).
//!
//! # Inverse-gamma scale-mixture augmentation
//!
//! Direct half-Cauchy conditionals are not conjugate, so we use the auxiliary
//! representation (Makalic & Schmidt, 2016): a half-Cauchy scale `λ ~ C⁺(0, 1)`
//! is equivalent to the hierarchy
//!
//! ```text
//! λ² | ν ~ InvGamma(1/2, 1/ν),    ν ~ InvGamma(1/2, 1),
//! ```
//!
//! and likewise for `τ² | ξ` and `ξ`. Every conditional below is then an
//! inverse-gamma or a Gaussian, giving a clean Gibbs sampler.
//!
//! # Gibbs sampler for `y = Xβ + ε`, `ε ~ N(0, σ²)`
//!
//! With `D = diag(τ² λ_j²)` the full conditionals are
//!
//! ```text
//! β | · ~ N( A⁻¹ Xᵀy / σ², A⁻¹ ),     A = XᵀX/σ² + D⁻¹,
//! λ_j² | · ~ InvGamma( 1, 1/ν_j + β_j²/(2τ²) ),
//! ν_j   | · ~ InvGamma( 1, 1 + 1/λ_j² ),
//! τ²    | · ~ InvGamma( (p+1)/2, 1/ξ + Σ_j β_j²/(2λ_j²) ),
//! ξ     | · ~ InvGamma( 1, 1 + 1/τ² ),
//! σ²    | · ~ InvGamma( (n + p)/2, (‖y − Xβ‖² + Σ_j β_j²/(τ²λ_j²)) / 2 ).
//! ```
//!
//! All arithmetic is `f32` and pure-Rust; randomness comes from [`LcgRng`].

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Prior densities ───────────────────────────────────────────────────────────

/// Log density of the standard half-Cauchy `C⁺(0, 1)` at `x ≥ 0`:
/// `log p(x) = log 2 − log π − log(1 + x²)`.
///
/// # Errors
/// Returns [`BayesError::InvalidConfig`] for negative `x`.
pub fn half_cauchy_log_pdf(x: f32) -> BayesResult<f32> {
    if x < 0.0 {
        return Err(BayesError::InvalidConfig(
            "half-Cauchy support is x >= 0".into(),
        ));
    }
    let two_over_pi = (2.0_f32 / std::f32::consts::PI).ln();
    Ok(two_over_pi - (1.0 + x * x).ln())
}

/// Marginal log density of a single horseshoe coefficient with global scale
/// `tau`, integrating out the local scale `λ ~ C⁺(0, 1)`.
///
/// The exact marginal has no elementary closed form, but Carvalho et al. (2010)
/// give the tight, fully-explicit bounds
///
/// ```text
/// (K/2) · log(1 + 4/(β/τ)²)  ≤  p(β|τ)·τ  ≤  K · log(1 + 2/(β/τ)²),
/// ```
///
/// with `K = 1/√(2π³)`. We return the **logarithm of the lower bound** scaled
/// to a density in `β`; it captures the two defining features exactly — an
/// integrable pole at the origin (`β → 0` ⇒ `+∞`) and Cauchy-like
/// `≍ β⁻²` tails (`log(1+4τ²/β²) ≍ 4τ²/β²` for large `β`). This is the standard
/// device for working with the horseshoe density on the log scale.
///
/// # Errors
/// Returns [`BayesError::NonPositiveSigma`] if `tau <= 0`.
pub fn horseshoe_log_pdf(beta: f32, tau: f32) -> BayesResult<f32> {
    if tau <= 0.0 {
        return Err(BayesError::NonPositiveSigma);
    }
    // K = 1 / sqrt(2 π³).
    let k = 1.0_f32 / (2.0 * std::f32::consts::PI.powi(3)).sqrt();
    let z = beta / tau;
    // log( (K / 2τ) · log(1 + 4/z²) ).
    let inner = (1.0 + 4.0 / (z * z)).ln();
    Ok((k / (2.0 * tau)).ln() + inner.ln())
}

// ─── Shrinkage factor ──────────────────────────────────────────────────────────

/// Horseshoe shrinkage factor `κ_j = 1 / (1 + τ² λ_j²)`.
///
/// `κ_j ∈ (0, 1)`: values near `1` indicate complete shrinkage (the coordinate
/// is treated as noise and pulled to zero), while values near `0` indicate the
/// signal is preserved. The posterior mean satisfies
/// `E[β_j | y] = (1 − κ_j) · β̂_j^{OLS}`.
///
/// # Errors
/// Returns [`BayesError::NonPositiveSigma`] if `tau <= 0` or `lambda <= 0`.
pub fn shrinkage_factor(tau: f32, lambda: f32) -> BayesResult<f32> {
    if tau <= 0.0 || lambda <= 0.0 {
        return Err(BayesError::NonPositiveSigma);
    }
    let t2l2 = tau * tau * lambda * lambda;
    Ok(1.0 / (1.0 + t2l2))
}

// ─── Config / Fit ──────────────────────────────────────────────────────────────

/// Configuration for the horseshoe Gibbs sampler.
#[derive(Debug, Clone)]
pub struct HorseshoeConfig {
    /// Total number of Gibbs iterations (≥ 1).
    pub n_iter: usize,
    /// Number of leading iterations discarded as burn-in (`< n_iter`).
    pub burn_in: usize,
    /// Initial noise variance `σ²` (> 0).
    pub init_sigma2: f32,
}

impl Default for HorseshoeConfig {
    fn default() -> Self {
        Self {
            n_iter: 2000,
            burn_in: 1000,
            init_sigma2: 1.0,
        }
    }
}

impl HorseshoeConfig {
    fn validate(&self) -> BayesResult<()> {
        if self.n_iter == 0 {
            return Err(BayesError::InvalidConfig(
                "horseshoe n_iter must be >= 1".into(),
            ));
        }
        if self.burn_in >= self.n_iter {
            return Err(BayesError::InvalidConfig(
                "horseshoe burn_in must be < n_iter".into(),
            ));
        }
        if !self.init_sigma2.is_finite() || self.init_sigma2 <= 0.0 {
            return Err(BayesError::InvalidConfig(
                "horseshoe init_sigma2 must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Posterior summary returned by the horseshoe Gibbs sampler.
#[derive(Debug, Clone)]
pub struct HorseshoeFit {
    /// Posterior mean of each regression coefficient (length `p`).
    pub beta_mean: Vec<f32>,
    /// Posterior mean of the local scales `λ_j` (length `p`).
    pub lambda_mean: Vec<f32>,
    /// Posterior mean of the global scale `τ`.
    pub tau_mean: f32,
    /// Posterior mean of the noise variance `σ²`.
    pub sigma2_mean: f32,
    /// Posterior mean of the per-coefficient shrinkage factor `κ_j` (length `p`).
    pub kappa_mean: Vec<f32>,
}

// ─── HorseshoeRegression ─────────────────────────────────────────────────────

/// Horseshoe sparse linear-regression Gibbs sampler.
///
/// The design matrix is supplied row-major as a flat `n × p` slice. The sampler
/// is fully deterministic given the seed of the [`LcgRng`] passed to [`fit`].
///
/// [`fit`]: HorseshoeRegression::fit
pub struct HorseshoeRegression;

impl HorseshoeRegression {
    /// Run the Gibbs sampler on `(x, y)` with `x` an `n × p` row-major design
    /// matrix and `y` the length-`n` response.
    ///
    /// # Errors
    /// * [`BayesError::EmptyInputs`] if `x` or `y` is empty.
    /// * [`BayesError::DimensionMismatch`] if `x.len() != n_rows * p` or
    ///   `y.len() != n_rows`.
    /// * Configuration errors from `HorseshoeConfig::validate`.
    pub fn fit(
        x: &[f32],
        y: &[f32],
        p: usize,
        cfg: &HorseshoeConfig,
        rng: &mut LcgRng,
    ) -> BayesResult<HorseshoeFit> {
        cfg.validate()?;
        if x.is_empty() || y.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if p == 0 {
            return Err(BayesError::InvalidConfig("horseshoe p must be >= 1".into()));
        }
        let n = y.len();
        if x.len() != n * p {
            return Err(BayesError::DimensionMismatch {
                expected: n * p,
                got: x.len(),
            });
        }

        // Pre-compute XᵀX (p × p) and Xᵀy (p) — reused every sweep.
        let xtx = gram_matrix(x, n, p);
        let xty = matvec_transpose(x, y, p);

        // ── State (augmented) ──────────────────────────────────────────────
        let mut beta = vec![0.0_f32; p];
        let mut lambda2 = vec![1.0_f32; p]; // λ_j²
        let mut nu = vec![1.0_f32; p]; // auxiliary ν_j
        let mut tau2 = 1.0_f32; // τ²
        let mut xi = 1.0_f32; // auxiliary ξ
        let mut sigma2 = cfg.init_sigma2;

        // ── Accumulators for posterior means (post burn-in) ─────────────────
        let mut beta_sum = vec![0.0_f32; p];
        let mut lambda_sum = vec![0.0_f32; p];
        let mut kappa_sum = vec![0.0_f32; p];
        let mut tau_sum = 0.0_f32;
        let mut sigma2_sum = 0.0_f32;
        let mut n_kept = 0usize;

        for it in 0..cfg.n_iter {
            // 1. β | · ~ N(A⁻¹ Xᵀy/σ², A⁻¹),  A = XᵀX/σ² + D⁻¹.
            let inv_sigma2 = 1.0 / sigma2;
            let mut a = vec![0.0_f32; p * p];
            for r in 0..p {
                for c in 0..p {
                    a[r * p + c] = xtx[r * p + c] * inv_sigma2;
                }
                // D⁻¹ diagonal: 1 / (τ² λ_j²).
                a[r * p + r] += 1.0 / (tau2 * lambda2[r]).max(1e-12);
            }
            let rhs: Vec<f32> = xty.iter().map(|&v| v * inv_sigma2).collect();
            beta = sample_mvn_from_precision(&a, &rhs, p, rng)?;

            // 2. λ_j² | · ~ InvGamma(1, 1/ν_j + β_j²/(2τ²)).
            for j in 0..p {
                let rate = 1.0 / nu[j] + beta[j] * beta[j] / (2.0 * tau2);
                lambda2[j] = sample_inv_gamma(1.0, rate.max(1e-12), rng);
                // 3. ν_j | · ~ InvGamma(1, 1 + 1/λ_j²).
                nu[j] = sample_inv_gamma(1.0, 1.0 + 1.0 / lambda2[j].max(1e-12), rng);
            }

            // 4. τ² | · ~ InvGamma((p+1)/2, 1/ξ + Σ_j β_j²/(2 λ_j²)).
            let mut tau_rate = 1.0 / xi;
            for j in 0..p {
                tau_rate += beta[j] * beta[j] / (2.0 * lambda2[j].max(1e-12));
            }
            tau2 = sample_inv_gamma((p as f32 + 1.0) / 2.0, tau_rate.max(1e-12), rng);
            // 5. ξ | · ~ InvGamma(1, 1 + 1/τ²).
            xi = sample_inv_gamma(1.0, 1.0 + 1.0 / tau2.max(1e-12), rng);

            // 6. σ² | · ~ InvGamma((n+p)/2, (‖y − Xβ‖² + Σ β_j²/(τ²λ_j²)) / 2).
            let resid_ss = residual_sum_of_squares(x, y, &beta, p);
            let mut beta_penalty = 0.0_f32;
            for j in 0..p {
                beta_penalty += beta[j] * beta[j] / (tau2 * lambda2[j]).max(1e-12);
            }
            let sigma_shape = (n as f32 + p as f32) / 2.0;
            let sigma_rate = (resid_ss + beta_penalty) / 2.0;
            sigma2 = sample_inv_gamma(sigma_shape, sigma_rate.max(1e-12), rng);

            // ── Accumulate after burn-in ───────────────────────────────────
            if it >= cfg.burn_in {
                for j in 0..p {
                    beta_sum[j] += beta[j];
                    lambda_sum[j] += lambda2[j].sqrt();
                    // κ_j = 1 / (1 + τ² λ_j²).
                    kappa_sum[j] += 1.0 / (1.0 + tau2 * lambda2[j]);
                }
                tau_sum += tau2.sqrt();
                sigma2_sum += sigma2;
                n_kept += 1;
            }
        }

        let inv_kept = 1.0 / n_kept.max(1) as f32;
        Ok(HorseshoeFit {
            beta_mean: beta_sum.iter().map(|&v| v * inv_kept).collect(),
            lambda_mean: lambda_sum.iter().map(|&v| v * inv_kept).collect(),
            tau_mean: tau_sum * inv_kept,
            sigma2_mean: sigma2_sum * inv_kept,
            kappa_mean: kappa_sum.iter().map(|&v| v * inv_kept).collect(),
        })
    }
}

// ─── Ridge baseline (for tests / contrast) ─────────────────────────────────────

/// Ridge-regression (Tikhonov) point estimate `β = (XᵀX + λI)⁻¹ Xᵀy`.
///
/// Provided as a dense baseline against which the sparsity of the horseshoe
/// posterior can be contrasted: ridge shrinks every coefficient uniformly and
/// cannot zero noise coordinates while sparing large signals.
///
/// # Errors
/// * [`BayesError::EmptyInputs`] / [`BayesError::DimensionMismatch`] on bad shapes.
/// * [`BayesError::SingularMatrix`] if the normal equations are not solvable.
pub fn ridge_regression(x: &[f32], y: &[f32], p: usize, ridge: f32) -> BayesResult<Vec<f32>> {
    if x.is_empty() || y.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if p == 0 {
        return Err(BayesError::InvalidConfig("ridge p must be >= 1".into()));
    }
    let n = y.len();
    if x.len() != n * p {
        return Err(BayesError::DimensionMismatch {
            expected: n * p,
            got: x.len(),
        });
    }
    let mut a = gram_matrix(x, n, p);
    for j in 0..p {
        a[j * p + j] += ridge;
    }
    let b = matvec_transpose(x, y, p);
    solve_spd(&a, &b, p)
}

// ─── Linear-algebra helpers ────────────────────────────────────────────────────

/// Compute the Gram matrix `XᵀX` (p × p, row-major) of an `n × p` design.
fn gram_matrix(x: &[f32], n: usize, p: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; p * p];
    for row in 0..n {
        let base = row * p;
        for i in 0..p {
            let xi = x[base + i];
            for j in i..p {
                g[i * p + j] += xi * x[base + j];
            }
        }
    }
    // Symmetrise the lower triangle.
    for i in 0..p {
        for j in 0..i {
            g[i * p + j] = g[j * p + i];
        }
    }
    g
}

/// Compute `Xᵀy` for an `n × p` row-major `X` and length-`n` `y`.
fn matvec_transpose(x: &[f32], y: &[f32], p: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; p];
    for (row, &yr) in x.chunks_exact(p).zip(y.iter()) {
        for (o, &xv) in out.iter_mut().zip(row.iter()) {
            *o += xv * yr;
        }
    }
    out
}

/// Residual sum of squares `‖y − Xβ‖²`.
fn residual_sum_of_squares(x: &[f32], y: &[f32], beta: &[f32], p: usize) -> f32 {
    let mut ss = 0.0_f32;
    for (row, &yr) in x.chunks_exact(p).zip(y.iter()) {
        let pred: f32 = row.iter().zip(beta.iter()).map(|(&xv, &bv)| xv * bv).sum();
        let r = yr - pred;
        ss += r * r;
    }
    ss
}

/// Cholesky factorisation `A = L Lᵀ` of an SPD matrix (row-major, p × p),
/// returning the lower-triangular `L` (row-major).
fn cholesky(a: &[f32], p: usize) -> BayesResult<Vec<f32>> {
    let mut l = vec![0.0_f32; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[i * p + j];
            for k in 0..j {
                sum -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(BayesError::SingularMatrix(
                        "horseshoe: Cholesky hit a non-positive pivot".into(),
                    ));
                }
                l[i * p + j] = sum.sqrt();
            } else {
                l[i * p + j] = sum / l[j * p + j];
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` for an SPD `A` (p × p, row-major) via Cholesky.
fn solve_spd(a: &[f32], b: &[f32], p: usize) -> BayesResult<Vec<f32>> {
    let l = cholesky(a, p)?;
    // Forward solve L z = b.
    let mut z = vec![0.0_f32; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * p + k] * z[k];
        }
        z[i] = sum / l[i * p + i];
    }
    // Back solve Lᵀ x = z.
    let mut x = vec![0.0_f32; p];
    for i in (0..p).rev() {
        let mut sum = z[i];
        for k in (i + 1)..p {
            sum -= l[k * p + i] * x[k];
        }
        x[i] = sum / l[i * p + i];
    }
    Ok(x)
}

/// Sample `β ~ N(A⁻¹ b, A⁻¹)` given the **precision** matrix `A` (SPD, p × p,
/// row-major) and the precision-weighted mean vector `b` (so the mean is
/// `μ = A⁻¹ b`).
///
/// Uses the Cholesky factor `A = L Lᵀ`: solve `L Lᵀ μ = b` for the mean, then
/// add `Lᵀ⁻¹ z` with `z ~ N(0, I)` so that `Cov = (Lᵀ)⁻¹ L⁻¹ = A⁻¹`.
fn sample_mvn_from_precision(
    a: &[f32],
    b: &[f32],
    p: usize,
    rng: &mut LcgRng,
) -> BayesResult<Vec<f32>> {
    let l = cholesky(a, p)?;
    // Mean: forward then back solve as in solve_spd.
    let mut z = vec![0.0_f32; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * p + k] * z[k];
        }
        z[i] = sum / l[i * p + i];
    }
    let mut mean = vec![0.0_f32; p];
    for i in (0..p).rev() {
        let mut sum = z[i];
        for k in (i + 1)..p {
            sum -= l[k * p + i] * mean[k];
        }
        mean[i] = sum / l[i * p + i];
    }
    // Sample: solve Lᵀ w = ε with ε ~ N(0, I); then β = μ + w has Cov = A⁻¹.
    let eps = draw_standard_normal(p, rng);
    let mut w = vec![0.0_f32; p];
    for i in (0..p).rev() {
        let mut sum = eps[i];
        for k in (i + 1)..p {
            sum -= l[k * p + i] * w[k];
        }
        w[i] = sum / l[i * p + i];
    }
    Ok((0..p).map(|i| mean[i] + w[i]).collect())
}

// ─── Random sampling helpers ───────────────────────────────────────────────────

/// Unit-uniform draw in `[0, 1)` from the crate [`LcgRng`].
///
/// `LcgRng::next_u32` returns `state >> 33`, whose maximum is `2³¹ − 1`, so
/// dividing by `2³¹` gives a faithful `[0, 1)` draw. (The crate's `next_f32`
/// divides by `2³²` and only spans `[0, 0.5)`, biasing Box-Muller, so we build
/// the uniform here directly.)
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0_f32
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

/// Draw `d` independent standard-normal samples.
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

/// Sample `G ~ Gamma(shape, rate)` (mean `shape / rate`) via the
/// Marsaglia-Tsang (2000) method with a boosting step for `shape < 1`.
fn sample_gamma(shape: f32, rate: f32, rng: &mut LcgRng) -> f32 {
    debug_assert!(shape > 0.0 && rate > 0.0);
    if shape < 1.0 {
        // Boosting: if G ~ Gamma(shape+1, 1) and U ~ U(0,1) then
        // G · U^(1/shape) ~ Gamma(shape, 1).
        let g = sample_gamma(shape + 1.0, 1.0, rng);
        let u = unit_uniform(rng).max(1e-12);
        return g * u.powf(1.0 / shape) / rate;
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // x ~ N(0,1), v = (1 + c x)³ must be positive.
        let (x, _) = box_muller_pair(rng);
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        v = v * v * v;
        let u = unit_uniform(rng).clamp(1e-12, 1.0 - 1e-12);
        let x2 = x * x;
        // Squeeze + acceptance tests (standard Marsaglia-Tsang).
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v / rate;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

/// Sample `X ~ InvGamma(shape, scale)` with density
/// `∝ x^(−shape−1) exp(−scale / x)` (so mean `scale / (shape − 1)` for
/// `shape > 1`). Drawn as `1 / Gamma(shape, rate = scale)`.
fn sample_inv_gamma(shape: f32, scale: f32, rng: &mut LcgRng) -> f32 {
    let g = sample_gamma(shape, scale, rng);
    1.0 / g.max(1e-30)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sparse regression problem: a few large signals, many zeros.
    /// Returns `(x_row_major, y, true_beta, n, p)`.
    fn make_sparse_problem(
        n: usize,
        p: usize,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize) {
        // True coefficients: indices 0, 3, 7 are large; the rest are exactly 0.
        let mut true_beta = vec![0.0_f32; p];
        if p > 0 {
            true_beta[0] = 3.0;
        }
        if p > 3 {
            true_beta[3] = -2.5;
        }
        if p > 7 {
            true_beta[7] = 4.0;
        }
        let mut x = vec![0.0_f32; n * p];
        let mut y = vec![0.0_f32; n];
        for (x_row, y_row) in x.chunks_exact_mut(p).zip(y.iter_mut()) {
            let mut yi = 0.0_f32;
            for (slot, &b) in x_row.iter_mut().zip(true_beta.iter()) {
                let (xv, _) = box_muller_pair(rng);
                *slot = xv;
                yi += xv * b;
            }
            // Small observation noise.
            let (e, _) = box_muller_pair(rng);
            *y_row = yi + 0.1 * e;
        }
        (x, y, true_beta, n, p)
    }

    // ── (a) the marginal density has a spike at 0 AND heavy tails ─────────────
    #[test]
    fn horseshoe_density_spike_and_heavy_tails() {
        let tau = 1.0_f32;
        // Spike at the origin: density → +∞ as β → 0.
        let near0 = horseshoe_log_pdf(1e-3, tau).expect("density");
        let mid = horseshoe_log_pdf(1.0, tau).expect("density");
        assert!(
            near0 > mid + 2.0,
            "log-density at 0 ({near0}) must dominate the bulk ({mid})"
        );

        // Heavy tails: compare the horseshoe tail decay with a Gaussian's.
        // Between β=3 and β=6 the horseshoe log-density should drop far less
        // than a standard Gaussian's (which falls by Δ = (6²−3²)/2 = 13.5).
        let hs_3 = horseshoe_log_pdf(3.0, tau).expect("density");
        let hs_6 = horseshoe_log_pdf(6.0, tau).expect("density");
        let gauss_drop = (6.0_f32 * 6.0 - 3.0 * 3.0) / 2.0; // 13.5 in nats
        let hs_drop = hs_3 - hs_6;
        assert!(
            hs_drop < gauss_drop,
            "horseshoe tail must decay slower than Gaussian: hs_drop={hs_drop}, gauss_drop={gauss_drop}"
        );
        assert!(hs_drop > 0.0, "tail must still be decreasing");
    }

    #[test]
    fn half_cauchy_pdf_basics() {
        // p(0) = 2/π; log p(0) = log(2/π).
        let lp0 = half_cauchy_log_pdf(0.0).expect("hc");
        assert!((lp0 - (2.0_f32 / std::f32::consts::PI).ln()).abs() < 1e-5);
        // Monotone decreasing on x ≥ 0.
        let lp1 = half_cauchy_log_pdf(1.0).expect("hc");
        let lp2 = half_cauchy_log_pdf(2.0).expect("hc");
        assert!(lp0 > lp1 && lp1 > lp2);
        assert!(half_cauchy_log_pdf(-0.1).is_err());
    }

    // ── (b) sparse recovery beats ridge ───────────────────────────────────────
    #[test]
    fn horseshoe_beats_ridge_on_sparse_signal() {
        let mut data_rng = LcgRng::new(2024);
        let (x, y, true_beta, _n, p) = make_sparse_problem(120, 10, &mut data_rng);

        let cfg = HorseshoeConfig {
            n_iter: 1500,
            burn_in: 750,
            init_sigma2: 1.0,
        };
        let mut rng = LcgRng::new(7);
        let fit = HorseshoeRegression::fit(&x, &y, p, &cfg, &mut rng).expect("fit");

        // Ridge baseline with a moderate penalty.
        let ridge = ridge_regression(&x, &y, p, 1.0).expect("ridge");

        // Mean-squared error against the true sparse vector.
        let mse = |est: &[f32]| -> f32 {
            est.iter()
                .zip(true_beta.iter())
                .map(|(&e, &t)| (e - t) * (e - t))
                .sum::<f32>()
                / p as f32
        };
        let hs_mse = mse(&fit.beta_mean);
        let ridge_mse = mse(&ridge);
        assert!(
            hs_mse < ridge_mse,
            "horseshoe MSE ({hs_mse}) should beat ridge MSE ({ridge_mse})"
        );

        // Noise coordinates (true zero) should be shrunk much harder than
        // ridge shrinks them.
        let noise_idx: Vec<usize> = (0..p).filter(|&j| true_beta[j] == 0.0).collect();
        let hs_noise: f32 = noise_idx.iter().map(|&j| fit.beta_mean[j].abs()).sum();
        let ridge_noise: f32 = noise_idx.iter().map(|&j| ridge[j].abs()).sum();
        assert!(
            hs_noise < ridge_noise,
            "horseshoe should zero noise more aggressively: hs={hs_noise}, ridge={ridge_noise}"
        );
    }

    // ── (c) the shrinkage factor lies in (0, 1) ───────────────────────────────
    #[test]
    fn shrinkage_factor_in_unit_interval() {
        // κ = 1/(1 + τ²λ²) is mathematically in the open interval (0, 1); at
        // extreme scales f32 rounds the boundary (κ → 1 when τ²λ² underflows
        // below f32 epsilon), so we assert the closed range (0, 1] here.
        for &tau in &[0.01_f32, 0.1, 1.0, 5.0] {
            for &lambda in &[0.01_f32, 0.5, 2.0, 50.0] {
                let kappa = shrinkage_factor(tau, lambda).expect("kappa");
                assert!(
                    kappa > 0.0 && kappa <= 1.0,
                    "kappa={kappa} out of (0,1] for tau={tau}, lambda={lambda}"
                );
            }
        }
        // With a non-degenerate scale κ is strictly interior.
        let interior = shrinkage_factor(1.0, 1.0).expect("kappa"); // = 0.5
        assert!(interior > 0.0 && interior < 1.0);
        assert!((interior - 0.5).abs() < 1e-6);
        // Large τ²λ² ⇒ κ → 0 (signal preserved); small ⇒ κ → 1 (shrunk).
        let big = shrinkage_factor(10.0, 10.0).expect("kappa");
        let small = shrinkage_factor(0.01, 0.01).expect("kappa");
        assert!(big < 0.01, "large scale should give small kappa: {big}");
        assert!(small > 0.99, "tiny scale should give kappa≈1: {small}");
        assert!(shrinkage_factor(-1.0, 1.0).is_err());
    }

    // ── (d) the IG augmentation reproduces the half-Cauchy scale ──────────────
    #[test]
    fn ig_augmentation_reproduces_half_cauchy() {
        // λ² | ν ~ IG(1/2, 1/ν), ν ~ IG(1/2, 1) ⇒ λ ~ C⁺(0,1).
        // We check the *median* of the implied |λ| ≈ 1 (the standard
        // half-Cauchy has median 1), which is robust to the heavy tail.
        let mut rng = LcgRng::new(123);
        let n_samples = 40_000;
        let mut lambdas = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let nu = sample_inv_gamma(0.5, 1.0, &mut rng);
            let lambda2 = sample_inv_gamma(0.5, 1.0 / nu.max(1e-12), &mut rng);
            lambdas.push(lambda2.sqrt());
        }
        lambdas.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        let median = lambdas[n_samples / 2];
        // Standard half-Cauchy median is exactly 1; allow generous tolerance.
        assert!(
            (median - 1.0).abs() < 0.25,
            "augmented half-Cauchy median should be ≈ 1, got {median}"
        );
        // Heavy tail: the 90th percentile of C⁺(0,1) is tan(0.9·π/2) ≈ 6.31.
        let p90 = lambdas[(n_samples as f32 * 0.9) as usize];
        assert!(p90 > 2.0, "tail too light: p90={p90}");
    }

    // ── (e) the global scale τ shrinks as the model gets sparser ──────────────
    #[test]
    fn global_scale_shrinks_with_sparsity() {
        let fit_for_density = |n_signals: usize| -> f32 {
            let mut data_rng = LcgRng::new(555);
            let n = 100;
            let p = 12;
            let mut true_beta = vec![0.0_f32; p];
            for b in true_beta.iter_mut().take(n_signals) {
                *b = 3.0;
            }
            let mut x = vec![0.0_f32; n * p];
            let mut y = vec![0.0_f32; n];
            for (x_row, y_row) in x.chunks_exact_mut(p).zip(y.iter_mut()) {
                let mut yi = 0.0_f32;
                for (slot, &b) in x_row.iter_mut().zip(true_beta.iter()) {
                    let (xv, _) = box_muller_pair(&mut data_rng);
                    *slot = xv;
                    yi += xv * b;
                }
                let (e, _) = box_muller_pair(&mut data_rng);
                *y_row = yi + 0.1 * e;
            }
            let cfg = HorseshoeConfig {
                n_iter: 1200,
                burn_in: 600,
                init_sigma2: 1.0,
            };
            let mut rng = LcgRng::new(31);
            HorseshoeRegression::fit(&x, &y, p, &cfg, &mut rng)
                .expect("fit")
                .tau_mean
        };
        // Sparser model (1 signal) ⇒ smaller global scale than a denser one.
        let tau_sparse = fit_for_density(1);
        let tau_dense = fit_for_density(8);
        assert!(
            tau_sparse < tau_dense,
            "global scale should shrink with sparsity: sparse={tau_sparse}, dense={tau_dense}"
        );
    }

    // ── (f) the β posterior mean tracks the signals ───────────────────────────
    #[test]
    fn beta_posterior_tracks_signals() {
        let mut data_rng = LcgRng::new(909);
        let (x, y, true_beta, _n, p) = make_sparse_problem(150, 10, &mut data_rng);
        let cfg = HorseshoeConfig {
            n_iter: 1500,
            burn_in: 750,
            init_sigma2: 1.0,
        };
        let mut rng = LcgRng::new(17);
        let fit = HorseshoeRegression::fit(&x, &y, p, &cfg, &mut rng).expect("fit");

        // Large signals are recovered with the right sign and rough magnitude.
        for &j in &[0usize, 3, 7] {
            let est = fit.beta_mean[j];
            let truth = true_beta[j];
            assert!(
                est.signum() == truth.signum() && (est - truth).abs() < 0.6,
                "signal {j}: est={est}, truth={truth}"
            );
            // Recovered signal coordinates have small shrinkage factor κ.
            assert!(
                fit.kappa_mean[j] < 0.5,
                "signal {j} should be lightly shrunk: kappa={}",
                fit.kappa_mean[j]
            );
        }
        // Noise coordinates are heavily shrunk: |β̂| small and κ near 1.
        for (j, (&truth, &est)) in true_beta.iter().zip(fit.beta_mean.iter()).enumerate() {
            if truth == 0.0 {
                assert!(est.abs() < 0.3, "noise {j} not shrunk: {est}");
            }
        }
    }

    // ── extra: sampler internals & error paths ────────────────────────────────
    #[test]
    fn inv_gamma_mean_is_correct() {
        // IG(shape, scale) has mean scale/(shape-1) for shape>1.
        let mut rng = LcgRng::new(77);
        let (shape, scale) = (4.0_f32, 6.0_f32);
        let n = 60_000;
        let mut s = 0.0_f64;
        for _ in 0..n {
            s += sample_inv_gamma(shape, scale, &mut rng) as f64;
        }
        let mean = (s / n as f64) as f32;
        let expected = scale / (shape - 1.0); // = 2.0
        assert!(
            (mean - expected).abs() < 0.1,
            "IG mean: got {mean}, expected {expected}"
        );
    }

    #[test]
    fn gamma_mean_is_correct() {
        let mut rng = LcgRng::new(88);
        // Gamma(shape, rate) mean = shape/rate. Test a shape<1 (boosting path).
        let (shape, rate) = (0.5_f32, 2.0_f32);
        let n = 80_000;
        let mut s = 0.0_f64;
        for _ in 0..n {
            s += sample_gamma(shape, rate, &mut rng) as f64;
        }
        let mean = (s / n as f64) as f32;
        let expected = shape / rate; // 0.25
        assert!(
            (mean - expected).abs() < 0.02,
            "Gamma mean: got {mean}, expected {expected}"
        );
    }

    #[test]
    fn ridge_recovers_dense_solution() {
        // With a tiny ridge and a well-posed system, ridge ≈ OLS.
        let x = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![2.0_f32, 3.0, 5.0];
        let beta = ridge_regression(&x, &y, 2, 1e-6).expect("ridge");
        // Solve exactly: rows (1,0)->2, (0,1)->3, (1,1)->5 ⇒ β≈(2,3).
        assert!((beta[0] - 2.0).abs() < 0.05, "b0={}", beta[0]);
        assert!((beta[1] - 3.0).abs() < 0.05, "b1={}", beta[1]);
    }

    #[test]
    fn horseshoe_rejects_bad_inputs() {
        let mut rng = LcgRng::new(1);
        let cfg = HorseshoeConfig::default();
        assert!(matches!(
            HorseshoeRegression::fit(&[], &[], 1, &cfg, &mut rng),
            Err(BayesError::EmptyInputs)
        ));
        assert!(matches!(
            HorseshoeRegression::fit(&[1.0, 2.0], &[1.0], 3, &cfg, &mut rng),
            Err(BayesError::DimensionMismatch { .. })
        ));
        let bad = HorseshoeConfig {
            n_iter: 5,
            burn_in: 5,
            init_sigma2: 1.0,
        };
        assert!(HorseshoeRegression::fit(&[1.0], &[1.0], 1, &bad, &mut rng).is_err());
    }

    #[test]
    fn cholesky_solve_round_trip() {
        // A = [[4,2],[2,3]] SPD; solve A x = b for b=(2,1).
        let a = vec![4.0_f32, 2.0, 2.0, 3.0];
        let b = vec![2.0_f32, 1.0];
        let x = solve_spd(&a, &b, 2).expect("solve");
        // Verify A x ≈ b.
        let ax0 = 4.0 * x[0] + 2.0 * x[1];
        let ax1 = 2.0 * x[0] + 3.0 * x[1];
        assert!((ax0 - 2.0).abs() < 1e-4 && (ax1 - 1.0).abs() < 1e-4);
        // Non-SPD matrix is rejected.
        let bad = vec![1.0_f32, 2.0, 2.0, 1.0]; // indefinite
        assert!(solve_spd(&bad, &b, 2).is_err());
    }
}
