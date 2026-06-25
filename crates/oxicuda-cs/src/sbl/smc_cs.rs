//! Sequential Compressed Sensing via Sequential Monte Carlo (Ji, Xue & Carin 2008).
//!
//! "Bayesian Compressive Sensing", IEEE Trans. Signal Processing 56(6):2346-2356 —
//! sequential / streaming variant.
//!
//! # Model
//!
//! Measurements arrive one row (or a small batch) at a time:
//!
//! ```text
//! y_t = φ_tᵀ x + ε_t,   ε_t ~ N(0, σ²),   t = 1, 2, …
//! ```
//!
//! with a sparsity-promoting Relevance-Vector-Machine (Automatic-Relevance-Determination)
//! prior on the unknown signal `x ∈ ℝ^d`:
//!
//! ```text
//! p(x_j | α_j) = N(0, α_j⁻¹),   A = diag(α),   p(x | A) = N(0, A⁻¹).
//! ```
//!
//! The per-coefficient precisions `α_j` control sparsity: a large `α_j` shrinks `x_j` to
//! zero, while a small `α_j` lets `x_j` be active. The exact posterior over `α` is
//! intractable, so we approximate it with a population of weighted **particles**. Each
//! particle fixes its own hyperparameter vector `α^{(p)}` and therefore carries an
//! *analytic* Gaussian posterior over `x`:
//!
//! ```text
//! p(x | y_{1:t}, α^{(p)}) = N(m^{(p)}_t, Σ^{(p)}_t).
//! ```
//!
//! # Sequential Monte Carlo recursion
//!
//! On receiving a new measurement `(φ_t, y_t)` we perform, per particle:
//!
//! 1. **Predictive likelihood** (Gaussian innovation):
//!    ```text
//!    ŷ   = φ_tᵀ m^{(p)}                  (predictive mean)
//!    s   = φ_tᵀ Σ^{(p)} φ_t + σ²         (innovation / predictive variance)
//!    p(y_t | particle, y_{1:t-1}) = N(y_t; ŷ, s).
//!    ```
//!
//! 2. **Rank-1 sequential Bayesian update** of the Gaussian posterior — the
//!    Kalman / recursive-least-squares (Sherman–Morrison) form:
//!    ```text
//!    g   = Σ^{(p)} φ_t            (un-normalised gain, length d)
//!    k   = g / s                  (Kalman gain)
//!    m^{(p)} ← m^{(p)} + k (y_t − ŷ)
//!    Σ^{(p)} ← Σ^{(p)} − g gᵀ / s     (rank-1 covariance downdate)
//!    ```
//!    This is exact: it is the closed-form Bayesian conditioning of a Gaussian on one
//!    linear-Gaussian observation, never re-touching past measurements.
//!
//! 3. **Reweighting** by the predictive likelihood (the SMC importance weight under the
//!    optimal/bootstrap proposal that leaves `α` fixed within a particle):
//!    ```text
//!    w^{(p)} ← w^{(p)} · N(y_t; ŷ, s),   then normalise Σ_p w^{(p)} = 1.
//!    ```
//!
//! 4. **Resampling** when the effective sample size drops:
//!    ```text
//!    ESS = 1 / Σ_p (w^{(p)})².
//!    ```
//!    If `ESS < ess_threshold · n_particles`, systematic resampling draws `n_particles`
//!    survivors proportional to their weights (RNG from the crate `LcgRng`), copies their
//!    Gaussian posteriors, and resets all weights to `1/n_particles`.
//!
//! # Estimate
//!
//! The posterior mean of `x` is the weighted particle average
//! `x̂ = Σ_p w^{(p)} m^{(p)}`, and the (mixture) posterior variance per coefficient is
//! `Var[x_j] = Σ_p w^{(p)} (Σ^{(p)}_{jj} + (m^{(p)}_j)²) − x̂_j²`.
//!
//! # References
//!
//! - S. Ji, Y. Xue & L. Carin (2008), "Bayesian Compressive Sensing",
//!   IEEE Trans. Signal Processing 56(6):2346-2356.
//! - M. E. Tipping (2001), "Sparse Bayesian Learning and the Relevance Vector Machine",
//!   JMLR 1:211-244.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

/// Configuration for the sequential-Monte-Carlo compressed-sensing recovery.
#[derive(Debug, Clone)]
pub struct SmcCsConfig {
    /// Number of particles in the population (`> 0`).
    pub n_particles: usize,
    /// Dimension `d` of the unknown signal `x` (`> 0`).
    pub signal_dim: usize,
    /// Observation-noise variance `σ²` (`> 0`).
    pub noise_variance: f64,
    /// Resampling fires when `ESS < ess_threshold · n_particles`. Must lie in `(0, 1]`.
    pub ess_threshold: f64,
    /// Shape parameter of the Gamma prior from which each particle samples its
    /// per-coefficient precisions `α_j` (`> 0`). Larger ⇒ tighter spread of `α`.
    pub alpha_shape: f64,
    /// Mean (scale·shape) of the Gamma prior over `α_j` (`> 0`). The *prior* belief about
    /// the typical precision; a large value encodes a strong sparsity preference.
    pub alpha_mean: f64,
    /// Whether to rejuvenate (re-perturb) particle hyperparameters after a resampling
    /// step to combat sample impoverishment.
    pub rejuvenate: bool,
}

impl Default for SmcCsConfig {
    fn default() -> Self {
        Self {
            n_particles: 64,
            signal_dim: 16,
            noise_variance: 1e-2,
            ess_threshold: 0.5,
            alpha_shape: 2.0,
            alpha_mean: 1.0,
            rejuvenate: true,
        }
    }
}

/// A single SMC particle: a fixed hyperparameter vector `α` together with the analytic
/// Gaussian posterior `N(mean, cov)` over `x` it induces, and an (un-normalised) weight.
#[derive(Debug, Clone)]
struct Particle {
    /// Per-coefficient precisions `α_j` (length `d`).
    alpha: Vec<f64>,
    /// Posterior mean `m ∈ ℝ^d`.
    mean: Vec<f64>,
    /// Posterior covariance `Σ ∈ ℝ^{d×d}`, row-major (length `d·d`).
    cov: Vec<f64>,
    /// Importance weight (kept normalised after each update).
    weight: f64,
}

impl Particle {
    /// Initialise a particle from its sampled precision vector: posterior = prior, i.e.
    /// `m = 0`, `Σ = diag(1/α_j)`.
    fn from_alpha(alpha: Vec<f64>, weight: f64) -> Self {
        let d = alpha.len();
        let mean = vec![0.0_f64; d];
        let mut cov = vec![0.0_f64; d * d];
        for j in 0..d {
            cov[j * d + j] = 1.0 / alpha[j];
        }
        Self {
            alpha,
            mean,
            cov,
            weight,
        }
    }

    /// Apply one rank-1 sequential Bayesian (Kalman) update for measurement `(phi, y)`.
    ///
    /// Returns the predictive likelihood `N(y; φᵀm, φᵀΣφ + σ²)` so the caller can reweight.
    fn sequential_update(&mut self, phi: &[f64], y: f64, noise_variance: f64) -> f64 {
        let d = self.mean.len();

        // g = Σ φ           (length d)
        let mut g = vec![0.0_f64; d];
        for i in 0..d {
            let row = i * d;
            let mut acc = 0.0_f64;
            for j in 0..d {
                acc += self.cov[row + j] * phi[j];
            }
            g[i] = acc;
        }

        // Innovation variance  s = φᵀ Σ φ + σ²  = φᵀ g + σ².
        let mut phi_sigma_phi = 0.0_f64;
        for j in 0..d {
            phi_sigma_phi += phi[j] * g[j];
        }
        // Σ is SPD, so φᵀΣφ ≥ 0; guard against tiny negative round-off.
        let s = phi_sigma_phi.max(0.0) + noise_variance;
        let s_safe = s.max(1e-300);

        // Predictive mean  ŷ = φᵀ m.
        let mut y_hat = 0.0_f64;
        for j in 0..d {
            y_hat += phi[j] * self.mean[j];
        }
        let innovation = y - y_hat;

        // Mean update  m ← m + (g / s) (y − ŷ).
        let scale = innovation / s_safe;
        for i in 0..d {
            self.mean[i] += g[i] * scale;
        }

        // Covariance downdate  Σ ← Σ − g gᵀ / s   (Sherman–Morrison rank-1).
        let inv_s = 1.0 / s_safe;
        for i in 0..d {
            let gi = g[i];
            if gi == 0.0 {
                continue;
            }
            let row = i * d;
            let gi_inv_s = gi * inv_s;
            for j in 0..d {
                self.cov[row + j] -= gi_inv_s * g[j];
            }
        }
        // Re-symmetrise to quench round-off drift (Σ must stay symmetric SPD).
        for i in 0..d {
            for j in (i + 1)..d {
                let avg = 0.5 * (self.cov[i * d + j] + self.cov[j * d + i]);
                self.cov[i * d + j] = avg;
                self.cov[j * d + i] = avg;
            }
        }

        // Predictive likelihood  N(y; ŷ, s).
        gaussian_pdf(innovation, s_safe)
    }
}

/// Sequential-Monte-Carlo compressed-sensing recovery engine.
///
/// Maintains a weighted particle approximation of the posterior over the RVM
/// hyperparameters `α` (and, through them, over the sparse signal `x`), updating it
/// online as measurement rows stream in.
#[derive(Debug, Clone)]
pub struct SmcCs {
    config: SmcCsConfig,
    particles: Vec<Particle>,
    rng: LcgRng,
    /// Number of measurements assimilated so far.
    n_observed: usize,
    /// Number of resampling events triggered so far.
    n_resamples: usize,
}

impl SmcCs {
    /// Create a new SMC recovery engine, sampling each particle's hyperparameter vector
    /// `α` from the configured Gamma prior using `rng`.
    ///
    /// # Errors
    /// * [`CsError::InvalidParameter`] if `n_particles == 0`, `signal_dim == 0`,
    ///   `noise_variance <= 0`, `ess_threshold` outside `(0, 1]`, or a non-positive /
    ///   non-finite Gamma-prior parameter.
    pub fn new(config: SmcCsConfig, rng: LcgRng) -> CsResult<Self> {
        if config.n_particles == 0 {
            return Err(CsError::InvalidParameter(
                "smc_cs: n_particles must be > 0".into(),
            ));
        }
        if config.signal_dim == 0 {
            return Err(CsError::InvalidParameter(
                "smc_cs: signal_dim must be > 0".into(),
            ));
        }
        if config.noise_variance <= 0.0 || !config.noise_variance.is_finite() {
            return Err(CsError::InvalidParameter(format!(
                "smc_cs: noise_variance must be finite and > 0, got {}",
                config.noise_variance
            )));
        }
        if config.ess_threshold <= 0.0 || config.ess_threshold > 1.0 {
            return Err(CsError::InvalidParameter(format!(
                "smc_cs: ess_threshold must be in (0, 1], got {}",
                config.ess_threshold
            )));
        }
        if config.alpha_shape <= 0.0 || !config.alpha_shape.is_finite() {
            return Err(CsError::InvalidParameter(format!(
                "smc_cs: alpha_shape must be finite and > 0, got {}",
                config.alpha_shape
            )));
        }
        if config.alpha_mean <= 0.0 || !config.alpha_mean.is_finite() {
            return Err(CsError::InvalidParameter(format!(
                "smc_cs: alpha_mean must be finite and > 0, got {}",
                config.alpha_mean
            )));
        }

        let mut rng = rng;
        let d = config.signal_dim;
        let n_p = config.n_particles;
        let w0 = 1.0 / n_p as f64;
        let mut particles = Vec::with_capacity(n_p);
        for _ in 0..n_p {
            let alpha = sample_alpha_vector(d, config.alpha_shape, config.alpha_mean, &mut rng);
            particles.push(Particle::from_alpha(alpha, w0));
        }

        Ok(Self {
            config,
            particles,
            rng,
            n_observed: 0,
            n_resamples: 0,
        })
    }

    /// Assimilate one streaming measurement `y = φ_rowᵀ x + noise`.
    ///
    /// Performs the per-particle predictive-likelihood evaluation, the rank-1 sequential
    /// Bayesian (Kalman) posterior update, weight re-normalisation, and conditional
    /// resampling.
    ///
    /// # Errors
    /// * [`CsError::DimensionMismatch`] if `phi_row.len() != signal_dim`.
    /// * [`CsError::InvalidParameter`] if `y` or any entry of `phi_row` is non-finite.
    /// * [`CsError::NumericalInstability`] if every particle weight underflows to zero
    ///   (the measurement is incompatible with the whole population).
    pub fn update(&mut self, phi_row: &[f64], y: f64) -> CsResult<()> {
        let d = self.config.signal_dim;
        if phi_row.len() != d {
            return Err(CsError::DimensionMismatch {
                a: phi_row.len(),
                b: d,
            });
        }
        if !y.is_finite() {
            return Err(CsError::InvalidParameter(
                "smc_cs: measurement y must be finite".into(),
            ));
        }
        if phi_row.iter().any(|v| !v.is_finite()) {
            return Err(CsError::InvalidParameter(
                "smc_cs: phi_row entries must be finite".into(),
            ));
        }

        let noise_variance = self.config.noise_variance;
        let mut weight_sum = 0.0_f64;
        for particle in &mut self.particles {
            let likelihood = particle.sequential_update(phi_row, y, noise_variance);
            // Multiply the (already normalised) weight by the predictive likelihood.
            particle.weight *= likelihood;
            weight_sum += particle.weight;
        }

        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            return Err(CsError::NumericalInstability(
                "smc_cs: all particle weights underflowed to zero".into(),
            ));
        }

        // Normalise weights.
        let inv = 1.0 / weight_sum;
        for particle in &mut self.particles {
            particle.weight *= inv;
        }

        self.n_observed += 1;

        // Resample if the effective sample size is too low.
        let ess = self.effective_sample_size();
        let threshold = self.config.ess_threshold * self.config.n_particles as f64;
        if ess < threshold {
            self.resample()?;
        }

        Ok(())
    }

    /// Assimilate a batch of measurements: row `t` of `phi` (row-major `[n_rows × d]`)
    /// paired with `y[t]`. Each row is processed by [`SmcCs::update`] in arrival order.
    ///
    /// # Errors
    /// * [`CsError::ShapeMismatch`] if `phi.len() != n_rows * signal_dim`.
    /// * [`CsError::DimensionMismatch`] if `y.len() != n_rows`.
    /// * any error propagated by [`SmcCs::update`].
    pub fn update_batch(&mut self, phi: &[f64], n_rows: usize, y: &[f64]) -> CsResult<()> {
        let d = self.config.signal_dim;
        if phi.len() != n_rows * d {
            return Err(CsError::ShapeMismatch {
                expected: vec![n_rows, d],
                got: vec![phi.len()],
            });
        }
        if y.len() != n_rows {
            return Err(CsError::DimensionMismatch {
                a: y.len(),
                b: n_rows,
            });
        }
        for t in 0..n_rows {
            let row = &phi[t * d..(t + 1) * d];
            self.update(row, y[t])?;
        }
        Ok(())
    }

    /// Posterior-mean estimate of `x`: the weight-averaged particle mean
    /// `x̂ = Σ_p w^{(p)} m^{(p)}`.
    #[must_use]
    pub fn estimate(&self) -> Vec<f64> {
        let d = self.config.signal_dim;
        let mut out = vec![0.0_f64; d];
        for particle in &self.particles {
            let w = particle.weight;
            for j in 0..d {
                out[j] += w * particle.mean[j];
            }
        }
        out
    }

    /// Per-coefficient posterior variance of the particle mixture:
    /// `Var[x_j] = Σ_p w^{(p)} (Σ^{(p)}_{jj} + (m^{(p)}_j)²) − x̂_j²`.
    #[must_use]
    pub fn posterior_variance(&self) -> Vec<f64> {
        let d = self.config.signal_dim;
        let mean = self.estimate();
        let mut second = vec![0.0_f64; d];
        for particle in &self.particles {
            let w = particle.weight;
            for j in 0..d {
                let m_j = particle.mean[j];
                let var_j = particle.cov[j * d + j];
                second[j] += w * (var_j + m_j * m_j);
            }
        }
        let mut out = vec![0.0_f64; d];
        for j in 0..d {
            out[j] = (second[j] - mean[j] * mean[j]).max(0.0);
        }
        out
    }

    /// Effective sample size `ESS = 1 / Σ_p (w^{(p)})²`, always in `(0, n_particles]`.
    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        let mut sum_sq = 0.0_f64;
        for particle in &self.particles {
            sum_sq += particle.weight * particle.weight;
        }
        if sum_sq <= 0.0 {
            // Degenerate guard (should not happen after a successful update).
            return self.config.n_particles as f64;
        }
        1.0 / sum_sq
    }

    /// Number of measurements assimilated so far.
    #[must_use]
    pub fn n_observed(&self) -> usize {
        self.n_observed
    }

    /// Number of resampling events that have fired so far.
    #[must_use]
    pub fn n_resamples(&self) -> usize {
        self.n_resamples
    }

    /// Number of particles in the population.
    #[must_use]
    pub fn n_particles(&self) -> usize {
        self.config.n_particles
    }

    /// Read-only access to the (normalised) particle weights.
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        self.particles.iter().map(|p| p.weight).collect()
    }

    /// Systematic resampling: replace the population with `n_particles` survivors drawn
    /// proportionally to the current weights, then reset all weights to `1/n_particles`.
    fn resample(&mut self) -> CsResult<()> {
        let n_p = self.config.n_particles;
        let n_p_f = n_p as f64;

        // Build the inclusive cumulative-weight ladder.
        let mut cumulative = Vec::with_capacity(n_p);
        let mut acc = 0.0_f64;
        for particle in &self.particles {
            acc += particle.weight;
            cumulative.push(acc);
        }
        let total = acc;
        if total <= 0.0 || !total.is_finite() {
            return Err(CsError::NumericalInstability(
                "smc_cs: cannot resample from a zero-mass population".into(),
            ));
        }

        // Systematic resampling: one uniform offset, evenly-spaced pointers.
        let step = total / n_p_f;
        let u0 = self.rng.next_f64() * step;

        let mut survivors: Vec<usize> = Vec::with_capacity(n_p);
        let mut cursor = 0usize;
        for k in 0..n_p {
            let pointer = u0 + step * k as f64;
            while cursor < n_p - 1 && cumulative[cursor] < pointer {
                cursor += 1;
            }
            survivors.push(cursor);
        }

        // Clone selected particles into the new population with uniform weights.
        let w0 = 1.0 / n_p_f;
        let mut new_particles: Vec<Particle> = Vec::with_capacity(n_p);
        for &idx in &survivors {
            let mut p = self.particles[idx].clone();
            p.weight = w0;
            new_particles.push(p);
        }

        // Optional rejuvenation: re-sample the duplicated particles' hyperparameters and
        // re-condition their Gaussian posteriors on all measurements implicitly retained
        // via the carried-over (mean, cov). We perturb α multiplicatively and re-anchor
        // the covariance scale so impoverishment after many resamples is mitigated while
        // preserving the already-learned posterior mean.
        if self.config.rejuvenate {
            self.rejuvenate_population(&mut new_particles, &survivors);
        }

        self.particles = new_particles;
        self.n_resamples += 1;
        Ok(())
    }

    /// Multiplicatively jitter duplicated particles' precisions to restore diversity.
    ///
    /// Only particles that are exact duplicates of an already-kept survivor are jittered,
    /// so unique survivors keep their learned posterior intact. The jitter rescales the
    /// posterior covariance consistently with the new prior precision while preserving the
    /// posterior mean (the data-driven point estimate).
    fn rejuvenate_population(&mut self, particles: &mut [Particle], survivors: &[usize]) {
        let d = self.config.signal_dim;
        let mut seen = vec![false; self.config.n_particles];
        for (slot, &idx) in survivors.iter().enumerate() {
            if !seen[idx] {
                seen[idx] = true;
                continue; // first copy of this survivor: leave untouched
            }
            // Duplicate copy: jitter its precisions and rescale covariance accordingly.
            let particle = &mut particles[slot];
            for j in 0..d {
                // Log-normal multiplicative jitter centred at 1 (σ_log ≈ 0.15).
                let factor = (0.15 * self.rng.next_normal()).exp();
                let old_alpha = particle.alpha[j];
                let new_alpha = (old_alpha * factor).clamp(1e-8, 1e12);
                // Rescale the j-th covariance row/col by √(old/new) so the (j,j) variance
                // scales by the full ratio old/new (consistent with the changed prior
                // precision) and cross-covariances scale by its square root, while the
                // posterior mean (point estimate) is left unchanged.
                let ratio = old_alpha / new_alpha;
                let sqrt_ratio = ratio.sqrt();
                for col in 0..d {
                    particle.cov[j * d + col] *= sqrt_ratio;
                    particle.cov[col * d + j] *= sqrt_ratio;
                }
                particle.alpha[j] = new_alpha;
            }
            // Re-symmetrise after the asymmetric row/col scaling round-off.
            for i in 0..d {
                for col in (i + 1)..d {
                    let avg = 0.5 * (particle.cov[i * d + col] + particle.cov[col * d + i]);
                    particle.cov[i * d + col] = avg;
                    particle.cov[col * d + i] = avg;
                }
            }
        }
    }
}

/// Standard univariate Gaussian density `N(x; 0, variance)` evaluated at `x`.
///
/// `variance` is assumed strictly positive (the caller guards this).
fn gaussian_pdf(x: f64, variance: f64) -> f64 {
    let inv_var = 1.0 / variance;
    let coeff = (inv_var / std::f64::consts::TAU).sqrt();
    coeff * (-0.5 * x * x * inv_var).exp()
}

/// Sample one Gamma-distributed precision via a moment-matched recipe built from the
/// crate `LcgRng`, then return its reciprocal-free value (the precision itself).
///
/// The Gamma prior has the requested `shape` (`k`) and a scale `θ = mean / k`, so its mean
/// equals `mean`. We use the Marsaglia–Tsang transformation-rejection method for `k ≥ 1`
/// and the boost trick `Γ(k) = Γ(k+1) · U^{1/k}` for `k < 1`, drawing standard normals and
/// uniforms from the LCG. This keeps the population of `α` diverse and strictly positive
/// without any external RNG.
fn sample_gamma(shape: f64, scale: f64, rng: &mut LcgRng) -> f64 {
    // Boost low shapes into the k ≥ 1 regime.
    let (k, boost) = if shape < 1.0 {
        let u = rng.next_f64().max(1e-300);
        (shape + 1.0, u.powf(1.0 / shape))
    } else {
        (shape, 1.0)
    };

    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    // Marsaglia–Tsang: loop until acceptance (expected ~1.0-1.5 iterations).
    let value;
    loop {
        // Draw a standard normal and form v = (1 + c·z)³.
        let z = rng.next_normal();
        let v_base = 1.0 + c * z;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = rng.next_f64().max(1e-300);
        // Squeeze + full acceptance tests.
        if u < 1.0 - 0.0331 * z * z * z * z {
            value = d * v;
            break;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            value = d * v;
            break;
        }
    }
    (value * boost * scale).max(1e-300)
}

/// Sample a per-coefficient precision vector `α ∈ ℝ^d` from the Gamma prior parameterised
/// by `shape` and target `mean`.
fn sample_alpha_vector(d: usize, shape: f64, mean: f64, rng: &mut LcgRng) -> Vec<f64> {
    let scale = mean / shape;
    let mut alpha = Vec::with_capacity(d);
    for _ in 0..d {
        let a = sample_gamma(shape, scale, rng).clamp(1e-8, 1e12);
        alpha.push(a);
    }
    alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sequential random Gaussian measurement `(φ, y)` for a known signal and
    /// stream `n_rows` of them into the engine. Returns the recovery error ‖x̂ − x‖₂.
    fn stream_and_error(engine: &mut SmcCs, x_true: &[f64], n_rows: usize, meas_seed: u64) -> f64 {
        let d = x_true.len();
        let mut mrng = LcgRng::new(meas_seed);
        for _ in 0..n_rows {
            let mut phi = vec![0.0_f64; d];
            for v in phi.iter_mut() {
                *v = mrng.next_normal();
            }
            let mut y = 0.0_f64;
            for j in 0..d {
                y += phi[j] * x_true[j];
            }
            // Small measurement noise.
            y += 0.01 * mrng.next_normal();
            engine.update(&phi, y).expect("update ok");
        }
        let est = engine.estimate();
        let mut err = 0.0_f64;
        for j in 0..d {
            let diff = est[j] - x_true[j];
            err += diff * diff;
        }
        err.sqrt()
    }

    fn default_config(d: usize, n_p: usize) -> SmcCsConfig {
        SmcCsConfig {
            n_particles: n_p,
            signal_dim: d,
            noise_variance: 1e-3,
            ess_threshold: 0.5,
            alpha_shape: 2.0,
            alpha_mean: 0.5,
            rejuvenate: true,
        }
    }

    // Test 1: error decreases as more measurement rows arrive.
    #[test]
    fn error_decreases_with_more_rows() {
        let d = 12usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[2] = 1.5;
        x_true[7] = -2.0;
        x_true[10] = 0.8;

        let cfg = default_config(d, 96);

        let mut eng_few = SmcCs::new(cfg.clone(), LcgRng::new(11)).expect("new ok");
        let err_few = stream_and_error(&mut eng_few, &x_true, 6, 4242);

        let mut eng_many = SmcCs::new(cfg, LcgRng::new(11)).expect("new ok");
        let err_many = stream_and_error(&mut eng_many, &x_true, 40, 4242);

        assert!(
            err_many < err_few,
            "more measurements should reduce error: few={err_few:.4}, many={err_many:.4}"
        );
        assert!(err_many < 0.5, "final error too large: {err_many:.4}");
    }

    // Test 2: ESS always lies in (0, n_particles].
    #[test]
    fn ess_in_valid_range() {
        let d = 8usize;
        let n_p = 64usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[1] = 1.0;
        x_true[5] = -1.2;

        let cfg = default_config(d, n_p);
        let mut eng = SmcCs::new(cfg, LcgRng::new(7)).expect("new ok");

        // Initial ESS == n_particles (uniform weights).
        let init_ess = eng.effective_sample_size();
        assert!(
            (init_ess - n_p as f64).abs() < 1e-9,
            "init ESS = {init_ess}"
        );

        let mut mrng = LcgRng::new(303);
        for _ in 0..30 {
            let mut phi = vec![0.0_f64; d];
            for v in phi.iter_mut() {
                *v = mrng.next_normal();
            }
            let mut y = 0.0_f64;
            for j in 0..d {
                y += phi[j] * x_true[j];
            }
            eng.update(&phi, y).expect("update ok");
            let ess = eng.effective_sample_size();
            assert!(
                ess > 0.0 && ess <= n_p as f64 + 1e-9,
                "ESS out of range: {ess}"
            );
        }
    }

    // Test 3: weights are normalised after every update.
    #[test]
    fn weights_normalised() {
        let d = 6usize;
        let cfg = default_config(d, 48);
        let mut eng = SmcCs::new(cfg, LcgRng::new(21)).expect("new ok");

        let mut x_true = vec![0.0_f64; d];
        x_true[3] = 2.0;

        let mut mrng = LcgRng::new(99);
        for _ in 0..20 {
            let mut phi = vec![0.0_f64; d];
            for v in phi.iter_mut() {
                *v = mrng.next_normal();
            }
            let mut y = 0.0_f64;
            for j in 0..d {
                y += phi[j] * x_true[j];
            }
            eng.update(&phi, y).expect("update ok");
            let sum: f64 = eng.weights().iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "weights sum = {sum}");
            assert!(
                eng.weights().iter().all(|w| *w >= 0.0 && w.is_finite()),
                "weights must be non-negative and finite"
            );
        }
    }

    // Test 4: resampling triggers when ESS falls below threshold.
    #[test]
    fn resampling_triggers() {
        let d = 10usize;
        // Aggressive threshold so resampling almost certainly fires.
        let cfg = SmcCsConfig {
            n_particles: 80,
            signal_dim: d,
            noise_variance: 1e-3,
            ess_threshold: 0.9,
            alpha_shape: 2.0,
            alpha_mean: 0.5,
            rejuvenate: true,
        };
        let mut eng = SmcCs::new(cfg, LcgRng::new(5)).expect("new ok");

        let mut x_true = vec![0.0_f64; d];
        x_true[0] = 3.0;
        x_true[6] = -2.5;

        let mut mrng = LcgRng::new(808);
        for _ in 0..25 {
            let mut phi = vec![0.0_f64; d];
            for v in phi.iter_mut() {
                *v = mrng.next_normal();
            }
            let mut y = 0.0_f64;
            for j in 0..d {
                y += phi[j] * x_true[j];
            }
            eng.update(&phi, y).expect("update ok");
        }
        assert!(
            eng.n_resamples() > 0,
            "resampling should have fired at least once"
        );
        // After resampling, weights remain normalised and ESS valid.
        let sum: f64 = eng.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "post-resample weights sum = {sum}"
        );
    }

    // Test 5: a 1-sparse signal is recovered accurately.
    #[test]
    fn one_sparse_recovery_accurate() {
        let d = 16usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[9] = 4.0;

        let cfg = default_config(d, 128);
        let mut eng = SmcCs::new(cfg, LcgRng::new(33)).expect("new ok");
        let err = stream_and_error(&mut eng, &x_true, 50, 123_456);

        assert!(err < 0.3, "1-sparse recovery error too large: {err:.4}");
        let est = eng.estimate();
        // The active coefficient dominates.
        assert!(est[9] > 3.0, "active coeff under-recovered: {}", est[9]);
        // Inactive coefficients are small.
        let max_off = (0..d)
            .filter(|&j| j != 9)
            .map(|j| est[j].abs())
            .fold(0.0_f64, f64::max);
        assert!(max_off < 0.5, "off-support leakage too large: {max_off:.4}");
    }

    // Test 6: posterior variance is finite, non-negative, and shrinks with data.
    #[test]
    fn posterior_variance_shrinks() {
        let d = 8usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[4] = 2.0;

        let cfg = default_config(d, 96);
        let mut eng = SmcCs::new(cfg, LcgRng::new(64)).expect("new ok");

        let var_before = eng.posterior_variance();
        assert!(
            var_before.iter().all(|v| *v >= 0.0 && v.is_finite()),
            "initial variance must be non-negative finite"
        );

        let _ = stream_and_error(&mut eng, &x_true, 40, 7777);
        let var_after = eng.posterior_variance();
        assert!(
            var_after.iter().all(|v| *v >= 0.0 && v.is_finite()),
            "posterior variance must stay non-negative finite"
        );
        // Total posterior uncertainty should drop after assimilating many measurements.
        let total_before: f64 = var_before.iter().sum();
        let total_after: f64 = var_after.iter().sum();
        assert!(
            total_after < total_before,
            "variance should shrink: before={total_before:.4}, after={total_after:.4}"
        );
    }

    // Test 7: batched update equals row-by-row update.
    #[test]
    fn batch_matches_sequential() {
        let d = 7usize;
        let n_rows = 15usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[2] = 1.0;
        x_true[5] = -1.0;

        // Build a fixed measurement set.
        let mut mrng = LcgRng::new(2024);
        let mut phi = vec![0.0_f64; n_rows * d];
        let mut y = vec![0.0_f64; n_rows];
        for t in 0..n_rows {
            for j in 0..d {
                phi[t * d + j] = mrng.next_normal();
            }
            let mut acc = 0.0_f64;
            for j in 0..d {
                acc += phi[t * d + j] * x_true[j];
            }
            y[t] = acc;
        }

        let cfg = default_config(d, 64);

        let mut eng_seq = SmcCs::new(cfg.clone(), LcgRng::new(50)).expect("new ok");
        for t in 0..n_rows {
            eng_seq
                .update(&phi[t * d..(t + 1) * d], y[t])
                .expect("update ok");
        }

        let mut eng_batch = SmcCs::new(cfg, LcgRng::new(50)).expect("new ok");
        eng_batch.update_batch(&phi, n_rows, &y).expect("batch ok");

        let est_seq = eng_seq.estimate();
        let est_batch = eng_batch.estimate();
        for j in 0..d {
            assert!(
                (est_seq[j] - est_batch[j]).abs() < 1e-12,
                "batch vs sequential mismatch at {j}: {} vs {}",
                est_seq[j],
                est_batch[j]
            );
        }
    }

    // Test 8: dimension-mismatch error path.
    #[test]
    fn update_dimension_mismatch() {
        let d = 5usize;
        let cfg = default_config(d, 16);
        let mut eng = SmcCs::new(cfg, LcgRng::new(1)).expect("new ok");
        // Wrong-length phi row.
        let phi = vec![1.0_f64, 2.0, 3.0];
        assert!(matches!(
            eng.update(&phi, 1.0),
            Err(CsError::DimensionMismatch { .. })
        ));
        // Non-finite y.
        let phi_ok = vec![1.0_f64; d];
        assert!(matches!(
            eng.update(&phi_ok, f64::NAN),
            Err(CsError::InvalidParameter(_))
        ));
        // Batch shape mismatch.
        assert!(matches!(
            eng.update_batch(&[1.0, 2.0, 3.0], 2, &[1.0, 2.0]),
            Err(CsError::ShapeMismatch { .. })
        ));
        // Batch y-length mismatch.
        assert!(matches!(
            eng.update_batch(&vec![0.0_f64; 2 * d], 2, &[1.0]),
            Err(CsError::DimensionMismatch { .. })
        ));
    }

    // Test 9: invalid configurations are rejected.
    #[test]
    fn invalid_config_rejected() {
        // Zero particles.
        let mut cfg = default_config(4, 0);
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        // Zero signal dim.
        cfg = default_config(0, 8);
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        // Negative noise variance.
        cfg = default_config(4, 8);
        cfg.noise_variance = -1.0;
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        // ess_threshold out of range.
        cfg = default_config(4, 8);
        cfg.ess_threshold = 1.5;
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        cfg = default_config(4, 8);
        cfg.ess_threshold = 0.0;
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        // Bad Gamma-prior parameters.
        cfg = default_config(4, 8);
        cfg.alpha_shape = 0.0;
        assert!(matches!(
            SmcCs::new(cfg.clone(), LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
        cfg = default_config(4, 8);
        cfg.alpha_mean = -2.0;
        assert!(matches!(
            SmcCs::new(cfg, LcgRng::new(1)),
            Err(CsError::InvalidParameter(_))
        ));
    }

    // Test 10: determinism — same seed yields identical estimates.
    #[test]
    fn deterministic_given_seed() {
        let d = 9usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[1] = 1.0;
        x_true[8] = -1.5;

        let cfg = default_config(d, 64);

        let mut eng_a = SmcCs::new(cfg.clone(), LcgRng::new(314)).expect("new ok");
        let err_a = stream_and_error(&mut eng_a, &x_true, 25, 271);

        let mut eng_b = SmcCs::new(cfg, LcgRng::new(314)).expect("new ok");
        let err_b = stream_and_error(&mut eng_b, &x_true, 25, 271);

        assert!(
            (err_a - err_b).abs() < 1e-12,
            "non-deterministic: {err_a} vs {err_b}"
        );
        let est_a = eng_a.estimate();
        let est_b = eng_b.estimate();
        for j in 0..d {
            assert!(
                (est_a[j] - est_b[j]).abs() < 1e-12,
                "estimate mismatch at {j}"
            );
        }
    }

    // Test 11: estimate and posterior variance are always finite.
    #[test]
    fn outputs_finite() {
        let d = 10usize;
        let mut x_true = vec![0.0_f64; d];
        x_true[0] = 1.0;
        x_true[3] = -1.0;
        x_true[9] = 2.0;

        let cfg = default_config(d, 80);
        let mut eng = SmcCs::new(cfg, LcgRng::new(123)).expect("new ok");
        let _ = stream_and_error(&mut eng, &x_true, 35, 4321);

        let est = eng.estimate();
        assert!(est.iter().all(|v| v.is_finite()), "estimate has non-finite");
        let var = eng.posterior_variance();
        assert!(var.iter().all(|v| v.is_finite()), "variance has non-finite");
        assert_eq!(est.len(), d);
        assert_eq!(var.len(), d);
        assert_eq!(eng.n_observed(), 35);
    }

    // Test 12: Gamma sampler stays positive with sensible mean.
    #[test]
    fn gamma_sampler_positive_mean() {
        let mut rng = LcgRng::new(2718);
        let shape = 2.0;
        let mean = 1.5;
        let scale = mean / shape;
        let n = 5000;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            let g = sample_gamma(shape, scale, &mut rng);
            assert!(
                g > 0.0 && g.is_finite(),
                "gamma draw must be positive finite"
            );
            sum += g;
        }
        let empirical = sum / n as f64;
        // Empirical mean within ~15% of the requested mean.
        assert!(
            (empirical - mean).abs() < 0.15 * mean,
            "empirical gamma mean {empirical:.3} far from {mean:.3}"
        );
    }
}
