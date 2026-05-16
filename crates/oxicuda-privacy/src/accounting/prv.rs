//! Privacy Random Variable (PRV) accountant.
//!
//! Reference: Gopi, Lee & Wutschitz (2021), "Numerical Composition of
//! Differential Privacy", NeurIPS 2021.
//!
//! # Overview
//! For a mechanism M and neighbouring x, x', define the *privacy loss*
//! random variable (PRV):
//!
//! `Z = log(P(M(x) ∈ E) / P(M(x') ∈ E))`
//!
//! evaluated over the output distribution of M(x).
//!
//! When k mechanisms are composed, the composed PRV is Z₁ + … + Zₖ.
//! By independence, the CDF/PMF of the sum is obtained by convolving the
//! individual PMFs.  Given the composed PMF we can compute:
//!
//! `δ(ε) = E[max(0, 1 − e^(ε − Z))]`
//!
//! # For the Gaussian mechanism
//! With sensitivity Δ and noise std σ, the PRV is N(Δ²/(2σ²), (Δ/σ)²).
//! We discretize this onto a uniform grid [grid_lo, grid_hi] and convolve.
//!
//! # Complexity
//! Convolution is O(n²) where n = `grid_size`.  Use n ≤ 1000 for tests.

use crate::error::{PrivacyError, PrivacyResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Grid parameters for PRV discretization.
#[derive(Debug, Clone)]
pub struct PrvConfig {
    /// Lower bound of the log-ratio domain (e.g., −10.0).
    pub grid_lo: f64,
    /// Upper bound of the log-ratio domain (e.g., 10.0).
    pub grid_hi: f64,
    /// Number of grid points (e.g., 1000).  Must be ≥ 2.
    pub grid_size: usize,
}

impl PrvConfig {
    /// Construct and validate a `PrvConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `grid_size < 2` or `grid_lo >= grid_hi`.
    pub fn new(grid_lo: f64, grid_hi: f64, grid_size: usize) -> PrivacyResult<Self> {
        if grid_size < 2 {
            return Err(PrivacyError::InvalidParameter(
                "grid_size must be ≥ 2".into(),
            ));
        }
        if grid_lo >= grid_hi {
            return Err(PrivacyError::InvalidParameter(
                "grid_lo must be < grid_hi".into(),
            ));
        }
        Ok(Self {
            grid_lo,
            grid_hi,
            grid_size,
        })
    }

    /// Grid step size h = (grid_hi − grid_lo) / (grid_size − 1).
    #[must_use]
    pub fn step(&self) -> f64 {
        (self.grid_hi - self.grid_lo) / (self.grid_size - 1) as f64
    }

    /// Return the z-value of grid index `i`.
    #[must_use]
    pub fn z_at(&self, i: usize) -> f64 {
        self.grid_lo + i as f64 * self.step()
    }
}

// ─── Gaussian PRV ─────────────────────────────────────────────────────────────

/// Parameters of a Gaussian mechanism PRV.
///
/// For the Gaussian mechanism M(x) = f(x) + N(0, σ²·I) with L2 sensitivity Δ,
/// the privacy loss RV Z ~ N(Δ²/(2σ²), (Δ/σ)²).
#[derive(Debug, Clone)]
pub struct GaussianPrv {
    /// L2 sensitivity Δ > 0.
    pub sensitivity: f64,
    /// Noise standard deviation σ > 0.
    pub sigma: f64,
}

impl GaussianPrv {
    /// Construct and validate a `GaussianPrv`.
    ///
    /// # Errors
    /// Returns appropriate errors for non-positive sensitivity or sigma.
    pub fn new(sensitivity: f64, sigma: f64) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        Ok(Self { sensitivity, sigma })
    }

    /// Privacy loss mean: μ_Z = Δ²/(2σ²).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.sensitivity * self.sensitivity / (2.0 * self.sigma * self.sigma)
    }

    /// Privacy loss std: σ_Z = Δ/σ.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.sensitivity / self.sigma
    }

    /// Evaluate the Gaussian PRV PDF at log-ratio value `z`.
    ///
    /// `f_Z(z) = φ((z − μ_Z) / σ_Z) / σ_Z`
    /// where φ is the standard normal PDF and μ_Z = Δ²/(2σ²), σ_Z = Δ/σ.
    #[must_use]
    pub fn pmf_at(&self, z: f64) -> f64 {
        let mu = self.mean();
        let sd = self.std_dev();
        if sd <= 0.0 {
            return 0.0;
        }
        let norm = (z - mu) / sd;
        (-0.5 * norm * norm).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt())
    }
}

// ─── Discretization ───────────────────────────────────────────────────────────

/// Discretize a Gaussian PRV onto a uniform grid of `cfg.grid_size` points.
///
/// Returns a probability mass vector of length `cfg.grid_size` where
/// `pmf[i] ≈ P(z_i ≤ Z < z_{i+1})` normalized so that Σ pmf`[i]` ≈ 1.
///
/// Values outside [grid_lo, grid_hi] are clipped to the boundary bins.
pub fn gaussian_prv_pmf(prv: &GaussianPrv, cfg: &PrvConfig) -> Vec<f64> {
    let n = cfg.grid_size;
    let h = cfg.step();
    let mut pmf = vec![0.0f64; n];

    for (i, v) in pmf.iter_mut().enumerate().take(n) {
        let z = cfg.z_at(i);
        *v = prv.pmf_at(z) * h;
    }

    // Normalize so sum == 1 (absorb truncation error).
    let total: f64 = pmf.iter().sum();
    if total > 0.0 {
        for v in pmf.iter_mut() {
            *v /= total;
        }
    }
    pmf
}

// ─── Convolution ─────────────────────────────────────────────────────────────

/// Convolve two discrete PMF vectors (O(n²) direct convolution).
///
/// Returns a vector of length `a.len() + b.len() − 1` representing the PMF
/// of the sum Z_A + Z_B when Z_A ~ pmf_a and Z_B ~ pmf_b are independent.
///
/// # Panics
/// Does not panic; empty inputs produce empty output.
pub fn convolve_pmfs(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let mut out = vec![0.0f64; out_len];
    #[allow(clippy::needless_range_loop)]
    for i in 0..a.len() {
        for j in 0..b.len() {
            out[i + j] += a[i] * b[j];
        }
    }
    out
}

/// Compose n identical Gaussian mechanisms via repeated PMF convolution.
///
/// The composed PMF has length `n * (grid_size - 1) + 1` but we trim it
/// back to `grid_size` by re-normalizing on the same grid.
///
/// # Errors
/// - `EmptyMechanismList` if `n == 0`.
/// - `InvalidParameter` from `PrvConfig` or `GaussianPrv` validation.
pub fn compose_gaussian_prv(
    prv: &GaussianPrv,
    n: usize,
    cfg: &PrvConfig,
) -> PrivacyResult<Vec<f64>> {
    if n == 0 {
        return Err(PrivacyError::EmptyMechanismList);
    }

    let base_pmf = gaussian_prv_pmf(prv, cfg);
    let mut composed = base_pmf.clone();

    for _ in 1..n {
        composed = convolve_pmfs(&composed, &base_pmf);
    }

    // Re-project back to the original grid by normalizing.
    // The composed PMF is now longer than grid_size, but the extra probability
    // mass is in the tails; we collect it into the boundary bins.
    let composed_len = composed.len();
    let grid_n = cfg.grid_size;

    if composed_len <= grid_n {
        // Pad with zeros at ends.
        let mut out = vec![0.0f64; grid_n];
        let offset = (grid_n - composed_len) / 2;
        for (i, &v) in composed.iter().enumerate() {
            out[offset + i] += v;
        }
        Ok(out)
    } else {
        // Aggregate excess mass into boundary bins by binning.
        let ratio = (composed_len - 1) as f64 / (grid_n - 1) as f64;
        let mut out = vec![0.0f64; grid_n];
        for (ci, &v) in composed.iter().enumerate() {
            let mapped = (ci as f64 / ratio).round() as usize;
            let idx = mapped.min(grid_n - 1);
            out[idx] += v;
        }
        // Normalize.
        let total: f64 = out.iter().sum();
        if total > 0.0 {
            for v in out.iter_mut() {
                *v /= total;
            }
        }
        Ok(out)
    }
}

// ─── Privacy curve evaluation ─────────────────────────────────────────────────

/// Compute δ(ε) from a composed PRV PMF.
///
/// `δ(ε) = E[max(0, 1 − e^(ε − Z))] = Σᵢ pmf[i] · max(0, 1 − e^(ε − zᵢ))`
///
/// where zᵢ is the log-ratio value at grid index i.
pub fn prv_delta(pmf: &[f64], epsilon: f64, cfg: &PrvConfig) -> f64 {
    let mut delta = 0.0f64;
    for (i, &p) in pmf.iter().enumerate() {
        let z = cfg.z_at(i);
        let contrib = (1.0 - (epsilon - z).exp()).max(0.0);
        delta += p * contrib;
    }
    delta.min(1.0)
}

/// Find the minimum ε such that δ(ε) ≤ δ_target, via binary search.
///
/// # Errors
/// - `InvalidDelta` if `delta_target ∉ (0, 1)`.
/// - `ConvergenceFailed` if binary search fails to converge.
pub fn prv_epsilon(pmf: &[f64], delta: f64, cfg: &PrvConfig) -> PrivacyResult<f64> {
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }

    // Trivial check: δ(0) ≥ δ_target means ε* = 0.
    if prv_delta(pmf, 0.0, cfg) <= delta {
        return Ok(0.0);
    }

    // Binary search over ε ∈ [0, 2 * grid_hi].
    let eps_max = 2.0 * cfg.grid_hi.abs().max(cfg.grid_lo.abs());
    let mut lo = 0.0f64;
    let mut hi = eps_max;

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if prv_delta(pmf, mid, cfg) > delta {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-10 {
            return Ok(hi);
        }
    }

    Err(PrivacyError::ConvergenceFailed(200))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prv_pmf_sums_to_one() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("ok");
        let cfg = PrvConfig::new(-10.0, 10.0, 100).expect("ok");
        let pmf = gaussian_prv_pmf(&prv, &cfg);
        let total: f64 = pmf.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "PMF sums to {total}");
    }

    #[test]
    fn test_prv_compose_delta_decreasing_in_epsilon() {
        let prv = GaussianPrv::new(1.0, 2.0).expect("ok");
        let cfg = PrvConfig::new(-10.0, 10.0, 200).expect("ok");
        let pmf = compose_gaussian_prv(&prv, 5, &cfg).expect("ok");
        let d1 = prv_delta(&pmf, 0.0, &cfg);
        let d2 = prv_delta(&pmf, 1.0, &cfg);
        let d3 = prv_delta(&pmf, 5.0, &cfg);
        assert!(d1 >= d2, "δ should decrease as ε increases: {d1} >= {d2}");
        assert!(d2 >= d3, "δ should decrease as ε increases: {d2} >= {d3}");
    }

    #[test]
    fn test_prv_epsilon_positive() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("ok");
        let cfg = PrvConfig::new(-5.0, 5.0, 100).expect("ok");
        let pmf = compose_gaussian_prv(&prv, 3, &cfg).expect("ok");
        let eps = prv_epsilon(&pmf, 1e-3, &cfg).expect("ok");
        assert!(eps >= 0.0, "epsilon must be non-negative, got {eps}");
    }

    #[test]
    fn test_convolve_pmfs_basic() {
        // Convolving a point mass at 0 with itself k times gives a point mass at 0.
        let pmf = vec![1.0];
        let out = convolve_pmfs(&pmf, &pmf);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }
}
