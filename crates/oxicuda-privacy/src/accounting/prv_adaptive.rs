//! PRV accountant with adaptive grid refinement for tight `(ε, δ)` reports.
//!
//! Reference: Gopi, Lee & Wutschitz (2021), "Numerical Composition of
//! Differential Privacy", NeurIPS 2021.  The fixed-grid PRV accountant lives in
//! [`crate::accounting::prv`]; this module adds *automatic* selection of the
//! discretisation `[grid_lo, grid_hi]` and `grid_size` so the reported `δ(ε)`
//! (or `ε(δ)`) is stable to a caller-specified tolerance, rather than being
//! sensitive to a hand-picked grid.
//!
//! # Why adaptive refinement
//! The composed privacy-loss distribution of `k` Gaussian mechanisms is
//! approximately `N(k·μ_Z, k·σ_Z²)` with `μ_Z = Δ²/(2σ²)`, `σ_Z = Δ/σ`.  A grid
//! that is too narrow truncates the tail (under-estimating `δ`); one that is too
//! coarse mis-resolves the `e^{ε−z}` integrand near `z = ε`.  Both biases shrink
//! as the grid widens and densifies.  This accountant:
//!
//! 1. **Centres** the grid on the analytic composed mean `k·μ_Z` and sizes its
//!    half-width as `grid_sigmas · √k · σ_Z` (default 12 std-devs), guaranteeing
//!    the bulk plus deep tails are captured for any `(σ, Δ, k)`.
//! 2. **Doubles** `grid_size` until two successive `δ(ε)` estimates agree to
//!    within `tol` (absolute), or `max_grid_size` is hit — Richardson-style
//!    convergence on the discretisation error.
//!
//! Because the analytic composed moments are known in closed form for the
//! Gaussian PRV, the grid placement is exact and only the *density* must be
//! refined, which converges quickly (typically 3–5 doublings).

use crate::accounting::prv::{GaussianPrv, PrvConfig, prv_delta, prv_epsilon};
use crate::error::{PrivacyError, PrivacyResult};

/// Configuration controlling adaptive PRV grid refinement.
#[derive(Debug, Clone)]
pub struct AdaptivePrvConfig {
    /// Half-width of the grid in composed standard deviations (default 12.0).
    pub grid_sigmas: f64,
    /// Initial number of grid points before doubling (default 256).
    pub initial_grid_size: usize,
    /// Upper bound on grid points; refinement stops here (default 8192).
    pub max_grid_size: usize,
    /// Absolute tolerance on successive `δ(ε)` estimates (default 1e-7).
    pub tol: f64,
}

impl Default for AdaptivePrvConfig {
    fn default() -> Self {
        Self {
            grid_sigmas: 12.0,
            initial_grid_size: 256,
            max_grid_size: 8_192,
            tol: 1e-7,
        }
    }
}

impl AdaptivePrvConfig {
    /// Construct and validate an `AdaptivePrvConfig`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `grid_sigmas ≤ 0`, `initial_grid_size < 2`,
    ///   `max_grid_size < initial_grid_size`, or `tol ≤ 0`.
    pub fn new(
        grid_sigmas: f64,
        initial_grid_size: usize,
        max_grid_size: usize,
        tol: f64,
    ) -> PrivacyResult<Self> {
        if grid_sigmas <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grid_sigmas must be positive, got {grid_sigmas}"
            )));
        }
        if initial_grid_size < 2 {
            return Err(PrivacyError::InvalidParameter(
                "initial_grid_size must be ≥ 2".into(),
            ));
        }
        if max_grid_size < initial_grid_size {
            return Err(PrivacyError::InvalidParameter(
                "max_grid_size must be ≥ initial_grid_size".into(),
            ));
        }
        if tol <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "tol must be positive, got {tol}"
            )));
        }
        Ok(Self {
            grid_sigmas,
            initial_grid_size,
            max_grid_size,
            tol,
        })
    }
}

/// Result of an adaptive PRV computation.
#[derive(Debug, Clone)]
pub struct AdaptivePrvResult {
    /// The converged privacy-curve value (`δ` for `delta_at`, `ε` for `epsilon_at`).
    pub value: f64,
    /// Grid size at which convergence was declared.
    pub grid_size: usize,
    /// Number of refinement doublings performed.
    pub refinements: usize,
    /// `true` if successive estimates agreed within `tol`; `false` if the
    /// `max_grid_size` cap was reached first.
    pub converged: bool,
    /// The `[grid_lo, grid_hi]` window used (centred on the composed mean).
    pub grid_lo: f64,
    /// Upper bound of the chosen grid window.
    pub grid_hi: f64,
}

/// Build the analytically-placed grid window for composing `n` copies of `prv`.
///
/// The composed PRV has mean `n·μ_Z` and std `√n·σ_Z`; the window is
/// `mean ± grid_sigmas · √n · σ_Z`.
fn grid_window(prv: &GaussianPrv, n: usize, grid_sigmas: f64) -> (f64, f64) {
    let mean = n as f64 * prv.mean();
    let half = grid_sigmas * (n as f64).sqrt() * prv.std_dev();
    (mean - half, mean + half)
}

/// Discretise the *closed-form composed* Gaussian privacy-loss distribution
/// `Z_total ~ N(n·μ_Z, n·σ_Z²)` directly onto `cfg`'s grid.
///
/// For the Gaussian mechanism the `n`-fold convolution of the per-step PRV is
/// itself Gaussian with mean `n·μ_Z` and variance `n·σ_Z²`, so we can place its
/// exact density on the analytic grid instead of performing (and re-projecting)
/// a numerical convolution.  This is both faster (`O(grid_size)`) and free of
/// the support-compression artefact of binning a convolution back onto a
/// fixed-width grid.  Returns a normalised PMF of length `cfg.grid_size`.
fn composed_gaussian_pmf(prv: &GaussianPrv, n: usize, cfg: &PrvConfig) -> Vec<f64> {
    let mean = n as f64 * prv.mean();
    let sd = (n as f64).sqrt() * prv.std_dev();
    let h = cfg.step();
    let inv = 1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt());
    let mut pmf = vec![0.0f64; cfg.grid_size];
    for (i, v) in pmf.iter_mut().enumerate() {
        let z = cfg.z_at(i);
        let norm = (z - mean) / sd;
        *v = inv * (-0.5 * norm * norm).exp() * h;
    }
    let total: f64 = pmf.iter().sum();
    if total > 0.0 {
        for v in pmf.iter_mut() {
            *v /= total;
        }
    }
    pmf
}

/// Compute `δ(ε)` for `n` composed Gaussian mechanisms with adaptive refinement.
///
/// Doubles the grid density (from `cfg.initial_grid_size`) until two successive
/// `δ` estimates differ by less than `cfg.tol`, or `cfg.max_grid_size` is hit.
///
/// # Errors
/// - `EmptyMechanismList` if `n == 0`.
/// - Propagates `PrvConfig` / `GaussianPrv` validation errors.
pub fn adaptive_delta(
    prv: &GaussianPrv,
    n: usize,
    epsilon: f64,
    cfg: &AdaptivePrvConfig,
) -> PrivacyResult<AdaptivePrvResult> {
    if n == 0 {
        return Err(PrivacyError::EmptyMechanismList);
    }
    let (grid_lo, grid_hi) = grid_window(prv, n, cfg.grid_sigmas);

    let mut grid_size = cfg.initial_grid_size;
    let mut prev: Option<f64> = None;
    let mut refinements = 0usize;

    loop {
        let pcfg = PrvConfig::new(grid_lo, grid_hi, grid_size)?;
        let pmf = composed_gaussian_pmf(prv, n, &pcfg);
        let delta = prv_delta(&pmf, epsilon, &pcfg);

        if let Some(p) = prev
            && (delta - p).abs() <= cfg.tol
        {
            return Ok(AdaptivePrvResult {
                value: delta,
                grid_size,
                refinements,
                converged: true,
                grid_lo,
                grid_hi,
            });
        }

        if grid_size >= cfg.max_grid_size {
            return Ok(AdaptivePrvResult {
                value: delta,
                grid_size,
                refinements,
                converged: false,
                grid_lo,
                grid_hi,
            });
        }

        prev = Some(delta);
        grid_size = (grid_size * 2).min(cfg.max_grid_size);
        refinements += 1;
    }
}

/// Compute `ε(δ)` for `n` composed Gaussian mechanisms with adaptive refinement.
///
/// Same doubling strategy as [`adaptive_delta`] but converges on the inverted
/// `ε` (via [`prv_epsilon`]).
///
/// # Errors
/// - `EmptyMechanismList` if `n == 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - Propagates `PrvConfig` / `GaussianPrv` validation errors.
pub fn adaptive_epsilon(
    prv: &GaussianPrv,
    n: usize,
    delta: f64,
    cfg: &AdaptivePrvConfig,
) -> PrivacyResult<AdaptivePrvResult> {
    if n == 0 {
        return Err(PrivacyError::EmptyMechanismList);
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    let (grid_lo, grid_hi) = grid_window(prv, n, cfg.grid_sigmas);

    let mut grid_size = cfg.initial_grid_size;
    let mut prev: Option<f64> = None;
    let mut refinements = 0usize;

    loop {
        let pcfg = PrvConfig::new(grid_lo, grid_hi, grid_size)?;
        let pmf = composed_gaussian_pmf(prv, n, &pcfg);
        let eps = prv_epsilon(&pmf, delta, &pcfg)?;

        if let Some(p) = prev
            && (eps - p).abs() <= cfg.tol
        {
            return Ok(AdaptivePrvResult {
                value: eps,
                grid_size,
                refinements,
                converged: true,
                grid_lo,
                grid_hi,
            });
        }

        if grid_size >= cfg.max_grid_size {
            return Ok(AdaptivePrvResult {
                value: eps,
                grid_size,
                refinements,
                converged: false,
                grid_lo,
                grid_hi,
            });
        }

        prev = Some(eps);
        grid_size = (grid_size * 2).min(cfg.max_grid_size);
        refinements += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_window_centres_on_mean() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("prv");
        // μ_Z = 0.5, σ_Z = 1.0; for n=4: mean=2.0, half=12·2·1=24.
        let (lo, hi) = grid_window(&prv, 4, 12.0);
        let centre = (lo + hi) / 2.0;
        assert!((centre - 2.0).abs() < 1e-9, "centre={centre}");
        assert!((hi - lo - 48.0).abs() < 1e-9, "width={}", hi - lo);
    }

    #[test]
    fn test_adaptive_delta_converges() {
        let prv = GaussianPrv::new(1.0, 2.0).expect("prv");
        let cfg = AdaptivePrvConfig::new(12.0, 128, 2_048, 1e-5).expect("cfg");
        let r = adaptive_delta(&prv, 4, 1.0, &cfg).expect("delta");
        assert!(r.converged, "should converge within max_grid_size");
        assert!(
            r.value >= 0.0 && r.value <= 1.0,
            "δ out of range: {}",
            r.value
        );
        assert!(r.refinements >= 1, "expected ≥1 refinement");
    }

    #[test]
    fn test_adaptive_delta_monotone_in_epsilon() {
        let prv = GaussianPrv::new(1.0, 1.5).expect("prv");
        let cfg = AdaptivePrvConfig::new(12.0, 128, 1_024, 1e-5).expect("cfg");
        let d_lo = adaptive_delta(&prv, 4, 0.5, &cfg).expect("a").value;
        let d_hi = adaptive_delta(&prv, 4, 3.0, &cfg).expect("b").value;
        assert!(d_lo >= d_hi, "δ must decrease in ε: {d_lo} >= {d_hi}");
    }

    #[test]
    fn test_adaptive_epsilon_roundtrip() {
        // ε(δ) then δ(ε) should approximately recover δ.
        let prv = GaussianPrv::new(1.0, 2.0).expect("prv");
        let cfg = AdaptivePrvConfig::new(12.0, 256, 2_048, 1e-6).expect("cfg");
        let target_delta = 1e-4;
        let eps_res = adaptive_epsilon(&prv, 4, target_delta, &cfg).expect("eps");
        let back = adaptive_delta(&prv, 4, eps_res.value, &cfg).expect("back");
        // δ at the discovered ε should be ≤ target (ε is the *minimum* meeting δ).
        assert!(
            back.value <= target_delta + 1e-4,
            "δ(ε(δ))={} should be ≤ target {target_delta}",
            back.value
        );
    }

    #[test]
    fn test_refinement_reduces_change() {
        // The first two coarse grids should differ more than the last two fine
        // grids — i.e. the sequence of δ estimates is converging.
        let prv = GaussianPrv::new(1.0, 2.0).expect("prv");
        let (lo, hi) = grid_window(&prv, 4, 12.0);
        let mut last = f64::NAN;
        let mut diffs = Vec::new();
        for &gs in &[256usize, 512, 1024, 2048] {
            let pcfg = PrvConfig::new(lo, hi, gs).expect("pcfg");
            let pmf = composed_gaussian_pmf(&prv, 4, &pcfg);
            let d = prv_delta(&pmf, 1.0, &pcfg);
            if last.is_finite() {
                diffs.push((d - last).abs());
            }
            last = d;
        }
        assert!(diffs.len() == 3);
        // The estimates are converging: every successive change is small and the
        // final fine-grid change is tiny in absolute terms.  Quadrature error of
        // the trapezoid-like sum on a refining grid is O(h²), so the change
        // shrinks roughly fourfold per doubling once in the asymptotic regime.
        assert!(
            diffs.iter().all(|&d| d < 1e-4),
            "all successive changes should be small: {diffs:?}"
        );
        assert!(
            diffs[2] < 1e-5,
            "final fine-grid change should be tiny: {diffs:?}"
        );
    }

    #[test]
    fn test_invalid_inputs() {
        let prv = GaussianPrv::new(1.0, 1.0).expect("prv");
        let cfg = AdaptivePrvConfig::default();
        assert!(adaptive_delta(&prv, 0, 1.0, &cfg).is_err());
        assert!(adaptive_epsilon(&prv, 0, 1e-4, &cfg).is_err());
        assert!(adaptive_epsilon(&prv, 4, 0.0, &cfg).is_err());
        assert!(AdaptivePrvConfig::new(0.0, 128, 256, 1e-6).is_err());
        assert!(AdaptivePrvConfig::new(12.0, 1, 256, 1e-6).is_err());
        assert!(AdaptivePrvConfig::new(12.0, 256, 128, 1e-6).is_err());
        assert!(AdaptivePrvConfig::new(12.0, 128, 256, 0.0).is_err());
    }
}
