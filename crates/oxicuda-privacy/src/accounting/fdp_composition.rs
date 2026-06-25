//! f-DP graphical composition via numerical privacy-loss-distribution (PLD)
//! convolution.
//!
//! Reference: Dong, Roth & Su (2022), "Gaussian Differential Privacy", JRSS-B
//! (the f-DP framework and its `(ε, δ)` ↔ trade-off-function duality); and
//! Koskela, Jälkö & Honkela (2020), "Computing Tight Differential Privacy
//! Guarantees Using FFT" (numerical PLD composition).
//!
//! # The trade-off function and its dual
//! A mechanism's privacy is captured by its **trade-off function**
//! `f(α) = inf{ β : type-I error ≤ α }` — the best achievable type-II error `β`
//! at type-I level `α`.  f-DP composition of `k` mechanisms is the
//! *tensor product* `f₁ ⊗ … ⊗ f_k`, which has **no closed form** in general
//! (only the Gaussian/GDP special case does, via `μ = √Σμᵢ²`).
//!
//! This module composes *arbitrary* mechanisms numerically through the duality
//! between the trade-off function and the **privacy-loss distribution**.  The
//! `(ε, δ)`-DP curve `δ(ε)` is the *convex conjugate* view of `f`:
//!
//! ```text
//!     δ(ε) = 1 + f*(−e^ε)     (supporting-line / Legendre duality),
//!     f(α) = sup_ε  1 − δ(ε) − e^ε · α        (recover trade-off from curve).
//! ```
//!
//! Composition is performed on the PLD (whose CDF yields `δ(ε)`), then the
//! composed `δ(ε)` curve is converted back to the trade-off function by the
//! supremum above — this is the "graphical" / curve-based f-composition.
//!
//! # What this provides over [`crate::accounting::fdp`]
//! `fdp.rs` only composes *Gaussian* μ-GDP via the central-limit `√Σμᵢ²`.  Here
//! we compose a **heterogeneous list** of mechanisms (each given as a discrete
//! PLD, or constructed from a Gaussian `μ`) and report the exact numerical
//! composed `δ(ε)` and trade-off `f(α)` — valid for non-Gaussian primitives such
//! as Laplace, randomised response, or empirically-tabulated mechanisms.

use crate::accounting::fdp::{phi, phi_inv};
use crate::error::{PrivacyError, PrivacyResult};

/// A discrete privacy-loss distribution on a uniform grid of log-likelihood
/// ratios (the "privacy loss" support points).
///
/// `support[i] = lo + i·step` and `mass[i] = P(L = support[i])` under the
/// dominating pair `(P, Q)`; the PLD is the law of `L = ln(dP/dQ)` under `P`.
#[derive(Debug, Clone)]
pub struct FdpPld {
    /// Lower support bound (log-ratio domain).
    pub lo: f64,
    /// Uniform grid spacing between adjacent support points.
    pub step: f64,
    /// Probability mass at each support point (should sum to ≈ 1).
    pub mass: Vec<f64>,
}

impl FdpPld {
    /// Construct and validate a PLD.
    ///
    /// # Errors
    /// - `InvalidParameter` if `step ≤ 0`.
    /// - `EmptyInput` if `mass` is empty.
    pub fn new(lo: f64, step: f64, mass: Vec<f64>) -> PrivacyResult<Self> {
        if step <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "step must be positive, got {step}"
            )));
        }
        if mass.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        Ok(Self { lo, step, mass })
    }

    /// Build the PLD of a μ-GDP (Gaussian) mechanism on a centred grid.
    ///
    /// The Gaussian dominating pair is `P = N(μ, 1)`, `Q = N(0, 1)`, giving the
    /// privacy loss `L = μ·X − μ²/2` for `X ~ N(μ, 1)` — itself Gaussian with
    /// mean `μ²/2` and std `μ`.  The grid spans `mean ± grid_sigmas·μ` with
    /// `grid_size` points.
    ///
    /// Use [`FdpPld::gaussian_on_grid`] when composing *heterogeneous*
    /// mechanisms, so they share an identical grid (required by [`compose_two`]).
    ///
    /// # Errors
    /// - `InvalidParameter` if `mu ≤ 0`, `grid_size < 2`, or `grid_sigmas ≤ 0`.
    pub fn gaussian(mu: f64, grid_size: usize, grid_sigmas: f64) -> PrivacyResult<Self> {
        if mu <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "mu must be positive, got {mu}"
            )));
        }
        if grid_sigmas <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grid_sigmas must be positive, got {grid_sigmas}"
            )));
        }
        let mean = mu * mu / 2.0;
        let half = grid_sigmas * mu;
        FdpPld::gaussian_on_grid(mu, mean - half, mean + half, grid_size)
    }

    /// Build the μ-GDP Gaussian PLD on an *explicit* `[lo, hi]` grid.
    ///
    /// All PLDs intended to be convolved together must be built on the **same**
    /// `(lo, hi, grid_size)` so their step matches, which makes [`compose_two`]
    /// exact (a shared step is required for the discrete convolution to align).
    ///
    /// # Errors
    /// - `InvalidParameter` if `mu ≤ 0`, `grid_size < 2`, or `lo ≥ hi`.
    pub fn gaussian_on_grid(mu: f64, lo: f64, hi: f64, grid_size: usize) -> PrivacyResult<Self> {
        if mu <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "mu must be positive, got {mu}"
            )));
        }
        if grid_size < 2 {
            return Err(PrivacyError::InvalidParameter(
                "grid_size must be ≥ 2".into(),
            ));
        }
        if lo >= hi {
            return Err(PrivacyError::InvalidParameter("lo must be < hi".into()));
        }
        let mean = mu * mu / 2.0;
        let step = (hi - lo) / (grid_size - 1) as f64;
        // Density of L ~ N(mean, mu²): φ((l−mean)/μ)/μ, then bin × step.
        let inv = 1.0 / (mu * (2.0 * std::f64::consts::PI).sqrt());
        let mut mass = vec![0.0f64; grid_size];
        for (i, m) in mass.iter_mut().enumerate() {
            let l = lo + i as f64 * step;
            let z = (l - mean) / mu;
            *m = inv * (-0.5 * z * z).exp() * step;
        }
        let total: f64 = mass.iter().sum();
        if total > 0.0 {
            for m in mass.iter_mut() {
                *m /= total;
            }
        }
        FdpPld::new(lo, step, mass)
    }

    /// Support value at grid index `i`.
    #[must_use]
    pub fn support_at(&self, i: usize) -> f64 {
        self.lo + i as f64 * self.step
    }

    /// Evaluate the `(ε, δ)`-DP curve `δ(ε) = E[(1 − e^{ε − L})₊]` from this PLD.
    ///
    /// This is the standard PLD-to-`δ` map (a.k.a. the "hockey-stick" divergence
    /// at slope `e^ε`).
    #[must_use]
    pub fn delta_at(&self, epsilon: f64) -> f64 {
        let mut delta = 0.0f64;
        for (i, &p) in self.mass.iter().enumerate() {
            let l = self.support_at(i);
            let contrib = (1.0 - (epsilon - l).exp()).max(0.0);
            delta += p * contrib;
        }
        delta.clamp(0.0, 1.0)
    }
}

/// Convolve two PLDs (the law of the *sum* of independent privacy losses).
///
/// O(n·m) direct convolution.  Both PLDs **must share the same step** `h`; the
/// result then lives exactly on the grid with origin `a.lo + b.lo` and the same
/// step `h`, because the privacy loss of a composition is the *sum* of the
/// per-mechanism losses (`L = L_a + L_b`) and the law of a sum of independent
/// lattice variables is the discrete convolution of their laws on the common
/// lattice.  Build all operands with [`FdpPld::gaussian_on_grid`] sharing one
/// `(lo, hi, grid_size)` to guarantee a matching step.
///
/// # Errors
/// - `InvalidParameter` if the two PLDs' steps differ by more than a tight
///   relative tolerance (1e-9), which would invalidate the lattice alignment.
pub fn compose_two(a: &FdpPld, b: &FdpPld) -> PrivacyResult<FdpPld> {
    let rel = (a.step - b.step).abs() / a.step.max(b.step).max(f64::EPSILON);
    if rel > 1e-9 {
        return Err(PrivacyError::InvalidParameter(format!(
            "PLDs must share the same step for exact convolution: {} vs {}",
            a.step, b.step
        )));
    }
    let step = a.step;
    let lo = a.lo + b.lo;
    let out_len = a.mass.len() + b.mass.len() - 1;
    let mut mass = vec![0.0f64; out_len];
    for (i, &ai) in a.mass.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.mass.iter().enumerate() {
            mass[i + j] += ai * bj;
        }
    }
    FdpPld::new(lo, step, mass)
}

/// Compose a list of PLDs into the PLD of the full sequential composition.
///
/// # Errors
/// - `EmptyMechanismList` if `plds` is empty.
/// - Propagates [`compose_two`] alignment errors.
pub fn compose_many(plds: &[FdpPld]) -> PrivacyResult<FdpPld> {
    let (first, rest) = plds.split_first().ok_or(PrivacyError::EmptyMechanismList)?;
    let mut acc = first.clone();
    for p in rest {
        acc = compose_two(&acc, p)?;
    }
    Ok(acc)
}

/// Compose `k` identical PLDs via repeated-squaring convolution (`O(log k)`
/// convolutions instead of `k`).
///
/// # Errors
/// - `EmptyMechanismList` if `k == 0`.
/// - Propagates [`compose_two`] errors.
pub fn compose_self(base: &FdpPld, k: usize) -> PrivacyResult<FdpPld> {
    if k == 0 {
        return Err(PrivacyError::EmptyMechanismList);
    }
    let mut result: Option<FdpPld> = None;
    let mut power = base.clone();
    let mut remaining = k;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = Some(match result {
                None => power.clone(),
                Some(r) => compose_two(&r, &power)?,
            });
        }
        remaining >>= 1;
        if remaining > 0 {
            power = compose_two(&power, &power)?;
        }
    }
    result.ok_or(PrivacyError::EmptyMechanismList)
}

/// Recover the trade-off function value `f(α)` from a composed `δ(ε)` curve.
///
/// By Legendre duality, `f(α) = sup_ε [ 1 − δ(ε) − e^ε · α ]₊`, evaluated by
/// scanning the PLD support points as candidate `ε` values (the curve is
/// piecewise-linear in this representation, so the supremum is attained at a
/// support point).  Returns a `β ∈ [0, 1−α]`.
///
/// # Errors
/// - `InvalidParameter` if `alpha ∉ [0, 1]`.
pub fn tradeoff_from_pld(pld: &FdpPld, alpha: f64) -> PrivacyResult<f64> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PrivacyError::InvalidParameter(format!(
            "alpha must be in [0,1], got {alpha}"
        )));
    }
    let mut best = 0.0f64;
    for i in 0..pld.mass.len() {
        let eps = pld.support_at(i);
        let delta = pld.delta_at(eps);
        let beta = 1.0 - delta - eps.exp() * alpha;
        if beta > best {
            best = beta;
        }
    }
    // Also test ε=0 explicitly (often the binding constraint for large α).
    let delta0 = pld.delta_at(0.0);
    let beta0 = 1.0 - delta0 - alpha;
    if beta0 > best {
        best = beta0;
    }
    Ok(best.clamp(0.0, 1.0 - alpha))
}

/// Find the minimum `ε` with composed `δ(ε) ≤ δ_target` via bisection on the
/// PLD-derived curve.
///
/// # Errors
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - `ConvergenceFailed` if bisection does not converge.
pub fn epsilon_at_delta(pld: &FdpPld, delta: f64) -> PrivacyResult<f64> {
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    if pld.delta_at(0.0) <= delta {
        return Ok(0.0);
    }
    let hi_support = pld.support_at(pld.mass.len() - 1);
    let mut lo = 0.0f64;
    let mut hi = hi_support.abs().max(1.0) * 2.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if pld.delta_at(mid) > delta {
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

/// Analytic Gaussian trade-off function `T_μ(α) = Φ(Φ⁻¹(1−α) − μ)`, for
/// cross-checking the numerical [`tradeoff_from_pld`] against the closed form.
#[must_use]
pub fn gaussian_tradeoff(mu: f64, alpha: f64) -> f64 {
    phi(phi_inv(1.0 - alpha) - mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_gaussian_pld_sums_to_one() {
        let p = FdpPld::gaussian(1.0, 512, 12.0).expect("pld");
        let total: f64 = p.mass.iter().sum();
        assert!(approx(total, 1.0, 1e-9), "PLD mass sums to {total}");
    }

    #[test]
    fn test_gaussian_self_composition_matches_gdp() {
        // Composing k copies of μ-GDP must give √k·μ-GDP (CLT, exact for
        // Gaussians).  We verify the composed δ(ε) matches the analytic
        // √k·μ-GDP δ(ε) closely.
        let mu = 0.8;
        let k = 4usize;
        let base = FdpPld::gaussian(mu, 2048, 14.0).expect("base");
        let composed = compose_self(&base, k).expect("composed");
        let mu_k = mu * (k as f64).sqrt();
        for &eps in &[0.5f64, 1.0, 2.0, 3.0] {
            let num = composed.delta_at(eps);
            // analytic μ_k-GDP δ(ε) = Φ(−ε/μ_k + μ_k/2) − e^ε Φ(−ε/μ_k − μ_k/2)
            let a = -eps / mu_k + mu_k / 2.0;
            let b = -eps / mu_k - mu_k / 2.0;
            let ana = phi(a) - eps.exp() * phi(b);
            assert!(
                approx(num, ana.max(0.0), 5e-3),
                "ε={eps}: numerical δ={num} vs analytic {ana}"
            );
        }
    }

    #[test]
    fn test_compose_two_equals_compose_self_2() {
        let base = FdpPld::gaussian(1.0, 1024, 12.0).expect("base");
        let two_a = compose_two(&base, &base).expect("two");
        let two_b = compose_self(&base, 2).expect("self2");
        let da = two_a.delta_at(1.0);
        let db = two_b.delta_at(1.0);
        assert!(approx(da, db, 1e-9), "{da} vs {db}");
    }

    #[test]
    fn test_tradeoff_recovers_gaussian() {
        // Single μ-GDP PLD → recovered trade-off f(α) should track the analytic
        // Gaussian trade-off T_μ(α).
        let mu = 1.0;
        let pld = FdpPld::gaussian(mu, 4096, 14.0).expect("pld");
        for &alpha in &[0.2f64, 0.4, 0.6, 0.8] {
            let num = tradeoff_from_pld(&pld, alpha).expect("f");
            let ana = gaussian_tradeoff(mu, alpha);
            assert!(
                approx(num, ana, 3e-2),
                "α={alpha}: numerical f={num} vs analytic {ana}"
            );
        }
    }

    #[test]
    fn test_heterogeneous_composition_costs_more_privacy() {
        // Composition can only *increase* the privacy cost: at a fixed target δ,
        // the composed mechanism's ε must be ≥ either component's ε.  This is
        // grid-robust (unlike a pointwise δ(ε) comparison) and is the defining
        // property of sequential composition.  Heterogeneous mechanisms must
        // share a common grid so the convolution aligns.
        let lo = -16.0;
        let hi = 16.0;
        let gs = 4096;
        let a = FdpPld::gaussian_on_grid(0.5, lo, hi, gs).expect("a");
        let b = FdpPld::gaussian_on_grid(0.7, lo, hi, gs).expect("b");
        let ab = compose_two(&a, &b).expect("ab");
        let target = 1e-3;
        let eps_ab = epsilon_at_delta(&ab, target).expect("ab eps");
        let eps_a = epsilon_at_delta(&a, target).expect("a eps");
        let eps_b = epsilon_at_delta(&b, target).expect("b eps");
        assert!(
            eps_ab >= eps_a - 1e-2 && eps_ab >= eps_b - 1e-2,
            "composed ε={eps_ab} must be ≥ components ε_a={eps_a}, ε_b={eps_b}"
        );
        // And the composed effective μ = √(0.5²+0.7²) ≈ 0.86 exceeds each.
        assert!(
            eps_ab > eps_a,
            "composed must strictly cost more than μ=0.5"
        );
    }

    #[test]
    fn test_epsilon_at_delta_roundtrip() {
        let base = FdpPld::gaussian(0.6, 2048, 14.0).expect("base");
        let composed = compose_self(&base, 3).expect("composed");
        let target = 1e-4;
        let eps = epsilon_at_delta(&composed, target).expect("eps");
        let d = composed.delta_at(eps);
        assert!(d <= target + 1e-5, "δ(ε)={d} should be ≤ {target}");
        assert!(eps >= 0.0);
    }

    #[test]
    fn test_delta_monotone_in_epsilon() {
        let pld = FdpPld::gaussian(1.0, 1024, 12.0).expect("pld");
        let d0 = pld.delta_at(0.0);
        let d1 = pld.delta_at(1.0);
        let d2 = pld.delta_at(2.0);
        assert!(d0 >= d1 && d1 >= d2, "δ monotone: {d0} {d1} {d2}");
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(FdpPld::new(0.0, 0.0, vec![1.0]).is_err());
        assert!(FdpPld::new(0.0, 1.0, vec![]).is_err());
        assert!(FdpPld::gaussian(0.0, 10, 12.0).is_err());
        assert!(FdpPld::gaussian(1.0, 1, 12.0).is_err());
        assert!(compose_many(&[]).is_err());
        let p = FdpPld::gaussian(1.0, 16, 12.0).expect("p");
        assert!(compose_self(&p, 0).is_err());
        assert!(tradeoff_from_pld(&p, 1.5).is_err());
        assert!(epsilon_at_delta(&p, 0.0).is_err());
    }
}
