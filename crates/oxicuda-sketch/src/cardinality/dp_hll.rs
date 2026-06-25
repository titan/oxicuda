//! Differentially-private HyperLogLog cardinality estimator.
//!
//! This wraps the integer [`crate::cardinality::hll::HyperLogLog`] sketch and
//! releases a cardinality estimate that satisfies **(ε)-** or **(ε, δ)-
//! differential privacy** (Dwork & Roth 2014). Two principled mechanisms are
//! offered, both backed by an explicit sensitivity analysis. The design follows
//! the DP Count-Min sketch in [`crate::frequency::cm_dp`] for budget handling,
//! noise sampling and RNG usage.
//!
//! # Background: the HyperLogLog state
//!
//! A HyperLogLog sketch over precision `p` keeps `m = 2^p` registers
//! `M_0, …, M_{m-1} ∈ {0, 1, …, B}` where `B = 64` is the maximum leading-zero
//! count plus one for a 64-bit hash. Each register holds the **maximum** rho
//! ever observed for the hashes that fall in its bucket. The (non-private)
//! cardinality estimate is the harmonic-mean form
//!
//! ```text
//! Z = Σ_j 2^{-M_j}            (the "inverse register-sum")
//! E = α_m · m² / Z ,
//! ```
//! with a small-range *linear-counting* correction `E = m · ln(m / m_0)` (where
//! `m_0` = number of zero registers) used when the harmonic estimate is small.
//!
//! # Neighboring datasets and sensitivity
//!
//! Two streams are *neighbors* if one is obtained from the other by adding or
//! removing a single element. Because each element hashes to exactly one bucket
//! and only ever *raises* that bucket's register (a `max`), adding one element:
//!
//! * changes **at most one** register `M_j` (L0 / change-count sensitivity = 1),
//!   and
//! * can only **increase** that register, by some amount `M_j → M_j' ≥ M_j`.
//!
//! These two facts are the crux of both mechanisms below.
//!
//! ## Mechanism A — per-register geometric (the register / "flip" mechanism)
//!
//! We add an independent two-sided geometric variate to **every** register and
//! release the noisy register vector (clamped back to `{0, …, B}`); the estimate
//! is then computed from the noisy registers. The two-sided geometric mechanism
//! (Ghosh, Roughgarden & Sundararajan 2009) with ratio `α = exp(-ε / Δ_reg)`
//! makes a single integer register `ε / Δ_reg`-DP, where `Δ_reg` is the largest
//! value change a single element can induce in one register. Since neighboring
//! streams differ in **at most one** register, the released vector inherits the
//! same `(Δ_reg · ε / Δ_reg) = ε` guarantee: the worst-case privacy loss is
//! confined to the single coordinate that can change, so no composition over the
//! `m` registers is incurred (cf. parallel composition / bounded L0 change).
//! Conservatively we take `Δ_reg = B` (a register may jump from `0` to `B`).
//! Releasing the noisy registers — and hence the estimate derived from them — is
//! **ε-DP**.
//!
//! ## Mechanism B — Laplace / Gaussian on the inverse register-sum
//!
//! Instead of perturbing every register we perturb the two scalars the estimate
//! depends on — the inverse register-sum `Z` and the zero-register count `m_0` —
//! and post-process. One element changes one register `M_j → M_j'` with
//! `M_j' > M_j`, so
//!
//! ```text
//! |ΔZ|   = |2^{-M_j} − 2^{-M_j'}| ≤ 2^{-M_j} ≤ 2^0 = 1 ,
//! |Δm_0| ≤ 1     (that one register may flip from zero to non-zero).
//! ```
//!
//! Hence the **L1/L2 sensitivity of `Z` is `1`** and of `m_0` is `1`, regardless
//! of the cardinality. We split the budget evenly, releasing
//! `Z̃ = Z + noise(ε/2)` and `m̃_0 = m_0 + noise(ε/2)`, then post-process them
//! into the estimate (harmonic form, clamped to `[Z_min, m]`, or linear counting
//! when small). By basic sequential composition the pair is `ε`-DP, and
//! post-processing preserves it.
//!
//! * **Laplace** noise of scale `b = Δ / (ε/2) = 2/ε` on each quantity gives
//!   **ε-DP** overall.
//! * **Gaussian** noise of standard deviation
//!   `σ = Δ · √(2 ln(1.25/δ')) / (ε/2)` per quantity (with `δ' = δ/2`) gives
//!   **(ε, δ)-DP** overall (the analytic Gaussian mechanism bound of Dwork &
//!   Roth 2014, Thm 3.22; valid for `ε ∈ (0, 1)`).
//!
//! # Utility
//!
//! All noise here is mean-zero, so the noisy quantities (and therefore the
//! estimate) are asymptotically unbiased and, for large cardinalities where
//! `Z ≫ ΔZ`, the relative perturbation vanishes — the noisy estimate
//! concentrates around the true HLL estimate, and a larger `ε` (smaller noise)
//! tightens that concentration.

use crate::cardinality::hll::HyperLogLog;
use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Maximum register value `B` for a 64-bit hash (leading-zeros + 1 ≤ 64).
const MAX_REGISTER: i64 = 64;

/// Noise mechanism used to privatise the HyperLogLog estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpHllMechanism {
    /// Per-register two-sided geometric ("flip") mechanism. Pure **ε-DP**;
    /// ignores `delta`.
    RegisterGeometric,
    /// Laplace noise on `(Z, m_0)`. Pure **ε-DP**; ignores `delta`.
    LaplaceEstimate,
    /// Gaussian noise on `(Z, m_0)`. **(ε, δ)-DP**; requires `0 < delta < 1`.
    GaussianEstimate,
}

/// Configuration for a [`DpHll`] sketch.
#[derive(Debug, Clone, Copy)]
pub struct DpHllConfig {
    /// HyperLogLog precision `p` (`m = 2^p` registers). Must be in `[4, 16]`.
    pub precision: u32,
    /// Privacy budget `ε > 0`. Smaller ε ⇒ more privacy ⇒ more noise.
    pub epsilon: f64,
    /// Privacy slack `δ`. Only used by [`DpHllMechanism::GaussianEstimate`],
    /// where it must satisfy `0 < δ < 1`.
    pub delta: f64,
    /// Which noise mechanism to apply.
    pub mechanism: DpHllMechanism,
}

impl DpHllConfig {
    /// Convenience constructor for the pure-ε Laplace mechanism (`δ` unused).
    #[must_use]
    pub fn laplace(precision: u32, epsilon: f64) -> Self {
        Self {
            precision,
            epsilon,
            delta: 0.0,
            mechanism: DpHllMechanism::LaplaceEstimate,
        }
    }

    /// Convenience constructor for the per-register geometric mechanism.
    #[must_use]
    pub fn register_geometric(precision: u32, epsilon: f64) -> Self {
        Self {
            precision,
            epsilon,
            delta: 0.0,
            mechanism: DpHllMechanism::RegisterGeometric,
        }
    }

    /// Convenience constructor for the (ε, δ) Gaussian mechanism.
    #[must_use]
    pub fn gaussian(precision: u32, epsilon: f64, delta: f64) -> Self {
        Self {
            precision,
            epsilon,
            delta,
            mechanism: DpHllMechanism::GaussianEstimate,
        }
    }
}

/// Differentially-private HyperLogLog cardinality estimator.
///
/// Inserts are accumulated into a plain [`HyperLogLog`]; [`DpHll::estimate`]
/// releases a noisy, differentially-private cardinality. See the module docs for
/// the (ε[, δ])-DP guarantee and the sensitivity argument for each mechanism.
#[derive(Debug, Clone)]
pub struct DpHll {
    /// Underlying exact HyperLogLog state.
    inner: HyperLogLog,
    /// Privacy budget ε.
    epsilon: f64,
    /// Privacy slack δ (Gaussian mechanism only).
    delta: f64,
    /// Active noise mechanism.
    mechanism: DpHllMechanism,
    /// Independent RNG used only for noise sampling (derived from the user RNG).
    noise_rng: LcgRng,
}

impl DpHll {
    /// Create a DP HyperLogLog from `cfg`, drawing an independent noise-RNG seed
    /// from `rng` (so the construction is reproducible from a single seed).
    ///
    /// # Errors
    ///
    /// * [`SketchError::InvalidPrecision`] — `precision` outside `[4, 16]`.
    /// * [`SketchError::InvalidParameter`] — non-finite or non-positive `ε`, or
    ///   (for the Gaussian mechanism) a `δ` outside `(0, 1)`.
    pub fn new(cfg: DpHllConfig, rng: &mut LcgRng) -> SketchResult<Self> {
        if !(cfg.epsilon.is_finite() && cfg.epsilon > 0.0) {
            return Err(SketchError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "must be finite and > 0".to_string(),
            });
        }
        if cfg.mechanism == DpHllMechanism::GaussianEstimate
            && !(cfg.delta.is_finite() && cfg.delta > 0.0 && cfg.delta < 1.0)
        {
            return Err(SketchError::InvalidParameter {
                name: "delta".to_string(),
                reason: "Gaussian mechanism requires 0 < delta < 1".to_string(),
            });
        }
        // HyperLogLog::new validates precision and returns InvalidPrecision.
        // The HLL seed is derived from the user RNG so it is reproducible.
        let hll_seed = rng.next_u64();
        let inner = HyperLogLog::new(cfg.precision, hll_seed)?;
        let noise_seed = rng.next_u64();
        Ok(Self {
            inner,
            epsilon: cfg.epsilon,
            delta: cfg.delta,
            mechanism: cfg.mechanism,
            noise_rng: LcgRng::new(noise_seed),
        })
    }

    /// Insert a pre-hashed 64-bit item into the sketch.
    pub fn add(&mut self, item_hash: u64) {
        self.inner.add_hash(item_hash);
    }

    /// Insert a `u64` value (hashed internally by the underlying HLL).
    pub fn add_u64(&mut self, x: u64) {
        self.inner.add_u64(x);
    }

    /// The configured privacy budget ε.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// The configured privacy slack δ (meaningful for the Gaussian mechanism).
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// The active noise mechanism.
    #[must_use]
    pub fn mechanism(&self) -> DpHllMechanism {
        self.mechanism
    }

    /// Number of registers `m = 2^p`.
    #[must_use]
    pub fn num_registers(&self) -> usize {
        self.inner.m
    }

    /// **Non-private** exact HyperLogLog estimate. For testing / diagnostics
    /// only — releasing this value carries **no** privacy guarantee.
    #[must_use]
    pub fn true_estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Release a **differentially-private** cardinality estimate.
    ///
    /// Each call draws fresh noise; under a fixed privacy budget the intended
    /// usage is a single release (repeated releases compose and spend ε each).
    /// Never returns a negative or non-finite value.
    pub fn estimate(&mut self) -> f64 {
        match self.mechanism {
            DpHllMechanism::RegisterGeometric => self.estimate_register_geometric(),
            DpHllMechanism::LaplaceEstimate => self.estimate_noisy_scalars(NoiseKind::Laplace),
            DpHllMechanism::GaussianEstimate => self.estimate_noisy_scalars(NoiseKind::Gaussian),
        }
    }

    /// Reset the underlying sketch to empty (does not reseed the noise RNG).
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    // ── Mechanism A: per-register geometric ───────────────────────────────────

    /// Compute the estimate from noisy registers under the two-sided geometric
    /// mechanism (Mechanism A). The ratio is `α = exp(-ε / B)` with `B = 64` the
    /// per-register value-sensitivity bound; this yields **ε-DP** because at most
    /// one register differs between neighboring streams.
    fn estimate_register_geometric(&mut self) -> f64 {
        let m = self.inner.m as f64;
        let alpha = alpha_m(self.inner.m);
        // Two-sided geometric ratio. Larger ε ⇒ α closer to 0 ⇒ less noise.
        let ratio = (-self.epsilon / MAX_REGISTER as f64).exp();
        let mut z = 0.0_f64;
        let mut zero_count = 0usize;
        for &reg in &self.inner.registers {
            let noisy =
                two_sided_geometric(reg as i64, ratio, &mut self.noise_rng).clamp(0, MAX_REGISTER);
            z += 2.0_f64.powi(-(noisy as i32));
            if noisy == 0 {
                zero_count += 1;
            }
        }
        finalize_estimate(alpha, m, z, zero_count as f64, self.inner.m)
    }

    // ── Mechanism B: Laplace / Gaussian on (Z, m_0) ──────────────────────────

    /// Compute the estimate from a noisy inverse register-sum `Z` and noisy
    /// zero-register count `m_0` (Mechanism B). Each scalar has sensitivity `1`;
    /// the budget is split evenly, so Laplace scale `2/ε` ⇒ ε-DP and Gaussian
    /// σ = √(2 ln(1.25/δ'))·2/ε with δ' = δ/2 ⇒ (ε, δ)-DP.
    fn estimate_noisy_scalars(&mut self, kind: NoiseKind) -> f64 {
        let m = self.inner.m as f64;
        let alpha = alpha_m(self.inner.m);
        let mut z = 0.0_f64;
        let mut zero_count = 0usize;
        for &reg in &self.inner.registers {
            z += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zero_count += 1;
            }
        }
        // Split the budget across the two released scalars (both sensitivity 1).
        let eps_half = self.epsilon / 2.0;
        let sensitivity = 1.0_f64;
        let (noise_z, noise_m0) = match kind {
            NoiseKind::Laplace => {
                let scale = sensitivity / eps_half;
                (
                    laplace_sample(scale, &mut self.noise_rng),
                    laplace_sample(scale, &mut self.noise_rng),
                )
            }
            NoiseKind::Gaussian => {
                let delta_half = self.delta / 2.0;
                let sigma = sensitivity * (2.0 * (1.25 / delta_half).ln()).sqrt() / eps_half;
                (
                    sigma * self.noise_rng.next_normal(),
                    sigma * self.noise_rng.next_normal(),
                )
            }
        };
        // Post-process: clamp Z to its physical range [Z_min, m] and m_0 to
        // [0, m]; this preserves the DP guarantee and stabilises the estimate.
        let z_min = m * 2.0_f64.powi(-(MAX_REGISTER as i32));
        let z_noisy = (z + noise_z).clamp(z_min, m);
        let m0_noisy = (zero_count as f64 + noise_m0).clamp(0.0, m);
        finalize_estimate(alpha, m, z_noisy, m0_noisy, self.inner.m)
    }
}

/// Which additive-noise distribution Mechanism B uses.
#[derive(Debug, Clone, Copy)]
enum NoiseKind {
    Laplace,
    Gaussian,
}

/// HyperLogLog `alpha_m` bias-correction constant (Flajolet 2007).
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

/// Finalise a harmonic-mean estimate with the standard small-range (linear
/// counting) correction, matching [`HyperLogLog::estimate`]. `zero_count` may be
/// fractional (a noisy estimate of `m_0`).
fn finalize_estimate(alpha: f64, m: f64, z: f64, zero_count: f64, m_int: usize) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let raw = alpha * m * m / z;
    if raw <= 2.5 * m && zero_count > 0.0 {
        let lc = m * (m_int as f64 / zero_count).ln();
        return if lc.is_finite() && lc >= 0.0 { lc } else { 0.0 };
    }
    if raw.is_finite() && raw >= 0.0 {
        raw
    } else {
        0.0
    }
}

/// Draw a two-sided (symmetric) geometric variate centred at `center`.
///
/// The two-sided geometric distribution has pmf `∝ ratio^{|k|}` over the
/// integers (`0 < ratio < 1`); this is the discrete analogue of the Laplace
/// distribution and is the noise of the geometric mechanism (Ghosh,
/// Roughgarden & Sundararajan 2009). Sampling uses the difference of two i.i.d.
/// geometric variates, each produced by inverse-CDF on a uniform.
fn two_sided_geometric(center: i64, ratio: f64, rng: &mut LcgRng) -> i64 {
    // Degenerate guards: ratio outside (0,1) ⇒ no usable noise.
    if !(ratio.is_finite() && ratio > 0.0 && ratio < 1.0) {
        return center;
    }
    let g1 = geometric_sample(ratio, rng);
    let g2 = geometric_sample(ratio, rng);
    center + g1 - g2
}

/// Draw a geometric variate on `{0, 1, 2, …}` with success parameter
/// `1 - ratio` (so `P[K = k] = (1 - ratio) · ratio^k`). Inverse-CDF on a
/// uniform `u ∈ [0, 1)`: `k = ⌊ln(1 - u) / ln(ratio)⌋`.
fn geometric_sample(ratio: f64, rng: &mut LcgRng) -> i64 {
    let u = rng.next_f64();
    // 1 - u ∈ (0, 1]; clamp away from 0 to avoid -inf from ln.
    let one_minus_u = (1.0 - u).max(1.0e-300);
    let k = (one_minus_u.ln() / ratio.ln()).floor();
    if k.is_finite() && k >= 0.0 {
        k as i64
    } else {
        0
    }
}

/// Draw a Laplace(0, `scale`) variate via inverse-CDF (matches `cm_dp`).
fn laplace_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    let u = rng.next_f64() - 0.5;
    let s = if u < 0.0 { -1.0 } else { 1.0 };
    -scale * s * (1.0 - 2.0 * u.abs()).max(1.0e-300).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_epsilon() {
        let mut rng = LcgRng::new(1);
        assert!(DpHll::new(DpHllConfig::laplace(10, 0.0), &mut rng).is_err());
        assert!(DpHll::new(DpHllConfig::laplace(10, -1.0), &mut rng).is_err());
        assert!(DpHll::new(DpHllConfig::laplace(10, f64::NAN), &mut rng).is_err());
    }

    #[test]
    fn rejects_bad_precision() {
        let mut rng = LcgRng::new(2);
        assert!(DpHll::new(DpHllConfig::laplace(2, 1.0), &mut rng).is_err());
        assert!(DpHll::new(DpHllConfig::laplace(20, 1.0), &mut rng).is_err());
    }

    #[test]
    fn rejects_bad_delta_for_gaussian() {
        let mut rng = LcgRng::new(3);
        assert!(DpHll::new(DpHllConfig::gaussian(10, 0.5, 0.0), &mut rng).is_err());
        assert!(DpHll::new(DpHllConfig::gaussian(10, 0.5, 1.0), &mut rng).is_err());
        assert!(DpHll::new(DpHllConfig::gaussian(10, 0.5, -0.1), &mut rng).is_err());
        // Valid delta should construct fine.
        assert!(DpHll::new(DpHllConfig::gaussian(10, 0.5, 1e-5), &mut rng).is_ok());
    }

    #[test]
    fn laplace_estimate_close_for_large_cardinality() {
        // Large ε ⇒ small noise; for large cardinality the relative error of
        // the noisy estimate should be modest.
        let mut rng = LcgRng::new(7);
        let mut dp = DpHll::new(DpHllConfig::laplace(14, 8.0), &mut rng).expect("ok");
        let n: u64 = 50_000;
        for i in 0..n {
            dp.add_u64(i);
        }
        let truth = n as f64;
        let est = dp.estimate();
        let rel = (est - truth).abs() / truth;
        assert!(
            rel < 0.2,
            "noisy estimate {est} vs truth {truth} (rel {rel})"
        );
    }

    #[test]
    fn register_geometric_within_reasonable_factor() {
        // The per-register geometric mechanism with the honest Δ_reg = 64
        // sensitivity is noisy; with a healthy budget it stays within a small
        // constant factor of the truth for large cardinality.
        let mut rng = LcgRng::new(11);
        let mut dp = DpHll::new(DpHllConfig::register_geometric(14, 30.0), &mut rng).expect("ok");
        let n: u64 = 50_000;
        for i in 0..n {
            dp.add_u64(i);
        }
        let truth = n as f64;
        let mut acc = 0.0;
        let reps = 25;
        for _ in 0..reps {
            acc += dp.estimate();
        }
        let mean = acc / reps as f64;
        let factor = mean / truth;
        assert!(
            (0.4..2.5).contains(&factor),
            "mean geom estimate {mean} not within factor of {truth} (factor {factor})"
        );
    }

    #[test]
    fn gaussian_estimate_close_for_large_cardinality() {
        let mut rng = LcgRng::new(13);
        let mut dp = DpHll::new(DpHllConfig::gaussian(14, 0.9, 1e-5), &mut rng).expect("ok");
        let n: u64 = 50_000;
        for i in 0..n {
            dp.add_u64(i);
        }
        let truth = n as f64;
        let mut acc = 0.0;
        let reps = 30;
        for _ in 0..reps {
            acc += dp.estimate();
        }
        let mean = acc / reps as f64;
        let rel = (mean - truth).abs() / truth;
        assert!(
            rel < 0.3,
            "mean gaussian estimate {mean} vs {truth} (rel {rel})"
        );
    }

    #[test]
    fn larger_epsilon_means_less_noise_laplace() {
        // Compare the average absolute deviation of the noisy estimate from the
        // true HLL estimate under a small vs large ε. Larger ε ⇒ less noise.
        let truth_n: u64 = 20_000;
        let avg_dev = |epsilon: f64, seed: u64| -> f64 {
            let mut rng = LcgRng::new(seed);
            let mut dp = DpHll::new(DpHllConfig::laplace(14, epsilon), &mut rng).expect("ok");
            for i in 0..truth_n {
                dp.add_u64(i);
            }
            let base = dp.true_estimate();
            let reps = 60;
            let mut acc = 0.0;
            for _ in 0..reps {
                acc += (dp.estimate() - base).abs();
            }
            acc / reps as f64
        };
        let dev_small = avg_dev(0.3, 100);
        let dev_large = avg_dev(20.0, 100);
        assert!(
            dev_large < dev_small,
            "larger ε should give less noise: dev_large={dev_large}, dev_small={dev_small}"
        );
    }

    #[test]
    fn larger_epsilon_means_less_noise_geometric() {
        let truth_n: u64 = 20_000;
        let avg_dev = |epsilon: f64, seed: u64| -> f64 {
            let mut rng = LcgRng::new(seed);
            let mut dp =
                DpHll::new(DpHllConfig::register_geometric(14, epsilon), &mut rng).expect("ok");
            for i in 0..truth_n {
                dp.add_u64(i);
            }
            let base = dp.true_estimate();
            let reps = 60;
            let mut acc = 0.0;
            for _ in 0..reps {
                acc += (dp.estimate() - base).abs();
            }
            acc / reps as f64
        };
        let dev_small = avg_dev(0.5, 222);
        let dev_large = avg_dev(60.0, 222);
        assert!(
            dev_large < dev_small,
            "larger ε should give less noise: dev_large={dev_large}, dev_small={dev_small}"
        );
    }

    #[test]
    fn estimate_never_negative_or_nan() {
        // Even on an empty sketch with heavy noise the release stays valid.
        for mech in [
            DpHllMechanism::RegisterGeometric,
            DpHllMechanism::LaplaceEstimate,
            DpHllMechanism::GaussianEstimate,
        ] {
            let cfg = DpHllConfig {
                precision: 8,
                epsilon: 0.1,
                delta: 1e-6,
                mechanism: mech,
            };
            let mut rng = LcgRng::new(31);
            let mut dp = DpHll::new(cfg, &mut rng).expect("ok");
            for _ in 0..100 {
                let v = dp.estimate();
                assert!(v.is_finite() && v >= 0.0, "bad release {v} for {mech:?}");
            }
        }
    }

    #[test]
    fn duplicates_do_not_inflate_private_estimate() {
        // True distinct count is 1; with the linear-counting correction in the
        // Laplace mechanism, the noisy estimate should remain small on average.
        let mut rng = LcgRng::new(41);
        let mut dp = DpHll::new(DpHllConfig::laplace(12, 5.0), &mut rng).expect("ok");
        for _ in 0..5000 {
            dp.add_u64(7);
        }
        let mut acc = 0.0;
        let reps = 40;
        for _ in 0..reps {
            acc += dp.estimate();
        }
        let mean = acc / reps as f64;
        assert!(mean < 60.0, "duplicate-only mean estimate {mean} too large");
    }

    #[test]
    fn fresh_noise_each_release() {
        let mut rng = LcgRng::new(53);
        let mut dp = DpHll::new(DpHllConfig::laplace(12, 0.5), &mut rng).expect("ok");
        for i in 0..1000u64 {
            dp.add_u64(i);
        }
        let a = dp.estimate();
        let b = dp.estimate();
        assert!((a - b).abs() > 1e-12, "noise not redrawn: {a} == {b}");
    }

    #[test]
    fn true_estimate_matches_inner_hll() {
        let mut rng = LcgRng::new(61);
        let mut dp = DpHll::new(DpHllConfig::laplace(14, 1.0), &mut rng).expect("ok");
        let n: u64 = 10_000;
        for i in 0..n {
            dp.add_u64(i);
        }
        let est = dp.true_estimate();
        let rel = (est - n as f64).abs() / n as f64;
        assert!(rel < 0.05, "true_estimate {est} vs {n} (rel {rel})");
        assert_eq!(dp.num_registers(), 1 << 14);
    }

    #[test]
    fn clear_resets_state() {
        let mut rng = LcgRng::new(71);
        let mut dp = DpHll::new(DpHllConfig::register_geometric(12, 50.0), &mut rng).expect("ok");
        for i in 0..2000u64 {
            dp.add_u64(i);
        }
        dp.clear();
        // After clear the true estimate is ~0; noisy stays small with big ε.
        assert!(dp.true_estimate() < 5.0);
    }

    #[test]
    fn geometric_sample_nonnegative() {
        let mut rng = LcgRng::new(83);
        for _ in 0..10_000 {
            let k = geometric_sample(0.5, &mut rng);
            assert!(k >= 0, "geometric sample negative: {k}");
        }
    }
}
