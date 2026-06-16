//! Manski partial-identification bounds for the Average Treatment Effect (ATE).
//!
//! References:
//! - Manski, C. F. (1990). "Nonparametric Bounds on Treatment Effects."
//!   *American Economic Review*, 80(2), 319-323.
//! - Manski, C. F., & Pepper, J. V. (2000). "Monotone Instrumental Variables:
//!   With an Application to the Returns to Schooling."
//!   *American Economic Review*, 90(4), 997-1010.
//!
//! # Sharp ATE Bounds
//!
//! Under partial identification, the ATE = E[Y(1)] − E[Y(0)] is not point-
//! identified without assumptions. Manski (1990) derives worst-case sharp
//! bounds by imputing counterfactual outcomes with the extreme values
//! `y_lower` and `y_upper` of the outcome support.
//!
//! ## NoAssumption (Manski 1990 Eq 1)
//!
//! ```text
//! E[Y(1)] ∈ [p₁·ȳ₁ + p₀·y_lo,  p₁·ȳ₁ + p₀·y_hi]
//! E[Y(0)] ∈ [p₀·ȳ₀ + p₁·y_lo,  p₀·ȳ₀ + p₁·y_hi]
//! ```
//!
//! The ATE interval width equals `y_upper − y_lower` when `p₁ · p₀ > 0`.
//!
//! ## MeanIndependence
//!
//! `E[Y(t)|T=t'] = E[Y(t)]` ⟹ point identification.
//!
//! ## MonotoneTreatmentResponse (MTR, Manski 1997)
//!
//! Y(1) ≥ Y(0) a.s. ⟹ ATE ≥ 0; otherwise same no-assumption bounds.
//!
//! ## MonotoneTreatmentSelection (MTS, Manski-Pepper 2000)
//!
//! E[Y(t)|T=1] ≥ E[Y(t)|T=0] ⟹ observed treated mean is an upper bound
//! for E[Y(1)] and observed control mean is a lower bound for E[Y(0)].

use crate::error::{CausalError, CausalResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Identifying assumptions that tighten the Manski ATE bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ManskiAssumption {
    /// No assumptions: worst-case sharp bounds (Manski 1990).
    NoAssumption,
    /// Mean independence: E[Y(t)|T] = E[Y(t)] — point identification.
    MeanIndependence,
    /// Monotone treatment response: Y(1) ≥ Y(0) a.s. (MTR).
    MonotoneTreatmentResponse,
    /// Monotone treatment selection: E[Y(t)|T=1] ≥ E[Y(t)|T=0] (MTS).
    MonotoneTreatmentSelection,
}

/// Configuration for [`ManskiBounds::ate_bounds`].
#[derive(Debug, Clone)]
pub struct ManskiConfig {
    /// Lower bound of the outcome support.
    pub y_lower: f64,
    /// Upper bound of the outcome support (must be strictly greater than
    /// `y_lower`).
    pub y_upper: f64,
    /// Identifying assumption to apply.
    pub assumption: ManskiAssumption,
}

/// Sharp ATE bounds returned by [`ManskiBounds::ate_bounds`].
#[derive(Debug, Clone)]
pub struct ManskiResult {
    /// Lower bound on the ATE.
    pub ate_lower: f64,
    /// Upper bound on the ATE.
    pub ate_upper: f64,
    /// Interval width: `ate_upper − ate_lower`.
    pub width: f64,
    /// Lower bound on E[Y(1)].
    pub e_y1_lower: f64,
    /// Upper bound on E[Y(1)].
    pub e_y1_upper: f64,
    /// Lower bound on E[Y(0)].
    pub e_y0_lower: f64,
    /// Upper bound on E[Y(0)].
    pub e_y0_upper: f64,
}

/// Stateless namespace for Manski partial-identification bound computations.
pub struct ManskiBounds;

impl ManskiBounds {
    /// Compute sharp ATE bounds from observed data.
    ///
    /// # Parameters
    /// - `y` — outcome observations (length `n`).
    /// - `t` — binary treatment indicator (0 or 1, same length as `y`).
    /// - `cfg` — outcome support bounds and identifying assumption.
    ///
    /// # Errors
    /// - [`CausalError::EmptyInput`] if `y` is empty.
    /// - [`CausalError::DimensionMismatch`] if `y.len() != t.len()`.
    /// - [`CausalError::InvalidParameter`] if `y_lower >= y_upper` or if
    ///   both the treated and control groups are simultaneously empty.
    pub fn ate_bounds(y: &[f64], t: &[u8], cfg: &ManskiConfig) -> CausalResult<ManskiResult> {
        // ── validation ────────────────────────────────────────────────────
        if y.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        if y.len() != t.len() {
            return Err(CausalError::DimensionMismatch {
                expected: y.len(),
                got: t.len(),
            });
        }
        if cfg.y_lower >= cfg.y_upper {
            return Err(CausalError::InvalidParameter {
                reason: "y_lower must be < y_upper".into(),
            });
        }

        let n = y.len();
        let y_lo = cfg.y_lower;
        let y_hi = cfg.y_upper;

        // ── observed group means ──────────────────────────────────────────
        let mut sum1 = 0.0_f64;
        let mut cnt1 = 0_usize;
        let mut sum0 = 0.0_f64;
        let mut cnt0 = 0_usize;

        for (i, &ti) in t.iter().enumerate() {
            if ti == 1 {
                sum1 += y[i];
                cnt1 += 1;
            } else {
                sum0 += y[i];
                cnt0 += 1;
            }
        }

        if cnt1 == 0 && cnt0 == 0 {
            return Err(CausalError::InvalidParameter {
                reason: "no observations in either treatment group".into(),
            });
        }

        let p1 = cnt1 as f64 / n as f64;
        let p0 = 1.0 - p1;
        let ybar_1 = if cnt1 > 0 { sum1 / cnt1 as f64 } else { 0.0 };
        let ybar_0 = if cnt0 > 0 { sum0 / cnt0 as f64 } else { 0.0 };

        // ── compute bounds per assumption ─────────────────────────────────
        let (e_y1_lower, e_y1_upper, e_y0_lower, e_y0_upper, ate_lower, ate_upper) =
            match cfg.assumption {
                ManskiAssumption::NoAssumption => {
                    no_assumption_bounds(p1, p0, ybar_1, ybar_0, y_lo, y_hi)
                }
                ManskiAssumption::MeanIndependence => mean_independence_bounds(ybar_1, ybar_0),
                ManskiAssumption::MonotoneTreatmentResponse => {
                    mtr_bounds(p1, p0, ybar_1, ybar_0, y_lo, y_hi)
                }
                ManskiAssumption::MonotoneTreatmentSelection => {
                    mts_bounds(p1, p0, ybar_1, ybar_0, y_lo, y_hi)
                }
            };

        let width = ate_upper - ate_lower;

        Ok(ManskiResult {
            ate_lower,
            ate_upper,
            width,
            e_y1_lower,
            e_y1_upper,
            e_y0_lower,
            e_y0_upper,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Assumption-specific bound computations
// ─────────────────────────────────────────────────────────────────────────────

/// Manski (1990) no-assumption sharp bounds.
///
/// E[Y(1)] ∈ [p₁·ȳ₁ + p₀·y_lo,  p₁·ȳ₁ + p₀·y_hi]
/// E[Y(0)] ∈ [p₀·ȳ₀ + p₁·y_lo,  p₀·ȳ₀ + p₁·y_hi]
fn no_assumption_bounds(
    p1: f64,
    p0: f64,
    ybar_1: f64,
    ybar_0: f64,
    y_lo: f64,
    y_hi: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let ey1_lo = p1 * ybar_1 + p0 * y_lo;
    let ey1_hi = p1 * ybar_1 + p0 * y_hi;
    let ey0_lo = p0 * ybar_0 + p1 * y_lo;
    let ey0_hi = p0 * ybar_0 + p1 * y_hi;
    let ate_lo = ey1_lo - ey0_hi;
    let ate_hi = ey1_hi - ey0_lo;
    (ey1_lo, ey1_hi, ey0_lo, ey0_hi, ate_lo, ate_hi)
}

/// Mean independence ⟹ point identification.
///
/// E[Y(1)] = ȳ₁ (no bias), E[Y(0)] = ȳ₀ (no bias).
fn mean_independence_bounds(ybar_1: f64, ybar_0: f64) -> (f64, f64, f64, f64, f64, f64) {
    let ate = ybar_1 - ybar_0;
    (ybar_1, ybar_1, ybar_0, ybar_0, ate, ate)
}

/// Monotone treatment response (MTR): Y(1) ≥ Y(0) ⟹ ATE ≥ 0.
///
/// No-assumption E[Y(t)] bounds apply; only ATE lower is lifted to 0.
fn mtr_bounds(
    p1: f64,
    p0: f64,
    ybar_1: f64,
    ybar_0: f64,
    y_lo: f64,
    y_hi: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let (ey1_lo, ey1_hi, ey0_lo, ey0_hi, ate_lo_na, ate_hi) =
        no_assumption_bounds(p1, p0, ybar_1, ybar_0, y_lo, y_hi);
    let ate_lo = ate_lo_na.max(0.0);
    (ey1_lo, ey1_hi, ey0_lo, ey0_hi, ate_lo, ate_hi)
}

/// Monotone treatment selection (MTS, Manski-Pepper 2000 Thm 1).
///
/// E[Y(t)|T=1] ≥ E[Y(t)|T=0] ⟹:
/// - E[Y(1)] upper ≤ ȳ₁ (MTS tightens: observable treated mean is an upper
///   bound for E[Y(1)|T=0]).
/// - E[Y(0)] lower ≥ ȳ₀ (MTS tightens: observable control mean is a lower
///   bound for E[Y(0)|T=1]).
fn mts_bounds(
    p1: f64,
    p0: f64,
    ybar_1: f64,
    ybar_0: f64,
    y_lo: f64,
    y_hi: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    // No-assumption lower bounds remain.
    let ey1_lo = p1 * ybar_1 + p0 * y_lo;
    // MTS tightens upper: E[Y(1)] ≤ ȳ₁ (since E[Y(1)|T=0] ≤ ȳ₁ under MTS).
    let ey1_hi = ybar_1;
    // MTS tightens lower: E[Y(0)] ≥ ȳ₀.
    let ey0_lo = ybar_0;
    // No-assumption upper bound for E[Y(0)] remains.
    let ey0_hi = p0 * ybar_0 + p1 * y_hi;
    let ate_lo = ey1_lo - ey0_hi;
    let ate_hi = ey1_hi - ey0_lo;
    (ey1_lo, ey1_hi, ey0_lo, ey0_hi, ate_lo, ate_hi)
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests (16 tests)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // ── helper DGPs ──────────────────────────────────────────────────────────

    fn make_balanced(n: usize) -> (Vec<f64>, Vec<u8>) {
        // n/2 treated (y=1.0), n/2 control (y=0.5).
        let half = n / 2;
        let y: Vec<f64> = (0..half)
            .map(|_| 1.0_f64)
            .chain((0..half).map(|_| 0.5_f64))
            .collect();
        let t: Vec<u8> = (0..half)
            .map(|_| 1u8)
            .chain((0..half).map(|_| 0u8))
            .collect();
        (y, t)
    }

    fn na_cfg() -> ManskiConfig {
        ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        }
    }

    // ── Test 1: all in control ───────────────────────────────────────────────
    /// All t=0: p0=1, p1=0. E[Y(0)]_lo = ȳ₀ (p0·ȳ₀ + 0·y_lo = ȳ₀).
    /// E[Y(1)]_lo = p1·ȳ₁ + p0·y_lo = 0 + 1·y_lo = y_lo.
    #[test]
    fn all_control_no_assumption() {
        let n = 10;
        let y = vec![0.6_f64; n];
        let t = vec![0u8; n];
        let cfg = na_cfg();
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        // p1=0, p0=1: E[Y(0)] = 1·0.6 + 0·y_lo = 0.6 (both bounds equal ȳ₀).
        assert!(approx(r.e_y0_lower, 0.6, 1e-12));
        assert!(approx(r.e_y0_upper, 0.6, 1e-12));
        // E[Y(1)] = 0·ȳ₁ + 1·y_lo = y_lo = 0.0 (lower) and y_hi=1.0 (upper).
        assert!(approx(r.e_y1_lower, 0.0, 1e-12));
        assert!(approx(r.e_y1_upper, 1.0, 1e-12));
    }

    // ── Test 2: balanced NoAssumption width ──────────────────────────────────
    /// Balanced 50/50: ATE interval width = y_upper − y_lower (Manski 1990).
    #[test]
    fn balanced_no_assumption_width() {
        let (y, t) = make_balanced(100);
        let cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        let expected_width = 1.0; // y_upper − y_lower = 1.0 − 0.0
        assert!(
            approx(r.width, expected_width, 1e-10),
            "width = {}, expected {}",
            r.width,
            expected_width
        );
    }

    // ── Test 3: MeanIndependence → point identification ───────────────────────
    #[test]
    fn mean_independence_point_id() {
        let (y, t) = make_balanced(100);
        let cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::MeanIndependence,
        };
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        let expected_ate = 0.5; // ȳ₁ − ȳ₀ = 1.0 − 0.5
        assert!(
            approx(r.ate_upper, expected_ate, 1e-10),
            "ate_upper = {}",
            r.ate_upper
        );
        assert!(
            approx(r.ate_lower, expected_ate, 1e-10),
            "ate_lower = {}",
            r.ate_lower
        );
        assert!(approx(r.width, 0.0, 1e-10));
    }

    // ── Test 4: MTR → ate_lower ≥ 0 ──────────────────────────────────────────
    #[test]
    fn mtr_ate_lower_non_negative() {
        let (y, t) = make_balanced(100);
        let cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::MonotoneTreatmentResponse,
        };
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        assert!(
            r.ate_lower >= 0.0,
            "MTR requires ate_lower >= 0, got {}",
            r.ate_lower
        );
    }

    // ── Test 5: MTS narrower than NoAssumption ────────────────────────────────
    #[test]
    fn mts_narrower_than_no_assumption() {
        let (y, t) = make_balanced(100);
        let na_cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 2.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        let mts_cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 2.0,
            assumption: ManskiAssumption::MonotoneTreatmentSelection,
        };
        let r_na = ManskiBounds::ate_bounds(&y, &t, &na_cfg).expect("ate_bounds should succeed");
        let r_mts = ManskiBounds::ate_bounds(&y, &t, &mts_cfg).expect("ate_bounds should succeed");
        // MTS tightens E[Y(1)] upper.
        assert!(
            r_mts.e_y1_upper <= r_na.e_y1_upper + 1e-10,
            "MTS e_y1_upper={} should be ≤ NA e_y1_upper={}",
            r_mts.e_y1_upper,
            r_na.e_y1_upper
        );
    }

    // ── Test 6: empty y → EmptyInput ─────────────────────────────────────────
    #[test]
    fn empty_y_returns_empty_input() {
        let cfg = na_cfg();
        let r = ManskiBounds::ate_bounds(&[], &[], &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    // ── Test 7: y.len() != t.len() → DimensionMismatch ───────────────────────
    #[test]
    fn mismatched_lengths_returns_dimension_mismatch() {
        let y = vec![1.0_f64, 2.0, 3.0];
        let t = vec![0u8, 1u8];
        let cfg = na_cfg();
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    // ── Test 8: y_lower >= y_upper → InvalidParameter ─────────────────────────
    #[test]
    fn y_lower_ge_y_upper_returns_invalid_parameter() {
        let y = vec![0.5_f64; 10];
        let t = vec![1u8, 0u8, 1u8, 0u8, 1u8, 0u8, 1u8, 0u8, 1u8, 0u8];
        // y_lower == y_upper
        let cfg = ManskiConfig {
            y_lower: 1.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        assert!(matches!(
            ManskiBounds::ate_bounds(&y, &t, &cfg),
            Err(CausalError::InvalidParameter { .. })
        ));
        // y_lower > y_upper
        let cfg2 = ManskiConfig {
            y_lower: 2.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        assert!(matches!(
            ManskiBounds::ate_bounds(&y, &t, &cfg2),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    // ── Test 9: all t=1 ──────────────────────────────────────────────────────
    /// All treated (p1=1): E[Y(1)] bounds degenerate to ȳ₁.
    #[test]
    fn all_treated_no_assumption() {
        let n = 10;
        let y = vec![0.8_f64; n];
        let t = vec![1u8; n];
        let cfg = na_cfg();
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        // p1=1, p0=0: E[Y(1)] = 1·ȳ₁ + 0·y_lo = ȳ₁ for both bounds.
        assert!(approx(r.e_y1_lower, 0.8, 1e-12));
        assert!(approx(r.e_y1_upper, 0.8, 1e-12));
        // E[Y(0)] = 0·ȳ₀ + 1·y_lo = y_lo (lower) / y_hi (upper).
        assert!(approx(r.e_y0_lower, 0.0, 1e-12));
        assert!(approx(r.e_y0_upper, 1.0, 1e-12));
    }

    // ── Test 10: MTS positive monotone DGP → tighter interval ────────────────
    #[test]
    fn mts_positive_monotone_tighter_interval() {
        // Treated units have higher outcome — MTS is empirically plausible.
        let y: Vec<f64> = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let t: Vec<u8> = vec![1, 1, 1, 0, 0, 0];
        let na = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        let mts = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::MonotoneTreatmentSelection,
        };
        let r_na = ManskiBounds::ate_bounds(&y, &t, &na).expect("ate_bounds should succeed");
        let r_mts = ManskiBounds::ate_bounds(&y, &t, &mts).expect("ate_bounds should succeed");
        // MTS should have a narrower (or equal) width.
        assert!(
            r_mts.width <= r_na.width + 1e-10,
            "MTS width={} should be ≤ NA width={}",
            r_mts.width,
            r_na.width
        );
    }

    // ── Test 11: width = ate_upper − ate_lower ────────────────────────────────
    #[test]
    fn width_equals_ate_upper_minus_lower() {
        let (y, t) = make_balanced(100);
        for assumption in [
            ManskiAssumption::NoAssumption,
            ManskiAssumption::MeanIndependence,
            ManskiAssumption::MonotoneTreatmentResponse,
            ManskiAssumption::MonotoneTreatmentSelection,
        ] {
            let cfg = ManskiConfig {
                y_lower: 0.0,
                y_upper: 1.0,
                assumption,
            };
            let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
            assert!(
                approx(r.width, r.ate_upper - r.ate_lower, 1e-10),
                "assumption={assumption:?}: width={} ≠ ate_upper - ate_lower={}",
                r.width,
                r.ate_upper - r.ate_lower
            );
        }
    }

    // ── Test 12: binary outcome within [y_lower, y_upper] ────────────────────
    #[test]
    fn binary_outcome_bounds_within_support() {
        let y: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let t: Vec<u8> = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
        // All E[Y(t)] bounds must lie within [y_lower, y_upper].
        assert!(r.e_y1_lower >= -1e-12);
        assert!(r.e_y1_upper <= 1.0 + 1e-12);
        assert!(r.e_y0_lower >= -1e-12);
        assert!(r.e_y0_upper <= 1.0 + 1e-12);
    }

    // ── Test 13: p1=0 or p1=1 → NA width = y_upper − y_lower ─────────────────
    #[test]
    fn extreme_propensity_na_width_equals_support() {
        let y_lo = 0.0_f64;
        let y_hi = 2.0_f64;
        let n = 10;

        // All control (p1=0).
        let y0 = vec![0.5_f64; n];
        let t0 = vec![0u8; n];
        let cfg = ManskiConfig {
            y_lower: y_lo,
            y_upper: y_hi,
            assumption: ManskiAssumption::NoAssumption,
        };
        let r0 = ManskiBounds::ate_bounds(&y0, &t0, &cfg).expect("ate_bounds should succeed");
        assert!(
            approx(r0.width, y_hi - y_lo, 1e-10),
            "p1=0 width={} expected {}",
            r0.width,
            y_hi - y_lo
        );

        // All treated (p1=1).
        let y1 = vec![1.5_f64; n];
        let t1 = vec![1u8; n];
        let r1 = ManskiBounds::ate_bounds(&y1, &t1, &cfg).expect("ate_bounds should succeed");
        assert!(
            approx(r1.width, y_hi - y_lo, 1e-10),
            "p1=1 width={} expected {}",
            r1.width,
            y_hi - y_lo
        );
    }

    // ── Test 14: e_y1_lower ≤ e_y1_upper for all assumptions ─────────────────
    #[test]
    fn e_y1_bounds_ordered() {
        let (y, t) = make_balanced(100);
        for assumption in [
            ManskiAssumption::NoAssumption,
            ManskiAssumption::MeanIndependence,
            ManskiAssumption::MonotoneTreatmentResponse,
            ManskiAssumption::MonotoneTreatmentSelection,
        ] {
            let cfg = ManskiConfig {
                y_lower: 0.0,
                y_upper: 1.0,
                assumption,
            };
            let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
            assert!(
                r.e_y1_lower <= r.e_y1_upper + 1e-12,
                "{assumption:?}: e_y1_lower={} > e_y1_upper={}",
                r.e_y1_lower,
                r.e_y1_upper
            );
        }
    }

    // ── Test 15: e_y0_lower ≤ e_y0_upper for all assumptions ─────────────────
    #[test]
    fn e_y0_bounds_ordered() {
        let (y, t) = make_balanced(100);
        for assumption in [
            ManskiAssumption::NoAssumption,
            ManskiAssumption::MeanIndependence,
            ManskiAssumption::MonotoneTreatmentResponse,
            ManskiAssumption::MonotoneTreatmentSelection,
        ] {
            let cfg = ManskiConfig {
                y_lower: 0.0,
                y_upper: 1.0,
                assumption,
            };
            let r = ManskiBounds::ate_bounds(&y, &t, &cfg).expect("ate_bounds should succeed");
            assert!(
                r.e_y0_lower <= r.e_y0_upper + 1e-12,
                "{assumption:?}: e_y0_lower={} > e_y0_upper={}",
                r.e_y0_lower,
                r.e_y0_upper
            );
        }
    }

    // ── Test 16: MTR ate_upper ≥ NoAssumption ate_lower ───────────────────────
    #[test]
    fn mtr_upper_ge_na_lower() {
        let (y, t) = make_balanced(100);
        let na_cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::NoAssumption,
        };
        let mtr_cfg = ManskiConfig {
            y_lower: 0.0,
            y_upper: 1.0,
            assumption: ManskiAssumption::MonotoneTreatmentResponse,
        };
        let r_na = ManskiBounds::ate_bounds(&y, &t, &na_cfg).expect("ate_bounds should succeed");
        let r_mtr = ManskiBounds::ate_bounds(&y, &t, &mtr_cfg).expect("ate_bounds should succeed");
        // MTR upper = NA upper (MTR does not change the upper bound).
        assert!(
            approx(r_mtr.ate_upper, r_na.ate_upper, 1e-10),
            "MTR ate_upper={} ≠ NA ate_upper={}",
            r_mtr.ate_upper,
            r_na.ate_upper
        );
        // MTR lower ≥ NA lower (MTR lifts lower to max(na_lower, 0)).
        assert!(
            r_mtr.ate_lower >= r_na.ate_lower - 1e-12,
            "MTR ate_lower={} < NA ate_lower={}",
            r_mtr.ate_lower,
            r_na.ate_lower
        );
    }
}
