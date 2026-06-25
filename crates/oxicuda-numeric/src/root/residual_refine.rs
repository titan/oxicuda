//! Multi-precision residual refinement for ill-conditioned root finding.
//!
//! When a root `x*` of `f(x) = 0` is *ill-conditioned* — for example a multiple or
//! tightly clustered polynomial root, or a function whose evaluation near the root
//! suffers catastrophic cancellation — the residual `f(x)` computed in working
//! (`f64`) precision is contaminated by rounding noise long before `x` reaches `x*`.
//! A plain Newton / Halley iteration then stalls at the *residual floor*: it cannot
//! drive `x` closer than the point where the rounding error in `f(x)` swamps the true
//! value, which for a root of multiplicity `m` is only `O(ε^{1/m})` away from `x*`.
//!
//! The classical cure (used in QUADPACK / GSL "polish" iterations and in numerical
//! libraries that need full machine accuracy on hard problems) is to evaluate the
//! *residual* in **extended precision** while keeping the iterate in `f64`. With a
//! residual that carries roughly twice the working precision, the Newton correction
//! `f(x) / f'(x)` carries enough correct bits to push `x` to the genuine `f64` limit
//! around `x*`, well past the naïve residual floor.
//!
//! This module provides two entry points.
//!
//! * [`refine_root_extended`] — a generic scalar refiner. The caller supplies a
//!   closure that returns the residual *and* its derivative as a [`Double`]
//!   compensated (double-double) pair. This lets a user feed any
//!   higher-precision evaluation of `f` and `f'` they can construct.
//!
//! * [`refine_polynomial_root`] — a turnkey refiner for *polynomial* roots that
//!   evaluates `p(x)` and `p'(x)` with a **compensated Horner scheme**
//!   (Graillat–Langlois–Louvet 2005), which is provably as accurate as evaluating
//!   the polynomial in twice the working precision. No user-supplied high-precision
//!   evaluator is required.
//!
//! The compensated arithmetic is built from error-free transformations
//! ([`two_sum`], [`two_prod`]) and represented by the [`Double`] struct, an
//! unevaluated sum `hi + lo` of two non-overlapping `f64` components (a "double-double"
//! number, Dekker 1971 / Knuth). All of this is pure `f64` arithmetic; no external
//! crates and no `unsafe`.
//!
//! References:
//! - T. J. Dekker, "A floating-point technique for extending the available precision",
//!   *Numer. Math.* 18 (1971), 224–242.
//! - S. Graillat, P. Langlois, N. Louvet, "Compensated Horner scheme",
//!   Research Report, Université de Perpignan (2005).
//! - W. H. Press et al., *Numerical Recipes*, 3rd ed., §9.5 (root polishing).

use core::ops::{Add, Mul, Sub};

use crate::error::{NumericError, NumericResult};

/// A double-double number: an unevaluated sum `hi + lo` of two `f64` components with
/// `|lo| ≤ ½ ulp(hi)`, giving roughly 106 bits of significand.
///
/// Only the operations needed by the residual refiner are implemented (construction
/// from an `f64`, compensated `+`, `-`, `*`, and conversion back to the nearest
/// `f64`). It is deliberately not a full general-purpose extended-precision type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Double {
    /// High-order component (the leading `f64`).
    pub hi: f64,
    /// Low-order correction such that the true value is `hi + lo`.
    pub lo: f64,
}

/// Error-free transformation of a sum: returns `(s, e)` with `s = fl(a + b)` and
/// `a + b = s + e` exactly (Knuth's `TwoSum`, valid for arbitrary `a`, `b`).
#[inline]
#[must_use]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Error-free transformation of a product: returns `(p, e)` with `p = fl(a · b)` and
/// `a · b = p + e` exactly, using a fused multiply-add (`f64::mul_add`) to recover the
/// rounding error (Ogita–Rump–Oishi 2005).
#[inline]
#[must_use]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // `mul_add` computes a*b - p with a single rounding; this is exactly the FMA-based
    // error-free transformation of a product.
    let err = a.mul_add(b, -p);
    (p, err)
}

impl Double {
    /// The additive identity `0`.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// Construct a double-double from an exact `f64` (the low component is zero).
    #[inline]
    #[must_use]
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// Construct from an explicit `(hi, lo)` pair, renormalising so that
    /// `|lo| ≤ ½ ulp(hi)`.
    #[inline]
    #[must_use]
    pub fn new(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    /// Round the double-double to the nearest representable `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

impl Add for Double {
    type Output = Self;

    /// Double-double addition, accurate to the combined precision.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let e2 = e + (self.lo + rhs.lo);
        let (hi, lo) = two_sum(s, e2);
        Self { hi, lo }
    }
}

impl Sub for Double {
    type Output = Self;

    /// Double-double subtraction.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + Self {
            hi: -rhs.hi,
            lo: -rhs.lo,
        }
    }
}

impl Mul for Double {
    type Output = Self;

    /// Double-double multiplication.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.hi, rhs.hi);
        let e2 = e + (self.hi * rhs.lo + self.lo * rhs.hi);
        let (hi, lo) = two_sum(p, e2);
        Self { hi, lo }
    }
}

impl Add<f64> for Double {
    type Output = Self;

    /// Add an exact `f64` to a double-double.
    #[inline]
    fn add(self, b: f64) -> Self {
        let (s, e) = two_sum(self.hi, b);
        let (hi, lo) = two_sum(s, e + self.lo);
        Self { hi, lo }
    }
}

impl Mul<f64> for Double {
    type Output = Self;

    /// Multiply a double-double by an exact `f64`.
    #[inline]
    fn mul(self, b: f64) -> Self {
        let (p, e) = two_prod(self.hi, b);
        let e2 = e + self.lo * b;
        let (hi, lo) = two_sum(p, e2);
        Self { hi, lo }
    }
}

/// Outcome of a residual-refinement run.
#[derive(Debug, Clone, Copy)]
pub struct RefineResult {
    /// The refined root.
    pub root: f64,
    /// `|f(root)|` measured in the high-precision residual.
    pub residual: f64,
    /// Number of refinement iterations actually performed.
    pub iterations: usize,
}

/// Configuration for [`refine_root_extended`] and [`refine_polynomial_root`].
#[derive(Debug, Clone, Copy)]
pub struct RefineConfig {
    /// Stop once the high-precision residual `|f(x)|` is below this absolute value.
    pub residual_tol: f64,
    /// Stop once a Newton correction is below `step_tol · max(|x|, 1)`.
    pub step_tol: f64,
    /// Maximum refinement iterations.
    pub max_iter: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            // Push to the genuine f64 floor: residuals near `u·|coeffs|` are expected.
            residual_tol: 0.0,
            step_tol: f64::EPSILON,
            max_iter: 50,
        }
    }
}

impl RefineConfig {
    /// Create and validate a configuration.
    ///
    /// # Errors
    /// Returns [`NumericError::InvalidConfiguration`] when `max_iter == 0`, when
    /// `residual_tol` is negative or non-finite, or when `step_tol` is non-positive or
    /// non-finite.
    pub fn new(residual_tol: f64, step_tol: f64, max_iter: usize) -> NumericResult<Self> {
        if max_iter == 0 {
            return Err(NumericError::InvalidConfiguration(
                "refine: max_iter must be ≥ 1".to_string(),
            ));
        }
        if !residual_tol.is_finite() || residual_tol < 0.0 {
            return Err(NumericError::InvalidConfiguration(format!(
                "refine: residual_tol must be finite and ≥ 0, got {residual_tol}"
            )));
        }
        if !step_tol.is_finite() || step_tol <= 0.0 {
            return Err(NumericError::InvalidConfiguration(format!(
                "refine: step_tol must be finite and > 0, got {step_tol}"
            )));
        }
        Ok(Self {
            residual_tol,
            step_tol,
            max_iter,
        })
    }
}

/// Refine a scalar root with a caller-supplied extended-precision residual.
///
/// Starting from an approximate root `x0` (typically the output of a standard
/// `f64` Newton / Brent / Aberth solve), this performs Newton iterations
/// `x ← x − f(x)/f'(x)` in which **both** `f(x)` and `f'(x)` are evaluated by the
/// user-supplied closure `eval` *in double-double precision*. Because the residual
/// is accurate to roughly twice the working precision, the iteration can advance
/// `x` past the point at which a naïve `f64` residual would have stalled, returning
/// the best `f64` approximation to the true root.
///
/// The closure must return `(f(x), f'(x))` as a pair of [`Double`] values for a given
/// `f64` argument `x`.
///
/// # Errors
/// Returns [`NumericError::InvalidParameter`] when `x0` is not finite, and
/// [`NumericError::NumericalInstability`] when the derivative underflows to zero or an
/// iterate becomes non-finite. The iteration does not error on failure to reach
/// `residual_tol`; it returns the best iterate found, since the residual floor itself
/// may legitimately sit above any positive tolerance for very ill-conditioned roots.
pub fn refine_root_extended<E>(
    eval: E,
    x0: f64,
    config: RefineConfig,
) -> NumericResult<RefineResult>
where
    E: Fn(f64) -> NumericResult<(Double, Double)>,
{
    if !x0.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "refine_root_extended: x0 must be finite, got {x0}"
        )));
    }

    let mut x = x0;
    let (mut fx, mut dfx) = eval(x)?;
    let mut best_x = x;
    let mut best_res = fx.to_f64().abs();
    let mut iterations = 0usize;

    for _ in 0..config.max_iter {
        iterations += 1;
        let res = fx.to_f64().abs();
        if res < best_res {
            best_res = res;
            best_x = x;
        }
        if res <= config.residual_tol {
            return Ok(RefineResult {
                root: x,
                residual: res,
                iterations,
            });
        }

        let dfx_val = dfx.to_f64();
        if dfx_val.abs() < 1.0e-300 {
            // Derivative underflow: the residual surface is flat (multiple root).
            // Return the best iterate rather than dividing by ~0.
            return Ok(RefineResult {
                root: best_x,
                residual: best_res,
                iterations,
            });
        }

        // Newton correction with the high-precision residual: dx = f(x) / f'(x).
        // The quotient itself only needs working precision; the *accuracy* comes from
        // f(x) and f'(x) being correct to ~2u.
        let dx = fx.to_f64() / dfx_val;
        let x_new = x - dx;

        if !x_new.is_finite() {
            return Err(NumericError::NumericalInstability(format!(
                "refine_root_extended: iterate became non-finite (x={x}, dx={dx})"
            )));
        }

        let step_floor = config.step_tol * x.abs().max(1.0);
        x = x_new;
        let (f_new, df_new) = eval(x)?;
        fx = f_new;
        dfx = df_new;

        // Converged when the step is at the f64 resolution of x around the root.
        if dx.abs() <= step_floor {
            let final_res = fx.to_f64().abs();
            let (root, residual) = if final_res <= best_res {
                (x, final_res)
            } else {
                (best_x, best_res)
            };
            return Ok(RefineResult {
                root,
                residual,
                iterations,
            });
        }
    }

    // Out of iterations: hand back the best iterate (lowest high-precision residual).
    let final_res = fx.to_f64().abs();
    let (root, residual) = if final_res <= best_res {
        (x, final_res)
    } else {
        (best_x, best_res)
    };
    Ok(RefineResult {
        root,
        residual,
        iterations,
    })
}

/// Compensated Horner evaluation of a polynomial and its derivative.
///
/// Given coefficients in *ascending* order (`coeffs[i]` is the coefficient of `xⁱ`),
/// returns `(p(x), p'(x))` each as a [`Double`]. The polynomial value is computed with
/// the compensated Horner scheme of Graillat–Langlois–Louvet (2005): a standard Horner
/// sweep accumulates, at every step, the rounding errors of the multiply and the add via
/// error-free transformations, and the accumulated correction is added back at the end.
/// The result is provably as accurate as Horner evaluated in twice the working
/// precision. The derivative is evaluated by the analogous compensated sweep on the
/// derivative recurrence.
#[must_use]
pub fn compensated_horner(coeffs: &[f64], x: f64) -> (Double, Double) {
    let n = coeffs.len();
    if n == 0 {
        return (Double::ZERO, Double::ZERO);
    }
    if n == 1 {
        return (Double::from_f64(coeffs[0]), Double::ZERO);
    }

    // Horner runs from the highest-degree coefficient downward.
    // s holds p(x); the compensation `corr` accumulates all rounding errors so that
    // the returned value is (s + corr) ≈ p(x) to ~2u.
    let mut s = coeffs[n - 1];
    let mut corr = 0.0_f64; // running compensation for p(x)
    // Derivative: d/dx of Horner. ds holds p'(x), dcorr its compensation.
    let mut ds = 0.0_f64;
    let mut dcorr = 0.0_f64;

    for &c in coeffs[..n - 1].iter().rev() {
        // Derivative recurrence (compensated): ds = ds·x + s, BEFORE updating s.
        let (dp, dpe) = two_prod(ds, x);
        let (dsum, dse) = two_sum(dp, s);
        // The derivative's own compensation also needs the previous p-compensation
        // `corr`, since p'(x) = d/dx of the compensated p(x): error terms dpe, dse and
        // the propagated dcorr·x and corr feed the derivative correction.
        dcorr = dcorr.mul_add(x, dpe + dse + corr);
        ds = dsum;

        // Value recurrence (compensated): s = s·x + c.
        let (p, pe) = two_prod(s, x);
        let (sum, se) = two_sum(p, c);
        corr = corr.mul_add(x, pe + se);
        s = sum;
    }

    let value = Double::new(s, corr);
    let deriv = Double::new(ds, dcorr);
    (value, deriv)
}

/// Refine a polynomial root to full `f64` accuracy with a compensated residual.
///
/// `coeffs` are in *ascending* order (`coeffs[i]` multiplies `xⁱ`). Starting from an
/// approximate root `x0`, Newton iterations are performed in which `p(x)` and `p'(x)`
/// are evaluated by [`compensated_horner`], i.e. with the accuracy of double the
/// working precision. This polishes roots that a plain `f64` solver leaves stranded at
/// the residual floor — most dramatically for multiple or clustered roots and for
/// high-degree polynomials with large coefficients.
///
/// # Errors
/// Returns [`NumericError::EmptyInput`] when `coeffs` is empty, and propagates the
/// errors of [`refine_root_extended`] (non-finite `x0` or iterate).
pub fn refine_polynomial_root(
    coeffs: &[f64],
    x0: f64,
    config: RefineConfig,
) -> NumericResult<RefineResult> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    refine_root_extended(|x| Ok(compensated_horner(coeffs, x)), x0, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Naïve `f64` Horner — the baseline whose residual floor we want to beat.
    fn naive_horner(coeffs: &[f64], x: f64) -> (f64, f64) {
        let n = coeffs.len();
        let mut p = coeffs[n - 1];
        let mut dp = 0.0;
        for &c in coeffs[..n - 1].iter().rev() {
            dp = dp * x + p;
            p = p * x + c;
        }
        (p, dp)
    }

    #[test]
    fn two_sum_is_error_free() {
        // 1 and a tiny number whose sum loses the small part in f64.
        let a = 1.0_f64;
        let b = 1.0e-20_f64;
        let (s, e) = two_sum(a, b);
        // The exact sum is recovered as s + e.
        assert_eq!(s, 1.0);
        assert_eq!(e, 1.0e-20);
    }

    #[test]
    fn two_prod_is_error_free() {
        let a = 1.0 + 2.0_f64.powi(-20);
        let b = 1.0 - 2.0_f64.powi(-20);
        let (p, e) = two_prod(a, b);
        // a·b = 1 - 2^-40 exactly; p rounds to 1.0, e carries -2^-40.
        let reconstructed = Double::new(p, e).to_f64();
        assert!((reconstructed - (1.0 - 2.0_f64.powi(-40))).abs() < 1.0e-30);
    }

    #[test]
    fn double_double_arithmetic_roundtrips() {
        let a = Double::from_f64(1.0 / 3.0);
        let b = Double::from_f64(3.0);
        // (1/3) · 3 should be extremely close to 1 in double-double.
        let prod = a * b;
        assert!((prod.to_f64() - 1.0).abs() < 1.0e-16);
        let sum = a + a + a;
        assert!((sum.to_f64() - 1.0).abs() < 1.0e-16);
    }

    #[test]
    fn compensated_horner_matches_exact_simple() {
        // p(x) = x^3 - 2x + 1, ascending coeffs [1, -2, 0, 1].
        let coeffs = [1.0, -2.0, 0.0, 1.0];
        let x = 1.5;
        let (val, der) = compensated_horner(&coeffs, x);
        let exact_val = x.powi(3) - 2.0 * x + 1.0;
        let exact_der = 3.0 * x * x - 2.0;
        assert!((val.to_f64() - exact_val).abs() < 1.0e-13, "val={val:?}");
        assert!((der.to_f64() - exact_der).abs() < 1.0e-13, "der={der:?}");
    }

    #[test]
    fn refine_simple_root_reaches_machine_precision() {
        // p(x) = x^2 - 2, root √2. Start slightly off.
        let coeffs = [-2.0, 0.0, 1.0];
        let cfg = RefineConfig::default();
        let res = refine_polynomial_root(&coeffs, 1.4, cfg).expect("ok");
        let exact = 2.0_f64.sqrt();
        assert!(
            (res.root - exact).abs() < 4.0 * f64::EPSILON,
            "root={}, err={:e}",
            res.root,
            (res.root - exact).abs()
        );
    }

    #[test]
    fn refine_beats_naive_residual_on_ill_conditioned_double_root() {
        // p(x) = (x - 1)^2 = x^2 - 2x + 1: a double root at x = 1.
        // Near a double root the f64 residual floor is ~√ε ≈ 1.5e-8 away from the root,
        // and naive Newton stalls there. The compensated residual should let us refine
        // much closer.
        let coeffs = [1.0, -2.0, 1.0];
        let x0 = 1.0 + 1.0e-3;

        // Baseline: naive f64 Newton from the same start.
        let mut x_naive = x0;
        for _ in 0..200 {
            let (p, dp) = naive_horner(&coeffs, x_naive);
            if dp.abs() < 1.0e-300 {
                break;
            }
            let step = p / dp;
            x_naive -= step;
            if step.abs() <= f64::EPSILON * x_naive.abs().max(1.0) {
                break;
            }
        }
        let naive_err = (x_naive - 1.0).abs();

        let cfg = RefineConfig::new(0.0, f64::EPSILON, 200).expect("cfg");
        let res = refine_polynomial_root(&coeffs, x0, cfg).expect("ok");
        let refined_err = (res.root - 1.0).abs();

        // The compensated refinement must get at least as close as naive, and in
        // practice an order of magnitude or more closer to the true double root.
        assert!(
            refined_err <= naive_err,
            "refined {refined_err:e} should beat naive {naive_err:e}"
        );
        assert!(
            refined_err < 1.0e-7,
            "double root not polished: err={refined_err:e}"
        );
    }

    #[test]
    fn refine_high_degree_random_root_stays_accurate() {
        // Build p(x) = ∏(x - r_k) for known roots, evaluate ascending coeffs, then
        // refine one perturbed root and confirm it returns to the true value.
        let roots = [0.5_f64, 1.3, -0.7, 2.1, -1.8];
        // Expand the product into ascending-order coefficients.
        let mut coeffs = vec![1.0_f64]; // start with polynomial "1"
        for &r in &roots {
            // multiply current poly by (x - r): new[i] = old[i-1] - r·old[i]
            let mut next = vec![0.0_f64; coeffs.len() + 1];
            for (i, &c) in coeffs.iter().enumerate() {
                next[i] += -r * c; // constant-shift term
                next[i + 1] += c; // x · term
            }
            coeffs = next;
        }

        let mut rng = LcgRng::new(20_260_621);
        for &true_root in &roots {
            // Perturb the root by a random ±1e-4 and refine.
            let perturb = (rng.next_f64() - 0.5) * 2.0e-4;
            let x0 = true_root + perturb;
            let cfg = RefineConfig::default();
            let res = refine_polynomial_root(&coeffs, x0, cfg).expect("ok");
            assert!(
                (res.root - true_root).abs() < 1.0e-10,
                "root {true_root}: refined={}, err={:e}",
                res.root,
                (res.root - true_root).abs()
            );
        }
    }

    #[test]
    fn refine_extended_with_custom_evaluator() {
        // f(x) = exp(x) - 2, root ln 2. Supply a double-double residual: exp(x) is hard
        // to do in extended precision cheaply, so use Double arithmetic around f64 exp.
        // Here the point is the generic API; f and f' coincide for exp.
        let ln2 = std::f64::consts::LN_2;
        let eval = |x: f64| -> NumericResult<(Double, Double)> {
            let ex = x.exp();
            let f = Double::from_f64(ex) + (-2.0);
            let df = Double::from_f64(ex);
            Ok((f, df))
        };
        let cfg = RefineConfig::new(1.0e-15, f64::EPSILON, 50).expect("cfg");
        let res = refine_root_extended(eval, 0.6, cfg).expect("ok");
        assert!((res.root - ln2).abs() < 1.0e-12, "root={}", res.root);
    }

    #[test]
    fn rejects_non_finite_start() {
        let coeffs = [-2.0, 0.0, 1.0];
        let cfg = RefineConfig::default();
        assert!(refine_polynomial_root(&coeffs, f64::NAN, cfg).is_err());
        assert!(refine_polynomial_root(&coeffs, f64::INFINITY, cfg).is_err());
    }

    #[test]
    fn rejects_empty_coeffs() {
        let cfg = RefineConfig::default();
        assert!(matches!(
            refine_polynomial_root(&[], 1.0, cfg),
            Err(NumericError::EmptyInput)
        ));
    }

    #[test]
    fn config_validation() {
        assert!(RefineConfig::new(1.0e-15, 1.0e-15, 0).is_err());
        assert!(RefineConfig::new(-1.0, 1.0e-15, 10).is_err());
        assert!(RefineConfig::new(1.0e-15, 0.0, 10).is_err());
        assert!(RefineConfig::new(1.0e-15, f64::NAN, 10).is_err());
        assert!(RefineConfig::new(0.0, f64::EPSILON, 10).is_ok());
    }
}
