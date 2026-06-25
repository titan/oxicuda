//! Complex-valued scalar root-finding for analytic functions `f: ℂ → ℂ`.
//!
//! This module is the *complex* analogue of the real-valued [`mod@crate::root::halley`]
//! and [`mod@crate::root::newton`] modules. It operates over
//! [`num_complex::Complex<f64>`] and provides:
//!
//! * [`complex_newton`] — Newton's method `z ← z − f(z)/f'(z)` (quadratic
//!   convergence near a simple root).
//! * [`complex_halley`] — Halley's method
//!   `z ← z − 2·f·f' / (2·f'² − f·f'')` (cubic convergence near a simple root).
//! * [`complex_poly_roots`] — a deflation-based all-roots finder for complex
//!   polynomials. It finds one root at a time by complex Halley iteration from a
//!   perturbed start, synthetically deflates `p(z)/(z − root)`, and repeats until
//!   all `n` roots are recovered. An optional final polishing pass refines every
//!   root against the *original* polynomial. This is the natural companion of the
//!   simultaneous Aberth–Ehrlich refinement in [`mod@crate::root::aberth_all_roots`].
//!
//! All iterations guard against a vanishing derivative / Halley denominator and
//! report failure through [`NumericError`] rather than panicking.

use crate::error::{NumericError, NumericResult};
use num_complex::Complex;

/// A 64-bit complex number (`re + im·i`).
pub type Cplx = Complex<f64>;

/// Absolute floor below which a complex magnitude is treated as zero.
///
/// Chosen well above the `f64` denormal range so that the *square* used in
/// Halley's denominator (`2·f'²`) cannot silently underflow to zero.
const ZERO_GUARD: f64 = 1.0e-290;

/// Configuration for the scalar complex root-finders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexRootConfig {
    /// Convergence tolerance applied to both the step size `|Δz|` and the
    /// residual `|f(z)|`.
    pub tol: f64,
    /// Maximum number of iterations before giving up.
    pub max_iter: usize,
}

impl Default for ComplexRootConfig {
    fn default() -> Self {
        Self {
            tol: 1.0e-12,
            max_iter: 100,
        }
    }
}

impl ComplexRootConfig {
    /// Construct a configuration, validating the tolerance and iteration cap.
    pub fn new(tol: f64, max_iter: usize) -> NumericResult<Self> {
        if !tol.is_finite() || tol <= 0.0 {
            return Err(NumericError::InvalidParameter(format!(
                "tol must be a positive finite number, got {tol}"
            )));
        }
        if max_iter == 0 {
            return Err(NumericError::InvalidParameter(
                "max_iter must be at least 1".into(),
            ));
        }
        Ok(Self { tol, max_iter })
    }
}

/// Outcome of a scalar complex root-finding run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexRoot {
    /// The estimated root `z`.
    pub root: Cplx,
    /// Number of iterations actually performed.
    pub iterations: usize,
    /// Final residual magnitude `|f(z)|`.
    pub residual: f64,
    /// `true` if the run met the convergence tolerance.
    pub converged: bool,
}

/// Validate a complex starting point.
fn check_start(z0: Cplx) -> NumericResult<()> {
    if !z0.re.is_finite() || !z0.im.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite starting point z0=({}, {})",
            z0.re, z0.im
        )));
    }
    Ok(())
}

/// `true` when the complex iterate has become non-finite (overflow / NaN).
fn is_diverged(z: Cplx) -> bool {
    !z.re.is_finite() || !z.im.is_finite()
}

/// Complex Newton iteration: `z ← z − f(z)/f'(z)`.
///
/// Converges quadratically toward a simple root of an analytic `f`. The
/// derivative is supplied by the caller. The iteration stops as soon as either
/// the residual `|f(z)| < tol` or the step `|Δz| < tol`.
///
/// # Errors
/// Returns [`NumericError::NumericalInstability`] if the derivative magnitude
/// falls below the zero guard (the Newton step is undefined) or the iterate
/// diverges, and [`NumericError::NotConverged`] if `max_iter` is exhausted.
pub fn complex_newton<F, DF>(
    f: F,
    df: DF,
    z0: Cplx,
    cfg: ComplexRootConfig,
) -> NumericResult<ComplexRoot>
where
    F: Fn(Cplx) -> NumericResult<Cplx>,
    DF: Fn(Cplx) -> NumericResult<Cplx>,
{
    let cfg = ComplexRootConfig::new(cfg.tol, cfg.max_iter)?;
    check_start(z0)?;

    let mut z = z0;
    let mut last_residual = f(z)?.norm();

    for k in 0..cfg.max_iter {
        let fz = f(z)?;
        let residual = fz.norm();
        last_residual = residual;
        if residual < cfg.tol {
            return Ok(ComplexRoot {
                root: z,
                iterations: k,
                residual,
                converged: true,
            });
        }

        let dfz = df(z)?;
        if dfz.norm() < ZERO_GUARD {
            return Err(NumericError::NumericalInstability(format!(
                "Newton derivative vanished at z=({}, {}) iter={k}",
                z.re, z.im
            )));
        }

        let step = fz / dfz;
        z -= step;
        if is_diverged(z) {
            return Err(NumericError::NumericalInstability(format!(
                "Newton iterate diverged at iter={k}"
            )));
        }
        if step.norm() < cfg.tol {
            let residual = f(z)?.norm();
            if residual_acceptable(residual, cfg.tol) {
                return Ok(ComplexRoot {
                    root: z,
                    iterations: k + 1,
                    residual,
                    converged: true,
                });
            }
            // A vanishing step that fails to drive `f` to zero means we are stuck
            // at a stationary point: report a stall instead of a false success.
            return Err(NumericError::NumericalInstability(format!(
                "Newton stalled at z=({}, {}) with residual {residual} iter={k}",
                z.re, z.im
            )));
        }
    }

    Err(NumericError::NotConverged {
        iter: cfg.max_iter,
        residual: last_residual,
    })
}

/// Decide whether a residual reached via a vanishing step counts as convergence.
///
/// A small step can occur either because `f ≈ 0` (true convergence) or because
/// the numerator of the correction collapsed at a stationary point while `f`
/// stays large. We accept only when the residual is itself small — at or below a
/// relaxed bound derived from `tol` (its square root, capped, so the very tight
/// `tol` values typical here still admit a sensible residual window).
fn residual_acceptable(residual: f64, tol: f64) -> bool {
    let relaxed = tol.sqrt().min(1.0e-6).max(tol);
    residual <= relaxed
}

/// Complex Halley iteration: `z ← z − 2·f·f' / (2·f'² − f·f'')`.
///
/// Converges cubically toward a simple root of an analytic `f`, using the first
/// and second derivatives supplied by the caller. The iteration stops as soon as
/// either the residual `|f(z)| < tol` or the step `|Δz| < tol`.
///
/// # Errors
/// Returns [`NumericError::NumericalInstability`] if the Halley denominator
/// magnitude falls below the zero guard or the iterate diverges, and
/// [`NumericError::NotConverged`] if `max_iter` is exhausted.
pub fn complex_halley<F, DF, D2F>(
    f: F,
    df: DF,
    d2f: D2F,
    z0: Cplx,
    cfg: ComplexRootConfig,
) -> NumericResult<ComplexRoot>
where
    F: Fn(Cplx) -> NumericResult<Cplx>,
    DF: Fn(Cplx) -> NumericResult<Cplx>,
    D2F: Fn(Cplx) -> NumericResult<Cplx>,
{
    let cfg = ComplexRootConfig::new(cfg.tol, cfg.max_iter)?;
    check_start(z0)?;

    let mut z = z0;
    let mut last_residual = f(z)?.norm();

    for k in 0..cfg.max_iter {
        let fz = f(z)?;
        let residual = fz.norm();
        last_residual = residual;
        if residual < cfg.tol {
            return Ok(ComplexRoot {
                root: z,
                iterations: k,
                residual,
                converged: true,
            });
        }

        let dfz = df(z)?;
        let d2fz = d2f(z)?;
        // denom = 2·f'² − f·f''
        let denom = (dfz * dfz) * 2.0 - fz * d2fz;
        if denom.norm() < ZERO_GUARD {
            return Err(NumericError::NumericalInstability(format!(
                "Halley denominator vanished at z=({}, {}) iter={k}",
                z.re, z.im
            )));
        }

        // step = 2·f·f' / denom
        let step = (fz * dfz) * 2.0 / denom;
        z -= step;
        if is_diverged(z) {
            return Err(NumericError::NumericalInstability(format!(
                "Halley iterate diverged at iter={k}"
            )));
        }
        if step.norm() < cfg.tol {
            let residual = f(z)?.norm();
            if residual_acceptable(residual, cfg.tol) {
                return Ok(ComplexRoot {
                    root: z,
                    iterations: k + 1,
                    residual,
                    converged: true,
                });
            }
            // Vanishing step but `f` still large ⇒ stuck at a stationary point
            // (e.g. f·f' = 0 with f ≠ 0): report a stall, never a false success.
            return Err(NumericError::NumericalInstability(format!(
                "Halley stalled at z=({}, {}) with residual {residual} iter={k}",
                z.re, z.im
            )));
        }
    }

    Err(NumericError::NotConverged {
        iter: cfg.max_iter,
        residual: last_residual,
    })
}

/// Evaluate `p(z)`, `p'(z)`, `p''(z)` for a polynomial given in ascending-power
/// coefficient order (`coeffs[i]` multiplies `z^i`) using a single Horner sweep.
fn poly_eval_012(coeffs: &[Cplx], z: Cplx) -> (Cplx, Cplx, Cplx) {
    // Horner recurrence carrying value, first and second derivative.
    // p   = ((a_n z + a_{n-1}) z + … ) z + a_0
    // p'  = derivative accumulated alongside
    // p'' = second derivative accumulated alongside
    let zero = Complex::new(0.0, 0.0);
    let n = coeffs.len();
    if n == 0 {
        return (zero, zero, zero);
    }
    let mut p = coeffs[n - 1];
    let mut dp = zero;
    let mut ddp = zero;
    for i in (0..(n - 1)).rev() {
        // Order matters: update second derivative from current first derivative,
        // then first derivative from current value, then value from coeff.
        ddp = ddp * z + dp * 2.0;
        dp = dp * z + p;
        p = p * z + coeffs[i];
    }
    (p, dp, ddp)
}

/// Evaluate just `p(z)` (ascending-power coefficients) via Horner.
fn poly_eval(coeffs: &[Cplx], z: Cplx) -> Cplx {
    let mut acc = Complex::new(0.0, 0.0);
    for c in coeffs.iter().rev() {
        acc = acc * z + *c;
    }
    acc
}

/// Synthetic division of `p(z)` (ascending-power) by the monic linear factor
/// `(z − root)`. Returns the `degree − 1` quotient coefficients (ascending).
///
/// For `p(z) = Σ a_i z^i`, the quotient `q(z) = Σ b_i z^i` of `p/(z − r)` is
/// obtained by the recurrence `b_{i-1} = a_i + r·b_i` running from the top down,
/// discarding the remainder (which is `p(r) ≈ 0`).
fn deflate_linear(coeffs: &[Cplx], root: Cplx) -> Vec<Cplx> {
    let n = coeffs.len();
    if n <= 1 {
        return Vec::new();
    }
    let mut quotient = vec![Complex::new(0.0, 0.0); n - 1];
    // b_{n-2} = a_{n-1}; then b_{i-1} = a_i + r·b_i.
    let mut carry = coeffs[n - 1];
    quotient[n - 2] = carry;
    for i in (1..(n - 1)).rev() {
        carry = coeffs[i] + root * carry;
        quotient[i - 1] = carry;
    }
    quotient
}

/// Trim trailing (high-order) coefficients whose magnitude is negligible so the
/// effective degree reflects the genuinely non-zero leading term.
fn effective_len(coeffs: &[Cplx]) -> usize {
    let mut len = coeffs.len();
    while len > 1 && coeffs[len - 1].norm() < ZERO_GUARD {
        len -= 1;
    }
    len
}

/// Find a single root of a complex polynomial by Halley iteration with a Newton
/// fallback, starting from `z0`. Operates directly on the (ascending-power)
/// coefficients so the analytic derivatives are exact.
fn poly_one_root(coeffs: &[Cplx], z0: Cplx, cfg: ComplexRootConfig) -> NumericResult<ComplexRoot> {
    check_start(z0)?;
    let mut z = z0;
    let mut last_residual = poly_eval(coeffs, z).norm();

    for k in 0..cfg.max_iter {
        let (p, dp, ddp) = poly_eval_012(coeffs, z);
        let residual = p.norm();
        last_residual = residual;
        if residual < cfg.tol {
            return Ok(ComplexRoot {
                root: z,
                iterations: k,
                residual,
                converged: true,
            });
        }

        // Prefer the cubic Halley step; fall back to the Newton step when the
        // Halley denominator is too small but the derivative is still usable.
        let denom = (dp * dp) * 2.0 - p * ddp;
        let step = if denom.norm() >= ZERO_GUARD {
            (p * dp) * 2.0 / denom
        } else if dp.norm() >= ZERO_GUARD {
            p / dp
        } else {
            return Err(NumericError::NumericalInstability(format!(
                "polynomial derivative vanished at z=({}, {}) iter={k}",
                z.re, z.im
            )));
        };

        z -= step;
        if is_diverged(z) {
            return Err(NumericError::NumericalInstability(format!(
                "polynomial iterate diverged at iter={k}"
            )));
        }
        if step.norm() < cfg.tol {
            let residual = poly_eval(coeffs, z).norm();
            if residual_acceptable(residual, cfg.tol) {
                return Ok(ComplexRoot {
                    root: z,
                    iterations: k + 1,
                    residual,
                    converged: true,
                });
            }
            // Stalled at a polynomial stationary point that is not a root; let the
            // caller (restart driver) try a different start.
            return Err(NumericError::NumericalInstability(format!(
                "polynomial root iteration stalled at z=({}, {}) residual {residual} iter={k}",
                z.re, z.im
            )));
        }
    }

    Err(NumericError::NotConverged {
        iter: cfg.max_iter,
        residual: last_residual,
    })
}

/// A handful of perturbed starting points used to escape stationary points and
/// real-axis symmetry traps when seeking the next deflated root.
fn start_candidates(seed: Cplx) -> [Cplx; 6] {
    [
        seed,
        seed + Complex::new(0.5, 0.5),
        seed + Complex::new(-0.5, 0.7),
        Complex::new(0.4, 0.9),
        Complex::new(-0.8, 0.3),
        Complex::new(0.1, -1.1),
    ]
}

/// Find **all** complex roots of a polynomial via repeated Halley/Newton plus
/// synthetic deflation.
///
/// The polynomial is supplied in **ascending-power** coefficient order, i.e.
/// `coeffs == [a_0, a_1, …, a_n]` represents `p(z) = a_0 + a_1 z + … + a_n z^n`.
/// The function returns the `n` roots (with multiplicity). After deflation each
/// root is optionally polished with a few Halley iterations against the
/// *original* polynomial, which removes the error accumulated by repeated
/// deflation and keeps the residual `|p(root)|` tiny.
///
/// # Errors
/// * [`NumericError::EmptyInput`] when `coeffs` is empty.
/// * [`NumericError::InvalidParameter`] when the leading coefficient is
///   (effectively) zero, so the degree is undefined.
/// * [`NumericError::NotConverged`] if a deflated root cannot be located from any
///   of the perturbed starts within `max_iter`.
///
/// A constant polynomial (degree `0`) has no roots and yields an empty vector.
pub fn complex_poly_roots(coeffs: &[Cplx], cfg: ComplexRootConfig) -> NumericResult<Vec<Cplx>> {
    let cfg = ComplexRootConfig::new(cfg.tol, cfg.max_iter)?;
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let len = effective_len(coeffs);
    if coeffs[len - 1].norm() < ZERO_GUARD {
        return Err(NumericError::InvalidParameter(
            "leading coefficient is zero; polynomial degree is undefined".into(),
        ));
    }
    let original: Vec<Cplx> = coeffs[..len].to_vec();
    let degree = len - 1;
    if degree == 0 {
        return Ok(Vec::new());
    }

    let mut working = original.clone();
    let mut roots: Vec<Cplx> = Vec::with_capacity(degree);

    // Peel off one root at a time until only a linear (or constant after the last
    // division) polynomial remains.
    while working.len() > 2 {
        let seed = next_seed(&roots);
        let found = find_with_restarts(&working, seed, cfg)?;
        // Polish against the ORIGINAL polynomial before deflating, so deflation
        // uses the most accurate root available.
        let polished = polish_root(&original, found.root, cfg);
        roots.push(polished);
        working = deflate_linear(&working, polished);
    }

    // Solve the final linear factor a_0 + a_1 z = 0 directly.
    if working.len() == 2 {
        let a1 = working[1];
        let a0 = working[0];
        if a1.norm() < ZERO_GUARD {
            return Err(NumericError::NumericalInstability(
                "deflation produced a degenerate linear factor".into(),
            ));
        }
        let last = -a0 / a1;
        roots.push(polish_root(&original, last, cfg));
    }

    // Final polishing sweep of every root against the original polynomial.
    for r in roots.iter_mut() {
        *r = polish_root(&original, *r, cfg);
    }

    Ok(roots)
}

/// Choose a starting seed for the next root, biased away from the roots already
/// found so we are unlikely to immediately rediscover one of them.
fn next_seed(found: &[Cplx]) -> Cplx {
    // A point off the real axis with a phase that advances per root keeps the
    // sequence of seeds well spread over the complex plane.
    let k = found.len() as f64;
    let theta = std::f64::consts::TAU * (k * 0.371) + 0.4;
    Complex::from_polar(0.9 + 0.15 * k, theta)
}

/// Try [`poly_one_root`] from several perturbed starts, returning the first
/// success. Fails with [`NumericError::NotConverged`] only if every restart fails.
fn find_with_restarts(
    coeffs: &[Cplx],
    seed: Cplx,
    cfg: ComplexRootConfig,
) -> NumericResult<ComplexRoot> {
    let mut last_err = NumericError::NotConverged {
        iter: cfg.max_iter,
        residual: f64::INFINITY,
    };
    for start in start_candidates(seed) {
        match poly_one_root(coeffs, start, cfg) {
            Ok(root) if root.converged => return Ok(root),
            Ok(root) => {
                last_err = NumericError::NotConverged {
                    iter: root.iterations,
                    residual: root.residual,
                };
            }
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// Run a bounded Halley/Newton polish of a single root against `coeffs`,
/// returning the improved estimate (or the input unchanged if it cannot be
/// improved, e.g. at a degenerate point).
fn polish_root(coeffs: &[Cplx], root: Cplx, cfg: ComplexRootConfig) -> Cplx {
    let polish_cfg = ComplexRootConfig {
        tol: cfg.tol,
        // A short, fixed number of refinement steps is plenty given cubic
        // convergence and a good starting estimate.
        max_iter: 16,
    };
    match poly_one_root(coeffs, root, polish_cfg) {
        Ok(r) => r.root,
        Err(_) => root,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default tolerance/iteration cap for the tests.
    fn cfg() -> ComplexRootConfig {
        ComplexRootConfig::new(1.0e-13, 100).expect("valid config")
    }

    /// Smallest distance from `z` to any element of `set`.
    fn min_dist(z: Cplx, set: &[Cplx]) -> f64 {
        set.iter()
            .map(|s| (z - *s).norm())
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn newton_z2_plus_one_reaches_both_roots() {
        // f(z) = z^2 + 1, f'(z) = 2z. Roots ±i.
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z + Complex::new(1.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z * 2.0) };

        // Start in the upper half-plane → +i.
        let up = complex_newton(f, df, Complex::new(0.1, 0.7), cfg()).expect("converges");
        assert!(up.converged);
        assert!(up.residual < 1.0e-12);
        assert!((up.root - Complex::new(0.0, 1.0)).norm() < 1.0e-10);

        // Start in the lower half-plane → -i.
        let down = complex_newton(f, df, Complex::new(0.1, -0.7), cfg()).expect("converges");
        assert!((down.root - Complex::new(0.0, -1.0)).norm() < 1.0e-10);
        assert!(down.residual < 1.0e-12);
    }

    #[test]
    fn halley_z3_minus_one_cube_roots_of_unity() {
        // f(z) = z^3 - 1; roots 1, e^{±2πi/3}.
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z * z - Complex::new(1.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z * 3.0) };
        let d2f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * 6.0) };

        let w_plus = Complex::from_polar(1.0, std::f64::consts::TAU / 3.0);
        let w_minus = Complex::from_polar(1.0, -std::f64::consts::TAU / 3.0);

        let near_one = complex_halley(f, df, d2f, Complex::new(1.3, 0.1), cfg()).expect("ok");
        assert!((near_one.root - Complex::new(1.0, 0.0)).norm() < 1.0e-10);

        let near_up = complex_halley(f, df, d2f, Complex::new(-0.6, 0.9), cfg()).expect("ok");
        assert!((near_up.root - w_plus).norm() < 1.0e-10);

        let near_down = complex_halley(f, df, d2f, Complex::new(-0.6, -0.9), cfg()).expect("ok");
        assert!((near_down.root - w_minus).norm() < 1.0e-10);
    }

    #[test]
    fn transcendental_exp_minus_one() {
        // f(z) = e^z - 1; roots z = 2πik. Start near 0 → 0; start near 2πi → 2πi.
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z.exp() - Complex::new(1.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z.exp()) };
        let d2f = |z: Cplx| -> NumericResult<Cplx> { Ok(z.exp()) };

        let to_zero = complex_halley(f, df, d2f, Complex::new(0.3, 0.2), cfg()).expect("ok");
        assert!(to_zero.root.norm() < 1.0e-10);
        assert!(to_zero.residual < 1.0e-12);

        let two_pi_i = Complex::new(0.0, std::f64::consts::TAU);
        let to_2pi = complex_newton(f, df, Complex::new(0.2, std::f64::consts::TAU - 0.3), cfg())
            .expect("ok");
        assert!((to_2pi.root - two_pi_i).norm() < 1.0e-9);
    }

    #[test]
    fn halley_never_slower_than_newton() {
        // On the same clean problem z^3 - 2 from the same start, Halley should
        // reach the tolerance in no more iterations than Newton.
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z * z - Complex::new(2.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z * 3.0) };
        let d2f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * 6.0) };
        let start = Complex::new(1.5, 0.4);

        let n = complex_newton(f, df, start, cfg()).expect("newton ok");
        let h = complex_halley(f, df, d2f, start, cfg()).expect("halley ok");
        assert!(
            h.iterations <= n.iterations,
            "Halley used {} iters, Newton used {}",
            h.iterations,
            n.iterations
        );
    }

    /// Run a fixed-point map `step` from `z0`, recording `|z_k − root|` until the
    /// error drops below `floor` (machine-noise regime) or `max_steps` is hit.
    fn error_sequence<S>(mut z: Cplx, root: Cplx, floor: f64, max_steps: usize, step: S) -> Vec<f64>
    where
        S: Fn(Cplx) -> Cplx,
    {
        let mut errors = Vec::new();
        for _ in 0..max_steps {
            let e = (z - root).norm();
            errors.push(e);
            if e < floor {
                break;
            }
            z = step(z);
        }
        errors
    }

    /// Estimate the empirical convergence order from three consecutive errors
    /// `p ≈ ln(e2/e1) / ln(e1/e0)` (valid in the asymptotic, pre-roundoff regime).
    fn order_estimate(e0: f64, e1: f64, e2: f64) -> f64 {
        (e2 / e1).ln() / (e1 / e0).ln()
    }

    #[test]
    fn halley_cubic_error_contraction() {
        // On the clean root √2 of f(z) = z^2 - 2, Halley's error sequence
        // contracts cubically while Newton's contracts quadratically. We run both
        // maps by hand from the same start, then compare the empirical order
        // estimate from the first asymptotic triple of each: Halley ≈ 3 > Newton ≈ 2.
        let root = Complex::new(std::f64::consts::SQRT_2, 0.0);
        let f = |z: Cplx| z * z - Complex::new(2.0, 0.0);
        let df = |z: Cplx| z * 2.0;
        let d2f = |_z: Cplx| Complex::new(2.0, 0.0);

        // Halley fixed-point map.
        let halley_step = |z: Cplx| {
            let denom = (df(z) * df(z)) * 2.0 - f(z) * d2f(z);
            z - (f(z) * df(z)) * 2.0 / denom
        };
        // Newton fixed-point map.
        let newton_step = |z: Cplx| z - f(z) / df(z);

        let start = Complex::new(1.0, 0.0);
        let halley_errs = error_sequence(start, root, 1.0e-11, 8, halley_step);
        let newton_errs = error_sequence(start, root, 1.0e-11, 8, newton_step);

        // The first three errors form a clean asymptotic window in both cases.
        assert!(halley_errs.len() >= 3 && newton_errs.len() >= 3);
        let halley_order = order_estimate(halley_errs[0], halley_errs[1], halley_errs[2]);
        let newton_order = order_estimate(newton_errs[0], newton_errs[1], newton_errs[2]);

        // Halley's measured order is near 3 (cubic), Newton's near 2 (quadratic).
        assert!(
            halley_order > 2.5,
            "Halley order estimate {halley_order} not cubic"
        );
        assert!(
            newton_order < 2.5,
            "Newton order estimate {newton_order} unexpectedly high"
        );
        // The decay rate genuinely "triples" relative to Newton: Halley's order is
        // meaningfully larger than Newton's.
        assert!(
            halley_order > newton_order + 0.5,
            "Halley {halley_order} should out-contract Newton {newton_order}"
        );
    }

    #[test]
    fn poly_roots_cube_roots_of_unity_set() {
        // p(z) = z^3 - 1 → {1, e^{2πi/3}, e^{-2πi/3}} as a set.
        let coeffs = vec![
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let roots = complex_poly_roots(&coeffs, cfg()).expect("roots");
        assert_eq!(roots.len(), 3);
        let expected = [
            Complex::new(1.0, 0.0),
            Complex::from_polar(1.0, std::f64::consts::TAU / 3.0),
            Complex::from_polar(1.0, -std::f64::consts::TAU / 3.0),
        ];
        for e in expected {
            assert!(
                min_dist(e, &roots) < 1.0e-10,
                "expected root {e} not matched by {roots:?}"
            );
        }
        // Each returned root has a tiny residual against the original poly.
        for r in &roots {
            assert!(poly_eval(&coeffs, *r).norm() < 1.0e-10);
        }
    }

    #[test]
    fn poly_roots_mixed_complex_factors() {
        // (z - 1)(z - 2i)(z + 3) expanded in ascending powers.
        // = z^3 + (2 - 2i) z^2 + (-3 - 4i) z + (6i)   ... compute below to be safe.
        let r1 = Complex::new(1.0, 0.0);
        let r2 = Complex::new(0.0, 2.0);
        let r3 = Complex::new(-3.0, 0.0);
        // Build coefficients by multiplying linear factors.
        let mut poly = vec![Complex::new(1.0, 0.0)]; // start with constant 1 (degree 0)
        for r in [r1, r2, r3] {
            // multiply current poly by (z - r): shift up (×z) minus r×poly.
            let mut next = vec![Complex::new(0.0, 0.0); poly.len() + 1];
            for (i, c) in poly.iter().enumerate() {
                next[i + 1] += *c; // z · term
                next[i] -= r * *c; // -r · term
            }
            poly = next;
        }

        let roots = complex_poly_roots(&poly, cfg()).expect("roots");
        assert_eq!(roots.len(), 3);
        for e in [r1, r2, r3] {
            assert!(
                min_dist(e, &roots) < 1.0e-9,
                "expected {e} not in {roots:?}"
            );
        }
        for r in &roots {
            assert!(
                poly_eval(&poly, *r).norm() < 1.0e-9,
                "residual too large at {r}"
            );
        }
    }

    #[test]
    fn zero_derivative_is_error_not_panic() {
        // f(z) = z^2 + 1 has f'(0) = 0 while f(0) = 1 ≠ 0, so z = 0 is a critical
        // point that is NOT a root. Newton must trip its vanishing-derivative
        // guard and return an error (not panic, not a false success).
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z + Complex::new(1.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z * 2.0) };
        let res = complex_newton(f, df, Complex::new(0.0, 0.0), cfg());
        assert!(matches!(res, Err(NumericError::NumericalInstability(_))));

        // At the same point the Halley correction 2·f·f' / (2·f'² − f·f'') has a
        // non-zero denominator (−2) but a zero numerator (f' = 0), so the step is
        // exactly 0. The iterate stalls without reaching a root: Halley must also
        // report a NumericalInstability (stall), never a residual≈1 "convergence".
        let d2f = |_z: Cplx| -> NumericResult<Cplx> { Ok(Complex::new(2.0, 0.0)) };
        let h = complex_halley(f, df, d2f, Complex::new(0.0, 0.0), cfg());
        assert!(matches!(h, Err(NumericError::NumericalInstability(_))));
    }

    #[test]
    fn non_convergence_within_max_iter_is_error() {
        // One iteration is far too few for z^2 + 1 from a poor start.
        let f = |z: Cplx| -> NumericResult<Cplx> { Ok(z * z + Complex::new(1.0, 0.0)) };
        let df = |z: Cplx| -> NumericResult<Cplx> { Ok(z * 2.0) };
        let tiny = ComplexRootConfig::new(1.0e-14, 1).expect("config");
        let res = complex_newton(f, df, Complex::new(5.0, 5.0), tiny);
        assert!(matches!(res, Err(NumericError::NotConverged { .. })));
    }

    #[test]
    fn empty_and_constant_polynomials() {
        // Empty input is an error.
        let empty: Vec<Cplx> = Vec::new();
        assert!(matches!(
            complex_poly_roots(&empty, cfg()),
            Err(NumericError::EmptyInput)
        ));

        // Non-zero constant polynomial has no roots.
        let constant = vec![Complex::new(3.0, -1.0)];
        let roots = complex_poly_roots(&constant, cfg()).expect("ok");
        assert!(roots.is_empty());

        // Zero leading coefficient (after trimming, all zero) is rejected.
        let zero_lead = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        assert!(matches!(
            complex_poly_roots(&zero_lead, cfg()),
            Err(NumericError::InvalidParameter(_))
        ));
    }

    #[test]
    fn config_validation_rejects_bad_input() {
        assert!(ComplexRootConfig::new(0.0, 10).is_err());
        assert!(ComplexRootConfig::new(-1.0, 10).is_err());
        assert!(ComplexRootConfig::new(1.0e-12, 0).is_err());
        assert!(ComplexRootConfig::new(1.0e-12, 10).is_ok());
    }
}
