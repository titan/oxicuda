//! Richardson Extrapolation and Romberg Integration.
//!
//! Richardson extrapolation (Richardson 1911) eliminates the leading error term from
//! a sequence of approximations at step sizes `h` and `h/2`:
//!
//! ```text
//! R(h, h/2) = (2^p · f(h/2) - f(h)) / (2^p - 1)
//! ```
//!
//! where `p` is the order of the method (i.e., the error behaves as `c·hᵖ`).
//!
//! Romberg integration applies Richardson extrapolation iteratively to the composite
//! trapezoidal rule, building a triangular table `T[i][j]` where:
//!
//! - `T[i][0]` is the trapezoidal approximation with `2^i` subintervals,
//! - `T[i][j] = Richardson(T[i-1][j-1], T[i][j-1], 2j)` eliminates the `h^{2j}` term.
//!
//! The resulting method converges faster than any finite power of `h` for smooth functions.

use crate::error::{NumericError, NumericResult};

// ---------------------------------------------------------------------------
// Richardson extrapolation
// ---------------------------------------------------------------------------

/// Single Richardson extrapolation step.
///
/// Given two approximations to the same quantity at step sizes `h` and `h/2`,
/// with leading error `c·hᵖ`, the extrapolated result cancels the `O(hᵖ)` term:
///
/// `R = (2^p · f(h/2) - f(h)) / (2^p - 1)`
///
/// # Arguments
/// - `f_h`: approximation at full step `h`.
/// - `f_h2`: approximation at half step `h/2`.
/// - `p`: order of the method (error ≈ c·hᵖ).
///
/// # Returns
/// The Richardson-extrapolated value (exact if error is exactly `c·hᵖ`).
#[must_use]
pub fn richardson_extrapolate(f_h: f64, f_h2: f64, p: u32) -> f64 {
    let pow2p = (1u64 << p) as f64; // 2^p
    (pow2p * f_h2 - f_h) / (pow2p - 1.0)
}

// ---------------------------------------------------------------------------
// Romberg integration
// ---------------------------------------------------------------------------

/// Romberg integration of `f` on `[a, b]` with `n_refinements` levels.
///
/// Builds a `(n_refinements + 1) × (n_refinements + 1)` triangular Romberg table.
/// Returns the highest-order estimate `T[n_refinements][n_refinements]`.
///
/// # Arguments
/// - `f`: integrand (must be smooth on `[a, b]`).
/// - `a`, `b`: limits of integration (`a < b` required).
/// - `n_refinements`: number of Richardson levels beyond the trapezoidal base.
///   Level 0 = trapezoidal with 1 panel. Level `n` = `2^n` panels at base.
///
/// # Errors
/// - [`NumericError::InvalidParameter`] if `n_refinements == 0` or limits are non-finite / reversed.
pub fn romberg_integration<F>(f: F, a: f64, b: f64, n_refinements: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> f64,
{
    let tbl = romberg_table(f, a, b, n_refinements)?;
    Ok(tbl[n_refinements][n_refinements])
}

/// Build the full `(n+1) × (n+1)` Romberg extrapolation table.
///
/// Entry `T[i][j]`:
/// - `T[i][0]` = composite trapezoidal with `2^i` intervals.
/// - `T[i][j]` = Richardson extrapolation of `T[i-1][j-1]` and `T[i][j-1]`
///   eliminating the leading `h^{2j}` error term (trapezoidal has even-order errors).
///
/// # Arguments
/// - `f`: integrand.
/// - `a`, `b`: integration bounds.
/// - `n`: number of refinement levels (table has size `(n+1) × (n+1)`).
///
/// # Errors
/// - [`NumericError::InvalidParameter`] if `n == 0` or limits are non-finite / reversed.
pub fn romberg_table<F>(f: F, a: f64, b: f64, n: usize) -> NumericResult<Vec<Vec<f64>>>
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter(format!(
            "non-finite integration limits: a={a}, b={b}"
        )));
    }
    if b <= a {
        return Err(NumericError::InvalidParameter(format!(
            "integration limits must satisfy a < b, got a={a}, b={b}"
        )));
    }
    if n == 0 {
        return Err(NumericError::InvalidParameter(
            "n_refinements must be ≥ 1".into(),
        ));
    }

    let h0 = b - a;
    // Allocate triangular table (use square for simplicity; entries [i][j] valid for j ≤ i).
    let size = n + 1;
    let mut t = vec![vec![0.0_f64; size]; size];

    // Row 0: simple trapezoidal with 1 interval (2 endpoints)
    t[0][0] = 0.5 * h0 * (f(a) + f(b));

    for i in 1..=n {
        // Halve the step: use 2^(i-1) new interior midpoints relative to the previous grid.
        let n_prev = 1_usize << (i - 1); // 2^{i-1} intervals at level i-1
        let h = h0 / (n_prev as f64 * 2.0); // half step
        // Sum function at new midpoints
        let mut mid_sum = 0.0_f64;
        for k in 0..n_prev {
            let x = a + h0 * (2 * k + 1) as f64 / (2 * n_prev) as f64;
            mid_sum += f(x);
        }
        // Composite trapezoidal with 2^i intervals:
        // T[i][0] = T[i-1][0] / 2 + h * sum_midpoints
        t[i][0] = 0.5 * t[i - 1][0] + h * mid_sum;

        // Richardson extrapolation columns: the trapezoidal rule has error in even powers of h,
        // so each extrapolation step doubles the order (O(h²) → O(h⁴) → …).
        // The extrapolation order for column j is p = 2j.
        let mut pow4 = 4.0_f64; // 4^j for j=1
        for j in 1..=i {
            t[i][j] = (pow4 * t[i][j - 1] - t[i - 1][j - 1]) / (pow4 - 1.0);
            pow4 *= 4.0;
        }
    }

    Ok(t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn richardson_p1_correct() {
        // Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h has error O(h).
        // For f(x) = x², f'(0) = 0.  At x=1, f'(1) = 2.
        // FD at h=0.1: (1.21 - 1) / 0.1 = 2.1
        // FD at h=0.05: (1.1025 - 1) / 0.05 = 2.05
        // Richardson p=1: (2*2.05 - 2.1) / (2-1) = 2.0
        let fd_h = (1.21_f64 - 1.0) / 0.1;
        let fd_h2 = (1.1025_f64 - 1.0) / 0.05;
        let ext = richardson_extrapolate(fd_h, fd_h2, 1);
        assert!((ext - 2.0).abs() < 1.0e-10, "got {ext}");
    }

    #[test]
    fn richardson_p2_correct() {
        // Trapezoidal rule for ∫₀¹ x dx = 0.5 has error O(h²).
        // trap(h=1): 0.5 * 1 * (f(0) + f(1)) = 0.5
        // trap(h=0.5): 0.5 * 0.5 * (f(0) + 2f(0.5) + f(1)) = 0.5
        // Both exact here. Use a harder integrand: ∫₀¹ x² dx = 1/3.
        // trap(h=1): (0 + 1) / 2 = 0.5, error = 0.5 - 1/3 = 1/6
        // trap(h=0.5): (0 + 2*(0.25) + 1) / (2*2) = 1.5/4 = 3/8, error = 3/8 - 1/3 = 1/24
        // p=2: (4*3/8 - 1/2) / (4-1) = (1.5 - 0.5) / 3 = 1/3 ✓
        let trap_h = 0.5_f64; // trapezoidal with 1 interval
        let trap_h2 = 3.0_f64 / 8.0; // trapezoidal with 2 intervals
        let ext = richardson_extrapolate(trap_h, trap_h2, 2);
        assert!((ext - 1.0 / 3.0).abs() < 1.0e-12, "got {ext}");
    }

    #[test]
    fn romberg_constant() {
        // ∫₀¹ 3 dx = 3
        let result = romberg_integration(|_x| 3.0, 0.0, 1.0, 4).expect("ok");
        assert!(
            (result - 3.0).abs() < 1.0e-10,
            "constant integral: got {result}"
        );
    }

    #[test]
    fn romberg_polynomial() {
        // ∫₀¹ x³ dx = 1/4
        let result = romberg_integration(|x| x.powi(3), 0.0, 1.0, 4).expect("ok");
        assert!((result - 0.25).abs() < 1.0e-10, "x³: got {result}");
    }

    #[test]
    fn romberg_sin() {
        // ∫₀^π sin(x) dx = 2
        let result = romberg_integration(|x| x.sin(), 0.0, PI, 6).expect("ok");
        assert!((result - 2.0).abs() < 1.0e-10, "sin: got {result}");
    }

    #[test]
    fn romberg_converges() {
        // ∫₀^π sin(x) dx = 2  with tight tolerance
        let r4 = romberg_integration(|x| x.sin(), 0.0, PI, 4).expect("ok");
        let r8 = romberg_integration(|x| x.sin(), 0.0, PI, 8).expect("ok");
        // Higher refinements should give same or better result
        assert!((r8 - 2.0).abs() <= (r4 - 2.0).abs() + 1.0e-14);
    }

    #[test]
    fn romberg_table_shape() {
        let tbl = romberg_table(|x| x.sin(), 0.0, PI, 5).expect("ok");
        assert_eq!(tbl.len(), 6, "n+1 rows");
        for row in &tbl {
            assert_eq!(row.len(), 6, "n+1 columns per row");
        }
    }

    #[test]
    fn romberg_vs_trap_more_accurate() {
        // Romberg at level 3 should beat simple trapezoidal at same # evaluations.
        let f = |x: f64| (x * x * x * x + 1.0).ln();
        let exact = {
            // Use a fine reference: Romberg with 12 levels
            romberg_integration(f, 0.0, 2.0, 12).expect("ref")
        };
        let tbl = romberg_table(f, 0.0, 2.0, 3).expect("ok");
        let trap_err = (tbl[3][0] - exact).abs(); // trapezoidal level 3
        let romberg_err = (tbl[3][3] - exact).abs(); // Romberg level 3
        assert!(
            romberg_err < trap_err,
            "Romberg (err={romberg_err:.2e}) should beat trapezoidal (err={trap_err:.2e})"
        );
    }

    #[test]
    fn richardson_order_matches() {
        // For the trapezoidal rule (p=2) applied to ∫₀¹ x⁴ dx = 1/5:
        // After Richardson (p=2), the result should be much closer to 1/5.
        let trap1 = 0.5 * (0.0 + 1.0); // 1 interval
        let trap2 = {
            let h = 0.5;
            h * (0.5 * 0.0_f64 + 0.5_f64.powi(4) + 0.5 * 1.0_f64)
        };
        let ext = richardson_extrapolate(trap1, trap2, 2);
        assert!(
            (ext - 0.2).abs() < (trap2 - 0.2).abs(),
            "Richardson should reduce error vs plain trapezoidal"
        );
    }
}
