//! Chebyshev series approximation of a function on an interval `[a, b]`.
//!
//! A smooth function `f` is approximated by a truncated Chebyshev expansion
//! `f(x) ≈ Σ_{j=0}^{n} c_j T_j(ξ)` where `ξ = (2x − a − b) / (b − a) ∈ [−1, 1]`
//! and `T_j` is the degree-`j` Chebyshev polynomial of the first kind. The
//! coefficients are obtained by the discrete-cosine (Chebyshev–Gauss)
//! transform sampling `f` at the `n + 1` Chebyshev nodes
//! `x_k = cos(π (k + ½) / (n + 1))`:
//!
//! ```text
//! c_j = (2 / (n + 1)) · Σ_{k=0}^{n} f(x_k) · cos(j · π (k + ½) / (n + 1))
//! ```
//!
//! Evaluation uses the numerically stable Clenshaw recurrence. Term-by-term
//! differentiation and integration of the Chebyshev series are provided in
//! closed form via the standard recurrences.

use crate::error::{NumericError, NumericResult};

/// A Chebyshev-series approximation on `[a, b]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevApprox {
    /// Chebyshev coefficients `c_0 .. c_n` (length = degree + 1).
    coeffs: Vec<f64>,
    /// Lower endpoint of the approximation interval.
    a: f64,
    /// Upper endpoint of the approximation interval.
    b: f64,
}

impl ChebyshevApprox {
    /// Fit a degree-`n` Chebyshev approximation of `f` on `[a, b]`.
    ///
    /// Produces `n + 1` coefficients from `n + 1` Chebyshev-node samples.
    ///
    /// # Errors
    /// Returns [`NumericError::InvalidParameter`] if `n == 0`, and
    /// [`NumericError::InvalidParameter`] if `a == b` or the interval is
    /// non-finite.
    pub fn fit(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> NumericResult<Self> {
        if n == 0 {
            return Err(NumericError::InvalidParameter(
                "Chebyshev degree n must be >= 1".to_string(),
            ));
        }
        if !a.is_finite() || !b.is_finite() || (a - b).abs() == 0.0 {
            return Err(NumericError::InvalidParameter(
                "Chebyshev interval [a, b] must be finite with a != b".to_string(),
            ));
        }

        let m = n + 1; // number of nodes and coefficients
        let half = 0.5 * (b + a);
        let span = 0.5 * (b - a);

        // Sample f at the Chebyshev nodes ξ_k = cos(π (k + ½) / m), mapped to [a, b].
        let mut fvals = vec![0.0_f64; m];
        for (k, fv) in fvals.iter_mut().enumerate() {
            let theta = std::f64::consts::PI * (k as f64 + 0.5) / m as f64;
            let xi = theta.cos();
            *fv = f(half + span * xi);
        }

        // Discrete cosine transform → Chebyshev coefficients.
        let mut coeffs = vec![0.0_f64; m];
        for (j, cj) in coeffs.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (k, &fv) in fvals.iter().enumerate() {
                let angle = std::f64::consts::PI * j as f64 * (k as f64 + 0.5) / m as f64;
                acc += fv * angle.cos();
            }
            *cj = 2.0 / m as f64 * acc;
        }

        Ok(Self { coeffs, a, b })
    }

    /// Construct directly from coefficients on `[a, b]` (mainly for derivative /
    /// integral results). Coefficients use the same convention as [`fit`].
    fn from_coeffs(coeffs: Vec<f64>, a: f64, b: f64) -> Self {
        let coeffs = if coeffs.is_empty() { vec![0.0] } else { coeffs };
        Self { coeffs, a, b }
    }

    /// Evaluate the approximation at `x` using the Clenshaw recurrence.
    ///
    /// `x` is mapped to `ξ ∈ [−1, 1]`; points outside `[a, b]` are evaluated by
    /// the (rapidly diverging) polynomial extrapolation and may be inaccurate.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let span = 0.5 * (self.b - self.a);
        let half = 0.5 * (self.b + self.a);
        // Map to ξ ∈ [-1, 1].
        let xi = (x - half) / span;
        let two_xi = 2.0 * xi;

        // Clenshaw: with the convention f(ξ) = Σ c_j T_j(ξ) and c_0 weighted ½.
        let n = self.coeffs.len();
        let mut d0 = 0.0_f64;
        let mut d1 = 0.0_f64;
        for j in (1..n).rev() {
            let tmp = d0;
            d0 = two_xi * d0 - d1 + self.coeffs[j];
            d1 = tmp;
        }
        // Final step uses the ½ weighting of c_0.
        xi * d0 - d1 + 0.5 * self.coeffs[0]
    }

    /// Chebyshev differentiation: return the approximation of `f'` on `[a, b]`.
    ///
    /// Uses the recurrence `c'_{j-1} = c'_{j+1} + 2 j c_j` (downward), then
    /// rescales by `2 / (b − a)` for the chain rule of the affine map.
    #[must_use]
    pub fn derivative(&self) -> ChebyshevApprox {
        let n = self.coeffs.len();
        if n <= 1 {
            return ChebyshevApprox::from_coeffs(vec![0.0], self.a, self.b);
        }
        // Derivative has degree n-2 → n-1 coefficients.
        let mut d = vec![0.0_f64; n - 1];
        // Downward recurrence over the ξ-domain coefficients.
        if n >= 2 {
            d[n - 2] = 2.0 * (n - 1) as f64 * self.coeffs[n - 1];
        }
        if n >= 3 {
            d[n - 3] = 2.0 * (n - 2) as f64 * self.coeffs[n - 2];
        }
        for j in (1..n - 2).rev() {
            d[j - 1] = d[j + 1] + 2.0 * j as f64 * self.coeffs[j];
        }
        // Chain rule: d/dx = (2 / (b - a)) d/dξ.
        let scale = 2.0 / (self.b - self.a);
        for v in &mut d {
            *v *= scale;
        }
        ChebyshevApprox::from_coeffs(d, self.a, self.b)
    }

    /// Chebyshev integration: return an antiderivative `F` of `f` on `[a, b]`
    /// with the convention `F(a) = 0` after the constant is fixed.
    ///
    /// Uses the recurrence `C_j = (c_{j-1} − c_{j+1}) / (2 j)` (in the ξ domain),
    /// rescaled by `(b − a) / 2` for the affine map, and sets the integration
    /// constant so the antiderivative vanishes at `x = a`.
    #[must_use]
    pub fn integral(&self) -> ChebyshevApprox {
        let n = self.coeffs.len();
        // Antiderivative has one more coefficient.
        let mut big = vec![0.0_f64; n + 1];
        let scale = 0.5 * (self.b - self.a);
        // Pad coefficients for convenient indexing (c_j = 0 for j >= n).
        let c = |j: usize| if j < n { self.coeffs[j] } else { 0.0 };
        for (j, slot) in big.iter_mut().enumerate().take(n + 1).skip(1) {
            let lower = c(j - 1);
            let upper = c(j + 1);
            *slot = scale * (lower - upper) / (2.0 * j as f64);
        }
        // Fix the constant term so F(a) = 0 (ξ = -1).
        let mut result = ChebyshevApprox::from_coeffs(big, self.a, self.b);
        let at_a = result.eval(self.a);
        result.coeffs[0] -= 2.0 * at_a; // c_0 contributes ½ c_0 → subtract 2·F(a).
        result
    }

    /// Return the polynomial degree of the approximation (= coeffs − 1).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Return the approximation interval `[a, b]`.
    #[must_use]
    pub fn interval(&self) -> (f64, f64) {
        (self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn eval_polynomial_exact() {
        // A degree-3 polynomial is reproduced exactly by a degree>=3 Chebyshev fit.
        let f = |x: f64| 2.0 * x * x * x - x * x + 3.0 * x - 5.0;
        let approx = ChebyshevApprox::fit(f, -2.0, 2.0, 6).expect("fit should succeed");
        for &x in &[-1.7, -0.3, 0.0, 0.8, 1.9] {
            assert!((approx.eval(x) - f(x)).abs() < 1e-9, "x={x}");
        }
    }

    #[test]
    fn eval_at_nodes() {
        // The approximation must match f closely at interior points.
        let f = |x: f64| (x).exp();
        let approx = ChebyshevApprox::fit(f, 0.0, 1.0, 16).expect("fit should succeed");
        for &x in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((approx.eval(x) - f(x)).abs() < 1e-10, "x={x}");
        }
    }

    #[test]
    fn fit_constant() {
        let approx = ChebyshevApprox::fit(|_x| 7.0, -1.0, 1.0, 4).expect("fit should succeed");
        for &x in &[-0.9, 0.0, 0.5, 0.99] {
            assert!((approx.eval(x) - 7.0).abs() < 1e-10, "x={x}");
        }
    }

    #[test]
    fn derivative_of_x2() {
        // d/dx (x²) = 2x.
        let approx = ChebyshevApprox::fit(|x| x * x, -3.0, 3.0, 6).expect("fit should succeed");
        let d = approx.derivative();
        for &x in &[-2.5, -1.0, 0.0, 1.5, 2.8] {
            assert!((d.eval(x) - 2.0 * x).abs() < 1e-7, "x={x}: {}", d.eval(x));
        }
    }

    #[test]
    fn integral_of_constant() {
        // ∫_a^x 4 dt = 4(x - a), with F(a) = 0.
        let a = -1.0;
        let approx = ChebyshevApprox::fit(|_x| 4.0, a, 2.0, 4).expect("fit should succeed");
        let integ = approx.integral();
        assert!((integ.eval(a)).abs() < 1e-9, "F(a) != 0");
        for &x in &[-0.5, 0.0, 1.0, 1.9] {
            assert!((integ.eval(x) - 4.0 * (x - a)).abs() < 1e-7, "x={x}");
        }
    }

    #[test]
    fn eval_in_range() {
        // Approximation of a bounded function stays bounded across the interval.
        let approx =
            ChebyshevApprox::fit(|x: f64| x.sin(), -PI, PI, 20).expect("value should be present");
        let mut x = -PI;
        while x <= PI {
            assert!(approx.eval(x).abs() <= 1.0 + 1e-6, "x={x}");
            x += 0.05;
        }
    }

    #[test]
    fn n_0_error() {
        let res = ChebyshevApprox::fit(|x| x, 0.0, 1.0, 0);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    #[test]
    fn a_eq_b_error() {
        let res = ChebyshevApprox::fit(|x| x, 1.0, 1.0, 4);
        assert!(matches!(res, Err(NumericError::InvalidParameter(_))));
    }

    #[test]
    fn sin_approximation() {
        // sin(x) on [0, π] to high accuracy with a moderate degree.
        let approx =
            ChebyshevApprox::fit(|x: f64| x.sin(), 0.0, PI, 24).expect("value should be present");
        for &x in &[0.0, 0.3, 1.0, std::f64::consts::FRAC_PI_2, 2.5, 3.0] {
            assert!((approx.eval(x) - x.sin()).abs() < 1e-9, "x={x}");
        }
    }

    #[test]
    fn degree_correct() {
        let approx = ChebyshevApprox::fit(|x| x, -1.0, 1.0, 5).expect("fit should succeed");
        assert_eq!(approx.degree(), 5);
        // Derivative drops the degree.
        assert_eq!(approx.derivative().degree(), 4);
        // Integral raises it.
        assert_eq!(approx.integral().degree(), 6);
    }

    #[test]
    fn derivative_of_sin_is_cos() {
        // d/dx sin = cos, checked via the Chebyshev derivative operator.
        let approx =
            ChebyshevApprox::fit(|x: f64| x.sin(), 0.0, PI, 24).expect("value should be present");
        let d = approx.derivative();
        for &x in &[0.2, 0.8, 1.5, 2.2, 2.9] {
            assert!((d.eval(x) - x.cos()).abs() < 1e-6, "x={x}: {}", d.eval(x));
        }
    }
}
