//! Padé rational approximation `[m/n]` from a Taylor series.
//!
//! Given the Taylor coefficients `c₀, c₁, …, c_{m+n}` of a function
//! `f(x) = Σ_{k≥0} c_k x^k`, the **Padé approximant** of type `[m/n]` is the
//! rational function
//!
//! ```text
//!            P(x)     p₀ + p₁ x + … + p_m x^m
//! R(x)  =   ────── = ─────────────────────────,     Q(0) = q₀ = 1,
//!            Q(x)     1  + q₁ x + … + q_n x^n
//! ```
//!
//! whose own Taylor expansion agrees with the given series through order
//! `m + n`, i.e. `f(x) − R(x) = 𝒪(x^{m+n+1})`. Writing
//! `f(x) Q(x) − P(x) = 𝒪(x^{m+n+1})` and matching coefficients gives two
//! coupled linear systems:
//!
//! * the **denominator** equations (orders `m+1 … m+n`) form an `n × n` Toeplitz
//!   system in `q₁ … q_n`,
//!
//!   ```text
//!   Σ_{j=1}^{n} c_{m+k−j} q_j = −c_{m+k},        k = 1 … n,
//!   ```
//!
//! * the **numerator** coefficients (orders `0 … m`) are then explicit,
//!
//!   ```text
//!   p_i = c_i + Σ_{j=1}^{min(i,n)} c_{i−j} q_j,   i = 0 … m
//!   ```
//!
//!   (with `q₀ = 1`).
//!
//! The denominator system is solved by the crate's LU factorisation with partial
//! pivoting. The diagonal `[m/0]` reduces to the truncated Taylor polynomial.
//!
//! Reference: G. A. Baker Jr. and P. Graves-Morris, *Padé Approximants*, 2nd ed.,
//! Encyclopedia of Mathematics and its Applications **59**, Cambridge University
//! Press (1996), §1.1–§1.4.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

/// A Padé approximant `R(x) = P(x) / Q(x)` of type `[m/n]`.
///
/// The numerator coefficients `p` have length `m + 1` (ascending powers) and the
/// denominator coefficients `q` have length `n + 1` with `q[0] == 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct PadeApprox {
    /// Numerator coefficients `p₀ … p_m` in ascending powers of `x`.
    numerator: Vec<f64>,
    /// Denominator coefficients `q₀ … q_n` in ascending powers of `x` (`q₀ = 1`).
    denominator: Vec<f64>,
}

impl PadeApprox {
    /// Construct the Padé approximant `[m/n]` from the Taylor coefficients
    /// `coeffs = [c₀, c₁, …]` (ascending powers).
    ///
    /// At least `m + n + 1` coefficients must be supplied; any beyond that index
    /// are ignored.
    ///
    /// # Errors
    /// * [`NumericError::InvalidParameter`] if `coeffs` is empty or contains a
    ///   non-finite value, or if `coeffs.len() < m + n + 1`.
    /// * [`NumericError::SingularMatrix`] if the `n × n` Padé (Toeplitz) system is
    ///   singular — the `[m/n]` approximant does not exist for this series.
    pub fn new(coeffs: &[f64], m: usize, n: usize) -> NumericResult<Self> {
        let need = m + n + 1;
        if coeffs.len() < need {
            return Err(NumericError::InvalidParameter(format!(
                "Padé [{m}/{n}] needs at least {need} Taylor coefficients, got {}",
                coeffs.len()
            )));
        }
        if coeffs[..need].iter().any(|v| !v.is_finite()) {
            return Err(NumericError::InvalidParameter(
                "Padé: Taylor coefficients must be finite".into(),
            ));
        }

        // Denominator coefficients q₀ … q_n with q₀ = 1.
        let mut q = vec![0.0_f64; n + 1];
        q[0] = 1.0;

        if n > 0 {
            // Assemble the n×n Toeplitz system  A q̃ = b  with
            //   A[k-1][j-1] = c_{m+k-j},   b[k-1] = -c_{m+k},   k,j = 1..n,
            // using the convention c_i = 0 for i < 0 (so the index m+k-j, which
            // can be negative when n > m+1, contributes zero there).
            let mut a = vec![0.0_f64; n * n];
            let mut b = vec![0.0_f64; n];
            for k in 1..=n {
                for j in 1..=n {
                    let signed = m as isize + k as isize - j as isize;
                    a[(k - 1) * n + (j - 1)] = if signed >= 0 {
                        coeffs[signed as usize]
                    } else {
                        0.0
                    };
                }
                b[k - 1] = -coeffs[m + k];
            }
            let (lu, piv, _) = lu_decompose(&a, n)?;
            let qtilde = lu_solve(&lu, &piv, n, &b)?;
            q[1..=n].copy_from_slice(&qtilde);
        }

        // Numerator coefficients p_i = Σ_{j=0}^{min(i,n)} c_{i-j} q_j, i = 0..m.
        let mut p = vec![0.0_f64; m + 1];
        for (i, p_i) in p.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for j in 0..=n.min(i) {
                acc += coeffs[i - j] * q[j];
            }
            *p_i = acc;
        }

        Ok(Self {
            numerator: p,
            denominator: q,
        })
    }

    /// Numerator coefficients `p₀ … p_m` (ascending powers).
    pub fn numerator(&self) -> &[f64] {
        &self.numerator
    }

    /// Denominator coefficients `q₀ … q_n` (ascending powers, `q₀ = 1`).
    pub fn denominator(&self) -> &[f64] {
        &self.denominator
    }

    /// Degree `m` of the numerator.
    pub fn numerator_degree(&self) -> usize {
        self.numerator.len() - 1
    }

    /// Degree `n` of the denominator.
    pub fn denominator_degree(&self) -> usize {
        self.denominator.len() - 1
    }

    /// Evaluate `R(x) = P(x) / Q(x)` at `x` using Horner's scheme.
    ///
    /// # Errors
    /// [`NumericError::SingularMatrix`] if the denominator vanishes at `x` (a pole).
    pub fn evaluate(&self, x: f64) -> NumericResult<f64> {
        let num = horner(&self.numerator, x);
        let den = horner(&self.denominator, x);
        if den == 0.0 || !den.is_finite() {
            return Err(NumericError::SingularMatrix(format!(
                "Padé denominator is zero (pole) at x = {x}"
            )));
        }
        Ok(num / den)
    }

    /// Recover the first `len` Taylor coefficients of `R(x)` by formal power-series
    /// division `P / Q`. By construction these match the input series through order
    /// `m + n`.
    ///
    /// Uses the recurrence `a_k = p_k − Σ_{j=1}^{min(k,n)} q_j a_{k−j}` (with
    /// `p_k = 0` for `k > m`), valid because `q₀ = 1`.
    pub fn taylor_coefficients(&self, len: usize) -> Vec<f64> {
        let m = self.numerator_degree();
        let n = self.denominator_degree();
        let mut a = vec![0.0_f64; len];
        for k in 0..len {
            let mut acc = if k <= m { self.numerator[k] } else { 0.0 };
            for j in 1..=n.min(k) {
                acc -= self.denominator[j] * a[k - j];
            }
            a[k] = acc;
        }
        a
    }
}

/// Horner evaluation of a polynomial with ascending-power coefficients.
fn horner(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0_f64;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Factorials 0!..N! as f64, for building exp/series coefficients.
    fn taylor_exp(order: usize) -> Vec<f64> {
        let mut c = vec![1.0_f64; order + 1];
        for k in 1..=order {
            c[k] = c[k - 1] / k as f64;
        }
        c
    }

    #[test]
    fn m_over_zero_is_truncated_taylor() {
        // [3/0] of exp(x): denominator = 1, numerator = c₀..c₃ exactly.
        let c = taylor_exp(3);
        let pade = PadeApprox::new(&c, 3, 0).expect("ok");
        assert_eq!(pade.denominator(), &[1.0]);
        assert_eq!(pade.numerator().len(), 4);
        for (got, want) in pade.numerator().iter().zip(c.iter()) {
            assert!((got - want).abs() < 1.0e-15, "got {got}, want {want}");
        }
        // Evaluation equals the Horner value of the Taylor polynomial.
        for &x in &[-0.7_f64, 0.0, 0.3, 1.1] {
            let series: f64 = c
                .iter()
                .enumerate()
                .map(|(k, ck)| ck * x.powi(k as i32))
                .sum();
            assert!((pade.evaluate(x).expect("ok") - series).abs() < 1.0e-14);
        }
    }

    #[test]
    fn geometric_series_zero_over_one_is_exact() {
        // 1/(1-x) = 1 + x + x² + … ; the [0/1] Padé must be P=1, Q=1-x exactly.
        let c = vec![1.0_f64; 8];
        let pade = PadeApprox::new(&c, 0, 1).expect("ok");
        assert!((pade.numerator()[0] - 1.0).abs() < 1.0e-15);
        assert!((pade.denominator()[0] - 1.0).abs() < 1.0e-15);
        assert!((pade.denominator()[1] - (-1.0)).abs() < 1.0e-15);
        for &x in &[-0.5_f64, -0.2, 0.0, 0.25, 0.6, 0.9] {
            let exact = 1.0 / (1.0 - x);
            assert!(
                (pade.evaluate(x).expect("ok") - exact).abs() < 1.0e-13,
                "x={x}"
            );
        }
    }

    #[test]
    fn pade_two_two_beats_taylor_four_for_exp() {
        // The [2/2] Padé of exp matches the known closed form (12+6x+x²)/(12−6x+x²)
        // and is markedly more accurate than the degree-4 Taylor at moderate x.
        let c = taylor_exp(4);
        let pade = PadeApprox::new(&c, 2, 2).expect("ok");

        let taylor4 = |x: f64| -> f64 {
            c.iter()
                .enumerate()
                .map(|(k, ck)| ck * x.powi(k as i32))
                .sum()
        };

        // Closed form holds for all x (verify at several points).
        for &x in &[0.5_f64, 1.0, 1.5, 2.0] {
            let closed = (12.0 + 6.0 * x + x * x) / (12.0 - 6.0 * x + x * x);
            assert!(
                (pade.evaluate(x).expect("ok") - closed).abs() < 1.0e-12,
                "x={x}"
            );
        }

        // Padé strictly beats Taylor4 throughout the open interval 0 < x < 2.
        for &x in &[0.25_f64, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75] {
            let exact = x.exp();
            let pade_err = (pade.evaluate(x).expect("ok") - exact).abs();
            let taylor_err = (taylor4(x) - exact).abs();
            assert!(
                pade_err < taylor_err,
                "x={x}: pade_err={pade_err:e} taylor_err={taylor_err:e}"
            );
        }

        // At x=1 the Padé error is at least twice as small (ratio ≈ 2.5).
        let x = 1.0_f64;
        let exact = x.exp();
        let pade_err = (pade.evaluate(x).expect("ok") - exact).abs();
        let taylor_err = (taylor4(x) - exact).abs();
        assert!(
            pade_err < 0.5 * taylor_err,
            "pade_err={pade_err:e} taylor_err={taylor_err:e}"
        );
    }

    #[test]
    fn recovered_series_matches_input_through_order_mn() {
        // The rational function's own Taylor series must reproduce c₀..c_{m+n}.
        let c = taylor_exp(6);
        let (m, n) = (3, 3);
        let pade = PadeApprox::new(&c, m, n).expect("ok");
        let rec = pade.taylor_coefficients(m + n + 1);
        for k in 0..=(m + n) {
            assert!(
                (rec[k] - c[k]).abs() < 1.0e-12,
                "coeff {k}: got {}, want {}",
                rec[k],
                c[k]
            );
        }
    }

    #[test]
    fn denominator_constant_is_unity() {
        let c = taylor_exp(5);
        for (m, n) in [(0_usize, 5_usize), (2, 3), (5, 0), (1, 4)] {
            let pade = PadeApprox::new(&c[..m + n + 1], m, n).expect("ok");
            assert!((pade.denominator()[0] - 1.0).abs() < 1.0e-15, "[{m}/{n}]");
        }
    }

    #[test]
    fn log1p_pade_is_accurate() {
        // ln(1+x) = x - x²/2 + x³/3 - … ; [2/2] is accurate well past the Taylor.
        let c = vec![0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2];
        let pade = PadeApprox::new(&c, 2, 2).expect("ok");
        for &x in &[0.2_f64, 0.5, 0.8, 1.0] {
            let exact = (1.0 + x).ln();
            let err = (pade.evaluate(x).expect("ok") - exact).abs();
            assert!(err < 5.0e-3, "x={x}, err={err:e}");
        }
    }

    #[test]
    fn insufficient_coefficients_errors() {
        let c = taylor_exp(3); // 4 coefficients
        // [2/2] needs 5 coefficients.
        assert!(PadeApprox::new(&c, 2, 2).is_err());
        // exactly enough is fine.
        assert!(PadeApprox::new(&c, 2, 1).is_ok());
    }

    #[test]
    fn non_finite_coefficient_errors() {
        let mut c = taylor_exp(4);
        c[2] = f64::NAN;
        assert!(PadeApprox::new(&c, 2, 2).is_err());
    }

    #[test]
    fn evaluate_at_pole_errors() {
        // 1/(1-x) as [0/1]: pole at x = 1.
        let c = vec![1.0_f64; 4];
        let pade = PadeApprox::new(&c, 0, 1).expect("ok");
        assert!(pade.evaluate(1.0).is_err());
    }
}
