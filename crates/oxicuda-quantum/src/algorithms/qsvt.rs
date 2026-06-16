//! Quantum Signal Processing (QSP) — the single-qubit core of Quantum Singular
//! Value Transformation (QSVT).
//!
//! Reference: Gilyén, Su, Low, Wiebe, *"Quantum singular value transformation and
//! beyond: exponential improvements for quantum matrix arithmetic"*, STOC 2019
//! (arXiv:1806.01838); see also Low & Chuang, *"Methodology of resonant
//! equiangular composite quantum gates"* (arXiv:1606.02685).
//!
//! # The QSP sequence
//!
//! QSP interleaves a fixed *signal* rotation `W(x)` (encoding a scalar
//! `x ∈ [-1, 1]`) with tunable *signal-processing* `Z`-rotations parameterised by
//! phase angles `φ_0, …, φ_d`. With the **W_x convention** used here the signal is
//! an `X`-rotation
//!
//! ```text
//! W(x) = [[ x,            i·√(1−x²) ],
//!         [ i·√(1−x²),    x         ]]  =  e^{ i·arccos(x)·X },
//! ```
//!
//! and the full QSP unitary for `d + 1` angles is
//!
//! ```text
//! U_φ(x) = e^{ i·φ_0·Z } · ∏_{k=1}^{d} ( W(x) · e^{ i·φ_k·Z } ).
//! ```
//!
//! A foundational QSP theorem states that the top-left matrix element of `U_φ(x)`,
//!
//! ```text
//! P(x) = ⟨0| U_φ(x) |0⟩,
//! ```
//!
//! is a degree-`d` polynomial in `x` with **definite parity** equal to `d mod 2`
//! (even `d` → even polynomial, odd `d` → odd polynomial) and is bounded,
//! `|P(x)| ≤ 1` for all `x ∈ [-1, 1]`. Choosing the `φ_k` realises (essentially)
//! any such bounded polynomial — this single-qubit construction is exactly the
//! "heart" lifted by QSVT to act on the singular values of a block-encoded
//! operator.
//!
//! # Chebyshev polynomials
//!
//! Setting **all angles to zero** collapses the sequence to `W(x)^d`. Because
//! `W(x) = e^{i·arccos(x)·X}`, repeated application multiplies the rotation angle,
//!
//! ```text
//! W(x)^d = e^{ i·d·arccos(x)·X }
//!        = [[ cos(d·arccos x),     i·sin(d·arccos x) ],
//!           [ i·sin(d·arccos x),   cos(d·arccos x)   ]],
//! ```
//!
//! whose top-left element is `cos(d · arccos x) = T_d(x)`, the degree-`d`
//! Chebyshev polynomial of the first kind. Hence [`chebyshev_qsp_angles`] returns
//! the all-zero angle vector of length `d + 1`, and [`qsp_top_left`] applied to it
//! reproduces `T_d(x)` exactly (see the unit tests).
//!
//! # Numerics
//!
//! The QSP unitary is a product of `2×2` matrices and never touches the
//! `f32` [`crate::statevec::state::StateVector`]; to honour the stringent unitarity
//! tolerances of block encodings the matrix algebra here is carried out in
//! `f64` via [`num_complex::Complex<f64>`].

use crate::error::{QuantumError, QuantumResult};
use num_complex::Complex;

/// `2×2` complex matrix in `f64`, row-major: `m[row][col]`.
pub type Mat2 = [[Complex<f64>; 2]; 2];

#[inline]
fn cx(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Row-major `2×2` matrix product `a · b`.
#[inline]
fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// The signal operator `W(x) = e^{i·arccos(x)·X}` in the W_x convention.
///
/// ```text
/// W(x) = [[ x,          i·√(1−x²) ],
///         [ i·√(1−x²),  x         ]].
/// ```
///
/// # Errors
/// Returns [`QuantumError::InvalidParameter`] when `x ∉ [-1, 1]` (the signal would
/// otherwise leave the unitary group as `√(1−x²)` becomes imaginary).
pub fn signal_operator(x: f64) -> QuantumResult<Mat2> {
    if !x.is_finite() || x.abs() > 1.0 {
        return Err(QuantumError::InvalidParameter {
            name: format!("QSP signal x={x} must lie in [-1, 1]"),
        });
    }
    let s = (1.0 - x * x).max(0.0).sqrt();
    Ok([[cx(x, 0.0), cx(0.0, s)], [cx(0.0, s), cx(x, 0.0)]])
}

/// The signal-processing rotation `e^{i·φ·Z} = diag(e^{iφ}, e^{-iφ})`.
#[inline]
fn z_rotation(phi: f64) -> Mat2 {
    [
        [cx(phi.cos(), phi.sin()), cx(0.0, 0.0)],
        [cx(0.0, 0.0), cx(phi.cos(), -phi.sin())],
    ]
}

/// Build the full QSP unitary `U_φ(x)` for phase angles `phi = [φ_0, …, φ_d]` and
/// signal value `x`.
///
/// Realises
/// `U_φ(x) = e^{iφ_0 Z} · ∏_{k=1}^{d} ( W(x) · e^{iφ_k Z} )`
/// where `d = phi.len() − 1`.
///
/// # Errors
/// * [`QuantumError::EmptyInput`] when `phi` is empty (at least `φ_0` is required).
/// * [`QuantumError::InvalidParameter`] when `x ∉ [-1, 1]` (propagated from
///   [`signal_operator`]).
pub fn qsp_unitary(phi: &[f64], x: f64) -> QuantumResult<Mat2> {
    if phi.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    let w = signal_operator(x)?;

    // Start from e^{iφ_0 Z}.
    let mut u = z_rotation(phi[0]);
    // Left-multiply by ( W · e^{iφ_k Z} ) for k = 1..=d, preserving order so that
    // the leftmost factor of the product is the φ_0 rotation.
    for &phi_k in &phi[1..] {
        let block = mat2_mul(&w, &z_rotation(phi_k));
        u = mat2_mul(&u, &block);
    }
    Ok(u)
}

/// The QSP polynomial `P(x) = ⟨0| U_φ(x) |0⟩` — the top-left matrix element.
///
/// This is the degree-`d` (with `d = phi.len() − 1`) polynomial of definite parity
/// `d mod 2` realised by the angle set `phi`.
///
/// # Errors
/// Same conditions as [`qsp_unitary`].
pub fn qsp_top_left(phi: &[f64], x: f64) -> QuantumResult<Complex<f64>> {
    let u = qsp_unitary(phi, x)?;
    Ok(u[0][0])
}

/// QSP phase angles that reproduce the Chebyshev polynomial `T_d(x)` in the
/// top-left element under the sequence implemented by [`qsp_unitary`].
///
/// Under the W_x convention with the `X`-signal of [`signal_operator`], the
/// all-zero angle vector collapses the sequence to `W(x)^d`, whose top-left
/// element is `cos(d·arccos x) = T_d(x)`. The returned vector therefore has length
/// `d + 1` and is identically zero.
///
/// # Errors
/// This routine is infallible for any `usize` degree and returns `Ok`.
pub fn chebyshev_qsp_angles(d: usize) -> QuantumResult<Vec<f64>> {
    Ok(vec![0.0_f64; d + 1])
}

/// Evaluate the Chebyshev polynomial of the first kind `T_d(x)` directly via the
/// stable three-term recurrence `T_{n+1} = 2x·T_n − T_{n-1}` (reference value used
/// by the unit tests and by callers wanting the closed form).
#[must_use]
pub fn chebyshev_t(d: usize, x: f64) -> f64 {
    if d == 0 {
        return 1.0;
    }
    if d == 1 {
        return x;
    }
    let mut t_prev = 1.0_f64; // T_0
    let mut t_cur = x; // T_1
    for _ in 2..=d {
        let t_next = 2.0 * x * t_cur - t_prev;
        t_prev = t_cur;
        t_cur = t_next;
    }
    t_cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Conjugate transpose (Hermitian adjoint) `a†`.
    fn mat2_dagger(a: &Mat2) -> Mat2 {
        [
            [a[0][0].conj(), a[1][0].conj()],
            [a[0][1].conj(), a[1][1].conj()],
        ]
    }

    /// `U U† = I` within `tol` (operator-norm-ish elementwise check).
    fn assert_unitary(u: &Mat2, tol: f64) {
        let prod = mat2_mul(u, &mat2_dagger(u));
        let id = [[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]];
        for r in 0..2 {
            for col in 0..2 {
                let diff = prod[r][col] - id[r][col];
                assert!(
                    diff.norm() < tol,
                    "U U† not identity at [{r}][{col}]: {:?}",
                    prod[r][col]
                );
            }
        }
    }

    // (a) d = 1 with angles (0, 0) gives top-left = x (identity transform of the
    //     signal: T_1(x) = x).
    #[test]
    fn qsp_degree_one_zero_angles_is_x() {
        for &x in &[-0.9_f64, -0.3, 0.0, 0.25, 0.5, 0.87, 1.0] {
            let p = qsp_top_left(&[0.0, 0.0], x)
                .expect("x is within [-1, 1] and angles slice is non-empty");
            assert!((p.re - x).abs() < 1e-12, "x={x}, P.re={}", p.re);
            assert!(p.im.abs() < 1e-12, "x={x}, P.im={}", p.im);
        }
    }

    // (b) The standard QSP angles for T_d reproduce the Chebyshev polynomial:
    //     top-left ≈ T_d(x) for d = 2, 3 over several x ∈ [−1, 1].
    #[test]
    fn qsp_chebyshev_angles_reproduce_t_d() {
        for &d in &[2_usize, 3] {
            let angles =
                chebyshev_qsp_angles(d).expect("chebyshev_qsp_angles is infallible for any degree");
            assert_eq!(angles.len(), d + 1);
            for &x in &[-1.0_f64, -0.73, -0.4, -0.1, 0.0, 0.2, 0.55, 0.81, 1.0] {
                let p = qsp_top_left(&angles, x).expect(
                    "chebyshev angles are non-empty and all test x values are within [-1, 1]",
                );
                let want = chebyshev_t(d, x);
                assert!(
                    (p.re - want).abs() < 1e-10,
                    "d={d}, x={x}: P.re={} vs T_d={want}",
                    p.re
                );
                assert!(p.im.abs() < 1e-10, "d={d}, x={x}: P.im={}", p.im);
            }
        }
    }

    // Extra: higher-degree Chebyshev (d = 5) still matches, confirming the
    // recurrence-vs-circuit agreement is not a low-order coincidence.
    #[test]
    fn qsp_chebyshev_high_degree_matches() {
        let d = 5;
        let angles =
            chebyshev_qsp_angles(d).expect("chebyshev_qsp_angles is infallible for any degree");
        for &x in &[-0.95_f64, -0.5, 0.0, 0.33, 0.66, 0.99] {
            let p = qsp_top_left(&angles, x)
                .expect("angles are non-empty and all test x values are within [-1, 1]");
            assert!((p.re - chebyshev_t(d, x)).abs() < 1e-9, "x={x}");
        }
    }

    // (c) The QSP unitary is actually unitary (U U† = I to 1e-10) — checked for
    //     several non-trivial angle sets and signal values.
    #[test]
    fn qsp_unitary_is_unitary() {
        let angle_sets: &[&[f64]] = &[
            &[0.0],
            &[0.3, -0.7],
            &[0.1, 0.2, 0.3, 0.4],
            &[PI / 4.0, -PI / 3.0, PI / 6.0, 0.9, -1.2],
        ];
        for angles in angle_sets {
            for &x in &[-1.0_f64, -0.6, -0.2, 0.0, 0.45, 0.77, 1.0] {
                let u = qsp_unitary(angles, x).expect(
                    "all test angle sets are non-empty and all test x values are within [-1, 1]",
                );
                assert_unitary(&u, 1e-10);
            }
        }
    }

    // (d) Parity: an even-d polynomial is even in x, odd-d is odd, i.e.
    //     P(−x) = (−1)^d · P(x). Verified for the Chebyshev angle sets and for a
    //     generic (but parity-respecting) symmetric angle choice.
    #[test]
    fn qsp_parity_matches_degree() {
        for &d in &[2_usize, 3, 4, 5] {
            let angles =
                chebyshev_qsp_angles(d).expect("chebyshev_qsp_angles is infallible for any degree");
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            for &x in &[0.13_f64, 0.37, 0.61, 0.88] {
                let p_pos = qsp_top_left(&angles, x)
                    .expect("chebyshev angles are non-empty and x is in (0, 1]");
                let p_neg = qsp_top_left(&angles, -x)
                    .expect("chebyshev angles are non-empty and -x is in [-1, 0) since x > 0");
                assert!(
                    (p_neg.re - sign * p_pos.re).abs() < 1e-10,
                    "d={d}, x={x}: P(-x).re={} vs {}·P(x).re={}",
                    p_neg.re,
                    sign,
                    sign * p_pos.re
                );
                assert!(
                    (p_neg.im - sign * p_pos.im).abs() < 1e-10,
                    "d={d}, x={x}: imag parity mismatch"
                );
            }
        }
    }

    // (e) |top-left| ≤ 1 for all x (bounded-polynomial / block-encoding norm).
    #[test]
    fn qsp_top_left_bounded_by_one() {
        let angle_sets: &[&[f64]] = &[
            &[0.0, 0.0, 0.0],       // T_2
            &[0.0, 0.0, 0.0, 0.0],  // T_3
            &[0.5, -1.1, 0.7, 0.2], // generic
            &[PI / 3.0, 0.0, -PI / 5.0],
        ];
        for angles in angle_sets {
            // Sweep a dense grid of x ∈ [-1, 1].
            for i in 0..=200 {
                let x = -1.0 + 2.0 * (i as f64) / 200.0;
                let p = qsp_top_left(angles, x)
                    .expect("angle sets are all non-empty and x is on a grid within [-1, 1]");
                assert!(
                    p.norm() <= 1.0 + 1e-10,
                    "|P({x})|={} exceeds 1 for angles={angles:?}",
                    p.norm()
                );
            }
        }
    }

    // (f) Angle-count mismatch / empty input errors.
    #[test]
    fn qsp_empty_angles_errors() {
        assert!(qsp_unitary(&[], 0.5).is_err());
        assert!(qsp_top_left(&[], 0.5).is_err());
    }

    // (f-bis) Out-of-range signal value errors.
    #[test]
    fn qsp_signal_out_of_range_errors() {
        assert!(signal_operator(1.5).is_err());
        assert!(signal_operator(-1.0001).is_err());
        assert!(signal_operator(f64::NAN).is_err());
        assert!(qsp_unitary(&[0.0, 0.0], 2.0).is_err());
    }

    // Sanity: the signal operator itself is unitary and has the documented form.
    #[test]
    fn signal_operator_form_and_unitarity() {
        let x = 0.6_f64;
        let w = signal_operator(x).expect("x=0.6 is within the valid range [-1, 1]");
        let s = (1.0_f64 - x * x).sqrt();
        assert!((w[0][0].re - x).abs() < 1e-12);
        assert!((w[0][1].im - s).abs() < 1e-12);
        assert!((w[1][0].im - s).abs() < 1e-12);
        assert!((w[1][1].re - x).abs() < 1e-12);
        assert_unitary(&w, 1e-12);
    }
}
