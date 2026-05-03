//! SSM discretization: convert continuous-time `(A, B)` to discrete `(Ā, B̄)`.
//!
//! Given a continuous-time LTI system
//!
//! ```text
//! ḣ(t) = A h(t) + B u(t)
//! ```
//!
//! with a diagonal `A` ∈ ℝ^{N×N} and input matrix `B` ∈ ℝ^N, this module
//! converts the pair into a discrete-time recurrence
//!
//! ```text
//! h[k] = Ā h[k-1] + B̄ u[k]
//! ```
//!
//! using one of three classical methods (ZOH, Bilinear, Euler).
//!
//! All arithmetic is f32 single-precision to match GPU kernels.

use crate::error::{MambaError, MambaResult};

// ─── Discretization method ───────────────────────────────────────────────────

/// Method used to convert a continuous-time SSM to discrete time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretization {
    /// **Zero-Order Hold** (exact exponential hold).
    ///
    /// `Ā[i] = exp(Δ * A[i])`
    /// `B̄[i] = (Ā[i] − 1) / A[i] * B[i]`  (limit `Δ*B[i]` when `A[i]` ≈ 0)
    Zoh,

    /// **Bilinear (Tustin)** approximation.
    ///
    /// `Ā[i] = (1 + Δ/2 * A[i]) / (1 − Δ/2 * A[i])`
    /// `B̄[i] = Δ * B[i] / (1 − Δ/2 * A[i])`
    Bilinear,

    /// **Forward Euler** approximation.
    ///
    /// `Ā[i] = 1 + Δ * A[i]`
    /// `B̄[i] = Δ * B[i]`
    Euler,
}

// ─── Threshold for near-zero A ───────────────────────────────────────────────

/// Absolute threshold below which |A[i]| is treated as zero for ZOH.
const ZOH_ZERO_THRESHOLD: f32 = 1e-6;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Discretize a diagonal SSM `(A, B)` with time-step `Δ`.
///
/// # Arguments
///
/// * `a_diag` — Diagonal entries of the A matrix, shape `[N]`.
/// * `b`      — B vector, shape `[N]`.
/// * `delta`  — Discretization time-step `Δ > 0`.
/// * `method` — Discretization method to use.
///
/// # Returns
///
/// `(a_bar, b_bar)`, both length `N`.
///
/// # Errors
///
/// * [`MambaError::NonPositiveDelta`] — if `delta ≤ 0`.
/// * [`MambaError::EmptyInput`]       — if slices are empty.
/// * [`MambaError::DimensionMismatch`] — if `a_diag.len() ≠ b.len()`.
pub fn discretize(
    a_diag: &[f32],
    b: &[f32],
    delta: f32,
    method: Discretization,
) -> MambaResult<(Vec<f32>, Vec<f32>)> {
    // ── Validation ────────────────────────────────────────────────────────────
    if delta <= 0.0 {
        return Err(MambaError::NonPositiveDelta(delta));
    }
    if a_diag.is_empty() {
        return Err(MambaError::EmptyInput("a_diag"));
    }
    if a_diag.len() != b.len() {
        return Err(MambaError::DimensionMismatch {
            expected: a_diag.len(),
            got: b.len(),
        });
    }

    let n = a_diag.len();
    let mut a_bar = Vec::with_capacity(n);
    let mut b_bar = Vec::with_capacity(n);

    match method {
        Discretization::Zoh => {
            for i in 0..n {
                let a_i = a_diag[i];
                let exp_val = (delta * a_i).exp();
                a_bar.push(exp_val);
                if a_i.abs() < ZOH_ZERO_THRESHOLD {
                    // L'Hôpital / Taylor limit: (exp(Δ*A) − 1)/A → Δ as A→0
                    b_bar.push(delta * b[i]);
                } else {
                    b_bar.push((exp_val - 1.0) / a_i * b[i]);
                }
            }
        }
        Discretization::Bilinear => {
            for i in 0..n {
                let a_i = a_diag[i];
                let half_delta_a = 0.5 * delta * a_i;
                let denom = 1.0 - half_delta_a;
                // Stable for negative A (denom > 1); still well-defined for
                // positive A as long as denom ≠ 0 (which requires Δ*A ≠ 2).
                a_bar.push((1.0 + half_delta_a) / denom);
                b_bar.push(delta * b[i] / denom);
            }
        }
        Discretization::Euler => {
            for i in 0..n {
                a_bar.push(1.0 + delta * a_diag[i]);
                b_bar.push(delta * b[i]);
            }
        }
    }

    Ok((a_bar, b_bar))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── ZOH ──────────────────────────────────────────────────────────────────

    /// ZOH, delta=1, A=[-1], B=[1].  Ā ≈ exp(-1) ≈ 0.3679.
    #[test]
    fn zoh_basic_negative_a() {
        let (a_bar, b_bar) =
            discretize(&[-1.0], &[1.0], 1.0, Discretization::Zoh).expect("valid input");
        let expected_a = (-1.0_f32).exp(); // ≈ 0.36788
        assert!(
            (a_bar[0] - expected_a).abs() < EPS,
            "a_bar={} expected≈{expected_a}",
            a_bar[0]
        );
        // B_bar = (exp(-1) - 1) / (-1) * 1 = (1 - exp(-1)) ≈ 0.63212
        let expected_b = (1.0 - expected_a) / 1.0;
        assert!(
            (b_bar[0] - expected_b).abs() < EPS,
            "b_bar={} expected≈{expected_b}",
            b_bar[0]
        );
    }

    /// ZOH with A=[0.0] must not produce NaN/Inf — hits the near-zero branch.
    #[test]
    fn zoh_near_zero_a_graceful() {
        let (a_bar, b_bar) =
            discretize(&[0.0], &[2.5], 0.5, Discretization::Zoh).expect("valid input");
        // A=0 branch: A_bar = exp(0) = 1, B_bar ≈ delta * B = 0.5 * 2.5 = 1.25
        assert!(a_bar[0].is_finite(), "a_bar must be finite");
        assert!(b_bar[0].is_finite(), "b_bar must be finite");
        assert!((a_bar[0] - 1.0).abs() < EPS, "A=0 => a_bar should be 1");
        let expected_b = 0.5 * 2.5;
        assert!(
            (b_bar[0] - expected_b).abs() < 1e-4,
            "b_bar={} expected≈{expected_b}",
            b_bar[0]
        );
    }

    /// ZOH with small but non-zero A close to the threshold, continuity check.
    #[test]
    fn zoh_b_bar_continuity_near_zero() {
        // ZOH limit as A→0 equals Euler: B_bar ≈ Δ*B
        let (_, b_bar_zoh) =
            discretize(&[1e-8], &[1.0], 1.0, Discretization::Zoh).expect("valid input");
        let (_, b_bar_euler) =
            discretize(&[1e-8], &[1.0], 1.0, Discretization::Euler).expect("valid input");
        assert!(
            (b_bar_zoh[0] - b_bar_euler[0]).abs() < 1e-4,
            "ZOH and Euler should agree when A≈0: zoh={} euler={}",
            b_bar_zoh[0],
            b_bar_euler[0]
        );
    }

    /// ZOH with positive A — A_bar > 1 (unstable, but still finite).
    #[test]
    fn zoh_positive_a_finite() {
        let (a_bar, b_bar) =
            discretize(&[2.0], &[1.0], 0.1, Discretization::Zoh).expect("valid input");
        assert!(a_bar[0].is_finite());
        assert!(b_bar[0].is_finite());
        let expected = (0.2_f32).exp();
        assert!((a_bar[0] - expected).abs() < EPS);
    }

    // ── Bilinear ─────────────────────────────────────────────────────────────

    /// Bilinear: stable input (negative A) must produce |Ā| < 1.
    #[test]
    fn bilinear_stable_a_magnitude_less_than_one() {
        let a_diag = [-1.0_f32, -2.0, -0.5, -5.0];
        let b = [1.0_f32; 4];
        let (a_bar, _) =
            discretize(&a_diag, &b, 0.1, Discretization::Bilinear).expect("valid input");
        for (i, &ab) in a_bar.iter().enumerate() {
            assert!(
                ab.abs() < 1.0,
                "bilinear a_bar[{i}]={ab} should be < 1 for stable A"
            );
        }
    }

    /// Bilinear formula: A_bar = (1 + Δ/2*A) / (1 - Δ/2*A).
    #[test]
    fn bilinear_formula_check() {
        let delta = 0.2_f32;
        let a = -3.0_f32;
        let b_val = 1.0_f32;
        let (a_bar, b_bar) =
            discretize(&[a], &[b_val], delta, Discretization::Bilinear).expect("valid input");
        let half = 0.5 * delta * a;
        let expected_a = (1.0 + half) / (1.0 - half);
        let expected_b = delta * b_val / (1.0 - half);
        assert!((a_bar[0] - expected_a).abs() < EPS, "a_bar mismatch");
        assert!((b_bar[0] - expected_b).abs() < EPS, "b_bar mismatch");
    }

    // ── Euler ─────────────────────────────────────────────────────────────────

    /// Euler: A_bar = 1 + Δ*A exactly.
    #[test]
    fn euler_formula_check() {
        let delta = 0.1_f32;
        let a_diag = [-1.0_f32, -2.0, 0.5, 3.0];
        let b = [1.0_f32, 2.0, 0.5, -1.0];
        let (a_bar, b_bar) =
            discretize(&a_diag, &b, delta, Discretization::Euler).expect("valid input");
        for i in 0..a_diag.len() {
            let expected_a = 1.0 + delta * a_diag[i];
            let expected_b = delta * b[i];
            assert!((a_bar[i] - expected_a).abs() < EPS, "euler a_bar[{i}]");
            assert!((b_bar[i] - expected_b).abs() < EPS, "euler b_bar[{i}]");
        }
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    /// delta = 0.0 must return NonPositiveDelta.
    #[test]
    fn error_zero_delta() {
        let err = discretize(&[-1.0], &[1.0], 0.0, Discretization::Zoh).expect_err("should fail");
        assert_eq!(err, MambaError::NonPositiveDelta(0.0));
    }

    /// delta = -0.1 must return NonPositiveDelta.
    #[test]
    fn error_negative_delta() {
        let err = discretize(&[-1.0], &[1.0], -0.1, Discretization::Zoh).expect_err("should fail");
        assert_eq!(err, MambaError::NonPositiveDelta(-0.1));
    }

    /// Empty a_diag must return EmptyInput.
    #[test]
    fn error_empty_input() {
        let err = discretize(&[], &[], 1.0, Discretization::Zoh).expect_err("should fail");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    /// Shape mismatch between a_diag and b.
    #[test]
    fn error_shape_mismatch() {
        let err =
            discretize(&[-1.0, -2.0], &[1.0], 0.1, Discretization::Zoh).expect_err("should fail");
        assert!(
            matches!(
                err,
                MambaError::DimensionMismatch {
                    expected: 2,
                    got: 1
                }
            ),
            "unexpected error: {err}"
        );
    }

    // ── Finite output for all methods, N=4 ────────────────────────────────────

    /// All three methods must produce finite output for N=4.
    #[test]
    fn all_methods_finite_n4() {
        let a = [-0.5_f32, -1.0, -2.0, -0.1];
        let b = [1.0_f32, 0.5, 2.0, -1.0];
        let delta = 0.05_f32;
        for method in [
            Discretization::Zoh,
            Discretization::Bilinear,
            Discretization::Euler,
        ] {
            let (a_bar, b_bar) = discretize(&a, &b, delta, method).expect("valid input");
            for (i, (&ab, &bb)) in a_bar.iter().zip(b_bar.iter()).enumerate() {
                assert!(ab.is_finite(), "{method:?} a_bar[{i}]={ab} not finite");
                assert!(bb.is_finite(), "{method:?} b_bar[{i}]={bb} not finite");
            }
        }
    }

    /// Lengths of returned vecs match input length.
    #[test]
    fn output_length_matches_input() {
        let a = [-1.0_f32; 16];
        let b = [1.0_f32; 16];
        let (a_bar, b_bar) = discretize(&a, &b, 0.01, Discretization::Zoh).expect("valid input");
        assert_eq!(a_bar.len(), 16);
        assert_eq!(b_bar.len(), 16);
    }

    /// ZOH A_bar for negative A must lie strictly in (0, 1) (stable).
    #[test]
    fn zoh_a_bar_stable_range() {
        let a = [-0.1_f32, -1.0, -10.0];
        let b = [1.0_f32; 3];
        let (a_bar, _) = discretize(&a, &b, 1.0, Discretization::Zoh).expect("valid input");
        for (i, &ab) in a_bar.iter().enumerate() {
            assert!(ab > 0.0 && ab < 1.0, "zoh a_bar[{i}]={ab} outside (0,1)");
        }
    }
}
