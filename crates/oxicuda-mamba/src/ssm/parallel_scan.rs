//! CPU reference implementation of the associative `(A, b)` prefix scan.
//!
//! The Mamba selective-scan recurrence is:
//!
//! ```text
//! h[t] = A_bar[t] * h[t-1] + B_bar[t] * u[t]
//! ```
//!
//! This can be expressed as a sequence of `ScanPair { a, b }` elements where
//! the associative binary operator `⊕` is defined by:
//!
//! ```text
//! (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)
//! ```
//!
//! Under this operator, the inclusive prefix scan gives exactly the states
//! `h[t]` starting from `h[-1] = 0`.
//!
//! This module provides sequential CPU implementations as reference/test
//! baselines, without requiring GPU hardware.

use crate::error::{MambaError, MambaResult};

// ─── ScanPair ────────────────────────────────────────────────────────────────

/// A single `(a, b)` element in the associative prefix scan.
///
/// Represents one step of the SSM recurrence: the state after this step is
/// `a * h_prev + b` where `h_prev` is the accumulated state from earlier steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanPair {
    /// Multiplicative decay factor (A_bar for this timestep).
    pub a: f32,
    /// Additive input term (B_bar * u for this timestep).
    pub b: f32,
}

impl ScanPair {
    /// The associative binary operator:
    ///
    /// `(a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)`
    ///
    /// Represents applying `right` after `left` in the SSM recurrence.
    #[inline]
    pub fn combine(left: ScanPair, right: ScanPair) -> ScanPair {
        ScanPair {
            a: right.a * left.a,
            b: right.a * left.b + right.b,
        }
    }

    /// The identity element for `combine`: `identity ⊕ x = x ⊕ identity = x`.
    ///
    /// `(1, 0)` is the identity because:
    /// - `(1, 0) ⊕ (a, b) = (a * 1, a * 0 + b) = (a, b)` ✓
    /// - `(a, b) ⊕ (1, 0) = (1 * a, 1 * b + 0) = (a, b)` ✓
    #[inline]
    pub fn identity() -> ScanPair {
        ScanPair { a: 1.0, b: 0.0 }
    }
}

// ─── Inclusive prefix scan ───────────────────────────────────────────────────

/// Sequential inclusive prefix scan over `ScanPair` elements.
///
/// Returns `output[t] = pairs[0] ⊕ pairs[1] ⊕ ... ⊕ pairs[t]`.
///
/// With initial hidden state `h[-1] = 0`, the SSM state at time `t` is
/// `output[t].b`.
///
/// Empty input returns an empty `Vec`.
pub fn inclusive_scan(pairs: &[ScanPair]) -> Vec<ScanPair> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pairs.len());
    let mut acc = pairs[0];
    out.push(acc);
    for &p in &pairs[1..] {
        acc = ScanPair::combine(acc, p);
        out.push(acc);
    }
    out
}

// ─── Exclusive prefix scan ───────────────────────────────────────────────────

/// Sequential exclusive prefix scan over `ScanPair` elements.
///
/// Returns `output[t] = identity ⊕ pairs[0] ⊕ ... ⊕ pairs[t-1]`.
///
/// - `output[0]` is always the identity `(1, 0)`.
/// - `output[t]` encodes the accumulated state just *before* step `t`.
///
/// Empty input returns an empty `Vec`.
pub fn exclusive_scan(pairs: &[ScanPair]) -> Vec<ScanPair> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pairs.len());
    let mut acc = ScanPair::identity();
    for &p in pairs {
        out.push(acc);
        acc = ScanPair::combine(acc, p);
    }
    out
}

// ─── SSM state computation ───────────────────────────────────────────────────

/// Compute SSM hidden states via inclusive prefix scan.
///
/// Given the per-timestep discrete parameters:
/// - `a_bar[t]`: discretized A decay at time `t`
/// - `b_bar_u[t]`: `B_bar[t] * u[t]` (combined input contribution)
///
/// Computes `h[t] = a_bar[t] * h[t-1] + b_bar_u[t]` for all `t`, assuming
/// `h[-1] = 0`.
///
/// Returns a vector of length `L` containing `h[0], h[1], ..., h[L-1]`.
///
/// # Errors
///
/// * [`MambaError::EmptyInput`]       — if input is empty.
/// * [`MambaError::DimensionMismatch`] — if `a_bar` and `b_bar_u` have different lengths.
pub fn ssm_state_scan(a_bar: &[f32], b_bar_u: &[f32]) -> MambaResult<Vec<f32>> {
    if a_bar.is_empty() {
        return Err(MambaError::EmptyInput("a_bar"));
    }
    if a_bar.len() != b_bar_u.len() {
        return Err(MambaError::DimensionMismatch {
            expected: a_bar.len(),
            got: b_bar_u.len(),
        });
    }
    let pairs: Vec<ScanPair> = a_bar
        .iter()
        .zip(b_bar_u.iter())
        .map(|(&a, &b)| ScanPair { a, b })
        .collect();
    let scanned = inclusive_scan(&pairs);
    Ok(scanned.into_iter().map(|p| p.b).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScanPair::identity ────────────────────────────────────────────────────

    /// `identity ⊕ x = x` and `x ⊕ identity = x`.
    #[test]
    fn identity_combine() {
        let id = ScanPair::identity();
        let x = ScanPair { a: 0.5, b: 1.3 };
        assert_eq!(ScanPair::combine(id, x), x, "identity ⊕ x should equal x");
        assert_eq!(ScanPair::combine(x, id), x, "x ⊕ identity should equal x");
    }

    // ── ScanPair::combine associativity ──────────────────────────────────────

    /// `(a⊕b)⊕c == a⊕(b⊕c)` for deterministic test values.
    #[test]
    fn combine_associativity() {
        let p1 = ScanPair { a: 0.9, b: 0.2 };
        let p2 = ScanPair { a: 0.8, b: 0.5 };
        let p3 = ScanPair { a: 0.7, b: 1.1 };
        let left_first = ScanPair::combine(ScanPair::combine(p1, p2), p3);
        let right_first = ScanPair::combine(p1, ScanPair::combine(p2, p3));
        assert!(
            (left_first.a - right_first.a).abs() < 1e-5,
            "associativity: a mismatch"
        );
        assert!(
            (left_first.b - right_first.b).abs() < 1e-5,
            "associativity: b mismatch"
        );
    }

    // ── inclusive_scan ────────────────────────────────────────────────────────

    /// Single-element scan returns that element.
    #[test]
    fn inclusive_scan_single() {
        let p = ScanPair { a: 0.5, b: 2.0 };
        let result = inclusive_scan(&[p]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], p);
    }

    /// Two-element scan: `[(2,3), (4,5)]` → `[(2,3), (8,17)]`.
    ///
    /// Step 0: `(2, 3)` unchanged.
    /// Step 1: `(2,3) ⊕ (4,5) = (4*2, 4*3+5) = (8, 17)`.
    #[test]
    fn inclusive_scan_two_elements() {
        let pairs = [ScanPair { a: 2.0, b: 3.0 }, ScanPair { a: 4.0, b: 5.0 }];
        let result = inclusive_scan(&pairs);
        assert_eq!(result.len(), 2);
        assert!(
            (result[0].a - 2.0).abs() < 1e-6,
            "result[0].a={}",
            result[0].a
        );
        assert!(
            (result[0].b - 3.0).abs() < 1e-6,
            "result[0].b={}",
            result[0].b
        );
        assert!(
            (result[1].a - 8.0).abs() < 1e-6,
            "result[1].a={}",
            result[1].a
        );
        assert!(
            (result[1].b - 17.0).abs() < 1e-6,
            "result[1].b={}",
            result[1].b
        );
    }

    /// States remain bounded when all `a` ∈ [0, 1) and `b` are bounded.
    #[test]
    fn inclusive_scan_all_stable() {
        // a = 0.9 everywhere, b = 1.0 everywhere, L=50
        // Geometric series: state converges to b/(1-a) = 10.0
        let l = 50_usize;
        let pairs: Vec<ScanPair> = (0..l).map(|_| ScanPair { a: 0.9, b: 1.0 }).collect();
        let result = inclusive_scan(&pairs);
        assert_eq!(result.len(), l);
        for (t, r) in result.iter().enumerate() {
            assert!(r.b.is_finite(), "state at t={t} not finite: {}", r.b);
            // Converges to ≤ 1/(1-0.9) = 10 from below
            assert!(r.b <= 11.0, "state at t={t} exceeds bound: {}", r.b);
        }
    }

    /// Empty input returns empty output.
    #[test]
    fn inclusive_scan_empty() {
        assert!(inclusive_scan(&[]).is_empty());
    }

    // ── exclusive_scan ────────────────────────────────────────────────────────

    /// First element of exclusive scan is always the identity.
    #[test]
    fn exclusive_scan_first_is_identity() {
        let pairs = [
            ScanPair { a: 0.5, b: 1.0 },
            ScanPair { a: 0.8, b: 2.0 },
            ScanPair { a: 0.3, b: 0.5 },
        ];
        let result = exclusive_scan(&pairs);
        let id = ScanPair::identity();
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0], id,
            "first element of exclusive scan must be identity"
        );
    }

    /// Exclusive scan agrees with shifted inclusive scan.
    #[test]
    fn exclusive_equals_shifted_inclusive() {
        let pairs: Vec<ScanPair> = (0..5)
            .map(|i| ScanPair {
                a: 0.7 + i as f32 * 0.05,
                b: i as f32 * 0.3,
            })
            .collect();
        let inc = inclusive_scan(&pairs);
        let exc = exclusive_scan(&pairs);
        assert_eq!(exc[0], ScanPair::identity());
        for i in 1..pairs.len() {
            assert!(
                (exc[i].a - inc[i - 1].a).abs() < 1e-5,
                "exc[{i}].a={} inc[{}].a={}",
                exc[i].a,
                i - 1,
                inc[i - 1].a
            );
            assert!(
                (exc[i].b - inc[i - 1].b).abs() < 1e-5,
                "exc[{i}].b={} inc[{}].b={}",
                exc[i].b,
                i - 1,
                inc[i - 1].b
            );
        }
    }

    // ── ssm_state_scan ────────────────────────────────────────────────────────

    /// Output shape matches input length.
    #[test]
    fn ssm_state_scan_shape() {
        let l = 7_usize;
        let a_bar = vec![0.9_f32; l];
        let b_bar_u = vec![1.0_f32; l];
        let h = ssm_state_scan(&a_bar, &b_bar_u).expect("valid input");
        assert_eq!(h.len(), l);
    }

    /// All `a_bar = 0` means state resets each step: `h[t] = b_bar_u[t]`.
    #[test]
    fn ssm_state_scan_constant_a_zero() {
        let b_bar_u = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let a_bar = vec![0.0_f32; 5];
        let h = ssm_state_scan(&a_bar, &b_bar_u).expect("valid input");
        for (t, (&expected, &got)) in b_bar_u.iter().zip(h.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "a=0 h[{t}]={got} expected {expected}"
            );
        }
    }

    /// All `a_bar = 1` means state is cumulative sum of `b_bar_u`.
    #[test]
    fn ssm_state_scan_constant_a_one() {
        let b_bar_u = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a_bar = vec![1.0_f32; 4];
        let h = ssm_state_scan(&a_bar, &b_bar_u).expect("valid input");
        // h[0]=1, h[1]=3, h[2]=6, h[3]=10
        let expected = [1.0_f32, 3.0, 6.0, 10.0];
        for (t, (&exp, &got)) in expected.iter().zip(h.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "cumsum h[{t}]={got} expected {exp}"
            );
        }
    }

    /// Error: empty a_bar.
    #[test]
    fn ssm_state_scan_error_empty() {
        let err = ssm_state_scan(&[], &[]).expect_err("should fail on empty");
        assert!(matches!(err, MambaError::EmptyInput(_)));
    }

    /// Error: length mismatch between a_bar and b_bar_u.
    #[test]
    fn ssm_state_scan_error_length_mismatch() {
        let err = ssm_state_scan(&[0.9, 0.9, 0.9], &[1.0, 2.0]).expect_err("should fail");
        assert!(
            matches!(
                err,
                MambaError::DimensionMismatch {
                    expected: 3,
                    got: 2
                }
            ),
            "unexpected error: {err}"
        );
    }

    /// L=1024 produces finite output.
    #[test]
    fn scan_large_l_finite() {
        let l = 1024_usize;
        let a_bar = vec![0.95_f32; l];
        let b_bar_u = vec![0.1_f32; l];
        let h = ssm_state_scan(&a_bar, &b_bar_u).expect("valid input");
        assert_eq!(h.len(), l);
        for (t, &v) in h.iter().enumerate() {
            assert!(v.is_finite(), "h[{t}]={v} not finite for large L");
        }
        // Geometric series converges to 0.1/(1-0.95) = 2.0
        // Final state should be close to limit
        assert!(
            h[l - 1] < 2.1,
            "state should converge near limit 2.0, got {}",
            h[l - 1]
        );
    }

    /// Manual recurrence check for 3 steps.
    #[test]
    fn ssm_state_scan_manual_recurrence() {
        // h[-1] = 0
        // h[0] = 0.5 * 0 + 1.0 = 1.0
        // h[1] = 0.5 * 1.0 + 2.0 = 2.5
        // h[2] = 0.5 * 2.5 + 0.5 = 1.75
        let a_bar = [0.5_f32, 0.5, 0.5];
        let b_bar_u = [1.0_f32, 2.0, 0.5];
        let h = ssm_state_scan(&a_bar, &b_bar_u).expect("valid input");
        assert!((h[0] - 1.0).abs() < 1e-6, "h[0]={}", h[0]);
        assert!((h[1] - 2.5).abs() < 1e-6, "h[1]={}", h[1]);
        assert!((h[2] - 1.75).abs() < 1e-6, "h[2]={}", h[2]);
    }
}
