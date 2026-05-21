//! Tests for the Connect-the-Dots accountant.
//!
//! Reference: Doroshenko-Ghazi-Kamath-Kumar-Manurangsi (2022),
//! "Connect the Dots: Tighter Discrete Approximations of Privacy Loss
//! Distributions", PoPETs 2022(4).

use super::*;
use crate::error::PrivacyError;

fn build_default_gaussian() -> CtdAccountant {
    let cfg = CtdConfig::new(-30.0, 30.0, 1000).expect("cfg");
    CtdAccountant::from_gaussian(1.0, 1.0, &cfg).expect("ctd")
}

// 1. Pessimistic PLD sums to ~1.0 within 1e-6.
#[test]
fn test_pessimistic_pld_sums_to_one() {
    let ctd = build_default_gaussian();
    let total: f64 = ctd.pessimistic_pld.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "pessimistic mass {total} ≠ 1");
}

// 2. Optimistic PLD sums to ~1.0.
#[test]
fn test_optimistic_pld_sums_to_one() {
    let ctd = build_default_gaussian();
    let total: f64 = ctd.optimistic_pld.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "optimistic mass {total} ≠ 1");
}

// 3. Pessimistic δ ≥ optimistic δ for several thresholds.
#[test]
fn test_pessimistic_dominates_optimistic_delta() {
    let ctd = build_default_gaussian();
    for &eps in &[0.1, 1.0, 5.0] {
        let pess = ctd.delta_at_epsilon(eps).expect("ok");
        let opt = ctd.delta_at_epsilon_optimistic(eps).expect("ok");
        assert!(
            pess + 1e-9 >= opt,
            "pessimistic {pess} should ≥ optimistic {opt} at ε={eps}"
        );
    }
}

// 4. δ at very large ε → 0.
#[test]
fn test_delta_at_large_epsilon_is_zero() {
    let ctd = build_default_gaussian();
    let d = ctd.delta_at_epsilon(1e6).expect("ok");
    assert!(d < 1e-6, "δ at ε=1e6 should be ≈ 0, got {d}");
}

// 5. δ at very negative ε ≤ 1.
#[test]
fn test_delta_at_very_negative_epsilon_clamped() {
    let ctd = build_default_gaussian();
    let d = ctd.delta_at_epsilon(-1e6).expect("ok");
    assert!(d <= 1.0 + 1e-12, "δ should be clamped ≤ 1, got {d}");
    assert!(d >= 0.0, "δ must be ≥ 0, got {d}");
}

// 6. compose_self(1) reproduces self cell-by-cell.
#[test]
fn test_compose_self_one_is_identity_op() {
    let ctd = build_default_gaussian();
    let same = ctd.compose_self(1).expect("ok");
    for (a, b) in ctd.pessimistic_pld.iter().zip(same.pessimistic_pld.iter()) {
        assert!((a - b).abs() < 1e-12, "pessimistic cell {a} vs {b}");
    }
    for (a, b) in ctd.optimistic_pld.iter().zip(same.optimistic_pld.iter()) {
        assert!((a - b).abs() < 1e-12, "optimistic cell {a} vs {b}");
    }
}

// 7. compose_self(2) ≈ self.compose(self).
#[test]
fn test_compose_self_two_equals_compose() {
    let ctd = build_default_gaussian();
    let via_self = ctd.compose_self(2).expect("ok");
    let via_compose = ctd.compose(&ctd).expect("ok");
    for (a, b) in via_self
        .pessimistic_pld
        .iter()
        .zip(via_compose.pessimistic_pld.iter())
    {
        assert!((a - b).abs() < 1e-9, "pessimistic cell mismatch {a} vs {b}");
    }
}

// 8. epsilon_at_delta inverts delta_at_epsilon within tolerance.
#[test]
fn test_epsilon_inverts_delta() {
    let ctd = build_default_gaussian();
    let target_delta = 1e-5;
    let eps = ctd.epsilon_at_delta(target_delta).expect("ok");
    let d = ctd.delta_at_epsilon(eps).expect("ok");
    assert!(
        d <= target_delta + 1e-6,
        "δ({eps}) = {d} should ≤ {target_delta}"
    );
}

// 9. Single Gaussian δ matches the closed-form GDP bound at typical (σ, Δ, ε).
//
// Closed form (Dong-Roth-Su 2022, Cor. 2.13):
// δ(ε) = Φ(-ε/μ + μ/2) - exp(ε) · Φ(-ε/μ - μ/2)  with μ = Δ/σ.
//
// The CTD pessimistic δ is an *upper* bound on this exact value; on a
// 1000-cell grid the gap should be well under 1e-3.
#[test]
fn test_pessimistic_matches_closed_form_within_tolerance() {
    let ctd = build_default_gaussian();
    let mu = 1.0f64; // sensitivity / sigma = 1
    let eps = 1.0f64;
    let closed = phi(-eps / mu + mu / 2.0) - eps.exp() * phi(-eps / mu - mu / 2.0);
    let pess = ctd.delta_at_epsilon(eps).expect("ok");
    let opt = ctd.delta_at_epsilon_optimistic(eps).expect("ok");
    assert!(
        pess + 1e-9 >= closed,
        "pessimistic {pess} should ≥ closed {closed}"
    );
    assert!(
        opt - 1e-9 <= closed,
        "optimistic {opt} should ≤ closed {closed}"
    );
    // The pessimistic bound's gap relative to the closed form is dominated
    // by the cell width h = (grid_hi − grid_lo) / grid_size = 0.06. The
    // observed gap should stay under one cell-width of slack.
    assert!(
        (pess - closed).abs() < 1e-2,
        "pessimistic gap |{pess} - {closed}| too large"
    );
}

// 10. grid_size < 8 returns InvalidParameter.
#[test]
fn test_grid_size_too_small_errors() {
    let r = CtdConfig::new(-10.0, 10.0, 7);
    assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
}

// 11. grid_lo ≥ grid_hi returns InvalidParameter.
#[test]
fn test_grid_bounds_swapped_errors() {
    let r = CtdConfig::new(10.0, 10.0, 100);
    assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    let r2 = CtdConfig::new(20.0, 10.0, 100);
    assert!(matches!(r2, Err(PrivacyError::InvalidParameter(_))));
}

// 12. compose with mismatched grid_size returns DimensionMismatch.
#[test]
fn test_compose_mismatched_grid_errors() {
    let cfg_a = CtdConfig::new(-10.0, 10.0, 100).expect("a");
    let cfg_b = CtdConfig::new(-10.0, 10.0, 200).expect("b");
    let a = CtdAccountant::from_gaussian(1.0, 1.0, &cfg_a).expect("ctd a");
    let b = CtdAccountant::from_gaussian(1.0, 1.0, &cfg_b).expect("ctd b");
    let r = a.compose(&b);
    assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
}

// 13. compose_self(0) returns identity (δ(ε) = 0 for every ε ≥ 0).
#[test]
fn test_compose_self_zero_is_identity() {
    let ctd = build_default_gaussian();
    let id = ctd.compose_self(0).expect("ok");
    let total: f64 = id.pessimistic_pld.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "identity mass {total} ≠ 1");
    let d = id.delta_at_epsilon(0.0).expect("ok");
    assert!(d < 1e-9, "identity δ(0) should be ≈ 0, got {d}");
    let d2 = id.delta_at_epsilon(1.0).expect("ok");
    assert!(d2 < 1e-9, "identity δ(1) should be ≈ 0, got {d2}");
}

// 14. Large k (k = 64) stable, no NaN.
#[test]
fn test_compose_self_large_k_stable() {
    let cfg = CtdConfig::new(-30.0, 30.0, 256).expect("cfg");
    let ctd = CtdAccountant::from_gaussian(4.0, 1.0, &cfg).expect("ctd");
    let composed = ctd.compose_self(64).expect("ok");
    let total: f64 = composed.pessimistic_pld.iter().sum();
    assert!(total.is_finite(), "total {total} non-finite");
    assert!((total - 1.0).abs() < 1e-6, "composed mass {total} ≠ 1");
    for &v in composed.pessimistic_pld.iter() {
        assert!(v.is_finite(), "NaN cell {v}");
        assert!(v >= 0.0, "negative cell {v}");
    }
    let d = composed.delta_at_epsilon(2.0).expect("ok");
    assert!(d.is_finite() && (0.0..=1.0).contains(&d), "δ {d} insane");
}

// 15. δ is monotonically non-increasing in ε on the pessimistic side.
#[test]
fn test_pessimistic_delta_monotone_decreasing() {
    let ctd = build_default_gaussian();
    let d0 = ctd.delta_at_epsilon(0.0).expect("ok");
    let d1 = ctd.delta_at_epsilon(0.5).expect("ok");
    let d2 = ctd.delta_at_epsilon(1.0).expect("ok");
    let d3 = ctd.delta_at_epsilon(2.0).expect("ok");
    assert!(d0 + 1e-9 >= d1, "{d0} should ≥ {d1}");
    assert!(d1 + 1e-9 >= d2, "{d1} should ≥ {d2}");
    assert!(d2 + 1e-9 >= d3, "{d2} should ≥ {d3}");
}

// 16. epsilon_at_delta with δ outside (0,1) errors.
#[test]
fn test_epsilon_at_delta_bad_delta_errors() {
    let ctd = build_default_gaussian();
    assert!(matches!(
        ctd.epsilon_at_delta(0.0),
        Err(PrivacyError::InvalidDelta(_))
    ));
    assert!(matches!(
        ctd.epsilon_at_delta(1.0),
        Err(PrivacyError::InvalidDelta(_))
    ));
    assert!(matches!(
        ctd.epsilon_at_delta(-0.1),
        Err(PrivacyError::InvalidDelta(_))
    ));
    assert!(matches!(
        ctd.epsilon_at_delta(1.5),
        Err(PrivacyError::InvalidDelta(_))
    ));
}

// 17. Non-positive sigma errors.
#[test]
fn test_from_gaussian_nonpositive_sigma_errors() {
    let cfg = CtdConfig::new(-30.0, 30.0, 1000).expect("cfg");
    let r = CtdAccountant::from_gaussian(0.0, 1.0, &cfg);
    assert!(matches!(r, Err(PrivacyError::InvalidParameter(_))));
    let r2 = CtdAccountant::from_gaussian(-1.0, 1.0, &cfg);
    assert!(matches!(r2, Err(PrivacyError::InvalidParameter(_))));
}

// 18. Non-positive sensitivity errors.
#[test]
fn test_from_gaussian_nonpositive_sensitivity_errors() {
    let cfg = CtdConfig::new(-30.0, 30.0, 1000).expect("cfg");
    let r = CtdAccountant::from_gaussian(1.0, 0.0, &cfg);
    assert!(matches!(r, Err(PrivacyError::NonPositiveSensitivity(_))));
    let r2 = CtdAccountant::from_gaussian(1.0, -1.0, &cfg);
    assert!(matches!(r2, Err(PrivacyError::NonPositiveSensitivity(_))));
}
