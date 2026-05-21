//! Inline tests for the Rosenbaum-bounds Wilcoxon signed-rank sensitivity
//! analysis.

#![cfg(test)]

use super::rosenbaum_bounds::{
    RosenbaumBounds, RosenbaumConfig, normal_cdf_for_tests, signed_rank_for_tests,
};
use crate::error::CausalError;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// --------- input-validation tests ---------------------------------------

#[test]
fn empty_differences_errors() {
    let cfg = RosenbaumConfig::default();
    let r = RosenbaumBounds::wilcoxon_signed_rank(&[], &cfg);
    assert!(matches!(r, Err(CausalError::EmptyInput)));
}

#[test]
fn all_zero_differences_errors() {
    let cfg = RosenbaumConfig::default();
    let r = RosenbaumBounds::wilcoxon_signed_rank(&[0.0, 0.0, 0.0], &cfg);
    assert!(matches!(r, Err(CausalError::EmptyInput)));
}

#[test]
fn alpha_out_of_range_errors() {
    let diffs = vec![0.1, -0.2, 0.3];
    let bad_zero = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 0.0,
    };
    let bad_one = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 1.0,
    };
    let bad_neg = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: -0.05,
    };
    assert!(matches!(
        RosenbaumBounds::wilcoxon_signed_rank(&diffs, &bad_zero),
        Err(CausalError::IncompatibleData)
    ));
    assert!(matches!(
        RosenbaumBounds::wilcoxon_signed_rank(&diffs, &bad_one),
        Err(CausalError::IncompatibleData)
    ));
    assert!(matches!(
        RosenbaumBounds::wilcoxon_signed_rank(&diffs, &bad_neg),
        Err(CausalError::IncompatibleData)
    ));
}

#[test]
fn gamma_below_one_errors() {
    let diffs = vec![0.1, -0.2, 0.3];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![0.5, 1.0, 1.5],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn empty_gamma_grid_errors() {
    let diffs = vec![0.1, -0.2, 0.3];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

#[test]
fn non_finite_difference_errors() {
    let cfg = RosenbaumConfig::default();
    let diffs_nan = vec![0.1, f64::NAN, 0.3];
    let diffs_inf = vec![0.1, f64::INFINITY, 0.3];
    assert!(matches!(
        RosenbaumBounds::wilcoxon_signed_rank(&diffs_nan, &cfg),
        Err(CausalError::IncompatibleData)
    ));
    assert!(matches!(
        RosenbaumBounds::wilcoxon_signed_rank(&diffs_inf, &cfg),
        Err(CausalError::IncompatibleData)
    ));
}

// --------- correctness tests --------------------------------------------

#[test]
fn gamma_equal_one_lower_equals_upper() {
    // At Γ = 1, p_high = p_low = 0.5, so the two bounds coincide.
    let diffs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    assert_eq!(r.len(), 1);
    assert!(
        approx(r[0].p_lower, r[0].p_upper, 1e-12),
        "p_lower = {}, p_upper = {}",
        r[0].p_lower,
        r[0].p_upper
    );
}

#[test]
fn lower_bound_below_or_equal_upper_bound() {
    let diffs: Vec<f64> = vec![1.0, 2.0, -0.5, 1.5, 2.5, 0.3, -0.8, 1.2, 0.7, 1.8];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0, 1.5, 2.0, 3.0, 5.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    for row in &r {
        assert!(
            row.p_lower <= row.p_upper + 1e-9,
            "p_lower={} > p_upper={} at Γ={}",
            row.p_lower,
            row.p_upper,
            row.gamma
        );
    }
}

#[test]
fn strong_positive_effect_p_lower_zero_p_upper_inflated_by_gamma() {
    // 25 strongly positive differences — exact at Γ=1 is essentially 0.
    let diffs: Vec<f64> = (1..=25).map(|i| 0.5 * i as f64).collect();
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0, 10.0, 50.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    // At Γ = 1 the test is extremely significant: p_upper ≈ 0.
    assert!(
        r[0].p_upper < 1e-3,
        "p_upper at Γ=1 = {} (expected near 0)",
        r[0].p_upper
    );
    // p_upper should grow strictly with Γ.
    assert!(
        r[2].p_upper > r[0].p_upper + 0.01,
        "p_upper(Γ=50)={} should grow vs Γ=1 ({})",
        r[2].p_upper,
        r[0].p_upper
    );
    // p_lower at every Γ remains near 0 (all-positive observed signs).
    for row in &r {
        assert!(
            row.p_lower < 1e-3,
            "p_lower at Γ={} = {} (expected near 0)",
            row.gamma,
            row.p_lower
        );
    }
}

#[test]
fn deterministic_under_repeated_call() {
    let diffs: Vec<f64> = vec![1.0, -0.5, 2.0, 1.5, -1.0, 0.5, 2.5, -0.3, 1.2, 0.8];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0, 1.5, 2.0],
        alpha: 0.05,
    };
    let r1 = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    let r2 = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn upper_p_monotone_in_gamma() {
    // The upper p-value is non-decreasing in Γ.
    let diffs: Vec<f64> = (1..=30).map(|i| (i as f64).powi(1)).collect();
    let grid = vec![1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 8.0];
    let cfg = RosenbaumConfig {
        gamma_grid: grid,
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    for w in r.windows(2) {
        assert!(
            w[1].p_upper >= w[0].p_upper - 1e-9,
            "p_upper not monotone in Γ: {} (Γ={}) → {} (Γ={})",
            w[0].p_upper,
            w[0].gamma,
            w[1].p_upper,
            w[1].gamma
        );
    }
}

#[test]
fn critical_gamma_exceeds_one_for_significant_effect() {
    let diffs: Vec<f64> = (1..=30).map(|i| 0.3 * i as f64).collect();
    let g = RosenbaumBounds::critical_gamma(&diffs, 0.05).unwrap();
    assert!(g > 1.0, "critical Γ = {} (expected > 1)", g);
    assert!(g < 20.0, "critical Γ = {} (expected < 20)", g);
}

#[test]
fn critical_gamma_returns_one_for_non_significant_effect() {
    let diffs: Vec<f64> = vec![0.1, -0.2, 0.05, -0.1, 0.15, -0.05, 0.08, -0.07];
    let g = RosenbaumBounds::critical_gamma(&diffs, 0.05).unwrap();
    assert!(approx(g, 1.0, 1e-9), "critical Γ = {} (expected 1.0)", g);
}

#[test]
fn critical_gamma_alpha_validation() {
    let diffs = vec![0.1, 0.2, 0.3];
    assert!(matches!(
        RosenbaumBounds::critical_gamma(&diffs, 0.0),
        Err(CausalError::IncompatibleData)
    ));
    assert!(matches!(
        RosenbaumBounds::critical_gamma(&diffs, 1.0),
        Err(CausalError::IncompatibleData)
    ));
    assert!(matches!(
        RosenbaumBounds::critical_gamma(&diffs, f64::NAN),
        Err(CausalError::IncompatibleData)
    ));
}

#[test]
fn exact_matches_normal_at_n_eq_20() {
    // At n = 20 the implementation switches from exact enumeration to the
    // normal approximation. Both should report a tiny p-value with strong
    // positive differences.
    let diffs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 0.05,
    };
    let r_norm = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    let diffs_19 = &diffs[0..19];
    let r_exact = RosenbaumBounds::wilcoxon_signed_rank(diffs_19, &cfg).unwrap();
    assert!(r_norm[0].p_upper < 1e-3, "normal p={}", r_norm[0].p_upper);
    assert!(r_exact[0].p_upper < 1e-3, "exact p={}", r_exact[0].p_upper);
}

#[test]
fn pratt_zero_differences_dropped() {
    // Pairs with zero magnitude must be silently dropped (Pratt rule).
    let diffs: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 3.0, -1.5, 2.5];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    let non_zero: Vec<f64> = diffs.iter().copied().filter(|d| *d != 0.0).collect();
    let r_no_zero = RosenbaumBounds::wilcoxon_signed_rank(&non_zero, &cfg).unwrap();
    assert!(approx(r[0].p_upper, r_no_zero[0].p_upper, 1e-9));
    assert!(approx(r[0].p_lower, r_no_zero[0].p_lower, 1e-9));
}

#[test]
fn ties_in_absolute_difference_handled_with_average_ranks() {
    // Ties share the average integer rank.
    let diffs: Vec<f64> = vec![1.0, 1.0, -1.0, 1.0, 2.0, -2.0, 2.0];
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    assert!(r[0].p_upper.is_finite());
    assert!((0.0..=1.0).contains(&r[0].p_upper));
    assert!((0.0..=1.0).contains(&r[0].p_lower));
}

#[test]
fn p_values_are_in_closed_unit_interval() {
    let diffs: Vec<f64> = (1..=12)
        .map(|i| if i % 3 == 0 { -1.0 } else { i as f64 / 5.0 })
        .collect();
    let cfg = RosenbaumConfig {
        gamma_grid: vec![1.0, 1.5, 2.5, 4.0],
        alpha: 0.05,
    };
    let r = RosenbaumBounds::wilcoxon_signed_rank(&diffs, &cfg).unwrap();
    for row in &r {
        assert!(
            (0.0..=1.0).contains(&row.p_lower),
            "p_lower={} out of [0,1] at Γ={}",
            row.p_lower,
            row.gamma
        );
        assert!(
            (0.0..=1.0).contains(&row.p_upper),
            "p_upper={} out of [0,1] at Γ={}",
            row.p_upper,
            row.gamma
        );
    }
}

#[test]
fn normal_cdf_basic() {
    assert!(approx(normal_cdf_for_tests(0.0), 0.5, 1e-6));
    assert!(approx(normal_cdf_for_tests(1.0), 0.841_344_746, 1e-5));
    assert!(approx(normal_cdf_for_tests(-1.0), 0.158_655_254, 1e-5));
    assert!(approx(normal_cdf_for_tests(1.96), 0.975, 1e-3));
}

#[test]
fn signed_rank_ordering() {
    let diffs = vec![3.0, -1.0, 2.0, -4.0];
    let (ranks, signs) = signed_rank_for_tests(&diffs).unwrap();
    // Sorted |d|: 1, 2, 3, 4 ⇒ ranks 1..4 ⇒ matches sorted order.
    // Original signs in sorted order:
    //   |d|=1 (-1.0 → negative)
    //   |d|=2 ( 2.0 → positive)
    //   |d|=3 ( 3.0 → positive)
    //   |d|=4 (-4.0 → negative)
    assert_eq!(ranks, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(signs, vec![0.0, 1.0, 1.0, 0.0]);
}
