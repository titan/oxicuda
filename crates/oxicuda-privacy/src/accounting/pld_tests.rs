use super::*;

fn small_grid() -> PldGrid {
    PldGrid::new(-1.0, 1.0, 0.1).expect("grid")
}

fn wide_grid() -> PldGrid {
    PldGrid::new(-10.0, 10.0, 0.05).expect("grid")
}

/// Analytical Gaussian δ(ε) for sensitivity Δ, noise σ (Balle-Wang 2018):
/// δ = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ).
fn analytical_gaussian_delta(sensitivity: f64, sigma: f64, epsilon: f64) -> f64 {
    let mu = sensitivity / sigma;
    let term1 = phi(mu / 2.0 - epsilon / mu);
    let term2 = epsilon.exp() * phi(-mu / 2.0 - epsilon / mu);
    (term1 - term2).max(0.0)
}

#[test]
fn test_from_gaussian_total_mass_within_tolerance() {
    let grid = wide_grid();
    let pld = Pld::from_gaussian(1.0, 1.0, grid).expect("pld");
    let total = pld.total_mass();
    assert!(
        (total - 1.0).abs() < 1e-3,
        "total mass {total} should be ≈ 1"
    );
}

#[test]
fn test_from_gaussian_zero_sigma_errors() {
    let grid = small_grid();
    assert!(Pld::from_gaussian(1.0, 0.0, grid).is_err());
}

#[test]
fn test_from_gaussian_non_positive_sensitivity_errors() {
    let grid = small_grid();
    assert!(Pld::from_gaussian(0.0, 1.0, grid.clone()).is_err());
    assert!(Pld::from_gaussian(-2.0, 1.0, grid).is_err());
}

#[test]
fn test_from_histogram_length_mismatch_errors() {
    let grid = small_grid();
    let n = grid.len();
    let probabilities = vec![0.0f64; n + 1];
    assert!(Pld::from_histogram(grid, probabilities, 0.0).is_err());
}

#[test]
fn test_from_histogram_negative_probabilities_error() {
    let grid = small_grid();
    let n = grid.len();
    let mut probabilities = vec![0.0f64; n];
    probabilities[0] = -0.1;
    assert!(Pld::from_histogram(grid, probabilities, 0.0).is_err());
}

#[test]
fn test_compose_mismatched_step_errors() {
    let grid_a = PldGrid::new(-1.0, 1.0, 0.1).expect("a");
    let grid_b = PldGrid::new(-1.0, 1.0, 0.2).expect("b");
    let na = grid_a.len();
    let nb = grid_b.len();
    let mut prob_a = vec![0.0f64; na];
    prob_a[na / 2] = 1.0;
    let mut prob_b = vec![0.0f64; nb];
    prob_b[nb / 2] = 1.0;
    let pld_a = Pld::from_histogram(grid_a, prob_a, 0.0).expect("a pld");
    let pld_b = Pld::from_histogram(grid_b, prob_b, 0.0).expect("b pld");
    assert!(pld_a.compose(&pld_b).is_err());
}

#[test]
fn test_compose_self_zero_is_identity() {
    let grid = PldGrid::new(-1.0, 1.0, 0.1).expect("g");
    let pld = Pld::from_gaussian(1.0, 1.0, grid).expect("p");
    let composed = pld.compose_self(0).expect("c");
    // Identity ⇒ delta_for_epsilon(0) = 0, epsilon_for_delta(0.5) = 0.
    assert!(composed.delta_for_epsilon(0.0) < 1e-9);
    assert!((composed.epsilon_for_delta(0.5) - 0.0).abs() < 1e-9);
    assert!((composed.epsilon_for_delta(1.0) - 0.0).abs() < 1e-9);
}

#[test]
fn test_compose_self_one_matches_self() {
    let grid = PldGrid::new(-2.0, 2.0, 0.05).expect("g");
    let pld = Pld::from_gaussian(1.0, 1.5, grid).expect("p");
    let composed = pld.compose_self(1).expect("c");
    assert_eq!(composed.probabilities().len(), pld.probabilities().len());
    let mut max_diff = 0.0f64;
    for (a, b) in composed
        .probabilities()
        .iter()
        .zip(pld.probabilities().iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff < 1e-12, "max diff {max_diff}");
}

#[test]
fn test_compose_self_two_matches_pairwise_compose() {
    let grid = PldGrid::new(-3.0, 3.0, 0.1).expect("g");
    let pld = Pld::from_gaussian(1.0, 1.5, grid).expect("p");
    let via_self = pld.compose_self(2).expect("c");
    let via_pair = pld.compose(&pld).expect("c2");
    assert_eq!(
        via_self.probabilities().len(),
        via_pair.probabilities().len()
    );
    let mut max_diff = 0.0f64;
    for (a, b) in via_self
        .probabilities()
        .iter()
        .zip(via_pair.probabilities().iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff < 1e-12, "max diff {max_diff}");
    assert!(
        (via_self.truncation_mass() - via_pair.truncation_mass()).abs() < 1e-12,
        "truncation diff"
    );
}

#[test]
fn test_delta_for_epsilon_monotone_in_epsilon() {
    let pld = Pld::from_gaussian(1.0, 1.0, wide_grid()).expect("p");
    let mut prev = pld.delta_for_epsilon(0.0);
    for k in 1..50 {
        let eps = k as f64 * 0.2;
        let d = pld.delta_for_epsilon(eps);
        assert!(
            d <= prev + 1e-12,
            "δ should be monotone non-increasing: {prev} vs {d} at ε={eps}"
        );
        prev = d;
    }
}

#[test]
fn test_epsilon_for_delta_monotone_in_delta() {
    let pld = Pld::from_gaussian(1.0, 1.0, wide_grid()).expect("p");
    let mut prev = pld.epsilon_for_delta(1e-8);
    for k in 1..20 {
        let delta = 1e-8 * (10f64.powi(k));
        if delta >= 1.0 {
            break;
        }
        let eps = pld.epsilon_for_delta(delta);
        assert!(
            eps <= prev + 1e-9,
            "ε should be non-increasing in δ: {prev} vs {eps} at δ={delta}"
        );
        prev = eps;
    }
}

#[test]
fn test_gaussian_pld_matches_analytical_delta() {
    let grid = PldGrid::new(-15.0, 15.0, 0.01).expect("g");
    let sensitivity = 1.0;
    let sigma = 1.0;
    let pld = Pld::from_gaussian(sensitivity, sigma, grid).expect("p");
    let epsilon = 3.0;
    let numerical = pld.delta_for_epsilon(epsilon);
    let analytical = analytical_gaussian_delta(sensitivity, sigma, epsilon);
    let rel = (numerical - analytical).abs() / analytical.max(1e-12);
    assert!(
        rel < 0.05,
        "numerical δ={numerical}, analytical={analytical}, rel diff {rel}"
    );
}

#[test]
fn test_compose_total_mass_symmetric() {
    let grid = PldGrid::new(-2.0, 2.0, 0.1).expect("g");
    let pld_a = Pld::from_gaussian(1.0, 1.0, grid.clone()).expect("a");
    let pld_b = Pld::from_gaussian(1.0, 2.0, grid).expect("b");
    let ab = pld_a.compose(&pld_b).expect("ab");
    let ba = pld_b.compose(&pld_a).expect("ba");
    assert!(
        (ab.total_mass() - ba.total_mass()).abs() < 1e-12,
        "total mass should be symmetric"
    );
}

#[test]
fn test_compose_self_four_matches_squared_compose() {
    let grid = PldGrid::new(-2.0, 2.0, 0.1).expect("g");
    let pld = Pld::from_gaussian(1.0, 1.0, grid).expect("p");
    let four = pld.compose_self(4).expect("4");
    let two = pld.compose_self(2).expect("2");
    let two_two = two.compose(&two).expect("2x2");
    let mut total_diff = 0.0f64;
    for (a, b) in four
        .probabilities()
        .iter()
        .zip(two_two.probabilities().iter())
    {
        total_diff += (a - b).abs();
    }
    total_diff += (four.truncation_mass() - two_two.truncation_mass()).abs();
    assert!(total_diff < 1e-6, "sum |diff| = {total_diff}");
}

#[test]
fn test_empty_grid_errors_on_construction() {
    assert!(PldGrid::new(0.0, 0.0, 0.1).is_err());
    assert!(PldGrid::new(1.0, 0.0, 0.1).is_err());
    assert!(PldGrid::new(0.0, 1.0, 0.0).is_err());
    assert!(PldGrid::new(0.0, 1.0, -0.1).is_err());
}

#[test]
fn test_from_histogram_total_mass_check() {
    let grid = small_grid();
    let n = grid.len();
    let probabilities = vec![1.0f64; n];
    assert!(Pld::from_histogram(grid, probabilities, 0.5).is_err());
}
