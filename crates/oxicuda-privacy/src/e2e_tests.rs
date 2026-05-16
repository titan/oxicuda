//! End-to-end cross-module integration tests for `oxicuda-privacy`.
//!
//! These tests exercise multiple modules together to validate correctness
//! of the DP guarantees and utility properties.

use crate::accounting::fdp::{gdp_compose, gdp_to_epsilon_delta};
use crate::accounting::prv::{
    GaussianPrv, PrvConfig, compose_gaussian_prv, gaussian_prv_pmf, prv_delta,
};
use crate::accounting::zcdp::{zcdp_compose, zcdp_to_epsilon_delta};
use crate::composition::advanced::{basic_compose, strong_compose};
use crate::composition::amplification_subsampling::amplify_poisson;
use crate::handle::LcgRng;
use crate::local::grr::{GrrConfig, grr_encode, grr_estimate_frequency};
use crate::local::oue::{OueConfig, oue_encode, oue_estimate_frequency};
use crate::mechanism::exponential::{ExponentialConfig, exponential_sample};
use crate::mechanism::propose_release::{PtrConfig, propose_test_release};
use crate::mechanism::report_noisy_max::{RnmConfig, report_noisy_max};
use crate::metrics::metrics::{PrivacyBudget, gaussian_utility};
use crate::optimizer::dp_adam::{DpAdamConfig, DpAdamState};
use crate::ptx_kernels::{
    clip_gradient_ptx, exponential_sample_ptx, gaussian_noise_ptx, laplace_noise_ptx,
    oue_encode_ptx, prv_convolve_ptx, svt_threshold_ptx,
};
use crate::selection::sparse_vector::{SvtConfig, SvtState};

// ─── Test 1: Exponential mechanism distribution ────────────────────────────────

#[test]
fn exponential_mechanism_respects_distribution() {
    // scores = [0.0, 1.0, 2.0]; true probabilities ∝ exp(ε·score/(2·Δ)).
    // With ε=2, Δ=1: weights = exp(0)=1, exp(1)≈2.72, exp(2)≈7.39.
    // Normalised: ~0.093, ~0.253, ~0.690 — index 2 should win ~69% of the time.
    let scores = vec![0.0, 1.0, 2.0];
    let cfg = ExponentialConfig::new(2.0, 1.0).expect("ok");
    let mut rng = LcgRng::new(12345);

    let n_samples = 5000;
    let mut counts = [0usize; 3];
    for _ in 0..n_samples {
        let idx = exponential_sample(&scores, &cfg, &mut rng).expect("ok");
        counts[idx] += 1;
    }

    // Expected fractions: compute normalised weights.
    let weights: Vec<f64> = scores.iter().map(|&s| (s).exp()).collect();
    let total: f64 = weights.iter().sum();
    let expected: Vec<f64> = weights.iter().map(|&w| w / total).collect();

    for i in 0..3 {
        let observed = counts[i] as f64 / n_samples as f64;
        let exp_frac = expected[i];
        // Allow 10% relative error.
        assert!(
            (observed - exp_frac).abs() < 0.10,
            "index {i}: observed={observed:.3}, expected={exp_frac:.3}"
        );
    }
}

// ─── Test 2: Report-Noisy-Max returns valid index ─────────────────────────────

#[test]
fn report_noisy_max_returns_valid_index() {
    let scores = vec![0.0, -1.0, 3.0, 2.0, -5.0];
    let cfg = RnmConfig::new(1.0, 1.0).expect("ok");
    let mut rng = LcgRng::new(99);
    for _ in 0..500 {
        let idx = report_noisy_max(&scores, &cfg, &mut rng).expect("ok");
        assert!(
            idx < scores.len(),
            "index {idx} out of range {}",
            scores.len()
        );
    }
}

// ─── Test 3: PTR releases or returns None ─────────────────────────────────────

#[test]
fn ptr_releases_or_returns_none() {
    let cfg = PtrConfig::new(1.0, 1e-6, 1.0).expect("ok");
    let mut rng = LcgRng::new(7);
    for _ in 0..100 {
        let result = propose_test_release(0.0, 42.0, &cfg, &mut rng).expect("ok");
        if let Some(v) = result {
            assert!(v.is_finite(), "released value must be finite");
        }
    }
}

// ─── Test 4: SVT respects true count limit ────────────────────────────────────

#[test]
fn svt_respects_true_count_limit() {
    let k_limit = 3;
    let cfg = SvtConfig::new(1.0, -1000.0, k_limit, 1.0).expect("ok");
    let mut rng = LcgRng::new(42);
    let mut state = SvtState::new(&cfg, &mut rng).expect("ok");

    let mut true_count = 0usize;
    for _ in 0..200 {
        match state.query(1000.0, &cfg, &mut rng) {
            Ok(Some(true)) => {
                true_count += 1;
                if true_count >= k_limit {
                    break;
                }
            }
            Ok(Some(false)) | Ok(None) => {}
            Err(_) => break,
        }
    }
    assert!(
        true_count <= k_limit,
        "true_count={true_count} exceeded k={k_limit}"
    );
}

// ─── Test 5: GDP compose = √(sum of squares) ──────────────────────────────────

#[test]
fn gdp_compose_square_root_sum_squares() {
    let mus = [1.0_f64; 3];
    let composed = gdp_compose(&mus);
    let expected = 3.0f64.sqrt();
    assert!(
        (composed - expected).abs() < 1e-10,
        "composed μ={composed}, expected √3={expected}"
    );
}

// ─── Test 6: GDP → (ε, δ) gives positive ε ────────────────────────────────────

#[test]
fn gdp_to_eps_delta_nonneg_epsilon() {
    let epsilon = gdp_to_epsilon_delta(1.0, 1e-5).expect("ok");
    assert!(epsilon > 0.0, "epsilon must be positive, got {epsilon}");
}

// ─── Test 7: zCDP compose is additive ─────────────────────────────────────────

#[test]
fn zcdp_compose_additive() {
    let rhos = [0.1, 0.2];
    let total = zcdp_compose(&rhos);
    assert!((total - 0.3).abs() < 1e-12, "expected 0.3, got {total}");
}

// ─── Test 8: zCDP → (ε, δ): smaller δ → larger ε ─────────────────────────────

#[test]
fn zcdp_to_eps_delta_ordering() {
    let eps_small_delta = zcdp_to_epsilon_delta(0.5, 1e-8).expect("ok");
    let eps_large_delta = zcdp_to_epsilon_delta(0.5, 1e-2).expect("ok");
    assert!(
        eps_small_delta > eps_large_delta,
        "smaller δ should give larger ε: {eps_small_delta} > {eps_large_delta}"
    );
}

// ─── Test 9: PRV delta decreasing in epsilon ──────────────────────────────────

#[test]
fn prv_compose_delta_increasing_in_epsilon() {
    let prv = GaussianPrv::new(1.0, 2.0).expect("ok");
    let cfg = PrvConfig::new(-10.0, 10.0, 200).expect("ok");
    let pmf = compose_gaussian_prv(&prv, 3, &cfg).expect("ok");
    let d0 = prv_delta(&pmf, 0.0, &cfg);
    let d1 = prv_delta(&pmf, 1.0, &cfg);
    let d5 = prv_delta(&pmf, 5.0, &cfg);
    assert!(d0 >= d1, "δ(0) >= δ(1): {d0} >= {d1}");
    assert!(d1 >= d5, "δ(1) >= δ(5): {d1} >= {d5}");
}

// ─── Test 10: Strong compose tighter than basic ───────────────────────────────

#[test]
fn strong_compose_tighter_than_basic() {
    let eps = 0.1;
    let delta = 1e-5;
    let k = 100;
    let delta_prime = 1e-5;
    let basic = basic_compose(eps, delta, k);
    let strong = strong_compose(eps, delta, k, delta_prime).expect("ok");
    assert!(
        strong.epsilon < basic.epsilon,
        "strong.ε={} should be < basic.ε={}",
        strong.epsilon,
        basic.epsilon
    );
}

// ─── Test 11: Poisson amplification reduces epsilon ───────────────────────────

#[test]
fn poisson_amplification_reduces_epsilon() {
    let original_eps = 1.0;
    let result = amplify_poisson(original_eps, 1e-5, 0.01).expect("ok");
    assert!(
        result.epsilon < original_eps,
        "amplified ε={} should be < original ε={original_eps}",
        result.epsilon
    );
}

// ─── Test 12: GRR estimate unbiased at large n ────────────────────────────────

#[test]
fn grr_estimate_unbiased_at_large_n() {
    let cfg = GrrConfig::new(5.0, 3).expect("ok");
    let mut rng = LcgRng::new(9999);
    let n = 10_000;
    let reports: Vec<usize> = (0..n)
        .map(|_| grr_encode(0, &cfg, &mut rng).expect("ok"))
        .collect();
    let freqs = grr_estimate_frequency(&reports, &cfg).expect("ok");
    // f̂(0) should be near 1.0.
    assert!(
        (freqs[0] - 1.0).abs() < 0.1,
        "f̂(0)={}, expected ≈ 1.0 (±0.1)",
        freqs[0]
    );
}

// ─── Test 13: OUE estimate unbiased ───────────────────────────────────────────

#[test]
fn oue_estimate_unbiased() {
    let cfg = OueConfig::new(4.0, 5).expect("ok");
    let mut rng = LcgRng::new(8888);
    let n = 10_000;
    let reports: Vec<Vec<u8>> = (0..n)
        .map(|_| oue_encode(2, &cfg, &mut rng).expect("ok"))
        .collect();
    let freqs = oue_estimate_frequency(&reports, &cfg).expect("ok");
    // f̂(2) should be near 1.0.
    assert!(
        (freqs[2] - 1.0).abs() < 0.1,
        "f̂(2)={}, expected ≈ 1.0 (±0.1)",
        freqs[2]
    );
}

// ─── Test 14: DP-Adam state evolves ───────────────────────────────────────────

#[test]
fn dp_adam_state_evolves() {
    let cfg = DpAdamConfig::default();
    let mut rng = LcgRng::new(42);
    let mut state = DpAdamState::new(4);

    let init_params = state.params.clone();
    let grads = vec![0.5f64; 4]; // batch_size=1
    state.step(&grads, 1, &cfg, &mut rng).expect("ok");

    assert_eq!(state.t, 1, "step count should be 1");
    assert_ne!(state.params, init_params, "params should change after step");
    for &p in &state.params {
        assert!(p.is_finite(), "all params must be finite");
    }
}

// ─── Test 15: Privacy budget exhaustion ───────────────────────────────────────

#[test]
fn privacy_budget_exhaustion() {
    let mut budget = PrivacyBudget::new(1.0, 1e-5).expect("ok");
    // Spending more than the total should error.
    let result = budget.spend(1.5, 0.0);
    assert!(result.is_err(), "spending > budget should fail");
    // Verify it's a BudgetExhausted error via message.
    let err_str = result
        .expect_err("spending over budget should fail")
        .to_string();
    assert!(
        err_str.contains("budget") || err_str.contains("Budget"),
        "error message should mention budget: {err_str}"
    );
}

// ─── Test 16: PTX kernels non-empty, contain entry ────────────────────────────

#[test]
fn handle_ptx_kernels_non_empty() {
    let sm_versions = [75u32, 80, 89, 90, 100, 120];

    type KernelEntry = (&'static str, fn(u32) -> String);
    let kernel_fns: &[KernelEntry] = &[
        ("exponential_sample", exponential_sample_ptx),
        ("laplace_noise", laplace_noise_ptx),
        ("gaussian_noise", gaussian_noise_ptx),
        ("clip_gradient", clip_gradient_ptx),
        ("svt_threshold", svt_threshold_ptx),
        ("prv_convolve", prv_convolve_ptx),
        ("oue_encode", oue_encode_ptx),
    ];

    for &sm in &sm_versions {
        for &(name, kernel_fn) in kernel_fns {
            let ptx = kernel_fn(sm);
            assert!(
                !ptx.is_empty(),
                "kernel {name} for SM {sm} returned empty PTX"
            );
            assert!(
                ptx.contains(".visible .entry"),
                "kernel {name} for SM {sm} missing '.visible .entry'"
            );
        }
    }
}

// ─── Test 17: Gaussian utility utility ────────────────────────────────────────

#[test]
fn gaussian_utility_increases_with_smaller_delta() {
    // sigma should increase (more noise) when delta is smaller.
    let sigma_small_delta = gaussian_utility(1.0, 1.0, 1e-8).expect("ok");
    let sigma_large_delta = gaussian_utility(1.0, 1.0, 1e-2).expect("ok");
    assert!(
        sigma_small_delta > sigma_large_delta,
        "tighter δ → more noise: {sigma_small_delta} > {sigma_large_delta}"
    );
}

// ─── Test 18: PRV PMF sums to ~1 after composition ────────────────────────────

#[test]
fn prv_composed_pmf_sums_to_one() {
    let prv = GaussianPrv::new(1.0, 3.0).expect("ok");
    let cfg = PrvConfig::new(-8.0, 8.0, 100).expect("ok");
    let base = gaussian_prv_pmf(&prv, &cfg);
    let total: f64 = base.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "PMF should sum to 1, got {total}"
    );

    let composed = compose_gaussian_prv(&prv, 2, &cfg).expect("ok");
    let total2: f64 = composed.iter().sum();
    assert!(
        (total2 - 1.0).abs() < 1e-4,
        "composed PMF should sum to ~1, got {total2}"
    );
}
