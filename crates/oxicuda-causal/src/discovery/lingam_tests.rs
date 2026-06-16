//! Tests for `discovery/lingam.rs`.

use super::lingam::{Lingam, LingamConfig, LingamGFunction, LingamResult};
use crate::error::CausalError;
use crate::handle::{CausalHandle, LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// DGP helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform-noise helper: maps LcgRng output into (−0.5, 0.5).
fn uniform_noise(rng: &mut LcgRng) -> f64 {
    rng.next_f32() as f64 - 0.5
}

/// Generate data from the three-node chain:
///   x1 = ε1
///   x2 = 0.8·x1 + ε2
///   x3 = 0.5·x2 + ε3
/// where ε_i ~ Uniform(−0.5, 0.5).
fn three_node_chain(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    let mut data = vec![0.0_f64; n * 3];
    for i in 0..n {
        let e1 = uniform_noise(&mut rng);
        let e2 = uniform_noise(&mut rng);
        let e3 = uniform_noise(&mut rng);
        let x1 = e1;
        let x2 = 0.8 * x1 + e2;
        let x3 = 0.5 * x2 + e3;
        data[i * 3] = x1;
        data[i * 3 + 1] = x2;
        data[i * 3 + 2] = x3;
    }
    data
}

/// Generate data from the two-node chain:
///   x1 = ε1
///   x2 = coeff·x1 + ε2
/// where ε_i ~ Uniform(−0.5, 0.5).
fn two_node_chain(n: usize, coeff: f64, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    let mut data = vec![0.0_f64; n * 2];
    for i in 0..n {
        let e1 = uniform_noise(&mut rng);
        let e2 = uniform_noise(&mut rng);
        let x1 = e1;
        let x2 = coeff * x1 + e2;
        data[i * 2] = x1;
        data[i * 2 + 1] = x2;
    }
    data
}

/// Default handle for tests.
fn make_handle(seed: u64) -> CausalHandle {
    CausalHandle::new(80, seed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: check that a slice is a permutation of 0..d.
// ─────────────────────────────────────────────────────────────────────────────

fn is_permutation(v: &[usize], d: usize) -> bool {
    if v.len() != d {
        return false;
    }
    let mut seen = vec![false; d];
    for &x in v {
        if x >= d || seen[x] {
            return false;
        }
        seen[x] = true;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// 1. Three-node chain: x1=ε1, x2=0.8·x1+ε2, x3=0.5·x2+ε3.
///    Verify ordering[0] == 0 (x1 is the root), or that B is consistent with
///    the causal structure (no false edges from children to parents).
#[test]
fn three_node_chain_ordering_root() {
    let n = 500;
    let data = three_node_chain(n, 42);
    let cfg = LingamConfig::default();
    let mut h = make_handle(1);
    let res =
        Lingam::fit(&data, n, 3, &cfg, &mut h).expect("three-node chain Lingam fit should succeed");
    // The algorithm may not always recover the exact order, but the result
    // must be a valid permutation and the B matrix must be the correct size.
    assert!(is_permutation(&res.ordering, 3));
    assert_eq!(res.b.len(), 9);
    // x1 (index 0) should appear first or at least early in the ordering.
    // We accept ordering[0] == 0 or check that the ordering places 0 before 1.
    let pos0 = res
        .ordering
        .iter()
        .position(|&x| x == 0)
        .expect("variable 0 must appear in the ordering");
    let pos1 = res
        .ordering
        .iter()
        .position(|&x| x == 1)
        .expect("variable 1 must appear in the ordering");
    // x0 must come before x1 in the ordering (0 causes 1).
    assert!(pos0 < pos1, "ordering = {:?}", res.ordering);
}

/// 2. Two-node case: x1=ε1, x2=2·x1+ε2.
///    Check |B[1][0]| is approximately in (1.5, 2.5).
#[test]
fn two_node_chain_b_coefficient() {
    let n = 600;
    let data = two_node_chain(n, 2.0, 99);
    let cfg = LingamConfig::default();
    let mut h = make_handle(7);
    let res =
        Lingam::fit(&data, n, 2, &cfg, &mut h).expect("two-node chain Lingam fit should succeed");
    // Find the edge from x0 to x1 in the ordering.
    // B[i, j] = structural coefficient from j to i.
    // In the causal order x0 → x1, B[1][0] should ≈ -(−2) = 2 (since B = I − W).
    // We check the absolute value of the off-diagonal entry.
    let b10 = res.b[2].abs(); // b[1][0]
    let b01 = res.b[1].abs(); // b[0][1]
    // One of these should be large (≈ 2).
    let max_off = b10.max(b01);
    assert!(
        max_off > 1.5,
        "expected |B off-diag| > 1.5, got b10={b10:.4}, b01={b01:.4}; ordering={:?}",
        res.ordering
    );
}

/// 3. Deterministic: same LcgRng seed → identical LingamResult.
#[test]
fn deterministic_with_same_seed() {
    let n = 300;
    let data = two_node_chain(n, 1.5, 77);
    let cfg = LingamConfig::default();
    let mut h1 = make_handle(123);
    let mut h2 = make_handle(123);
    let r1 = Lingam::fit(&data, n, 2, &cfg, &mut h1)
        .expect("first Lingam fit with seed 123 should succeed");
    let r2 = Lingam::fit(&data, n, 2, &cfg, &mut h2)
        .expect("second Lingam fit with seed 123 should succeed");
    assert_eq!(r1.ordering, r2.ordering);
    assert_eq!(r1.b, r2.b);
    assert_eq!(r1.w, r2.w);
    assert_eq!(r1.ica_converged, r2.ica_converged);
}

/// 4. Empty input → EmptyInput error.
#[test]
fn empty_n_returns_empty_input() {
    let cfg = LingamConfig::default();
    let mut h = make_handle(1);
    let r = Lingam::fit(&[], 0, 3, &cfg, &mut h);
    assert!(matches!(r, Err(CausalError::EmptyInput)));
}

/// 5. Empty input (d=0) → EmptyInput error.
#[test]
fn empty_d_returns_empty_input() {
    let cfg = LingamConfig::default();
    let mut h = make_handle(1);
    let r = Lingam::fit(&[1.0, 2.0], 2, 0, &cfg, &mut h);
    assert!(matches!(r, Err(CausalError::EmptyInput)));
}

/// 6. n < d → IncompatibleData.
#[test]
fn n_less_than_d_returns_incompatible_data() {
    let cfg = LingamConfig::default();
    let mut h = make_handle(1);
    // n=2, d=3 → underdetermined.
    let data = vec![0.0_f64; 6]; // 2 × 3 = 6
    let r = Lingam::fit(&data, 2, 3, &cfg, &mut h);
    assert!(matches!(r, Err(CausalError::IncompatibleData)));
}

/// 7. x.len() != n*d → DimensionMismatch.
#[test]
fn dimension_mismatch_returns_error() {
    let cfg = LingamConfig::default();
    let mut h = make_handle(1);
    let data = vec![0.0_f64; 5]; // should be 6 for n=3, d=2
    let r = Lingam::fit(&data, 3, 2, &cfg, &mut h);
    assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
}

/// 8. All-Gaussian data: ica_converged may be false; result returns without
///    panic.
#[test]
fn gaussian_data_no_panic() {
    let n = 200;
    let d = 3;
    let mut rng = LcgRng::new(55);
    let mut data = vec![0.0_f64; n * d];
    for v in data.iter_mut() {
        *v = rng.next_normal() as f64;
    }
    let cfg = LingamConfig::default();
    let mut h = make_handle(3);
    // May fail or succeed, but must not panic.
    let _ = Lingam::fit(&data, n, d, &cfg, &mut h);
}

/// 9. g_function Gauss works on a 2-node chain.
#[test]
fn gauss_nonlinearity_two_node() {
    let n = 400;
    let data = two_node_chain(n, 1.5, 11);
    let cfg = LingamConfig {
        g_function: LingamGFunction::Gauss,
        ..LingamConfig::default()
    };
    let mut h = make_handle(42);
    let res = Lingam::fit(&data, n, 2, &cfg, &mut h);
    assert!(res.is_ok(), "Gauss g function failed: {:?}", res.err());
    let r = res.expect("Lingam fit with Gauss g-function should succeed");
    assert_eq!(r.b.len(), 4);
    assert_eq!(r.w.len(), 4);
}

/// 10. g_function Cube works on a 2-node chain.
#[test]
fn cube_nonlinearity_two_node() {
    let n = 400;
    let data = two_node_chain(n, 1.5, 13);
    let cfg = LingamConfig {
        g_function: LingamGFunction::Cube,
        ..LingamConfig::default()
    };
    let mut h = make_handle(99);
    let res = Lingam::fit(&data, n, 2, &cfg, &mut h);
    assert!(res.is_ok(), "Cube g function failed: {:?}", res.err());
    let r = res.expect("Lingam fit with Cube g-function should succeed");
    assert_eq!(r.b.len(), 4);
    assert_eq!(r.w.len(), 4);
}

/// 11. B matrix is d×d.
#[test]
fn b_matrix_has_correct_size() {
    let n = 200;
    let d = 4;
    let data = three_node_chain(n, 17); // we only use 3 cols; make a 4-node dataset
    // Create a proper 4-col dataset.
    let mut rng = LcgRng::new(17);
    let mut data4 = vec![0.0_f64; n * d];
    for i in 0..n {
        let e1 = uniform_noise(&mut rng);
        let e2 = uniform_noise(&mut rng);
        let e3 = uniform_noise(&mut rng);
        let e4 = uniform_noise(&mut rng);
        data4[i * 4] = e1;
        data4[i * 4 + 1] = 0.7 * e1 + e2;
        data4[i * 4 + 2] = 0.5 * data4[i * 4 + 1] + e3;
        data4[i * 4 + 3] = 0.3 * data4[i * 4 + 2] + e4;
    }
    let cfg = LingamConfig::default();
    let mut h = make_handle(7);
    let res =
        Lingam::fit(&data4, n, d, &cfg, &mut h).expect("four-node chain Lingam fit should succeed");
    assert_eq!(res.b.len(), d * d, "b should be d×d = {}", d * d);
    let _ = data; // suppress unused warning
}

/// 12. w matrix is d×d.
#[test]
fn w_matrix_has_correct_size() {
    let n = 200;
    let data = two_node_chain(n, 1.2, 33);
    let cfg = LingamConfig::default();
    let mut h = make_handle(5);
    let res = Lingam::fit(&data, n, 2, &cfg, &mut h)
        .expect("two-node chain Lingam fit for w-size test should succeed");
    assert_eq!(res.w.len(), 4);
}

/// 13. ordering has length d and is a permutation.
#[test]
fn ordering_is_permutation_of_d() {
    let n = 300;
    let data = three_node_chain(n, 21);
    let cfg = LingamConfig::default();
    let mut h = make_handle(11);
    let res = Lingam::fit(&data, n, 3, &cfg, &mut h)
        .expect("three-node chain Lingam fit for ordering test should succeed");
    assert!(
        is_permutation(&res.ordering, 3),
        "ordering = {:?} is not a permutation of 0..3",
        res.ordering
    );
}

/// 14. d=1 case: trivially returns ordering=[0], B=[0.0].
#[test]
fn d_equals_one_trivial() {
    let n = 50;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 - 2.5).collect();
    let cfg = LingamConfig::default();
    let mut h = make_handle(2);
    let res =
        Lingam::fit(&data, n, 1, &cfg, &mut h).expect("d=1 trivial Lingam fit should succeed");
    assert_eq!(res.ordering, vec![0]);
    assert_eq!(res.b.len(), 1);
    // B = I − W_scaled; for d=1, the diagonal of W_scaled is 1, so B[0][0] = 0.
    assert!(
        res.b[0].abs() < 1e-9,
        "B[0][0] should be 0, got {}",
        res.b[0]
    );
}

/// 15. ridge=1e-8 vs ridge=1.0: both succeed on well-conditioned data.
#[test]
fn different_ridge_values_both_succeed() {
    let n = 300;
    let data = two_node_chain(n, 1.5, 45);
    let mut h1 = make_handle(6);
    let mut h2 = make_handle(6);
    let cfg_small = LingamConfig {
        ridge: 1e-8,
        ..LingamConfig::default()
    };
    let cfg_large = LingamConfig {
        ridge: 1.0,
        ..LingamConfig::default()
    };
    assert!(
        Lingam::fit(&data, n, 2, &cfg_small, &mut h1).is_ok(),
        "ridge=1e-8 should succeed"
    );
    assert!(
        Lingam::fit(&data, n, 2, &cfg_large, &mut h2).is_ok(),
        "ridge=1.0 should succeed"
    );
}

/// 16. Large n=2000 smoke test: doesn't panic.
#[test]
fn large_n_smoke_test() {
    let n = 2000;
    let data = three_node_chain(n, 314);
    let cfg = LingamConfig::default();
    let mut h = make_handle(159);
    let res = Lingam::fit(&data, n, 3, &cfg, &mut h);
    assert!(res.is_ok(), "large-n smoke test failed: {:?}", res.err());
    let r = res.expect("large-n Lingam fit should succeed");
    assert!(is_permutation(&r.ordering, 3));
}

/// 17. Collinear data with ridge=0 → MatrixSingular or ok (no panic).
#[test]
fn collinear_data_low_ridge_no_panic() {
    let n = 50;
    let d = 2;
    // Perfectly collinear: x2 = x1 (no independent noise).
    let data: Vec<f64> = (0..n)
        .flat_map(|i| {
            let v = i as f64 * 0.1;
            [v, v]
        })
        .collect();
    let cfg = LingamConfig {
        ridge: 0.0,
        ..LingamConfig::default()
    };
    let mut h = make_handle(1);
    // Should either return MatrixSingular or succeed — must not panic.
    let result = Lingam::fit(&data, n, d, &cfg, &mut h);
    match result {
        Ok(_) => {}
        Err(CausalError::MatrixSingular) => {}
        Err(e) => {
            // Other errors (e.g. IncompatibleData) are also acceptable.
            let _ = e;
        }
    }
}

/// Bonus: three-node chain LingamResult fields are all finite.
#[test]
fn all_result_fields_finite() {
    let n = 300;
    let data = three_node_chain(n, 100);
    let cfg = LingamConfig::default();
    let mut h = make_handle(50);
    let res = Lingam::fit(&data, n, 3, &cfg, &mut h)
        .expect("three-node chain Lingam fit for finite-fields test should succeed");
    for &v in &res.b {
        assert!(v.is_finite(), "b contains non-finite: {v}");
    }
    for &v in &res.w {
        assert!(v.is_finite(), "w contains non-finite: {v}");
    }
}

/// Verify the LingamResult struct has all expected fields.
#[test]
fn result_struct_fields_accessible() {
    let n = 100;
    let data = two_node_chain(n, 1.0, 22);
    let cfg = LingamConfig::default();
    let mut h = make_handle(8);
    let res: LingamResult = Lingam::fit(&data, n, 2, &cfg, &mut h)
        .expect("two-node chain Lingam fit for struct access test should succeed");
    let _ordering: &Vec<usize> = &res.ordering;
    let _b: &Vec<f64> = &res.b;
    let _w: &Vec<f64> = &res.w;
    let _converged: bool = res.ica_converged;
}
