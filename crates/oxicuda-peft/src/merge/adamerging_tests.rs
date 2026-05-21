//! Unit tests for [`super::adamerging`]. Split into a sibling file to keep
//! `adamerging.rs` within the per-file budget while bundling a comprehensive
//! battery of correctness, configuration, and simplex-invariant checks.

use super::adamerging::{AdaMerging, AdaMergingConfig};
use crate::error::PeftError;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

fn default_cfg() -> AdaMergingConfig {
    AdaMergingConfig::per_task(1e-3, 20, 1.0)
}

#[test]
fn single_task_vector_returns_base_plus_tau() {
    let base = vec![0.1_f32, 0.2, 0.3];
    let tv = vec![1.0_f32, 2.0, 3.0];
    let logits = vec![vec![0.5_f32, 0.4]];
    let cfg = default_cfg();
    let res = AdaMerging::merge(&base, std::slice::from_ref(&tv), &logits, &cfg).expect("merge");
    // With K=1 the simplex degenerates to {1.0}, so coefficient is 1.
    assert!(approx_eq(res.coefficients[0], 1.0, 1e-5));
    let expected: Vec<f32> = base.iter().zip(tv.iter()).map(|(b, t)| b + t).collect();
    assert!(approx_eq_slice(&res.merged, &expected, 1e-5));
}

#[test]
fn two_identical_task_vectors_split_uniformly() {
    let base = vec![0.0_f32, 0.0, 0.0, 0.0];
    let tv = vec![1.0_f32, 1.0, 1.0, 1.0];
    let logits = vec![vec![0.5_f32, 0.4], vec![0.5_f32, 0.4]];
    let cfg = default_cfg();
    let res = AdaMerging::merge(&base, &[tv.clone(), tv.clone()], &logits, &cfg).expect("merge");
    // Symmetry: both coefficients should be ~0.5.
    assert!(approx_eq(res.coefficients[0], 0.5, 1e-4));
    assert!(approx_eq(res.coefficients[1], 0.5, 1e-4));
}

#[test]
fn empty_base_errors() {
    let res = AdaMerging::merge(&[], &[vec![1.0_f32]], &[vec![0.5_f32, 0.4]], &default_cfg());
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn empty_task_vectors_errors() {
    let res = AdaMerging::merge(&[0.1_f32, 0.2], &[], &[vec![0.5_f32, 0.4]], &default_cfg());
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn task_vector_length_mismatch_errors() {
    let base = vec![0.0_f32, 0.0, 0.0];
    let tv = vec![1.0_f32, 2.0]; // length 2 != base length 3
    let logits = vec![vec![0.5_f32, 0.4]];
    let res = AdaMerging::merge(&base, &[tv], &logits, &default_cfg());
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn empty_unlabeled_logits_errors() {
    let base = vec![0.0_f32, 0.0];
    let tv = vec![1.0_f32, 1.0];
    let res = AdaMerging::merge(&base, &[tv], &[], &default_cfg());
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn non_positive_learning_rate_errors() {
    let cfg = AdaMergingConfig::per_task(0.0, 10, 1.0);
    let res = AdaMerging::merge(&[0.0_f32], &[vec![1.0_f32]], &[vec![0.5_f32, 0.4]], &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn zero_iterations_errors() {
    let cfg = AdaMergingConfig::per_task(1e-3, 0, 1.0);
    let res = AdaMerging::merge(&[0.0_f32], &[vec![1.0_f32]], &[vec![0.5_f32, 0.4]], &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn non_positive_temperature_errors() {
    let cfg = AdaMergingConfig::per_task(1e-3, 10, 0.0);
    let res = AdaMerging::merge(&[0.0_f32], &[vec![1.0_f32]], &[vec![0.5_f32, 0.4]], &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn final_entropy_does_not_exceed_initial() {
    // For two distinct logit vectors entropy minimisation should make
    // progress (or at least not regress beyond the start).
    let base = vec![0.0_f32; 4];
    let tv_a = vec![1.0_f32, -1.0, 0.5, -0.5];
    let tv_b = vec![-0.5_f32, 0.5, -1.0, 1.0];
    let logits = vec![vec![2.0_f32, 0.0, 0.0], vec![0.0_f32, 0.0, 2.0]];
    let cfg = AdaMergingConfig::per_task(0.5, 50, 1.0);
    let res = AdaMerging::merge(&base, &[tv_a, tv_b], &logits, &cfg).expect("merge");
    let h0 = res.iter_history[0];
    let h_last = *res.iter_history.last().expect("non-empty history");
    assert!(
        h_last <= h0 + 1e-5,
        "entropy increased: h0={h0}, h_last={h_last}"
    );
}

#[test]
fn coefficients_sum_to_one_per_task_variant() {
    let base = vec![0.0_f32, 0.0, 0.0];
    let tvs = vec![
        vec![1.0_f32, 0.5, -0.5],
        vec![-1.0_f32, 0.3, 0.4],
        vec![0.2_f32, -0.7, 1.0],
    ];
    let logits = vec![
        vec![1.0_f32, 0.5, -0.5],
        vec![0.0_f32, 1.0, 0.0],
        vec![-1.0_f32, 0.0, 0.5],
    ];
    let cfg = AdaMergingConfig::per_task(1e-3, 30, 1.0);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    let sum: f32 = res.coefficients.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5), "sum={sum}");
}

#[test]
fn coefficients_are_non_negative_simplex_invariant() {
    let base = vec![0.0_f32; 3];
    let tvs = vec![
        vec![1.0_f32, 0.0, -1.0],
        vec![0.0_f32, 1.0, 0.0],
        vec![-1.0_f32, 0.5, 1.0],
    ];
    let logits = vec![vec![1.0_f32, -1.0], vec![-1.0_f32, 1.0], vec![0.5_f32, 0.5]];
    let cfg = AdaMergingConfig::per_task(1e-3, 50, 1.0);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    for c in res.coefficients {
        assert!((0.0..=1.0 + 1e-6).contains(&c), "coef out of range: {c}");
    }
}

#[test]
fn deterministic_under_repeated_calls() {
    let base = vec![0.1_f32, 0.2, -0.1];
    let tvs = vec![vec![0.5_f32, 0.0, 0.5], vec![-0.5_f32, 0.5, 0.0]];
    let logits = vec![vec![1.0_f32, 0.5], vec![0.5_f32, 1.0]];
    let cfg = AdaMergingConfig::per_task(1e-3, 20, 1.0);
    let r1 = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    let r2 = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    assert_eq!(r1.coefficients, r2.coefficients);
    assert_eq!(r1.merged, r2.merged);
}

#[test]
fn layer_wise_coefficient_count_is_k_times_l() {
    // base has 4 params split into two layers of 2 each.
    let base = vec![0.0_f32; 4];
    let tvs = vec![
        vec![1.0_f32, 0.0, -1.0, 0.5],
        vec![-0.5_f32, 1.0, 0.0, -1.0],
    ];
    let logits = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let cfg = AdaMergingConfig::layer_wise(1e-3, 5, 1.0, vec![2, 4]);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    assert_eq!(res.coefficients.len(), 2 * 2);
}

#[test]
fn layer_wise_offsets_strictly_increasing_required() {
    let base = vec![0.0_f32; 4];
    let tvs = vec![vec![1.0_f32; 4], vec![-1.0_f32; 4]];
    let logits = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    // Non-increasing offsets must be rejected.
    let cfg = AdaMergingConfig::layer_wise(1e-3, 5, 1.0, vec![2, 2]);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn layer_wise_offsets_must_cover_full_length() {
    let base = vec![0.0_f32; 4];
    let tvs = vec![vec![1.0_f32; 4], vec![-1.0_f32; 4]];
    let logits = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    // Final offset is 3, not 4 → reject.
    let cfg = AdaMergingConfig::layer_wise(1e-3, 5, 1.0, vec![2, 3]);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn merged_length_matches_base_length() {
    let base = vec![0.0_f32; 7];
    let tvs = vec![vec![0.1_f32; 7], vec![-0.1_f32; 7]];
    let logits = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let cfg = AdaMergingConfig::per_task(1e-3, 10, 1.0);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    assert_eq!(res.merged.len(), base.len());
}

#[test]
fn iter_history_length_matches_n_iters() {
    let base = vec![0.0_f32; 3];
    let tvs = vec![vec![1.0_f32; 3], vec![-1.0_f32; 3]];
    let logits = vec![vec![1.0_f32, 0.5], vec![0.5_f32, 1.0]];
    let cfg = AdaMergingConfig::per_task(1e-3, 17, 1.0);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    assert_eq!(res.iter_history.len(), 17);
}

#[test]
fn final_entropy_matches_returned_field() {
    let base = vec![0.0_f32; 3];
    let tvs = vec![vec![1.0_f32, 2.0, 3.0], vec![-1.0_f32, 0.0, 1.0]];
    let logits = vec![vec![1.0_f32, 0.5], vec![0.5_f32, 1.0]];
    let cfg = AdaMergingConfig::per_task(1e-3, 10, 1.0);
    let res = AdaMerging::merge(&base, &tvs, &logits, &cfg).expect("merge");
    // final_entropy is recomputed *after* the last update from the post-step
    // lambda — sanity-check that it is a finite non-negative number ≤ ln(C).
    let upper = (logits[0].len() as f32).ln() + 1e-3;
    assert!(res.final_entropy.is_finite());
    assert!(res.final_entropy >= 0.0);
    assert!(res.final_entropy <= upper);
}

#[test]
fn unlabeled_logit_length_mismatch_errors() {
    // Logit-vectors of differing length must be rejected.
    let base = vec![0.0_f32, 0.0];
    let tvs = vec![vec![1.0_f32, 1.0], vec![-1.0_f32, -1.0]];
    let logits = vec![vec![0.5_f32, 0.4], vec![0.5_f32, 0.4, 0.1]];
    let res = AdaMerging::merge(&base, &tvs, &logits, &default_cfg());
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}
