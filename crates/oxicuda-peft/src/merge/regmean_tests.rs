//! Unit tests for [`super::regmean`]. Split into a sibling file to keep
//! `regmean.rs` within the per-file budget while bundling a comprehensive
//! battery of correctness, dimension, and numerical-stability checks.

use super::regmean::{RegMean, RegMeanConfig, solve_gauss_jordan_for_test};
use crate::error::PeftError;

fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

fn identity_gram(n: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; n * n];
    for i in 0..n {
        g[i * n + i] = 1.0;
    }
    g
}

fn zero_gram(n: usize) -> Vec<f32> {
    vec![0.0_f32; n * n]
}

/// Naïve OLS solver `W = (XᵀX + ε·I)⁻¹ · XᵀY` for benchmark comparison.
fn ols_reference(x: &[Vec<f32>], y: &[Vec<f32>], d_in: usize, d_out: usize, eps: f64) -> Vec<f32> {
    let mut xtx = vec![0.0_f64; d_in * d_in];
    let mut xty = vec![0.0_f64; d_in * d_out];
    for (row, target) in x.iter().zip(y.iter()) {
        for a in 0..d_in {
            let xa = row[a] as f64;
            for b in 0..d_in {
                xtx[a * d_in + b] += xa * (row[b] as f64);
            }
            for j in 0..d_out {
                xty[a * d_out + j] += xa * (target[j] as f64);
            }
        }
    }
    for d in 0..d_in {
        xtx[d * d_in + d] += eps;
    }
    let w = solve_gauss_jordan_for_test(xtx, xty, d_in, d_out).expect("ols solve");
    w.into_iter().map(|v| v as f32).collect()
}

#[test]
fn single_model_returns_itself() {
    let d_in = 3;
    let d_out = 2;
    let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g = identity_gram(d_in);
    let cfg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 1e-12,
    };
    let merged = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg).expect("merge");
    assert!(
        approx_eq_slice(&merged, &w, 1e-4),
        "single model not recovered: {merged:?} vs {w:?}"
    );
}

#[test]
fn two_identical_models_returns_the_same_model() {
    let d_in = 3;
    let d_out = 2;
    let w = [1.0_f32, -2.0, 3.0, 4.0, -5.0, 6.0];
    let g = identity_gram(d_in);
    let cfg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 1e-12,
    };
    let merged =
        RegMean::merge(&[(&w[..], &g[..]), (&w[..], &g[..])], d_in, d_out, &cfg).expect("merge");
    assert!(approx_eq_slice(&merged, &w, 1e-4));
}

#[test]
fn equal_gram_yields_unweighted_mean_of_weights() {
    let d_in = 3;
    let d_out = 2;
    let w1 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w2 = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let g = identity_gram(d_in);
    let cfg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 1e-12,
    };
    let merged =
        RegMean::merge(&[(&w1[..], &g[..]), (&w2[..], &g[..])], d_in, d_out, &cfg).expect("merge");
    let expected: Vec<f32> = w1
        .iter()
        .zip(w2.iter())
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    assert!(approx_eq_slice(&merged, &expected, 1e-4));
}

#[test]
fn identity_gram_alpha_zero_vs_one_same_result() {
    let d_in = 3;
    let d_out = 2;
    let w = [1.0_f32, -1.0, 2.0, -2.0, 3.0, -3.0];
    let g = identity_gram(d_in);
    let cfg_zero = RegMeanConfig {
        non_diag_alpha: 0.0,
        eps: 1e-12,
    };
    let cfg_one = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 1e-12,
    };
    let merged_zero = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_zero).expect("a");
    let merged_one = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_one).expect("b");
    assert!(approx_eq_slice(&merged_zero, &merged_one, 1e-4));
}

#[test]
fn alpha_zero_ignores_off_diagonals() {
    let d_in = 3;
    let d_out = 2;
    let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g_a = [
        1.0_f32, 0.7, 0.3, //
        0.7, 2.0, 0.4, //
        0.3, 0.4, 3.0, //
    ];
    let g_b = [
        1.0_f32, -0.5, 0.0, //
        -0.5, 2.0, 0.9, //
        0.0, 0.9, 3.0, //
    ];
    let cfg = RegMeanConfig {
        non_diag_alpha: 0.0,
        eps: 1e-12,
    };
    let merged_a = RegMean::merge(&[(&w[..], &g_a[..])], d_in, d_out, &cfg).expect("a");
    let merged_b = RegMean::merge(&[(&w[..], &g_b[..])], d_in, d_out, &cfg).expect("b");
    assert!(
        approx_eq_slice(&merged_a, &merged_b, 1e-4),
        "α=0 depended on off-diagonal entries: {merged_a:?} vs {merged_b:?}"
    );
}

#[test]
fn ols_recovery_on_pooled_data() {
    let d_in = 3;
    let d_out = 2;
    let x: Vec<Vec<f32>> = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![0.5_f32, -1.0, 2.5],
        vec![2.0_f32, 0.5, -1.0],
        vec![1.5_f32, 1.0, 1.0],
    ];
    let y_1: Vec<Vec<f32>> = vec![
        vec![0.5_f32, 1.0],
        vec![-0.3_f32, 0.8],
        vec![0.7_f32, -0.2],
        vec![0.1_f32, 0.6],
    ];
    let y_2: Vec<Vec<f32>> = vec![
        vec![0.2_f32, -1.0],
        vec![-0.6_f32, 0.4],
        vec![1.1_f32, 0.0],
        vec![0.3_f32, -0.5],
    ];

    let eps = 1e-6_f32;
    let cfg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps,
    };

    let w_1 = ols_reference(&x, &y_1, d_in, d_out, eps as f64 / 2.0);
    let w_2 = ols_reference(&x, &y_2, d_in, d_out, eps as f64 / 2.0);

    let gram = RegMean::compute_gram(&x).expect("gram");

    let merged = RegMean::merge(
        &[(&w_1[..], &gram[..]), (&w_2[..], &gram[..])],
        d_in,
        d_out,
        &cfg,
    )
    .expect("merge");

    let y_avg: Vec<Vec<f32>> = y_1
        .iter()
        .zip(y_2.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .map(|(va, vb)| 0.5 * (va + vb))
                .collect()
        })
        .collect();
    let w_ref = ols_reference(&x, &y_avg, d_in, d_out, eps as f64 / 2.0);
    assert!(
        approx_eq_slice(&merged, &w_ref, 1e-2),
        "RegMean failed to recover pooled OLS: merged={merged:?} ref={w_ref:?}"
    );
}

#[test]
fn weight_dim_mismatch_errors() {
    let d_in = 3;
    let d_out = 2;
    let w = vec![0.0_f32; d_in * d_out + 1];
    let g = identity_gram(d_in);
    let cfg = RegMeanConfig::default();
    let res = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn gram_dim_mismatch_errors() {
    let d_in = 3;
    let d_out = 2;
    let w = vec![0.0_f32; d_in * d_out];
    let g = vec![0.0_f32; d_in * d_in + 1];
    let cfg = RegMeanConfig::default();
    let res = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn empty_models_errors() {
    let cfg = RegMeanConfig::default();
    let res = RegMean::merge(&[], 2, 2, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn negative_or_zero_eps_errors() {
    let d_in = 2;
    let d_out = 2;
    let w = vec![0.0_f32; d_in * d_out];
    let g = identity_gram(d_in);
    let cfg_zero = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 0.0,
    };
    let cfg_neg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: -1e-6,
    };
    assert!(matches!(
        RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_zero),
        Err(PeftError::Internal { .. })
    ));
    assert!(matches!(
        RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_neg),
        Err(PeftError::Internal { .. })
    ));
}

#[test]
fn alpha_out_of_range_errors() {
    let d_in = 2;
    let d_out = 2;
    let w = vec![0.0_f32; d_in * d_out];
    let g = identity_gram(d_in);
    let cfg_lo = RegMeanConfig {
        non_diag_alpha: -0.1,
        eps: 1e-6,
    };
    let cfg_hi = RegMeanConfig {
        non_diag_alpha: 1.1,
        eps: 1e-6,
    };
    let cfg_nan = RegMeanConfig {
        non_diag_alpha: f32::NAN,
        eps: 1e-6,
    };
    assert!(matches!(
        RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_lo),
        Err(PeftError::Internal { .. })
    ));
    assert!(matches!(
        RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_hi),
        Err(PeftError::Internal { .. })
    ));
    assert!(matches!(
        RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg_nan),
        Err(PeftError::Internal { .. })
    ));
}

#[test]
fn compute_gram_empty_errors() {
    let res = RegMean::compute_gram(&[]);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn compute_gram_zero_dim_errors() {
    let res = RegMean::compute_gram(&[Vec::<f32>::new()]);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn compute_gram_row_length_mismatch_errors() {
    let x = vec![vec![1.0_f32, 2.0], vec![1.0_f32, 2.0, 3.0]];
    let res = RegMean::compute_gram(&x);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn compute_gram_correctness_2x2() {
    // X = [[1, 2], [3, 4]] → XᵀX = [[10, 14], [14, 20]]
    let x = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
    let g = RegMean::compute_gram(&x).expect("gram");
    let expected = [10.0_f32, 14.0, 14.0, 20.0];
    assert!(
        approx_eq_slice(&g, &expected, 1e-5),
        "gram mismatch: {g:?} vs {expected:?}"
    );
}

#[test]
fn singular_gram_regularised_by_eps_stays_finite() {
    let d_in = 3;
    let d_out = 2;
    let w = vec![5.0_f32; d_in * d_out];
    let g = zero_gram(d_in);
    let cfg = RegMeanConfig {
        non_diag_alpha: 1.0,
        eps: 1e-6,
    };
    let merged = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg).expect("merge");
    for &v in &merged {
        assert!(v.is_finite(), "non-finite value {v} in zero-Gram merge");
        assert!(v.abs() < 1e-3, "expected ≈ 0 for zero Gram, got {v}");
    }
}

#[test]
fn non_finite_weight_errors() {
    let d_in = 2;
    let d_out = 2;
    let mut w = vec![0.0_f32; d_in * d_out];
    w[0] = f32::NAN;
    let g = identity_gram(d_in);
    let cfg = RegMeanConfig::default();
    let res = RegMean::merge(&[(&w[..], &g[..])], d_in, d_out, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn deterministic_merge() {
    let d_in = 4;
    let d_out = 3;
    let w_1: Vec<f32> = (0..d_in * d_out).map(|i| (i as f32) * 0.1).collect();
    let w_2: Vec<f32> = (0..d_in * d_out).map(|i| 1.0 - (i as f32) * 0.05).collect();
    let g_1 = identity_gram(d_in);
    let mut g_2 = identity_gram(d_in);
    for i in 0..d_in {
        g_2[i * d_in + i] = 0.5;
    }
    let cfg = RegMeanConfig::default();
    let a = RegMean::merge(
        &[(&w_1[..], &g_1[..]), (&w_2[..], &g_2[..])],
        d_in,
        d_out,
        &cfg,
    )
    .expect("a");
    let b = RegMean::merge(
        &[(&w_1[..], &g_1[..]), (&w_2[..], &g_2[..])],
        d_in,
        d_out,
        &cfg,
    )
    .expect("b");
    assert_eq!(a, b);
}

#[test]
fn zero_d_in_or_d_out_errors() {
    let cfg = RegMeanConfig::default();
    let res_in = RegMean::merge(&[(&[][..], &[][..])], 0, 2, &cfg);
    assert!(matches!(res_in, Err(PeftError::Internal { .. })));
    let res_out = RegMean::merge(&[(&[][..], &[][..])], 2, 0, &cfg);
    assert!(matches!(res_out, Err(PeftError::Internal { .. })));
}
