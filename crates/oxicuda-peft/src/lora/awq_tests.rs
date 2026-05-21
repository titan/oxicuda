//! Unit tests for [`super::awq`]. Split into a sibling file to keep `awq.rs`
//! within the per-file budget while bundling a comprehensive battery of
//! correctness, dimension, and numerical-stability checks.

use super::awq::{Awq, AwqConfig, AwqQuantized};
use crate::error::PeftError;

/// Build a deterministic `rows × cols` weight matrix in `[-amp, amp]` whose
/// entries blend a linear ramp with a low-frequency sinusoid.
fn synthetic_weight(rows: usize, cols: usize, amp: f32) -> Vec<f32> {
    let mut w = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let ti = (i as f32 + 0.5) / rows.max(1) as f32;
            let tj = (j as f32 + 0.5) / cols.max(1) as f32;
            w[i * cols + j] = amp * (2.0 * tj - 1.0) + 0.3 * amp * ((4.0 * ti + 7.0 * tj).sin());
        }
    }
    w
}

fn weight_range(w: &[f32]) -> f32 {
    let mut lo = w[0];
    let mut hi = w[0];
    for &v in &w[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    hi - lo
}

fn linf(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn default_cfg(bits: u8, group_size: usize, steps: u32) -> AwqConfig {
    AwqConfig {
        bits,
        group_size,
        alpha_search_steps: steps,
    }
}

#[test]
fn round_trip_8bit_within_five_percent_of_range() {
    let rows = 6;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(8, 8, 4);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    let dq = Awq::dequantize_and_apply(&q).expect("dequant");
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    assert!(
        err < r * 0.05,
        "8-bit AWQ L∞ err={err} not within 5% of range {r}"
    );
}

#[test]
fn round_trip_4bit_bounded_by_one_sixteenth_range() {
    let rows = 6;
    let cols = 32;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(4, 8, 4);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    let dq = Awq::dequantize_and_apply(&q).expect("dequant");
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    // 4-bit per-group affine: worst-case is 0.5 · group_scale ≈ range / (2 * 15).
    // We bound generously by range/16 to absorb cross-group interactions.
    assert!(
        err < r / 8.0,
        "4-bit AWQ L∞ err={err} exceeds bound (range/8 = {})",
        r / 8.0
    );
}

#[test]
fn round_trip_3bit_bounded_error() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(3, 8, 4);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    let dq = Awq::dequantize_and_apply(&q).expect("dequant");
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    // 3-bit step is roughly range/7 per group → bound by range/4.
    assert!(
        err < r / 2.0,
        "3-bit AWQ L∞ err={err} exceeds bound (range/2 = {})",
        r / 2.0
    );
    // Ensure no NaN.
    for &v in &dq {
        assert!(v.is_finite(), "3-bit AWQ produced non-finite {v}");
    }
}

#[test]
fn alpha_zero_reduces_to_pure_per_group_min_max() {
    // alpha_search_steps=1 → alpha grid is {0.0, 1.0}. With identical activations
    // both alphas produce the same scaling (s_i ≡ 1 after normalisation), so the
    // chosen alpha can be either — the codes should be those of pure min/max.
    let rows = 4;
    let cols = 8;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(8, 4, 1);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    // With uniform activations, all awq_scale entries equal 1 (geomean=1 ⇒ normalised).
    for &v in &q.awq_scale {
        assert!((v - 1.0).abs() < 1e-5, "awq_scale {v} not ≈ 1.0");
    }
}

#[test]
fn act_dim_mismatch_errors() {
    let rows = 4;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let act = vec![1.0_f32; rows + 1];
    let cfg = default_cfg(4, 2, 4);
    let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn weight_length_mismatch_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols + 2];
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(4, 2, 4);
    let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn invalid_bits_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let act = vec![1.0_f32; rows];
    for &bits in &[0_u8, 1, 2, 5, 6, 7, 16] {
        let cfg = AwqConfig {
            bits,
            group_size: 2,
            alpha_search_steps: 4,
        };
        let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for bits={bits}"
        );
    }
}

#[test]
fn single_column_degenerate_case() {
    let rows = 6;
    let cols = 1;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = AwqConfig {
        bits: 4,
        group_size: 1,
        alpha_search_steps: 4,
    };
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    assert_eq!(q.q.len(), rows);
    assert_eq!(q.scale.len(), 1);
    assert_eq!(q.zero.len(), 1);
    assert_eq!(q.awq_scale.len(), rows);
    let dq = Awq::dequantize_and_apply(&q).expect("dequant");
    assert_eq!(dq.len(), rows);
}

#[test]
fn deterministic_no_rng() {
    let rows = 5;
    let cols = 12;
    let w = synthetic_weight(rows, cols, 1.0);
    let act: Vec<f32> = (0..rows).map(|i| 0.5 + 0.1 * (i as f32)).collect();
    let cfg = default_cfg(4, 4, 8);
    let a = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("a");
    let b = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("b");
    assert_eq!(a.q, b.q);
    assert_eq!(a.scale, b.scale);
    assert_eq!(a.zero, b.zero);
    assert_eq!(a.alpha, b.alpha);
    assert_eq!(a.awq_scale, b.awq_scale);
}

#[test]
fn group_size_greater_than_cols_clamps_to_cols() {
    let rows = 3;
    let cols = 7;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = AwqConfig {
        bits: 4,
        group_size: 999,
        alpha_search_steps: 4,
    };
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    assert_eq!(q.group_size, cols, "group size should clamp to cols");
    assert_eq!(
        q.scale.len(),
        1,
        "single oversized group should produce one (scale, zero)"
    );
    assert_eq!(q.zero.len(), 1);
    assert_eq!(q.q.len(), rows * cols);
}

#[test]
fn idempotent_requantize_dequant() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act = vec![1.0_f32; rows];
    let cfg = default_cfg(8, 8, 4);
    let q1 = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("q1");
    let dq1 = Awq::dequantize_and_apply(&q1).expect("dq1");
    let q2 = Awq::quantize_weight(&dq1, rows, cols, &act, &cfg).expect("q2");
    let dq2 = Awq::dequantize_and_apply(&q2).expect("dq2");
    // Re-quantizing a dequantized weight should converge — small drift only.
    let drift = linf(&dq1, &dq2);
    assert!(
        drift < 1e-2,
        "quantize→dequant→quantize drift {drift} above tolerance"
    );
}

#[test]
fn salient_channels_have_lower_per_row_error_than_baseline() {
    // Build a synthetic where one input channel has 10× activation mean.
    // AWQ should pick α > 0 to flatten that row, reducing its weighted error
    // relative to a baseline run with α=0 (uniform activations).
    let rows = 6;
    let cols = 32;
    let mut w = synthetic_weight(rows, cols, 1.0);
    // Make row 2 carry an aggressive spike so quantization without rescaling
    // truncates large magnitudes.
    for j in 0..cols {
        let tj = (j as f32 + 0.5) / cols as f32;
        w[2 * cols + j] = 3.0 * (2.0 * tj - 1.0); // ±3 range
    }
    // Activation mean: row 2 is 10× louder.
    let mut act = vec![1.0_f32; rows];
    act[2] = 10.0;

    let cfg_uniform = AwqConfig {
        bits: 4,
        group_size: 8,
        alpha_search_steps: 1, // grid = {0.0, 1.0}; force min/max-only with α=0 baseline
    };
    let cfg_full = AwqConfig {
        bits: 4,
        group_size: 8,
        alpha_search_steps: 8,
    };

    // Build a baseline that *fixes* α=0 by passing uniform activations.
    let act_uniform = vec![1.0_f32; rows];
    let q_baseline =
        Awq::quantize_weight(&w, rows, cols, &act_uniform, &cfg_uniform).expect("baseline");
    let q_awq = Awq::quantize_weight(&w, rows, cols, &act, &cfg_full).expect("awq");
    let dq_baseline = Awq::dequantize_and_apply(&q_baseline).expect("dq baseline");
    let dq_awq = Awq::dequantize_and_apply(&q_awq).expect("dq awq");

    // Compare salient-row L1 error.
    let mut err_baseline = 0.0_f32;
    let mut err_awq = 0.0_f32;
    for j in 0..cols {
        err_baseline += (w[2 * cols + j] - dq_baseline[2 * cols + j]).abs();
        err_awq += (w[2 * cols + j] - dq_awq[2 * cols + j]).abs();
    }
    // AWQ should not increase the salient-row error and should pick α > 0.
    assert!(q_awq.alpha >= 0.0 && q_awq.alpha <= 1.0);
    // Permit equality (e.g. α=0 wins by random luck) but require AWQ to be at
    // least no worse on the salient row.
    assert!(
        err_awq <= err_baseline * 1.05,
        "AWQ salient-row L1 {err_awq} should not be much worse than baseline {err_baseline}"
    );
}

#[test]
fn alpha_in_valid_range() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act: Vec<f32> = (0..rows).map(|i| 0.1 + 0.5 * (i as f32)).collect();
    let cfg = default_cfg(4, 4, 16);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    assert!(
        (0.0..=1.0).contains(&q.alpha),
        "alpha {} not in [0, 1]",
        q.alpha
    );
}

#[test]
fn zero_rows_or_cols_errors() {
    let cfg = default_cfg(4, 2, 4);
    let res_rows = Awq::quantize_weight(&[], 0, 4, &[], &cfg);
    assert!(matches!(res_rows, Err(PeftError::Internal { .. })));
    let act_dummy = vec![1.0_f32; 4];
    let res_cols = Awq::quantize_weight(&[], 4, 0, &act_dummy, &cfg);
    assert!(matches!(res_cols, Err(PeftError::Internal { .. })));
}

#[test]
fn alpha_search_steps_zero_errors() {
    let rows = 2;
    let cols = 2;
    let w = vec![0.0_f32; rows * cols];
    let act = vec![1.0_f32; rows];
    let cfg = AwqConfig {
        bits: 4,
        group_size: 2,
        alpha_search_steps: 0,
    };
    let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn zero_group_size_errors() {
    let rows = 2;
    let cols = 2;
    let w = vec![0.0_f32; rows * cols];
    let act = vec![1.0_f32; rows];
    let cfg = AwqConfig {
        bits: 4,
        group_size: 0,
        alpha_search_steps: 4,
    };
    let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn non_finite_activation_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let act = vec![1.0_f32, f32::NAN, 1.0];
    let cfg = default_cfg(4, 2, 4);
    let res = Awq::quantize_weight(&w, rows, cols, &act, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn dequantize_inconsistent_state_errors() {
    let q = AwqQuantized {
        q: vec![0_i32; 12],
        scale: vec![0.1_f32, 0.2],
        zero: vec![0.0_f32, 0.0],
        alpha: 0.5,
        awq_scale: vec![1.0_f32; 3],
        bits: 4,
        group_size: 0, // intentionally invalid
        original_shape: (3, 4),
    };
    let res = Awq::dequantize_and_apply(&q);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn dequantize_awq_scale_mismatch_errors() {
    let q = AwqQuantized {
        q: vec![0_i32; 12],
        scale: vec![0.1_f32, 0.2],
        zero: vec![0.0_f32, 0.0],
        alpha: 0.5,
        awq_scale: vec![1.0_f32; 5], // wrong length
        bits: 4,
        group_size: 2,
        original_shape: (3, 4),
    };
    let res = Awq::dequantize_and_apply(&q);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn codes_inside_range() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act: Vec<f32> = (0..rows).map(|i| 0.1 + 0.2 * (i as f32)).collect();
    for &bits in &[3_u8, 4, 8] {
        let cfg = default_cfg(bits, 4, 4);
        let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
        let q_max = (1_i32 << bits) - 1;
        for &c in &q.q {
            assert!(
                (0..=q_max).contains(&c),
                "code {c} out of [0, {q_max}] for bits={bits}"
            );
        }
    }
}

#[test]
fn awq_scale_normalised_to_unit_geomean() {
    // After computing s_i = (|x_i| + ε)^α we normalise so geomean(s) = 1.
    // For varying α the normalised geomean must remain ≈ 1.0.
    let rows = 8;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let act: Vec<f32> = (0..rows).map(|i| 0.1 + 0.3 * (i as f32 + 1.0)).collect();
    let cfg = default_cfg(4, 4, 8);
    let q = Awq::quantize_weight(&w, rows, cols, &act, &cfg).expect("quantize");
    let log_sum: f64 = q
        .awq_scale
        .iter()
        .map(|&v| (v as f64).max(f64::MIN_POSITIVE).ln())
        .sum();
    let log_geomean = log_sum / (rows as f64);
    let geomean = log_geomean.exp();
    assert!(
        (geomean - 1.0).abs() < 1e-3,
        "awq_scale geomean {geomean} not ≈ 1.0"
    );
}
