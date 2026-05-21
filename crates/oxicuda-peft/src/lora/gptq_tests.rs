//! Unit tests for [`super::gptq`]. Split into a sibling file to keep `gptq.rs`
//! well within the per-file budget while still bundling a comprehensive battery
//! of correctness, dimension, and numerical-stability checks.

use super::gptq::{Gptq, GptqConfig, GptqQuantized};
use crate::error::PeftError;

/// Build a deterministic `rows × cols` weight matrix in `[-amp, amp]` whose entries
/// blend a linear ramp with a low-frequency sinusoid.
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

fn default_cfg(bits: u8, group_size: usize) -> GptqConfig {
    GptqConfig {
        bits,
        group_size,
        damp_percent: 0.01,
        act_order: false,
        blocksize: 64,
    }
}

#[test]
fn identity_hessian_matches_naive_minmax_quantization() {
    let rows = 6;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols]; // identical diagonal — no preferential ordering
    let cfg = default_cfg(8, cols); // single group covers all columns
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    // For a single group, dequantization should track the per-column min/max grid.
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    assert!(
        err < r * 1e-2,
        "identity Hessian L∞ err={err} not within 1% of weight range {r}"
    );
}

#[test]
fn round_trip_8bit_below_one_percent() {
    let rows = 8;
    let cols = 32;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(8, 16);
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    assert!(
        err < r * 1e-2,
        "8-bit GPTQ L∞ err={err} not within 1% of range {r}"
    );
}

#[test]
fn round_trip_2bit_error_bounded_by_scale() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(2, 8);
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    let err = linf(&w, &dq);
    // For 2-bit affine over an 8-column group, the worst-case error is bounded by the
    // largest per-group scale. We bound by the matrix range itself, which must hold.
    let r = weight_range(&w);
    assert!(err.is_finite(), "2-bit GPTQ produced non-finite output");
    assert!(err <= r + 1e-6, "2-bit GPTQ err={err} exceeds range {r}");
    // Sanity: bits-2 should *not* be exact.
    assert!(
        err > 0.0,
        "expected 2-bit quantization to leave residual error"
    );
}

#[test]
fn group_size_greater_than_cols_clamps_to_cols() {
    let rows = 3;
    let cols = 7;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    let mut cfg = default_cfg(4, 999); // wildly oversized
    cfg.blocksize = 999;
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
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
fn damp_percent_no_nan_with_zero_hessian_proxy() {
    let rows = 4;
    let cols = 8;
    let w = synthetic_weight(rows, cols, 0.5);
    let xtx = vec![0.0_f32; cols]; // all-zero diagonal → relies entirely on damping
    let mut cfg = default_cfg(4, 4);
    cfg.damp_percent = 0.05;
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    for &v in &dq {
        assert!(v.is_finite(), "GPTQ produced non-finite value {v}");
    }
}

#[test]
fn xtx_diag_length_mismatch_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let xtx = vec![1.0_f32; cols + 1]; // wrong length
    let cfg = default_cfg(4, 2);
    let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn weight_length_mismatch_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols + 1]; // off by one
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(4, 2);
    let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn zero_rows_or_cols_errors() {
    let w = vec![0.0_f32; 4];
    let xtx = vec![1.0_f32; 4];
    let cfg = default_cfg(4, 2);
    let res_rows = Gptq::quantize_weight(&w, 0, 4, &xtx, &cfg);
    assert!(matches!(res_rows, Err(PeftError::Internal { .. })));
    let res_cols = Gptq::quantize_weight(&w, 4, 0, &[], &cfg);
    assert!(matches!(res_cols, Err(PeftError::Internal { .. })));
}

#[test]
fn deterministic_quantization() {
    let rows = 5;
    let cols = 12;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx: Vec<f32> = (0..cols).map(|j| 0.1 + j as f32 * 0.01).collect();
    let cfg = default_cfg(4, 4);
    let a = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("a");
    let b = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("b");
    assert_eq!(a.q, b.q);
    assert_eq!(a.scale, b.scale);
    assert_eq!(a.zero, b.zero);
    assert_eq!(a.bits, b.bits);
    assert_eq!(a.group_size, b.group_size);
    assert_eq!(a.original_shape, b.original_shape);
}

#[test]
fn idempotent_under_requantization() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(4, 8);
    let q1 = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("q1");
    let dq1 = Gptq::dequantize(&q1).expect("dq1");
    let q2 = Gptq::quantize_weight(&dq1, rows, cols, &xtx, &cfg).expect("q2");
    assert_eq!(
        q1.q, q2.q,
        "re-quantizing a dequantized weight should give same codes"
    );
    for (a, b) in q1.scale.iter().zip(q2.scale.iter()) {
        assert!((a - b).abs() < 1e-5, "scale drift {} → {}", a, b);
    }
    for (a, b) in q1.zero.iter().zip(q2.zero.iter()) {
        assert!((a - b).abs() < 1e-5, "zero drift {} → {}", a, b);
    }
}

#[test]
fn act_order_preserves_shape() {
    let rows = 4;
    let cols = 12;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx: Vec<f32> = (0..cols).map(|j| 1.0 + j as f32).collect();
    let mut cfg = default_cfg(4, 4);
    cfg.act_order = true;
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    assert_eq!(q.original_shape, (rows, cols));
    assert_eq!(q.q.len(), rows * cols);
    let dq = Gptq::dequantize(&q).expect("dequant");
    assert_eq!(dq.len(), rows * cols);
}

#[test]
fn single_column_degenerate_case() {
    let rows = 6;
    let cols = 1;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(4, 1);
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    assert_eq!(q.q.len(), rows);
    assert_eq!(q.scale.len(), 1);
    assert_eq!(q.zero.len(), 1);
    let dq = Gptq::dequantize(&q).expect("dequant");
    assert_eq!(dq.len(), rows);
}

#[test]
fn invalid_bits_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let xtx = vec![1.0_f32; cols];
    for &bits in &[0_u8, 1, 5, 6, 7, 9, 16] {
        let cfg = GptqConfig {
            bits,
            group_size: 4,
            damp_percent: 0.01,
            act_order: false,
            blocksize: 4,
        };
        let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for bits={bits}"
        );
    }
}

#[test]
fn zero_range_column_does_not_divide_by_zero() {
    // First column is constant zero (zero range); the rest carries useful weight.
    let rows = 5;
    let cols = 6;
    let mut w = synthetic_weight(rows, cols, 1.0);
    for i in 0..rows {
        w[i * cols] = 0.0;
    }
    let xtx = vec![1.0_f32; cols];
    let cfg = default_cfg(4, cols); // single group so zero column is part of the (scale, zero)
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    for &v in &dq {
        assert!(v.is_finite(), "zero-range column produced non-finite {v}");
    }
}

#[test]
fn dequantize_inconsistent_state_errors() {
    let q = GptqQuantized {
        q: vec![0_i32; 12],
        scale: vec![0.1_f32, 0.2],
        zero: vec![0.0_f32, 0.0],
        bits: 4,
        group_size: 0, // intentionally invalid
        original_shape: (3, 4),
        perm: None,
    };
    let res = Gptq::dequantize(&q);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn dequantize_wrong_length_errors() {
    let q = GptqQuantized {
        q: vec![0_i32; 11], // wrong: should be 3*4=12
        scale: vec![0.1_f32],
        zero: vec![0.0_f32],
        bits: 4,
        group_size: 4,
        original_shape: (3, 4),
        perm: None,
    };
    let res = Gptq::dequantize(&q);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn codes_inside_legal_range() {
    let rows = 4;
    let cols = 16;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx = vec![1.0_f32; cols];
    for &bits in &[2_u8, 3, 4, 8] {
        let cfg = default_cfg(bits, 4);
        let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
        let q_max = (1_i32 << bits) - 1;
        for &code in &q.q {
            assert!(
                (0..=q_max).contains(&code),
                "code {code} out of [0, {q_max}] for bits={bits}"
            );
        }
    }
}

#[test]
fn blocksize_zero_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let xtx = vec![1.0_f32; cols];
    let cfg = GptqConfig {
        bits: 4,
        group_size: 2,
        damp_percent: 0.01,
        act_order: false,
        blocksize: 0,
    };
    let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn non_positive_damp_percent_errors() {
    let rows = 3;
    let cols = 4;
    let w = vec![0.0_f32; rows * cols];
    let xtx = vec![1.0_f32; cols];
    let mut cfg = default_cfg(4, 2);
    cfg.damp_percent = 0.0;
    let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));

    let mut cfg = default_cfg(4, 2);
    cfg.damp_percent = -1e-3;
    let res = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg);
    assert!(matches!(res, Err(PeftError::Internal { .. })));
}

#[test]
fn act_order_dequant_close_to_w_for_8bit() {
    let rows = 4;
    let cols = 12;
    let w = synthetic_weight(rows, cols, 1.0);
    let xtx: Vec<f32> = (0..cols).map(|j| (j as f32 + 1.0).powi(2)).collect();
    let mut cfg = default_cfg(8, 4);
    cfg.act_order = true;
    let q = Gptq::quantize_weight(&w, rows, cols, &xtx, &cfg).expect("quantize");
    let dq = Gptq::dequantize(&q).expect("dequant");
    let r = weight_range(&w);
    let err = linf(&w, &dq);
    assert!(
        err < r * 5e-2,
        "act_order 8-bit L∞ err={err} not within 5% of range {r}"
    );
}
