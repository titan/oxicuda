//! Tests for LST (Ladder Side-Tuning) — [`super::lst`].

use super::lst::{LadderSideTuning, LstConfig};
use crate::error::PeftError;
use crate::handle::PeftHandle;

fn make_handle(seed: u64) -> PeftHandle {
    PeftHandle::new(80, seed)
}

fn default_cfg(d_trunk: usize, d_side: usize, num_layers: usize) -> LstConfig {
    LstConfig {
        d_trunk,
        d_side,
        num_layers,
        gate_init: 0.5,
    }
}

// ── Test 1 ─────────────────────────────────────────────────────────────────
/// `new` with `num_layers = 0` must return an error.
#[test]
fn new_zero_layers_errors() {
    let cfg = default_cfg(8, 4, 0);
    let mut h = make_handle(1);
    let res = LadderSideTuning::new(cfg, &mut h);
    assert!(
        matches!(res, Err(PeftError::InvalidTargetRank { .. })),
        "expected InvalidTargetRank, got {:?}",
        res
    );
}

// ── Test 2 ─────────────────────────────────────────────────────────────────
/// `new` with `d_side = 0` must return an error.
#[test]
fn new_zero_d_side_errors() {
    let cfg = default_cfg(8, 0, 2);
    let mut h = make_handle(2);
    let res = LadderSideTuning::new(cfg, &mut h);
    assert!(
        matches!(res, Err(PeftError::InvalidTargetRank { .. })),
        "expected InvalidTargetRank, got {:?}",
        res
    );
}

// ── Test 3 ─────────────────────────────────────────────────────────────────
/// `new` with `d_trunk = 0` must return an error.
#[test]
fn new_zero_d_trunk_errors() {
    let cfg = default_cfg(0, 4, 2);
    let mut h = make_handle(3);
    let res = LadderSideTuning::new(cfg, &mut h);
    assert!(
        matches!(res, Err(PeftError::InvalidTargetRank { .. })),
        "expected InvalidTargetRank, got {:?}",
        res
    );
}

// ── Test 4 ─────────────────────────────────────────────────────────────────
/// `forward_layer` output has length `seq_len × d_side`.
#[test]
fn forward_layer_output_length() {
    let d_trunk = 8;
    let d_side = 4;
    let seq_len = 3;
    let cfg = default_cfg(d_trunk, d_side, 1);
    let mut h = make_handle(4);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let trunk_hidden = vec![0.1_f32; seq_len * d_trunk];
    let side_state = vec![0.0_f32; seq_len * d_side];
    let out = lst
        .forward_layer(0, &trunk_hidden, &side_state, seq_len)
        .unwrap();
    assert_eq!(out.len(), seq_len * d_side);
}

// ── Test 5 ─────────────────────────────────────────────────────────────────
/// When `gate = 1.0` and `up_w` is zero-initialized, `final_output` returns
/// exactly `trunk_final` (since `(1−1)*up(·) = 0`).
#[test]
fn final_output_gate_one_returns_trunk() {
    let d_trunk = 6;
    let d_side = 3;
    let seq_len = 2;
    let cfg = LstConfig {
        d_trunk,
        d_side,
        num_layers: 1,
        gate_init: 1.0,
    };
    let mut h = make_handle(5);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    // up_w is zero by construction, so the up-projection contributes only up_b (=0).
    let side_state = vec![0.5_f32; seq_len * d_side];
    let trunk_final: Vec<f32> = (0..seq_len * d_trunk).map(|i| i as f32 * 0.1).collect();

    let out = lst
        .final_output(&side_state, &trunk_final, seq_len)
        .unwrap();
    assert_eq!(out.len(), trunk_final.len());
    for (a, b) in out.iter().zip(trunk_final.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "gate=1.0 output differs from trunk: {a} vs {b}"
        );
    }
}

// ── Test 6 ─────────────────────────────────────────────────────────────────
/// When `gate = 0.0` and `up_w = 0`, `final_output` returns the up-projected
/// side state (which equals `up_b = 0` since `up_w` is zero-initialized).
#[test]
fn final_output_gate_zero_uses_side() {
    let d_trunk = 4;
    let d_side = 2;
    let seq_len = 2;
    let cfg = LstConfig {
        d_trunk,
        d_side,
        num_layers: 1,
        gate_init: 0.0,
    };
    let mut h = make_handle(6);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let side_state = vec![1.0_f32; seq_len * d_side];
    let trunk_final = vec![99.0_f32; seq_len * d_trunk];

    // up_w = 0 and up_b = 0 → output should be 0 (gate=0 ignores trunk).
    let out = lst
        .final_output(&side_state, &trunk_final, seq_len)
        .unwrap();
    for &v in &out {
        assert!(
            v.abs() < 1e-6,
            "gate=0.0 with zero up_w should give zero output, got {v}"
        );
    }
}

// ── Test 7 ─────────────────────────────────────────────────────────────────
/// Zero side state: output is determined solely by the trunk hidden state
/// through the down-projection path.
#[test]
fn zero_side_state_output_determined_by_trunk() {
    let d_trunk = 6;
    let d_side = 3;
    let seq_len = 1;
    let cfg = default_cfg(d_trunk, d_side, 1);
    let mut h = make_handle(7);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let trunk_hidden: Vec<f32> = (0..d_trunk).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let zero_side = vec![0.0_f32; d_side];

    let out1 = lst
        .forward_layer(0, &trunk_hidden, &zero_side, seq_len)
        .unwrap();
    // Non-zero trunk + zero side → non-zero output (down_w is Kaiming-initialized).
    // The output is purely from the down projection through side_w, starting from zero side.
    assert_eq!(out1.len(), seq_len * d_side);
    // With all-zero trunk this would also be zero (just bias path).
    let zero_trunk = vec![0.0_f32; d_trunk];
    let out2 = lst
        .forward_layer(0, &zero_trunk, &zero_side, seq_len)
        .unwrap();
    // side residual from zero everything is just 0 + side_b (=0) → all zero.
    for &v in &out2 {
        assert!(
            v.abs() < 1e-6,
            "all-zero inputs should give zero output, got {v}"
        );
    }
}

// ── Test 8 ─────────────────────────────────────────────────────────────────
/// `total_params` matches the analytic formula.
#[test]
fn total_params_analytic() {
    let d_trunk = 8;
    let d_side = 4;
    let num_layers = 3;
    let cfg = default_cfg(d_trunk, d_side, num_layers);
    let mut h = make_handle(8);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let per_block = d_side * d_trunk   // down_w
        + d_side                        // down_b
        + d_side * d_side               // side_w
        + d_side                        // side_b
        + d_trunk * d_side              // up_w
        + d_trunk                       // up_b
        + 1; // gate
    assert_eq!(lst.total_params(), per_block * num_layers);
}

// ── Test 9 ─────────────────────────────────────────────────────────────────
/// Deterministic output for the same PeftHandle seed.
#[test]
fn deterministic_same_seed() {
    let cfg = default_cfg(8, 4, 2);
    let mut h1 = make_handle(9);
    let mut h2 = make_handle(9);
    let lst1 = LadderSideTuning::new(cfg.clone(), &mut h1).unwrap();
    let lst2 = LadderSideTuning::new(cfg, &mut h2).unwrap();

    let trunk = vec![0.3_f32; 8];
    let side = vec![0.1_f32; 4];
    let out1 = lst1.forward_layer(0, &trunk, &side, 1).unwrap();
    let out2 = lst2.forward_layer(0, &trunk, &side, 1).unwrap();
    assert_eq!(out1, out2, "same seed must produce identical output");
}

// ── Test 10 ────────────────────────────────────────────────────────────────
/// `layer >= num_layers` → `LayerOutOfRange`.
#[test]
fn forward_layer_out_of_range() {
    let cfg = default_cfg(8, 4, 2);
    let mut h = make_handle(10);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let trunk = vec![0.0_f32; 8];
    let side = vec![0.0_f32; 4];
    let res = lst.forward_layer(2, &trunk, &side, 1);
    assert!(
        matches!(res, Err(PeftError::LayerOutOfRange { .. })),
        "expected LayerOutOfRange, got {:?}",
        res
    );
}

// ── Test 11 ────────────────────────────────────────────────────────────────
/// Wrong `trunk_hidden` length → `DimensionMismatch`.
#[test]
fn forward_layer_trunk_dim_mismatch() {
    let cfg = default_cfg(8, 4, 1);
    let mut h = make_handle(11);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let bad_trunk = vec![0.0_f32; 5]; // should be 1 * 8 = 8
    let side = vec![0.0_f32; 4];
    let res = lst.forward_layer(0, &bad_trunk, &side, 1);
    assert!(
        matches!(res, Err(PeftError::DimensionMismatch { .. })),
        "expected DimensionMismatch for trunk_hidden, got {:?}",
        res
    );
}

// ── Test 12 ────────────────────────────────────────────────────────────────
/// Wrong `side_state` length in `forward_layer` → `DimensionMismatch`.
#[test]
fn forward_layer_side_dim_mismatch() {
    let cfg = default_cfg(8, 4, 1);
    let mut h = make_handle(12);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let trunk = vec![0.0_f32; 8];
    let bad_side = vec![0.0_f32; 3]; // should be 1 * 4 = 4
    let res = lst.forward_layer(0, &trunk, &bad_side, 1);
    assert!(
        matches!(res, Err(PeftError::DimensionMismatch { .. })),
        "expected DimensionMismatch for side_state, got {:?}",
        res
    );
}

// ── Test 13 ────────────────────────────────────────────────────────────────
/// Wrong `side_state` length in `final_output` → `DimensionMismatch`.
#[test]
fn final_output_side_dim_mismatch() {
    let cfg = default_cfg(8, 4, 1);
    let mut h = make_handle(13);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let bad_side = vec![0.0_f32; 3]; // should be 1 * 4 = 4
    let trunk_final = vec![0.0_f32; 8];
    let res = lst.final_output(&bad_side, &trunk_final, 1);
    assert!(
        matches!(res, Err(PeftError::DimensionMismatch { .. })),
        "expected DimensionMismatch for side_state in final_output, got {:?}",
        res
    );
}

// ── Test 14 ────────────────────────────────────────────────────────────────
/// After running `forward_layer` through all layers, `final_output` succeeds.
#[test]
fn full_forward_through_all_layers_then_final_output() {
    let d_trunk = 6;
    let d_side = 3;
    let num_layers = 3;
    let seq_len = 2;
    let cfg = default_cfg(d_trunk, d_side, num_layers);
    let mut h = make_handle(14);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    // Simulate frozen trunk hidden states per layer.
    let trunk_hiddens: Vec<Vec<f32>> = (0..num_layers)
        .map(|l| vec![0.1_f32 * (l as f32 + 1.0); seq_len * d_trunk])
        .collect();

    let mut side = vec![0.0_f32; seq_len * d_side];
    for (layer, trunk) in trunk_hiddens.iter().enumerate() {
        side = lst.forward_layer(layer, trunk, &side, seq_len).unwrap();
        assert_eq!(side.len(), seq_len * d_side);
    }

    let trunk_final = vec![0.5_f32; seq_len * d_trunk];
    let out = lst.final_output(&side, &trunk_final, seq_len).unwrap();
    assert_eq!(out.len(), seq_len * d_trunk);
}

// ── Test 15 ────────────────────────────────────────────────────────────────
/// 3-layer LST: each `forward_layer` output has dimension `seq_len × d_side`.
#[test]
fn three_layer_forward_correct_dims() {
    let d_trunk = 10;
    let d_side = 5;
    let num_layers = 3;
    let seq_len = 4;
    let cfg = default_cfg(d_trunk, d_side, num_layers);
    let mut h = make_handle(15);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let mut side = vec![0.0_f32; seq_len * d_side];
    for layer in 0..num_layers {
        let trunk = vec![0.0_f32; seq_len * d_trunk];
        let new_side = lst.forward_layer(layer, &trunk, &side, seq_len).unwrap();
        assert_eq!(
            new_side.len(),
            seq_len * d_side,
            "layer {layer} output dim wrong"
        );
        side = new_side;
    }
}

// ── Test 16 ────────────────────────────────────────────────────────────────
/// `total_params` increases with more layers.
#[test]
fn total_params_increases_with_layers() {
    let d_trunk = 8;
    let d_side = 4;

    let cfg1 = default_cfg(d_trunk, d_side, 1);
    let cfg2 = default_cfg(d_trunk, d_side, 2);
    let mut h1 = make_handle(16);
    let mut h2 = make_handle(16);
    let lst1 = LadderSideTuning::new(cfg1, &mut h1).unwrap();
    let lst2 = LadderSideTuning::new(cfg2, &mut h2).unwrap();
    assert!(
        lst2.total_params() > lst1.total_params(),
        "2-layer should have more params than 1-layer"
    );
}

// ── Test 17 ────────────────────────────────────────────────────────────────
/// Wrong `trunk_final` length in `final_output` → `DimensionMismatch`.
#[test]
fn final_output_trunk_dim_mismatch() {
    let cfg = default_cfg(8, 4, 1);
    let mut h = make_handle(17);
    let lst = LadderSideTuning::new(cfg, &mut h).unwrap();

    let side = vec![0.0_f32; 4]; // seq_len=1, d_side=4
    let bad_trunk = vec![0.0_f32; 5]; // should be 1 * 8 = 8
    let res = lst.final_output(&side, &bad_trunk, 1);
    assert!(
        matches!(res, Err(PeftError::DimensionMismatch { .. })),
        "expected DimensionMismatch for trunk_final, got {:?}",
        res
    );
}
