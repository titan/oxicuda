//! Unit tests for [`super::loha`]. Split out via `#[path]` to keep `loha.rs` under
//! the 600-line per-file budget.

use super::loha::{LoHaAdapter, LoHaConfig};
use crate::error::PeftError;

fn default_cfg(in_f: usize, out_f: usize, rank: usize, alpha: f64) -> LoHaConfig {
    LoHaConfig {
        in_features: in_f,
        out_features: out_f,
        rank,
        alpha,
    }
}

#[test]
fn initial_forward_is_zero_with_zero_b() {
    let cfg = default_cfg(6, 4, 2, 4.0);
    let adapter = LoHaAdapter::new(cfg, 7).unwrap();
    let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
    let y = adapter.forward(&x).unwrap();
    assert_eq!(y.len(), 4);
    for &v in &y {
        assert!(v.abs() < 1e-15, "expected zero output, got {v}");
    }
}

#[test]
fn a_factors_reproducible_by_seed() {
    let cfg = default_cfg(8, 5, 3, 6.0);
    let a = LoHaAdapter::new(cfg.clone(), 42).unwrap();
    let b = LoHaAdapter::new(cfg, 42).unwrap();
    assert_eq!(a.a1, b.a1);
    assert_eq!(a.a2, b.a2);
    assert_eq!(a.b1, b.b1);
    assert_eq!(a.b2, b.b2);
}

#[test]
fn a1_and_a2_differ_within_same_seed() {
    let cfg = default_cfg(8, 5, 3, 6.0);
    let adapter = LoHaAdapter::new(cfg, 17).unwrap();
    let diff: f64 = adapter
        .a1
        .iter()
        .zip(adapter.a2.iter())
        .map(|(p, q)| (p - q).abs())
        .sum();
    assert!(diff > 1e-6, "A₁ and A₂ should differ within one seed");
}

#[test]
fn forward_dimensions_correct() {
    let cfg = default_cfg(7, 9, 3, 6.0);
    let mut adapter = LoHaAdapter::new(cfg, 11).unwrap();
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.05 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.03 * (i as f64 + 2.0);
    }
    let x = vec![1.0_f64; 7];
    let y = adapter.forward(&x).unwrap();
    assert_eq!(y.len(), 9);
}

#[test]
fn backward_grad_shapes_correct() {
    let cfg = default_cfg(5, 4, 2, 4.0);
    let adapter = LoHaAdapter::new(cfg, 3).unwrap();
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
    let grad_y = vec![0.1_f64, -0.2, 0.3, 0.4];
    let (da1, db1, da2, db2) = adapter.backward(&x, &grad_y).unwrap();
    assert_eq!(da1.len(), 2 * 5);
    assert_eq!(db1.len(), 4 * 2);
    assert_eq!(da2.len(), 2 * 5);
    assert_eq!(db2.len(), 4 * 2);
}

fn loss_at(a: &LoHaAdapter, x: &[f64], gy: &[f64]) -> f64 {
    gy.iter()
        .zip(a.forward(x).unwrap().iter())
        .map(|(g, y)| g * y)
        .sum()
}

fn check_fd(
    adapter: &mut LoHaAdapter,
    x: &[f64],
    gy: &[f64],
    selector: fn(&mut LoHaAdapter) -> &mut [f64],
    grads: &[f64],
    label: &str,
) {
    let eps = 1e-6_f64;
    for (k, &g_k) in grads.iter().enumerate() {
        let saved = selector(adapter)[k];
        selector(adapter)[k] = saved + eps;
        let lp = loss_at(adapter, x, gy);
        selector(adapter)[k] = saved - eps;
        let lm = loss_at(adapter, x, gy);
        selector(adapter)[k] = saved;
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - g_k).abs() < 1e-5,
            "{label}[{k}] FD={fd} analytic={g_k}"
        );
    }
}

#[test]
fn backward_matches_finite_differences() {
    let cfg = default_cfg(4, 3, 2, 4.0);
    let mut adapter = LoHaAdapter::new(cfg, 99).unwrap();
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.07 * (i as f64 + 2.0);
    }
    let x = vec![0.5_f64, -1.0, 0.25, 0.75];
    let gy = vec![1.0_f64, -0.5, 0.25];
    let (da1, db1, da2, db2) = adapter.backward(&x, &gy).unwrap();
    check_fd(&mut adapter, &x, &gy, |a| &mut a.a1, &da1, "a1");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.b1, &db1, "b1");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.a2, &da2, "a2");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.b2, &db2, "b2");
}

#[test]
fn sgd_reduces_loss_on_small_fit() {
    let mut adapter = LoHaAdapter::new(default_cfg(6, 4, 2, 4.0), 21).unwrap();
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
    // Both Hadamard branches need non-zero starting B so each receives gradient signal.
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.2 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.15 * (i as f64 + 1.0);
    }
    let target = {
        let mut probe = adapter.clone();
        for (i, b) in probe.b1.iter_mut().enumerate() {
            *b += 0.4 * (i as f64 + 1.0);
        }
        for (i, b) in probe.b2.iter_mut().enumerate() {
            *b += 0.3 * (i as f64 + 1.0);
        }
        probe.forward(&x).unwrap()
    };
    let mse = |a: &LoHaAdapter| -> f64 {
        a.forward(&x)
            .unwrap()
            .iter()
            .zip(target.iter())
            .map(|(p, q)| (p - q).powi(2))
            .sum()
    };
    let initial = mse(&adapter);
    for _ in 0..200 {
        let y = adapter.forward(&x).unwrap();
        let gy: Vec<f64> = y.iter().zip(target.iter()).map(|(p, q)| p - q).collect();
        let (da1, db1, da2, db2) = adapter.backward(&x, &gy).unwrap();
        adapter.apply_grads(&da1, &db1, &da2, &db2, 0.02).unwrap();
    }
    let final_loss = mse(&adapter);
    assert!(
        final_loss * 10.0 < initial,
        "loss {final_loss} should drop >10x from {initial}"
    );
}

#[test]
fn alpha_zero_produces_zero_forward() {
    let mut adapter = LoHaAdapter::new(default_cfg(5, 4, 2, 0.0), 77).unwrap();
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.07 * (i as f64 + 1.0);
    }
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
    let y = adapter.forward(&x).unwrap();
    for &v in &y {
        assert!(v.abs() < 1e-15, "α=0 must zero out adapter, got {v}");
    }
}

#[test]
fn invalid_configs_rejected() {
    for (i, o, r) in [(0, 4, 2), (4, 0, 2), (4, 4, 0)] {
        assert!(matches!(
            LoHaAdapter::new(default_cfg(i, o, r, 1.0), 0),
            Err(PeftError::EmptyInput)
        ));
    }
    for (i, o, r) in [(3, 8, 5), (8, 3, 5)] {
        assert!(matches!(
            LoHaAdapter::new(default_cfg(i, o, r, 1.0), 0),
            Err(PeftError::RankTooLarge { .. })
        ));
    }
}

#[test]
fn dim_mismatch_in_forward_and_backward_rejected() {
    let adapter = LoHaAdapter::new(default_cfg(5, 3, 2, 2.0), 0).unwrap();
    assert!(matches!(
        adapter.forward(&[1.0_f64, 2.0, 3.0]),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.backward(&[0.1_f64; 5], &[0.1_f64; 2]),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.backward(&[0.1_f64; 4], &[0.1_f64; 3]),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn apply_grads_dim_mismatch_rejected() {
    let mut adapter = LoHaAdapter::new(default_cfg(5, 3, 2, 2.0), 0).unwrap();
    let good_a = vec![0.0_f64; 2 * 5];
    let good_b = vec![0.0_f64; 3 * 2];
    let bad_a = vec![0.0_f64; 5];
    let bad_b = vec![0.0_f64; 5];
    assert!(matches!(
        adapter.apply_grads(&bad_a, &good_b, &good_a, &good_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.apply_grads(&good_a, &bad_b, &good_a, &good_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.apply_grads(&good_a, &good_b, &bad_a, &good_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.apply_grads(&good_a, &good_b, &good_a, &bad_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn scale_alpha_over_rank_applied() {
    let mut a1 = LoHaAdapter::new(default_cfg(5, 3, 2, 4.0), 33).unwrap();
    let mut a2 = LoHaAdapter::new(default_cfg(5, 3, 2, 8.0), 33).unwrap();
    let b_seed: Vec<f64> = (0..a1.b1.len()).map(|i| 0.05 * (i as f64 + 1.0)).collect();
    a1.b1.copy_from_slice(&b_seed);
    a2.b1.copy_from_slice(&b_seed);
    let b_seed2: Vec<f64> = (0..a1.b2.len()).map(|i| 0.07 * (i as f64 + 1.0)).collect();
    a1.b2.copy_from_slice(&b_seed2);
    a2.b2.copy_from_slice(&b_seed2);
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
    let y1 = a1.forward(&x).unwrap();
    let y2 = a2.forward(&x).unwrap();
    for (v1, v2) in y1.iter().zip(y2.iter()) {
        assert!((2.0 * v1 - v2).abs() < 1e-12, "α doubled → y doubled");
    }
    assert!((a1.scale() - 2.0).abs() < 1e-15);
    assert!((a2.scale() - 4.0).abs() < 1e-15);
}

#[test]
fn zero_input_yields_zero_output() {
    let mut adapter = LoHaAdapter::new(default_cfg(5, 4, 2, 4.0), 13).unwrap();
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.07 * (i as f64 + 2.0);
    }
    let x = vec![0.0_f64; 5];
    let y = adapter.forward(&x).unwrap();
    for &v in &y {
        assert!(v.abs() < 1e-15, "zero x must yield zero y, got {v}");
    }
}

#[test]
fn multiple_forward_calls_dont_mutate_state() {
    let mut adapter = LoHaAdapter::new(default_cfg(6, 5, 3, 6.0), 13).unwrap();
    for (i, b) in adapter.b1.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    for (i, b) in adapter.b2.iter_mut().enumerate() {
        *b = 0.05 * (i as f64 + 2.0);
    }
    let snap_a1 = adapter.a1.clone();
    let snap_b1 = adapter.b1.clone();
    let snap_a2 = adapter.a2.clone();
    let snap_b2 = adapter.b2.clone();
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
    let _ = adapter.forward(&x).unwrap();
    let _ = adapter.forward(&x).unwrap();
    let _ = adapter.forward(&x).unwrap();
    assert_eq!(adapter.a1, snap_a1);
    assert_eq!(adapter.b1, snap_b1);
    assert_eq!(adapter.a2, snap_a2);
    assert_eq!(adapter.b2, snap_b2);
}

#[test]
fn n_trainable_counts_all_four_factors() {
    let adapter = LoHaAdapter::new(default_cfg(8, 12, 4, 8.0), 0).unwrap();
    assert_eq!(adapter.n_trainable(), 2 * 4 * (8 + 12));
}
