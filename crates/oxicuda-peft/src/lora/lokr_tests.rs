//! Unit tests for [`super::lokr`]. Split out via `#[path]` to keep `lokr.rs` under
//! the 600-line per-file budget.

use super::lokr::{LoKrAdapter, LoKrConfig};
use crate::error::PeftError;

fn default_cfg(
    in_f: usize,
    out_f: usize,
    m: (usize, usize),
    n: (usize, usize),
    rank: usize,
    alpha: f64,
) -> LoKrConfig {
    LoKrConfig {
        in_features: in_f,
        out_features: out_f,
        m1: m.0,
        m2: m.1,
        n1: n.0,
        n2: n.1,
        rank,
        alpha,
    }
}

#[test]
fn initial_forward_is_zero_with_zero_b() {
    let cfg = default_cfg(6, 8, (2, 4), (3, 2), 2, 4.0);
    let adapter =
        LoKrAdapter::new(cfg, 7).expect("LoKrAdapter creation should succeed with valid config");
    let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
    let y = adapter
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    assert_eq!(y.len(), 8);
    for &v in &y {
        assert!(v.abs() < 1e-15, "expected zero output, got {v}");
    }
}

#[test]
fn reproducible_by_seed() {
    let cfg = default_cfg(6, 8, (2, 4), (3, 2), 2, 4.0);
    let a = LoKrAdapter::new(cfg.clone(), 42)
        .expect("LoKrAdapter a creation should succeed with valid config");
    let b =
        LoKrAdapter::new(cfg, 42).expect("LoKrAdapter b creation should succeed with valid config");
    assert_eq!(a.w1, b.w1);
    assert_eq!(a.a, b.a);
    assert_eq!(a.b, b.b);
}

#[test]
fn out_features_kronecker_mismatch_rejected() {
    let cfg = default_cfg(6, 9, (2, 4), (3, 2), 2, 4.0);
    assert!(matches!(
        LoKrAdapter::new(cfg, 0),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn in_features_kronecker_mismatch_rejected() {
    let cfg = default_cfg(7, 8, (2, 4), (3, 2), 2, 4.0);
    assert!(matches!(
        LoKrAdapter::new(cfg, 0),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn forward_dimensions_correct() {
    let cfg = default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0);
    let mut adapter =
        LoKrAdapter::new(cfg, 11).expect("LoKrAdapter creation should succeed with valid config");
    for (i, b) in adapter.b.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    let x = vec![1.0_f64; 6];
    let y = adapter
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    assert_eq!(y.len(), 6);
}

#[test]
fn backward_shapes_correct() {
    let cfg = default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0);
    let adapter =
        LoKrAdapter::new(cfg, 3).expect("LoKrAdapter creation should succeed with valid config");
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.7];
    let grad_y = vec![0.1_f64, -0.2, 0.3, 0.4, 0.1, -0.05];
    let (dw1, da, db) = adapter
        .backward(&x, &grad_y)
        .expect("backward pass should succeed with valid inputs");
    assert_eq!(dw1.len(), 3 * 2);
    assert_eq!(da.len(), 2 * 3);
    assert_eq!(db.len(), 2 * 2);
}

/// Build the explicit Kronecker product `M = W₁ ⊗ W₂` row-major.
fn naive_kron(w1: &[f64], w2: &[f64], m1: usize, n1: usize, m2: usize, n2: usize) -> Vec<f64> {
    let out = m1 * m2;
    let in_f = n1 * n2;
    let mut m = vec![0.0_f64; out * in_f];
    for o1 in 0..m1 {
        for o2 in 0..m2 {
            let row = (o1 * m2 + o2) * in_f;
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    m[row + i1 * n2 + i2] = w1[o1 * n1 + i1] * w2[o2 * n2 + i2];
                }
            }
        }
    }
    m
}

#[test]
fn block_form_matches_naive_kronecker() {
    let cfg = default_cfg(4, 4, (2, 2), (2, 2), 2, 4.0);
    let mut adapter =
        LoKrAdapter::new(cfg, 55).expect("LoKrAdapter creation should succeed with valid config");
    for (i, b) in adapter.b.iter_mut().enumerate() {
        *b = 0.2 * (i as f64 + 1.0);
    }
    let x = vec![0.5_f64, -1.0, 0.25, 0.75];
    let s = adapter.scale();
    let w2 = adapter.w2();
    let m = naive_kron(&adapter.w1, &w2, 2, 2, 2, 2);
    let mut y_naive = [0.0_f64; 4];
    for (o, y_o) in y_naive.iter_mut().enumerate() {
        let row = o * 4;
        let mut acc = 0.0_f64;
        for (i, x_i) in x.iter().enumerate() {
            acc += m[row + i] * x_i;
        }
        *y_o = s * acc;
    }
    let y_block = adapter
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    for (a, b) in y_naive.iter().zip(y_block.iter()) {
        assert!((a - b).abs() < 1e-8, "block={b} naive={a}");
    }
}

fn loss_at(a: &LoKrAdapter, x: &[f64], gy: &[f64]) -> f64 {
    gy.iter()
        .zip(
            a.forward(x)
                .expect("forward pass should succeed in loss_at helper")
                .iter(),
        )
        .map(|(g, y)| g * y)
        .sum()
}

fn check_fd(
    adapter: &mut LoKrAdapter,
    x: &[f64],
    gy: &[f64],
    selector: fn(&mut LoKrAdapter) -> &mut [f64],
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
    let cfg = default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0);
    let mut adapter =
        LoKrAdapter::new(cfg, 99).expect("LoKrAdapter creation should succeed with valid config");
    for (i, b) in adapter.b.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    adapter.a[0] += 0.05;
    adapter.a[3] -= 0.07;
    let x = vec![0.5_f64, -1.0, 0.25, 0.75, 0.4, -0.6];
    let gy = vec![1.0_f64, -0.5, 0.25, 0.4, -0.3, 0.2];
    let (dw1, da, db) = adapter
        .backward(&x, &gy)
        .expect("backward pass should succeed with valid inputs");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.w1, &dw1, "w1");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.a, &da, "a");
    check_fd(&mut adapter, &x, &gy, |a| &mut a.b, &db, "b");
}

#[test]
fn sgd_reduces_loss_on_small_fit() {
    let mut adapter = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0), 21)
        .expect("LoKrAdapter creation should succeed with valid config");
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
    let target = {
        let mut probe = adapter.clone();
        for (i, b) in probe.b.iter_mut().enumerate() {
            *b = 0.4 * (i as f64 + 1.0);
        }
        probe
            .forward(&x)
            .expect("probe forward pass should succeed")
    };
    let mse = |a: &LoKrAdapter| -> f64 {
        a.forward(&x)
            .expect("forward pass should succeed in mse closure")
            .iter()
            .zip(target.iter())
            .map(|(p, q)| (p - q).powi(2))
            .sum()
    };
    // Nudge B off zero so dW₁ is non-trivial in the first step.
    for (i, b) in adapter.b.iter_mut().enumerate() {
        *b = 0.01 * (i as f64 + 1.0);
    }
    let initial = mse(&adapter);
    for _ in 0..200 {
        let y = adapter
            .forward(&x)
            .expect("forward pass should succeed in training loop");
        let gy: Vec<f64> = y.iter().zip(target.iter()).map(|(p, q)| p - q).collect();
        let (dw1, da, db) = adapter
            .backward(&x, &gy)
            .expect("backward pass should succeed in training loop");
        adapter
            .apply_grads(&dw1, &da, &db, 0.05)
            .expect("gradient application should succeed");
    }
    let final_loss = mse(&adapter);
    assert!(
        final_loss * 10.0 < initial,
        "loss {final_loss} should drop >10x from {initial}"
    );
}

#[test]
fn alpha_zero_produces_zero_forward() {
    let mut adapter = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 0.0), 77)
        .expect("LoKrAdapter creation should succeed with alpha=0");
    for (i, b) in adapter.b.iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.7];
    let y = adapter
        .forward(&x)
        .expect("forward pass should succeed with alpha=0");
    for &v in &y {
        assert!(v.abs() < 1e-15, "α=0 must zero out adapter, got {v}");
    }
}

#[test]
fn dim_mismatch_in_forward_and_backward_rejected() {
    let adapter = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 2.0), 0)
        .expect("LoKrAdapter creation should succeed with valid config");
    assert!(matches!(
        adapter.forward(&[1.0_f64; 5]),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.backward(&[0.1_f64; 5], &[0.1_f64; 6]),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.backward(&[0.1_f64; 6], &[0.1_f64; 4]),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn apply_grads_dim_mismatch_rejected() {
    let mut adapter = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 2.0), 0)
        .expect("LoKrAdapter creation should succeed with valid config");
    let good_w1 = vec![0.0_f64; 3 * 2];
    let good_a = vec![0.0_f64; 2 * 3];
    let good_b = vec![0.0_f64; 2 * 2];
    let bad = vec![0.0_f64; 5];
    assert!(matches!(
        adapter.apply_grads(&bad, &good_a, &good_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.apply_grads(&good_w1, &bad, &good_b, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        adapter.apply_grads(&good_w1, &good_a, &bad, 0.1),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn invalid_configs_rejected() {
    for cfg in [
        default_cfg(0, 6, (3, 2), (2, 3), 2, 1.0),
        default_cfg(6, 0, (3, 2), (2, 3), 2, 1.0),
        default_cfg(6, 6, (0, 2), (2, 3), 2, 1.0),
        default_cfg(6, 6, (3, 0), (2, 3), 2, 1.0),
        default_cfg(6, 6, (3, 2), (0, 3), 2, 1.0),
        default_cfg(6, 6, (3, 2), (2, 0), 2, 1.0),
        default_cfg(6, 6, (3, 2), (2, 3), 0, 1.0),
    ] {
        assert!(matches!(
            LoKrAdapter::new(cfg, 0),
            Err(PeftError::EmptyInput)
        ));
    }
    let cfg = default_cfg(4, 4, (2, 2), (2, 2), 3, 1.0);
    assert!(matches!(
        LoKrAdapter::new(cfg, 0),
        Err(PeftError::RankTooLarge { .. })
    ));
}

#[test]
fn scale_alpha_over_rank_applied() {
    let mut a1 = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0), 33)
        .expect("a1 LoKrAdapter creation should succeed with valid config");
    let mut a2 = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 8.0), 33)
        .expect("a2 LoKrAdapter creation should succeed with valid config");
    let b_seed: Vec<f64> = (0..a1.b.len()).map(|i| 0.1 * (i as f64 + 1.0)).collect();
    a1.b.copy_from_slice(&b_seed);
    a2.b.copy_from_slice(&b_seed);
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.7];
    let y1 = a1
        .forward(&x)
        .expect("a1 forward pass should succeed with valid input");
    let y2 = a2
        .forward(&x)
        .expect("a2 forward pass should succeed with valid input");
    for (v1, v2) in y1.iter().zip(y2.iter()) {
        assert!((2.0 * v1 - v2).abs() < 1e-12, "α doubled → y doubled");
    }
    assert!((a1.scale() - 2.0).abs() < 1e-15);
    assert!((a2.scale() - 4.0).abs() < 1e-15);
}

#[test]
fn n_trainable_counts_w1_a_b() {
    let adapter = LoKrAdapter::new(default_cfg(6, 6, (3, 2), (2, 3), 2, 4.0), 0)
        .expect("LoKrAdapter creation should succeed with valid config");
    assert_eq!(adapter.n_trainable(), 3 * 2 + 2 * (2 + 3));
}
