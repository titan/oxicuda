//! Unit tests for [`super::molora`]. Split out to keep `molora.rs` under the
//! 600-line per-file budget.

use super::molora::{MoLoraAdapter, MoLoraConfig};
use crate::error::PeftError;

fn cfg(
    in_f: usize,
    out_f: usize,
    rank: usize,
    alpha: f64,
    n_experts: usize,
    top_k: usize,
    temperature: f64,
) -> MoLoraConfig {
    MoLoraConfig {
        in_features: in_f,
        out_features: out_f,
        rank,
        alpha,
        n_experts,
        top_k,
        temperature,
    }
}

#[test]
fn single_expert_matches_standard_lora_forward() {
    let mut mo = MoLoraAdapter::new(cfg(5, 4, 2, 4.0, 1, 1, 1.0), 7)
        .expect("MoLoRA adapter creation should succeed with valid config");
    for (i, b) in mo.b_experts[0].iter_mut().enumerate() {
        *b = 0.1 * (i as f64 + 1.0);
    }
    let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
    let y_mo = mo
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    let s = 4.0_f64 / 2.0;
    let t: Vec<f64> = (0..2)
        .map(|k| {
            let row = k * 5;
            let mut acc = 0.0_f64;
            for (j, x_j) in x.iter().enumerate() {
                acc += mo.a_experts[0][row + j] * x_j;
            }
            acc
        })
        .collect();
    let y_ref: Vec<f64> = (0..4)
        .map(|i| {
            let row = i * 2;
            let mut acc = 0.0_f64;
            for (k, t_k) in t.iter().enumerate() {
                acc += mo.b_experts[0][row + k] * t_k;
            }
            s * acc
        })
        .collect();
    for (a, b) in y_mo.iter().zip(y_ref.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "single-expert MoLoRA mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn two_experts_top2_zero_b_gives_zero_output() {
    let mo = MoLoraAdapter::new(cfg(6, 4, 2, 4.0, 2, 2, 1.0), 11)
        .expect("MoLoRA adapter creation should succeed with valid config");
    let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
    let y = mo
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    for &v in &y {
        assert!(v.abs() < 1e-15, "expected zero output (B=0), got {v}");
    }
}

#[test]
fn top_k_one_selects_single_expert() {
    let mo = MoLoraAdapter::new(cfg(6, 4, 2, 4.0, 2, 1, 1.0), 11)
        .expect("MoLoRA adapter creation should succeed with valid config");
    let x = vec![1.0_f64; 6];
    let (_, info) = mo
        .forward_with_route(&x)
        .expect("forward with route should succeed");
    assert_eq!(info.selected.len(), 1);
    let total: f64 = info.gates.iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
    let non_zero = info.gates.iter().filter(|g| **g > 0.0).count();
    assert_eq!(non_zero, 1);
}

#[test]
fn rejects_top_k_greater_than_n_experts() {
    let bad = cfg(4, 4, 2, 1.0, 2, 5, 1.0);
    assert!(matches!(
        MoLoraAdapter::new(bad, 0),
        Err(PeftError::Internal { .. })
    ));
}

#[test]
fn rejects_zero_experts() {
    let bad = cfg(4, 4, 2, 1.0, 0, 1, 1.0);
    assert!(matches!(
        MoLoraAdapter::new(bad, 0),
        Err(PeftError::EmptyInput)
    ));
}

#[test]
fn rejects_zero_rank() {
    let bad = cfg(4, 4, 0, 1.0, 2, 2, 1.0);
    assert!(matches!(
        MoLoraAdapter::new(bad, 0),
        Err(PeftError::EmptyInput)
    ));
}

#[test]
fn rejects_non_positive_temperature() {
    let bad = cfg(4, 4, 2, 1.0, 2, 2, 0.0);
    assert!(matches!(
        MoLoraAdapter::new(bad, 0),
        Err(PeftError::Internal { .. })
    ));
    let bad_neg = cfg(4, 4, 2, 1.0, 2, 2, -0.5);
    assert!(matches!(
        MoLoraAdapter::new(bad_neg, 0),
        Err(PeftError::Internal { .. })
    ));
}

#[test]
fn forward_output_length_equals_out_features() {
    let mut mo = MoLoraAdapter::new(cfg(5, 9, 2, 4.0, 3, 2, 1.0), 17)
        .expect("MoLoRA adapter creation should succeed with valid config");
    for b in mo.b_experts.iter_mut() {
        for (i, v) in b.iter_mut().enumerate() {
            *v = 0.05 * (i as f64 + 1.0);
        }
    }
    let x = vec![1.0_f64; 5];
    let y = mo
        .forward(&x)
        .expect("forward pass should succeed with valid input");
    assert_eq!(y.len(), 9);
}

#[test]
fn forward_rejects_wrong_length_x() {
    let mo = MoLoraAdapter::new(cfg(5, 4, 2, 4.0, 2, 2, 1.0), 0)
        .expect("MoLoRA adapter creation should succeed with valid config");
    assert!(matches!(
        mo.forward(&[1.0_f64, 2.0, 3.0]),
        Err(PeftError::DimensionMismatch { .. })
    ));
}

#[test]
fn backward_matches_finite_differences_on_b_expert() {
    let mut mo = MoLoraAdapter::new(cfg(4, 3, 2, 4.0, 2, 2, 1.5), 99)
        .expect("MoLoRA adapter creation should succeed with valid config");
    for (k, b_k) in mo.b_experts.iter_mut().enumerate() {
        for (i, v) in b_k.iter_mut().enumerate() {
            *v = 0.1 * (i as f64 + 1.0) * (k as f64 + 1.0);
        }
    }
    let x = vec![0.5_f64, -1.0, 0.25, 0.75];
    let gy = vec![1.0_f64, -0.5, 0.25];
    let (_, grad_b_all, _) = mo.backward(&x, &gy).expect("backward pass should succeed");
    let eps = 1e-6_f64;
    let target_expert = 0_usize;
    let target_grad = grad_b_all[target_expert].clone();
    for (k, &analytic) in target_grad.iter().enumerate() {
        let saved = mo.b_experts[target_expert][k];
        mo.b_experts[target_expert][k] = saved + eps;
        let yp = mo
            .forward(&x)
            .expect("forward pass should succeed in finite-difference check");
        mo.b_experts[target_expert][k] = saved - eps;
        let ym = mo
            .forward(&x)
            .expect("forward pass should succeed in finite-difference check");
        mo.b_experts[target_expert][k] = saved;
        let lp: f64 = gy.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
        let lm: f64 = gy.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - analytic).abs() < 1e-4,
            "B[{target_expert}][{k}] FD={fd} analytic={analytic}"
        );
    }
}

#[test]
fn backward_matches_finite_differences_on_w_gate() {
    let mut mo = MoLoraAdapter::new(cfg(4, 3, 2, 4.0, 3, 2, 1.5), 99)
        .expect("MoLoRA adapter creation should succeed with valid config");
    for (k, b_k) in mo.b_experts.iter_mut().enumerate() {
        for (i, v) in b_k.iter_mut().enumerate() {
            *v = 0.1 * (i as f64 + 1.0) * (k as f64 + 1.0);
        }
    }
    let x = vec![0.5_f64, -1.0, 0.25, 0.75];
    let gy = vec![1.0_f64, -0.5, 0.25];
    let (_, _, grad_w) = mo.backward(&x, &gy).expect("backward pass should succeed");
    let eps = 1e-6_f64;
    let grad_w_snapshot = grad_w.clone();
    for (k, &analytic) in grad_w_snapshot.iter().enumerate() {
        let saved = mo.w_gate[k];
        mo.w_gate[k] = saved + eps;
        let yp = mo
            .forward(&x)
            .expect("forward pass should succeed in finite-difference check");
        mo.w_gate[k] = saved - eps;
        let ym = mo
            .forward(&x)
            .expect("forward pass should succeed in finite-difference check");
        mo.w_gate[k] = saved;
        let lp: f64 = gy.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
        let lm: f64 = gy.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - analytic).abs() < 1e-3,
            "W_g[{k}] FD={fd} analytic={analytic}"
        );
    }
}

fn entropy(gates: &[f64]) -> f64 {
    let mut h = 0.0_f64;
    for &g in gates {
        if g > 0.0 {
            h -= g * g.ln();
        }
    }
    h
}

#[test]
fn lower_temperature_sharpens_gate() {
    let warm = MoLoraAdapter::new(cfg(8, 4, 2, 4.0, 4, 4, 2.0), 23)
        .expect("warm MoLoRA adapter creation should succeed with valid config");
    let cold = MoLoraAdapter::new(cfg(8, 4, 2, 4.0, 4, 4, 0.1), 23)
        .expect("cold MoLoRA adapter creation should succeed with valid config");
    let x: Vec<f64> = (0..8).map(|i| (i as f64 - 3.5) * 0.5).collect();
    let (_, w_info) = warm
        .forward_with_route(&x)
        .expect("warm adapter forward with route should succeed");
    let (_, c_info) = cold
        .forward_with_route(&x)
        .expect("cold adapter forward with route should succeed");
    let h_warm = entropy(&w_info.gates);
    let h_cold = entropy(&c_info.gates);
    assert!(
        h_cold < h_warm,
        "cold gate entropy {h_cold} should be < warm {h_warm}"
    );
}

#[test]
fn load_balance_var_zero_when_batch_empty() {
    let mo = MoLoraAdapter::new(cfg(4, 4, 2, 4.0, 2, 2, 1.0), 0)
        .expect("MoLoRA adapter creation should succeed with valid config");
    let (ys, infos) = mo
        .forward_batch(&[])
        .expect("batch forward should succeed with empty batch");
    assert!(ys.is_empty());
    assert!(infos.is_empty());
    let xs: Vec<Vec<f64>> = (0..3)
        .map(|i| (0..4).map(|j| (i * 4 + j) as f64 * 0.1).collect())
        .collect();
    let (_, infos) = mo
        .forward_batch(&xs)
        .expect("batch forward should succeed with valid inputs");
    let var = infos[0].load_balance_var;
    assert!(var >= 0.0, "load_balance_var must be non-negative");
    for info in &infos {
        assert!((info.load_balance_var - var).abs() < 1e-15);
    }
}

#[test]
fn forward_batch_returns_n_outputs_and_infos() {
    let mo = MoLoraAdapter::new(cfg(4, 3, 2, 4.0, 3, 2, 1.0), 5)
        .expect("MoLoRA adapter creation should succeed with valid config");
    let xs: Vec<Vec<f64>> = (0..5)
        .map(|i| (0..4).map(|j| (i + j) as f64 * 0.1).collect())
        .collect();
    let (ys, infos) = mo
        .forward_batch(&xs)
        .expect("batch forward should succeed with valid inputs");
    assert_eq!(ys.len(), 5);
    assert_eq!(infos.len(), 5);
    for y in &ys {
        assert_eq!(y.len(), 3);
    }
}

#[test]
fn deterministic_given_same_seed() {
    let c = cfg(6, 4, 2, 4.0, 3, 2, 1.0);
    let m1 = MoLoraAdapter::new(c.clone(), 42)
        .expect("MoLoRA adapter m1 creation should succeed with valid config");
    let m2 = MoLoraAdapter::new(c, 42)
        .expect("MoLoRA adapter m2 creation should succeed with valid config");
    assert_eq!(m1.a_experts, m2.a_experts);
    assert_eq!(m1.b_experts, m2.b_experts);
    assert_eq!(m1.w_gate, m2.w_gate);
}
