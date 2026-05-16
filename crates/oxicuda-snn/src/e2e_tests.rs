//! End-to-end integration tests across the `oxicuda-snn` modules. The test
//! body is filled out as more modules land in parallel patches; for now we
//! exercise the `neuron` and `surrogate` foundations.

use crate::neuron::integrate_fire::{IfConfig, IfState, if_step};
use crate::neuron::izhikevich::{IzhConfig, IzhState, izh_step};
use crate::neuron::lif::{LifConfig, LifState, ResetMode, lif_step};
use crate::neuron::poisson::poisson_step;
use crate::surrogate::atan::atan_grad;
use crate::surrogate::fast_sigmoid::fast_sigmoid_grad;
use crate::surrogate::sigmoid::sigmoid_grad;
use crate::surrogate::super_spike::super_spike_grad;
use crate::surrogate::triangle::triangle_grad;

#[test]
fn lif_then_sigmoid_grad_e2e() {
    let cfg = LifConfig::default();
    let mut state = LifState::new(8);
    let current = vec![0.5_f32; 8];
    let mut spikes = vec![0.0_f32; 8];
    for _ in 0..20 {
        lif_step(&mut state, &current, &cfg, &mut spikes).expect("step");
    }
    let mut grad = vec![0.0_f32; 8];
    sigmoid_grad(&state.v, cfg.v_th, 4.0, &mut grad).expect("grad");
    for &g in &grad {
        assert!(g.is_finite() && g >= 0.0);
    }
}

#[test]
fn if_then_atan_grad_e2e() {
    let cfg = IfConfig::default();
    let mut state = IfState::new(4);
    let current = vec![0.3_f32; 4];
    let mut spikes = vec![0.0_f32; 4];
    if_step(&mut state, &current, &cfg, &mut spikes).expect("step");
    let mut grad = vec![0.0_f32; 4];
    atan_grad(&state.v, cfg.v_th, 1.0, &mut grad).expect("grad");
    for &g in &grad {
        assert!(g.is_finite() && g >= 0.0);
    }
}

#[test]
fn izh_then_super_spike_grad_e2e() {
    let cfg = IzhConfig::regular_spiking();
    let mut state = IzhState::new(4, cfg.b);
    let current = vec![10.0_f32; 4];
    let mut spikes = vec![0.0_f32; 4];
    for _ in 0..50 {
        izh_step(&mut state, &current, &cfg, &mut spikes).expect("step");
    }
    let mut grad = vec![0.0_f32; 4];
    super_spike_grad(&state.v, 30.0, 1.0, &mut grad).expect("grad");
    for &g in &grad {
        assert!(g.is_finite() && g > 0.0);
    }
}

#[test]
fn poisson_input_drives_lif() {
    use crate::handle::LcgRng;
    let mut rng = LcgRng::new(123);
    let n = 16_usize;
    let rates = vec![0.2_f32; n];
    let mut input_spikes = vec![0.0_f32; n];
    let mut lif_state = LifState::new(n);
    let mut lif_out = vec![0.0_f32; n];
    let cfg = LifConfig::default();
    let mut total_lif_spikes = 0_usize;
    for _ in 0..200 {
        poisson_step(&rates, 1.0, &mut rng, &mut input_spikes).expect("poisson");
        lif_step(&mut lif_state, &input_spikes, &cfg, &mut lif_out).expect("lif");
        for &s in &lif_out {
            if s == 1.0 {
                total_lif_spikes += 1;
            }
        }
    }
    assert!(total_lif_spikes > 0);
}

#[test]
fn triangle_and_fast_sigmoid_grad_finite() {
    let v: Vec<f32> = (-10..=10).map(|i| 0.1 * i as f32).collect();
    let n = v.len();
    let mut g_tri = vec![0.0_f32; n];
    let mut g_fs = vec![0.0_f32; n];
    triangle_grad(&v, 0.0, 1.0, &mut g_tri).expect("tri");
    fast_sigmoid_grad(&v, 0.0, 2.0, &mut g_fs).expect("fs");
    for (&a, &b) in g_tri.iter().zip(g_fs.iter()) {
        assert!(a.is_finite() && a >= 0.0);
        assert!(b.is_finite() && b > 0.0);
    }
}

#[test]
fn handle_constructs_with_rng() {
    use crate::handle::SnnHandle;
    let h = SnnHandle::new(80, 42);
    assert_eq!(h.sm().as_u32(), 80);
}

#[test]
fn lif_soft_vs_hard_reset_diverges() {
    let cfg_h = LifConfig {
        reset: ResetMode::Hard,
        ..Default::default()
    };
    let cfg_s = LifConfig {
        reset: ResetMode::Soft,
        ..Default::default()
    };
    let mut s_h = LifState::new(1);
    let mut s_s = LifState::new(1);
    let current = vec![1.5_f32; 1];
    let mut sp_h = vec![0.0_f32; 1];
    let mut sp_s = vec![0.0_f32; 1];
    lif_step(&mut s_h, &current, &cfg_h, &mut sp_h).expect("h");
    lif_step(&mut s_s, &current, &cfg_s, &mut sp_s).expect("s");
    assert_eq!(sp_h, sp_s);
    assert!((s_h.v[0] - s_s.v[0]).abs() > 1e-6);
}

#[test]
fn ptx_kernels_include_targets() {
    use crate::ptx_kernels::*;
    for sm in [75u32, 80, 86, 89, 90, 100] {
        assert!(lif_step_ptx(sm).contains(".visible .entry"));
        assert!(surrogate_grad_ptx(sm).contains(".visible .entry"));
        assert!(stdp_update_ptx(sm).contains(".visible .entry"));
        assert!(spike_conv_ptx(sm).contains(".visible .entry"));
        assert!(rate_encode_ptx(sm).contains(".visible .entry"));
        assert!(poisson_sample_ptx(sm).contains(".visible .entry"));
        assert!(bptt_accum_ptx(sm).contains(".visible .entry"));
    }
}
