//! End-to-end integration tests across the `oxicuda-snn` modules.
//!
//! These tests deliberately span module boundaries: each one wires together an
//! encoder, one or more neuron/synapse/layer stages, and (where applicable) a
//! plasticity rule or analysis metric, then asserts an invariant that can only
//! hold if every stage in the chain agrees on data layout and semantics. The
//! coverage exercises the `neuron`, `surrogate`, `synapse`, `layer`,
//! `encoding`, `plasticity`, `metrics`, and `conversion` modules in realistic
//! cross-module flows.

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

// ---------------------------------------------------------------------------
// Cross-module integration: encoding -> layer -> plasticity -> metrics.
// ---------------------------------------------------------------------------

/// Rate-encode an analogue input, push every timestep through a `SpikingLinear`
/// layer, drive a pair-STDP rule with the resulting pre/post spikes, and verify
/// (a) STDP keeps the weights bounded and (b) the layer actually emitted
/// spikes that the firing-rate metric can recover. This wires `encoding::rate`,
/// `layer::spiking_linear`, `plasticity::stdp`, and `metrics` together.
#[test]
fn rate_encoded_input_through_linear_layer_with_stdp_and_metrics() {
    use crate::encoding::rate::rate_encode;
    use crate::handle::LcgRng;
    use crate::layer::spiking_linear::SpikingLinear;
    use crate::metrics::metrics::firing_rate;
    use crate::plasticity::stdp::{StdpConfig, StdpTraces, stdp_step};

    let in_dim = 12_usize;
    let out_dim = 6_usize;
    let t_steps = 64_usize;

    // Encode a moderately active analogue input as a Bernoulli spike train.
    let mut rng = LcgRng::new(2025);
    let values = vec![0.6_f32; in_dim];
    let mut train = vec![0.0_f32; t_steps * in_dim];
    rate_encode(&values, t_steps, &mut rng, &mut train).expect("rate encode");

    // A fully-connected spiking layer with biases lifted so it reliably fires.
    let mut layer = SpikingLinear::new(in_dim, out_dim, LifConfig::default(), &mut rng);
    for b in &mut layer.b {
        *b = 0.4;
    }

    // Pair-STDP between the (rate-encoded) pre-population and the layer output.
    let stdp_cfg = StdpConfig::default();
    let mut traces = StdpTraces::new(in_dim, out_dim);
    let mut weights = vec![0.5_f32; in_dim * out_dim];

    // Collect the layer's output spike train so a metric can analyse it.
    let mut out_train = vec![0.0_f32; t_steps * out_dim];
    let mut out_row = vec![0.0_f32; out_dim];

    for t in 0..t_steps {
        let pre = &train[t * in_dim..(t + 1) * in_dim];
        layer.forward_step(pre, &mut out_row).expect("layer step");
        out_train[t * out_dim..(t + 1) * out_dim].copy_from_slice(&out_row);
        stdp_step(
            &mut weights,
            &mut traces,
            pre,
            &out_row,
            in_dim,
            out_dim,
            &stdp_cfg,
        )
        .expect("stdp step");
    }

    // STDP must keep every weight inside the configured clip range.
    for &w in &weights {
        assert!(
            w.is_finite() && (stdp_cfg.w_min..=stdp_cfg.w_max).contains(&w),
            "weight {w} escaped [{}, {}]",
            stdp_cfg.w_min,
            stdp_cfg.w_max
        );
    }

    // The eligibility traces are non-negative and finite after the run.
    for &x in &traces.x_pre {
        assert!(x.is_finite() && x >= 0.0);
    }
    for &y in &traces.y_post {
        assert!(y.is_finite() && y >= 0.0);
    }

    // The output is a valid binary spike train and the firing-rate metric can
    // recover a non-trivial rate from it.
    for &s in &out_train {
        assert!(s == 0.0 || s == 1.0);
    }
    let rates = firing_rate(&out_train, t_steps, out_dim, 1.0).expect("firing rate");
    assert_eq!(rates.len(), out_dim);
    let total: f32 = rates.iter().sum();
    assert!(total > 0.0, "layer never spiked across {t_steps} steps");
    for &r in &rates {
        assert!((0.0..=1.0).contains(&r), "rate {r} out of [0,1]");
    }
}

/// Drive a `SpikingRecurrent` layer with a constant input and confirm the
/// recurrent self-connections produce temporal structure: the membrane state
/// after a driven phase differs from its zero-initialised value, and the
/// recorded spike train passes the van-Rossum self-distance sanity check
/// (a train is at zero distance from itself). Wires `layer::recurrent`,
/// `neuron::lif`, and `metrics`.
#[test]
fn recurrent_layer_temporal_dynamics_and_van_rossum() {
    use crate::handle::LcgRng;
    use crate::layer::recurrent::SpikingRecurrent;
    use crate::metrics::metrics::van_rossum_distance;

    let in_dim = 5_usize;
    let n = 4_usize;
    let t_steps = 48_usize;

    let mut rng = LcgRng::new(77);
    let mut layer = SpikingRecurrent::new(in_dim, n, LifConfig::default(), &mut rng);

    let drive = vec![0.9_f32; in_dim];
    let quiet = vec![0.0_f32; in_dim];
    let mut out = vec![0.0_f32; n];

    // Record neuron 0's spike train across the whole run.
    let mut neuron0_train = vec![0.0_f32; t_steps];
    let mut any_spike = false;

    for (t, slot) in neuron0_train.iter_mut().enumerate() {
        // Drive the first half, then let recurrence carry activity.
        let input = if t < t_steps / 2 { &drive } else { &quiet };
        layer.forward_step(input, &mut out).expect("recurrent step");
        *slot = out[0];
        if out.contains(&1.0) {
            any_spike = true;
        }
        // `last_spikes` is the recurrent feedback buffer; it must mirror `out`.
        assert_eq!(layer.last_spikes, out);
    }

    assert!(any_spike, "recurrent layer never spiked");

    // A spike train is at zero van-Rossum distance from itself.
    let self_dist =
        van_rossum_distance(&neuron0_train, &neuron0_train, t_steps, 5.0, 1.0).expect("van rossum");
    assert!(self_dist.abs() < 1e-5, "self distance was {self_dist}");

    // The membrane state is finite and (because of leak + reset) bounded.
    for &v in &layer.state.v {
        assert!(v.is_finite(), "membrane potential diverged: {v}");
    }
}

/// Build a current-based (CUBA) synapse bank, feed it a TTFS-encoded spike
/// train, and integrate the resulting synaptic current into a LIF neuron
/// population. Confirms the latency-coded input (early-spiking = high value)
/// produces an exponentially-decaying post-synaptic current and that the LIF
/// pool downstream eventually fires. Wires `encoding::temporal`,
/// `synapse::conductance`, and `neuron::lif`.
#[test]
fn ttfs_encoded_input_through_cuba_synapse_into_lif() {
    use crate::encoding::temporal::ttfs_encode;
    use crate::neuron::lif::beta;
    use crate::synapse::conductance::{CubaConfig, CubaState, cuba_step_batch};

    let n = 6_usize;
    let t_steps = 40_usize;

    // High-amplitude inputs spike early under TTFS coding.
    let values = vec![0.95_f32; n];
    let mut train = vec![0.0_f32; t_steps * n];
    ttfs_encode(&values, t_steps, &mut train).expect("ttfs encode");

    // One CUBA synapse per channel; strong positive weights.
    let syn_cfg = CubaConfig::default();
    let mut syn_states = vec![CubaState::new(); n];
    let weights = vec![1.5_f64; n];
    let mut i_syn = vec![0.0_f64; n];

    // Downstream LIF pool integrating the synaptic current.
    let lif_cfg = LifConfig::default();
    let mut lif_state = LifState::new(n);
    let mut lif_out = vec![0.0_f32; n];

    let mut peak_current = 0.0_f64;
    let mut current_at_peak_step = 0_usize;
    let mut total_lif_spikes = 0_usize;
    let mut current_after_input: Option<f64> = None;
    let mut current_one_step_later: Option<f64> = None;

    for t in 0..t_steps {
        let row = &train[t * n..(t + 1) * n];
        let spikes_in: Vec<bool> = row.iter().map(|&s| s != 0.0).collect();
        cuba_step_batch(&mut syn_states, &spikes_in, &weights, &mut i_syn, &syn_cfg)
            .expect("cuba batch");

        // Current as f32 fed into the LIF pool.
        let current_f32: Vec<f32> = i_syn.iter().map(|&v| v as f32).collect();
        lif_step(&mut lif_state, &current_f32, &lif_cfg, &mut lif_out).expect("lif");
        total_lif_spikes += lif_out.iter().filter(|&&s| s == 1.0).count();

        let row_max = i_syn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if row_max > peak_current {
            peak_current = row_max;
            current_at_peak_step = t;
        }
        // Sample the conductance one step after the input spike and again the
        // next step to verify pure exponential decay (no further input).
        if t == current_at_peak_step + 1 && current_after_input.is_none() {
            current_after_input = Some(i_syn[0]);
        } else if current_after_input.is_some() && current_one_step_later.is_none() {
            current_one_step_later = Some(i_syn[0]);
        }
    }

    // The synapse must have charged up from the input spikes.
    assert!(peak_current > 0.0, "CUBA synapse never charged");

    // After the (single) input spike, the current decays by exactly the CUBA
    // decay factor each step — the synapse and LIF share the exp-decay model.
    if let (Some(a), Some(b)) = (current_after_input, current_one_step_later) {
        let alpha = (-syn_cfg.dt / syn_cfg.tau_syn).exp();
        assert!(
            (b - a * alpha).abs() < 1e-9,
            "CUBA decay broken: {b} != {a} * {alpha}"
        );
    }

    // The LIF leak factor is in (0,1): a sanity cross-check that both modules
    // agree on the exponential-Euler convention.
    let lif_beta = beta(&lif_cfg);
    assert!((0.0..1.0).contains(&lif_beta));

    // Strong synaptic drive should make the downstream pool fire.
    assert!(
        total_lif_spikes > 0,
        "LIF pool never fired under CUBA drive"
    );
}

/// Conductance-based (COBA) synapse driving force test: the same arriving
/// spike train produces depolarising current at an excitatory reversal
/// potential and hyperpolarising current at an inhibitory one. This exercises
/// the `synapse::conductance` COBA path in a cross-config comparison.
#[test]
fn coba_excitatory_vs_inhibitory_driving_force() {
    use crate::synapse::conductance::{CobaConfig, CobaState, coba_step_batch};

    let n = 4_usize;
    let exc_cfg = CobaConfig::excitatory();
    let inh_cfg = CobaConfig::inhibitory();

    let mut exc_states = vec![CobaState::new(); n];
    let mut inh_states = vec![CobaState::new(); n];
    let weights = vec![0.8_f64; n];
    // A resting membrane potential between the two reversal potentials.
    let v = vec![-65.0_f64; n];
    let mut exc_i = vec![0.0_f64; n];
    let mut inh_i = vec![0.0_f64; n];

    // First step: every channel receives a spike, charging conductance.
    let all_spike = vec![true; n];
    coba_step_batch(
        &mut exc_states,
        &all_spike,
        &weights,
        &v,
        &mut exc_i,
        &exc_cfg,
    )
    .expect("coba exc");
    coba_step_batch(
        &mut inh_states,
        &all_spike,
        &weights,
        &v,
        &mut inh_i,
        &inh_cfg,
    )
    .expect("coba inh");

    for &i in &exc_i {
        // E_rev (0 mV) > V (-65 mV) => depolarising (positive) current.
        assert!(i > 0.0, "excitatory COBA current should be positive: {i}");
    }
    for &i in &inh_i {
        // E_rev (-80 mV) < V (-65 mV) => hyperpolarising (negative) current.
        assert!(i < 0.0, "inhibitory COBA current should be negative: {i}");
    }

    // With no further spikes, the conductance decays and the current
    // magnitude shrinks monotonically toward zero.
    let no_spike = vec![false; n];
    let prev_exc = exc_i[0].abs();
    coba_step_batch(
        &mut exc_states,
        &no_spike,
        &weights,
        &v,
        &mut exc_i,
        &exc_cfg,
    )
    .expect("coba exc decay");
    assert!(
        exc_i[0].abs() < prev_exc,
        "COBA current did not decay: {} -> {}",
        prev_exc,
        exc_i[0].abs()
    );
}

/// Full ANN->SNN conversion flow: balance a two-layer ReLU chain with the
/// percentile method, then run a rate-encoded input through the rescaled
/// weights as a spiking `SpikingLinear` stack and confirm the output rates
/// stay finite and bounded. Wires `conversion::threshold_balance`,
/// `layer::spiking_linear`, `encoding::rate`, and `metrics`.
#[test]
fn ann_to_snn_chain_balanced_then_run_as_spiking_layers() {
    use crate::conversion::threshold_balance::balance_layer_chain;
    use crate::encoding::rate::rate_encode;
    use crate::handle::LcgRng;
    use crate::layer::spiking_linear::SpikingLinear;
    use crate::metrics::metrics::firing_rate;

    // A 8 -> 5 -> 3 feed-forward chain.
    let dims = [8_usize, 5, 3];
    let mut weights: Vec<Vec<f32>> = vec![
        (0..dims[0] * dims[1])
            .map(|k| 0.05 + 0.01 * (k as f32))
            .collect(),
        (0..dims[1] * dims[2])
            .map(|k| 0.04 + 0.015 * (k as f32))
            .collect(),
    ];
    let mut biases: Vec<Vec<f32>> = vec![vec![0.1_f32; dims[1]], vec![0.1_f32; dims[2]]];
    // Representative per-layer activation samples for the percentile estimate.
    let activations: Vec<Vec<f32>> = vec![
        (0..32).map(|k| 0.1 + 0.05 * (k % 7) as f32).collect(),
        (0..32).map(|k| 0.2 + 0.04 * (k % 5) as f32).collect(),
    ];

    let lambdas = balance_layer_chain(&mut weights, &mut biases, &activations, &dims, 0.99)
        .expect("threshold balance");
    assert_eq!(lambdas.len(), 2);
    for &lam in &lambdas {
        assert!(lam.is_finite() && lam > 0.0, "bad lambda {lam}");
    }

    // Build spiking layers from the *balanced* weights.
    let mut rng = LcgRng::new(404);
    let mut layer0 = SpikingLinear::new(dims[0], dims[1], LifConfig::default(), &mut rng);
    let mut layer1 = SpikingLinear::new(dims[1], dims[2], LifConfig::default(), &mut rng);
    layer0.w.copy_from_slice(&weights[0]);
    layer0.b.copy_from_slice(&biases[0]);
    layer1.w.copy_from_slice(&weights[1]);
    layer1.b.copy_from_slice(&biases[1]);

    // Rate-encode an input and run it through the two-layer spiking stack.
    let t_steps = 50_usize;
    let input_values = vec![0.7_f32; dims[0]];
    let mut input_train = vec![0.0_f32; t_steps * dims[0]];
    rate_encode(&input_values, t_steps, &mut rng, &mut input_train).expect("rate encode");

    let mut h0 = vec![0.0_f32; dims[1]];
    let mut h1 = vec![0.0_f32; dims[2]];
    let mut out_train = vec![0.0_f32; t_steps * dims[2]];

    for t in 0..t_steps {
        let x = &input_train[t * dims[0]..(t + 1) * dims[0]];
        layer0.forward_step(x, &mut h0).expect("layer0 step");
        layer1.forward_step(&h0, &mut h1).expect("layer1 step");
        out_train[t * dims[2]..(t + 1) * dims[2]].copy_from_slice(&h1);
    }

    // The converted network produces a valid, bounded output spike train.
    for &s in &out_train {
        assert!(s == 0.0 || s == 1.0);
    }
    let rates = firing_rate(&out_train, t_steps, dims[2], 1.0).expect("firing rate");
    for &r in &rates {
        assert!(
            r.is_finite() && (0.0..=1.0).contains(&r),
            "rate {r} invalid"
        );
    }
}

/// Reward-modulated learning loop: encode a Poisson input, run it through a
/// recurrent reservoir-style layer, then apply pair-STDP and confirm the
/// weight matrix moves away from its initialisation in a bounded way while the
/// post-synaptic activity remains analysable by the ISI / CV metrics. Wires
/// `encoding::poisson_input`, `layer::recurrent`, `plasticity::stdp`, and
/// `metrics`.
#[test]
fn poisson_encoded_recurrent_learning_loop_changes_weights() {
    use crate::encoding::poisson_input::poisson_input_encode;
    use crate::handle::LcgRng;
    use crate::layer::recurrent::SpikingRecurrent;
    use crate::metrics::metrics::{cv_isi, isi};
    use crate::plasticity::stdp::{StdpConfig, StdpTraces, stdp_step};

    let in_dim = 8_usize;
    let n = 6_usize;
    let t_steps = 96_usize;

    // Poisson-encoded input train.
    let mut rng = LcgRng::new(909);
    let rates = vec![0.35_f32; in_dim];
    let mut input_train = vec![0.0_f32; t_steps * in_dim];
    poisson_input_encode(&rates, t_steps, 1.0, &mut rng, &mut input_train).expect("poisson encode");

    // Recurrent layer; lift the input weights so it fires regularly.
    let mut layer = SpikingRecurrent::new(in_dim, n, LifConfig::default(), &mut rng);
    for w in &mut layer.w_in {
        *w = w.abs() + 0.25;
    }

    // STDP between the input population and the recurrent neurons.
    let stdp_cfg = StdpConfig::default();
    let mut traces = StdpTraces::new(in_dim, n);
    let initial_weights = vec![0.5_f32; in_dim * n];
    let mut weights = initial_weights.clone();

    let mut out = vec![0.0_f32; n];
    let mut neuron0_train = vec![0.0_f32; t_steps];

    for t in 0..t_steps {
        let pre = &input_train[t * in_dim..(t + 1) * in_dim];
        layer.forward_step(pre, &mut out).expect("recurrent step");
        neuron0_train[t] = out[0];
        stdp_step(&mut weights, &mut traces, pre, &out, in_dim, n, &stdp_cfg).expect("stdp step");
    }

    // Learning must have moved at least one weight while keeping all of them
    // inside the clip range.
    let mut moved = false;
    for (&w, &w0) in weights.iter().zip(initial_weights.iter()) {
        assert!(
            w.is_finite() && (stdp_cfg.w_min..=stdp_cfg.w_max).contains(&w),
            "weight {w} escaped clip range"
        );
        if (w - w0).abs() > 1e-6 {
            moved = true;
        }
    }
    assert!(moved, "STDP did not change any weight over {t_steps} steps");

    // The post-synaptic spike train is analysable: ISI values are positive and
    // the coefficient of variation is finite and non-negative.
    let intervals = isi(&neuron0_train, 1.0);
    for &iv in &intervals {
        assert!(iv > 0.0, "non-positive inter-spike interval {iv}");
    }
    let cv = cv_isi(&intervals);
    assert!(cv.is_finite() && cv >= 0.0, "bad CV {cv}");
}

// ---------------------------------------------------------------------------
// Cross-module integration: neuron / encoding -> advanced metrics.
// ---------------------------------------------------------------------------

/// Drive a population of LIF neurons with a Poisson input, collect a time-major
/// population spike buffer, and run the avalanche detector over it. Confirms the
/// detector accepts the buffer and that every avalanche obeys the
/// layout-consistency invariant `size == Σ bins` (the only way this can hold is
/// if the population coarse-graining agrees with the time-major raster layout
/// the LIF stage wrote). Wires `neuron::poisson`, `neuron::lif`, and
/// `metrics::avalanche`.
#[test]
fn poisson_lif_population_avalanche_detection() {
    use crate::handle::LcgRng;
    use crate::metrics::avalanche::detect_avalanches;

    let n = 24_usize;
    let t_steps = 300_usize;
    let mut rng = LcgRng::new(54321);
    // Heterogeneous drive so the population activity is bursty (some bins busy,
    // some quiet) — the regime where avalanches are well defined.
    let rates: Vec<f32> = (0..n).map(|i| 0.05 + 0.02 * (i % 5) as f32).collect();
    let mut input_spikes = vec![0.0_f32; n];
    let mut lif_state = LifState::new(n);
    let mut lif_out = vec![0.0_f32; n];
    let cfg = LifConfig::default();

    // Time-major population buffer: row t holds the n neuron spikes for step t.
    let mut population = vec![0.0_f32; t_steps * n];
    for t in 0..t_steps {
        poisson_step(&rates, 1.0, &mut rng, &mut input_spikes).expect("poisson");
        // Scale the Poisson input up so the LIF pool actually fires.
        let drive: Vec<f32> = input_spikes.iter().map(|&s| s * 1.5).collect();
        lif_step(&mut lif_state, &drive, &cfg, &mut lif_out).expect("lif");
        population[t * n..(t + 1) * n].copy_from_slice(&lif_out);
    }

    let stats = detect_avalanches(&population, t_steps, n, 2).expect("avalanche detect");
    for av in &stats.avalanches {
        assert!(av.duration >= 1, "avalanche with zero duration");
        assert!(av.size >= 1, "avalanche with zero size");
        let bin_sum: usize = av.bins.iter().sum();
        assert_eq!(
            av.size, bin_sum,
            "layout inconsistency: size {} != Σ bins {}",
            av.size, bin_sum
        );
        assert_eq!(
            av.duration,
            av.bins.len(),
            "duration must equal bins length"
        );
    }
}

/// Run a single LIF neuron under a constant supra-threshold drive, take its
/// spike train, and confirm the information-theoretic identity `MI(X; X) = H(X)`
/// holds numerically for the word-binned estimators, and that the value is a
/// non-degenerate (strictly positive) entropy. Wires `neuron::lif` and
/// `metrics::information`.
#[test]
fn lif_train_self_mutual_information_equals_entropy() {
    use crate::metrics::information::{MiCorrection, mutual_information, spike_train_entropy};

    let t_steps = 120_usize;
    let cfg = LifConfig {
        tau_m: 20.0,
        v_th: 1.0,
        v_rest: 0.0,
        dt: 1.0,
        reset: ResetMode::Hard,
    };
    let mut state = LifState::new(1);
    // A current that makes the neuron fire periodically (but not every step),
    // so the word-binned symbol distribution is non-degenerate.
    let current = vec![0.18_f32; 1];
    let mut spikes = vec![0.0_f32; 1];
    let mut train = vec![0.0_f32; t_steps];
    for (t, slot) in train.iter_mut().enumerate() {
        lif_step(&mut state, &current, &cfg, &mut spikes).expect("lif");
        *slot = spikes[0];
        let _ = t;
    }

    // Word-bin parameters chosen so the bit pattern splits across symbols.
    let bin_steps = 2_usize;
    let word_bits = 1_usize;
    let h = spike_train_entropy(&train, t_steps, bin_steps, word_bits).expect("entropy");
    let mi = mutual_information(
        &train,
        &train,
        t_steps,
        bin_steps,
        word_bits,
        MiCorrection::None,
    )
    .expect("mi");

    assert!(mi >= 0.0, "MI must be non-negative, got {mi}");
    assert!(h > 0.0, "entropy degenerate (train never split): H={h}");
    assert!(
        (mi - h).abs() < 1e-5,
        "self-MI {mi} should equal entropy {h}"
    );
}

/// Build a 1-D stimulus with a temporal bump preceding each of several locked
/// spikes and confirm the spike-triggered average peaks at the bump centre.
/// Wires `metrics::decoding` with a hand-built spike train whose timing locks to
/// the stimulus feature.
#[test]
fn locked_spikes_sta_peaks_at_bump_centre() {
    use crate::metrics::decoding::spike_triggered_average;

    let stim_dim = 1_usize;
    let window = 7_usize;
    // A symmetric triangular bump; its peak sits at index 3 (the centre).
    let bump = [0.0_f32, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0];
    let repeats = 5_usize;
    let gap = 5_usize;
    let mut stim: Vec<f32> = Vec::new();
    let mut spikes: Vec<f32> = Vec::new();
    for _ in 0..repeats {
        for (k, &b) in bump.iter().enumerate() {
            stim.push(b);
            // Spike on the final bump sample so its window == the bump.
            spikes.push(if k == bump.len() - 1 { 1.0 } else { 0.0 });
        }
        for _ in 0..gap {
            stim.push(0.0);
            spikes.push(0.0);
        }
    }
    let t_steps = stim.len();
    let sta = spike_triggered_average(&stim, &spikes, t_steps, stim_dim, window).expect("sta");

    // The STA must reproduce the bump and therefore peak at its centre index.
    let mut peak_idx = 0_usize;
    let mut peak_val = f32::NEG_INFINITY;
    for (k, &v) in sta.iter().enumerate() {
        if v > peak_val {
            peak_val = v;
            peak_idx = k;
        }
    }
    assert_eq!(
        peak_idx, 3,
        "STA peak at {peak_idx}, expected bump centre 3"
    );
    assert!((peak_val - 1.0).abs() < 1e-5, "STA peak value {peak_val}");
}
