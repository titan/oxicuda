# oxicuda-snn

Spiking Neural Network primitives -- a pure Rust simulation and training stack.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-snn` is a pure-Rust CPU-side library for simulating, training, and
analysing spiking neural networks (SNNs). It implements classical neuron
families -- Leaky Integrate-and-Fire, Integrate-and-Fire, Izhikevich, Adaptive
Exponential (AdEx), Hodgkin-Huxley, adaptive-LIF, heterogeneous LIF, and
Poisson sources -- alongside the synaptic kinetics, layers, and training
machinery required to build and evaluate full spiking models end-to-end.

Training is supported through surrogate-gradient backpropagation: BPTT, STBP
(Wu et al. 2018), SLAYER (Shrestha & Orchard 2018), and e-prop, with sigmoid,
atan, triangle, super-spike, and fast-sigmoid surrogate functions. Online
learning is covered by pair- and triplet- STDP with reward-modulated
variants, and ANN-to-SNN conversion is provided via quantile threshold
balancing for layer chains. Spiking layers include linear, convolutional,
pooling, and recurrent variants on top of the LIF kernel; a Liquid State
Machine reservoir is available for echo-state-style workflows.

Spike-train inputs are produced via rate, time-to-first-spike (TTFS), phase,
and Poisson encodings; outputs are analysed with firing-rate, ISI, CV,
van Rossum, Victor-Purpura, and synchrony metrics. Each domain module is
paired with PTX kernels emitted at runtime for SM 7.5 through SM 10.0, which
the higher OxiCUDA stack can dispatch when a CUDA device is available.

## Modules

| Module | Description |
|--------|-------------|
| `neuron` | LIF, IF, Izhikevich, AdEx, Hodgkin-Huxley, Poisson neuron models |
| `synapse` | Synaptic kinetics models for spiking neural networks |
| `surrogate` | Sigmoid, atan, triangle, super-spike, fast-sigmoid surrogate gradients |
| `training` | BPTT, STBP, SLAYER, e-prop supervised training |
| `plasticity` | Spike-timing-dependent plasticity (pair, triplet, reward-modulated) |
| `conversion` | ANN to SNN conversion via threshold balancing |
| `encoding` | Rate, TTFS, phase, Poisson spike-train encodings |
| `layer` | Spiking linear, convolutional, pooling, recurrent layers |
| `reservoir` | Liquid State Machine and random recurrent reservoirs |
| `metrics` | Firing rate, ISI, CV, van Rossum, Victor-Purpura, sync index |
| `handle` | `SnnHandle`, `SmVersion`, deterministic `LcgRng` |
| `ptx_kernels` | GPU PTX kernel strings for SM 7.5 through SM 10.0 |
| `error` | `SnnError` / `SnnResult` |

## Quick Start

```rust,no_run
use oxicuda_snn::neuron::lif::{LifConfig, LifState, lif_step};
use oxicuda_snn::error::SnnResult;

fn main() -> SnnResult<()> {
    let n = 8;
    let cfg = LifConfig::default();
    let mut state = LifState::new(n);

    // Constant input current and an output buffer for binary spike flags.
    let input: Vec<f32> = vec![0.5; n];
    let mut spikes = vec![0.0_f32; n];

    // One discrete-time LIF step:
    //   v_{t+1} = beta * v_t + I_t, spike if v >= v_th, then reset.
    lif_step(&mut state, &input, &cfg, &mut spikes)?;

    let total_spikes: f32 = spikes.iter().sum();
    println!("spikes this step: {total_spikes}");
    Ok(())
}
```

## Status

**Alpha** -- 10,683 SLoC, 329 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
