# oxicuda-snn TODO

Pure Rust Spiking Neural Network primitives covering classical neuron models, surrogate gradients, BPTT / STBP / SLAYER training, STDP-family plasticity, ANN→SNN conversion, input encoding, spiking layers (linear / conv / pool / recurrent), Liquid State Machines, and analytical spike metrics, with PTX kernel templates for SM 7.5 through SM 10.0. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.45).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: ~32,700 SLoC (95 files)**

Current implementation covers the classical spiking ML stack: LIF / IF / Izhikevich / AdEx / Poisson neurons; five surrogate-gradient families; BPTT / STBP / SLAYER training; pair-STDP, triplet-STDP, reward-modulated STDP plasticity; ANN→SNN rate conversion with threshold balancing; rate / TTFS / phase / Poisson input encodings; spiking linear / conv / pool / recurrent layers; Liquid State Machine reservoir; and analytical spike-train metrics (firing rate, ISI, CV, van Rossum, Victor-Purpura, sync index, neuronal-avalanche criticality, entropy / mutual information, population-vector decoding, spike-triggered average / covariance).

### Completed [x]

#### Core Infrastructure
- [x] `lib.rs` — Crate root, module declarations
- [x] `error.rs` — `SnnError` enum + `SnnResult<T>` alias
- [x] `handle.rs` — `SmVersion`, `LcgRng` (MMIX 64-bit + Box-Muller), `SnnHandle`
- [x] `ptx_kernels.rs` — 7 GPU kernels × 6 SM versions (75 / 80 / 86 / 89 / 90 / 100)
- [x] `e2e_tests.rs` — 11 cross-module integration tests

#### Neuron Models (neuron/)
- [x] `neuron/lif.rs` — `LifConfig {tau_m, v_th, v_rest, dt, reset:ResetMode}`, `LifState {v}`, `beta() = exp(−dt/τ_m)`, `lif_step` with Hard / Soft reset
- [x] `neuron/integrate_fire.rs` — `IfConfig`, `IfState`, `if_step` (no leak, threshold + reset)
- [x] `neuron/izhikevich.rs` — `IzhConfig` with `regular_spiking` / `fast_spiking` / `chattering` / `intrinsically_bursting` presets, two-half-step Euler `izh_step`, post-update clamp
- [x] `neuron/adex.rs` — Brette-Gerstner defaults, `AdexConfig` / `AdexState`, `adex_step` with `(V−V_T)/Δ_T ≤ 50` exponential clamp
- [x] `neuron/poisson.rs` — `poisson_step(rates, dt, rng, out)` with non-negative rate validation

#### Surrogate Gradients (surrogate/)
- [x] `surrogate/sigmoid.rs` — `stable_sigmoid` two-branch numerically stable formulation + `sigmoid_grad = α·σ·(1−σ)`
- [x] `surrogate/atan.rs` — `α/(π·(1+(α(v−v_th))²))`
- [x] `surrogate/triangle.rs` — `max(0, 1−|v−v_th|/α)` with exact compact support
- [x] `surrogate/super_spike.rs` — Zenke-Ganguli `α/(1+|v−v_th|·α)²`
- [x] `surrogate/fast_sigmoid.rs` — `α/(1+|α(v−v_th)|)²`

#### Training (training/)
- [x] `training/bptt.rs` — `BpttConfig {t_steps, surrogate, alpha}`, `surrogate_eval` dispatcher, `bptt_unroll` with `dL/dv_t = surrogate'·dL/ds_t + β·dL/dv_{t+1}` and weight outer-product accumulation; supports Hard and Soft reset gradients
- [x] `training/stbp.rs` — Explicit reset gradient `(1−s_t)·…` for hard reset (matches BPTT when no spikes occur)
- [x] `training/slayer.rs` — `SlayerConfig {tau_s, dt}`, `epsilon_psp` ε-kernel, truncated convolution `convolve_psp`, `slayer_loss` MSE over spike-response

#### Plasticity (plasticity/)
- [x] `plasticity/stdp.rs` — `StdpConfig`, `StdpTraces`, pair-rule with exponential decay traces and `[w_min, w_max]` weight clamping
- [x] `plasticity/r_stdp.rs` — `RStdpConfig` / `RStdpState`, eligibility-trace decay `e ← e·exp(−dt/τ_e) + STDP_event` gated by reward signal
- [x] `plasticity/triplet_stdp.rs` — `TripletStdpConfig` / `TripletTraces`, additional long pre/post traces (Pfister-Gerstner), reduces to pair STDP when `a2_*=0`

#### ANN→SNN Conversion (conversion/)
- [x] `conversion/ann2snn.rs` — `SnnLayer`, `quantile`, `ann_to_snn_layer` 99-percentile rescale `W' = W·(λ_prev/λ); b' = b/λ`
- [x] `conversion/threshold_balance.rs` — `balance_layer_chain` propagating `λ` across layer chain

#### Input Encoding (encoding/)
- [x] `encoding/rate.rs` — Bernoulli `out[t,i] = (rng < value[i])`, `rate_decode` time-average
- [x] `encoding/temporal.rs` — TTFS `t_spike = floor((1−clamp(v, 0, 1))·(T−1))`
- [x] `encoding/phase.rs` — Phase-coded oscillatory reference signal
- [x] `encoding/poisson_input.rs` — Wraps `poisson_step` for input layers

#### Spiking Layers (layer/)
- [x] `layer/spiking_linear.rs` — `SpikingLinear` with Kaiming-normal init, `forward_step` (W·x + b → LIF)
- [x] `layer/spiking_conv.rs` — `SpikingConv2d` naive direct sliding-window convolution + per-output-pixel LIF
- [x] `layer/spiking_pool.rs` — `PoolKind {Max, Avg}`, `spike_pool` 2-D windowed reduction
- [x] `layer/recurrent.rs` — `SpikingRecurrent` with `W_in·x + W_rec·s_{t-1}` + LIF, persistent `last_spikes`

#### Reservoir Computing (reservoir/)
- [x] `reservoir/lsm.rs` — `LsmConfig {n_neurons, density, spectral_radius, w_in_scale, seed}`, `power_iteration_spectral_radius`, sparse-random `W_rec` rescaled to target ρ(W)

#### Analytical Metrics (metrics/)
- [x] `metrics/metrics.rs` — `firing_rate`, `isi`, `cv_isi`, `van_rossum_distance` (exp-filter L²), `victor_purpura_distance` (DP recurrence), `sync_index` (peak normalised cross-correlation)
- [x] `metrics/avalanche.rs` — `detect_avalanches` (Beggs & Plenz 2003 coarse-grained population raster), `branching_parameter` / `branching_parameter_global`, `powerlaw_mle_exponent` (discrete Clauset-Shalizi-Newman MLE)
- [x] `metrics/information.rs` — `spike_train_entropy`, `mutual_information` (word-binned joint histogram, `MiCorrection::{None, MillerMadow}`)
- [x] `metrics/decoding.rs` — `population_vector` (Georgopoulos vector sum), `cosine_tuning_rate`, `spike_triggered_average` (window-major), `spike_triggered_covariance` (sample covariance)

#### GPU PTX Kernels
- [x] `lif_step_ptx` — LIF discrete-time step with Hard / Soft reset
- [x] `surrogate_grad_ptx` — 5-mode dispatch (sigmoid / atan / triangle / super-spike / fast-sigmoid)
- [x] `stdp_update_ptx` — Pair-STDP weight update
- [x] `spike_conv_ptx` — Direct spiking 2-D convolution
- [x] `rate_encode_ptx` — Bernoulli rate encoding
- [x] `poisson_sample_ptx` — Poisson rate-to-spike sampling
- [x] `bptt_accum_ptx` — BPTT gradient accumulation

### Future Enhancements [ ]

#### P0 — Verification on GPU Hardware (requires GPU hardware)
- [ ] End-to-end GPU verification of all PTX kernels under Linux + NVIDIA driver 525+ (requires GPU hardware)
- [ ] Criterion benchmark suite executed on real hardware (requires GPU hardware)
- [ ] Numerical equivalence between CPU reference and GPU PTX path within FP32 tolerance for the 7 kernels (requires GPU hardware)
- [ ] Memory-leak audit for long-time-window BPTT (T > 1000 steps) (requires GPU hardware)

#### P1 — Algorithm Coverage
- [x] Adaptive-threshold LIF (ALIF) with learnable adaptive threshold (`neuron/alif.rs` -- Bellec et al. 2018)
- [x] Heterogeneous LIF (per-neuron `tau_m` / `v_th`) for population diversity (`neuron/het_lif.rs`)
- [x] Conductance-based synapses (CUBA / COBA variants)
- [x] Multi-compartment neuron models (Hodgkin-Huxley, two-compartment Pinsky-Rinzel) (`neuron/hodgkin_huxley.rs` -- full HH with RK4 voltage integration + Pinsky-Rinzel 1994 CA3 soma/dendrite model: `hh_step`/`hh_run`/`pr_step`)
- [x] Backpropagation Through Spike-Time (DECOLLE, e-prop) for online learning (`training/eprop.rs` -- Bellec et al. 2020 e-prop eligibility traces + Kaiser et al. 2020 DECOLLE local readout)
- [x] Learnable surrogate gradients (parametric `α(t)`) (`surrogate/learnable.rs` -- α with ∂loss/∂α gradient)
- [x] Spatio-temporal Backpropagation with Random Feedback Alignment (`training/feedback_alignment.rs`)
- [x] STDP-based homeostatic plasticity (BCM, Oja, intrinsic plasticity) (`plasticity/homeostatic.rs` BCM+Oja; `plasticity/intrinsic.rs` Triesch 2005 intrinsic plasticity)
- [x] Reward-modulated triplet STDP with eligibility kernels (`plasticity/reward_triplet_stdp.rs`)
- [x] Local learning rules with three-factor neuromodulation (`training/eligibility_consolidation.rs` -- three-factor pre×post×neuromod)
- [x] ANN→SNN conversion with bias absorption and BatchNorm folding (`conversion/bn_fold.rs`)
- [x] Quantisation-aware SNN training (INT8 / FP8 weight + spike) (`training/quantization.rs` -- fake-quant STE)
- [x] Population coding for output (rate-decode + winner-take-all) (`metrics/population_coding.rs`)

#### P1 — Spiking Layer Coverage
- [x] Spiking transposed convolution (deconv) for spike-based generative models (`layer/spiking_deconv.rs`)
- [x] Spiking attention layer (spike-driven query / key / value) (`layer/spiking_attention.rs` -- Spikformer SSA, Zhou et al. 2023)
- [x] Spiking transformer blocks (Spikformer, SpikeGPT-style) (`layer/spiking_transformer.rs` -- SSA + spiking MLP/FFN + residual encoder block, Zhou et al. 2023)
- [x] Spiking residual / skip connection layers (`layer/spiking_residual.rs` -- SEW/MS-ResNet additive skip)
- [x] Spiking batch normalisation (tdBN) for training stability (`layer/td_bn.rs` -- Zheng et al. 2021)
- [x] Spiking dropout and stochastic depth (`layer/spiking_regularization.rs`)
- [x] Recurrent spiking layers with multiple time constants (multi-τ LIF) (`layer/multi_tau_lif.rs`)

#### P2 — Optimisations and Tooling
- [ ] Fused LIF + surrogate-gradient kernel (forward + backward in one pass)
- [ ] Persistent CTA scheduling for long-time-window BPTT
- [ ] CUDA-graph capture for repeated spike-time-step iterations
- [ ] Mixed-precision (FP16 / BF16) training kernels
- [ ] On-device Poisson sampling with hardware RNG (Philox)
- [x] Sparse spike encoding (CSR-style spike packets to reduce bandwidth) (`encoding/sparse_spike.rs` -- `SparseSpikes` CSR per-timestep row_ptr/col_idx/values; `encode_dense_to_sparse` + `to_dense` exact round-trip; `forward` sparse-spike × Wᵀ membrane current touching only active spikes, bit-exact vs `dense_forward`; `nnz`/`density` tracked)
- [x] Event-driven simulation backend for very sparse spiking regimes (`neuron/event_driven.rs` -- `EventDrivenLif` BinaryHeap time-ordered event queue, exact analytic `exp(−Δ/τ)` lazy decay between events, threshold-crossing reset + delayed recurrent propagation; `clock_stepped_spike_times` reference matched within tolerance while doing ≪ t_steps·n membrane updates)
- [ ] CUDA-graph capture for ANN→SNN conversion inference loop

#### P2 — Reservoir / LSM
- [x] Echo State Network (ESN) with leaky integrator units
- [x] Dendritic computation model (`neuron/dendritic.rs`) — Poirazi 2003 Neuron: multi-compartment nonlinear dendritic integration with sigmoid dendritic sub-unit followed by somatic integration; `DendriticNeuron`
- [x] Eligibility trace consolidation (`training/eligibility_consolidation.rs`) — Zenke 2021: three-factor eligibility-trace synaptic tags + neuromodulatory consolidation signal for spike-timing-dependent plasticity with delay; `EligibilityConsolidation`
- [x] Temporal contrast encoding (`encoding/temporal_contrast.rs`) — Brandli 2014: event-camera-inspired temporal-contrast spike encoder with threshold-crossing hysteresis for asynchronous frame encoding; `TemporalContrastEncoder`
- [x] Adaptive spectral-radius scheduling during training (`reservoir/adaptive_spectral.rs`)
- [x] Multi-reservoir hierarchical LSM (`reservoir/hierarchical_lsm.rs`)
- [x] Online ridge-regression readout training (`reservoir/ridge_readout.rs` -- RLS)
- [x] Force-learning readout (FORCE-trained LSM) (`reservoir/ridge_readout.rs` -- Sussillo & Abbott 2009 RLS)

#### P2 — Metrics and Analysis
- [x] Spike-triggered average / covariance analysis (`metrics/decoding.rs` -- STA window-major + STC sample covariance, de Boer & Kuyper 1968 / Schwartz et al. 2006)
- [x] Population vector decoding (`metrics/decoding.rs` -- Georgopoulos et al. 1986 vector sum + cosine tuning curve)
- [x] Time-resolved firing-rate estimation via Kernel Density (`metrics/kde_rate.rs`)
- [x] Mutual information between spike trains (`metrics/information.rs` -- word-binned joint histogram MI with optional Miller-Madow correction)
- [x] Avalanche statistics (criticality, branching parameter) (`metrics/avalanche.rs` -- Beggs & Plenz 2003 detection, per/global branching parameter, discrete CSN power-law MLE)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | Device / Pinned memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Tests: 833 passing (unit + 11 e2e integration tests in `e2e_tests.rs`; includes 23 reservoir-task / STDP-verification tests in `tasks/`)
- Warnings: 0 (`cargo clippy --all-features --all-targets -- -D warnings` clean)
- `unwrap()` in production code: 0
- macOS: compiles, runtime returns `UnsupportedPlatform` for GPU launches
- All PTX kernels validated as non-empty strings for SM 75 / 80 / 86 / 89 / 90 / 100

## Performance Targets

Spiking-neural-network kernels exhibit two regimes: dense (every neuron updated each timestep) and event-driven (only spiking neurons trigger downstream updates). Current PTX kernels target the dense regime; a pure-Rust CPU event-driven backend (`neuron/event_driven.rs`) and CSR sparse-spike packing (`encoding/sparse_spike.rs`) now cover the sparse regime on the host side.

| Operation | Target Reference | Notes |
|-----------|------------------|-------|
| LIF step (N=10K) | ≥ 95% of cuBLAS axpy + threshold | trivially bandwidth-bound |
| Surrogate gradient (N=10K) | ≥ 95% of cuBLAS axpy | trivially bandwidth-bound |
| STDP update (N_pre × N_post=1K×1K) | ≥ 90% of cuBLAS ger | rank-1 trace product |
| Spiking conv (B=32, C=64, H=W=32, K=3) | ≥ 90% of cuDNN conv2d | dominated by im2col |
| BPTT unroll (T=100, N=1K) | ≥ 90% of T separate gemv+axpy | sequential time-step |
| Poisson sample (N=100K, T=100) | ≥ 90% of cuRAND Philox | RNG-bound |

## Notes

- All neuron dynamics are deterministic given an `LcgRng` seed (used for Poisson sampling and reservoir initialisation).
- The surrogate-gradient `α` parameter controls slope; smaller `α` = wider gradient (more numerical stability), larger `α` = sharper Heaviside approximation.
- The Brette-Gerstner AdEx model includes an exponential-clamp at `(V−V_T)/Δ_T ≤ 50` to prevent overflow during spike onset.
- LSM spectral-radius rescaling uses power iteration (default 30 iterations) for largest eigenvalue magnitude estimation.
- ANN→SNN conversion preserves prediction accuracy by per-layer threshold rescaling using 99-percentile activation as λ.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [ ] Validate LIF and surrogate-gradient kernels on T4 (FP16 storage)
- [ ] Block-size autotuning for spike-driven sparse layers

### Ampere (sm_80 / sm_86)
- [ ] `cp.async` 3-stage prefetch of spike trains for BPTT temporal accumulation on A100
- [ ] Tensor-Core (mma.sync) acceleration of spiking convolution and recurrent layer
- [ ] Persistent CTA scheduling for long T-steps simulations

### Ada (sm_89)
- [ ] FP8 (e4m3 / e5m2) weight + spike storage for inference deployment
- [ ] Sparse Tensor-Core path for very sparse spike trains (2:4 spike mask)

### Hopper (sm_90)
- [ ] TMA-based bulk spike-train staging for very long T-step BPTT
- [ ] `wgmma.mma_async` for spiking convolution and recurrent matmul
- [ ] Distributed shared memory across CTA cluster for tiled BPTT accumulation
- [ ] Cluster-wide reduce for population firing-rate metrics

### Blackwell (sm_100)
- [ ] `tcgen05` tensor memory layout for FP4 / FP6 spiking inference
- [ ] 5th-generation Tensor Core for spiking attention and convolution

---

## Deepening Opportunities

### Verification Gaps
- [ ] All 7 PTX kernels executed end-to-end on GPU hardware (currently only string-content verified)
- [ ] Performance bench numbers (lif_step, surrogate_grad on A100 / H100) recorded in `benches/snn_ops.rs`
- [ ] Numerical equivalence between CPU reference and GPU PTX path within FP32 tolerance
- [ ] Long-horizon BPTT (T > 1000) numerical-stability verification
- [ ] ANN→SNN conversion accuracy-vs-time-steps curve documented for representative networks

### Algorithmic Deepening
- [ ] Surrogate-gradient training on a large benchmark (N-MNIST, DVS128, SHD) with reference accuracy numbers
- [x] Pair-STDP convergence under realistic Poisson input statistics (`tasks/stdp_protocols.rs:pair_stdp_window`/`pair_stdp_poisson_final_weight` / tests -- measured: window sign LTP Δt>0 / LTD Δt<0 with `A·exp(−|Δt|/τ)` shape; uncorrelated Poisson drifts 0.5→0.038 (LTD-dominated competition), causal pre→post correlation converges higher (0.151 vs 0.038))
- [x] Triplet-STDP rate-dependent BCM-like behaviour empirically verified (`tasks/stdp_protocols.rs:triplet_pairing_dw` / tests -- measured: triplet-specific extra potentiation grows with pairing rate, +0.339 at period=5 vs +0.199 at period=50, a third-factor effect the pure pair rule cannot produce)
- [ ] R-STDP RL agent (cart-pole, MountainCar) demo
- [x] LSM polynomial-readout task suite (NARMA-10, memory capacity) (`tasks/reservoir_tasks.rs:narma10_sequence`/`narma10_lsm_nmse`/`memory_capacity` / tests -- measured: NARMA-10 test NMSE ≈0.48 vs mean-baseline 1.0; linear memory capacity ≈2.5 (N=40) → ≈6.4 (N=300), rising with reservoir size and bounded by readout-feature count)
- [ ] ANN→SNN conversion on a CIFAR-10 ResNet with empirical accuracy-vs-time-step curve
- [x] Multi-time-constant LIF for temporal multiplexing (`layer/multi_tau_lif.rs`)

### Coverage Gaps vs Literature
- [x] Brette and Gerstner LIF variations (alpha-function synapses, refractory periods)
- [x] Spike-Timing-Dependent Long-Term Potentiation / Depression with metaplasticity (`plasticity/metaplastic_stdp.rs` -- BCM-style sliding metaplastic state)
- [x] Spike-Timing-Dependent Heterosynaptic Plasticity (`plasticity/heterosynaptic.rs` -- weight-normalising heterosynaptic decay, Chistiakova et al. 2014)
- [x] STDP-driven self-organising maps (Kohonen-style) (`plasticity/stdp_som.rs`)
- [x] Liquid Time-Constant Networks (LTC) — non-spiking but related to LSM (`reservoir/ltc.rs` -- Hasani et al. 2021 fused-ODE solver)
- [ ] Neuromorphic-hardware ports (Loihi, TrueNorth instruction emulation) (requires hardware ISA emulation -- deferred)
- [x] Spiking-VAE / Spiking-GAN (`layer/spiking_vae.rs` -- spiking variational autoencoder + reparameterised latent)
- [x] Differentiable spike encoders (learned rate / TTFS) (`encoding/differentiable.rs`)
- [x] Online learning rules: e-prop, RFLO, BPTT-with-feedback-alignment (e-prop `training/eprop.rs`; RFLO `training/rflo.rs`; feedback-alignment `training/feedback_alignment.rs`)
- [x] Bayesian SNN training with variational posterior over spikes (`training/bayesian_snn.rs`)
