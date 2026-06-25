# oxicuda-rl TODO

GPU-accelerated reinforcement learning primitives: experience replay buffers, policy distributions, return/advantage estimators, RL algorithm losses (DQN/PPO/SAC/TD3), normalization, and environment abstractions. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.9).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 11,280 SLoC across 50 files (includes Markdown doc-comments) / 4,652 pure Rust SLoC**

First-class GPU-ready RL library implementing every major modern algorithm from DQN to SAC/TD3/PPO,
including prioritized experience replay (PER), n-step returns, GAE, V-trace, Retrace,
observation/reward normalization, and a vectorized environment abstraction.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `RlError` (12 variants): DimensionMismatch, InsufficientTransitions, InvalidPriority, InvalidConfig, EmptyBatch, NanEncountered, InvalidLogProb, NanLoss, EpisodeError, InvalidStateSize, InvalidAction, Other
- [x] `handle.rs` -- `RlHandle`, `SmVersion`, `LcgRng`
  - `SmVersion(u32)` with `ptx_version_str()` mapping sm>=100->"8.7", sm>=90->"8.4", sm>=80->"8.0", else "7.5"
  - `LcgRng` -- 64-bit LCG (multiplier 6364136223846793005) with `next_u32()`, `next_f32()`, `next_usize(n)`
  - `RlHandle::default_handle()` -- sm=80, device=0, seed=42
- [x] `lib.rs` -- prelude module and 5 E2E integration tests

#### PTX Kernel Sources
- [x] `ptx_kernels.rs` -- 5 GPU-side RL kernels (23.4 KB of PTX)
  - `td_error_ptx` -- TD-error `delta = r + gamma * (1 - done) * V' - V`, grid-stride
  - `normalize_advantages_ptx` -- mean/variance normalisation pass
  - `ppo_ratio_ptx` -- clipped importance ratio `exp(lp_new - lp_old)` with `ex2.approx.f32`
  - `sac_target_ptx` -- soft Bellman target `y = r + gamma * (1 - done) * (min(Q1, Q2) - alpha * lp)`
  - `per_is_weight_ptx` -- IS weight `(N * p_i)^(-beta)` normalised by max; `lg2.approx.f32`

#### Experience Replay Buffers (`buffer/`)
- [x] `buffer/mod.rs` -- module organization
- [x] `buffer/replay.rs` -- `UniformReplayBuffer` -- struct-of-arrays circular buffer; rejection sampling without replacement
- [x] `buffer/prioritized.rs` -- `PrioritizedReplayBuffer` -- dual sum + min segment tree O(log N); stratified sampling across strata; IS weight computation
- [x] `buffer/n_step.rs` -- `NStepBuffer` -- circular buffer of `Option<Step>`; n-step return accumulation with gamma^n bootstrap; flush on episode end

#### Policy Distributions (`policy/`)
- [x] `policy/mod.rs` -- module organization
- [x] `policy/categorical.rs` -- `CategoricalPolicy` -- Gumbel-max sampling; log-prob; entropy; KL-divergence; greedy; log_prob_batch
- [x] `policy/gaussian.rs` -- `GaussianPolicy` -- Box-Muller N(0,1); reparameterisation `mu + sigma * eps`; Tanh squashing with Jacobian correction; log-prob batch
- [x] `policy/deterministic.rs` -- `DeterministicPolicy` -- DDPG exploration noise; TD3 target policy smoothing (clipped Gaussian); `OrnsteinUhlenbeck` OU process

#### Return / Advantage Estimators (`estimator/`)
- [x] `estimator/mod.rs` -- module organization
- [x] `estimator/gae.rs` -- `compute_gae` -- backward scan `A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}`; optional Welford normalisation; `GaeConfig`
- [x] `estimator/td.rs` -- `compute_td_lambda` -- `G_t = r_t + gamma * mask * [(1 - lambda) * v_{t+1} + lambda * G_{t+1}]`; takes values[T+1] bootstrap
- [x] `estimator/vtrace.rs` -- `compute_vtrace` -- IMPALA V-trace: `c_t = min(c_bar, rho_t)`, `rho_bar_t = min(rho_bar, rho_t)`; backward scan advantages
- [x] `estimator/retrace.rs` -- `compute_retrace` -- safe off-policy Q-targets: `c_t = lambda * min(1, rho_t)`; `Q^ret_t = Q_t + delta_t + gamma * c_{t+1} * (Q^ret_{t+1} - Q_{t+1})`

#### RL Algorithm Loss Functions (`loss/`)
- [x] `loss/mod.rs` -- module organization
- [x] `loss/ppo.rs` -- `ppo_loss` -- clip `ratio * A` + `PpoConfig{clip_eps=0.2, c_v=0.5, c_e=0.01}`; `approx_kl`, `clip_fraction` metrics
- [x] `loss/dqn.rs` -- `dqn_loss` / `double_dqn_loss` -- Bellman MSE/Huber (kappa=1.0); IS-weighted; Double-DQN decoupled selection
- [x] `loss/sac.rs` -- `sac_critic_loss` / `sac_actor_loss` / `sac_temperature_loss` -- entropy-regularized; log-space temperature
- [x] `loss/td3.rs` -- `td3_critic_loss` / `td3_actor_loss` -- twin-Q Bellman error + deterministic actor `-mean(Q1_mu)`

#### Normalization (`normalize/`)
- [x] `normalize/mod.rs` -- module organization
- [x] `normalize/running_stats.rs` -- `RunningStats` -- Welford online N-dim: `delta = x - mean`; `mean += delta / n`; `M2 += delta * delta2`; batch update
- [x] `normalize/obs_norm.rs` -- `ObservationNormalizer` -- wraps `RunningStats`; clip=5.0; enable/disable; eval mode (no stat update)
- [x] `normalize/reward_norm.rs` -- `RewardNormalizer` -- `ReturnNorm` (G_t = gamma * G_{t-1} + r_t, divide by std), `Clip`, `None` modes; n_envs parallel

#### Environment Abstractions (`env/`)
- [x] `env/mod.rs` -- module organization
- [x] `env/env.rs` -- `Env` trait -- `obs_dim()`, `act_dim()`, `reset()`, `step()`, `is_continuous()`; `LinearQuadraticEnv` reference env (Box-Muller noise, NaN-safe)
- [x] `env/vectorized.rs` -- `VecEnv<E: Env>` -- batched `reset_all()`, `step()` (auto-reset on done), `foreach()`, `terminal_obs` tracking

#### Integration Tests
- [x] 5 E2E tests in `lib.rs`:
  - `e2e_dqn_style_loop` -- collect 200 transitions + DQN loss on LQ env
  - `e2e_ppo_gae_loss` -- 128-step GAE + PPO clip+value+entropy loss
  - `e2e_sac_style_update` -- PER buffer + SAC critic loss with IS weights
  - `e2e_vecenv_with_obs_norm` -- 4x VecEnv 20 steps + ObservationNormalizer Welford update
  - `e2e_n_step_buffer` -- 3-step return verification: R ~ 1 + 0.99 + 0.99^2

### Future Enhancements

#### P0 -- Critical (Algorithm Coverage)
- [x] PPO with clip + value + entropy loss (`loss/ppo.rs`)
- [x] DQN + Double-DQN Bellman losses (`loss/dqn.rs`)
- [x] SAC twin-Q + temperature auto-tuning (`loss/sac.rs`)
- [x] TD3 twin-Q + delayed policy (`loss/td3.rs`)

#### P1 -- Important (Sample Efficiency)
- [x] Prioritized Experience Replay with O(log N) segment tree (`buffer/prioritized.rs`)
- [x] N-step return accumulator with episode-boundary handling (`buffer/n_step.rs`)
- [x] GAE-lambda advantage estimator with optional Welford normalisation (`estimator/gae.rs`)
- [x] V-trace off-policy correction for IMPALA-style training (`estimator/vtrace.rs`)
- [x] Retrace safe off-policy Q-targets (`estimator/retrace.rs`)

#### P2 -- Nice-to-Have (Stability)
- [x] Observation/reward normalization with Welford stats (`normalize/`)
- [x] Vectorized environment wrapper with auto-reset (`env/vectorized.rs`)
- [x] OU-noise + clipped Gaussian exploration (`policy/deterministic.rs`)
- [x] (P2) Distributional RL (C51, QR-DQN) value losses -- no current implementation
- [ ] (P2) Apex / R2D2 distributed actor-learner pattern -- requires Vol.12 collective integration
- [x] (P2) DreamerV3 world-model RL (`world_model/dreamer_v3.rs`) — Hafner 2023: RSSM recurrent state-space model with symlog encoding, KL balancing, two-hot critic targets; `DreamerV3`
- [x] (P2) Decision Transformer (`policy/decision_transformer.rs`) — Chen 2021 NeurIPS: offline RL via causal Transformer conditioned on return-to-go + state + action; `DecisionTransformer`
- [x] (P2) Discrete SAC with Gumbel-Softmax (`loss/discrete_sac.rs`) — Christodoulou 2019: SAC extended to discrete action spaces; temperature auto-tuning via target entropy H_target=-|A|; `DiscreteSacLoss`
- [x] (P2) Plan2Explore unsupervised exploration (`policy/plan2explore.rs`) — Sekar 2020: ensemble disagreement intrinsic reward + one-step world-model disagreement maximisation; `Plan2Explore` (already implemented: `Plan2Explore`/`Plan2ExploreConfig`, K MLP ensemble members with latent-disagreement variance bonus)

#### Offline (Batch) RL Value Correction (`loss/offline.rs`)
- [x] (P2) CQL — Conservative Q-Learning (`loss/offline.rs`) — Kumar 2020: `logsumexp_a Q(s,a) − E_{a~D}[Q(s,a)]` conservative gap + Bellman MSE/Huber; `cql_loss`/`CqlConfig`/`CqlLoss`
- [x] (P2) IQL — Implicit Q-Learning (`loss/offline.rs`) — Kostrikov 2021: expectile value regression `|τ−1(u<0)|·u²` + in-sample-V critic bootstrap + advantage-weighted policy extraction; `iql_value_loss`/`iql_critic_loss`/`expectile_weight`/`advantage_weighted_policy_loss`/`IqlConfig`
- [x] (P2) AWAC — Advantage-Weighted Actor-Critic (`loss/offline.rs`) — Nair 2020: `−E[log π(a|s)·exp(A/λ)]` weighted MLE with clamp/normalise; `awac_actor_loss`/`AwacConfig`
- [x] (P2) BCQ — Batch-Constrained Q-Learning (`loss/offline.rs`) — Fujimoto 2019: soft clipped-double-Q target `λ·min+(1−λ)·max` + cVAE ELBO (recon + β·KL); `bcq_target`/`bcq_critic_loss`/`bcq_vae_loss`/`BcqConfig`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| thiserror | Error derive macros | Yes |
| tracing | Structured logging for episode rollouts | Yes |
| num-traits | Numeric trait bounds | Yes |

## Quality Status

- Warnings: 0 (clippy clean, `#![warn(missing_docs)]`)
- Tests: 425 passing (root TODO.md count)
- unwrap() calls: 0 (production code)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, returns `UnsupportedPlatform` at runtime

## Performance Targets

| Operation | Target |
|-----------|--------|
| `td_error_ptx` -- 1M transition batch | >= 90% bandwidth-limited peak on sm_80+ |
| `ppo_ratio_ptx` -- 65k-step PPO mini-batch | >= 85% bandwidth-limited peak |
| `per_is_weight_ptx` -- 1M sample PER batch | >= 80% bandwidth-limited peak |
| `compute_gae` (CPU reference) -- 4096-step rollout | < 1 ms single-threaded |
| `PrioritizedReplayBuffer::sample` -- 100k buffer | O(log N) per sample, < 1 us amortised |

## Numerical Accuracy Requirements

| Operation | Tolerance vs FP64 reference |
|-----------|-----------------------------|
| `compute_gae` advantages | rel < 1e-5 on rollouts <= 1000 steps |
| `compute_vtrace` advantages | rel < 1e-5 with clamp ratios at default 1.0 |
| PPO clip fraction | exact within [0, 1] |
| DQN Huber loss (kappa=1) | abs < 1e-6 vs PyTorch reference |
| SAC entropy temperature `alpha = exp(log_alpha)` | rel < 1e-5 |

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86 / sm_89)
- [x] PTX header selection emits `.target sm_80` for cp.async-capable reduction kernels
- [ ] Cooperative-group reductions in `normalize_advantages_ptx` (deferred -- requires hardware verification)

### Hopper (sm_90 / sm_90a)
- [x] PTX header selection emits `.target sm_90`
- [ ] Warp-grouped MMA for batched policy log-prob computation (deferred -- requires `oxicuda-blas` integration)

### Bandwidth Considerations
- All PTX kernels use grid-stride loops; replay-buffer sampling is bandwidth-bound on host side.

## Deepening Opportunities

### Verification Gaps
- [x] All 4 loss functions exercised by E2E tests with realistic batch shapes
- [x] GAE backward scan validated against analytic single-step TD limit
- [x] PER sampling distribution verified to match priority weights statistically
- [x] V-trace clamp ratio sweep across [0.1, 10.0] (`estimator/vtrace.rs::tests::vtrace_clamp_ratio_sweep_monotone` — verifies finiteness + monotonicity of v_s in c̄/ρ̄ while capping, and saturation above ρ)
- [x] Retrace traced through multi-episode boundary (`estimator/retrace.rs::tests::retrace_multi_episode_boundary_severs_trace` — done flag severs the backward trace; pre-boundary step matches episode-only batch, second-episode start matches B-only off-policy trace)

### Implementation Deepening
- [x] `CategoricalPolicy` supports Gumbel-max for parallel batch sampling
- [x] `GaussianPolicy` Tanh squashing applies log-determinant Jacobian correction
- [x] PER segment tree handles dynamic priority updates in O(log N)
- [x] VecEnv auto-reset preserves terminal observation for correct bootstrap
- [x] Distributional RL (C51/QR-DQN) policy + loss extension
- [x] Multi-discrete and tuple action space support (`spaces/multi_discrete.rs` — `Discrete`/`MultiDiscrete`/`TupleSpace`/`Space` trait with factorised log-prob, joint entropy, flat-prob sampling)

## Notes

- `LinearQuadraticEnv` reference environment uses Box-Muller noise with explicit NaN guard (`>> 41 bit shift`).
- `LcgRng` is deterministic across platforms; all sampling is seedable.
- Benchmarks not configured (no `criterion` dev-dep) -- algorithmic complexity is bounded analytically.
- Future integration with `oxicuda-train` for end-to-end RL training loops tracked in root TODO.md.
