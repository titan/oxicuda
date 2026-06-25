# oxicuda-rlhf TODO

Pure-Rust RLHF and alignment primitives covering all modern alignment
algorithms: Direct Preference Optimization (DPO), Identity Preference
Optimization (IPO), Kahneman-Tversky Optimization (KTO), Odds Ratio Preference
Optimization (ORPO), Simple Preference Optimization (SimPO), Bradley-Terry
reward modelling, reward normalisation (Welford), PPO-RLHF rollout with GAE +
KL penalty, adaptive KL controller, and masked SFT cross-entropy loss. Part of
[OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.32).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: ~15,700 SLoC (57 files)** -- compact implementation with 12 E2E
integration tests

The crate is intentionally compact: every modern preference-alignment loss
is implemented as a small, focused function operating on log-probabilities,
with PTX kernels for GPU acceleration on NVIDIA SM 7.5 through SM 12.0. The
PPO-RLHF pipeline is also covered (rollout + GAE + KL controller + clipped
surrogate).

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `RlhfError` (15 variants: DimensionMismatch, EmptyInput,
      InvalidBeta, InvalidTemp, NanEncountered, InvalidLambda,
      LogProbsRequired, MismatchedPairLength, InvalidMargin, KlDivergence,
      InvalidReferenceLogProb, RewardNormFailed, InvalidMaskValue, Internal,
      InvalidClipRatio); `RlhfResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Knuth MMIX 64-bit LCG),
      `RlhfHandle::default_handle()` (SM 8.0, device 0, seed 42)
- [x] `lib.rs` -- module exports + prelude + 12 E2E integration tests

#### PTX Kernels (7 kernels x 6 SM versions = 42 generators)
- [x] `ptx_kernels.rs::bt_reward_loss_ptx` --
      -sum log(sigmoid(r_chosen - r_rejected)) per pair
- [x] `ptx_kernels.rs::dpo_loss_ptx` -- DPO:
      -log sigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l))) per pair
- [x] `ptx_kernels.rs::ipo_loss_ptx` -- IPO squared:
      ((lp_w - ref_w) - (lp_l - ref_l) - 1 / (2 * beta))^2
- [x] `ptx_kernels.rs::kto_loss_ptx` -- KTO: desirable
      (1 - sigmoid(beta * (r - z_0))) + undesirable with lambda weights;
      z_0 = ln 2
- [x] `ptx_kernels.rs::orpo_odds_ptx` -- ORPO log-odds:
      log(exp(lp) / (1 - exp(lp) + eps)) per sequence
- [x] `ptx_kernels.rs::rlhf_kl_ptx` -- forward KL penalty per token:
      exp(lp) * (lp - ref_lp)
- [x] `ptx_kernels.rs::sft_mask_ptx` -- masked cross-entropy per token;
      division by mask-sum in host code
- [x] `ptx_kernels.rs::f32_hex` -- f32 to 0F-prefixed hex literal helper

#### Preference Data (preference/)
- [x] `pair.rs::PreferencePair` / `PairBatch` -- paired chosen / rejected
      log-probs + reference-model log-probs; length validation via
      `PairBatch::new()`
- [x] `bradley_terry.rs::bt_reward_loss` / `RewardHead` -- Bradley-Terry
      pairwise loss -E[log sigmoid(r_w - r_l)]; linear reward head with
      Xavier initialisation

#### Reward Modelling (reward/)
- [x] `model.rs::RewardModel` -- multi-layer MLP with ReLU activations ->
      scalar reward
- [x] `normalize.rs::RewardNormalizer` -- Welford online mean / variance;
      `normalize()` whitens to zero-mean unit-variance

#### Preference Alignment Losses (dpo/)
- [x] `dpo.rs::dpo_loss` / `DpoConfig` / `dpo_loss_per_pair` /
      `dpo_log_ratio` -- DPO loss with per-pair and batch variants
- [x] `ipo.rs::ipo_loss` / `IpoConfig` -- IPO squared loss
      (log_ratio_diff - 1 / (2 * beta))^2
- [x] `kto.rs::kto_loss` / `KtoConfig` -- KTO with desirable lambda_d and
      undesirable lambda_u; KL reference point z_0 = ln 2

#### Reference-Free Alignment (orpo/)
- [x] `orpo.rs::orpo_loss` / `OrpoConfig` / `log_odds` -- ORPO:
      L_SFT + lambda * (-log sigmoid(log_odds_w - log_odds_l)); no reference
      model
- [x] `simpo.rs::simpo_loss` / `SimpoConfig` -- SimPO: length-normalised
      -log sigmoid(beta / |y_w| * sum lp_w - beta / |y_l| * sum lp_l - gamma);
      margin gamma

#### RLHF-PPO Utilities (ppo_rlhf/)
- [x] `rollout.rs::RlhfRollout` -- rollout buffer with log_probs, ref_log_probs,
      rewards, values; `compute_advantages()` (GAE), `apply_kl_penalty()`
      (reward -= beta * KL)
- [x] `kl_control.rs::KlController` / `kl_divergence_from_logps` -- adaptive KL
      beta: beta *= (1 + k * (kl - target) / target)
- [x] `ppo_step.rs::rlhf_ppo_loss` / `RlhfPpoConfig` -- clipped PPO surrogate
      + value loss + entropy bonus -> (policy, value, entropy, approx_kl)

#### SFT Loss (sft/)
- [x] `loss.rs::sft_loss` / `masked_token_ce` -- cross-entropy with attention
      mask; logsumexp trick for numerical stability; division by sum of mask

#### Stateful Loss Objects (loss/) -- wired orphan modules
- [x] `loss/dpo_loss.rs::DpoLoss` / `DpoConfig` -- stateful DPO loss object with
      label smoothing, implicit-reward, and reward-margin helpers; 12 tests
      (wired orphan: `pub mod loss;` added to lib.rs; reachable via
      `loss::dpo_loss::` -- kept OUT of prelude to avoid `DpoConfig` clash with
      `dpo::dpo::DpoConfig`)
- [x] `loss/ppo_loss.rs::PpoLoss` / `PpoConfig` -- clipped PPO surrogate + MSE
      value loss + entropy-regularised total loss; 10 tests
      (wired orphan: `pub mod loss;` added to lib.rs; reachable via
      `loss::ppo_loss::`)

#### Reward Normalisation Stats (utils/) -- wired orphan module
- [x] `utils/reward_norm.rs::RunningRewardStats` / `RewardNormConfig` -- Welford
      online slice-based reward mean/variance with eps-stabilised whitening;
      8 tests (wired orphan: `pub mod reward_norm;` added to utils/mod.rs)

#### Metrics (metrics/)
- [x] `alignment.rs::win_rate` / `reward_gap` / `kl_from_ref` / `perplexity` /
      `AlignmentMetrics` / `compute_alignment_metrics` -- standard RLHF
      evaluation metrics; batch helper

#### Integration Tests (lib.rs e2e_tests)
- [x] `e2e_bt_loss_zero_equal_rewards` -- equal rewards -> loss = -log(0.5)
- [x] `e2e_bt_loss_decreases_with_gap` -- larger reward gap -> lower BT loss
- [x] `e2e_dpo_loss_finite` -- DPO loss on three-pair batch is finite
- [x] `e2e_dpo_lower_for_aligned_pairs` -- aligned pairs produce lower DPO
      loss than unaligned
- [x] `e2e_ipo_loss_finite` -- IPO loss is finite and non-negative
- [x] `e2e_kto_loss_nonneg` -- KTO loss is finite and >= 0
- [x] `e2e_orpo_structure` -- ORPO >= SFT loss when lambda > 0 and penalty > 0
- [x] `e2e_simpo_length_normalized` -- SimPO loss finite on different-length
      sequences
- [x] `e2e_sft_loss_correct_prediction` -- strong correct prediction -> loss
      < 0.01
- [x] `e2e_kl_zero_at_ref` -- KL(p || p) = 0
- [x] `e2e_reward_normalizer_unit_variance` -- Welford normaliser gives
      zero-mean unit-variance
- [x] `e2e_ptx_kernels_all_sm_versions` -- all 7 kernels x 6 SM versions
      contain `.version`, `.visible .entry`, `sm_X`, and kernel name

#### Benchmarks (benches/rlhf_ops.rs)
- [x] 7 PTX kernel groups x 4 SM versions (PTX generation throughput)
- [x] `dpo_batch_256` -- DPO loss on batch of 256 pairs
- [x] `ipo_batch_256` -- IPO loss on batch of 256 pairs
- [x] `kto_batch_256` -- KTO loss on batch of 256 examples
- [x] `sft_512tokens_32kvocab` -- masked SFT cross-entropy
- [x] `reward_norm_update` -- Welford online normaliser update
- [x] `bt_loss_batch_256` -- Bradley-Terry reward loss on batch of 256

### Future Enhancements [ ]

#### P0 -- Critical (Performance-Sensitive Paths)
- [x] Fused DPO loss + softplus-stable BCE in a single PTX kernel
      (currently host-side sigmoid + log)
- [ ] Tensor-Core path for `RewardModel` MLP forward + backward
- [x] Fused KL-penalty + advantage computation in PPO rollout
- [ ] Token-level GAE on GPU (currently CPU loop in `compute_advantages`)

#### P1 -- Important (Feature Completeness)
- [x] Constitutional AI (CAI) self-critique loop primitives
      (constitutional/mod.rs -- critique→revise pipeline with principle rotation,
      pluggable critique_fn/revise_fn closures, revision trace, SL-record +
      preference-pair harvest, model-free heuristic_revision; 14 tests)
- [x] RLAIF (RL from AI feedback) reward modelling helpers
      (reward/rlaif.rs -- soft AI-preference labels from judge logits,
      position-bias debiasing, self-consistency mean/variance, soft-label
      Bradley-Terry cross-entropy reward loss; 15 tests)
- [x] Cringe Loss -- negative-log-likelihood for forbidden continuations
- [x] Step-DPO -- step-wise preference optimisation for chain-of-thought
- [x] sDPO -- staged DPO with reference-model updates between stages
      (dpo/sdpo.rs -- StagedDpo driver with reference handoff; 22 tests; STALE
      checkbox -- already implemented)
- [x] DPO with Identity Preference Optimisation regularisation
      (DPO + IPO blend) (dpo/dpo_ipo_blend.rs -- convex blend
      (1-α)·L_DPO + α·λ·L_IPO; α=0→DPO, α=1→IPO bit-exact; 10 tests)
- [x] Length-controlled DPO -- penalty on response-length difference
- [x] Process-Reward Modelling (PRM) loss for step-level rewards
- [x] Best-of-N sampling helpers with score aggregation
      (reward/best_of_n.rs -- generate-N + reward-score + select; Max/Mean/SoftmaxWeighted/TopKMean aggregation; order-statistic expected-best-reward)
- [x] `rlhf/grpo.rs` — GRPO (Shao 2024): Group Relative Policy Optimisation; advantages computed from group of outputs without separate value model; KL regularisation vs reference; `GrpoConfig { group_size: usize, kl_coeff: f32 }`
- [x] `rlhf/rebel.rs` — REBEL (Gao 2024): Regression-Based RL; direct regression of reward differences onto token log-prob differences; no policy gradient variance; `RebelConfig { tau: f32 }`
- [x] `reward/rm_calibration.rs` — Reward model calibration (Touvron 2023): temperature scaling + margin-based reliability; isotonic regression on held-out preference pairs; `RewardModelCalibrator` (already implemented in reward/rm_calibration.rs -- STALE checkbox)
- [x] `constitutional/mod.rs` — Constitutional AI (Bai 2022): revision-based self-critique pipeline; apply critique prompt → revise → collect revised samples as SL data; `ConstitutionalConfig { principles: Vec<String>, rounds }` + `run_constitutional_revision` (filename is constitutional/mod.rs, not rlhf/constitutional.rs)

#### P2 -- Nice-to-Have (Advanced Features)
- [x] Online DPO with rejection sampling
      (dpo/online_dpo.rs -- N-candidate scored pairs, BestWorst/BestVsRest/Threshold
      pairing; 22 tests; STALE checkbox -- already implemented)
- [x] Reward model ensembling and uncertainty estimation
      (reward/ensemble.rs -- Mean/Min/WeightedMean aggregation, cross-model std uncertainty, pessimistic penalized_reward = aggregate − λ·std)
- [x] DPO with offline policy regularisation
      (dpo/dpo_sft_mix.rs -- DPO + λ_reg·mean[(π_chosen − μ_behaviour)²] quadratic
      trust region around the offline behaviour policy; 11 tests)
- [x] Soft-Actor-Critic-style entropy-regularised RLHF
      (ppo_rlhf/sac_rlhf.rs -- soft target r − α·logπ, soft policy/value loss,
      auto temperature tuning toward target entropy; 15 tests)
- [x] Multi-objective alignment (helpfulness + harmlessness Pareto front)
      (metrics/multi_objective.rs -- Pareto-front extraction via dominance,
      weighted-sum + Chebyshev scalarisation, front selection; 14 tests)
- [ ] FP8 inference path for `RewardModel` on Hopper / Ada (requires GPU hardware)
- [ ] Distributed PPO with parameter server / all-reduce stubs (requires GPU/multi-device runtime)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |

No CUDA SDK, no C, no Fortran. The crate compiles standalone and produces PTX
strings that can be consumed by `oxicuda-driver` / `oxicuda-launch` at runtime.

## Quality Status

- Warnings: 0 (clippy clean, `-D warnings` all-features all-targets)
- Tests: 604 passing (focused, high-coverage; includes 12 E2E; +30 from wired
  `loss/dpo_loss` + `loss/ppo_loss` + `utils/reward_norm` orphan modules; +31
  finite-difference gradient tests for DPO / IPO / KTO / SimPO / ORPO / PPO;
  +76 finite-difference gradient tests for the extended family BCO / DPOP / SLiC
  / Step-DPO / sDPO / RRHF / DPO+IPO blend / DPO+SFT mix / length-DPO / online-DPO
  / Bradley-Terry / soft-BT RLAIF / PRM / GRPO / REBEL / RLOO / SAC-RLHF)
- unwrap() calls: 0 (production code)
- All public APIs return `RlhfResult<T>` or `Result<T, RlhfError>`

## Performance Targets

Reference shapes (DPO / IPO / KTO are dominated by per-pair sigmoid + reduce;
SFT is dominated by softmax over vocabulary):

| Kernel | Shape | Target |
|--------|-------|--------|
| bt_reward_loss | batch = 256 | reduction-bound |
| dpo_loss | batch = 256 | sigmoid + reduce |
| ipo_loss | batch = 256 | reduction-bound |
| kto_loss | batch = 256 | sigmoid + reduce |
| orpo_odds | batch = 256, seq_len = 1024 | log + reduce |
| rlhf_kl | batch = 256, seq_len = 1024 | exp + reduce |
| sft_mask | seq_len = 512, vocab = 32768 | softmax-bound |

## Notes

- All losses operate on log-probabilities (not logits) to avoid double-softmax
  bugs; the `PairBatch` constructor validates that
  chosen_logps.len() == rejected_logps.len() == chosen_ref_logps.len() ==
  rejected_ref_logps.len().
- `DpoConfig::beta`, `KtoConfig::beta`, `IpoConfig::beta`, and
  `SimpoConfig::beta` all must be positive; validation occurs at first-use
  time inside the loss function.
- `RewardNormalizer` uses Welford's algorithm so single-pass online updates
  are numerically stable for very large reward streams.
- `KlController` updates beta multiplicatively as in the original Ziegler et
  al. (2019) PPO-RLHF formulation; the proportionality constant `k` can be
  tuned per training run.
- `sft_loss` divides by the sum of the attention mask, so padding tokens do
  not contribute to the reported loss.
- `OrpoConfig::lambda` weights the odds-ratio penalty added to SFT loss;
  setting lambda = 0 reduces ORPO to plain SFT (verified by E2E test layout).

---

## Architecture-Specific Deepening

### Hopper (sm_90 / sm_90a)
- [ ] `wgmma.mma_async` path for `RewardModel` MLP and SFT logits projection
- [ ] TMA (`cp.async.bulk`) loading of preference-pair batches

### Ampere (sm_80 / sm_86) / Ada (sm_89)
- [ ] `cp.async` prefetch of log-probability tensors
- [ ] Warp-shuffle reduction in `bt_reward_loss_ptx` and `rlhf_kl_ptx`

### Blackwell (sm_100 / sm_120)
- [ ] 5th-gen Tensor Core path for `RewardModel` inference
- [ ] Cluster launch for cross-CTA SFT cross-entropy across very large vocab
      (>= 256k tokens)

---

## Deepening Opportunities

### Verification Gaps
- [x] All 7 PTX generators emit `.version`, `.target sm_X`, and named entry per
      SM version (verified by `e2e_ptx_kernels_all_sm_versions`)
- [x] BT loss = -log(0.5) for equal rewards (E2E)
- [x] BT loss is monotone in reward gap (E2E)
- [x] DPO loss is lower for aligned pairs (E2E)
- [x] KL(p || p) = 0 (E2E)
- [x] Welford normaliser gives mean approx 0 and variance approx 1 (E2E)
- [ ] DPO / IPO / KTO numerical parity vs. reference TRL implementation within
      1e-4
- [ ] PPO clipped surrogate gradient cross-checked against PyTorch reference
      (NOTE: gradient now implemented and finite-difference-verified against the
      crate's OWN forward -- loss/ppo_loss.rs::policy_grad, incl. zero-gradient
      in the clipped-and-binding region; external PyTorch parity still TODO)
- [ ] Length-normalisation correctness of SimPO across mixed-length batches

### Implementation Deepening
- [x] All preference losses are batch-vectorised (no per-pair Python-style
      loops in hot paths)
- [x] All losses validate input shapes before computation (`PairBatch::new`)
- [x] Numerically stable sigmoid via softplus formulation (DPO / IPO / KTO)
- [x] Logsumexp trick in `sft_loss` for numerical stability
- [ ] Mixed-precision (bf16 storage, fp32 accumulate) variants of all losses
      (requires GPU/bf16 hardware path)
- [x] Gradient implementations for the CORE preference-optimisation losses
      (analytic, each finite-difference-verified against that loss's OWN f32
      forward; central-difference max relative error measured ≤ 1e-4 for every
      gradient):
      - [x] DPO (dpo/dpo.rs::dpo_grad_per_pair, dpo_grad, DpoGrad) -- ∂L/∂(four
            logprobs) via dL/dΔ = −β·σ(−β·Δ); max FD rel-err 1.8e-5
      - [x] DPO + label smoothing (loss/dpo_loss.rs::DpoLoss::grad, DpoGradients)
            -- slice-wise ∂L/∂(lp_w, lp_l, ref_w, ref_l); max FD rel-err 2.2e-5
      - [x] IPO (dpo/ipo.rs::ipo_grad, IpoGrad) -- dL/dh = 2(h − 1/2β) chained;
            max FD rel-err 9.5e-7
      - [x] KTO (dpo/kto.rs::kto_grad, KtoGrad) -- ∂L/∂reward, ∓λ·β·σ(a)(1−σ(a))
            for desirable/undesirable; max FD rel-err 4.6e-5
      - [x] SimPO (orpo/simpo.rs::simpo_grad, SimpoGrad) -- length-normalised
            ∂L/∂(summed logp); max FD rel-err 4.5e-5
      - [x] ORPO (orpo/orpo.rs::orpo_grad, log_odds_grad, OrpoGrad) -- odds-ratio
            chain incl. clamp/floor-aware log-odds derivative + ∂L/∂L_SFT = 1;
            max FD rel-err 7.4e-5
      - [x] PPO (loss/ppo_loss.rs::policy_grad / value_grad / entropy_grad /
            total_grad, PpoGrad) -- clip-aware policy surrogate (gradient is
            EXACTLY 0 where clipped-and-binding), value-MSE (v−ret)/N, entropy
            bonus −entropy_coeff; max FD rel-err 8.1e-5
      Extended preference / reward / RL gradients (each analytic, FD-verified
      against the crate's OWN f32 forward; measured central-difference max
      rel-err shown):
      - [x] BCO (dpo/bco.rs::bco_grad, BcoGrad) -- BCE on implicit rewards,
            ∂L/∂r_des = −σ(δ−r)/N_d, ∂L/∂r_und = +σ(r−δ)/N_u; FD-verified, max
            rel-err 2.1e-5
      - [x] DPOP (dpo/dpop.rs::dpop_grad_per_pair, dpop_grad, DpopGrad) --
            penalty-aware ∂L/∂(four logp): chosen-side slope β+λ where the
            deficit max(0,rc−c) is active, β elsewhere; FD-verified, max rel-err
            2.7e-5
      - [x] SLiC (dpo/slic.rs::slic_grad, slic_grad_batch, SlicGrad) -- hinge
            sub-gradient (∓1 binding / 0 in satisfied region) + linear −λ on the
            reference term; FD-verified, max rel-err 7.0e-6
      - [x] Step-DPO (dpo/step_dpo.rs::step_dpo_grad, StepDpoGrad) -- per-step
            DPO grad weighted by the WeightedMean/WeightedSum/LastStep aggregation
            coefficient; FD-verified, max rel-err 4.6e-5
      - [x] sDPO (dpo/sdpo.rs::sdpo_stage_grad, SdpoStageGrad) -- DPO closed form
            against the handed-off stage reference; FD-verified, max rel-err
            2.0e-5
      - [x] RRHF ranking (dpo/rrhf.rs::ranking_grad, rrhf_grad, RrhfGrad) --
            pairwise-hinge sub-gradient (±1 per active mis-ordered pair) + best-
            response ft term, length-normalised; FD-verified, max rel-err 2.3e-5
      - [x] DPO+IPO blend (dpo/dpo_ipo_blend.rs::blend_grad_per_pair,
            dpo_ipo_blend_grad, BlendGrad) -- (1−α)·DPO + α·λ·IPO chained through
            Δ; FD-verified, max rel-err 2.2e-6
      - [x] DPO+SFT mix (dpo/dpo_sft_mix.rs::dpo_sft_mix_grad, MixGrad) -- DPO +
            −λ_sft/N SFT anchor + 2λ_reg(c−μ)/N offline regulariser on the chosen
            logp; FD-verified, max rel-err 9.1e-5
      - [x] length-DPO (dpo/length_dpo.rs::LengthDpo::loss_grad_per_pair,
            loss_grad, LengthDpoGrad) -- length-normalised DPO grad (penalty is
            constant in the logps); FD-verified, max rel-err 6.6e-5
      - [x] online-DPO (dpo/online_dpo.rs::online_dpo_grad, OnlineDpoGrad) -- DPO
            grad on the reward-selected best/worst pair (selection held fixed),
            zero on unselected candidates; FD-verified, max rel-err 1.1e-5
      - [x] Bradley-Terry reward (preference/bradley_terry.rs::bt_reward_grad,
            BtRewardGrad) -- ∂/∂(r_w,r_l) of −log σ(r_w−r_l) = ∓σ(−(r_w−r_l))/N;
            FD-verified, max rel-err 7.8e-6
      - [x] soft-BT RLAIF (reward/rlaif.rs::soft_bt_pair_grad, soft_bt_reward_grad,
            SoftBtGrad) -- cross-entropy grad dL/dΔ = σ(Δ)−p; FD-verified, max
            rel-err 1.7e-4
      - [x] PRM (reward/process_reward.rs::prm_grad, PrmGrad) -- step-BCE
            ∂/∂z = σ(z)−ỹ (label-smoothed) + last-step solution term; FD-verified,
            max rel-err 1.4e-5
      RL estimators -- assessed case-by-case; each forward takes the stochastic
      quantities (advantages / rewards / old & ref logps / q-values) as already-
      computed inputs, so each reduces to a deterministic FD-checkable scalar
      gradient and NONE required honest deferral:
      - [x] GRPO (grpo/mod.rs::output_surrogate_grad) -- clipped surrogate wrt
            token logps with group-advantage held as input (EXACTLY 0 where the
            clip binds) plus the k3-KL term ∂KL/∂lp = 1−exp(ref−lp); FD-verified,
            max rel-err 1.8e-4
      - [x] REBEL (rebel/mod.rs::rebel_pair_grad, rebel_grad, RebelGrad) --
            squared-regression residual: ∂L/∂logp_a = 2·residual/η,
            ∂L/∂logp_b = −2·residual/η; FD-verified, max rel-err 1.6e-6
      - [x] RLOO (ppo_rlhf/rloo.rs::rloo_grad) -- leave-one-out advantage-weighted
            logprob grad ∂L/∂logp_i = −Âᵢ/k with advantages held as input;
            FD-verified, max rel-err 1.4e-6
      - [x] SAC-RLHF (ppo_rlhf/sac_rlhf.rs::sac_policy_grad/sac_value_grad/
            sac_temperature_grad, SacPolicyGrad, SacValueGrad) -- soft policy loss
            (α/N, −1/N), soft-target value MSE ((v−y)/N, α(v−y)/N), temperature
            (−α/N); FD-verified, max rel-err 2.8e-5
      (Cringe already exposes the negative-hinge subgradient,
      dpo/cringe.rs::negative_hinge_grad; a crate-wide autodiff-graph redesign
      remains future work.)
- [x] Top-k / top-p sampling helpers for rollout generation
      (ppo_rlhf/sampling.rs -- temperature softmax + TopK/TopP/TopKThenP
      truncation + renormalise, inverse-CDF sampling via LcgRng, greedy decode;
      16 tests)
- [x] Reference-model log-probability caching for stationary reference
      (avoid re-forwarding on every step) (utils/ref_cache.rs -- id-keyed
      scalar/per-token cache with get_or_compute + hit/miss accounting +
      capacity bound; 11 tests)
- [x] Mixed DPO + SFT auxiliary loss combiner
      (dpo/dpo_sft_mix.rs -- L_DPO + λ_sft·NLL(chosen) (+ optional offline-reg);
      11 tests)
