//! AdaLoRA continuous importance EMA + scheduled (cubic) budget pruning.
//!
//! Reference: Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W., & Zhao, T.
//! (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*.
//! International Conference on Learning Representations (ICLR).
//! <https://arxiv.org/abs/2303.10512>
//!
//! The base [`AdaloraLinear`](crate::lora::adalora::AdaloraLinear) supports a *one-shot* prune to
//! a fixed target rank. This module adds the paper's actual training-time mechanism:
//!
//! 1. **Sensitivity** of every triplet `(P[:,i], λ_i, Q[i,:])` is summarised by a per-rank
//!    sensitivity `I_i = |λ_i · g_i|` where `g_i` is the gradient of the loss w.r.t. `λ_i`
//!    (a cheap proxy for the full triplet importance that the paper also uses).
//! 2. Because mini-batch sensitivities are noisy, AdaLoRA *smooths* them with an exponential
//!    moving average and an uncertainty term:
//!
//!    ```text
//!      Ī_i  ← β₁·Ī_i + (1-β₁)·I_i                (smoothed sensitivity)
//!      Ū_i  ← β₂·Ū_i + (1-β₂)·|I_i − Ī_i|        (uncertainty / instability)
//!      s_i   = Ī_i · Ū_i                          (importance score)
//!    ```
//!
//! 3. The total *budget* (number of retained singular triplets across the whole adapter) follows
//!    a **cubic schedule** that warms up from the initial budget `b⁽⁰⁾`, holds, then anneals to
//!    the final budget `b⁽ᵀ⁾`:
//!
//!    ```text
//!              ⎧ b⁽⁰⁾                                                 t < t_i
//!      b(t) =  ⎨ b⁽ᵀ⁾ + (b⁽⁰⁾ − b⁽ᵀ⁾)·(1 − (t−t_i)/(T−t_i−t_f))³     t_i ≤ t < T−t_f
//!              ⎩ b⁽ᵀ⁾                                                 t ≥ T−t_f
//!    ```
//!
//! 4. At each pruning step the lowest-importance singular values are masked (set to zero) so that
//!    exactly `b(t)` of them remain non-zero, mirroring `prune_to_target` but driven by the
//!    schedule and the smoothed scores instead of a static target.
//!
//! Everything here is deterministic and gradient-driven (no RNG); the caller supplies the
//! per-rank `λ`-gradient each step.

use crate::error::{PeftError, PeftResult};
use crate::lora::adalora::AdaloraLinear;

/// Hyper-parameters for the AdaLoRA importance schedule.
#[derive(Debug, Clone)]
pub struct AdaloraScheduleConfig {
    /// Initial budget `b⁽⁰⁾`: number of retained singular triplets at the start (usually `r`,
    /// i.e. nothing pruned during warm-up).
    pub init_budget: usize,
    /// Final budget `b⁽ᵀ⁾`: number of retained triplets after annealing (the target rank).
    pub final_budget: usize,
    /// Total number of training steps `T`.
    pub total_steps: usize,
    /// Warm-up steps `t_i` before any pruning begins.
    pub warmup_steps: usize,
    /// Final-warm-up steps `t_f`: the budget is held at `final_budget` for the last `t_f` steps.
    pub final_warmup_steps: usize,
    /// EMA coefficient `β₁` for the sensitivity smoothing, in `[0, 1)`.
    pub beta1: f32,
    /// EMA coefficient `β₂` for the uncertainty smoothing, in `[0, 1)`.
    pub beta2: f32,
}

impl AdaloraScheduleConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] when `total_steps == 0`.
    /// - [`PeftError::InvalidTargetRank`] when `final_budget > init_budget`.
    /// - [`PeftError::InvalidDensity`] when either β is outside `[0, 1)`.
    /// - [`PeftError::DimensionMismatch`] when the warm-up windows overlap
    ///   (`warmup_steps + final_warmup_steps >= total_steps`).
    pub fn validate(&self) -> PeftResult<()> {
        if self.total_steps == 0 {
            return Err(PeftError::EmptyInput);
        }
        if self.final_budget > self.init_budget {
            return Err(PeftError::InvalidTargetRank {
                target_r: self.final_budget,
                r: self.init_budget,
            });
        }
        for b in [self.beta1, self.beta2] {
            if !(0.0..1.0).contains(&b) {
                return Err(PeftError::InvalidDensity { density: b });
            }
        }
        if self.warmup_steps + self.final_warmup_steps >= self.total_steps {
            return Err(PeftError::DimensionMismatch {
                expected: self.total_steps,
                got: self.warmup_steps + self.final_warmup_steps,
            });
        }
        Ok(())
    }

    /// Cubic budget schedule `b(t)` (number of retained triplets at step `t`).
    ///
    /// Returns `init_budget` during warm-up, anneals cubically toward `final_budget`, then holds
    /// at `final_budget` during the final warm-up window. `t` is clamped to `[0, total_steps]`.
    #[must_use]
    pub fn budget_at(&self, t: usize) -> usize {
        let t_i = self.warmup_steps;
        let t_f = self.final_warmup_steps;
        let total = self.total_steps;
        if t <= t_i {
            return self.init_budget;
        }
        let anneal_end = total.saturating_sub(t_f);
        if t >= anneal_end {
            return self.final_budget;
        }
        // Cubic decay over (t_i, anneal_end).
        let span = (anneal_end - t_i) as f32;
        let progress = (t - t_i) as f32 / span; // (0, 1)
        let cubic = (1.0 - progress).powi(3); // 1 → 0
        let b0 = self.init_budget as f32;
        let bt = self.final_budget as f32;
        let value = bt + (b0 - bt) * cubic;
        // Round to nearest, clamp into [final_budget, init_budget].
        let rounded = value.round() as i64;
        let lo = self.final_budget as i64;
        let hi = self.init_budget as i64;
        rounded.clamp(lo, hi) as usize
    }
}

/// Stateful importance tracker driving an [`AdaloraLinear`] toward its budget over training.
#[derive(Debug, Clone)]
pub struct AdaloraScheduler {
    cfg: AdaloraScheduleConfig,
    /// Smoothed sensitivity `Ī_i`, one per singular value.
    sensitivity: Vec<f32>,
    /// Smoothed uncertainty `Ū_i`, one per singular value.
    uncertainty: Vec<f32>,
    /// Whether any update has been observed yet (first step seeds the EMAs directly).
    initialised: bool,
    /// Number of [`Self::step`] calls performed.
    step_idx: usize,
}

impl AdaloraScheduler {
    /// Create a scheduler for an adapter of `rank` singular values.
    ///
    /// # Errors
    ///
    /// - Forwards [`AdaloraScheduleConfig::validate`] errors.
    /// - [`PeftError::EmptyInput`] when `rank == 0`.
    /// - [`PeftError::RankTooLarge`] when `init_budget > rank`.
    pub fn new(rank: usize, cfg: AdaloraScheduleConfig) -> PeftResult<Self> {
        cfg.validate()?;
        if rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        if cfg.init_budget > rank {
            return Err(PeftError::RankTooLarge {
                rank: cfg.init_budget,
                dim: rank,
            });
        }
        Ok(Self {
            cfg,
            sensitivity: vec![0.0; rank],
            uncertainty: vec![0.0; rank],
            initialised: false,
            step_idx: 0,
        })
    }

    /// Current step index (number of [`Self::step`] calls).
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step_idx
    }

    /// Borrow the smoothed sensitivity `Ī`.
    #[must_use]
    pub fn sensitivity(&self) -> &[f32] {
        &self.sensitivity
    }

    /// Borrow the smoothed uncertainty `Ū`.
    #[must_use]
    pub fn uncertainty(&self) -> &[f32] {
        &self.uncertainty
    }

    /// Importance score `s_i = Ī_i · Ū_i` for every singular value.
    #[must_use]
    pub fn importance(&self) -> Vec<f32> {
        self.sensitivity
            .iter()
            .zip(self.uncertainty.iter())
            .map(|(&s, &u)| s * u)
            .collect()
    }

    /// Update the smoothed sensitivity/uncertainty from this step's raw sensitivities `I_i`.
    ///
    /// `raw` must hold one sensitivity per singular value (`|λ_i · g_i|`, supplied by the caller).
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `raw.len()` differs from the tracked rank.
    pub fn observe(&mut self, raw: &[f32]) -> PeftResult<()> {
        if raw.len() != self.sensitivity.len() {
            return Err(PeftError::DimensionMismatch {
                expected: self.sensitivity.len(),
                got: raw.len(),
            });
        }
        let b1 = self.cfg.beta1;
        let b2 = self.cfg.beta2;
        if !self.initialised {
            // Seed EMAs so a single observation already yields meaningful importance.
            for (i, &r) in raw.iter().enumerate() {
                self.sensitivity[i] = r;
                self.uncertainty[i] = 0.0;
            }
            self.initialised = true;
        } else {
            for (i, &r) in raw.iter().enumerate() {
                let new_sens = b1 * self.sensitivity[i] + (1.0 - b1) * r;
                let instab = (r - new_sens).abs();
                self.uncertainty[i] = b2 * self.uncertainty[i] + (1.0 - b2) * instab;
                self.sensitivity[i] = new_sens;
            }
        }
        Ok(())
    }

    /// Compute per-rank raw sensitivity `I_i = |λ_i · g_i|` from an adapter and its `λ`-gradient.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `lambda_grad.len()` differs from `adapter.rank`.
    pub fn raw_sensitivity(adapter: &AdaloraLinear, lambda_grad: &[f32]) -> PeftResult<Vec<f32>> {
        if lambda_grad.len() != adapter.rank {
            return Err(PeftError::DimensionMismatch {
                expected: adapter.rank,
                got: lambda_grad.len(),
            });
        }
        Ok(adapter
            .lambda
            .iter()
            .zip(lambda_grad.iter())
            .map(|(&l, &g)| (l * g).abs())
            .collect())
    }

    /// Perform one scheduled-pruning step: ingest `raw` sensitivities, advance the step counter,
    /// then mask `adapter.lambda` so exactly `budget_at(step)` singular values remain non-zero,
    /// keeping the highest-importance ones.
    ///
    /// Returns the budget that was applied at this step.
    ///
    /// # Errors
    ///
    /// - [`PeftError::DimensionMismatch`] when `raw.len()` differs from the tracked rank, or when
    ///   the adapter's rank differs from the tracked rank.
    pub fn step(&mut self, adapter: &mut AdaloraLinear, raw: &[f32]) -> PeftResult<usize> {
        if adapter.lambda.len() != self.sensitivity.len() {
            return Err(PeftError::DimensionMismatch {
                expected: self.sensitivity.len(),
                got: adapter.lambda.len(),
            });
        }
        self.observe(raw)?;
        self.step_idx += 1;
        let budget = self.cfg.budget_at(self.step_idx).min(adapter.lambda.len());
        self.apply_budget(adapter, budget);
        Ok(budget)
    }

    /// Mask `adapter.lambda` down to `budget` non-zero entries, retaining the highest-importance
    /// singular values per the current smoothed scores.
    fn apply_budget(&self, adapter: &mut AdaloraLinear, budget: usize) {
        let scores = self.importance();
        let rank = adapter.lambda.len();
        // Indices sorted by importance ascending (least important first).
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let to_prune = rank.saturating_sub(budget);
        for &(i, _) in indexed.iter().take(to_prune) {
            adapter.lambda[i] = 0.0;
        }
    }

    /// Number of currently non-zero singular values in an adapter (a convenience for tests/tools).
    #[must_use]
    pub fn active_rank(adapter: &AdaloraLinear) -> usize {
        adapter.lambda.iter().filter(|&&v| v != 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::lora::adalora::AdaloraConfig;

    fn cfg() -> AdaloraScheduleConfig {
        AdaloraScheduleConfig {
            init_budget: 8,
            final_budget: 2,
            total_steps: 100,
            warmup_steps: 10,
            final_warmup_steps: 10,
            beta1: 0.85,
            beta2: 0.85,
        }
    }

    #[test]
    fn budget_schedule_is_monotone_non_increasing_and_bounded() {
        let c = cfg();
        let mut prev = c.budget_at(0);
        assert_eq!(prev, c.init_budget, "warm-up holds initial budget");
        for t in 0..=c.total_steps {
            let b = c.budget_at(t);
            assert!(
                b <= c.init_budget && b >= c.final_budget,
                "budget {b} out of range at t={t}"
            );
            assert!(
                b <= prev,
                "budget must be non-increasing: {b} > {prev} at t={t}"
            );
            prev = b;
        }
        assert_eq!(
            c.budget_at(c.total_steps),
            c.final_budget,
            "ends at final budget"
        );
        // During warm-up nothing is pruned.
        assert_eq!(c.budget_at(c.warmup_steps), c.init_budget);
        // After the anneal window the budget is final.
        assert_eq!(
            c.budget_at(c.total_steps - c.final_warmup_steps),
            c.final_budget
        );
    }

    #[test]
    fn cubic_decays_faster_early() {
        // The cubic (1-p)^3 shape means the budget drops slowly right after warm-up and fast
        // near the end. Check the midpoint budget sits below the linear interpolant.
        let c = cfg();
        let t_mid = (c.warmup_steps + (c.total_steps - c.final_warmup_steps)) / 2;
        let b_mid = c.budget_at(t_mid) as f32;
        let linear_mid = (c.init_budget as f32 + c.final_budget as f32) / 2.0;
        // (1-0.5)^3 = 0.125, so value = 2 + 6*0.125 = 2.75 < linear 5.0.
        assert!(
            b_mid < linear_mid,
            "cubic midpoint {b_mid} not below linear {linear_mid}"
        );
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        let mut bad = cfg();
        bad.final_budget = 100;
        assert!(matches!(
            bad.validate(),
            Err(PeftError::InvalidTargetRank { .. })
        ));
        let mut bad2 = cfg();
        bad2.beta1 = 1.5;
        assert!(matches!(
            bad2.validate(),
            Err(PeftError::InvalidDensity { .. })
        ));
        let mut bad3 = cfg();
        bad3.total_steps = 0;
        assert!(matches!(bad3.validate(), Err(PeftError::EmptyInput)));
        let mut bad4 = cfg();
        bad4.warmup_steps = 60;
        bad4.final_warmup_steps = 60;
        assert!(matches!(
            bad4.validate(),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    /// A schedule config sized for a rank-3 adapter (so `init_budget <= rank`).
    fn small_cfg(rank: usize) -> AdaloraScheduleConfig {
        AdaloraScheduleConfig {
            init_budget: rank,
            final_budget: 1,
            total_steps: 100,
            warmup_steps: 10,
            final_warmup_steps: 10,
            beta1: 0.85,
            beta2: 0.85,
        }
    }

    #[test]
    fn ema_smooths_noisy_sensitivities() {
        let mut sched = AdaloraScheduler::new(3, small_cfg(3)).expect("valid scheduler");
        // First observation seeds the EMA directly.
        sched.observe(&[1.0, 0.0, 0.5]).expect("observe ok");
        assert_eq!(sched.sensitivity(), &[1.0, 0.0, 0.5]);
        assert_eq!(sched.uncertainty(), &[0.0, 0.0, 0.0]);
        // Second, noisy observation must move the EMA only partway (β=0.85).
        sched.observe(&[0.0, 1.0, 0.5]).expect("observe ok");
        let s = sched.sensitivity();
        assert!((s[0] - 0.85).abs() < 1e-5, "Ī[0]={}", s[0]);
        assert!((s[1] - 0.15).abs() < 1e-5, "Ī[1]={}", s[1]);
        assert!((s[2] - 0.5).abs() < 1e-5, "Ī[2]={}", s[2]);
        // Uncertainty grew where the signal was unstable (ranks 0 and 1) and stayed 0 at rank 2.
        let u = sched.uncertainty();
        assert!(u[0] > 0.0 && u[1] > 0.0);
        assert!(
            u[2].abs() < 1e-6,
            "stable rank should have ~0 uncertainty, got {}",
            u[2]
        );
    }

    #[test]
    fn dimension_mismatch_is_caught() {
        let mut sched = AdaloraScheduler::new(4, small_cfg(4)).expect("valid");
        assert!(matches!(
            sched.observe(&[1.0, 2.0]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn raw_sensitivity_is_abs_lambda_times_grad() {
        let mut rng = LcgRng::new(42);
        let acfg = AdaloraConfig {
            r: 4,
            alpha: 8.0,
            target_r: 2,
        };
        let adapter = AdaloraLinear::new(8, 8, &acfg, &mut rng);
        // lambda is 0.01·ones at init.
        let grad = vec![2.0, -3.0, 0.0, 10.0];
        let raw = AdaloraScheduler::raw_sensitivity(&adapter, &grad).expect("ok");
        assert!((raw[0] - 0.02).abs() < 1e-6);
        assert!((raw[1] - 0.03).abs() < 1e-6);
        assert!((raw[2] - 0.0).abs() < 1e-6);
        assert!((raw[3] - 0.10).abs() < 1e-6);
    }

    #[test]
    fn scheduled_pruning_reaches_final_budget_and_keeps_important_ranks() {
        let mut rng = LcgRng::new(7);
        let acfg = AdaloraConfig {
            r: 8,
            alpha: 16.0,
            target_r: 2,
        };
        let mut adapter = AdaloraLinear::new(16, 16, &acfg, &mut rng);
        let schedule = AdaloraScheduleConfig {
            init_budget: 8,
            final_budget: 2,
            total_steps: 40,
            warmup_steps: 5,
            final_warmup_steps: 5,
            beta1: 0.8,
            beta2: 0.8,
        };
        let mut sched = AdaloraScheduler::new(8, schedule).expect("valid");
        // AdaLoRA importance is Ī·Ū (smoothed sensitivity × uncertainty), so a rank only scores
        // high if it is BOTH sensitive and unstable. Drive ranks 0 and 1 with large oscillating
        // sensitivities (high Ī, high Ū); keep the rest at a tiny constant (low Ī, ~0 Ū).
        let mut last_budget = 8;
        for t in 0..40 {
            let big = if t % 2 == 0 { 8.0 } else { 2.0 };
            let raw: Vec<f32> = (0..8).map(|i| if i < 2 { big } else { 0.001 }).collect();
            last_budget = sched.step(&mut adapter, &raw).expect("step ok");
        }
        assert_eq!(last_budget, 2, "schedule must end at the final budget");
        let active = AdaloraScheduler::active_rank(&adapter);
        assert!(active <= 2, "active rank {active} exceeds final budget");
        // The two consistently-important ranks must have survived (non-zero λ).
        assert!(adapter.lambda[0] != 0.0, "important rank 0 was pruned");
        assert!(adapter.lambda[1] != 0.0, "important rank 1 was pruned");
    }

    #[test]
    fn budget_only_shrinks_during_training() {
        let mut rng = LcgRng::new(11);
        let acfg = AdaloraConfig {
            r: 6,
            alpha: 12.0,
            target_r: 3,
        };
        let mut adapter = AdaloraLinear::new(8, 8, &acfg, &mut rng);
        let schedule = AdaloraScheduleConfig {
            init_budget: 6,
            final_budget: 3,
            total_steps: 30,
            warmup_steps: 3,
            final_warmup_steps: 3,
            beta1: 0.9,
            beta2: 0.9,
        };
        let mut sched = AdaloraScheduler::new(6, schedule).expect("valid");
        let mut prev_active = 6usize;
        for _ in 0..30 {
            let raw: Vec<f32> = (0..6).map(|i| (i as f32 + 1.0) * 0.5).collect();
            sched.step(&mut adapter, &raw).expect("step ok");
            let active = AdaloraScheduler::active_rank(&adapter);
            assert!(
                active <= prev_active,
                "active rank grew: {active} > {prev_active}"
            );
            prev_active = active;
        }
        assert!(
            prev_active <= 3,
            "did not reach final budget, active={prev_active}"
        );
    }

    #[test]
    fn step_rejects_mismatched_adapter_rank() {
        let mut rng = LcgRng::new(3);
        let acfg = AdaloraConfig {
            r: 4,
            alpha: 8.0,
            target_r: 2,
        };
        let mut adapter = AdaloraLinear::new(8, 8, &acfg, &mut rng);
        let mut sched = AdaloraScheduler::new(8, cfg()).expect("valid"); // rank 8 ≠ adapter rank 4
        assert!(matches!(
            sched.step(&mut adapter, &[0.0; 8]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn pruning_trajectory_param_count_matches_budget_times_dims() {
        // Drive the AdaLoRA importance scheduler through a full cubic-budget
        // trajectory and assert, at *every* step, that (a) the active singular-rank
        // count equals the scheduled budget, (b) the retained-parameter count equals
        // the analytic `budget × (out + in + 1)` accounting, and (c) the survivors
        // are exactly the highest-importance triplets.
        let in_features = 4usize;
        let out_features = 5usize;
        let rank = 6usize;
        let mut rng = LcgRng::new(123);
        let acfg = AdaloraConfig {
            r: rank,
            alpha: 12.0,
            target_r: 2,
        };
        let mut adapter = AdaloraLinear::new(in_features, out_features, &acfg, &mut rng);

        // Pin the per-rank parameter geometry directly from the adapter storage:
        // one P column (out_features) + one Q row (in_features) + one singular value.
        assert_eq!(adapter.p.len(), out_features * rank);
        assert_eq!(adapter.q.len(), in_features * rank);
        let p_col = adapter.p.len() / rank; // = out_features
        let q_row = adapter.q.len() / rank; // = in_features
        let params_per_rank = p_col + q_row + 1;

        let schedule_cfg = AdaloraScheduleConfig {
            init_budget: rank,
            final_budget: 2,
            total_steps: 24,
            warmup_steps: 4,
            final_warmup_steps: 4,
            beta1: 0.8,
            beta2: 0.8,
        };
        let schedule_for_budget = schedule_cfg.clone();
        let mut sched = AdaloraScheduler::new(rank, schedule_cfg).expect("valid scheduler");

        for t in 0..schedule_for_budget.total_steps {
            // Oscillating, index-scaled sensitivities make the smoothed importance
            // s_i ∝ (i+1)² — a strict, stable ordering — so the prune set is nested
            // (lowest indices pruned first) and the active rank equals the budget.
            let mult = if t % 2 == 0 { 2.0_f32 } else { 1.0_f32 };
            let raw: Vec<f32> = (0..rank).map(|i| (i as f32 + 1.0) * mult).collect();
            let budget = sched.step(&mut adapter, &raw).expect("step ok");

            let step_idx = t + 1;
            assert_eq!(
                budget,
                schedule_for_budget.budget_at(step_idx),
                "returned budget disagrees with the schedule at step {step_idx}"
            );

            let active_indices: Vec<usize> = adapter
                .lambda
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v != 0.0)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(
                active_indices.len(),
                budget,
                "active rank != budget at step {step_idx}"
            );

            let retained_params = active_indices.len() * params_per_rank;
            assert_eq!(
                retained_params,
                budget * params_per_rank,
                "retained-parameter count != budget×dims at step {step_idx}"
            );

            let expected_survivors: Vec<usize> = (rank - budget..rank).collect();
            assert_eq!(
                active_indices, expected_survivors,
                "survivors are not the highest-importance triplets at step {step_idx}"
            );
        }

        assert_eq!(
            AdaloraScheduler::active_rank(&adapter),
            schedule_for_budget.final_budget,
            "trajectory must terminate at the final budget"
        );
    }
}
