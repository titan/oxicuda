//! ReLoRA — Periodic merge-and-restart low-rank adaptation.
//!
//! Reference: Lialin, V., Muckatira, S., Shivagunde, N., & Rumshisky, A. (2024).
//! *ReLoRA: High-Rank Training Through Low-Rank Updates*. ICLR 2024.
//! <https://arxiv.org/abs/2307.05695>
//!
//! A single LoRA adapter `ΔW = scale · B·A` is at most rank `r`. ReLoRA reaches an effectively
//! *higher* rank by **periodically merging** the current adapter into the frozen base weight and
//! then **resetting** the factors:
//!
//! ```text
//!   every `merge_every` steps:
//!       W₀  ← W₀ + scale · B·A          (merge accumulated low-rank update)
//!       A   ← fresh N(0, init_scale²)   (re-randomise the down-projection)
//!       B   ← 0                          (zero the up-projection)
//!       (optimiser state for A, B is also reset; modelled here by the factor reset)
//! ```
//!
//! Summing `k` independent rank-`r` updates can produce a weight delta of rank up to `k·r`,
//! recovering much of full-rank training's expressivity at a fraction of the optimiser memory.
//!
//! ReLoRA pairs this with a **jagged cosine** learning-rate schedule: after every reset the LR
//! warms up linearly for `warmup` steps and then follows a cosine decay until the next reset.
//! [`ReloraSchedule`] provides this multiplier in `[0, 1]`.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a [`ReloraLinear`] adapter.
#[derive(Debug, Clone)]
pub struct ReloraConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension `r` of each restart (`> 0`, `≤ min(in, out)`).
    pub rank: usize,
    /// Scaling factor α; the effective scale is `α / r`.
    pub alpha: f32,
    /// Standard deviation used when (re-)initialising `A`.
    pub init_scale: f32,
    /// Number of optimisation steps between successive merge-and-reset events (`> 0`).
    pub merge_every: usize,
}

/// ReLoRA adapter for a single linear layer.
#[derive(Debug, Clone)]
pub struct ReloraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension `r`.
    pub rank: usize,
    /// Effective scale `α / r`.
    pub scale: f32,
    /// Std-dev for re-initialising `A`.
    pub init_scale: f32,
    /// Frozen base weight `W₀`, shape `[out_features × in_features]` (row-major).
    pub w: Vec<f32>,
    /// Down-projection `A`, shape `[r × in_features]` (row-major).
    pub a: Vec<f32>,
    /// Up-projection `B`, shape `[out_features × r]` (row-major). Zero after each reset.
    pub b: Vec<f32>,
    /// Steps between merge-and-reset events.
    pub merge_every: usize,
    /// Number of merge-and-reset events performed so far.
    pub num_restarts: usize,
}

impl ReloraLinear {
    /// Construct a new ReLoRA adapter.
    ///
    /// `W₀` is zero-initialised, `A ~ N(0, init_scale²)`, `B = 0`.
    ///
    /// # Errors
    /// - [`PeftError::ZeroBlockSize`] if `rank == 0` or `merge_every == 0`.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
    pub fn new(cfg: &ReloraConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        if cfg.rank == 0 || cfg.merge_every == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        let upper = cfg.in_features.min(cfg.out_features);
        if cfg.rank > upper {
            return Err(PeftError::RankTooLarge {
                rank: cfg.rank,
                dim: upper,
            });
        }
        let scale = cfg.alpha / cfg.rank as f32;
        let mut a = vec![0.0_f32; cfg.rank * cfg.in_features];
        rng.fill_normal(&mut a);
        for v in a.iter_mut() {
            *v *= cfg.init_scale;
        }
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            rank: cfg.rank,
            scale,
            init_scale: cfg.init_scale,
            w: vec![0.0_f32; cfg.out_features * cfg.in_features],
            a,
            b: vec![0.0_f32; cfg.out_features * cfg.rank],
            merge_every: cfg.merge_every,
            num_restarts: 0,
        })
    }

    /// Forward pass `y = (W₀ + scale · B·A)·x`.
    ///
    /// # Errors
    /// [`PeftError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = mat_vec(&self.w, x, self.out_features, self.in_features);
        let tmp = mat_vec(&self.a, x, self.rank, self.in_features);
        let delta = mat_vec(&self.b, &tmp, self.out_features, self.rank);
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += self.scale * d;
        }
        Ok(out)
    }

    /// The current low-rank delta `scale · B·A` as a flat `[out × in]` matrix.
    #[must_use]
    pub fn lora_delta(&self) -> Vec<f32> {
        let mut result = vec![0.0_f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let b_ik = self.b[i * self.rank + k];
                for j in 0..self.in_features {
                    result[i * self.in_features + j] +=
                        self.scale * b_ik * self.a[k * self.in_features + j];
                }
            }
        }
        result
    }

    /// Merge the current adapter into `W₀` and reset the factors (one ReLoRA restart).
    ///
    /// `W₀ += scale · B·A`; then `A` is re-randomised and `B` is zeroed. Increments
    /// [`ReloraLinear::num_restarts`].
    pub fn merge_and_reset(&mut self, rng: &mut LcgRng) {
        let delta = self.lora_delta();
        for (w, d) in self.w.iter_mut().zip(delta.iter()) {
            *w += d;
        }
        // Re-initialise A ~ N(0, init_scale²).
        rng.fill_normal(&mut self.a);
        for v in self.a.iter_mut() {
            *v *= self.init_scale;
        }
        // Zero B.
        for v in self.b.iter_mut() {
            *v = 0.0;
        }
        self.num_restarts += 1;
    }

    /// Advance the global step counter and, if it lands on a merge boundary, perform a restart.
    ///
    /// Returns `true` iff a merge-and-reset occurred. A restart happens when `step > 0` and
    /// `step` is an exact multiple of `merge_every`. Divisibility is tested via integer division
    /// (`(step / m) * m == step`) to stay compatible with the workspace MSRV (1.85).
    pub fn step(&mut self, step: usize, rng: &mut LcgRng) -> bool {
        let on_boundary = step > 0 && (step / self.merge_every) * self.merge_every == step;
        if on_boundary {
            self.merge_and_reset(rng);
            true
        } else {
            false
        }
    }

    /// Effective maximum rank of the accumulated update after the restarts so far.
    ///
    /// Each restart contributes up to `rank`, plus the current (un-merged) adapter, capped by the
    /// full matrix rank `min(in, out)`.
    #[must_use]
    pub fn effective_rank_bound(&self) -> usize {
        let raw = (self.num_restarts + 1) * self.rank;
        raw.min(self.in_features.min(self.out_features))
    }
}

/// Jagged-cosine learning-rate schedule for ReLoRA.
///
/// Within each `merge_every`-long cycle the multiplier warms up linearly from 0 to 1 over
/// `warmup` steps, then decays following a half-cosine back toward 0 by the end of the cycle.
#[derive(Debug, Clone)]
pub struct ReloraSchedule {
    /// Cycle length in steps (equal to `merge_every`).
    pub cycle_len: usize,
    /// Warmup length in steps within each cycle (`< cycle_len`).
    pub warmup: usize,
}

impl ReloraSchedule {
    /// Construct a schedule.
    ///
    /// # Errors
    /// - [`PeftError::ZeroBlockSize`] if `cycle_len == 0`.
    /// - [`PeftError::Internal`] if `warmup >= cycle_len`.
    pub fn new(cycle_len: usize, warmup: usize) -> PeftResult<Self> {
        if cycle_len == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        if warmup >= cycle_len {
            return Err(PeftError::Internal {
                msg: format!("warmup {warmup} must be < cycle_len {cycle_len}"),
            });
        }
        Ok(Self { cycle_len, warmup })
    }

    /// Learning-rate multiplier in `[0, 1]` for the given global `step`.
    #[must_use]
    pub fn multiplier(&self, step: usize) -> f32 {
        let pos = step % self.cycle_len;
        if pos < self.warmup {
            // Linear warmup: 0 → 1 over `warmup` steps (pos+1 so step 0 is non-zero).
            (pos as f32 + 1.0) / self.warmup.max(1) as f32
        } else {
            // Cosine decay from 1 → 0 across the remainder of the cycle.
            let decay_len = (self.cycle_len - self.warmup) as f32;
            let t = (pos - self.warmup) as f32 / decay_len;
            0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        }
    }
}

/// Multiply matrix `m` (`[rows × cols]`, row-major) by vector `v` (length `cols`).
fn mat_vec(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            let start = i * cols;
            m[start..start + cols]
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(in_f: usize, out_f: usize, r: usize, every: usize) -> ReloraConfig {
        ReloraConfig {
            in_features: in_f,
            out_features: out_f,
            rank: r,
            alpha: 8.0,
            init_scale: 0.05,
            merge_every: every,
        }
    }

    #[test]
    fn new_b_zero_init_delta_zero() {
        let mut rng = LcgRng::new(1);
        let r = ReloraLinear::new(&cfg(8, 8, 4, 100), &mut rng)
            .expect("ReloraLinear::new should succeed with valid ReLoRA config");
        // B=0 → delta zero → with W₀=0 output is zero.
        let x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let out = r
            .forward(&x)
            .expect("ReLoRA forward pass should succeed with matching input dimension");
        for &v in &out {
            assert!(v.abs() < 1e-6, "B=0 → zero output, got {v}");
        }
    }

    #[test]
    fn new_zero_rank_errors() {
        let mut rng = LcgRng::new(2);
        assert!(matches!(
            ReloraLinear::new(&cfg(8, 8, 0, 100), &mut rng),
            Err(PeftError::ZeroBlockSize)
        ));
    }

    #[test]
    fn new_zero_merge_every_errors() {
        let mut rng = LcgRng::new(3);
        assert!(matches!(
            ReloraLinear::new(&cfg(8, 8, 4, 0), &mut rng),
            Err(PeftError::ZeroBlockSize)
        ));
    }

    #[test]
    fn new_rank_too_large_errors() {
        let mut rng = LcgRng::new(4);
        assert!(matches!(
            ReloraLinear::new(&cfg(4, 8, 6, 100), &mut rng),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(5);
        let r = ReloraLinear::new(&cfg(8, 8, 4, 100), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        assert!(matches!(
            r.forward(&[1.0, 2.0]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn merge_and_reset_zeros_b_and_changes_a() {
        let mut rng = LcgRng::new(6);
        let mut r = ReloraLinear::new(&cfg(8, 8, 4, 100), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        // Make B non-zero so the merge actually moves mass into W₀.
        for (i, v) in r.b.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 + 0.1;
        }
        let a_before = r.a.clone();
        let delta = r.lora_delta();
        r.merge_and_reset(&mut rng);
        // B must be all zeros.
        for &v in &r.b {
            assert!(v.abs() < 1e-9, "B must be zeroed after reset, got {v}");
        }
        // A must have been re-randomised.
        assert_ne!(r.a, a_before, "A must change on reset");
        // W₀ must equal the merged delta (was zero before).
        for (w, d) in r.w.iter().zip(delta.iter()) {
            assert!((w - d).abs() < 1e-5, "W₀ must absorb the delta: {w} vs {d}");
        }
        assert_eq!(r.num_restarts, 1);
    }

    #[test]
    fn merge_preserves_function_at_reset_point() {
        // Immediately after merge, the function value (W₀ + scale·B·A)·x must be unchanged,
        // because B becomes zero but W₀ absorbed exactly the old delta.
        let mut rng = LcgRng::new(7);
        let mut r = ReloraLinear::new(&cfg(8, 8, 4, 100), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        for (i, v) in r.b.iter_mut().enumerate() {
            *v = ((i * 3 + 1) % 7) as f32 * 0.05;
        }
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let before = r
            .forward(&x)
            .expect("forward pass should succeed before merge");
        r.merge_and_reset(&mut rng);
        let after = r
            .forward(&x)
            .expect("forward pass should succeed after merge");
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 1e-4,
                "function must be preserved at merge: {b} vs {a}"
            );
        }
    }

    #[test]
    fn step_triggers_on_boundary_only() {
        let mut rng = LcgRng::new(8);
        let mut r = ReloraLinear::new(&cfg(8, 8, 4, 5), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        assert!(!r.step(0, &mut rng), "step 0 must not merge");
        assert!(!r.step(3, &mut rng), "step 3 must not merge");
        assert!(r.step(5, &mut rng), "step 5 must merge (multiple of 5)");
        assert!(!r.step(7, &mut rng), "step 7 must not merge");
        assert!(r.step(10, &mut rng), "step 10 must merge");
        assert_eq!(r.num_restarts, 2);
    }

    #[test]
    fn effective_rank_bound_grows_with_restarts() {
        let mut rng = LcgRng::new(9);
        let mut r = ReloraLinear::new(&cfg(16, 16, 2, 5), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        let r0 = r.effective_rank_bound();
        for (i, v) in r.b.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 + 0.01;
        }
        r.merge_and_reset(&mut rng);
        let r1 = r.effective_rank_bound();
        assert!(r1 > r0, "effective rank bound must grow: {r0} → {r1}");
    }

    #[test]
    fn effective_rank_bound_capped_at_full_rank() {
        let mut rng = LcgRng::new(10);
        let mut r = ReloraLinear::new(&cfg(8, 8, 4, 5), &mut rng)
            .expect("ReloraLinear::new should succeed with valid config");
        for _ in 0..10 {
            r.num_restarts += 1;
        }
        assert!(
            r.effective_rank_bound() <= 8,
            "bound must be capped at min(in,out)=8"
        );
    }

    #[test]
    fn schedule_warmup_linear() {
        let s = ReloraSchedule::new(10, 4)
            .expect("ReloraSchedule::new should succeed with valid cycle and warmup");
        // pos 0..4 warmup: (pos+1)/4
        assert!((s.multiplier(0) - 0.25).abs() < 1e-6);
        assert!((s.multiplier(1) - 0.5).abs() < 1e-6);
        assert!((s.multiplier(3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn schedule_cosine_decay_reaches_low_at_cycle_end() {
        let s = ReloraSchedule::new(10, 4)
            .expect("ReloraSchedule::new should succeed with valid cycle and warmup");
        // At the last step of the cycle (pos=9) the cosine has decayed near 0.
        let last = s.multiplier(9);
        assert!(
            last < 0.1,
            "decay should be near 0 at cycle end, got {last}"
        );
    }

    #[test]
    fn schedule_repeats_each_cycle() {
        let s = ReloraSchedule::new(10, 4)
            .expect("ReloraSchedule::new should succeed with valid cycle and warmup");
        for step in 0..10 {
            let a = s.multiplier(step);
            let b = s.multiplier(step + 10);
            assert!(
                (a - b).abs() < 1e-6,
                "schedule must repeat each cycle at step {step}"
            );
        }
    }

    #[test]
    fn schedule_in_unit_range() {
        let s = ReloraSchedule::new(20, 5)
            .expect("ReloraSchedule::new should succeed with valid cycle and warmup");
        for step in 0..60 {
            let m = s.multiplier(step);
            assert!(
                (0.0..=1.0).contains(&m),
                "multiplier out of range at {step}: {m}"
            );
        }
    }

    #[test]
    fn schedule_zero_cycle_errors() {
        assert!(matches!(
            ReloraSchedule::new(0, 0),
            Err(PeftError::ZeroBlockSize)
        ));
    }

    #[test]
    fn schedule_warmup_too_large_errors() {
        assert!(matches!(
            ReloraSchedule::new(5, 5),
            Err(PeftError::Internal { .. })
        ));
    }
}
