//! Learnable surrogate gradient with a trainable slope parameter `α`.
//!
//! Spiking neural networks replace the non-differentiable Heaviside spike
//! function `s = Θ(v − v_th)` by a smooth surrogate during the backward pass.
//! The sigmoid-family surrogate gradient is
//!
//! ```text
//! s'(v) = α · σ(α(v − v_th)) · (1 − σ(α(v − v_th))),   σ(x) = 1 / (1 + e^{−x}).
//! ```
//!
//! The slope `α > 0` controls the sharpness of the surrogate: large `α`
//! approaches the true (Dirac) derivative, while small `α` spreads the gradient
//! over a wider voltage range. Rather than fixing `α` as a hyper-parameter, this
//! module treats it as a **trainable** parameter that is itself optimised by
//! gradient descent (cf. "Differentiable Spike", Li et al. 2021).
//!
//! To learn `α` we need the partial derivative of the surrogate output with
//! respect to `α`. Writing `u = v − v_th`, `x = α u`, `σ = σ(x)` and
//! `σ' = σ(1 − σ)`, the surrogate is `s' = α σ'`. Using `dσ/dx = σ'` and
//! `dx/dα = u`,
//!
//! ```text
//! d(σ')/dα = σ'(1 − 2σ) · u,
//! ∂s'/∂α   = σ' + α · d(σ')/dα = σ' + α u · σ'(1 − 2σ)
//!          = σ'·[1 + α u (1 − 2σ)].
//! ```
//!
//! Given an upstream loss gradient `dL/ds'` flowing into each element of the
//! surrogate buffer, the gradient with respect to the scalar slope is the chain
//! rule accumulated over all elements:
//!
//! ```text
//! dL/dα = Σ_i (dL/ds'_i) · (∂s'_i/∂α).
//! ```
//!
//! A single SGD step then updates `α ← α − lr · dL/dα`, after which `α` is
//! projected back into the strictly-positive region.

use crate::error::{SnnError, SnnResult};

/// Smallest slope value the projection step is allowed to clamp `α` down to.
///
/// Keeping a strictly-positive floor prevents the surrogate from collapsing to
/// the zero map (which would stop all gradient flow) and avoids division-style
/// numerical issues when `α` is driven toward zero by the optimiser.
pub const ALPHA_FLOOR: f32 = 1e-3;

/// Largest slope value the projection step allows; bounds the peak gradient
/// magnitude `α/4` so a runaway update cannot produce a non-finite surrogate.
pub const ALPHA_CEIL: f32 = 1e4;

/// Numerically stable logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`.
///
/// Uses the two-branch formulation so that large-magnitude inputs of either
/// sign evaluate without overflow: for `x ≥ 0` the denominator stays bounded,
/// and for `x < 0` the numerator `e^x` underflows gracefully to zero.
#[must_use]
#[inline]
pub fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Configuration for a [`LearnableSurrogate`].
///
/// All fields are plain scalars so the config is trivially copyable. Validation
/// happens in [`LearnableSurrogate::new`]; constructing the config directly does
/// not enforce the invariants.
#[derive(Debug, Clone, Copy)]
pub struct LearnableSurrogateConfig {
    /// Initial trainable slope `α` (must be finite and strictly positive).
    pub alpha: f32,
    /// Learning rate used by [`LearnableSurrogate::update_alpha`].
    pub lr: f32,
    /// Lower projection bound for `α` after each update.
    pub alpha_min: f32,
    /// Upper projection bound for `α` after each update.
    pub alpha_max: f32,
}

impl Default for LearnableSurrogateConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            lr: 1e-2,
            alpha_min: ALPHA_FLOOR,
            alpha_max: ALPHA_CEIL,
        }
    }
}

/// Learnable sigmoid-family surrogate gradient with a trainable slope `α`.
///
/// The struct owns the current slope and an optional per-timestep schedule. When
/// a schedule is present, [`LearnableSurrogate::alpha_at`] returns the scheduled
/// value for a given timestep (falling back to the base `α` outside the schedule
/// range); the schedule itself is treated as a fixed annealing curve and is not
/// modified by [`LearnableSurrogate::update_alpha`], which only adjusts the base
/// slope.
#[derive(Debug, Clone)]
pub struct LearnableSurrogate {
    /// Current trainable slope `α` (always kept in `[alpha_min, alpha_max]`).
    alpha: f32,
    /// SGD learning rate for the slope update.
    lr: f32,
    /// Lower projection bound applied after each slope update.
    alpha_min: f32,
    /// Upper projection bound applied after each slope update.
    alpha_max: f32,
    /// Optional per-timestep slope schedule (e.g. an annealing curve).
    schedule: Option<Vec<f32>>,
}

impl LearnableSurrogate {
    /// Construct a learnable surrogate from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::OutOfRange`] if `alpha` is non-finite or not strictly
    /// positive, if `lr` is negative or non-finite, or if the projection bounds
    /// are inconsistent (`alpha_min <= 0`, non-finite, or `alpha_min > alpha_max`).
    pub fn new(cfg: LearnableSurrogateConfig) -> SnnResult<Self> {
        if !cfg.alpha.is_finite() || cfg.alpha <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "alpha".into(),
                val: cfg.alpha,
            });
        }
        if !cfg.lr.is_finite() || cfg.lr < 0.0 {
            return Err(SnnError::OutOfRange {
                name: "lr".into(),
                val: cfg.lr,
            });
        }
        if !cfg.alpha_min.is_finite() || cfg.alpha_min <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "alpha_min".into(),
                val: cfg.alpha_min,
            });
        }
        if !cfg.alpha_max.is_finite() || cfg.alpha_max < cfg.alpha_min {
            return Err(SnnError::OutOfRange {
                name: "alpha_max".into(),
                val: cfg.alpha_max,
            });
        }
        let alpha = cfg.alpha.clamp(cfg.alpha_min, cfg.alpha_max);
        Ok(Self {
            alpha,
            lr: cfg.lr,
            alpha_min: cfg.alpha_min,
            alpha_max: cfg.alpha_max,
            schedule: None,
        })
    }

    /// Attach an optional per-timestep slope schedule.
    ///
    /// Each entry must be finite and strictly positive. The schedule is consulted
    /// by [`LearnableSurrogate::alpha_at`]; it overrides the base slope only for
    /// the timesteps it covers.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `schedule` is empty, or
    /// [`SnnError::OutOfRange`] if any entry is non-finite or non-positive.
    pub fn with_schedule(mut self, schedule: Vec<f32>) -> SnnResult<Self> {
        if schedule.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        for &a in &schedule {
            if !a.is_finite() || a <= 0.0 {
                return Err(SnnError::OutOfRange {
                    name: "schedule_alpha".into(),
                    val: a,
                });
            }
        }
        self.schedule = Some(schedule);
        Ok(self)
    }

    /// Return the current base slope `α`.
    #[must_use]
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Return the SGD learning rate used for slope updates.
    #[must_use]
    #[inline]
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Return the slope used at timestep `t`.
    ///
    /// If a schedule is present and covers `t`, its scheduled value is returned;
    /// otherwise the base trainable slope `α` is used.
    #[must_use]
    pub fn alpha_at(&self, t: usize) -> f32 {
        match &self.schedule {
            Some(sched) if t < sched.len() => sched[t],
            _ => self.alpha,
        }
    }

    /// Evaluate the surrogate gradient `s'(v)` element-wise into a fresh buffer.
    ///
    /// Uses the base slope `α`. See [`LearnableSurrogate::forward_with_alpha`]
    /// to evaluate with an arbitrary slope (e.g. a scheduled one).
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `v` is empty.
    pub fn forward(&self, v: &[f32], v_th: f32) -> SnnResult<Vec<f32>> {
        self.forward_with_alpha(v, v_th, self.alpha)
    }

    /// Evaluate the surrogate gradient `s'(v)` with an explicit slope.
    ///
    /// `s'(v) = α · σ(α(v − v_th)) · (1 − σ(α(v − v_th)))`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `v` is empty, or
    /// [`SnnError::OutOfRange`] if `alpha` is non-finite or non-positive.
    pub fn forward_with_alpha(&self, v: &[f32], v_th: f32, alpha: f32) -> SnnResult<Vec<f32>> {
        if v.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "alpha".into(),
                val: alpha,
            });
        }
        let out = v
            .iter()
            .map(|&vi| {
                let s = stable_sigmoid(alpha * (vi - v_th));
                alpha * s * (1.0 - s)
            })
            .collect();
        Ok(out)
    }

    /// Evaluate the partial derivative of the surrogate output w.r.t. `α`.
    ///
    /// With `u = v − v_th`, `σ = σ(α u)` and `σ' = σ(1 − σ)`:
    ///
    /// ```text
    /// ∂s'/∂α = σ' · [1 + α u (1 − 2σ)].
    /// ```
    ///
    /// Returns one value per element of `v`, evaluated at the base slope `α`.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `v` is empty.
    pub fn grad_wrt_alpha(&self, v: &[f32], v_th: f32) -> SnnResult<Vec<f32>> {
        self.grad_wrt_alpha_with(v, v_th, self.alpha)
    }

    /// Evaluate `∂s'/∂α` with an explicit slope.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if `v` is empty, or
    /// [`SnnError::OutOfRange`] if `alpha` is non-finite or non-positive.
    pub fn grad_wrt_alpha_with(&self, v: &[f32], v_th: f32, alpha: f32) -> SnnResult<Vec<f32>> {
        if v.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(SnnError::OutOfRange {
                name: "alpha".into(),
                val: alpha,
            });
        }
        let out = v
            .iter()
            .map(|&vi| {
                let u = vi - v_th;
                let s = stable_sigmoid(alpha * u);
                let sp = s * (1.0 - s);
                sp * (1.0 + alpha * u * (1.0 - 2.0 * s))
            })
            .collect();
        Ok(out)
    }

    /// Perform one SGD step on `α` given the upstream surrogate gradient.
    ///
    /// The arguments are the membrane potentials `v` (and threshold `v_th`) that
    /// produced the surrogate during the forward pass, and the upstream loss
    /// gradient `dL/ds'` for each surrogate element. The slope gradient is
    ///
    /// ```text
    /// dL/dα = Σ_i (dL/ds'_i) · (∂s'_i/∂α),
    /// ```
    ///
    /// after which `α ← clamp(α − lr · dL/dα, alpha_min, alpha_max)`. Returns the
    /// computed `dL/dα` so callers can monitor convergence.
    ///
    /// # Errors
    ///
    /// Returns [`SnnError::EmptyInput`] if either slice is empty, or
    /// [`SnnError::IncompatibleLength`] if the two slices differ in length.
    pub fn update_alpha(
        &mut self,
        v: &[f32],
        v_th: f32,
        grad_loss_wrt_surrogate: &[f32],
    ) -> SnnResult<f32> {
        if v.is_empty() || grad_loss_wrt_surrogate.is_empty() {
            return Err(SnnError::EmptyInput);
        }
        if v.len() != grad_loss_wrt_surrogate.len() {
            return Err(SnnError::IncompatibleLength {
                a: v.len(),
                b: grad_loss_wrt_surrogate.len(),
            });
        }
        let dsp_dalpha = self.grad_wrt_alpha(v, v_th)?;
        let grad_alpha: f32 = grad_loss_wrt_surrogate
            .iter()
            .zip(dsp_dalpha.iter())
            .map(|(&dl, &da)| dl * da)
            .sum();
        let updated = self.alpha - self.lr * grad_alpha;
        // Project back into the strictly-positive admissible interval. A
        // non-finite update (e.g. from an exploding upstream gradient) is
        // clamped to the floor rather than poisoning the slope.
        self.alpha = if updated.is_finite() {
            updated.clamp(self.alpha_min, self.alpha_max)
        } else {
            self.alpha_min
        };
        Ok(grad_alpha)
    }
}

impl Default for LearnableSurrogate {
    /// A default surrogate is always constructible because the default config is
    /// valid; the fallback floor slope is used only in the impossible event that
    /// validation fails, keeping `Default` infallible.
    fn default() -> Self {
        Self::new(LearnableSurrogateConfig::default()).unwrap_or(Self {
            alpha: ALPHA_FLOOR,
            lr: 1e-2,
            alpha_min: ALPHA_FLOOR,
            alpha_max: ALPHA_CEIL,
            schedule: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(alpha: f32, lr: f32) -> LearnableSurrogate {
        LearnableSurrogate::new(LearnableSurrogateConfig {
            alpha,
            lr,
            ..LearnableSurrogateConfig::default()
        })
        .expect("valid config")
    }

    #[test]
    fn peak_at_threshold_equals_alpha_over_four() {
        let s = make(2.0, 1e-2);
        let v = vec![0.5_f32];
        let g = s.forward(&v, 0.5).expect("ok");
        assert!((g[0] - 2.0 / 4.0).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn forward_symmetric_about_threshold() {
        let s = make(1.5, 1e-2);
        let v = vec![-0.7_f32, 0.7];
        let g = s.forward(&v, 0.0).expect("ok");
        assert!((g[0] - g[1]).abs() < 1e-6, "{} vs {}", g[0], g[1]);
    }

    #[test]
    fn forward_finite_at_extremes() {
        let s = make(1.0, 1e-2);
        let v = vec![-1e6_f32, 1e6];
        let g = s.forward(&v, 0.0).expect("ok");
        for &gi in &g {
            assert!(gi.is_finite() && gi >= 0.0, "g={gi}");
        }
    }

    #[test]
    fn grad_wrt_alpha_matches_finite_difference() {
        // Compare the analytic ∂s'/∂α to a central finite difference.
        let s = make(2.3, 1e-2);
        let v_th = 0.1_f32;
        let v = vec![-0.4_f32, 0.0, 0.25, 0.9];
        let analytic = s.grad_wrt_alpha(&v, v_th).expect("ok");
        let eps = 1e-3_f32;
        let plus = s.forward_with_alpha(&v, v_th, s.alpha() + eps).expect("ok");
        let minus = s.forward_with_alpha(&v, v_th, s.alpha() - eps).expect("ok");
        for ((&a, &p), &m) in analytic.iter().zip(plus.iter()).zip(minus.iter()) {
            let fd = (p - m) / (2.0 * eps);
            assert!((a - fd).abs() < 1e-2, "analytic={a} fd={fd}");
        }
    }

    #[test]
    fn grad_wrt_alpha_zero_at_threshold() {
        // At v = v_th, u = 0, σ = 1/2, σ' = 1/4, so ∂s'/∂α = σ'·[1 + 0] = 1/4.
        let s = make(3.0, 1e-2);
        let g = s.grad_wrt_alpha(&[0.2_f32], 0.2).expect("ok");
        assert!((g[0] - 0.25).abs() < 1e-6, "g={}", g[0]);
    }

    #[test]
    fn update_alpha_reduces_surrogate_when_target_smaller() {
        // Drive α so the surrogate output at threshold (= α/4) approaches a
        // target peak of 0.25 (i.e. α → 1). Start above and watch α decrease.
        let mut s = make(4.0, 0.5);
        let v_th = 0.0_f32;
        let v = vec![0.0_f32]; // exactly at threshold → s' = α/4
        let target_peak = 0.25_f32;
        let mut last = s.alpha();
        for _ in 0..200 {
            let surrogate = s.forward(&v, v_th).expect("ok");
            // L = ½ (s' − target)², dL/ds' = (s' − target).
            let dl: Vec<f32> = surrogate.iter().map(|&sp| sp - target_peak).collect();
            s.update_alpha(&v, v_th, &dl).expect("ok");
            assert!(s.alpha() > 0.0, "alpha must stay positive");
            last = s.alpha();
        }
        // Peak α/4 should have converged to ~0.25 → α ≈ 1.0.
        assert!((last - 1.0).abs() < 0.1, "alpha converged to {last}");
    }

    #[test]
    fn update_alpha_is_deterministic() {
        let v = vec![-0.3_f32, 0.1, 0.5];
        let v_th = 0.0_f32;
        let mut a = make(2.0, 0.1);
        let mut b = make(2.0, 0.1);
        for _ in 0..50 {
            let ga = {
                let sg = a.forward(&v, v_th).expect("ok");
                let dl: Vec<f32> = sg.iter().map(|&x| x - 0.1).collect();
                a.update_alpha(&v, v_th, &dl).expect("ok")
            };
            let gb = {
                let sg = b.forward(&v, v_th).expect("ok");
                let dl: Vec<f32> = sg.iter().map(|&x| x - 0.1).collect();
                b.update_alpha(&v, v_th, &dl).expect("ok")
            };
            assert_eq!(ga.to_bits(), gb.to_bits());
            assert_eq!(a.alpha().to_bits(), b.alpha().to_bits());
        }
    }

    #[test]
    fn alpha_stays_positive_under_huge_gradient() {
        let mut s = make(2.0, 1e9);
        let v = vec![0.0_f32];
        // Enormous positive dL/ds' would push α far negative; projection holds.
        let dl = vec![1e9_f32];
        s.update_alpha(&v, 0.0, &dl).expect("ok");
        assert!(s.alpha() >= ALPHA_FLOOR, "alpha={}", s.alpha());
    }

    #[test]
    fn schedule_overrides_base_alpha() {
        let s = make(2.0, 1e-2)
            .with_schedule(vec![5.0_f32, 4.0, 3.0])
            .expect("valid schedule");
        assert_eq!(s.alpha_at(0), 5.0);
        assert_eq!(s.alpha_at(2), 3.0);
        // Outside the schedule range falls back to the base slope.
        assert_eq!(s.alpha_at(7), 2.0);
    }

    #[test]
    fn rejects_bad_alpha() {
        let err = LearnableSurrogate::new(LearnableSurrogateConfig {
            alpha: -1.0,
            ..LearnableSurrogateConfig::default()
        });
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_inconsistent_bounds() {
        let err = LearnableSurrogate::new(LearnableSurrogateConfig {
            alpha: 2.0,
            alpha_min: 5.0,
            alpha_max: 1.0,
            ..LearnableSurrogateConfig::default()
        });
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }

    #[test]
    fn rejects_empty_input() {
        let s = make(2.0, 1e-2);
        assert!(matches!(s.forward(&[], 0.0), Err(SnnError::EmptyInput)));
        assert!(matches!(
            s.grad_wrt_alpha(&[], 0.0),
            Err(SnnError::EmptyInput)
        ));
    }

    #[test]
    fn update_rejects_length_mismatch() {
        let mut s = make(2.0, 1e-2);
        let v = vec![0.0_f32; 3];
        let dl = vec![0.0_f32; 2];
        assert!(matches!(
            s.update_alpha(&v, 0.0, &dl),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    #[test]
    fn default_is_valid() {
        let s = LearnableSurrogate::default();
        assert!(s.alpha() > 0.0);
    }
}
