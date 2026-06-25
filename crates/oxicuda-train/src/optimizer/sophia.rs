//! Sophia optimizer — Liu, Li, Hall, Liang & Ma, 2023.
//!
//! "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
//! Pre-training" (arXiv:2305.14342).
//!
//! Sophia is a light-weight diagonal second-order method.  It maintains an
//! exponential moving average (EMA) of the gradient `m` and an EMA of a
//! *diagonal Hessian estimate* `h`, then takes a **per-coordinate clipped**
//! pre-conditioned step.  The clipping bounds the worst-case update so that
//! inaccurate curvature estimates and non-convex (negative-curvature)
//! directions cannot blow up training:
//!
//! ```text
//! t   ← t + 1
//! m_t ← β₁·m_{t-1} + (1−β₁)·g_t                       // gradient EMA
//!
//! if t mod k == 0:                                     // refresh curvature
//!     h_t ← β₂·h_{t-1} + (1−β₂)·ĥ_t                    // Hessian-diag EMA
//! else:
//!     h_t ← h_{t-1}                                    // hold between updates
//!
//! θ_t ← θ_{t-1} − lr · clip( m_t / max(γ·h_t, ε), ρ )  // element-wise
//! ```
//!
//! where `clip(z, ρ) = max(−ρ, min(ρ, z))`.  A *decoupled* (AdamW-style)
//! weight-decay term `−lr·λ·θ` is applied every step before the curvature
//! step.  Because the pre-conditioned ratio is clipped to `[−ρ, ρ]`, the per
//! coordinate parameter change never exceeds `lr·ρ` in magnitude (plus the
//! decoupled decay), which is the property that makes Sophia robust.
//!
//! ## Diagonal Hessian estimators
//!
//! Sophia is agnostic to *how* `ĥ_t` is produced; the paper proposes two
//! unbiased / positive-biased diagonal estimators:
//!
//! * **Sophia-H (Hutchinson)** — `ĥ = u ⊙ (H·u)` with a Rademacher vector
//!   `u ∈ {−1, +1}^d`.  `E[u ⊙ (H u)] = diag(H)`.  Use
//!   [`Sophia::step_hutchinson`], which draws `u` from the crate RNG and asks
//!   the caller for the Hessian-vector product `H·u` via a closure.
//!
//! * **Sophia-G (Gauss-Newton-Bartlett)** — for a cross-entropy loss the
//!   Gauss-Newton matrix is `Jᵀ diag(p) − p pᵀ ... ` ; Bartlett's identity lets
//!   one form an unbiased diagonal estimate from sampled labels.  The estimate
//!   is model-specific, so it is supplied by the caller and forwarded through
//!   [`Sophia::step_gauss_newton`] / the generic [`Sophia::step`].
//!
//! All host-side state (`m`, `h`) is stored as flat `f64` slices for the high
//! numerical accuracy a second-order method benefits from.

use crate::error::{TrainError, TrainResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Sophia`] optimizer.
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Learning rate `lr` (must be > 0).
    pub lr: f64,
    /// Gradient EMA decay `β₁` (default 0.965), in `[0, 1)`.
    pub beta1: f64,
    /// Hessian-diagonal EMA decay `β₂` (default 0.99), in `[0, 1)`.
    pub beta2: f64,
    /// Per-coordinate clipping threshold `ρ` (must be > 0; default 0.04).
    pub rho: f64,
    /// Denominator floor `ε` for numerical stability (must be > 0).
    pub eps: f64,
    /// Decoupled (AdamW-style) weight-decay coefficient `λ` (default 0; ≥ 0).
    pub weight_decay: f64,
    /// Refresh the Hessian-diagonal EMA every `k` steps (must be ≥ 1; default 10).
    pub hessian_update_interval: usize,
    /// Curvature scaling `γ` multiplying `h` in the denominator (must be > 0;
    /// default 1.0).  Larger `γ` shrinks steps in high-curvature directions.
    pub gamma: f64,
}

impl Default for SophiaConfig {
    /// Defaults follow the language-model pre-training recipe in the paper
    /// (`β₁ = 0.965`, `β₂ = 0.99`, `ρ = 0.04`, `k = 10`).
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.965,
            beta2: 0.99,
            rho: 0.04,
            eps: 1e-12,
            weight_decay: 0.0,
            hessian_update_interval: 10,
            gamma: 1.0,
        }
    }
}

impl SophiaConfig {
    /// Validate every field, returning the crate error type on violation.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if `lr <= 0`.
    /// * [`TrainError::Internal`] for any other out-of-range field (`eps`,
    ///   `rho`, `gamma`, betas, `weight_decay`, `hessian_update_interval`).
    fn validate(&self) -> TrainResult<()> {
        // `<= 0.0 || is_nan()` rejects non-positive *and* NaN values without a
        // negated partial-ord comparison (clippy::neg_cmp_op_on_partial_ord).
        if self.lr <= 0.0 || self.lr.is_nan() {
            return Err(TrainError::InvalidLearningRate { lr: self.lr });
        }
        if self.eps <= 0.0 || self.eps.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("eps must be positive, got {}", self.eps),
            });
        }
        if self.rho <= 0.0 || self.rho.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("rho must be positive, got {}", self.rho),
            });
        }
        if self.gamma <= 0.0 || self.gamma.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("gamma must be positive, got {}", self.gamma),
            });
        }
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            return Err(TrainError::Internal {
                msg: format!(
                    "beta1/beta2 must be in [0, 1), got {} / {}",
                    self.beta1, self.beta2
                ),
            });
        }
        if self.weight_decay < 0.0 {
            return Err(TrainError::Internal {
                msg: format!("weight_decay must be >= 0, got {}", self.weight_decay),
            });
        }
        if self.hessian_update_interval == 0 {
            return Err(TrainError::Internal {
                msg: "hessian_update_interval must be >= 1".to_string(),
            });
        }
        Ok(())
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Sophia diagonal second-order optimizer.
///
/// Holds the gradient EMA `m` and Hessian-diagonal EMA `h` as host-side `f64`
/// buffers and operates on flat `f64` parameter / gradient slices.
///
/// # Example
///
/// ```rust
/// use oxicuda_train::optimizer::{Sophia, SophiaConfig};
///
/// // Minimise f(x) = ½·xᵀ A x with A = diag(2, 4); ∇f = A x, diag(H) = (2, 4).
/// let cfg = SophiaConfig {
///     lr: 0.5,
///     beta1: 0.9,
///     hessian_update_interval: 1,
///     ..Default::default()
/// };
/// let mut opt = Sophia::new(2, cfg).expect("config is valid");
/// let mut theta = vec![1.0_f64, 1.0];
/// for _ in 0..500 {
///     let grad = vec![2.0 * theta[0], 4.0 * theta[1]];
///     let hess = vec![2.0_f64, 4.0];
///     opt.step(&mut theta, &grad, Some(&hess)).expect("step succeeds");
/// }
/// assert!(theta[0].abs() < 1e-2 && theta[1].abs() < 1e-2);
/// ```
pub struct Sophia {
    /// Gradient EMA `m_t`.
    m: Vec<f64>,
    /// Hessian-diagonal EMA `h_t` (held between refreshes).
    h: Vec<f64>,
    /// Number of completed steps `t`.
    t: usize,
    /// Deterministic RNG for Hutchinson Rademacher draws.
    rng: LcgRng,
    /// Validated configuration.
    config: SophiaConfig,
}

impl Sophia {
    /// Default seed for the internal Rademacher RNG.
    const DEFAULT_SEED: u64 = 0x5350_4849_4100_0001; // "SOPHIA\0\x01"

    /// Create a new `Sophia` optimizer for `n_params` parameters.
    ///
    /// # Errors
    ///
    /// Returns the crate error type if any [`SophiaConfig`] field is
    /// out of range (see `SophiaConfig::validate`), or
    /// [`TrainError::EmptyParams`] if `n_params == 0`.
    pub fn new(n_params: usize, config: SophiaConfig) -> TrainResult<Self> {
        config.validate()?;
        if n_params == 0 {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self {
            m: vec![0.0; n_params],
            h: vec![0.0; n_params],
            t: 0,
            rng: LcgRng::new(Self::DEFAULT_SEED),
            config,
        })
    }

    /// Create a new `Sophia` optimizer with an explicit RNG seed for the
    /// Hutchinson Rademacher draws (useful for reproducible experiments).
    ///
    /// # Errors
    ///
    /// Same as [`Sophia::new`].
    pub fn with_seed(n_params: usize, config: SophiaConfig, seed: u64) -> TrainResult<Self> {
        let mut opt = Self::new(n_params, config)?;
        opt.rng = LcgRng::new(seed);
        Ok(opt)
    }

    /// Number of parameters this optimizer was constructed for.
    #[must_use]
    pub fn len(&self) -> usize {
        self.m.len()
    }

    /// Always `false` — `new` rejects a zero-length parameter set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }

    /// Number of completed optimizer steps `t`.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.t
    }

    /// Reference to the gradient-EMA buffer `m`.
    #[must_use]
    pub fn m(&self) -> &[f64] {
        &self.m
    }

    /// Reference to the Hessian-diagonal-EMA buffer `h`.
    #[must_use]
    pub fn h(&self) -> &[f64] {
        &self.h
    }

    /// Read-only view of the configuration.
    #[must_use]
    pub fn config(&self) -> &SophiaConfig {
        &self.config
    }

    /// Whether the *next* step (i.e. step `t + 1`) will refresh the Hessian
    /// EMA, given the configured `hessian_update_interval`.
    #[must_use]
    pub fn next_step_refreshes_hessian(&self) -> bool {
        (self.t + 1) % self.config.hessian_update_interval == 0
    }

    /// Reset all optimizer state (`m`, `h`, step counter) to zero.  The RNG
    /// stream is **not** reset; use [`Sophia::reseed`] for that.
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.h.fill(0.0);
        self.t = 0;
    }

    /// Reseed the internal Rademacher RNG.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = LcgRng::new(seed);
    }

    /// Perform one Sophia step, updating `params` in-place.
    ///
    /// * `grads` — the mini-batch gradient `g_t`.
    /// * `hessian_diag` — the diagonal Hessian estimate `ĥ_t`.  On a step that
    ///   refreshes the curvature EMA (every `hessian_update_interval` steps)
    ///   this **must** be `Some(_)`; on the in-between steps it is ignored and
    ///   the stored `h` is reused, so passing `None` is fine there.  Passing
    ///   `Some(_)` on a non-refresh step is allowed but the value is ignored.
    ///
    /// The estimate may come from either the Sophia-H or Sophia-G estimator;
    /// [`Sophia::step_hutchinson`] is a convenience that builds the Sophia-H
    /// estimate internally.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if `params`, `grads`, or (when
    ///   supplied) `hessian_diag` lengths disagree with each other or with the
    ///   configured parameter count.
    /// * [`TrainError::StateNotInitialised`] if a curvature refresh is due but
    ///   `hessian_diag` is `None`.
    pub fn step(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        hessian_diag: Option<&[f64]>,
    ) -> TrainResult<()> {
        self.check_len(params.len(), grads.len())?;
        if let Some(hd) = hessian_diag {
            if hd.len() != self.m.len() {
                return Err(TrainError::ParamCountMismatch {
                    expected: self.m.len(),
                    got: hd.len(),
                });
            }
        }

        self.t += 1;
        let refresh = self.t % self.config.hessian_update_interval == 0;
        if refresh {
            match hessian_diag {
                Some(hd) => self.update_hessian(hd),
                None => {
                    // Roll back the step counter so a retry behaves correctly.
                    self.t -= 1;
                    return Err(TrainError::StateNotInitialised);
                }
            }
        }

        self.apply(params, grads);
        Ok(())
    }

    /// Perform one **Sophia-H** step.  A fresh Rademacher vector `u` is drawn
    /// from the internal RNG; the caller-supplied `hvp` closure must write the
    /// Hessian-vector product `H·u` into its second argument given `u` in the
    /// first.  The diagonal estimate `ĥ = u ⊙ (H u)` is then formed and the EMA
    /// refreshed before the parameter update.
    ///
    /// Unlike [`Sophia::step`], the Hessian estimate is built **every** call
    /// (the closure cost is the dominant term), but the EMA is still only
    /// folded in on the configured interval — on the in-between steps the
    /// closure is *not* invoked and the held `h` is reused.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if `params`/`grads` lengths
    ///   disagree, or if `hvp` writes a slice of the wrong length.
    pub fn step_hutchinson<F>(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        mut hvp: F,
    ) -> TrainResult<()>
    where
        F: FnMut(&[f64], &mut [f64]) -> TrainResult<()>,
    {
        self.check_len(params.len(), grads.len())?;

        self.t += 1;
        let refresh = self.t % self.config.hessian_update_interval == 0;
        if refresh {
            let n = self.m.len();
            let mut u = vec![0.0_f64; n];
            self.rng.fill_rademacher(&mut u);
            let mut hu = vec![0.0_f64; n];
            hvp(&u, &mut hu)?;
            if hu.len() != n {
                // Defensive: a well-behaved closure preserves the length, but
                // `Vec` reallocation inside the closure could change it.
                self.t -= 1;
                return Err(TrainError::ParamCountMismatch {
                    expected: n,
                    got: hu.len(),
                });
            }
            // ĥ = u ⊙ (H u); since u_i ∈ {−1, +1}, u_i·(Hu)_i = sign-folded HVP.
            for i in 0..n {
                let h_hat = u[i] * hu[i];
                self.h[i] = self.config.beta2 * self.h[i] + (1.0 - self.config.beta2) * h_hat;
            }
        }

        self.apply(params, grads);
        Ok(())
    }

    /// Perform one **Sophia-G** (Gauss-Newton-Bartlett) step.  The GNB diagonal
    /// estimate is model-specific and supplied by the caller; this is a thin,
    /// explicitly-named wrapper over [`Sophia::step`] that documents intent.
    ///
    /// On a curvature-refresh step `gnb_diag` must be `Some(_)`.
    ///
    /// # Errors
    ///
    /// Same as [`Sophia::step`].
    pub fn step_gauss_newton(
        &mut self,
        params: &mut [f64],
        grads: &[f64],
        gnb_diag: Option<&[f64]>,
    ) -> TrainResult<()> {
        self.step(params, grads, gnb_diag)
    }

    // ─── Internals ─────────────────────────────────────────────────────────

    /// Validate that `params`/`grads` lengths agree with each other and with
    /// the configured parameter count.
    fn check_len(&self, params_len: usize, grads_len: usize) -> TrainResult<()> {
        if params_len != grads_len {
            return Err(TrainError::ParamCountMismatch {
                expected: params_len,
                got: grads_len,
            });
        }
        if params_len != self.m.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.m.len(),
                got: params_len,
            });
        }
        Ok(())
    }

    /// Fold a caller-provided diagonal estimate into the Hessian EMA `h`.
    fn update_hessian(&mut self, hessian_diag: &[f64]) {
        let b2 = self.config.beta2;
        for (h, &h_hat) in self.h.iter_mut().zip(hessian_diag.iter()) {
            *h = b2 * *h + (1.0 - b2) * h_hat;
        }
    }

    /// Update `m` and apply the clipped, decoupled-weight-decayed step to
    /// `params`.  Assumes lengths have already been validated.
    fn apply(&mut self, params: &mut [f64], grads: &[f64]) {
        let b1 = self.config.beta1;
        let lr = self.config.lr;
        let rho = self.config.rho;
        let eps = self.config.eps;
        let gamma = self.config.gamma;
        let wd = self.config.weight_decay;

        for i in 0..params.len() {
            // Gradient EMA.
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];

            // Decoupled (AdamW-style) weight decay.
            if wd != 0.0 {
                params[i] -= lr * wd * params[i];
            }

            // Pre-conditioned ratio with a strictly-positive floored denominator.
            // max(γ·h, ε) ≥ ε > 0, so this never divides by zero and tames
            // negative-curvature directions (their tiny/negative h floors to ε).
            let denom = (gamma * self.h[i]).max(eps);
            let ratio = self.m[i] / denom;

            // Per-coordinate clip to [−ρ, ρ]; |update| ≤ lr·ρ by construction.
            let clipped = clip(ratio, rho);
            params[i] -= lr * clipped;
        }
    }
}

/// Symmetric clip of `z` into `[−rho, rho]`.  `rho` is assumed positive
/// (enforced by [`SophiaConfig::validate`]).
#[inline]
fn clip(z: f64, rho: f64) -> f64 {
    if z > rho {
        rho
    } else if z < -rho {
        -rho
    } else {
        z
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SophiaConfig {
        SophiaConfig {
            lr: 0.5,
            beta1: 0.9,
            beta2: 0.99,
            rho: 0.04,
            eps: 1e-12,
            weight_decay: 0.0,
            hessian_update_interval: 1,
            gamma: 1.0,
        }
    }

    /// Converges on a convex quadratic f(x)=½xᵀAx with the exact diagonal
    /// Hessian: loss decreases monotonically (overall) and θ → minimiser 0.
    #[test]
    fn converges_quadratic_exact_hessian() {
        // A = diag(2, 4, 6); ∇f = A·x; diag(H) = (2, 4, 6); minimiser = 0.
        let diag_a = [2.0_f64, 4.0, 6.0];
        let mut opt = Sophia::new(3, cfg()).expect("valid config");
        let mut theta = vec![3.0_f64, -2.0, 1.5];

        let loss = |x: &[f64]| 0.5 * (0..x.len()).map(|i| diag_a[i] * x[i] * x[i]).sum::<f64>();
        let initial = loss(&theta);

        for _ in 0..400 {
            let grad: Vec<f64> = (0..3).map(|i| diag_a[i] * theta[i]).collect();
            let hess: Vec<f64> = diag_a.to_vec();
            opt.step(&mut theta, &grad, Some(&hess)).expect("step ok");
        }

        let final_loss = loss(&theta);
        assert!(
            final_loss < initial * 1e-4,
            "loss should shrink dramatically: {initial} -> {final_loss}"
        );
        for (i, &v) in theta.iter().enumerate() {
            assert!(v.abs() < 1e-2, "theta[{i}] = {v} should approach 0");
        }
    }

    /// On the convex quadratic the loss decreases monotonically while it is
    /// still meaningfully above the noise floor.  (An EMA-momentum optimizer
    /// may jitter at the very bottom once the loss is ~machine-precision; the
    /// guard below stops checking once we are already essentially at the
    /// minimum, which is the physically meaningful claim.)
    #[test]
    fn loss_decreases_while_above_floor() {
        let diag_a = [2.0_f64, 5.0];
        let mut opt = Sophia::new(2, cfg()).expect("valid");
        let mut theta = vec![1.0_f64, 1.0];
        let loss = |x: &[f64]| 0.5 * (diag_a[0] * x[0] * x[0] + diag_a[1] * x[1] * x[1]);

        let mut prev = loss(&theta);
        for _ in 0..100 {
            if prev < 1e-8 {
                break; // already at the minimum; stop the monotonicity check
            }
            let grad = vec![diag_a[0] * theta[0], diag_a[1] * theta[1]];
            opt.step(&mut theta, &grad, Some(&diag_a)).expect("step ok");
            let now = loss(&theta);
            assert!(
                now <= prev + 1e-12,
                "loss must not increase above the floor: {prev} -> {now}"
            );
            prev = now;
        }
        assert!(prev < 1e-8, "loss should reach the minimum, got {prev}");
    }

    /// The per-coordinate update magnitude never exceeds lr·ρ (clipping bound)
    /// even with huge gradients and tiny curvature.
    #[test]
    fn update_bounded_by_lr_rho() {
        let mut config = cfg();
        config.weight_decay = 0.0; // isolate the curvature step
        let lr = config.lr;
        let rho = config.rho;
        let bound = lr * rho;

        let mut opt = Sophia::new(4, config).expect("valid");
        let mut theta = vec![0.0_f64; 4];
        // Enormous gradients, near-zero Hessian → ratio saturates the clip.
        let grads = vec![1e9_f64, -1e9, 5e8, -7e7];
        let hess = vec![1e-9_f64; 4];

        for _ in 0..50 {
            let before = theta.clone();
            opt.step(&mut theta, &grads, Some(&hess)).expect("step ok");
            for i in 0..4 {
                let delta = (theta[i] - before[i]).abs();
                assert!(
                    delta <= bound + 1e-9,
                    "|Δθ[{i}]| = {delta} exceeded lr·ρ = {bound}"
                );
            }
        }
    }

    /// The Hessian EMA tracks a constant diagonal: h → the constant value.
    #[test]
    fn hessian_ema_tracks_constant() {
        let mut config = cfg();
        config.beta2 = 0.9;
        config.hessian_update_interval = 1;
        let mut opt = Sophia::new(2, config).expect("valid");
        let mut theta = vec![1.0_f64, 1.0];
        let target = [3.0_f64, 7.0];

        for _ in 0..300 {
            let grad = vec![0.01 * theta[0], 0.01 * theta[1]];
            opt.step(&mut theta, &grad, Some(&target)).expect("step ok");
        }
        for (i, &h) in opt.h().iter().enumerate() {
            assert!(
                (h - target[i]).abs() < 1e-3,
                "h[{i}] = {h} should track constant {}",
                target[i]
            );
        }
    }

    /// Weight decay with zero gradient shrinks parameters toward 0.
    #[test]
    fn weight_decay_shrinks_params() {
        let mut config = cfg();
        config.weight_decay = 0.1;
        config.lr = 0.1;
        let mut opt = Sophia::new(3, config).expect("valid");
        let mut theta = vec![1.0_f64, -2.0, 3.0];
        let grads = vec![0.0_f64; 3];
        let hess = vec![1.0_f64; 3];

        let before = theta.clone();
        for _ in 0..10 {
            opt.step(&mut theta, &grads, Some(&hess)).expect("step ok");
        }
        for i in 0..3 {
            assert!(
                theta[i].abs() < before[i].abs(),
                "weight decay should shrink |theta[{i}]|: {} -> {}",
                before[i].abs(),
                theta[i].abs()
            );
            // Pure decoupled decay only scales toward 0, so the sign of each
            // coordinate is preserved (no curvature step with zero gradient).
            assert!(
                theta[i].signum() == before[i].signum(),
                "weight decay must preserve the sign of theta[{i}]"
            );
        }
    }

    /// `hessian_update_interval` actually gates the EMA refresh: with k=5 the
    /// Hessian only changes on the 5th, 10th, ... step.
    #[test]
    fn hessian_update_interval_gates() {
        let mut config = cfg();
        config.hessian_update_interval = 5;
        config.beta2 = 0.5; // make refreshes visible quickly
        let mut opt = Sophia::new(1, config).expect("valid");
        let mut theta = vec![1.0_f64];
        let grad = vec![0.001_f64];
        let hess = [10.0_f64];

        for step in 1..=12 {
            let h_before = opt.h()[0];
            // On non-refresh steps pass None to prove it is ignored/reused.
            let provide = step % 5 == 0;
            let hd = if provide { Some(&hess[..]) } else { None };
            opt.step(&mut theta, &grad, hd).expect("step ok");
            let h_after = opt.h()[0];
            if provide {
                assert!(
                    (h_after - h_before).abs() > 0.0,
                    "h should change on refresh step {step}"
                );
            } else {
                assert_eq!(
                    h_after, h_before,
                    "h must be held on non-refresh step {step}"
                );
            }
        }
        assert_eq!(opt.step_count(), 12);
    }

    /// A curvature refresh with `None` Hessian is an error, and the step
    /// counter is rolled back so a retry is well-defined.
    #[test]
    fn refresh_without_hessian_errors() {
        let mut config = cfg();
        config.hessian_update_interval = 1; // every step refreshes
        let mut opt = Sophia::new(2, config).expect("valid");
        let mut theta = vec![1.0_f64, 1.0];
        let grad = vec![0.1_f64, 0.1];
        let err = opt.step(&mut theta, &grad, None);
        assert!(matches!(err, Err(TrainError::StateNotInitialised)));
        assert_eq!(opt.step_count(), 0, "failed step must not advance counter");
        // Retry with a Hessian succeeds.
        let hess = vec![1.0_f64, 1.0];
        opt.step(&mut theta, &grad, Some(&hess)).expect("retry ok");
        assert_eq!(opt.step_count(), 1);
    }

    /// Sophia-H Hutchinson path: with an exact diagonal Hessian the HVP
    /// `H·u = diag ⊙ u`, so `u ⊙ (H u) = diag` exactly, and the optimizer
    /// converges on the quadratic.
    #[test]
    fn hutchinson_converges() {
        let diag_a = [2.0_f64, 4.0];
        let mut opt = Sophia::with_seed(2, cfg(), 2024).expect("valid");
        let mut theta = vec![2.0_f64, -1.0];
        // Enough steps that the un-bias-corrected EMA `h` has essentially
        // saturated to the constant diagonal (0.99^1600 ≈ 1e-7).
        for _ in 0..1600 {
            let grad = vec![diag_a[0] * theta[0], diag_a[1] * theta[1]];
            opt.step_hutchinson(&mut theta, &grad, |u, hu| {
                // Diagonal Hessian: (H u)_i = diag_i · u_i.
                for i in 0..u.len() {
                    hu[i] = diag_a[i] * u[i];
                }
                Ok(())
            })
            .expect("step ok");
        }
        for (i, &v) in theta.iter().enumerate() {
            assert!(v.abs() < 1e-2, "theta[{i}] = {v} should approach 0");
        }
        // The recovered Hessian EMA matches the true diagonal: u ⊙ (H u) = diag
        // exactly for *any* Rademacher u when H is diagonal (u_i² = 1), so the
        // estimate is noise-free and the saturated EMA equals the diagonal.
        for (i, &h) in opt.h().iter().enumerate() {
            assert!(
                (h - diag_a[i]).abs() < 1e-5,
                "h[{i}] = {h} should equal diag {}",
                diag_a[i]
            );
        }
    }

    /// Sophia-G wrapper forwards to `step` identically.
    #[test]
    fn gauss_newton_matches_step() {
        let diag_a = [2.0_f64, 3.0];
        let mut a = Sophia::new(2, cfg()).expect("valid");
        let mut b = Sophia::new(2, cfg()).expect("valid");
        let mut ta = vec![1.0_f64, 1.0];
        let mut tb = vec![1.0_f64, 1.0];
        for _ in 0..20 {
            let ga = vec![diag_a[0] * ta[0], diag_a[1] * ta[1]];
            let gb = vec![diag_a[0] * tb[0], diag_a[1] * tb[1]];
            a.step(&mut ta, &ga, Some(&diag_a)).expect("ok");
            b.step_gauss_newton(&mut tb, &gb, Some(&diag_a))
                .expect("ok");
        }
        for i in 0..2 {
            assert!((ta[i] - tb[i]).abs() < 1e-12, "paths must agree at {i}");
        }
    }

    /// Determinism: identical seed + inputs ⇒ identical trajectories.
    #[test]
    fn deterministic() {
        let run = || {
            let mut opt = Sophia::with_seed(3, cfg(), 99).expect("valid");
            let mut theta = vec![1.0_f64, 2.0, 3.0];
            for _ in 0..30 {
                let grad = vec![2.0 * theta[0], 4.0 * theta[1], 6.0 * theta[2]];
                opt.step_hutchinson(&mut theta, &grad, |u, hu| {
                    hu[0] = 2.0 * u[0];
                    hu[1] = 4.0 * u[1];
                    hu[2] = 6.0 * u[2];
                    Ok(())
                })
                .expect("ok");
            }
            (theta, opt.m().to_vec(), opt.h().to_vec())
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "same seed and inputs must be bit-identical");
    }

    /// Reset clears state but not parameters.
    #[test]
    fn reset_clears_state() {
        let mut opt = Sophia::new(2, cfg()).expect("valid");
        let mut theta = vec![1.0_f64, 1.0];
        opt.step(&mut theta, &[0.5, 0.5], Some(&[1.0, 1.0]))
            .expect("ok");
        assert_eq!(opt.step_count(), 1);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        assert!(opt.m().iter().all(|&v| v == 0.0));
        assert!(opt.h().iter().all(|&v| v == 0.0));
    }

    // ── Error paths ──────────────────────────────────────────────────────

    #[test]
    fn zero_params_errors() {
        assert!(matches!(
            Sophia::new(0, cfg()),
            Err(TrainError::EmptyParams)
        ));
    }

    #[test]
    fn bad_lr_errors() {
        let bad = SophiaConfig { lr: 0.0, ..cfg() };
        assert!(matches!(
            Sophia::new(4, bad),
            Err(TrainError::InvalidLearningRate { .. })
        ));
        let neg = SophiaConfig { lr: -1.0, ..cfg() };
        assert!(matches!(
            Sophia::new(4, neg),
            Err(TrainError::InvalidLearningRate { .. })
        ));
    }

    #[test]
    fn bad_beta_errors() {
        let b1 = SophiaConfig {
            beta1: 1.0,
            ..cfg()
        };
        assert!(matches!(
            Sophia::new(4, b1),
            Err(TrainError::Internal { .. })
        ));
        let b2 = SophiaConfig {
            beta2: -0.1,
            ..cfg()
        };
        assert!(matches!(
            Sophia::new(4, b2),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn bad_rho_gamma_eps_interval_errors() {
        for bad in [
            SophiaConfig { rho: 0.0, ..cfg() },
            SophiaConfig { rho: -1.0, ..cfg() },
            SophiaConfig {
                gamma: 0.0,
                ..cfg()
            },
            SophiaConfig { eps: 0.0, ..cfg() },
            SophiaConfig {
                eps: -1e-9,
                ..cfg()
            },
            SophiaConfig {
                weight_decay: -0.5,
                ..cfg()
            },
            SophiaConfig {
                hessian_update_interval: 0,
                ..cfg()
            },
        ] {
            assert!(
                matches!(
                    Sophia::new(4, bad.clone()),
                    Err(TrainError::Internal { .. })
                ),
                "config {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn len_mismatch_errors() {
        let mut opt = Sophia::new(3, cfg()).expect("valid");
        let mut theta = vec![1.0_f64; 3];
        // grads too short
        assert!(matches!(
            opt.step(&mut theta, &[0.1, 0.1], Some(&[1.0, 1.0, 1.0])),
            Err(TrainError::ParamCountMismatch { .. })
        ));
        // hessian wrong length on a refresh step (k=1)
        assert!(matches!(
            opt.step(&mut theta, &[0.1, 0.1, 0.1], Some(&[1.0, 1.0])),
            Err(TrainError::ParamCountMismatch { .. })
        ));
    }
}
