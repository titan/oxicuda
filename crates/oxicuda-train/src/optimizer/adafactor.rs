//! Adafactor optimizer — Shazeer & Stern, 2018.
//!
//! "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
//! (arXiv:1804.04235).
//!
//! Adafactor reduces the `O(m·n)` second-moment memory of Adam to `O(m + n)`
//! for 2-D parameters by maintaining only **per-row** and **per-column**
//! running averages `R ∈ ℝ^m` and `C ∈ ℝ^n` of the squared gradient, and
//! reconstructing the full second-moment estimate on the fly as a rank-1
//! outer product:
//!
//! ```text
//! V̂[i,j] = R[i] · C[j] / (1ᵀ R)          (factored second moment)
//! ```
//!
//! For 1-D (vector) parameters factorisation is impossible, so a dense
//! second-moment vector `V ∈ ℝ^d` is kept instead (still cheap).
//!
//! The full update for step `t` is:
//!
//! ```text
//! β̂₂(t) = 1 − t^(−c)                                  (decay schedule, c = 0.8)
//! 2-D:  R ← β̂₂·R + (1−β̂₂)·(G² + ε₁)·1                 (row mean of G²)
//!       C ← β̂₂·C + (1−β̂₂)·1ᵀ(G² + ε₁)                 (col mean of G²)
//!       V̂ = (R·Cᵀ) / (1ᵀ R)
//! 1-D:  V ← β̂₂·V + (1−β̂₂)·(G² + ε₁)
//! U = G / √V̂                                           (pre-conditioned grad)
//! Û = U / max(1, RMS(U)/d_clip)                        (update clipping, d_clip = 1)
//! α = max(ε₂, RMS(θ)) · ρ(t)                           (relative step size)
//! θ ← θ − α·Û − α·λ·θ                                  (decoupled weight decay)
//! ```
//!
//! where `ρ(t) = min(10⁻², 1/√t)` is the default *relative* step-size schedule
//! (used when `lr` is `None`); when an explicit `lr` is supplied the relative
//! step is replaced by that fixed value and `scale_parameter` controls whether
//! it is further multiplied by `max(ε₂, RMS(θ))`.
//!
//! All running state is stored as `f64` for the numerical robustness the
//! factored estimate benefits from.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Adafactor`] optimizer.
#[derive(Debug, Clone)]
pub struct AdafactorConfig {
    /// Optional fixed external learning rate.  When `None`, Adafactor uses its
    /// internal relative step-size schedule `ρ(t) = min(eps_relative, 1/√t)`.
    pub lr: Option<f64>,
    /// Exponent `c` of the second-moment decay schedule `β̂₂(t) = 1 − t^(−c)`
    /// (default 0.8; must be in `(0, 1)`).
    pub decay_rate: f64,
    /// Optional fixed `β₁` first-moment EMA.  `None` (the default) disables the
    /// first moment entirely, matching the memory-lean Adafactor recipe.
    pub beta1: Option<f64>,
    /// Regularisation constant `ε₁` added to `G²` before averaging (default 1e-30).
    pub eps1: f64,
    /// Relative step-size floor `ε₂` used in `α = max(ε₂, RMS(θ))·ρ(t)`
    /// (default 1e-3; must be > 0).
    pub eps2: f64,
    /// Update-clipping threshold `d` on the RMS of the pre-conditioned update
    /// (default 1.0; must be > 0).
    pub clip_threshold: f64,
    /// Decoupled (AdamW-style) weight-decay coefficient `λ` (default 0; ≥ 0).
    pub weight_decay: f64,
    /// Whether to multiply the (relative or fixed) step size by
    /// `max(ε₂, RMS(θ))`.  `true` (default) reproduces the paper's
    /// parameter-scaled behaviour.
    pub scale_parameter: bool,
    /// Cap on the internal relative step schedule `ρ(t) = min(eps_relative, 1/√t)`
    /// (default 1e-2; only used when `lr` is `None`).
    pub eps_relative: f64,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: None,
            decay_rate: 0.8,
            beta1: None,
            eps1: 1e-30,
            eps2: 1e-3,
            clip_threshold: 1.0,
            weight_decay: 0.0,
            scale_parameter: true,
            eps_relative: 1e-2,
        }
    }
}

impl AdafactorConfig {
    /// Validate every field.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if an explicit `lr` is `≤ 0`.
    /// * [`TrainError::Internal`] for any other out-of-range field.
    fn validate(&self) -> TrainResult<()> {
        if let Some(lr) = self.lr {
            if lr <= 0.0 || lr.is_nan() {
                return Err(TrainError::InvalidLearningRate { lr });
            }
        }
        if !(0.0..1.0).contains(&self.decay_rate) || self.decay_rate <= 0.0 {
            return Err(TrainError::Internal {
                msg: format!("decay_rate must be in (0, 1), got {}", self.decay_rate),
            });
        }
        if let Some(b1) = self.beta1 {
            if !(0.0..1.0).contains(&b1) {
                return Err(TrainError::Internal {
                    msg: format!("beta1 must be in [0, 1), got {b1}"),
                });
            }
        }
        if self.eps1 <= 0.0 || self.eps1.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("eps1 must be positive, got {}", self.eps1),
            });
        }
        if self.eps2 <= 0.0 || self.eps2.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("eps2 must be positive, got {}", self.eps2),
            });
        }
        if self.clip_threshold <= 0.0 || self.clip_threshold.is_nan() {
            return Err(TrainError::Internal {
                msg: format!(
                    "clip_threshold must be positive, got {}",
                    self.clip_threshold
                ),
            });
        }
        if self.weight_decay < 0.0 || self.weight_decay.is_nan() {
            return Err(TrainError::Internal {
                msg: format!(
                    "weight_decay must be non-negative, got {}",
                    self.weight_decay
                ),
            });
        }
        if self.eps_relative <= 0.0 || self.eps_relative.is_nan() {
            return Err(TrainError::Internal {
                msg: format!("eps_relative must be positive, got {}", self.eps_relative),
            });
        }
        Ok(())
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Adafactor second-moment storage: factored (2-D) or dense (1-D).
#[derive(Debug, Clone)]
enum SecondMoment {
    /// 2-D parameter of shape `(rows, cols)`: per-row `R` and per-column `C`.
    Factored {
        rows: usize,
        cols: usize,
        row: Vec<f64>,
        col: Vec<f64>,
    },
    /// 1-D parameter: dense second moment.
    Dense { v: Vec<f64> },
}

/// Adafactor optimizer operating on a single flat parameter tensor.
///
/// A 2-D parameter is declared by passing `Some((rows, cols))` to
/// [`Adafactor::new`]; passing `None` treats the parameter as a flat vector and
/// keeps a dense second moment.
#[derive(Debug, Clone)]
pub struct Adafactor {
    config: AdafactorConfig,
    second: SecondMoment,
    /// Optional first moment (only allocated when `config.beta1` is `Some`).
    m: Option<Vec<f64>>,
    dim: usize,
    t: u64,
}

impl Adafactor {
    /// Create an Adafactor optimizer for a parameter of `dim` elements.
    ///
    /// * `shape = Some((rows, cols))` enables the factored (`O(rows+cols)`)
    ///   second moment and requires `rows * cols == dim`.
    /// * `shape = None` keeps a dense second moment of length `dim`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `dim == 0`.
    /// * Any error from `AdafactorConfig::validate`.
    /// * [`TrainError::ShapeMismatch`] if `rows * cols != dim`.
    pub fn new(
        dim: usize,
        shape: Option<(usize, usize)>,
        config: AdafactorConfig,
    ) -> TrainResult<Self> {
        if dim == 0 {
            return Err(TrainError::EmptyParams);
        }
        config.validate()?;
        let second = match shape {
            Some((rows, cols)) => {
                if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(dim) {
                    return Err(TrainError::ShapeMismatch {
                        expected: vec![dim],
                        got: vec![rows, cols],
                    });
                }
                SecondMoment::Factored {
                    rows,
                    cols,
                    row: vec![0.0; rows],
                    col: vec![0.0; cols],
                }
            }
            None => SecondMoment::Dense { v: vec![0.0; dim] },
        };
        let m = config.beta1.map(|_| vec![0.0_f64; dim]);
        Ok(Self {
            config,
            second,
            m,
            dim,
            t: 0,
        })
    }

    /// Root-mean-square of a slice (treated as `f64`).
    #[inline]
    fn rms(slice: &[f64]) -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = slice.iter().map(|&x| x * x).sum();
        (sum_sq / slice.len() as f64).sqrt()
    }

    /// Perform one Adafactor update in-place on `params` using `grads`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ShapeMismatch`] if `params.len()`/`grads.len()` differ
    ///   from the configured dimension.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> TrainResult<()> {
        if params.len() != self.dim || grads.len() != self.dim {
            return Err(TrainError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![params.len(), grads.len()],
            });
        }
        self.t += 1;
        let t = self.t as f64;
        let beta2_hat = 1.0 - t.powf(-self.config.decay_rate);
        let eps1 = self.config.eps1;

        // Pre-conditioned update buffer Û (before clipping / step scaling).
        let mut update = vec![0.0_f64; self.dim];

        match &mut self.second {
            SecondMoment::Factored {
                rows,
                cols,
                row,
                col,
            } => {
                let rows = *rows;
                let cols = *cols;
                // Row-mean and col-mean of (G² + ε₁).
                let mut new_row = vec![0.0_f64; rows];
                let mut new_col = vec![0.0_f64; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        let g = f64::from(grads[i * cols + j]);
                        let gsq = g * g + eps1;
                        new_row[i] += gsq;
                        new_col[j] += gsq;
                    }
                }
                for r in &mut new_row {
                    *r /= cols as f64;
                }
                for c in &mut new_col {
                    *c /= rows as f64;
                }
                for (r, nr) in row.iter_mut().zip(new_row.iter()) {
                    *r = beta2_hat * *r + (1.0 - beta2_hat) * *nr;
                }
                for (c, nc) in col.iter_mut().zip(new_col.iter()) {
                    *c = beta2_hat * *c + (1.0 - beta2_hat) * *nc;
                }
                // Reconstruct V̂ = (R · Cᵀ) / mean(R) and U = G / √V̂.
                let row_mean: f64 = row.iter().sum::<f64>() / rows as f64;
                let inv_row_mean = if row_mean > 0.0 { 1.0 / row_mean } else { 0.0 };
                for i in 0..rows {
                    for j in 0..cols {
                        let v_hat = row[i] * col[j] * inv_row_mean;
                        let denom = v_hat.sqrt().max(1e-30);
                        update[i * cols + j] = f64::from(grads[i * cols + j]) / denom;
                    }
                }
            }
            SecondMoment::Dense { v } => {
                for (idx, vi) in v.iter_mut().enumerate() {
                    let g = f64::from(grads[idx]);
                    let gsq = g * g + eps1;
                    *vi = beta2_hat * *vi + (1.0 - beta2_hat) * gsq;
                    let denom = vi.sqrt().max(1e-30);
                    update[idx] = g / denom;
                }
            }
        }

        // Optional first-moment EMA on the pre-conditioned update.
        if let (Some(beta1), Some(m)) = (self.config.beta1, self.m.as_mut()) {
            for (mi, u) in m.iter_mut().zip(update.iter_mut()) {
                *mi = beta1 * *mi + (1.0 - beta1) * *u;
                *u = *mi;
            }
        }

        // Update clipping: Û ← U / max(1, RMS(U)/d).
        let update_rms = Self::rms(&update);
        let clip = (update_rms / self.config.clip_threshold).max(1.0);
        if clip > 1.0 {
            let inv = 1.0 / clip;
            for u in &mut update {
                *u *= inv;
            }
        }

        // Step size: explicit lr or relative schedule, optionally parameter-scaled.
        let rho = match self.config.lr {
            Some(lr) => lr,
            None => self.config.eps_relative.min(1.0 / t.sqrt()),
        };
        let param_scale = if self.config.scale_parameter {
            // RMS over f64-promoted params.
            let psum_sq: f64 = params.iter().map(|&p| f64::from(p) * f64::from(p)).sum();
            let prms = (psum_sq / self.dim as f64).sqrt();
            prms.max(self.config.eps2)
        } else {
            1.0
        };
        let alpha = rho * param_scale;
        let wd = self.config.weight_decay;

        for (p, u) in params.iter_mut().zip(update.iter()) {
            let mut val = f64::from(*p);
            if wd > 0.0 {
                val -= alpha * wd * val;
            }
            val -= alpha * *u;
            *p = val as f32;
        }
        Ok(())
    }

    /// Reset all running state and the step counter.
    pub fn reset(&mut self) {
        match &mut self.second {
            SecondMoment::Factored { row, col, .. } => {
                row.iter_mut().for_each(|x| *x = 0.0);
                col.iter_mut().for_each(|x| *x = 0.0);
            }
            SecondMoment::Dense { v } => v.iter_mut().for_each(|x| *x = 0.0),
        }
        if let Some(m) = self.m.as_mut() {
            m.iter_mut().for_each(|x| *x = 0.0);
        }
        self.t = 0;
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Whether the optimizer uses the factored (`O(rows+cols)`) second moment.
    #[must_use]
    pub fn is_factored(&self) -> bool {
        matches!(self.second, SecondMoment::Factored { .. })
    }

    /// Parameter dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg_lr(lr: f64) -> AdafactorConfig {
        AdafactorConfig {
            lr: Some(lr),
            scale_parameter: false,
            ..Default::default()
        }
    }

    /// Parameter-scaled config (the paper's recipe): the step is multiplied by
    /// `max(eps2, RMS(θ))`, so it shrinks as the parameters do, giving genuine
    /// geometric convergence rather than the sign-descent floor of a fixed lr.
    fn cfg_scaled(lr: f64) -> AdafactorConfig {
        AdafactorConfig {
            lr: Some(lr),
            scale_parameter: true,
            eps2: 1e-8,
            ..Default::default()
        }
    }

    #[test]
    fn rejects_bad_shape() {
        let r = Adafactor::new(6, Some((4, 4)), AdafactorConfig::default());
        assert!(matches!(r, Err(TrainError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(matches!(
            Adafactor::new(0, None, AdafactorConfig::default()),
            Err(TrainError::EmptyParams)
        ));
    }

    #[test]
    fn rejects_bad_lr() {
        assert!(matches!(
            Adafactor::new(4, None, cfg_lr(0.0)),
            Err(TrainError::InvalidLearningRate { .. })
        ));
    }

    #[test]
    fn factored_flag_set_for_matrix() {
        let opt = Adafactor::new(12, Some((3, 4)), AdafactorConfig::default()).expect("valid");
        assert!(opt.is_factored());
        let dense = Adafactor::new(12, None, AdafactorConfig::default()).expect("valid");
        assert!(!dense.is_factored());
    }

    #[test]
    fn step_changes_params() {
        let mut opt = Adafactor::new(4, None, cfg_lr(1e-1)).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.5_f32; 4];
        opt.step(&mut params, &grads).expect("step ok");
        for &p in &params {
            assert!(p < 1.0, "params should decrease, got {p}");
        }
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn wrong_length_errors() {
        let mut opt = Adafactor::new(4, None, cfg_lr(1e-2)).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let r = opt.step(&mut params, &[0.1, 0.2]);
        assert!(matches!(r, Err(TrainError::ShapeMismatch { .. })));
    }

    /// Dense Adafactor with parameter scaling minimises a convex quadratic
    /// f(x) = Σ xᵢ² well below tolerance (the step shrinks with RMS(θ)).
    #[test]
    fn dense_converges_quadratic() {
        let dim = 8;
        let mut opt = Adafactor::new(dim, None, cfg_scaled(0.3)).expect("valid");
        let mut params = vec![1.5_f32; dim];
        for _ in 0..600 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("step ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(max_abs < 1e-2, "dense Adafactor not converged: {max_abs}");
    }

    /// Factored Adafactor (parameter-scaled) on a 2-D quadratic with a random
    /// initialisation also converges below tolerance.
    #[test]
    fn factored_converges_quadratic() {
        let (rows, cols) = (4, 5);
        let dim = rows * cols;
        let mut opt = Adafactor::new(dim, Some((rows, cols)), cfg_scaled(0.3)).expect("valid");
        let mut rng = LcgRng::new(2024);
        let mut params: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        for _ in 0..800 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("step ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(
            max_abs < 5e-2,
            "factored Adafactor not converged: {max_abs}"
        );
    }

    /// With `beta1 = Some(..)` the first moment is allocated; combined with
    /// parameter scaling the optimizer converges on a quadratic.
    #[test]
    fn with_momentum_converges() {
        let cfg = AdafactorConfig {
            lr: Some(0.3),
            beta1: Some(0.9),
            scale_parameter: true,
            eps2: 1e-8,
            ..Default::default()
        };
        let mut opt = Adafactor::new(4, None, cfg).expect("valid");
        let mut params = vec![1.0_f32; 4];
        for _ in 0..1000 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("step ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(
            max_abs < 5e-2,
            "momentum Adafactor not converged: {max_abs}"
        );
    }

    /// With a fixed lr and no parameter scaling, Adafactor's preconditioned
    /// update is sign-like, so it reduces the loss to a floor of order `lr`
    /// without diverging — verifies the fixed-lr / no-scale path stays stable.
    #[test]
    fn fixed_lr_makes_progress() {
        let mut opt = Adafactor::new(4, None, cfg_lr(0.05)).expect("valid");
        let mut params = vec![1.5_f32; 4];
        for _ in 0..400 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let max_abs = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        // Floor is ~lr=0.05; just below the 1.5 start and bounded.
        assert!(max_abs < 0.3, "fixed-lr Adafactor stalled high: {max_abs}");
    }

    #[test]
    fn reset_clears_state() {
        let mut opt = Adafactor::new(3, Some((1, 3)), cfg_lr(1e-2)).expect("valid");
        let mut p = vec![1.0_f32; 3];
        opt.step(&mut p, &[0.5, 0.5, 0.5]).expect("ok");
        assert_eq!(opt.step_count(), 1);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
    }

    /// Relative step schedule (lr = None) still drives a quadratic down,
    /// demonstrating the internal ρ(t) = min(eps_relative, 1/√t) path.
    #[test]
    fn relative_step_makes_progress() {
        let mut opt = Adafactor::new(4, None, AdafactorConfig::default()).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let start = params[0].abs();
        for _ in 0..2000 {
            let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
            opt.step(&mut params, &grads).expect("ok");
        }
        let end = params.iter().fold(0.0_f32, |a, &p| a.max(p.abs()));
        assert!(end < start, "relative-step Adafactor made no progress");
        assert!(end < 0.5, "relative-step Adafactor stalled at {end}");
    }
}
