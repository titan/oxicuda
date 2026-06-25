//! Sparse-gradient AdamW optimizer for large embedding tables.
//!
//! Reference: Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay
//! Regularization" (AdamW), ICLR 2019; Diederik Kingma and Jimmy Ba, "Adam:
//! A Method for Stochastic Optimization", ICLR 2015.
//!
//! # Why a sparse optimizer
//!
//! Recommender embedding tables have millions of rows but each mini-batch only
//! touches the handful of user / item rows that appear in it. A dense Adam step
//! would read and write the *entire* table (and its two moment buffers) every
//! update, which is bandwidth-prohibitive. This optimizer keeps the
//! `[n_rows × dim]` first- and second-moment buffers but only **time-steps and
//! updates the rows that received a gradient**, using a *per-row* step counter
//! for bias correction. A row that is updated `t_r` times is corrected exactly
//! as if it had run for `t_r` dense steps, which is the standard "lazy / sparse
//! Adam" semantics used by production embedding optimizers.
//!
//! Decoupled weight decay (AdamW) is applied multiplicatively to the touched
//! rows only: `w ← w · (1 − lr · λ)` *before* the adaptive gradient step, so the
//! decay is independent of the gradient magnitude (unlike L2 folded into the
//! gradient). Rows that are never touched are left untouched — exactly matching
//! dense AdamW for the sparse-access pattern, because their effective step count
//! is zero.
//!
//! All state is FP32. No randomness is required; the optimizer is fully
//! deterministic for a fixed sequence of gradient rows.

use std::collections::HashMap;

use crate::error::{RecsysError, RecsysResult};

/// Hyper-parameters for [`SparseAdamW`].
#[derive(Debug, Clone)]
pub struct SparseAdamWConfig {
    /// Embedding dimension (columns per row).
    pub dim: usize,
    /// Number of rows in the embedding table.
    pub n_rows: usize,
    /// Learning rate `α > 0`.
    pub lr: f32,
    /// First-moment decay `β₁ ∈ [0, 1)`.
    pub beta1: f32,
    /// Second-moment decay `β₂ ∈ [0, 1)`.
    pub beta2: f32,
    /// Numerical-stability constant `ε > 0`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient `λ ≥ 0`.
    pub weight_decay: f32,
}

impl SparseAdamWConfig {
    /// Reasonable defaults (`lr = 1e-3`, `β₁ = 0.9`, `β₂ = 0.999`,
    /// `ε = 1e-8`, `λ = 0`) for a table of the given shape.
    #[must_use]
    pub fn new(n_rows: usize, dim: usize) -> Self {
        Self {
            dim,
            n_rows,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Validates the configuration.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidEmbeddingDim`] if `dim == 0`.
    /// - [`RecsysError::InvalidNumItems`] if `n_rows == 0`.
    /// - [`RecsysError::InvalidConfig`] for out-of-range hyper-parameters.
    pub fn validate(&self) -> RecsysResult<()> {
        if self.dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: self.dim });
        }
        if self.n_rows == 0 {
            return Err(RecsysError::InvalidNumItems { n: self.n_rows });
        }
        if self.lr <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "lr must be > 0".into(),
            });
        }
        if !(0.0..1.0).contains(&self.beta1) {
            return Err(RecsysError::InvalidConfig {
                msg: "beta1 must be in [0, 1)".into(),
            });
        }
        if !(0.0..1.0).contains(&self.beta2) {
            return Err(RecsysError::InvalidConfig {
                msg: "beta2 must be in [0, 1)".into(),
            });
        }
        if self.eps <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "eps must be > 0".into(),
            });
        }
        if self.weight_decay < 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: "weight_decay must be >= 0".into(),
            });
        }
        Ok(())
    }
}

/// A `(row, gradient)` pair, the unit of a sparse update.
#[derive(Debug, Clone)]
pub struct RowGrad {
    /// Row index into the embedding table.
    pub row: usize,
    /// Gradient for this row (length must equal `dim`).
    pub grad: Vec<f32>,
}

/// Sparse-gradient AdamW optimizer with per-row moment state and bias counters.
#[derive(Debug, Clone)]
pub struct SparseAdamW {
    cfg: SparseAdamWConfig,
    /// First-moment estimate `m`, `[n_rows × dim]`, lazily nonzero.
    m: Vec<f32>,
    /// Second-moment estimate `v`, `[n_rows × dim]`, lazily nonzero.
    v: Vec<f32>,
    /// Per-row update count `t_r` for bias correction (0 until first touch).
    steps: Vec<u64>,
}

impl SparseAdamW {
    /// Allocates moment buffers for a fresh table.
    ///
    /// # Errors
    /// Propagates [`SparseAdamWConfig::validate`].
    pub fn new(cfg: SparseAdamWConfig) -> RecsysResult<Self> {
        cfg.validate()?;
        let total = cfg.n_rows * cfg.dim;
        Ok(Self {
            m: vec![0.0_f32; total],
            v: vec![0.0_f32; total],
            steps: vec![0_u64; cfg.n_rows],
            cfg,
        })
    }

    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.cfg.n_rows
    }

    /// Number of times `row` has been updated so far.
    #[must_use]
    pub fn row_step(&self, row: usize) -> u64 {
        self.steps.get(row).copied().unwrap_or(0)
    }

    /// Apply a single sparse AdamW update to `params[row]` in place.
    ///
    /// `params` is the flat `[n_rows × dim]` embedding table being optimized.
    ///
    /// # Errors
    /// - [`RecsysError::DimensionMismatch`] if `params.len() != n_rows · dim`
    ///   or `grad.len() != dim`.
    /// - [`RecsysError::ItemOutOfBounds`] if `row >= n_rows`.
    pub fn step_row(&mut self, params: &mut [f32], row: usize, grad: &[f32]) -> RecsysResult<()> {
        let dim = self.cfg.dim;
        if params.len() != self.cfg.n_rows * dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.cfg.n_rows * dim,
                got: params.len(),
            });
        }
        if grad.len() != dim {
            return Err(RecsysError::DimensionMismatch {
                expected: dim,
                got: grad.len(),
            });
        }
        if row >= self.cfg.n_rows {
            return Err(RecsysError::ItemOutOfBounds {
                idx: row,
                n: self.cfg.n_rows,
            });
        }

        let t = self.steps[row] + 1;
        self.steps[row] = t;

        let SparseAdamWConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            ..
        } = self.cfg;

        // Per-row bias-correction denominators using this row's own step count.
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        let base = row * dim;
        for (j, &g) in grad.iter().enumerate() {
            let idx = base + j;

            // Decoupled weight decay (applied to the parameter, not the grad).
            if weight_decay > 0.0 {
                params[idx] *= 1.0 - lr * weight_decay;
            }

            // Moment updates.
            let m = beta1 * self.m[idx] + (1.0 - beta1) * g;
            let v = beta2 * self.v[idx] + (1.0 - beta2) * g * g;
            self.m[idx] = m;
            self.v[idx] = v;

            // Bias-corrected step.
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            params[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        Ok(())
    }

    /// Apply a batch of sparse updates. Gradients that share a row are summed
    /// first (the natural semantics when a row appears multiple times in a
    /// mini-batch) and then applied as a *single* AdamW step for that row, so
    /// the per-row step counter advances once per call regardless of how many
    /// times the row appeared.
    ///
    /// # Errors
    /// Propagates [`Self::step_row`]; also
    /// [`RecsysError::DimensionMismatch`] if any `grad.len() != dim`.
    pub fn step_batch(&mut self, params: &mut [f32], grads: &[RowGrad]) -> RecsysResult<()> {
        let dim = self.cfg.dim;
        // Accumulate duplicate rows into a single summed gradient.
        let mut acc: HashMap<usize, Vec<f32>> = HashMap::new();
        for rg in grads {
            if rg.grad.len() != dim {
                return Err(RecsysError::DimensionMismatch {
                    expected: dim,
                    got: rg.grad.len(),
                });
            }
            if rg.row >= self.cfg.n_rows {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: rg.row,
                    n: self.cfg.n_rows,
                });
            }
            let entry = acc.entry(rg.row).or_insert_with(|| vec![0.0_f32; dim]);
            for (a, &g) in entry.iter_mut().zip(rg.grad.iter()) {
                *a += g;
            }
        }
        // Deterministic application order (sorted rows) for reproducibility.
        let mut rows: Vec<usize> = acc.keys().copied().collect();
        rows.sort_unstable();
        for row in rows {
            let g = &acc[&row];
            self.step_row(params, row, g)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_rows: usize, dim: usize) -> SparseAdamWConfig {
        SparseAdamWConfig {
            dim,
            n_rows,
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(SparseAdamW::new(cfg(0, 4)).is_err());
        assert!(SparseAdamW::new(cfg(4, 0)).is_err());
        let mut bad = cfg(4, 4);
        bad.beta1 = 1.0;
        assert!(SparseAdamW::new(bad).is_err());
        let mut bad2 = cfg(4, 4);
        bad2.lr = 0.0;
        assert!(SparseAdamW::new(bad2).is_err());
    }

    #[test]
    fn first_step_moves_against_gradient() {
        // A positive gradient must decrease the parameter; first-step magnitude
        // equals lr (bias-corrected Adam: m_hat/sqrt(v_hat) = sign(g)).
        let mut opt = SparseAdamW::new(cfg(3, 2)).expect("ok");
        let mut params = vec![0.0_f32; 6];
        opt.step_row(&mut params, 1, &[1.0, -2.0]).expect("step ok");
        // Row 1 entries move opposite to gradient sign, magnitude ≈ lr.
        assert!((params[2] + 0.1).abs() < 1e-4, "got {}", params[2]);
        assert!((params[3] - 0.1).abs() < 1e-4, "got {}", params[3]);
        // Untouched rows stay exactly zero (sparsity).
        assert_eq!(&params[0..2], &[0.0, 0.0]);
        assert_eq!(&params[4..6], &[0.0, 0.0]);
        assert_eq!(opt.row_step(1), 1);
        assert_eq!(opt.row_step(0), 0);
    }

    #[test]
    fn untouched_rows_keep_zero_step_count() {
        let mut opt = SparseAdamW::new(cfg(5, 3)).expect("ok");
        let mut params = vec![0.5_f32; 15];
        for _ in 0..4 {
            opt.step_row(&mut params, 2, &[0.1, 0.1, 0.1]).expect("ok");
        }
        assert_eq!(opt.row_step(2), 4);
        for r in [0usize, 1, 3, 4] {
            assert_eq!(opt.row_step(r), 0);
            // Their parameters are unchanged.
            assert!(params[r * 3..(r + 1) * 3].iter().all(|&p| p == 0.5));
        }
    }

    #[test]
    fn converges_toward_target_on_quadratic() {
        // Minimise 0.5*(w - t)^2 ⇒ grad = w - t. AdamW should drive w → t.
        let mut opt = SparseAdamW::new(cfg(1, 1)).expect("ok");
        let mut params = vec![0.0_f32];
        let target = 3.0_f32;
        for _ in 0..500 {
            let g = params[0] - target;
            opt.step_row(&mut params, 0, &[g]).expect("ok");
        }
        assert!(
            (params[0] - target).abs() < 1e-2,
            "expected convergence to {target}, got {}",
            params[0]
        );
    }

    #[test]
    fn weight_decay_shrinks_param_with_zero_grad() {
        // With zero gradient, decoupled decay must shrink the parameter toward 0.
        let mut c = cfg(1, 1);
        c.weight_decay = 0.5;
        let mut opt = SparseAdamW::new(c).expect("ok");
        let mut params = vec![10.0_f32];
        let before = params[0];
        opt.step_row(&mut params, 0, &[0.0]).expect("ok");
        // decay multiplies by (1 - lr*wd) = 1 - 0.05 = 0.95; m/v stay 0 so the
        // adaptive term is ~0 ⇒ param ≈ 0.95 * before.
        assert!(params[0] < before, "decay should shrink: {}", params[0]);
        assert!(
            (params[0] - before * 0.95).abs() < 1e-3,
            "got {}",
            params[0]
        );
    }

    #[test]
    fn batch_sums_duplicate_rows() {
        // Two grads on the same row in one batch == one grad of their sum,
        // applied as a single step (step count advances once).
        let mut opt_batch = SparseAdamW::new(cfg(2, 2)).expect("ok");
        let mut p_batch = vec![1.0_f32; 4];
        let grads = vec![
            RowGrad {
                row: 0,
                grad: vec![0.3, -0.1],
            },
            RowGrad {
                row: 0,
                grad: vec![0.2, -0.4],
            },
        ];
        opt_batch.step_batch(&mut p_batch, &grads).expect("ok");

        let mut opt_single = SparseAdamW::new(cfg(2, 2)).expect("ok");
        let mut p_single = vec![1.0_f32; 4];
        opt_single
            .step_row(&mut p_single, 0, &[0.5, -0.5])
            .expect("ok");

        for (a, b) in p_batch.iter().zip(p_single.iter()) {
            assert!((a - b).abs() < 1e-6, "batch != summed-single: {a} vs {b}");
        }
        assert_eq!(opt_batch.row_step(0), 1);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut opt = SparseAdamW::new(cfg(2, 3)).expect("ok");
        let mut params = vec![0.0_f32; 6];
        assert!(opt.step_row(&mut params, 0, &[1.0, 2.0]).is_err());
        let mut wrong = vec![0.0_f32; 5];
        assert!(opt.step_row(&mut wrong, 0, &[1.0, 2.0, 3.0]).is_err());
        assert!(opt.step_row(&mut params, 9, &[1.0, 2.0, 3.0]).is_err());
    }
}
