//! Flora — Low-Rank Adam via Random Projection (Hao et al. 2024).
//!
//! Reference: Hao, Y., Cao, Y., & Mou, L. (2024). *Flora: Low-Rank Adapters Are
//! Secretly Gradient Compressors*. ICML 2024. <https://arxiv.org/abs/2402.03293>
//!
//! Flora observes that the random down-projection used at LoRA initialisation can
//! be re-interpreted as a *gradient compressor*. Instead of optimising low-rank
//! factors, Flora keeps the full-rank weight but compresses the optimiser state by
//! projecting the gradient `G ∈ ℝ^{m×n}` into an `r`-dimensional subspace with a
//! frozen random Gaussian matrix `P ∈ ℝ^{r×m}`:
//!
//! ```text
//!   compressed = (1/√r) · P · G          (shape r × n, stored)
//!   reconstructed = (1/√r) · Pᵀ · compressed   (shape m × n, unbiased estimate of G)
//! ```
//!
//! With `P_{ij} ∼ 𝒩(0, 1)` the estimator is **unbiased**: `E[Pᵀ P] = r · I`, so
//! `E[(1/r) Pᵀ P G] = G`. The optimiser momentum / variance are kept in the
//! compressed `r × n` space, slashing optimiser memory from `O(m·n)` to `O(r·n)`.
//!
//! ## Re-sampling with state rescaling
//!
//! To remain an unbiased estimator over a long horizon Flora periodically draws a
//! fresh projection `P'`. The momentum stored under the old basis is re-projected
//! into the new basis so training continues smoothly:
//!
//! ```text
//!   m_new = (1/√r) · P' · ((1/√r) · Pᵀ · m_old) .
//! ```
//!
//! This module implements the deterministic CPU core: projection, reconstruction,
//! a compressed Adam step, and basis re-sampling — all driven by [`crate::handle::LcgRng`].

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration of a [`FloraCompressor`].
#[derive(Debug, Clone)]
pub struct FloraConfig {
    /// Leading dimension of the gradient matrix (`m`).
    pub rows: usize,
    /// Trailing dimension of the gradient matrix (`n`).
    pub cols: usize,
    /// Projection rank `r` (must satisfy `1 ≤ r ≤ rows`).
    pub rank: usize,
    /// Adam first-moment decay `β₁`.
    pub beta1: f64,
    /// Adam second-moment decay `β₂`.
    pub beta2: f64,
    /// Adam numerical-stability constant `ε`.
    pub eps: f64,
    /// Seed for the initial random projection.
    pub seed: u64,
}

/// Low-rank gradient compressor with a compressed Adam optimiser state.
///
/// The frozen random projection `p` lives here; the compressed first / second
/// moments are stored in the `r × n` subspace.
#[derive(Debug, Clone)]
pub struct FloraCompressor {
    /// `m` (gradient rows).
    pub rows: usize,
    /// `n` (gradient cols).
    pub cols: usize,
    /// `r` (projection rank).
    pub rank: usize,
    /// Adam β₁.
    pub beta1: f64,
    /// Adam β₂.
    pub beta2: f64,
    /// Adam ε.
    pub eps: f64,
    /// Current random projection `P`, row-major `[rank × rows]`.
    pub p: Vec<f64>,
    /// Compressed first moment `M`, row-major `[rank × cols]`.
    pub m_moment: Vec<f64>,
    /// Compressed second moment `V`, row-major `[rank × cols]`.
    pub v_moment: Vec<f64>,
    /// Number of optimiser steps taken (for bias correction).
    pub step: u64,
    /// RNG state, advanced on every re-sample.
    rng: LcgRng,
}

impl FloraCompressor {
    /// Build a compressor with a freshly sampled projection and zeroed moments.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > rows`.
    pub fn new(cfg: &FloraConfig) -> PeftResult<Self> {
        if cfg.rows == 0 || cfg.cols == 0 || cfg.rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        if cfg.rank > cfg.rows {
            return Err(PeftError::RankTooLarge {
                rank: cfg.rank,
                dim: cfg.rows,
            });
        }
        let mut rng = LcgRng::new(cfg.seed);
        let p = sample_projection(cfg.rank, cfg.rows, &mut rng);
        Ok(Self {
            rows: cfg.rows,
            cols: cfg.cols,
            rank: cfg.rank,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            p,
            m_moment: vec![0.0_f64; cfg.rank * cfg.cols],
            v_moment: vec![0.0_f64; cfg.rank * cfg.cols],
            step: 0,
            rng,
        })
    }

    /// Compress a full gradient `g` (`rows × cols`) to `(1/√r) · P · g` (`rank × cols`).
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] if `g.len() != rows * cols`.
    pub fn compress(&self, g: &[f64]) -> PeftResult<Vec<f64>> {
        if g.len() != self.rows * self.cols {
            return Err(PeftError::DimensionMismatch {
                expected: self.rows * self.cols,
                got: g.len(),
            });
        }
        let inv_sqrt_r = 1.0 / (self.rank as f64).sqrt();
        // out[a, j] = (1/√r) Σ_i P[a, i] · g[i, j]
        let mut out = vec![0.0_f64; self.rank * self.cols];
        for a in 0..self.rank {
            let p_row = a * self.rows;
            for i in 0..self.rows {
                let pai = self.p[p_row + i];
                if pai == 0.0 {
                    continue;
                }
                let g_row = i * self.cols;
                let out_row = a * self.cols;
                for j in 0..self.cols {
                    out[out_row + j] += pai * g[g_row + j];
                }
            }
        }
        for v in out.iter_mut() {
            *v *= inv_sqrt_r;
        }
        Ok(out)
    }

    /// Reconstruct a full-shape estimate `(1/√r) · Pᵀ · c` (`rows × cols`) from a
    /// compressed matrix `c` (`rank × cols`).
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] if `c.len() != rank * cols`.
    pub fn reconstruct(&self, c: &[f64]) -> PeftResult<Vec<f64>> {
        if c.len() != self.rank * self.cols {
            return Err(PeftError::DimensionMismatch {
                expected: self.rank * self.cols,
                got: c.len(),
            });
        }
        let inv_sqrt_r = 1.0 / (self.rank as f64).sqrt();
        // out[i, j] = (1/√r) Σ_a P[a, i] · c[a, j]
        let mut out = vec![0.0_f64; self.rows * self.cols];
        for a in 0..self.rank {
            let p_row = a * self.rows;
            let c_row = a * self.cols;
            for i in 0..self.rows {
                let pai = self.p[p_row + i];
                if pai == 0.0 {
                    continue;
                }
                let out_row = i * self.cols;
                for j in 0..self.cols {
                    out[out_row + j] += pai * c[c_row + j];
                }
            }
        }
        for v in out.iter_mut() {
            *v *= inv_sqrt_r;
        }
        Ok(out)
    }

    /// One Flora-Adam step: compress `g`, update the compressed moments, and return
    /// the *full-shape* parameter update `−lr · reconstruct(m̂ / (√v̂ + ε))`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] if `g.len() != rows * cols`.
    pub fn adam_update(&mut self, g: &[f64], lr: f64) -> PeftResult<Vec<f64>> {
        let c = self.compress(g)?;
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        let mut precond = vec![0.0_f64; self.rank * self.cols];
        for idx in 0..self.m_moment.len() {
            let gi = c[idx];
            self.m_moment[idx] = b1 * self.m_moment[idx] + (1.0 - b1) * gi;
            self.v_moment[idx] = b2 * self.v_moment[idx] + (1.0 - b2) * gi * gi;
            let m_hat = self.m_moment[idx] / bc1;
            let v_hat = self.v_moment[idx] / bc2;
            precond[idx] = m_hat / (v_hat.sqrt() + self.eps);
        }
        let full = self.reconstruct(&precond)?;
        let update: Vec<f64> = full.iter().map(|&v| -lr * v).collect();
        Ok(update)
    }

    /// Draw a fresh projection `P'` and re-project the stored moments into the new
    /// basis: `M ← (1/√r) P' · ((1/√r) Pᵀ M)` (and likewise for `V`).
    ///
    /// This keeps the optimiser state consistent with the new random subspace so the
    /// estimator stays unbiased over long training runs.
    pub fn resample_projection(&mut self) {
        // Reconstruct moments to full space under the *old* basis.
        let full_m = self
            .reconstruct(&self.m_moment.clone())
            .unwrap_or_else(|_| vec![0.0_f64; self.rows * self.cols]);
        let full_v = self
            .reconstruct(&self.v_moment.clone())
            .unwrap_or_else(|_| vec![0.0_f64; self.rows * self.cols]);
        // Sample new projection.
        self.p = sample_projection(self.rank, self.rows, &mut self.rng);
        // Re-compress under the new basis.
        self.m_moment = self
            .compress(&full_m)
            .unwrap_or_else(|_| vec![0.0_f64; self.rank * self.cols]);
        self.v_moment = self
            .compress(&full_v)
            .unwrap_or_else(|_| vec![0.0_f64; self.rank * self.cols]);
    }

    /// Compressed optimiser-state element count: `2 · rank · cols`.
    #[must_use]
    pub fn state_size(&self) -> usize {
        2 * self.rank * self.cols
    }

    /// Full-rank state element count that vanilla Adam would need: `2 · rows · cols`.
    #[must_use]
    pub fn full_state_size(&self) -> usize {
        2 * self.rows * self.cols
    }
}

/// Sample a row-major `[rank × rows]` standard-normal projection matrix.
fn sample_projection(rank: usize, rows: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut p = vec![0.0_f64; rank * rows];
    for v in p.iter_mut() {
        *v = rng.next_normal() as f64;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(rows: usize, cols: usize, rank: usize) -> FloraConfig {
        FloraConfig {
            rows,
            cols,
            rank,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            seed: 42,
        }
    }

    #[test]
    fn rejects_zero_dims() {
        assert!(matches!(
            FloraCompressor::new(&cfg(0, 4, 1)),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            FloraCompressor::new(&cfg(4, 0, 1)),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            FloraCompressor::new(&cfg(4, 4, 0)),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_rank_above_rows() {
        assert!(matches!(
            FloraCompressor::new(&cfg(3, 4, 5)),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn projection_shapes() {
        let f = FloraCompressor::new(&cfg(6, 5, 3))
            .expect("FloraCompressor::new should succeed with valid config");
        assert_eq!(f.p.len(), 3 * 6);
        assert_eq!(f.m_moment.len(), 3 * 5);
        assert_eq!(f.v_moment.len(), 3 * 5);
    }

    #[test]
    fn compress_then_reconstruct_shapes() {
        let f = FloraCompressor::new(&cfg(6, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let g: Vec<f64> = (0..24).map(|i| i as f64 * 0.1).collect();
        let c = f
            .compress(&g)
            .expect("compress should succeed with correctly sized gradient");
        assert_eq!(c.len(), 2 * 4);
        let r = f
            .reconstruct(&c)
            .expect("reconstruct should succeed with valid compressed matrix");
        assert_eq!(r.len(), 6 * 4);
    }

    #[test]
    fn compress_dim_mismatch_errors() {
        let f = FloraCompressor::new(&cfg(4, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let bad = vec![0.0_f64; 15]; // should be 16
        assert!(matches!(
            f.compress(&bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn reconstruct_dim_mismatch_errors() {
        let f = FloraCompressor::new(&cfg(4, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let bad = vec![0.0_f64; 7]; // should be 8 (rank*cols)
        assert!(matches!(
            f.reconstruct(&bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn estimator_is_unbiased_in_expectation() {
        // Averaging reconstruct(compress(G)) over many random projections must
        // converge to G. We use rank = rows so each draw is full-rank but still
        // exercises the (1/r) PᵀP ≈ I property in expectation.
        let rows = 4;
        let cols = 3;
        let rank = 4;
        let g: Vec<f64> = (0..rows * cols).map(|i| (i as f64) * 0.25 - 1.0).collect();
        let n_samples = 4000;
        let mut acc = vec![0.0_f64; rows * cols];
        let mut rng = LcgRng::new(2024);
        for _ in 0..n_samples {
            let p = sample_projection(rank, rows, &mut rng);
            let f = FloraCompressor {
                rows,
                cols,
                rank,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                p,
                m_moment: vec![0.0; rank * cols],
                v_moment: vec![0.0; rank * cols],
                step: 0,
                rng: LcgRng::new(0),
            };
            let c = f
                .compress(&g)
                .expect("compress should succeed with correctly sized gradient");
            let r = f
                .reconstruct(&c)
                .expect("reconstruct should succeed with valid compressed matrix");
            for (a, ri) in acc.iter_mut().zip(r.iter()) {
                *a += ri;
            }
        }
        for v in acc.iter_mut() {
            *v /= n_samples as f64;
        }
        // Monte-Carlo estimate should be close to G.
        let max_err = acc
            .iter()
            .zip(g.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 0.25,
            "unbiased estimator should recover G on average, max_err={max_err}"
        );
    }

    #[test]
    fn adam_update_shape_and_sign() {
        let mut f = FloraCompressor::new(&cfg(5, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let g: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1 - 1.0).collect();
        let upd = f
            .adam_update(&g, 0.01)
            .expect("Adam update should succeed with valid gradient");
        assert_eq!(upd.len(), 5 * 4);
        assert_eq!(f.step, 1);
        assert!(upd.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn adam_first_step_descends_loss_direction() {
        // For a constant gradient on a single column the update should move opposite
        // to the gradient (descent). We check the sign of the reconstructed update
        // correlates negatively with the gradient on average.
        let mut f = FloraCompressor::new(&cfg(4, 1, 4))
            .expect("FloraCompressor::new should succeed with valid config");
        let g = vec![1.0_f64, 1.0, 1.0, 1.0];
        let upd = f
            .adam_update(&g, 0.1)
            .expect("Adam update should succeed with valid gradient");
        // Sum of update should be negative (descending a positive gradient).
        let s: f64 = upd.iter().sum();
        assert!(
            s < 0.0,
            "update should descend a positive gradient, sum={s}"
        );
    }

    #[test]
    fn resample_keeps_moment_shapes() {
        let mut f = FloraCompressor::new(&cfg(6, 4, 3))
            .expect("FloraCompressor::new should succeed with valid config");
        let g: Vec<f64> = (0..24).map(|i| (i as f64) * 0.05).collect();
        let _ = f
            .adam_update(&g, 0.01)
            .expect("Adam update should succeed with valid gradient");
        let p_before = f.p.clone();
        f.resample_projection();
        assert_eq!(f.m_moment.len(), 3 * 4);
        assert_eq!(f.v_moment.len(), 3 * 4);
        assert_eq!(f.p.len(), 3 * 6);
        assert_ne!(f.p, p_before, "projection must change after resample");
    }

    #[test]
    fn state_size_is_smaller_than_full() {
        let f = FloraCompressor::new(&cfg(128, 64, 8))
            .expect("FloraCompressor::new should succeed with valid config");
        assert_eq!(f.state_size(), 2 * 8 * 64);
        assert_eq!(f.full_state_size(), 2 * 128 * 64);
        assert!(
            f.state_size() < f.full_state_size(),
            "compressed Adam state must be smaller than full-rank state"
        );
    }

    #[test]
    fn multiple_steps_accumulate_moments() {
        let mut f = FloraCompressor::new(&cfg(5, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let g: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let _ = f
            .adam_update(&g, 0.01)
            .expect("Adam update should succeed with valid gradient");
        let m1 = f.m_moment.clone();
        let _ = f
            .adam_update(&g, 0.01)
            .expect("Adam update should succeed with valid gradient");
        assert_eq!(f.step, 2);
        assert_ne!(f.m_moment, m1, "moments must evolve across steps");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let f1 = FloraCompressor::new(&cfg(5, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        let f2 = FloraCompressor::new(&cfg(5, 4, 2))
            .expect("FloraCompressor::new should succeed with valid config");
        assert_eq!(f1.p, f2.p, "same seed must yield same projection");
    }
}
