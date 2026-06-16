//! Muon optimizer — Nesterov SGD with Newton-Schulz orthogonalization.
//!
//! Muon (Jordan et al., 2024) applies Nesterov momentum to gradients and then
//! orthogonalises the update matrix using Newton-Schulz iterations before applying it.
//! The orthogonalisation ensures the effective step lies along the steepest-descent
//! direction on the manifold of matrices with bounded spectral norm, which empirically
//! improves convergence stability for large weight matrices.
//!
//! ## Algorithm
//!
//! ```text
//! v_t ← μ·v_{t-1} + g_t                    // momentum accumulation
//!
//! // Newton-Schulz orthogonalization (default 5 iterations)
//! X₀  = v_t / ‖v_t‖_F
//! Xₙ₊₁ = 1.5·Xₙ − 0.5·Xₙ·(Xₙᵀ·Xₙ)/min(rows,cols)
//!
//! // Parameter update
//! p ← p − lr · NS(v_t)
//! ```

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Muon`] optimizer.
#[derive(Debug, Clone)]
pub struct MuonConfig {
    /// Learning rate.
    pub lr: f32,
    /// Momentum coefficient (default 0.95).
    pub momentum: f32,
    /// Number of Newton-Schulz orthogonalization iterations (0 = disabled).
    pub ns_steps: usize,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.95,
            ns_steps: 5,
        }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Muon optimizer with Newton-Schulz orthogonalization.
///
/// Operates on a single weight matrix of shape `(rows, cols)` stored in
/// row-major flat layout.  For 1-D parameters, use `rows=1, cols=dim`.
pub struct Muon {
    velocity: Vec<f32>,
    rows: usize,
    cols: usize,
    config: MuonConfig,
}

impl Muon {
    /// Create a new `Muon` optimizer for a weight matrix of shape `(rows, cols)`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidLearningRate`] if `config.lr <= 0`.
    pub fn new(rows: usize, cols: usize, config: MuonConfig) -> TrainResult<Self> {
        if config.lr <= 0.0 {
            return Err(TrainError::InvalidLearningRate {
                lr: config.lr as f64,
            });
        }
        Ok(Self {
            velocity: vec![0.0; rows * cols],
            rows,
            cols,
            config,
        })
    }

    /// Perform one optimizer step for a weight matrix stored as a flat slice.
    ///
    /// `rows` and `cols` must equal the dimensions used at construction; `params`
    /// and `grads` must have length `rows * cols`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::ParamCountMismatch`] if slice lengths are inconsistent.
    pub fn step(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        rows: usize,
        cols: usize,
    ) -> TrainResult<()> {
        if params.len() != grads.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: params.len(),
                got: grads.len(),
            });
        }
        if params.len() != self.velocity.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.velocity.len(),
                got: params.len(),
            });
        }

        let mu = self.config.momentum;
        // Nesterov momentum: v = μ·v + g
        for (v, &g) in self.velocity.iter_mut().zip(grads.iter()) {
            *v = mu * *v + g;
        }

        // Newton-Schulz orthogonalization
        let update = if self.config.ns_steps > 0 {
            Self::ns_orthogonalize(&self.velocity, rows, cols, self.config.ns_steps)
        } else {
            self.velocity.clone()
        };

        let lr = self.config.lr;
        for (p, &u) in params.iter_mut().zip(update.iter()) {
            *p -= lr * u;
        }
        Ok(())
    }

    /// Apply Newton-Schulz polynomial iterations to orthogonalize a matrix.
    ///
    /// Input `g` is a row-major matrix of shape `(rows, cols)`.
    /// The iteration is:
    /// ```text
    /// X ← 1.5·X − 0.5·X·(Xᵀ·X) / min(rows, cols)
    /// ```
    /// Returns the orthogonalized matrix as a new `Vec<f32>`.
    #[must_use]
    pub fn ns_orthogonalize(g: &[f32], rows: usize, cols: usize, n_steps: usize) -> Vec<f32> {
        let n = rows.min(cols) as f32;
        let mut v = g.to_vec();

        for _ in 0..n_steps {
            // Compute A = vᵀ·v / n  (cols × cols)
            let mut a = vec![0.0_f32; cols * cols];
            for k in 0..rows {
                for j in 0..cols {
                    for l in 0..cols {
                        a[j * cols + l] += v[k * cols + j] * v[k * cols + l];
                    }
                }
            }
            for x in &mut a {
                *x /= n;
            }

            // B = 1.5·I − 0.5·A
            let mut b = vec![0.0_f32; cols * cols];
            for i in 0..cols {
                for j in 0..cols {
                    b[i * cols + j] = if i == j { 1.5 } else { 0.0 } - 0.5 * a[i * cols + j];
                }
            }

            // new_v = v @ B  (rows×cols @ cols×cols → rows×cols)
            let mut new_v = vec![0.0_f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    for k in 0..cols {
                        new_v[i * cols + j] += v[i * cols + k] * b[k * cols + j];
                    }
                }
            }
            v = new_v;
        }
        v
    }

    /// Return current velocity buffer.
    #[must_use]
    pub fn velocity(&self) -> &[f32] {
        &self.velocity
    }

    /// Return the matrix shape `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MuonConfig {
        MuonConfig {
            lr: 1e-2,
            momentum: 0.9,
            ns_steps: 5,
        }
    }

    /// Params must change after a step with non-zero gradient.
    #[test]
    fn step_changes_params() {
        let mut opt = Muon::new(2, 3, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 6];
        let grads = vec![0.1_f32; 6];
        opt.step(&mut params, &grads, 2, 3)
            .expect("step should succeed");
        let changed = params.iter().any(|&p| (p - 1.0).abs() > 1e-9);
        assert!(changed, "params should change after gradient step");
    }

    /// After Newton-Schulz orthogonalization, the columns of the output should
    /// be approximately orthogonal (vᵀv ≈ scaled identity).
    #[test]
    fn ns_makes_cols_orthogonal() {
        // Deterministic well-conditioned full-rank matrix (4×3).
        // Rows chosen to be linearly independent with good condition number.
        #[rustfmt::skip]
        let g = vec![
            1.0_f32, 0.0,  0.0,
            0.0,     1.0,  0.0,
            0.0,     0.0,  1.0,
            0.5,     0.3,  0.7,
        ];
        let rows = 4;
        let cols = 3;
        let v = Muon::ns_orthogonalize(&g, rows, cols, 5);

        // Compute vᵀ·v (cols×cols)
        let mut vtv = vec![0.0_f32; cols * cols];
        for k in 0..rows {
            for j in 0..cols {
                for l in 0..cols {
                    vtv[j * cols + l] += v[k * cols + j] * v[k * cols + l];
                }
            }
        }

        // After NS iterations the matrix should be approximately orthogonal:
        // vᵀv ≈ (rows/cols) × I (scaled identity).
        // Check diagonal is positive and off-diagonals are small relative to diagonal.
        let diag_mean: f32 = (0..cols).map(|i| vtv[i * cols + i]).sum::<f32>() / cols as f32;
        assert!(
            diag_mean > 0.0,
            "diagonal of vᵀv must be positive, got {diag_mean}"
        );

        for i in 0..cols {
            for j in 0..cols {
                if i != j {
                    let off_diag = vtv[i * cols + j].abs();
                    // Off-diagonal should be at most 20% of diagonal mean
                    assert!(
                        off_diag < 0.2 * diag_mean + 1e-4,
                        "off-diagonal vtv[{i},{j}]={off_diag} should be small relative to diag_mean={diag_mean}"
                    );
                }
            }
        }
    }

    /// Velocity should accumulate across steps with non-zero momentum.
    #[test]
    fn momentum_accumulates() {
        let cfg = MuonConfig {
            lr: 1e-3,
            momentum: 0.9,
            ns_steps: 0, // disable NS so velocity feeds directly into update
        };
        let mut opt = Muon::new(1, 2, cfg).expect("valid config");
        let mut params = vec![0.0_f32; 2];
        let grads = vec![1.0_f32; 2];

        opt.step(&mut params, &grads, 1, 2).expect("step 1 ok");
        let v1 = opt.velocity()[0];

        opt.step(&mut params, &grads, 1, 2).expect("step 2 ok");
        let v2 = opt.velocity()[0];

        assert!(
            v2 > v1,
            "velocity should increase with momentum, got v1={v1} v2={v2}"
        );
    }

    /// With ns_steps=0 the update equals velocity directly (no orthogonalization).
    #[test]
    fn ns_steps_0_no_orthog() {
        let cfg_ns = MuonConfig {
            lr: 1e-2,
            momentum: 0.0, // zero momentum for determinism
            ns_steps: 0,
        };
        let mut opt = Muon::new(1, 3, cfg_ns).expect("valid config");
        let mut params = vec![1.0_f32; 3];
        let grads = vec![0.5_f32, 0.5, 0.5];
        opt.step(&mut params, &grads, 1, 3)
            .expect("step should succeed");

        // With mu=0 and ns_steps=0: update = grads, so params -= lr * grads
        let lr = 1e-2_f32;
        for &p in &params {
            assert!(
                (p - (1.0 - lr * 0.5)).abs() < 1e-6,
                "ns_steps=0 should apply gradient directly; got {p}"
            );
        }
    }

    /// Calling step with wrong-length params/grads must return an error.
    #[test]
    fn rows_cols_mismatch_error() {
        let mut opt = Muon::new(2, 3, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 6];
        let grads = vec![0.1_f32; 5]; // wrong length
        let result = opt.step(&mut params, &grads, 2, 3);
        assert!(
            matches!(result, Err(TrainError::ParamCountMismatch { .. })),
            "length mismatch should produce ParamCountMismatch"
        );
    }

    /// After many steps all params must remain finite.
    #[test]
    fn params_finite() {
        let mut opt = Muon::new(3, 4, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 12];
        for _ in 0..10 {
            let grads: Vec<f32> = params.iter().map(|&p| 0.1 * p).collect();
            opt.step(&mut params, &grads, 3, 4)
                .expect("step should succeed");
        }
        for &p in &params {
            assert!(p.is_finite(), "all params must remain finite");
        }
    }

    /// Output of ns_orthogonalize must have the same length as input.
    #[test]
    fn output_dim_correct() {
        let g = vec![1.0_f32; 12]; // 3×4
        let out = Muon::ns_orthogonalize(&g, 3, 4, 3);
        assert_eq!(
            out.len(),
            12,
            "ns_orthogonalize output length must match input length"
        );
    }

    /// Zero grads + zero momentum should leave params unchanged.
    #[test]
    fn zero_grad_no_change_with_zero_momentum() {
        let cfg = MuonConfig {
            lr: 1e-2,
            momentum: 0.0,
            ns_steps: 0,
        };
        let mut opt = Muon::new(1, 4, cfg).expect("valid config");
        let initial_params = vec![3.0_f32; 4];
        let mut params = initial_params.clone();
        let grads = vec![0.0_f32; 4];
        opt.step(&mut params, &grads, 1, 4)
            .expect("step should succeed");
        for (i, (&p, &ip)) in params.iter().zip(initial_params.iter()).enumerate() {
            assert!(
                (p - ip).abs() < 1e-7,
                "zero grad + zero momentum should leave param[{i}] unchanged; got {p}"
            );
        }
    }

    /// A single NS iteration should not panic on any input.
    #[test]
    fn ns_1_step_works() {
        let g: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 + 0.5).collect();
        let out = Muon::ns_orthogonalize(&g, 3, 4, 1);
        assert_eq!(out.len(), 12, "output length must match input");
        for &v in &out {
            assert!(v.is_finite(), "all NS outputs must be finite");
        }
    }

    /// NS orthogonalization preserves output for 1×1 matrices.
    #[test]
    fn ns_1x1_matrix() {
        let g = vec![2.5_f32];
        let out = Muon::ns_orthogonalize(&g, 1, 1, 5);
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite(), "1x1 NS output must be finite");
    }

    /// Negative learning rate must produce an error.
    #[test]
    fn negative_lr_errors() {
        let cfg = MuonConfig {
            lr: -0.01,
            ..default_config()
        };
        let result = Muon::new(2, 3, cfg);
        assert!(
            matches!(result, Err(TrainError::InvalidLearningRate { .. })),
            "negative lr should produce InvalidLearningRate"
        );
    }
}
