//! Sharpness-Aware Minimization (SAM) — Foret et al., 2021.
//!
//! "Sharpness-Aware Minimization for Efficiently Improving Generalization",
//! ICLR 2021.
//!
//! SAM seeks parameters that lie in *flat* loss basins by minimising the
//! worst-case loss within an ε-ball.  Each optimisation iteration is split into
//! two gradient evaluations:
//!
//! 1. **First step** — ascend to the local worst-case point inside the ball:
//!    ```text
//!    ε̂ = ρ · g / (‖g‖₂ + δ)
//!    θ ← θ + ε̂
//!    ```
//! 2. The caller re-evaluates the gradient at the perturbed point `θ + ε̂`.
//! 3. **Second step** — restore the original parameters so the base optimizer
//!    can apply the sharpness-aware gradient:
//!    ```text
//!    θ ← θ − ε̂
//!    ```
//!    The caller then runs its base optimizer (Adam, SGD, …) using the
//!    gradient computed at the perturbed point.
//!
//! This wrapper is base-optimizer-agnostic and operates on flat `f32` slices.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`Sam`].
#[derive(Debug, Clone)]
pub struct SamConfig {
    /// Neighbourhood radius ρ > 0 controlling the size of the ascent step.
    pub rho: f32,
    /// Numerical-stability epsilon added to the gradient norm (default 1e-12).
    pub eps: f32,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            rho: 0.05,
            eps: 1e-12,
        }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// Sharpness-Aware Minimization perturbation tracker.
///
/// Stores the per-step perturbation `ε̂` so it can be added (first step) and
/// removed (second step) around the caller's second gradient evaluation.
pub struct Sam {
    e_w: Vec<f32>,
    config: SamConfig,
}

impl Sam {
    /// Create a SAM wrapper for a parameter vector of length `n_params`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `n_params == 0`.
    /// * [`TrainError::Internal`] if `rho <= 0` or `eps < 0`.
    pub fn new(n_params: usize, config: SamConfig) -> TrainResult<Self> {
        if n_params == 0 {
            return Err(TrainError::EmptyParams);
        }
        if config.rho <= 0.0 || !config.rho.is_finite() {
            return Err(TrainError::Internal {
                msg: format!("SAM rho must be finite and > 0, got {}", config.rho),
            });
        }
        if config.eps < 0.0 {
            return Err(TrainError::Internal {
                msg: format!("SAM eps must be >= 0, got {}", config.eps),
            });
        }
        Ok(Self {
            e_w: vec![0.0; n_params],
            config,
        })
    }

    /// First SAM step: compute `ε̂ = ρ·g/‖g‖` and ascend `params += ε̂`.
    ///
    /// `grads` are the gradients evaluated at the *current* parameters.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if any length disagrees.
    pub fn first_step(&mut self, params: &mut [f32], grads: &[f32]) -> TrainResult<()> {
        if params.len() != self.e_w.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.e_w.len(),
                got: params.len(),
            });
        }
        if grads.len() != self.e_w.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.e_w.len(),
                got: grads.len(),
            });
        }
        // Gradient L2 norm accumulated in f64 for stability.
        let grad_norm = grads
            .iter()
            .map(|&g| (g as f64) * (g as f64))
            .sum::<f64>()
            .sqrt() as f32;
        let scale = self.config.rho / (grad_norm + self.config.eps);
        for ((e, &g), p) in self.e_w.iter_mut().zip(grads.iter()).zip(params.iter_mut()) {
            *e = g * scale;
            *p += *e;
        }
        Ok(())
    }

    /// Second SAM step: restore parameters by subtracting the stored
    /// perturbation, `params -= ε̂`.
    ///
    /// After this call the caller applies its base optimizer using the gradient
    /// evaluated at the perturbed point.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if `params.len()` disagrees.
    pub fn second_step(&self, params: &mut [f32]) -> TrainResult<()> {
        if params.len() != self.e_w.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.e_w.len(),
                got: params.len(),
            });
        }
        for (p, &e) in params.iter_mut().zip(self.e_w.iter()) {
            *p -= e;
        }
        Ok(())
    }

    /// Immutable view of the most recent perturbation `ε̂`.
    #[must_use]
    #[inline]
    pub fn perturbation(&self) -> &[f32] {
        &self.e_w
    }

    /// L2 norm of the most recent perturbation (should be ≈ `rho`).
    #[must_use]
    pub fn perturbation_norm(&self) -> f32 {
        self.e_w
            .iter()
            .map(|&e| (e as f64) * (e as f64))
            .sum::<f64>()
            .sqrt() as f32
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(rho: f32) -> SamConfig {
        SamConfig { rho, eps: 1e-12 }
    }

    #[test]
    fn first_step_perturbs() {
        let mut sam = Sam::new(3, cfg(0.1)).expect("valid");
        let mut params = vec![1.0_f32, 2.0, 3.0];
        let grads = vec![1.0_f32, 0.0, 0.0];
        sam.first_step(&mut params, &grads).expect("valid");
        // Only the first param (non-zero grad) should move.
        assert!(params[0] > 1.0, "param moved along gradient: {}", params[0]);
        assert!((params[1] - 2.0).abs() < 1e-6);
        assert!((params[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn second_step_restores() {
        let mut sam = Sam::new(3, cfg(0.1)).expect("valid");
        let original = vec![1.0_f32, -2.0, 0.5];
        let mut params = original.clone();
        let grads = vec![0.3_f32, -0.4, 0.5];
        sam.first_step(&mut params, &grads).expect("valid");
        sam.second_step(&mut params).expect("valid");
        for (i, (&p, &o)) in params.iter().zip(original.iter()).enumerate() {
            assert!((p - o).abs() < 1e-5, "param[{i}] not restored: {p} vs {o}");
        }
    }

    #[test]
    fn roundtrip_identity() {
        let mut sam = Sam::new(8, cfg(0.5)).expect("valid");
        let original: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut params = original.clone();
        let grads: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        sam.first_step(&mut params, &grads).expect("valid");
        sam.second_step(&mut params).expect("valid");
        for (&p, &o) in params.iter().zip(original.iter()) {
            assert!((p - o).abs() < 1e-5);
        }
    }

    #[test]
    fn perturbation_norm_equals_rho() {
        let rho = 0.25_f32;
        let mut sam = Sam::new(4, cfg(rho)).expect("valid");
        let mut params = vec![0.0_f32; 4];
        let grads = vec![3.0_f32, 4.0, 0.0, 0.0]; // norm 5
        sam.first_step(&mut params, &grads).expect("valid");
        // ‖ε̂‖ = ρ·‖g‖/‖g‖ = ρ.
        assert!(
            (sam.perturbation_norm() - rho).abs() < 1e-5,
            "‖ε̂‖ should be ρ={rho}, got {}",
            sam.perturbation_norm()
        );
    }

    #[test]
    fn rho_scales_perturbation() {
        let mut sam_small = Sam::new(2, cfg(0.1)).expect("valid");
        let mut sam_large = Sam::new(2, cfg(0.5)).expect("valid");
        let grads = vec![1.0_f32, 1.0];
        let mut p1 = vec![0.0_f32; 2];
        let mut p2 = vec![0.0_f32; 2];
        sam_small.first_step(&mut p1, &grads).expect("valid");
        sam_large.first_step(&mut p2, &grads).expect("valid");
        assert!(
            sam_large.perturbation_norm() > sam_small.perturbation_norm(),
            "larger rho should give larger perturbation"
        );
        // Ratio of norms ≈ ratio of rho.
        let ratio = sam_large.perturbation_norm() / sam_small.perturbation_norm();
        assert!((ratio - 5.0).abs() < 1e-3, "ratio should be 5, got {ratio}");
    }

    #[test]
    fn zero_grad_no_perturb() {
        let mut sam = Sam::new(4, cfg(0.1)).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.0_f32; 4];
        sam.first_step(&mut params, &grads).expect("valid");
        // Zero gradient → ε̂ = 0 (no division blow-up thanks to eps).
        for &p in &params {
            assert!((p - 1.0).abs() < 1e-6, "zero grad should not move params");
        }
        assert!(sam.perturbation_norm() < 1e-6);
    }

    #[test]
    fn n_params_zero_error() {
        assert!(matches!(
            Sam::new(0, cfg(0.1)),
            Err(TrainError::EmptyParams)
        ));
    }

    #[test]
    fn invalid_rho_error() {
        assert!(matches!(
            Sam::new(4, cfg(0.0)),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            Sam::new(4, cfg(-0.1)),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn len_mismatch_error() {
        let mut sam = Sam::new(4, cfg(0.1)).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let bad_grads = vec![0.1_f32; 3];
        assert!(matches!(
            sam.first_step(&mut params, &bad_grads),
            Err(TrainError::ParamCountMismatch { .. })
        ));
        let mut bad_params = vec![1.0_f32; 3];
        assert!(matches!(
            sam.second_step(&mut bad_params),
            Err(TrainError::ParamCountMismatch { .. })
        ));
    }

    #[test]
    fn perturbation_finite() {
        let mut sam = Sam::new(6, cfg(0.05)).expect("valid");
        let mut params = vec![1.0_f32; 6];
        let grads = vec![1e-30_f32; 6]; // tiny norm
        sam.first_step(&mut params, &grads).expect("valid");
        assert!(sam.perturbation().iter().all(|v| v.is_finite()));
        assert!(params.iter().all(|v| v.is_finite()));
    }
}
