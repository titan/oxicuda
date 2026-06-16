//! Conservative Physics-Informed Neural Network loss.
//!
//! Enforces a one-dimensional scalar conservation law
//!
//! ```text
//! ∂_t u(x, t) + ∂_x F(u(x, t)) = 0
//! ```
//!
//! through its INTEGRAL (weak / control-volume) form over a rectangular
//! space–time subdomain `[x_L, x_R] × [t_1, t_2]`. Applying the divergence
//! theorem to the space–time PDE yields
//!
//! ```text
//! 0 = ∫_{x_L}^{x_R} [u(x, t_2) − u(x, t_1)] dx
//!   + ∫_{t_1}^{t_2}    [F(x_R, t)  − F(x_L, t)] dt .
//! ```
//!
//! This loss computes the *integral residual* on the right-hand side for a
//! collection of subdomain boxes using the composite trapezoid rule along
//! each axis, squares each residual, optionally averages, and weights by a
//! scalar weight.
//!
//! Reference: Liu, Z., Cai, W., & Xu, Z.-Q. J. (2020).
//! *Conservative Physics-Informed Neural Networks on Discrete Domains for
//! Conservation Laws: Applications to Forward and Inverse Problems.*
//! Communications in Computational Physics 28(5): 1970–2001.

use crate::error::{PinnError, PinnResult};

// ───────────────────────────── Configuration ─────────────────────────────

/// Configuration for `ConservativeLoss`.
#[derive(Debug, Clone)]
pub struct ConservativeConfig {
    /// Number of evenly-spaced trapezoid nodes used along EACH integration
    /// axis (one axis for the space-integrals, one axis for the time-flux
    /// integrals). Must be `≥ 2`.
    pub n_quadrature: usize,
    /// Non-negative scalar multiplier applied to the summed-squared
    /// subdomain residuals to produce the total loss. Setting `weight = 0`
    /// disables the contribution from this loss term.
    pub weight: f32,
}

impl Default for ConservativeConfig {
    fn default() -> Self {
        Self {
            n_quadrature: 16,
            weight: 1.0,
        }
    }
}

/// A rectangular space–time control volume on which the integral
/// conservation identity is enforced.
#[derive(Debug, Clone, Copy)]
pub struct SubdomainBox {
    /// Left boundary in `x`. Required to be strictly less than `x_r`.
    pub x_l: f32,
    /// Right boundary in `x`.
    pub x_r: f32,
    /// Initial time. Required to be strictly less than `t_2`.
    pub t_1: f32,
    /// Final time.
    pub t_2: f32,
}

// ─────────────────────────────── Loss object ─────────────────────────────

/// Conservative PINN loss.
#[derive(Debug, Clone)]
pub struct ConservativeLoss {
    cfg: ConservativeConfig,
}

impl ConservativeLoss {
    /// Construct a new conservative PINN loss.
    ///
    /// # Errors
    /// - `InvalidGridResolution { n: n_quadrature }` if `n_quadrature < 2`.
    /// - `InvalidWeight { weight }` if `weight < 0` or not finite.
    pub fn new(cfg: ConservativeConfig) -> PinnResult<Self> {
        if cfg.n_quadrature < 2 {
            return Err(PinnError::InvalidGridResolution {
                n: cfg.n_quadrature,
            });
        }
        if !cfg.weight.is_finite() || cfg.weight < 0.0 {
            return Err(PinnError::InvalidWeight { weight: cfg.weight });
        }
        Ok(Self { cfg })
    }

    /// Read-only configuration accessor.
    #[must_use]
    pub fn config(&self) -> &ConservativeConfig {
        &self.cfg
    }

    /// `∫_{x_L}^{x_R} u(x, t) dx` via composite trapezoid quadrature using
    /// `n_quadrature` evenly-spaced nodes in `[x_L, x_R]`.
    ///
    /// # Errors
    /// - `InvalidTimeInterval { t0: x_l, t1: x_r }` if `x_r ≤ x_l` (the
    ///   error variant is reused for spatial bounds).
    /// - `NanEncountered` if `u_at` produces a non-finite value or the
    ///   resulting integral is non-finite.
    pub fn integrate_u_space<U>(&self, x_l: f32, x_r: f32, t: f32, u_at: &U) -> PinnResult<f32>
    where
        U: Fn(f32, f32) -> f32,
    {
        if !x_l.is_finite() || !x_r.is_finite() || x_r <= x_l {
            return Err(PinnError::InvalidTimeInterval { t0: x_l, t1: x_r });
        }
        let n = self.cfg.n_quadrature;
        let h = (x_r - x_l) / ((n - 1) as f32);
        let mut acc = 0.0_f32;
        for i in 0..n {
            let x = x_l + (i as f32) * h;
            let u = u_at(x, t);
            if !u.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "ConservativeLoss::integrate_u_space",
                });
            }
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            acc += w * u;
        }
        let result = acc * h;
        if !result.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "ConservativeLoss::integrate_u_space",
            });
        }
        Ok(result)
    }

    /// `∫_{t_1}^{t_2} F(x, t) dt` via composite trapezoid quadrature using
    /// `n_quadrature` evenly-spaced nodes in `[t_1, t_2]`.
    ///
    /// # Errors
    /// - `InvalidTimeInterval { t0: t_1, t1: t_2 }` if `t_2 ≤ t_1`.
    /// - `NanEncountered` if `flux_at` produces a non-finite value or the
    ///   resulting integral is non-finite.
    pub fn integrate_flux_time<F>(&self, x: f32, t_1: f32, t_2: f32, flux_at: &F) -> PinnResult<f32>
    where
        F: Fn(f32, f32) -> f32,
    {
        if !t_1.is_finite() || !t_2.is_finite() || t_2 <= t_1 {
            return Err(PinnError::InvalidTimeInterval { t0: t_1, t1: t_2 });
        }
        let n = self.cfg.n_quadrature;
        let h = (t_2 - t_1) / ((n - 1) as f32);
        let mut acc = 0.0_f32;
        for i in 0..n {
            let t = t_1 + (i as f32) * h;
            let f = flux_at(x, t);
            if !f.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "ConservativeLoss::integrate_flux_time",
                });
            }
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            acc += w * f;
        }
        let result = acc * h;
        if !result.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "ConservativeLoss::integrate_flux_time",
            });
        }
        Ok(result)
    }

    /// Squared integral conservation residual on a single subdomain box.
    ///
    /// ```text
    /// r(B) = (∫u(·, t_2) − ∫u(·, t_1)) + (∫_t F(x_R, ·) − ∫_t F(x_L, ·))
    /// L(B) = r(B)²
    /// ```
    ///
    /// # Errors
    /// - `InvalidTimeInterval` if `x_r ≤ x_l` or `t_2 ≤ t_1`.
    /// - `NanEncountered` if any of the underlying integrals is non-finite.
    pub fn subdomain_residual<U, F>(
        &self,
        sub: &SubdomainBox,
        u_at: &U,
        flux_at: &F,
    ) -> PinnResult<f32>
    where
        U: Fn(f32, f32) -> f32,
        F: Fn(f32, f32) -> f32,
    {
        if !sub.x_l.is_finite() || !sub.x_r.is_finite() || sub.x_r <= sub.x_l {
            return Err(PinnError::InvalidTimeInterval {
                t0: sub.x_l,
                t1: sub.x_r,
            });
        }
        if !sub.t_1.is_finite() || !sub.t_2.is_finite() || sub.t_2 <= sub.t_1 {
            return Err(PinnError::InvalidTimeInterval {
                t0: sub.t_1,
                t1: sub.t_2,
            });
        }
        let u_t2 = self.integrate_u_space(sub.x_l, sub.x_r, sub.t_2, u_at)?;
        let u_t1 = self.integrate_u_space(sub.x_l, sub.x_r, sub.t_1, u_at)?;
        let f_xr = self.integrate_flux_time(sub.x_r, sub.t_1, sub.t_2, flux_at)?;
        let f_xl = self.integrate_flux_time(sub.x_l, sub.t_1, sub.t_2, flux_at)?;
        let r = (u_t2 - u_t1) + (f_xr - f_xl);
        let sq = r * r;
        if !sq.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "ConservativeLoss::subdomain_residual",
            });
        }
        Ok(sq)
    }

    /// Sum of subdomain squared residuals multiplied by the configured
    /// `weight`.
    ///
    /// ```text
    /// L_total = weight · Σ_B r(B)²
    /// ```
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `subdomains` is empty.
    /// - Errors from `subdomain_residual` propagate.
    pub fn total_loss<U, F>(
        &self,
        subdomains: &[SubdomainBox],
        u_at: &U,
        flux_at: &F,
    ) -> PinnResult<f32>
    where
        U: Fn(f32, f32) -> f32,
        F: Fn(f32, f32) -> f32,
    {
        if subdomains.is_empty() {
            return Err(PinnError::EmptyCollocationSet);
        }
        let mut acc = 0.0_f32;
        for sub in subdomains {
            acc += self.subdomain_residual(sub, u_at, flux_at)?;
        }
        let result = self.cfg.weight * acc;
        if !result.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "ConservativeLoss::total_loss",
            });
        }
        Ok(result)
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_loss() -> ConservativeLoss {
        ConservativeLoss::new(ConservativeConfig::default())
            .expect("ConservativeLoss construction with default config should succeed")
    }

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn conservative_new_default_ok() {
        let l = default_loss();
        assert_eq!(l.config().n_quadrature, 16);
        assert!((l.config().weight - 1.0).abs() < 1e-7);
    }

    #[test]
    fn conservative_new_invalid_n_quadrature_err() {
        let r = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 1,
            weight: 1.0,
        });
        assert!(matches!(r, Err(PinnError::InvalidGridResolution { .. })));
    }

    #[test]
    fn conservative_new_negative_weight_err() {
        let r = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 4,
            weight: -0.1,
        });
        assert!(matches!(r, Err(PinnError::InvalidWeight { .. })));
    }

    #[test]
    fn conservative_new_nan_weight_err() {
        let r = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 4,
            weight: f32::NAN,
        });
        assert!(matches!(r, Err(PinnError::InvalidWeight { .. })));
    }

    // ── integrate_u_space ─────────────────────────────────────────────────────

    #[test]
    fn integrate_u_space_constant_one() {
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 32,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with valid params should succeed");
        let u = |_x: f32, _t: f32| 1.0_f32;
        let v = l
            .integrate_u_space(0.0, 1.0, 0.0, &u)
            .expect("space integration over valid bounds should succeed");
        assert!((v - 1.0).abs() < 1e-5, "∫1 over [0,1] = 1, got {v}");
    }

    #[test]
    fn integrate_u_space_x_linear() {
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 64,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with valid params should succeed");
        let u = |x: f32, _t: f32| x;
        let v = l
            .integrate_u_space(0.0, 1.0, 0.0, &u)
            .expect("space integration over valid bounds should succeed");
        assert!(
            (v - 0.5).abs() < 1e-5,
            "∫x dx over [0,1] = 1/2 (trapezoid is exact for linear), got {v}"
        );
    }

    #[test]
    fn integrate_u_space_invalid_bounds_err() {
        let l = default_loss();
        let u = |_x: f32, _t: f32| 1.0_f32;
        let r = l.integrate_u_space(1.0, 0.5, 0.0, &u);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
        let r2 = l.integrate_u_space(0.5, 0.5, 0.0, &u);
        assert!(matches!(r2, Err(PinnError::InvalidTimeInterval { .. })));
    }

    // ── integrate_flux_time ───────────────────────────────────────────────────

    #[test]
    fn integrate_flux_time_constant() {
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 16,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with valid params should succeed");
        let c = 2.3_f32;
        let f = |_x: f32, _t: f32| c;
        let v = l
            .integrate_flux_time(0.0, 0.0, 1.5, &f)
            .expect("flux time integration over valid bounds should succeed");
        let expected = c * (1.5 - 0.0);
        assert!(
            (v - expected).abs() < 1e-4,
            "∫c dt over [0, 1.5] = c·1.5; got {v} vs {expected}"
        );
    }

    #[test]
    fn integrate_flux_time_invalid_bounds_err() {
        let l = default_loss();
        let f = |_x: f32, _t: f32| 1.0_f32;
        let r = l.integrate_flux_time(0.0, 1.0, 0.5, &f);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
        let r2 = l.integrate_flux_time(0.0, 0.0, 0.0, &f);
        assert!(matches!(r2, Err(PinnError::InvalidTimeInterval { .. })));
    }

    // ── subdomain_residual (conserved solution) ───────────────────────────────

    #[test]
    fn subdomain_residual_advection_conserved_solution_near_zero() {
        // u(x, t) = cos(x − t), F(u) = u → advection ∂_t u + ∂_x F = 0
        // exactly. The integral identity must hold up to quadrature error.
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 128,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with valid params should succeed");
        let u = |x: f32, t: f32| (x - t).cos();
        let f = |x: f32, t: f32| (x - t).cos();
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let r = l
            .subdomain_residual(&sub, &u, &f)
            .expect("conservation law computation should succeed for advection conserved solution");
        assert!(r < 1e-3, "Conserved solution → residual² ≈ 0; got {r}");
    }

    #[test]
    fn subdomain_residual_non_conserved_positive() {
        // u(x, t) = x · t  with flux F = 0
        // Δ∫u(x, t) dx = (t2 − t1) · ∫x dx ≠ 0 over a non-zero interval.
        let l = default_loss();
        let u = |x: f32, t: f32| x * t;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let r = l
            .subdomain_residual(&sub, &u, &f)
            .expect("subdomain residual computation should succeed for non-conserved solution");
        assert!(r > 1e-4, "Non-conserved solution: residual² > 0, got {r}");
    }

    #[test]
    fn subdomain_residual_invalid_bounds_err() {
        let l = default_loss();
        let u = |_x: f32, _t: f32| 0.0_f32;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let bad_x = SubdomainBox {
            x_l: 1.0,
            x_r: 0.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let r = l.subdomain_residual(&bad_x, &u, &f);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));

        let bad_t = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.5,
            t_2: 0.5,
        };
        let r2 = l.subdomain_residual(&bad_t, &u, &f);
        assert!(matches!(r2, Err(PinnError::InvalidTimeInterval { .. })));
    }

    // ── total_loss ────────────────────────────────────────────────────────────

    #[test]
    fn total_loss_sum_of_subdomain_residuals_times_weight() {
        let w = 2.5_f32;
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 32,
            weight: w,
        })
        .expect("ConservativeLoss construction with weight=2.5 and n_quadrature=32 should succeed");
        let u = |x: f32, t: f32| x * t;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let s1 = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let s2 = SubdomainBox {
            x_l: 0.5,
            x_r: 1.5,
            t_1: 0.1,
            t_2: 0.6,
        };
        let r1 = l
            .subdomain_residual(&s1, &u, &f)
            .expect("subdomain residual computation for first box should succeed");
        let r2 = l
            .subdomain_residual(&s2, &u, &f)
            .expect("subdomain residual computation for second box should succeed");
        let tot = l
            .total_loss(&[s1, s2], &u, &f)
            .expect("total loss computation for two subdomains should succeed");
        let expected = w * (r1 + r2);
        assert!(
            (tot - expected).abs() < 1e-3,
            "total_loss = weight·Σr², got {tot} vs {expected}"
        );
    }

    #[test]
    fn total_loss_zero_weight_is_zero() {
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 16,
            weight: 0.0,
        })
        .expect("ConservativeLoss construction with weight=0.0 should succeed");
        let u = |x: f32, t: f32| (x * t).sin();
        let f = |x: f32, _t: f32| x;
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 1.0,
        };
        let tot = l
            .total_loss(&[sub], &u, &f)
            .expect("total loss computation with zero weight should succeed");
        assert!(tot.abs() < 1e-7, "weight = 0 → total = 0, got {tot}");
    }

    #[test]
    fn total_loss_empty_subdomains_err() {
        let l = default_loss();
        let u = |_x: f32, _t: f32| 0.0_f32;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let r = l.total_loss(&[], &u, &f);
        assert!(matches!(r, Err(PinnError::EmptyCollocationSet)));
    }

    #[test]
    fn total_loss_single_subdomain_matches_residual_times_weight() {
        let w = 0.7_f32;
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 32,
            weight: w,
        })
        .expect("ConservativeLoss construction with weight=0.7 and n_quadrature=32 should succeed");
        let u = |x: f32, t: f32| x + t;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let r = l
            .subdomain_residual(&sub, &u, &f)
            .expect("subdomain residual computation should succeed for valid input");
        let tot = l
            .total_loss(&[sub], &u, &f)
            .expect("total loss computation for single subdomain should succeed");
        assert!(
            (tot - w * r).abs() < 1e-4,
            "Single-box total = w · r²; got {tot} vs {}",
            w * r
        );
    }

    // ── determinism ───────────────────────────────────────────────────────────

    #[test]
    fn total_loss_deterministic() {
        let l = default_loss();
        let u = |x: f32, t: f32| (x * 0.7 + t).sin();
        let f = |x: f32, t: f32| (x - t).cos();
        let sub = SubdomainBox {
            x_l: 0.1,
            x_r: 0.9,
            t_1: 0.0,
            t_2: 0.4,
        };
        let a = l
            .total_loss(&[sub], &u, &f)
            .expect("total loss computation should succeed for determinism check (first call)");
        let b = l
            .total_loss(&[sub], &u, &f)
            .expect("total loss computation should succeed for determinism check (second call)");
        assert!((a - b).abs() < 1e-9, "deterministic: {a} vs {b}");
    }

    // ── trapezoid convergence on a smooth conserved solution ──────────────────

    #[test]
    fn subdomain_residual_trapezoid_convergence() {
        let u = |x: f32, t: f32| (x - t).cos();
        let f = |x: f32, t: f32| (x - t).cos();
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.4,
        };
        let coarse = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 4,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with n_quadrature=4 should succeed");
        let fine = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 256,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with n_quadrature=256 should succeed");
        let r_coarse = coarse
            .subdomain_residual(&sub, &u, &f)
            .expect("subdomain residual computation with coarse quadrature should succeed");
        let r_fine = fine
            .subdomain_residual(&sub, &u, &f)
            .expect("subdomain residual computation with fine quadrature should succeed");
        assert!(
            r_fine <= r_coarse + 1e-6,
            "Finer quadrature should not increase residual: fine={r_fine} > coarse={r_coarse}"
        );
        assert!(
            r_fine < 1e-5,
            "Fine quadrature on smooth conserved soln → near-zero residual; got {r_fine}"
        );
    }

    // ── identity flux F = 0 ──────────────────────────────────────────────────

    #[test]
    fn subdomain_residual_zero_flux_equals_delta_u() {
        // With F = 0, residual = Δ∫u; specifically for u(x, t) = t over
        // x ∈ [0, 1], ∫u dx = t and Δ over [0, 0.5] = 0.5 → r² = 0.25.
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 32,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with n_quadrature=32 should succeed");
        let u = |_x: f32, t: f32| t;
        let f = |_x: f32, _t: f32| 0.0_f32;
        let sub = SubdomainBox {
            x_l: 0.0,
            x_r: 1.0,
            t_1: 0.0,
            t_2: 0.5,
        };
        let r = l
            .subdomain_residual(&sub, &u, &f)
            .expect("subdomain residual computation with zero flux should succeed");
        assert!(
            (r - 0.25).abs() < 1e-4,
            "F=0 residual = (Δ∫u)² = (0.5)² = 0.25; got {r}"
        );
    }

    // ── more error cases ──────────────────────────────────────────────────────

    #[test]
    fn integrate_u_space_min_quadrature_two_works() {
        // n=2 → straight-line trapezoid; exact for linear integrands.
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 2,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with minimum n_quadrature=2 should succeed");
        let u = |x: f32, _t: f32| x;
        let v = l
            .integrate_u_space(0.0, 2.0, 0.0, &u)
            .expect("space integration with minimum quadrature nodes should succeed");
        assert!(
            (v - 2.0).abs() < 1e-5,
            "Trapezoid (n=2) on linear u: exact ∫x dx [0,2] = 2; got {v}"
        );
    }

    #[test]
    fn integrate_u_space_quadratic_close_with_many_nodes() {
        // ∫_0^1 x² dx = 1/3; trapezoid on a smooth function converges as O(h²).
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 200,
            weight: 1.0,
        })
        .expect("ConservativeLoss construction with n_quadrature=200 should succeed");
        let u = |x: f32, _t: f32| x * x;
        let v = l
            .integrate_u_space(0.0, 1.0, 0.0, &u)
            .expect("space integration for quadratic function should succeed");
        assert!((v - 1.0_f32 / 3.0).abs() < 1e-3, "∫x² dx ≈ 1/3; got {v}");
    }

    #[test]
    fn config_accessor_round_trip() {
        let l = ConservativeLoss::new(ConservativeConfig {
            n_quadrature: 17,
            weight: 0.42,
        })
        .expect(
            "ConservativeLoss construction with n_quadrature=17 and weight=0.42 should succeed",
        );
        let c = l.config();
        assert_eq!(c.n_quadrature, 17);
        assert!((c.weight - 0.42).abs() < 1e-7);
    }
}
