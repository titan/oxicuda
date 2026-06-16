//! Gradient-enhanced Physics-Informed Neural Networks (gPINN).
//!
//! Yu, Lu, Meng & Karniadakis (2022) "Gradient-enhanced physics-informed neural
//! networks for forward and inverse PDE problems" (CMAME 393, 114823).
//!
//! A standard PINN drives the strong-form PDE residual `r(x) = N[u](x) − f(x)` to
//! zero at a set of collocation points by minimising
//!
//! ```text
//! L_pinn = L_data + L_pde,        L_pde = (1/N) Σ_n r(x_n)² .
//! ```
//!
//! gPINN observes that if `r ≡ 0` on the whole domain then **every spatial
//! derivative of the residual also vanishes**, `∂r/∂x_i ≡ 0`. Enforcing this extra
//! (necessary) condition supplies the optimiser with additional, physically
//! consistent gradient information and empirically yields more accurate solutions
//! with fewer collocation points. The gPINN loss augments the PINN loss with a
//! residual-gradient penalty, one term per input coordinate `i = 1 .. d`:
//!
//! ```text
//! L_gpinn = L_data + L_pde + Σ_{i=1}^{d} w_i · L_g,i ,
//! L_g,i   = (1/N) Σ_n ( ∂r/∂x_i (x_n) )² .
//! ```
//!
//! The residual gradient `∂r/∂x_i` is evaluated by a central finite difference of
//! the user-supplied residual closure with respect to each collocation coordinate,
//! so the formulation works with any field representation (an [`crate::network::mlp::Mlp`],
//! an analytic candidate, a Fourier-feature network, …) whose residual can be
//! sampled pointwise. Setting all `w_i = 0` recovers a plain PINN exactly.

use crate::error::{PinnError, PinnResult};

/// Default finite-difference step used to probe the residual gradient.
const DEFAULT_FD_STEP: f32 = 1e-3;

/// Configuration for a gradient-enhanced PINN loss.
#[derive(Debug, Clone)]
pub struct GPinnConfig {
    /// Spatial / temporal dimensionality `d` of a collocation point.
    pub input_dim: usize,
    /// Per-coordinate weights `w_i` on the residual-gradient terms (length `d`).
    /// `w_i = 0` disables enhancement along coordinate `i`; all-zero ⇒ plain PINN.
    pub grad_weights: Vec<f32>,
    /// Central finite-difference step `h` for `∂r/∂x_i ≈ (r(x+h e_i) − r(x−h e_i))/2h`.
    /// Must be finite and `> 0`.
    pub fd_step: f32,
}

impl GPinnConfig {
    /// Uniform-weight constructor: every coordinate gets the same weight `w`.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `input_dim == 0`.
    pub fn uniform(input_dim: usize, w: f32) -> PinnResult<Self> {
        if input_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        Ok(Self {
            input_dim,
            grad_weights: vec![w; input_dim],
            fd_step: DEFAULT_FD_STEP,
        })
    }
}

/// A decomposed gPINN loss: the PINN part, the gradient part, and their sum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GPinnLossTerms {
    /// Data (supervised) mean-squared loss `L_data`.
    pub data: f32,
    /// PDE residual mean-squared loss `L_pde = (1/N) Σ r²`.
    pub pde: f32,
    /// Weighted residual-gradient penalty `Σ_i w_i · L_g,i`.
    pub grad: f32,
    /// Total gPINN loss `L_data + L_pde + Σ_i w_i L_g,i`.
    pub total: f32,
}

/// Gradient-enhanced PINN loss assembler.
#[derive(Debug, Clone)]
pub struct GPinnLoss {
    config: GPinnConfig,
}

impl GPinnLoss {
    /// Construct a gPINN loss assembler.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `input_dim == 0`.
    /// - [`PinnError::DimensionMismatch`] if `grad_weights.len() != input_dim`.
    /// - [`PinnError::InvalidStepSize`] if `fd_step` is not finite or `<= 0`.
    /// - [`PinnError::InvalidWeight`] if any weight is negative or not finite.
    pub fn new(config: GPinnConfig) -> PinnResult<Self> {
        if config.input_dim == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if config.grad_weights.len() != config.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: config.input_dim,
                got: config.grad_weights.len(),
            });
        }
        if !config.fd_step.is_finite() || config.fd_step <= 0.0 {
            return Err(PinnError::InvalidStepSize { h: config.fd_step });
        }
        for &w in &config.grad_weights {
            if !w.is_finite() || w < 0.0 {
                return Err(PinnError::InvalidWeight { weight: w });
            }
        }
        Ok(Self { config })
    }

    /// Input dimensionality `d`.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Per-coordinate residual-gradient weights.
    #[must_use]
    pub fn grad_weights(&self) -> &[f32] {
        &self.config.grad_weights
    }

    /// Residual gradient `∇r(x)` at a single point via central finite differences.
    ///
    /// Returns `[∂r/∂x_0, …, ∂r/∂x_{d-1}]`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `x.len() != input_dim`.
    /// - [`PinnError::NanEncountered`] if any component is not finite.
    pub fn residual_gradient<F>(&self, x: &[f32], residual_fn: &F) -> PinnResult<Vec<f32>>
    where
        F: Fn(&[f32]) -> f32,
    {
        if x.len() != self.config.input_dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.input_dim,
                got: x.len(),
            });
        }
        let h = self.config.fd_step;
        let mut grad = vec![0.0_f32; self.config.input_dim];
        let mut probe = x.to_vec();
        for i in 0..self.config.input_dim {
            let xi = probe[i];
            probe[i] = xi + h;
            let r_plus = residual_fn(&probe);
            probe[i] = xi - h;
            let r_minus = residual_fn(&probe);
            probe[i] = xi; // restore
            grad[i] = (r_plus - r_minus) / (2.0 * h);
        }
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "gpinn::residual_gradient",
            });
        }
        Ok(grad)
    }

    /// Per-coordinate residual-gradient mean-squared terms `L_g,i`.
    ///
    /// `points` is a flat `[n × d]` collocation array. Returns a length-`d`
    /// vector whose `i`-th entry is `(1/n) Σ_n (∂r/∂x_i(x_n))²`.
    ///
    /// # Errors
    /// - [`PinnError::EmptyCollocationSet`] if `points` is empty or `n == 0`.
    /// - [`PinnError::DimensionMismatch`] if `points.len() != n * input_dim`.
    pub fn grad_terms<F>(&self, points: &[f32], n: usize, residual_fn: &F) -> PinnResult<Vec<f32>>
    where
        F: Fn(&[f32]) -> f32,
    {
        let d = self.config.input_dim;
        if points.is_empty() || n == 0 {
            return Err(PinnError::EmptyCollocationSet);
        }
        if points.len() != n * d {
            return Err(PinnError::DimensionMismatch {
                expected: n * d,
                got: points.len(),
            });
        }
        let mut sum_sq = vec![0.0_f32; d];
        for k in 0..n {
            let x = &points[k * d..(k + 1) * d];
            let g = self.residual_gradient(x, residual_fn)?;
            for i in 0..d {
                sum_sq[i] += g[i] * g[i];
            }
        }
        let inv_n = 1.0 / n as f32;
        for s in &mut sum_sq {
            *s *= inv_n;
        }
        Ok(sum_sq)
    }

    /// Mean-squared PDE residual `L_pde = (1/N) Σ r²` at the collocation points.
    ///
    /// # Errors
    /// - [`PinnError::EmptyCollocationSet`] if `points` is empty or `n == 0`.
    /// - [`PinnError::DimensionMismatch`] if `points.len() != n * input_dim`.
    /// - [`PinnError::NanEncountered`] if the result is not finite.
    pub fn pde_loss<F>(&self, points: &[f32], n: usize, residual_fn: &F) -> PinnResult<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let d = self.config.input_dim;
        if points.is_empty() || n == 0 {
            return Err(PinnError::EmptyCollocationSet);
        }
        if points.len() != n * d {
            return Err(PinnError::DimensionMismatch {
                expected: n * d,
                got: points.len(),
            });
        }
        let mut acc = 0.0_f32;
        for k in 0..n {
            let r = residual_fn(&points[k * d..(k + 1) * d]);
            acc += r * r;
        }
        let mse = acc / n as f32;
        if !mse.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "gpinn::pde_loss",
            });
        }
        Ok(mse)
    }

    /// Assemble the full gPINN loss decomposition.
    ///
    /// `data_loss` is the (already computed) supervised term `L_data` (pass `0.0`
    /// for a purely physics-driven problem). `residual_fn` returns the strong-form
    /// PDE residual at a `d`-dimensional collocation point; it is sampled both for
    /// the value loss and, via finite differences, for the gradient terms.
    ///
    /// # Errors
    /// Propagates errors from [`Self::pde_loss`] / [`Self::grad_terms`]; also
    /// returns [`PinnError::NanEncountered`] if `data_loss` or the total is not finite.
    pub fn loss<F>(
        &self,
        points: &[f32],
        n: usize,
        data_loss: f32,
        residual_fn: &F,
    ) -> PinnResult<GPinnLossTerms>
    where
        F: Fn(&[f32]) -> f32,
    {
        if !data_loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "gpinn::loss(data_loss)",
            });
        }
        let pde = self.pde_loss(points, n, residual_fn)?;
        let grad_terms = self.grad_terms(points, n, residual_fn)?;
        let grad: f32 = self
            .config
            .grad_weights
            .iter()
            .zip(grad_terms.iter())
            .map(|(&w, &g)| w * g)
            .sum();
        let total = data_loss + pde + grad;
        if !total.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "gpinn::loss(total)",
            });
        }
        Ok(GPinnLossTerms {
            data: data_loss,
            pde,
            grad,
            total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // (a) residual-gradient term matches a finite difference of the residual --------
    #[test]
    fn residual_gradient_matches_finite_difference() {
        // r(x, t) = x² + sin(3 t): ∂r/∂x = 2x, ∂r/∂t = 3 cos(3 t).
        let res = |p: &[f32]| p[0] * p[0] + (3.0 * p[1]).sin();
        let g = GPinnLoss::new(
            GPinnConfig::uniform(2, 1.0)
                .expect("2D uniform GPinnConfig with weight 1.0 should be valid"),
        )
        .expect("GPinnLoss with 2D weight-1.0 config should construct successfully");
        let x = [0.4_f32, 0.2];
        let grad = g.residual_gradient(&x, &res).expect(
            "residual gradient of x²+sin(3t) at (0.4, 0.2) should be finite and computable",
        );
        assert!(approx(grad[0], 2.0 * x[0], 1e-2), "∂r/∂x got {}", grad[0]);
        assert!(
            approx(grad[1], 3.0 * (3.0 * x[1]).cos(), 1e-2),
            "∂r/∂t got {}",
            grad[1]
        );
    }

    // (b) gPINN loss ≥ plain PINN loss (adds a non-negative term) --------------------
    #[test]
    fn gpinn_loss_at_least_plain_pinn() {
        // A spatially-varying residual so the gradient term is strictly positive.
        let res = |p: &[f32]| 0.5 * p[0] + 0.3 * p[1] * p[1];
        let cfg = GPinnConfig::uniform(2, 0.7)
            .expect("2D uniform GPinnConfig with weight 0.7 should be valid");
        let g = GPinnLoss::new(cfg)
            .expect("GPinnLoss with 2D weight-0.7 config should construct successfully");
        let n = 9;
        let mut pts = Vec::with_capacity(n * 2);
        for i in 0..3 {
            for j in 0..3 {
                pts.push(i as f32 * 0.25);
                pts.push(j as f32 * 0.25);
            }
        }
        let terms = g.loss(&pts, n, 0.0, &res).expect(
            "gPINN loss for spatially-varying residual on 3×3 grid should compute successfully",
        );
        let plain_pde = g.pde_loss(&pts, n, &res).expect("plain PDE MSE loss for spatially-varying residual on 3×3 grid should compute successfully");
        assert!(terms.grad >= 0.0, "gradient penalty must be non-negative");
        assert!(
            terms.total >= plain_pde - 1e-6,
            "gPINN total {} should be ≥ plain PINN {}",
            terms.total,
            plain_pde
        );
        assert!(
            terms.grad > 1e-6,
            "varying residual ⇒ strictly positive gradient term, got {}",
            terms.grad
        );
    }

    // (c) network == EXACT solution ⇒ residual and its gradient ~0 -------------------
    #[test]
    fn exact_solution_zero_residual_and_gradient_ode() {
        // ODE u'' + u = 0 with the exact solution u(x) = sin(x).
        // Strong residual r(x) = u''(x) + u(x). For u = sin the analytic second
        // derivative is u'' = -sin(x), so r(x) = -sin(x) + sin(x) ≡ 0 as a *smooth*
        // function of x. gPINN's own central difference of this smooth residual must
        // then return ∂r/∂x ≈ 0 everywhere. (We pass the analytic u'' so the test
        // probes the gPINN gradient machinery, not a nested finite-difference stencil.)
        let upp = |x: f32| -x.sin(); // exact u''(x)
        let u = |x: f32| x.sin(); // exact u(x)
        let residual = move |p: &[f32]| {
            let x = p[0];
            upp(x) + u(x) // u'' + u  ≡ 0 for the exact solution
        };
        let g = GPinnLoss::new(
            GPinnConfig::uniform(1, 1.0)
                .expect("1D uniform GPinnConfig with weight 1.0 should be valid"),
        )
        .expect("GPinnLoss with 1D weight-1.0 config should construct successfully");
        for k in 0..10 {
            let x = [0.2 * k as f32];
            let r = residual(&x);
            assert!(
                r.abs() < 1e-5,
                "residual should be ~0 at x={}, got {r}",
                x[0]
            );
            let grad = g.residual_gradient(&x, &residual).expect("residual gradient of sin(x) ODE exact solution should be computable via central FD");
            assert!(
                grad[0].abs() < 1e-3,
                "residual gradient should be ~0 at x={}, got {}",
                x[0],
                grad[0]
            );
        }
    }

    #[test]
    fn exact_solution_zero_residual_and_gradient_exp() {
        // ODE u' = u with the exact solution u(x) = exp(x).
        // r(x) = u'(x) − u(x); for u = exp, u' = exp ⇒ r ≡ 0 and ∂r/∂x ≡ 0.
        let u = |x: f32| x.exp();
        let h = 1e-3_f32;
        let residual = move |p: &[f32]| {
            let x = p[0];
            let up = (u(x + h) - u(x - h)) / (2.0 * h); // u'(x)
            up - u(x)
        };
        let g = GPinnLoss::new(
            GPinnConfig::uniform(1, 2.0)
                .expect("1D uniform GPinnConfig with weight 2.0 should be valid"),
        )
        .expect("GPinnLoss with 1D weight-2.0 config should construct successfully");
        let n = 8;
        let pts: Vec<f32> = (0..n).map(|k| k as f32 * 0.1).collect();
        let pde = g
            .pde_loss(&pts, n, &residual)
            .expect("PDE MSE loss for u'=u exp(x) exact solution should compute successfully");
        let gt = g
            .grad_terms(&pts, n, &residual)
            .expect("gradient terms for u'=u exp(x) exact solution should compute successfully");
        assert!(pde < 1e-3, "PDE residual loss should be ~0, got {pde}");
        assert!(gt[0] < 1e-2, "gradient term should be ~0, got {}", gt[0]);
    }

    // (d) weight w = 0 reduces gPINN to standard PINN exactly ------------------------
    #[test]
    fn zero_weight_reduces_to_plain_pinn() {
        let res = |p: &[f32]| (2.0 * p[0]).cos() + p[1];
        let n = 6;
        let pts: Vec<f32> = (0..n)
            .flat_map(|k| vec![k as f32 * 0.2, k as f32 * 0.1])
            .collect();

        let g0 = GPinnLoss::new(GPinnConfig::uniform(2, 0.0).expect("2D uniform GPinnConfig with zero weights should be valid")).expect("GPinnLoss with 2D zero-weight config (plain PINN reduction) should construct successfully");
        let terms = g0
            .loss(&pts, n, 0.0, &res)
            .expect("zero-weight gPINN loss should compute successfully and match plain PINN");
        let plain = g0.pde_loss(&pts, n, &res).expect(
            "plain PDE loss for zero-weight reference comparison should compute successfully",
        );
        assert!(approx(terms.grad, 0.0, 1e-12), "w=0 ⇒ no gradient term");
        assert!(
            approx(terms.total, plain, 1e-6),
            "w=0 gPINN total {} must equal plain PINN {}",
            terms.total,
            plain
        );
    }

    // (e) gradient term penalizes a spatially-varying residual -----------------------
    #[test]
    fn gradient_term_penalizes_varying_residual() {
        // Constant residual ⇒ zero gradient term; varying residual ⇒ positive term.
        let const_res = |_p: &[f32]| 0.9_f32;
        let vary_res = |p: &[f32]| 0.9 + 1.5 * p[0];
        let cfg = GPinnConfig::uniform(1, 1.0)
            .expect("1D uniform GPinnConfig with weight 1.0 should be valid");
        let g = GPinnLoss::new(cfg).expect("GPinnLoss for constant-vs-varying residual comparison test should construct successfully");
        let n = 5;
        let pts: Vec<f32> = (0..n).map(|k| k as f32 * 0.2).collect();

        let gt_const = g
            .grad_terms(&pts, n, &const_res)
            .expect("gradient terms for constant residual 0.9 should compute successfully");
        let gt_vary = g.grad_terms(&pts, n, &vary_res).expect(
            "gradient terms for linearly-varying residual 0.9+1.5x should compute successfully",
        );
        assert!(
            gt_const[0] < 1e-6,
            "constant residual ⇒ ~zero gradient term, got {}",
            gt_const[0]
        );
        assert!(
            gt_vary[0] > gt_const[0] + 1e-3,
            "varying residual gradient term {} must exceed constant {}",
            gt_vary[0],
            gt_const[0]
        );
        // Slope is 1.5 ⇒ (∂r/∂x)² = 2.25 at every point ⇒ mean term ≈ 2.25.
        assert!(
            approx(gt_vary[0], 2.25, 1e-2),
            "mean squared gradient should be 1.5² = 2.25, got {}",
            gt_vary[0]
        );
    }

    // (f) shapes / finiteness -------------------------------------------------------
    #[test]
    fn loss_terms_shapes_and_finiteness() {
        let res = |p: &[f32]| p[0] * p[1] - 0.3;
        let cfg = GPinnConfig {
            input_dim: 2,
            grad_weights: vec![0.2, 0.5],
            fd_step: 1e-3,
        };
        let g = GPinnLoss::new(cfg).expect("GPinnLoss with non-uniform weights [0.2, 0.5] and 2D bilinear residual should construct successfully");
        let n = 4;
        let pts = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let terms = g.loss(&pts, n, 0.05, &res).expect(
            "full gPINN loss with data_loss=0.05 for bilinear residual should compute successfully",
        );
        assert!(terms.data.is_finite() && terms.pde.is_finite());
        assert!(terms.grad.is_finite() && terms.total.is_finite());
        assert!(approx(terms.data, 0.05, 1e-9), "data loss passes through");
        // total = data + pde + grad
        assert!(
            approx(terms.total, terms.data + terms.pde + terms.grad, 1e-5),
            "total must equal sum of parts"
        );
        let gt = g.grad_terms(&pts, n, &res).expect("gradient terms for bilinear residual should return one finite term per input coordinate");
        assert_eq!(gt.len(), 2, "one gradient term per input coordinate");
    }

    // ── validation guards ──────────────────────────────────────────────────────────
    #[test]
    fn construction_validation() {
        // input_dim == 0
        assert!(matches!(
            GPinnConfig::uniform(0, 1.0),
            Err(PinnError::InvalidLayerWidth)
        ));
        // grad_weights length mismatch
        assert!(matches!(
            GPinnLoss::new(GPinnConfig {
                input_dim: 2,
                grad_weights: vec![1.0],
                fd_step: 1e-3,
            }),
            Err(PinnError::DimensionMismatch { .. })
        ));
        // bad step
        assert!(matches!(
            GPinnLoss::new(GPinnConfig {
                input_dim: 1,
                grad_weights: vec![1.0],
                fd_step: 0.0,
            }),
            Err(PinnError::InvalidStepSize { .. })
        ));
        // negative weight
        assert!(matches!(
            GPinnLoss::new(GPinnConfig {
                input_dim: 1,
                grad_weights: vec![-1.0],
                fd_step: 1e-3,
            }),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn empty_and_mismatched_points_error() {
        let res = |p: &[f32]| p[0];
        let g = GPinnLoss::new(
            GPinnConfig::uniform(1, 1.0)
                .expect("1D uniform GPinnConfig with weight 1.0 should be valid"),
        )
        .expect(
            "GPinnLoss for empty/mismatched-points validation test should construct successfully",
        );
        assert!(matches!(
            g.pde_loss(&[], 0, &res),
            Err(PinnError::EmptyCollocationSet)
        ));
        assert!(matches!(
            g.grad_terms(&[1.0, 2.0, 3.0], 2, &res),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }
}
