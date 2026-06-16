//! PI-DeepONet — physics-informed Deep Operator Network with joint training.
//!
//! Wang, Wang & Perdikaris (2021) "Learning the solution operator of parametric
//! PDEs with physics-informed DeepONets", Science Advances 7(40), eabi8605.
//!
//! A [`DeepONet`] learns a solution operator `G: u ↦ s` mapping an input function
//! `u` (sampled at fixed sensors) to the PDE solution `s`. A *physics-informed*
//! DeepONet additionally enforces that `G(u)` satisfies the governing equation by
//! differentiating the operator output with respect to the query coordinate and
//! penalising the PDE residual at collocation points — so the operator can be
//! trained with few (or no) paired examples.
//!
//! ## Scope of this module
//! This module reuses the crate's [`DeepONet`] as a fixed random feature backbone
//! `b_k(u)·t_k(y)` and trains the **output basis coefficients** `c_k` on top:
//! ```text
//! G(u)(y) = Σ_k c_k · b_k(u) · t_k(y).
//! ```
//! Because `G` is *linear* in `c`, the joint physics + IC + data objective is a
//! convex quadratic in `c` with a closed-form ridge solution (a physics-informed
//! "extreme-learning-machine" readout on DeepONet features) and an exact analytic
//! gradient for first-order descent. Derivatives of `G` with respect to the query
//! coordinate are taken by central finite differences of the (fixed) trunk net.
//!
//! ## Canonical benchmark: the antiderivative operator
//! For the 1-D ODE `ds/dy = u(y)`, `s(0) = 0`, the operator `G` is the
//! antiderivative `G(u)(y) = ∫₀ʸ u(τ) dτ`. The physics residual at query `y` is
//! `r(y) = dG/dy − u(y)`, the initial-condition residual is `G(u)(0)`, and the
//! data residual (where labels exist) is `G(u)(y) − s(y)`.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use crate::neural_op::deeponet::{DeepONet, DeepONetConfig};
use crate::pinn_loss::residual::pde_residual_loss;

/// Configuration for [`PiDeepONet`] training.
#[derive(Debug, Clone)]
pub struct PiDeepONetConfig {
    /// Finite-difference step `h` for the trunk-coordinate derivative `dG/dy`.
    pub fd_step: f32,
    /// Weight of the PDE-residual (physics) loss term.
    pub physics_weight: f32,
    /// Weight of the initial-condition loss term `G(u)(0)²`.
    pub ic_weight: f32,
    /// Weight of the data (supervised) loss term.
    pub data_weight: f32,
    /// Learning rate for the coefficient gradient-descent step.
    pub coeff_lr: f32,
    /// Tikhonov (ridge) regularisation `λ ≥ 0` for the closed-form solve.
    pub ridge_lambda: f32,
}

impl PiDeepONetConfig {
    /// Default configuration: `h = 1e-3`, physics / IC weights `1.0`,
    /// data weight `1.0`, `lr = 1e-2`, `λ = 1e-6`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fd_step: 1e-3,
            physics_weight: 1.0,
            ic_weight: 1.0,
            data_weight: 1.0,
            coeff_lr: 1e-3,
            ridge_lambda: 1e-6,
        }
    }
}

impl Default for PiDeepONetConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Physics-informed DeepONet with a trainable output-coefficient readout.
pub struct PiDeepONet {
    backbone: DeepONet,
    /// Output basis coefficients `c_k`, one per DeepONet basis dimension `p`.
    coeffs: Vec<f32>,
    p: usize,
    d_query: usize,
    config: PiDeepONetConfig,
}

impl PiDeepONet {
    /// Construct a PI-DeepONet from a DeepONet configuration and training config.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if the basis width `p` is zero.
    /// - [`PinnError::EmptyInput`] if `d_query` is zero.
    /// - [`PinnError::InvalidStepSize`] if `fd_step <= 0` or non-finite.
    /// - [`PinnError::InvalidWeight`] if `coeff_lr <= 0` or `ridge_lambda < 0`.
    pub fn new(
        deeponet_config: DeepONetConfig,
        config: PiDeepONetConfig,
        rng: &mut LcgRng,
    ) -> PinnResult<Self> {
        if deeponet_config.p == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if deeponet_config.d_query == 0 {
            return Err(PinnError::EmptyInput);
        }
        if !config.fd_step.is_finite() || config.fd_step <= 0.0 {
            return Err(PinnError::InvalidStepSize { h: config.fd_step });
        }
        if !config.coeff_lr.is_finite() || config.coeff_lr <= 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.coeff_lr,
            });
        }
        if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.ridge_lambda,
            });
        }
        let p = deeponet_config.p;
        let d_query = deeponet_config.d_query;
        let backbone = DeepONet::new(deeponet_config, rng);
        Ok(Self {
            backbone,
            coeffs: vec![1.0_f32; p],
            p,
            d_query,
            config,
        })
    }

    /// Number of output basis coefficients `p`.
    #[must_use]
    pub fn basis_dim(&self) -> usize {
        self.p
    }

    /// Current output coefficients `c`.
    #[must_use]
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Overwrite the output coefficients.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `coeffs.len() != p`.
    pub fn set_coeffs(&mut self, coeffs: &[f32]) -> PinnResult<()> {
        if coeffs.len() != self.p {
            return Err(PinnError::DimensionMismatch {
                expected: self.p,
                got: coeffs.len(),
            });
        }
        self.coeffs.copy_from_slice(coeffs);
        Ok(())
    }

    /// Combine a branch / trunk pair with the current coefficients:
    /// `Σ_k c_k · b_k · t_k`.
    fn combine(&self, branch: &[f32], trunk: &[f32]) -> f32 {
        self.coeffs
            .iter()
            .zip(branch.iter())
            .zip(trunk.iter())
            .map(|((&c, &b), &t)| c * b * t)
            .sum()
    }

    /// Operator value `G(u)(y)`.
    ///
    /// # Errors
    /// Propagates dimension errors from the DeepONet branch / trunk nets.
    pub fn value(&self, func_samples: &[f32], query: &[f32]) -> PinnResult<f32> {
        let b = self.backbone.branch_forward(func_samples)?;
        let t = self.backbone.trunk_forward(query)?;
        Ok(self.combine(&b, &t))
    }

    /// First derivative `dG/dy` along query coordinate 0 (central difference).
    ///
    /// # Errors
    /// Propagates dimension errors from the DeepONet nets.
    pub fn value_dy(&self, func_samples: &[f32], query: &[f32]) -> PinnResult<f32> {
        let b = self.backbone.branch_forward(func_samples)?;
        let h = self.config.fd_step;
        let mut yp = query.to_vec();
        let mut ym = query.to_vec();
        yp[0] += h;
        ym[0] -= h;
        let tp = self.backbone.trunk_forward(&yp)?;
        let tm = self.backbone.trunk_forward(&ym)?;
        let gp = self.combine(&b, &tp);
        let gm = self.combine(&b, &tm);
        Ok((gp - gm) / (2.0 * h))
    }

    /// Second derivative `d²G/dy²` along query coordinate 0 (central difference).
    ///
    /// Useful for second-order operators (e.g. diffusion). Provided alongside the
    /// first-order antiderivative benchmark.
    ///
    /// # Errors
    /// Propagates dimension errors from the DeepONet nets.
    pub fn value_dyy(&self, func_samples: &[f32], query: &[f32]) -> PinnResult<f32> {
        let b = self.backbone.branch_forward(func_samples)?;
        let h = self.config.fd_step;
        let mut yp = query.to_vec();
        let mut ym = query.to_vec();
        yp[0] += h;
        ym[0] -= h;
        let tp = self.backbone.trunk_forward(&yp)?;
        let t0 = self.backbone.trunk_forward(query)?;
        let tm = self.backbone.trunk_forward(&ym)?;
        let gp = self.combine(&b, &tp);
        let g0 = self.combine(&b, &t0);
        let gm = self.combine(&b, &tm);
        Ok((gp - 2.0 * g0 + gm) / (h * h))
    }

    /// Antiderivative physics residual `r(y) = dG/dy − u(y)`.
    ///
    /// `u_at_query` is the value of the input function at the query coordinate
    /// (the operator only sees `u` through its sensor samples, so the caller
    /// supplies `u(y)` directly).
    ///
    /// # Errors
    /// Propagates dimension errors from the DeepONet nets.
    pub fn antiderivative_residual(
        &self,
        func_samples: &[f32],
        query: &[f32],
        u_at_query: f32,
    ) -> PinnResult<f32> {
        Ok(self.value_dy(func_samples, query)? - u_at_query)
    }

    /// Initial-condition value `G(u)(0)`.
    ///
    /// # Errors
    /// Propagates dimension errors from the DeepONet nets.
    pub fn ic_value(&self, func_samples: &[f32]) -> PinnResult<f32> {
        let origin = vec![0.0_f32; self.d_query];
        self.value(func_samples, &origin)
    }

    /// Assemble the per-query value features `φᴳ_k(y) = b_k·t_k(y)` and derivative
    /// features `φᴰ_k(y) = b_k·(t_k(y+h) − t_k(y−h))/(2h)`.
    ///
    /// Both are returned as flat row-major `[n_queries × p]` matrices.
    fn assemble(
        &self,
        branch: &[f32],
        queries: &[f32],
        n_queries: usize,
    ) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        let p = self.p;
        let dq = self.d_query;
        if queries.len() != n_queries * dq {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries * dq,
                got: queries.len(),
            });
        }
        let h = self.config.fd_step;
        let mut phi_g = vec![0.0_f32; n_queries * p];
        let mut phi_d = vec![0.0_f32; n_queries * p];
        for i in 0..n_queries {
            let y = &queries[i * dq..(i + 1) * dq];
            let t0 = self.backbone.trunk_forward(y)?;
            let mut yp = y.to_vec();
            let mut ym = y.to_vec();
            yp[0] += h;
            ym[0] -= h;
            let tp = self.backbone.trunk_forward(&yp)?;
            let tm = self.backbone.trunk_forward(&ym)?;
            for k in 0..p {
                phi_g[i * p + k] = branch[k] * t0[k];
                phi_d[i * p + k] = branch[k] * (tp[k] - tm[k]) / (2.0 * h);
            }
        }
        Ok((phi_g, phi_d))
    }

    /// Dot of the coefficient vector with one row `[i·p .. i·p+p]` of a feature
    /// matrix.
    fn coeff_dot(&self, phi: &[f32], i: usize) -> f32 {
        let p = self.p;
        self.coeffs
            .iter()
            .zip(phi[i * p..i * p + p].iter())
            .map(|(&c, &f)| c * f)
            .sum()
    }

    /// PDE (antiderivative) residuals at the collocation queries.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if the query / `u` lengths disagree.
    pub fn physics_residuals(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        n_queries: usize,
    ) -> PinnResult<Vec<f32>> {
        if u_at_queries.len() != n_queries {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries,
                got: u_at_queries.len(),
            });
        }
        let b = self.backbone.branch_forward(func_samples)?;
        let (_, phi_d) = self.assemble(&b, queries, n_queries)?;
        let res: Vec<f32> = (0..n_queries)
            .map(|i| self.coeff_dot(&phi_d, i) - u_at_queries[i])
            .collect();
        Ok(res)
    }

    /// Mean-squared physics-residual loss.
    ///
    /// # Errors
    /// Propagates from [`PiDeepONet::physics_residuals`] / [`pde_residual_loss`].
    pub fn physics_loss(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        n_queries: usize,
    ) -> PinnResult<f32> {
        let res = self.physics_residuals(func_samples, queries, u_at_queries, n_queries)?;
        pde_residual_loss(&res)
    }

    /// Data (supervised) residuals `G(u)(yᵢ) − sᵢ` at labelled queries.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if the query / target lengths disagree.
    pub fn data_residuals(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        targets: &[f32],
        n_queries: usize,
    ) -> PinnResult<Vec<f32>> {
        if targets.len() != n_queries {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries,
                got: targets.len(),
            });
        }
        let b = self.backbone.branch_forward(func_samples)?;
        let (phi_g, _) = self.assemble(&b, queries, n_queries)?;
        let res: Vec<f32> = (0..n_queries)
            .map(|i| self.coeff_dot(&phi_g, i) - targets[i])
            .collect();
        Ok(res)
    }

    /// Mean-squared data loss.
    ///
    /// # Errors
    /// Propagates from [`PiDeepONet::data_residuals`] / [`pde_residual_loss`].
    pub fn data_loss(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        targets: &[f32],
        n_queries: usize,
    ) -> PinnResult<f32> {
        let res = self.data_residuals(func_samples, queries, targets, n_queries)?;
        pde_residual_loss(&res)
    }

    /// Weighted joint loss
    /// `w_phys·L_phys + w_ic·G(u)(0)² + w_data·L_data`.
    ///
    /// `targets` are optional; when `None` the data term is omitted.
    ///
    /// # Errors
    /// Propagates from the underlying residual computations.
    pub fn joint_loss(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        targets: Option<&[f32]>,
        n_queries: usize,
    ) -> PinnResult<f32> {
        let phys = self.physics_loss(func_samples, queries, u_at_queries, n_queries)?;
        let ic = self.ic_value(func_samples)?;
        let mut total = self.config.physics_weight * phys + self.config.ic_weight * ic * ic;
        if let Some(tgt) = targets {
            let data = self.data_loss(func_samples, queries, tgt, n_queries)?;
            total += self.config.data_weight * data;
        }
        if !total.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "pi_deeponet_joint_loss",
            });
        }
        Ok(total)
    }

    /// Closed-form analytic gradient of the joint loss with respect to the output
    /// coefficients `c` (exact, since the loss is quadratic in `c`).
    ///
    /// # Errors
    /// Propagates from the feature assembly.
    pub fn coeff_gradient(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        targets: Option<&[f32]>,
        n_queries: usize,
    ) -> PinnResult<Vec<f32>> {
        if u_at_queries.len() != n_queries {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries,
                got: u_at_queries.len(),
            });
        }
        let p = self.p;
        let b = self.backbone.branch_forward(func_samples)?;
        let (phi_g, phi_d) = self.assemble(&b, queries, n_queries)?;
        let mut grad = vec![0.0_f32; p];

        // Physics term: w_phys · (2/N) Σ_i r_i φᴰ_i.
        let phys_factor = self.config.physics_weight * 2.0 / n_queries as f32;
        for i in 0..n_queries {
            let r = self.coeff_dot(&phi_d, i) - u_at_queries[i];
            for k in 0..p {
                grad[k] += phys_factor * r * phi_d[i * p + k];
            }
        }

        // Data term: w_data · (2/N) Σ_i d_i φᴳ_i.
        if let Some(tgt) = targets {
            if tgt.len() != n_queries {
                return Err(PinnError::DimensionMismatch {
                    expected: n_queries,
                    got: tgt.len(),
                });
            }
            let data_factor = self.config.data_weight * 2.0 / n_queries as f32;
            for i in 0..n_queries {
                let d = self.coeff_dot(&phi_g, i) - tgt[i];
                for k in 0..p {
                    grad[k] += data_factor * d * phi_g[i * p + k];
                }
            }
        }

        // IC term: w_ic · 2 r₀ φᴳ(0).
        let origin = vec![0.0_f32; self.d_query];
        let (phi_g0, _) = self.assemble(&b, &origin, 1)?;
        let r0 = self.coeff_dot(&phi_g0, 0);
        for k in 0..p {
            grad[k] += self.config.ic_weight * 2.0 * r0 * phi_g0[k];
        }

        if grad.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "pi_deeponet_coeff_gradient",
            });
        }
        Ok(grad)
    }

    /// One gradient-descent step on the output coefficients. Returns the joint
    /// loss *after* the update.
    ///
    /// # Errors
    /// Propagates from gradient / loss computation.
    pub fn train_step(
        &mut self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        targets: Option<&[f32]>,
        n_queries: usize,
    ) -> PinnResult<f32> {
        let grad = self.coeff_gradient(func_samples, queries, u_at_queries, targets, n_queries)?;
        let lr = self.config.coeff_lr;
        for (c, g) in self.coeffs.iter_mut().zip(grad.iter()) {
            *c -= lr * g;
        }
        self.joint_loss(func_samples, queries, u_at_queries, targets, n_queries)
    }

    /// Solve for the output coefficients in closed form by ridge least squares on
    /// the weighted joint objective (a physics-informed ELM readout).
    ///
    /// # Errors
    /// - Propagates from feature assembly.
    /// - [`PinnError::SolverDivergence`] if the normal-equation system is
    ///   singular.
    pub fn fit_least_squares(
        &mut self,
        func_samples: &[f32],
        queries: &[f32],
        u_at_queries: &[f32],
        targets: Option<&[f32]>,
        n_queries: usize,
    ) -> PinnResult<()> {
        if u_at_queries.len() != n_queries {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries,
                got: u_at_queries.len(),
            });
        }
        let p = self.p;
        let b = self.backbone.branch_forward(func_samples)?;
        let (phi_g, phi_d) = self.assemble(&b, queries, n_queries)?;
        let origin = vec![0.0_f32; self.d_query];
        let (phi_g0, _) = self.assemble(&b, &origin, 1)?;

        let mut m = vec![0.0_f32; p * p];
        let mut rhs = vec![0.0_f32; p];

        // Physics: (w_phys / N) Σ φᴰ φᴰᵀ c = (w_phys / N) Σ u_i φᴰ.
        let wphys = self.config.physics_weight / n_queries as f32;
        for i in 0..n_queries {
            let row = &phi_d[i * p..i * p + p];
            for a in 0..p {
                for c in 0..p {
                    m[a * p + c] += wphys * row[a] * row[c];
                }
                rhs[a] += wphys * u_at_queries[i] * row[a];
            }
        }

        // Data: (w_data / N) Σ φᴳ φᴳᵀ c = (w_data / N) Σ s_i φᴳ.
        if let Some(tgt) = targets {
            if tgt.len() != n_queries {
                return Err(PinnError::DimensionMismatch {
                    expected: n_queries,
                    got: tgt.len(),
                });
            }
            let wdata = self.config.data_weight / n_queries as f32;
            for i in 0..n_queries {
                let row = &phi_g[i * p..i * p + p];
                for a in 0..p {
                    for c in 0..p {
                        m[a * p + c] += wdata * row[a] * row[c];
                    }
                    rhs[a] += wdata * tgt[i] * row[a];
                }
            }
        }

        // IC: w_ic φᴳ(0) φᴳ(0)ᵀ c = 0 (target is 0).
        let wic = self.config.ic_weight;
        for a in 0..p {
            for c in 0..p {
                m[a * p + c] += wic * phi_g0[a] * phi_g0[c];
            }
        }

        // Ridge.
        for a in 0..p {
            m[a * p + a] += self.config.ridge_lambda;
        }

        let c = solve_linear_system(&mut m, &mut rhs, p)?;
        self.coeffs = c;
        Ok(())
    }
}

/// Solve the `n×n` linear system `A·x = b` by Gaussian elimination with partial
/// pivoting (`a` and `b` are overwritten).
///
/// # Errors
/// - [`PinnError::SolverDivergence`] if `A` is (numerically) singular.
/// - [`PinnError::NanEncountered`] if the solution is not finite.
fn solve_linear_system(a: &mut [f32], b: &mut [f32], n: usize) -> PinnResult<Vec<f32>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let val = a[r * n + col].abs();
            if val > best {
                best = val;
                pivot = r;
            }
        }
        if best <= 1e-20 {
            return Err(PinnError::SolverDivergence {
                reason: "singular matrix in PI-DeepONet ridge solve",
            });
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f32; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for k in (col + 1)..n {
            s -= a[col * n + k] * x[k];
        }
        x[col] = s / a[col * n + col];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "pi_deeponet_solve_linear_system",
        });
    }
    Ok(x)
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pideeponet(seed: u64) -> PiDeepONet {
        let mut rng = LcgRng::new(seed);
        let don_cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 8,
            d_query: 1,
            p: 12,
            branch_hidden: vec![16],
            trunk_hidden: vec![16],
        };
        PiDeepONet::new(don_cfg, PiDeepONetConfig::new(), &mut rng)
            .expect("PiDeepONet construction with valid config should succeed")
    }

    #[test]
    fn pideeponet_construct() {
        let model = make_pideeponet(1);
        assert_eq!(model.basis_dim(), 12);
        assert_eq!(model.coeffs().len(), 12);
    }

    #[test]
    fn pideeponet_bad_fd_step() {
        let mut rng = LcgRng::new(1);
        let don_cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 4,
            d_query: 1,
            p: 4,
            branch_hidden: vec![],
            trunk_hidden: vec![],
        };
        let mut cfg = PiDeepONetConfig::new();
        cfg.fd_step = 0.0;
        assert!(matches!(
            PiDeepONet::new(don_cfg, cfg, &mut rng),
            Err(PinnError::InvalidStepSize { .. })
        ));
    }

    #[test]
    fn pideeponet_value_finite() {
        let model = make_pideeponet(2);
        let u = vec![0.5_f32; 8];
        let v = model
            .value(&u, &[0.3])
            .expect("operator value should be finite for valid input");
        assert!(v.is_finite());
    }

    #[test]
    fn pideeponet_value_linear_in_coeffs() {
        // G is linear in c (no bias) → scaling c by s scales G by s.
        let mut model = make_pideeponet(3);
        let u = vec![0.4_f32; 8];
        let base = model.coeffs().to_vec();
        let v1 = model
            .value(&u, &[0.25])
            .expect("operator value should be finite for valid input");
        let doubled: Vec<f32> = base.iter().map(|&c| 2.0 * c).collect();
        model
            .set_coeffs(&doubled)
            .expect("set_coeffs should succeed when length matches basis dim");
        let v2 = model
            .value(&u, &[0.25])
            .expect("operator value should be finite after doubling coefficients");
        assert!(
            (v2 - 2.0 * v1).abs() < 1e-4,
            "G should be linear in coeffs: {v2} vs 2·{v1}"
        );
    }

    #[test]
    fn pideeponet_dy_matches_central_difference() {
        // value_dy must equal the central difference of value() at the same h.
        let model = make_pideeponet(4);
        let u = vec![0.3_f32; 8];
        let y = [0.4_f32];
        let h = model.config.fd_step;
        let gp = model
            .value(&u, &[y[0] + h])
            .expect("operator value at y+h should succeed");
        let gm = model
            .value(&u, &[y[0] - h])
            .expect("operator value at y-h should succeed");
        let manual = (gp - gm) / (2.0 * h);
        let method = model
            .value_dy(&u, &y)
            .expect("first derivative value should be finite");
        assert!((manual - method).abs() < 1e-4, "{manual} vs {method}");
    }

    #[test]
    fn pideeponet_dyy_finite() {
        let model = make_pideeponet(5);
        let u = vec![0.3_f32; 8];
        let d2 = model
            .value_dyy(&u, &[0.5])
            .expect("second derivative value should be finite");
        assert!(d2.is_finite());
    }

    #[test]
    fn pideeponet_data_loss_zero_at_self_targets() {
        // Setting targets to the model's own predictions gives zero data loss.
        let model = make_pideeponet(6);
        let u = vec![0.6_f32; 8];
        let ys: Vec<f32> = (0..5).map(|i| i as f32 * 0.2).collect();
        let preds: Vec<f32> = ys
            .iter()
            .map(|&y| {
                model
                    .value(&u, &[y])
                    .expect("operator value should succeed for each query point")
            })
            .collect();
        let loss = model
            .data_loss(&u, &ys, &preds, 5)
            .expect("data loss against self-predictions should be zero");
        assert!(
            loss < 1e-8,
            "self-target data loss should be ~0, got {loss}"
        );
    }

    #[test]
    fn pideeponet_analytic_gradient_matches_numeric() {
        // joint_loss is exactly quadratic in c, so the closed-form gradient must
        // match a central finite-difference gradient of joint_loss.
        let mut model = make_pideeponet(7);
        let u = vec![0.5_f32; 8];
        let ys: Vec<f32> = (0..6).map(|i| 0.1 + i as f32 * 0.15).collect();
        let u_at: Vec<f32> = vec![0.5_f32; 6];
        let targets: Vec<f32> = ys.iter().map(|&y| 0.5 * y).collect();

        let analytic = model
            .coeff_gradient(&u, &ys, &u_at, Some(&targets), 6)
            .expect("analytic gradient computation should succeed");

        let base = model.coeffs().to_vec();
        let delta = 1e-3_f32;
        let mut numeric = vec![0.0_f32; base.len()];
        for k in 0..base.len() {
            let mut cp = base.clone();
            let mut cm = base.clone();
            cp[k] += delta;
            cm[k] -= delta;
            model
                .set_coeffs(&cp)
                .expect("set_coeffs with perturbed coefficients should succeed");
            let lp = model
                .joint_loss(&u, &ys, &u_at, Some(&targets), 6)
                .expect("joint loss at perturbed coefficients should succeed");
            model
                .set_coeffs(&cm)
                .expect("set_coeffs with negatively perturbed coefficients should succeed");
            let lm = model
                .joint_loss(&u, &ys, &u_at, Some(&targets), 6)
                .expect("joint loss at negatively perturbed coefficients should succeed");
            numeric[k] = (lp - lm) / (2.0 * delta);
        }
        model
            .set_coeffs(&base)
            .expect("restoring original coefficients should succeed");

        for (a, n) in analytic.iter().zip(numeric.iter()) {
            assert!(
                (a - n).abs() < 1e-2 + 1e-2 * a.abs(),
                "gradient mismatch: analytic={a} numeric={n}"
            );
        }
    }

    #[test]
    fn pideeponet_train_step_decreases_loss() {
        // Gradient descent with a small step is monotone on the convex quadratic
        // joint objective (the physics features have sharp curvature, so a small
        // learning rate is used).
        let mut rng = LcgRng::new(8);
        let don_cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 8,
            d_query: 1,
            p: 12,
            branch_hidden: vec![16],
            trunk_hidden: vec![16],
        };
        let mut cfg = PiDeepONetConfig::new();
        cfg.coeff_lr = 1e-4;
        let mut model = PiDeepONet::new(don_cfg, cfg, &mut rng)
            .expect("PiDeepONet construction with small learning rate should succeed");

        let u = vec![0.5_f32; 8];
        let ys: Vec<f32> = (0..6).map(|i| 0.1 + i as f32 * 0.15).collect();
        let u_at = vec![0.5_f32; 6];
        let targets: Vec<f32> = ys.iter().map(|&y| 0.5 * y).collect();
        let l0 = model
            .joint_loss(&u, &ys, &u_at, Some(&targets), 6)
            .expect("initial joint loss should be finite and computable");
        let mut last = l0;
        for _ in 0..15 {
            let next = model
                .train_step(&u, &ys, &u_at, Some(&targets), 6)
                .expect("each gradient descent step should succeed");
            assert!(
                next <= last * (1.0 + 1e-4),
                "gradient descent should be monotone: {next} !<= {last}"
            );
            last = next;
        }
        assert!(
            last < l0,
            "joint loss should decrease overall: {last} !< {l0}"
        );
    }

    #[test]
    fn pideeponet_least_squares_zero_gradient() {
        // At the ridge LS optimum the (unregularised) joint-loss gradient ≈ 0.
        let mut model = make_pideeponet(9);
        let u = vec![0.5_f32; 8];
        let ys: Vec<f32> = (0..8).map(|i| 0.05 + i as f32 * 0.12).collect();
        let u_at = vec![0.5_f32; 8];
        let targets: Vec<f32> = ys.iter().map(|&y| 0.5 * y).collect();
        model
            .fit_least_squares(&u, &ys, &u_at, Some(&targets), 8)
            .expect("least-squares fit should converge");
        let grad = model
            .coeff_gradient(&u, &ys, &u_at, Some(&targets), 8)
            .expect("gradient at LS optimum should be computable");
        let gnorm: f32 = grad.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!(
            gnorm < 1e-2,
            "gradient at LS optimum should be ~0, got {gnorm}"
        );
    }

    #[test]
    fn pideeponet_antiderivative_physics_decreases() {
        // u ≡ 1 → antiderivative s(y) = y. Fitting the physics objective should
        // reduce the PDE residual below the arbitrary initial coefficients.
        let mut rng = LcgRng::new(21);
        let don_cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 8,
            d_query: 1,
            p: 16,
            branch_hidden: vec![24],
            trunk_hidden: vec![24],
        };
        let mut cfg = PiDeepONetConfig::new();
        cfg.data_weight = 0.0;
        cfg.ic_weight = 0.0; // physics-only fit for a clean monotone check
        let mut model = PiDeepONet::new(don_cfg, cfg, &mut rng)
            .expect("PiDeepONet construction for antiderivative benchmark should succeed");

        let u = vec![1.0_f32; 8];
        let ys: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
        let u_at = vec![1.0_f32; 12];
        let before = model
            .physics_loss(&u, &ys, &u_at, 12)
            .expect("physics loss before LS fit should be computable");
        model
            .fit_least_squares(&u, &ys, &u_at, None, 12)
            .expect("physics-only LS fit should succeed");
        let after = model
            .physics_loss(&u, &ys, &u_at, 12)
            .expect("physics loss after LS fit should be computable");
        assert!(
            after < before,
            "physics residual should drop after LS fit: {after} !< {before}"
        );
    }

    #[test]
    fn pideeponet_physics_residual_shape_error() {
        let model = make_pideeponet(10);
        let u = vec![0.5_f32; 8];
        let ys = vec![0.1_f32, 0.2, 0.3];
        let u_at = vec![0.5_f32; 2]; // wrong length
        assert!(matches!(
            model.physics_residuals(&u, &ys, &u_at, 3),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn pideeponet_set_coeffs_shape_error() {
        let mut model = make_pideeponet(11);
        assert!(matches!(
            model.set_coeffs(&[1.0, 2.0]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }
}
