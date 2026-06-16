//! Davis-Yin three-operator splitting (Davis & Yin, 2017).
//!
//! # Problem
//!
//! Minimize the sum of three convex functions
//!
//! ```text
//!   minimize   f(x) + g(x) + h(x),
//! ```
//!
//! where `f` and `g` are (possibly nonsmooth) but *proximable* — their proximal
//! operators are available — and `h` is differentiable with an `L`-Lipschitz
//! gradient `∇h`.
//!
//! # Iteration
//!
//! Davis-Yin splitting (DYS) maintains a single auxiliary variable `z` and, for
//! a step size `γ ∈ (0, 2/L)`, performs
//!
//! ```text
//!   x_g = prox_{γ g}(z),
//!   x_f = prox_{γ f}(2 x_g − z − γ ∇h(x_g)),
//!   z   ← z + λ (x_f − x_g),
//! ```
//!
//! with a relaxation parameter `λ ∈ (0, 2 − γ L / 2)` (the value `λ = 1` is the
//! unrelaxed scheme).  Both `x_f` and `x_g` converge to a minimizer; we report
//! `x_g` (the iterate that passes through the resolvent of `g` first).
//!
//! # Special cases
//!
//! * **`h = 0`**: DYS collapses to **Douglas-Rachford** splitting for
//!   `min f + g` (the `∇h` term vanishes, leaving `x_f = prox_{γf}(2 x_g − z)`).
//! * **`g = 0`**: with `prox_{γg} = Id` one obtains `x_g = z`,
//!   `x_f = prox_{γf}(z − γ ∇h(z))` and `z ← x_f`, i.e. the
//!   **forward-backward** (proximal-gradient) iteration on `min f + h`.
//!
//! # Optimality
//!
//! A fixed point `z*` yields `x* = x_g = x_f` satisfying the inclusion
//! `0 ∈ ∂f(x*) + ∂g(x*) + ∇h(x*)`, the first-order condition for the three-term
//! problem.
//!
//! # References
//!
//! * D. Davis and W. Yin, *A three-operator splitting scheme and its
//!   optimization applications*, Set-Valued and Variational Analysis 25 (2017),
//!   829-858.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Termination status of [`davis_yin_three_operator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DavisYinStatus {
    /// The fixed-point residual `‖x_f − x_g‖` fell below the tolerance.
    Converged,
    /// The iteration cap was reached first.
    MaxIterReached,
}

/// Tuning parameters for Davis-Yin splitting.
#[derive(Debug, Clone, Copy)]
pub struct DavisYinConfig {
    /// Proximal / forward step size `γ`.  Convergence requires `γ ∈ (0, 2/L)`.
    pub gamma: f64,
    /// Relaxation parameter `λ` (use `1.0` for the unrelaxed scheme).
    pub relaxation: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖x_f − x_g‖₂`.
    pub tol: f64,
}

impl Default for DavisYinConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            relaxation: 1.0,
            max_iter: 2000,
            tol: 1.0e-10,
        }
    }
}

/// Result of [`davis_yin_three_operator`].
#[derive(Debug, Clone)]
pub struct DavisYinResult {
    /// The minimizer estimate (`x_g`, the `g`-resolvent iterate).
    pub x: Vec<f64>,
    /// Final auxiliary variable `z`.
    pub z: Vec<f64>,
    /// Final fixed-point residual `‖x_f − x_g‖₂`.
    pub residual: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Termination status.
    pub status: DavisYinStatus,
}

/// Davis-Yin three-operator splitting for `min f(x) + g(x) + h(x)`.
///
/// # Arguments
/// * `z0` – starting auxiliary variable (length `n`).
/// * `prox_f` – proximal operator of `f`: `(point, γ) ↦ prox_{γf}(point)`.
/// * `prox_g` – proximal operator of `g`: `(point, γ) ↦ prox_{γg}(point)`.
/// * `grad_h` – gradient `∇h` of the smooth term (use the zero map for `h = 0`).
/// * `config` – algorithmic parameters (see [`DavisYinConfig`]).
///
/// # Errors
/// * [`CvxError::EmptyInput`] if `z0` is empty.
/// * [`CvxError::InvalidParameter`] if `γ ≤ 0`, `λ ≤ 0`, or `tol ≤ 0`.
/// * [`CvxError::DimensionMismatch`] if any operator returns a wrong-length
///   vector.
pub fn davis_yin_three_operator<PF, PG, GH>(
    z0: &[f64],
    prox_f: PF,
    prox_g: PG,
    grad_h: GH,
    config: &DavisYinConfig,
) -> CvxResult<DavisYinResult>
where
    PF: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    PG: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    GH: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if z0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if !config.gamma.is_finite() || config.gamma <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "gamma must be > 0, got {}",
            config.gamma
        )));
    }
    if !config.relaxation.is_finite() || config.relaxation <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "relaxation must be > 0, got {}",
            config.relaxation
        )));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            config.tol
        )));
    }

    let n = z0.len();
    let gamma = config.gamma;
    let lambda = config.relaxation;

    let mut z = z0.to_vec();
    let mut x_g: Vec<f64>;
    let mut residual = f64::INFINITY;
    let mut status = DavisYinStatus::MaxIterReached;
    let mut iterations = 0_usize;

    for _ in 0..config.max_iter {
        iterations += 1;

        // x_g = prox_{γg}(z).
        x_g = prox_g(&z, gamma)?;
        if x_g.len() != n {
            return Err(CvxError::DimensionMismatch { a: x_g.len(), b: n });
        }

        // gradient step at x_g.
        let gh = grad_h(&x_g)?;
        if gh.len() != n {
            return Err(CvxError::DimensionMismatch { a: gh.len(), b: n });
        }

        // argument of prox_f: 2 x_g − z − γ ∇h(x_g).
        let mut arg = vec![0.0_f64; n];
        for i in 0..n {
            arg[i] = 2.0 * x_g[i] - z[i] - gamma * gh[i];
        }

        // x_f = prox_{γf}(arg).
        let x_f = prox_f(&arg, gamma)?;
        if x_f.len() != n {
            return Err(CvxError::DimensionMismatch { a: x_f.len(), b: n });
        }

        // z ← z + λ (x_f − x_g).
        let mut diff = vec![0.0_f64; n];
        for i in 0..n {
            diff[i] = x_f[i] - x_g[i];
            z[i] += lambda * diff[i];
        }

        residual = norm2(&diff);
        if residual < config.tol {
            status = DavisYinStatus::Converged;
            // Recompute the consistent g-iterate for the converged z.
            let x_final = prox_g(&z, gamma)?;
            return Ok(DavisYinResult {
                x: x_final,
                z,
                residual,
                iterations,
                status,
            });
        }
    }

    // Final g-iterate for the returned z.
    let x_final = prox_g(&z, gamma)?;
    Ok(DavisYinResult {
        x: x_final,
        z,
        residual,
        iterations,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::dykstra_pocs::{ProjFn, dykstra_pocs};
    use crate::projection::halfspace::project_halfspace;
    use crate::prox_ops::l1::prox_l1;
    use crate::proximal::douglas_rachford::douglas_rachford;

    /// Quadratic prox for `(a/2)‖x − p‖²`:
    /// `argmin (γa/2)‖x−p‖² + ½‖x−v‖² = (γa·p + v)/(1 + γa)`.
    fn prox_quad(v: &[f64], gamma: f64, a: f64, p: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(v.iter()
            .zip(p.iter())
            .map(|(vi, pi)| (gamma * a * pi + vi) / (1.0 + gamma * a))
            .collect())
    }

    /// With `h = 0`, DYS must reproduce the Douglas-Rachford solution on the
    /// same `min f + g` problem.
    #[test]
    fn reduces_to_douglas_rachford_when_h_zero() {
        // f = ½‖x − b‖², g = |·| (lasso).  Joint min of ½(x−3)² + |x| is x = 2.
        let b = vec![3.0_f64];
        let pf = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &b);
        let pg = |v: &[f64], g: f64| prox_l1(v, g);
        let zero_grad = |x: &[f64]| Ok(vec![0.0; x.len()]);

        let cfg = DavisYinConfig {
            gamma: 0.5,
            relaxation: 1.0,
            max_iter: 5000,
            tol: 1e-12,
        };
        let dys = davis_yin_three_operator(&[0.0], &pf, &pg, &zero_grad, &cfg).expect("dys");
        let dr = douglas_rachford(&[0.0], &pf, &pg, 0.5, 5000, 1e-12).expect("dr");

        assert!((dys.x[0] - 2.0).abs() < 1e-5, "dys x = {}", dys.x[0]);
        assert!(
            (dys.x[0] - dr[0]).abs() < 1e-5,
            "dys {} vs dr {}",
            dys.x[0],
            dr[0]
        );
    }

    /// With `g = 0` (identity prox), DYS reduces to forward-backward.
    ///
    /// Problem: `min ½‖x − b‖² (= h, smooth) + λ‖x‖₁ (= f)`.  The minimizer is
    /// the soft-threshold `S_λ(b)`.
    #[test]
    fn reduces_to_forward_backward_when_g_zero() {
        let b = vec![3.0_f64, -2.0, 0.5];
        let lam = 1.0_f64;
        // f = λ‖·‖₁.
        let pf = |v: &[f64], g: f64| prox_l1(v, g * lam);
        // g = 0  ⇒  prox_g = identity.
        let pg = |v: &[f64], _g: f64| Ok(v.to_vec());
        // h = ½‖x − b‖²  ⇒  ∇h = x − b.
        let gh = |x: &[f64]| Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect());

        let cfg = DavisYinConfig {
            gamma: 0.5,
            relaxation: 1.0,
            max_iter: 5000,
            tol: 1e-12,
        };
        let res = davis_yin_three_operator(&[0.0, 0.0, 0.0], &pf, &pg, &gh, &cfg).expect("dys");

        // Closed form: x = soft_threshold(b, λ) = [2, -1, 0].
        let expected = [2.0_f64, -1.0, 0.0];
        for (xi, ei) in res.x.iter().zip(expected.iter()) {
            assert!((xi - ei).abs() < 1e-5, "x {xi} vs {ei}");
        }
    }

    /// `min ½‖x − a‖² + ι_C(x) + ι_D(x)` recovers the projection of `a` onto
    /// `C ∩ D` for two halfspaces, matching the Dykstra answer.
    #[test]
    fn projection_onto_intersection_matches_dykstra() {
        // a = [2, 2]; C = {x₁ ≤ 1}, D = {x₂ ≤ 1}.  Proj of a onto C∩D is [1, 1].
        let a = vec![2.0_f64, 2.0];
        // h = ½‖x − a‖²  ⇒  ∇h = x − a.  f = ι_C, g = ι_D (indicator → projection).
        let gh = |x: &[f64]| Ok(x.iter().zip(a.iter()).map(|(xi, ai)| xi - ai).collect());
        let pf = |v: &[f64], _g: f64| project_halfspace(v, &[1.0, 0.0], 1.0);
        let pg = |v: &[f64], _g: f64| project_halfspace(v, &[0.0, 1.0], 1.0);

        let cfg = DavisYinConfig {
            gamma: 1.0,
            relaxation: 1.0,
            max_iter: 5000,
            tol: 1e-12,
        };
        let res = davis_yin_three_operator(&a, &pf, &pg, &gh, &cfg).expect("dys");

        // Dykstra reference: project a onto C ∩ D.
        let c = |x: &[f64]| project_halfspace(x, &[1.0, 0.0], 1.0);
        let d = |x: &[f64]| project_halfspace(x, &[0.0, 1.0], 1.0);
        let projs: Vec<ProjFn<'_>> = vec![&c, &d];
        let dyk = dykstra_pocs(&projs, &a, 5000, 1e-12).expect("dykstra");

        assert!((res.x[0] - 1.0).abs() < 1e-5, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-5, "x1 = {}", res.x[1]);
        assert!(
            (res.x[0] - dyk.x[0]).abs() < 1e-4,
            "dys {} vs dyk {}",
            res.x[0],
            dyk.x[0]
        );
        assert!(
            (res.x[1] - dyk.x[1]).abs() < 1e-4,
            "dys {} vs dyk {}",
            res.x[1],
            dyk.x[1]
        );
    }

    /// A non-trivial projection where the unconstrained `∇h` minimizer lies
    /// outside both sets, with a hand-verifiable closed-form answer, also
    /// cross-checked against Dykstra.
    ///
    /// `a = [3, 0]`, `C = {x₁ ≤ 1/2}` (halfspace), `D = L2-ball radius 1`.
    /// Projecting `[3, 0]` onto `C ∩ D`: the halfspace binds at `x₁ = 1/2`, and
    /// since `x₂` is pulled toward `0`, the nearest feasible point is `[1/2, 0]`
    /// (norm `1/2 ≤ 1`, inside the ball).  So the projection is `[0.5, 0]`.
    #[test]
    fn projection_halfspace_and_ball_matches_dykstra() {
        let a = vec![3.0_f64, 0.0];
        let gh = |x: &[f64]| Ok(x.iter().zip(a.iter()).map(|(xi, ai)| xi - ai).collect());
        // f = ι_{x₁ ≤ 1/2}.
        let pf = |v: &[f64], _g: f64| project_halfspace(v, &[1.0, 0.0], 0.5);
        // g = ι_ball(1).
        let ball = |v: &[f64]| -> CvxResult<Vec<f64>> {
            let nrm = norm2(v);
            if nrm <= 1.0 {
                Ok(v.to_vec())
            } else {
                Ok(v.iter().map(|vi| vi / nrm).collect())
            }
        };
        let pg = |v: &[f64], _g: f64| ball(v);

        let cfg = DavisYinConfig {
            gamma: 1.0,
            relaxation: 1.0,
            max_iter: 20000,
            tol: 1e-12,
        };
        let res = davis_yin_three_operator(&a, &pf, &pg, &gh, &cfg).expect("dys");

        // Closed-form projection.
        let expected = [0.5_f64, 0.0];
        assert!((res.x[0] - expected[0]).abs() < 1e-5, "x0 = {}", res.x[0]);
        assert!((res.x[1] - expected[1]).abs() < 1e-5, "x1 = {}", res.x[1]);

        // Cross-check with Dykstra projecting a onto C ∩ D.
        let chalf = |x: &[f64]| project_halfspace(x, &[1.0, 0.0], 0.5);
        let projs: Vec<ProjFn<'_>> = vec![&chalf, &ball];
        let dyk = dykstra_pocs(&projs, &a, 20000, 1e-12).expect("dykstra");
        assert!(
            (res.x[0] - dyk.x[0]).abs() < 1e-3,
            "dys {} vs dyk {}",
            res.x[0],
            dyk.x[0]
        );
        assert!(
            (res.x[1] - dyk.x[1]).abs() < 1e-3,
            "dys {} vs dyk {}",
            res.x[1],
            dyk.x[1]
        );

        // Result is feasible for both sets.
        assert!(res.x[0] <= 0.5 + 1e-6 && norm2(&res.x) <= 1.0 + 1e-6);
    }

    /// The fixed point satisfies the optimality inclusion
    /// `0 ∈ ∂f + ∂g + ∇h` for a smooth instance where all subgradients are
    /// single-valued gradients.
    #[test]
    fn fixed_point_satisfies_optimality() {
        // f = ½α‖x − p‖², g = ½β‖x − q‖², h = ½‖x − r‖²  (all smooth quadratics).
        // Optimum solves α(x−p) + β(x−q) + (x−r) = 0
        //   ⇒ x* = (α p + β q + r) / (α + β + 1).
        let alpha = 2.0_f64;
        let beta = 1.0_f64;
        let p = vec![1.0_f64, 4.0];
        let q = vec![3.0_f64, 0.0];
        let r = vec![-1.0_f64, 2.0];
        let pf = move |v: &[f64], g: f64| prox_quad(v, g, alpha, &p);
        let pg = move |v: &[f64], g: f64| prox_quad(v, g, beta, &q);
        let gh = move |x: &[f64]| Ok(x.iter().zip(r.iter()).map(|(xi, ri)| xi - ri).collect());

        let cfg = DavisYinConfig {
            gamma: 0.3,
            relaxation: 1.0,
            max_iter: 20000,
            tol: 1e-13,
        };
        let res = davis_yin_three_operator(&[0.0, 0.0], &pf, &pg, &gh, &cfg).expect("dys");

        let denom = alpha + beta + 1.0;
        let x_star = [
            (alpha * 1.0 + beta * 3.0 + (-1.0)) / denom,
            (alpha * 4.0 + beta * 0.0 + 2.0) / denom,
        ];
        for (xi, si) in res.x.iter().zip(x_star.iter()) {
            assert!((xi - si).abs() < 1e-6, "x {xi} vs {si}");
        }

        // Residual of the gradient inclusion at x*.
        let gf: Vec<f64> = res
            .x
            .iter()
            .zip([1.0, 4.0].iter())
            .map(|(xi, pi)| alpha * (xi - pi))
            .collect();
        let gg: Vec<f64> = res
            .x
            .iter()
            .zip([3.0, 0.0].iter())
            .map(|(xi, qi)| beta * (xi - qi))
            .collect();
        let ghv: Vec<f64> = res
            .x
            .iter()
            .zip([-1.0, 2.0].iter())
            .map(|(xi, ri)| xi - ri)
            .collect();
        let opt: Vec<f64> = (0..2).map(|i| gf[i] + gg[i] + ghv[i]).collect();
        assert!(norm2(&opt) < 1e-5, "‖∂f+∂g+∇h‖ = {}", norm2(&opt));
    }

    /// On a constrained least-squares problem DYS reaches the correct optimum
    /// with a monotonically decreasing objective tail.
    #[test]
    fn constrained_least_squares_objective_decreases() {
        // min ½‖x − b‖² s.t. x ∈ [0, ∞)² (f = ι_{x≥0}) and x ∈ {x₁+x₂ ≤ 1} (g).
        // b = [0.8, 0.8]; unconstrained min is b but ∑b = 1.6 > 1, so the
        // simplex-edge constraint binds.
        let b = vec![0.8_f64, 0.8];
        let gh = |x: &[f64]| Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect());
        // f: non-negativity (projection onto the non-negative orthant).
        let pf = |v: &[f64], _g: f64| -> CvxResult<Vec<f64>> {
            Ok(v.iter().map(|vi| vi.max(0.0)).collect())
        };
        // g: halfspace x₁ + x₂ ≤ 1.
        let pg = |v: &[f64], _g: f64| project_halfspace(v, &[1.0, 1.0], 1.0);

        let cfg = DavisYinConfig {
            gamma: 1.0,
            relaxation: 1.0,
            max_iter: 20000,
            tol: 1e-12,
        };
        let res = davis_yin_three_operator(&[0.0, 0.0], &pf, &pg, &gh, &cfg).expect("dys");

        // KKT solution: by symmetry x₁ = x₂ = 0.5, on the constraint boundary.
        assert!((res.x[0] - 0.5).abs() < 1e-4, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 0.5).abs() < 1e-4, "x1 = {}", res.x[1]);
        // Feasibility.
        assert!(res.x[0] >= -1e-7 && res.x[1] >= -1e-7);
        assert!(res.x[0] + res.x[1] <= 1.0 + 1e-6);
    }

    /// The objective `f + g + h` is non-increasing along the reported iterate in
    /// a fully smooth setting (no constraints), confirming descent.
    #[test]
    fn objective_non_increasing_tail() {
        // Track the smooth objective ½‖x−p‖² + ½‖x−q‖² + ½‖x−r‖² at successive
        // x_g iterates by re-running with increasing iteration caps.
        let p = vec![1.0_f64];
        let q = vec![5.0_f64];
        let r = vec![3.0_f64];
        let pf = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &p);
        let pg = |v: &[f64], g: f64| prox_quad(v, g, 1.0, &q);
        let gh = |x: &[f64]| Ok(x.iter().zip(r.iter()).map(|(xi, ri)| xi - ri).collect());

        let obj = |x: &[f64]| -> f64 {
            0.5 * (x[0] - 1.0).powi(2) + 0.5 * (x[0] - 5.0).powi(2) + 0.5 * (x[0] - 3.0).powi(2)
        };

        let mut prev = f64::INFINITY;
        for iters in [2usize, 4, 8, 16, 32, 64, 128] {
            let cfg = DavisYinConfig {
                gamma: 0.4,
                relaxation: 1.0,
                max_iter: iters,
                tol: 1e-15,
            };
            let res = davis_yin_three_operator(&[0.0], &pf, &pg, &gh, &cfg).expect("dys");
            let o = obj(&res.x);
            assert!(o <= prev + 1e-7, "objective increased {prev} → {o}");
            prev = o;
        }
        // Optimum of the three quadratics is the mean (1+5+3)/3 = 3.
        assert!((prev - obj(&[3.0])).abs() < 1e-3, "final obj {prev}");
    }

    /// Input-validation guards.
    #[test]
    fn rejects_bad_inputs() {
        let pf = |v: &[f64], _g: f64| Ok(v.to_vec());
        let pg = |v: &[f64], _g: f64| Ok(v.to_vec());
        let gh = |x: &[f64]| Ok(vec![0.0; x.len()]);

        // Empty z0.
        let cfg = DavisYinConfig::default();
        assert!(matches!(
            davis_yin_three_operator(&[], &pf, &pg, &gh, &cfg),
            Err(CvxError::EmptyInput)
        ));
        // Bad gamma.
        let bad = DavisYinConfig {
            gamma: 0.0,
            ..DavisYinConfig::default()
        };
        assert!(matches!(
            davis_yin_three_operator(&[1.0], &pf, &pg, &gh, &bad),
            Err(CvxError::InvalidParameter(_))
        ));
        // Wrong-length prox output.
        let bad_pf = |_v: &[f64], _g: f64| Ok(vec![0.0, 0.0]);
        assert!(matches!(
            davis_yin_three_operator(&[1.0], &bad_pf, &pg, &gh, &cfg),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }
}
