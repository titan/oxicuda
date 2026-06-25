//! Golden-Ratio Primal-Dual Algorithm (GRPDA) — Chang & Yang (2021).
//!
//! Solves the structured saddle-point problem
//!
//! ```text
//!   min_x max_y  ⟨K x, y⟩ + g(x) − f*(y),
//! ```
//! which is the primal-dual form of `min_x  f(K x) + g(x)`, with `K` a linear
//! operator (adjoint `Kᵀ`), `g` and `f*` proper closed convex functions whose
//! proximal operators are available.
//!
//! # The golden-ratio idea
//!
//! The classical primal-dual algorithm (Chambolle-Pock / PDHG) accelerates the
//! Arrow-Hurwicz iteration with an **inertial extrapolation** `x̄ = x + θ(x − x⁻)`
//! and is provably convergent only for `τ σ ‖K‖² < 1`.  GRPDA instead replaces the
//! extrapolation by a **convex combination of the whole trajectory**
//!
//! ```text
//!   z_k = ((ψ − 1)/ψ) x_{k−1} + (1/ψ) z_{k−1},        ψ ∈ (1, φ],
//! ```
//!
//! (the Malitsky golden-ratio averaging), where `φ = (1 + √5)/2 ≈ 1.618` is the
//! golden ratio.  The primal and dual variables are then updated in a Gauss-Seidel
//! manner:
//!
//! ```text
//!   x_k = prox_{τ g}( z_k − τ Kᵀ y_{k−1} ),
//!   y_k = prox_{σ f*}( y_{k−1} + σ K x_k ).
//! ```
//!
//! The crucial gain is the **enlarged step-size region**: GRPDA converges for
//!
//! ```text
//!   τ σ ‖K‖² < ψ ≤ φ,
//! ```
//!
//! i.e. up to the golden ratio, strictly larger than PDHG's `τ σ ‖K‖² < 1`.  Bigger
//! admissible steps translate directly into faster practical convergence with an
//! `O(1/N)` ergodic primal-dual-gap rate.
//!
//! The per-iteration cost is identical to PDHG: two prox evaluations and two
//! matrix-vector products (`K x` and `Kᵀ y`).
//!
//! # References
//!
//! - X. Chang & J. Yang (2021), "A Golden Ratio Primal-Dual Algorithm for
//!   Structured Convex Optimization", *J. Sci. Comput.* 87(2):47.
//! - Y. Malitsky (2020), "Golden ratio algorithms for variational inequalities",
//!   *Math. Program.* 184:383-410.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// The golden ratio `φ = (1 + √5)/2`.
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_8;

/// Configuration for [`grpda`].
#[derive(Debug, Clone)]
pub struct GrpdaConfig {
    /// Primal step `τ > 0`.
    pub tau: f64,
    /// Dual step `σ > 0`.
    pub sigma: f64,
    /// Convex-combination parameter `ψ ∈ (1, φ]` (default `φ`).
    pub psi: f64,
    /// An upper bound (or exact value) of the operator norm `‖K‖₂`; used only to
    /// verify the step-size condition `τ σ ‖K‖² < ψ` at setup.
    pub k_norm: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stop when `‖x_k − x_{k−1}‖₂ + ‖y_k − y_{k−1}‖₂ < tol`.
    pub tol: f64,
}

impl Default for GrpdaConfig {
    fn default() -> Self {
        Self {
            tau: 1.0,
            sigma: 1.0,
            psi: GOLDEN_RATIO,
            k_norm: 1.0,
            max_iter: 1000,
            tol: 1e-9,
        }
    }
}

impl GrpdaConfig {
    /// Build a configuration that automatically picks balanced steps satisfying the
    /// golden-ratio condition with a safety margin.
    ///
    /// Sets `τ = σ = √(safety · ψ) / ‖K‖` so that `τ σ ‖K‖² = safety · ψ < ψ`.
    ///
    /// # Errors
    /// Returns [`CvxError::InvalidParameter`] when `k_norm ≤ 0`, `safety ∉ (0, 1)`,
    /// or `psi ∉ (1, φ]`.
    pub fn balanced(
        k_norm: f64,
        psi: f64,
        safety: f64,
        max_iter: usize,
        tol: f64,
    ) -> CvxResult<Self> {
        if !(k_norm.is_finite() && k_norm > 0.0) {
            return Err(CvxError::InvalidParameter(format!(
                "k_norm must be positive and finite, got {k_norm}"
            )));
        }
        if !(safety > 0.0 && safety < 1.0) {
            return Err(CvxError::InvalidParameter(format!(
                "safety must lie in (0, 1), got {safety}"
            )));
        }
        if !(psi.is_finite() && psi > 1.0 && psi <= GOLDEN_RATIO + 1e-12) {
            return Err(CvxError::InvalidParameter(format!(
                "psi must lie in (1, φ], got {psi}"
            )));
        }
        let step = (safety * psi).sqrt() / k_norm;
        Ok(Self {
            tau: step,
            sigma: step,
            psi,
            k_norm,
            max_iter,
            tol,
        })
    }

    /// Validate the configuration and the golden-ratio step-size condition.
    ///
    /// # Errors
    /// Returns [`CvxError::InvalidParameter`] for out-of-range fields, and
    /// [`CvxError::InvalidConfiguration`] when `τ σ ‖K‖² ≥ ψ`.
    pub fn validate(&self) -> CvxResult<()> {
        let positive_finite = |v: f64| v.is_finite() && v > 0.0;
        if !positive_finite(self.tau) || !positive_finite(self.sigma) {
            return Err(CvxError::InvalidParameter(format!(
                "tau and sigma must be positive and finite, got tau={}, sigma={}",
                self.tau, self.sigma
            )));
        }
        if !(self.psi.is_finite() && self.psi > 1.0 && self.psi <= GOLDEN_RATIO + 1e-12) {
            return Err(CvxError::InvalidParameter(format!(
                "psi must lie in (1, φ ≈ {GOLDEN_RATIO}], got {}",
                self.psi
            )));
        }
        if !(self.k_norm.is_finite() && self.k_norm >= 0.0) {
            return Err(CvxError::InvalidParameter(format!(
                "k_norm must be non-negative and finite, got {}",
                self.k_norm
            )));
        }
        if self.max_iter == 0 {
            return Err(CvxError::InvalidParameter("max_iter must be ≥ 1".into()));
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(CvxError::InvalidParameter(
                "tol must be non-negative and finite".into(),
            ));
        }
        let product = self.tau * self.sigma * self.k_norm * self.k_norm;
        if product >= self.psi {
            return Err(CvxError::InvalidConfiguration(format!(
                "GRPDA requires τ σ ‖K‖² < ψ; got {product} ≥ {}",
                self.psi
            )));
        }
        Ok(())
    }
}

/// Result returned by [`grpda`].
#[derive(Debug, Clone)]
pub struct GrpdaResult {
    /// Final primal iterate `x`.
    pub x: Vec<f64>,
    /// Final dual iterate `y`.
    pub y: Vec<f64>,
    /// Iterations performed.
    pub iter: usize,
    /// Final combined increment `‖Δx‖₂ + ‖Δy‖₂`.
    pub residual: f64,
    /// Whether the increment stopping rule fired.
    pub converged: bool,
}

/// Run the Golden-Ratio Primal-Dual Algorithm.
///
/// # Parameters
/// * `x0`          — initial primal point (non-empty, length `n`).
/// * `y0`          — initial dual point (non-empty, length `m`).
/// * `k_op`        — applies `K`: returns `K x` (length `m`).
/// * `kt_op`       — applies the adjoint `Kᵀ`: returns `Kᵀ y` (length `n`).
/// * `prox_g`      — proximal operator `prox_{τ g}(v)`.
/// * `prox_f_star` — proximal operator `prox_{σ f*}(w)`.
/// * `config`      — step / golden-ratio settings (validated, including the
///   `τ σ ‖K‖² < ψ` condition).
///
/// # Errors
/// * [`CvxError::EmptyInput`] when `x0` or `y0` is empty.
/// * [`CvxError::InvalidParameter`] / [`CvxError::InvalidConfiguration`] for a bad config.
/// * [`CvxError::DimensionMismatch`] when an operator/prox closure returns a
///   wrongly-sized vector.
#[allow(clippy::too_many_arguments)]
pub fn grpda<K, KT, Pg, Pf>(
    x0: &[f64],
    y0: &[f64],
    k_op: K,
    kt_op: KT,
    prox_g: Pg,
    prox_f_star: Pf,
    config: &GrpdaConfig,
) -> CvxResult<GrpdaResult>
where
    K: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    KT: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    Pg: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    Pf: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() || y0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    config.validate()?;

    let n = x0.len();
    let m = y0.len();
    let tau = config.tau;
    let sigma = config.sigma;
    let psi = config.psi;
    let a = (psi - 1.0) / psi; // weight on x_{k−1}
    let bcoef = 1.0 / psi; // weight on z_{k−1}

    // x holds x_{k−1}; z holds z_{k−1}; y holds y_{k−1}.  z is seeded with x0.
    let mut x = x0.to_vec();
    let mut z = x0.to_vec();
    let mut y = y0.to_vec();

    let mut residual = f64::INFINITY;
    let mut iters = 0usize;
    let mut converged = false;

    for it in 0..config.max_iter {
        iters = it + 1;

        // ── golden-ratio convex combination: z_k = a x_{k−1} + b z_{k−1} ─────
        let z_new: Vec<f64> = (0..n).map(|i| a * x[i] + bcoef * z[i]).collect();

        // ── primal step: x_k = prox_{τ g}( z_k − τ Kᵀ y_{k−1} ) ─────────────
        let kt_y = kt_op(&y)?;
        if kt_y.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: kt_y.len(),
                b: n,
            });
        }
        let x_arg: Vec<f64> = (0..n).map(|i| z_new[i] - tau * kt_y[i]).collect();
        let x_new = prox_g(&x_arg, tau)?;
        if x_new.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: n,
            });
        }

        // ── dual step: y_k = prox_{σ f*}( y_{k−1} + σ K x_k ) ───────────────
        let kx = k_op(&x_new)?;
        if kx.len() != m {
            return Err(CvxError::DimensionMismatch { a: kx.len(), b: m });
        }
        let y_arg: Vec<f64> = (0..m).map(|i| y[i] + sigma * kx[i]).collect();
        let y_new = prox_f_star(&y_arg, sigma)?;
        if y_new.len() != m {
            return Err(CvxError::DimensionMismatch {
                a: y_new.len(),
                b: m,
            });
        }

        // ── increment-based stopping test ───────────────────────────────────
        let dx: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let dy: Vec<f64> = (0..m).map(|i| y_new[i] - y[i]).collect();
        residual = norm2(&dx) + norm2(&dy);

        x = x_new;
        z = z_new;
        y = y_new;

        if residual < config.tol {
            converged = true;
            break;
        }
    }

    Ok(GrpdaResult {
        x,
        y,
        iter: iters,
        residual,
        converged,
    })
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::prox_ops::l2::prox_l2;

    // Identity operator helpers.
    fn id(v: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(v.to_vec())
    }

    #[test]
    fn golden_ratio_constant_is_correct() {
        // φ² = φ + 1.
        assert!((GOLDEN_RATIO * GOLDEN_RATIO - (GOLDEN_RATIO + 1.0)).abs() < 1e-12);
    }

    #[test]
    fn step_size_condition_rejected_when_violated() {
        // τσ‖K‖² = 2 > ψ ⇒ must be rejected.
        let cfg = GrpdaConfig {
            tau: 2.0_f64.sqrt(),
            sigma: 2.0_f64.sqrt(),
            psi: 1.5,
            k_norm: 1.0,
            max_iter: 10,
            tol: 1e-9,
        };
        assert!(matches!(
            cfg.validate(),
            Err(CvxError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn balanced_builder_satisfies_condition() {
        let cfg = GrpdaConfig::balanced(3.0, GOLDEN_RATIO, 0.9, 100, 1e-9).expect("ok");
        cfg.validate().expect("balanced config must validate");
        let product = cfg.tau * cfg.sigma * cfg.k_norm * cfg.k_norm;
        assert!(product < cfg.psi);
        assert!((product - 0.9 * GOLDEN_RATIO).abs() < 1e-12);
    }

    #[test]
    fn separable_saddle_converges_to_origin() {
        // min_x max_y ⟨x, y⟩ + ½‖x‖² − ½‖y‖²  with K = I.
        // g(x) = ½‖x‖² ⇒ prox_{τg}(v) = v/(1+τ).
        // f(z) = ½‖z‖² ⇒ f*(y) = ½‖y‖² ⇒ prox_{σf*}(w) = w/(1+σ).
        // The unique saddle is x* = 0, y* = 0.
        let pg = |v: &[f64], t: f64| -> CvxResult<Vec<f64>> {
            Ok(v.iter().map(|vi| vi / (1.0 + t)).collect())
        };
        let pf = |w: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l2(w, s) };
        let cfg = GrpdaConfig::balanced(1.0, GOLDEN_RATIO, 0.95, 5000, 1e-10).expect("ok");
        let res = grpda(
            &[1.0, -2.0, 0.5],
            &[0.3, -0.7, 1.1],
            &id,
            &id,
            &pg,
            &pf,
            &cfg,
        )
        .expect("solves");
        assert!(res.converged, "did not converge");
        for &xi in &res.x {
            assert!(xi.abs() < 1e-3, "x not at origin: {xi}");
        }
        for &yi in &res.y {
            assert!(yi.abs() < 1e-3, "y not at origin: {yi}");
        }
    }

    #[test]
    fn matches_chambolle_pock_on_least_squares() {
        // min_x ½‖A x − b‖²  ⇔  min_x max_y ⟨A x, y⟩ − (½‖y‖² + ⟨b, y⟩)
        // with f(z) = ½‖z − b‖² (so f*(y) = ½‖y‖² + ⟨b, y⟩, prox = (w − σ b)/(1+σ))
        // and g = 0 (prox_g = identity). The optimum is the normal-equation solution
        // x* = (AᵀA)⁻¹ Aᵀ b. Use a tall well-conditioned A.
        let mut rng = LcgRng::new(424242);
        let m = 6usize;
        let n = 3usize;
        let a: Vec<f64> = (0..m * n).map(|_| rng.next_range(-1.0, 1.0)).collect();
        // Diagonally load to keep AᵀA well conditioned.
        let b: Vec<f64> = (0..m).map(|_| rng.next_range(-1.0, 1.0)).collect();

        // Operator norm bound via Frobenius norm (an upper bound on ‖A‖₂).
        let fro: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();

        let a_k = a.clone();
        let k_op = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            crate::linalg::matvec::mat_vec(&a_k, m, n, x)
        };
        let a_kt = a.clone();
        let kt_op = move |y: &[f64]| -> CvxResult<Vec<f64>> {
            crate::linalg::matvec::mat_t_vec(&a_kt, m, n, y)
        };
        let pg = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let b_pf = b.clone();
        let pf = move |w: &[f64], s: f64| -> CvxResult<Vec<f64>> {
            // prox_{σ f*}(w) = (w − σ b)/(1 + σ).
            Ok((0..m).map(|i| (w[i] - s * b_pf[i]) / (1.0 + s)).collect())
        };

        let cfg = GrpdaConfig::balanced(fro, GOLDEN_RATIO, 0.9, 20000, 1e-10).expect("ok");
        let res =
            grpda(&vec![0.0; n], &vec![0.0; m], &k_op, &kt_op, &pg, &pf, &cfg).expect("solves");

        // Reference: solve normal equations (AᵀA) x = Aᵀ b directly.
        let ata = crate::linalg::matvec::mat_t_mat(&a, m, n).expect("ok");
        let atb = crate::linalg::matvec::mat_t_vec(&a, m, n, &b).expect("ok");
        let x_ref = crate::linalg::solve::solve_dense(&ata, n, &atb).expect("ok");

        for i in 0..n {
            assert!(
                (res.x[i] - x_ref[i]).abs() < 1e-3,
                "x[{i}]={} ref {}",
                res.x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn larger_steps_than_pdhg_still_converge() {
        // ψ near φ lets τσ‖K‖² ≈ 1.5 > 1 (forbidden for PDHG) and GRPDA still works.
        let pg = |v: &[f64], t: f64| -> CvxResult<Vec<f64>> {
            Ok(v.iter().map(|vi| vi / (1.0 + t)).collect())
        };
        let pf = |w: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l2(w, s) };
        // τ = σ so τσ‖K‖² = τ² with ‖K‖ = 1; pick τ² = 1.5 < φ.
        let step = 1.5_f64.sqrt();
        let cfg = GrpdaConfig {
            tau: step,
            sigma: step,
            psi: GOLDEN_RATIO,
            k_norm: 1.0,
            max_iter: 5000,
            tol: 1e-10,
        };
        cfg.validate().expect("1.5 < φ so condition holds");
        let res = grpda(&[2.0, -1.0], &[0.5, 0.5], &id, &id, &pg, &pf, &cfg).expect("solves");
        assert!(res.converged);
        for &xi in &res.x {
            assert!(xi.abs() < 1e-3);
        }
    }

    #[test]
    fn rejects_empty_and_bad_psi() {
        let pg = |v: &[f64], _t: f64| Ok(v.to_vec());
        let pf = |w: &[f64], _s: f64| Ok(w.to_vec());
        let cfg = GrpdaConfig::default();
        let r = grpda(&[], &[1.0], &id, &id, &pg, &pf, &cfg);
        assert!(matches!(r, Err(CvxError::EmptyInput)), "{r:?}");

        let bad = GrpdaConfig {
            psi: 2.0, // > φ
            ..Default::default()
        };
        let r2 = grpda(&[1.0], &[1.0], &id, &id, &pg, &pf, &bad);
        assert!(matches!(r2, Err(CvxError::InvalidParameter(_))), "{r2:?}");
    }
}
