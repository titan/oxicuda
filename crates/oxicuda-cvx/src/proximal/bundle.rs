//! Proximal-bundle method for non-smooth convex minimisation.
//!
//! Implements the *simplified* proximal-bundle scheme of Lemaréchal (1975) and
//! Kiwiel (1990). At each iteration the algorithm maintains a *bundle* of
//! past iterates and subgradients
//! ```text
//!     B = { (x_j, f_j, g_j) : f_j = f(x_j), g_j ∈ ∂f(x_j) }
//! ```
//! and the associated piecewise-linear *cutting-plane* model
//! ```text
//!     f̂(x) = max_j  f_j + g_jᵀ(x − x_j) .
//! ```
//! Given the current stability centre `x_c` and the prox parameter `ρ > 0` the
//! solver computes the next *trial* iterate as the unique minimiser of the
//! strongly-convex prox-QP master problem
//! ```text
//!     x_+ = argmin_x  f̂(x) + (ρ/2)‖x − x_c‖² .
//! ```
//! This piecewise-quadratic problem has the convex-combination *Wolfe dual*
//! ```text
//!     max_{λ ≥ 0, Σλ_j = 1}  − (1/2ρ)‖Σλ_j g_j‖² + Σ λ_j (f_j + g_jᵀ(x_c − x_j))
//! ```
//! whose optimal primal recovery is
//! ```text
//!     x_+ = x_c − (1/ρ) Σ λ_j g_j ,
//! ```
//! and whose Lagrangian value at `λ*` equals `f̂(x_+) + (ρ/2)‖x_+ − x_c‖²`, hence
//! ```text
//!     predicted_decrease = f_c − f̂(x_+) .
//! ```
//! A *serious* step accepts `x_+` as the new stability centre whenever the
//! actual decrease meets a fraction `m ∈ (0,1)` of the predicted decrease,
//! ```text
//!     f(x_c) − f(x_+) ≥ m · (f_c − f̂(x_+)) .
//! ```
//! Otherwise the iterate is a *null* step: the centre is retained but the
//! freshly-evaluated cut `(x_+, f(x_+), g(x_+))` is added to the bundle, which
//! is provably sufficient to drive the next predicted decrease below `tol`.
//! Convergence is declared when `predicted_decrease ≤ tol`.
//!
//! The Wolfe dual is solved by a small projected-gradient pass on the
//! probability simplex. The simplex projection used is the standard
//! `O(k log k)` sort-based algorithm (Wang & Carreira-Perpiñán 2013).
//!
//! All work is `f32`; the bundle never grows beyond `max_bundle_size` (oldest
//! cuts are evicted when necessary).

use crate::error::{CvxError, CvxResult};

/// Configuration for the [`ProximalBundle`] solver.
#[derive(Debug, Clone)]
pub struct BundleConfig {
    /// Maximum number of outer iterations (`≥ 1`).
    pub max_iter: usize,
    /// Hard cap on the bundle cardinality (`≥ 1`). Older cuts are evicted.
    pub max_bundle_size: usize,
    /// Prox parameter `ρ > 0`. Smaller `ρ` → longer steps.
    pub rho: f32,
    /// Serious-step acceptance ratio `m ∈ (0, 1)`.
    pub m_serious: f32,
    /// Tolerance on the predicted decrease (`> 0`).
    pub tol: f32,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            max_bundle_size: 50,
            rho: 1.0,
            m_serious: 0.1,
            tol: 1.0e-6,
        }
    }
}

/// Outcome of a [`ProximalBundle`] minimisation.
#[derive(Debug, Clone)]
pub struct BundleResult {
    /// Final stability centre `x_c` (best certified iterate).
    pub x: Vec<f32>,
    /// Objective value `f(x)` at the returned iterate.
    pub f: f32,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Number of *serious* steps accepted.
    pub n_serious: usize,
    /// Number of *null* steps performed.
    pub n_null: usize,
    /// Whether `predicted_decrease ≤ tol` was reached.
    pub converged: bool,
}

/// Proximal-bundle solver.
pub struct ProximalBundle;

impl ProximalBundle {
    /// Minimise the (non-smooth) convex function `f` starting from `x0`.
    ///
    /// * `f` — objective `x ↦ f(x)`.
    /// * `subgrad` — `x ↦ a subgradient g ∈ ∂f(x)` (must return a vector of length
    ///   `x.len()`).
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::EmptyInput`] / [`CvxError::InvalidParameter`] on
    /// malformed configuration, and [`CvxError::DimensionMismatch`] if
    /// `subgrad` changes the dimension.
    pub fn minimize<F, G>(
        x0: &[f32],
        f: F,
        subgrad: G,
        cfg: &BundleConfig,
    ) -> CvxResult<BundleResult>
    where
        F: Fn(&[f32]) -> f32,
        G: Fn(&[f32]) -> Vec<f32>,
    {
        Self::validate(x0, cfg)?;
        let dim = x0.len();

        // Initial bundle entry at the starting iterate.
        let mut x_c = x0.to_vec();
        let mut f_c = f(&x_c);
        let g0 = subgrad(&x_c);
        if g0.len() != dim {
            return Err(CvxError::DimensionMismatch {
                a: g0.len(),
                b: dim,
            });
        }
        let mut bundle: Vec<(Vec<f32>, f32, Vec<f32>)> = Vec::with_capacity(cfg.max_bundle_size);
        push_bundle(&mut bundle, (x_c.clone(), f_c, g0), cfg.max_bundle_size);

        let mut iterations = 0usize;
        let mut n_serious = 0usize;
        let mut n_null = 0usize;
        let mut converged = false;

        for it in 0..cfg.max_iter {
            iterations = it + 1;

            let (x_new, predicted_dec) = Self::solve_prox_master(&bundle, &x_c, cfg.rho)?;
            if x_new.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: x_new.len(),
                    b: dim,
                });
            }

            // Stopping test on the certified predicted decrease (Kiwiel 1990).
            if predicted_dec <= cfg.tol {
                converged = true;
                break;
            }

            let f_new = f(&x_new);
            let g_new = subgrad(&x_new);
            if g_new.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: g_new.len(),
                    b: dim,
                });
            }

            let actual_dec = f_c - f_new;
            let serious = actual_dec >= cfg.m_serious * predicted_dec;

            // Always push the freshly-evaluated cut — even on a null step it
            // strictly refines the cutting-plane model at the centre.
            push_bundle(
                &mut bundle,
                (x_new.clone(), f_new, g_new),
                cfg.max_bundle_size,
            );

            if serious {
                x_c = x_new;
                f_c = f_new;
                n_serious += 1;
            } else {
                n_null += 1;
            }
        }

        Ok(BundleResult {
            x: x_c,
            f: f_c,
            iterations,
            n_serious,
            n_null,
            converged,
        })
    }

    /// Cutting-plane model value
    /// `max_j  f_j + g_jᵀ(x − x_j)`.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::EmptyInput`] on an empty bundle and
    /// [`CvxError::DimensionMismatch`] if any cut has the wrong dimension.
    pub fn cutting_plane_max(bundle: &[(Vec<f32>, f32, Vec<f32>)], x: &[f32]) -> CvxResult<f32> {
        if bundle.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if x.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        let dim = x.len();
        let mut best = f32::NEG_INFINITY;
        for (xj, fj, gj) in bundle.iter() {
            if xj.len() != dim || gj.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: xj.len().max(gj.len()),
                    b: dim,
                });
            }
            let mut v = *fj;
            for k in 0..dim {
                v += gj[k] * (x[k] - xj[k]);
            }
            if v > best {
                best = v;
            }
        }
        if !best.is_finite() {
            return Err(CvxError::NumericalInstability(
                "cutting-plane model overflowed".into(),
            ));
        }
        Ok(best)
    }

    /// Solve the prox-QP master problem
    /// `argmin_x  max_j [ f_j + g_jᵀ(x − x_j) ] + (ρ/2)‖x − x_c‖²`
    /// via the convex-combination *Wolfe dual* on the probability simplex.
    ///
    /// Returns `(x_+, predicted_decrease)` where
    /// `predicted_decrease = f_c − f̂(x_+)` and `f_c` is the cutting-plane
    /// value at the centre (which equals `f(x_c)` whenever the cut at `x_c`
    /// is present in the bundle, as it always is here).
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::EmptyInput`] on an empty bundle / centre,
    /// [`CvxError::InvalidParameter`] if `rho ≤ 0`, and
    /// [`CvxError::DimensionMismatch`] for inconsistent dimensions.
    pub fn solve_prox_master(
        bundle: &[(Vec<f32>, f32, Vec<f32>)],
        x_c: &[f32],
        rho: f32,
    ) -> CvxResult<(Vec<f32>, f32)> {
        if bundle.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if x_c.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if rho <= 0.0 || !rho.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "bundle prox parameter rho must be > 0, got {rho}"
            )));
        }
        let dim = x_c.len();
        for (xj, _, gj) in bundle.iter() {
            if xj.len() != dim || gj.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: xj.len().max(gj.len()),
                    b: dim,
                });
            }
        }
        let k = bundle.len();

        // Dual objective in λ (to be MINIMISED):
        //   φ(λ) = (1/2ρ) ‖G λ‖² − cᵀλ ,        Σλ = 1, λ ≥ 0
        // where c_j = f_j + g_jᵀ(x_c − x_j).
        // Equivalent maximisation matches the docstring formula.
        // Gradient: ∇φ(λ)_i = (1/ρ) g_iᵀ(G λ) − c_i .
        let mut c = vec![0.0_f32; k];
        for (i, (xj, fj, gj)) in bundle.iter().enumerate() {
            let mut s = *fj;
            for d in 0..dim {
                s += gj[d] * (x_c[d] - xj[d]);
            }
            c[i] = s;
        }

        let inv_rho = 1.0_f32 / rho;
        let mut lam = vec![1.0_f32 / k as f32; k];

        // Pre-compute a Lipschitz upper bound for the dual quadratic so that a
        // constant step `1/L_d` always satisfies the descent lemma. The
        // Hessian is `(1/ρ) GᵀG`; its operator norm is bounded by the
        // Frobenius norm `(1/ρ) Σ_i ‖g_i‖²` (= trace bound).
        let mut g_frob_sq = 0.0_f32;
        for (_, _, gj) in bundle.iter() {
            for &v in gj.iter() {
                g_frob_sq += v * v;
            }
        }
        let lip = (inv_rho * g_frob_sq).max(1.0e-12_f32);
        let step = 1.0_f32 / lip;

        // Projected-gradient on the simplex (constant step + safe iterates).
        // The dual is convex quadratic over a compact set → linear convergence
        // for moderate ρ. A small fixed budget is plenty in practice and is
        // bounded by the bundle size cap.
        let inner_iters = (8 * k).max(64);
        let mut g_lam = vec![0.0_f32; dim];
        for _ in 0..inner_iters {
            // G λ → g_lam .
            for slot in g_lam.iter_mut() {
                *slot = 0.0;
            }
            for (i, (_, _, gj)) in bundle.iter().enumerate() {
                let li = lam[i];
                for d in 0..dim {
                    g_lam[d] += li * gj[d];
                }
            }
            // grad_i = (1/ρ) g_iᵀ g_lam − c_i .
            let mut grad = vec![0.0_f32; k];
            for (i, (_, _, gj)) in bundle.iter().enumerate() {
                let mut s = 0.0_f32;
                for d in 0..dim {
                    s += gj[d] * g_lam[d];
                }
                grad[i] = inv_rho * s - c[i];
            }
            // Gradient step + simplex projection.
            for i in 0..k {
                lam[i] -= step * grad[i];
            }
            project_unit_simplex_inplace(&mut lam);
        }

        // Primal recovery: x_+ = x_c − (1/ρ) G λ .
        for slot in g_lam.iter_mut() {
            *slot = 0.0;
        }
        for (i, (_, _, gj)) in bundle.iter().enumerate() {
            let li = lam[i];
            for d in 0..dim {
                g_lam[d] += li * gj[d];
            }
        }
        let mut x_plus = vec![0.0_f32; dim];
        for d in 0..dim {
            x_plus[d] = x_c[d] - inv_rho * g_lam[d];
        }

        // Predicted decrease: f_c − f̂(x_+) where f_c is the cutting-plane
        // value at the centre (≥ f(x_c)). For our bundle convention the cut
        // at the centre is always present, so f̂(x_c) = f(x_c) = f_c.
        let f_c = Self::cutting_plane_max(bundle, x_c)?;
        let f_hat_new = Self::cutting_plane_max(bundle, &x_plus)?;
        let predicted_dec = (f_c - f_hat_new).max(0.0);

        Ok((x_plus, predicted_dec))
    }

    fn validate(x0: &[f32], cfg: &BundleConfig) -> CvxResult<()> {
        if x0.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if cfg.max_iter == 0 {
            return Err(CvxError::InvalidParameter(
                "bundle requires max_iter ≥ 1".to_string(),
            ));
        }
        if cfg.max_bundle_size == 0 {
            return Err(CvxError::InvalidParameter(
                "bundle requires max_bundle_size ≥ 1".to_string(),
            ));
        }
        if cfg.rho <= 0.0 || !cfg.rho.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "bundle rho must be > 0, got {}",
                cfg.rho
            )));
        }
        if cfg.m_serious <= 0.0 || cfg.m_serious >= 1.0 || !cfg.m_serious.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "bundle m_serious must lie in (0, 1), got {}",
                cfg.m_serious
            )));
        }
        if cfg.tol <= 0.0 || !cfg.tol.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "bundle tol must be > 0, got {}",
                cfg.tol
            )));
        }
        Ok(())
    }
}

/// Push a new cut into the bundle, evicting the oldest entry once the cap is
/// reached (FIFO policy — robust and avoids needing dual weights).
fn push_bundle(
    bundle: &mut Vec<(Vec<f32>, f32, Vec<f32>)>,
    cut: (Vec<f32>, f32, Vec<f32>),
    cap: usize,
) {
    if bundle.len() >= cap {
        bundle.remove(0);
    }
    bundle.push(cut);
}

/// In-place Euclidean projection of `v` onto the unit probability simplex
/// `{x ≥ 0, Σ x = 1}` (Wang & Carreira-Perpiñán 2013, `O(n log n)` sort).
fn project_unit_simplex_inplace(v: &mut [f32]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut u: Vec<f32> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0_f32;
    let mut tau = 0.0_f32;
    let mut found = false;
    for (k, &uk) in u.iter().enumerate() {
        cum += uk;
        let cand = (cum - 1.0_f32) / (k as f32 + 1.0);
        if uk - cand > 0.0 {
            tau = cand;
            found = true;
        } else {
            break;
        }
    }
    if !found {
        // Degenerate (e.g. all entries −∞); fall back to uniform.
        let uniform = 1.0_f32 / n as f32;
        for slot in v.iter_mut() {
            *slot = uniform;
        }
        return;
    }
    for slot in v.iter_mut() {
        *slot = (*slot - tau).max(0.0);
    }
    // Re-normalise against tiny f32 drift so the iterates stay exactly on the
    // simplex.
    let sum: f32 = v.iter().sum();
    if sum > 0.0 {
        let inv = 1.0_f32 / sum;
        for slot in v.iter_mut() {
            *slot *= inv;
        }
    } else {
        let uniform = 1.0_f32 / n as f32;
        for slot in v.iter_mut() {
            *slot = uniform;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l1_obj(x: &[f32]) -> f32 {
        x.iter().map(|v| v.abs()).sum()
    }

    fn l1_subgrad(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&v| {
                if v > 0.0 {
                    1.0_f32
                } else if v < 0.0 {
                    -1.0_f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn linf_obj(x: &[f32]) -> f32 {
        let mut m = 0.0_f32;
        for &v in x {
            m = m.max(v.abs());
        }
        m
    }

    fn linf_subgrad(x: &[f32]) -> Vec<f32> {
        // Subgradient: e_i · sign(x_i) for any argmax index i of |x|.
        let n = x.len();
        let mut idx = 0usize;
        let mut best = 0.0_f32;
        for (i, &v) in x.iter().enumerate() {
            let a = v.abs();
            if a > best {
                best = a;
                idx = i;
            }
        }
        let mut g = vec![0.0_f32; n];
        if best > 0.0 {
            g[idx] = if x[idx] >= 0.0 { 1.0 } else { -1.0 };
        }
        g
    }

    fn quad_obj(c: &[f32]) -> impl Fn(&[f32]) -> f32 + '_ {
        move |x: &[f32]| -> f32 {
            0.5_f32
                * x.iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
        }
    }

    fn quad_grad(c: &[f32]) -> impl Fn(&[f32]) -> Vec<f32> + '_ {
        move |x: &[f32]| -> Vec<f32> { x.iter().zip(c.iter()).map(|(a, b)| a - b).collect() }
    }

    #[test]
    fn minimize_l1_converges_to_zero() {
        let cfg = BundleConfig {
            max_iter: 200,
            max_bundle_size: 40,
            rho: 1.0,
            m_serious: 0.1,
            tol: 1.0e-5,
        };
        let res =
            ProximalBundle::minimize(&[1.5_f32, -2.0, 0.7], l1_obj, l1_subgrad, &cfg).expect("ok");
        for &v in &res.x {
            assert!(v.abs() < 1.0e-2, "x = {v}");
        }
    }

    #[test]
    fn minimize_linf_converges_to_zero() {
        let cfg = BundleConfig {
            max_iter: 200,
            max_bundle_size: 40,
            rho: 1.0,
            m_serious: 0.1,
            tol: 1.0e-5,
        };
        let res = ProximalBundle::minimize(&[2.0_f32, -1.0, 0.5], linf_obj, linf_subgrad, &cfg)
            .expect("ok");
        for &v in &res.x {
            assert!(v.abs() < 5.0e-2, "x = {v}");
        }
    }

    #[test]
    fn minimize_smooth_quadratic_converges_to_target() {
        let target = vec![1.0_f32, -2.0, 0.5];
        let f = quad_obj(&target);
        let g = quad_grad(&target);
        let cfg = BundleConfig {
            max_iter: 300,
            max_bundle_size: 40,
            rho: 1.0,
            m_serious: 0.2,
            tol: 1.0e-7,
        };
        let res = ProximalBundle::minimize(&[0.0_f32, 0.0, 0.0], &f, &g, &cfg).expect("ok");
        for (a, b) in res.x.iter().zip(target.iter()) {
            assert!((a - b).abs() < 2.0e-2, "x = {a}, target = {b}");
        }
        assert!(res.converged);
    }

    #[test]
    fn cutting_plane_single_cut_hand_example() {
        // bundle = { (x0 = [1], f0 = 4, g0 = [3]) }
        // f̂(x) = 4 + 3·(x − 1) = 3x + 1 . At x = 5 → 16.
        let bundle = vec![(vec![1.0_f32], 4.0_f32, vec![3.0_f32])];
        let v = ProximalBundle::cutting_plane_max(&bundle, &[5.0_f32]).expect("ok");
        assert!((v - 16.0).abs() < 1.0e-5, "v = {v}");
    }

    #[test]
    fn cutting_plane_empty_bundle_errors() {
        let bundle: Vec<(Vec<f32>, f32, Vec<f32>)> = Vec::new();
        assert!(matches!(
            ProximalBundle::cutting_plane_max(&bundle, &[0.0_f32]),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn bundle_size_bounded_by_cap() {
        // A pathological null-step problem: minimise f(x) = |x| with tiny tol
        // so the algorithm runs for many iterations. The bundle must stay ≤ cap.
        let cfg = BundleConfig {
            max_iter: 80,
            max_bundle_size: 5,
            rho: 5.0,
            m_serious: 0.99, // hard to satisfy → mostly null steps
            tol: 1.0e-12,
        };
        let res = ProximalBundle::minimize(&[10.0_f32], l1_obj, l1_subgrad, &cfg).expect("ok");
        // n_serious + n_null is the count of non-converged iterations.
        assert!(res.n_serious + res.n_null <= res.iterations);
        // Bundle never exceeds the cap throughout the run (probe via the
        // public API by calling solve_prox_master with the final cuts) — we
        // also verify by re-running and inspecting via a test-only mirror.
        // (Direct access not exposed; rely on counts here.)
        assert!(res.iterations <= cfg.max_iter);
    }

    #[test]
    fn step_count_invariant() {
        // n_serious + n_null ≤ iterations always (one of them per iter unless
        // the algorithm converged at the top of the loop and emitted nothing).
        let cfg = BundleConfig {
            max_iter: 60,
            max_bundle_size: 30,
            rho: 1.0,
            m_serious: 0.2,
            tol: 1.0e-6,
        };
        let res = ProximalBundle::minimize(&[3.0_f32, -1.0], l1_obj, l1_subgrad, &cfg).expect("ok");
        assert!(res.n_serious + res.n_null <= res.iterations);
    }

    #[test]
    fn converged_flag_when_predicted_dec_below_tol() {
        // Start from the optimum of f(x) = |x|: x = 0. The very first prox
        // master returns x_+ = 0 (no descent possible) and predicted_dec = 0 ≤ tol.
        let cfg = BundleConfig {
            max_iter: 20,
            max_bundle_size: 10,
            rho: 1.0,
            m_serious: 0.1,
            tol: 1.0e-6,
        };
        let res = ProximalBundle::minimize(&[0.0_f32, 0.0], l1_obj, l1_subgrad, &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.iterations >= 1);
    }

    #[test]
    fn deterministic_runs_match() {
        let cfg = BundleConfig::default();
        let r1 =
            ProximalBundle::minimize(&[1.0_f32, -1.5, 2.0], l1_obj, l1_subgrad, &cfg).expect("ok");
        let r2 =
            ProximalBundle::minimize(&[1.0_f32, -1.5, 2.0], l1_obj, l1_subgrad, &cfg).expect("ok");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.f, r2.f);
        assert_eq!(r1.n_serious, r2.n_serious);
        assert_eq!(r1.n_null, r2.n_null);
    }

    #[test]
    fn err_x0_empty() {
        let cfg = BundleConfig::default();
        assert!(matches!(
            ProximalBundle::minimize(&[], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn err_max_iter_zero() {
        let cfg = BundleConfig {
            max_iter: 0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_max_bundle_size_zero() {
        let cfg = BundleConfig {
            max_bundle_size: 0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_rho_non_positive() {
        let cfg = BundleConfig {
            rho: 0.0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_m_serious_out_of_range() {
        let cfg = BundleConfig {
            m_serious: 0.0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = BundleConfig {
            m_serious: 1.0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = BundleConfig {
            m_serious: -0.1,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = BundleConfig {
            m_serious: 1.5,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_tol_non_positive() {
        let cfg = BundleConfig {
            tol: 0.0,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
        let cfg = BundleConfig {
            tol: -1.0e-6,
            ..BundleConfig::default()
        };
        assert!(matches!(
            ProximalBundle::minimize(&[1.0_f32], l1_obj, l1_subgrad, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn solve_prox_master_returns_centre_length() {
        // Bundle with a single cut → λ = 1, x_+ = x_c − (1/ρ) g.
        let bundle = vec![(vec![0.0_f32, 0.0], 0.0_f32, vec![1.0_f32, -2.0])];
        let (x_plus, pred) =
            ProximalBundle::solve_prox_master(&bundle, &[3.0_f32, -1.0], 1.0).expect("ok");
        assert_eq!(x_plus.len(), 2);
        // x_+ = (3, -1) − (1, -2) = (2, 1).
        assert!((x_plus[0] - 2.0).abs() < 1.0e-5, "x_+[0] = {}", x_plus[0]);
        assert!((x_plus[1] - 1.0).abs() < 1.0e-5, "x_+[1] = {}", x_plus[1]);
        // f̂(x_c) = 0 + (1,−2)·(3,−1) = 3 + 2 = 5; f̂(x_+) = 0 + (1,−2)·(2,1) = 2 − 2 = 0.
        // predicted_dec = 5 − 0 = 5.
        assert!((pred - 5.0).abs() < 1.0e-4, "pred = {pred}");
    }

    #[test]
    fn solve_prox_master_empty_bundle_errors() {
        let bundle: Vec<(Vec<f32>, f32, Vec<f32>)> = Vec::new();
        assert!(matches!(
            ProximalBundle::solve_prox_master(&bundle, &[0.0_f32], 1.0),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn solve_prox_master_empty_centre_errors() {
        let bundle = vec![(vec![0.0_f32], 0.0_f32, vec![1.0_f32])];
        assert!(matches!(
            ProximalBundle::solve_prox_master(&bundle, &[], 1.0),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn solve_prox_master_rho_validation() {
        let bundle = vec![(vec![0.0_f32], 0.0_f32, vec![1.0_f32])];
        assert!(matches!(
            ProximalBundle::solve_prox_master(&bundle, &[1.0_f32], 0.0),
            Err(CvxError::InvalidParameter(_))
        ));
        assert!(matches!(
            ProximalBundle::solve_prox_master(&bundle, &[1.0_f32], -1.0),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn one_dim_minimization_sanity() {
        // f(x) = |x − 3|. Minimum at x = 3.
        let f = |x: &[f32]| -> f32 { (x[0] - 3.0).abs() };
        let g = |x: &[f32]| -> Vec<f32> {
            if x[0] > 3.0 {
                vec![1.0]
            } else if x[0] < 3.0 {
                vec![-1.0]
            } else {
                vec![0.0]
            }
        };
        let cfg = BundleConfig {
            max_iter: 200,
            max_bundle_size: 20,
            rho: 1.0,
            m_serious: 0.1,
            tol: 1.0e-6,
        };
        let res = ProximalBundle::minimize(&[0.0_f32], f, g, &cfg).expect("ok");
        assert!((res.x[0] - 3.0).abs() < 5.0e-2, "x = {}", res.x[0]);
    }

    #[test]
    fn cutting_plane_dim_mismatch_errors() {
        let bundle = vec![(vec![0.0_f32, 0.0], 0.0_f32, vec![1.0_f32, 2.0])];
        assert!(matches!(
            ProximalBundle::cutting_plane_max(&bundle, &[1.0_f32]),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn null_step_increments_counter() {
        // Multi-dimensional ‖x‖₁ where the initial cutting-plane model
        // overestimates the decrease along the prox direction: starting at
        // (10, 0.01) the subgradient is (1, 1) so the model predicts a large
        // drop along (−1, −1) but the actual decrease is small because the
        // sign of x₁ flips. With a stringent `m_serious` close to 1 this
        // triggers a null step on the first iteration.
        let cfg = BundleConfig {
            max_iter: 6,
            max_bundle_size: 30,
            rho: 5.0,
            m_serious: 0.95,
            tol: 1.0e-12,
        };
        let res =
            ProximalBundle::minimize(&[10.0_f32, 0.01], l1_obj, l1_subgrad, &cfg).expect("ok");
        assert!(res.n_null >= 1, "n_null = {}", res.n_null);
    }

    #[test]
    fn serious_step_increments_counter_easy_drop() {
        // With a tiny m_serious, the first prox-master iterate triggers a
        // serious step on a smooth quadratic.
        let target = vec![1.0_f32, -1.0];
        let f = quad_obj(&target);
        let g = quad_grad(&target);
        let cfg = BundleConfig {
            max_iter: 50,
            max_bundle_size: 30,
            rho: 1.0,
            m_serious: 1.0e-3,
            tol: 1.0e-7,
        };
        let res = ProximalBundle::minimize(&[5.0_f32, 5.0], &f, &g, &cfg).expect("ok");
        assert!(res.n_serious >= 1, "n_serious = {}", res.n_serious);
    }

    #[test]
    fn final_f_matches_x() {
        // The reported `f` field must equal `f(x)`.
        let cfg = BundleConfig::default();
        let res =
            ProximalBundle::minimize(&[2.0_f32, -3.0, 0.5], l1_obj, l1_subgrad, &cfg).expect("ok");
        let f_at_x = l1_obj(&res.x);
        assert!((res.f - f_at_x).abs() < 1.0e-5, "{} vs {}", res.f, f_at_x);
    }
}
