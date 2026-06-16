//! Synthetic Control Method (SCM) — Abadie-Diamond-Hainmueller (2010).
//!
//! Reference: Abadie, A., Diamond, A. & Hainmueller, J. (2010). "Synthetic
//! control methods for comparative case studies: estimating the effect of
//! California's tobacco control program." *Journal of the American
//! Statistical Association*, 105(490), 493-505.  See also Abadie, A. (2021).
//! "Using synthetic controls: feasibility, data requirements, and
//! methodological aspects." *Journal of Economic Literature*, 59(2), 391-425.
//!
//! # Setting
//!
//! Panel data are observed for `J + 1` units over `T` time periods.  Unit
//! `0` is the *treated* unit, units `1..=J` form the **donor pool**, and
//! treatment starts at period `T₀ ≥ 1` (so periods `[0, T₀)` are pre-
//! treatment, periods `[T₀, T)` are post-treatment).  Outcomes are stored
//! in a row-major `(n_units × n_periods)` matrix `Y`.
//!
//! # Algorithm
//!
//! The synthetic control method approximates the counterfactual outcome of
//! the treated unit by a *convex combination* of donor outcomes,
//!
//! ```text
//!   Ŷ_0,t  =  Σ_{j=1..J}  w_j  Y_{j, t},
//! ```
//!
//! where the weights `w = (w_1, …, w_J)` are constrained to the probability
//! simplex `Δ^J = { w : w_j ≥ 0, Σ_j w_j = 1 }`.  The weights are chosen to
//! match the treated trajectory in the **pre-treatment** window by
//! minimising
//!
//! ```text
//!   w* = argmin_{w ∈ Δ^J}  ‖ Y_{0, [0,T₀)}  −  Σ_j w_j Y_{j, [0,T₀)} ‖².
//! ```
//!
//! Let `D ∈ R^{T₀ × J}` denote the donor sub-matrix `D_{t, j} = Y_{j, t}`
//! for `t < T₀`, and let `y* = Y_{0, [0,T₀)} ∈ R^{T₀}`.  The QP reads
//!
//! ```text
//!   minimise   ‖ D w − y* ‖²
//!   subject to  w ∈ Δ^J.
//! ```
//!
//! This is a strictly convex QP (whenever `D^T D` is positive definite on
//! the affine hull of the simplex) and we solve it by **projected gradient
//! descent**: starting from the uniform weights `w = (1/J, …, 1/J)`, we
//! iterate
//!
//! ```text
//!   g  =  2 · D^T (D w − y*)
//!   w_new  =  Π_{Δ^J} ( w − η · g ),
//! ```
//!
//! where the simplex projection `Π_{Δ^J}(·)` is implemented exactly via the
//! Wang-Carreira-Perpiñán (2013) sort-based O(J log J) algorithm (sort
//! descending, find the largest `ρ` such that
//! `u_ρ − (Σ_{i≤ρ} u_i − 1)/(ρ+1) > 0`, threshold and clamp).
//!
//! Iteration stops when `‖w_new − w_old‖_∞ < tol` or after `n_iter`
//! iterations, whichever comes first.
//!
//! # Post-treatment causal effects
//!
//! Once `w*` is fixed, the **synthetic counterfactual** at every period
//! `t` is `Σ_j w*_j Y_{j, t}`, and the **causal effect** at post-treatment
//! period `t ≥ T₀` is
//!
//! ```text
//!   τ̂_t  =  Y_{0, t}  −  Σ_j w*_j Y_{j, t}.
//! ```
//!
//! Following Abadie-Diamond-Hainmueller, a small post-treatment effect
//! together with a small pre-treatment RMSE is interpreted as evidence
//! against a causal effect, whereas a large τ̂_t coupled with small pre-
//! treatment RMSE is interpreted as evidence of a treatment effect.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`synthetic_control`].
#[derive(Debug, Clone)]
pub struct SyntheticControlConfig {
    /// Maximum number of projected-gradient iterations.  Must be `≥ 1`.
    pub n_iter: usize,
    /// Gradient step size `η > 0`.  Smaller values are more stable but
    /// slower; `1e-3` is usually a good starting point for unit-scale data.
    pub step_size: f64,
    /// Convergence tolerance on `‖w_new − w_old‖_∞`.  Must be `> 0`.
    pub tol: f64,
}

impl Default for SyntheticControlConfig {
    fn default() -> Self {
        Self {
            n_iter: 2000,
            step_size: 1e-3,
            tol: 1e-9,
        }
    }
}

/// Result of [`synthetic_control`].
#[derive(Debug, Clone)]
pub struct SyntheticControlResult {
    /// Donor weights `w* ∈ Δ^J`.  Length `J = n_units − 1`.  Each entry is
    /// non-negative and the entries sum to `1` (up to numerical rounding).
    pub weights: Vec<f64>,
    /// Root mean square pre-treatment fit
    /// `‖Y_{0,[0,T₀)} − D w*‖₂ / √T₀`.
    pub pretreatment_rmse: f64,
    /// Post-treatment causal effects `τ̂_t = Y_{0,t} − Σ_j w*_j Y_{j,t}`
    /// for `t ∈ [T₀, T)`.  Length `T − T₀`.
    pub effects: Vec<f64>,
    /// Synthetic counterfactual trajectory `Σ_j w*_j Y_{j, t}` for every
    /// `t ∈ [0, T)`.  Length `T`.
    pub synthetic: Vec<f64>,
}

/// Compute synthetic-control weights and post-treatment effects.
///
/// # Parameters
/// - `y`: row-major `n_units × n_periods` outcome panel.  Unit index `0` is
///   the treated unit; units `1..n_units` are donors.
/// - `n_units`: total number of units (treated + donors).  Must be `≥ 2`.
/// - `n_periods`: total number of time periods.  Must be `≥ 2`.
/// - `t0`: first **post**-treatment period.  Must satisfy `1 ≤ t0 ≤ n_periods − 1`.
/// - `cfg`: see [`SyntheticControlConfig`].
///
/// # Errors
/// - [`CausalError::EmptyInput`] if `y` is empty or any size is zero.
/// - [`CausalError::DimensionMismatch`] if `y.len() != n_units · n_periods`.
/// - [`CausalError::IncompatibleData`] if `n_units < 2`, `n_periods < 2`,
///   `t0 == 0`, `t0 >= n_periods`, `cfg.n_iter == 0`, `cfg.step_size ≤ 0`,
///   or `cfg.tol ≤ 0`.
pub fn synthetic_control(
    y: &[f64],
    n_units: usize,
    n_periods: usize,
    t0: usize,
    cfg: &SyntheticControlConfig,
) -> CausalResult<SyntheticControlResult> {
    // ---- input validation -----------------------------------------------
    if y.is_empty() || n_units == 0 || n_periods == 0 {
        return Err(CausalError::EmptyInput);
    }
    if y.len() != n_units * n_periods {
        return Err(CausalError::DimensionMismatch {
            expected: n_units * n_periods,
            got: y.len(),
        });
    }
    if n_units < 2 || n_periods < 2 {
        return Err(CausalError::IncompatibleData);
    }
    if t0 == 0 || t0 >= n_periods {
        return Err(CausalError::IncompatibleData);
    }
    if cfg.n_iter == 0 || cfg.step_size <= 0.0 || cfg.tol <= 0.0 {
        return Err(CausalError::IncompatibleData);
    }

    let n_donors = n_units - 1;

    // ---- pre-treatment target y* and donor matrix D --------------------
    // y* has shape (t0,);  D has shape (t0, n_donors), row-major.
    // Treated unit lives at row 0 of `y`, so its row offset is just 0.
    let mut y_star = vec![0.0_f64; t0];
    y_star[..t0].copy_from_slice(&y[..t0]);
    let mut d_mat = vec![0.0_f64; t0 * n_donors];
    for t in 0..t0 {
        for j in 0..n_donors {
            let unit = j + 1; // donor units start at index 1
            d_mat[t * n_donors + j] = y[unit * n_periods + t];
        }
    }

    // ---- projected gradient descent ------------------------------------
    let mut w = vec![1.0_f64 / n_donors as f64; n_donors];
    let mut w_prev = w.clone();
    let mut residual = vec![0.0_f64; t0];
    let mut grad = vec![0.0_f64; n_donors];

    for _iter in 0..cfg.n_iter {
        // residual r = D w − y*  (length t0)
        for (t, r_t) in residual.iter_mut().enumerate() {
            let mut s = 0.0_f64;
            for (j, &w_j) in w.iter().enumerate() {
                s += d_mat[t * n_donors + j] * w_j;
            }
            *r_t = s - y_star[t];
        }
        // gradient g = 2 D^T r  (length n_donors)
        for (j, g_j) in grad.iter_mut().enumerate() {
            let mut g = 0.0_f64;
            for (t, &r_t) in residual.iter().enumerate() {
                g += d_mat[t * n_donors + j] * r_t;
            }
            *g_j = 2.0 * g;
        }
        // tentative descent step then project onto Δ^J
        w_prev.copy_from_slice(&w);
        for (w_j, g_j) in w.iter_mut().zip(grad.iter()) {
            *w_j -= cfg.step_size * g_j;
        }
        project_to_simplex_in_place(&mut w);

        // convergence check on L∞ norm of weight change
        let mut max_change = 0.0_f64;
        for (w_j, w_p) in w.iter().zip(w_prev.iter()) {
            let dlt = (w_j - w_p).abs();
            if dlt > max_change {
                max_change = dlt;
            }
        }
        if max_change < cfg.tol {
            break;
        }
    }

    // ---- pre-treatment RMSE --------------------------------------------
    let mut sse = 0.0_f64;
    for t in 0..t0 {
        let mut yhat = 0.0_f64;
        for (j, &w_j) in w.iter().enumerate() {
            yhat += d_mat[t * n_donors + j] * w_j;
        }
        let r = y[t] - yhat; // treated row offset is 0
        sse += r * r;
    }
    let pretreatment_rmse = (sse / t0 as f64).sqrt();

    // ---- synthetic trajectory and post-treatment effects ---------------
    let mut synthetic = vec![0.0_f64; n_periods];
    for (t, syn_t) in synthetic.iter_mut().enumerate() {
        let mut s = 0.0_f64;
        for (j, &w_j) in w.iter().enumerate() {
            let unit = j + 1;
            s += w_j * y[unit * n_periods + t];
        }
        *syn_t = s;
    }
    let mut effects = vec![0.0_f64; n_periods - t0];
    for (k, eff) in effects.iter_mut().enumerate() {
        let t = t0 + k;
        *eff = y[t] - synthetic[t]; // treated row offset is 0
    }

    Ok(SyntheticControlResult {
        weights: w,
        pretreatment_rmse,
        effects,
        synthetic,
    })
}

// =====================================================================
// helpers — exact Euclidean projection onto the probability simplex
// =====================================================================

/// Project `w` onto `{x ∈ R^J : x_j ≥ 0, Σ_j x_j = 1}` in place.
///
/// Implements the O(J log J) sort-based algorithm of
/// Wang & Carreira-Perpiñán (2013): sort `w` descending into `u`, find
/// `ρ = max{ k : u_k − (Σ_{i ≤ k} u_i − 1) / (k+1) > 0 }`, set
/// `τ = (Σ_{i ≤ ρ} u_i − 1) / (ρ + 1)`, then `w_j ← max(w_j − τ, 0)`.
///
/// When all candidates fail the positivity test (which can happen if every
/// `u_i` is identical and equals `1/J`), the function falls back to
/// `τ = (Σ_i u_i − 1) / J`, which gives the closed-form projection of a
/// constant vector and recovers the uniform distribution `1/J` for the
/// degenerate input.
fn project_to_simplex_in_place(w: &mut [f64]) {
    let n = w.len();
    if n == 0 {
        return;
    }
    let mut u: Vec<f64> = w.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0_f64;
    let mut tau = (u.iter().sum::<f64>() - 1.0) / n as f64;
    let mut found = false;
    for (k, &uk) in u.iter().enumerate() {
        cum += uk;
        let candidate = (cum - 1.0) / (k as f64 + 1.0);
        if uk - candidate > 0.0 {
            tau = candidate;
            found = true;
        } else {
            break;
        }
    }
    // If no k satisfies the positivity test we keep the fall-back τ above,
    // which still yields a valid simplex projection for degenerate inputs.
    let _ = found;
    for w_j in w.iter_mut() {
        *w_j = (*w_j - tau).max(0.0);
    }
}

// =====================================================================
// tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a (n_units × n_periods) panel row-major.
    fn build_panel(rows: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
        let n_units = rows.len();
        let n_periods = rows[0].len();
        let mut y = Vec::with_capacity(n_units * n_periods);
        for r in rows {
            assert_eq!(r.len(), n_periods);
            y.extend_from_slice(r);
        }
        (y, n_units, n_periods)
    }

    // -------------------- input validation tests ---------------------------

    #[test]
    fn invalid_empty() {
        let cfg = SyntheticControlConfig::default();
        let r = synthetic_control(&[], 0, 0, 0, &cfg);
        assert!(matches!(r, Err(CausalError::EmptyInput)));
    }

    #[test]
    fn invalid_dim_mismatch() {
        let cfg = SyntheticControlConfig::default();
        // y.len() = 5 but 3*4 = 12
        let r = synthetic_control(&[1.0; 5], 3, 4, 2, &cfg);
        assert!(matches!(r, Err(CausalError::DimensionMismatch { .. })));
    }

    #[test]
    fn invalid_n_units_less_than_two() {
        let cfg = SyntheticControlConfig::default();
        let r = synthetic_control(&[1.0, 2.0, 3.0], 1, 3, 1, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_n_periods_less_than_two() {
        let cfg = SyntheticControlConfig::default();
        let r = synthetic_control(&[1.0, 2.0], 2, 1, 0, &cfg);
        // n_periods < 2  →  IncompatibleData (even though t0=0 also fails).
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_t0_zero() {
        let cfg = SyntheticControlConfig::default();
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 0, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_t0_equals_n_periods() {
        let cfg = SyntheticControlConfig::default();
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 4, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_t0_greater_than_n_periods() {
        let cfg = SyntheticControlConfig::default();
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 5, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_zero_iterations() {
        let cfg = SyntheticControlConfig {
            n_iter: 0,
            ..SyntheticControlConfig::default()
        };
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 2, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_step_size_zero() {
        let cfg = SyntheticControlConfig {
            step_size: 0.0,
            ..SyntheticControlConfig::default()
        };
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 2, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_step_size_negative() {
        let cfg = SyntheticControlConfig {
            step_size: -1.0,
            ..SyntheticControlConfig::default()
        };
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 2, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    #[test]
    fn invalid_tol_zero() {
        let cfg = SyntheticControlConfig {
            tol: 0.0,
            ..SyntheticControlConfig::default()
        };
        let (y, n_u, n_t) = build_panel(&[vec![1.0; 4], vec![2.0; 4]]);
        let r = synthetic_control(&y, n_u, n_t, 2, &cfg);
        assert!(matches!(r, Err(CausalError::IncompatibleData)));
    }

    // -------------------- correctness tests --------------------------------

    /// "Treated = 0.5·donor1 + 0.5·donor2": recovered weights ≈ (0.5, 0.5).
    /// Two-donor SCM, pre-treatment fit exact.
    #[test]
    fn recovers_half_half_weights_two_donors() {
        let t_total = 10;
        let t0 = 6;
        // Donor 1: linear up, donor 2: linear down.
        let donor1: Vec<f64> = (0..t_total).map(|t| 1.0 + 0.3 * t as f64).collect();
        let donor2: Vec<f64> = (0..t_total).map(|t| 5.0 - 0.2 * t as f64).collect();
        // Treated trajectory = average of the two donors.
        let treated: Vec<f64> = donor1
            .iter()
            .zip(donor2.iter())
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect();
        let (y, n_u, n_t) = build_panel(&[treated, donor1.clone(), donor2.clone()]);
        let cfg = SyntheticControlConfig {
            n_iter: 5000,
            step_size: 5e-3,
            tol: 1e-12,
        };
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        assert_eq!(res.weights.len(), 2);
        let s: f64 = res.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "weights sum = {s}");
        assert!(
            (res.weights[0] - 0.5).abs() < 0.05,
            "w_1 = {} (expected ~0.5)",
            res.weights[0]
        );
        assert!(
            (res.weights[1] - 0.5).abs() < 0.05,
            "w_2 = {} (expected ~0.5)",
            res.weights[1]
        );
        // No treatment effect was injected → post-treatment effects ≈ 0.
        for &e in &res.effects {
            assert!(e.abs() < 0.05, "post-treatment effect = {e}");
        }
        assert!(
            res.pretreatment_rmse < 0.05,
            "pre RMSE = {}",
            res.pretreatment_rmse
        );
    }

    /// Detect a positive shock injected post-treatment.
    #[test]
    fn detects_post_treatment_shock() {
        let t_total = 12;
        let t0 = 7;
        let shock = 4.0_f64;
        let donor1: Vec<f64> = (0..t_total).map(|t| 1.0 + 0.4 * t as f64).collect();
        let donor2: Vec<f64> = (0..t_total).map(|t| 6.0 - 0.3 * t as f64).collect();
        // Treated = 0.5·donor1 + 0.5·donor2 in pre-period, plus +shock after t0.
        let mut treated = vec![0.0_f64; t_total];
        for t in 0..t_total {
            let baseline = 0.5 * donor1[t] + 0.5 * donor2[t];
            treated[t] = if t >= t0 { baseline + shock } else { baseline };
        }
        let (y, n_u, n_t) = build_panel(&[treated, donor1, donor2]);
        let cfg = SyntheticControlConfig {
            n_iter: 5000,
            step_size: 5e-3,
            tol: 1e-12,
        };
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        // Each post-treatment effect should be ≈ +shock.
        for &e in &res.effects {
            assert!((e - shock).abs() < 0.20, "effect = {e} expected ~{shock}");
        }
        // Pre-treatment fit should be excellent.
        assert!(
            res.pretreatment_rmse < 0.05,
            "pre RMSE = {}",
            res.pretreatment_rmse
        );
    }

    #[test]
    fn weights_sum_to_one_and_nonneg() {
        let t_total = 8;
        let t0 = 5;
        let mut rng = LcgRng::new(7);
        let mut rows = Vec::new();
        for _ in 0..6 {
            let r: Vec<f64> = (0..t_total).map(|_| rng.next_f32() as f64).collect();
            rows.push(r);
        }
        let (y, n_u, n_t) = build_panel(&rows);
        let cfg = SyntheticControlConfig::default();
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        for &w in &res.weights {
            assert!(w >= -1e-12, "negative weight: {w}");
        }
        let s: f64 = res.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "weights sum = {s}");
    }

    #[test]
    fn deterministic() {
        let t_total = 10;
        let t0 = 6;
        let mut rng = LcgRng::new(99);
        let mut rows = Vec::new();
        for _ in 0..5 {
            let r: Vec<f64> = (0..t_total).map(|_| rng.next_f32() as f64).collect();
            rows.push(r);
        }
        let (y, n_u, n_t) = build_panel(&rows);
        let cfg = SyntheticControlConfig::default();
        let r1 =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        let r2 =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        assert_eq!(r1.weights, r2.weights);
        assert_eq!(r1.effects, r2.effects);
        assert_eq!(r1.synthetic, r2.synthetic);
        assert_eq!(r1.pretreatment_rmse, r2.pretreatment_rmse);
    }

    #[test]
    fn convergence_improves_with_more_iterations() {
        let t_total = 10;
        let t0 = 6;
        let donor1: Vec<f64> = (0..t_total).map(|t| 0.5 + 0.4 * t as f64).collect();
        let donor2: Vec<f64> = (0..t_total).map(|t| 5.0 - 0.3 * t as f64).collect();
        let donor3: Vec<f64> = (0..t_total).map(|t| 2.0 + 0.05 * t as f64).collect();
        let treated: Vec<f64> = (0..t_total)
            .map(|t| 0.4 * donor1[t] + 0.4 * donor2[t] + 0.2 * donor3[t])
            .collect();
        let (y, n_u, n_t) = build_panel(&[treated, donor1, donor2, donor3]);
        let cfg_few = SyntheticControlConfig {
            n_iter: 5,
            step_size: 5e-3,
            tol: 1e-30,
        };
        let cfg_many = SyntheticControlConfig {
            n_iter: 5000,
            step_size: 5e-3,
            tol: 1e-30,
        };
        let r_few = synthetic_control(&y, n_u, n_t, t0, &cfg_few)
            .expect("synthetic_control should succeed");
        let r_many = synthetic_control(&y, n_u, n_t, t0, &cfg_many)
            .expect("synthetic_control should succeed");
        assert!(
            r_many.pretreatment_rmse <= r_few.pretreatment_rmse + 1e-9,
            "RMSE did not decrease: few = {}, many = {}",
            r_few.pretreatment_rmse,
            r_many.pretreatment_rmse
        );
        // With enough iterations the fit should be tight.
        assert!(
            r_many.pretreatment_rmse < 0.10,
            "many-iter RMSE = {}",
            r_many.pretreatment_rmse
        );
    }

    #[test]
    fn synthetic_length_equals_n_periods() {
        let t_total = 7;
        let t0 = 4;
        let (y, n_u, n_t) =
            build_panel(&[vec![1.0; t_total], vec![1.0; t_total], vec![2.0; t_total]]);
        let cfg = SyntheticControlConfig::default();
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        assert_eq!(res.synthetic.len(), n_t);
        assert_eq!(res.effects.len(), n_t - t0);
    }

    #[test]
    fn effects_length_zero_when_t0_is_last_minus_one() {
        // t0 = n_periods - 1  →  one post-treatment period.
        let t0 = 3;
        let (y, n_u, n_t) = build_panel(&[
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 5.0],
            vec![1.0, 2.0, 3.0, 3.0],
        ]);
        let cfg = SyntheticControlConfig::default();
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        assert_eq!(res.effects.len(), 1);
        assert_eq!(res.synthetic.len(), n_t);
        assert!(res.effects[0].is_finite());
    }

    #[test]
    fn large_n_units_runs() {
        let t_total = 20;
        let t0 = 14;
        let n_donors = 20;
        let mut rng = LcgRng::new(7777);
        let mut rows = Vec::new();
        // Donors: random walks.
        let mut donor_rows = Vec::with_capacity(n_donors);
        for _ in 0..n_donors {
            let mut row = vec![0.0_f64; t_total];
            let mut v = 0.5;
            for x in &mut row {
                v += (rng.next_f32() as f64 - 0.5) * 0.2;
                *x = v;
            }
            donor_rows.push(row);
        }
        // Treated: convex combination of the first three donors.
        let mut treated = vec![0.0_f64; t_total];
        for t in 0..t_total {
            treated[t] = 0.5 * donor_rows[0][t] + 0.3 * donor_rows[1][t] + 0.2 * donor_rows[2][t];
        }
        rows.push(treated);
        rows.extend(donor_rows);
        let (y, n_u, n_t) = build_panel(&rows);
        assert_eq!(n_u, 1 + n_donors);
        let cfg = SyntheticControlConfig {
            n_iter: 4000,
            step_size: 5e-3,
            tol: 1e-10,
        };
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        assert_eq!(res.weights.len(), n_donors);
        let s: f64 = res.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "weights sum = {s}");
        for &w in &res.weights {
            assert!(w >= -1e-9);
        }
        // The pre-treatment fit should be tight.
        assert!(
            res.pretreatment_rmse < 0.10,
            "large RMSE = {}",
            res.pretreatment_rmse
        );
    }

    #[test]
    fn synthetic_matches_weighted_donors() {
        let t_total = 8;
        let t0 = 5;
        let donor1: Vec<f64> = (0..t_total).map(|t| 1.0 + 0.5 * t as f64).collect();
        let donor2: Vec<f64> = (0..t_total).map(|t| 3.0 - 0.2 * t as f64).collect();
        let treated: Vec<f64> = donor1
            .iter()
            .zip(donor2.iter())
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect();
        let (y, n_u, n_t) = build_panel(&[treated, donor1.clone(), donor2.clone()]);
        let cfg = SyntheticControlConfig {
            n_iter: 5000,
            step_size: 5e-3,
            tol: 1e-12,
        };
        let res =
            synthetic_control(&y, n_u, n_t, t0, &cfg).expect("synthetic_control should succeed");
        // Verify synthetic[t] = w_1·donor1[t] + w_2·donor2[t] for every t.
        for t in 0..t_total {
            let expected = res.weights[0] * donor1[t] + res.weights[1] * donor2[t];
            assert!(
                (res.synthetic[t] - expected).abs() < 1e-9,
                "synthetic[{}] = {} expected {}",
                t,
                res.synthetic[t],
                expected
            );
        }
        // Effects = treated − synthetic for post-treatment periods.
        for k in 0..(t_total - t0) {
            let t = t0 + k;
            let treated_t = 0.5 * donor1[t] + 0.5 * donor2[t]; // we built it this way
            let expected_eff = treated_t - res.synthetic[t];
            assert!(
                (res.effects[k] - expected_eff).abs() < 1e-9,
                "effects[{}] mismatch",
                k
            );
        }
    }

    #[test]
    fn config_default_is_sane() {
        let cfg = SyntheticControlConfig::default();
        assert!(cfg.n_iter >= 1);
        assert!(cfg.step_size > 0.0);
        assert!(cfg.tol > 0.0);
    }
}
