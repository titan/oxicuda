//! Trust-region Newton optimisation for Cox proportional hazards.
//!
//! Implements:
//! - [`steihaug_cg`]: Steihaug truncated CG for the trust-region sub-problem.
//! - [`trust_region_cox`]: Outer trust-region loop using the negative partial log-likelihood.
//!
//! # Algorithm
//! The outer loop follows the Dogleg / Steihaug-CG trust-region Newton method (Nocedal &
//! Wright (2006), Algorithm 4.3 + 7.2).  The Hessian and gradient are the Fisher information
//! matrix and score vector from the Cox partial likelihood (Breslow or Efron ties).

use crate::cox::newton_raphson::TieMethod;
use crate::data::{Dataset, Observation};
use crate::error::{SurvivalError, SurvivalResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the trust-region Newton algorithm.
#[derive(Debug, Clone, Copy)]
pub struct TrustRegionConfig {
    /// Initial trust-region radius.
    pub delta_init: f64,
    /// Maximum allowed trust-region radius.
    pub delta_max: f64,
    /// Acceptance threshold for the reduction ratio ρ.
    pub eta: f64,
    /// Maximum outer (Newton) iterations.
    pub max_outer: usize,
    /// Maximum conjugate-gradient iterations per outer step.
    pub cg_max_iter: usize,
    /// CG convergence tolerance (`||r|| < cg_tol * ||r0||`).
    pub cg_tol: f64,
    /// Outer convergence tolerance (on `||gradient||_inf`).
    pub tol: f64,
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        Self {
            delta_init: 1.0,
            delta_max: 100.0,
            eta: 0.1,
            max_outer: 100,
            cg_max_iter: 50,
            cg_tol: 1.0e-6,
            tol: 1.0e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a trust-region Newton Cox fit.
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    /// Estimated coefficient vector β.
    pub coef: Vec<f64>,
    /// Number of outer (trust-region) iterations performed.
    pub n_outer_iters: usize,
    /// Whether the algorithm converged within `max_outer` iterations.
    pub converged: bool,
    /// Partial log-likelihood at the returned coefficient vector.
    pub log_lik: f64,
}

// ---------------------------------------------------------------------------
// Matrix-vector helpers
// ---------------------------------------------------------------------------

/// Compute `H * v` where `H` is a `p×p` row-major matrix and `v` has length `p`.
fn mat_vec(h: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let p = h.len();
    let mut out = vec![0.0_f64; p];
    for i in 0..p {
        for j in 0..p {
            out[i] += h[i][j] * v[j];
        }
    }
    out
}

/// Dot product of two length-`p` vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Euclidean norm.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Scale: `alpha * v`.
fn scale(alpha: f64, v: &[f64]) -> Vec<f64> {
    v.iter().map(|vi| alpha * vi).collect()
}

/// `a + b` element-wise.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

/// Find τ ≥ 0 such that `||p + τ d||² = delta²`.
/// Returns the positive root of the quadratic in τ.
fn step_to_boundary(p: &[f64], d: &[f64], delta: f64) -> f64 {
    let pp = dot(p, p);
    let pd = dot(p, d);
    let dd = dot(d, d);
    let disc = pd * pd - dd * (pp - delta * delta);
    if disc < 0.0 || dd < f64::EPSILON {
        return 0.0;
    }
    (-pd + disc.sqrt()) / dd
}

// ---------------------------------------------------------------------------
// Steihaug truncated CG
// ---------------------------------------------------------------------------

/// Solve the trust-region sub-problem via Steihaug's CG method.
///
/// Minimises the quadratic model `m(p) = 0.5 pᵀHp + gᵀp` subject to `||p|| ≤ delta`.
///
/// The Hessian `H` is provided as a `Vec<Vec<f64>>` (row-major list of rows).  Gradient `g`
/// is the score from the partial log-likelihood (note: we *maximise* log-likelihood, so the
/// gradient of the *negative* log-likelihood is `-g`; callers pass the gradient of the
/// *objective to minimise*, which is `-score`).
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] if shapes are inconsistent.
/// - [`SurvivalError::NumericalInstability`] if CG encounters a NaN.
pub fn steihaug_cg(
    gradient: &[f64],
    hessian: &[Vec<f64>],
    delta: f64,
    max_iter: usize,
    tol: f64,
) -> SurvivalResult<Vec<f64>> {
    let p = gradient.len();
    if hessian.len() != p || hessian.iter().any(|row| row.len() != p) {
        return Err(SurvivalError::DimensionMismatch {
            a: p,
            b: hessian.len(),
        });
    }

    // p_0 = 0, r_0 = g, d_0 = -g
    let mut step = vec![0.0_f64; p]; // current step vector
    let mut r = gradient.to_vec(); // residual r = g + H * step = g at step=0
    let mut d: Vec<f64> = gradient.iter().map(|gi| -gi).collect(); // conjugate direction

    let r0_norm = norm2(&r);
    if r0_norm < f64::EPSILON {
        // Already at critical point.
        return Ok(step);
    }
    let stop_tol = tol * r0_norm;

    for _iter in 0..max_iter {
        let hd = mat_vec(hessian, &d);
        let dhd = dot(&d, &hd);

        // Negative curvature or zero curvature → step to boundary along d.
        if dhd <= 0.0 {
            let tau = step_to_boundary(&step, &d, delta);
            for i in 0..p {
                step[i] += tau * d[i];
            }
            return Ok(step);
        }

        let rr = dot(&r, &r);
        let alpha_cg = rr / dhd;

        // Would the new step exceed the trust-region boundary?
        let p_new = vec_add(&step, &scale(alpha_cg, &d));
        if norm2(&p_new) >= delta {
            // Step to boundary along d.
            let tau = step_to_boundary(&step, &d, delta);
            for i in 0..p {
                step[i] += tau * d[i];
            }
            return Ok(step);
        }

        // Accept the CG step.
        step = p_new;
        // r_{k+1} = r_k + alpha * H d
        let r_new: Vec<f64> = r
            .iter()
            .zip(hd.iter())
            .map(|(ri, hdi)| ri + alpha_cg * hdi)
            .collect();

        if norm2(&r_new) < stop_tol {
            return Ok(step);
        }

        let rr_new = dot(&r_new, &r_new);
        if !rr_new.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "Steihaug-CG: NaN in residual norm".to_string(),
            ));
        }
        let beta_cg = rr_new / rr;
        // d_{k+1} = -r_{k+1} + beta * d_k
        d = r_new
            .iter()
            .zip(d.iter())
            .map(|(rni, di)| -rni + beta_cg * di)
            .collect();
        r = r_new;
    }

    Ok(step)
}

// ---------------------------------------------------------------------------
// Internal helpers for Cox log-likelihood / gradient / Hessian
// ---------------------------------------------------------------------------

/// Build a [`Dataset`] from raw `(time, event, covariates)` triples.
fn triples_to_dataset(data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<Dataset> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let mut obs = Vec::with_capacity(data.len());
    let mut cov = Vec::with_capacity(data.len());
    for (t, e, x) in data.iter() {
        obs.push(Observation::new(*t, *e)?);
        cov.push(x.clone());
    }
    Dataset::new(obs, Some(cov), None)
}

/// Compute the negative partial log-likelihood, its gradient (negative score) and the
/// negative Hessian (Fisher information) from the partial log-likelihood functions.
///
/// Returns `(neg_loglik, neg_score, fisher_info_2d)`.
/// `fisher_info_2d` is returned as `Vec<Vec<f64>>` for use in the CG sub-problem.
fn neg_loglik_grad_hess(
    ds: &Dataset,
    beta: &[f64],
    tie: TieMethod,
) -> SurvivalResult<(f64, Vec<f64>, Vec<Vec<f64>>)> {
    let p = beta.len();
    let (ll, score, info_flat) = match tie {
        TieMethod::Breslow => crate::cox::breslow_ties::breslow_log_likelihood(ds, beta)?,
        TieMethod::Efron => crate::cox::efron_ties::efron_log_likelihood(ds, beta)?,
    };
    // We minimise -loglik, so gradient = -score.
    let neg_score: Vec<f64> = score.iter().map(|s| -s).collect();
    // Fisher information is the *negative* Hessian of loglik, so it is the Hessian of -loglik.
    let mut hessian = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..p {
            hessian[i][j] = info_flat[i * p + j];
        }
    }
    Ok((-ll, neg_score, hessian))
}

// ---------------------------------------------------------------------------
// Trust-region Newton for Cox
// ---------------------------------------------------------------------------

/// Trust-region Newton for Cox proportional hazards.
///
/// Fits the Cox PH model by minimising the negative partial log-likelihood using a
/// Steihaug-CG trust-region Newton method.  This is particularly effective for
/// ill-conditioned design matrices where standard Newton-Raphson may diverge.
///
/// # Arguments
/// - `data`: slice of `(time, event, covariates)` tuples.
/// - `config`: trust-region hyperparameters.
/// - `tie_method`: Breslow or Efron tie handling.
///
/// # Errors
/// - [`SurvivalError::EmptyDataset`] if `data` is empty.
/// - [`SurvivalError::NoEvents`] if there are no events.
/// - Propagates errors from the likelihood computation.
pub fn trust_region_cox(
    data: &[(f64, bool, Vec<f64>)],
    config: &TrustRegionConfig,
    tie_method: TieMethod,
) -> SurvivalResult<TrustRegionResult> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let n_events = data.iter().filter(|(_, e, _)| *e).count();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let ds = triples_to_dataset(data)?;
    let p = ds.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "trust_region_cox: no covariates".to_string(),
        ));
    }

    let mut beta = vec![0.0_f64; p];
    let mut delta = config.delta_init;
    let mut converged = false;
    let mut n_outer_iters = 0usize;
    let mut last_log_lik = 0.0_f64;

    for outer in 0..config.max_outer {
        n_outer_iters = outer + 1;

        let (neg_ll, neg_score, hessian) = neg_loglik_grad_hess(&ds, &beta, tie_method)?;
        last_log_lik = -neg_ll;

        // Check convergence: ||gradient||_inf < tol.
        let g_inf = neg_score.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if g_inf < config.tol {
            converged = true;
            break;
        }

        // Solve trust-region sub-problem.
        let step = steihaug_cg(
            &neg_score,
            &hessian,
            delta,
            config.cg_max_iter,
            config.cg_tol,
        )?;

        let step_norm = norm2(&step);

        // Predicted reduction: m(0) - m(p) = -(gᵀp + 0.5 pᵀHp)
        let gp = dot(&neg_score, &step);
        let hp = mat_vec(&hessian, &step);
        let phpp = dot(&step, &hp);
        let predicted = -(gp + 0.5 * phpp);

        // Actual reduction.
        let beta_new: Vec<f64> = beta.iter().zip(step.iter()).map(|(b, s)| b + s).collect();
        let (neg_ll_new, _, _) = neg_loglik_grad_hess(&ds, &beta_new, tie_method)?;
        let actual = neg_ll - neg_ll_new;

        // Compute reduction ratio ρ.
        let rho = if predicted.abs() < f64::EPSILON {
            1.0
        } else {
            actual / predicted
        };

        // Update trust-region radius.
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 && (step_norm - delta).abs() < 1.0e-8 * delta {
            delta = (2.0 * delta).min(config.delta_max);
        }

        // Accept or reject the step.
        if rho > config.eta {
            beta = beta_new;
            last_log_lik = -neg_ll_new;
        }

        // Guard against degenerate radius.
        if delta < 1.0e-15 {
            break;
        }
    }

    // Final convergence check.
    if !converged {
        let (neg_ll_f, neg_score_f, _) = neg_loglik_grad_hess(&ds, &beta, tie_method)?;
        last_log_lik = -neg_ll_f;
        let g_inf = neg_score_f.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if g_inf < config.tol {
            converged = true;
        }
    }

    Ok(TrustRegionResult {
        coef: beta,
        n_outer_iters,
        converged,
        log_lik: last_log_lik,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn make_data(n: usize, beta_true: f64, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let lambda = (beta_true * x).exp();
                let t = rng.next_exponential(lambda).max(1.0e-6);
                (t, true, vec![x])
            })
            .collect()
    }

    fn make_data_censored(n: usize, beta_true: f64, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let lambda = (beta_true * x).exp();
                let t = rng.next_exponential(lambda).max(1.0e-6);
                let c = rng.next_exponential(0.5).max(1.0e-6);
                let event = t <= c;
                (t.min(c), event, vec![x])
            })
            .collect()
    }

    // ---------------------------------------------------------------------------
    // steihaug_cg tests
    // ---------------------------------------------------------------------------

    #[test]
    fn steihaug_cg_identity_hessian_recovers_newton_step() {
        // H = I, g = [2, 3] → Newton step = -H^{-1} g = [-2, -3].
        let g = vec![2.0_f64, 3.0_f64];
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let delta = 100.0; // large enough not to hit boundary
        let step = steihaug_cg(&g, &h, delta, 50, 1.0e-10).expect("ok");
        assert!((step[0] - (-2.0)).abs() < 1.0e-8, "step[0]={}", step[0]);
        assert!((step[1] - (-3.0)).abs() < 1.0e-8, "step[1]={}", step[1]);
    }

    #[test]
    fn steihaug_cg_hits_boundary_when_delta_small() {
        // H = I, g = [1, 0], delta = 0.5 → step must be clipped to ||p||=0.5.
        let g = vec![1.0_f64, 0.0_f64];
        let h = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let delta = 0.5;
        let step = steihaug_cg(&g, &h, delta, 50, 1.0e-10).expect("ok");
        let n = norm2(&step);
        assert!(
            (n - delta).abs() < 1.0e-8,
            "step norm={n}, expected delta={delta}"
        );
    }

    #[test]
    fn steihaug_cg_negative_curvature_steps_to_boundary() {
        // H = diag(-1, 1): negative curvature in first direction.
        let g = vec![0.0_f64, 1.0_f64];
        let h = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        let delta = 2.0;
        let step = steihaug_cg(&g, &h, delta, 50, 1.0e-10).expect("ok");
        // Step must satisfy ||step|| <= delta.
        let n = norm2(&step);
        assert!(n <= delta + 1.0e-8, "step norm={n} > delta={delta}");
    }

    #[test]
    fn steihaug_cg_zero_gradient_returns_zero_step() {
        let g = vec![0.0_f64, 0.0_f64];
        let h = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let step = steihaug_cg(&g, &h, 10.0, 50, 1.0e-10).expect("ok");
        assert!(norm2(&step) < 1.0e-12);
    }

    #[test]
    fn steihaug_cg_step_inside_delta() {
        // Large delta → full Newton step should be accepted.
        let g = vec![4.0_f64, 2.0_f64];
        let h = vec![vec![4.0, 0.0], vec![0.0, 2.0]];
        let delta = 100.0;
        let step = steihaug_cg(&g, &h, delta, 100, 1.0e-12).expect("ok");
        // Newton step = H^{-1}(-g) = [-1, -1].
        assert!((step[0] - (-1.0)).abs() < 1.0e-6, "step[0]={}", step[0]);
        assert!((step[1] - (-1.0)).abs() < 1.0e-6, "step[1]={}", step[1]);
    }

    #[test]
    fn steihaug_cg_dimension_mismatch_error() {
        let g = vec![1.0_f64, 2.0_f64];
        let h = vec![vec![1.0]]; // wrong size
        let err = steihaug_cg(&g, &h, 1.0, 10, 1.0e-6);
        assert!(err.is_err());
    }

    // ---------------------------------------------------------------------------
    // trust_region_cox tests
    // ---------------------------------------------------------------------------

    #[test]
    fn trust_region_cox_empty_data_error() {
        let err = trust_region_cox(&[], &TrustRegionConfig::default(), TieMethod::Breslow);
        assert!(matches!(err, Err(SurvivalError::EmptyDataset)));
    }

    #[test]
    fn trust_region_cox_no_events_error() {
        let data: Vec<(f64, bool, Vec<f64>)> =
            vec![(1.0, false, vec![0.5]), (2.0, false, vec![-0.5])];
        let err = trust_region_cox(&data, &TrustRegionConfig::default(), TieMethod::Breslow);
        assert!(matches!(err, Err(SurvivalError::NoEvents)));
    }

    #[test]
    fn trust_region_cox_converges_on_synthetic_data() {
        let data = make_data(200, 0.5, 42);
        let config = TrustRegionConfig::default();
        let result = trust_region_cox(&data, &config, TieMethod::Breslow).expect("ok");
        assert!(result.converged, "did not converge");
        assert!(result.coef[0] > 0.0, "coef sign wrong");
    }

    #[test]
    fn trust_region_cox_coef_sign_matches_breslow() {
        // beta_true=1 → risk increases with x → β̂ should be positive.
        let data = make_data(300, 1.0, 77);
        let result =
            trust_region_cox(&data, &TrustRegionConfig::default(), TieMethod::Breslow).expect("ok");
        assert!(result.coef[0] > 0.2, "coef={}", result.coef[0]);
    }

    #[test]
    fn trust_region_cox_efron_ties_also_converges() {
        let data = make_data(150, -0.6, 99);
        let result =
            trust_region_cox(&data, &TrustRegionConfig::default(), TieMethod::Efron).expect("ok");
        assert!(result.converged);
        assert!(result.coef[0] < 0.0, "negative beta_true → negative coef");
    }

    #[test]
    fn trust_region_cox_log_lik_is_finite() {
        let data = make_data(100, 0.3, 111);
        let result =
            trust_region_cox(&data, &TrustRegionConfig::default(), TieMethod::Breslow).expect("ok");
        assert!(result.log_lik.is_finite());
    }

    #[test]
    fn trust_region_cox_censored_data_ok() {
        let data = make_data_censored(200, 0.8, 222);
        let result =
            trust_region_cox(&data, &TrustRegionConfig::default(), TieMethod::Breslow).expect("ok");
        assert!(result.coef[0] > 0.0);
    }

    #[test]
    fn trust_region_cox_iters_within_bound() {
        let data = make_data(100, 0.5, 333);
        let config = TrustRegionConfig {
            max_outer: 200,
            ..Default::default()
        };
        let result = trust_region_cox(&data, &config, TieMethod::Breslow).expect("ok");
        assert!(result.n_outer_iters <= 200);
    }
}
