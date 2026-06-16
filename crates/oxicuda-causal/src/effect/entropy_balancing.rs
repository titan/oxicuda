//! Entropy balancing (Hainmueller 2012 *Political Analysis* 20:25).
//!
//! Entropy balancing is a covariate-balancing reweighting scheme for estimating
//! the Average Treatment effect on the Treated (ATT). Instead of estimating a
//! propensity model and inverting it, it solves directly for the unit weights
//! `w_i` on the control sample that:
//!
//! 1. **exactly** reproduce the treated group's covariate moments
//!    `Σ_i w_i c_{ij} = m̄_j` (the balance constraints), and
//! 2. stay as close as possible to uniform base weights `q_i` in the
//!    Kullback–Leibler / maximum-entropy sense `min Σ w_i log(w_i / q_i)`.
//!
//! The Lagrangian dual is an unconstrained smooth convex problem in the
//! multipliers `λ ∈ R^J` (one per balanced moment). The weights have the
//! closed form
//!
//! ```text
//! w_i(λ) = q_i · exp(−Σ_j λ_j c_{ij}) / Z(λ)
//! ```
//!
//! and `λ` is found by Newton's method on the dual objective whose gradient is
//! exactly the moment imbalance `Σ_i w_i c_{ij} − m̄_j`. At the optimum the
//! constraints hold to numerical tolerance and the weights are guaranteed
//! non-negative and to sum to one.
//!
//! The reweighted control outcome mean is contrasted with the (unweighted)
//! treated mean to form the ATT.

use crate::error::{CausalError, CausalResult};

/// Configuration for entropy balancing.
#[derive(Debug, Clone)]
pub struct EntropyBalancingConfig {
    /// Maximum Newton iterations on the dual.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum absolute moment imbalance.
    pub tol: f32,
    /// Ridge added to the Hessian diagonal for numerical stability.
    pub ridge: f32,
    /// Also balance the second (centred) moments (variances) of each covariate.
    pub balance_variance: bool,
}

impl Default for EntropyBalancingConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            ridge: 1e-8,
            balance_variance: false,
        }
    }
}

/// Result of entropy balancing.
#[derive(Debug, Clone)]
pub struct EntropyBalancingResult {
    /// Estimated ATT.
    pub att: f32,
    /// Balancing weights on the control units (length = number of controls,
    /// in control-index order). Sum to 1.
    pub weights: Vec<f32>,
    /// Maximum absolute residual moment imbalance at the solution.
    pub max_imbalance: f32,
    /// Whether the dual Newton iteration converged within `max_iter`.
    pub converged: bool,
    /// Number of balanced moment constraints `J`.
    pub n_constraints: usize,
}

/// Solve a small symmetric linear system `A x = b` by Gaussian elimination with
/// partial pivoting, in `f64`. Returns `None` if the system is singular. Used by
/// the entropy-balancing dual Newton optimiser.
fn solve_linear_f64(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-300 {
            return None;
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
    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s -= a[col * n + c] * x[c];
        }
        x[col] = s / a[col * n + col];
    }
    Some(x)
}

/// Estimate the ATT by entropy balancing.
///
/// # Arguments
/// * `y` — outcomes, length `n`.
/// * `t` — binary treatment (`1.0` treated / `0.0` control), length `n`.
/// * `x` — covariates row-major `[n × d]`.
/// * `n`, `d` — sample size and covariate dimension.
/// * `config` — balancing configuration.
///
/// # Errors
/// Returns [`CausalError::EmptyInput`] for empty inputs,
/// [`CausalError::DimensionMismatch`] on a `[n×d]` size error,
/// [`CausalError::NotFitted`] if a treated or control group is empty, and
/// [`CausalError::MatrixSingular`] if the Newton Hessian becomes singular.
pub fn entropy_balancing(
    y: &[f32],
    t: &[f32],
    x: &[f32],
    n: usize,
    d: usize,
    config: &EntropyBalancingConfig,
) -> CausalResult<EntropyBalancingResult> {
    if n == 0 || d == 0 {
        return Err(CausalError::EmptyInput);
    }
    if y.len() != n || t.len() != n || x.len() != n * d {
        return Err(CausalError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let treated: Vec<usize> = (0..n).filter(|&i| t[i] > 0.5).collect();
    let controls: Vec<usize> = (0..n).filter(|&i| t[i] <= 0.5).collect();
    if treated.is_empty() || controls.is_empty() {
        return Err(CausalError::NotFitted);
    }

    // Number of balanced moments: d means (+ d variances if requested).
    let j = if config.balance_variance { 2 * d } else { d };

    // Build the control "constraint design" matrix C: rows = controls, cols = J.
    // First d columns are the raw covariates; optional next d are squared.
    let n_ctrl = controls.len();
    let mut c_mat = vec![0.0_f32; n_ctrl * j];
    for (row, &ci) in controls.iter().enumerate() {
        for col in 0..d {
            let v = x[ci * d + col];
            c_mat[row * j + col] = v;
            if config.balance_variance {
                c_mat[row * j + d + col] = v * v;
            }
        }
    }

    // Targets: treated-group moments m̄_j.
    let n_treat_f = treated.len() as f32;
    let mut target = vec![0.0_f32; j];
    for &ti in &treated {
        for col in 0..d {
            let v = x[ti * d + col];
            target[col] += v;
            if config.balance_variance {
                target[d + col] += v * v;
            }
        }
    }
    for tg in &mut target {
        *tg /= n_treat_f;
    }

    // The dual objective minimised in λ is the convex log-partition function
    //   L(λ) = log( Σ_i q_i exp(−λ·C_i) ) + λ·target,
    // whose gradient is `Σ_i w_i C_i − target` (the moment imbalance) and whose
    // weights are `w_i(λ) = q_i exp(−λ·C_i) / Z`. The optimisation is carried
    // out in f64 for accuracy (the f32 tolerance is 1e-5) with damped Newton
    // plus Armijo backtracking for global stability; weights are cast back to
    // f32 at the end.
    let c_mat64: Vec<f64> = c_mat.iter().map(|&v| v as f64).collect();
    let target64: Vec<f64> = target.iter().map(|&v| v as f64).collect();
    let q = 1.0_f64 / n_ctrl as f64;
    let log_q = q.ln();
    let tol64 = config.tol as f64;
    let ridge64 = config.ridge as f64;

    // Given λ, compute the normalised weights, the dual objective `L(λ)`, and
    // the gradient `grad = Σ w_i C_i − target` via a stable log-sum-exp.
    let eval = |lambda: &[f64]| -> Option<(Vec<f64>, f64, Vec<f64>)> {
        let mut logits = vec![0.0_f64; n_ctrl];
        let mut max_logit = f64::NEG_INFINITY;
        for (row, lg) in logits.iter_mut().enumerate() {
            let mut dot = 0.0_f64;
            for col in 0..j {
                dot += lambda[col] * c_mat64[row * j + col];
            }
            let v = log_q - dot;
            *lg = v;
            if v > max_logit {
                max_logit = v;
            }
        }
        if !max_logit.is_finite() {
            return None;
        }
        let mut sum_exp = 0.0_f64;
        for lg in &logits {
            sum_exp += (lg - max_logit).exp();
        }
        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return None;
        }
        let log_z = max_logit + sum_exp.ln();
        let mut weights = vec![0.0_f64; n_ctrl];
        for (row, w) in weights.iter_mut().enumerate() {
            *w = (logits[row] - log_z).exp();
        }
        let mut obj = log_z;
        for col in 0..j {
            obj += lambda[col] * target64[col];
        }
        let mut grad = vec![0.0_f64; j];
        for col in 0..j {
            let mut acc = 0.0_f64;
            for (row, &w) in weights.iter().enumerate() {
                acc += w * c_mat64[row * j + col];
            }
            grad[col] = acc - target64[col];
        }
        Some((weights, obj, grad))
    };

    let mut lambda = vec![0.0_f64; j];
    let (mut weights64, mut obj, mut grad) = eval(&lambda).ok_or(CausalError::MatrixSingular)?;
    let mut max_imbalance = grad.iter().fold(0.0_f64, |m, &g| m.max(g.abs()));

    for _iter in 0..config.max_iter {
        if max_imbalance < tol64 {
            break;
        }

        // Hessian H_jk = Σ_i w_i C_ij C_ik − (Σ_i w_i C_ij)(Σ_i w_i C_ik):
        // the weighted covariance of the moment functions (positive definite).
        let mut wmean = vec![0.0_f64; j];
        for col in 0..j {
            wmean[col] = grad[col] + target64[col]; // = Σ w_i C_ij
        }
        let mut hess = vec![0.0_f64; j * j];
        for (row, &wi) in weights64.iter().enumerate() {
            for a in 0..j {
                let ca = c_mat64[row * j + a];
                if ca == 0.0 {
                    continue;
                }
                for b in 0..j {
                    hess[a * j + b] += wi * ca * c_mat64[row * j + b];
                }
            }
        }
        for a in 0..j {
            for b in 0..j {
                hess[a * j + b] -= wmean[a] * wmean[b];
            }
            hess[a * j + a] += ridge64;
        }

        // Newton descent direction Δ = H⁻¹·grad (∇L = grad here).
        let mut rhs: Vec<f64> = grad.clone();
        let direction =
            solve_linear_f64(&mut hess, &mut rhs, j).ok_or(CausalError::MatrixSingular)?;

        // Armijo backtracking with an imbalance-decrease fallback.
        let dir_deriv: f64 = grad
            .iter()
            .zip(direction.iter())
            .map(|(&g, &d)| -g * d)
            .sum();
        let c_armijo = 1e-4_f64;
        let mut t_step = 1.0_f64;
        let mut accepted = false;
        for _ls in 0..50 {
            let trial: Vec<f64> = lambda
                .iter()
                .zip(direction.iter())
                .map(|(&l, &dlt)| l + t_step * dlt)
                .collect();
            if let Some((w_new, obj_new, grad_new)) = eval(&trial) {
                let imb_new = grad_new.iter().fold(0.0_f64, |m, &g| m.max(g.abs()));
                let armijo_ok = obj_new <= obj + c_armijo * t_step * dir_deriv;
                if obj_new.is_finite() && (armijo_ok || imb_new < max_imbalance) {
                    lambda = trial;
                    weights64 = w_new;
                    obj = obj_new;
                    grad = grad_new;
                    max_imbalance = imb_new;
                    accepted = true;
                    break;
                }
            }
            t_step *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    let weights: Vec<f32> = weights64.iter().map(|&w| w as f32).collect();
    let max_imbalance = max_imbalance as f32;
    let converged = max_imbalance < config.tol;

    // ATT = mean treated outcome − weighted mean control outcome.
    let treated_y_mean = treated.iter().map(|&i| y[i]).sum::<f32>() / n_treat_f;
    let weighted_ctrl_y: f32 = controls
        .iter()
        .zip(weights.iter())
        .map(|(&ci, &wi)| wi * y[ci])
        .sum();
    let att = treated_y_mean - weighted_ctrl_y;

    Ok(EntropyBalancingResult {
        att,
        weights,
        max_imbalance,
        converged,
        n_constraints: j,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Construct data where a single covariate confounds selection and outcome.
    /// Treated units have higher `x`; outcome `y = beta·x + tau·T + noise`.
    fn confounded_data(
        n: usize,
        beta: f32,
        tau: f32,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; n];
        let mut t = vec![0.0_f32; n];
        let mut x = vec![0.0_f32; n];
        for i in 0..n {
            let xi = rng.next_normal();
            x[i] = xi;
            // Moderate logistic selection: higher x → more likely treated, but
            // with substantial overlap so the treated covariate mean stays well
            // inside the control convex hull (entropy balancing is then feasible).
            let prob = 1.0 / (1.0 + (-0.8 * xi).exp());
            let treated = if rng.next_f32() < prob { 1.0 } else { 0.0 };
            t[i] = treated;
            y[i] = beta * xi + tau * treated + 0.05 * rng.next_normal();
        }
        (y, t, x)
    }

    #[test]
    fn eb_balances_means() {
        let mut rng = LcgRng::new(1);
        let (y, t, x) = confounded_data(200, 1.0, 2.0, &mut rng);
        let cfg = EntropyBalancingConfig::default();
        let r = entropy_balancing(&y, &t, &x, 200, 1, &cfg).expect("ok");
        assert!(
            r.max_imbalance < 1e-3,
            "covariate means not balanced: imbalance {}",
            r.max_imbalance
        );
    }

    #[test]
    fn eb_weights_sum_to_one() {
        let mut rng = LcgRng::new(2);
        let (y, t, x) = confounded_data(120, 1.0, 1.0, &mut rng);
        let r =
            entropy_balancing(&y, &t, &x, 120, 1, &EntropyBalancingConfig::default()).expect("ok");
        let s: f32 = r.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "weights sum {s} != 1");
    }

    #[test]
    fn eb_weights_nonnegative() {
        let mut rng = LcgRng::new(3);
        let (y, t, x) = confounded_data(120, 1.0, 1.0, &mut rng);
        let r =
            entropy_balancing(&y, &t, &x, 120, 1, &EntropyBalancingConfig::default()).expect("ok");
        assert!(r.weights.iter().all(|&w| w >= 0.0), "negative weight found");
    }

    #[test]
    fn eb_reduces_confounding_bias() {
        // Naive ATT (difference in means) is biased upward by the confounder;
        // entropy balancing should be closer to the true tau.
        let mut rng = LcgRng::new(4);
        let n = 300;
        let (y, t, x) = confounded_data(n, 1.5, 2.0, &mut rng);
        let r =
            entropy_balancing(&y, &t, &x, n, 1, &EntropyBalancingConfig::default()).expect("ok");
        let ty: f32 = (0..n).filter(|&i| t[i] > 0.5).map(|i| y[i]).sum();
        let tn = (0..n).filter(|&i| t[i] > 0.5).count() as f32;
        let cy: f32 = (0..n).filter(|&i| t[i] <= 0.5).map(|i| y[i]).sum();
        let cn = (0..n).filter(|&i| t[i] <= 0.5).count() as f32;
        let naive = ty / tn - cy / cn;
        assert!(
            (r.att - 2.0).abs() < (naive - 2.0).abs(),
            "EB ATT {} not closer to 2.0 than naive {naive}",
            r.att
        );
    }

    #[test]
    fn eb_recovers_tau_approx() {
        let mut rng = LcgRng::new(5);
        let n = 400;
        let (y, t, x) = confounded_data(n, 1.0, 3.0, &mut rng);
        let r =
            entropy_balancing(&y, &t, &x, n, 1, &EntropyBalancingConfig::default()).expect("ok");
        assert!(
            (r.att - 3.0).abs() < 0.6,
            "ATT {} should be near 3.0",
            r.att
        );
    }

    #[test]
    fn eb_multivariate_balances_all() {
        let mut rng = LcgRng::new(6);
        let n = 200;
        let d = 3;
        let mut x = vec![0.0_f32; n * d];
        let mut t = vec![0.0_f32; n];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let mut s = 0.0_f32;
            for k in 0..d {
                let v = rng.next_normal();
                x[i * d + k] = v;
                s += v;
            }
            let prob = 1.0 / (1.0 + (-0.5 * s).exp());
            t[i] = if rng.next_f32() < prob { 1.0 } else { 0.0 };
            y[i] = s + 2.0 * t[i] + 0.05 * rng.next_normal();
        }
        let r =
            entropy_balancing(&y, &t, &x, n, d, &EntropyBalancingConfig::default()).expect("ok");
        assert_eq!(r.n_constraints, d);
        assert!(r.max_imbalance < 1e-3, "imbalance {}", r.max_imbalance);
    }

    #[test]
    fn eb_variance_balancing_doubles_constraints() {
        let mut rng = LcgRng::new(7);
        let n = 200;
        let d = 2;
        let (y, t, x) = {
            let mut x = vec![0.0_f32; n * d];
            let mut t = vec![0.0_f32; n];
            let mut y = vec![0.0_f32; n];
            for i in 0..n {
                let a = rng.next_normal();
                let b = rng.next_normal();
                x[i * d] = a;
                x[i * d + 1] = b;
                t[i] = if a + b > 0.0 { 1.0 } else { 0.0 };
                y[i] = a + b + 1.0 * t[i];
            }
            (y, t, x)
        };
        let cfg = EntropyBalancingConfig {
            balance_variance: true,
            ..EntropyBalancingConfig::default()
        };
        let r = entropy_balancing(&y, &t, &x, n, d, &cfg).expect("ok");
        assert_eq!(r.n_constraints, 2 * d);
    }

    #[test]
    fn eb_empty_errors() {
        let cfg = EntropyBalancingConfig::default();
        assert!(matches!(
            entropy_balancing(&[], &[], &[], 0, 0, &cfg),
            Err(CausalError::EmptyInput)
        ));
    }

    #[test]
    fn eb_dim_mismatch_errors() {
        let cfg = EntropyBalancingConfig::default();
        let y = vec![1.0, 2.0];
        let t = vec![1.0, 0.0];
        let x = vec![0.5]; // should be 2*1 = 2
        assert!(matches!(
            entropy_balancing(&y, &t, &x, 2, 1, &cfg),
            Err(CausalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn eb_no_treated_errors() {
        let cfg = EntropyBalancingConfig::default();
        let y = vec![1.0, 2.0, 3.0];
        let t = vec![0.0, 0.0, 0.0];
        let x = vec![0.1, 0.2, 0.3];
        assert!(matches!(
            entropy_balancing(&y, &t, &x, 3, 1, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn eb_no_control_errors() {
        let cfg = EntropyBalancingConfig::default();
        let y = vec![1.0, 2.0, 3.0];
        let t = vec![1.0, 1.0, 1.0];
        let x = vec![0.1, 0.2, 0.3];
        assert!(matches!(
            entropy_balancing(&y, &t, &x, 3, 1, &cfg),
            Err(CausalError::NotFitted)
        ));
    }

    #[test]
    fn eb_converges_flag_set() {
        let mut rng = LcgRng::new(8);
        let (y, t, x) = confounded_data(150, 1.0, 1.0, &mut rng);
        let r =
            entropy_balancing(&y, &t, &x, 150, 1, &EntropyBalancingConfig::default()).expect("ok");
        assert!(r.converged, "expected convergence on well-posed problem");
    }

    #[test]
    fn eb_solve_linear_identity() {
        // 2x2 identity system.
        let mut a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let mut b = vec![3.0_f64, -2.0];
        let x = solve_linear_f64(&mut a, &mut b, 2).expect("solvable");
        assert!((x[0] - 3.0).abs() < 1e-9 && (x[1] + 2.0).abs() < 1e-9);
    }

    #[test]
    fn eb_solve_linear_singular_none() {
        let mut a = vec![1.0_f64, 2.0, 2.0, 4.0]; // rank 1
        let mut b = vec![1.0_f64, 2.0];
        assert!(solve_linear_f64(&mut a, &mut b, 2).is_none());
    }

    #[test]
    fn eb_weighted_control_mean_matches_treated_x() {
        // After balancing, the *weighted* control covariate mean equals the
        // treated covariate mean (the defining property).
        let mut rng = LcgRng::new(9);
        let n = 200;
        let (_, t, x) = confounded_data(n, 1.0, 1.0, &mut rng);
        let y = vec![0.0_f32; n]; // outcome irrelevant for this check
        let r =
            entropy_balancing(&y, &t, &x, n, 1, &EntropyBalancingConfig::default()).expect("ok");
        let treated_mean = {
            let s: f32 = (0..n).filter(|&i| t[i] > 0.5).map(|i| x[i]).sum();
            let c = (0..n).filter(|&i| t[i] > 0.5).count() as f32;
            s / c
        };
        let controls: Vec<usize> = (0..n).filter(|&i| t[i] <= 0.5).collect();
        let weighted_ctrl_mean: f32 = controls
            .iter()
            .zip(r.weights.iter())
            .map(|(&ci, &w)| w * x[ci])
            .sum();
        assert!(
            (treated_mean - weighted_ctrl_mean).abs() < 1e-3,
            "weighted control mean {weighted_ctrl_mean} != treated mean {treated_mean}"
        );
    }
}
