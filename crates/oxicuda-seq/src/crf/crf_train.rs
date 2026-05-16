//! CRF training: log-likelihood + gradient via forward-backward in score space,
//! plus a limited-memory BFGS (L-BFGS) optimiser with backtracking line search.

use super::linear_chain_crf::LinearChainCrf;
use crate::error::{SeqError, SeqResult};
use crate::hmm::forward_backward::logsumexp;

/// L-BFGS configuration.
#[derive(Debug, Clone)]
pub struct LbfgsConfig {
    /// Number of (s,y) history pairs retained.
    pub memory: usize,
    /// Maximum optimiser iterations.
    pub max_iter: usize,
    /// Gradient-norm convergence tolerance.
    pub grad_tol: f64,
    /// Line-search backtracking factor (multiplicative shrink).
    pub backtrack: f64,
    /// Maximum line-search trials per iteration.
    pub max_line_search: usize,
    /// L2 regularisation strength on parameters.
    pub l2: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 5,
            max_iter: 50,
            grad_tol: 1e-6,
            backtrack: 0.5,
            max_line_search: 30,
            l2: 1e-3,
        }
    }
}

/// Forward in score space.  α_t(j) = logsumexp_i(α_{t-1}(i) + tr[i,j]) + emit_t(j).
fn forward_scores(crf: &LinearChainCrf, emit: &[f64]) -> Vec<f64> {
    let n = crf.n_labels;
    let t_max = emit.len() / n;
    let mut alpha = vec![f64::NEG_INFINITY; t_max * n];
    alpha[..n].copy_from_slice(&emit[..n]);
    let mut tmp = vec![0.0; n];
    for t in 1..t_max {
        for j in 0..n {
            for i in 0..n {
                tmp[i] = alpha[(t - 1) * n + i] + crf.transitions[i * n + j];
            }
            alpha[t * n + j] = logsumexp(&tmp) + emit[t * n + j];
        }
    }
    alpha
}

/// Backward in score space.  β_{T-1}(i) = 0;
/// β_t(i) = logsumexp_j(tr[i,j] + emit_{t+1}(j) + β_{t+1}(j)).
fn backward_scores(crf: &LinearChainCrf, emit: &[f64]) -> Vec<f64> {
    let n = crf.n_labels;
    let t_max = emit.len() / n;
    let mut beta = vec![0.0; t_max * n];
    let mut tmp = vec![0.0; n];
    for t in (0..t_max - 1).rev() {
        for i in 0..n {
            for j in 0..n {
                tmp[j] = crf.transitions[i * n + j] + emit[(t + 1) * n + j] + beta[(t + 1) * n + j];
            }
            beta[t * n + i] = logsumexp(&tmp);
        }
    }
    beta
}

/// Compute log-likelihood and gradient for a single (x, y) example.
///
/// Returns `(log_likelihood, grad_emissions, grad_transitions)` where the gradient
/// has the **same sign as the objective** — i.e. an *ascent* direction.
pub fn crf_log_likelihood_and_gradient(
    crf: &LinearChainCrf,
    x: &[f64],
    y: &[usize],
) -> SeqResult<(f64, Vec<f64>, Vec<f64>)> {
    let n = crf.n_labels;
    let k = crf.n_features;
    if y.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = y.len();
    if x.len() != t_max * k {
        return Err(SeqError::ShapeMismatch {
            expected: t_max * k,
            got: x.len(),
        });
    }

    // Pre-compute emission scores for every (t, j).
    let mut emit = vec![0.0; t_max * n];
    for t in 0..t_max {
        for j in 0..n {
            emit[t * n + j] = crf.emit_score(j, &x[t * k..(t + 1) * k])?;
        }
    }

    let alpha = forward_scores(crf, &emit);
    let beta = backward_scores(crf, &emit);

    // log Z(x) = logsumexp_j α_{T-1}(j)
    let last_alpha = &alpha[(t_max - 1) * n..];
    let log_z = logsumexp(last_alpha);

    // score(y, x)
    let true_score = crf.sequence_score(x, y)?;
    let ll = true_score - log_z;

    // Marginals: p_t(j) ∝ exp(α_t(j) + β_t(j) − log Z)
    let mut p_node = vec![0.0; t_max * n];
    for t in 0..t_max {
        for j in 0..n {
            p_node[t * n + j] = (alpha[t * n + j] + beta[t * n + j] - log_z).exp();
        }
        let s: f64 = p_node[t * n..t * n + n].iter().sum();
        if s > 0.0 {
            for v in p_node[t * n..t * n + n].iter_mut() {
                *v /= s;
            }
        }
    }
    // Edge marginals: p_t(i,j) ∝ exp(α_t(i) + tr[i,j] + emit_{t+1}(j) + β_{t+1}(j) − log Z)
    let mut p_edge = vec![0.0; t_max.saturating_sub(1) * n * n];
    for t in 0..t_max.saturating_sub(1) {
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = (alpha[t * n + i]
                    + crf.transitions[i * n + j]
                    + emit[(t + 1) * n + j]
                    + beta[(t + 1) * n + j]
                    - log_z)
                    .exp();
                p_edge[t * n * n + i * n + j] = v;
                s += v;
            }
        }
        if s > 0.0 {
            for v in p_edge[t * n * n..(t + 1) * n * n].iter_mut() {
                *v /= s;
            }
        }
    }

    // Gradient = empirical features − expected features
    let mut grad_emit = vec![0.0; n * k];
    let mut grad_trans = vec![0.0; n * n];

    // Empirical: +1 at observed pairs
    for t in 0..t_max {
        let yt = y[t];
        for f in 0..k {
            grad_emit[yt * k + f] += x[t * k + f];
        }
        if t > 0 {
            grad_trans[y[t - 1] * n + y[t]] += 1.0;
        }
    }
    // Expected: -p
    for t in 0..t_max {
        for j in 0..n {
            let p = p_node[t * n + j];
            for f in 0..k {
                grad_emit[j * k + f] -= p * x[t * k + f];
            }
        }
        if t < t_max - 1 {
            for i in 0..n {
                for j in 0..n {
                    grad_trans[i * n + j] -= p_edge[t * n * n + i * n + j];
                }
            }
        }
    }

    Ok((ll, grad_emit, grad_trans))
}

/// Aggregate log-likelihood and gradient over a dataset, with L2 regularisation.
fn objective_and_grad(
    crf: &LinearChainCrf,
    examples: &[(Vec<f64>, Vec<usize>)],
    l2: f64,
) -> SeqResult<(f64, Vec<f64>)> {
    let mut total_ll = 0.0;
    let mut g_emit = vec![0.0; crf.emissions.len()];
    let mut g_trans = vec![0.0; crf.transitions.len()];
    for (x, y) in examples {
        let (ll, ge, gt) = crf_log_likelihood_and_gradient(crf, x, y)?;
        total_ll += ll;
        for (a, b) in g_emit.iter_mut().zip(ge.iter()) {
            *a += *b;
        }
        for (a, b) in g_trans.iter_mut().zip(gt.iter()) {
            *a += *b;
        }
    }
    // L2 regularisation: −0.5 λ ||w||² to objective; gradient −= λ w
    let mut reg = 0.0;
    for &e in &crf.emissions {
        reg += e * e;
    }
    for &t in &crf.transitions {
        reg += t * t;
    }
    total_ll -= 0.5 * l2 * reg;
    for (g, w) in g_emit.iter_mut().zip(crf.emissions.iter()) {
        *g -= l2 * *w;
    }
    for (g, w) in g_trans.iter_mut().zip(crf.transitions.iter()) {
        *g -= l2 * *w;
    }

    let mut grad = Vec::with_capacity(g_emit.len() + g_trans.len());
    grad.extend(g_emit);
    grad.extend(g_trans);
    Ok((total_ll, grad))
}

/// L-BFGS two-loop recursion direction computation.
///
/// Returns the search direction `d` (an *ascent* direction since we maximise
/// the log-likelihood).
fn lbfgs_direction(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho: &[f64],
) -> Vec<f64> {
    let m = s_hist.len();
    let n = grad.len();
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0; m];

    // First loop: i = m-1 .. 0
    for i in (0..m).rev() {
        let r = rho[i];
        let mut dot = 0.0;
        for k in 0..n {
            dot += s_hist[i][k] * q[k];
        }
        alpha[i] = r * dot;
        for k in 0..n {
            q[k] -= alpha[i] * y_hist[i][k];
        }
    }

    // Initial Hessian approximation: H₀ = γ I with γ = (s·y)/(y·y)
    let mut gamma = 1.0;
    if m > 0 {
        let last_s = &s_hist[m - 1];
        let last_y = &y_hist[m - 1];
        let mut sy = 0.0;
        let mut yy = 0.0;
        for k in 0..n {
            sy += last_s[k] * last_y[k];
            yy += last_y[k] * last_y[k];
        }
        if yy > 1e-30 {
            gamma = sy / yy;
        }
    }
    let mut r = q;
    for v in r.iter_mut() {
        *v *= gamma;
    }

    // Second loop: i = 0 .. m
    for i in 0..m {
        let mut dot = 0.0;
        for k in 0..n {
            dot += y_hist[i][k] * r[k];
        }
        let beta = rho[i] * dot;
        for k in 0..n {
            r[k] += s_hist[i][k] * (alpha[i] - beta);
        }
    }
    r
}

/// Train a linear-chain CRF by maximising the log-likelihood with L-BFGS.
///
/// Returns the final log-likelihood and updates `crf` in place.
pub fn train_crf_lbfgs(
    crf: &mut LinearChainCrf,
    examples: &[(Vec<f64>, Vec<usize>)],
    cfg: &LbfgsConfig,
) -> SeqResult<f64> {
    if examples.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let n_params = crf.param_count();
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.memory);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(cfg.memory);

    let (mut f_val, mut grad) = objective_and_grad(crf, examples, cfg.l2)?;

    for _it in 0..cfg.max_iter {
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < cfg.grad_tol {
            break;
        }

        // Compute direction.  Negative-gradient ascent search direction for the
        // first iteration when no history yet.
        let mut dir = if s_hist.is_empty() {
            grad.clone()
        } else {
            lbfgs_direction(&grad, &s_hist, &y_hist, &rho_hist)
        };

        // Make sure dir is an ascent direction (g·d > 0); otherwise flip to +grad
        let mut dot_gd: f64 = grad.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        if dot_gd <= 0.0 {
            dir = grad.clone();
            dot_gd = grad.iter().map(|g| g * g).sum();
        }
        // Normalise direction to a sane magnitude for the first step
        let dir_norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
        if dir_norm > 0.0 && s_hist.is_empty() {
            let scale = 1.0_f64 / dir_norm.max(1.0);
            for v in dir.iter_mut() {
                *v *= scale;
            }
        }

        // Backtracking line search (Armijo).
        let armijo = 1e-4_f64;
        let mut step = 1.0_f64;
        let p_old = crf.to_params();
        let mut accepted = false;
        let mut f_new = f_val;
        let mut grad_new = grad.clone();
        for _ls in 0..cfg.max_line_search {
            let mut p_try = p_old.clone();
            for k in 0..n_params {
                p_try[k] = p_old[k] + step * dir[k];
            }
            crf.from_params(&p_try)?;
            let (fc, gc) = objective_and_grad(crf, examples, cfg.l2)?;
            if fc >= f_val + armijo * step * dot_gd {
                f_new = fc;
                grad_new = gc;
                accepted = true;
                break;
            }
            step *= cfg.backtrack;
        }
        if !accepted {
            // Restore and stop
            crf.from_params(&p_old)?;
            return Ok(f_val);
        }

        // Update L-BFGS history
        let p_new = crf.to_params();
        let s_vec: Vec<f64> = p_new
            .iter()
            .zip(p_old.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        let y_vec: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        let ys: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if ys.abs() > 1e-30 {
            if s_hist.len() == cfg.memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / ys);
        }
        f_val = f_new;
        grad = grad_new;
    }

    Ok(f_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_finite_difference() {
        let mut crf = LinearChainCrf::zeros(2, 2).expect("ok");
        crf.emissions = vec![0.5, -0.3, 0.1, 0.4];
        crf.transitions = vec![0.2, -0.1, -0.4, 0.3];

        let x = vec![1.0, 0.5, 0.0, 1.0, 0.7, 0.2];
        let y = vec![0usize, 1, 0];

        let (ll0, ge, gt) = crf_log_likelihood_and_gradient(&crf, &x, &y).expect("ok");

        let eps = 1e-5;
        // Check emission gradient via finite difference.
        for idx in 0..crf.emissions.len() {
            let mut c2 = crf.clone();
            c2.emissions[idx] += eps;
            let (llp, _, _) = crf_log_likelihood_and_gradient(&c2, &x, &y).expect("ok");
            let mut c3 = crf.clone();
            c3.emissions[idx] -= eps;
            let (llm, _, _) = crf_log_likelihood_and_gradient(&c3, &x, &y).expect("ok");
            let num = (llp - llm) / (2.0 * eps);
            assert!(
                (num - ge[idx]).abs() < 1e-3,
                "emit grad {idx}: num={num}, ana={}",
                ge[idx]
            );
        }
        for idx in 0..crf.transitions.len() {
            let mut c2 = crf.clone();
            c2.transitions[idx] += eps;
            let (llp, _, _) = crf_log_likelihood_and_gradient(&c2, &x, &y).expect("ok");
            let mut c3 = crf.clone();
            c3.transitions[idx] -= eps;
            let (llm, _, _) = crf_log_likelihood_and_gradient(&c3, &x, &y).expect("ok");
            let num = (llp - llm) / (2.0 * eps);
            assert!(
                (num - gt[idx]).abs() < 1e-3,
                "trans grad {idx}: num={num}, ana={}",
                gt[idx]
            );
        }
        let _ = ll0; // sanity used
    }

    #[test]
    fn train_increases_likelihood() {
        let mut crf = LinearChainCrf::zeros(2, 2).expect("ok");
        let x1 = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let y1 = vec![0usize, 0, 1];
        let x2 = vec![0.0, 1.0, 1.0, 0.0];
        let y2 = vec![1usize, 0];
        let examples = vec![(x1, y1), (x2, y2)];
        let (ll0, _) = objective_and_grad(&crf, &examples, 0.0).expect("ok");
        let cfg = LbfgsConfig {
            memory: 3,
            max_iter: 20,
            grad_tol: 1e-8,
            backtrack: 0.5,
            max_line_search: 20,
            l2: 0.0,
        };
        let ll_final = train_crf_lbfgs(&mut crf, &examples, &cfg).expect("ok");
        assert!(ll_final >= ll0 - 1e-6, "ll0={ll0}, ll_final={ll_final}");
    }
}
