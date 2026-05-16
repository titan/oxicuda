//! Baum-Welch (EM) parameter learning for discrete HMMs.

use super::forward_backward::forward_backward;
use super::hmm::HmmDiscrete;
use crate::error::{SeqError, SeqResult};

/// Result of Baum-Welch training: re-estimated HMM and the per-iteration log-likelihood trace.
#[derive(Debug, Clone)]
pub struct BaumWelchResult {
    pub model: HmmDiscrete,
    pub log_likelihoods: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Run Baum-Welch (single-sequence variant) until convergence.
///
/// * `init` — starting HMM
/// * `obs`  — observation sequence
/// * `max_iter` — maximum EM iterations
/// * `tol`  — convergence threshold on Δlog-likelihood
pub fn baum_welch_discrete(
    init: &HmmDiscrete,
    obs: &[usize],
    max_iter: usize,
    tol: f64,
) -> SeqResult<BaumWelchResult> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut model = init.clone();
    let n = model.n_states;
    let k = model.n_obs;
    let t_max = obs.len();

    let mut history: Vec<f64> = Vec::with_capacity(max_iter + 1);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter_used = 0;

    for it in 0..max_iter {
        iter_used = it + 1;
        let fb = forward_backward(&model, obs)?;
        history.push(fb.log_likelihood);

        // Check convergence
        if (fb.log_likelihood - prev_ll).abs() < tol && it > 0 {
            converged = true;
            break;
        }
        prev_ll = fb.log_likelihood;

        // M-step
        // π_i = γ₀(i)
        for i in 0..n {
            model.pi[i] = fb.gamma[i];
        }

        // A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i) (over t = 0..T-1, since ξ is T-1)
        for i in 0..n {
            let denom: f64 = (0..t_max - 1).map(|t| fb.gamma[t * n + i]).sum();
            for j in 0..n {
                let num: f64 = (0..t_max - 1).map(|t| fb.xi[t * n * n + i * n + j]).sum();
                model.a[i * n + j] = if denom > 1e-300 {
                    num / denom
                } else {
                    1.0 / n as f64
                };
            }
            // Re-normalise A row (defensive)
            let row_sum: f64 = model.a[i * n..i * n + n].iter().sum();
            if row_sum > 1e-300 {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v /= row_sum;
                }
            } else {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v = 1.0 / n as f64;
                }
            }
        }

        // B_j(k) = Σ_{t: o_t=k} γ_t(j) / Σ_t γ_t(j) (over all t)
        for j in 0..n {
            let denom: f64 = (0..t_max).map(|t| fb.gamma[t * n + j]).sum();
            for sym in 0..k {
                let num: f64 = (0..t_max)
                    .filter(|&t| obs[t] == sym)
                    .map(|t| fb.gamma[t * n + j])
                    .sum();
                model.b[j * k + sym] = if denom > 1e-300 {
                    num / denom
                } else {
                    1.0 / k as f64
                };
            }
            let row_sum: f64 = model.b[j * k..j * k + k].iter().sum();
            if row_sum > 1e-300 {
                for v in model.b[j * k..j * k + k].iter_mut() {
                    *v /= row_sum;
                }
            } else {
                for v in model.b[j * k..j * k + k].iter_mut() {
                    *v = 1.0 / k as f64;
                }
            }
        }

        // Re-normalise π
        let s: f64 = model.pi.iter().sum();
        if s > 1e-300 {
            for v in model.pi.iter_mut() {
                *v /= s;
            }
        } else {
            for v in model.pi.iter_mut() {
                *v = 1.0 / n as f64;
            }
        }
    }

    // Final likelihood for the trace
    let fb_final = forward_backward(&model, obs)?;
    history.push(fb_final.log_likelihood);

    Ok(BaumWelchResult {
        model,
        log_likelihoods: history,
        iterations: iter_used,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baum_welch_monotone_nondecreasing() {
        let init = HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.6, 0.4, 0.4, 0.6],
            vec![0.7, 0.3, 0.3, 0.7],
        )
        .expect("ok");
        let obs = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 0];
        let r = baum_welch_discrete(&init, &obs, 20, 1e-6).expect("ok");
        // Likelihood should be non-decreasing across iterations.
        for w in r.log_likelihoods.windows(2) {
            assert!(
                w[1] + 1e-6 >= w[0],
                "log-lik decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }
}
