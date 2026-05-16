//! Log-space forward-backward for discrete HMMs and Gaussian HMMs.

use super::hmm::{HmmDiscrete, HmmGaussian, log_safe};
use crate::error::{SeqError, SeqResult};

/// Result of forward-backward inference.
#[derive(Debug, Clone)]
pub struct ForwardBackward {
    /// log α (T × n_states)
    pub log_alpha: Vec<f64>,
    /// log β (T × n_states)
    pub log_beta: Vec<f64>,
    /// γ (T × n_states): state posteriors
    pub gamma: Vec<f64>,
    /// ξ ((T-1) × n_states × n_states): edge posteriors
    pub xi: Vec<f64>,
    /// log p(o₁…o_T | model)
    pub log_likelihood: f64,
}

/// log-sum-exp on a slice; handles `-∞` cleanly.
#[inline]
pub fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// Run forward-backward on a discrete HMM.
pub fn forward_backward(hmm: &HmmDiscrete, obs: &[usize]) -> SeqResult<ForwardBackward> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = obs.len();
    let n = hmm.n_states;

    // Pre-compute log-emissions per (t, j).
    let mut log_em = vec![f64::NEG_INFINITY; t_max * n];
    for t in 0..t_max {
        for j in 0..n {
            log_em[t * n + j] = hmm.log_emission(j, obs[t])?;
        }
    }

    // Pre-compute log-transitions and log-init.
    let mut log_a = vec![f64::NEG_INFINITY; n * n];
    for i in 0..n {
        for j in 0..n {
            log_a[i * n + j] = log_safe(hmm.a[i * n + j]);
        }
    }
    let log_pi: Vec<f64> = hmm.pi.iter().map(|&p| log_safe(p)).collect();

    forward_backward_log(&log_pi, &log_a, &log_em, n, t_max)
}

/// Run forward-backward on a Gaussian HMM with observation sequence `x`
/// (T × dim row-major).
pub fn forward_backward_gaussian(hmm: &HmmGaussian, x: &[f64]) -> SeqResult<ForwardBackward> {
    if x.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if x.len() % hmm.dim != 0 {
        return Err(SeqError::DimensionMismatch {
            a: x.len(),
            b: hmm.dim,
        });
    }
    let t_max = x.len() / hmm.dim;
    let n = hmm.n_states;
    let mut log_em = vec![f64::NEG_INFINITY; t_max * n];
    for t in 0..t_max {
        let row = &x[t * hmm.dim..(t + 1) * hmm.dim];
        for j in 0..n {
            log_em[t * n + j] = hmm.log_emission(j, row)?;
        }
    }
    let mut log_a = vec![f64::NEG_INFINITY; n * n];
    for i in 0..n {
        for j in 0..n {
            log_a[i * n + j] = log_safe(hmm.a[i * n + j]);
        }
    }
    let log_pi: Vec<f64> = hmm.pi.iter().map(|&p| log_safe(p)).collect();
    forward_backward_log(&log_pi, &log_a, &log_em, n, t_max)
}

/// Generic log-space forward-backward given pre-computed log arrays.
fn forward_backward_log(
    log_pi: &[f64],
    log_a: &[f64],
    log_em: &[f64],
    n: usize,
    t_max: usize,
) -> SeqResult<ForwardBackward> {
    let mut log_alpha = vec![f64::NEG_INFINITY; t_max * n];
    let mut log_beta = vec![f64::NEG_INFINITY; t_max * n];

    // α₀(j) = log π_j + log B_j(o₀)
    for j in 0..n {
        log_alpha[j] = log_pi[j] + log_em[j];
    }

    // α_t(j) = logsumexp_i(α_{t-1}(i) + log A_{ij}) + log B_j(o_t)
    let mut tmp = vec![0.0; n];
    for t in 1..t_max {
        for j in 0..n {
            for i in 0..n {
                tmp[i] = log_alpha[(t - 1) * n + i] + log_a[i * n + j];
            }
            log_alpha[t * n + j] = logsumexp(&tmp) + log_em[t * n + j];
        }
    }

    // β_{T-1}(i) = 0
    for i in 0..n {
        log_beta[(t_max - 1) * n + i] = 0.0;
    }
    // β_t(i) = logsumexp_j(log A_{ij} + log B_j(o_{t+1}) + β_{t+1}(j))
    for t in (0..t_max - 1).rev() {
        for i in 0..n {
            for j in 0..n {
                tmp[j] = log_a[i * n + j] + log_em[(t + 1) * n + j] + log_beta[(t + 1) * n + j];
            }
            log_beta[t * n + i] = logsumexp(&tmp);
        }
    }

    // log_likelihood = logsumexp_j α_{T-1}(j)
    let last_alpha = &log_alpha[(t_max - 1) * n..t_max * n];
    let ll = logsumexp(last_alpha);

    // γ_t(i) = α_t(i) + β_t(i) − ll
    let mut gamma = vec![0.0; t_max * n];
    for t in 0..t_max {
        for i in 0..n {
            gamma[t * n + i] = (log_alpha[t * n + i] + log_beta[t * n + i] - ll).exp();
        }
        // Re-normalise to keep numerical purity
        let s: f64 = gamma[t * n..t * n + n].iter().sum();
        if s > 0.0 {
            for i in 0..n {
                gamma[t * n + i] /= s;
            }
        }
    }

    // ξ_t(i,j) = α_t(i) + log A_{ij} + log B_j(o_{t+1}) + β_{t+1}(j) − ll
    let mut xi = vec![0.0; (t_max.saturating_sub(1)) * n * n];
    for t in 0..t_max.saturating_sub(1) {
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = (log_alpha[t * n + i]
                    + log_a[i * n + j]
                    + log_em[(t + 1) * n + j]
                    + log_beta[(t + 1) * n + j]
                    - ll)
                    .exp();
                xi[t * n * n + i * n + j] = v;
                s += v;
            }
        }
        if s > 0.0 {
            for v in xi[t * n * n..(t + 1) * n * n].iter_mut() {
                *v /= s;
            }
        }
    }

    Ok(ForwardBackward {
        log_alpha,
        log_beta,
        gamma,
        xi,
        log_likelihood: ll,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_hmm() -> HmmDiscrete {
        HmmDiscrete::new(
            2,
            2,
            vec![0.6, 0.4],
            vec![0.7, 0.3, 0.4, 0.6],
            vec![0.1, 0.9, 0.8, 0.2],
        )
        .expect("ok")
    }

    #[test]
    fn forward_alpha_dimensions() {
        let h = small_hmm();
        let fb = forward_backward(&h, &[0, 1, 0]).expect("ok");
        assert_eq!(fb.log_alpha.len(), 6);
        assert_eq!(fb.gamma.len(), 6);
        assert_eq!(fb.xi.len(), 8);
    }

    #[test]
    fn gamma_rows_sum_to_one() {
        let h = small_hmm();
        let fb = forward_backward(&h, &[0, 1, 0, 1]).expect("ok");
        for t in 0..4 {
            let s: f64 = fb.gamma[t * 2..(t + 1) * 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "γ_{t} sums to {s}");
        }
    }

    #[test]
    fn logsumexp_neg_inf() {
        let xs = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert!(logsumexp(&xs).is_infinite());
    }

    #[test]
    fn logsumexp_single() {
        let xs = vec![5.0];
        assert!((logsumexp(&xs) - 5.0).abs() < 1e-12);
    }
}
