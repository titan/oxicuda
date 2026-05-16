//! Log-space Viterbi decoding for discrete HMMs.

use super::hmm::{HmmDiscrete, log_safe};
use crate::error::{SeqError, SeqResult};

/// Viterbi decoding output.
#[derive(Debug, Clone)]
pub struct ViterbiResult {
    /// Maximum-likelihood state sequence.
    pub path: Vec<usize>,
    /// Score (log-probability) of the best path.
    pub log_score: f64,
}

/// Run the Viterbi algorithm.  Returns the most-likely state path and its
/// log-probability.
pub fn viterbi(hmm: &HmmDiscrete, obs: &[usize]) -> SeqResult<ViterbiResult> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = obs.len();
    let n = hmm.n_states;

    let mut log_em = vec![f64::NEG_INFINITY; t_max * n];
    for t in 0..t_max {
        for j in 0..n {
            log_em[t * n + j] = hmm.log_emission(j, obs[t])?;
        }
    }
    let mut log_a = vec![f64::NEG_INFINITY; n * n];
    for i in 0..n {
        for j in 0..n {
            log_a[i * n + j] = log_safe(hmm.a[i * n + j]);
        }
    }

    let mut delta = vec![f64::NEG_INFINITY; t_max * n];
    let mut psi = vec![0usize; t_max * n];

    // δ₀(j) = log π_j + log B_j(o₀)
    for j in 0..n {
        delta[j] = log_safe(hmm.pi[j]) + log_em[j];
    }

    for t in 1..t_max {
        for j in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut argmax = 0usize;
            for i in 0..n {
                let v = delta[(t - 1) * n + i] + log_a[i * n + j];
                if v > best {
                    best = v;
                    argmax = i;
                }
            }
            delta[t * n + j] = best + log_em[t * n + j];
            psi[t * n + j] = argmax;
        }
    }

    // Termination
    let mut best = f64::NEG_INFINITY;
    let mut last = 0usize;
    for j in 0..n {
        let v = delta[(t_max - 1) * n + j];
        if v > best {
            best = v;
            last = j;
        }
    }

    // Trace-back
    let mut path = vec![0usize; t_max];
    path[t_max - 1] = last;
    for t in (1..t_max).rev() {
        path[t - 1] = psi[t * n + path[t]];
    }

    Ok(ViterbiResult {
        path,
        log_score: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_two_state() {
        let h = HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.9, 0.1, 0.1, 0.9],
            vec![0.9, 0.1, 0.1, 0.9],
        )
        .expect("ok");
        // Observing [0, 0, 1, 1] — should prefer states [0,0,1,1] strongly.
        let r = viterbi(&h, &[0, 0, 1, 1]).expect("ok");
        assert_eq!(r.path, vec![0, 0, 1, 1]);
        assert!(r.log_score > -10.0);
    }

    #[test]
    fn viterbi_empty_errors() {
        let h = HmmDiscrete::new(1, 1, vec![1.0], vec![1.0], vec![1.0]).expect("ok");
        assert!(viterbi(&h, &[]).is_err());
    }
}
