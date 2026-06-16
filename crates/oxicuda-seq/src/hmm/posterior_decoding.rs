//! Posterior (max-marginal) decoding for discrete HMMs.
//!
//! Viterbi decoding returns the single jointly-most-probable state path
//! `argmax_y P(y | o)`.  **Posterior decoding** (also "maximum-marginal" or
//! "MPM — maximum posterior marginal" decoding) instead chooses, independently
//! at each time step, the state with the largest *marginal* posterior:
//!
//! ```text
//! ŷ_t = argmax_j  γ_t(j) ,   γ_t(j) = P(s_t = j | o_1…o_T)
//! ```
//!
//! This minimises the expected number of individually-mislabelled positions
//! (per-symbol error / Hamming risk), whereas Viterbi minimises the
//! whole-sequence 0/1 error.  The two can disagree, and the posterior path is
//! not guaranteed to be a *legal* path (it may traverse a zero-probability
//! transition), so we also expose [`posterior_path_is_feasible`] to check it.
//!
//! The marginals `γ` come from log-space forward-backward; this module also
//! returns the **expected number of correct labels** under the model, which is
//! a useful confidence proxy: `Σ_t max_j γ_t(j)`.

use super::forward_backward::forward_backward;
use super::hmm::HmmDiscrete;
use crate::error::{SeqError, SeqResult};

/// Result of posterior (max-marginal) decoding.
#[derive(Debug, Clone)]
pub struct PosteriorDecode {
    /// Per-position max-marginal labels, length `T`.
    pub path: Vec<usize>,
    /// The marginal posterior of the chosen label at each position, length `T`.
    pub marginal: Vec<f64>,
    /// State posteriors `γ` (`T × n_states`, row-major) for downstream use.
    pub gamma: Vec<f64>,
    /// Expected number of correctly-labelled positions, `Σ_t γ_t(ŷ_t)`.
    pub expected_correct: f64,
}

/// Decode an observation sequence by maximum posterior marginal.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] — if `obs` is empty.
/// * Propagates emission/transition errors from forward-backward (e.g. an
///   observation symbol out of range).
pub fn posterior_decode(hmm: &HmmDiscrete, obs: &[usize]) -> SeqResult<PosteriorDecode> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let n = hmm.n_states;
    let fb = forward_backward(hmm, obs)?;
    let t_max = obs.len();

    let mut path = vec![0usize; t_max];
    let mut marginal = vec![0.0_f64; t_max];
    let mut expected_correct = 0.0_f64;

    for t in 0..t_max {
        let row = &fb.gamma[t * n..t * n + n];
        let mut best = f64::NEG_INFINITY;
        let mut argmax = 0usize;
        for (j, &g) in row.iter().enumerate() {
            if g > best {
                best = g;
                argmax = j;
            }
        }
        path[t] = argmax;
        marginal[t] = best;
        expected_correct += best;
    }

    Ok(PosteriorDecode {
        path,
        marginal,
        gamma: fb.gamma,
        expected_correct,
    })
}

/// Whether a decoded path is *feasible* under the model — i.e. every visited
/// transition has positive probability and the initial state has positive `π`.
///
/// Posterior decoding can yield infeasible paths (Viterbi cannot); this lets a
/// caller detect that and fall back to Viterbi if a legal path is required.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`]       — if `path` is empty.
/// * [`SeqError::IndexOutOfBounds`] — if any state index `≥ n_states`.
pub fn posterior_path_is_feasible(hmm: &HmmDiscrete, path: &[usize]) -> SeqResult<bool> {
    if path.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let n = hmm.n_states;
    for &s in path {
        if s >= n {
            return Err(SeqError::IndexOutOfBounds { index: s, len: n });
        }
    }
    if hmm.pi[path[0]] <= 0.0 {
        return Ok(false);
    }
    for w in path.windows(2) {
        if hmm.a[w[0] * n + w[1]] <= 0.0 {
            return Ok(false);
        }
    }
    Ok(true)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple 2-state, 2-symbol HMM.
    fn small_hmm() -> HmmDiscrete {
        HmmDiscrete::new(
            2,
            2,
            vec![0.6, 0.4],
            vec![0.7, 0.3, 0.4, 0.6],
            vec![0.1, 0.9, 0.8, 0.2],
        )
        .expect("hmm")
    }

    /// A near-deterministic HMM whose states are strongly tied to symbols.
    fn deterministic_hmm() -> HmmDiscrete {
        HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.9, 0.1, 0.1, 0.9],
            // state 0 almost always emits symbol 0; state 1 → symbol 1.
            vec![0.99, 0.01, 0.01, 0.99],
        )
        .expect("hmm")
    }

    #[test]
    fn decode_rejects_empty() {
        let h = small_hmm();
        assert!(matches!(
            posterior_decode(&h, &[]),
            Err(SeqError::EmptyInput)
        ));
    }

    #[test]
    fn decode_shapes_correct() {
        let h = small_hmm();
        let d = posterior_decode(&h, &[0, 1, 0, 1]).expect("ok");
        assert_eq!(d.path.len(), 4);
        assert_eq!(d.marginal.len(), 4);
        assert_eq!(d.gamma.len(), 4 * 2);
    }

    #[test]
    fn marginals_are_probabilities() {
        let h = small_hmm();
        let d = posterior_decode(&h, &[0, 1, 1, 0]).expect("ok");
        for &m in &d.marginal {
            assert!((0.0..=1.0).contains(&m), "marginal {m} out of [0,1]");
        }
    }

    #[test]
    fn chosen_label_is_argmax_of_gamma() {
        let h = small_hmm();
        let d = posterior_decode(&h, &[0, 1, 0]).expect("ok");
        let n = h.n_states;
        for t in 0..3 {
            let row = &d.gamma[t * n..t * n + n];
            let true_arg = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
                .map(|(i, _)| i)
                .expect("nonempty");
            assert_eq!(d.path[t], true_arg);
            assert!((d.marginal[t] - row[true_arg]).abs() < 1e-12);
        }
    }

    #[test]
    fn gamma_rows_sum_to_one() {
        let h = small_hmm();
        let d = posterior_decode(&h, &[0, 0, 1, 1]).expect("ok");
        let n = h.n_states;
        for t in 0..4 {
            let s: f64 = d.gamma[t * n..t * n + n].iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn expected_correct_bounds() {
        let h = small_hmm();
        let obs = [0, 1, 0, 1, 0];
        let d = posterior_decode(&h, &obs).expect("ok");
        // Σ_t max_j γ ≥ T/n_states (uniform lower bound) and ≤ T.
        assert!(d.expected_correct <= obs.len() as f64 + 1e-9);
        assert!(d.expected_correct >= obs.len() as f64 / 2.0 - 1e-9);
        // Also equals the sum of the per-position marginals.
        let s: f64 = d.marginal.iter().sum();
        assert!((d.expected_correct - s).abs() < 1e-12);
    }

    #[test]
    fn deterministic_recovers_symbol_states() {
        // With near-deterministic emissions, the max-marginal state at each
        // position matches the emitted symbol.
        let h = deterministic_hmm();
        let obs = [0usize, 1, 0, 1, 1, 0];
        let d = posterior_decode(&h, &obs).expect("ok");
        for (t, &o) in obs.iter().enumerate() {
            assert_eq!(d.path[t], o, "pos {t}: expected state {o}");
            // Each marginal is the winner over two states, hence ≥ 0.5.
            assert!(d.marginal[t] >= 0.5, "winner marginal must be ≥ 0.5");
        }
        // On average the model is highly confident on this clean sequence.
        let mean_conf: f64 = d.marginal.iter().sum::<f64>() / obs.len() as f64;
        assert!(
            mean_conf > 0.8,
            "mean confidence {mean_conf} should be high"
        );
    }

    #[test]
    fn feasible_path_check_accepts_valid() {
        let h = small_hmm();
        // All transitions are positive in this HMM, so any path is feasible.
        assert!(posterior_path_is_feasible(&h, &[0, 1, 0, 1]).expect("ok"));
    }

    #[test]
    fn feasible_path_check_rejects_zero_transition() {
        // Build an HMM with a forbidden 0→0 self-transition.
        let h = HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.0, 1.0, 1.0, 0.0], // A[0,0] = 0
            vec![0.6, 0.4, 0.4, 0.6],
        )
        .expect("hmm");
        assert!(!posterior_path_is_feasible(&h, &[0, 0]).expect("ok"));
        assert!(posterior_path_is_feasible(&h, &[0, 1]).expect("ok"));
    }

    #[test]
    fn feasible_rejects_zero_initial() {
        let h = HmmDiscrete::new(
            2,
            2,
            vec![1.0, 0.0], // state 1 has zero prior
            vec![0.5, 0.5, 0.5, 0.5],
            vec![0.6, 0.4, 0.4, 0.6],
        )
        .expect("hmm");
        assert!(!posterior_path_is_feasible(&h, &[1, 0]).expect("ok"));
    }

    #[test]
    fn feasible_rejects_empty_and_oob() {
        let h = small_hmm();
        assert!(matches!(
            posterior_path_is_feasible(&h, &[]),
            Err(SeqError::EmptyInput)
        ));
        assert!(matches!(
            posterior_path_is_feasible(&h, &[0, 5]),
            Err(SeqError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn single_observation_decodes() {
        let h = small_hmm();
        let d = posterior_decode(&h, &[1]).expect("ok");
        assert_eq!(d.path.len(), 1);
        // γ_0 ∝ π_j B_j(o_0); the larger product wins.
        assert!(d.path[0] < 2);
    }

    #[test]
    fn out_of_range_symbol_errors() {
        let h = small_hmm();
        // symbol 5 ≥ n_obs=2 → emission lookup fails inside forward-backward.
        assert!(posterior_decode(&h, &[0, 5]).is_err());
    }
}
