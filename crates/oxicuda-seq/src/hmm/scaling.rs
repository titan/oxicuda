use super::hmm::{HmmDiscrete, log_safe};
use crate::error::{SeqError, SeqResult};

/// Scaled forward variables and associated per-step scaling coefficients.
#[derive(Debug, Clone)]
pub struct ScaledForwardResult {
    /// T × n_states, scaled forward variables α_t(j).
    pub alpha: Vec<f64>,
    /// T scaling coefficients c_t (one per time step).
    pub scales: Vec<f64>,
    /// Σ_t log(1 / c_t) = -Σ_t log(c_t).
    pub log_likelihood: f64,
}

/// Scaled backward variables.
#[derive(Debug, Clone)]
pub struct ScaledBackwardResult {
    /// T × n_states, scaled backward variables β_t(i).
    pub beta: Vec<f64>,
}

/// Combined scaled forward-backward with posterior statistics.
#[derive(Debug, Clone)]
pub struct ScaledForwardBackwardResult {
    /// T × n_states scaled forward variables.
    pub alpha: Vec<f64>,
    /// T × n_states scaled backward variables.
    pub beta: Vec<f64>,
    /// T scaling coefficients.
    pub scales: Vec<f64>,
    /// T × n_states state posteriors γ_t(i) = P(q_t=i | O, λ).
    pub gamma: Vec<f64>,
    /// (T-1) × n_states × n_states edge posteriors ξ_t(i,j).
    pub xi: Vec<f64>,
    pub log_likelihood: f64,
}

/// Rabiner (1989) §VI: scaled forward pass for a discrete HMM.
///
/// Prevents arithmetic underflow on long sequences by normalising each time
/// step so that the α row sums to 1.  The per-step scale factors carry the
/// magnitude information needed to recover the log-likelihood exactly.
pub fn scaled_forward(hmm: &HmmDiscrete, obs: &[usize]) -> SeqResult<ScaledForwardResult> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = obs.len();
    let n = hmm.n_states;

    let mut alpha = vec![0.0f64; t_max * n];
    let mut scales = vec![0.0f64; t_max];

    // t=0: α̂_0(j) = π_j · b_j(o_0); then normalise.
    for j in 0..n {
        let em = hmm.b[j * hmm.n_obs + obs[0]];
        alpha[j] = hmm.pi[j] * em;
    }
    let c0: f64 = alpha[..n].iter().sum();
    if c0 < f64::MIN_POSITIVE {
        return Err(SeqError::NumericalInstability(
            "all initial emissions are zero for obs[0]".to_string(),
        ));
    }
    let c0 = 1.0 / c0;
    scales[0] = c0;
    for j in 0..n {
        alpha[j] *= c0;
    }

    // t>0: recursive step.
    let mut tmp_row = vec![0.0f64; n];
    for t in 1..t_max {
        // Copy previous alpha row to avoid simultaneous borrow.
        tmp_row.copy_from_slice(&alpha[(t - 1) * n..t * n]);
        for j in 0..n {
            let em = hmm.b[j * hmm.n_obs + obs[t]];
            let sum: f64 = (0..n).map(|i| tmp_row[i] * hmm.a[i * n + j]).sum();
            alpha[t * n + j] = sum * em;
        }
        let row_sum: f64 = alpha[t * n..t * n + n].iter().sum();
        if row_sum < f64::MIN_POSITIVE {
            return Err(SeqError::NumericalInstability(format!(
                "all scaled forward values vanished at t={t}"
            )));
        }
        let ct = 1.0 / row_sum;
        scales[t] = ct;
        for j in 0..n {
            alpha[t * n + j] *= ct;
        }
    }

    // log p(O|λ) = -Σ_t log(c_t) = Σ_t log(1/c_t).
    let log_likelihood: f64 = scales.iter().map(|&c| -log_safe(c)).sum();

    Ok(ScaledForwardResult {
        alpha,
        scales,
        log_likelihood,
    })
}

/// Scaled backward pass using the scales from a prior scaled forward pass.
///
/// Reusing the same c_t as forward ensures the combined α·β products remain
/// in a numerically safe range without a separate normalisation pass.
pub fn scaled_backward(
    hmm: &HmmDiscrete,
    obs: &[usize],
    scales: &[f64],
) -> SeqResult<ScaledBackwardResult> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = obs.len();
    if scales.len() != t_max {
        return Err(SeqError::LengthMismatch {
            a: scales.len(),
            b: t_max,
        });
    }
    let n = hmm.n_states;
    let mut beta = vec![0.0f64; t_max * n];

    // β_{T-1}(i) is set to c_{T-1} (scaled by the last scale factor).
    let last_c = scales[t_max - 1];
    for i in 0..n {
        beta[(t_max - 1) * n + i] = last_c;
    }

    // β_t(i) = c_t · Σ_j a_{ij} · b_j(o_{t+1}) · β_{t+1}(j)
    let mut tmp_next = vec![0.0f64; n];
    for t in (0..t_max - 1).rev() {
        // Copy next beta row to avoid simultaneous borrow.
        tmp_next.copy_from_slice(&beta[(t + 1) * n..(t + 2) * n]);
        let ct = scales[t];
        for i in 0..n {
            let mut s = 0.0f64;
            for j in 0..n {
                let em = hmm.b[j * hmm.n_obs + obs[t + 1]];
                s += hmm.a[i * n + j] * em * tmp_next[j];
            }
            beta[t * n + i] = ct * s;
        }
    }

    Ok(ScaledBackwardResult { beta })
}

/// Full scaled forward-backward, yielding posteriors γ and ξ.
pub fn scaled_forward_backward(
    hmm: &HmmDiscrete,
    obs: &[usize],
) -> SeqResult<ScaledForwardBackwardResult> {
    let sf = scaled_forward(hmm, obs)?;
    let sb = scaled_backward(hmm, obs, &sf.scales)?;

    let t_max = obs.len();
    let n = hmm.n_states;

    // γ_t(i) = α_t(i) · β_t(i), normalised per time step.
    let mut gamma = vec![0.0f64; t_max * n];
    for t in 0..t_max {
        let mut row_sum = 0.0f64;
        for i in 0..n {
            let v = sf.alpha[t * n + i] * sb.beta[t * n + i];
            gamma[t * n + i] = v;
            row_sum += v;
        }
        if row_sum > 0.0 {
            for i in 0..n {
                gamma[t * n + i] /= row_sum;
            }
        }
    }

    // ξ_t(i,j) = α_t(i) · a_{ij} · b_j(o_{t+1}) · β_{t+1}(j), normalised per t.
    let xi_len = t_max.saturating_sub(1) * n * n;
    let mut xi = vec![0.0f64; xi_len];
    for t in 0..t_max.saturating_sub(1) {
        let mut total = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let em = hmm.b[j * hmm.n_obs + obs[t + 1]];
                let v = sf.alpha[t * n + i] * hmm.a[i * n + j] * em * sb.beta[(t + 1) * n + j];
                xi[t * n * n + i * n + j] = v;
                total += v;
            }
        }
        if total > 0.0 {
            for v in xi[t * n * n..(t + 1) * n * n].iter_mut() {
                *v /= total;
            }
        }
    }

    Ok(ScaledForwardBackwardResult {
        alpha: sf.alpha,
        beta: sb.beta,
        scales: sf.scales,
        gamma,
        xi,
        log_likelihood: sf.log_likelihood,
    })
}

/// Compute Baum-Welch parameter updates using scaled forward-backward.
///
/// Returns unnormalised `(new_pi, A_numerator, B_numerator)`.
/// Caller is responsible for row-normalising A and B before creating a new HMM.
pub fn scaled_baum_welch_step(
    hmm: &HmmDiscrete,
    obs: &[usize],
    sfb: &ScaledForwardBackwardResult,
) -> SeqResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let t_max = obs.len();
    let n = hmm.n_states;
    let n_obs = hmm.n_obs;

    // new_pi = γ_0
    let new_pi: Vec<f64> = sfb.gamma[..n].to_vec();

    // A_num[i*n+j] = Σ_{t=0}^{T-2} ξ_t(i,j)
    let mut a_num = vec![0.0f64; n * n];
    for t in 0..t_max.saturating_sub(1) {
        for i in 0..n {
            for j in 0..n {
                a_num[i * n + j] += sfb.xi[t * n * n + i * n + j];
            }
        }
    }

    // B_num[j*n_obs+o] = Σ_{t: obs[t]==o} γ_t(j)
    let mut b_num = vec![0.0f64; n * n_obs];
    for (t, &o) in obs.iter().enumerate() {
        if o >= n_obs {
            return Err(SeqError::IndexOutOfBounds {
                index: o,
                len: n_obs,
            });
        }
        for j in 0..n {
            b_num[j * n_obs + o] += sfb.gamma[t * n + j];
        }
    }

    Ok((new_pi, a_num, b_num))
}

/// Log-space Viterbi decoding, provided as a companion to the scaled algorithms.
///
/// The scaling approach only helps the forward-backward pass; Viterbi's
/// max-product recursion is already numerically safe in log-space.
pub fn scaled_viterbi(hmm: &HmmDiscrete, obs: &[usize]) -> SeqResult<Vec<usize>> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let t_max = obs.len();
    let n = hmm.n_states;

    let mut delta = vec![f64::NEG_INFINITY; t_max * n];
    let mut psi = vec![0usize; t_max * n];

    // δ_0(j) = log π_j + log b_j(o_0)
    for j in 0..n {
        delta[j] = log_safe(hmm.pi[j]) + log_safe(hmm.b[j * hmm.n_obs + obs[0]]);
    }

    for t in 1..t_max {
        for j in 0..n {
            let log_em = log_safe(hmm.b[j * hmm.n_obs + obs[t]]);
            let mut best = f64::NEG_INFINITY;
            let mut argmax = 0usize;
            for i in 0..n {
                let v = delta[(t - 1) * n + i] + log_safe(hmm.a[i * n + j]);
                if v > best {
                    best = v;
                    argmax = i;
                }
            }
            delta[t * n + j] = best + log_em;
            psi[t * n + j] = argmax;
        }
    }

    // Termination: find the state with the highest score at T-1.
    let mut best = f64::NEG_INFINITY;
    let mut last = 0usize;
    for j in 0..n {
        let v = delta[(t_max - 1) * n + j];
        if v > best {
            best = v;
            last = j;
        }
    }

    // Traceback.
    let mut path = vec![0usize; t_max];
    path[t_max - 1] = last;
    for t in (1..t_max).rev() {
        path[t - 1] = psi[t * n + path[t]];
    }

    Ok(path)
}

// ─── inline tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::forward_backward::forward_backward;
    use crate::hmm::viterbi::viterbi;

    fn small_hmm() -> HmmDiscrete {
        HmmDiscrete::new(
            2,
            2,
            vec![0.6, 0.4],
            vec![0.7, 0.3, 0.4, 0.6],
            vec![0.1, 0.9, 0.8, 0.2],
        )
        .expect("small_hmm ok")
    }

    fn hmm_2s_2o() -> HmmDiscrete {
        HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.9, 0.1, 0.1, 0.9],
            vec![0.9, 0.1, 0.1, 0.9],
        )
        .expect("hmm_2s_2o ok")
    }

    fn single_state_hmm() -> HmmDiscrete {
        HmmDiscrete::new(1, 2, vec![1.0], vec![1.0], vec![0.5, 0.5]).expect("single ok")
    }

    #[test]
    fn scaled_forward_likelihood_matches_log_space() {
        let h = small_hmm();
        let obs = vec![0usize, 1, 0, 1, 0];
        let sf = scaled_forward(&h, &obs).expect("ok");
        let fb = forward_backward(&h, &obs).expect("ok");
        assert!(
            (sf.log_likelihood - fb.log_likelihood).abs() < 1e-6,
            "scaled ll={} log-space ll={}",
            sf.log_likelihood,
            fb.log_likelihood
        );
    }

    #[test]
    fn scaled_forward_scales_all_positive() {
        let h = small_hmm();
        let sf = scaled_forward(&h, &[0, 1, 0, 1]).expect("ok");
        for (t, &c) in sf.scales.iter().enumerate() {
            assert!(c > 0.0, "c[{t}]={c} not positive");
        }
    }

    #[test]
    fn scaled_forward_alpha_rows_sum_to_one() {
        let h = small_hmm();
        let obs = vec![0, 1, 0, 1];
        let sf = scaled_forward(&h, &obs).expect("ok");
        let n = h.n_states;
        for t in 0..obs.len() {
            let s: f64 = sf.alpha[t * n..(t + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "t={t} row sum={s}");
        }
    }

    #[test]
    fn scaled_backward_beta_finite() {
        let h = small_hmm();
        let obs = vec![0, 1, 0];
        let sf = scaled_forward(&h, &obs).expect("ok");
        let sb = scaled_backward(&h, &obs, &sf.scales).expect("ok");
        for &v in &sb.beta {
            assert!(v.is_finite(), "beta value not finite: {v}");
        }
    }

    #[test]
    fn scaled_forward_backward_gamma_sum() {
        let h = small_hmm();
        let obs = vec![0, 1, 0, 1];
        let sfb = scaled_forward_backward(&h, &obs).expect("ok");
        let n = h.n_states;
        for t in 0..obs.len() {
            let s: f64 = sfb.gamma[t * n..(t + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "gamma t={t} sum={s}");
        }
    }

    #[test]
    fn scaled_forward_backward_xi_sum() {
        let h = small_hmm();
        let obs = vec![0, 1, 0, 1];
        let sfb = scaled_forward_backward(&h, &obs).expect("ok");
        let n = h.n_states;
        for t in 0..obs.len() - 1 {
            let s: f64 = sfb.xi[t * n * n..(t + 1) * n * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "xi t={t} sum={s}");
        }
    }

    #[test]
    fn scaled_ll_equals_log_space_ll() {
        let h = hmm_2s_2o();
        let obs = vec![0, 0, 1, 1, 0];
        let sf = scaled_forward(&h, &obs).expect("ok");
        let fb = forward_backward(&h, &obs).expect("ok");
        assert!(
            (sf.log_likelihood - fb.log_likelihood).abs() < 1e-6,
            "scaled={} log-space={}",
            sf.log_likelihood,
            fb.log_likelihood
        );
    }

    #[test]
    fn scaled_forward_empty_obs_err() {
        let h = small_hmm();
        let res = scaled_forward(&h, &[]);
        assert!(matches!(res, Err(SeqError::EmptyInput)));
    }

    #[test]
    fn scaled_viterbi_consistent_with_standard_viterbi() {
        let h = hmm_2s_2o();
        let obs = vec![0, 0, 1, 1];
        let sv = scaled_viterbi(&h, &obs).expect("ok");
        let lv = viterbi(&h, &obs).expect("ok");
        assert_eq!(
            sv, lv.path,
            "scaled_viterbi path diverges from log-space viterbi"
        );
    }

    #[test]
    fn scaled_forward_single_obs() {
        let h = small_hmm();
        let sf = scaled_forward(&h, &[0]).expect("ok");
        assert_eq!(sf.alpha.len(), h.n_states);
        assert_eq!(sf.scales.len(), 1);
        let s: f64 = sf.alpha.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaled_forward_long_sequence_no_underflow() {
        let h = hmm_2s_2o();
        let obs: Vec<usize> = (0..1000).map(|i| i % 2).collect();
        let sf = scaled_forward(&h, &obs);
        assert!(sf.is_ok(), "scaled_forward failed on length-1000 sequence");
        let sf = sf.expect("ok");
        assert!(sf.log_likelihood.is_finite());
        assert!(sf.log_likelihood < 0.0, "log-likelihood must be negative");
    }

    #[test]
    fn scaled_backward_wrong_scales_len_err() {
        let h = small_hmm();
        let obs = vec![0, 1, 0];
        let bad_scales = vec![1.0, 1.0];
        let res = scaled_backward(&h, &obs, &bad_scales);
        assert!(
            matches!(res, Err(SeqError::LengthMismatch { .. })),
            "expected LengthMismatch"
        );
    }

    #[test]
    fn scaled_baum_welch_step_pi_sums_to_1() {
        let h = small_hmm();
        let obs = vec![0, 1, 0, 1];
        let sfb = scaled_forward_backward(&h, &obs).expect("ok");
        let (new_pi, _, _) = scaled_baum_welch_step(&h, &obs, &sfb).expect("ok");
        let s: f64 = new_pi.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "new_pi sum={s}");
    }

    #[test]
    fn scaled_baum_welch_step_shapes_correct() {
        let h = small_hmm();
        let obs = vec![0, 1, 0, 1];
        let sfb = scaled_forward_backward(&h, &obs).expect("ok");
        let (pi, a_num, b_num) = scaled_baum_welch_step(&h, &obs, &sfb).expect("ok");
        assert_eq!(pi.len(), h.n_states);
        assert_eq!(a_num.len(), h.n_states * h.n_states);
        assert_eq!(b_num.len(), h.n_states * h.n_obs);
    }

    #[test]
    fn scaled_forward_backward_2state_2obs() {
        // Simple known HMM: deterministic transitions, near-deterministic emissions.
        // State 0 emits obs 0 with probability ~1, state 1 emits obs 1 with probability ~1.
        let h = HmmDiscrete::new(
            2,
            2,
            vec![1.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.99, 0.01, 0.01, 0.99],
        )
        .expect("ok");
        let obs = vec![0, 1, 0, 1];
        let sfb = scaled_forward_backward(&h, &obs).expect("ok");
        // At t=0 state 0 is strongly preferred (γ_0(0) ≈ 1).
        assert!(sfb.gamma[0] > 0.9, "gamma[0][0]={}", sfb.gamma[0]);
        // At t=1 state 1 is strongly preferred.
        let n = h.n_states;
        assert!(sfb.gamma[n + 1] > 0.9, "gamma[1][1]={}", sfb.gamma[n + 1]);
    }

    #[test]
    fn scaled_forward_single_state() {
        let h = single_state_hmm();
        let obs = vec![0, 1, 0];
        let sf = scaled_forward(&h, &obs).expect("ok");
        assert_eq!(sf.scales.len(), 3);
        assert_eq!(sf.alpha.len(), 3);
        for &a in &sf.alpha {
            assert!(
                (a - 1.0).abs() < 1e-12,
                "single-state alpha must be 1.0, got {a}"
            );
        }
    }

    #[test]
    fn scaled_viterbi_single_state() {
        let h = single_state_hmm();
        let obs = vec![0, 1, 0, 1];
        let path = scaled_viterbi(&h, &obs).expect("ok");
        assert_eq!(
            path,
            vec![0, 0, 0, 0],
            "single-state path must be all zeros"
        );
    }
}
