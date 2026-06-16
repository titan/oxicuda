//! Connectionist Temporal Classification (CTC) loss (Graves et al. 2006).
//!
//! CTC trains sequence models that emit a per-frame distribution over an
//! alphabet augmented with a special **blank** symbol, without requiring a
//! frame-level alignment between the input and the target label sequence. The
//! loss is the negative log-probability of the target, summed over **all**
//! alignments (frame-to-label paths) that collapse to the target via the
//! CTC mapping `B`:
//!
//! 1. merge consecutive identical labels, then
//! 2. remove all blanks.
//!
//! ## Forward-backward algorithm
//! The target label sequence `l` of length `U` is expanded into a
//! blank-augmented sequence `l'` of length `S = 2U + 1`:
//! `[blank, l_0, blank, l_1, …, blank, l_{U-1}, blank]`. The forward variable
//! `α_t(s)` is the total probability of all paths ending in symbol `l'_s` at
//! frame `t`; the backward variable `β_t(s)` is the symmetric quantity. Both
//! are accumulated in **log-space** for numerical stability over long
//! sequences and small per-frame probabilities.
//!
//! The total log-probability is
//! `log p(l | x) = logaddexp(α_{T-1}(S-1), α_{T-1}(S-2))`
//! and the loss is `−log p(l | x)`.
//!
//! ## Inputs
//! All emissions are supplied as **log-probabilities** with shape `[T, C]`
//! (row-major: frame `t`, symbol `c` lives at `t·C + c`). `blank` is the symbol
//! index of the blank (commonly `0` or `C − 1`). The target labels must lie in
//! `0..C` and must not equal `blank`.

use crate::error::{SeqError, SeqResult};

/// Numerically-stable `log(exp(a) + exp(b))` with `-inf` handling.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Validated CTC problem instance.
struct CtcProblem<'a> {
    /// Log-probability emissions `[T, C]`, row-major.
    log_probs: &'a [f64],
    /// Number of frames `T`.
    t_len: usize,
    /// Alphabet size `C` (including blank).
    n_symbols: usize,
    /// Blank symbol index.
    blank: usize,
    /// Blank-augmented target `l'` of length `S = 2U + 1`.
    extended: Vec<usize>,
}

impl<'a> CtcProblem<'a> {
    fn new(
        log_probs: &'a [f64],
        t_len: usize,
        n_symbols: usize,
        targets: &[usize],
        blank: usize,
    ) -> SeqResult<Self> {
        if t_len == 0 || n_symbols == 0 {
            return Err(SeqError::EmptyInput);
        }
        if log_probs.len() != t_len * n_symbols {
            return Err(SeqError::ShapeMismatch {
                expected: t_len * n_symbols,
                got: log_probs.len(),
            });
        }
        if blank >= n_symbols {
            return Err(SeqError::IndexOutOfBounds {
                index: blank,
                len: n_symbols,
            });
        }
        for &v in log_probs {
            if v.is_nan() {
                return Err(SeqError::NumericalInstability(
                    "NaN in CTC log-probabilities".into(),
                ));
            }
        }
        for &lab in targets {
            if lab >= n_symbols {
                return Err(SeqError::IndexOutOfBounds {
                    index: lab,
                    len: n_symbols,
                });
            }
            if lab == blank {
                return Err(SeqError::InvalidConfiguration(
                    "CTC target label must not equal the blank symbol".into(),
                ));
            }
        }
        // CTC requires T >= U + (number of consecutive repeats), with a minimum
        // of T >= U. A target longer than the number of frames is infeasible.
        if targets.len() > t_len {
            return Err(SeqError::InvalidConfiguration(format!(
                "CTC target length {} exceeds frame count {t_len}",
                targets.len()
            )));
        }

        let mut extended = Vec::with_capacity(2 * targets.len() + 1);
        extended.push(blank);
        for &lab in targets {
            extended.push(lab);
            extended.push(blank);
        }
        Ok(Self {
            log_probs,
            t_len,
            n_symbols,
            blank,
            extended,
        })
    }

    #[inline]
    fn lp(&self, t: usize, c: usize) -> f64 {
        self.log_probs[t * self.n_symbols + c]
    }

    /// Forward pass returning `α` of shape `[T, S]` (log-space).
    fn forward(&self) -> Vec<f64> {
        let s = self.extended.len();
        let t = self.t_len;
        let mut alpha = vec![f64::NEG_INFINITY; t * s];
        // Initialisation at t = 0: only blank (s=0) or first label (s=1) may
        // start a valid path; all deeper positions remain -inf.
        alpha[0] = self.lp(0, self.blank);
        if s > 1 {
            alpha[1] = self.lp(0, self.extended[1]);
        }
        for ti in 1..t {
            let row = ti * s;
            let prow = (ti - 1) * s;
            for si in 0..s {
                let sym = self.extended[si];
                // Path stays at s or comes from s-1.
                let mut acc = alpha[prow + si];
                if si >= 1 {
                    acc = log_add_exp(acc, alpha[prow + si - 1]);
                }
                // The skip transition s-2 → s is allowed only when the current
                // symbol is a non-blank label different from l'_{s-2}.
                if si >= 2 && sym != self.blank && self.extended[si - 2] != sym {
                    acc = log_add_exp(acc, alpha[prow + si - 2]);
                }
                alpha[row + si] = acc + self.lp(ti, sym);
            }
        }
        alpha
    }

    /// Backward pass returning `β` of shape `[T, S]` (log-space).
    fn backward(&self) -> Vec<f64> {
        let s = self.extended.len();
        let t = self.t_len;
        let mut beta = vec![f64::NEG_INFINITY; t * s];
        let last = (t - 1) * s;
        beta[last + s - 1] = 0.0;
        if s >= 2 {
            beta[last + s - 2] = 0.0;
        }
        for ti in (0..t - 1).rev() {
            let row = ti * s;
            let nrow = (ti + 1) * s;
            for si in 0..s {
                let sym = self.extended[si];
                let mut acc = beta[nrow + si] + self.lp(ti + 1, sym);
                if si + 1 < s {
                    let sym1 = self.extended[si + 1];
                    acc = log_add_exp(acc, beta[nrow + si + 1] + self.lp(ti + 1, sym1));
                }
                if si + 2 < s {
                    let sym2 = self.extended[si + 2];
                    if sym2 != self.blank && sym2 != sym {
                        acc = log_add_exp(acc, beta[nrow + si + 2] + self.lp(ti + 1, sym2));
                    }
                }
                beta[row + si] = acc;
            }
        }
        beta
    }
}

/// CTC negative-log-likelihood loss for a single sequence.
///
/// * `log_probs` — `[T, C]` row-major log-probabilities.
/// * `t_len` — number of frames `T`.
/// * `n_symbols` — alphabet size `C` (including blank).
/// * `targets` — target label indices (each in `0..C`, none equal to `blank`).
/// * `blank` — blank symbol index.
///
/// Returns `−log p(targets | log_probs) ≥ 0`.
pub fn ctc_loss(
    log_probs: &[f64],
    t_len: usize,
    n_symbols: usize,
    targets: &[usize],
    blank: usize,
) -> SeqResult<f64> {
    let problem = CtcProblem::new(log_probs, t_len, n_symbols, targets, blank)?;
    let s = problem.extended.len();
    let alpha = problem.forward();
    let last = (problem.t_len - 1) * s;
    // Total probability terminates in the final blank (s-1) or final label (s-2).
    let log_prob = if s >= 2 {
        log_add_exp(alpha[last + s - 1], alpha[last + s - 2])
    } else {
        alpha[last + s - 1]
    };
    if log_prob == f64::NEG_INFINITY {
        return Err(SeqError::NumericalInstability(
            "CTC alignment has zero probability (infeasible target)".into(),
        ));
    }
    Ok(-log_prob)
}

/// Mean CTC loss over a batch of variable-length sequences.
///
/// Each batch element is `(log_probs, t_len, targets)`; `n_symbols` and `blank`
/// are shared. Returns the mean of the per-sequence losses.
pub fn ctc_loss_batch(
    batch: &[(Vec<f64>, usize, Vec<usize>)],
    n_symbols: usize,
    blank: usize,
) -> SeqResult<f64> {
    if batch.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut acc = 0.0;
    for (log_probs, t_len, targets) in batch {
        acc += ctc_loss(log_probs, *t_len, n_symbols, targets, blank)?;
    }
    Ok(acc / batch.len() as f64)
}

/// Posterior occupancy `γ_t(s) = α_t(s)·β_t(s) / p(l)` in **log-space**,
/// returned as a `[T, S]` matrix where `S = 2U + 1`.
///
/// These are the per-symbol responsibilities used to form the CTC gradient with
/// respect to the emission log-probabilities; they are exposed for testing and
/// for downstream training code.
pub fn ctc_log_occupancy(
    log_probs: &[f64],
    t_len: usize,
    n_symbols: usize,
    targets: &[usize],
    blank: usize,
) -> SeqResult<(Vec<f64>, usize)> {
    let problem = CtcProblem::new(log_probs, t_len, n_symbols, targets, blank)?;
    let s = problem.extended.len();
    let alpha = problem.forward();
    let beta = problem.backward();
    let last = (problem.t_len - 1) * s;
    let log_prob = if s >= 2 {
        log_add_exp(alpha[last + s - 1], alpha[last + s - 2])
    } else {
        alpha[last + s - 1]
    };
    if log_prob == f64::NEG_INFINITY {
        return Err(SeqError::NumericalInstability(
            "CTC alignment has zero probability".into(),
        ));
    }
    let mut gamma = vec![f64::NEG_INFINITY; t_len * s];
    for i in 0..t_len * s {
        let v = alpha[i] + beta[i];
        gamma[i] = if v == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            v - log_prob
        };
    }
    Ok((gamma, s))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert a `[T, C]` matrix of probabilities to log-probabilities.
    fn to_log(probs: &[f64], t: usize, c: usize) -> Vec<f64> {
        assert_eq!(probs.len(), t * c);
        probs.iter().map(|&p| p.max(1e-30).ln()).collect()
    }

    /// Brute-force CTC probability by enumerating all length-T paths over the
    /// alphabet and collapsing each via the CTC mapping. Used as ground truth.
    fn brute_force_prob(probs: &[f64], t: usize, c: usize, target: &[usize], blank: usize) -> f64 {
        let mut total = 0.0;
        let mut path = vec![0usize; t];
        let n_paths = c.pow(t as u32);
        for code in 0..n_paths {
            let mut rem = code;
            for ti in 0..t {
                path[ti] = rem % c;
                rem /= c;
            }
            // Collapse: merge repeats then drop blanks.
            let mut collapsed: Vec<usize> = Vec::new();
            let mut prev = usize::MAX;
            for &sym in &path {
                if sym != prev {
                    collapsed.push(sym);
                }
                prev = sym;
            }
            collapsed.retain(|&x| x != blank);
            if collapsed == target {
                let mut prob = 1.0;
                for ti in 0..t {
                    prob *= probs[ti * c + path[ti]];
                }
                total += prob;
            }
        }
        total
    }

    #[test]
    fn loss_is_finite_and_nonnegative() {
        // T=4, C=3, blank=0, target=[1,2].
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
            0.4, 0.3, 0.3, //
        ];
        let lp = to_log(&probs, 4, 3);
        let loss = ctc_loss(&lp, 4, 3, &[1, 2], 0).expect("loss");
        assert!(loss.is_finite() && loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn matches_brute_force_small() {
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
        ];
        let lp = to_log(&probs, 3, 3);
        let loss = ctc_loss(&lp, 3, 3, &[1, 2], 0).expect("loss");
        let bf = brute_force_prob(&probs, 3, 3, &[1, 2], 0);
        assert!(bf > 0.0);
        let expected = -bf.ln();
        assert!(
            (loss - expected).abs() < 1e-9,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn matches_brute_force_with_repeat_target() {
        // Target with a repeated label [1,1] requires an interleaving blank.
        let probs = vec![
            0.1, 0.7, 0.2, //
            0.5, 0.3, 0.2, //
            0.2, 0.6, 0.2, //
            0.3, 0.3, 0.4, //
        ];
        let lp = to_log(&probs, 4, 3);
        let loss = ctc_loss(&lp, 4, 3, &[1, 1], 0).expect("loss");
        let bf = brute_force_prob(&probs, 4, 3, &[1, 1], 0);
        assert!(bf > 0.0);
        let expected = -bf.ln();
        assert!(
            (loss - expected).abs() < 1e-9,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn empty_target_is_all_blanks() {
        // For an empty target, the only valid path is all-blank.
        let probs = vec![
            0.7, 0.2, 0.1, //
            0.6, 0.3, 0.1, //
        ];
        let lp = to_log(&probs, 2, 3);
        let loss = ctc_loss(&lp, 2, 3, &[], 0).expect("loss");
        let expected = -(0.7_f64.ln() + 0.6_f64.ln());
        assert!(
            (loss - expected).abs() < 1e-9,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn deterministic_single_frame_single_label() {
        // T=1, target=[1], blank=0: probability is exactly p(frame0, label1).
        let probs = vec![0.3, 0.7, 0.0];
        let lp = to_log(&probs, 1, 3);
        let loss = ctc_loss(&lp, 1, 3, &[1], 0).expect("loss");
        assert!((loss - (-0.7_f64.ln())).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn blank_at_last_index_works() {
        // Use blank = C-1 to confirm blank index is not hard-coded.
        let probs = vec![
            0.5, 0.3, 0.2, //
            0.4, 0.4, 0.2, //
            0.2, 0.5, 0.3, //
        ];
        let lp = to_log(&probs, 3, 3);
        let blank = 2;
        let loss = ctc_loss(&lp, 3, 3, &[0, 1], blank).expect("loss");
        let bf = brute_force_prob(&probs, 3, 3, &[0, 1], blank);
        let expected = -bf.ln();
        assert!(
            (loss - expected).abs() < 1e-9,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn loss_decreases_when_target_probability_increases() {
        // Sharpen the emissions toward the target → loss must drop.
        let probs_a = vec![
            0.3, 0.4, 0.3, //
            0.3, 0.4, 0.3, //
            0.3, 0.3, 0.4, //
        ];
        let probs_b = vec![
            0.1, 0.8, 0.1, //
            0.1, 0.8, 0.1, //
            0.1, 0.1, 0.8, //
        ];
        let la = ctc_loss(&to_log(&probs_a, 3, 3), 3, 3, &[1, 2], 0).expect("a");
        let lb = ctc_loss(&to_log(&probs_b, 3, 3), 3, 3, &[1, 2], 0).expect("b");
        assert!(lb < la, "la={la} lb={lb}");
    }

    #[test]
    fn target_longer_than_frames_errors() {
        let lp = to_log(&[0.5, 0.5], 1, 2);
        let res = ctc_loss(&lp, 1, 2, &[0, 1], 1);
        assert!(res.is_err());
    }

    #[test]
    fn target_equal_blank_errors() {
        let lp = to_log(&[0.5, 0.5, 0.5, 0.5], 2, 2);
        // blank = 0, target contains 0 → invalid.
        assert!(ctc_loss(&lp, 2, 2, &[0], 0).is_err());
    }

    #[test]
    fn shape_mismatch_errors() {
        let lp = vec![0.0; 5]; // not 2*3
        assert!(ctc_loss(&lp, 2, 3, &[1], 0).is_err());
    }

    #[test]
    fn blank_index_out_of_range_errors() {
        let lp = to_log(&[0.5, 0.5, 0.5, 0.5], 2, 2);
        assert!(ctc_loss(&lp, 2, 2, &[1], 5).is_err());
    }

    #[test]
    fn nan_in_log_probs_errors() {
        let lp = vec![f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(ctc_loss(&lp, 2, 3, &[1], 0).is_err());
    }

    #[test]
    fn forward_backward_agree_on_total() {
        // log p computed from α at the last frame must equal that from β at frame 0.
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
            0.4, 0.3, 0.3, //
        ];
        let lp = to_log(&probs, 4, 3);
        let problem = CtcProblem::new(&lp, 4, 3, &[1, 2], 0).expect("problem");
        let s = problem.extended.len();
        let alpha = problem.forward();
        let beta = problem.backward();
        let last = 3 * s;
        let from_alpha = log_add_exp(alpha[last + s - 1], alpha[last + s - 2]);
        // From β: sum over s of β_0(s) + α_0(s); equivalently α_0(s)·β_0(s).
        let mut from_beta = f64::NEG_INFINITY;
        for si in 0..s {
            from_beta = log_add_exp(from_beta, alpha[si] + beta[si]);
        }
        assert!(
            (from_alpha - from_beta).abs() < 1e-9,
            "a={from_alpha} b={from_beta}"
        );
    }

    #[test]
    fn occupancy_normalises_per_frame() {
        // Σ_s exp(γ_t(s)) = 1 for every frame t.
        let probs = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
            0.4, 0.3, 0.3, //
        ];
        let lp = to_log(&probs, 4, 3);
        let (gamma, s) = ctc_log_occupancy(&lp, 4, 3, &[1, 2], 0).expect("gamma");
        for ti in 0..4 {
            let mut sum = 0.0;
            for si in 0..s {
                let g = gamma[ti * s + si];
                if g != f64::NEG_INFINITY {
                    sum += g.exp();
                }
            }
            assert!((sum - 1.0).abs() < 1e-6, "frame {ti}: sum={sum}");
        }
    }

    #[test]
    fn batch_mean_matches_manual() {
        let probs0 = vec![
            0.2, 0.5, 0.3, //
            0.1, 0.6, 0.3, //
            0.3, 0.2, 0.5, //
        ];
        let probs1 = vec![
            0.4, 0.4, 0.2, //
            0.2, 0.5, 0.3, //
            0.3, 0.3, 0.4, //
        ];
        let lp0 = to_log(&probs0, 3, 3);
        let lp1 = to_log(&probs1, 3, 3);
        let batch = vec![(lp0.clone(), 3, vec![1, 2]), (lp1.clone(), 3, vec![1, 2])];
        let mean = ctc_loss_batch(&batch, 3, 0).expect("mean");
        let l0 = ctc_loss(&lp0, 3, 3, &[1, 2], 0).expect("l0");
        let l1 = ctc_loss(&lp1, 3, 3, &[1, 2], 0).expect("l1");
        assert!((mean - 0.5 * (l0 + l1)).abs() < 1e-12);
    }
}
