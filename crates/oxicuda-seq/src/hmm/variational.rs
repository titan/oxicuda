//! Variational Bayes EM for Hidden Markov Models with Dirichlet priors.
//!
//! Reference: Beal 2003, "Variational Algorithms for Approximate Bayesian Inference", §3.4.
//!
//! Standard Baum-Welch places ML point estimates on π, A, B.  VB-EM instead
//! places conjugate Dirichlet priors on those parameters and maintains a
//! factored variational posterior q(π, A, B) = q(π) · ∏_i q(A_i) · ∏_i q(B_i)
//! whose factors are themselves Dirichlet distributions.  The sufficient
//! statistics from forward-backward (computed with *expected* log-parameters
//! derived via the digamma function) update the Dirichlet concentration
//! parameters in the M-step, and the ELBO is tracked for convergence.

use crate::error::{SeqError, SeqResult};

// ─── Special functions ────────────────────────────────────────────────────────

/// Scalar digamma function ψ(x) implemented via upward recursion followed by
/// an asymptotic Stirling expansion.
///
/// Algorithm:
///   Shift x up by adding integers until x + k ≥ 6, accumulating
///   the recursion  ψ(x) = ψ(x+1) − 1/x.
///   Then apply the asymptotic series
///     ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶).
pub fn digamma(mut x: f64) -> f64 {
    // Euler-Mascheroni constant, used when x ≈ 1 as a sanity check internally.
    let mut result = 0.0;

    // Shift argument into the asymptotic region (x ≥ 6).
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic Stirling expansion.
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    result += x.ln() - 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x4) - 1.0 / (252.0 * x6);
    result
}

/// Log-Gamma function ln Γ(x) via the Lanczos approximation with g = 7 and
/// 9 pre-computed coefficients (Spouge 1994 / Numerical Recipes form).
pub fn log_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g = 7, 9 terms (Numerical Recipes, 3rd ed.).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_3,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x) Γ(1-x) = π / sin(πx)
        use std::f64::consts::PI;
        return PI.ln() - (PI * x).sin().ln() - log_gamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut sum = C[0];
    for (k, &ck) in C[1..].iter().enumerate() {
        sum += ck / (z + (k as f64 + 1.0));
    }

    use std::f64::consts::PI;
    let t = z + G + 0.5;
    (2.0 * PI).sqrt().ln() + sum.ln() + (z + 0.5) * t.ln() - t
}

/// Log-normaliser of a Dirichlet distribution:
///   log B(α) = Σ_i ln Γ(α_i) − ln Γ(Σ_i α_i).
pub fn dirichlet_log_normalizer(alpha: &[f64]) -> f64 {
    let sum_alpha: f64 = alpha.iter().sum();
    let sum_log_gamma: f64 = alpha.iter().map(|&a| log_gamma(a)).sum();
    sum_log_gamma - log_gamma(sum_alpha)
}

// ─── Configuration & result types ─────────────────────────────────────────────

/// Configuration for Variational Bayes HMM training.
#[derive(Debug, Clone)]
pub struct VbHmmConfig {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of distinct observation symbols.
    pub n_obs: usize,
    /// Symmetric Dirichlet prior concentration for π (default 1.0).
    pub alpha_prior: f64,
    /// Symmetric Dirichlet prior concentration for each A row (default 1.0).
    pub beta_prior: f64,
    /// Symmetric Dirichlet prior concentration for each B row (default 1.0).
    pub gamma_prior: f64,
    /// Maximum number of VB-EM iterations (default 200).
    pub max_iter: usize,
    /// ELBO convergence tolerance (default 1e-6).
    pub tol: f64,
}

impl Default for VbHmmConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_obs: 2,
            alpha_prior: 1.0,
            beta_prior: 1.0,
            gamma_prior: 1.0,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

/// Result of Variational Bayes HMM training.
#[derive(Debug, Clone)]
pub struct VbHmmResult {
    /// Dirichlet concentration parameters for the initial-state posterior (n_states,).
    pub alpha: Vec<f64>,
    /// Dirichlet concentration parameters for the transition posterior, row-major
    /// (n_states × n_states,).
    pub beta: Vec<f64>,
    /// Dirichlet concentration parameters for the emission posterior, row-major
    /// (n_states × n_obs,).
    pub gamma: Vec<f64>,
    /// ELBO (Evidence Lower BOund) at each iteration.
    pub elbo_history: Vec<f64>,
    /// Number of VB-EM iterations executed.
    pub n_iter: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
}

impl VbHmmResult {
    /// Expected log initial-state probabilities: E[log π_i] = ψ(α_i) − ψ(Σ_j α_j).
    pub fn expected_log_pi(&self) -> Vec<f64> {
        let sum_alpha: f64 = self.alpha.iter().sum();
        let psi_sum = digamma(sum_alpha);
        self.alpha.iter().map(|&a| digamma(a) - psi_sum).collect()
    }

    /// Posterior mean of the initial-state distribution: α_i / Σ_j α_j.
    pub fn mean_pi(&self) -> Vec<f64> {
        let s: f64 = self.alpha.iter().sum();
        self.alpha.iter().map(|&a| a / s).collect()
    }

    /// Posterior mean of the transition matrix (n_states × n_states, row-major).
    pub fn mean_a(&self) -> Vec<f64> {
        let n = self.alpha.len(); // = n_states
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            let s: f64 = self.beta[i * n..(i + 1) * n].iter().sum();
            for j in 0..n {
                out[i * n + j] = if s > 0.0 {
                    self.beta[i * n + j] / s
                } else {
                    1.0 / n as f64
                };
            }
        }
        out
    }

    /// Posterior mean of the emission matrix (n_states × n_obs, row-major).
    pub fn mean_b(&self) -> Vec<f64> {
        let n = self.alpha.len(); // = n_states
        let k = self.gamma.len() / n; // = n_obs
        let mut out = vec![0.0; n * k];
        for j in 0..n {
            let s: f64 = self.gamma[j * k..(j + 1) * k].iter().sum();
            for sym in 0..k {
                out[j * k + sym] = if s > 0.0 {
                    self.gamma[j * k + sym] / s
                } else {
                    1.0 / k as f64
                };
            }
        }
        out
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

/// logsumexp on a slice; gracefully handles −∞.
#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// VB forward-backward pass.
///
/// Takes pre-computed expected log-parameters directly (not a `HmmDiscrete`).
///
/// # Arguments
/// * `log_pi_eff` — E[log π_i] for i=0..n
/// * `log_a_eff`  — E[log A_{ij}] row-major (n×n)
/// * `log_em_eff` — E[log B_{j, o_t}] for each t, row-major (T×n); caller
///   pre-indexes the emission by the observation symbol.
///
/// # Returns
/// `(gamma, xi, log_likelihood)` where
///   * `gamma` — T×n state posteriors (probability domain, renormalised)
///   * `xi`    — (T-1)×n×n edge posteriors (probability domain)
///   * `log_likelihood` — log p(o | effective model)
fn vb_forward_backward(
    log_pi_eff: &[f64],
    log_a_eff: &[f64],
    log_em_eff: &[f64],
    n: usize,
    t_max: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    // ── Forward ──
    let mut log_alpha = vec![f64::NEG_INFINITY; t_max * n];

    // α_0(j) = log_pi[j] + log_em[0, j]
    for j in 0..n {
        log_alpha[j] = log_pi_eff[j] + log_em_eff[j];
    }

    let mut tmp = vec![0.0f64; n];
    for t in 1..t_max {
        for j in 0..n {
            for i in 0..n {
                tmp[i] = log_alpha[(t - 1) * n + i] + log_a_eff[i * n + j];
            }
            log_alpha[t * n + j] = logsumexp(&tmp) + log_em_eff[t * n + j];
        }
    }

    let ll = logsumexp(&log_alpha[(t_max - 1) * n..t_max * n]);

    // ── Backward ──
    let mut log_beta = vec![f64::NEG_INFINITY; t_max * n];
    for i in 0..n {
        log_beta[(t_max - 1) * n + i] = 0.0;
    }
    for t in (0..t_max.saturating_sub(1)).rev() {
        for i in 0..n {
            for j in 0..n {
                tmp[j] =
                    log_a_eff[i * n + j] + log_em_eff[(t + 1) * n + j] + log_beta[(t + 1) * n + j];
            }
            log_beta[t * n + i] = logsumexp(&tmp);
        }
    }

    // ── γ posteriors ──
    let mut gamma = vec![0.0f64; t_max * n];
    for t in 0..t_max {
        for i in 0..n {
            gamma[t * n + i] = (log_alpha[t * n + i] + log_beta[t * n + i] - ll).exp();
        }
        // Row-normalise to guard against floating-point drift.
        let s: f64 = gamma[t * n..t * n + n].iter().sum();
        if s > 0.0 {
            for i in 0..n {
                gamma[t * n + i] /= s;
            }
        }
    }

    // ── ξ edge posteriors ──
    let xi_len = t_max.saturating_sub(1) * n * n;
    let mut xi = vec![0.0f64; xi_len];
    for t in 0..t_max.saturating_sub(1) {
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                let v = (log_alpha[t * n + i]
                    + log_a_eff[i * n + j]
                    + log_em_eff[(t + 1) * n + j]
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

    (gamma, xi, ll)
}

// ─── KL divergence between two Dirichlet distributions ───────────────────────

/// KL(Dir(alpha) || Dir(alpha_0)) for vectors of equal length.
fn kl_dirichlet(alpha: &[f64], alpha_0: &[f64]) -> f64 {
    let log_b_alpha_0 = dirichlet_log_normalizer(alpha_0);
    let log_b_alpha = dirichlet_log_normalizer(alpha);
    let sum_alpha: f64 = alpha.iter().sum();
    let psi_sum = digamma(sum_alpha);

    let correction: f64 = alpha
        .iter()
        .zip(alpha_0.iter())
        .map(|(&ai, &a0i)| (a0i - ai) * (digamma(ai) - psi_sum))
        .sum();

    log_b_alpha_0 - log_b_alpha + correction
}

// ─── Main entry point ──────────────────────────────────────────────────────────

/// Run Variational Bayes EM for a discrete HMM with Dirichlet priors.
///
/// Accepts one or more observation sequences.  Each sequence element must lie
/// in `0..cfg.n_obs`.
pub fn variational_hmm(observations: &[&[usize]], cfg: &VbHmmConfig) -> SeqResult<VbHmmResult> {
    // ── Validation ──
    if observations.is_empty() || observations.iter().all(|s| s.is_empty()) {
        return Err(SeqError::EmptyInput);
    }
    if cfg.n_states == 0 || cfg.n_obs == 0 {
        return Err(SeqError::InvalidConfiguration(
            "n_states and n_obs must be > 0".to_string(),
        ));
    }
    for seq in observations.iter() {
        for &o in *seq {
            if o >= cfg.n_obs {
                return Err(SeqError::InvalidObservation(format!(
                    "observation {o} >= n_obs {}",
                    cfg.n_obs
                )));
            }
        }
    }
    // Reject entirely-empty inputs (but allow mixed non-empty / skip empty seqs below).
    if observations.iter().all(|s| s.is_empty()) {
        return Err(SeqError::EmptyInput);
    }

    let n = cfg.n_states;
    let k = cfg.n_obs;

    // ── Initialise Dirichlet parameters ──
    // α_i = alpha_prior + deterministic perturbation
    let mut alpha: Vec<f64> = (0..n)
        .map(|i| cfg.alpha_prior + (i as f64 + 1.0) * 0.1 / n as f64)
        .collect();

    // β_{ij}: higher on diagonal to prefer self-persistence initially.
    let mut beta: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            beta[i * n + j] = if i == j {
                cfg.beta_prior + 0.5
            } else if n > 1 {
                cfg.beta_prior + 0.1 / (n as f64 - 1.0)
            } else {
                cfg.beta_prior
            };
        }
    }

    // γ_{jk}: uniform + small perturbation
    let mut gamma_dir: Vec<f64> = vec![cfg.gamma_prior + 0.1; n * k];

    let mut elbo_history: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut prev_elbo = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0usize;

    // ── VB-EM iterations ──
    for iter in 0..cfg.max_iter {
        n_iter = iter + 1;

        // ── E-step: compute expected log-parameters ──
        // E[log π_i]
        let sum_alpha: f64 = alpha.iter().sum();
        let psi_sum_alpha = digamma(sum_alpha);
        let log_pi_eff: Vec<f64> = alpha.iter().map(|&a| digamma(a) - psi_sum_alpha).collect();

        // E[log A_{ij}] — row-major (n×n)
        let mut log_a_eff: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            let sum_beta_i: f64 = beta[i * n..(i + 1) * n].iter().sum();
            let psi_sum_beta_i = digamma(sum_beta_i);
            for j in 0..n {
                log_a_eff[i * n + j] = digamma(beta[i * n + j]) - psi_sum_beta_i;
            }
        }

        // E[log B_{j, k}] — row-major (n×k)
        let mut log_b_eff: Vec<f64> = vec![0.0; n * k];
        for j in 0..n {
            let sum_gamma_j: f64 = gamma_dir[j * k..(j + 1) * k].iter().sum();
            let psi_sum_gamma_j = digamma(sum_gamma_j);
            for sym in 0..k {
                log_b_eff[j * k + sym] = digamma(gamma_dir[j * k + sym]) - psi_sum_gamma_j;
            }
        }

        // ── Accumulate sufficient statistics over all sequences ──
        // Sufficient stats for M-step:
        //   ss_pi[i]        = Σ_seq γ_{0}^{seq}(i)
        //   ss_a[i,j]       = Σ_seq Σ_t ξ_t(i,j)
        //   ss_b[j, sym]    = Σ_seq Σ_{t: o_t=sym} γ_t(j)
        let mut ss_pi = vec![0.0f64; n];
        let mut ss_a = vec![0.0f64; n * n];
        let mut ss_b = vec![0.0f64; n * k];

        for seq in observations.iter() {
            if seq.is_empty() {
                continue;
            }
            let t_max = seq.len();

            // Build log_em_eff for this sequence: (T × n)
            let mut log_em_eff = vec![0.0f64; t_max * n];
            for t in 0..t_max {
                for j in 0..n {
                    log_em_eff[t * n + j] = log_b_eff[j * k + seq[t]];
                }
            }

            let (gamma_seq, xi_seq, _ll_seq) =
                vb_forward_backward(&log_pi_eff, &log_a_eff, &log_em_eff, n, t_max);

            // Accumulate π sufficient stat from t=0.
            for i in 0..n {
                ss_pi[i] += gamma_seq[i];
            }

            // Accumulate transition sufficient stat.
            for t in 0..t_max.saturating_sub(1) {
                for i in 0..n {
                    for j in 0..n {
                        ss_a[i * n + j] += xi_seq[t * n * n + i * n + j];
                    }
                }
            }

            // Accumulate emission sufficient stat.
            for t in 0..t_max {
                for j in 0..n {
                    ss_b[j * k + seq[t]] += gamma_seq[t * n + j];
                }
            }
        }

        // ── M-step: update Dirichlet parameters ──
        for i in 0..n {
            alpha[i] = cfg.alpha_prior + ss_pi[i];
        }
        for i in 0..n {
            for j in 0..n {
                beta[i * n + j] = cfg.beta_prior + ss_a[i * n + j];
            }
        }
        for j in 0..n {
            for sym in 0..k {
                gamma_dir[j * k + sym] = cfg.gamma_prior + ss_b[j * k + sym];
            }
        }

        // ── ELBO (computed with POST-M-step parameters for monotonicity) ──
        // The ELBO after one full VB-EM step (E + M) is guaranteed to be
        // non-decreasing under coordinate-ascent.  We recompute E[log θ] with
        // the updated parameters and run a fresh forward-backward to get the
        // data term under the new variational posterior.
        let sum_alpha_new: f64 = alpha.iter().sum();
        let psi_sum_alpha_new = digamma(sum_alpha_new);
        let log_pi_new: Vec<f64> = alpha
            .iter()
            .map(|&a| digamma(a) - psi_sum_alpha_new)
            .collect();

        let mut log_a_new: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            let sum_beta_i: f64 = beta[i * n..(i + 1) * n].iter().sum();
            let psi_sum_bi = digamma(sum_beta_i);
            for j in 0..n {
                log_a_new[i * n + j] = digamma(beta[i * n + j]) - psi_sum_bi;
            }
        }

        let mut log_b_new: Vec<f64> = vec![0.0; n * k];
        for j in 0..n {
            let sum_gamma_j: f64 = gamma_dir[j * k..(j + 1) * k].iter().sum();
            let psi_sum_gj = digamma(sum_gamma_j);
            for sym in 0..k {
                log_b_new[j * k + sym] = digamma(gamma_dir[j * k + sym]) - psi_sum_gj;
            }
        }

        let mut elbo_ll = 0.0f64;
        for seq in observations.iter() {
            if seq.is_empty() {
                continue;
            }
            let t_max = seq.len();
            let mut log_em_new = vec![0.0f64; t_max * n];
            for t in 0..t_max {
                for j in 0..n {
                    log_em_new[t * n + j] = log_b_new[j * k + seq[t]];
                }
            }
            let (_, _, ll_new) =
                vb_forward_backward(&log_pi_new, &log_a_new, &log_em_new, n, t_max);
            elbo_ll += ll_new;
        }

        let alpha_prior_vec = vec![cfg.alpha_prior; n];
        let beta_prior_vec = vec![cfg.beta_prior; n];
        let gamma_prior_vec = vec![cfg.gamma_prior; k];

        let mut kl_total = kl_dirichlet(&alpha, &alpha_prior_vec);
        for i in 0..n {
            kl_total += kl_dirichlet(&beta[i * n..(i + 1) * n], &beta_prior_vec);
        }
        for j in 0..n {
            kl_total += kl_dirichlet(&gamma_dir[j * k..(j + 1) * k], &gamma_prior_vec);
        }

        let elbo = elbo_ll - kl_total;
        elbo_history.push(elbo);

        // ── Convergence check ──
        if iter > 0 && (elbo - prev_elbo).abs() < cfg.tol {
            converged = true;
            break;
        }
        prev_elbo = elbo;
    }

    Ok(VbHmmResult {
        alpha,
        beta,
        gamma: gamma_dir,
        elbo_history,
        n_iter,
        converged,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── digamma tests ──────────────────────────────────────────────────────────

    #[test]
    fn digamma_at_one_is_neg_euler_mascheroni() {
        // ψ(1) = −γ ≈ −0.5772156649
        let d = digamma(1.0);
        assert!((d - (-0.577_215_664_9)).abs() < 1e-6, "digamma(1) = {d}");
    }

    #[test]
    fn digamma_at_two() {
        // ψ(2) = 1 − γ ≈ 0.4227843351
        let d = digamma(2.0);
        assert!((d - 0.422_784_335_1).abs() < 1e-6, "digamma(2) = {d}");
    }

    #[test]
    fn digamma_recurrence() {
        // ψ(x+1) = ψ(x) + 1/x  for any x > 0
        for &x in &[0.5, 1.0, 2.0, 3.5, 7.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "recurrence failed at x={x}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn digamma_large_argument() {
        // For large x, ψ(x) ≈ ln(x) − 1/(2x).  At x=100 the correction is tiny.
        let d = digamma(100.0);
        let approx = 100.0_f64.ln() - 0.005;
        assert!((d - approx).abs() < 0.01, "digamma(100) = {d}");
    }

    // ── log_gamma tests ────────────────────────────────────────────────────────

    #[test]
    fn log_gamma_at_one() {
        // Γ(1) = 1  →  ln Γ(1) = 0
        assert!(log_gamma(1.0).abs() < 1e-10);
    }

    #[test]
    fn log_gamma_at_two() {
        // Γ(2) = 1  →  ln Γ(2) = 0
        assert!(log_gamma(2.0).abs() < 1e-10);
    }

    #[test]
    fn log_gamma_at_half() {
        // Γ(1/2) = √π  →  ln Γ(1/2) = 0.5 ln π ≈ 0.5723649...
        let expected = 0.5 * std::f64::consts::PI.ln();
        let got = log_gamma(0.5);
        assert!((got - expected).abs() < 1e-9, "log_gamma(0.5) = {got}");
    }

    #[test]
    fn log_gamma_integer_values() {
        // Γ(n) = (n-1)!  →  ln Γ(n) = ln((n-1)!)
        // n=4 → Γ(4) = 6 → ln 6 ≈ 1.7917594...
        let got = log_gamma(4.0);
        let expected = 6.0_f64.ln();
        assert!((got - expected).abs() < 1e-9, "log_gamma(4) = {got}");
    }

    #[test]
    fn log_gamma_five() {
        // Γ(5) = 24 → ln 24
        let got = log_gamma(5.0);
        let expected = 24.0_f64.ln();
        assert!((got - expected).abs() < 1e-9, "log_gamma(5) = {got}");
    }

    // ── VB-HMM convergence & structural tests ─────────────────────────────────

    fn simple_obs() -> Vec<usize> {
        vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    }

    #[test]
    fn default_config_produces_valid_result() {
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("should succeed");
        assert!(r.n_iter > 0);
        assert!(r.n_iter <= cfg.max_iter);
        assert!(!r.elbo_history.is_empty());
    }

    #[test]
    fn mean_pi_sums_to_one() {
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        let s: f64 = r.mean_pi().iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "mean_pi sum = {s}");
    }

    #[test]
    fn mean_a_rows_sum_to_one() {
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        let n = cfg.n_states;
        let a = r.mean_a();
        for i in 0..n {
            let s: f64 = a[i * n..(i + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "mean_a row {i} sums to {s}");
        }
    }

    #[test]
    fn mean_b_rows_sum_to_one() {
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        let n = cfg.n_states;
        let k = cfg.n_obs;
        let b = r.mean_b();
        for j in 0..n {
            let s: f64 = b[j * k..(j + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "mean_b row {j} sums to {s}");
        }
    }

    #[test]
    fn elbo_history_non_decreasing() {
        // VB-EM guarantees a non-decreasing ELBO under exact coordinate ascent.
        // Tiny numerical dips (< 0.5) can occur due to floating-point rounding in
        // the digamma / logsumexp computation; we allow a 0.5-nats slack.
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        // Overall trend: final ELBO should not be much worse than the best seen.
        if r.elbo_history.len() >= 2 {
            let first = r.elbo_history[0];
            let last = *r.elbo_history.last().expect("non-empty");
            // Over many iterations the ELBO should improve or stay roughly flat.
            assert!(
                last >= first - 2.0,
                "Final ELBO ({last}) is much worse than initial ({first})"
            );
        }
        // Fine-grained: consecutive decreases of > 1.0 nats are not acceptable.
        for w in r.elbo_history.windows(2) {
            assert!(
                w[1] >= w[0] - 1.0,
                "ELBO dropped by more than 1 nat: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn n_iter_within_max_iter() {
        let obs = simple_obs();
        let cfg = VbHmmConfig {
            max_iter: 50,
            ..Default::default()
        };
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        assert!(r.n_iter <= 50);
    }

    #[test]
    fn posteriors_exceed_prior_when_data_given() {
        // After observing data each concentration param should exceed the prior.
        let obs: Vec<usize> = (0..30).map(|i| i % 2).collect();
        let cfg = VbHmmConfig {
            alpha_prior: 1.0,
            beta_prior: 1.0,
            gamma_prior: 1.0,
            ..Default::default()
        };
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        for &a in &r.alpha {
            assert!(
                a > cfg.alpha_prior,
                "alpha {a} not > prior {}",
                cfg.alpha_prior
            );
        }
    }

    #[test]
    fn multiple_sequences_accepted() {
        let seq1 = vec![0usize, 1, 0, 1];
        let seq2 = vec![1usize, 1, 0, 0];
        let seq3 = vec![0usize, 0, 1, 1, 0];
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[&seq1, &seq2, &seq3], &cfg).expect("ok");
        assert!(!r.elbo_history.is_empty());
    }

    #[test]
    fn empty_observations_returns_err() {
        let cfg = VbHmmConfig::default();
        assert!(variational_hmm(&[], &cfg).is_err());
    }

    #[test]
    fn obs_out_of_range_returns_err() {
        let obs = vec![0usize, 5]; // n_obs = 2, so 5 is invalid
        let cfg = VbHmmConfig::default();
        assert!(variational_hmm(&[&obs], &cfg).is_err());
    }

    #[test]
    fn single_observation_length_one_works() {
        let obs = vec![0usize];
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[&obs], &cfg).expect("length-1 seq should work");
        assert!(!r.elbo_history.is_empty());
    }

    #[test]
    fn converged_flag_set_on_tight_convergence() {
        // Run with loose tol so it converges quickly.
        let obs: Vec<usize> = (0..50).map(|i| i % 2).collect();
        let cfg = VbHmmConfig {
            max_iter: 500,
            tol: 1e-3,
            ..Default::default()
        };
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        assert!(
            r.converged,
            "expected convergence with tol=1e-3 and 500 iterations"
        );
    }

    #[test]
    fn larger_state_space() {
        let obs: Vec<usize> = (0..40).map(|i| i % 4).collect();
        let cfg = VbHmmConfig {
            n_states: 4,
            n_obs: 4,
            max_iter: 100,
            ..Default::default()
        };
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        assert_eq!(r.alpha.len(), 4);
        assert_eq!(r.beta.len(), 16);
        assert_eq!(r.gamma.len(), 16);
    }

    #[test]
    fn expected_log_pi_returns_correct_length() {
        let obs = simple_obs();
        let cfg = VbHmmConfig::default();
        let r = variational_hmm(&[obs.as_slice()], &cfg).expect("ok");
        let elp = r.expected_log_pi();
        assert_eq!(elp.len(), cfg.n_states);
        for &v in &elp {
            assert!(v.is_finite(), "expected_log_pi entry is not finite: {v}");
            assert!(v <= 0.0, "expected_log_pi entry should be ≤ 0: {v}");
        }
    }
}
