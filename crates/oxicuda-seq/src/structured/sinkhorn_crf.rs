//! Sinkhorn CRF: entropy-regularised optimal-transport normalisation for
//! structured prediction.
//!
//! Reference: Shi, Y., Cornish, R., Doucet, A. & Teh, Y. W. (2020).
//! *Sinkhorn permutation variational marginal inference / differentiable
//! structured prediction with Sinkhorn normalisation*. The idea, used across a
//! family of papers (Adams & Zemel 2011 "Ranking via Sinkhorn propagation";
//! Mena et al. 2018 "Learning latent permutations with Gumbel-Sinkhorn";
//! Cuturi 2013 "Sinkhorn distances"), is to replace the usual
//! **row-softmax / sum-product** normalisation of a score matrix by the
//! **doubly-stochastic** projection obtained from entropic optimal transport.
//!
//! # What this module computes
//!
//! Given an `n × m` score matrix `s[i][j]` (log-affinity of source position `i`
//! to target position `j`) and a temperature `ε > 0`, the **Sinkhorn operator**
//! returns the entropy-regularised transport plan
//!
//! ```text
//! P* = argmin_{P ∈ U(a,b)}  − Σ_{ij} P[i][j] · s[i][j]  +  ε · Σ_{ij} P[i][j] (log P[i][j] − 1)
//! ```
//!
//! over the transport polytope `U(a,b) = { P ≥ 0 : P·1 = a, Pᵀ·1 = b }` with
//! prescribed margins `a ∈ ℝ^n`, `b ∈ ℝ^m` (each summing to the same mass). The
//! minimiser has the Gibbs form `P*[i][j] = u[i] · K[i][j] · v[j]` with
//! `K = exp(s/ε)`, and the scaling vectors `u, v` are found by the
//! **Sinkhorn–Knopp** fixed-point iteration
//!
//! ```text
//! u ← a ⊘ (K · v),     v ← b ⊘ (Kᵀ · u)
//! ```
//!
//! which we run **entirely in the log domain** (`log u`, `log v`, log-sum-exp)
//! for numerical stability at small `ε`.
//!
//! When `a = 1_n`, `b = 1_m` and `n = m` the plan converges to a
//! doubly-stochastic matrix; as `ε → 0` it approaches a hard permutation /
//! assignment, giving a *differentiable relaxation* of the discrete matching
//! used in structured prediction (alignment, matching CRFs, permutation
//! learning). The dual potentials `(f, g)` with `f = ε·log u`, `g = ε·log v`
//! also yield the **regularised optimal-transport cost** and a Sinkhorn-divergence
//! style structured loss.
//!
//! Production code never panics: every fallible path validates its inputs and
//! returns [`SeqError`].

use crate::error::{SeqError, SeqResult};
use crate::hmm::forward_backward::logsumexp;

/// Configuration for the Sinkhorn normalisation operator.
#[derive(Debug, Clone)]
pub struct SinkhornConfig {
    /// Entropic regularisation temperature `ε > 0`. Smaller values give a
    /// sharper (closer to a hard assignment) transport plan but need more
    /// iterations and more numerical care.
    pub epsilon: f64,
    /// Maximum number of Sinkhorn–Knopp scaling sweeps.
    pub max_iter: usize,
    /// Convergence threshold on the largest absolute change of the dual
    /// potential `log v` between successive sweeps.
    pub tol: f64,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 1000,
            tol: 1e-9,
        }
    }
}

impl SinkhornConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// * [`SeqError::InvalidParameter`] if `epsilon` is not strictly positive
    ///   and finite, or if `tol` is negative / non-finite.
    /// * [`SeqError::InvalidConfiguration`] if `max_iter == 0`.
    pub fn validate(&self) -> SeqResult<()> {
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(SeqError::InvalidParameter {
                name: "epsilon".into(),
                value: self.epsilon,
            });
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(SeqError::InvalidParameter {
                name: "tol".into(),
                value: self.tol,
            });
        }
        if self.max_iter == 0 {
            return Err(SeqError::InvalidConfiguration(
                "max_iter must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Result of running the Sinkhorn operator on a score matrix.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// Number of source rows `n`.
    pub n: usize,
    /// Number of target columns `m`.
    pub m: usize,
    /// The entropy-regularised transport plan `P*`, row-major `[n * m]`. Row
    /// sums equal the source margins `a`, column sums equal the target margins
    /// `b` (to within `tol`).
    pub plan: Vec<f64>,
    /// Source dual potential `f = ε · log u` (`[n]`).
    pub f: Vec<f64>,
    /// Target dual potential `g = ε · log v` (`[m]`).
    pub g: Vec<f64>,
    /// Regularised optimal-transport value `Σ_{ij} P*[i][j] · s[i][j]` (the
    /// expected score under the optimal plan — *larger is better* because `s`
    /// is a log-affinity / reward, not a cost).
    pub transport_score: f64,
    /// Number of scaling sweeps actually performed.
    pub iterations: usize,
    /// Largest absolute `log v` change at the final sweep (≤ `tol` on success).
    pub residual: f64,
}

impl SinkhornResult {
    /// Borrow row `i` of the transport plan (`m` entries). Returns
    /// [`SeqError::IndexOutOfBounds`] if `i >= n`.
    pub fn row(&self, i: usize) -> SeqResult<&[f64]> {
        if i >= self.n {
            return Err(SeqError::IndexOutOfBounds {
                index: i,
                len: self.n,
            });
        }
        Ok(&self.plan[i * self.m..(i + 1) * self.m])
    }

    /// Hard assignment obtained by taking, for each source row, the column of
    /// maximum transport mass (greedy arg-max decode of the soft plan). Returns
    /// a length-`n` vector of column indices.
    pub fn argmax_assignment(&self) -> Vec<usize> {
        let mut out = vec![0usize; self.n];
        for i in 0..self.n {
            let base = i * self.m;
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for j in 0..self.m {
                let v = self.plan[base + j];
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            out[i] = best;
        }
        out
    }
}

/// Run the log-domain Sinkhorn operator on an `n × m` **score** matrix `s`
/// (row-major, log-affinities; larger = stronger match) with uniform margins.
///
/// Source margins default to `1/n` per row and target margins to `1/m` per
/// column (so total mass is 1 and the plan is a coupling of two uniform
/// distributions). For the square `n == m` case this yields a doubly-stochastic
/// matrix scaled by `1/n`.
///
/// # Errors
/// * [`SeqError::EmptyInput`] if `n == 0` or `m == 0`.
/// * [`SeqError::ShapeMismatch`] if `s.len() != n * m`.
/// * [`SeqError::NumericalInstability`] if `s` contains a non-finite entry.
/// * Propagates [`SinkhornConfig::validate`] errors.
/// * [`SeqError::NotConverged`] only if the iteration produces a non-finite
///   potential (true divergence); reaching `max_iter` without hitting `tol` is
///   *not* an error — the best plan so far is returned.
pub fn sinkhorn_normalize(
    s: &[f64],
    n: usize,
    m: usize,
    config: &SinkhornConfig,
) -> SeqResult<SinkhornResult> {
    let a = vec![1.0 / n.max(1) as f64; n];
    let b = vec![1.0 / m.max(1) as f64; m];
    sinkhorn_normalize_with_margins(s, n, m, &a, &b, config)
}

/// Run the log-domain Sinkhorn operator with explicit, arbitrary margins
/// `a ∈ ℝ^n_{>0}` and `b ∈ ℝ^m_{>0}`. The margins must carry equal total mass
/// (`Σa ≈ Σb`); they are *not* required to sum to one.
///
/// # Errors
/// In addition to the error cases of [`sinkhorn_normalize`]:
/// * [`SeqError::LengthMismatch`] if `a.len() != n` or `b.len() != m`.
/// * [`SeqError::InvalidParameter`] if any margin entry is non-positive /
///   non-finite.
/// * [`SeqError::NumericalInstability`] if the total masses of `a` and `b`
///   differ by more than `1e-6` (no feasible coupling exists otherwise).
pub fn sinkhorn_normalize_with_margins(
    s: &[f64],
    n: usize,
    m: usize,
    a: &[f64],
    b: &[f64],
    config: &SinkhornConfig,
) -> SeqResult<SinkhornResult> {
    config.validate()?;
    if n == 0 || m == 0 {
        return Err(SeqError::EmptyInput);
    }
    if s.len() != n * m {
        return Err(SeqError::ShapeMismatch {
            expected: n * m,
            got: s.len(),
        });
    }
    if a.len() != n {
        return Err(SeqError::LengthMismatch { a: a.len(), b: n });
    }
    if b.len() != m {
        return Err(SeqError::LengthMismatch { a: b.len(), b: m });
    }
    for &v in s {
        if !v.is_finite() {
            return Err(SeqError::NumericalInstability(
                "score matrix contains a non-finite entry".into(),
            ));
        }
    }
    let mut mass_a = 0.0;
    for (idx, &v) in a.iter().enumerate() {
        if !(v.is_finite() && v > 0.0) {
            return Err(SeqError::InvalidParameter {
                name: format!("a[{idx}]"),
                value: v,
            });
        }
        mass_a += v;
    }
    let mut mass_b = 0.0;
    for (idx, &v) in b.iter().enumerate() {
        if !(v.is_finite() && v > 0.0) {
            return Err(SeqError::InvalidParameter {
                name: format!("b[{idx}]"),
                value: v,
            });
        }
        mass_b += v;
    }
    if (mass_a - mass_b).abs() > 1e-6 {
        return Err(SeqError::NumericalInstability(format!(
            "margin masses differ: Σa={mass_a}, Σb={mass_b}"
        )));
    }

    let eps = config.epsilon;
    // log-kernel  log K[i][j] = s[i][j] / ε
    // dual potentials in nats: log_u (length n), log_v (length m)
    let log_a: Vec<f64> = a.iter().map(|&x| x.ln()).collect();
    let log_b: Vec<f64> = b.iter().map(|&x| x.ln()).collect();

    let mut log_u = vec![0.0f64; n];
    let mut log_v = vec![0.0f64; m];

    // Scratch buffers reused each sweep to avoid per-iteration allocation.
    let mut col_buf = vec![0.0f64; n]; // over rows i, for fixed column j
    let mut row_buf = vec![0.0f64; m]; // over columns j, for fixed row i

    let mut iterations = 0usize;
    let mut residual = f64::INFINITY;
    for sweep in 0..config.max_iter {
        iterations = sweep + 1;

        // Row update:  log_u[i] = log_a[i] − logsumexp_j ( logK[i][j] + log_v[j] )
        for i in 0..n {
            let base = i * m;
            for j in 0..m {
                row_buf[j] = s[base + j] / eps + log_v[j];
            }
            let lse = logsumexp(&row_buf);
            log_u[i] = log_a[i] - lse;
            if !log_u[i].is_finite() {
                return Err(SeqError::NotConverged { iter: iterations });
            }
        }

        // Column update:  log_v[j] = log_b[j] − logsumexp_i ( logK[i][j] + log_u[i] )
        let mut max_delta = 0.0f64;
        for j in 0..m {
            for i in 0..n {
                col_buf[i] = s[i * m + j] / eps + log_u[i];
            }
            let lse = logsumexp(&col_buf);
            let new_v = log_b[j] - lse;
            if !new_v.is_finite() {
                return Err(SeqError::NotConverged { iter: iterations });
            }
            let delta = (new_v - log_v[j]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            log_v[j] = new_v;
        }

        residual = max_delta;
        if max_delta <= config.tol {
            break;
        }
    }

    // Assemble the plan  P[i][j] = exp(logK[i][j] + log_u[i] + log_v[j]).
    let mut plan = vec![0.0f64; n * m];
    let mut transport_score = 0.0f64;
    for i in 0..n {
        let base = i * m;
        for j in 0..m {
            let log_p = s[base + j] / eps + log_u[i] + log_v[j];
            let p = log_p.exp();
            plan[base + j] = p;
            transport_score += p * s[base + j];
        }
    }

    let f: Vec<f64> = log_u.iter().map(|&x| eps * x).collect();
    let g: Vec<f64> = log_v.iter().map(|&x| eps * x).collect();

    Ok(SinkhornResult {
        n,
        m,
        plan,
        f,
        g,
        transport_score,
        iterations,
        residual,
    })
}

/// A Sinkhorn-normalised structured-prediction layer over an `n × m` log-score
/// matrix.
///
/// Holds a validated [`SinkhornConfig`] and exposes the forward Sinkhorn pass
/// plus a **structured loss** that compares the predicted soft transport plan
/// against a target (gold) coupling. The loss is the (regularised) cross-style
/// disagreement
///
/// ```text
/// L = Σ_{ij} ( P_gold[i][j] − P_pred[i][j] ) · s[i][j]
/// ```
///
/// i.e. the difference between the score the gold coupling collects and the
/// score the model's optimal plan collects. It is `≥ 0` at a fixed `s` because
/// `P_pred` maximises the regularised transport score over the polytope, and it
/// is `0` exactly when the model's plan reproduces the gold coupling — the
/// structured-perceptron / max-margin signal in soft, differentiable form.
#[derive(Debug, Clone)]
pub struct SinkhornCrf {
    config: SinkhornConfig,
}

impl SinkhornCrf {
    /// Construct a layer from a configuration, validating it.
    ///
    /// # Errors
    /// Propagates [`SinkhornConfig::validate`].
    pub fn new(config: SinkhornConfig) -> SeqResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Borrow the validated configuration.
    pub fn config(&self) -> &SinkhornConfig {
        &self.config
    }

    /// Forward Sinkhorn pass with uniform margins. See
    /// [`sinkhorn_normalize`].
    ///
    /// # Errors
    /// Propagates [`sinkhorn_normalize`].
    pub fn forward(&self, s: &[f64], n: usize, m: usize) -> SeqResult<SinkhornResult> {
        sinkhorn_normalize(s, n, m, &self.config)
    }

    /// Forward Sinkhorn pass with explicit margins. See
    /// [`sinkhorn_normalize_with_margins`].
    ///
    /// # Errors
    /// Propagates [`sinkhorn_normalize_with_margins`].
    pub fn forward_with_margins(
        &self,
        s: &[f64],
        n: usize,
        m: usize,
        a: &[f64],
        b: &[f64],
    ) -> SeqResult<SinkhornResult> {
        sinkhorn_normalize_with_margins(s, n, m, a, b, &self.config)
    }

    /// Structured loss between the model's predicted plan over score matrix `s`
    /// and a `gold` coupling (row-major `[n * m]`, e.g. a hard permutation
    /// matrix). Returns `(loss, predicted_plan_result)`.
    ///
    /// The accompanying gradient of `L` w.r.t. the score matrix is
    /// `∂L/∂s[i][j] = P_gold[i][j] − P_pred[i][j]` (the classic
    /// expectation-minus-target structured gradient), available from the
    /// returned plan and `gold`.
    ///
    /// # Errors
    /// * [`SeqError::ShapeMismatch`] if `gold.len() != n * m`.
    /// * Propagates [`sinkhorn_normalize`].
    pub fn structured_loss(
        &self,
        s: &[f64],
        n: usize,
        m: usize,
        gold: &[f64],
    ) -> SeqResult<(f64, SinkhornResult)> {
        if gold.len() != n * m {
            return Err(SeqError::ShapeMismatch {
                expected: n * m,
                got: gold.len(),
            });
        }
        let pred = self.forward(s, n, m)?;
        let mut loss = 0.0f64;
        for k in 0..n * m {
            loss += (gold[k] - pred.plan[k]) * s[k];
        }
        Ok((loss, pred))
    }

    /// Gradient of the [`structured_loss`](Self::structured_loss) with respect
    /// to the score matrix, `∂L/∂s = P_gold − P_pred`, returned row-major
    /// `[n * m]` together with the loss value.
    ///
    /// # Errors
    /// Propagates [`structured_loss`](Self::structured_loss).
    pub fn structured_loss_grad(
        &self,
        s: &[f64],
        n: usize,
        m: usize,
        gold: &[f64],
    ) -> SeqResult<(f64, Vec<f64>)> {
        let (loss, pred) = self.structured_loss(s, n, m, gold)?;
        let mut grad = vec![0.0f64; n * m];
        for k in 0..n * m {
            grad[k] = gold[k] - pred.plan[k];
        }
        Ok((loss, grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row sums of the plan must match the source margins and column sums the
    /// target margins (this is the defining property of a Sinkhorn coupling).
    #[test]
    fn plan_respects_uniform_margins() {
        let n = 4;
        let m = 4;
        // Identity-favouring scores: diagonal high, off-diagonal low.
        let mut s = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                s[i * m + j] = if i == j { 3.0 } else { 0.0 };
            }
        }
        let cfg = SinkhornConfig {
            epsilon: 0.05,
            max_iter: 2000,
            tol: 1e-12,
        };
        let res = sinkhorn_normalize(&s, n, m, &cfg).expect("sinkhorn");
        // Each row sums to 1/n, each column sums to 1/m.
        for i in 0..n {
            let row_sum: f64 = res.row(i).expect("row").iter().sum();
            assert!(
                (row_sum - 1.0 / n as f64).abs() < 1e-7,
                "row {i} sum {row_sum}"
            );
        }
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| res.plan[i * m + j]).sum();
            assert!(
                (col_sum - 1.0 / m as f64).abs() < 1e-7,
                "col {j} sum {col_sum}"
            );
        }
    }

    /// With a strongly diagonal score and small ε, the argmax assignment must
    /// recover the identity permutation, and the doubly-stochastic plan times n
    /// must approach the identity matrix.
    #[test]
    fn sharpens_to_permutation() {
        let n = 5;
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = if i == j { 5.0 } else { 0.0 };
            }
        }
        let cfg = SinkhornConfig {
            epsilon: 0.02,
            max_iter: 5000,
            tol: 1e-12,
        };
        let res = sinkhorn_normalize(&s, n, n, &cfg).expect("sinkhorn");
        assert_eq!(res.argmax_assignment(), vec![0, 1, 2, 3, 4]);
        // n * P should be ~identity on the diagonal.
        for i in 0..n {
            let on_diag = n as f64 * res.plan[i * n + i];
            assert!(on_diag > 0.95, "diag {i} = {on_diag}");
        }
    }

    /// Convergence: the residual must fall below the tolerance for a
    /// well-conditioned problem.
    #[test]
    fn converges_below_tolerance() {
        let n = 3;
        let m = 3;
        let s = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let cfg = SinkhornConfig {
            epsilon: 0.2,
            max_iter: 1000,
            tol: 1e-10,
        };
        let res = sinkhorn_normalize(&s, n, m, &cfg).expect("sinkhorn");
        assert!(res.residual <= 1e-10, "residual {}", res.residual);
        assert!(res.iterations < cfg.max_iter);
    }

    /// Non-square problem with explicit non-uniform margins.
    #[test]
    fn rectangular_with_margins() {
        let n = 2;
        let m = 3;
        let s = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // a sums to 1.0, b sums to 1.0.
        let a = vec![0.6, 0.4];
        let b = vec![0.3, 0.3, 0.4];
        let cfg = SinkhornConfig {
            epsilon: 0.1,
            max_iter: 4000,
            tol: 1e-12,
        };
        let res = sinkhorn_normalize_with_margins(&s, n, m, &a, &b, &cfg).expect("sinkhorn");
        for i in 0..n {
            let row_sum: f64 = res.row(i).expect("row").iter().sum();
            assert!((row_sum - a[i]).abs() < 1e-7, "row {i} = {row_sum}");
        }
        for j in 0..m {
            let col_sum: f64 = (0..n).map(|i| res.plan[i * m + j]).sum();
            assert!((col_sum - b[j]).abs() < 1e-7, "col {j} = {col_sum}");
        }
    }

    /// Structured loss is zero when gold == predicted plan, and the gradient
    /// equals `gold − pred`. We also check the loss is non-negative when gold is
    /// the (mass-1/n) identity coupling against a diagonal score.
    #[test]
    fn structured_loss_and_grad() {
        let n = 3;
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            s[i * n + i] = 2.0;
        }
        let cfg = SinkhornConfig {
            epsilon: 0.1,
            max_iter: 3000,
            tol: 1e-12,
        };
        let layer = SinkhornCrf::new(cfg).expect("layer");
        // gold = predicted plan ⇒ loss 0, grad 0.
        let pred = layer.forward(&s, n, n).expect("fwd");
        let (loss0, grad0) = layer
            .structured_loss_grad(&s, n, n, &pred.plan)
            .expect("loss");
        assert!(loss0.abs() < 1e-9, "loss0 = {loss0}");
        for g in &grad0 {
            assert!(g.abs() < 1e-9);
        }
        // gold = mass-1/n identity coupling against diagonal score ⇒ loss ≥ 0.
        let mut gold = vec![0.0; n * n];
        for i in 0..n {
            gold[i * n + i] = 1.0 / n as f64;
        }
        let (loss1, _) = layer.structured_loss(&s, n, n, &gold).expect("loss1");
        assert!(loss1 >= -1e-9, "loss1 = {loss1}");
    }

    /// Determinism: identical inputs produce bitwise-identical plans.
    #[test]
    fn deterministic() {
        let n = 4;
        let m = 4;
        let s: Vec<f64> = (0..n * m).map(|k| (k as f64 * 0.37).sin()).collect();
        let cfg = SinkhornConfig::default();
        let r1 = sinkhorn_normalize(&s, n, m, &cfg).expect("r1");
        let r2 = sinkhorn_normalize(&s, n, m, &cfg).expect("r2");
        assert_eq!(r1.plan, r2.plan);
        assert_eq!(r1.f, r2.f);
        assert_eq!(r1.g, r2.g);
    }

    /// Input validation paths.
    #[test]
    fn validation_errors() {
        let cfg = SinkhornConfig::default();
        // empty
        assert!(sinkhorn_normalize(&[], 0, 3, &cfg).is_err());
        // shape mismatch
        assert!(sinkhorn_normalize(&[1.0, 2.0], 2, 2, &cfg).is_err());
        // non-finite score
        assert!(sinkhorn_normalize(&[f64::NAN, 0.0, 0.0, 0.0], 2, 2, &cfg).is_err());
        // bad epsilon
        let bad = SinkhornConfig {
            epsilon: 0.0,
            ..SinkhornConfig::default()
        };
        assert!(bad.validate().is_err());
        // mismatched margin masses
        let a = vec![0.5, 0.5];
        let b = vec![1.0, 1.0];
        assert!(sinkhorn_normalize_with_margins(&[0.0; 4], 2, 2, &a, &b, &cfg).is_err());
    }
}
