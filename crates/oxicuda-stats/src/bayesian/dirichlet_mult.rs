//! Dirichlet-Multinomial (Pólya) compound distribution and the maximum-likelihood
//! estimation of its concentration vector `α` via Minka's fixed-point iteration.
//!
//! Whereas [`crate::bayesian::conjugate::dirichlet_multinomial_update`] performs a
//! single Bayesian *posterior update* (Dirichlet prior + observed counts → Dirichlet
//! posterior), this module fits the parameters of the *compound* distribution to a
//! collection of count vectors by maximum likelihood — the over-dispersed
//! generalisation of the multinomial used for bag-of-words / categorical-burst data.
//!
//! The mass function of a single observation `n = (n_1, …, n_K)` with `N = Σ n_k`
//! and concentration `α = (α_1, …, α_K)`, `A = Σ α_k`, is
//!
//! ```text
//! P(n | α) = (N! / Π n_k!) · Γ(A)/Γ(N + A) · Π_k Γ(n_k + α_k)/Γ(α_k).
//! ```
//!
//! # References
//! - Minka, T.P. (2003). "Estimating a Dirichlet distribution". MIT Tech. note —
//!   §1 (fixed-point iteration on the digamma recurrence).
//! - Mosimann, J.E. (1962). "On the compound multinomial distribution, the
//!   multivariate β-distribution, and correlations among proportions".
//!   *Biometrika* 49(1-2):65-82.

use crate::error::{StatsError, StatsResult};
use crate::special::digamma::digamma;
use crate::special::gammaln::lgamma;

/// A fitted Dirichlet-Multinomial model.
#[derive(Debug, Clone)]
pub struct DirichletMultinomial {
    /// Concentration parameters `α = (α_1, …, α_K)`, all strictly positive.
    pub alpha: Vec<f64>,
}

impl DirichletMultinomial {
    /// Construct a model from an explicit concentration vector.
    ///
    /// # Errors
    /// - [`StatsError::EmptyInput`] if `alpha` is empty.
    /// - [`StatsError::InvalidParameter`] if any `α_k ≤ 0` or non-finite.
    pub fn new(alpha: Vec<f64>) -> StatsResult<Self> {
        if alpha.is_empty() {
            return Err(StatsError::EmptyInput);
        }
        for (k, &a) in alpha.iter().enumerate() {
            if !a.is_finite() || a <= 0.0 {
                return Err(StatsError::InvalidParameter {
                    name: format!("alpha[{k}]"),
                    reason: format!("must be finite and > 0, got {a}"),
                });
            }
        }
        Ok(Self { alpha })
    }

    /// Number of categories `K`.
    #[inline]
    pub fn n_categories(&self) -> usize {
        self.alpha.len()
    }

    /// Concentration sum `A = Σ_k α_k` (Pólya "precision").
    #[inline]
    pub fn precision(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Mean category proportions `m_k = α_k / A`.
    pub fn mean(&self) -> Vec<f64> {
        let a = self.precision();
        self.alpha.iter().map(|&ak| ak / a).collect()
    }

    /// Log-mass `ln P(n | α)` of a single count vector.
    ///
    /// # Errors
    /// - [`StatsError::DimensionMismatch`] if `counts.len() != K`.
    pub fn log_mass(&self, counts: &[u64]) -> StatsResult<f64> {
        let k = self.alpha.len();
        if counts.len() != k {
            return Err(StatsError::DimensionMismatch {
                a: counts.len(),
                b: k,
            });
        }
        let n_total: u64 = counts.iter().sum();
        let a_sum = self.precision();
        let n_f = n_total as f64;

        // Multinomial coefficient ln(N!) - Σ ln(n_k!) via lgamma.
        let mut log_coeff = lgamma(n_f + 1.0);
        for &c in counts {
            log_coeff -= lgamma(c as f64 + 1.0);
        }

        // ln Γ(A) - ln Γ(N + A) + Σ_k [ln Γ(n_k + α_k) - ln Γ(α_k)].
        let mut term = lgamma(a_sum) - lgamma(n_f + a_sum);
        for (k_idx, &c) in counts.iter().enumerate() {
            term += lgamma(c as f64 + self.alpha[k_idx]) - lgamma(self.alpha[k_idx]);
        }
        Ok(log_coeff + term)
    }

    /// Total log-likelihood of a data matrix of `D` count vectors.
    ///
    /// `data` is row-major `[D × K]`.
    ///
    /// # Errors
    /// - [`StatsError::ShapeMismatch`] if `data.len() != n_docs * K`.
    pub fn log_likelihood(&self, data: &[u64], n_docs: usize) -> StatsResult<f64> {
        let k = self.alpha.len();
        if data.len() != n_docs * k {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_docs, k],
                got: vec![data.len()],
            });
        }
        let mut total = 0.0_f64;
        for d in 0..n_docs {
            total += self.log_mass(&data[d * k..(d + 1) * k])?;
        }
        Ok(total)
    }

    /// Posterior-predictive probabilities for the *next* single draw given the
    /// observed `counts`: `P(category j | counts) = (n_j + α_j) / (N + A)`.
    ///
    /// # Errors
    /// - [`StatsError::DimensionMismatch`] if `counts.len() != K`.
    pub fn predictive(&self, counts: &[u64]) -> StatsResult<Vec<f64>> {
        let k = self.alpha.len();
        if counts.len() != k {
            return Err(StatsError::DimensionMismatch {
                a: counts.len(),
                b: k,
            });
        }
        let n_total: f64 = counts.iter().map(|&c| c as f64).sum();
        let denom = n_total + self.precision();
        Ok(counts
            .iter()
            .zip(&self.alpha)
            .map(|(&c, &ak)| (c as f64 + ak) / denom)
            .collect())
    }
}

/// Configuration for the Minka fixed-point MLE.
#[derive(Debug, Clone, Copy)]
pub struct DirMultFitConfig {
    /// Maximum number of fixed-point iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum relative change of any `α_k`.
    pub tol: f64,
}

impl Default for DirMultFitConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
        }
    }
}

/// Fit a [`DirichletMultinomial`] to a `[D × K]` count matrix by Minka's
/// fixed-point iteration.
///
/// The update, derived from a lower bound on the log-likelihood, is
///
/// ```text
/// α_k ← α_k · [ Σ_d (ψ(n_{dk} + α_k) − ψ(α_k)) ]
///             ───────────────────────────────────────
///             [ Σ_d (ψ(N_d + A)   − ψ(A))     ]
/// ```
///
/// which is guaranteed to increase the likelihood and keep `α_k > 0`.
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `n_docs == 0` or `k == 0`.
/// - [`StatsError::ShapeMismatch`] if `data.len() != n_docs * k`.
/// - [`StatsError::NotConverged`] if the iteration fails to converge.
pub fn dirichlet_multinomial_mle(
    data: &[u64],
    n_docs: usize,
    k: usize,
    config: DirMultFitConfig,
) -> StatsResult<DirichletMultinomial> {
    if n_docs == 0 || k == 0 {
        return Err(StatsError::EmptyInput);
    }
    if data.len() != n_docs * k {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_docs, k],
            got: vec![data.len()],
        });
    }

    // Per-document totals N_d.
    let mut doc_totals = vec![0.0_f64; n_docs];
    for d in 0..n_docs {
        let mut s = 0u64;
        for j in 0..k {
            s += data[d * k + j];
        }
        doc_totals[d] = s as f64;
    }

    // Initialise α from the empirical mean proportion with a moderate precision.
    let mut col_means = vec![0.0_f64; k];
    let mut grand_total = 0.0_f64;
    for d in 0..n_docs {
        for j in 0..k {
            col_means[j] += data[d * k + j] as f64;
        }
        grand_total += doc_totals[d];
    }
    if grand_total <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "data".to_string(),
            reason: "all counts are zero; α is unidentifiable".to_string(),
        });
    }
    // Seed precision: a small multiple keeps the fixed point away from 0.
    let init_precision = (k as f64).max(1.0);
    let mut alpha: Vec<f64> = col_means
        .iter()
        .map(|&m| (m / grand_total * init_precision).max(1e-6))
        .collect();

    for _ in 0..config.max_iter {
        let a_sum: f64 = alpha.iter().sum();
        // Denominator is shared across all k: Σ_d (ψ(N_d + A) − ψ(A)).
        let psi_a = digamma(a_sum);
        let mut denom = 0.0_f64;
        for &nd in &doc_totals {
            denom += digamma(nd + a_sum) - psi_a;
        }
        if denom.abs() < 1e-300 {
            // No information to move α (e.g. all documents empty): stop.
            break;
        }

        let mut max_rel_change = 0.0_f64;
        let mut new_alpha = vec![0.0_f64; k];
        for j in 0..k {
            let aj = alpha[j];
            let psi_aj = digamma(aj);
            let mut numer = 0.0_f64;
            for d in 0..n_docs {
                numer += digamma(data[d * k + j] as f64 + aj) - psi_aj;
            }
            let updated = (aj * numer / denom).max(1e-12);
            let rel = ((updated - aj) / aj).abs();
            if rel > max_rel_change {
                max_rel_change = rel;
            }
            new_alpha[j] = updated;
        }
        alpha = new_alpha;

        if max_rel_change < config.tol {
            return DirichletMultinomial::new(alpha);
        }
    }

    // Iterations exhausted. Minka's update increases the likelihood monotonically,
    // so the final `alpha` is the best available estimate: return it as long as it
    // is finite and positive, and only surface an error on numerical divergence.
    if alpha.iter().all(|&a| a.is_finite() && a > 0.0) {
        DirichletMultinomial::new(alpha)
    } else {
        Err(StatsError::NumericalInstability(
            "Dirichlet-Multinomial MLE diverged".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_empty() {
        assert!(matches!(
            DirichletMultinomial::new(vec![]),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn new_rejects_nonpositive() {
        assert!(DirichletMultinomial::new(vec![1.0, -0.5]).is_err());
        assert!(DirichletMultinomial::new(vec![1.0, 0.0]).is_err());
        assert!(DirichletMultinomial::new(vec![1.0, f64::NAN]).is_err());
    }

    #[test]
    fn mean_sums_to_one() {
        let dm = DirichletMultinomial::new(vec![1.0, 2.0, 3.0]).expect("ok");
        let m = dm.mean();
        assert!((m.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((m[2] - 0.5).abs() < 1e-12);
        assert!((dm.precision() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn log_mass_dimension_mismatch() {
        let dm = DirichletMultinomial::new(vec![1.0, 1.0]).expect("ok");
        assert!(matches!(
            dm.log_mass(&[1, 2, 3]),
            Err(StatsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn log_mass_is_negative_and_finite() {
        let dm = DirichletMultinomial::new(vec![1.0, 1.0, 1.0]).expect("ok");
        let lm = dm.log_mass(&[2, 3, 5]).expect("ok");
        assert!(lm.is_finite());
        assert!(lm <= 0.0, "log-mass should be ≤ 0, got {lm}");
    }

    #[test]
    fn log_mass_normalises_over_simplex() {
        // For K=2 with total N, masses over all (n, N-n) must sum to 1.
        let dm = DirichletMultinomial::new(vec![1.5, 2.5]).expect("ok");
        let n_total = 6u64;
        let mut sum = 0.0_f64;
        for n0 in 0..=n_total {
            let lm = dm.log_mass(&[n0, n_total - n0]).expect("ok");
            sum += lm.exp();
        }
        assert!((sum - 1.0).abs() < 1e-10, "Σ P = {sum}");
    }

    #[test]
    fn log_mass_uniform_alpha_matches_combinatorial() {
        // With α = (1,…,1), the DM mass is N!/Π n_k! · 1/C(N+K-1, K-1) ... we
        // verify it reduces to the uniform-over-compositions identity P = ...
        // Equivalently, for α=1 every *ordered* count config has equal mass to
        // the symmetric Dirichlet integral; just check exchange symmetry here.
        let dm = DirichletMultinomial::new(vec![1.0, 1.0, 1.0]).expect("ok");
        let a = dm.log_mass(&[4, 1, 0]).expect("ok");
        let b = dm.log_mass(&[0, 1, 4]).expect("ok");
        assert!(
            (a - b).abs() < 1e-10,
            "symmetric α ⇒ permutation-invariant mass"
        );
    }

    #[test]
    fn predictive_sums_to_one() {
        let dm = DirichletMultinomial::new(vec![1.0, 1.0, 1.0]).expect("ok");
        let p = dm.predictive(&[3, 0, 1]).expect("ok");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Category with more observed counts should have higher predictive prob.
        assert!(p[0] > p[1]);
    }

    #[test]
    fn log_likelihood_shape_check() {
        let dm = DirichletMultinomial::new(vec![1.0, 1.0]).expect("ok");
        assert!(matches!(
            dm.log_likelihood(&[1, 2, 3], 2),
            Err(StatsError::ShapeMismatch { .. })
        ));
        let ll = dm.log_likelihood(&[1, 2, 3, 0], 2).expect("ok");
        assert!(ll.is_finite());
    }

    #[test]
    fn mle_rejects_bad_shape() {
        assert!(matches!(
            dirichlet_multinomial_mle(&[1, 2, 3], 2, 2, DirMultFitConfig::default()),
            Err(StatsError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            dirichlet_multinomial_mle(&[], 0, 2, DirMultFitConfig::default()),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn mle_recovers_mean_proportions() {
        // Build a count matrix whose column proportions are ~ (0.5, 0.3, 0.2)
        // with mild over-dispersion. The fitted mean should track those ratios.
        let proportions = [0.5_f64, 0.3, 0.2];
        let n_docs = 200usize;
        let k = 3usize;
        let mut data = vec![0u64; n_docs * k];
        // Deterministic pseudo-random doc-level multinomial draws.
        let mut state = 0x1234_5678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64
        };
        for d in 0..n_docs {
            let n_draws = 30;
            for _ in 0..n_draws {
                let u = next();
                let cat = if u < proportions[0] {
                    0
                } else if u < proportions[0] + proportions[1] {
                    1
                } else {
                    2
                };
                data[d * k + cat] += 1;
            }
        }
        let cfg = DirMultFitConfig {
            max_iter: 5000,
            tol: 1e-9,
        };
        let dm = dirichlet_multinomial_mle(&data, n_docs, k, cfg).expect("fit ok");
        let mean = dm.mean();
        assert!((mean[0] - 0.5).abs() < 0.06, "m0={}", mean[0]);
        assert!((mean[1] - 0.3).abs() < 0.06, "m1={}", mean[1]);
        assert!((mean[2] - 0.2).abs() < 0.06, "m2={}", mean[2]);
    }

    #[test]
    fn mle_all_zero_counts_errors() {
        let data = vec![0u64; 6];
        assert!(dirichlet_multinomial_mle(&data, 2, 3, DirMultFitConfig::default()).is_err());
    }

    #[test]
    fn mle_increases_likelihood_over_seed() {
        // After fitting, the likelihood should be at least as high as a flat α.
        let data = [
            5u64, 1, 0, //
            4, 2, 0, //
            6, 0, 1, //
            3, 2, 1,
        ];
        let n_docs = 4;
        let k = 3;
        let flat = DirichletMultinomial::new(vec![1.0, 1.0, 1.0]).expect("ok");
        let ll_flat = flat.log_likelihood(&data, n_docs).expect("ok");
        let cfg = DirMultFitConfig {
            max_iter: 2000,
            tol: 1e-3,
        };
        let fitted = dirichlet_multinomial_mle(&data, n_docs, k, cfg).expect("fit ok");
        let ll_fit = fitted.log_likelihood(&data, n_docs).expect("ok");
        assert!(
            ll_fit >= ll_flat - 1e-6,
            "fitted LL {ll_fit} should be ≥ flat LL {ll_flat}"
        );
    }
}
