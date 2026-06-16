//! Dirichlet process: Chinese Restaurant Process, stick-breaking (GEM), and a
//! collapsed-Gibbs DP-Gaussian-mixture sampler.
//!
//! The Dirichlet process `DP(α, G₀)` is a distribution over distributions, used
//! as a non-parametric Bayesian prior for mixture models with an unbounded
//! number of components. This module gives three concrete, interoperable views:
//!
//! - **Chinese Restaurant Process (CRP)** — the predictive (Pólya-urn) rule for
//!   partitions: customer `n+1` joins an occupied table with probability
//!   proportional to its size and starts a new table with probability
//!   `α / (α + n)` (Blackwell & MacQueen 1973; Aldous 1985).
//! - **Stick-breaking / GEM weights** — Sethuraman's (1994) constructive
//!   representation: `β_k ~ Beta(1, α)`, `π_k = β_k ∏_{j<k}(1 − β_j)`, giving an
//!   explicit (a.s. summing-to-one) infinite weight vector.
//! - **DP-mixture Gibbs sampler** — Neal's (2000) Algorithm 3 collapsed Gibbs
//!   sampler for a conjugate Normal–Normal mixture, which clusters 1-D data with
//!   an automatically inferred number of components.
//!
//! # References
//! - Ferguson, T.S. (1973). *A Bayesian analysis of some nonparametric problems.*
//!   Ann. Statist. 1(2):209-230.
//! - Sethuraman, J. (1994). *A constructive definition of Dirichlet priors.*
//!   Statist. Sinica 4:639-650.
//! - Blackwell, D. & MacQueen, J.B. (1973). *Ferguson distributions via Pólya urn
//!   schemes.* Ann. Statist. 1(2):353-355.
//! - Neal, R.M. (2000). *Markov chain sampling methods for Dirichlet process
//!   mixture models.* J. Comput. Graph. Statist. 9(2):249-265.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Beta(1, α) sampling — the stick-breaking proportions.
// ---------------------------------------------------------------------------

/// Draw from `Beta(1, α)` using the inverse-CDF: if `β ~ Beta(1, α)` then
/// `1 − β = U^{1/α}` with `U ~ Uniform(0,1)`, i.e. `β = 1 − U^{1/α}`.
fn sample_beta_1_alpha(alpha: f64, rng: &mut LcgRng) -> f64 {
    let u = rng.next_f64().clamp(1e-300, 1.0);
    (1.0 - u.powf(1.0 / alpha)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Stick-breaking (GEM) weights
// ---------------------------------------------------------------------------

/// A truncated stick-breaking weight vector and its remaining mass.
#[derive(Debug, Clone)]
pub struct StickBreakingWeights {
    /// Weights `π₁,…,π_K` (truncated to `K` sticks).
    pub weights: Vec<f64>,
    /// The unbroken remainder `∏_{k=1}^{K}(1 − β_k)` (the tail mass `< ε`).
    pub remaining_mass: f64,
    /// Concentration parameter `α` used to generate the sticks.
    pub alpha: f64,
}

impl StickBreakingWeights {
    /// Sum of the explicit (truncated) weights.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.weights.iter().sum()
    }
}

/// Generate `k_max` stick-breaking weights from `GEM(α)`.
///
/// `β_k ~ Beta(1, α)`, `π_k = β_k ∏_{j<k}(1 − β_j)`. The returned
/// [`StickBreakingWeights::remaining_mass`] is the unbroken tail, which decays
/// geometrically (expected factor `α/(α+1)` per stick).
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if `alpha <= 0`.
/// - [`StatsError::InsufficientSampleSize`] if `k_max == 0`.
pub fn stick_breaking_weights(
    alpha: f64,
    k_max: usize,
    rng: &mut LcgRng,
) -> StatsResult<StickBreakingWeights> {
    if !positive_finite(alpha) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".to_owned(),
            reason: format!("concentration must be > 0, got {alpha}"),
        });
    }
    if k_max == 0 {
        return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
    }
    let mut weights = Vec::with_capacity(k_max);
    let mut remaining = 1.0_f64;
    for _ in 0..k_max {
        let beta = sample_beta_1_alpha(alpha, rng);
        let pi = beta * remaining;
        weights.push(pi);
        remaining *= 1.0 - beta;
        if remaining < 1e-300 {
            remaining = 0.0;
            break;
        }
    }
    // Pad to k_max with zeros if we exited early.
    while weights.len() < k_max {
        weights.push(0.0);
    }
    Ok(StickBreakingWeights {
        weights,
        remaining_mass: remaining,
        alpha,
    })
}

// ---------------------------------------------------------------------------
// Chinese Restaurant Process
// ---------------------------------------------------------------------------

/// State of a Chinese Restaurant Process: the table index of each seated
/// customer and the per-table occupancy counts.
#[derive(Debug, Clone)]
pub struct ChineseRestaurant {
    /// Concentration parameter `α`.
    pub alpha: f64,
    /// `assignment[i]` = table index (0-based) of customer `i`.
    pub assignment: Vec<usize>,
    /// `table_counts[k]` = number of customers at table `k`.
    pub table_counts: Vec<usize>,
}

impl ChineseRestaurant {
    /// Empty restaurant with concentration `α`.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if `alpha <= 0`.
    pub fn new(alpha: f64) -> StatsResult<Self> {
        if !positive_finite(alpha) {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_owned(),
                reason: format!("concentration must be > 0, got {alpha}"),
            });
        }
        Ok(Self {
            alpha,
            assignment: Vec::new(),
            table_counts: Vec::new(),
        })
    }

    /// Number of seated customers.
    #[must_use]
    pub fn n_customers(&self) -> usize {
        self.assignment.len()
    }

    /// Number of occupied tables (clusters).
    #[must_use]
    pub fn n_tables(&self) -> usize {
        self.table_counts.len()
    }

    /// Seating probabilities for the *next* customer over the existing tables
    /// followed by the new-table probability.
    ///
    /// Returns a vector of length `n_tables() + 1`: index `k < n_tables()` holds
    /// `count_k / (α + n)`; the final entry holds the new-table probability
    /// `α / (α + n)`. The entries sum to 1.
    #[must_use]
    pub fn next_probabilities(&self) -> Vec<f64> {
        let n = self.n_customers() as f64;
        let denom = self.alpha + n;
        let mut probs = Vec::with_capacity(self.n_tables() + 1);
        for &c in &self.table_counts {
            probs.push(c as f64 / denom);
        }
        probs.push(self.alpha / denom);
        probs
    }

    /// Seat the next customer by sampling from [`Self::next_probabilities`];
    /// returns the chosen table index.
    pub fn seat_next(&mut self, rng: &mut LcgRng) -> usize {
        let probs = self.next_probabilities();
        let table = categorical_sample(&probs, rng);
        if table == self.n_tables() {
            self.table_counts.push(1);
        } else {
            self.table_counts[table] += 1;
        }
        self.assignment.push(table);
        table
    }

    /// Seat the customer at an explicit table index (creating the next new table
    /// when `table == n_tables()`), used to evaluate insertion-order invariance.
    ///
    /// # Errors
    /// [`StatsError::IndexOutOfBounds`] if `table > n_tables()`.
    pub fn seat_at(&mut self, table: usize) -> StatsResult<()> {
        let nt = self.n_tables();
        if table > nt {
            return Err(StatsError::IndexOutOfBounds {
                index: table,
                len: nt,
            });
        }
        if table == nt {
            self.table_counts.push(1);
        } else {
            self.table_counts[table] += 1;
        }
        self.assignment.push(table);
        Ok(())
    }

    /// Log-probability of the *partition* induced by the current seating under
    /// the CRP / Ewens sampling formula. This depends only on the multiset of
    /// table sizes (it is exchangeable / insertion-order invariant):
    ///
    /// ```text
    /// P(partition) = α^K · Γ(α) / Γ(α + n) · ∏_{k} (n_k − 1)!
    /// ```
    ///
    /// where `K` is the number of tables and `n_k` their sizes.
    #[must_use]
    pub fn log_partition_probability(&self) -> f64 {
        let n = self.n_customers();
        if n == 0 {
            return 0.0;
        }
        let k = self.n_tables() as f64;
        let mut lp = k * self.alpha.ln();
        // Γ(α) / Γ(α+n) = 1 / ∏_{i=0}^{n-1}(α+i).
        for i in 0..n {
            lp -= (self.alpha + i as f64).ln();
        }
        // ∏_k (n_k − 1)!  →  Σ_k ln Γ(n_k).
        for &c in &self.table_counts {
            lp += ln_factorial(c.saturating_sub(1));
        }
        lp
    }
}

/// Simulate a CRP for `n` customers from an empty restaurant.
///
/// # Errors
/// [`StatsError::InvalidParameter`] if `alpha <= 0`.
pub fn crp_simulate(alpha: f64, n: usize, rng: &mut LcgRng) -> StatsResult<ChineseRestaurant> {
    let mut crp = ChineseRestaurant::new(alpha)?;
    for _ in 0..n {
        crp.seat_next(rng);
    }
    Ok(crp)
}

// ---------------------------------------------------------------------------
// DP Gaussian mixture: collapsed Gibbs (Neal 2000, Algorithm 3)
// ---------------------------------------------------------------------------

/// Conjugate Normal base measure `G₀` for a 1-D DP-Gaussian mixture with known
/// observation variance `sigma2`. The component means have a Normal prior
/// `N(mu0, tau2)`.
#[derive(Debug, Clone)]
pub struct NormalBaseMeasure {
    /// Prior mean of the component means.
    pub mu0: f64,
    /// Prior variance of the component means.
    pub tau2: f64,
    /// Known (shared) observation variance.
    pub sigma2: f64,
}

impl NormalBaseMeasure {
    /// Construct a base measure, validating positivity of the variances.
    ///
    /// # Errors
    /// [`StatsError::InvalidParameter`] if `tau2 <= 0` or `sigma2 <= 0`.
    pub fn new(mu0: f64, tau2: f64, sigma2: f64) -> StatsResult<Self> {
        if !positive_finite(tau2) {
            return Err(StatsError::InvalidParameter {
                name: "tau2".to_owned(),
                reason: format!("prior variance must be > 0, got {tau2}"),
            });
        }
        if !positive_finite(sigma2) {
            return Err(StatsError::InvalidParameter {
                name: "sigma2".to_owned(),
                reason: format!("observation variance must be > 0, got {sigma2}"),
            });
        }
        Ok(Self { mu0, tau2, sigma2 })
    }

    /// Draw a component mean from the base measure `N(mu0, tau2)`.
    #[must_use]
    pub fn sample_mean(&self, rng: &mut LcgRng) -> f64 {
        self.mu0 + self.tau2.sqrt() * rng.next_normal()
    }

    /// Marginal (prior-predictive) density of a single observation `x` with the
    /// component mean integrated out: `N(x; mu0, tau2 + sigma2)`.
    #[must_use]
    pub fn marginal_density(&self, x: f64) -> f64 {
        normal_pdf(x, self.mu0, self.tau2 + self.sigma2)
    }
}

/// Result of a DP-mixture Gibbs run.
///
/// The reported [`Self::n_clusters`] / [`Self::assignment`] / [`Self::cluster_means`]
/// summarise the chain by the **posterior mode of `K`** (the most frequently
/// visited cluster count over the post-burn-in sweeps); the representative
/// assignment is the highest-joint-density sweep among those with `K = mode`.
/// This is the standard point summary for a DP-mixture, robust to the transient
/// singletons a single final sweep can contain.
#[derive(Debug, Clone)]
pub struct DpMixtureResult {
    /// Cluster label of each data point (relabelled `0..K`) at the representative
    /// (modal-`K`, highest-density) sweep.
    pub assignment: Vec<usize>,
    /// Posterior-mean estimate of each occupied cluster's mean.
    pub cluster_means: Vec<f64>,
    /// Posterior-modal number of occupied clusters.
    pub n_clusters: usize,
    /// Posterior distribution of `K`: `cluster_count_freq[k]` is the number of
    /// post-burn-in sweeps that visited exactly `k` clusters (`index 0` unused).
    pub cluster_count_freq: Vec<usize>,
    /// Number of Gibbs sweeps performed.
    pub iterations: usize,
}

/// Configuration for the DP Gaussian-mixture Gibbs sampler.
#[derive(Debug, Clone)]
pub struct DpMixtureConfig {
    /// DP concentration `α`.
    pub alpha: f64,
    /// Number of Gibbs sweeps.
    pub iterations: usize,
    /// Number of initial sweeps to discard as burn-in before collecting the
    /// posterior summary of `K`. Clamped to `< iterations`.
    pub burn_in: usize,
}

impl Default for DpMixtureConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            iterations: 300,
            burn_in: 100,
        }
    }
}

/// Internal mutable per-cluster sufficient statistics (count + sum of x).
struct ClusterStats {
    count: usize,
    sum: f64,
}

/// Fit a 1-D DP Gaussian mixture by collapsed Gibbs sampling (Neal 2000, Alg. 3).
///
/// Each sweep reassigns every point to an existing cluster (probability ∝
/// `count_k · N(x; posterior-predictive)`) or to a brand-new cluster
/// (probability ∝ `α · marginal_density(x)`). Empty clusters are removed.
///
/// # Errors
/// - [`StatsError::EmptyInput`] if `data` is empty.
/// - [`StatsError::InvalidParameter`] if `alpha <= 0`.
pub fn dp_mixture_fit(
    data: &[f64],
    base: &NormalBaseMeasure,
    config: &DpMixtureConfig,
    rng: &mut LcgRng,
) -> StatsResult<DpMixtureResult> {
    if data.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if !positive_finite(config.alpha) {
        return Err(StatsError::InvalidParameter {
            name: "alpha".to_owned(),
            reason: format!("concentration must be > 0, got {}", config.alpha),
        });
    }
    let n = data.len();
    let alpha = config.alpha;
    let burn_in = config.burn_in.min(config.iterations.saturating_sub(1));

    // Initialise: everyone in cluster 0.
    let mut labels = vec![0usize; n];
    let mut clusters: Vec<ClusterStats> = vec![ClusterStats {
        count: n,
        sum: data.iter().sum(),
    }];

    // Posterior summary of K and, for each visited K, the highest-density
    // labelling seen at that K (so the representative can match the modal K).
    let mut cluster_count_freq: Vec<usize> = vec![0; n + 1];
    let mut best_per_k: Vec<Option<(f64, Vec<usize>)>> = vec![None; n + 1];

    for sweep in 0..config.iterations {
        for i in 0..n {
            let xi = data[i];
            // Remove point i from its current cluster.
            let ci = labels[i];
            clusters[ci].count -= 1;
            clusters[ci].sum -= xi;
            if clusters[ci].count == 0 {
                // Drop the empty cluster and relabel the trailing one.
                clusters.remove(ci);
                for lbl in labels.iter_mut() {
                    if *lbl > ci {
                        *lbl -= 1;
                    }
                }
            }

            // Build assignment weights for existing clusters + a new cluster.
            let k = clusters.len();
            let mut weights = Vec::with_capacity(k + 1);
            for c in &clusters {
                // Posterior-predictive N(x; m_post, s2_post + sigma2) of cluster c.
                let (m_post, s2_post) = posterior_mean_var(c.count, c.sum, base);
                let pred = normal_pdf(xi, m_post, s2_post + base.sigma2);
                weights.push(c.count as f64 * pred);
            }
            // New cluster: α · prior-predictive.
            weights.push(alpha * base.marginal_density(xi));

            let choice = categorical_sample(&weights, rng);
            if choice == k {
                clusters.push(ClusterStats { count: 1, sum: xi });
                labels[i] = k;
            } else {
                clusters[choice].count += 1;
                clusters[choice].sum += xi;
                labels[i] = choice;
            }
        }

        // Collect the posterior summary only after burn-in.
        if sweep >= burn_in {
            let k = clusters.len();
            cluster_count_freq[k] += 1;
            let logdens = joint_log_density(data, &labels, k, base);
            let slot = &mut best_per_k[k];
            if slot.as_ref().is_none_or(|(prev, _)| logdens > *prev) {
                *slot = Some((logdens, labels.clone()));
            }
        }
    }

    // Posterior mode of K over the collected sweeps; fall back to the final
    // state's K if no post-burn-in sweep was recorded.
    let modal_k = cluster_count_freq
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|&(_, &f)| f)
        .map(|(k, _)| k)
        .filter(|&k| cluster_count_freq[k] > 0)
        .unwrap_or(clusters.len());

    // Representative assignment = best-density labelling recorded at the modal K
    // (guaranteed present when modal_k came from the frequency table).
    let rep_labels = best_per_k[modal_k]
        .as_ref()
        .map(|(_, lbls)| lbls.clone())
        .unwrap_or_else(|| labels.clone());
    let rep_k = modal_k;

    // Recompute per-cluster sufficient statistics for the representative labels.
    let mut sums = vec![0.0_f64; rep_k];
    let mut counts = vec![0usize; rep_k];
    for (i, &lbl) in rep_labels.iter().enumerate() {
        if lbl < rep_k {
            sums[lbl] += data[i];
            counts[lbl] += 1;
        }
    }
    let mut cluster_means = Vec::with_capacity(rep_k);
    for j in 0..rep_k {
        let (m_post, _) = posterior_mean_var(counts[j], sums[j], base);
        cluster_means.push(m_post);
    }

    Ok(DpMixtureResult {
        n_clusters: rep_k,
        assignment: rep_labels,
        cluster_means,
        cluster_count_freq,
        iterations: config.iterations,
    })
}

/// Posterior mean and variance of a cluster mean given `count` observations with
/// running `sum`, under the conjugate Normal base measure.
///
/// `1/s² = 1/τ² + count/σ²`, `m = s² · (mu0/τ² + sum/σ²)`.
fn posterior_mean_var(count: usize, sum: f64, base: &NormalBaseMeasure) -> (f64, f64) {
    let prec = 1.0 / base.tau2 + count as f64 / base.sigma2;
    let s2 = 1.0 / prec;
    let m = s2 * (base.mu0 / base.tau2 + sum / base.sigma2);
    (m, s2)
}

/// Marginal log-likelihood of `n_k` i.i.d. Normal observations sharing one
/// (integrated-out) mean under the conjugate Normal base measure.
///
/// For `x ~ N(μ, σ²)`, `μ ~ N(μ₀, τ²)`, the marginal of a block with sample sum
/// `Σx` and sum-of-squares `Σx²` is (Murphy 2007, Eq. for Normal–Normal):
///
/// ```text
/// log p(block) = −(n_k/2) log(2π σ²) + ½ log(σ²/(σ² + n_k τ²))
///                − Σx²/(2σ²) + (τ² (Σx/σ²)² ) / (2 (1 + n_k τ²/σ²))
///                + μ₀(…) terms.
/// ```
///
/// We evaluate it directly from the posterior precision to keep it numerically
/// straightforward.
fn cluster_log_marginal(n_k: usize, sum_x: f64, sum_x2: f64, base: &NormalBaseMeasure) -> f64 {
    if n_k == 0 {
        return 0.0;
    }
    let nk = n_k as f64;
    let sigma2 = base.sigma2;
    let tau2 = base.tau2;
    let mu0 = base.mu0;
    // Posterior precision/variance of the cluster mean.
    let post_prec = 1.0 / tau2 + nk / sigma2;
    let post_var = 1.0 / post_prec;
    let post_mean = post_var * (mu0 / tau2 + sum_x / sigma2);
    // log marginal = data term + prior term − posterior term + normaliser.
    //   = −n_k/2·log(2π σ²) − Σx²/(2σ²)
    //     + ½ log(post_var/τ²) − μ₀²/(2τ²) + post_mean²/(2·post_var)·post_var?
    // Use the standard completion-of-squares result:
    let data_term = -0.5 * nk * (2.0 * PI * sigma2).ln() - sum_x2 / (2.0 * sigma2);
    let prior_term = -0.5 * (tau2).ln() - mu0 * mu0 / (2.0 * tau2);
    let post_term = -0.5 * post_var.ln() - post_mean * post_mean / (2.0 * post_var);
    data_term + prior_term - post_term
}

/// Joint log marginal-likelihood of the data given a partition `labels` with `k`
/// clusters (cluster means integrated out). Used to choose a representative
/// sweep among those with the modal number of clusters.
fn joint_log_density(data: &[f64], labels: &[usize], k: usize, base: &NormalBaseMeasure) -> f64 {
    let mut sum_x = vec![0.0_f64; k];
    let mut sum_x2 = vec![0.0_f64; k];
    let mut counts = vec![0usize; k];
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl < k {
            let x = data[i];
            sum_x[lbl] += x;
            sum_x2[lbl] += x * x;
            counts[lbl] += 1;
        }
    }
    let mut lp = 0.0;
    for j in 0..k {
        lp += cluster_log_marginal(counts[j], sum_x[j], sum_x2[j], base);
    }
    lp
}

// ---------------------------------------------------------------------------
// Small numerical helpers
// ---------------------------------------------------------------------------

/// Whether `x` is finite and strictly positive (used for hyper-parameter checks
/// in a way that also rejects `NaN`, avoiding negated partial-order comparisons).
#[inline]
fn positive_finite(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// `N(x; mean, var)` density with variance (not std-dev).
fn normal_pdf(x: f64, mean: f64, var: f64) -> f64 {
    let v = var.max(1e-300);
    let z = x - mean;
    (-(z * z) / (2.0 * v)).exp() / (2.0 * PI * v).sqrt()
}

/// `ln(n!)` for small non-negative `n` via direct summation.
fn ln_factorial(n: usize) -> f64 {
    let mut s = 0.0;
    for i in 2..=n {
        s += (i as f64).ln();
    }
    s
}

/// Sample an index from non-negative (unnormalised) `weights` via the inverse
/// CDF. Falls back to the last index if all weights are zero/non-finite.
fn categorical_sample(weights: &[f64], rng: &mut LcgRng) -> usize {
    let total: f64 = weights.iter().map(|w| w.max(0.0)).sum();
    if !(total.is_finite() && total > 0.0) {
        return weights.len().saturating_sub(1);
    }
    let mut u = rng.next_f64() * total;
    for (i, &w) in weights.iter().enumerate() {
        u -= w.max(0.0);
        if u <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- (a) stick-breaking weights sum ≈ 1; E[#clusters] grows ~ α log n --
    #[test]
    fn stick_breaking_sums_to_one() {
        let mut rng = LcgRng::new(7);
        let sb = stick_breaking_weights(2.0, 400, &mut rng)
            .expect("stick_breaking_weights should succeed");
        let s = sb.sum();
        assert!(
            sb.remaining_mass < 1e-6,
            "tail mass {} too big",
            sb.remaining_mass
        );
        assert!(
            (s + sb.remaining_mass - 1.0).abs() < 1e-9,
            "sum+tail≠1: {s}"
        );
        assert!((s - 1.0).abs() < 1e-6, "truncated sum {s} ≉ 1");
        for &w in &sb.weights {
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn expected_clusters_grow_with_log_n() {
        // E[#tables] for the CRP ≈ α·ln(1 + n/α); should increase with n.
        let alpha = 1.5_f64;
        let mut small = 0.0;
        let mut large = 0.0;
        let reps = 40;
        for s in 0..reps {
            let mut rng = LcgRng::new(1000 + s);
            small += crp_simulate(alpha, 50, &mut rng)
                .expect("crp_simulate should succeed")
                .n_tables() as f64;
            let mut rng2 = LcgRng::new(5000 + s);
            large += crp_simulate(alpha, 800, &mut rng2)
                .expect("crp_simulate should succeed")
                .n_tables() as f64;
        }
        small /= reps as f64;
        large /= reps as f64;
        assert!(
            large > small,
            "more customers should give more tables: {small} vs {large}"
        );
        // Loose check against the α·ln(1+n/α) growth law.
        let predicted_large = alpha * (1.0 + 800.0 / alpha).ln();
        assert!(
            (large - predicted_large).abs() < 0.4 * predicted_large,
            "E[#tables]={large} far from α·ln(1+n/α)={predicted_large}"
        );
    }

    // ---- (b) CRP assignment probabilities correct ------------------------
    #[test]
    fn crp_next_probabilities_correct() {
        let mut crp = ChineseRestaurant::new(2.0).expect("new should succeed");
        // Seat customers to make tables of sizes [3, 1].
        crp.seat_at(0).expect("seat_at should succeed");
        crp.seat_at(0).expect("seat_at should succeed");
        crp.seat_at(0).expect("seat_at should succeed");
        crp.seat_at(1).expect("seat_at should succeed");
        // n = 4, α = 2 ⇒ denom = 6.
        let probs = crp.next_probabilities();
        assert_eq!(probs.len(), 3); // tables 0,1 + new
        assert!((probs[0] - 3.0 / 6.0).abs() < 1e-12, "table0 {}", probs[0]);
        assert!((probs[1] - 1.0 / 6.0).abs() < 1e-12, "table1 {}", probs[1]);
        assert!((probs[2] - 2.0 / 6.0).abs() < 1e-12, "new {}", probs[2]);
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "probs sum {s} ≠ 1");
    }

    // ---- (c) α controls cluster count ------------------------------------
    #[test]
    fn concentration_controls_cluster_count() {
        let n = 500usize;
        let reps = 30;
        let mut low = 0.0;
        let mut high = 0.0;
        for s in 0..reps {
            let mut r1 = LcgRng::new(20_000 + s);
            low += crp_simulate(0.3, n, &mut r1)
                .expect("crp_simulate should succeed")
                .n_tables() as f64;
            let mut r2 = LcgRng::new(40_000 + s);
            high += crp_simulate(8.0, n, &mut r2)
                .expect("crp_simulate should succeed")
                .n_tables() as f64;
        }
        low /= reps as f64;
        high /= reps as f64;
        assert!(
            high > low + 2.0,
            "higher α should give more clusters: α=0.3→{low}, α=8→{high}"
        );
    }

    // ---- (d) DP-mixture recovers true cluster count on separated Gaussians -
    #[test]
    fn dp_mixture_recovers_cluster_count() {
        // Three well-separated clusters around -8, 0, +8 with σ=0.6.
        let mut rng = LcgRng::new(2718);
        let centres = [-8.0, 0.0, 8.0];
        let mut data = Vec::new();
        for &c in &centres {
            for _ in 0..40 {
                data.push(c + 0.6 * rng.next_normal());
            }
        }
        // Base-measure scale reflects the spread of the cluster means (±8) and a
        // small concentration strongly favours a parsimonious clustering.
        let base = NormalBaseMeasure::new(0.0, 50.0, 0.6 * 0.6).expect("new should succeed");
        let cfg = DpMixtureConfig {
            alpha: 0.1,
            iterations: 400,
            burn_in: 150,
        };
        let res =
            dp_mixture_fit(&data, &base, &cfg, &mut rng).expect("dp_mixture_fit should succeed");
        assert_eq!(
            res.n_clusters, 3,
            "expected 3 clusters (modal K), got {} (means {:?}, K-freq {:?})",
            res.n_clusters, res.cluster_means, res.cluster_count_freq
        );
        // The modal K must indeed be the most frequent post-burn-in count.
        let modal = res
            .cluster_count_freq
            .iter()
            .enumerate()
            .skip(1)
            .max_by_key(|&(_, &f)| f)
            .map(|(k, _)| k)
            .expect("value should be present");
        assert_eq!(modal, 3, "posterior mode of K should be 3");
        // Recovered means should be near the true centres (order-agnostic).
        let mut means = res.cluster_means.clone();
        means.sort_by(|a, b| a.partial_cmp(b).expect("partial_cmp should succeed"));
        for (m, c) in means.iter().zip(centres.iter()) {
            assert!((m - c).abs() < 1.0, "cluster mean {m} far from {c}");
        }
    }

    // ---- (e) CRP exchangeability: partition prob invariant to order ------
    #[test]
    fn crp_partition_probability_exchangeable() {
        // Order A: customers seated 0,0,1,0,1,2 (sizes 3,2,1).
        let mut a = ChineseRestaurant::new(1.7).expect("new should succeed");
        for &t in &[0usize, 0, 1, 0, 1, 2] {
            a.seat_at(t).expect("seat_at should succeed");
        }
        // Order B: a permuted seating yielding the SAME partition sizes (3,2,1).
        let mut b = ChineseRestaurant::new(1.7).expect("new should succeed");
        for &t in &[0usize, 1, 2, 0, 1, 0] {
            b.seat_at(t).expect("seat_at should succeed");
        }
        // Both must have identical multiset of table sizes.
        let mut sa = a.table_counts.clone();
        let mut sb = b.table_counts.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(sa, sb, "constructed partitions must match in sizes");
        let pa = a.log_partition_probability();
        let pb = b.log_partition_probability();
        assert!(
            (pa - pb).abs() < 1e-12,
            "partition log-prob must be order-invariant: {pa} vs {pb}"
        );
    }

    #[test]
    fn crp_partition_probability_matches_closed_form() {
        // Single explicit check of the Ewens formula for a small partition.
        // n=3, α=2, partition {0,0,1} (sizes 2,1):
        //   P = α^2 · Γ(α)/Γ(α+3) · (2-1)!·(1-1)!
        //     = 4 / [(α)(α+1)(α+2)] · 1 = 4 / (2·3·4) = 4/24 = 1/6.
        let mut crp = ChineseRestaurant::new(2.0).expect("new should succeed");
        crp.seat_at(0).expect("seat_at should succeed");
        crp.seat_at(0).expect("seat_at should succeed");
        crp.seat_at(1).expect("seat_at should succeed");
        let p = crp.log_partition_probability().exp();
        assert!((p - 1.0 / 6.0).abs() < 1e-12, "Ewens prob {p} ≠ 1/6");
    }

    // ---- (f) base-measure draws are finite/valid -------------------------
    #[test]
    fn base_measure_draws_finite() {
        let base = NormalBaseMeasure::new(1.0, 4.0, 0.25).expect("new should succeed");
        let mut rng = LcgRng::new(99);
        for _ in 0..1000 {
            let m = base.sample_mean(&mut rng);
            assert!(m.is_finite(), "base draw {m} not finite");
        }
        // Marginal density is a proper (positive, finite) density.
        for &x in &[-5.0, 0.0, 1.0, 5.0] {
            let d = base.marginal_density(x);
            assert!(
                d.is_finite() && d > 0.0,
                "marginal density {d} invalid at {x}"
            );
        }
    }

    // ---- validation paths -------------------------------------------------
    #[test]
    fn invalid_alpha_rejected() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            stick_breaking_weights(0.0, 10, &mut rng),
            Err(StatsError::InvalidParameter { .. })
        ));
        assert!(ChineseRestaurant::new(-1.0).is_err());
    }

    #[test]
    fn base_measure_rejects_bad_variance() {
        assert!(NormalBaseMeasure::new(0.0, 0.0, 1.0).is_err());
        assert!(NormalBaseMeasure::new(0.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn dp_mixture_empty_input_errors() {
        let base = NormalBaseMeasure::new(0.0, 1.0, 1.0).expect("new should succeed");
        let cfg = DpMixtureConfig::default();
        let mut rng = LcgRng::new(3);
        assert!(matches!(
            dp_mixture_fit(&[], &base, &cfg, &mut rng),
            Err(StatsError::EmptyInput)
        ));
    }

    #[test]
    fn crp_assignment_consistent_with_counts() {
        let mut rng = LcgRng::new(123);
        let crp = crp_simulate(2.0, 200, &mut rng).expect("crp_simulate should succeed");
        // Counts derived from assignments must equal table_counts.
        let mut derived = vec![0usize; crp.n_tables()];
        for &a in &crp.assignment {
            derived[a] += 1;
        }
        assert_eq!(derived, crp.table_counts);
        assert_eq!(crp.assignment.len(), 200);
    }
}
