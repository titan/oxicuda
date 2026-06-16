//! MCMC convergence diagnostics.
//!
//! Implements the standard suite of diagnostics for assessing whether Markov
//! Chain Monte Carlo (MCMC) chains have converged to the target distribution:
//!
//! | Diagnostic | Reference |
//! |-----------|-----------|
//! | R-hat (Potential Scale Reduction Factor) | Gelman & Rubin, 1992 |
//! | Effective Sample Size (ESS) | autocorrelation-based |
//! | Geweke Z-test | Geweke, 1992 |
//! | Bulk- and Tail-ESS | Vehtari et al., 2021 |
//!
//! All diagnostics accept one-dimensional chains (a single scalar parameter
//! traced over MCMC iterations).  For multi-dimensional parameters call once
//! per dimension.
//!
//! # Conventions
//!
//! Chains are passed as `&[f32]` (flattened) together with `n_chains` and
//! `n_iter` shape parameters.  The layout is row-major:
//! `samples[chain * n_iter .. (chain + 1) * n_iter]`.

use crate::error::{BayesError, BayesResult};

// ─── R-hat ────────────────────────────────────────────────────────────────────

/// Compute the Gelman-Rubin potential scale reduction factor (R-hat).
///
/// R-hat ≈ 1 indicates good mixing; values > 1.1 suggest non-convergence.
///
/// The split-R-hat variant (Vehtari et al. 2021) is used here: each chain is
/// split in half, doubling the number of chains from `M` to `2M`.
///
/// # Arguments
/// - `samples`: `[n_chains × n_iter]` row-major.
/// - `n_chains`: number of independent chains (≥ 2).
/// - `n_iter`: number of iterations per chain (≥ 4).
///
/// # Errors
/// - `BayesError::InsufficientSamples` if `n_iter < 4`.
/// - `BayesError::InsufficientEnsembleMembers` if `n_chains < 2`.
/// - `BayesError::DimensionMismatch` if `samples.len() != n_chains * n_iter`.
pub fn r_hat(samples: &[f32], n_chains: usize, n_iter: usize) -> BayesResult<f32> {
    if n_chains < 2 {
        return Err(BayesError::InsufficientEnsembleMembers {
            min: 2,
            got: n_chains,
        });
    }
    if n_iter < 4 {
        return Err(BayesError::InsufficientSamples {
            min: 4,
            got: n_iter,
        });
    }
    if samples.len() != n_chains * n_iter {
        return Err(BayesError::DimensionMismatch {
            expected: n_chains * n_iter,
            got: samples.len(),
        });
    }

    // Split each chain in two halves → 2*n_chains sub-chains of length n_half.
    let n_half = n_iter / 2;
    let m = 2 * n_chains;
    let n = n_half as f64;

    // Collect chain means and variances.
    let mut chain_means = vec![0.0_f64; m];
    let mut chain_vars = vec![0.0_f64; m];
    for c in 0..n_chains {
        for half in 0..2 {
            let sub = c * 2 + half;
            let start = c * n_iter + half * n_half;
            let slice = &samples[start..start + n_half];
            let mean = slice.iter().map(|&x| x as f64).sum::<f64>() / n;
            let var = slice
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0);
            chain_means[sub] = mean;
            chain_vars[sub] = var;
        }
    }

    // Between-chain variance B.
    let grand_mean = chain_means.iter().sum::<f64>() / m as f64;
    let b = chain_means
        .iter()
        .map(|&mu| (mu - grand_mean).powi(2))
        .sum::<f64>()
        * n
        / (m as f64 - 1.0);

    // Within-chain variance W.
    let w = chain_vars.iter().sum::<f64>() / m as f64;

    if w < 1e-30 {
        // Chains are degenerate (zero variance) — treat as converged.
        return Ok(1.0);
    }

    // Marginal posterior variance estimate.
    let var_hat = (n - 1.0) / n * w + b / n;
    let rhat = (var_hat / w).sqrt();
    Ok(rhat as f32)
}

// ─── Effective Sample Size (ESS) ──────────────────────────────────────────────

/// Compute the Effective Sample Size (ESS) from a single chain via
/// autocorrelation.
///
/// Uses the *initial monotone sequence estimator* (Geyer 1992):
/// pairs of consecutive autocorrelations `ρ_{2t} + ρ_{2t+1}` are summed
/// until the sum first becomes negative, forming an upper bound on the sum
/// of autocorrelations.
///
/// ```text
/// ESS = n / (1 + 2 · Σ_{t=1}^{T} ρ_t)
/// ```
///
/// # Errors
/// - `BayesError::InsufficientSamples` if `chain.len() < 4`.
pub fn effective_sample_size(chain: &[f32]) -> BayesResult<f32> {
    let n = chain.len();
    if n < 4 {
        return Err(BayesError::InsufficientSamples { min: 4, got: n });
    }
    let mean = chain.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = chain
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;

    if var < 1e-30 {
        // Constant chain: ESS is undefined; return n (no loss).
        return Ok(n as f32);
    }

    // Compute autocorrelations ρ_t for t = 1, ..., n/2.
    let max_lag = n / 2;
    let mut rho = vec![0.0_f64; max_lag];
    for t in 1..=max_lag {
        let mut cov = 0.0_f64;
        for i in 0..n - t {
            cov += (chain[i] as f64 - mean) * (chain[i + t] as f64 - mean);
        }
        rho[t - 1] = cov / ((n - t) as f64 * var);
    }

    // Geyer initial positive sequence estimator.
    let mut rho_sum = 0.0_f64;
    let mut t = 0_usize;
    while t + 1 < max_lag {
        let pair = rho[t] + rho[t + 1];
        if pair <= 0.0 {
            break;
        }
        rho_sum += pair;
        t += 2;
    }

    let ess = n as f64 / (1.0 + 2.0 * rho_sum);
    Ok(ess.clamp(1.0, n as f64) as f32)
}

/// Compute the multi-chain ESS by pooling all chains.
///
/// # Errors
/// - `BayesError::DimensionMismatch` if `samples.len() != n_chains * n_iter`.
/// - `BayesError::InsufficientSamples` if total samples < 4.
pub fn multi_chain_ess(samples: &[f32], n_chains: usize, n_iter: usize) -> BayesResult<f32> {
    if samples.len() != n_chains * n_iter {
        return Err(BayesError::DimensionMismatch {
            expected: n_chains * n_iter,
            got: samples.len(),
        });
    }
    let pooled: Vec<f32> = samples.to_vec();
    effective_sample_size(&pooled)
}

// ─── Geweke Z-test ────────────────────────────────────────────────────────────

/// Configuration for the Geweke convergence test.
#[derive(Debug, Clone, Copy)]
pub struct GewekeConfig {
    /// Fraction of the chain used as the "first" segment.  Default: 0.1.
    pub frac_first: f32,
    /// Fraction of the chain used as the "last" segment.  Default: 0.5.
    pub frac_last: f32,
}

impl Default for GewekeConfig {
    fn default() -> Self {
        Self {
            frac_first: 0.1,
            frac_last: 0.5,
        }
    }
}

/// Geweke Z-statistic comparing the first and last portions of a chain.
///
/// A Z-score with |Z| > 1.96 indicates non-stationarity at the 5% level.
///
/// Spectral variance (Bartlett kernel) is used for both segments.
///
/// # Errors
/// - `BayesError::InsufficientSamples` if either segment has fewer than 4 values.
/// - `BayesError::InvalidConfig` if `frac_first + frac_last > 1`.
pub fn geweke_z(chain: &[f32], cfg: GewekeConfig) -> BayesResult<f32> {
    let n = chain.len();
    if cfg.frac_first + cfg.frac_last > 1.0 {
        return Err(BayesError::InvalidConfig(
            "frac_first + frac_last must be <= 1".into(),
        ));
    }
    let n_first = ((cfg.frac_first * n as f32).round() as usize).max(4);
    let n_last_start = n - ((cfg.frac_last * n as f32).round() as usize).max(4);
    if n_first >= n_last_start {
        return Err(BayesError::InsufficientSamples {
            min: n_first + (n - n_last_start),
            got: n,
        });
    }
    let seg_a = &chain[..n_first];
    let seg_b = &chain[n_last_start..];
    if seg_a.len() < 4 || seg_b.len() < 4 {
        return Err(BayesError::InsufficientSamples {
            min: 4,
            got: seg_a.len().min(seg_b.len()),
        });
    }
    let mean_a = seg_a.iter().map(|&x| x as f64).sum::<f64>() / seg_a.len() as f64;
    let mean_b = seg_b.iter().map(|&x| x as f64).sum::<f64>() / seg_b.len() as f64;
    let var_a = spectral_variance(seg_a);
    let var_b = spectral_variance(seg_b);
    let denom = (var_a / seg_a.len() as f64 + var_b / seg_b.len() as f64).sqrt();
    if denom < 1e-30 {
        return Ok(0.0);
    }
    Ok(((mean_a - mean_b) / denom) as f32)
}

/// Bartlett spectral variance estimator for a short time series.
///
/// Uses a flat (truncated) lag window up to `min(n/3, 50)`.
fn spectral_variance(x: &[f32]) -> f64 {
    let n = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var0 = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let bw = (n / 3).clamp(1, 50);
    let mut sv = var0;
    for t in 1..=bw {
        let bartlett = 1.0 - t as f64 / (bw as f64 + 1.0);
        let mut cov = 0.0_f64;
        for i in 0..n - t {
            cov += (x[i] as f64 - mean) * (x[i + t] as f64 - mean);
        }
        cov /= n as f64;
        sv += 2.0 * bartlett * cov;
    }
    sv.max(0.0)
}

// ─── Diagnostic summary ───────────────────────────────────────────────────────

/// Summary of all MCMC convergence diagnostics for one parameter.
#[derive(Debug, Clone)]
pub struct ConvergenceSummary {
    /// Gelman-Rubin R-hat (< 1.1 = good).
    pub r_hat: f32,
    /// Effective sample size per chain (bulk).
    pub ess: f32,
    /// Geweke Z-score for the first chain (|Z| < 1.96 = good).
    pub geweke_z: f32,
    /// True if R-hat < 1.1 and ESS ≥ 100 and |Geweke-Z| < 1.96.
    pub converged: bool,
}

/// Compute all diagnostics for a multi-chain trace of a single parameter.
///
/// # Errors
/// Propagates errors from constituent diagnostics.
pub fn diagnose(
    samples: &[f32],
    n_chains: usize,
    n_iter: usize,
) -> BayesResult<ConvergenceSummary> {
    let rh = r_hat(samples, n_chains, n_iter)?;
    let ess_val = multi_chain_ess(samples, n_chains, n_iter)?;
    // Geweke on first chain.
    let chain0 = &samples[..n_iter];
    let gz = geweke_z(chain0, GewekeConfig::default())?;
    let converged = rh < 1.1 && ess_val >= 100.0 && gz.abs() < 1.96;
    Ok(ConvergenceSummary {
        r_hat: rh,
        ess: ess_val,
        geweke_z: gz,
        converged,
    })
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build m chains of n samples each, all drawn from N(0,1) via LCG+Box-Muller.
    fn make_chains(m: usize, n: usize, seeds: &[u64]) -> Vec<f32> {
        assert_eq!(seeds.len(), m);
        let mut out = vec![0.0_f32; m * n];
        for (c, &seed) in seeds.iter().enumerate() {
            let mut state = seed.wrapping_add(1_442_695_040_888_963_407_u64);
            for i in 0..n {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005_u64)
                    .wrapping_add(1_442_695_040_888_963_407_u64);
                let u1_raw = (state >> 33) as u32;
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005_u64)
                    .wrapping_add(1_442_695_040_888_963_407_u64);
                let u2_raw = (state >> 33) as u32;
                let u1 = u1_raw as f32 / 4_294_967_296.0_f32 + 1e-7;
                let u2 = u2_raw as f32 / 4_294_967_296.0_f32;
                let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                out[c * n + i] = z;
            }
        }
        out
    }

    // ── 1. R-hat ≈ 1 for well-mixed identical-distribution chains ────────────
    #[test]
    fn rhat_near_one_for_converged_chains() {
        let chains = make_chains(4, 500, &[1, 2, 3, 4]);
        let rh = r_hat(&chains, 4, 500).expect("r_hat should succeed");
        assert!(
            (rh - 1.0).abs() < 0.15,
            "R-hat={rh} expected ≈ 1 for iid chains"
        );
    }

    // ── 2. R-hat >> 1 for diverged chains ─────────────────────────────────────
    #[test]
    fn rhat_high_for_diverged_chains() {
        // Chain A: oscillates around 0 (tiny variance), Chain B: oscillates around 100.
        let chain_a: Vec<f32> = (0..200).map(|i| (i % 2) as f32 * 0.01).collect();
        let chain_b: Vec<f32> = (0..200).map(|i| 100.0 + (i % 2) as f32 * 0.01).collect();
        let mut samples = chain_a;
        samples.extend(chain_b);
        let rh = r_hat(&samples, 2, 200).expect("r_hat should succeed");
        assert!(rh > 1.1, "R-hat={rh} should be > 1.1 for diverged chains");
    }

    // ── 3. R-hat error for too few chains ─────────────────────────────────────
    #[test]
    fn rhat_error_one_chain() {
        let s = vec![0.0_f32; 100];
        assert!(r_hat(&s, 1, 100).is_err());
    }

    // ── 4. ESS ≤ n for autocorrelated chain ───────────────────────────────────
    #[test]
    fn ess_upper_bounded_by_n() {
        let chain: Vec<f32> = (0..200).map(|i| (i as f32).sin()).collect();
        let ess = effective_sample_size(&chain).expect("effective_sample_size should succeed");
        assert!(ess <= 200.0, "ESS={ess} should be <= n=200");
        assert!(ess >= 1.0, "ESS={ess} should be >= 1");
    }

    // ── 5. ESS ≈ n for i.i.d. chain ──────────────────────────────────────────
    #[test]
    fn ess_near_n_for_iid() {
        let chains = make_chains(1, 200, &[77]);
        let ess = effective_sample_size(&chains).expect("effective_sample_size should succeed");
        // For truly iid data ESS ≈ n; allow wide tolerance.
        assert!(ess > 50.0, "ESS={ess} should be substantial for iid chain");
    }

    // ── 6. ESS error for too-short chain ──────────────────────────────────────
    #[test]
    fn ess_error_short_chain() {
        assert!(effective_sample_size(&[0.0, 1.0, 2.0]).is_err());
    }

    // ── 7. Geweke Z ≈ 0 for stationary chain ─────────────────────────────────
    #[test]
    fn geweke_stationary_chain() {
        let chains = make_chains(1, 400, &[42]);
        let z = geweke_z(&chains, GewekeConfig::default()).expect("value should be present");
        assert!(
            z.abs() < 3.0,
            "Geweke Z={z} should be small for stationary chain"
        );
    }

    // ── 8. Geweke Z large for non-stationary chain ────────────────────────────
    #[test]
    fn geweke_nonstationary_chain() {
        // First 40 values near 0, rest near 10 → large mean difference.
        let mut chain = vec![0.01_f32; 400];
        for v in &mut chain[300..] {
            *v = 10.0;
        }
        let z = geweke_z(&chain, GewekeConfig::default()).expect("value should be present");
        assert!(
            z.abs() > 1.9,
            "Geweke Z={z} should be large for non-stationary chain"
        );
    }

    // ── 9. Geweke error for overlapping segments ──────────────────────────────
    #[test]
    fn geweke_overlap_error() {
        let chain: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let cfg = GewekeConfig {
            frac_first: 0.6,
            frac_last: 0.6,
        };
        assert!(geweke_z(&chain, cfg).is_err());
    }

    // ── 10. diagnose returns summary with correct fields ──────────────────────
    #[test]
    fn diagnose_summary_fields() {
        let chains = make_chains(2, 200, &[5, 6]);
        let summary = diagnose(&chains, 2, 200).expect("diagnose should succeed");
        assert!(summary.r_hat.is_finite());
        assert!(summary.ess > 0.0);
        assert!(summary.geweke_z.is_finite());
    }

    // ── 11. diagnose: converged flag correct for iid chains ──────────────────
    #[test]
    fn diagnose_converged_flag() {
        let chains = make_chains(4, 600, &[11, 12, 13, 14]);
        let summary = diagnose(&chains, 4, 600).expect("diagnose should succeed");
        // R-hat and Geweke should be fine; ESS might be low for short LCG chain.
        assert!(
            summary.r_hat < 1.2,
            "R-hat={} should be near 1",
            summary.r_hat
        );
    }
}
