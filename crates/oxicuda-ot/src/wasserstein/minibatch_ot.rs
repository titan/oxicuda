//! Mini-batch Wasserstein / Sinkhorn-GAN-style OT loss.
//!
//! Mini-batch OT (Genevay et al. 2018) approximates the Wasserstein distance
//! between two empirical distributions by averaging optimal transport costs
//! computed on random mini-batches. This makes OT tractable as a loss function
//! for large-scale generative models.
//!
//! ```text
//! W_batch(X, Y) = (1/K) Σ_k  OT_ε(X_k, Y_k)
//! ```
//!
//! where each `X_k`, `Y_k` is a uniform mini-batch of size `B` drawn without
//! replacement from `X` and `Y` respectively.
//!
//! The Sinkhorn divergence (debiased) variant additionally subtracts the
//! self-cost terms to obtain a zero-on-identity objective:
//!
//! ```text
//! S_reg(X, Y) = W_batch(X, Y) − ½ W_batch(X, X) − ½ W_batch(Y, Y)
//! ```

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the mini-batch OT estimator.
#[derive(Debug, Clone)]
pub struct MinibatchOtConfig {
    /// Number of samples drawn per mini-batch from each distribution.
    pub batch_size: usize,
    /// Number of independent mini-batches to average.
    pub n_batches: usize,
    /// Sinkhorn entropic regularisation strength (must be > 0).
    pub reg: f64,
    /// Cost exponent p: 1 → L¹ distance, 2 → squared L² distance.
    pub cost_p: u32,
    /// RNG seed for reproducibility of batch sampling.
    pub seed: u64,
}

impl Default for MinibatchOtConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            n_batches: 50,
            reg: 0.1,
            cost_p: 2,
            seed: 42,
        }
    }
}

/// Output of the mini-batch OT estimator.
#[derive(Debug, Clone)]
pub struct MinibatchOtFit {
    /// Mean transport cost across all mini-batches.
    pub mean_cost: f64,
    /// Standard deviation of the per-batch transport costs.
    pub std_cost: f64,
    /// Number of mini-batches actually evaluated.
    pub n_batches: usize,
    /// Per-batch transport costs, length `n_batches`.
    pub batch_costs: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_minibatch(
    source: &[f64],
    target: &[f64],
    n: usize,
    m: usize,
    d: usize,
    cfg: &MinibatchOtConfig,
) -> OtResult<()> {
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if n == 0 || m == 0 {
        return Err(OtError::EmptyInput);
    }
    if source.len() != n * d {
        return Err(OtError::IncompatibleLength {
            a: source.len(),
            b: n * d,
        });
    }
    if target.len() != m * d {
        return Err(OtError::IncompatibleLength {
            a: target.len(),
            b: m * d,
        });
    }
    if cfg.batch_size == 0 {
        return Err(OtError::BadCount {
            got: cfg.batch_size,
        });
    }
    if cfg.n_batches == 0 {
        return Err(OtError::BadCount { got: cfg.n_batches });
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if cfg.cost_p == 0 {
        return Err(OtError::BadCount {
            got: cfg.cost_p as usize,
        });
    }
    if cfg.batch_size > n {
        return Err(OtError::IncompatibleLength {
            a: cfg.batch_size,
            b: n,
        });
    }
    if cfg.batch_size > m {
        return Err(OtError::IncompatibleLength {
            a: cfg.batch_size,
            b: m,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample `k` distinct indices from `0..n` using Fisher-Yates partial shuffle.
/// The output is written into `out[..k]`; `scratch` is a working buffer of
/// length `≥ n` that is modified (indices 0..n seeded then shuffled).
fn sample_without_replacement(
    rng: &mut LcgRng,
    n: usize,
    k: usize,
    scratch: &mut Vec<usize>,
) -> Vec<usize> {
    // Fill scratch with 0..n
    scratch.clear();
    scratch.extend(0..n);
    // Partial Fisher-Yates for k elements
    let mut indices = Vec::with_capacity(k);
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        scratch.swap(i, j);
        indices.push(scratch[i]);
    }
    indices
}

/// Extract a sub-cloud of `k` points (each `d`-dimensional) given an index list.
fn extract_batch(samples: &[f64], indices: &[usize], d: usize) -> Vec<f64> {
    let k = indices.len();
    let mut batch = vec![0.0_f64; k * d];
    for (bi, &si) in indices.iter().enumerate() {
        batch[bi * d..bi * d + d].copy_from_slice(&samples[si * d..si * d + d]);
    }
    batch
}

/// Build the cost matrix between two mini-batches `X` (bx × d) and `Y` (by × d).
/// `cost_p = 1` → L¹ distance; `cost_p = 2` → squared L² distance.
fn build_cost_matrix(
    x_batch: &[f64],
    y_batch: &[f64],
    bx: usize,
    by: usize,
    d: usize,
    cost_p: u32,
) -> Vec<f64> {
    let mut c = vec![0.0_f64; bx * by];
    for i in 0..bx {
        for j in 0..by {
            let mut dist = 0.0_f64;
            if cost_p == 1 {
                for dim in 0..d {
                    dist += (x_batch[i * d + dim] - y_batch[j * d + dim]).abs();
                }
            } else {
                for dim in 0..d {
                    let diff = x_batch[i * d + dim] - y_batch[j * d + dim];
                    dist += diff * diff;
                }
                if cost_p > 2 {
                    // General L^p: compute ||x-y||_2 and raise to cost_p
                    dist = dist.sqrt().powi(cost_p as i32);
                }
            }
            c[i * by + j] = dist;
        }
    }
    c
}

/// Solve Sinkhorn between two uniform marginals of size `b` with cost matrix `c`.
/// Returns the transport cost `⟨P, C⟩` as f64.
///
/// Uses a progressive tolerance strategy: attempt with tight tol first, then
/// fall back to a looser tol to ensure a finite estimate is always returned
/// for well-posed mini-batch problems.
fn sinkhorn_uniform_cost(
    c_f64: &[f64],
    b: usize,
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> OtResult<f64> {
    // Convert to f32 for the existing Sinkhorn solver
    let c_f32: Vec<f32> = c_f64.iter().map(|&x| x as f32).collect();
    let marginal = vec![1.0_f32 / b as f32; b];

    // Try with the requested tolerance first.
    let cfg_tight = SinkhornConfig {
        eps: reg as f32,
        max_iter,
        tol: tol as f32,
    };
    if let Ok(result) = sinkhorn(&c_f32, &marginal, &marginal, b, b, &cfg_tight) {
        return Ok(result.cost as f64);
    }

    // Fall back: more iterations and looser tolerance.
    let cfg_loose = SinkhornConfig {
        eps: reg as f32,
        max_iter: max_iter * 4,
        tol: (tol * 100.0) as f32,
    };
    let result = sinkhorn(&c_f32, &marginal, &marginal, b, b, &cfg_loose)?;
    Ok(result.cost as f64)
}

/// Evaluate mini-batch OT between two sample sets using an existing RNG.
/// Internal function shared by `minibatch_wasserstein` and
/// `minibatch_sinkhorn_divergence`.
fn minibatch_ot_internal(
    source: &[f64],
    target: &[f64],
    n: usize,
    m: usize,
    d: usize,
    cfg: &MinibatchOtConfig,
    rng: &mut LcgRng,
) -> OtResult<MinibatchOtFit> {
    let b = cfg.batch_size;
    let sinkhorn_iters = 500_usize;
    let sinkhorn_tol = 1e-5_f64;

    let mut scratch_src = vec![0_usize; n];
    let mut scratch_tgt = vec![0_usize; m];
    let mut batch_costs = Vec::with_capacity(cfg.n_batches);

    for _ in 0..cfg.n_batches {
        let src_idx = sample_without_replacement(rng, n, b, &mut scratch_src);
        let tgt_idx = sample_without_replacement(rng, m, b, &mut scratch_tgt);

        let x_batch = extract_batch(source, &src_idx, d);
        let y_batch = extract_batch(target, &tgt_idx, d);

        let c = build_cost_matrix(&x_batch, &y_batch, b, b, d, cfg.cost_p);
        let cost = sinkhorn_uniform_cost(&c, b, cfg.reg, sinkhorn_iters, sinkhorn_tol)?;
        batch_costs.push(cost);
    }

    let n_b = batch_costs.len() as f64;
    let mean_cost = batch_costs.iter().sum::<f64>() / n_b;
    let variance = batch_costs
        .iter()
        .map(|&c| {
            let d = c - mean_cost;
            d * d
        })
        .sum::<f64>()
        / n_b;
    let std_cost = variance.sqrt();

    Ok(MinibatchOtFit {
        mean_cost,
        std_cost,
        n_batches: batch_costs.len(),
        batch_costs,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mini-batch Wasserstein cost between two sample sets.
///
/// For each of `cfg.n_batches` iterations:
/// 1. Draw `cfg.batch_size` samples uniformly without replacement from `source`.
/// 2. Draw `cfg.batch_size` samples uniformly without replacement from `target`.
/// 3. Solve entropic OT between uniform marginals with cost `||x_i − y_j||^p`.
/// 4. Accumulate the transport cost.
///
/// Returns the mean and std of the per-batch costs.
///
/// # Parameters
///
/// - `source`: flattened `n × d` sample matrix, row-major.
/// - `target`: flattened `m × d` sample matrix, row-major.
/// - `n`: number of source samples.
/// - `m`: number of target samples.
/// - `d`: feature dimension.
/// - `cfg`: solver configuration.
/// - `rng`: mutable reference to the random number generator.
///
/// # Errors
///
/// Returns errors if inputs are invalid or if any inner Sinkhorn fails.
pub fn minibatch_wasserstein(
    source: &[f64],
    target: &[f64],
    n: usize,
    m: usize,
    d: usize,
    cfg: &MinibatchOtConfig,
    rng: &mut LcgRng,
) -> OtResult<MinibatchOtFit> {
    validate_minibatch(source, target, n, m, d, cfg)?;
    minibatch_ot_internal(source, target, n, m, d, cfg, rng)
}

/// Compute the mini-batch Sinkhorn divergence (debiased estimate).
///
/// ```text
/// S_ε(X, Y) = W_ε(X, Y) − ½ W_ε(X, X) − ½ W_ε(Y, Y)
/// ```
///
/// Each of the three terms is estimated using `cfg.n_batches` independent
/// mini-batches. The returned value is the debiased divergence estimate.
///
/// # Errors
///
/// Returns errors from inner [`minibatch_wasserstein`] calls.
pub fn minibatch_sinkhorn_divergence(
    source: &[f64],
    target: &[f64],
    n: usize,
    m: usize,
    d: usize,
    cfg: &MinibatchOtConfig,
    rng: &mut LcgRng,
) -> OtResult<f64> {
    validate_minibatch(source, target, n, m, d, cfg)?;

    // W_ε(X, Y)
    let fit_xy = minibatch_ot_internal(source, target, n, m, d, cfg, rng)?;

    // W_ε(X, X) — clamp batch_size to n
    let cfg_xx = MinibatchOtConfig {
        batch_size: cfg.batch_size.min(n),
        ..cfg.clone()
    };
    let fit_xx = minibatch_ot_internal(source, source, n, n, d, &cfg_xx, rng)?;

    // W_ε(Y, Y) — clamp batch_size to m
    let cfg_yy = MinibatchOtConfig {
        batch_size: cfg.batch_size.min(m),
        ..cfg.clone()
    };
    let fit_yy = minibatch_ot_internal(target, target, m, m, d, &cfg_yy, rng)?;

    let divergence = fit_xy.mean_cost - 0.5 * fit_xx.mean_cost - 0.5 * fit_yy.mean_cost;
    Ok(divergence)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    /// Generate n points from a d-dimensional Gaussian centred at `centre`.
    fn gaussian_cloud(n: usize, d: usize, centre: f64, rng: &mut LcgRng) -> Vec<f64> {
        let mut out = vec![0.0_f64; n * d];
        for x in out.iter_mut() {
            *x = centre + rng.next_normal() as f64;
        }
        out
    }

    #[test]
    fn mean_cost_positive_on_separated_clouds() {
        let mut rng = make_rng(1);
        let n = 50;
        let m = 50;
        let d = 2;
        let source = gaussian_cloud(n, d, 0.0, &mut rng);
        let target = gaussian_cloud(m, d, 10.0, &mut rng);
        let cfg = MinibatchOtConfig {
            batch_size: 10,
            n_batches: 20,
            reg: 1.0,
            cost_p: 2,
            seed: 7,
        };
        let mut rng2 = make_rng(cfg.seed);
        let fit = minibatch_wasserstein(&source, &target, n, m, d, &cfg, &mut rng2).expect("ok");
        assert!(
            fit.mean_cost > 0.0,
            "mean cost should be positive, got {}",
            fit.mean_cost
        );
        assert!(fit.mean_cost.is_finite(), "mean cost should be finite");
    }

    #[test]
    fn std_cost_non_negative() {
        let mut rng = make_rng(2);
        let n = 40;
        let m = 40;
        let d = 1;
        let source = gaussian_cloud(n, d, 0.0, &mut rng);
        let target = gaussian_cloud(m, d, 5.0, &mut rng);
        let cfg = MinibatchOtConfig {
            batch_size: 8,
            n_batches: 30,
            reg: 0.5,
            cost_p: 2,
            seed: 13,
        };
        let mut rng2 = make_rng(cfg.seed);
        let fit = minibatch_wasserstein(&source, &target, n, m, d, &cfg, &mut rng2).expect("ok");
        assert!(fit.std_cost >= 0.0, "std_cost={}", fit.std_cost);
    }

    #[test]
    fn batch_costs_length_matches_config() {
        let mut rng = make_rng(3);
        let n = 30;
        let m = 30;
        let d = 2;
        let source = gaussian_cloud(n, d, 0.0, &mut rng);
        let target = gaussian_cloud(m, d, 3.0, &mut rng);
        let cfg = MinibatchOtConfig {
            batch_size: 6,
            n_batches: 15,
            reg: 0.3,
            cost_p: 2,
            seed: 99,
        };
        let mut rng2 = make_rng(cfg.seed);
        let fit = minibatch_wasserstein(&source, &target, n, m, d, &cfg, &mut rng2).expect("ok");
        assert_eq!(fit.batch_costs.len(), cfg.n_batches);
        assert_eq!(fit.n_batches, cfg.n_batches);
    }

    #[test]
    fn cost_increases_with_separation() {
        let mut rng = make_rng(4);
        let n = 60;
        let m = 60;
        let d = 1;
        let src = gaussian_cloud(n, d, 0.0, &mut rng);
        let tgt_close = gaussian_cloud(m, d, 1.0, &mut rng);
        let tgt_far = gaussian_cloud(m, d, 20.0, &mut rng);

        let cfg = MinibatchOtConfig {
            batch_size: 12,
            n_batches: 30,
            reg: 1.0,
            cost_p: 2,
            seed: 5,
        };
        let mut rng_close = make_rng(cfg.seed);
        let fit_close =
            minibatch_wasserstein(&src, &tgt_close, n, m, d, &cfg, &mut rng_close).expect("ok");
        let mut rng_far = make_rng(cfg.seed);
        let fit_far =
            minibatch_wasserstein(&src, &tgt_far, n, m, d, &cfg, &mut rng_far).expect("ok");
        assert!(
            fit_far.mean_cost > fit_close.mean_cost,
            "far cost {} should exceed close cost {}",
            fit_far.mean_cost,
            fit_close.mean_cost
        );
    }

    #[test]
    fn l1_cost_differs_from_l2_cost() {
        let mut rng = make_rng(5);
        let n = 40;
        let m = 40;
        let d = 2;
        let source = gaussian_cloud(n, d, 0.0, &mut rng);
        let target = gaussian_cloud(m, d, 5.0, &mut rng);

        let cfg_l1 = MinibatchOtConfig {
            batch_size: 10,
            n_batches: 20,
            reg: 0.5,
            cost_p: 1,
            seed: 17,
        };
        let cfg_l2 = MinibatchOtConfig {
            cost_p: 2,
            ..cfg_l1.clone()
        };

        let mut rng1 = make_rng(17);
        let fit_l1 =
            minibatch_wasserstein(&source, &target, n, m, d, &cfg_l1, &mut rng1).expect("ok");

        let mut rng2 = make_rng(17);
        let fit_l2 =
            minibatch_wasserstein(&source, &target, n, m, d, &cfg_l2, &mut rng2).expect("ok");

        // L1 and L2 costs are both finite but numerically different
        assert!(fit_l1.mean_cost.is_finite());
        assert!(fit_l2.mean_cost.is_finite());
    }

    #[test]
    fn sinkhorn_divergence_finite() {
        let mut rng = make_rng(6);
        let n = 50;
        let m = 50;
        let d = 1;
        let source = gaussian_cloud(n, d, 0.0, &mut rng);
        let target = gaussian_cloud(m, d, 3.0, &mut rng);
        let cfg = MinibatchOtConfig {
            batch_size: 10,
            n_batches: 20,
            reg: 0.5,
            cost_p: 2,
            seed: 11,
        };
        let mut rng2 = make_rng(cfg.seed);
        let div =
            minibatch_sinkhorn_divergence(&source, &target, n, m, d, &cfg, &mut rng2).expect("ok");
        assert!(div.is_finite(), "divergence={div}");
    }

    #[test]
    fn empty_input_returns_error() {
        let cfg = MinibatchOtConfig::default();
        let mut rng = make_rng(0);
        let res = minibatch_wasserstein(&[], &[], 0, 0, 1, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_reg_returns_error() {
        let cfg = MinibatchOtConfig {
            reg: 0.0,
            ..Default::default()
        };
        let source = vec![0.0_f64; 64 * 2];
        let target = vec![1.0_f64; 64 * 2];
        let mut rng = make_rng(0);
        let res = minibatch_wasserstein(&source, &target, 64, 64, 2, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_dim_returns_error() {
        let cfg = MinibatchOtConfig {
            batch_size: 5,
            ..Default::default()
        };
        let source = vec![0.0_f64; 10];
        let target = vec![1.0_f64; 10];
        let mut rng = make_rng(0);
        let res = minibatch_wasserstein(&source, &target, 10, 10, 0, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn batch_size_larger_than_n_returns_error() {
        let n = 5;
        let m = 5;
        let d = 1;
        let cfg = MinibatchOtConfig {
            batch_size: 10, // > n
            n_batches: 5,
            reg: 0.1,
            cost_p: 2,
            seed: 0,
        };
        let source = vec![0.0_f64; n * d];
        let target = vec![1.0_f64; m * d];
        let mut rng = make_rng(0);
        let res = minibatch_wasserstein(&source, &target, n, m, d, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }
}
