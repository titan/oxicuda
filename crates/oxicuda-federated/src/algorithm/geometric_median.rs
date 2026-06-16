//! Geometric-median (RFA) robust aggregation for federated learning.
//!
//! Pillutla, Kakade & Harchaoui, "Robust Aggregation for Federated Learning",
//! IEEE Transactions on Signal Processing 2022 (a.k.a. *RFA*).
//!
//! The geometric median (a.k.a. the spatial / L1 median) of a set of client
//! updates `{w_i}` with weights `{α_i}` is the point `z` minimising the
//! weighted sum of Euclidean distances
//!
//! `z* = argmin_z  Σ_i α_i · ‖z − w_i‖₂`.
//!
//! Unlike the (coordinate-wise) mean, the geometric median has a breakdown
//! point of 1/2: it tolerates up to (almost) half of the clients being
//! arbitrarily corrupted, which makes it a strong Byzantine-robust aggregator.
//!
//! # Smoothed Weiszfeld iteration
//! The geometric median has no closed form for `n > 2`; RFA computes it with the
//! *smoothed Weiszfeld* algorithm. Starting from the weighted mean, repeat
//!
//! `β_i ← α_i / max(ν, ‖z − w_i‖₂)`,  `z ← (Σ_i β_i · w_i) / (Σ_i β_i)`
//!
//! where `ν > 0` is a small smoothing constant that keeps the update
//! well-defined when an iterate coincides with a client point. The objective is
//! convex, so the iteration converges to the (smoothed) geometric median.

use crate::error::{FedError, FedResult};

/// Configuration for the geometric-median aggregator.
#[derive(Debug, Clone)]
pub struct GeometricMedianConfig {
    /// Maximum number of smoothed-Weiszfeld iterations.
    pub max_iters: usize,
    /// Convergence tolerance on the relative movement of the iterate.
    pub tol: f64,
    /// Smoothing constant `ν > 0` (lower bound on distances in the reweighting).
    pub smoothing: f64,
}

impl GeometricMedianConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// Returns `Internal` if `max_iters == 0`, `tol < 0`, or `smoothing ≤ 0`.
    pub fn new(max_iters: usize, tol: f64, smoothing: f64) -> FedResult<Self> {
        if max_iters == 0 {
            return Err(FedError::Internal(
                "geometric_median: max_iters must be ≥ 1".into(),
            ));
        }
        if !(tol >= 0.0 && tol.is_finite()) {
            return Err(FedError::Internal(
                "geometric_median: tol must be finite and ≥ 0".into(),
            ));
        }
        if !(smoothing > 0.0 && smoothing.is_finite()) {
            return Err(FedError::Internal(
                "geometric_median: smoothing must be finite and > 0".into(),
            ));
        }
        Ok(Self {
            max_iters,
            tol,
            smoothing,
        })
    }
}

impl Default for GeometricMedianConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol: 1e-7,
            smoothing: 1e-6,
        }
    }
}

/// Result of a geometric-median aggregation.
#[derive(Debug, Clone)]
pub struct GeometricMedianResult {
    /// The aggregated (geometric-median) update.
    pub aggregated: Vec<f32>,
    /// Number of Weiszfeld iterations actually performed.
    pub iterations: usize,
    /// Weighted sum of distances `Σ α_i ‖z − w_i‖` at the returned point
    /// (the objective value; lower is better).
    pub objective: f64,
}

/// Weighted Euclidean distance between two equal-length vectors (in f64).
fn distance(a: &[f64], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Validate the `(grads, weights)` inputs and return the gradient dimension.
fn validate(grads: &[Vec<f32>], weights: &[f32]) -> FedResult<usize> {
    if grads.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    if weights.len() != grads.len() {
        return Err(FedError::DimensionMismatch {
            expected: grads.len(),
            got: weights.len(),
        });
    }
    let dim = grads[0].len();
    if dim == 0 {
        return Err(FedError::EmptyClientList);
    }
    for g in grads.iter().skip(1) {
        if g.len() != dim {
            return Err(FedError::DimensionMismatch {
                expected: dim,
                got: g.len(),
            });
        }
    }
    for &w in weights {
        if !(w >= 0.0 && w.is_finite()) {
            return Err(FedError::InvalidWeight { weight: w });
        }
    }
    if weights.iter().map(|&w| w as f64).sum::<f64>() <= 0.0 {
        return Err(FedError::InvalidWeight { weight: 0.0 });
    }
    Ok(dim)
}

/// Compute the weighted geometric median of `grads` via the smoothed Weiszfeld
/// algorithm (RFA, Pillutla et al. 2022).
///
/// `weights[i]` is the (non-negative) aggregation weight of client `i` — e.g.
/// its sample count. They need not sum to one; they are renormalised
/// internally.
///
/// # Errors
/// - `EmptyClientList` if `grads` is empty or gradients are zero-length.
/// - `DimensionMismatch` if `weights.len() != grads.len()` or gradients differ
///   in length.
/// - `InvalidWeight` if a weight is negative / non-finite or all weights are
///   zero.
pub fn geometric_median(
    grads: &[Vec<f32>],
    weights: &[f32],
    cfg: &GeometricMedianConfig,
) -> FedResult<GeometricMedianResult> {
    let dim = validate(grads, weights)?;
    let w64: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    let total_w: f64 = w64.iter().sum();

    // Initialise z at the weighted mean (RFA's recommended warm start).
    let mut z = vec![0.0_f64; dim];
    for (g, &wi) in grads.iter().zip(w64.iter()) {
        for (zj, &gj) in z.iter_mut().zip(g.iter()) {
            *zj += wi * gj as f64;
        }
    }
    for zj in z.iter_mut() {
        *zj /= total_w;
    }

    let mut iterations = 0usize;
    for _ in 0..cfg.max_iters {
        iterations += 1;

        // Reweight: β_i = α_i / max(ν, ‖z − w_i‖).
        let mut beta_sum = 0.0_f64;
        let mut numer = vec![0.0_f64; dim];
        for (g, &wi) in grads.iter().zip(w64.iter()) {
            let d = distance(&z, g).max(cfg.smoothing);
            let beta = wi / d;
            beta_sum += beta;
            for (nj, &gj) in numer.iter_mut().zip(g.iter()) {
                *nj += beta * gj as f64;
            }
        }

        if beta_sum <= 0.0 {
            // All weights zero after reweighting (degenerate); stop early.
            break;
        }

        // z_new = numer / beta_sum; track movement for the stopping rule.
        let mut shift_sq = 0.0_f64;
        let mut z_norm_sq = 0.0_f64;
        for (zj, &nj) in z.iter_mut().zip(numer.iter()) {
            let new = nj / beta_sum;
            let d = new - *zj;
            shift_sq += d * d;
            z_norm_sq += new * new;
            *zj = new;
        }

        let denom = z_norm_sq.sqrt().max(1e-12);
        if shift_sq.sqrt() / denom <= cfg.tol {
            break;
        }
    }

    // Objective Σ α_i ‖z − w_i‖ at the final iterate.
    let objective: f64 = grads
        .iter()
        .zip(w64.iter())
        .map(|(g, &wi)| wi * distance(&z, g))
        .sum();

    let aggregated: Vec<f32> = z.iter().map(|&v| v as f32).collect();
    Ok(GeometricMedianResult {
        aggregated,
        iterations,
        objective,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    #[test]
    fn config_validation() {
        assert!(GeometricMedianConfig::new(0, 1e-6, 1e-6).is_err());
        assert!(GeometricMedianConfig::new(10, -1.0, 1e-6).is_err());
        assert!(GeometricMedianConfig::new(10, 1e-6, 0.0).is_err());
        assert!(GeometricMedianConfig::new(10, 1e-6, -1.0).is_err());
        assert!(GeometricMedianConfig::new(10, 1e-6, 1e-6).is_ok());
    }

    #[test]
    fn empty_client_list_errors() {
        let cfg = GeometricMedianConfig::default();
        assert!(matches!(
            geometric_median(&[], &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn dimension_mismatch_weights() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let cfg = GeometricMedianConfig::default();
        assert!(matches!(
            geometric_median(&grads, &[1.0], &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_gradients() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32]];
        let cfg = GeometricMedianConfig::default();
        assert!(matches!(
            geometric_median(&grads, &[1.0, 1.0], &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_weight_errors() {
        let grads = vec![vec![1.0_f32], vec![2.0_f32]];
        let cfg = GeometricMedianConfig::default();
        assert!(matches!(
            geometric_median(&grads, &[1.0, -1.0], &cfg),
            Err(FedError::InvalidWeight { .. })
        ));
        assert!(matches!(
            geometric_median(&grads, &[0.0, 0.0], &cfg),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn single_client_returns_itself() {
        let grads = vec![vec![1.0_f32, -2.0, 3.5]];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &[1.0], &cfg).expect("ok");
        for (a, &b) in res.aggregated.iter().zip(grads[0].iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn output_shape_matches_dimension() {
        let grads = vec![vec![0.0_f32; 7], vec![1.0_f32; 7], vec![2.0_f32; 7]];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &uniform(3), &cfg).expect("ok");
        assert_eq!(res.aggregated.len(), 7);
    }

    #[test]
    fn symmetric_points_yield_centroid() {
        // Four symmetric points about the origin → geometric median ≈ origin.
        let grads = vec![
            vec![1.0_f32, 0.0],
            vec![-1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![0.0_f32, -1.0],
        ];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &uniform(4), &cfg).expect("ok");
        assert!(res.aggregated[0].abs() < 1e-3, "x={}", res.aggregated[0]);
        assert!(res.aggregated[1].abs() < 1e-3, "y={}", res.aggregated[1]);
    }

    #[test]
    fn collinear_median_is_middle_point() {
        // Points 0, 1, 100 on a line → geometric median is the middle one (1),
        // since 1 minimises Σ|z − x_i| for an odd number of collinear points.
        let grads = vec![vec![0.0_f32], vec![1.0_f32], vec![100.0_f32]];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &uniform(3), &cfg).expect("ok");
        assert!(
            (res.aggregated[0] - 1.0).abs() < 1e-2,
            "median = {}",
            res.aggregated[0]
        );
    }

    #[test]
    fn byzantine_outlier_rejected() {
        // Nine honest clients near (0,0), one massive Byzantine outlier.
        let mut grads = Vec::new();
        for i in 0..9 {
            let off = (i as f32 - 4.0) * 0.01;
            grads.push(vec![off, off]);
        }
        grads.push(vec![1_000.0_f32, -1_000.0]);
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &uniform(10), &cfg).expect("ok");
        // The geometric median should stay close to the honest cluster, far from
        // the outlier (whereas the mean would be pulled to ≈ ±100).
        assert!(res.aggregated[0].abs() < 1.0, "x={}", res.aggregated[0]);
        assert!(res.aggregated[1].abs() < 1.0, "y={}", res.aggregated[1]);
    }

    #[test]
    fn weighted_pulls_toward_heavier_client() {
        // Two points at 0 and 10; weight 9:1 toward 0 → median near 0.
        let grads = vec![vec![0.0_f32], vec![10.0_f32]];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &[9.0, 1.0], &cfg).expect("ok");
        // For two points the geometric median is the heavier endpoint when one
        // weight dominates (objective is piecewise-linear, minimised at 0).
        assert!(res.aggregated[0] < 1.0, "z={}", res.aggregated[0]);
    }

    #[test]
    fn deterministic_repeatable() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 1.0], vec![2.0_f32, 5.0]];
        let cfg = GeometricMedianConfig::default();
        let a = geometric_median(&grads, &uniform(3), &cfg).expect("ok");
        let b = geometric_median(&grads, &uniform(3), &cfg).expect("ok");
        assert_eq!(a.aggregated, b.aggregated);
        assert_eq!(a.iterations, b.iterations);
    }

    #[test]
    fn objective_is_minimised_vs_mean() {
        // The geometric-median objective at z* must be ≤ the objective at the mean.
        let grads = vec![
            vec![0.0_f32, 0.0],
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![5.0_f32, 5.0],
        ];
        let cfg = GeometricMedianConfig::default();
        let res = geometric_median(&grads, &uniform(4), &cfg).expect("ok");

        // Mean objective.
        let dim = 2;
        let mut mean = vec![0.0_f64; dim];
        for g in &grads {
            for (m, &gj) in mean.iter_mut().zip(g.iter()) {
                *m += gj as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= grads.len() as f64;
        }
        let mean_obj: f64 = grads.iter().map(|g| distance(&mean, g)).sum();
        assert!(
            res.objective <= mean_obj + 1e-6,
            "gm obj {} should be ≤ mean obj {}",
            res.objective,
            mean_obj
        );
    }

    #[test]
    fn converges_within_iteration_budget() {
        let grads = vec![vec![1.0_f32, 1.0], vec![2.0_f32, 2.0], vec![3.0_f32, 3.0]];
        let cfg = GeometricMedianConfig::new(100, 1e-8, 1e-7).expect("ok");
        let res = geometric_median(&grads, &uniform(3), &cfg).expect("ok");
        assert!(res.iterations <= 100);
        assert!(res.iterations >= 1);
    }
}
