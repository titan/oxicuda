//! Centered Clipping (CClip) robust aggregation for federated learning.
//!
//! Karimireddy, He & Jaggi, "Learning from History for Byzantine Robust
//! Optimization", ICML 2021.
//!
//! Centered Clipping is a simple yet provably Byzantine-robust aggregator. Given
//! client updates `{w_i}` and a *center* `v` (typically the previous round's
//! aggregate / momentum), it iterates
//!
//! `v ← v + (1/n) Σ_i (w_i − v) · min(1, τ / ‖w_i − v‖₂)`,
//!
//! i.e. each client's deviation from the center is clipped to an L2 ball of
//! radius `τ` before averaging. A handful of clipping iterations `L` refine the
//! estimate. Clipping bounds the influence any single (possibly Byzantine)
//! client can exert to `τ`, giving order-optimal robustness while costing only
//! `O(n·d)` per iteration — far cheaper than Krum's `O(n²·d)`.
//!
//! Using last round's aggregate as the center couples robustness with *history*
//! (momentum), which the paper shows is essential to defeat time-coupled
//! attacks.

use crate::error::{FedError, FedResult};

/// Configuration for the Centered-Clipping aggregator.
#[derive(Debug, Clone)]
pub struct CenteredClippingConfig {
    /// Clipping radius `τ > 0`: client deviations are clipped to this L2 norm.
    pub clip_radius: f64,
    /// Number of refinement iterations `L ≥ 1` (1 already gives a valid update).
    pub n_iters: usize,
}

impl CenteredClippingConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// Returns `InvalidClipNorm` if `clip_radius ≤ 0` / non-finite, or `Internal`
    /// if `n_iters == 0`.
    pub fn new(clip_radius: f64, n_iters: usize) -> FedResult<Self> {
        if !(clip_radius > 0.0 && clip_radius.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        if n_iters == 0 {
            return Err(FedError::Internal(
                "centered_clipping: n_iters must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            clip_radius,
            n_iters,
        })
    }
}

impl Default for CenteredClippingConfig {
    fn default() -> Self {
        Self {
            clip_radius: 1.0,
            n_iters: 3,
        }
    }
}

/// Result of a Centered-Clipping aggregation.
#[derive(Debug, Clone)]
pub struct CenteredClippingResult {
    /// The aggregated update.
    pub aggregated: Vec<f32>,
    /// Number of clients whose deviation was actually clipped in the final
    /// iteration (a Byzantine-activity diagnostic).
    pub n_clipped: usize,
}

/// Validate inputs and return the gradient dimension.
fn validate(grads: &[Vec<f32>], center: &[f32]) -> FedResult<usize> {
    if grads.is_empty() {
        return Err(FedError::EmptyClientList);
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
    if center.len() != dim {
        return Err(FedError::DimensionMismatch {
            expected: dim,
            got: center.len(),
        });
    }
    Ok(dim)
}

/// Run Centered Clipping with an explicit center `v`.
///
/// `center` is the point around which deviations are clipped (e.g. the previous
/// round's aggregate or server momentum). The returned aggregate refines the
/// center using clipped client deviations.
///
/// # Errors
/// - `EmptyClientList` if `grads` is empty or zero-length.
/// - `DimensionMismatch` if gradients or `center` have inconsistent lengths.
pub fn centered_clipping(
    grads: &[Vec<f32>],
    center: &[f32],
    cfg: &CenteredClippingConfig,
) -> FedResult<CenteredClippingResult> {
    let dim = validate(grads, center)?;
    let n = grads.len() as f64;

    let mut v: Vec<f64> = center.iter().map(|&c| c as f64).collect();
    let mut n_clipped = 0usize;

    for _iter in 0..cfg.n_iters {
        n_clipped = 0;
        let mut accum = vec![0.0_f64; dim];

        for g in grads {
            // Deviation d = w_i − v and its L2 norm.
            let mut norm_sq = 0.0_f64;
            for (j, &gj) in g.iter().enumerate() {
                let d = gj as f64 - v[j];
                norm_sq += d * d;
            }
            let norm = norm_sq.sqrt();

            // Clip factor min(1, τ / ‖d‖). Guard the zero-deviation case.
            let factor = if norm > cfg.clip_radius {
                n_clipped += 1;
                cfg.clip_radius / norm
            } else {
                1.0
            };

            for (j, &gj) in g.iter().enumerate() {
                let d = gj as f64 - v[j];
                accum[j] += d * factor;
            }
        }

        // v ← v + (1/n) Σ clipped deviations.
        for (vj, &aj) in v.iter_mut().zip(accum.iter()) {
            *vj += aj / n;
        }
    }

    Ok(CenteredClippingResult {
        aggregated: v.iter().map(|&x| x as f32).collect(),
        n_clipped,
    })
}

/// Convenience wrapper that centers the clipping at the (coordinate-wise) mean
/// of the client updates — a reasonable history-free default for the first
/// round when no momentum is available yet.
///
/// # Errors
/// Same as [`centered_clipping`].
pub fn centered_clipping_at_mean(
    grads: &[Vec<f32>],
    cfg: &CenteredClippingConfig,
) -> FedResult<CenteredClippingResult> {
    let dim = validate(grads, &vec![0.0_f32; grads.first().map_or(0, |g| g.len())])?;
    let n = grads.len() as f64;
    let mut mean = vec![0.0_f32; dim];
    {
        let mut acc = vec![0.0_f64; dim];
        for g in grads {
            for (a, &gj) in acc.iter_mut().zip(g.iter()) {
                *a += gj as f64;
            }
        }
        for (m, &a) in mean.iter_mut().zip(acc.iter()) {
            *m = (a / n) as f32;
        }
    }
    centered_clipping(grads, &mean, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0_f32; n]
    }

    #[test]
    fn config_validation() {
        assert!(CenteredClippingConfig::new(0.0, 3).is_err());
        assert!(CenteredClippingConfig::new(-1.0, 3).is_err());
        assert!(CenteredClippingConfig::new(1.0, 0).is_err());
        assert!(CenteredClippingConfig::new(1.0, 3).is_ok());
    }

    #[test]
    fn empty_client_list_errors() {
        let cfg = CenteredClippingConfig::default();
        assert!(matches!(
            centered_clipping(&[], &[], &cfg),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn dimension_mismatch_gradients() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32]];
        let cfg = CenteredClippingConfig::default();
        assert!(matches!(
            centered_clipping(&grads, &zeros(2), &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_center() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let cfg = CenteredClippingConfig::default();
        assert!(matches!(
            centered_clipping(&grads, &zeros(3), &cfg),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn output_shape_matches_dimension() {
        let grads = vec![vec![0.5_f32; 6], vec![1.0_f32; 6]];
        let cfg = CenteredClippingConfig::default();
        let res = centered_clipping(&grads, &zeros(6), &cfg).expect("ok");
        assert_eq!(res.aggregated.len(), 6);
    }

    #[test]
    fn single_client_recovers_clipped_update() {
        // One client at norm 0.5 < τ=1: nothing clipped, aggregate = client.
        let grads = vec![vec![0.3_f32, 0.4]]; // ‖·‖ = 0.5
        let cfg = CenteredClippingConfig::new(1.0, 1).expect("ok");
        let res = centered_clipping(&grads, &zeros(2), &cfg).expect("ok");
        assert!((res.aggregated[0] - 0.3).abs() < 1e-5);
        assert!((res.aggregated[1] - 0.4).abs() < 1e-5);
        assert_eq!(res.n_clipped, 0);
    }

    #[test]
    fn averaging_correct_when_within_radius() {
        // Three small clients (all within τ) centered at 0 → result ≈ mean.
        let grads = vec![vec![0.1_f32], vec![0.2_f32], vec![0.3_f32]];
        let cfg = CenteredClippingConfig::new(10.0, 5).expect("ok");
        let res = centered_clipping(&grads, &zeros(1), &cfg).expect("ok");
        assert!(
            (res.aggregated[0] - 0.2).abs() < 1e-4,
            "{}",
            res.aggregated[0]
        );
        assert_eq!(res.n_clipped, 0);
    }

    #[test]
    fn large_deviation_is_clipped() {
        // Client far outside τ → counted as clipped.
        let grads = vec![vec![100.0_f32, 0.0]];
        let cfg = CenteredClippingConfig::new(1.0, 1).expect("ok");
        let res = centered_clipping(&grads, &zeros(2), &cfg).expect("ok");
        assert_eq!(res.n_clipped, 1);
        // After one iteration v = 0 + (1/1)·(d·τ/‖d‖) = unit·1 = (1, 0).
        assert!(
            (res.aggregated[0] - 1.0).abs() < 1e-4,
            "{}",
            res.aggregated[0]
        );
    }

    #[test]
    fn byzantine_resistance_outlier_bounded() {
        // Many honest clients near 0, one enormous Byzantine update.
        let mut grads: Vec<Vec<f32>> = (0..9).map(|_| vec![0.01_f32, 0.01]).collect();
        grads.push(vec![1_000.0_f32, 1_000.0]);
        let cfg = CenteredClippingConfig::new(0.5, 3).expect("ok");
        let res = centered_clipping(&grads, &zeros(2), &cfg).expect("ok");
        // Byzantine influence is bounded by τ/n per iteration, so the result
        // stays near the honest cluster, nowhere near ±100.
        assert!(res.aggregated[0].abs() < 1.0, "x={}", res.aggregated[0]);
        assert!(res.aggregated[1].abs() < 1.0, "y={}", res.aggregated[1]);
        assert!(res.n_clipped >= 1);
    }

    #[test]
    fn deterministic_repeatable() {
        let grads = vec![vec![1.0_f32, -1.0], vec![2.0_f32, 0.5]];
        let cfg = CenteredClippingConfig::default();
        let a = centered_clipping(&grads, &zeros(2), &cfg).expect("ok");
        let b = centered_clipping(&grads, &zeros(2), &cfg).expect("ok");
        assert_eq!(a.aggregated, b.aggregated);
        assert_eq!(a.n_clipped, b.n_clipped);
    }

    #[test]
    fn more_iterations_converge_toward_clipped_mean() {
        // With clients within τ the iteration must reach the mean exactly.
        let grads = vec![vec![1.0_f32], vec![3.0_f32]];
        let big_tau = CenteredClippingConfig::new(100.0, 10).expect("ok");
        let res = centered_clipping(&grads, &zeros(1), &big_tau).expect("ok");
        assert!(
            (res.aggregated[0] - 2.0).abs() < 1e-3,
            "{}",
            res.aggregated[0]
        );
    }

    #[test]
    fn center_at_mean_helper_matches_mean_when_unclipped() {
        let grads = vec![vec![2.0_f32, 4.0], vec![4.0_f32, 6.0], vec![6.0_f32, 8.0]];
        let cfg = CenteredClippingConfig::new(100.0, 3).expect("ok");
        let res = centered_clipping_at_mean(&grads, &cfg).expect("ok");
        assert!(
            (res.aggregated[0] - 4.0).abs() < 1e-3,
            "{}",
            res.aggregated[0]
        );
        assert!(
            (res.aggregated[1] - 6.0).abs() < 1e-3,
            "{}",
            res.aggregated[1]
        );
    }

    #[test]
    fn convergence_direction_moves_toward_honest_mean() {
        // Starting from a center offset from the honest mean, the aggregate
        // should move toward the honest cluster.
        let grads = vec![vec![5.0_f32], vec![5.0_f32], vec![5.0_f32]];
        let cfg = CenteredClippingConfig::new(10.0, 5).expect("ok");
        let res = centered_clipping(&grads, &[0.0_f32], &cfg).expect("ok");
        assert!(res.aggregated[0] > 4.0, "moved to {}", res.aggregated[0]);
    }
}
