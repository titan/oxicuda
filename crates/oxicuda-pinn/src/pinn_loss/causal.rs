//! Causal PINN training (Wang et al. 2022, ICLR 2022).
//!
//! Standard PINNs treat all collocation points equally, which violates physical causality
//! in time-dependent PDEs. Causal PINN assigns exponentially-decaying weights to later
//! collocation points based on accumulated early residuals: points sorted by time t_1 ≤ t_2 ≤ …
//! receive weight `w_i = exp(-ε · Σ_{j<i} r_j²)`. As training progresses, early residuals
//! decrease → causality weights increase for later points → full temporal coverage.
//!
//! Reference: Wang, S., Sankaran, S., & Perdikaris, P. (2022).
//! "Respecting Causality for Training Physics-Informed Neural Networks." ICLR 2022.

use crate::error::{PinnError, PinnResult};

/// Configuration for causal PINN training.
#[derive(Debug, Clone)]
pub struct CausalPinnConfig {
    /// Causality parameter ε (epsilon). Controls how strongly earlier residuals suppress later
    /// weights. Larger ε → stricter causality (later points get near-zero weight until early
    /// ones are small). Typical range: 1.0 to 1000.0. Default: 1.0.
    pub epsilon: f32,
    /// Convergence tolerance: considered converged when `min(weights) > 1 - tol`.
    pub convergence_tol: f32,
}

impl Default for CausalPinnConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            convergence_tol: 0.01,
        }
    }
}

/// Causal PINN loss computer.
///
/// Residuals must be provided in temporal order (sorted by ascending time). Each residual
/// `r_i` corresponds to a collocation point at time `t_i` with `t_1 ≤ t_2 ≤ … ≤ t_N`.
///
/// # Algorithm
/// ```text
/// w_0 = 1
/// w_i = exp(-ε · Σ_{j=0}^{i-1} r_j²)    for i > 0
/// L_causal = (1/N) Σ_i w_i · r_i²
/// ```
#[derive(Debug, Clone)]
pub struct CausalPinnLoss {
    pub config: CausalPinnConfig,
}

impl CausalPinnLoss {
    /// Create a new causal PINN loss with the given configuration.
    ///
    /// # Errors
    /// - `InvalidWeight { weight: epsilon }` if `epsilon ≤ 0` or not finite.
    /// - `InvalidWeight { weight: convergence_tol }` if `convergence_tol` not in `(0, 1)`.
    pub fn new(config: CausalPinnConfig) -> PinnResult<Self> {
        if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.epsilon,
            });
        }
        if !config.convergence_tol.is_finite()
            || config.convergence_tol <= 0.0
            || config.convergence_tol >= 1.0
        {
            return Err(PinnError::InvalidWeight {
                weight: config.convergence_tol,
            });
        }
        Ok(Self { config })
    }

    /// Compute causality weights for residuals provided in temporal order.
    ///
    /// ```text
    /// w_0 = 1.0
    /// w_i = exp(-ε · Σ_{j<i} r_j²)    for i > 0
    /// ```
    ///
    /// Returns a `Vec<f32>` of the same length as `residuals`.
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `residuals` is empty.
    /// - `NanEncountered` if any computed weight is not finite.
    pub fn causality_weights(&self, residuals: &[f32]) -> PinnResult<Vec<f32>> {
        if residuals.is_empty() {
            return Err(PinnError::EmptyCollocationSet);
        }
        let mut weights = Vec::with_capacity(residuals.len());
        let mut cumsum = 0.0_f32;
        weights.push(1.0_f32); // w_0 = 1
        for i in 1..residuals.len() {
            cumsum += residuals[i - 1] * residuals[i - 1]; // Σ_{j<i} r_j²
            let w = (-self.config.epsilon * cumsum).exp();
            if !w.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "causality_weights",
                });
            }
            weights.push(w);
        }
        Ok(weights)
    }

    /// Compute causally-weighted PDE residual loss.
    ///
    /// ```text
    /// L = (1/N) Σ_i w_i · r_i²
    /// ```
    /// where `w_i` are the causality weights.
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `residuals` is empty.
    /// - `NanEncountered` if the resulting loss is not finite.
    pub fn weighted_loss(&self, residuals: &[f32]) -> PinnResult<f32> {
        let weights = self.causality_weights(residuals)?;
        let loss = weights
            .iter()
            .zip(residuals.iter())
            .map(|(&w, &r)| w * r * r)
            .sum::<f32>()
            / residuals.len() as f32;
        if !loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "causal_weighted_loss",
            });
        }
        Ok(loss)
    }

    /// Check convergence criterion: `min(weights) > 1 - convergence_tol`.
    ///
    /// When the minimum causality weight (the weight on the last collocation point, since
    /// weights are monotone non-increasing) is close to 1, the causality enforcement has
    /// relaxed sufficiently to indicate approximate temporal convergence.
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `residuals` is empty.
    /// - `NanEncountered` if weights cannot be computed.
    pub fn is_converged(&self, residuals: &[f32]) -> PinnResult<bool> {
        let weights = self.causality_weights(residuals)?;
        let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        Ok(min_w > 1.0 - self.config.convergence_tol)
    }

    /// Compute weighted loss up to (inclusive) time index `k`.
    ///
    /// Useful for analysing how much of the domain has been "activated" by causality.
    /// The sub-slice `residuals[0..=k]` is treated as a self-contained temporal segment.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `k >= residuals.len()`.
    /// - `EmptyCollocationSet` if `residuals` is empty.
    /// - `NanEncountered` if the loss is not finite.
    pub fn partial_loss(&self, residuals: &[f32], k: usize) -> PinnResult<f32> {
        if residuals.is_empty() {
            return Err(PinnError::EmptyCollocationSet);
        }
        if k >= residuals.len() {
            return Err(PinnError::DimensionMismatch {
                expected: residuals.len() - 1,
                got: k,
            });
        }
        let sub = &residuals[..=k];
        let weights = self.causality_weights(sub)?;
        let loss = weights
            .iter()
            .zip(sub.iter())
            .map(|(&w, &r)| w * r * r)
            .sum::<f32>()
            / (k + 1) as f32;
        if !loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "partial_loss",
            });
        }
        Ok(loss)
    }

    /// Compute the effective fraction of the domain with non-negligible causality weight.
    ///
    /// Returns the fraction of collocation points whose causality weight exceeds
    /// `weight_threshold`.
    ///
    /// # Errors
    /// - `InvalidWeight` if `weight_threshold` is not in `[0, 1]`.
    /// - `EmptyCollocationSet` if `residuals` is empty.
    pub fn effective_coverage(&self, residuals: &[f32], weight_threshold: f32) -> PinnResult<f32> {
        if !weight_threshold.is_finite() || !(0.0..=1.0).contains(&weight_threshold) {
            return Err(PinnError::InvalidWeight {
                weight: weight_threshold,
            });
        }
        let weights = self.causality_weights(residuals)?;
        let n_active = weights.iter().filter(|&&w| w > weight_threshold).count();
        Ok(n_active as f32 / residuals.len() as f32)
    }

    /// Compute the cumulative sum of squared residuals up to each point.
    ///
    /// ```text
    /// cum_r2[i] = Σ_{j=0}^{i} r_j²
    /// ```
    ///
    /// The causality weight for point `i+1` is `exp(-ε · cum_r2[i])`.
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `residuals` is empty.
    pub fn cumulative_squared_residuals(&self, residuals: &[f32]) -> PinnResult<Vec<f32>> {
        if residuals.is_empty() {
            return Err(PinnError::EmptyCollocationSet);
        }
        let mut cum = Vec::with_capacity(residuals.len());
        let mut acc = 0.0_f32;
        for &r in residuals {
            acc += r * r;
            cum.push(acc);
        }
        Ok(cum)
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_loss() -> CausalPinnLoss {
        CausalPinnLoss::new(CausalPinnConfig::default()).unwrap()
    }

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn causal_config_err_negative_epsilon() {
        let cfg = CausalPinnConfig {
            epsilon: -1.0,
            convergence_tol: 0.01,
        };
        assert!(matches!(
            CausalPinnLoss::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn causal_config_err_zero_epsilon() {
        let cfg = CausalPinnConfig {
            epsilon: 0.0,
            convergence_tol: 0.01,
        };
        assert!(matches!(
            CausalPinnLoss::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn causal_config_err_nan_epsilon() {
        let cfg = CausalPinnConfig {
            epsilon: f32::NAN,
            convergence_tol: 0.01,
        };
        assert!(matches!(
            CausalPinnLoss::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn causal_config_err_bad_convergence_tol_zero() {
        let cfg = CausalPinnConfig {
            epsilon: 1.0,
            convergence_tol: 0.0,
        };
        assert!(matches!(
            CausalPinnLoss::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn causal_config_err_bad_convergence_tol_one() {
        let cfg = CausalPinnConfig {
            epsilon: 1.0,
            convergence_tol: 1.0,
        };
        assert!(matches!(
            CausalPinnLoss::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    // ── causality_weights ─────────────────────────────────────────────────────

    #[test]
    fn causal_weights_first_is_one() {
        let cl = default_loss();
        let r = vec![0.5_f32, 1.0, 2.0];
        let w = cl.causality_weights(&r).unwrap();
        assert!(
            (w[0] - 1.0_f32).abs() < 1e-7,
            "w[0] must be exactly 1.0, got {}",
            w[0]
        );
    }

    #[test]
    fn causal_weights_zero_residuals() {
        let cl = default_loss();
        let r = vec![0.0_f32; 8];
        let w = cl.causality_weights(&r).unwrap();
        for (i, &wi) in w.iter().enumerate() {
            assert!(
                (wi - 1.0_f32).abs() < 1e-7,
                "All-zero residuals → all weights = 1, w[{i}] = {wi}"
            );
        }
    }

    #[test]
    fn causal_weights_large_residuals() {
        // With epsilon=1 and residuals=10, w_1 = exp(-100) ≈ 0
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 1.0,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![10.0_f32, 10.0, 10.0];
        let w = cl.causality_weights(&r).unwrap();
        assert!(w[1] < 1e-10, "w[1] should be near zero, got {}", w[1]);
        assert!(w[2] < 1e-10, "w[2] should be near zero, got {}", w[2]);
    }

    #[test]
    fn causal_weights_monotone_decreasing() {
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 2.0,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![0.5_f32, 0.8, 1.2, 0.3, 0.6];
        let w = cl.causality_weights(&r).unwrap();
        for i in 1..w.len() {
            assert!(
                w[i] <= w[i - 1],
                "Weights must be monotone non-increasing: w[{}]={} > w[{}]={}",
                i,
                w[i],
                i - 1,
                w[i - 1]
            );
        }
    }

    #[test]
    fn causal_weights_length_equals_residuals() {
        let cl = default_loss();
        let r: Vec<f32> = (0..13).map(|i| i as f32 * 0.1).collect();
        let w = cl.causality_weights(&r).unwrap();
        assert_eq!(w.len(), r.len(), "Output length must match input length");
    }

    #[test]
    fn causal_weights_empty_returns_err() {
        let cl = default_loss();
        assert!(matches!(
            cl.causality_weights(&[]),
            Err(PinnError::EmptyCollocationSet)
        ));
    }

    // ── weighted_loss ─────────────────────────────────────────────────────────

    #[test]
    fn causal_weighted_loss_zero_residuals() {
        let cl = default_loss();
        let r = vec![0.0_f32; 10];
        let loss = cl.weighted_loss(&r).unwrap();
        assert!(loss.abs() < 1e-8, "Zero residuals → loss = 0, got {loss}");
    }

    #[test]
    fn causal_weighted_loss_equals_standard_when_eps_tiny() {
        // With very small ε, weights ≈ 1, so causal loss ≈ standard MSE
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 1e-6,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![0.5_f32, 1.0, -0.3, 0.8];
        let causal_loss = cl.weighted_loss(&r).unwrap();
        let mse: f32 = r.iter().map(|&x| x * x).sum::<f32>() / r.len() as f32;
        assert!(
            (causal_loss - mse).abs() < 1e-4,
            "Tiny epsilon: causal={causal_loss} vs mse={mse}"
        );
    }

    #[test]
    fn causal_weighted_loss_always_nonneg() {
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 5.0,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![-2.0_f32, 3.0, -1.5, 0.7, -0.1];
        let loss = cl.weighted_loss(&r).unwrap();
        assert!(loss >= 0.0, "Loss must be non-negative, got {loss}");
    }

    #[test]
    fn causal_weighted_loss_decreases_with_better_early() {
        // Better early residuals (smaller r[0]) → later weights increase, but early loss is lower.
        // The total causal loss should be strictly less with near-zero early residuals.
        let cl = default_loss();
        let r_bad_early = vec![10.0_f32, 1.0, 1.0, 1.0];
        let r_good_early = vec![0.01_f32, 1.0, 1.0, 1.0];
        let loss_bad = cl.weighted_loss(&r_bad_early).unwrap();
        let loss_good = cl.weighted_loss(&r_good_early).unwrap();
        assert!(
            loss_good < loss_bad,
            "Good early residuals should reduce causal loss: {loss_good} < {loss_bad}"
        );
    }

    #[test]
    fn causal_loss_single_point() {
        let cl = default_loss();
        let r = vec![3.0_f32];
        // Single point: w[0] = 1, loss = r[0]² / 1 = 9
        let loss = cl.weighted_loss(&r).unwrap();
        assert!(
            (loss - 9.0_f32).abs() < 1e-6,
            "Single-point loss = r², got {loss}"
        );
    }

    // ── is_converged ──────────────────────────────────────────────────────────

    #[test]
    fn is_converged_true_when_residuals_small() {
        // All-zero residuals → all weights = 1 → min weight = 1 > 1 - tol
        let cl = default_loss();
        let r = vec![0.0_f32; 20];
        assert!(cl.is_converged(&r).unwrap(), "Should be converged");
    }

    #[test]
    fn is_converged_false_when_residuals_large() {
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 10.0,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![5.0_f32, 5.0, 5.0];
        assert!(!cl.is_converged(&r).unwrap(), "Should not be converged");
    }

    // ── partial_loss ──────────────────────────────────────────────────────────

    #[test]
    fn partial_loss_first_point() {
        let cl = default_loss();
        let r = vec![3.0_f32, 10.0, 10.0]; // only index 0 matters
        // partial_loss(r, 0) uses sub=[r[0]], w[0]=1, loss = r[0]²/1 = 9
        let loss = cl.partial_loss(&r, 0).unwrap();
        assert!(
            (loss - 9.0_f32).abs() < 1e-6,
            "partial_loss with k=0 = r[0]², got {loss}"
        );
    }

    #[test]
    fn partial_loss_out_of_bounds() {
        let cl = default_loss();
        let r = vec![1.0_f32, 2.0, 3.0];
        // k=3 is out of bounds (len=3, valid range 0..=2)
        assert!(matches!(
            cl.partial_loss(&r, 3),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn partial_loss_nonneg() {
        let cl = default_loss();
        let r = vec![-3.0_f32, 1.5, -2.0, 0.5];
        for k in 0..r.len() {
            let loss = cl.partial_loss(&r, k).unwrap();
            assert!(
                loss >= 0.0,
                "partial_loss(k={k}) must be non-negative, got {loss}"
            );
        }
    }

    // ── effective_coverage ────────────────────────────────────────────────────

    #[test]
    fn effective_coverage_all_zero_residuals() {
        let cl = default_loss();
        let r = vec![0.0_f32; 12];
        // All weights = 1 > any threshold in [0, 1)
        let cov = cl.effective_coverage(&r, 0.5).unwrap();
        assert!(
            (cov - 1.0_f32).abs() < 1e-7,
            "All-zero residuals → coverage = 1.0, got {cov}"
        );
    }

    #[test]
    fn effective_coverage_large_residuals() {
        let cl = CausalPinnLoss::new(CausalPinnConfig {
            epsilon: 100.0,
            convergence_tol: 0.01,
        })
        .unwrap();
        let r = vec![5.0_f32, 5.0, 5.0, 5.0, 5.0];
        // Later weights ≈ 0; only first point has weight = 1
        let cov = cl.effective_coverage(&r, 0.5).unwrap();
        assert!(
            cov < 1.0,
            "Large residuals with high epsilon → partial coverage, got {cov}"
        );
    }

    #[test]
    fn effective_coverage_invalid_threshold() {
        let cl = default_loss();
        let r = vec![0.5_f32; 5];
        assert!(matches!(
            cl.effective_coverage(&r, 1.5),
            Err(PinnError::InvalidWeight { .. })
        ));
        assert!(matches!(
            cl.effective_coverage(&r, -0.1),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    // ── cumulative_squared_residuals ──────────────────────────────────────────

    #[test]
    fn cumulative_squared_residuals_shape() {
        let cl = default_loss();
        let r = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cum = cl.cumulative_squared_residuals(&r).unwrap();
        assert_eq!(cum.len(), r.len(), "Output length must match input");
    }

    #[test]
    fn cumulative_squared_residuals_nondecreasing() {
        let cl = default_loss();
        let r = vec![0.1_f32, 0.5, 0.0, 2.0, 0.3];
        let cum = cl.cumulative_squared_residuals(&r).unwrap();
        for i in 1..cum.len() {
            assert!(
                cum[i] >= cum[i - 1],
                "cum_r2 must be non-decreasing: cum[{}]={} < cum[{}]={}",
                i,
                cum[i],
                i - 1,
                cum[i - 1]
            );
        }
    }

    #[test]
    fn cumulative_squared_residuals_values() {
        let cl = default_loss();
        let r = vec![2.0_f32, 3.0]; // r² = [4, 9]
        let cum = cl.cumulative_squared_residuals(&r).unwrap();
        assert!(
            (cum[0] - 4.0_f32).abs() < 1e-6,
            "cum[0] = r[0]²=4, got {}",
            cum[0]
        );
        assert!(
            (cum[1] - 13.0_f32).abs() < 1e-6,
            "cum[1] = r[0]²+r[1]²=13, got {}",
            cum[1]
        );
    }

    #[test]
    fn cumulative_squared_residuals_empty_err() {
        let cl = default_loss();
        assert!(matches!(
            cl.cumulative_squared_residuals(&[]),
            Err(PinnError::EmptyCollocationSet)
        ));
    }
}
