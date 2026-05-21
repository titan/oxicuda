//! Fisher-Weighted Averaging for model merging.
//!
//! Reference: Matena MS, Raffel C (2022) "Merging Models with Fisher-Weighted
//! Averaging", NeurIPS.
//! <https://arxiv.org/abs/2111.09832>
//!
//! Matena & Raffel approximate the Hessian of the per-task log-likelihood with
//! its (diagonal) **Fisher information matrix** and use those per-coordinate
//! curvatures to weight the parameter contributions of every ingredient when
//! merging. Intuitively, coordinates that are *sharp* (large Fisher) in
//! task `i` pull the merged parameter toward `θᵢ`, while *flat* coordinates
//! (small Fisher) defer to the other models.
//!
//! For each parameter index `j`:
//!
//! ```text
//! θ̄_j  =  Σᵢ F̂ᵢⱼ · θᵢⱼ
//!          ─────────────────
//!          Σᵢ F̂ᵢⱼ + ε
//! ```
//!
//! The small `ε > 0` is a ridge guard against degenerate columns (e.g. all
//! Fishers zero for that coordinate); when `ε = 0` and every `F̂ᵢⱼ = 0`, the
//! merge is mathematically undefined. Following the original implementation
//! we recommend `ε ≈ 1e-8`.
//!
//! The diagonal Fisher estimator is the empirical-Fisher form
//! `F̂_j = (1/N) Σ_n (∂ℓ_n/∂θ_j)²` (Matena & Raffel 2022, §3): the caller
//! provides the pre-summed squared gradient and sample count.

use crate::error::{PeftError, PeftResult};

/// Diagonal Fisher information estimate for one model.
///
/// `diag.len()` must match the parameter dimensionality.
#[derive(Debug, Clone, PartialEq)]
pub struct FisherEstimate {
    /// Per-coordinate Fisher value (non-negative under empirical estimation).
    pub diag: Vec<f32>,
}

impl FisherEstimate {
    /// Dimensionality of the underlying parameter vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.diag.len()
    }

    /// Whether the estimate is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diag.is_empty()
    }
}

/// Configuration for the Fisher merge.
#[derive(Debug, Clone, Copy)]
pub struct FisherConfig {
    /// Ridge guard added to the denominator to prevent division by zero when
    /// every model assigns the coordinate zero Fisher.
    pub eps: f32,
}

impl Default for FisherConfig {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

/// Fisher-weighted merging namespace.
pub struct FisherMerging;

impl FisherMerging {
    /// Build a [`FisherEstimate`] from `Σ_n (∂ℓ_n/∂θ_j)²` and the sample count.
    ///
    /// This is the diagonal *empirical* Fisher of Matena & Raffel (2022, §3)
    /// — the loss-gradient outer product diagonal averaged over `n_samples`.
    ///
    /// # Errors
    /// * [`PeftError::EmptyInput`] when `grads_squared_sum` is empty.
    /// * [`PeftError::Internal`] when `n_samples == 0`.
    pub fn estimate_diagonal(
        grads_squared_sum: &[f32],
        n_samples: usize,
    ) -> PeftResult<FisherEstimate> {
        if grads_squared_sum.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        if n_samples == 0 {
            return Err(PeftError::Internal {
                msg: "Fisher estimator requires n_samples > 0".to_string(),
            });
        }
        let inv = 1.0_f32 / (n_samples as f32);
        let diag = grads_squared_sum.iter().map(|&g2| g2 * inv).collect();
        Ok(FisherEstimate { diag })
    }

    /// Merge `(θᵢ, F̂ᵢ)` pairs with Fisher-weighted averaging.
    ///
    /// All parameter slices must share the same length `p`, and each Fisher
    /// estimate must also have length `p`. `eps ≥ 0` is required.
    ///
    /// # Errors
    /// * [`PeftError::EmptyInput`] when `models` is empty or the first model
    ///   has zero parameters.
    /// * [`PeftError::DimensionMismatch`] when parameter or Fisher lengths
    ///   disagree.
    /// * [`PeftError::Internal`] when `cfg.eps` is negative.
    pub fn merge(models: &[(&[f32], &FisherEstimate)], cfg: &FisherConfig) -> PeftResult<Vec<f32>> {
        if models.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let n = models[0].0.len();
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        if cfg.eps.is_nan() || cfg.eps < 0.0 {
            return Err(PeftError::Internal {
                msg: format!("Fisher-merge eps must be non-negative, got {}", cfg.eps),
            });
        }
        for &(params, fisher) in models {
            if params.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: params.len(),
                });
            }
            if fisher.diag.len() != n {
                return Err(PeftError::DimensionMismatch {
                    expected: n,
                    got: fisher.diag.len(),
                });
            }
        }

        let mut numerator = vec![0.0_f32; n];
        let mut denominator = vec![cfg.eps; n];
        for &(params, fisher) in models {
            for ((num, den), (&f, &p)) in numerator
                .iter_mut()
                .zip(denominator.iter_mut())
                .zip(fisher.diag.iter().zip(params.iter()))
            {
                *num += f * p;
                *den += f;
            }
        }
        let merged: Vec<f32> = numerator
            .iter()
            .zip(denominator.iter())
            .map(|(num, den)| {
                // Guard against ε==0 + all-zero Fisher → 0/0 → fall back to 0.
                if *den == 0.0 { 0.0 } else { num / den }
            })
            .collect();
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn fisher_ones(n: usize) -> FisherEstimate {
        FisherEstimate {
            diag: vec![1.0_f32; n],
        }
    }

    #[test]
    fn estimate_diagonal_divides_by_n_samples() {
        let g2 = [4.0_f32, 9.0, 25.0];
        let est = FisherMerging::estimate_diagonal(&g2, 4).expect("estimate");
        let expected = [1.0_f32, 2.25, 6.25];
        assert!(approx_eq_slice(&est.diag, &expected, 1e-7));
    }

    #[test]
    fn estimate_diagonal_zero_samples_errors() {
        let g2 = [1.0_f32, 2.0, 3.0];
        let res = FisherMerging::estimate_diagonal(&g2, 0);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn estimate_diagonal_empty_errors() {
        let res = FisherMerging::estimate_diagonal(&[], 10);
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn merge_single_model_returns_itself_up_to_eps_perturbation() {
        let theta = [1.0_f32, -2.0, 3.5, 0.25];
        let fisher = FisherEstimate {
            diag: vec![2.0_f32, 2.0, 2.0, 2.0],
        };
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&theta[..], &fisher)], &cfg).expect("merge");
        assert!(approx_eq_slice(&merged, &theta, 1e-6));
    }

    #[test]
    fn merge_two_equal_fisher_returns_arithmetic_mean() {
        let a = [0.0_f32, 6.0, 12.0];
        let b = [6.0_f32, 6.0, 0.0];
        let fa = fisher_ones(3);
        let fb = fisher_ones(3);
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge");
        let expected = [3.0_f32, 6.0, 6.0];
        assert!(approx_eq_slice(&merged, &expected, 1e-6));
    }

    #[test]
    fn merge_empty_list_errors() {
        let res = FisherMerging::merge(&[], &FisherConfig::default());
        assert!(matches!(res, Err(PeftError::EmptyInput)));
    }

    #[test]
    fn merge_param_dimension_mismatch_errors() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0];
        let fa = fisher_ones(3);
        let fb = fisher_ones(2);
        let res = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &FisherConfig::default());
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn merge_fisher_param_length_mismatch_errors() {
        let a = [1.0_f32, 2.0, 3.0];
        let fa = FisherEstimate {
            diag: vec![1.0_f32, 1.0],
        };
        let res = FisherMerging::merge(&[(&a[..], &fa)], &FisherConfig::default());
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
    }

    #[test]
    fn merge_zero_fisher_falls_back_to_eps_regularised_mean() {
        // With all-zero Fisher per coordinate, numerator is 0 and denominator
        // is ε, so the merged value is 0 / ε = 0.
        let a = [3.0_f32, 4.0];
        let b = [5.0_f32, 6.0];
        let zero = FisherEstimate {
            diag: vec![0.0_f32, 0.0],
        };
        let cfg = FisherConfig { eps: 1e-3 };
        let merged =
            FisherMerging::merge(&[(&a[..], &zero), (&b[..], &zero)], &cfg).expect("merge");
        for &v in &merged {
            assert!(approx_eq(v, 0.0, 1e-6));
        }
    }

    #[test]
    fn merge_zero_fisher_zero_eps_returns_zero() {
        // ε=0 and all-zero Fisher → 0/0 guard returns 0.
        let a = [1.0_f32, 2.0];
        let zero = FisherEstimate {
            diag: vec![0.0_f32, 0.0],
        };
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&a[..], &zero)], &cfg).expect("merge");
        for &v in &merged {
            assert!(approx_eq(v, 0.0, 1e-7));
        }
    }

    #[test]
    fn merge_dominant_fisher_pulls_toward_that_model() {
        // a has tiny Fisher, b has huge Fisher → merged ≈ b.
        let a = [10.0_f32, 10.0];
        let b = [1.0_f32, 2.0];
        let fa = FisherEstimate {
            diag: vec![1e-4_f32, 1e-4],
        };
        let fb = FisherEstimate {
            diag: vec![1e4_f32, 1e4],
        };
        let cfg = FisherConfig { eps: 1e-8 };
        let merged = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge");
        assert!(
            approx_eq(merged[0], 1.0, 1e-2),
            "expected ≈1.0, got {}",
            merged[0]
        );
        assert!(
            approx_eq(merged[1], 2.0, 1e-2),
            "expected ≈2.0, got {}",
            merged[1]
        );
    }

    #[test]
    fn merge_per_coordinate_normalisation_correct() {
        // Two-coordinate case where we can hand-verify the formula.
        let a = [1.0_f32, 5.0];
        let b = [3.0_f32, 5.0];
        let fa = FisherEstimate {
            diag: vec![2.0_f32, 0.0],
        };
        let fb = FisherEstimate {
            diag: vec![1.0_f32, 4.0],
        };
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge");
        // j=0: (2*1 + 1*3)/(2+1) = 5/3
        // j=1: (0*5 + 4*5)/(0+4) = 5
        assert!(approx_eq(merged[0], 5.0 / 3.0, 1e-6));
        assert!(approx_eq(merged[1], 5.0, 1e-6));
    }

    #[test]
    fn merge_is_deterministic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let fa = FisherEstimate {
            diag: vec![1.0_f32, 0.5, 2.0],
        };
        let fb = FisherEstimate {
            diag: vec![0.5_f32, 2.0, 1.0],
        };
        let cfg = FisherConfig { eps: 1e-6 };
        let first = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge1");
        let second = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge2");
        assert_eq!(first, second);
    }

    #[test]
    fn merge_idempotent_remerge_equals_first_merge() {
        // Re-merging the merged result with itself (Fisher=any-equal) should
        // return the same merged vector.
        let a = [1.0_f32, 2.0];
        let b = [3.0_f32, 5.0];
        let fa = FisherEstimate {
            diag: vec![1.0_f32, 2.0],
        };
        let fb = FisherEstimate {
            diag: vec![3.0_f32, 1.0],
        };
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge1");
        // Re-merge merged with itself, both having Fisher=ones.
        let f_self = fisher_ones(merged.len());
        let again = FisherMerging::merge(&[(&merged[..], &f_self), (&merged[..], &f_self)], &cfg)
            .expect("merge2");
        assert!(approx_eq_slice(&merged, &again, 1e-6));
    }

    #[test]
    fn merge_negative_eps_errors() {
        let a = [1.0_f32, 2.0];
        let fa = fisher_ones(2);
        let cfg = FisherConfig { eps: -1.0 };
        let res = FisherMerging::merge(&[(&a[..], &fa)], &cfg);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    #[test]
    fn merge_tiny_fisher_with_eps_guard_stays_finite() {
        let a = [100.0_f32, -200.0];
        let fa = FisherEstimate {
            diag: vec![1e-30_f32, 1e-30],
        };
        let cfg = FisherConfig { eps: 1e-8 };
        let merged = FisherMerging::merge(&[(&a[..], &fa)], &cfg).expect("merge");
        for v in &merged {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn merge_balanced_fisher_recovers_average() {
        // Both models have unit Fisher, ε=0 → arithmetic mean.
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let fa = fisher_ones(4);
        let fb = fisher_ones(4);
        let cfg = FisherConfig { eps: 0.0 };
        let merged = FisherMerging::merge(&[(&a[..], &fa), (&b[..], &fb)], &cfg).expect("merge");
        let expected = [3.0_f32, 4.0, 5.0, 6.0];
        assert!(approx_eq_slice(&merged, &expected, 1e-6));
    }

    #[test]
    fn fisher_estimate_helpers_len_and_is_empty() {
        let est = FisherEstimate {
            diag: vec![1.0_f32, 2.0],
        };
        assert_eq!(est.len(), 2);
        assert!(!est.is_empty());
        let empty = FisherEstimate { diag: Vec::new() };
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
