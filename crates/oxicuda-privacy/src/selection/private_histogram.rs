//! PrivateHistogram with stability-based release.
//!
//! # Reference
//! - Vadhan (2017), "The Complexity of Differential Privacy",
//!   Foundations of DP Lecture 12, Theorem 12.4.
//!
//! # Algorithm
//! For a histogram with `k` bins:
//!
//! 1. Add independent Laplace noise `Lap(1/ε)` to each bin count.
//!    The noise scale `1/ε` corresponds to sensitivity-1 queries.
//!
//! 2. Compute the stability threshold
//!    `T = 1 + (2/ε) · ln(2 / (δ · k))`.
//!
//! 3. Release only bins whose noisy count exceeds `T`.
//!    Bins that do not clear the threshold are zeroed out in the output
//!    counts vector.
//!
//! 4. If no bins clear the threshold, return `Suppressed` instead of leaking
//!    that the histogram was empty.
//!
//! The `release_top_k` variant first selects the top-`top_k` bins by noisy
//! count, then applies the same threshold test to those bins only, limiting
//! data-independent output size.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the stability-based private histogram release.
#[derive(Debug, Clone)]
pub struct PrivateHistogramConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Privacy parameter δ ∈ (0, 1).
    pub delta: f64,
    /// Number of histogram bins `k ≥ 1`.
    pub k: usize,
    /// If > 0.0, use this as the stability threshold directly.
    /// If 0.0, compute `T = 1 + (2/ε) · ln(2 / (δ · k))` automatically.
    pub stability_threshold: f64,
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Output of a private histogram release.
#[derive(Debug)]
pub enum PrivateHistogramOutput {
    /// At least one bin cleared the stability threshold.
    Released {
        /// Noisy counts for all `k` bins; suppressed bins are set to 0.0.
        counts: Vec<f64>,
        /// Zero-based indices of bins whose noisy count exceeded the threshold.
        released_bins: Vec<usize>,
    },
    /// No bins cleared the threshold; the histogram is suppressed.
    Suppressed {
        /// Number of bins that were below the threshold (equals k).
        reason_bin_count: usize,
    },
}

// ─── PrivateHistogram ─────────────────────────────────────────────────────────

/// Stability-based private histogram release mechanism.
///
/// One instance corresponds to one configuration and can be reused for
/// multiple releases against different bin-count vectors.
pub struct PrivateHistogram {
    cfg: PrivateHistogramConfig,
}

impl PrivateHistogram {
    /// Validate and construct a `PrivateHistogram`.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `ε ≤ 0`.
    /// - `InvalidDelta` if `δ ≤ 0` or `δ ≥ 1`.
    /// - `InvalidParameter` if `k == 0` or `stability_threshold < 0`.
    pub fn new(cfg: PrivateHistogramConfig) -> PrivacyResult<Self> {
        if cfg.epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
        }
        if cfg.delta <= 0.0 || cfg.delta >= 1.0 {
            return Err(PrivacyError::InvalidDelta(cfg.delta));
        }
        if cfg.k == 0 {
            return Err(PrivacyError::InvalidParameter("k must be ≥ 1".into()));
        }
        if cfg.stability_threshold < 0.0 {
            return Err(PrivacyError::InvalidParameter(
                "stability_threshold must be ≥ 0".into(),
            ));
        }
        Ok(Self { cfg })
    }

    /// Compute the stability threshold `T`.
    ///
    /// Returns `cfg.stability_threshold` when it was set > 0 at construction.
    /// Otherwise computes `T = 1 + (2/ε) · ln(2 / (δ · k))`, clamped to ≥ 0.
    #[must_use]
    pub fn compute_threshold(&self) -> f64 {
        if self.cfg.stability_threshold > 0.0 {
            return self.cfg.stability_threshold;
        }
        let eps = self.cfg.epsilon;
        let delta = self.cfg.delta;
        let k = self.cfg.k as f64;
        let t = 1.0 + (2.0 / eps) * (2.0 / (delta * k)).ln();
        t.max(0.0)
    }

    /// Release a private histogram with stability-based suppression.
    ///
    /// Each bin receives independent Laplace noise `Lap(1/ε)`.  Only bins
    /// with noisy count > `T` are included in the release.  If no bins pass
    /// the test, returns `Suppressed`.
    ///
    /// # Errors
    /// - `EmptyInput` if `bin_counts` is empty.
    /// - `DimensionMismatch` if `bin_counts.len() != k`.
    pub fn release(
        &self,
        bin_counts: &[u64],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<PrivateHistogramOutput> {
        self.validate_bin_counts(bin_counts)?;
        let noisy_counts = self.add_laplace_noise(bin_counts, handle)?;
        self.threshold_and_build(&noisy_counts)
    }

    /// Release a private histogram restricted to the top-`top_k` bins by
    /// noisy count, then applying the stability threshold.
    ///
    /// All `k` bins receive Laplace noise, but only the `top_k` noisiest bins
    /// are candidates for the threshold test.  This limits output size while
    /// retaining correctness.
    ///
    /// # Errors
    /// - `EmptyInput` if `bin_counts` is empty.
    /// - `DimensionMismatch` if `bin_counts.len() != k`.
    /// - `InvalidParameter` if `top_k == 0` or `top_k > k`.
    pub fn release_top_k(
        &self,
        bin_counts: &[u64],
        top_k: usize,
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<PrivateHistogramOutput> {
        if top_k == 0 {
            return Err(PrivacyError::InvalidParameter("top_k must be ≥ 1".into()));
        }
        if top_k > self.cfg.k {
            return Err(PrivacyError::InvalidParameter(
                "top_k cannot exceed k".into(),
            ));
        }
        self.validate_bin_counts(bin_counts)?;
        let noisy_counts = self.add_laplace_noise(bin_counts, handle)?;

        // Sort indices by noisy count descending, take first top_k.
        let mut indices: Vec<usize> = (0..self.cfg.k).collect();
        indices.sort_by(|&a, &b| {
            noisy_counts[b]
                .partial_cmp(&noisy_counts[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_indices = &indices[..top_k];

        // Apply threshold test to the top-k candidates only.
        let threshold = self.compute_threshold();
        let mut counts = vec![0.0f64; self.cfg.k];
        let mut released_bins = Vec::new();

        for &i in top_indices {
            if noisy_counts[i] > threshold {
                counts[i] = noisy_counts[i];
                released_bins.push(i);
            }
        }

        released_bins.sort_unstable();

        if released_bins.is_empty() {
            Ok(PrivateHistogramOutput::Suppressed {
                reason_bin_count: self.cfg.k,
            })
        } else {
            Ok(PrivateHistogramOutput::Released {
                counts,
                released_bins,
            })
        }
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    fn validate_bin_counts(&self, bin_counts: &[u64]) -> PrivacyResult<()> {
        if bin_counts.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if bin_counts.len() != self.cfg.k {
            return Err(PrivacyError::DimensionMismatch {
                expected: self.cfg.k,
                got: bin_counts.len(),
            });
        }
        Ok(())
    }

    /// Add independent `Lap(1/ε)` noise to each bin count.
    fn add_laplace_noise(
        &self,
        bin_counts: &[u64],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<Vec<f64>> {
        let scale = 1.0 / self.cfg.epsilon;
        let noise_vec = handle.generate_laplace_noise(scale, bin_counts.len())?;
        let noisy: Vec<f64> = bin_counts
            .iter()
            .zip(noise_vec.iter())
            .map(|(&c, &n)| c as f64 + n)
            .collect();
        Ok(noisy)
    }

    /// Apply the stability threshold and build the output.
    fn threshold_and_build(&self, noisy_counts: &[f64]) -> PrivacyResult<PrivateHistogramOutput> {
        let threshold = self.compute_threshold();
        let mut counts = vec![0.0f64; self.cfg.k];
        let mut released_bins = Vec::new();

        for (i, &nc) in noisy_counts.iter().enumerate() {
            if nc > threshold {
                counts[i] = nc;
                released_bins.push(i);
            }
        }

        if released_bins.is_empty() {
            Ok(PrivateHistogramOutput::Suppressed {
                reason_bin_count: self.cfg.k,
            })
        } else {
            Ok(PrivateHistogramOutput::Released {
                counts,
                released_bins,
            })
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handle(seed: u64) -> PrivacyHandle {
        PrivacyHandle::new(80, seed)
    }

    fn default_cfg(k: usize) -> PrivateHistogramConfig {
        PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-5,
            k,
            stability_threshold: 0.0,
        }
    }

    // 1. new() with ε ≤ 0 → error.
    #[test]
    fn test_new_rejects_nonpositive_epsilon() {
        let cfg = PrivateHistogramConfig {
            epsilon: 0.0,
            delta: 1e-5,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg).is_err());

        let cfg2 = PrivateHistogramConfig {
            epsilon: -1.0,
            delta: 1e-5,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg2).is_err());
    }

    // 2. new() with δ ≤ 0 → error.
    #[test]
    fn test_new_rejects_nonpositive_delta() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 0.0,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg).is_err());

        let cfg2 = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: -0.1,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg2).is_err());
    }

    // 3. new() with δ ≥ 1 → error.
    #[test]
    fn test_new_rejects_delta_geq_one() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1.0,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg).is_err());

        let cfg2 = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 2.0,
            k: 4,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg2).is_err());
    }

    // 4. new() with k=0 → error.
    #[test]
    fn test_new_rejects_zero_k() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-5,
            k: 0,
            stability_threshold: 0.0,
        };
        assert!(PrivateHistogram::new(cfg).is_err());
    }

    // 5. release with empty bin_counts → EmptyInput.
    #[test]
    fn test_release_empty_bin_counts_yields_error() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let mut handle = make_handle(42);
        let result = hist.release(&[], &mut handle);
        assert!(
            matches!(result, Err(PrivacyError::EmptyInput)),
            "expected EmptyInput, got {result:?}"
        );
    }

    // 6. release with bin_counts.len() != k → DimensionMismatch.
    #[test]
    fn test_release_dimension_mismatch_yields_error() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let mut handle = make_handle(42);
        let result = hist.release(&[1, 2, 3], &mut handle); // len=3, k=4
        assert!(
            matches!(
                result,
                Err(PrivacyError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "expected DimensionMismatch(4,3), got {result:?}"
        );
    }

    // 7. release_top_k with top_k=0 → error.
    #[test]
    fn test_release_top_k_zero_yields_error() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let mut handle = make_handle(42);
        let result = hist.release_top_k(&[10, 20, 30, 40], 0, &mut handle);
        assert!(result.is_err(), "expected error for top_k=0");
    }

    // 8. release_top_k with top_k > k → error.
    #[test]
    fn test_release_top_k_exceeds_k_yields_error() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let mut handle = make_handle(42);
        let result = hist.release_top_k(&[10, 20, 30, 40], 5, &mut handle);
        assert!(result.is_err(), "expected error for top_k=5 > k=4");
    }

    // 9. All-zero counts with large threshold → Suppressed.
    #[test]
    fn test_zero_counts_large_threshold_yields_suppressed() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-5,
            k: 4,
            stability_threshold: 1_000_000.0, // absurdly large
        };
        let hist = PrivateHistogram::new(cfg).expect("ok");
        let mut handle = make_handle(42);
        let result = hist.release(&[0, 0, 0, 0], &mut handle).expect("ok");
        assert!(
            matches!(result, PrivateHistogramOutput::Suppressed { .. }),
            "expected Suppressed with large threshold, got {result:?}"
        );
    }

    // 10. Single bin with very high count (1_000_000) → Released with that bin.
    #[test]
    fn test_very_high_count_yields_released() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-5,
            k: 1,
            stability_threshold: 0.0, // auto compute; will be finite and small
        };
        let hist = PrivateHistogram::new(cfg).expect("ok");
        let mut handle = make_handle(7);
        let result = hist.release(&[1_000_000], &mut handle).expect("ok");
        assert!(
            matches!(result, PrivateHistogramOutput::Released { .. }),
            "expected Released for bin with count 1_000_000"
        );
        if let PrivateHistogramOutput::Released { released_bins, .. } = result {
            assert_eq!(released_bins, vec![0]);
        }
    }

    // 11. Deterministic: same handle state + same bin_counts → same noisy values.
    #[test]
    fn test_deterministic_release() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let counts = [100u64, 200, 300, 400];

        let mut handle_a = make_handle(123);
        let mut handle_b = make_handle(123);

        let out_a = hist.release(&counts, &mut handle_a).expect("ok");
        let out_b = hist.release(&counts, &mut handle_b).expect("ok");

        // Both should have the same released_bins.
        match (out_a, out_b) {
            (
                PrivateHistogramOutput::Released {
                    released_bins: rb_a,
                    counts: c_a,
                },
                PrivateHistogramOutput::Released {
                    released_bins: rb_b,
                    counts: c_b,
                },
            ) => {
                assert_eq!(rb_a, rb_b, "released bins must be deterministic");
                for (ca, cb) in c_a.iter().zip(c_b.iter()) {
                    assert!((ca - cb).abs() < 1e-12, "count mismatch: {ca} vs {cb}");
                }
            }
            (
                PrivateHistogramOutput::Suppressed { .. },
                PrivateHistogramOutput::Suppressed { .. },
            ) => {
                // Both suppressed — consistent.
            }
            (a, b) => panic!("inconsistent outputs: {a:?} vs {b:?}"),
        }
    }

    // 12. release_top_k(k=cfg.k) qualitatively equivalent to release().
    #[test]
    fn test_release_top_k_full_k_matches_release() {
        let hist = PrivateHistogram::new(default_cfg(4)).expect("ok");
        let counts = [50u64, 200, 5, 300];

        let mut handle_a = make_handle(77);
        let mut handle_b = make_handle(77);

        let out_full = hist.release(&counts, &mut handle_a).expect("ok");
        let out_topk = hist.release_top_k(&counts, 4, &mut handle_b).expect("ok");

        // Both should agree on released_bins since the same noise is generated.
        match (out_full, out_topk) {
            (
                PrivateHistogramOutput::Released {
                    released_bins: rb_full,
                    ..
                },
                PrivateHistogramOutput::Released {
                    released_bins: rb_topk,
                    ..
                },
            ) => {
                assert_eq!(rb_full, rb_topk, "full and top-k release should agree");
            }
            (
                PrivateHistogramOutput::Suppressed { .. },
                PrivateHistogramOutput::Suppressed { .. },
            ) => {
                // Both suppressed — consistent.
            }
            (a, b) => panic!("inconsistent: {a:?} vs {b:?}"),
        }
    }

    // 13. threshold formula: compute_threshold() > 1.0 for reasonable ε/δ/k.
    #[test]
    fn test_threshold_formula_exceeds_one() {
        let cfg = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-5,
            k: 10,
            stability_threshold: 0.0,
        };
        let hist = PrivateHistogram::new(cfg).expect("ok");
        let t = hist.compute_threshold();
        assert!(t > 1.0, "expected threshold > 1.0, got {t}");
    }

    // 14. Released counts vec has length k.
    #[test]
    fn test_released_counts_length_equals_k() {
        let k = 6usize;
        let hist = PrivateHistogram::new(default_cfg(k)).expect("ok");
        let counts: Vec<u64> = (1..=k as u64).map(|i| i * 100_000).collect();
        let mut handle = make_handle(31);
        let result = hist.release(&counts, &mut handle).expect("ok");
        if let PrivateHistogramOutput::Released {
            counts: released_counts,
            ..
        } = result
        {
            assert_eq!(released_counts.len(), k, "counts vec length must equal k");
        }
    }

    // 15. released_bins indices are in 0..k.
    #[test]
    fn test_released_bins_indices_in_range() {
        let k = 5usize;
        let hist = PrivateHistogram::new(default_cfg(k)).expect("ok");
        let counts: Vec<u64> = vec![10_000_000; k];
        let mut handle = make_handle(99);
        let result = hist.release(&counts, &mut handle).expect("ok");
        if let PrivateHistogramOutput::Released { released_bins, .. } = result {
            for &idx in &released_bins {
                assert!(idx < k, "released bin index {idx} out of range [0, {k})");
            }
        }
    }

    // 16. Smaller δ → larger threshold (more suppression).
    #[test]
    fn test_smaller_delta_yields_larger_threshold() {
        let cfg_small_delta = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-9,
            k: 10,
            stability_threshold: 0.0,
        };
        let cfg_large_delta = PrivateHistogramConfig {
            epsilon: 1.0,
            delta: 1e-3,
            k: 10,
            stability_threshold: 0.0,
        };
        let hist_small = PrivateHistogram::new(cfg_small_delta).expect("ok");
        let hist_large = PrivateHistogram::new(cfg_large_delta).expect("ok");
        let t_small = hist_small.compute_threshold();
        let t_large = hist_large.compute_threshold();
        assert!(
            t_small > t_large,
            "smaller delta should yield larger threshold: {t_small} vs {t_large}"
        );
    }
}
