//! Target encoding with smoothing and leave-one-out (LOO) variant.
//!
//! Replaces each categorical level with a smoothed version of `E[y | cat = c]`.
//! LOO removes the current sample from the group estimate during training to
//! prevent target leakage.
//!
//! Reference: Micci-Barreca (2001), "A preprocessing scheme for high-cardinality
//! categorical attributes in classification and prediction problems."

use std::collections::HashMap;

use crate::error::{TabularError, TabularResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`TargetEncoder`].
#[derive(Debug, Clone)]
pub struct TargetEncoderConfig {
    /// Smoothing strength. High `k` pulls the encoding toward the global mean.
    pub k: f32,
    /// Minimum group count before using the global mean instead of smoothed mean.
    pub min_count: usize,
}

impl Default for TargetEncoderConfig {
    fn default() -> Self {
        Self {
            k: 1.0,
            min_count: 1,
        }
    }
}

// ─── TargetEncoder ────────────────────────────────────────────────────────────

/// Smoothed target encoder: maps each categorical level to an estimate of
/// `E[y | cat = c]` regularised toward the global mean.
///
/// # Layout convention
/// `x_cat` is a flat row-major slice of shape `[n_samples × n_cat_features]`.
/// Entry at sample `i`, feature `f` is `x_cat[i * n_cat_features + f]`.
pub struct TargetEncoder {
    global_mean: f32,
    /// `category_means[f][c]` = smoothed mean for feature `f`, category `c`.
    category_means: Vec<HashMap<usize, f32>>,
    /// `group_sums[f][c]` = sum of `y` for category `c` of feature `f`.
    group_sums: Vec<HashMap<usize, f32>>,
    /// `group_counts[f][c]` = number of samples with category `c` for feature `f`.
    group_counts: Vec<HashMap<usize, usize>>,
    n_cat_features: usize,
    config: TargetEncoderConfig,
}

impl TargetEncoder {
    /// Fit on categorical inputs `x_cat` and continuous targets `y`.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] when `n_samples == 0`.
    /// - [`TabularError::DimensionMismatch`] when slice lengths do not match.
    pub fn fit(
        x_cat: &[usize],
        y: &[f32],
        n_samples: usize,
        n_cat_features: usize,
        config: TargetEncoderConfig,
    ) -> TabularResult<Self> {
        if n_samples == 0 {
            return Err(TabularError::EmptyInput);
        }
        let expected_x = n_samples * n_cat_features;
        if x_cat.len() != expected_x {
            return Err(TabularError::DimensionMismatch {
                expected: expected_x,
                got: x_cat.len(),
            });
        }
        if y.len() != n_samples {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples,
                got: y.len(),
            });
        }

        // Global mean.
        let global_mean = {
            let sum: f32 = y.iter().sum();
            sum / n_samples as f32
        };

        // Accumulate per-feature group sums and counts.
        let mut group_sums: Vec<HashMap<usize, f32>> =
            (0..n_cat_features).map(|_| HashMap::new()).collect();
        let mut group_counts: Vec<HashMap<usize, usize>> =
            (0..n_cat_features).map(|_| HashMap::new()).collect();

        for i in 0..n_samples {
            let yi = y[i];
            for f in 0..n_cat_features {
                let c = x_cat[i * n_cat_features + f];
                *group_sums[f].entry(c).or_insert(0.0) += yi;
                *group_counts[f].entry(c).or_insert(0) += 1;
            }
        }

        // Compute smoothed category means.
        let mut category_means: Vec<HashMap<usize, f32>> =
            (0..n_cat_features).map(|_| HashMap::new()).collect();

        for f in 0..n_cat_features {
            for (&c, &n_c) in &group_counts[f] {
                let smoothed = if n_c >= config.min_count {
                    let group_sum = group_sums[f].get(&c).copied().unwrap_or(0.0);
                    let group_mean_c = group_sum / n_c as f32;
                    (n_c as f32 * group_mean_c + config.k * global_mean) / (n_c as f32 + config.k)
                } else {
                    global_mean
                };
                category_means[f].insert(c, smoothed);
            }
        }

        Ok(Self {
            global_mean,
            category_means,
            group_sums,
            group_counts,
            n_cat_features,
            config,
        })
    }

    /// Transform `x_cat` to a flat `[n_samples × n_cat_features]` float matrix
    /// of smoothed target means. Unseen categories fall back to `global_mean`.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] when slice length does not match.
    pub fn transform(&self, x_cat: &[usize], n_samples: usize) -> TabularResult<Vec<f32>> {
        let expected = n_samples * self.n_cat_features;
        if x_cat.len() != expected {
            return Err(TabularError::DimensionMismatch {
                expected,
                got: x_cat.len(),
            });
        }
        let mut out = Vec::with_capacity(expected);
        for i in 0..n_samples {
            for f in 0..self.n_cat_features {
                let c = x_cat[i * self.n_cat_features + f];
                let enc = self.category_means[f]
                    .get(&c)
                    .copied()
                    .unwrap_or(self.global_mean);
                out.push(enc);
            }
        }
        Ok(out)
    }

    /// Leave-one-out transform.
    ///
    /// For sample `i`, feature `f`, category `c`, the current sample is
    /// subtracted from the group statistics before computing the smoothed mean.
    /// This prevents target leakage when the encoder is used on the training set.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] when slice lengths do not match.
    pub fn transform_loo(
        &self,
        x_cat: &[usize],
        y: &[f32],
        n_samples: usize,
    ) -> TabularResult<Vec<f32>> {
        let expected_x = n_samples * self.n_cat_features;
        if x_cat.len() != expected_x {
            return Err(TabularError::DimensionMismatch {
                expected: expected_x,
                got: x_cat.len(),
            });
        }
        if y.len() != n_samples {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples,
                got: y.len(),
            });
        }

        let mut out = Vec::with_capacity(expected_x);
        for i in 0..n_samples {
            let yi = y[i];
            for f in 0..self.n_cat_features {
                let c = x_cat[i * self.n_cat_features + f];
                let n_c = self.group_counts[f].get(&c).copied().unwrap_or(1);
                let sum_c = self.group_sums[f].get(&c).copied().unwrap_or(yi);

                let loo_sum = sum_c - yi;
                // Avoid division by zero: at least 1.
                let loo_n = n_c.saturating_sub(1).max(1);
                let loo_mean = loo_sum / loo_n as f32;
                let enc = (loo_n as f32 * loo_mean + self.config.k * self.global_mean)
                    / (loo_n as f32 + self.config.k);
                out.push(enc);
            }
        }
        Ok(out)
    }

    /// The global target mean computed during [`fit`](Self::fit).
    #[must_use]
    pub fn global_mean(&self) -> f32 {
        self.global_mean
    }

    /// Number of categorical features.
    #[must_use]
    pub fn n_cat_features(&self) -> usize {
        self.n_cat_features
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: single feature, n categories.
    fn fit1(cats: &[usize], y: &[f32], k: f32, min_count: usize) -> TargetEncoder {
        let cfg = TargetEncoderConfig { k, min_count };
        TargetEncoder::fit(cats, y, cats.len(), 1, cfg).expect("value should be present")
    }

    // ── 1. Binary target: cat_0 → 0.0, cat_1 → 1.0 ─────────────────────────
    #[test]
    fn binary_target_encodes_proba() {
        let cats = vec![0usize, 0, 0, 1, 1, 1];
        let y = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let enc = fit1(&cats, &y, 0.0, 1);
        let out = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        // cat 0 → mean = 0.0, cat 1 → mean = 1.0
        assert!((out[0] - 0.0).abs() < 1e-5, "cat=0 should encode to 0.0");
        assert!((out[3] - 1.0).abs() < 1e-5, "cat=1 should encode to 1.0");
    }

    // ── 2. Very high k collapses to global mean ──────────────────────────────
    #[test]
    fn smoothing_collapses_at_high_k() {
        let cats = vec![0usize, 0, 1, 1, 2];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0, 0.5];
        let global_mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        let enc = fit1(&cats, &y, 1e9, 1);
        let out = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        for &v in &out {
            assert!(
                (v - global_mean).abs() < 1e-3,
                "k=1e9: expected {global_mean}, got {v}"
            );
        }
    }

    // ── 3. k=0 gives exact group mean ───────────────────────────────────────
    #[test]
    fn no_smoothing_at_k_zero() {
        let cats = vec![0usize, 0, 0, 1, 1];
        let y = vec![1.0_f32, 3.0, 5.0, 10.0, 20.0];
        let enc = fit1(&cats, &y, 0.0, 1);
        let out = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        // cat 0: exact mean = (1+3+5)/3 = 3.0
        assert!((out[0] - 3.0).abs() < 1e-5, "cat=0 exact mean={}", out[0]);
        // cat 1: exact mean = (10+20)/2 = 15.0
        assert!((out[3] - 15.0).abs() < 1e-5, "cat=1 exact mean={}", out[3]);
    }

    // ── 4. LOO differs from regular transform ────────────────────────────────
    #[test]
    fn loo_differs_from_regular() {
        let cats = vec![0usize, 0, 0, 1, 1, 1];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0, 1.0, 0.0];
        let cfg = TargetEncoderConfig {
            k: 0.5,
            min_count: 1,
        };
        let enc =
            TargetEncoder::fit(&cats, &y, cats.len(), 1, cfg).expect("value should be present");
        let regular = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        let loo = enc
            .transform_loo(&cats, &y, cats.len())
            .expect("value should be present");
        // At least some values should differ.
        let any_differ = regular
            .iter()
            .zip(loo.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_differ,
            "LOO must differ from regular for non-trivial input"
        );
    }

    // ── 5. Unseen category returns global mean ───────────────────────────────
    #[test]
    fn unseen_category_fallback() {
        let cats = vec![0usize, 0, 1, 1];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0];
        let enc = fit1(&cats, &y, 0.0, 1);
        let global_mean = enc.global_mean();
        // Category 99 was never seen.
        let unseen = vec![99usize];
        let out = enc.transform(&unseen, 1).expect("transform should succeed");
        assert!(
            (out[0] - global_mean).abs() < 1e-5,
            "unseen cat should return global_mean"
        );
    }

    // ── 6. Output shape correct ──────────────────────────────────────────────
    #[test]
    fn output_shape_correct() {
        let n_samples = 10;
        let n_cat_features = 3;
        let cats: Vec<usize> = (0..(n_samples * n_cat_features)).map(|i| i % 4).collect();
        let y: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        let cfg = TargetEncoderConfig::default();
        let enc = TargetEncoder::fit(&cats, &y, n_samples, n_cat_features, cfg)
            .expect("fit should succeed");
        let out = enc
            .transform(&cats, n_samples)
            .expect("transform should succeed");
        assert_eq!(
            out.len(),
            n_samples * n_cat_features,
            "output shape mismatch"
        );
    }

    // ── 7. Constant target: all encodings ≈ constant ────────────────────────
    #[test]
    fn constant_target_all_same() {
        let cats = vec![0usize, 1, 2, 0, 1, 2];
        let y = vec![5.0_f32; 6];
        let enc = fit1(&cats, &y, 1.0, 1);
        let out = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        for &v in &out {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "constant target: expected 5.0, got {v}"
            );
        }
    }

    // ── 8. min_count: rare category falls back to global mean ────────────────
    #[test]
    fn min_count_uses_global_mean() {
        // cat=0 appears 4 times, cat=1 appears once.
        let cats = vec![0usize, 0, 0, 0, 1];
        let y = vec![0.0_f32, 0.0, 0.0, 0.0, 100.0];
        let cfg = TargetEncoderConfig {
            k: 0.0,
            min_count: 5,
        }; // cat=1 with count=1 < min_count=5
        let enc =
            TargetEncoder::fit(&cats, &y, cats.len(), 1, cfg).expect("value should be present");
        let global_mean = enc.global_mean();
        let out = enc
            .transform(&cats, cats.len())
            .expect("value should be present");
        // cat=1 at index 4 should be global_mean, not 100.0
        assert!(
            (out[4] - global_mean).abs() < 1e-5,
            "rare cat should use global_mean={global_mean}, got {}",
            out[4]
        );
    }

    // ── 9. Empty input returns error ─────────────────────────────────────────
    #[test]
    fn empty_input_error() {
        let result = TargetEncoder::fit(&[], &[], 0, 1, TargetEncoderConfig::default());
        assert!(
            matches!(result, Err(TabularError::EmptyInput)),
            "expected EmptyInput error"
        );
    }

    // ── 10. Dimension mismatch ────────────────────────────────────────────────
    #[test]
    fn dimension_mismatch_error() {
        // x_cat.len() should be 3*2=6, but we give only 3.
        let x_cat = vec![0usize, 1, 2];
        let y = vec![1.0_f32, 2.0, 3.0];
        let result = TargetEncoder::fit(&x_cat, &y, 3, 2, TargetEncoderConfig::default());
        assert!(
            matches!(result, Err(TabularError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    // ── 11. Multiple features are encoded independently ──────────────────────
    #[test]
    fn multi_feature_independence() {
        // Feature 0: cats [0, 0, 1, 1], y=[0,0,1,1] → cat0→0.0, cat1→1.0
        // Feature 1: cats [0, 1, 0, 1], y same    → each cat sees both 0 and 1
        //   So feature 1, cat 0 group: y=[0,1] mean=0.5; feature 1, cat 1 group: y=[0,1] mean=0.5
        // With k=0 the encodings for feature 0 differ from feature 1.
        let x_cat = vec![
            0usize, 0, // sample 0: feat0=0, feat1=0
            0, 1, // sample 1: feat0=0, feat1=1
            1, 0, // sample 2: feat0=1, feat1=0
            1, 1, // sample 3: feat0=1, feat1=1
        ];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0];
        let cfg = TargetEncoderConfig {
            k: 0.0,
            min_count: 1,
        };
        let enc = TargetEncoder::fit(&x_cat, &y, 4, 2, cfg).expect("fit should succeed");
        let out = enc.transform(&x_cat, 4).expect("transform should succeed");
        // Feature 0 (stride=2, offset=0): sample 0 → enc for feat0=0 = 0.0
        // Feature 1 (stride=2, offset=1): sample 0 → enc for feat1=0 = 0.5
        assert!((out[0] - 0.0).abs() < 1e-5, "feat0 cat0 should be 0.0");
        assert!((out[1] - 0.5).abs() < 1e-5, "feat1 cat0 should be 0.5");
        // Sample 2: feat0=1→1.0, feat1=0→0.5
        assert!((out[4] - 1.0).abs() < 1e-5, "feat0 cat1 should be 1.0");
        assert!(
            (out[5] - 0.5).abs() < 1e-5,
            "feat1 cat0 should be 0.5 again"
        );
    }
}
