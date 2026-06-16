//! Weight-of-Evidence (WoE) encoding and Information Value (IV).
//!
//! For a binary target (`0 = "good"`, `1 = "bad"` in credit-scoring parlance) a
//! categorical feature's level `c` is replaced by its **weight of evidence**
//!
//! ```text
//! WoE(c) = ln( (good_c / good_total) / (bad_c / bad_total) )
//! ```
//!
//! i.e. the log-ratio of the within-level *good* distribution to the *bad*
//! distribution. Positive WoE means the level is associated with the good class.
//! A small Laplace smoothing term (`alpha`) prevents `0`/`0` or `ln(0)` when a
//! level contains only one class.
//!
//! The **information value** of the whole feature aggregates the per-level
//! separation
//!
//! ```text
//! IV = Σ_c (good_c/good_total − bad_c/bad_total) · WoE(c)
//! ```
//!
//! and is the standard univariate predictive-power score: `< 0.02` useless,
//! `0.1–0.3` medium, `> 0.5` suspiciously strong. WoE encoding is monotone in
//! the bad-rate and keeps logistic-regression coefficients interpretable.
//!
//! ## References
//! - Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.

use crate::error::{TabularError, TabularResult};

/// A fitted Weight-of-Evidence encoder for a single categorical feature.
#[derive(Debug, Clone)]
pub struct WoeEncoder {
    /// WoE value for each category index `0..n_categories`.
    woe: Vec<f64>,
    /// Information value of the feature.
    information_value: f64,
    /// Number of categories.
    n_categories: usize,
    /// WoE assigned to unseen / out-of-range category indices at transform time.
    default_woe: f64,
}

impl WoeEncoder {
    /// Fit a WoE encoder from integer category codes `x` and a binary target
    /// `y` (`0`/`1`; any non-zero `y` is treated as the positive/"bad" class).
    ///
    /// `n_categories` is the number of distinct category codes; every entry of
    /// `x` must be `< n_categories`. `alpha` is the Laplace smoothing count
    /// added to each cell (use `0.5` for the common Haldane–Anscombe correction).
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] if `x` is empty.
    /// - [`TabularError::DimensionMismatch`] if `x.len() != y.len()`.
    /// - [`TabularError::InvalidFeatureCount`] if `n_categories == 0`.
    /// - [`TabularError::CategoricalOutOfRange`] if any code `≥ n_categories`.
    /// - [`TabularError::InvalidParameter`] if `alpha < 0`, or if the target has
    ///   no positives or no negatives (WoE is undefined).
    pub fn fit(x: &[usize], y: &[u8], n_categories: usize, alpha: f64) -> TabularResult<Self> {
        if x.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        if x.len() != y.len() {
            return Err(TabularError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        if n_categories == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if alpha < 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "alpha".into(),
                msg: "smoothing must be ≥ 0".into(),
            });
        }

        // Per-category good (y=0) and bad (y=1) counts.
        let mut good = vec![0.0_f64; n_categories];
        let mut bad = vec![0.0_f64; n_categories];
        let mut good_total = 0.0_f64;
        let mut bad_total = 0.0_f64;
        for (&c, &label) in x.iter().zip(y.iter()) {
            if c >= n_categories {
                return Err(TabularError::CategoricalOutOfRange {
                    feat: 0,
                    val: c,
                    n: n_categories,
                });
            }
            if label != 0 {
                bad[c] += 1.0;
                bad_total += 1.0;
            } else {
                good[c] += 1.0;
                good_total += 1.0;
            }
        }
        if good_total == 0.0 || bad_total == 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "y".into(),
                msg: "target must contain both classes for WoE".into(),
            });
        }

        // Smoothed totals so the distributions still sum to ≈ 1.
        let k = n_categories as f64;
        let good_denom = good_total + alpha * k;
        let bad_denom = bad_total + alpha * k;

        let mut woe = vec![0.0_f64; n_categories];
        let mut iv = 0.0_f64;
        for c in 0..n_categories {
            let dist_good = (good[c] + alpha) / good_denom;
            let dist_bad = (bad[c] + alpha) / bad_denom;
            let w = (dist_good / dist_bad).ln();
            woe[c] = w;
            iv += (dist_good - dist_bad) * w;
        }

        Ok(Self {
            woe,
            information_value: iv,
            n_categories,
            default_woe: 0.0,
        })
    }

    /// The information value of the fitted feature.
    #[must_use]
    pub fn information_value(&self) -> f64 {
        self.information_value
    }

    /// The WoE value for a single category index, or [`Self::default_woe`] for
    /// an out-of-range index.
    #[must_use]
    pub fn woe_for(&self, category: usize) -> f64 {
        if category < self.n_categories {
            self.woe[category]
        } else {
            self.default_woe
        }
    }

    /// The per-category WoE table.
    #[must_use]
    pub fn woe_table(&self) -> &[f64] {
        &self.woe
    }

    /// The WoE assigned to unseen categories at transform time.
    #[must_use]
    pub fn default_woe(&self) -> f64 {
        self.default_woe
    }

    /// Transform a slice of category codes into their WoE values. Codes that are
    /// out of range receive [`Self::default_woe`] rather than erroring, so the
    /// encoder is robust to categories unseen during fitting.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] if `x` is empty.
    pub fn transform(&self, x: &[usize]) -> TabularResult<Vec<f64>> {
        if x.is_empty() {
            return Err(TabularError::EmptyInput);
        }
        Ok(x.iter().map(|&c| self.woe_for(c)).collect())
    }

    /// Convenience: fit on `(x, y)` and immediately transform `x`.
    ///
    /// # Errors
    /// As [`WoeEncoder::fit`].
    pub fn fit_transform(
        x: &[usize],
        y: &[u8],
        n_categories: usize,
        alpha: f64,
    ) -> TabularResult<(Self, Vec<f64>)> {
        let enc = Self::fit(x, y, n_categories, alpha)?;
        let transformed = enc.transform(x)?;
        Ok((enc, transformed))
    }
}

/// Standalone information value of a categorical feature (without keeping the
/// encoder). Equivalent to `WoeEncoder::fit(...).information_value()`.
///
/// # Errors
/// As [`WoeEncoder::fit`].
pub fn information_value(
    x: &[usize],
    y: &[u8],
    n_categories: usize,
    alpha: f64,
) -> TabularResult<f64> {
    Ok(WoeEncoder::fit(x, y, n_categories, alpha)?.information_value())
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn woe_sign_follows_class_association() {
        // Category 0 is almost all good (y=0), category 1 almost all bad (y=1).
        let x = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
        let y = vec![0u8, 0, 0, 1, 1, 1, 1, 0];
        let enc = WoeEncoder::fit(&x, &y, 2, 0.5).expect("ok");
        // Good-associated category → positive WoE; bad-associated → negative.
        assert!(enc.woe_for(0) > 0.0, "woe0={}", enc.woe_for(0));
        assert!(enc.woe_for(1) < 0.0, "woe1={}", enc.woe_for(1));
    }

    #[test]
    fn woe_zero_for_neutral_category() {
        // A category whose good/bad rate matches the overall rate → WoE ≈ 0.
        // Overall: 4 good, 4 bad. Category 0 has 2 good 2 bad (matches), as does 1.
        let x = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
        let y = vec![0u8, 0, 1, 1, 0, 0, 1, 1];
        let enc = WoeEncoder::fit(&x, &y, 2, 0.0).expect("ok");
        assert!(enc.woe_for(0).abs() < 1e-9, "woe0={}", enc.woe_for(0));
        assert!(enc.woe_for(1).abs() < 1e-9, "woe1={}", enc.woe_for(1));
    }

    #[test]
    fn information_value_positive_for_predictive_feature() {
        let x = vec![0usize, 0, 0, 0, 1, 1, 1, 1];
        let y = vec![0u8, 0, 0, 0, 1, 1, 1, 1]; // perfect separation
        let iv = information_value(&x, &y, 2, 0.5).expect("ok");
        assert!(iv > 0.5, "iv={iv}");
    }

    #[test]
    fn information_value_near_zero_for_useless_feature() {
        // Both categories have identical good/bad distribution → IV ≈ 0.
        let x = vec![0usize, 0, 1, 1, 0, 0, 1, 1];
        let y = vec![0u8, 1, 0, 1, 0, 1, 0, 1];
        let iv = information_value(&x, &y, 2, 0.0).expect("ok");
        assert!(iv.abs() < 1e-9, "iv={iv}");
    }

    #[test]
    fn iv_matches_encoder() {
        let x = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2];
        let y = vec![0u8, 1, 1, 0, 0, 1, 0, 1, 0];
        let enc = WoeEncoder::fit(&x, &y, 3, 0.5).expect("ok");
        let iv = information_value(&x, &y, 3, 0.5).expect("ok");
        assert!((enc.information_value() - iv).abs() < 1e-12);
    }

    #[test]
    fn transform_maps_codes_to_woe() {
        let x = vec![0usize, 0, 1, 1, 2, 2];
        let y = vec![0u8, 0, 1, 1, 0, 1];
        let (enc, out) = WoeEncoder::fit_transform(&x, &y, 3, 0.5).expect("ok");
        assert_eq!(out.len(), x.len());
        for (i, &c) in x.iter().enumerate() {
            assert!((out[i] - enc.woe_for(c)).abs() < 1e-12);
        }
    }

    #[test]
    fn transform_unseen_category_uses_default() {
        let x = vec![0usize, 0, 1, 1];
        let y = vec![0u8, 0, 1, 1];
        let enc = WoeEncoder::fit(&x, &y, 2, 0.5).expect("ok");
        // Category 5 was never fit; transform must not error.
        let out = enc.transform(&[5usize]).expect("ok");
        assert_eq!(out[0], enc.default_woe());
    }

    #[test]
    fn woe_table_length() {
        let x = vec![0usize, 1, 2, 0, 1, 2];
        let y = vec![0u8, 1, 0, 1, 0, 1];
        let enc = WoeEncoder::fit(&x, &y, 3, 0.5).expect("ok");
        assert_eq!(enc.woe_table().len(), 3);
    }

    #[test]
    fn smoothing_handles_single_class_category() {
        // Category 2 appears only with y=1 (bad). Without smoothing this would
        // produce −∞; with alpha>0 it is finite.
        let x = vec![0usize, 0, 1, 1, 2, 2];
        let y = vec![0u8, 1, 0, 1, 1, 1];
        let enc = WoeEncoder::fit(&x, &y, 3, 0.5).expect("ok");
        assert!(enc.woe_for(2).is_finite());
        assert!(
            enc.woe_for(2) < 0.0,
            "single-bad category should be negative"
        );
    }

    #[test]
    fn nonzero_target_treated_as_bad() {
        let x = vec![0usize, 0, 1, 1];
        let y_one = vec![0u8, 0, 1, 1];
        let y_two = vec![0u8, 0, 7, 3]; // non-zero codes → bad
        let e1 = WoeEncoder::fit(&x, &y_one, 2, 0.5).expect("ok");
        let e2 = WoeEncoder::fit(&x, &y_two, 2, 0.5).expect("ok");
        assert!((e1.woe_for(1) - e2.woe_for(1)).abs() < 1e-12);
    }

    #[test]
    fn empty_input_error() {
        assert!(matches!(
            WoeEncoder::fit(&[], &[], 2, 0.5),
            Err(TabularError::EmptyInput)
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let x = vec![0usize, 1, 2];
        let y = vec![0u8, 1];
        assert!(matches!(
            WoeEncoder::fit(&x, &y, 3, 0.5),
            Err(TabularError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn zero_categories_error() {
        let x = vec![0usize, 0];
        let y = vec![0u8, 1];
        assert!(matches!(
            WoeEncoder::fit(&x, &y, 0, 0.5),
            Err(TabularError::InvalidFeatureCount { .. })
        ));
    }

    #[test]
    fn category_out_of_range_error() {
        let x = vec![0usize, 3]; // 3 ≥ n_categories=2
        let y = vec![0u8, 1];
        assert!(matches!(
            WoeEncoder::fit(&x, &y, 2, 0.5),
            Err(TabularError::CategoricalOutOfRange { .. })
        ));
    }

    #[test]
    fn negative_alpha_error() {
        let x = vec![0usize, 1];
        let y = vec![0u8, 1];
        assert!(matches!(
            WoeEncoder::fit(&x, &y, 2, -0.1),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn single_class_target_error() {
        let x = vec![0usize, 1, 0, 1];
        let y = vec![0u8, 0, 0, 0]; // all good → undefined WoE
        assert!(matches!(
            WoeEncoder::fit(&x, &y, 2, 0.5),
            Err(TabularError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn transform_empty_error() {
        let x = vec![0usize, 1];
        let y = vec![0u8, 1];
        let enc = WoeEncoder::fit(&x, &y, 2, 0.5).expect("ok");
        assert!(matches!(enc.transform(&[]), Err(TabularError::EmptyInput)));
    }

    #[test]
    fn deterministic() {
        let x = vec![0usize, 1, 2, 0, 1, 2];
        let y = vec![0u8, 1, 0, 1, 1, 0];
        let a = WoeEncoder::fit(&x, &y, 3, 0.5).expect("ok");
        let b = WoeEncoder::fit(&x, &y, 3, 0.5).expect("ok");
        assert_eq!(a.woe_table(), b.woe_table());
        assert_eq!(a.information_value(), b.information_value());
    }
}
