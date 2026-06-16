//! Quantile feature transformation.
//!
//! Maps each feature to its empirical quantile → Gaussian or uniform output,
//! mirroring scikit-learn's `QuantileTransformer`.
//!
//! # Algorithm
//!
//! 1. **Fit**: for each feature `j`, store the sorted training values as a
//!    reference distribution (quantile function).
//! 2. **Transform**: for each new value `x_j`, binary-search the sorted
//!    reference column to get the empirical quantile `q ∈ [0, 1]`.
//!    - `QuantileDist::Uniform` → output `q` directly.
//!    - `QuantileDist::Normal` → apply the probit transform
//!      `Φ⁻¹(q)` via the rational Beasley-Springer-Moro approximation
//!      (clipped at `±8` to avoid infinite tails).
//!
//! The resulting features have near-uniform (or near-Gaussian) marginal
//! distributions, which regularises distance-based models and prevents
//! extreme-value features from dominating.

use crate::error::{TabularError, TabularResult};

// ─── QuantileDist ─────────────────────────────────────────────────────────────

/// Target output distribution after quantile transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileDist {
    /// Map to `[0, 1]` uniform — identical to the empirical CDF value.
    Uniform,
    /// Map to standard-normal via the probit (inverse-CDF) transform.
    Normal,
}

// ─── QuantileTransformer ──────────────────────────────────────────────────────

/// Quantile feature transformer.
///
/// Learned from training data; each feature gets its own sorted reference
/// array of size `n_quantiles`.
#[derive(Debug, Clone)]
pub struct QuantileTransformer {
    /// Number of quantile nodes used for interpolation.
    pub n_quantiles: usize,
    /// Target output distribution.
    pub output_dist: QuantileDist,
    /// `n_features` sorted reference arrays, each of length `n_quantiles`.
    /// Stored row-major: `quantiles[j * n_quantiles .. (j+1) * n_quantiles]`.
    pub quantiles: Vec<f32>,
    /// Number of input features.
    pub n_features: usize,
}

impl QuantileTransformer {
    /// Fit the transformer from a `[n_samples × n_features]` row-major matrix.
    ///
    /// Stores up to `n_quantiles` evenly-spaced quantile nodes per feature.
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] if `n_samples == 0` or `n_features == 0`.
    /// - [`TabularError::InvalidParameter`] if `n_quantiles == 0`.
    /// - [`TabularError::DimensionMismatch`] if `data.len() != n_samples * n_features`.
    pub fn fit(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
        n_quantiles: usize,
        output_dist: QuantileDist,
    ) -> TabularResult<Self> {
        if n_samples == 0 || n_features == 0 {
            return Err(TabularError::EmptyInput);
        }
        if n_quantiles == 0 {
            return Err(TabularError::InvalidParameter {
                name: "n_quantiles".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if data.len() != n_samples * n_features {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        // Effective number of quantile nodes — clamp to n_samples so we don't
        // request more nodes than there are data points.
        let n_q = n_quantiles.min(n_samples);
        let mut quantiles = vec![0.0_f32; n_features * n_q];

        for j in 0..n_features {
            // Extract feature column.
            let mut col: Vec<f32> = (0..n_samples).map(|i| data[i * n_features + j]).collect();
            // Sort (NaN-safe: push NaNs to end).
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));

            // Sample n_q evenly-spaced nodes from the sorted column.
            for qi in 0..n_q {
                let frac = if n_q == 1 {
                    0.0
                } else {
                    qi as f32 / (n_q - 1) as f32
                };
                let exact_idx = frac * (n_samples - 1) as f32;
                let lo = exact_idx.floor() as usize;
                let hi = (lo + 1).min(n_samples - 1);
                let t = exact_idx - lo as f32;
                quantiles[j * n_q + qi] = col[lo] + t * (col[hi] - col[lo]);
            }
        }

        Ok(Self {
            n_quantiles: n_q,
            output_dist,
            quantiles,
            n_features,
        })
    }

    /// Fit and immediately transform the training data.
    ///
    /// Returns `(transformer, transformed_data)` where `transformed_data` has
    /// the same layout as `data`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::fit`] or [`Self::transform_row`].
    pub fn fit_transform(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
        n_quantiles: usize,
        output_dist: QuantileDist,
    ) -> TabularResult<(Self, Vec<f32>)> {
        let qt = Self::fit(data, n_samples, n_features, n_quantiles, output_dist)?;
        let mut out = vec![0.0_f32; data.len()];
        for i in 0..n_samples {
            let row = &data[i * n_features..(i + 1) * n_features];
            let t = qt.transform_row(row)?;
            out[i * n_features..(i + 1) * n_features].copy_from_slice(&t);
        }
        Ok((qt, out))
    }

    /// Transform a single feature row (length `n_features`).
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `row.len() != n_features`.
    pub fn transform_row(&self, row: &[f32]) -> TabularResult<Vec<f32>> {
        if row.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: row.len(),
            });
        }
        let mut out = vec![0.0_f32; self.n_features];
        for j in 0..self.n_features {
            let q = self.empirical_quantile(j, row[j]);
            out[j] = match self.output_dist {
                QuantileDist::Uniform => q,
                QuantileDist::Normal => probit(q),
            };
        }
        Ok(out)
    }

    /// Transform a full `[n_samples × n_features]` row-major matrix.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `data.len() != n_samples * n_features`.
    pub fn transform(&self, data: &[f32], n_samples: usize) -> TabularResult<Vec<f32>> {
        if data.len() != n_samples * self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: n_samples * self.n_features,
                got: data.len(),
            });
        }
        let mut out = vec![0.0_f32; data.len()];
        for i in 0..n_samples {
            let row = &data[i * self.n_features..(i + 1) * self.n_features];
            let t = self.transform_row(row)?;
            out[i * self.n_features..(i + 1) * self.n_features].copy_from_slice(&t);
        }
        Ok(out)
    }

    /// Inverse-transform: map each output value back to original feature space.
    ///
    /// For `Uniform` output: interpolates the quantile table in reverse.
    /// For `Normal` output: applies the standard-normal CDF first to get `q`,
    /// then interpolates.
    ///
    /// # Errors
    /// - [`TabularError::DimensionMismatch`] if `row.len() != n_features`.
    pub fn inverse_transform_row(&self, row: &[f32]) -> TabularResult<Vec<f32>> {
        if row.len() != self.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_features,
                got: row.len(),
            });
        }
        let mut out = vec![0.0_f32; self.n_features];
        for j in 0..self.n_features {
            let q = match self.output_dist {
                QuantileDist::Uniform => row[j].clamp(0.0, 1.0),
                QuantileDist::Normal => std_normal_cdf(row[j]),
            };
            out[j] = self.quantile_value(j, q);
        }
        Ok(out)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Binary-search the stored quantile nodes to find the empirical quantile
    /// `q ∈ [0, 1]` for value `v` in feature column `j`.
    fn empirical_quantile(&self, j: usize, v: f32) -> f32 {
        let q_slice = &self.quantiles[j * self.n_quantiles..(j + 1) * self.n_quantiles];
        let n = q_slice.len();
        if n == 0 {
            return 0.0;
        }
        // Handle boundary cases.
        if v <= q_slice[0] {
            return 0.0;
        }
        if v >= q_slice[n - 1] {
            return 1.0;
        }
        // Binary search for bracketing indices.
        let mut lo = 0_usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if q_slice[mid] <= v {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let lo_val = q_slice[lo];
        let hi_val = q_slice[hi];
        let q_lo = lo as f32 / (n - 1) as f32;
        let q_hi = hi as f32 / (n - 1) as f32;
        if (hi_val - lo_val).abs() < 1e-12 {
            q_lo
        } else {
            let t = (v - lo_val) / (hi_val - lo_val);
            q_lo + t * (q_hi - q_lo)
        }
    }

    /// Interpolate the stored quantile nodes to get the original value for
    /// quantile `q ∈ [0, 1]` in feature column `j`.
    fn quantile_value(&self, j: usize, q: f32) -> f32 {
        let q_slice = &self.quantiles[j * self.n_quantiles..(j + 1) * self.n_quantiles];
        let n = q_slice.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return q_slice[0];
        }
        let q_clamped = q.clamp(0.0, 1.0);
        let pos = q_clamped * (n - 1) as f32;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let t = pos - lo as f32;
        q_slice[lo] + t * (q_slice[hi] - q_slice[lo])
    }
}

// ─── Probit (inverse normal CDF) ──────────────────────────────────────────────

/// Rational approximation of the probit function Φ⁻¹(p) for `p ∈ (0, 1)`.
///
/// Uses the Beasley-Springer-Moro rational approximation (1977/1994), which
/// achieves ~6 significant figures across the full domain.  Extreme tails
/// (`p < 1e-7` or `p > 1 − 1e-7`) are clipped to ±8.
pub fn probit(p: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    // Coefficients from the BSM approximation (central region).
    const A: [f64; 4] = [2.515_517, 0.802_853, 0.010_328, 0.0];
    const B: [f64; 3] = [1.432_788, 0.189_269, 0.001_308];
    let p64 = p as f64;
    let (sign, prob) = if p64 < 0.5 {
        (-1.0_f64, p64)
    } else {
        (1.0_f64, 1.0 - p64)
    };
    let t = (-2.0 * prob.ln()).sqrt();
    let num = A[0] + t * (A[1] + t * (A[2] + t * A[3]));
    let den = 1.0 + t * (B[0] + t * (B[1] + t * B[2]));
    let z = sign * (t - num / den);
    (z as f32).clamp(-8.0, 8.0)
}

/// Standard normal CDF Φ(x) via the Horner-form rational approximation.
///
/// Abramowitz and Stegun, formula 26.2.17.  Accurate to ~7 decimal places.
pub fn std_normal_cdf(x: f32) -> f32 {
    let x64 = x as f64;
    let t = 1.0 / (1.0 + 0.2316_419 * x64.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = 1.0 - ((-0.5 * x64 * x64).exp() / std::f64::consts::TAU.sqrt()) * poly;
    let result = if x64 >= 0.0 { phi } else { 1.0 - phi };
    result.clamp(0.0, 1.0) as f32
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    // ── 1. Uniform output lies in [0, 1] for in-distribution data ─────────────
    #[test]
    fn uniform_output_in_range() {
        let data = linspace(20);
        let qt = QuantileTransformer::fit(&data, 20, 1, 20, QuantileDist::Uniform)
            .expect("fit should succeed");
        let out = qt.transform(&data, 20).expect("transform should succeed");
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "uniform output {v} not in [0, 1]");
        }
    }

    // ── 2. Normal output: all finite for in-distribution data ─────────────────
    #[test]
    fn normal_output_finite() {
        let data: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let qt = QuantileTransformer::fit(&data, 50, 1, 20, QuantileDist::Normal)
            .expect("fit should succeed");
        let out = qt.transform(&data, 50).expect("transform should succeed");
        for &v in &out {
            assert!(v.is_finite(), "normal output {v} is not finite");
        }
    }

    // ── 3. Fit-transform is consistent with separate fit then transform ────────
    #[test]
    fn fit_transform_consistent() {
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let (qt, out1) =
            QuantileTransformer::fit_transform(&data, 30, 1, 15, QuantileDist::Uniform)
                .expect("fit_transform should succeed");
        let out2 = qt.transform(&data, 30).expect("transform should succeed");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    // ── 4. Values strictly beyond training range → 0 or 1 ─────────────────────
    #[test]
    fn boundary_clamping() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 0..9
        let qt = QuantileTransformer::fit(&data, 10, 1, 10, QuantileDist::Uniform)
            .expect("fit should succeed");
        let below = qt
            .transform_row(&[-100.0])
            .expect("transform_row should succeed");
        let above = qt
            .transform_row(&[1000.0])
            .expect("transform_row should succeed");
        assert!((below[0] - 0.0).abs() < 1e-6);
        assert!((above[0] - 1.0).abs() < 1e-6);
    }

    // ── 5. Monotonicity: larger input → larger quantile (uniform mode) ─────────
    #[test]
    fn monotone_uniform() {
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let qt = QuantileTransformer::fit(&data, 40, 1, 20, QuantileDist::Uniform)
            .expect("fit should succeed");
        let mut prev = -1.0_f32;
        for &v in &data {
            let q = qt
                .transform_row(&[v])
                .expect("transform_row should succeed")[0];
            assert!(q >= prev - 1e-6, "not monotone: {q} < {prev}");
            prev = q;
        }
    }

    // ── 6. Multi-feature transform shape ──────────────────────────────────────
    #[test]
    fn multi_feature_shape() {
        let n_s = 20_usize;
        let n_f = 4_usize;
        let data: Vec<f32> = (0..n_s * n_f).map(|i| i as f32 * 0.1).collect();
        let qt = QuantileTransformer::fit(&data, n_s, n_f, 10, QuantileDist::Uniform)
            .expect("fit should succeed");
        let out = qt.transform(&data, n_s).expect("transform should succeed");
        assert_eq!(out.len(), n_s * n_f);
    }

    // ── 7. Probit / CDF are approximate inverses ───────────────────────────────
    #[test]
    fn probit_cdf_inverse() {
        for &p in &[0.01_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let z = probit(p);
            let p2 = std_normal_cdf(z);
            assert!((p2 - p).abs() < 0.01, "p={p} → z={z} → p2={p2}");
        }
    }

    // ── 8. Inverse transform round-trips for Uniform mode ─────────────────────
    #[test]
    fn inverse_transform_roundtrip_uniform() {
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let qt = QuantileTransformer::fit(&data, 30, 1, 30, QuantileDist::Uniform)
            .expect("fit should succeed");
        // Take mid-range values; boundary artefacts are acceptable.
        for &v in &data[2..28] {
            let q_row = qt
                .transform_row(&[v])
                .expect("transform_row should succeed");
            let v2 = qt
                .inverse_transform_row(&q_row)
                .expect("inverse_transform_row should succeed")[0];
            assert!((v2 - v).abs() < 0.6, "round-trip failed: {v} → {v2}");
        }
    }

    // ── 9. dimension mismatch errors ──────────────────────────────────────────
    #[test]
    fn dimension_mismatch_errors() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let qt = QuantileTransformer::fit(&data, 20, 1, 10, QuantileDist::Uniform)
            .expect("fit should succeed");
        // Wrong row length.
        assert!(qt.transform_row(&[0.0, 1.0]).is_err());
        // Wrong matrix size.
        assert!(qt.transform(&data, 25).is_err());
    }

    // ── 10. n_quantiles clamped to n_samples ──────────────────────────────────
    #[test]
    fn n_quantiles_clamped() {
        let data: Vec<f32> = (0..5).map(|i| i as f32).collect();
        // Request more quantiles than samples — should succeed and clamp.
        let qt = QuantileTransformer::fit(&data, 5, 1, 100, QuantileDist::Uniform)
            .expect("fit should succeed");
        assert_eq!(qt.n_quantiles, 5);
    }

    // ── 11. Empty-input and zero-quantile errors ───────────────────────────────
    #[test]
    fn empty_and_invalid_errors() {
        assert!(QuantileTransformer::fit(&[], 0, 1, 10, QuantileDist::Uniform).is_err());
        let data: Vec<f32> = vec![1.0, 2.0];
        assert!(QuantileTransformer::fit(&data, 2, 1, 0, QuantileDist::Uniform).is_err());
    }
}
