//! # LLM.int8() — Outlier-Aware Mixed-Precision Quantization
//!
//! Dettmers et al. (2022): "LLM.int8(): 8-bit Matrix Multiplication for
//! Transformers at Scale" <https://arxiv.org/abs/2208.07339>
//!
//! ## Key idea
//!
//! At ≥6.7B parameters, transformer activations develop a small set of
//! *emergent outlier features*: a handful of input (hidden) dimensions whose
//! magnitudes are 10–100× larger than the rest. Naïvely quantizing those
//! dimensions to INT8 destroys accuracy, because the few huge values stretch
//! the per-tensor scale and crush all the small values to zero.
//!
//! LLM.int8() decomposes the linear layer `Y = X · Wᵀ` into two paths:
//!
//! ```text
//! Y = X[:, reg] · Wᵀ[reg, :]   (INT8, vector-wise quantized)
//!   + X[:, out] · Wᵀ[out, :]   (FP16, kept at full precision)
//! ```
//!
//! where `out` is the set of *outlier columns* — input dimensions in which
//! any activation exceeds an absolute `threshold` (default 6.0) — and `reg`
//! is the remaining "regular" dimensions. Only ~0.1 % of dimensions are
//! outliers, so the FP16 path is tiny while accuracy is preserved.
//!
//! ## Vector-wise (row/column) quantization
//!
//! Each row of `X` and each column of `Wᵀ` gets its own INT8 scale:
//!
//! ```text
//! sx[r] = max_c |X[r, c]| / 127        (per activation row / token)
//! sw[k] = max_r |W[k, r]| / 127        (per weight row of W = column of Wᵀ)
//! ```
//!
//! The INT8 product accumulates in INT32 and is rescaled by `sx[r]·sw[k]`.
//! This module computes the decomposition entirely on CPU `&[f32]` slices and
//! is fully unit-testable without any GPU INT8 tensor cores.

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for LLM.int8() outlier-aware decomposition.
#[derive(Debug, Clone)]
pub struct LlmInt8Config {
    /// Absolute activation magnitude above which an input dimension is treated
    /// as an outlier and kept in FP16. The paper uses 6.0.
    pub threshold: f32,
    /// If `true`, quantize activations row-wise (per token); otherwise use a
    /// single per-tensor scale for the activations.
    pub row_wise_activations: bool,
}

impl Default for LlmInt8Config {
    fn default() -> Self {
        Self {
            threshold: 6.0,
            row_wise_activations: true,
        }
    }
}

impl LlmInt8Config {
    /// Create a new configuration, validating the threshold.
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidConfig`] — non-finite or non-positive threshold.
    pub fn new(threshold: f32, row_wise_activations: bool) -> QuantResult<Self> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(QuantError::InvalidConfig(format!(
                "LLM.int8 threshold must be finite and positive, got {threshold}"
            )));
        }
        Ok(Self {
            threshold,
            row_wise_activations,
        })
    }
}

// ─── Decomposition result ──────────────────────────────────────────────────────

/// Result of an LLM.int8() decomposed matrix multiplication.
#[derive(Debug, Clone)]
pub struct LlmInt8Output {
    /// Layer output `Y`, row-major `(n_samples × n_rows)`.
    pub output: Vec<f32>,
    /// Indices of the detected outlier input dimensions (columns), sorted.
    pub outlier_columns: Vec<usize>,
    /// Fraction of input dimensions classified as outliers.
    pub outlier_fraction: f32,
}

// ─── Quantizer ─────────────────────────────────────────────────────────────────

/// LLM.int8() mixed-precision matrix-multiply engine.
#[derive(Debug, Clone)]
pub struct LlmInt8Quantizer {
    config: LlmInt8Config,
}

impl LlmInt8Quantizer {
    /// Create a new quantizer.
    #[must_use]
    pub fn new(config: LlmInt8Config) -> Self {
        Self { config }
    }

    /// Detect outlier input dimensions from a calibration activation matrix.
    ///
    /// A column `c` is an outlier if `max_r |X[r, c]| ≥ threshold`.
    ///
    /// # Parameters
    ///
    /// * `activations` — row-major `(n_samples × n_cols)`.
    /// * `n_samples`   — number of rows.
    /// * `n_cols`      — number of input dimensions (columns).
    ///
    /// # Returns
    ///
    /// Sorted indices of the outlier columns.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — empty activation slice.
    /// * [`QuantError::DimensionMismatch`] — slice length ≠ `n_samples·n_cols`.
    pub fn detect_outliers(
        &self,
        activations: &[f32],
        n_samples: usize,
        n_cols: usize,
    ) -> QuantResult<Vec<usize>> {
        if activations.is_empty() {
            return Err(QuantError::EmptyInput("LlmInt8::detect_outliers"));
        }
        if activations.len() != n_samples * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_samples * n_cols,
                got: activations.len(),
            });
        }
        let mut col_max = vec![0.0_f32; n_cols];
        for r in 0..n_samples {
            let row = &activations[r * n_cols..r * n_cols + n_cols];
            for (m, &v) in col_max.iter_mut().zip(row.iter()) {
                let a = v.abs();
                if a > *m {
                    *m = a;
                }
            }
        }
        let outliers: Vec<usize> = (0..n_cols)
            .filter(|&c| col_max[c] >= self.config.threshold)
            .collect();
        Ok(outliers)
    }

    /// Perform an LLM.int8() decomposed linear layer: `Y = X · Wᵀ`.
    ///
    /// Regular dimensions are vector-wise INT8 quantized; outlier dimensions
    /// (detected from `x` itself) are kept in FP16 (here, f32) precision.
    ///
    /// # Parameters
    ///
    /// * `x`        — activations, row-major `(n_samples × n_cols)`.
    /// * `w`        — weight matrix `W`, row-major `(n_rows × n_cols)`. Output
    ///   feature `k` of sample `r` is `Σ_c X[r,c]·W[k,c]`.
    /// * `n_samples`, `n_rows`, `n_cols` — matrix dimensions.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — empty `x` or `w`.
    /// * [`QuantError::DimensionMismatch`] — inconsistent slice lengths.
    pub fn matmul(
        &self,
        x: &[f32],
        w: &[f32],
        n_samples: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> QuantResult<LlmInt8Output> {
        if x.is_empty() || w.is_empty() {
            return Err(QuantError::EmptyInput("LlmInt8::matmul"));
        }
        if x.len() != n_samples * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_samples * n_cols,
                got: x.len(),
            });
        }
        if w.len() != n_rows * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows * n_cols,
                got: w.len(),
            });
        }

        // ── 1. Outlier detection ────────────────────────────────────────────
        let outliers = self.detect_outliers(x, n_samples, n_cols)?;
        let mut is_outlier = vec![false; n_cols];
        for &c in &outliers {
            is_outlier[c] = true;
        }
        let outlier_fraction = outliers.len() as f32 / n_cols as f32;

        // ── 2. Per-weight-row INT8 scales over regular columns ──────────────
        // sw[k] = max over regular columns of |W[k, c]| / 127.
        let mut sw = vec![0.0_f32; n_rows];
        for k in 0..n_rows {
            let mut amax = 0.0_f32;
            for c in 0..n_cols {
                if !is_outlier[c] {
                    amax = amax.max(w[k * n_cols + c].abs());
                }
            }
            sw[k] = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        }
        // Quantize regular weight entries to INT8 once.
        let mut w_q = vec![0_i8; n_rows * n_cols];
        for k in 0..n_rows {
            let inv = 1.0 / sw[k];
            for c in 0..n_cols {
                if !is_outlier[c] {
                    let q = (w[k * n_cols + c] * inv).round().clamp(-127.0, 127.0);
                    w_q[k * n_cols + c] = q as i8;
                }
            }
        }

        // ── 3. Per-token activation INT8 scales over regular columns ────────
        let mut output = vec![0.0_f32; n_samples * n_rows];
        for r in 0..n_samples {
            let xr = &x[r * n_cols..r * n_cols + n_cols];

            // Activation row scale (vector-wise, regular columns only).
            let sx = if self.config.row_wise_activations {
                let mut amax = 0.0_f32;
                for (c, &v) in xr.iter().enumerate() {
                    if !is_outlier[c] {
                        amax = amax.max(v.abs());
                    }
                }
                if amax > 0.0 { amax / 127.0 } else { 1.0 }
            } else {
                // Per-tensor scale already absorbed into a constant; recompute
                // a global activation scale once (cheap enough for CPU).
                let mut amax = 0.0_f32;
                for (i, &v) in x.iter().enumerate() {
                    let c = i % n_cols;
                    if !is_outlier[c] {
                        amax = amax.max(v.abs());
                    }
                }
                if amax > 0.0 { amax / 127.0 } else { 1.0 }
            };
            let inv_sx = 1.0 / sx;

            // Quantize this activation row's regular entries.
            let mut x_q = vec![0_i32; n_cols];
            for (c, &v) in xr.iter().enumerate() {
                if !is_outlier[c] {
                    x_q[c] = (v * inv_sx).round().clamp(-127.0, 127.0) as i32;
                }
            }

            for k in 0..n_rows {
                // INT8 path: accumulate in i32, rescale by sx·sw[k].
                let mut acc_int: i64 = 0;
                // FP16 (here f32) path: outlier columns at full precision.
                let mut acc_fp: f32 = 0.0;
                for c in 0..n_cols {
                    if is_outlier[c] {
                        acc_fp += xr[c] * w[k * n_cols + c];
                    } else {
                        acc_int += x_q[c] as i64 * w_q[k * n_cols + c] as i64;
                    }
                }
                output[r * n_rows + k] = acc_int as f32 * (sx * sw[k]) + acc_fp;
            }
        }

        Ok(LlmInt8Output {
            output,
            outlier_columns: outliers,
            outlier_fraction,
        })
    }

    /// Reference full-precision matmul `Y = X · Wᵀ` (for error comparison).
    #[must_use]
    pub fn matmul_fp32(
        x: &[f32],
        w: &[f32],
        n_samples: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; n_samples * n_rows];
        for r in 0..n_samples {
            for k in 0..n_rows {
                let mut acc = 0.0_f32;
                for c in 0..n_cols {
                    acc += x[r * n_cols + c] * w[k * n_cols + c];
                }
                y[r * n_rows + k] = acc;
            }
        }
        y
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG pseudo-random f32 in `[-scale, scale]`.
    fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 32) as u32;
                let f = (bits as f32) / (u32::MAX as f32);
                (f * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().max(1) as f32;
        (a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / n)
            .sqrt()
    }

    // ── Config ─────────────────────────────────────────────────────────────────

    #[test]
    fn default_config_sane() {
        let cfg = LlmInt8Config::default();
        assert_eq!(cfg.threshold, 6.0);
        assert!(cfg.row_wise_activations);
    }

    #[test]
    fn config_rejects_bad_threshold() {
        assert!(LlmInt8Config::new(-1.0, true).is_err());
        assert!(LlmInt8Config::new(0.0, true).is_err());
        assert!(LlmInt8Config::new(f32::NAN, true).is_err());
        assert!(LlmInt8Config::new(6.0, true).is_ok());
    }

    // ── Outlier detection ──────────────────────────────────────────────────────

    #[test]
    fn detects_outlier_column() {
        // 4 samples × 5 cols; column 2 has a big value.
        let mut x = lcg(4 * 5, 1, 1.0); // all in [-1, 1]
        x[2] = 9.0; // sample 0, column 2 is an outlier
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let out = q.detect_outliers(&x, 4, 5).expect("detect");
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn no_outliers_when_below_threshold() {
        let x = lcg(4 * 5, 2, 1.0); // all small
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let out = q.detect_outliers(&x, 4, 5).expect("detect");
        assert!(out.is_empty());
    }

    #[test]
    fn detect_outliers_dimension_error() {
        let x = vec![1.0_f32; 10];
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        assert!(matches!(
            q.detect_outliers(&x, 3, 5),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    // ── Decomposed matmul ──────────────────────────────────────────────────────

    #[test]
    fn matmul_matches_fp32_when_no_outliers() {
        // Without outliers it is pure vector-wise INT8 — should be close to FP32.
        let n_samples = 6;
        let n_rows = 4;
        let n_cols = 8;
        let x = lcg(n_samples * n_cols, 10, 1.0);
        let w = lcg(n_rows * n_cols, 20, 0.5);
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let out = q.matmul(&x, &w, n_samples, n_rows, n_cols).expect("matmul");
        assert!(out.outlier_columns.is_empty());
        let reference = LlmInt8Quantizer::matmul_fp32(&x, &w, n_samples, n_rows, n_cols);
        let err = rmse(&out.output, &reference);
        let scale = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(err < 0.05 * scale.max(1e-6), "INT8 rmse {err} too high");
    }

    #[test]
    fn decomposition_beats_naive_int8_on_outliers() {
        // Construct activations with a strong outlier column, then show the
        // LLM.int8() decomposition reconstructs the output far better than a
        // naïve per-tensor INT8 matmul that quantizes the outlier too.
        let n_samples = 8;
        let n_rows = 5;
        let n_cols = 16;
        let mut x = lcg(n_samples * n_cols, 30, 1.0);
        // Inject a large outlier in column 7 for every sample.
        for r in 0..n_samples {
            x[r * n_cols + 7] = 40.0 + (r as f32);
        }
        let w = lcg(n_rows * n_cols, 40, 0.5);

        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let out = q.matmul(&x, &w, n_samples, n_rows, n_cols).expect("matmul");
        assert!(
            out.outlier_columns.contains(&7),
            "column 7 must be flagged as outlier"
        );

        let reference = LlmInt8Quantizer::matmul_fp32(&x, &w, n_samples, n_rows, n_cols);
        let err_decomp = rmse(&out.output, &reference);

        // Naïve INT8: per-tensor activation scale dominated by the outlier.
        let amax_x = x.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        let sx = amax_x / 127.0;
        let amax_w = w.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        let sw = amax_w / 127.0;
        let mut naive = vec![0.0_f32; n_samples * n_rows];
        for r in 0..n_samples {
            for k in 0..n_rows {
                let mut acc: i64 = 0;
                for c in 0..n_cols {
                    let xq = (x[r * n_cols + c] / sx).round().clamp(-127.0, 127.0) as i64;
                    let wq = (w[k * n_cols + c] / sw).round().clamp(-127.0, 127.0) as i64;
                    acc += xq * wq;
                }
                naive[r * n_rows + k] = acc as f32 * sx * sw;
            }
        }
        let err_naive = rmse(&naive, &reference);

        assert!(
            err_decomp < err_naive,
            "decomposition rmse {err_decomp} should beat naïve INT8 {err_naive}"
        );
    }

    #[test]
    fn outlier_fraction_reported() {
        let n_samples = 4;
        let n_rows = 3;
        let n_cols = 10;
        let mut x = lcg(n_samples * n_cols, 50, 1.0);
        x[1] = 20.0;
        x[5] = 20.0;
        let w = lcg(n_rows * n_cols, 60, 0.5);
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let out = q.matmul(&x, &w, n_samples, n_rows, n_cols).expect("matmul");
        assert!((out.outlier_fraction - 0.2).abs() < 1e-6);
    }

    #[test]
    fn matmul_dimension_errors() {
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        let x = vec![1.0_f32; 6];
        let w = vec![1.0_f32; 6];
        assert!(matches!(
            q.matmul(&x, &w, 2, 2, 5),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn matmul_empty_errors() {
        let q = LlmInt8Quantizer::new(LlmInt8Config::default());
        assert!(matches!(
            q.matmul(&[], &[1.0], 0, 1, 1),
            Err(QuantError::EmptyInput(_))
        ));
    }
}
