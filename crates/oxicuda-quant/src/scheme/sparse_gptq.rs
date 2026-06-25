//! # Sparse-GPTQ — Joint Pruning + Quantization via OBC
//!
//! Frantar & Alistarh (2023): "SparseGPT: Massive Language Models Can Be
//! Accurately Pruned in One-Shot" <https://arxiv.org/abs/2301.00774>
//!
//! Sparse-GPTQ unifies the Optimal Brain Quantization (OBC) error-compensation
//! machinery of GPTQ with magnitude pruning, so that a layer can be **pruned
//! and quantized in a single pass** while minimizing the second-order
//! layer-output error.
//!
//! ## Algorithm (column-wise OBC with pruning)
//!
//! Using the layer Hessian `H = 2 XᵀX` and its inverse Cholesky factor:
//!
//! ```text
//! H_d  = H + λ I
//! H⁻¹  = (chol(H_d))⁻¹ machinery   (same as GPTQ)
//!
//! for j = 0 .. n_cols:
//!     # 1. Decide which rows to prune in this column using the OBC saliency
//!     #    metric  s_i = W[i,j]² / H⁻¹[j,j]  (cost of removing the weight).
//!     prune lowest-saliency fraction → set those W[i,j] = 0.
//!     # 2. Quantize the survivors; pruned entries contribute their *full*
//!     #    value as error (since 0 − W[i,j] = −W[i,j]).
//!     err[i] = (W[i,j] − dequant_or_0[i]) / H⁻¹[j,j]
//!     # 3. Propagate the combined error to all later columns.
//!     W[:, j+1:] −= outer(err, H⁻¹[j, j+1:])
//! ```
//!
//! Pruning may be driven either by a caller-supplied boolean mask (one entry
//! per weight, `true` = keep) or by a target *unstructured sparsity* ratio,
//! in which case Sparse-GPTQ selects the per-column victims itself using the
//! Hessian-aware saliency metric.
//!
//! All linear algebra is performed in `f32` on flat slices; the module is fully
//! CPU-verifiable.

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Pruning strategy for [`SparseGptqQuantizer`].
#[derive(Debug, Clone)]
pub enum SparsityTarget {
    /// Prune a fixed fraction of weights per column using OBC saliency.
    ///
    /// The value is in `[0, 1)`; e.g. `0.5` yields 50 % unstructured sparsity.
    Unstructured(f32),
    /// Prune using a caller-supplied boolean keep-mask, row-major
    /// `(n_rows × n_cols)`; `true` keeps the weight, `false` prunes it.
    Mask(Vec<bool>),
}

/// Configuration for Sparse-GPTQ.
#[derive(Debug, Clone)]
pub struct SparseGptqConfig {
    /// Quantization bit-width (2, 3, 4, or 8 typical).
    pub bits: u32,
    /// Relative Hessian dampening: λ = `percdamp` × mean(diag H).
    pub percdamp: f64,
    /// Symmetric (zero-point = 0) vs asymmetric quantization.
    pub symmetric: bool,
    /// Pruning target.
    pub sparsity: SparsityTarget,
}

impl Default for SparseGptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            percdamp: 0.01,
            symmetric: true,
            sparsity: SparsityTarget::Unstructured(0.5),
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Output of Sparse-GPTQ: pruned-and-quantized weight matrix.
#[derive(Debug, Clone)]
pub struct SparseGptqOutput {
    /// Integer codes, row-major `(n_rows × n_cols)`. Pruned positions hold the
    /// integer that dequantizes to 0 (`zero_point`).
    pub quantized: Vec<i32>,
    /// Boolean keep-mask actually applied (row-major `(n_rows × n_cols)`).
    pub mask: Vec<bool>,
    /// Per-column scales.
    pub scales: Vec<f32>,
    /// Per-column zero-points (0 for symmetric).
    pub zero_points: Vec<i32>,
    /// Row count.
    pub n_rows: usize,
    /// Column count.
    pub n_cols: usize,
}

impl SparseGptqOutput {
    /// Dequantize to `f32`, forcing pruned positions to exactly 0.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                let idx = i * self.n_cols + j;
                if self.mask[idx] {
                    let q = self.quantized[idx] as f32;
                    out[idx] = (q - self.zero_points[j] as f32) * self.scales[j];
                }
            }
        }
        out
    }

    /// Achieved unstructured sparsity (fraction of pruned weights).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let pruned = self.mask.iter().filter(|&&k| !k).count();
        pruned as f32 / self.mask.len().max(1) as f32
    }

    /// Mean squared reconstruction error against the original dense weights.
    #[must_use]
    pub fn reconstruction_mse(&self, original: &[f32]) -> f32 {
        let deq = self.dequantize();
        let n = deq.len().max(1) as f32;
        deq.iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n
    }
}

// ─── Quantizer ─────────────────────────────────────────────────────────────────

/// Sparse-GPTQ quantizer combining pruning and quantization through OBC.
#[derive(Debug, Clone)]
pub struct SparseGptqQuantizer {
    config: SparseGptqConfig,
}

impl SparseGptqQuantizer {
    /// Create a new quantizer.
    #[must_use]
    pub fn new(config: SparseGptqConfig) -> Self {
        Self { config }
    }

    /// Jointly prune and quantize a weight matrix using its layer Hessian.
    ///
    /// # Parameters
    ///
    /// * `weights`  — row-major `(n_rows × n_cols)`.
    /// * `n_rows`, `n_cols` — matrix dimensions.
    /// * `hessian`  — row-major symmetric PSD `(n_cols × n_cols)`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — empty weights.
    /// * [`QuantError::DimensionMismatch`] — inconsistent slice/mask lengths.
    /// * [`QuantError::InvalidBitWidth`] — `bits` is 0 or > 16.
    /// * [`QuantError::InvalidConfig`] — sparsity fraction outside `[0, 1)`.
    /// * [`QuantError::SingularHessian`] — H not positive definite after damping.
    pub fn quantize_layer(
        &self,
        weights: &[f32],
        n_rows: usize,
        n_cols: usize,
        hessian: &[f32],
    ) -> QuantResult<SparseGptqOutput> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("SparseGptq::quantize_layer"));
        }
        if weights.len() != n_rows * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_rows * n_cols,
                got: weights.len(),
            });
        }
        if hessian.len() != n_cols * n_cols {
            return Err(QuantError::DimensionMismatch {
                expected: n_cols * n_cols,
                got: hessian.len(),
            });
        }
        let bits = self.config.bits;
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { bits });
        }

        // Resolve the pruning specification.
        let prune_frac = match &self.config.sparsity {
            SparsityTarget::Unstructured(f) => {
                if !(0.0..1.0).contains(f) || !f.is_finite() {
                    return Err(QuantError::InvalidConfig(format!(
                        "sparsity fraction must be in [0, 1), got {f}"
                    )));
                }
                Some(*f)
            }
            SparsityTarget::Mask(m) => {
                if m.len() != n_rows * n_cols {
                    return Err(QuantError::DimensionMismatch {
                        expected: n_rows * n_cols,
                        got: m.len(),
                    });
                }
                None
            }
        };

        // ── Dampen + Cholesky ────────────────────────────────────────────────
        let mean_diag = (0..n_cols).map(|j| hessian[j * n_cols + j]).sum::<f32>() / n_cols as f32;
        let lambda = (self.config.percdamp as f32) * mean_diag.abs().max(1e-8);
        let mut h_damp = hessian.to_vec();
        for j in 0..n_cols {
            h_damp[j * n_cols + j] += lambda;
        }
        let l = cholesky_lower(&h_damp, n_cols).ok_or_else(|| {
            let min_diag = (0..n_cols)
                .map(|j| h_damp[j * n_cols + j])
                .fold(f32::INFINITY, f32::min);
            QuantError::SingularHessian { min_diag }
        })?;
        let l_inv = invert_lower(&l, n_cols);

        // ── Column-wise OBC with pruning ─────────────────────────────────────
        let (q_min, q_max) = quant_range(bits, self.config.symmetric);
        let mut w = weights.to_vec();
        let mut quantized = vec![0_i32; n_rows * n_cols];
        let mut mask = vec![true; n_rows * n_cols];
        let mut scales = vec![0.0_f32; n_cols];
        let mut zero_points = vec![0_i32; n_cols];

        for j in 0..n_cols {
            let (scale_j, zp_j) = col_quant_params(
                &w,
                n_rows,
                n_cols,
                j,
                bits,
                self.config.symmetric,
                q_min,
                q_max,
            );
            scales[j] = scale_j;
            zero_points[j] = zp_j;

            let hinv_j = hinv_row_starting(&l_inv, n_cols, j);
            let hinv_jj = hinv_j[0].max(1e-12);

            // Decide pruned rows for this column.
            let prune_this = if let Some(frac) = prune_frac {
                prune_by_saliency(&w, n_rows, n_cols, j, hinv_jj, frac)
            } else {
                // Mask supplied by caller: keep == !prune.
                let m = match &self.config.sparsity {
                    SparsityTarget::Mask(m) => m,
                    SparsityTarget::Unstructured(_) => unreachable!(),
                };
                (0..n_rows).map(|i| !m[i * n_cols + j]).collect()
            };

            let mut errs = vec![0.0_f32; n_rows];
            for i in 0..n_rows {
                let idx = i * n_cols + j;
                let wij = w[idx];
                if prune_this[i] {
                    // Pruned → reconstructs to 0; full value becomes error.
                    mask[idx] = false;
                    quantized[idx] = zp_j; // dequantizes to 0
                    errs[i] = wij / hinv_jj;
                } else {
                    let q = quant_scalar(wij, scale_j, zp_j, q_min, q_max);
                    quantized[idx] = q;
                    let q_dq = (q as f32 - zp_j as f32) * scale_j;
                    errs[i] = (wij - q_dq) / hinv_jj;
                }
            }

            // OBC error propagation to later columns.
            for (dk, k) in ((j + 1)..n_cols).enumerate() {
                let hinv_jk = hinv_j[dk + 1];
                for i in 0..n_rows {
                    w[i * n_cols + k] -= errs[i] * hinv_jk;
                }
            }
        }

        Ok(SparseGptqOutput {
            quantized,
            mask,
            scales,
            zero_points,
            n_rows,
            n_cols,
        })
    }
}

// ─── Private numeric helpers (self-contained) ──────────────────────────────────

/// Choose which rows of column `j` to prune using OBC saliency
/// `s_i = W[i,j]² / H⁻¹[j,j]`. The `frac` lowest-saliency entries are pruned.
fn prune_by_saliency(
    w: &[f32],
    n_rows: usize,
    n_cols: usize,
    j: usize,
    hinv_jj: f32,
    frac: f32,
) -> Vec<bool> {
    let n_prune = ((n_rows as f32) * frac).round() as usize;
    let mut prune = vec![false; n_rows];
    if n_prune == 0 {
        return prune;
    }
    if n_prune >= n_rows {
        return vec![true; n_rows];
    }
    // saliency = (W²)/Hinv_jj; smaller = cheaper to remove.
    let mut sal: Vec<(f32, usize)> = (0..n_rows)
        .map(|i| {
            let wij = w[i * n_cols + j];
            (wij * wij / hinv_jj, i)
        })
        .collect();
    // Partial selection of the `n_prune` smallest saliencies.
    sal.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, i) in sal.iter().take(n_prune) {
        prune[i] = true;
    }
    prune
}

/// Integer quantization range for `bits` bits.
fn quant_range(bits: u32, symmetric: bool) -> (i32, i32) {
    if symmetric {
        let half = 1i32 << (bits - 1);
        (-half, half - 1)
    } else {
        (0i32, (1i32 << bits) - 1)
    }
}

/// Round-to-nearest quantization of a single scalar.
fn quant_scalar(x: f32, scale: f32, zp: i32, q_min: i32, q_max: i32) -> i32 {
    (x / scale + zp as f32)
        .round()
        .clamp(q_min as f32, q_max as f32) as i32
}

/// Per-column quantization scale and zero-point.
fn col_quant_params(
    w: &[f32],
    n_rows: usize,
    n_cols: usize,
    j: usize,
    bits: u32,
    symmetric: bool,
    q_min: i32,
    q_max: i32,
) -> (f32, i32) {
    if symmetric {
        let abs_max = (0..n_rows)
            .map(|i| w[i * n_cols + j].abs())
            .fold(0.0_f32, f32::max)
            .max(1e-8);
        (abs_max / q_max as f32, 0)
    } else {
        let fmin = (0..n_rows)
            .map(|i| w[i * n_cols + j])
            .fold(f32::INFINITY, f32::min);
        let fmax = (0..n_rows)
            .map(|i| w[i * n_cols + j])
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (fmax - fmin).max(1e-8);
        let scale = range / ((1i32 << bits) - 1) as f32;
        let zp = (-fmin / scale).round().clamp(q_min as f32, q_max as f32) as i32;
        (scale, zp)
    }
}

/// Cholesky decomposition `H = L Lᵀ` (lower triangular). `None` if not PD.
fn cholesky_lower(h: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = h[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Invert a lower-triangular matrix via forward substitution.
fn invert_lower(l: &[f32], n: usize) -> Vec<f32> {
    let mut li = vec![0.0_f32; n * n];
    for i in 0..n {
        li[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0.0_f32;
            for k in j..i {
                s += l[i * n + k] * li[k * n + j];
            }
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    li
}

/// `H⁻¹[j, k]` for `k = j .. n` from the lower Cholesky inverse.
fn hinv_row_starting(l_inv: &[f32], n: usize, j: usize) -> Vec<f32> {
    (j..n)
        .map(|k| (k..n).map(|m| l_inv[m * n + j] * l_inv[m * n + k]).sum())
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(n: usize) -> Vec<f32> {
        let mut h = vec![0.0_f32; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        h
    }

    /// LCG pseudo-random f32 in `[-1, 1]`.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 32) as u32;
                let f = (bits as f32) / (u32::MAX as f32);
                f * 2.0 - 1.0
            })
            .collect()
    }

    // ── Config ─────────────────────────────────────────────────────────────────

    #[test]
    fn default_config_sane() {
        let cfg = SparseGptqConfig::default();
        assert_eq!(cfg.bits, 4);
        assert!(cfg.symmetric);
        match cfg.sparsity {
            SparsityTarget::Unstructured(f) => assert!((f - 0.5).abs() < 1e-6),
            SparsityTarget::Mask(_) => panic!("default should be unstructured"),
        }
    }

    // ── Sparsity achievement ───────────────────────────────────────────────────

    #[test]
    fn achieves_requested_sparsity() {
        let n_rows = 8;
        let n_cols = 8;
        let w = lcg(n_rows * n_cols, 1);
        let h = eye(n_cols);
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Unstructured(0.5),
            ..SparseGptqConfig::default()
        };
        let q = SparseGptqQuantizer::new(cfg);
        let out = q.quantize_layer(&w, n_rows, n_cols, &h).expect("quantize");
        // Per-column 50 % of 8 rows = exactly 4 pruned per column.
        assert!(
            (out.sparsity() - 0.5).abs() < 1e-6,
            "sparsity {}",
            out.sparsity()
        );
    }

    #[test]
    fn pruned_positions_dequantize_to_zero() {
        let n_rows = 6;
        let n_cols = 6;
        let w = lcg(n_rows * n_cols, 2);
        let h = eye(n_cols);
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Unstructured(0.5),
            bits: 8,
            ..SparseGptqConfig::default()
        };
        let out = SparseGptqQuantizer::new(cfg)
            .quantize_layer(&w, n_rows, n_cols, &h)
            .expect("quantize");
        let deq = out.dequantize();
        for (idx, &keep) in out.mask.iter().enumerate() {
            if !keep {
                assert_eq!(deq[idx], 0.0, "pruned position {idx} must be 0");
            }
        }
    }

    // ── Error compensation works ───────────────────────────────────────────────

    #[test]
    fn obc_beats_naive_prune_then_quant() {
        // Sparse-GPTQ's error propagation should beat a naïve baseline that
        // simply prunes the smallest weights and round-to-nearest quantizes the
        // rest, when measured by *layer-output* error YᵀY with Y = W X.
        let n_rows = 6;
        let n_cols = 8;
        let n_samples = 32;
        let w = lcg(n_rows * n_cols, 3);
        // Build a correlated activation matrix X (n_cols × n_samples).
        let x = lcg(n_cols * n_samples, 4);
        // Hessian H = X Xᵀ (n_cols × n_cols).
        let mut h = vec![0.0_f32; n_cols * n_cols];
        for a in 0..n_cols {
            for b in 0..n_cols {
                let mut s = 0.0_f32;
                for t in 0..n_samples {
                    s += x[a * n_samples + t] * x[b * n_samples + t];
                }
                h[a * n_cols + b] = s;
            }
        }

        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Unstructured(0.5),
            bits: 4,
            symmetric: true,
            ..SparseGptqConfig::default()
        };
        let out = SparseGptqQuantizer::new(cfg)
            .quantize_layer(&w, n_rows, n_cols, &h)
            .expect("quantize");
        let w_sparse = out.dequantize();

        // Naïve baseline: same mask, no error compensation.
        let mut w_naive = vec![0.0_f32; n_rows * n_cols];
        for i in 0..n_rows {
            // per-column symmetric INT4 scale from original weights
            for j in 0..n_cols {
                let idx = i * n_cols + j;
                if out.mask[idx] {
                    let scale = out.scales[j].max(1e-8);
                    let q = (w[idx] / scale).round().clamp(-8.0, 7.0);
                    w_naive[idx] = q * scale;
                }
            }
        }

        // Output error E = (W_hat − W) X, measured in Frobenius norm.
        let out_err = |what: &[f32]| -> f32 {
            let mut e = 0.0_f32;
            for i in 0..n_rows {
                for t in 0..n_samples {
                    let mut d = 0.0_f32;
                    for j in 0..n_cols {
                        d += (what[i * n_cols + j] - w[i * n_cols + j]) * x[j * n_samples + t];
                    }
                    e += d * d;
                }
            }
            e
        };
        let e_sparse = out_err(&w_sparse);
        let e_naive = out_err(&w_naive);
        assert!(
            e_sparse <= e_naive,
            "Sparse-GPTQ output err {e_sparse} should be ≤ naïve {e_naive}"
        );
    }

    // ── Caller-supplied mask ───────────────────────────────────────────────────

    #[test]
    fn respects_supplied_mask() {
        let n_rows = 4;
        let n_cols = 4;
        let w = lcg(n_rows * n_cols, 5);
        let h = eye(n_cols);
        // Keep mask: prune the entire first column.
        let mut keep = vec![true; n_rows * n_cols];
        for i in 0..n_rows {
            keep[i * n_cols] = false;
        }
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Mask(keep),
            bits: 8,
            ..SparseGptqConfig::default()
        };
        let out = SparseGptqQuantizer::new(cfg)
            .quantize_layer(&w, n_rows, n_cols, &h)
            .expect("quantize");
        let deq = out.dequantize();
        for i in 0..n_rows {
            assert_eq!(deq[i * n_cols], 0.0, "column 0 must be pruned");
            assert!(out.mask[i * n_cols + 1], "column 1 must be kept");
        }
    }

    #[test]
    fn mask_wrong_length_errors() {
        let n_rows = 4;
        let n_cols = 4;
        let w = lcg(n_rows * n_cols, 6);
        let h = eye(n_cols);
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Mask(vec![true; 10]),
            ..SparseGptqConfig::default()
        };
        assert!(matches!(
            SparseGptqQuantizer::new(cfg).quantize_layer(&w, n_rows, n_cols, &h),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    // ── Validation ─────────────────────────────────────────────────────────────

    #[test]
    fn invalid_sparsity_fraction_errors() {
        let w = lcg(4, 7);
        let h = eye(2);
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Unstructured(1.5),
            ..SparseGptqConfig::default()
        };
        assert!(matches!(
            SparseGptqQuantizer::new(cfg).quantize_layer(&w, 2, 2, &h),
            Err(QuantError::InvalidConfig(_))
        ));
    }

    #[test]
    fn singular_hessian_errors() {
        let w = lcg(4, 8);
        let h = vec![0.0_f32; 4];
        let cfg = SparseGptqConfig {
            percdamp: 0.0,
            ..SparseGptqConfig::default()
        };
        assert!(matches!(
            SparseGptqQuantizer::new(cfg).quantize_layer(&w, 2, 2, &h),
            Err(QuantError::SingularHessian { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_weight() {
        let cfg = SparseGptqConfig::default();
        let h = eye(4);
        let w = vec![0.5_f32; 3];
        assert!(matches!(
            SparseGptqQuantizer::new(cfg).quantize_layer(&w, 1, 4, &h),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn zero_sparsity_keeps_everything() {
        let n_rows = 4;
        let n_cols = 4;
        let w = lcg(n_rows * n_cols, 9);
        let h = eye(n_cols);
        let cfg = SparseGptqConfig {
            sparsity: SparsityTarget::Unstructured(0.0),
            bits: 8,
            ..SparseGptqConfig::default()
        };
        let out = SparseGptqQuantizer::new(cfg)
            .quantize_layer(&w, n_rows, n_cols, &h)
            .expect("quantize");
        assert_eq!(out.sparsity(), 0.0);
        assert!(out.mask.iter().all(|&k| k));
    }
}
