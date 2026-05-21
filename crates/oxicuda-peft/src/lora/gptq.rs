//! GPTQ — Accurate Post-Training Quantization for Generative Pre-trained Transformers.
//!
//! Reference: Frantar E, Ashkboos S, Hoefler T, Alistarh D (2023)
//! "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
//! Transformers", ICLR 2023. <https://arxiv.org/abs/2210.17323>
//!
//! GPTQ is a one-shot, layer-wise weight-only quantization scheme derived from
//! Optimal Brain Surgeon (OBS). Given a weight matrix `W ∈ ℝ^{rows × cols}` and a
//! second-moment proxy `H = X' X` (here taken in diagonal form as `xtx_diag`), the
//! algorithm:
//!
//! 1. Damps the Hessian: `H[j, j] += λ · mean(xtx_diag)` where `λ = damp_percent`.
//! 2. Computes the upper-triangular Cholesky factor of `H⁻¹`.
//! 3. Walks the columns left-to-right in blocks of `blocksize`, picking per-group
//!    `(scale, zero)` pairs from min/max and re-distributing the quantization
//!    error to the remaining unquantized columns through the Cholesky factor.
//! 4. Optionally permutes columns by descending diagonal magnitude (`act_order`)
//!    before the sweep and inverts the permutation at the end.
//!
//! Because only the diagonal of `H` is supplied, the off-block columns of
//! `Hinv_chol` are zero and step (3)'s cross-block update is a no-op. The
//! generalised loop body is kept in place so the same module can be re-used
//! with a full Hessian in the future. All arithmetic in the inner loop runs in
//! `f64`; the final `(scale, zero)` codes are cast back to `f32` for storage.

use crate::error::{PeftError, PeftResult};

/// Configuration for GPTQ activation-aware quantization.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    /// Bits per quantized value; must be 2, 3, 4, or 8.
    pub bits: u8,
    /// Number of consecutive columns sharing one `(scale, zero)` pair.
    pub group_size: usize,
    /// Damping percentage added to the Hessian diagonal (relative to its mean).
    pub damp_percent: f64,
    /// Whether to permute columns by descending diagonal magnitude.
    pub act_order: bool,
    /// Number of columns processed before flushing the cross-block error update.
    pub blocksize: usize,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            damp_percent: 0.01,
            act_order: false,
            blocksize: 128,
        }
    }
}

/// Output of [`Gptq::quantize_weight`].
///
/// When `perm` is `Some(p)`, the quantizer ran in `act_order` mode and the
/// per-group `(scale, zero)` entries are indexed by the *permuted* column
/// position. The vector `p` maps permuted-column index → original-column
/// index, so each original column `j` looks up its group via
/// `p.iter().position(|&old| old == j)`. The [`Gptq::dequantize`] helper
/// handles this lookup transparently.
#[derive(Debug, Clone)]
pub struct GptqQuantized {
    /// Integer codes in `[0, 2^bits − 1]`, row-major `rows × cols`.
    pub q: Vec<i32>,
    /// Per-group affine scale (in permuted-column order if `perm` is `Some`).
    pub scale: Vec<f32>,
    /// Per-group affine zero (continuous, in dequantized space).
    pub zero: Vec<f32>,
    /// Bits per quantized value.
    pub bits: u8,
    /// Group size used by the quantizer.
    pub group_size: usize,
    /// Original `(rows, cols)` shape of the weight matrix.
    pub original_shape: (usize, usize),
    /// Optional activation-order permutation: `perm[k] = old column index of the k-th permuted column`.
    pub perm: Option<Vec<usize>>,
}

/// GPTQ algorithm namespace.
pub struct Gptq;

impl Gptq {
    /// Quantize `w` (row-major `rows × cols`) using GPTQ with the diagonal Hessian
    /// proxy `xtx_diag` (length `cols`).
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] for any configuration or dimension
    /// violation (bits not in {2, 3, 4, 8}, non-positive group size /
    /// blocksize / damp percent, zero-sized matrix, or length mismatches).
    pub fn quantize_weight(
        w: &[f32],
        rows: usize,
        cols: usize,
        xtx_diag: &[f32],
        cfg: &GptqConfig,
    ) -> PeftResult<GptqQuantized> {
        validate(w, rows, cols, xtx_diag, cfg)?;

        // Clamp group_size to cols so a single oversized group is honoured.
        let group_size = cfg.group_size.min(cols);
        let n_groups = cols.div_ceil(group_size);
        let q_max = (1_i32 << cfg.bits) - 1;
        let q_max_f = q_max as f64;
        let blocksize = cfg.blocksize.min(cols);

        // Build a working copy of W in f64 for numerically stable error propagation.
        let mut w_work = vec![0.0_f64; rows * cols];
        for (slot, &v) in w_work.iter_mut().zip(w.iter()) {
            *slot = v as f64;
        }

        // Activation ordering (optional).
        let perm: Vec<usize> = if cfg.act_order {
            permutation_by_descending(xtx_diag)
        } else {
            (0..cols).collect()
        };

        // Damp the diagonal: H[j, j] = xtx_diag[j] + damp_percent · mean(xtx_diag).
        let mean_diag = mean_f64(xtx_diag);
        let damp_amt = cfg.damp_percent * mean_diag;
        let mut h_diag = vec![0.0_f64; cols];
        for j in 0..cols {
            let raw = xtx_diag[perm[j]] as f64;
            h_diag[j] = (raw + damp_amt).max(f64::EPSILON);
        }

        // If act_order is on we also need W in the permuted column order.
        let w_perm = if cfg.act_order {
            permute_columns(&w_work, rows, cols, &perm)
        } else {
            w_work
        };
        let mut w_work = w_perm;

        // Upper-triangular Cholesky factor of H⁻¹.
        // With diagonal H, the result is diag(1 / sqrt(H[j, j])) and the off-diagonals are zero.
        // We still store the full upper triangle so the inner loop can transparently support a
        // dense Hessian in future versions.
        let hinv_chol = cholesky_of_inverse_upper(&h_diag)?;

        // Group state (scale, zero) accumulators, allocated up-front.
        let mut scale = vec![0.0_f32; n_groups];
        let mut zero = vec![0.0_f32; n_groups];
        // Track whether each group has had its (scale, zero) initialised yet.
        let mut group_init = vec![false; n_groups];

        // Quantized codes in the permuted column order, row-major.
        let mut q_perm = vec![0_i32; rows * cols];

        // Walk blocks of `blocksize` columns left-to-right.
        let mut col_start = 0_usize;
        while col_start < cols {
            let col_end = (col_start + blocksize).min(cols);
            // Per-column local block buffer (already lives inside w_work — we modify in place).
            for j_global in col_start..col_end {
                let g = (j_global / group_size).min(n_groups - 1);

                // Initialise (scale, zero) for group `g` if not yet done. Scan the
                // entire group range so the affine grid bounds every column in
                // the group, not just the first.
                if !group_init[g] {
                    let g_start = g * group_size;
                    let g_end = (g_start + group_size).min(cols);
                    let (lo, hi) = group_min_max(&w_work, rows, cols, g_start, g_end);
                    let span = (hi - lo).max(f64::EPSILON);
                    let s = span / q_max_f;
                    scale[g] = s as f32;
                    zero[g] = lo as f32;
                    group_init[g] = true;
                }

                let s = scale[g] as f64;
                let z = zero[g] as f64;
                let inv_s = if s.abs() > f64::EPSILON { 1.0 / s } else { 0.0 };
                let d = hinv_chol[j_global * cols + j_global].max(f64::EPSILON);

                for i in 0..rows {
                    let w_ij = w_work[i * cols + j_global];
                    let code = ((w_ij - z) * inv_s).round();
                    let code_i = clamp_i64_to_code(code as i64, q_max);
                    q_perm[i * cols + j_global] = code_i;
                    let wq = s * (code_i as f64) + z;
                    let err = (w_ij - wq) / d;

                    // Propagate intra-block to remaining columns of the block.
                    for k in (j_global + 1)..col_end {
                        w_work[i * cols + k] -= err * hinv_chol[j_global * cols + k];
                    }
                    // Stash residual back so cross-block update can pick it up.
                    // We re-use w_work[i, j_global] = err so a later sweep can access it.
                    w_work[i * cols + j_global] = err;
                }
            }

            // Cross-block error propagation to columns past the current block.
            // With a diagonal Hessian Hinv_chol[j, k] = 0 for j != k, so this is a no-op.
            // Written generally so a full-Hessian variant can re-use the routine.
            if col_end < cols {
                for j_global in col_start..col_end {
                    let row_offset = j_global * cols;
                    for k in col_end..cols {
                        let coeff = hinv_chol[row_offset + k];
                        if coeff == 0.0 {
                            continue;
                        }
                        for i in 0..rows {
                            let err_i = w_work[i * cols + j_global];
                            w_work[i * cols + k] -= err_i * coeff;
                        }
                    }
                }
            }

            col_start = col_end;
        }

        // Codes are stored in *original* column order so end-users see a
        // standard row-major `rows × cols` layout. The per-group scale/zero
        // remain in permuted-column order and are paired with the saved
        // permutation so dequant can resolve the correct group per column.
        let (q_out, perm_out) = if cfg.act_order {
            (unpermute_codes(&q_perm, &perm, rows, cols), Some(perm))
        } else {
            (q_perm, None)
        };

        Ok(GptqQuantized {
            q: q_out,
            scale,
            zero,
            bits: cfg.bits,
            group_size,
            original_shape: (rows, cols),
            perm: perm_out,
        })
    }

    /// Dequantize a [`GptqQuantized`] back into a row-major `rows × cols` `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if the stored state is internally
    /// inconsistent (`group_size == 0`, missing group entries, code count
    /// disagreement, …).
    pub fn dequantize(q: &GptqQuantized) -> PeftResult<Vec<f32>> {
        let (rows, cols) = q.original_shape;
        if q.group_size == 0 {
            return Err(PeftError::Internal {
                msg: "GPTQ stored group_size is zero".to_string(),
            });
        }
        let expected = rows * cols;
        if q.q.len() != expected {
            return Err(PeftError::Internal {
                msg: format!("GPTQ q length {} != rows*cols {}", q.q.len(), expected),
            });
        }
        let n_groups = cols.div_ceil(q.group_size);
        if q.scale.len() != n_groups || q.zero.len() != n_groups {
            return Err(PeftError::Internal {
                msg: format!(
                    "GPTQ group meta length mismatch (expected {n_groups}, got scale={}, zero={})",
                    q.scale.len(),
                    q.zero.len()
                ),
            });
        }
        // Pre-compute per-original-column group index: in act_order mode the
        // groups are formed over the permuted column space, so original column
        // `j` looks up `inv_perm[j] / group_size`.
        let inv_perm = q.perm.as_ref().map(|p| invert_permutation(p, cols));
        if let Some(ref ip) = inv_perm
            && ip.len() != cols
        {
            return Err(PeftError::Internal {
                msg: format!("GPTQ permutation length {} != cols {cols}", ip.len()),
            });
        }
        let mut out = vec![0.0_f32; expected];
        for i in 0..rows {
            for j in 0..cols {
                let col_in_perm = match &inv_perm {
                    Some(ip) => ip[j],
                    None => j,
                };
                let g = (col_in_perm / q.group_size).min(n_groups - 1);
                let s = q.scale[g];
                let z = q.zero[g];
                out[i * cols + j] = s * (q.q[i * cols + j] as f32) + z;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Validate input slices and the user-supplied configuration.
fn validate(
    w: &[f32],
    rows: usize,
    cols: usize,
    xtx_diag: &[f32],
    cfg: &GptqConfig,
) -> PeftResult<()> {
    if rows == 0 || cols == 0 {
        return Err(PeftError::Internal {
            msg: format!("GPTQ requires non-zero rows and cols (rows={rows}, cols={cols})"),
        });
    }
    if !matches!(cfg.bits, 2 | 3 | 4 | 8) {
        return Err(PeftError::Internal {
            msg: format!("GPTQ bits must be one of {{2, 3, 4, 8}}, got {}", cfg.bits),
        });
    }
    if cfg.group_size == 0 {
        return Err(PeftError::Internal {
            msg: "GPTQ group_size must be > 0".to_string(),
        });
    }
    if cfg.blocksize == 0 {
        return Err(PeftError::Internal {
            msg: "GPTQ blocksize must be > 0".to_string(),
        });
    }
    if cfg.damp_percent.is_nan() || cfg.damp_percent <= 0.0 {
        return Err(PeftError::Internal {
            msg: format!(
                "GPTQ damp_percent must be > 0 and finite, got {}",
                cfg.damp_percent
            ),
        });
    }
    if w.len() != rows * cols {
        return Err(PeftError::Internal {
            msg: format!(
                "GPTQ weight length {} != rows*cols {}",
                w.len(),
                rows * cols
            ),
        });
    }
    if xtx_diag.len() != cols {
        return Err(PeftError::Internal {
            msg: format!("GPTQ xtx_diag length {} != cols {cols}", xtx_diag.len()),
        });
    }
    for (j, &v) in xtx_diag.iter().enumerate() {
        if !v.is_finite() {
            return Err(PeftError::Internal {
                msg: format!("GPTQ xtx_diag[{j}]={v} is not finite"),
            });
        }
    }
    Ok(())
}

/// Saturating clamp of a (possibly negative) integer into the code interval `[0, q_max]`.
#[inline]
fn clamp_i64_to_code(code: i64, q_max: i32) -> i32 {
    let lo = 0_i64;
    let hi = q_max as i64;
    code.clamp(lo, hi) as i32
}

/// Mean of an `f32` slice (returning `0.0` for an empty slice).
#[inline]
fn mean_f64(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for &v in xs {
        acc += v as f64;
    }
    acc / xs.len() as f64
}

/// Return the indices of `xs` sorted by descending magnitude.
fn permutation_by_descending(xs: &[f32]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..xs.len()).collect();
    perm.sort_by(|&i, &j| {
        let a = xs[i].abs();
        let b = xs[j].abs();
        b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
    });
    perm
}

/// Build a row-major `rows × cols` matrix whose column `k` equals the original
/// column `perm[k]` of `w` (also row-major `rows × cols`).
fn permute_columns(w: &[f64], rows: usize, cols: usize, perm: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for i in 0..rows {
        for (new_col, &old_col) in perm.iter().enumerate().take(cols) {
            out[i * cols + new_col] = w[i * cols + old_col];
        }
    }
    out
}

/// Min/max of every entry in columns `[j_start, j_end)` of a row-major `rows × cols`
/// `f64` matrix. Assumes `j_start < j_end ≤ cols` and `rows ≥ 1`.
fn group_min_max(w: &[f64], rows: usize, cols: usize, j_start: usize, j_end: usize) -> (f64, f64) {
    let mut lo = w[j_start];
    let mut hi = w[j_start];
    for i in 0..rows {
        for j in j_start..j_end {
            let v = w[i * cols + j];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    (lo, hi)
}

/// Upper-triangular Cholesky factor of `H⁻¹` for a diagonal `H` passed via its
/// diagonal `h_diag`. Returns a row-major `cols × cols` upper-triangular matrix.
///
/// Generalised note: the off-diagonal entries are zero for a diagonal `H`. The
/// inner GPTQ loop reads `hinv_chol[j, k]` for `k > j`, so leaving those
/// entries at zero correctly turns the cross-column update into a no-op without
/// special-casing.
fn cholesky_of_inverse_upper(h_diag: &[f64]) -> PeftResult<Vec<f64>> {
    let n = h_diag.len();
    if n == 0 {
        return Err(PeftError::Internal {
            msg: "Cholesky-of-inverse received empty diagonal".to_string(),
        });
    }
    let mut out = vec![0.0_f64; n * n];
    for j in 0..n {
        let hd = h_diag[j];
        if hd <= 0.0 || !hd.is_finite() {
            return Err(PeftError::Internal {
                msg: format!("Cholesky needs H[{j}, {j}] > 0, got {hd}"),
            });
        }
        out[j * n + j] = 1.0 / hd.sqrt();
    }
    Ok(out)
}

/// Move quantized codes from the permuted column order back to the original
/// column order so callers see a standard row-major `rows × cols` layout.
fn unpermute_codes(q_perm: &[i32], perm: &[usize], rows: usize, cols: usize) -> Vec<i32> {
    let mut q_out = vec![0_i32; rows * cols];
    for i in 0..rows {
        for (new_col, &old_col) in perm.iter().enumerate().take(cols) {
            q_out[i * cols + old_col] = q_perm[i * cols + new_col];
        }
    }
    q_out
}

/// Compute the inverse permutation: if `perm[k] = old`, then `inv[old] = k`.
fn invert_permutation(perm: &[usize], cols: usize) -> Vec<usize> {
    let mut inv = vec![0_usize; cols];
    for (k, &old) in perm.iter().enumerate().take(cols) {
        if old < cols {
            inv[old] = k;
        }
    }
    inv
}
