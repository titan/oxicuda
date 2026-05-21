//! RegMean — Regression-based Model Merging.
//!
//! Reference: Jin X, Ren X, Preotiuc-Pietro D, Cheng P (2023) "Dataless
//! Knowledge Fusion by Merging Weights of Language Models", ICLR 2023.
//! <https://arxiv.org/abs/2212.09849>
//!
//! RegMean treats each linear layer's per-task weight as the solution of a
//! linear-regression problem with input Gram matrix `G_i = X_iᵀ X_i`. Merging
//! `K` task-specific weights `{W_i}` and Grams `{G_i}` reduces to the
//! closed-form ridge-regularised least-squares estimate
//!
//! ```text
//! W̄  =  (Σᵢ G_i' + ε·I)⁻¹  ·  Σᵢ G_i' · W_i
//! ```
//!
//! where `G_i' = G_i ⊙ M(α)` is the user-supplied Gram with off-diagonal
//! entries scaled by `α ∈ [0, 1]` (`α = 0` keeps only the diagonal; `α = 1`
//! uses the full matrix). The ridge term `ε·I` guards against singular Grams.
//!
//! This implementation generalises the per-coordinate weighted-average pattern
//! in [`crate::merge::fisher_merging`] to a matrix inverse by solving the
//! `(d_in × d_in)` linear system in-place via Gauss-Jordan elimination on the
//! augmented `[A | B]` matrix of size `d_in × (d_in + d_out)`.

use crate::error::{PeftError, PeftResult};

/// Configuration for the RegMean merge.
#[derive(Debug, Clone, Copy)]
pub struct RegMeanConfig {
    /// Off-diagonal scaling factor `α ∈ [0, 1]` applied to every Gram.
    /// `α = 0` reduces the merger to a diagonal-only weighted average;
    /// `α = 1` uses the full Gram matrix unchanged.
    pub non_diag_alpha: f32,
    /// Ridge regulariser added to the diagonal of `Σ G_i'` to keep the linear
    /// solve well-conditioned. Must be strictly positive.
    pub eps: f32,
}

impl Default for RegMeanConfig {
    fn default() -> Self {
        Self {
            non_diag_alpha: 0.1,
            eps: 1e-6,
        }
    }
}

/// RegMean algorithm namespace.
pub struct RegMean;

impl RegMean {
    /// Merge a collection of `(weight, gram)` pairs using the RegMean
    /// closed-form solution.
    ///
    /// `models[i].0` is the row-major `d_in × d_out` weight matrix for model
    /// `i`; `models[i].1` is the row-major `d_in × d_in` input Gram.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] for any configuration or dimension
    /// violation (empty input, `eps ≤ 0`, `α ∉ [0, 1]`, mismatched lengths,
    /// or `d_in == 0` / `d_out == 0`).
    pub fn merge(
        models: &[(&[f32], &[f32])],
        d_in: usize,
        d_out: usize,
        cfg: &RegMeanConfig,
    ) -> PeftResult<Vec<f32>> {
        validate(models, d_in, d_out, cfg)?;

        let alpha = cfg.non_diag_alpha as f64;

        // A = Σ G_i' + ε·I, with G_i' = G_i ⊙ M(α).
        let mut a = vec![0.0_f64; d_in * d_in];
        // B = Σ G_i' · W_i (d_in × d_out, row-major).
        let mut b = vec![0.0_f64; d_in * d_out];

        for &(w, g) in models {
            // Accumulate masked Gram into A.
            for r in 0..d_in {
                let row = r * d_in;
                for c in 0..d_in {
                    let raw = g[row + c] as f64;
                    let scaled = if r == c { raw } else { alpha * raw };
                    a[row + c] += scaled;
                }
            }
            // Accumulate G_i' · W_i into B. We compute G_i' on-the-fly to
            // avoid allocating a per-model masked Gram.
            for r in 0..d_in {
                let row = r * d_in;
                for k in 0..d_in {
                    let raw = g[row + k] as f64;
                    let gv = if r == k { raw } else { alpha * raw };
                    if gv == 0.0 {
                        continue;
                    }
                    let w_row = k * d_out;
                    let b_row = r * d_out;
                    for j in 0..d_out {
                        b[b_row + j] += gv * (w[w_row + j] as f64);
                    }
                }
            }
        }

        // Add ε·I to A.
        let eps = cfg.eps as f64;
        for d in 0..d_in {
            a[d * d_in + d] += eps;
        }

        // Solve A · W = B in-place via Gauss-Jordan on [A | B].
        let merged_f64 = solve_gauss_jordan(a, b, d_in, d_out)?;
        let merged: Vec<f32> = merged_f64.into_iter().map(|v| v as f32).collect();
        Ok(merged)
    }

    /// Compute the Gram matrix `G = Σᵢ xᵢ · xᵢᵀ` for a collection of row
    /// vectors. `x[i]` must have length equal to `d_in`. Returns a row-major
    /// `d_in × d_in` matrix.
    ///
    /// # Errors
    /// Returns [`PeftError::Internal`] for empty input or row-length mismatch.
    pub fn compute_gram(x: &[Vec<f32>]) -> PeftResult<Vec<f32>> {
        if x.is_empty() {
            return Err(PeftError::Internal {
                msg: "RegMean compute_gram requires at least one row".to_string(),
            });
        }
        let d_in = x[0].len();
        if d_in == 0 {
            return Err(PeftError::Internal {
                msg: "RegMean compute_gram requires non-zero d_in".to_string(),
            });
        }
        for (idx, row) in x.iter().enumerate() {
            if row.len() != d_in {
                return Err(PeftError::Internal {
                    msg: format!(
                        "RegMean compute_gram row {idx} length {} != d_in {d_in}",
                        row.len()
                    ),
                });
            }
            for (j, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(PeftError::Internal {
                        msg: format!("RegMean compute_gram x[{idx}][{j}]={v} is not finite"),
                    });
                }
            }
        }
        let mut gram = vec![0.0_f64; d_in * d_in];
        for row in x {
            for a in 0..d_in {
                let xa = row[a] as f64;
                if xa == 0.0 {
                    continue;
                }
                let row_off = a * d_in;
                for b in 0..d_in {
                    gram[row_off + b] += xa * (row[b] as f64);
                }
            }
        }
        Ok(gram.into_iter().map(|v| v as f32).collect())
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Validate every input slice and the user-supplied configuration.
fn validate(
    models: &[(&[f32], &[f32])],
    d_in: usize,
    d_out: usize,
    cfg: &RegMeanConfig,
) -> PeftResult<()> {
    if models.is_empty() {
        return Err(PeftError::Internal {
            msg: "RegMean merge requires at least one model".to_string(),
        });
    }
    if d_in == 0 || d_out == 0 {
        return Err(PeftError::Internal {
            msg: format!("RegMean requires non-zero d_in and d_out (d_in={d_in}, d_out={d_out})"),
        });
    }
    if cfg.eps.is_nan() || cfg.eps <= 0.0 {
        return Err(PeftError::Internal {
            msg: format!("RegMean eps must be > 0 and finite, got {}", cfg.eps),
        });
    }
    if cfg.non_diag_alpha.is_nan() || !(0.0..=1.0).contains(&cfg.non_diag_alpha) {
        return Err(PeftError::Internal {
            msg: format!(
                "RegMean non_diag_alpha must be in [0, 1], got {}",
                cfg.non_diag_alpha
            ),
        });
    }
    let expected_w = d_in * d_out;
    let expected_g = d_in * d_in;
    for (idx, &(w, g)) in models.iter().enumerate() {
        if w.len() != expected_w {
            return Err(PeftError::Internal {
                msg: format!(
                    "RegMean model[{idx}] weight length {} != d_in*d_out {expected_w}",
                    w.len()
                ),
            });
        }
        if g.len() != expected_g {
            return Err(PeftError::Internal {
                msg: format!(
                    "RegMean model[{idx}] gram length {} != d_in*d_in {expected_g}",
                    g.len()
                ),
            });
        }
        for (j, &v) in w.iter().enumerate() {
            if !v.is_finite() {
                return Err(PeftError::Internal {
                    msg: format!("RegMean model[{idx}] weight[{j}]={v} is not finite"),
                });
            }
        }
        for (j, &v) in g.iter().enumerate() {
            if !v.is_finite() {
                return Err(PeftError::Internal {
                    msg: format!("RegMean model[{idx}] gram[{j}]={v} is not finite"),
                });
            }
        }
    }
    Ok(())
}

/// Solve the linear system `A · X = B` via in-place Gauss-Jordan elimination
/// with partial pivoting on the augmented matrix `[A | B]`.
///
/// * `a` is `d_in × d_in` (row-major).
/// * `b` is `d_in × d_out` (row-major).
///
/// Returns the solution `X` as a row-major `d_in × d_out` vector.
fn solve_gauss_jordan(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    d_in: usize,
    d_out: usize,
) -> PeftResult<Vec<f64>> {
    for col in 0..d_in {
        // Partial pivot: find the row with the largest |A[r, col]| for r >= col.
        let mut pivot = col;
        let mut best = a[col * d_in + col].abs();
        for r in (col + 1)..d_in {
            let v = a[r * d_in + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best <= 1e-30 {
            return Err(PeftError::Internal {
                msg: format!(
                    "RegMean linear solve hit singular pivot at column {col} (|A[{col},{col}]|≈0)"
                ),
            });
        }
        if pivot != col {
            swap_rows(&mut a, d_in, col, pivot);
            swap_rows(&mut b, d_out, col, pivot);
        }
        // Scale pivot row so A[col, col] = 1.
        let p = a[col * d_in + col];
        let inv = 1.0_f64 / p;
        for k in 0..d_in {
            a[col * d_in + k] *= inv;
        }
        for k in 0..d_out {
            b[col * d_out + k] *= inv;
        }
        // Eliminate column `col` from every other row.
        for r in 0..d_in {
            if r == col {
                continue;
            }
            let factor = a[r * d_in + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..d_in {
                a[r * d_in + k] -= factor * a[col * d_in + k];
            }
            for k in 0..d_out {
                b[r * d_out + k] -= factor * b[col * d_out + k];
            }
        }
    }
    Ok(b)
}

/// Swap rows `r1` and `r2` of a row-major `(n × stride)` matrix.
fn swap_rows(m: &mut [f64], stride: usize, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }
    let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
    let (head, tail) = m.split_at_mut(hi * stride);
    let row_a = &mut head[lo * stride..lo * stride + stride];
    let row_b = &mut tail[..stride];
    row_a.swap_with_slice(row_b);
}

/// Test-only entrypoint that exposes the internal Gauss-Jordan solver to the
/// sibling `regmean_tests` module (used for the OLS reference baseline).
#[cfg(test)]
pub(super) fn solve_gauss_jordan_for_test(
    a: Vec<f64>,
    b: Vec<f64>,
    d_in: usize,
    d_out: usize,
) -> PeftResult<Vec<f64>> {
    solve_gauss_jordan(a, b, d_in, d_out)
}
