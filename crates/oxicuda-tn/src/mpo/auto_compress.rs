//! Automatic MPO compression via SVD-based bond reduction.
//!
//! This module implements a sweep-based algorithm that compresses a Matrix Product Operator
//! by sequentially merging adjacent site tensors into a "super-tensor", performing a
//! truncated SVD, and factoring the result back into two site tensors with a reduced shared
//! bond dimension.
//!
//! ## Representation
//!
//! An MPO on N sites is encoded as:
//! - `cores[k]`: rank-4 tensor of shape `(r_{k-1}, d_in_k, d_out_k, r_k)` stored row-major.
//!   Element `[r_l, i, j, r_r]` is at flat index
//!   `r_l * (d_in * d_out * r_r) + i * (d_out * r_r) + j * r_r + r_r_idx`.
//! - `bond_dims`: length N+1 vector `[r_0, r_1, ..., r_N]` with `r_0 = r_N = 1` (OBC).
//! - `phys_dims`: length N vector of `(d_in_k, d_out_k)` pairs.
//!
//! ## Algorithm
//!
//! Single left-to-right sweep: for each bond `k ↔ k+1`:
//! 1. Merge cores k and k+1 by contracting over the shared virtual bond `r_k`.
//! 2. Reshape the result to a matrix `(r_l * d_in_k * d_out_k) × (r_r * d_in_{k+1} * d_out_{k+1})`.
//! 3. Perform a thin SVD and truncate to `min(max_bond, #{ s_i : s_i > tol * s_max })`.
//! 4. Write core k ← `U` reshaped, core k+1 ← `diag(S) · V^T` reshaped.

use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for MPO compression.
#[derive(Debug, Clone)]
pub struct MpoCompressConfig {
    /// Maximum virtual bond dimension to retain after truncation.
    pub max_bond: usize,
    /// Relative truncation tolerance: singular values `s_i < tol * s_max` are discarded.
    pub tol: f64,
}

impl Default for MpoCompressConfig {
    fn default() -> Self {
        Self {
            max_bond: 32,
            tol: 1e-8,
        }
    }
}

// ─── MPO data ────────────────────────────────────────────────────────────────

/// Flat representation of a Matrix Product Operator.
///
/// Core `k` has shape `(bond_dims[k], phys_dims[k].0, phys_dims[k].1, bond_dims[k+1])`.
/// Element `[r_l, i, j, r_r]` is at flat index
/// ```text
/// r_l * (d_in * d_out * r_r) + i * (d_out * r_r) + j * r_r + r_r_idx
/// ```
/// where `d_in = phys_dims[k].0`, `d_out = phys_dims[k].1`.
///
/// Open boundary conditions: `bond_dims[0] == 1` and `bond_dims[N] == 1`.
#[derive(Debug, Clone)]
pub struct MpoData {
    /// Flattened core tensors, one per site.
    pub cores: Vec<Vec<f64>>,
    /// Virtual bond dimensions, length `N + 1`.
    pub bond_dims: Vec<usize>,
    /// Physical dimensions `(d_in, d_out)` per site, length `N`.
    pub phys_dims: Vec<(usize, usize)>,
}

impl MpoData {
    /// Validate internal consistency.
    fn validate(&self) -> TnResult<()> {
        let n = self.cores.len();
        if n == 0 {
            return Err(TnError::EmptyInput);
        }
        if self.bond_dims.len() != n + 1 {
            return Err(TnError::ShapeMismatch {
                expected: vec![n + 1],
                got: vec![self.bond_dims.len()],
            });
        }
        if self.phys_dims.len() != n {
            return Err(TnError::ShapeMismatch {
                expected: vec![n],
                got: vec![self.phys_dims.len()],
            });
        }
        for k in 0..n {
            let r_l = self.bond_dims[k];
            let r_r = self.bond_dims[k + 1];
            let (d_in, d_out) = self.phys_dims[k];
            let expected = r_l * d_in * d_out * r_r;
            if self.cores[k].len() != expected {
                return Err(TnError::ShapeMismatch {
                    expected: vec![r_l, d_in, d_out, r_r],
                    got: vec![self.cores[k].len()],
                });
            }
        }
        Ok(())
    }
}

// ─── Public functions ─────────────────────────────────────────────────────────

/// Return a copy of the current virtual bond dimensions.
///
/// The returned vector has length `N + 1` and matches `mpo.bond_dims`.
pub fn mpo_bond_dims(mpo: &MpoData) -> Vec<usize> {
    mpo.bond_dims.clone()
}

/// Compute the squared Frobenius norm of the MPO: `||W||_F^2 = Σ w_{ij...}^2`.
pub fn mpo_operator_norm_sq(mpo: &MpoData) -> TnResult<f64> {
    mpo.validate()?;
    let norm_sq = mpo
        .cores
        .iter()
        .flat_map(|c| c.iter())
        .map(|&x| x * x)
        .sum::<f64>();
    Ok(norm_sq)
}

/// Compress an MPO via a single left-to-right SVD sweep.
///
/// For each bond between site `k` and `k+1`, the two cores are merged into a
/// super-tensor, reshaped into a matrix, and an SVD truncation is applied.
/// The new bond dimension is `min(config.max_bond, #{s_i : s_i > config.tol * s_max})`,
/// with at least 1 kept.
///
/// # Errors
///
/// - [`TnError::EmptyInput`] — `mpo.cores` is empty.
/// - [`TnError::ShapeMismatch`] — internal dimension inconsistency.
/// - [`TnError::InvalidParameter`] — `max_bond == 0` or `tol < 0`.
/// - [`TnError::NotConverged`] — Jacobi SVD did not converge (rare).
pub fn mpo_compress(mpo: &MpoData, config: &MpoCompressConfig) -> TnResult<MpoData> {
    mpo.validate()?;
    if config.max_bond == 0 {
        return Err(TnError::InvalidParameter {
            name: "max_bond".into(),
            reason: "must be >= 1".into(),
        });
    }
    if config.tol < 0.0 {
        return Err(TnError::InvalidParameter {
            name: "tol".into(),
            reason: "must be non-negative".into(),
        });
    }

    let n = mpo.cores.len();

    // Single-site MPO: no bonds to compress; return a clone.
    if n == 1 {
        return Ok(mpo.clone());
    }

    // Working copies that will be mutated site-by-site during the sweep.
    let mut cores: Vec<Vec<f64>> = mpo.cores.clone();
    let mut bond_dims: Vec<usize> = mpo.bond_dims.clone();
    let phys_dims = mpo.phys_dims.clone();

    // Left-to-right sweep over bonds 0..(N-1).
    for k in 0..n - 1 {
        let r_l = bond_dims[k];
        let r_mid = bond_dims[k + 1];
        let r_r = bond_dims[k + 2];
        let (d_in_k, d_out_k) = phys_dims[k];
        let (d_in_k1, d_out_k1) = phys_dims[k + 1];

        // Matrix dimensions after merging + reshape.
        // Row index   encodes (r_l_idx, i_k,  j_k)  → natural flat order of core_k  rows.
        // Column index encodes (i_{k+1}, j_{k+1}, r_r_idx) → natural flat order of core_{k+1}
        // without the leading r_mid index, so that S·V^T can be used directly as core_{k+1}.
        let nrows = r_l * d_in_k * d_out_k;
        let ncols = d_in_k1 * d_out_k1 * r_r;

        // Build the merged "theta" matrix of shape (nrows, ncols):
        //   theta[r_l_idx, i_k, j_k ; i_{k+1}, j_{k+1}, r_r_idx]
        //   = Σ_{r_mid_idx} core_k[r_l, i_k, j_k, r_mid] * core_{k+1}[r_mid, i_{k+1}, j_{k+1}, r_r]
        let theta = merge_cores(
            &cores[k],
            r_l,
            d_in_k,
            d_out_k,
            r_mid,
            &cores[k + 1],
            d_in_k1,
            d_out_k1,
            r_r,
        );

        // SVD of the theta matrix.
        let svd = svd_jacobi(&theta, nrows, ncols)?;

        // Determine how many singular values to keep.
        let keep = truncation_rank(&svd.s, config.max_bond, config.tol);

        // New virtual bond dimension at position k+1.
        bond_dims[k + 1] = keep;

        // Update core k ← U[:, :keep]  reshaped to (r_l, d_in_k, d_out_k, keep).
        // U is (nrows, k_full) row-major; we take first `keep` columns.
        cores[k] = extract_u_columns(&svd.u, nrows, svd.k, keep);

        // Update core k+1 ← diag(S[:keep]) · V^T[:keep, :]  reshaped to (keep, d_in_{k+1}, d_out_{k+1}, r_r).
        // V^T is (k_full, ncols) row-major; we take first `keep` rows, scaled by S.
        cores[k + 1] = extract_sv_rows(&svd.s, &svd.vt, svd.k, ncols, keep);
    }

    Ok(MpoData {
        cores,
        bond_dims,
        phys_dims,
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Merge two adjacent MPO cores by contracting over their shared virtual bond.
///
/// Core k has shape `(r_l, d_in_k, d_out_k, r_mid)`.
/// Core k+1 has shape `(r_mid, d_in_k1, d_out_k1, r_r)`.
///
/// The contracted result is reshaped to a matrix of shape
/// `(r_l * d_in_k * d_out_k, d_in_k1 * d_out_k1 * r_r)`.
///
/// Row index encodes `(r_l_idx, i_k, j_k)` — matching core_k's natural row layout
/// `r_l_idx * (d_in_k * d_out_k) + i_k * d_out_k + j_k`.
///
/// Column index encodes `(i_{k+1}, j_{k+1}, r_r_idx)` — matching core_{k+1}'s natural
/// storage order `i_{k+1} * (d_out_{k+1} * r_r) + j_{k+1} * r_r + r_r_idx`, so that
/// after SVD the `S · V^T` factor can be used as core_{k+1} without any extra permutation.
#[allow(clippy::too_many_arguments)]
fn merge_cores(
    core_k: &[f64],
    r_l: usize,
    d_in_k: usize,
    d_out_k: usize,
    r_mid: usize,
    core_k1: &[f64],
    d_in_k1: usize,
    d_out_k1: usize,
    r_r: usize,
) -> Vec<f64> {
    let nrows = r_l * d_in_k * d_out_k;
    let ncols = d_in_k1 * d_out_k1 * r_r;
    let mut mat = vec![0.0f64; nrows * ncols];

    // Flat index into core_k:  [r_l_idx, i_k, j_k, r_m]
    //   = r_l_idx * (d_in_k * d_out_k * r_mid) + i_k * (d_out_k * r_mid) + j_k * r_mid + r_m
    let stride_k_rl = d_in_k * d_out_k * r_mid;
    let stride_k_ik = d_out_k * r_mid;

    // Flat index into core_k1: [r_m, i_k1, j_k1, r_r_idx]
    //   = r_m * (d_in_k1 * d_out_k1 * r_r) + i_k1 * (d_out_k1 * r_r) + j_k1 * r_r + r_r_idx
    let stride_k1_rm = d_in_k1 * d_out_k1 * r_r;
    let stride_k1_ik1 = d_out_k1 * r_r;

    for r_l_idx in 0..r_l {
        for i_k in 0..d_in_k {
            for j_k in 0..d_out_k {
                // Row index in theta: natural flat order for the left-core "row" indices.
                let row = r_l_idx * d_in_k * d_out_k + i_k * d_out_k + j_k;
                let base_k = r_l_idx * stride_k_rl + i_k * stride_k_ik + j_k * r_mid;
                for i_k1 in 0..d_in_k1 {
                    for j_k1 in 0..d_out_k1 {
                        for r_r_idx in 0..r_r {
                            // Column index: natural flat storage of core_{k+1} rows
                            // (excluding the leading r_mid index which is summed out).
                            let col = i_k1 * stride_k1_ik1 + j_k1 * r_r + r_r_idx;
                            let mut acc = 0.0f64;
                            for r_m in 0..r_mid {
                                // core_k1 at [r_m, i_k1, j_k1, r_r_idx]
                                let k1_idx = r_m * stride_k1_rm + col;
                                acc += core_k[base_k + r_m] * core_k1[k1_idx];
                            }
                            mat[row * ncols + col] = acc;
                        }
                    }
                }
            }
        }
    }
    mat
}

/// Determine the number of singular values to keep.
///
/// Keep at most `max_bond` singular values, and drop any `s_i < tol * s_max`.
/// Always keeps at least 1.
fn truncation_rank(s: &[f64], max_bond: usize, tol: f64) -> usize {
    if s.is_empty() {
        return 1;
    }
    let s_max = s[0]; // descending order guaranteed by svd_jacobi
    let abs_tol = tol * s_max;
    let count_above = s.iter().filter(|&&v| v > abs_tol).count();
    // If tol == 0.0, every non-negative value is "above" threshold, so count_above == s.len().
    count_above.min(max_bond).max(1)
}

/// Extract the first `keep` columns of a `(nrows × k_full)` row-major U matrix.
fn extract_u_columns(u: &[f64], nrows: usize, k_full: usize, keep: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nrows * keep];
    for i in 0..nrows {
        for j in 0..keep {
            out[i * keep + j] = u[i * k_full + j];
        }
    }
    out
}

/// Scale the first `keep` rows of `vt` (shape `k_full × ncols`) by the corresponding
/// singular values, producing a `(keep × ncols)` matrix.
fn extract_sv_rows(s: &[f64], vt: &[f64], k_full: usize, ncols: usize, keep: usize) -> Vec<f64> {
    let _ = k_full; // used implicitly via indexing convention
    let mut out = vec![0.0f64; keep * ncols];
    for i in 0..keep {
        let sv = s[i];
        for j in 0..ncols {
            out[i * ncols + j] = sv * vt[i * ncols + j];
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a trivial identity MPO with n sites of physical dim d.
    /// Each core has shape (1, d, d, 1) with δ_{i,j} entries.
    fn identity_mpo_data(n: usize, d: usize) -> MpoData {
        let mut data = vec![0.0f64; d * d];
        for p in 0..d {
            data[p * d + p] = 1.0;
        }
        MpoData {
            cores: vec![data; n],
            bond_dims: vec![1usize; n + 1],
            phys_dims: vec![(d, d); n],
        }
    }

    /// Build a random-like MPO using a simple LCG, for testing bond-dim reduction.
    /// Cores have shape `(r_l, d_in, d_out, r_r)`.
    fn random_mpo_data(n: usize, d_in: usize, d_out: usize, inner_bond: usize) -> MpoData {
        let mut lcg: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut next = || -> f64 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) as f64) / (u32::MAX as f64) - 0.5
        };

        let mut cores = Vec::with_capacity(n);
        let mut bond_dims = Vec::with_capacity(n + 1);
        bond_dims.push(1usize);

        for k in 0..n {
            let r_l = *bond_dims.last().expect("last should succeed");
            let r_r = if k == n - 1 { 1 } else { inner_bond };
            bond_dims.push(r_r);
            let size = r_l * d_in * d_out * r_r;
            cores.push((0..size).map(|_| next()).collect::<Vec<f64>>());
        }

        MpoData {
            cores,
            bond_dims,
            phys_dims: vec![(d_in, d_out); n],
        }
    }

    /// Reconstruct the dense operator matrix from MpoData by contracting all cores.
    /// Returns a flat `(D^n_in, D^n_out)` matrix — only feasible for small n, d.
    fn dense_operator(mpo: &MpoData) -> Vec<f64> {
        let n = mpo.cores.len();
        // Start with a 1-element "transfer" vector of value 1.0.
        // We accumulate the full multi-site operator by repeatedly tensoring in new sites.
        // State: matrix of shape (d_in_acc, d_out_acc) — stored as flat row-major.
        let mut d_in_acc = 1usize;
        let mut d_out_acc = 1usize;
        let mut mat = vec![1.0f64]; // scalar 1

        for k in 0..n {
            let r_l = mpo.bond_dims[k];
            let r_r = mpo.bond_dims[k + 1];
            let (d_in_k, d_out_k) = mpo.phys_dims[k];

            // New dimensions: d_in_acc' = d_in_acc * d_in_k, etc.
            // But we also need to sum over virtual bond r_l (already handled by sequential
            // processing — we absorb r_l into the accumulated state).
            //
            // More precisely: the accumulated tensor tracks shape
            // (d_in_acc, d_out_acc, r_l_current). Initially r_l_current = 1.
            // We multiply by core k and contract over r_l.
            //
            // For simplicity in tests, use a direct contraction approach:
            // after site k the accumulated tensor has shape (r_r, d_in_total, d_out_total).

            let new_d_in = d_in_acc * d_in_k;
            let new_d_out = d_out_acc * d_out_k;

            // mat currently has shape (r_l, d_in_acc, d_out_acc) stored as flat.
            // We contract: new_mat[r_r, d_in_acc, i_k, d_out_acc, j_k]
            //   = Σ_{r_l} mat[r_l, d_in_acc, d_out_acc] * core[r_l, i_k, j_k, r_r]
            let mut new_mat = vec![0.0f64; r_r * new_d_in * new_d_out];

            for r_l_idx in 0..r_l {
                for i_acc in 0..d_in_acc {
                    for j_acc in 0..d_out_acc {
                        let m_val = mat[r_l_idx * d_in_acc * d_out_acc + i_acc * d_out_acc + j_acc];
                        for i_k in 0..d_in_k {
                            for j_k in 0..d_out_k {
                                let c_idx = r_l_idx * (d_in_k * d_out_k * r_r)
                                    + i_k * (d_out_k * r_r)
                                    + j_k * r_r;
                                for r_r_idx in 0..r_r {
                                    let c_val = mpo.cores[k][c_idx + r_r_idx];
                                    // new index: [r_r_idx, i_acc * d_in_k + i_k, j_acc * d_out_k + j_k]
                                    let new_i = i_acc * d_in_k + i_k;
                                    let new_j = j_acc * d_out_k + j_k;
                                    new_mat[r_r_idx * new_d_in * new_d_out
                                        + new_i * new_d_out
                                        + new_j] += m_val * c_val;
                                }
                            }
                        }
                    }
                }
            }

            d_in_acc = new_d_in;
            d_out_acc = new_d_out;
            mat = new_mat;
        }
        // Final: mat has shape (1, d_in_total, d_out_total); drop the r dimension.
        mat[0..d_in_acc * d_out_acc].to_vec()
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn mpo_compress_identity_2site() {
        // 2-site identity MPO has bond dims [1, 1, 1]. After compression, still [1, 1, 1].
        let mpo = identity_mpo_data(2, 2);
        let config = MpoCompressConfig::default();
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");
        assert_eq!(compressed.bond_dims, vec![1, 1, 1]);
        assert_eq!(compressed.cores.len(), 2);
    }

    #[test]
    fn mpo_compress_reduces_bond_dim() {
        // Random MPO with inner_bond = 8; compress to max_bond = 3.
        let mpo = random_mpo_data(4, 2, 2, 8);
        let config = MpoCompressConfig {
            max_bond: 3,
            tol: 1e-12,
        };
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");
        for &bd in &compressed.bond_dims {
            assert!(bd <= 3, "bond dim {} exceeds max_bond 3", bd);
        }
    }

    #[test]
    fn mpo_compress_norm_preserved() {
        // The Frobenius norm of the dense operator (not the per-core tensor norm) should
        // be preserved after compression that does not truncate any significant singular values.
        // Use the 3-site identity MPO which is exactly rank-1 in bond space.
        let n = 3;
        let d = 2;
        let mpo = identity_mpo_data(n, d);

        let config = MpoCompressConfig {
            max_bond: 4,
            tol: 1e-10,
        };
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");

        // Compare the dense operator: both must represent the same linear map.
        let orig_dense = dense_operator(&mpo);
        let comp_dense = dense_operator(&compressed);
        let orig_op_norm_sq: f64 = orig_dense.iter().map(|&x| x * x).sum();
        let comp_op_norm_sq: f64 = comp_dense.iter().map(|&x| x * x).sum();
        let rel_err = (orig_op_norm_sq - comp_op_norm_sq).abs() / (orig_op_norm_sq + 1e-30);
        assert!(
            rel_err < 1e-6,
            "operator norm not preserved: orig_sq={orig_op_norm_sq}, comp_sq={comp_op_norm_sq}, rel_err={rel_err}"
        );

        // Also check element-wise agreement of the dense maps.
        let diff: f64 = orig_dense
            .iter()
            .zip(&comp_dense)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(diff < 1e-9, "dense operators differ: diff={diff}");
    }

    #[test]
    fn mpo_bond_dims_after_compress() {
        // All bond dims (excluding boundary 1s at 0 and N) must be <= max_bond.
        let mpo = random_mpo_data(5, 2, 2, 10);
        let config = MpoCompressConfig {
            max_bond: 4,
            tol: 1e-12,
        };
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");
        let dims = mpo_bond_dims(&compressed);
        for &bd in &dims {
            assert!(bd <= 4 || bd == 1, "bond dim {} violates max_bond=4", bd);
        }
    }

    #[test]
    fn mpo_compress_single_site_is_noop() {
        // A single-site MPO has no bonds; compression should return an identical copy.
        let mpo = identity_mpo_data(1, 3);
        let config = MpoCompressConfig::default();
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");
        assert_eq!(compressed.bond_dims, vec![1, 1]);
        assert_eq!(compressed.cores[0], mpo.cores[0]);
    }

    #[test]
    fn mpo_operator_norm_sq_positive() {
        // Non-zero MPO has positive Frobenius norm squared.
        let mpo = identity_mpo_data(3, 2);
        let norm_sq = mpo_operator_norm_sq(&mpo).expect("norm ok");
        assert!(norm_sq > 0.0, "expected positive norm, got {norm_sq}");
    }

    #[test]
    fn mpo_operator_norm_sq_zero_tensor() {
        // All-zero cores give norm 0.
        let mpo = MpoData {
            cores: vec![vec![0.0; 4], vec![0.0; 4]],
            bond_dims: vec![1, 1, 1],
            phys_dims: vec![(2, 2), (2, 2)],
        };
        let norm_sq = mpo_operator_norm_sq(&mpo).expect("norm ok");
        assert_eq!(norm_sq, 0.0);
    }

    #[test]
    fn mpo_compress_empty_returns_error() {
        // Empty cores vector should produce EmptyInput.
        let mpo = MpoData {
            cores: vec![],
            bond_dims: vec![1],
            phys_dims: vec![],
        };
        let config = MpoCompressConfig::default();
        let result = mpo_compress(&mpo, &config);
        assert!(
            matches!(result, Err(TnError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    #[test]
    fn mpo_compress_mismatched_dims_returns_error() {
        // bond_dims length does not match cores.len() + 1.
        let mpo = MpoData {
            cores: vec![vec![1.0; 4], vec![1.0; 4]],
            bond_dims: vec![1, 1], // should be length 3
            phys_dims: vec![(2, 2), (2, 2)],
        };
        let config = MpoCompressConfig::default();
        let result = mpo_compress(&mpo, &config);
        assert!(
            matches!(result, Err(TnError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn mpo_compress_tight_tol_exact() {
        // With tol = 0.0, no singular value is dropped by threshold (only max_bond matters).
        // Use a large enough max_bond so no truncation occurs at all.
        // We verify that the dense operator is reproduced exactly.
        //
        // Note: `mpo_operator_norm_sq` sums per-core entry squares, which is NOT gauge-
        // invariant. Only the dense operator's Frobenius norm is preserved by SVD
        // (since SVD is a unitary transformation on each bond).
        let mpo = random_mpo_data(3, 2, 2, 4);
        let config = MpoCompressConfig {
            max_bond: 64,
            tol: 0.0,
        };
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");

        // Dense operator must match the original exactly.
        let orig_dense = dense_operator(&mpo);
        let comp_dense = dense_operator(&compressed);
        let diff: f64 = orig_dense
            .iter()
            .zip(&comp_dense)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-9,
            "dense operators differ after tol=0 compression: diff={diff}"
        );

        // Also check that the dense operator Frobenius norm is preserved (gauge-invariant).
        let orig_op_norm_sq: f64 = orig_dense.iter().map(|&x| x * x).sum();
        let comp_op_norm_sq: f64 = comp_dense.iter().map(|&x| x * x).sum();
        let rel_err = (orig_op_norm_sq - comp_op_norm_sq).abs() / (orig_op_norm_sq + 1e-30);
        assert!(
            rel_err < 1e-6,
            "dense op norm mismatch: orig_sq={orig_op_norm_sq}, comp_sq={comp_op_norm_sq}"
        );
    }

    #[test]
    fn mpo_compress_dense_operator_preserved_identity() {
        // After compressing a 3-site d=2 identity MPO, the dense operator must equal
        // the 4x4 identity (up to reordering of physical legs that we define consistently).
        let mpo = identity_mpo_data(3, 2);
        let config = MpoCompressConfig {
            max_bond: 16,
            tol: 1e-10,
        };
        let compressed = mpo_compress(&mpo, &config).expect("compress ok");

        let orig_dense = dense_operator(&mpo);
        let comp_dense = dense_operator(&compressed);
        let diff: f64 = orig_dense
            .iter()
            .zip(&comp_dense)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-9,
            "dense identity operator changed after compression: diff={diff}"
        );
    }
}
