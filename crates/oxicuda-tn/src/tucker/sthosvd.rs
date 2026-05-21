//! Sequentially-Truncated Higher-Order Singular Value Decomposition (ST-HOSVD).
//!
//! ## Reference
//!
//! Vannieuwenhoven, N., Vandebril, R., & Meerbergen, K. (2012).
//! *A new truncation strategy for the higher-order singular value decomposition.*
//! SIAM Journal on Scientific Computing, 34(2), A1027–A1052.
//!
//! ## Algorithm Overview
//!
//! Given an N-mode tensor `T` of shape `(d_0, d_1, ..., d_{N-1})` stored in C-order
//! (last index varies fastest), ST-HOSVD computes a Tucker decomposition
//! `T ≈ G ×_0 U_0 ×_1 U_1 ... ×_{N-1} U_{N-1}`
//! where `U_k` is of shape `(d_k, r_k)` with orthonormal columns and `G` (the core) is
//! of shape `(r_0, r_1, ..., r_{N-1})`.
//!
//! Unlike standard HOSVD (which unfolds the **original** tensor for every mode), ST-HOSVD
//! projects the tensor after each factor matrix computation, so subsequent SVDs operate on
//! **smaller** matrices: at step `k`, mode `k` has already been reduced from `d_k` to `r_k`
//! through projection with the accumulated modes `0..k`.
//!
//! ## Mode-k Unfolding (C-order)
//!
//! For a tensor of shape `(d_0, ..., d_{N-1})` stored row-major (C-order), the mode-k
//! unfolding is a matrix of shape `(d_k, D/d_k)` where `D = Π d_i`, and the column index
//! enumerates all multi-index combinations with the k-th index fixed at the row index.
//! The ordering follows the convention used by NumPy / standard Tucker literature:
//! column order cycles through all indices **other than** mode k, with later modes
//! varying fastest (right-to-left within the non-k index set).

use crate::svd::randomised_svd::{RsvdConfig, randomised_svd};
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the ST-HOSVD algorithm.
///
/// `max_rank[k]` is the target number of columns to retain from the mode-k SVD.
/// If `max_rank[k] >= d_k` the mode is kept at full rank (no truncation for that mode).
#[derive(Debug, Clone)]
pub struct SthosvdConfig {
    /// Target Tucker ranks (one per mode). Must have the same length as the number of modes.
    pub max_rank: Vec<usize>,
    /// If `true`, use the randomised SVD (faster for large modes); otherwise use Jacobi.
    pub use_randomised_svd: bool,
    /// Extra columns for the randomised SVD sketch (`l = r_k + oversampling`).
    pub rsvd_oversampling: usize,
    /// Seed for the randomised SVD RNG.
    pub rsvd_seed: u64,
}

impl Default for SthosvdConfig {
    fn default() -> Self {
        Self {
            max_rank: vec![2, 2],
            use_randomised_svd: false,
            rsvd_oversampling: 10,
            rsvd_seed: 42,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Output of the ST-HOSVD Tucker decomposition for an N-mode tensor.
///
/// The Tucker decomposition satisfies `T ≈ G ×_0 U_0 ×_1 U_1 ... ×_{N-1} U_{N-1}`.
#[derive(Debug, Clone)]
pub struct SthosvdResult {
    /// Core tensor of shape `core_shape`, stored C-order (row-major, last index fastest).
    pub core: Vec<f64>,
    /// Shape of the core tensor: `(r_0, r_1, ..., r_{N-1})`.
    pub core_shape: Vec<usize>,
    /// Factor matrices: `factors[k]` has shape `(d_k, r_k)`, stored row-major.
    pub factors: Vec<Vec<f64>>,
    /// Shapes of the factor matrices: `factor_shapes[k] == (d_k, r_k)`.
    pub factor_shapes: Vec<(usize, usize)>,
    /// Number of modes `N`.
    pub n_modes: usize,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the ST-HOSVD Tucker decomposition of an N-mode tensor.
///
/// # Parameters
///
/// - `tensor`: flat C-order storage of the input tensor.
/// - `shape`: slice of length `N` giving the size of each mode.
/// - `config`: algorithm parameters (ranks, SVD choice, etc.).
///
/// # Errors
///
/// - [`TnError::EmptyInput`] if any dimension is zero or the tensor is empty.
/// - [`TnError::ShapeMismatch`] if `tensor.len()` does not match `Π shape[k]`.
/// - [`TnError::InvalidParameter`] if `config.max_rank` has the wrong length or a rank is zero.
pub fn sthosvd(tensor: &[f64], shape: &[usize], config: &SthosvdConfig) -> TnResult<SthosvdResult> {
    // ── Validate ──────────────────────────────────────────────────────────
    let n_modes = shape.len();
    if n_modes == 0 || tensor.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if shape.contains(&0) {
        return Err(TnError::EmptyInput);
    }
    let total: usize = shape.iter().product();
    if tensor.len() != total {
        return Err(TnError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![tensor.len()],
        });
    }
    if config.max_rank.len() != n_modes {
        return Err(TnError::InvalidParameter {
            name: "max_rank".to_string(),
            reason: format!(
                "length {} does not match number of modes {}",
                config.max_rank.len(),
                n_modes
            ),
        });
    }
    for (k, &r) in config.max_rank.iter().enumerate() {
        if r == 0 {
            return Err(TnError::InvalidParameter {
                name: format!("max_rank[{k}]"),
                reason: "rank must be at least 1".to_string(),
            });
        }
    }

    // ── ST-HOSVD main loop ────────────────────────────────────────────────
    let mut current_tensor = tensor.to_vec();
    let mut current_shape = shape.to_vec();
    let mut factors: Vec<Vec<f64>> = Vec::with_capacity(n_modes);
    let mut factor_shapes: Vec<(usize, usize)> = Vec::with_capacity(n_modes);

    for k in 0..n_modes {
        let d_k = current_shape[k];
        // Clamp the target rank to the actual dimension
        let r_k = config.max_rank[k].min(d_k);

        // 1. Unfold current tensor along mode k → matrix M of shape (d_k, D_rest)
        let unfolded = unfold_mode(&current_tensor, &current_shape, k);
        let d_rest = unfolded.len() / d_k; // cols = D / d_k

        // 2. Compute top-r_k left singular vectors of M
        let u_k = compute_left_svecs(&unfolded, d_k, d_rest, r_k, config)?;
        // u_k is (d_k, r_k) row-major

        // 3. Project current tensor: current ×_k U_k^T
        //    U_k^T has shape (r_k, d_k); mode k shrinks from d_k to r_k.
        //    We build U_k^T explicitly (transpose of u_k).
        let u_k_t = transpose_matrix(&u_k, d_k, r_k);
        current_tensor = mode_product(&current_tensor, &current_shape, &u_k_t, r_k, k);
        current_shape[k] = r_k;

        factors.push(u_k);
        factor_shapes.push((d_k, r_k));
    }

    let core_shape = current_shape;
    Ok(SthosvdResult {
        core: current_tensor,
        core_shape,
        factors,
        factor_shapes,
        n_modes,
    })
}

/// Reconstruct the original tensor from a [`SthosvdResult`].
///
/// Applies `G ×_0 U_0 ×_1 U_1 ... ×_{N-1} U_{N-1}`.
///
/// # Errors
///
/// - [`TnError::ShapeMismatch`] if internal consistency fails.
pub fn sthosvd_reconstruct(result: &SthosvdResult) -> TnResult<Vec<f64>> {
    let n_modes = result.n_modes;
    if n_modes == 0 {
        return Err(TnError::EmptyInput);
    }
    let mut current_tensor = result.core.clone();
    let mut current_shape = result.core_shape.clone();

    for k in 0..n_modes {
        let (d_k, _r_k) = result.factor_shapes[k];
        let u_k = &result.factors[k]; // (d_k, r_k) row-major

        // Expand mode k from r_k to d_k via U_k (shape (d_k, r_k))
        current_tensor = mode_product(&current_tensor, &current_shape, u_k, d_k, k);
        current_shape[k] = d_k;
    }
    Ok(current_tensor)
}

// ─── Core tensor operations ───────────────────────────────────────────────────

/// Mode-k unfolding of an N-mode tensor stored in C-order.
///
/// Returns a matrix of shape `(d_k, D / d_k)` stored row-major, where `D = Π shape[i]`.
///
/// The column index enumerates all combinations of non-k indices with the last index of the
/// original tensor varying fastest. Concretely, for a 4-mode tensor with modes `(a, b, c, d)`
/// and `k = 1` (mode `b`), the unfolding is `(b, a * c * d)` and the column index is
/// `a * (c * d) + c * d + d` in C-order (which skips mode b from the full multi-index).
pub fn unfold_mode(tensor: &[f64], shape: &[usize], mode: usize) -> Vec<f64> {
    let n_modes = shape.len();
    let total: usize = shape.iter().product();
    let d_k = shape[mode];
    let d_rest = total / d_k;
    let mut out = vec![0.0f64; d_k * d_rest];

    // Strides for the original C-order tensor
    // stride[i] = Π shape[j] for j > i
    let mut strides = vec![1usize; n_modes];
    for i in (0..n_modes - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // For each element of the tensor (flat index `flat`), determine:
    //   row = multi_index[mode]           (the mode-k index)
    //   col = column in the unfolded matrix
    // The column is computed by enumerating all non-k indices in C-order.
    //
    // We iterate over all flat indices and decompose them.
    for (flat, &val) in tensor.iter().enumerate().take(total) {
        // Decompose `flat` into multi-index
        let mut remaining = flat;
        let mut row = 0usize;
        let mut col = 0usize;
        // Compute strides for non-mode indices (C-order among non-k indices)
        // We accumulate col by multiplying by the size of subsequent non-k dimensions.
        // To do this cleanly we precompute the "col stride" for each non-k mode.
        //
        // non-k indices in order: 0, 1, ..., mode-1, mode+1, ..., N-1
        // their C-order stride in the unfolded column space:
        //   col_stride[i] = Π shape[j] for j in (non-k modes after i)
        // We compute the multi-index inline.
        let mut col_stride = d_rest; // will be divided as we go
        for i in 0..n_modes {
            let idx = remaining / strides[i];
            remaining %= strides[i];
            if i == mode {
                row = idx;
            } else {
                col_stride /= shape[i];
                col += idx * col_stride;
            }
        }
        out[row * d_rest + col] = val;
    }
    out
}

/// Mode-k product: contract a matrix `u` of shape `(r, d_k)` with the tensor along mode k.
///
/// The output tensor has the same shape as the input except `shape[k]` is replaced by `r`.
///
/// Computes `(T ×_k U)[i_0,...,i_{k-1}, a, i_{k+1},...,i_{N-1}]
///         = Σ_{i_k} U[a, i_k] * T[i_0,...,i_k,...,i_{N-1}]`
pub fn mode_product(tensor: &[f64], shape: &[usize], u: &[f64], r: usize, mode: usize) -> Vec<f64> {
    let n_modes = shape.len();
    let d_k = shape[mode];
    let total_in: usize = shape.iter().product();
    debug_assert_eq!(tensor.len(), total_in);
    debug_assert_eq!(u.len(), r * d_k);

    // Output shape: same as input with shape[mode] = r
    let mut out_shape = shape.to_vec();
    out_shape[mode] = r;
    let total_out: usize = out_shape.iter().product();
    let mut out = vec![0.0f64; total_out];

    // C-order strides for input tensor
    let mut in_strides = vec![1usize; n_modes];
    for i in (0..n_modes - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    // C-order strides for output tensor
    let mut out_strides = vec![1usize; n_modes];
    out_strides[n_modes - 1] = 1;
    for i in (0..n_modes - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Iterate over all output elements
    for (out_flat, out_val) in out.iter_mut().enumerate() {
        // Decompose out_flat into multi-index for output tensor
        let mut remaining = out_flat;
        let mut multi_idx = vec![0usize; n_modes];
        for i in 0..n_modes {
            multi_idx[i] = remaining / out_strides[i];
            remaining %= out_strides[i];
        }

        // The output index along mode k is multi_idx[mode] = a (ranges 0..r)
        let a = multi_idx[mode];

        // Sum over mode-k input dimension: Σ_{i_k} U[a, i_k] * T[..., i_k, ...]
        let mut acc = 0.0f64;
        for i_k in 0..d_k {
            multi_idx[mode] = i_k;
            // Compute flat input index
            let in_flat: usize = multi_idx
                .iter()
                .zip(in_strides.iter())
                .map(|(&idx, &s)| idx * s)
                .sum();
            acc += u[a * d_k + i_k] * tensor[in_flat];
        }

        // Restore mode index for next iteration
        multi_idx[mode] = a;
        *out_val = acc;
    }
    out
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Transpose a `(rows, cols)` row-major matrix to `(cols, rows)` row-major.
fn transpose_matrix(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols * rows];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = mat[i * cols + j];
        }
    }
    out
}

/// Compute the top-`r` left singular vectors of an `(m, n)` matrix.
///
/// Chooses between Jacobi SVD and randomised SVD based on `config.use_randomised_svd`.
/// Returns a matrix of shape `(m, r)` with orthonormal columns.
fn compute_left_svecs(
    matrix: &[f64],
    m: usize,
    n: usize,
    r: usize,
    config: &SthosvdConfig,
) -> TnResult<Vec<f64>> {
    if config.use_randomised_svd && r < m.min(n) {
        let rsvd_cfg = RsvdConfig {
            k: r,
            oversampling: config.rsvd_oversampling,
            n_power_iter: 2,
            seed: config.rsvd_seed,
        };
        let svd = randomised_svd(matrix, m, n, &rsvd_cfg)?;
        // svd.u is (m, r) — already the first r left singular vectors
        Ok(svd.u)
    } else {
        let svd = svd_jacobi(matrix, m, n)?;
        // svd.u is (m, k) with k = min(m,n); extract first r columns
        let k = svd.k;
        if r > k {
            return Err(TnError::InvalidParameter {
                name: "max_rank".to_string(),
                reason: format!("requested rank {r} exceeds SVD rank {k}"),
            });
        }
        let mut u = vec![0.0f64; m * r];
        for i in 0..m {
            for j in 0..r {
                u[i * r + j] = svd.u[i * k + j];
            }
        }
        Ok(u)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn fro_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Build a rank-1 tensor: outer product a ⊗ b ⊗ c.
    fn rank1_tensor(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        let d0 = a.len();
        let d1 = b.len();
        let d2 = c.len();
        let mut t = vec![0.0f64; d0 * d1 * d2];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    t[(i * d1 + j) * d2 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        t
    }

    /// Build a random tensor via LcgRng with the given shape.
    fn random_tensor(shape: &[usize], seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        let total: usize = shape.iter().product();
        (0..total).map(|_| rng.next_normal()).collect()
    }

    // ── Test 1: rank-1 tensor reconstructs exactly with max_rank=[1,1,1] ─

    #[test]
    fn sthosvd_rank1_reconstructs_exactly() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let c = [6.0, 7.0, 8.0, 9.0];
        let tensor = rank1_tensor(&a, &b, &c);
        let shape = [3usize, 2, 4];

        let config = SthosvdConfig {
            max_rank: vec![1, 1, 1],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("sthosvd should succeed");
        let rec = sthosvd_reconstruct(&result).expect("reconstruct should succeed");

        let err = fro_diff(&tensor, &rec);
        assert!(
            err < 1e-8,
            "Rank-1 reconstruction error {err:.2e} should be < 1e-8"
        );
    }

    // ── Test 2: core shape matches configured max_rank ────────────────────

    #[test]
    fn sthosvd_output_core_shape() {
        let shape = [4usize, 5, 6];
        let tensor = random_tensor(&shape, 11);
        let config = SthosvdConfig {
            max_rank: vec![2, 3, 4],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("ok");
        assert_eq!(result.core_shape, vec![2, 3, 4]);
        assert_eq!(result.core.len(), 2 * 3 * 4);
    }

    // ── Test 3: factor shapes match (d_k, r_k) ────────────────────────────

    #[test]
    fn sthosvd_factor_shapes() {
        let shape = [4usize, 5, 6];
        let tensor = random_tensor(&shape, 22);
        let config = SthosvdConfig {
            max_rank: vec![2, 3, 4],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("ok");
        assert_eq!(result.factor_shapes[0], (4, 2));
        assert_eq!(result.factor_shapes[1], (5, 3));
        assert_eq!(result.factor_shapes[2], (6, 4));
        assert_eq!(result.factors[0].len(), 4 * 2);
        assert_eq!(result.factors[1].len(), 5 * 3);
        assert_eq!(result.factors[2].len(), 6 * 4);
    }

    // ── Test 4: factor matrices are column-orthonormal ────────────────────

    #[test]
    fn sthosvd_factors_orthonormal() {
        let shape = [5usize, 6, 7];
        let tensor = random_tensor(&shape, 33);
        let config = SthosvdConfig {
            max_rank: vec![3, 4, 5],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("ok");

        for k in 0..result.n_modes {
            let (d_k, r_k) = result.factor_shapes[k];
            let u = &result.factors[k]; // (d_k, r_k) row-major
            // Check U_k^T U_k ≈ I_{r_k}
            for i in 0..r_k {
                for j in 0..r_k {
                    let dot: f64 = (0..d_k)
                        .map(|row| u[row * r_k + i] * u[row * r_k + j])
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-9,
                        "mode {k}: U^T U [{i},{j}] = {dot:.2e}, expected {expected}"
                    );
                }
            }
        }
    }

    // ── Test 5: reconstruction error < original tensor norm ───────────────

    #[test]
    fn sthosvd_reconstruct_decreases_error() {
        let shape = [6usize, 7, 8];
        let tensor = random_tensor(&shape, 44);
        let config = SthosvdConfig {
            max_rank: vec![2, 2, 2],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("ok");
        let rec = sthosvd_reconstruct(&result).expect("reconstruct ok");

        let err = fro_diff(&tensor, &rec);
        let norm = fro_norm(&tensor);
        assert!(
            err < norm,
            "Reconstruction error {err:.4} should be less than tensor norm {norm:.4}"
        );
    }

    // ── Test 6: N=2 case (matrix) matches truncated SVD reconstruction ────

    #[test]
    fn sthosvd_2mode_tensor() {
        // For N=2, ST-HOSVD should give a good low-rank approximation of a matrix.
        // We build a rank-2 matrix and check near-zero reconstruction error with rank 2.
        let m = 6usize;
        let n = 8usize;
        let mut rng = LcgRng::new(55);
        // Rank-2 matrix: a1 * b1^T + a2 * b2^T
        let a1: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let a2: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let b1: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let b2: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let mut matrix = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                matrix[i * n + j] = a1[i] * b1[j] + a2[i] * b2[j];
            }
        }

        let config = SthosvdConfig {
            max_rank: vec![2, 2],
            ..Default::default()
        };
        let result = sthosvd(&matrix, &[m, n], &config).expect("ok");
        let rec = sthosvd_reconstruct(&result).expect("ok");

        let err = fro_diff(&matrix, &rec);
        assert!(
            err < 1e-8,
            "Rank-2 matrix reconstruction error {err:.2e} should be < 1e-8"
        );
    }

    // ── Test 7: full rank recovers tensor exactly ─────────────────────────

    #[test]
    fn sthosvd_large_rank_equals_full() {
        let shape = [3usize, 4, 5];
        let tensor = random_tensor(&shape, 66);

        // max_rank >= each dimension: should reconstruct exactly
        let config = SthosvdConfig {
            max_rank: vec![3, 4, 5],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("ok");
        let rec = sthosvd_reconstruct(&result).expect("ok");

        let err = fro_diff(&tensor, &rec);
        assert!(
            err < 1e-8,
            "Full-rank reconstruction error {err:.2e} should be < 1e-8"
        );
    }

    // ── Test 8: ST-HOSVD error comparable to HOSVD error ─────────────────

    #[test]
    fn sthosvd_vs_hosvd_error_comparable() {
        use crate::tucker::hosvd::{hosvd, tucker_reconstruct};

        let d0 = 5usize;
        let d1 = 6usize;
        let d2 = 7usize;
        let tensor = random_tensor(&[d0, d1, d2], 77);

        let r0 = 3usize;
        let r1 = 3usize;
        let r2 = 3usize;

        // Standard HOSVD
        let hosvd_res = hosvd(&tensor, d0, d1, d2, r0, r1, r2).expect("hosvd ok");
        let hosvd_rec = tucker_reconstruct(&hosvd_res);
        let hosvd_err = fro_diff(&tensor, &hosvd_rec);

        // ST-HOSVD
        let config = SthosvdConfig {
            max_rank: vec![r0, r1, r2],
            ..Default::default()
        };
        let st_res = sthosvd(&tensor, &[d0, d1, d2], &config).expect("sthosvd ok");
        let st_rec = sthosvd_reconstruct(&st_res).expect("ok");
        let st_err = fro_diff(&tensor, &st_rec);

        // ST-HOSVD error should be within 2× of HOSVD error (both are good approximations)
        assert!(
            st_err < 2.0 * hosvd_err + 1e-8,
            "ST-HOSVD error {st_err:.4} should be within 2× of HOSVD error {hosvd_err:.4}"
        );
    }

    // ── Test 9: empty input returns error ─────────────────────────────────

    #[test]
    fn sthosvd_empty_returns_error() {
        let config = SthosvdConfig {
            max_rank: vec![2, 2],
            ..Default::default()
        };
        // Zero-length tensor
        let err = sthosvd(&[], &[3usize, 4], &config).unwrap_err();
        assert!(
            matches!(err, TnError::EmptyInput),
            "Expected EmptyInput, got {err:?}"
        );

        // Shape contains a zero dimension
        let config2 = SthosvdConfig {
            max_rank: vec![2, 2],
            ..Default::default()
        };
        let err2 = sthosvd(&[1.0, 2.0], &[0usize, 2], &config2).unwrap_err();
        assert!(
            matches!(err2, TnError::EmptyInput),
            "Expected EmptyInput from zero-dim, got {err2:?}"
        );
    }

    // ── Test 10: max_rank > d_k is clamped to d_k (no error) ─────────────

    #[test]
    fn sthosvd_rank_exceeds_dim_clamped_or_errored() {
        let shape = [3usize, 4, 5];
        let tensor = random_tensor(&shape, 88);

        // max_rank[0] = 10 > d_0 = 3: should clamp to 3 and succeed
        let config = SthosvdConfig {
            max_rank: vec![10, 10, 10],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("clamping should succeed");
        // Actual ranks used are clamped to min(max_rank[k], d_k)
        assert_eq!(result.core_shape[0], 3);
        assert_eq!(result.core_shape[1], 4);
        assert_eq!(result.core_shape[2], 5);

        // Verify reconstruction is exact (since we used full rank after clamping)
        let rec = sthosvd_reconstruct(&result).expect("ok");
        let err = fro_diff(&tensor, &rec);
        assert!(
            err < 1e-8,
            "Clamped full-rank error {err:.2e} should be < 1e-8"
        );
    }

    // ── Test 11: 4-mode tensor decomposition ─────────────────────────────

    #[test]
    fn sthosvd_4mode_tensor() {
        let shape = [3usize, 4, 5, 2];
        let tensor = random_tensor(&shape, 99);

        let config = SthosvdConfig {
            max_rank: vec![2, 3, 4, 2],
            ..Default::default()
        };
        let result = sthosvd(&tensor, &shape, &config).expect("4-mode sthosvd ok");
        assert_eq!(result.n_modes, 4);
        assert_eq!(result.core_shape, vec![2, 3, 4, 2]);
        assert_eq!(result.factors.len(), 4);

        // Verify orthonormality of all factor matrices
        for k in 0..4 {
            let (d_k, r_k) = result.factor_shapes[k];
            let u = &result.factors[k];
            for i in 0..r_k {
                for j in i..r_k {
                    let dot: f64 = (0..d_k)
                        .map(|row| u[row * r_k + i] * u[row * r_k + j])
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-8,
                        "mode {k}: U^T U [{i},{j}] = {dot:.2e}"
                    );
                }
            }
        }
    }

    // ── Test 12: unfold_mode + mode_product round-trips ──────────────────

    #[test]
    fn unfold_mode_product_consistency() {
        // Verify that mode-k unfolding layout is consistent with mode_product.
        // For a matrix (N=2), mode-0 unfolding should be the matrix itself.
        let m = 4usize;
        let n = 5usize;
        let mut rng = LcgRng::new(123);
        let matrix: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();

        let unfolded = unfold_mode(&matrix, &[m, n], 0);
        assert_eq!(unfolded.len(), m * n);
        // Mode-0 unfolding of a matrix should match the matrix in row-major order.
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (unfolded[i * n + j] - matrix[i * n + j]).abs() < 1e-14,
                    "unfold mode-0 mismatch at ({i},{j})"
                );
            }
        }

        // Mode-1 unfolding of a matrix should be its transpose.
        let unfolded1 = unfold_mode(&matrix, &[m, n], 1);
        for i in 0..n {
            for j in 0..m {
                assert!(
                    (unfolded1[i * m + j] - matrix[j * n + i]).abs() < 1e-14,
                    "unfold mode-1 (transpose) mismatch at ({i},{j})"
                );
            }
        }
    }

    // ── Test 13: randomised SVD path produces valid orthonormal factors ───

    #[test]
    fn sthosvd_rsvd_path_orthonormal() {
        let shape = [8usize, 10, 6];
        let tensor = random_tensor(&shape, 200);

        let config = SthosvdConfig {
            max_rank: vec![3, 4, 3],
            use_randomised_svd: true,
            rsvd_oversampling: 5,
            rsvd_seed: 7,
        };
        let result = sthosvd(&tensor, &shape, &config).expect("rsvd path ok");

        for k in 0..result.n_modes {
            let (d_k, r_k) = result.factor_shapes[k];
            let u = &result.factors[k];
            for i in 0..r_k {
                for j in i..r_k {
                    let dot: f64 = (0..d_k)
                        .map(|row| u[row * r_k + i] * u[row * r_k + j])
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-5,
                        "mode {k}: rsvd U^T U [{i},{j}] = {dot:.2e}"
                    );
                }
            }
        }
    }
}
