//! Non-negative CP / PARAFAC decomposition via Lee-Seung multiplicative update rules.
//!
//! Factorises a non-negative N-mode tensor T as:
//! ```text
//! T ≈ Σ_{r=1}^{R} λ_r · a_r^(1) ⊗ a_r^(2) ⊗ … ⊗ a_r^(N)
//! ```
//! where every factor matrix A^(n) ∈ ℝ^{d_n × R} has non-negative entries and
//! each weight λ_r ≥ 0.
//!
//! The multiplicative update rule for mode n is:
//! ```text
//! A^(n)  ←  A^(n)  ⊙  ( T_(n) Z  ⊘  ( A^(n) (Z^T Z) + ε ) )
//! ```
//! where Z is the Khatri-Rao product of all factor matrices except mode n,
//! T_(n) is the mode-n unfolding of T, and ⊙ / ⊘ are element-wise multiply/divide.
//! The rule guarantees non-negativity as long as the initialisation is non-negative.

use crate::handle::LcgRng;
use crate::{TnError, TnResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public structs
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for non-negative CP decomposition.
#[derive(Debug, Clone)]
pub struct NnCpConfig {
    /// Number of rank-1 components (R).
    pub n_components: usize,
    /// Maximum number of ALS sweeps.
    pub max_iter: usize,
    /// Convergence tolerance on relative change of Frobenius residual.
    pub tol: f64,
    /// RNG seed for factor initialisation.
    pub seed: u64,
    /// Additive floor ε in the denominator of the multiplicative update.
    pub epsilon: f64,
}

impl Default for NnCpConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            tol: 1e-6,
            seed: 42,
            epsilon: 1e-10,
        }
    }
}

/// Result of a non-negative CP decomposition.
#[derive(Debug, Clone)]
pub struct NnCpResult {
    /// One factor matrix per mode, each stored row-major as a flat `Vec<f64>`.
    pub factors: Vec<Vec<f64>>,
    /// `(d_n, R)` for every mode n.
    pub factor_shapes: Vec<(usize, usize)>,
    /// Non-negative column weights λ_r, length R.
    pub weights: Vec<f64>,
    /// Frobenius residual ‖T − T̂‖_F at termination.
    pub residual: f64,
    /// Number of iterations executed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Number of modes N.
    pub n_modes: usize,
    /// Number of components R.
    pub n_components: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Decompose a non-negative tensor `tensor` (shape `shape`) into a non-negative
/// CP model using Lee-Seung multiplicative update rules.
///
/// # Errors
/// - [`TnError::EmptyInput`] if the tensor or any dimension is zero.
/// - [`TnError::ShapeMismatch`] if `tensor.len() ≠ Π shape[n]`.
/// - [`TnError::InvalidParameter`] if `n_components == 0`.
pub fn nn_cp_decomp(tensor: &[f64], shape: &[usize], config: &NnCpConfig) -> TnResult<NnCpResult> {
    // ── Validation ──────────────────────────────────────────────────────────
    if tensor.is_empty() || shape.is_empty() {
        return Err(TnError::EmptyInput);
    }
    for &d in shape {
        if d == 0 {
            return Err(TnError::EmptyInput);
        }
    }
    let total: usize = shape.iter().product();
    if tensor.len() != total {
        return Err(TnError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![tensor.len()],
        });
    }
    if config.n_components == 0 {
        return Err(TnError::InvalidParameter {
            name: "n_components".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }

    let n_modes = shape.len();
    let r = config.n_components;
    let eps = config.epsilon;

    // ── Initialise factors with Uniform([0, 1]) via LCG ─────────────────────
    let mut rng = LcgRng::new(config.seed);
    let mut factors: Vec<Vec<f64>> = shape
        .iter()
        .map(|&d| (0..d * r).map(|_| rng.next_f64() + eps).collect())
        .collect();

    // Pre-compute all mode-n unfoldings once (they are fixed throughout).
    let unfoldings: Vec<Vec<f64>> = (0..n_modes)
        .map(|n| mode_n_unfold(tensor, shape, n))
        .collect();

    let mut prev_residual = f64::INFINITY;
    let mut n_iter = 0usize;
    let mut converged = false;

    // ── Main iteration ───────────────────────────────────────────────────────
    for it in 0..config.max_iter {
        n_iter = it + 1;

        for n in 0..n_modes {
            // Build Khatri-Rao product Z of all factors except mode n.
            // Z has shape (total / shape[n], r).
            let z = khatri_rao_all_except(&factors, shape, n, r);
            let d_n = shape[n];
            let z_rows = total / d_n; // = Π_{m ≠ n} d_m

            // Gram matrix G = Z^T Z  (r × r).
            let gram = gram_matrix(&z, z_rows, r);

            // Numerator  N_mat = T_(n) · Z          shape (d_n, r).
            let num = matmul_rect(&unfoldings[n], d_n, z_rows, &z, z_rows, r);

            // Denominator D_mat = A^(n) · G         shape (d_n, r).
            let a_n = &factors[n];
            let den = matmul_rect(a_n, d_n, r, &gram, r, r);

            // Multiplicative update: A^(n) ← A^(n) ⊙ (num ⊘ (den + ε)).
            let new_factor: Vec<f64> = a_n
                .iter()
                .zip(num.iter())
                .zip(den.iter())
                .map(|((&old, &n_val), &d_val)| (old * n_val / (d_val + eps)).max(0.0))
                .collect();
            factors[n] = new_factor;
        }

        // ── Convergence check ────────────────────────────────────────────────
        let residual = frobenius_residual(tensor, &factors, shape, r);
        let rel_change = if prev_residual.is_finite() && prev_residual > 0.0 {
            (prev_residual - residual).abs() / (prev_residual + eps)
        } else {
            f64::INFINITY
        };
        prev_residual = residual;
        if rel_change < config.tol && it > 0 {
            converged = true;
            break;
        }
    }

    // ── Extract column norms as weights, then normalise factors ──────────────
    let mut weights = vec![1.0f64; r];
    for (n, factor) in factors.iter_mut().enumerate() {
        let d_n = shape[n];
        for col in 0..r {
            let sq: f64 = (0..d_n).map(|row| factor[row * r + col].powi(2)).sum();
            let nrm = sq.sqrt();
            if nrm > 1e-300 {
                for row in 0..d_n {
                    factor[row * r + col] /= nrm;
                }
                weights[col] *= nrm;
            }
        }
    }
    // Clamp tiny/negative weights to zero (should not occur, but guard anyway).
    for w in &mut weights {
        if *w < 0.0 {
            *w = 0.0;
        }
    }

    let factor_shapes: Vec<(usize, usize)> = shape.iter().map(|&d| (d, r)).collect();
    let residual = frobenius_residual_weighted(tensor, &factors, &weights, shape, r);

    Ok(NnCpResult {
        factors,
        factor_shapes,
        weights,
        residual,
        n_iter,
        converged,
        n_modes,
        n_components: r,
    })
}

/// Reconstruct the approximated tensor from `NnCpResult`.
///
/// The reconstructed tensor at multi-index (i_1, …, i_N) is:
/// ```text
/// T̂[i_1, …, i_N] = Σ_r λ_r · A^(1)[i_1, r] · A^(2)[i_2, r] · … · A^(N)[i_N, r]
/// ```
///
/// # Errors
/// - [`TnError::ShapeMismatch`] if `shape` is inconsistent with `result`.
pub fn nn_cp_reconstruct(result: &NnCpResult, shape: &[usize]) -> TnResult<Vec<f64>> {
    let n_modes = shape.len();
    if n_modes != result.n_modes {
        return Err(TnError::ShapeMismatch {
            expected: result.factor_shapes.iter().map(|&(d, _)| d).collect(),
            got: shape.to_vec(),
        });
    }
    for (n, &d) in shape.iter().enumerate() {
        let (exp_d, _) = result.factor_shapes[n];
        if d != exp_d {
            return Err(TnError::ShapeMismatch {
                expected: result.factor_shapes.iter().map(|&(d, _)| d).collect(),
                got: shape.to_vec(),
            });
        }
    }

    let total: usize = shape.iter().product();
    let r = result.n_components;
    let mut out = vec![0.0f64; total];
    let mut multi_idx = vec![0usize; n_modes];

    for (flat, out_elem) in out.iter_mut().enumerate() {
        decode_multi_idx(flat, shape, &mut multi_idx);
        let mut val = 0.0;
        for comp in 0..r {
            let prod = result.weights[comp]
                * multi_idx
                    .iter()
                    .zip(result.factors.iter())
                    .map(|(&idx, fac)| fac[idx * r + comp])
                    .product::<f64>();
            val += prod;
        }
        *out_elem = val;
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Decode a flat row-major index into a multi-index vector (in-place).
#[inline]
fn decode_multi_idx(flat: usize, shape: &[usize], multi_idx: &mut [usize]) {
    let mut rem = flat;
    for (idx, &d) in multi_idx.iter_mut().zip(shape.iter()).rev() {
        *idx = rem % d;
        rem /= d;
    }
}

/// Mode-n unfolding of `tensor` with shape `shape`.
///
/// Returns a `(shape[n], total / shape[n])` matrix stored row-major.
/// Column index corresponds to multi-index `(i_{N-1}, …, i_{n+1}, i_{n-1}, …, i_0)`
/// in the Kolda–Bader convention — matching the Khatri-Rao product ordering.
fn mode_n_unfold(tensor: &[f64], shape: &[usize], mode: usize) -> Vec<f64> {
    let n_modes = shape.len();
    let d_n = shape[mode];
    let total: usize = shape.iter().product();
    let cols = total / d_n;
    let mut out = vec![0.0f64; d_n * cols];

    // Build strides for the column index.
    //
    // The column ordering must MATCH the row ordering of the Khatri-Rao matrix produced
    // by `khatri_rao_all_except`.  That function processes modes in the order
    // (N-1, N-2, …, mode+1, mode-1, …, 0) and makes the FIRST listed mode the most
    // significant (outer) index of its output rows.  To match this, assign strides
    // by sweeping the same `order` list in REVERSE so that `order[0]` (the first /
    // most-significant index) receives the LARGEST stride.
    let mut strides_col = vec![1usize; n_modes];
    {
        let mut stride = 1usize;
        let order: Vec<usize> = (0..n_modes).filter(|&m| m != mode).rev().collect();
        // Sweep order in reverse: last entry gets stride 1, next gets shape[last], etc.
        for &m in order.iter().rev() {
            strides_col[m] = stride;
            stride *= shape[m];
        }
    }

    let mut multi_idx = vec![0usize; n_modes];
    for (flat, &t_val) in tensor.iter().enumerate() {
        decode_multi_idx(flat, shape, &mut multi_idx);
        let row = multi_idx[mode];
        let col: usize = multi_idx
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != mode)
            .map(|(m, &idx)| idx * strides_col[m])
            .sum();
        out[row * cols + col] = t_val;
    }
    out
}

/// Khatri-Rao product of all factor matrices except mode `skip`.
///
/// The product order (for consistency with the mode-n unfolding column ordering)
/// is `A^(N-1) ⊙ … ⊙ A^(skip+1) ⊙ A^(skip-1) ⊙ … ⊙ A^(0)`.
///
/// Returns a `(total / shape[skip], r)` matrix stored row-major.
fn khatri_rao_all_except(factors: &[Vec<f64>], shape: &[usize], skip: usize, r: usize) -> Vec<f64> {
    let n_modes = shape.len();
    // Column order in unfolding: (N-1, …, skip+1, skip-1, …, 0) — reverse, skip mode.
    let order: Vec<usize> = (0..n_modes).filter(|&m| m != skip).rev().collect();

    if order.is_empty() {
        // Single-mode tensor: Khatri-Rao is a (1, r) identity-like matrix.
        return vec![1.0f64; r];
    }

    // Start with the first factor in the order.
    let first_mode = order[0];
    let mut z: Vec<f64> = factors[first_mode].clone();
    let mut z_rows = shape[first_mode];

    // Fold in remaining modes one at a time.
    for &m in &order[1..] {
        let a = &factors[m];
        let a_rows = shape[m];
        z = khatri_rao_pair(&z, z_rows, a, a_rows, r);
        z_rows *= a_rows;
    }
    z
}

/// Khatri-Rao (column-wise Kronecker) product of two matrices.
///
/// `A` has shape `(m_a, r)`, `B` has shape `(m_b, r)`.
/// Result has shape `(m_a * m_b, r)` with `out[i * m_b + j, c] = A[i, c] * B[j, c]`.
fn khatri_rao_pair(a: &[f64], m_a: usize, b: &[f64], m_b: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m_a * m_b * r];
    for i in 0..m_a {
        for j in 0..m_b {
            for c in 0..r {
                out[(i * m_b + j) * r + c] = a[i * r + c] * b[j * r + c];
            }
        }
    }
    out
}

/// Gram matrix G = Z^T Z  of shape `(r, r)`.
fn gram_matrix(z: &[f64], z_rows: usize, r: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut acc = 0.0;
            for k in 0..z_rows {
                acc += z[k * r + i] * z[k * r + j];
            }
            g[i * r + j] = acc;
        }
    }
    g
}

/// Dense matrix multiplication: `A (m × k) · B (k × n) → C (m × n)`.
/// All matrices stored row-major.
fn matmul_rect(a: &[f64], m: usize, k: usize, b: &[f64], k2: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(k, k2, "matmul_rect inner dimension mismatch");
    let _ = k2;
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    out
}

/// Evaluate T̂[multi_idx] = Σ_comp prod_n factors[n][multi_idx[n]*r + comp].
#[inline]
fn eval_approx_unweighted(multi_idx: &[usize], factors: &[Vec<f64>], r: usize) -> f64 {
    (0..r)
        .map(|comp| {
            multi_idx
                .iter()
                .zip(factors.iter())
                .map(|(&idx, fac)| fac[idx * r + comp])
                .product::<f64>()
        })
        .sum()
}

/// Evaluate T̂[multi_idx] = Σ_comp weights[comp] * prod_n factors[n][multi_idx[n]*r + comp].
#[inline]
fn eval_approx_weighted(
    multi_idx: &[usize],
    factors: &[Vec<f64>],
    weights: &[f64],
    r: usize,
) -> f64 {
    (0..r)
        .map(|comp| {
            weights[comp]
                * multi_idx
                    .iter()
                    .zip(factors.iter())
                    .map(|(&idx, fac)| fac[idx * r + comp])
                    .product::<f64>()
        })
        .sum()
}

/// Frobenius residual ‖T − T̂‖_F where weights are all 1 (used during iteration).
fn frobenius_residual(tensor: &[f64], factors: &[Vec<f64>], shape: &[usize], r: usize) -> f64 {
    let n_modes = shape.len();
    let mut multi_idx = vec![0usize; n_modes];
    tensor
        .iter()
        .enumerate()
        .map(|(flat, &t_val)| {
            decode_multi_idx(flat, shape, &mut multi_idx);
            let approx = eval_approx_unweighted(&multi_idx, factors, r);
            let d = t_val - approx;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Frobenius residual ‖T − T̂‖_F with explicit weights λ.
fn frobenius_residual_weighted(
    tensor: &[f64],
    factors: &[Vec<f64>],
    weights: &[f64],
    shape: &[usize],
    r: usize,
) -> f64 {
    let n_modes = shape.len();
    let mut multi_idx = vec![0usize; n_modes];
    tensor
        .iter()
        .enumerate()
        .map(|(flat, &t_val)| {
            decode_multi_idx(flat, shape, &mut multi_idx);
            let approx = eval_approx_weighted(&multi_idx, factors, weights, r);
            let d = t_val - approx;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a rank-1 non-negative tensor from three vectors ─────────
    fn rank1_tensor(a: &[f64], b: &[f64], c: &[f64]) -> (Vec<f64>, [usize; 3]) {
        let shape = [a.len(), b.len(), c.len()];
        let mut t = vec![0.0f64; a.len() * b.len() * c.len()];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                for (k, &ck) in c.iter().enumerate() {
                    t[(i * b.len() + j) * c.len() + k] = ai * bj * ck;
                }
            }
        }
        (t, shape)
    }

    // ── 1. All factor entries must be non-negative after fitting ──────────────
    #[test]
    fn nncp_factors_non_negative() {
        let (t, shape) = rank1_tensor(&[1.0, 2.0, 3.0], &[0.5, 1.5], &[2.0, 1.0, 0.5]);
        let cfg = NnCpConfig {
            n_components: 2,
            max_iter: 100,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        for (n, factor) in res.factors.iter().enumerate() {
            for &v in factor {
                assert!(v >= 0.0, "factor[{n}] has negative entry {v}");
            }
        }
    }

    // ── 2. Weights must be non-negative ───────────────────────────────────────
    #[test]
    fn nncp_weights_non_negative() {
        let (t, shape) = rank1_tensor(&[1.0, 0.5], &[2.0, 1.0, 3.0], &[0.1, 4.0]);
        let cfg = NnCpConfig {
            n_components: 2,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        for (r, &w) in res.weights.iter().enumerate() {
            assert!(w >= 0.0, "weight[{r}] = {w} is negative");
        }
    }

    // ── 3. Reconstructed tensor has the correct shape (length) ────────────────
    #[test]
    fn nncp_reconstruct_shape_correct() {
        let shape = [3usize, 4, 5];
        let t = vec![0.5f64; shape.iter().product()];
        let cfg = NnCpConfig {
            n_components: 2,
            max_iter: 50,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        let recon = nn_cp_reconstruct(&res, &shape).expect("recon ok");
        assert_eq!(recon.len(), shape.iter().product::<usize>());
    }

    // ── 4. Low residual on an exact rank-1 non-negative tensor ────────────────
    #[test]
    fn nncp_low_residual_on_rank1_non_negative() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [0.5f64, 1.0, 2.0];
        let c = [4.0f64, 1.0];
        let (t, shape) = rank1_tensor(&a, &b, &c);
        let cfg = NnCpConfig {
            n_components: 1,
            max_iter: 500,
            tol: 1e-9,
            seed: 7,
            epsilon: 1e-10,
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        assert!(res.residual < 0.01, "residual too large: {}", res.residual);
    }

    // ── 5. N=2 (matrix) non-negative factorisation ────────────────────────────
    #[test]
    fn nncp_2mode_matrix_factorization() {
        // Build a rank-2 non-negative matrix: M = u1 v1^T + u2 v2^T.
        let u1 = [1.0f64, 2.0, 3.0];
        let v1 = [0.5f64, 1.0, 1.5, 2.0];
        let u2 = [0.1f64, 0.4, 0.9];
        let v2 = [2.0f64, 1.0, 0.5, 0.25];
        let rows = u1.len();
        let cols = v1.len();
        let mut m = vec![0.0f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                m[i * cols + j] = u1[i] * v1[j] + u2[i] * v2[j];
            }
        }
        let shape = [rows, cols];
        let cfg = NnCpConfig {
            n_components: 2,
            max_iter: 400,
            tol: 1e-8,
            seed: 17,
            epsilon: 1e-10,
        };
        let res = nn_cp_decomp(&m, &shape, &cfg).expect("decomp ok");
        assert!(
            res.residual < 0.5,
            "matrix factorisation residual too large: {}",
            res.residual
        );
    }

    // ── 6. factors.len() == N ─────────────────────────────────────────────────
    #[test]
    fn nncp_result_factors_count() {
        let shape = [2usize, 3, 4, 5];
        let t = vec![1.0f64; shape.iter().product()];
        let cfg = NnCpConfig {
            n_components: 3,
            max_iter: 30,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        assert_eq!(res.factors.len(), shape.len());
        assert_eq!(res.n_modes, shape.len());
    }

    // ── 7. factor_shapes[n] == (d_n, R) ──────────────────────────────────────
    #[test]
    fn nncp_factor_shapes_correct() {
        let shape = [3usize, 5, 7];
        let t = vec![0.25f64; shape.iter().product()];
        let r = 4usize;
        let cfg = NnCpConfig {
            n_components: r,
            max_iter: 30,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        for (n, &d_n) in shape.iter().enumerate() {
            assert_eq!(
                res.factor_shapes[n],
                (d_n, r),
                "factor_shapes[{n}] mismatch"
            );
            assert_eq!(
                res.factors[n].len(),
                d_n * r,
                "factors[{n}] length mismatch"
            );
        }
    }

    // ── 8. weights.len() == n_components ─────────────────────────────────────
    #[test]
    fn nncp_weights_length() {
        let shape = [4usize, 6];
        let t = vec![1.0f64; shape.iter().product()];
        let r = 5usize;
        let cfg = NnCpConfig {
            n_components: r,
            max_iter: 20,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        assert_eq!(res.weights.len(), r);
        assert_eq!(res.n_components, r);
    }

    // ── 9. Empty tensor returns TnError::EmptyInput ───────────────────────────
    #[test]
    fn nncp_empty_tensor_returns_error() {
        let t: Vec<f64> = vec![];
        let shape: &[usize] = &[0, 3, 2];
        let cfg = NnCpConfig::default();
        let err = nn_cp_decomp(&t, shape, &cfg).expect_err("should fail");
        assert!(
            matches!(err, TnError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    // ── 10. Zero rank returns TnError::InvalidParameter ──────────────────────
    #[test]
    fn nncp_zero_rank_returns_error() {
        let t = vec![1.0f64; 6];
        let shape = [2usize, 3];
        let cfg = NnCpConfig {
            n_components: 0,
            ..Default::default()
        };
        let err = nn_cp_decomp(&t, &shape, &cfg).expect_err("should fail");
        assert!(
            matches!(err, TnError::InvalidParameter { .. }),
            "expected InvalidParameter, got {err:?}"
        );
    }

    // ── 11. Reconstruct returns error on shape mismatch ───────────────────────
    #[test]
    fn nncp_reconstruct_shape_mismatch_error() {
        let (t, shape) = rank1_tensor(&[1.0, 2.0], &[1.0, 3.0], &[0.5, 1.5, 2.5]);
        let cfg = NnCpConfig {
            n_components: 1,
            max_iter: 50,
            ..Default::default()
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        let bad_shape = [2usize, 3]; // wrong number of modes
        let err = nn_cp_reconstruct(&res, &bad_shape).expect_err("should fail");
        assert!(matches!(err, TnError::ShapeMismatch { .. }));
    }

    // ── 12. Reconstruction is close to original for rank-1 tensor ─────────────
    #[test]
    fn nncp_reconstruct_close_rank1() {
        let a = [2.0f64, 1.0, 0.5];
        let b = [1.0f64, 3.0];
        let c = [0.5f64, 1.5, 2.5];
        let (t, shape) = rank1_tensor(&a, &b, &c);
        let cfg = NnCpConfig {
            n_components: 1,
            max_iter: 600,
            tol: 1e-9,
            seed: 13,
            epsilon: 1e-10,
        };
        let res = nn_cp_decomp(&t, &shape, &cfg).expect("decomp ok");
        let recon = nn_cp_reconstruct(&res, &shape).expect("recon ok");
        let fro: f64 = t
            .iter()
            .zip(recon.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        assert!(fro < 0.05, "reconstruction error too large: {fro}");
    }
}
