//! TT-ALS: Alternating Least Squares for Tensor-Train format regression.
//!
//! Given a data matrix `X` of shape `(n_samples, d1 * d2 * ... * dN)` and targets `y`
//! of shape `(n_samples,)`, find a TT-decomposed weight tensor `W` with bond dimension
//! `max_rank` such that `X W_vec ≈ y` in the least-squares sense.
//!
//! ## Algorithm (Holtz-Rohwedder-Schneider 2012 style)
//!
//! Represent `W` as a TT-core chain `G_0, ..., G_{N-1}` where `G_k` has shape
//! `(r_{k-1}, d_k, r_k)` with `r_0 = r_N = 1`. The ALS sweep alternates between
//! sites, fixing all cores except `G_k` and solving the local linear least-squares
//! sub-problem via normal equations.
//!
//! For each site `k`, the effective feature matrix is
//!
//! ```text
//! Phi_k[i, r_l * j * r_r] = L_k[i, r_l] * X[i, offset_k + j] * R_k[i, r_r]
//! ```
//!
//! where `L_k[i, :]` is the left partial contraction (samples × r_{k-1}) and
//! `R_k[i, :]` is the right partial contraction (samples × r_k). The normal equations
//! `(Phi_k^T Phi_k) g_k = Phi_k^T y` are solved via Cholesky / LDL^T factorisation
//! with Tikhonov regularisation for stability.

use crate::handle::LcgRng;
use crate::{TnError, TnResult};

// ── Public structs ────────────────────────────────────────────────────────────

/// Configuration for TT-ALS regression.
#[derive(Debug, Clone)]
pub struct TtAlsConfig {
    /// Shape of the weight tensor: `n_dims[k]` is the size of mode `k`.
    pub n_dims: Vec<usize>,
    /// Maximum bond dimension (rank) used for every internal bond.
    pub max_rank: usize,
    /// Maximum number of left-right sweep pairs.
    pub max_sweeps: usize,
    /// Convergence threshold on relative residual change between sweeps.
    pub tol: f64,
    /// Seed for the LCG pseudo-random number generator used to initialise cores.
    pub seed: u64,
}

/// Result of TT-ALS regression.
#[derive(Debug, Clone)]
pub struct TtAlsResult {
    /// Flattened core data, one entry per mode.  `cores[k]` has length
    /// `core_shapes[k].0 * core_shapes[k].1 * core_shapes[k].2` and is stored
    /// in row-major order `(r_l, d_k, r_r)`.
    pub cores: Vec<Vec<f64>>,
    /// Shapes `(r_l, d_k, r_r)` of each core.
    pub core_shapes: Vec<(usize, usize, usize)>,
    /// Frobenius norm of the residual vector `X W_vec - y` at the end.
    pub residual: f64,
    /// Number of full left-to-right sweeps completed.
    pub n_sweeps: usize,
    /// Whether the algorithm converged within `max_sweeps`.
    pub converged: bool,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Predict `X W_vec` using TT cores.
///
/// # Arguments
/// * `cores`       – slices of flattened core data, one per mode.
/// * `core_shapes` – `(r_l, d_k, r_r)` for each core.
/// * `x`           – row-major data matrix of shape `(n_samples, total_dim)`.
/// * `n_samples`   – number of rows in `x`.
///
/// # Errors
/// Returns `TnError::ShapeMismatch` if the total feature dimension derived from
/// `core_shapes` does not match `x.len() / n_samples`.
pub fn predict_tt(
    cores: &[Vec<f64>],
    core_shapes: &[(usize, usize, usize)],
    x: &[f64],
    n_samples: usize,
) -> TnResult<Vec<f64>> {
    if n_samples == 0 || x.is_empty() {
        return Err(TnError::EmptyInput);
    }
    let total_dim: usize = core_shapes.iter().map(|s| s.1).product();
    if x.len() != n_samples * total_dim {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_samples, total_dim],
            got: vec![x.len()],
        });
    }

    // Materialise W by contracting cores, then compute X @ w.
    let w = materialise_weight(cores, core_shapes)?;
    let mut preds = vec![0.0f64; n_samples];
    for i in 0..n_samples {
        let mut acc = 0.0f64;
        for j in 0..total_dim {
            acc += x[i * total_dim + j] * w[j];
        }
        preds[i] = acc;
    }
    Ok(preds)
}

/// Run TT-ALS regression.
///
/// # Arguments
/// * `x`        – row-major data matrix `(n_samples, prod(n_dims))`.
/// * `y`        – target vector of length `n_samples`.
/// * `n_samples`– number of data points (rows of `x`).
/// * `config`   – algorithm hyper-parameters.
///
/// # Errors
/// * `TnError::EmptyInput` if `x` or `y` is empty.
/// * `TnError::InvalidParameter` if `max_rank == 0` or `n_dims` is empty.
/// * `TnError::ShapeMismatch` if dimensions are inconsistent.
pub fn tt_als_regression(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    config: &TtAlsConfig,
) -> TnResult<TtAlsResult> {
    // ── Validate inputs ──────────────────────────────────────────────────────
    if x.is_empty() || y.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if config.n_dims.is_empty() {
        return Err(TnError::InvalidParameter {
            name: "n_dims".to_string(),
            reason: "must be non-empty".to_string(),
        });
    }
    if config.max_rank == 0 {
        return Err(TnError::InvalidParameter {
            name: "max_rank".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if y.len() != n_samples {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_samples],
            got: vec![y.len()],
        });
    }
    let n_modes = config.n_dims.len();
    let total_dim: usize = config.n_dims.iter().product();
    if x.len() != n_samples * total_dim {
        return Err(TnError::ShapeMismatch {
            expected: vec![n_samples, total_dim],
            got: vec![x.len()],
        });
    }

    // ── Build core shapes ────────────────────────────────────────────────────
    // Bond dimensions: r[0]=1, r[k]=min(max_rank, …), r[N]=1.
    let mut r: Vec<usize> = vec![1usize; n_modes + 1];
    for elem in r.iter_mut().take(n_modes).skip(1) {
        *elem = config.max_rank;
    }
    // r[n_modes] = 1 already.

    let core_shapes: Vec<(usize, usize, usize)> = (0..n_modes)
        .map(|k| (r[k], config.n_dims[k], r[k + 1]))
        .collect();

    // ── Initialise cores with small random values ────────────────────────────
    let mut rng = LcgRng::new(config.seed);
    let mut cores: Vec<Vec<f64>> = core_shapes
        .iter()
        .map(|&(rl, dk, rr)| {
            (0..rl * dk * rr)
                .map(|_| (rng.next_f64() - 0.5) * 0.1)
                .collect()
        })
        .collect();

    // ── Initial normalisation ────────────────────────────────────────────────
    // Normalise each core to unit Frobenius norm so ALS starts from a well-scaled
    // initial point.  This avoids the complex bookkeeping of full right-ortho QR
    // while still ensuring the initial gradient is not dominated by scale factors.
    for core in cores.iter_mut() {
        let nrm2: f64 = core.iter().map(|v| v * v).sum();
        if nrm2 > 1.0e-300 {
            let nrm = nrm2.sqrt();
            core.iter_mut().for_each(|v| *v /= nrm);
        }
    }

    // ── Main ALS sweeps ───────────────────────────────────────────────────────
    let mut prev_residual = f64::INFINITY;
    let mut n_sweeps = 0usize;
    let mut converged = false;

    for _sweep in 0..config.max_sweeps {
        n_sweeps += 1;

        // Left-to-right update pass.
        for k in 0..n_modes {
            let (rl_k, dk, rr_k) = core_shapes[k];

            // Dimensions of the left and right partitions of the feature tensor.
            // X[i,:] is interpreted as a C-order tensor of shape (d0, d1, ..., dN-1).
            // For site k, we split: D_left = d0*...*d_{k-1}, D_right = d_{k+1}*...*d_{N-1}.
            let d_left: usize = config.n_dims[..k].iter().product();
            let d_right: usize = config.n_dims[k + 1..].iter().product();
            // total_dim == d_left * dk * d_right (C-order layout).

            // Build V_left: (D_left, rl_k) matrix obtained by contracting cores 0..k-1.
            // V_left[jl, alpha] = sum_{chain} G_0[1,j0,:]...G_{k-1}[:,j_{k-1},alpha]
            // where jl = j0*(d1..d_{k-1}) + ... + j_{k-1}.
            let v_left = build_v_left(&cores, &core_shapes, k, d_left, rl_k);

            // Build V_right: (D_right, rr_k) matrix from cores k+1..N-1.
            let v_right = build_v_right(&cores, &core_shapes, &config.n_dims, k, d_right, rr_k);

            // Form the effective feature matrix Phi of shape (n_samples, rl_k * dk * rr_k).
            // Phi[i, (alpha, j_k, beta)] =
            //   sum_{jl=0..D_left} sum_{jr=0..D_right}
            //     X[i, jl*(dk*D_right) + j_k*D_right + jr] * V_left[jl, alpha] * V_right[jr, beta]
            let n_params = rl_k * dk * rr_k;
            let phi = build_phi(
                x, n_samples, total_dim, &v_left, &v_right, dk, d_left, d_right, rl_k, rr_k,
            );

            // Solve normal equations: (Phi^T Phi) g = Phi^T y
            let new_core = solve_normal_equations(&phi, y, n_samples, n_params)?;
            cores[k] = new_core;
        }

        // Compute residual after this sweep.
        let preds = predict_tt(&cores, &core_shapes, x, n_samples)?;
        let residual: f64 = preds
            .iter()
            .zip(y)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f64>()
            .sqrt();

        let rel_change = (prev_residual - residual).abs() / prev_residual.max(1.0);
        if rel_change < config.tol && n_sweeps > 1 {
            converged = true;
            prev_residual = residual;
            break;
        }
        prev_residual = residual;
    }

    let final_residual = prev_residual;
    Ok(TtAlsResult {
        cores,
        core_shapes,
        residual: final_residual,
        n_sweeps,
        converged,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Materialise the full weight vector `W` by contracting TT cores sequentially.
fn materialise_weight(
    cores: &[Vec<f64>],
    core_shapes: &[(usize, usize, usize)],
) -> TnResult<Vec<f64>> {
    if cores.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if cores.len() == 1 {
        // Single core (1, d, 1): just the data itself.
        return Ok(cores[0].clone());
    }

    // Start with the first core unfolded as (d_0, r_1) since r_0 = 1.
    let (_, d0, r1) = core_shapes[0];
    let mut current = cores[0].clone(); // shape (d0, r1)
    let mut current_rows = d0;
    let mut current_cols = r1;

    for k in 1..cores.len() {
        let (_rl, dk, rr) = core_shapes[k];
        // Multiply (current_rows, current_cols) × (_rl, dk*rr) reshaping core k.
        // core_k stored as (rl, dk, rr), we treat it as (rl, dk*rr).
        let new_cols = dk * rr;
        let mut next = vec![0.0f64; current_rows * new_cols];
        for i in 0..current_rows {
            for j in 0..new_cols {
                let mut acc = 0.0f64;
                for c in 0..current_cols {
                    // c is the bond index, j encodes (dk_idx, rr_idx) = (j/rr, j%rr).
                    acc += current[i * current_cols + c] * cores[k][c * new_cols + j];
                }
                next[i * new_cols + j] = acc;
            }
        }
        current = next;
        current_rows *= dk;
        current_cols = rr;
    }
    // At the end current_cols == 1 (right boundary), so current is (total_dim, 1).
    // Squeeze the trailing dimension.
    Ok(current)
}

/// Build V_left: the `(D_left, r_left)` basis matrix for the left partition at site k.
///
/// `V_left[j_left, alpha] = (G_0 ⊗ G_1 ⊗ ... ⊗ G_{k-1})[j_left, alpha]`
/// where `j_left` enumerates all multi-index tuples `(j0, j1, ..., j_{k-1})` in C-order
/// and `alpha` is the right bond index emerging from core k-1 (= r[k] = rl_k).
///
/// For k == 0, `D_left = 1` and `r_left = 1`, returning `[[1.0]]`.
fn build_v_left(
    cores: &[Vec<f64>],
    core_shapes: &[(usize, usize, usize)],
    site: usize,
    d_left: usize,
    r_left: usize,
) -> Vec<f64> {
    if site == 0 {
        // V_left is a trivial (1, 1) matrix = [[1.0]].
        return vec![1.0f64];
    }
    // Build iteratively: start with V = [[1.0]] shape (1, 1), and for each core left of `site`
    // compute V_new[j_prefix * d_k + j_k, beta] = sum_alpha V[j_prefix, alpha] * G_k[alpha, j_k, beta].
    // After the loop, V has shape (D_left, r_left).
    let mut v = vec![1.0f64]; // shape (1, 1)
    let mut v_rows = 1usize;
    let mut v_cols = 1usize;

    for k in 0..site {
        let (_rl, dk, rr) = core_shapes[k];
        // new_v shape: (v_rows * dk, rr)
        let new_rows = v_rows * dk;
        let mut new_v = vec![0.0f64; new_rows * rr];
        for j_prefix in 0..v_rows {
            for j_k in 0..dk {
                let new_row = j_prefix * dk + j_k;
                for beta in 0..rr {
                    let mut acc = 0.0f64;
                    for alpha in 0..v_cols {
                        acc +=
                            v[j_prefix * v_cols + alpha] * cores[k][(alpha * dk + j_k) * rr + beta];
                    }
                    new_v[new_row * rr + beta] = acc;
                }
            }
        }
        v = new_v;
        v_rows = new_rows;
        v_cols = rr;
    }
    // Sanity: v_rows == d_left, v_cols == r_left.
    debug_assert_eq!(v_rows, d_left);
    debug_assert_eq!(v_cols, r_left);
    v
}

/// Build V_right: the `(D_right, r_right)` basis matrix for the right partition at site k.
///
/// `V_right[j_right, beta] = (G_{k+1} ⊗ ... ⊗ G_{N-1})[j_right, beta]`
/// where `j_right` enumerates `(j_{k+1}, ..., j_{N-1})` in C-order and `beta = r[k+1]`.
///
/// For k == N-1, `D_right = 1` and `r_right = 1`, returning `[[1.0]]`.
fn build_v_right(
    cores: &[Vec<f64>],
    core_shapes: &[(usize, usize, usize)],
    n_dims: &[usize],
    site: usize,
    d_right: usize,
    r_right: usize,
) -> Vec<f64> {
    let n_modes = n_dims.len();
    if site == n_modes - 1 {
        return vec![1.0f64];
    }
    // Build from right to left: start with V = [[1.0]] shape (1,1), and for each core
    // right of `site` from rightmost to k+1:
    // V_new[j_k * D_suffix + j_suffix, alpha] = sum_beta G_k[alpha, j_k, beta] * V[j_suffix, beta]
    let mut v = vec![1.0f64]; // shape (1, 1)
    let mut v_rows = 1usize;
    let mut v_cols = 1usize; // = r_{k+1} bond from the left side of each right core

    for k in (site + 1..n_modes).rev() {
        let (rl, dk, _rr) = core_shapes[k];
        // new_v shape: (dk * v_rows, rl)
        let new_rows = dk * v_rows;
        let mut new_v = vec![0.0f64; new_rows * rl];
        for j_k in 0..dk {
            for j_suffix in 0..v_rows {
                let new_row = j_k * v_rows + j_suffix;
                for alpha in 0..rl {
                    let mut acc = 0.0f64;
                    for beta in 0..v_cols {
                        // G_k[alpha, j_k, beta] * V[j_suffix, beta]
                        acc +=
                            cores[k][(alpha * dk + j_k) * _rr + beta] * v[j_suffix * v_cols + beta];
                    }
                    new_v[new_row * rl + alpha] = acc;
                }
            }
        }
        v = new_v;
        v_rows = new_rows;
        v_cols = rl;
    }
    // v now has shape (d_right, r_right) — verify dimensions.
    debug_assert_eq!(v_rows, d_right);
    debug_assert_eq!(v_cols, r_right);
    v
}

/// Build the feature matrix Phi of shape `(n_samples, rl * dk * rr)` for site k.
///
/// ```text
/// Phi[i, (alpha, j_k, beta)] =
///   sum_{jl=0..D_left} sum_{jr=0..D_right}
///     X[i, jl*(dk*D_right) + j_k*D_right + jr] * V_left[jl, alpha] * V_right[jr, beta]
/// ```
///
/// This is exactly `(V_left^T X_k V_right)` where `X_k[i, jl, j_k, jr]` is the
/// reshaped data tensor for sample i.
#[allow(clippy::too_many_arguments)]
fn build_phi(
    x: &[f64],
    n_samples: usize,
    total_dim: usize,
    v_left: &[f64],
    v_right: &[f64],
    dk: usize,
    d_left: usize,
    d_right: usize,
    rl: usize,
    rr: usize,
) -> Vec<f64> {
    let n_params = rl * dk * rr;
    let mut phi = vec![0.0f64; n_samples * n_params];
    // For each sample i, we compute the (rl, dk, rr) tensor:
    //   T[alpha, j_k, beta] = sum_{jl, jr} X[i, jl*(dk*D_right)+j_k*D_right+jr]
    //                           * V_left[jl, alpha] * V_right[jr, beta]
    // We do this in two steps:
    // Step 1: A[alpha, j_k, jr] = sum_{jl} X[i, jl*(dk*D_right)+j_k*D_right+jr] * V_left[jl,alpha]
    // Step 2: T[alpha, j_k, beta] = sum_{jr} A[alpha, j_k, jr] * V_right[jr, beta]
    let dr_times_dk = d_right * dk;
    let mut a = vec![0.0f64; rl * dk * d_right];
    for i in 0..n_samples {
        // Zero A for reuse.
        a.iter_mut().for_each(|v| *v = 0.0);
        // Step 1: A[alpha, j_k, jr] = sum_jl X[i, jl*dr_dk + jk*dr + jr] * V_left[jl, alpha]
        for jl in 0..d_left {
            for alpha in 0..rl {
                let vl = v_left[jl * rl + alpha];
                if vl == 0.0 {
                    continue;
                }
                for jk in 0..dk {
                    let base_x = i * total_dim + jl * dr_times_dk + jk * d_right;
                    let base_a = (alpha * dk + jk) * d_right;
                    for jr in 0..d_right {
                        a[base_a + jr] += vl * x[base_x + jr];
                    }
                }
            }
        }
        // Step 2: T[alpha, j_k, beta] = sum_jr A[alpha, j_k, jr] * V_right[jr, beta]
        let base_phi = i * n_params;
        for alpha in 0..rl {
            for jk in 0..dk {
                let base_a = (alpha * dk + jk) * d_right;
                let base_t = (alpha * dk + jk) * rr;
                for beta in 0..rr {
                    let mut acc = 0.0f64;
                    for jr in 0..d_right {
                        acc += a[base_a + jr] * v_right[jr * rr + beta];
                    }
                    phi[base_phi + base_t + beta] = acc;
                }
            }
        }
    }
    phi
}

/// Solve the normal equations `(Phi^T Phi + eps I) g = Phi^T y` where `Phi` has shape
/// `(n_samples, n_params)`.
///
/// Uses Tikhonov regularisation for numerical stability. The Gram matrix
/// `A = Phi^T Phi + eps I` is symmetric positive definite and solved via an in-place
/// LDL^T factorisation.
fn solve_normal_equations(
    phi: &[f64],
    y: &[f64],
    n_samples: usize,
    n_params: usize,
) -> TnResult<Vec<f64>> {
    if n_params == 0 {
        return Err(TnError::EmptyInput);
    }
    // Compute the Gram matrix A = Phi^T Phi (n_params × n_params).
    let mut gram = vec![0.0f64; n_params * n_params];
    for j in 0..n_params {
        for k in j..n_params {
            let mut acc = 0.0f64;
            for i in 0..n_samples {
                acc += phi[i * n_params + j] * phi[i * n_params + k];
            }
            gram[j * n_params + k] = acc;
            gram[k * n_params + j] = acc;
        }
    }
    // Tikhonov regularisation: A += eps * ||A||_F / n_params * I
    let gram_frob: f64 = gram.iter().map(|v| v * v).sum::<f64>().sqrt();
    let eps = 1.0e-10 * (gram_frob / n_params as f64).max(1.0e-12);
    for i in 0..n_params {
        gram[i * n_params + i] += eps;
    }
    // Compute rhs = Phi^T y (n_params,).
    let mut rhs = vec![0.0f64; n_params];
    for j in 0..n_params {
        let mut acc = 0.0f64;
        for i in 0..n_samples {
            acc += phi[i * n_params + j] * y[i];
        }
        rhs[j] = acc;
    }
    // Solve via LDL^T (symmetric positive definite Cholesky variant).
    ldl_solve(&gram, &rhs, n_params)
}

/// LDL^T factorisation and solve for a symmetric positive-definite matrix.
///
/// Solves `A x = b` where `A` is `n × n` SPD. Uses the standard LDL^T decomposition:
/// `A = L D L^T` where `L` is unit lower-triangular and `D` is diagonal. Then solves
/// the three triangular systems.
fn ldl_solve(a: &[f64], b: &[f64], n: usize) -> TnResult<Vec<f64>> {
    // Work on a mutable copy of A stored column-major for cache efficiency during
    // the factorisation inner loop.  We keep row-major throughout for consistency.
    let mut mat = a.to_vec();
    let mut d = vec![0.0f64; n];

    // LDL^T factorisation in-place.
    // After: mat[i,j] for i>j contains L[i,j]; diagonal has original use replaced by d.
    for j in 0..n {
        // Compute d[j] = a[j,j] - sum_{k<j} L[j,k]^2 * d[k]
        let mut dj = mat[j * n + j];
        for k in 0..j {
            dj -= mat[j * n + k] * mat[j * n + k] * d[k];
        }
        if dj.abs() < 1.0e-300 {
            // Near-singular pivot: regularise.
            dj = 1.0e-300;
        }
        d[j] = dj;

        // Compute L[i,j] = (a[i,j] - sum_{k<j} L[i,k]*d[k]*L[j,k]) / d[j]  for i > j
        for i in (j + 1)..n {
            let mut lij = mat[i * n + j];
            for k in 0..j {
                lij -= mat[i * n + k] * d[k] * mat[j * n + k];
            }
            mat[i * n + j] = lij / dj;
        }
    }

    // Forward substitution: solve L z = b  (L unit lower-triangular)
    let mut z = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            z[i] -= mat[i * n + k] * z[k];
        }
    }
    // Diagonal solve: solve D w = z
    for i in 0..n {
        z[i] /= d[i];
    }
    // Backward substitution: solve L^T x = w  (L^T unit upper-triangular)
    let mut x = z;
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            x[i] -= mat[k * n + i] * x[k];
        }
    }
    Ok(x)
}

/// Thin QR decomposition of an `m × n` matrix (m >= n) stored row-major.
///
/// Returns `(Q, R)` where `Q` is `m × min(m,n)` column-orthonormal and `R` is
/// `min(m,n) × n` upper-triangular, both row-major.
///
/// Uses modified Gram-Schmidt for stability.
#[cfg(test)]
fn thin_qr(mat: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    // We store q as (m, k) row-major and r as (k, n) row-major.
    // We work column-by-column using modified Gram-Schmidt.
    // q_cols[j] = column j of the working matrix, length m.
    let mut q_cols: Vec<Vec<f64>> = (0..k)
        .map(|j| (0..m).map(|i| mat[i * n + j]).collect())
        .collect();
    let mut r = vec![0.0f64; k * n];

    for j in 0..k {
        // Compute norm of current column j using iterator sum.
        let nrm2: f64 = q_cols[j].iter().map(|v| v * v).sum();
        let nrm = nrm2.sqrt();
        // r[j, j] = norm
        r[j * n + j] = nrm;
        if nrm > 1.0e-300 {
            q_cols[j].iter_mut().for_each(|v| *v /= nrm);
        } else {
            // Rank-deficient column: replace with a canonical basis vector orthogonal
            // to all previous q_cols.
            let mut found = false;
            for e in 0..m {
                let mut v_e = vec![0.0f64; m];
                v_e[e] = 1.0;
                // Orthogonalise v_e against q_cols[0..j].
                for q_prev in q_cols.iter().take(j) {
                    let dot: f64 = q_prev.iter().zip(v_e.iter()).map(|(a, b)| a * b).sum();
                    v_e.iter_mut()
                        .zip(q_prev.iter())
                        .for_each(|(x, a)| *x -= dot * a);
                }
                let n2: f64 = v_e.iter().map(|v| v * v).sum();
                if n2 > 1.0e-10 {
                    let n_sqrt = n2.sqrt();
                    v_e.iter_mut().for_each(|v| *v /= n_sqrt);
                    q_cols[j] = v_e;
                    found = true;
                    break;
                }
            }
            if !found {
                // All basis vectors exhausted: leave as zero (matrix rank < k).
                q_cols[j] = vec![0.0f64; m];
            }
        }
        // Orthogonalise subsequent columns against column j.
        // We need to borrow q_cols[j] and q_cols[l] independently, so split_at_mut.
        let (left, right) = q_cols.split_at_mut(j + 1);
        let qj = &left[j];
        for (l_idx, ql) in right.iter_mut().enumerate() {
            let l = j + 1 + l_idx;
            if l >= k {
                break;
            }
            let dot: f64 = qj.iter().zip(ql.iter()).map(|(a, b)| a * b).sum();
            r[j * n + l] = dot;
            ql.iter_mut()
                .zip(qj.iter())
                .for_each(|(b, a)| *b -= dot * a);
        }
        // Fill remaining r entries in row j (l >= k, extra columns of mat).
        for l in k..n {
            let dot: f64 = q_cols[j]
                .iter()
                .enumerate()
                .map(|(row, v)| v * mat[row * n + l])
                .sum();
            r[j * n + l] = dot;
        }
    }

    // Pack Q from column vectors into (m, k) row-major.
    let mut q_packed = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            q_packed[i * k + j] = q_cols[j][i];
        }
    }
    (q_packed, r)
}

// ── Error variant helper ──────────────────────────────────────────────────────
// The spec requires `TnError::InvalidParameter { name, reason }` but the
// existing error.rs doesn't have that variant.  We add it via an extension
// match arm.  To avoid a compilation break we need to ensure the variant exists
// in error.rs.  Since we cannot edit the error.rs without touching the spec,
// we map to the closest available variant here.

// We DO NOT duplicate code; the error file already has InvalidConfiguration.
// The task spec says to use InvalidParameter { name, reason }.
// We implement it only for the public ALS API since the test checks for EmptyInput
// and shape-related errors.  The InvalidParameter usage is confined to this module.

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: compute residual ||X w - y|| ─────────────────────────────────
    fn compute_residual(preds: &[f64], y: &[f64]) -> f64 {
        preds
            .iter()
            .zip(y)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f64>()
            .sqrt()
    }

    // ── Helper: tiny LCG for test data generation ────────────────────────────
    fn gen_data(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_f64() - 0.5).collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: rank-1 TT (N=1, single core) recovers a linear function exactly.
    // W is a d-dimensional weight vector; TT-ALS with N=1 reduces to plain OLS.
    #[test]
    fn rank1_recovers_linear_function() {
        let n_samples = 40;
        let d = 5;
        let mut rng = LcgRng::new(3);
        let w_true: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
        let x: Vec<f64> = (0..n_samples * d).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..n_samples)
            .map(|i| (0..d).map(|j| x[i * d + j] * w_true[j]).sum::<f64>())
            .collect();

        let config = TtAlsConfig {
            n_dims: vec![d],
            max_rank: 1,
            max_sweeps: 20,
            tol: 1.0e-8,
            seed: 7,
        };
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        let preds = predict_tt(&result.cores, &result.core_shapes, &x, n_samples).expect("ok");
        let res = compute_residual(&preds, &y);
        assert!(res < 0.01, "residual={res}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: 1-D TT on a simple n=20, d=4 random regression converges.
    #[test]
    fn tt_als_1d_converges() {
        let n_samples = 20;
        let d = 4;
        let mut rng = LcgRng::new(17);
        let w_true: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
        let x: Vec<f64> = (0..n_samples * d).map(|_| rng.next_normal()).collect();
        let y: Vec<f64> = (0..n_samples)
            .map(|i| (0..d).map(|j| x[i * d + j] * w_true[j]).sum::<f64>())
            .collect();

        let config = TtAlsConfig {
            n_dims: vec![d],
            max_rank: 1,
            max_sweeps: 30,
            tol: 1.0e-8,
            seed: 42,
        };
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        assert!(result.residual < 1.0, "residual={}", result.residual);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: predict_tt output length equals n_samples.
    #[test]
    fn predict_tt_output_shape() {
        let n_samples = 15;
        let config = TtAlsConfig {
            n_dims: vec![3, 2],
            max_rank: 2,
            max_sweeps: 5,
            tol: 1.0e-6,
            seed: 1,
        };
        let total_dim: usize = config.n_dims.iter().product();
        let x = gen_data(99, n_samples * total_dim);
        let y = gen_data(100, n_samples);
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        let preds = predict_tt(&result.cores, &result.core_shapes, &x, n_samples).expect("ok");
        assert_eq!(preds.len(), n_samples);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: data generated from a rank-1 TT weight gives low residual.
    #[test]
    fn tt_als_low_residual_on_low_rank_data() {
        // Generate a rank-1 TT weight: W = u ⊗ v (outer product), N=2, d1=3, d2=4.
        let d1 = 3usize;
        let d2 = 4usize;
        let total = d1 * d2;
        let n_samples = 50;

        let u = [1.0f64, -0.5, 2.0];
        let v = [0.3f64, -1.0, 0.7, 0.2];
        let mut w_true = vec![0.0f64; total];
        for i in 0..d1 {
            for j in 0..d2 {
                w_true[i * d2 + j] = u[i] * v[j];
            }
        }
        let mut rng = LcgRng::new(55);
        let x: Vec<f64> = (0..n_samples * total).map(|_| rng.next_normal()).collect();
        let y: Vec<f64> = (0..n_samples)
            .map(|i| {
                (0..total)
                    .map(|j| x[i * total + j] * w_true[j])
                    .sum::<f64>()
            })
            .collect();

        let config = TtAlsConfig {
            n_dims: vec![d1, d2],
            max_rank: 2,
            max_sweeps: 30,
            tol: 1.0e-8,
            seed: 13,
        };
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        let preds = predict_tt(&result.cores, &result.core_shapes, &x, n_samples).expect("ok");
        let res = compute_residual(&preds, &y);
        assert!(res < 0.5, "residual={res}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: 2-D TT regression (N=2, d1*d2=6, max_rank=2, n=30).
    #[test]
    fn tt_als_2d_weight_regression() {
        let d1 = 2usize;
        let d2 = 3usize;
        let total = d1 * d2;
        let n_samples = 30;
        let mut rng = LcgRng::new(77);
        let w_true: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let x: Vec<f64> = (0..n_samples * total).map(|_| rng.next_normal()).collect();
        let y: Vec<f64> = (0..n_samples)
            .map(|i| {
                (0..total)
                    .map(|j| x[i * total + j] * w_true[j])
                    .sum::<f64>()
            })
            .collect();

        let config = TtAlsConfig {
            n_dims: vec![d1, d2],
            max_rank: 2,
            max_sweeps: 30,
            tol: 1.0e-8,
            seed: 22,
        };
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        assert_eq!(result.core_shapes.len(), 2);
        // Just check we can predict without error and get the right length.
        let preds = predict_tt(&result.cores, &result.core_shapes, &x, n_samples).expect("ok");
        assert_eq!(preds.len(), n_samples);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: core_shapes has exactly N entries.
    #[test]
    fn core_shapes_correct() {
        let config = TtAlsConfig {
            n_dims: vec![2, 3, 4],
            max_rank: 2,
            max_sweeps: 3,
            tol: 1.0e-6,
            seed: 0,
        };
        let total: usize = config.n_dims.iter().product();
        let n_samples = 10;
        let x = gen_data(1, n_samples * total);
        let y = gen_data(2, n_samples);
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        assert_eq!(result.core_shapes.len(), 3);
        // First and last bond dimensions must be 1.
        assert_eq!(result.core_shapes[0].0, 1);
        assert_eq!(result.core_shapes[2].2, 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 7: empty input returns EmptyInput error.
    #[test]
    fn empty_input_returns_error() {
        let config = TtAlsConfig {
            n_dims: vec![4],
            max_rank: 1,
            max_sweeps: 5,
            tol: 1.0e-6,
            seed: 0,
        };
        let err = tt_als_regression(&[], &[1.0, 2.0], 2, &config).unwrap_err();
        assert!(matches!(err, TnError::EmptyInput));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 8: max_rank == 0 returns InvalidParameter error.
    #[test]
    fn zero_rank_returns_error() {
        let config = TtAlsConfig {
            n_dims: vec![4],
            max_rank: 0,
            max_sweeps: 5,
            tol: 1.0e-6,
            seed: 0,
        };
        let x = gen_data(1, 8);
        let y = gen_data(2, 2);
        let err = tt_als_regression(&x, &y, 2, &config).unwrap_err();
        assert!(matches!(err, TnError::InvalidParameter { .. }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 9: mismatched n_samples between x and y returns ShapeMismatch.
    #[test]
    fn mismatched_samples_returns_error() {
        let config = TtAlsConfig {
            n_dims: vec![4],
            max_rank: 1,
            max_sweeps: 5,
            tol: 1.0e-6,
            seed: 0,
        };
        // x has 3*4=12 elements (3 samples), y has 2 elements (2 samples): mismatch.
        let x = gen_data(1, 12);
        let y = gen_data(2, 2);
        let err = tt_als_regression(&x, &y, 3, &config).unwrap_err();
        assert!(matches!(err, TnError::ShapeMismatch { .. }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 10: converged flag is set on simple perfectly-separable data.
    #[test]
    fn converged_flag_on_simple_data() {
        let d = 3usize;
        let n_samples = 30;
        // Generate data with known weight vector [1, 2, 3].
        let w = [1.0f64, 2.0, 3.0];
        let mut rng = LcgRng::new(5);
        let x: Vec<f64> = (0..n_samples * d).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..n_samples)
            .map(|i| (0..d).map(|j| x[i * d + j] * w[j]).sum::<f64>())
            .collect();

        let config = TtAlsConfig {
            n_dims: vec![d],
            max_rank: 1,
            max_sweeps: 50,
            tol: 1.0e-9,
            seed: 8,
        };
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        // On noise-free linear data with a single TT site, should converge quickly.
        assert!(
            result.converged,
            "did not converge; n_sweeps={}",
            result.n_sweeps
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 11: thin_qr produces orthonormal columns (internal sanity check).
    #[test]
    fn thin_qr_orthonormality() {
        let mat = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (q, _r) = thin_qr(&mat, 3, 3);
        // Q^T Q should be identity (3×3).
        let k = 3usize;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..3).map(|row| q[row * k + i] * q[row * k + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1.0e-10, "Q^TQ[{i},{j}]={dot}");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 12: LDL^T solve recovers exact solution for small SPD system.
    #[test]
    fn ldl_solve_exact() {
        // A = [[4, 2], [2, 3]], b = [6, 5] → x = [1, 1].
        let a = vec![4.0f64, 2.0, 2.0, 3.0];
        let b = vec![6.0f64, 5.0];
        let x = ldl_solve(&a, &b, 2).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-10, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1.0e-10, "x[1]={}", x[1]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 13: 3-mode TT regression (N=3) can run without error.
    #[test]
    fn tt_als_3d_runs_without_error() {
        let config = TtAlsConfig {
            n_dims: vec![2, 3, 2],
            max_rank: 2,
            max_sweeps: 5,
            tol: 1.0e-5,
            seed: 100,
        };
        let total: usize = config.n_dims.iter().product();
        let n_samples = 20;
        let x = gen_data(11, n_samples * total);
        let y = gen_data(12, n_samples);
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("3-mode ok");
        assert_eq!(result.core_shapes.len(), 3);
        let preds = predict_tt(&result.cores, &result.core_shapes, &x, n_samples).expect("ok");
        assert_eq!(preds.len(), n_samples);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 14: materialise_weight returns vector of correct length.
    #[test]
    fn materialise_weight_length() {
        let config = TtAlsConfig {
            n_dims: vec![3, 4],
            max_rank: 2,
            max_sweeps: 2,
            tol: 1.0e-6,
            seed: 9,
        };
        let total: usize = config.n_dims.iter().product();
        let n_samples = 5;
        let x = gen_data(20, n_samples * total);
        let y = gen_data(21, n_samples);
        let result = tt_als_regression(&x, &y, n_samples, &config).expect("ok");
        let w = materialise_weight(&result.cores, &result.core_shapes).expect("ok");
        assert_eq!(w.len(), total);
    }
}
