//! Block-Lanczos eigensolver for degenerate DMRG ground states.
//!
//! Implements the Golub-Underwood (1977) / Wu-Simon (2000) block-Lanczos
//! algorithm that simultaneously finds the `n_target` lowest eigenpairs of a
//! symmetric operator.  It is essential for DMRG when the ground state is
//! degenerate (common in systems with symmetry), where the standard
//! single-vector Lanczos collapses onto a single linear combination of the
//! degenerate eigenvectors and misses the others.
//!
//! ## Algorithm outline
//!
//! Instead of iterating with a *single* Krylov vector, the block variant
//! maintains a **block** of `p` orthonormal vectors `Vⱼ ∈ ℝ^{dim × p}` at
//! each step.  This produces a block-tridiagonal projected matrix
//!
//! ```text
//! T_m = [ A₁  B₁ᵀ              ]
//!        [ B₁  A₂  B₂ᵀ         ]
//!        [     B₂  A₃  B₃ᵀ     ]
//!        [         ⋱   ⋱   ⋱   ]
//!        [             Bₘ₋₁ Aₘ ]
//! ```
//!
//! with `p×p` diagonal blocks `Aⱼ` and upper-triangular off-diagonal blocks
//! `Bⱼ`.  Eigenvalues of `T_m` (Ritz values) approximate those of `A`.
//!
//! ## References
//!
//! * G. H. Golub and R. Underwood, "The block Lanczos method for computing
//!   eigenvalues," in *Mathematical Software III*, Academic Press, 1977.
//! * K. Wu and H. D. Simon, "Thick-restart Lanczos method for large symmetric
//!   eigenvalue problems," SIAM J. Matrix Anal. Appl. 22(2), 2000.

use crate::error::{TnError, TnResult};
use crate::handle::LcgRng;

// ─── Public configuration & result types ────────────────────────────────────

/// Configuration for the block-Lanczos eigensolver.
#[derive(Debug, Clone)]
pub struct BlockLanczosConfig {
    /// `p`: number of vectors in each Lanczos block.
    ///
    /// Setting `p ≥ degeneracy` ensures the degenerate sub-space is spanned
    /// at each step and never collapses onto a single eigenstate.  Default 2.
    pub block_size: usize,
    /// `m`: maximum number of Lanczos block steps.  Default 30.
    pub max_iter: usize,
    /// Convergence tolerance on Ritz-value change *and* residual bound.
    /// Default 1e-8.
    pub tol: f64,
    /// Number of lowest eigenpairs to compute.  Must satisfy
    /// `n_target ≤ block_size`.  Default 2.
    pub n_target: usize,
}

impl Default for BlockLanczosConfig {
    fn default() -> Self {
        Self {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 2,
        }
    }
}

/// Result returned by [`block_lanczos`].
#[derive(Debug, Clone)]
pub struct BlockLanczosResult {
    /// Lowest `n_target` eigenvalues, sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors packed as a `[dim × n_target]` column-major array.
    ///
    /// Column `i` (0-based) is `eigenvectors[i*dim .. (i+1)*dim]`.
    pub eigenvectors: Vec<f64>,
    /// Number of block-Lanczos steps actually performed.
    pub n_iter: usize,
    /// `true` if the Ritz values and residual bounds satisfied the tolerance.
    pub converged: bool,
    /// Residual `‖A·vᵢ − λᵢ·vᵢ‖` for each returned eigenpair.
    pub residuals: Vec<f64>,
}

// ─── Public entry point ──────────────────────────────────────────────────────

/// Block-Lanczos eigensolver for finding the `config.n_target` lowest
/// eigenpairs of a symmetric operator.
///
/// # Arguments
///
/// * `matvec` — closure `|v: &[f64]| -> Vec<f64>` applying the operator to a
///   single vector of length `dim`.  The operator is accessed only through
///   this closure (matrix-free formulation).
/// * `dim` — dimension of the operator.
/// * `config` — tuning parameters (block size, iterations, tolerance, …).
/// * `rng`    — random source for the initial block and rank-deficiency padding.
///
/// # Errors
///
/// * [`TnError::EmptyInput`] if `dim == 0`.
/// * [`TnError::InvalidConfiguration`] if `block_size == 0`, `n_target == 0`,
///   or `n_target > dim`.
/// * [`TnError::NotConverged`] if the Jacobi eigendecomposition of the
///   projected matrix fails.
pub fn block_lanczos<F>(
    matvec: F,
    dim: usize,
    config: &BlockLanczosConfig,
    rng: &mut LcgRng,
) -> TnResult<BlockLanczosResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    // ── Validate ────────────────────────────────────────────────────────────
    if dim == 0 {
        return Err(TnError::EmptyInput);
    }
    let p = config.block_size;
    if p == 0 {
        return Err(TnError::InvalidConfiguration(
            "block_size must be ≥ 1".into(),
        ));
    }
    let n_target = config.n_target;
    if n_target == 0 {
        return Err(TnError::InvalidConfiguration("n_target must be ≥ 1".into()));
    }
    if n_target > dim {
        return Err(TnError::InvalidConfiguration(
            "n_target must not exceed dim".into(),
        ));
    }
    // Clamp block_size to dim so we never exceed the full space
    let p = p.min(dim);
    // Maximum number of block steps.  The projected space grows by p per step.
    // We allow up to ceil(dim / p) steps so the projected space can span all
    // of R^dim.  However, if rank deficiency is detected (Krylov space exhausted),
    // we stop early.  The caller may further restrict via config.max_iter.
    let max_steps_dim = dim.div_ceil(p);
    let max_iter = config.max_iter.max(1).min(max_steps_dim);

    // ── Generate initial orthonormal block V₀ ─────────────────────────────
    let v1 = generate_random_block(dim, p, rng);
    // v_all[j] = Vⱼ, a flat [dim × p] column-major array (col i at offset i*dim)
    let mut v_all: Vec<Vec<f64>> = vec![v1];
    // Diagonal blocks Aⱼ (p×p row-major); indexed 0-based
    let mut diag_blocks: Vec<Vec<f64>> = Vec::new();
    // Off-diagonal blocks Bⱼ from QR (p×p row-major, upper-triangular).
    // off_blocks[j] = Bⱼ; Bⱼ connects diagonal block j to j+1 in T.
    // Only B₀..B_{m-2} appear in the final T_m; Bₘ₋₁ gives the residual bound.
    let mut off_blocks: Vec<Vec<f64>> = Vec::new();
    // For each step j, record which columns of V_{j+1} are "real" (non-padded).
    // v_real_cols[j] = list of column indices of V_{j+1} with non-zero QR diagonal.
    // V₀ always has p real columns (from generate_random_block).
    let v0_real: Vec<usize> = (0..p).collect();
    let mut v_real_cols: Vec<Vec<usize>> = vec![v0_real];

    let mut prev_ritz: Vec<f64> = vec![f64::INFINITY; n_target];
    let mut converged = false;
    let mut n_iter = 0usize;

    for j in 0..max_iter {
        let vj = v_all[j].clone(); // dim×p col-major

        // ── Step 1: W = A·Vⱼ  (p matvec calls) ────────────────────────────
        let mut w = apply_block(&matvec, &vj, dim, p)?;

        // ── Step 2: W ← W − Vⱼ₋₁ · Bⱼ₋₁ᵀ  (three-term recurrence) ────────
        // off_blocks[j-1] = B_{j-1}  (p×p upper-triangular row-major)
        if j > 0 {
            let b_prev = off_blocks[j - 1].clone();
            let v_prev = v_all[j - 1].clone();
            let bt = transpose_square(&b_prev, p);
            let vbt = mat_mul_ab(&v_prev, &bt, dim, p, p);
            for idx in 0..dim * p {
                w[idx] -= vbt[idx];
            }
        }

        // ── Step 3: Aⱼ = Vⱼᵀ · W  (p×p symmetric Rayleigh-Ritz block) ─────
        let a_j = mat_mul_atb(&vj, &w, dim, p, p);

        // ── Step 4: W ← W − Vⱼ · Aⱼ ────────────────────────────────────────
        let vaj = mat_mul_ab(&vj, &a_j, dim, p, p);
        for idx in 0..dim * p {
            w[idx] -= vaj[idx];
        }

        // ── Step 5: Full reorthogonalization (twice, Daniel et al. 1976) ────
        for _ in 0..2 {
            reorthogonalize(&mut w, &v_all, dim, p);
        }

        // ── Step 6: Thin QR of W → (Vⱼ₊₁, Bⱼ) ──────────────────────────────
        // Bⱼ is p×p upper-triangular; columns of Vⱼ₊₁ are orthonormal.
        // Rank deficiency is handled by random-vector padding in qr_thin.
        let (q_new, b_j) = qr_thin(&w, dim, p, rng);

        // Identify which columns of Q are genuinely new (non-zero QR diagonal)
        // vs padded (zero QR diagonal → random filler).
        let real_new_cols: Vec<usize> =
            (0..p).filter(|&k| b_j[k * p + k].abs() > 1.0e-10).collect();
        let krylov_exhausted = real_new_cols.is_empty();
        // krylov_partial: true if some but not all columns are padded
        let krylov_partial = !krylov_exhausted && real_new_cols.len() < p;

        // Always push the current step's diagonal block and B block.
        diag_blocks.push(a_j);
        off_blocks.push(b_j);
        n_iter = j + 1;

        if krylov_exhausted {
            // All columns are padded: the Krylov space is fully exhausted.
            // Include this step so the projected matrix can capture all directions,
            // but mark V_{j+1} as having zero real columns.
            v_real_cols.push(Vec::new());
            v_all.push(q_new);
            converged = true;
            break;
        }

        // Push real column info for V_{j+1} (clone so we can use it again below)
        let real_new_cols_saved = real_new_cols.clone();
        v_real_cols.push(real_new_cols.clone());

        // ── Step 7: Convergence check ─────────────────────────────────────────
        // T_m has diagonal blocks A₀..Aⱼ (total m_steps = j+1 blocks) and
        // off-diagonal blocks B₀..B_{j-1} (off_blocks[..j]).  Bⱼ is NOT part
        // of T_m; it supplies the residual bound ‖Bⱼ·sᵢ‖.
        let m_steps = j + 1;
        if let Ok((ritz_vals, ritz_vecs)) =
            block_tridiag_eigh(&diag_blocks, &off_blocks[..j], p, m_steps)
        {
            let take = n_target.min(ritz_vals.len());
            let curr_ritz: Vec<f64> = ritz_vals[..take].to_vec();
            let total_cols = m_steps * p;

            // Residual bound: ‖Bⱼ · sᵢ‖ where sᵢ are the last p components
            // of the i-th eigenvector of T_m (Wu & Simon 2000).
            let last_b = &off_blocks[j]; // = Bⱼ
            let res_bound_ok = (0..take).all(|i| {
                let s_start = (m_steps - 1) * p;
                let ev_start = i * total_cols + s_start;
                let ev_end = i * total_cols + total_cols;
                if ev_end > ritz_vecs.len() {
                    return true;
                }
                let s_i = &ritz_vecs[ev_start..ev_end];
                let bsi: f64 = (0..p)
                    .map(|row| {
                        let v: f64 = (0..p).map(|col| last_b[row * p + col] * s_i[col]).sum();
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt();
                bsi < config.tol
            });

            // Eigenvalue stability
            let val_ok = curr_ritz
                .iter()
                .zip(prev_ritz.iter())
                .all(|(cur, prv)| (cur - prv).abs() < config.tol);

            if val_ok && res_bound_ok && j > 0 {
                converged = true;
                // Vⱼ₊₁ and its real_cols already pushed above; just push q_new
                v_all.push(q_new);
                break;
            }
            prev_ritz = curr_ritz;
        }

        // Continue to next step if not converged and Krylov space is not fully exhausted.
        // krylov_partial means some columns are padded but we still push q_new.
        let _ = (krylov_partial, real_new_cols_saved); // tracked above for diagnostics only
        v_all.push(q_new);
    }

    // ── Collect the genuine (non-padded) Krylov vectors ─────────────────────
    // Build the basis matrix Q_basis whose columns are all real Krylov vectors,
    // in block order: columns of V₀ (all real), then real columns of V₁, etc.,
    // up to V_{m_steps-1}.  This excludes any padded (random) columns that were
    // inserted to handle rank deficiency during QR.
    let m_steps = diag_blocks.len();
    let mut basis: Vec<f64> = Vec::new(); // dim × n_basis, col-major
    let mut n_basis = 0usize;
    for blk in 0..m_steps {
        if blk >= v_all.len() || blk >= v_real_cols.len() {
            break;
        }
        let vblk = &v_all[blk];
        for &col in &v_real_cols[blk] {
            // Append column `col` of V_blk to basis
            basis.extend_from_slice(&vblk[col * dim..(col + 1) * dim]);
            n_basis += 1;
        }
    }

    if n_basis == 0 {
        return Err(TnError::NumericalInstability(
            "block_lanczos: no real Krylov vectors".into(),
        ));
    }

    // ── Rayleigh-Ritz: assemble and diagonalize the projected matrix ─────────
    // H_rr[i,j] = ψᵢᵀ · A · ψⱼ  where ψᵢ = basis[:,i]
    // (ψᵢ are orthonormal by construction, so S = I)
    let mut h_rr = vec![0.0f64; n_basis * n_basis];
    // For each basis vector ψⱼ, compute A·ψⱼ and then take inner products.
    for j in 0..n_basis {
        let psi_j = &basis[j * dim..(j + 1) * dim];
        let apsi_j = matvec(psi_j);
        for i in 0..=j {
            // H_rr[i,j] = <ψᵢ, A ψⱼ>
            let psi_i = &basis[i * dim..(i + 1) * dim];
            let val: f64 = psi_i.iter().zip(apsi_j.iter()).map(|(a, b)| a * b).sum();
            h_rr[i * n_basis + j] = val;
            h_rr[j * n_basis + i] = val; // symmetry
        }
    }

    // Diagonalise H_rr via symmetric Jacobi
    let (ritz_vals, ritz_vecs_rr) = jacobi_symm_sorted(&mut h_rr, n_basis)?;
    // ritz_vecs_rr[k * n_basis + i] = k-th component of i-th eigenvector

    let take = n_target.min(ritz_vals.len());

    // ── Recover eigenvectors: yᵢ = Q_basis · eᵢ ─────────────────────────────
    // eᵢ = ritz_vecs_rr[:,i] (i-th eigenvector of H_rr)
    let mut eigenvectors: Vec<f64> = vec![0.0; dim * take];
    for i in 0..take {
        for k in 0..n_basis {
            let coeff = ritz_vecs_rr[k * n_basis + i];
            let psi_k = &basis[k * dim..(k + 1) * dim];
            for row in 0..dim {
                eigenvectors[i * dim + row] += coeff * psi_k[row];
            }
        }
        // Normalise eigenvector i
        let nrm: f64 = eigenvectors[i * dim..(i + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if nrm > 1.0e-300 {
            for v in &mut eigenvectors[i * dim..(i + 1) * dim] {
                *v /= nrm;
            }
        }
    }

    // ── Compute true residuals ‖A·vᵢ − λᵢ·vᵢ‖ ───────────────────────────
    let eigenvalues: Vec<f64> = ritz_vals[..take].to_vec();
    let mut residuals = Vec::with_capacity(take);
    for i in 0..take {
        let ev = &eigenvectors[i * dim..(i + 1) * dim];
        let av = matvec(ev);
        let lam = eigenvalues[i];
        let res: f64 = av
            .iter()
            .zip(ev.iter())
            .map(|(a, v)| (a - lam * v).powi(2))
            .sum::<f64>()
            .sqrt();
        residuals.push(res);
    }

    Ok(BlockLanczosResult {
        eigenvalues,
        eigenvectors,
        n_iter,
        converged,
        residuals,
    })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Generate a random `[dim × p]` column-major block with orthonormal columns,
/// using the provided `LcgRng` and modified Gram-Schmidt orthogonalisation.
fn generate_random_block(dim: usize, p: usize, rng: &mut LcgRng) -> Vec<f64> {
    // Draw p random columns (stored column-major: col i → a[i*dim..(i+1)*dim])
    let mut a = vec![0.0; dim * p];
    for col in 0..p {
        for row in 0..dim {
            a[col * dim + row] = rng.next_normal();
        }
    }
    // Modified Gram-Schmidt in-place
    for col in 0..p {
        // Orthogonalise column `col` against all previous columns
        for prev in 0..col {
            let mut dot = 0.0;
            for row in 0..dim {
                dot += a[prev * dim + row] * a[col * dim + row];
            }
            for row in 0..dim {
                let sub = dot * a[prev * dim + row];
                a[col * dim + row] -= sub;
            }
        }
        // Normalise
        let nrm: f64 = a[col * dim..(col + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if nrm > 1.0e-14 {
            for row in 0..dim {
                a[col * dim + row] /= nrm;
            }
        } else {
            // Near-zero column: replace with a random unit vector orthogonal to previous ones
            replace_deficient_column(&mut a, col, dim, rng);
        }
    }
    a
}

/// Replace column `col` in `a` (column-major, `dim × p`) with a random unit
/// vector orthogonal to columns `0..col`.
fn replace_deficient_column(a: &mut [f64], col: usize, dim: usize, rng: &mut LcgRng) {
    const MAX_TRIES: usize = 32;
    for _ in 0..MAX_TRIES {
        // Fill with Gaussian noise
        for row in 0..dim {
            a[col * dim + row] = rng.next_normal();
        }
        // Orthogonalise against previous columns
        for prev in 0..col {
            let mut d = 0.0;
            for row in 0..dim {
                d += a[prev * dim + row] * a[col * dim + row];
            }
            for row in 0..dim {
                let sub = d * a[prev * dim + row];
                a[col * dim + row] -= sub;
            }
        }
        let nrm: f64 = a[col * dim..(col + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if nrm > 1.0e-14 {
            for row in 0..dim {
                a[col * dim + row] /= nrm;
            }
            return;
        }
    }
    // Fallback: use basis vector eₙ (n = col % dim) if random sampling keeps failing
    let row = col % dim;
    for r in 0..dim {
        a[col * dim + r] = 0.0;
    }
    a[col * dim + row] = 1.0;
}

/// Thin QR decomposition of `w` (m×n, column-major, **not** standard row-major).
///
/// Here `w` has `m` rows and `n` columns stored as `w[col * m + row]`.
/// Returns `(Q [m×n col-major], R [n×n row-major])`.
/// If any column becomes near-zero, the corresponding column is replaced with
/// a random orthonormal vector (rank-deficiency padding).
fn qr_thin(w: &[f64], m: usize, n: usize, rng: &mut LcgRng) -> (Vec<f64>, Vec<f64>) {
    // Copy w → q so we can work in-place
    let mut q = w.to_vec();
    // R is n×n, upper-triangular (row-major)
    let mut r = vec![0.0; n * n];

    for col in 0..n {
        // Orthogonalise column `col` against all previous
        for prev in 0..col {
            let mut dot = 0.0;
            for row in 0..m {
                dot += q[prev * m + row] * q[col * m + row];
            }
            r[prev * n + col] = dot;
            for row in 0..m {
                q[col * m + row] -= dot * q[prev * m + row];
            }
        }
        // Compute column norm
        let nrm: f64 = q[col * m..(col + 1) * m]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        r[col * n + col] = nrm;
        if nrm > 1.0e-14 {
            for row in 0..m {
                q[col * m + row] /= nrm;
            }
        } else {
            // Rank-deficient column: pad with random orthonormal vector
            r[col * n + col] = 0.0;
            replace_deficient_column(&mut q, col, m, rng);
        }
    }
    (q, r)
}

/// Apply `matvec` to each column of `v_block` (dim×p col-major) and return
/// the result as a new dim×p col-major matrix.
fn apply_block<F>(matvec: &F, v_block: &[f64], dim: usize, p: usize) -> TnResult<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut w = vec![0.0; dim * p];
    for col in 0..p {
        let vc = &v_block[col * dim..(col + 1) * dim];
        let out = matvec(vc);
        if out.len() != dim {
            return Err(TnError::ShapeMismatch {
                expected: vec![dim],
                got: vec![out.len()],
            });
        }
        w[col * dim..(col + 1) * dim].copy_from_slice(&out);
    }
    Ok(w)
}

/// Full reorthogonalisation of `w` (dim×p col-major) against all blocks in
/// `v_all` (each dim×p col-major).
///
/// Computes `W ← W − Σₖ (Vₖ · (Vₖᵀ · W))`.
fn reorthogonalize(w: &mut [f64], v_all: &[Vec<f64>], dim: usize, p: usize) {
    for vk in v_all {
        // Compute Sₖ = Vₖᵀ · W  (p×p)
        let s = mat_mul_atb(vk, w, dim, p, p);
        // W ← W − Vₖ · Sₖ
        let vs = mat_mul_ab(vk, &s, dim, p, p);
        for idx in 0..dim * p {
            w[idx] -= vs[idx];
        }
    }
}

/// Compute `Aᵀ · B` where `A` is `m×n` col-major and `B` is `m×p` col-major.
/// Result is `n×p` row-major.
///
/// `Aᵀ` has shape `n×m`, so the product `(n×m)·(m×p)` = `n×p`.
fn mat_mul_atb(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    // a: col-major [n cols of m rows]  →  a[j*m + i] = A[i,j]
    // b: col-major [p cols of m rows]  →  b[k*m + i] = B[i,k]
    // result c[j,k] = sum_i A[i,j]*B[i,k]  → row-major c[j*p + k]
    let mut c = vec![0.0; n * p];
    for j in 0..n {
        for k in 0..p {
            let mut acc = 0.0;
            for i in 0..m {
                acc += a[j * m + i] * b[k * m + i];
            }
            c[j * p + k] = acc;
        }
    }
    c
}

/// Compute `A · B` where `A` is `m×k` col-major and `B` is `k×n` row-major.
/// Result is `m×n` col-major.
fn mat_mul_ab(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // a: col-major [k cols of m rows]  →  a[j*m + i] = A[i,j]
    // b: row-major [k rows of n cols]  →  b[j*n + l] = B[j,l]
    // result[l*m + i] = C[i,l] = sum_j A[i,j] * B[j,l]
    let mut c = vec![0.0; m * n];
    for j in 0..k {
        for l in 0..n {
            let bjl = b[j * n + l];
            for i in 0..m {
                c[l * m + i] += a[j * m + i] * bjl;
            }
        }
    }
    c
}

/// Transpose a square `n×n` row-major matrix.
fn transpose_square(a: &[f64], n: usize) -> Vec<f64> {
    let mut at = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            at[j * n + i] = a[i * n + j];
        }
    }
    at
}

/// Assemble the full `(m·p × m·p)` block-tridiagonal matrix from the diagonal
/// and off-diagonal blocks and diagonalise it with the symmetric Jacobi method.
///
/// # Arguments
///
/// * `diag_blocks`  — `Aⱼ` blocks (each `p×p` row-major), length `m`.
/// * `off_blocks`   — `Bⱼ` blocks (each `p×p` row-major, upper-triangular
///   from QR), length `m-1`.  `Bⱼ` occupies the off-diagonal position
///   `(j, j+1)` and its transpose at `(j+1, j)`.
/// * `p` — block size.
/// * `m_steps` — number of diagonal blocks (`m`).
///
/// Returns `(eigenvalues_ascending, eigenvectors)` where eigenvectors are
/// stored as `(n_eig × total)` row-major with `total = m * p`.
fn block_tridiag_eigh(
    diag_blocks: &[Vec<f64>],
    off_blocks: &[Vec<f64>],
    p: usize,
    m_steps: usize,
) -> TnResult<(Vec<f64>, Vec<f64>)> {
    if m_steps == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let total = m_steps * p;
    // Build the dense matrix T (total × total, row-major)
    let mut t = vec![0.0; total * total];

    // Fill diagonal blocks
    for (blk, a) in diag_blocks.iter().enumerate().take(m_steps) {
        for i in 0..p {
            for j in 0..p {
                let row = blk * p + i;
                let col = blk * p + j;
                t[row * total + col] = a[i * p + j];
            }
        }
    }

    // Fill off-diagonal blocks (Bⱼ at (j, j+1) and Bⱼᵀ at (j+1, j))
    let n_off_diag = off_blocks.len().min(m_steps.saturating_sub(1));
    for (blk, b) in off_blocks.iter().enumerate().take(n_off_diag) {
        let row_base = blk * p;
        let col_base = (blk + 1) * p;
        for i in 0..p {
            for j in 0..p {
                // B occupies upper triangle of QR decomp
                t[(row_base + i) * total + (col_base + j)] = b[i * p + j];
                // Bᵀ
                t[(col_base + j) * total + (row_base + i)] = b[i * p + j];
            }
        }
    }

    // Diagonalise with Jacobi
    let (eig_vals, eig_vecs) = jacobi_symm_sorted(&mut t, total)?;

    // Return eigenvectors as (n_eig × total) row-major
    // jacobi_symm_sorted returns V as (total × total) row-major (columns = eigenvectors)
    // We need to transpose: output[i * total + k] = V[k * total + i]
    let n_eig = eig_vals.len();
    let mut out_vecs = vec![0.0; n_eig * total];
    for i in 0..n_eig {
        for k in 0..total {
            out_vecs[i * total + k] = eig_vecs[k * total + i];
        }
    }

    Ok((eig_vals, out_vecs))
}

/// Symmetric Jacobi eigendecomposition of an `n × n` row-major matrix
/// (modified in-place). Returns `(eigenvalues_ascending, V)` where columns of
/// `V` (stored row-major as `n × n`) are the eigenvectors.
///
/// This mirrors the implementation in `dmrg/lanczos.rs` but is kept local to
/// avoid exposing it from that module.
fn jacobi_symm_sorted(a: &mut [f64], n: usize) -> TnResult<(Vec<f64>, Vec<f64>)> {
    const MAX_SWEEPS: usize = 300;
    const EPS: f64 = 1.0e-14;
    // Accumulate rotations in V (identity initially)
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    'outer: for sweep in 0..MAX_SWEEPS {
        let mut max_off: f64 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let av = a[p * n + q].abs();
                if av > max_off {
                    max_off = av;
                }
            }
        }
        if max_off < EPS {
            break 'outer;
        }
        if sweep == MAX_SWEEPS - 1 {
            return Err(TnError::NotConverged { iter: MAX_SWEEPS });
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < EPS {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t_rot = if theta.abs() > 1.0e10 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t_rot * t_rot).sqrt();
                let s = t_rot * c;
                let h = t_rot * apq;
                a[p * n + p] = app - h;
                a[q * n + q] = aqq + h;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = s * arp + c * arq;
                    a[q * n + r] = a[r * n + q];
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    // Extract diagonal → eigenvalues, sort ascending
    let eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        eigs[i]
            .partial_cmp(&eigs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_eigs: Vec<f64> = order.iter().map(|&i| eigs[i]).collect();
    // Reorder columns of V accordingly (row-major)
    let mut sorted_v = vec![0.0; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            sorted_v[row * n + new_col] = v[row * n + old_col];
        }
    }
    Ok((sorted_eigs, sorted_v))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: return a `matvec` closure for a diagonal matrix with entries `diag`.
    fn diag_matvec(diag: Vec<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
        move |v: &[f64]| v.iter().zip(diag.iter()).map(|(x, d)| x * d).collect()
    }

    // ── 1. Config defaults ──────────────────────────────────────────────────
    #[test]
    fn config_defaults() {
        let cfg = BlockLanczosConfig::default();
        assert_eq!(cfg.block_size, 2);
        assert_eq!(cfg.max_iter, 30);
        assert_eq!(cfg.n_target, 2);
        assert!((cfg.tol - 1.0e-8).abs() < 1.0e-20);
    }

    // ── 2. Diagonal matrix eigenvalues ──────────────────────────────────────
    #[test]
    fn diagonal_matrix_eigenvalues() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-10,
            n_target: 2,
        };
        let mut rng = LcgRng::new(42);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), 2);
        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1.0e-6,
            "λ₀={}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 2.0).abs() < 1.0e-6,
            "λ₁={}",
            res.eigenvalues[1]
        );
    }

    // ── 3. Degenerate eigenvalues ────────────────────────────────────────────
    #[test]
    fn degenerate_eigenvalues() {
        let diag = vec![1.0, 1.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 40,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(7);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), 2);
        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1.0e-6,
            "λ₀={}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 1.0).abs() < 1.0e-6,
            "λ₁={}",
            res.eigenvalues[1]
        );
    }

    // ── 4. Near-degenerate eigenvalues ──────────────────────────────────────
    #[test]
    fn near_degenerate() {
        let diag = vec![1.0, 1.001, 3.0, 5.0, 7.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 50,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(13);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), 2);
        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1.0e-5,
            "λ₀={}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 1.001).abs() < 1.0e-5,
            "λ₁={}",
            res.eigenvalues[1]
        );
    }

    // ── 5. Eigenvalue ordering (ascending) ──────────────────────────────────
    #[test]
    fn eigenvalue_ordering() {
        let diag = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(17);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        for i in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[i] >= res.eigenvalues[i - 1] - 1.0e-10,
                "eigenvalues not sorted: {:?}",
                res.eigenvalues
            );
        }
    }

    // ── 6. Eigenvectors orthonormality ──────────────────────────────────────
    #[test]
    fn eigenvectors_orthonormal() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-10,
            n_target: 2,
        };
        let mut rng = LcgRng::new(99);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        let n_t = res.eigenvalues.len();
        // Check v_i · v_j ≈ δᵢⱼ
        for i in 0..n_t {
            for j in 0..n_t {
                let vi = &res.eigenvectors[i * dim..(i + 1) * dim];
                let vj = &res.eigenvectors[j * dim..(j + 1) * dim];
                let dot: f64 = vi.iter().zip(vj.iter()).map(|(a, b)| a * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1.0e-6,
                    "v[{i}]·v[{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    // ── 7. Eigenvector residual ──────────────────────────────────────────────
    #[test]
    fn eigenvectors_residual_small() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag.clone());
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-10,
            n_target: 2,
        };
        let mut rng = LcgRng::new(55);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        let mv2 = diag_matvec(diag);
        for i in 0..res.eigenvalues.len() {
            let lam = res.eigenvalues[i];
            let v = &res.eigenvectors[i * dim..(i + 1) * dim];
            let av = mv2(v);
            let res_norm: f64 = av
                .iter()
                .zip(v.iter())
                .map(|(a, vi)| (a - lam * vi).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(res_norm < 1.0e-6, "residual[{i}] = {res_norm} (λ={lam})");
        }
    }

    // ── 8. Result dimension correct ──────────────────────────────────────────
    #[test]
    fn result_dim_correct() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(11);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), cfg.n_target);
        assert_eq!(res.eigenvectors.len(), dim * cfg.n_target);
        assert_eq!(res.residuals.len(), cfg.n_target);
    }

    // ── 9. Converges flag ───────────────────────────────────────────────────
    #[test]
    fn converges_flag() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(37);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        // Diagonal matrix with well-separated eigenvalues should converge
        assert!(res.converged, "expected converged=true, got false");
    }

    // ── 10. Larger matrix ────────────────────────────────────────────────────
    #[test]
    fn larger_matrix() {
        let n = 50;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 3,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 3,
        };
        let mut rng = LcgRng::new(71);
        let res = block_lanczos(mv, n, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), 3);
        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1.0e-5,
            "λ₀={}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 2.0).abs() < 1.0e-5,
            "λ₁={}",
            res.eigenvalues[1]
        );
        assert!(
            (res.eigenvalues[2] - 3.0).abs() < 1.0e-5,
            "λ₂={}",
            res.eigenvalues[2]
        );
    }

    // ── 11. Empty matrix error ────────────────────────────────────────────────
    #[test]
    fn empty_matrix_error() {
        let mv = |v: &[f64]| v.to_vec();
        let cfg = BlockLanczosConfig::default();
        let mut rng = LcgRng::new(1);
        let err = block_lanczos(mv, 0, &cfg, &mut rng);
        assert!(err.is_err(), "expected Err for dim=0");
        matches!(err.unwrap_err(), TnError::EmptyInput);
    }

    // ── 12. Block size larger than n_target ──────────────────────────────────
    #[test]
    fn block_size_larger_than_n_target() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 4,
            max_iter: 30,
            tol: 1.0e-8,
            n_target: 2,
        };
        let mut rng = LcgRng::new(23);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        assert_eq!(res.eigenvalues.len(), 2);
        assert!(
            (res.eigenvalues[0] - 1.0).abs() < 1.0e-5,
            "λ₀={}",
            res.eigenvalues[0]
        );
        assert!(
            (res.eigenvalues[1] - 2.0).abs() < 1.0e-5,
            "λ₁={}",
            res.eigenvalues[1]
        );
    }

    // ── 13. Residuals close to zero ──────────────────────────────────────────
    #[test]
    fn residuals_close_to_zero() {
        let diag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = diag.len();
        let mv = diag_matvec(diag);
        let cfg = BlockLanczosConfig {
            block_size: 2,
            max_iter: 30,
            tol: 1.0e-10,
            n_target: 2,
        };
        let mut rng = LcgRng::new(19);
        let res = block_lanczos(mv, dim, &cfg, &mut rng).expect("should succeed");
        for (i, &r) in res.residuals.iter().enumerate() {
            assert!(r < 1.0e-6, "residuals[{i}] = {r} is too large");
        }
    }
}
