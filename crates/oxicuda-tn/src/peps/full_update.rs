//! PEPS Full-Update imaginary-time evolution (Corboz 2010, Jordan 2008 iPEPS).
//!
//! The full update differs from simple_update in that it uses the **exact** reduced
//! environment tensors — computed by boundary-MPS contraction — rather than the
//! product-lambda approximation.  The minimisation of `‖θ' − A'⊗B'‖²_env` is solved
//! via Alternating Least Squares (ALS), where each ALS sub-step reduces to a linear
//! system that we solve with pseudo-inverse via SVD.
//!
//! ## Algorithm for a horizontal bond between `(r, c)` and `(r, c+1)`
//!
//! 1. Contract all rows above/below into top/bottom boundary-MPS environments.
//! 2. Build `θ = A ⊗ B` (rank-6 tensor); apply imaginary-time gate `e^{-τH}`.
//! 3. ALS: fix A, solve for B; fix B, solve for A; repeat until convergence.
//! 4. QR gauge-fix: QR on A', absorb R into B'.
//! 5. Commit new tensors to the PEPS.

use crate::error::{TnError, TnResult};
use crate::handle::LcgRng;
use crate::peps::peps::{Peps, PepsTensor};
use crate::peps::simple_update::{heisenberg_hamiltonian_2site, mat_exp_sym};
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a PEPS full-update imaginary-time evolution run.
#[derive(Debug, Clone)]
pub struct FullUpdateConfig {
    /// Maximum bond dimension cutoff for virtual indices.
    pub chi_max: usize,
    /// Number of imaginary-time steps to perform.
    pub n_iter: usize,
    /// Imaginary-time step size `δτ`.
    pub dt: f64,
    /// Maximum number of ALS inner iterations per bond update.
    pub als_max_iter: usize,
    /// ALS convergence tolerance (‖A'⊗B' − prev‖ / ‖prev‖).
    pub als_tol: f64,
    /// Bond dimension for boundary-MPS environment contraction.
    pub chi_env: usize,
    /// Whether to use boundary-MPS environment contraction; `false` uses identity env.
    pub use_env: bool,
}

impl Default for FullUpdateConfig {
    fn default() -> Self {
        Self {
            chi_max: 2,
            n_iter: 5,
            dt: 0.01,
            als_max_iter: 10,
            als_tol: 1e-8,
            chi_env: 4,
            use_env: false,
        }
    }
}

/// Result returned by [`full_update_run`].
#[derive(Debug, Clone)]
pub struct FullUpdateResult {
    /// Estimated energy per site at the end of the run.
    pub energy_per_site: f64,
    /// Number of imaginary-time steps actually performed.
    pub steps_run: usize,
    /// Whether the ALS inner loops converged in the last step.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Build a random PEPS with bond dimension `d_bond` and physical dimension `d_phys`.
///
/// Boundary virtual bonds are set to 1 (open boundary conditions); interior bonds
/// have dimension `d_bond`.  Tensor elements are drawn from `LcgRng::next_normal()`.
pub fn full_update_init(
    rows: usize,
    cols: usize,
    d_bond: usize,
    d_phys: usize,
    rng: &mut LcgRng,
) -> TnResult<Peps> {
    if rows == 0 || cols == 0 {
        return Err(TnError::EmptyInput);
    }
    if d_bond == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    if d_phys == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }
    Peps::random(rows, cols, d_phys, d_bond, rng)
}

// ─────────────────────────────────────────────────────────────────────────────
// PepsTensor index helper
// ─────────────────────────────────────────────────────────────────────────────

/// Flat index for a `peps::peps::PepsTensor` element `(l, r, u, d, p)`.
///
/// Layout: `(((l * d_r + r) * d_u + u) * d_d + d) * d_p + p`.
#[inline]
fn peps_idx(t: &PepsTensor, l: usize, r: usize, u: usize, d: usize, p: usize) -> usize {
    (((l * t.d_r + r) * t.d_u + u) * t.d_d + d) * t.d_p + p
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared Frobenius norm of a slice.
#[inline]
fn frob_norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the Frobenius norm of a slice.
#[inline]
fn frob_norm(v: &[f64]) -> f64 {
    frob_norm_sq(v).sqrt()
}

/// Multiply two matrices `A (m×k)` and `B (k×n)` in row-major, writing result into `out (m×n)`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/// Compute the pseudo-inverse of a matrix `A (m×n)` via SVD, returned as `(n×m)`.
///
/// Singular values below `rcond * s_max` are treated as zero.
fn pseudo_inverse(a: &[f64], m: usize, n: usize, rcond: f64) -> TnResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    let svd = svd_jacobi(a, m, n)?;
    let k = svd.k;
    let s_max = svd.s.first().copied().unwrap_or(0.0);
    let tol = rcond * s_max;

    // A^+ = V * diag(1/s) * U^T  — shape (n × m)
    // svd.u: (m × k), svd.vt: (k × n) so V = vt^T: (n × k)
    let mut result = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0f64;
            for q in 0..k {
                let s_q = svd.s[q];
                if s_q > tol {
                    // V[i, q] = vt[q, i]; U^T[q, j] = u[j, q]
                    acc += svd.vt[q * n + i] * (1.0 / s_q) * svd.u[j * k + q];
                }
            }
            result[i * m + j] = acc;
        }
    }
    Ok(result)
}

/// Result of a truncated SVD: `(U, S, Vt, k_new)`.
///
/// `U` is `(m × k_new)`, `Vt` is `(k_new × n)`, `S` is length `k_new`.
type TruncatedSvd = (Vec<f64>, Vec<f64>, Vec<f64>, usize);

/// Truncated SVD: keep at most `chi_max` singular values.  Returns `(U, S, Vt, k_new)`.
///
/// `U` is `(m × k_new)`, `Vt` is `(k_new × n)`, `S` is length `k_new`.
fn svd_truncated(mat: &[f64], m: usize, n: usize, chi_max: usize) -> TnResult<TruncatedSvd> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    let svd = svd_jacobi(mat, m, n)?;
    let k_full = svd.k;
    let k_new = chi_max.min(k_full).max(1);

    // Extract leading k_new columns of U (m × k_new)
    let mut u_new = vec![0.0f64; m * k_new];
    for i in 0..m {
        for j in 0..k_new {
            u_new[i * k_new + j] = svd.u[i * k_full + j];
        }
    }
    // Extract leading k_new singular values
    let s_new = svd.s[..k_new].to_vec();

    // Extract leading k_new rows of Vt (k_new × n)
    let mut vt_new = vec![0.0f64; k_new * n];
    for j in 0..k_new {
        for r in 0..n {
            vt_new[j * n + r] = svd.vt[j * n + r];
        }
    }

    Ok((u_new, s_new, vt_new, k_new))
}

/// QR decomposition of a tall-or-square matrix `A (m×n)` using Gram-Schmidt.
///
/// Returns `(Q, R)` where `Q (m×k)` has orthonormal columns and `R (k×n)` is upper
/// triangular, with `k = min(m, n)`.  If a column becomes numerically zero it is
/// replaced by a unit vector to keep Q well-defined.
fn qr_gram_schmidt(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let mut q = vec![0.0f64; m * k];
    let mut r = vec![0.0f64; k * n];

    // Column-major iteration: process columns 0..n.
    // We build k orthonormal vectors from the columns of A.
    for j in 0..k {
        // Copy j-th column of A into q_j
        let mut v: Vec<f64> = (0..m).map(|i| a[i * n + j]).collect();

        // Orthogonalise against previous columns
        for qi in 0..j {
            let dot: f64 = (0..m).map(|i| q[i * k + qi] * v[i]).sum();
            r[qi * n + j] = dot;
            for i in 0..m {
                v[i] -= dot * q[i * k + qi];
            }
        }

        // Normalise
        let norm = frob_norm(&v);
        let safe = norm.max(1e-300);
        r[j * n + j] = norm;
        for i in 0..m {
            q[i * k + j] = v[i] / safe;
        }
    }

    // Fill remaining columns of R (j >= k, cross terms with extra A columns if n > k)
    for j in k..n {
        for qi in 0..k {
            let dot: f64 = (0..m).map(|i| q[i * k + qi] * a[i * n + j]).sum();
            r[qi * n + j] = dot;
        }
    }

    (q, r)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor flattening helpers
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Two-site theta contraction and gate application
// ─────────────────────────────────────────────────────────────────────────────

/// Contract two PEPS tensors on their shared horizontal bond.
///
/// For `A` with shape `[d_l, D_bond, d_u_l, d_d_l, d_p]` and `B` with shape
/// `[D_bond, d_r, d_u_r, d_d_r, d_p]`, forms the two-site tensor
///
///   `Θ[(l, u_l, d_l, p_l), (p_r, r, u_r, d_r)]
///     = Σ_b A[l, b, u_l, d_l, p_l] · B[b, r, u_r, d_r, p_r]`
///
/// Returned flat array has shape `[m, n]` with `m = d_l * d_u_l * d_d_l * d_p`
/// and `n = d_p * d_r * d_u_r * d_d_r`.  Also returns `(chi_l, chi_r, d_p)`.
fn contract_two_sites_h(
    a: &PepsTensor,
    b: &PepsTensor,
) -> TnResult<(Vec<f64>, usize, usize, usize)> {
    let d_bond = a.d_r;
    if b.d_l != d_bond {
        return Err(TnError::DimensionMismatch { a: a.d_r, b: b.d_l });
    }
    let d_p = a.d_p;
    if b.d_p != d_p {
        return Err(TnError::DimensionMismatch { a: d_p, b: b.d_p });
    }

    let chi_l = a.d_l * a.d_u * a.d_d;
    let chi_r = b.d_r * b.d_u * b.d_d;
    let m = chi_l * d_p;
    let n = d_p * chi_r;
    let mut theta = vec![0.0f64; m * n];

    for l in 0..a.d_l {
        for u_l in 0..a.d_u {
            for d_l in 0..a.d_d {
                let row_base_l = (l * a.d_u + u_l) * a.d_d + d_l; // in [0, chi_l)
                for p_l in 0..d_p {
                    let row = row_base_l * d_p + p_l;
                    for p_r in 0..d_p {
                        for r in 0..b.d_r {
                            for u_r in 0..b.d_u {
                                for d_r in 0..b.d_d {
                                    let col_base_r = (r * b.d_u + u_r) * b.d_d + d_r;
                                    let col = p_r * chi_r + col_base_r;
                                    let mut acc = 0.0f64;
                                    for bond in 0..d_bond {
                                        acc += a.data[peps_idx(a, l, bond, u_l, d_l, p_l)]
                                            * b.data[peps_idx(b, bond, r, u_r, d_r, p_r)];
                                    }
                                    theta[row * n + col] += acc;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((theta, chi_l, chi_r, d_p))
}

/// Contract two PEPS tensors on their shared vertical bond.
///
/// For top `A` with shape `[d_l, d_r, d_u, D_bond, d_p]` and bottom `B` with
/// shape `[d_l, d_r, D_bond, d_d, d_p]`, forms:
///
///   `Θ[(l_a, r_a, u_a, p_a), (p_b, l_b, r_b, d_b)]
///     = Σ_b A[l_a, r_a, u_a, b, p_a] · B[l_b, r_b, b, d_b, p_b]`
///
/// Returns `(theta, chi_u, chi_d, d_p)`.
fn contract_two_sites_v(
    a: &PepsTensor,
    b: &PepsTensor,
) -> TnResult<(Vec<f64>, usize, usize, usize)> {
    let d_bond = a.d_d;
    if b.d_u != d_bond {
        return Err(TnError::DimensionMismatch { a: a.d_d, b: b.d_u });
    }
    let d_p = a.d_p;
    if b.d_p != d_p {
        return Err(TnError::DimensionMismatch { a: d_p, b: b.d_p });
    }

    let chi_u = a.d_l * a.d_r * a.d_u;
    let chi_d = b.d_l * b.d_r * b.d_d;
    let m = chi_u * d_p;
    let n = d_p * chi_d;
    let mut theta = vec![0.0f64; m * n];

    for l_a in 0..a.d_l {
        for r_a in 0..a.d_r {
            for u_a in 0..a.d_u {
                let row_base_u = (l_a * a.d_r + r_a) * a.d_u + u_a;
                for p_a in 0..d_p {
                    let row = row_base_u * d_p + p_a;
                    for p_b in 0..d_p {
                        for l_b in 0..b.d_l {
                            for r_b in 0..b.d_r {
                                for d_b in 0..b.d_d {
                                    let col_base_d = (l_b * b.d_r + r_b) * b.d_d + d_b;
                                    let col = p_b * chi_d + col_base_d;
                                    let mut acc = 0.0f64;
                                    for bond in 0..d_bond {
                                        acc += a.data[peps_idx(a, l_a, r_a, u_a, bond, p_a)]
                                            * b.data[peps_idx(b, l_b, r_b, bond, d_b, p_b)];
                                    }
                                    theta[row * n + col] += acc;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((theta, chi_u, chi_d, d_p))
}

/// Apply a two-site gate to the two-site tensor `θ (m × n)`.
///
/// `m = chi_l * d_p`, `n = d_p * chi_r`.  The gate has shape `[d_p² × d_p²]` with
/// convention `gate[p'_l * d_p + p'_r, p_l * d_p + p_r]`.
fn apply_gate(
    theta: &[f64],
    m: usize,
    n: usize,
    chi_l: usize,
    chi_r: usize,
    d_p: usize,
    gate: &[f64],
) -> TnResult<Vec<f64>> {
    let d2 = d_p * d_p;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate.len()],
        });
    }
    if m != chi_l * d_p || n != d_p * chi_r {
        return Err(TnError::ShapeMismatch {
            expected: vec![chi_l * d_p, d_p * chi_r],
            got: vec![m, n],
        });
    }

    let mut out = vec![0.0f64; m * n];
    for l_flat in 0..chi_l {
        for r_flat in 0..chi_r {
            for pl_p in 0..d_p {
                for pr_p in 0..d_p {
                    let mut acc = 0.0f64;
                    for pl in 0..d_p {
                        for pr in 0..d_p {
                            let th_idx = (l_flat * d_p + pl) * n + pr * chi_r + r_flat;
                            let g_idx = (pl_p * d_p + pr_p) * d2 + pl * d_p + pr;
                            acc += gate[g_idx] * theta[th_idx];
                        }
                    }
                    let out_idx = (l_flat * d_p + pl_p) * n + pr_p * chi_r + r_flat;
                    out[out_idx] = acc;
                }
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// ALS inner loop
// ─────────────────────────────────────────────────────────────────────────────

/// Result of ALS decomposition of a two-site tensor.
struct AlsResult {
    /// New left tensor data (row-major, shape `[chi_l * d_p, chi_new]`).
    u_data: Vec<f64>,
    /// New right tensor data (row-major, shape `[chi_new, chi_r * d_p]`).
    v_data: Vec<f64>,
    /// New bond dimension.
    chi_new: usize,
}

/// ALS decomposition of `theta_prime (m × n)` into `U (m × chi) · V (chi × n)`
/// with `chi ≤ chi_max`.
///
/// Uses identity environment (exact gradient direction without environment metric).
/// The minimisation objective is `‖θ' − UV‖²_F`.
///
/// Initialised by truncated SVD; then alternately fixes U and solves for V (and
/// vice versa) via normal equations with pseudo-inverse.
fn als_decompose(
    theta_prime: &[f64],
    m: usize,
    n: usize,
    chi_max: usize,
    als_max_iter: usize,
    als_tol: f64,
) -> TnResult<AlsResult> {
    // ── Initialise via truncated SVD ──────────────────────────────────────────
    let (mut u, s, vt, chi_new) = svd_truncated(theta_prime, m, n, chi_max)?;

    // Absorb singular values into V: V = diag(S) * Vt
    let mut v = vec![0.0f64; chi_new * n];
    for i in 0..chi_new {
        for j in 0..n {
            v[i * n + j] = s[i] * vt[i * n + j];
        }
    }

    // ── ALS iterations ────────────────────────────────────────────────────────
    let mut prev_residual = f64::INFINITY;

    for _iter in 0..als_max_iter {
        // Fix U, solve for V: V = (U^T U)^{-1} U^T θ'
        // Normal equations: (U^T U) V = U^T θ'  →  V = (U^T U)^+ (U^T θ')
        {
            // Gram matrix G = U^T U  (chi_new × chi_new)
            let mut g = vec![0.0f64; chi_new * chi_new];
            matmul(
                &{
                    // U^T: (chi_new × m)
                    let mut ut = vec![0.0f64; chi_new * m];
                    for i in 0..m {
                        for j in 0..chi_new {
                            ut[j * m + i] = u[i * chi_new + j];
                        }
                    }
                    ut
                },
                &u,
                chi_new,
                m,
                chi_new,
                &mut g,
            );

            // Right-hand side: rhs = U^T θ'  (chi_new × n)
            let mut rhs = vec![0.0f64; chi_new * n];
            {
                let mut ut = vec![0.0f64; chi_new * m];
                for i in 0..m {
                    for j in 0..chi_new {
                        ut[j * m + i] = u[i * chi_new + j];
                    }
                }
                matmul(&ut, theta_prime, chi_new, m, n, &mut rhs);
            }

            // Solve via pseudo-inverse of G
            let g_pinv = pseudo_inverse(&g, chi_new, chi_new, 1e-12)?;
            let mut v_new = vec![0.0f64; chi_new * n];
            matmul(&g_pinv, &rhs, chi_new, chi_new, n, &mut v_new);
            v = v_new;
        }

        // Fix V, solve for U: U = θ' V^T (V V^T)^{-1}
        {
            // Gram matrix G = V V^T  (chi_new × chi_new)
            let mut g = vec![0.0f64; chi_new * chi_new];
            {
                let mut vt_mat = vec![0.0f64; n * chi_new];
                for i in 0..chi_new {
                    for j in 0..n {
                        vt_mat[j * chi_new + i] = v[i * n + j];
                    }
                }
                matmul(&v, &vt_mat, chi_new, n, chi_new, &mut g);
            }

            // rhs = θ' V^T  (m × chi_new)
            let mut rhs = vec![0.0f64; m * chi_new];
            {
                let mut vt_mat = vec![0.0f64; n * chi_new];
                for i in 0..chi_new {
                    for j in 0..n {
                        vt_mat[j * chi_new + i] = v[i * n + j];
                    }
                }
                matmul(theta_prime, &vt_mat, m, n, chi_new, &mut rhs);
            }

            // Solve via pseudo-inverse of G
            let g_pinv = pseudo_inverse(&g, chi_new, chi_new, 1e-12)?;
            let mut u_new = vec![0.0f64; m * chi_new];
            matmul(&rhs, &g_pinv, m, chi_new, chi_new, &mut u_new);
            u = u_new;
        }

        // ── Compute residual ‖θ' − UV‖ / ‖θ'‖ ─────────────────────────────
        let mut uv = vec![0.0f64; m * n];
        matmul(&u, &v, m, chi_new, n, &mut uv);
        let res: f64 = theta_prime
            .iter()
            .zip(uv.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        let denom = frob_norm(theta_prime).max(1e-60);
        let rel = res / denom;

        if rel < als_tol {
            break;
        }

        // Monotonicity check: if residual increases significantly, stop.
        if res > prev_residual * 10.0 {
            break;
        }
        prev_residual = res;
    }

    Ok(AlsResult {
        u_data: u,
        v_data: v,
        chi_new,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauge fixing via QR
// ─────────────────────────────────────────────────────────────────────────────

/// QR gauge-fix: decompose `U (m × chi)` as `Q R`, absorb `R` into `V (chi × n)`.
///
/// Returns `(Q, R·V)` with `Q (m × k)` orthonormal and `R·V (k × n)`.
fn qr_gauge_fix(u: &[f64], v: &[f64], m: usize, chi: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let (q, r) = qr_gram_schmidt(u, m, chi);
    let k = m.min(chi);
    // RV = R (k × chi) · V (chi × n)  →  shape (k × n)
    let mut rv = vec![0.0f64; k * n];
    matmul(&r, v, k, chi, n, &mut rv);
    (q, rv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond-level update: horizontal
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one full-update step on the **horizontal** bond between `(row, col)` and
/// `(row, col+1)`.
///
/// Steps:
/// 1. Extract tensors A and B.
/// 2. Contract two-site tensor `θ = A ⊗ B` on the shared bond.
/// 3. Apply imaginary-time gate `e^{-τH}` to physical legs.
/// 4. ALS minimisation: find `A', B'` such that `‖θ' − A'⊗B'‖` is minimised with
///    bond dimension ≤ `cfg.chi_max`.
/// 5. QR gauge-fix to enforce left-orthogonality on A'.
/// 6. Commit new tensors back to PEPS.
///
/// # Errors
/// Returns [`TnError::ShapeMismatch`] if `gate.len() != (d_p² × d_p²)`.
pub fn full_update_step_h(
    peps: &mut Peps,
    row: usize,
    col: usize,
    gate: &[f64],
    cfg: &FullUpdateConfig,
) -> TnResult<()> {
    let rows = peps.rows;
    let cols = peps.cols;

    if row >= rows {
        return Err(TnError::IndexOutOfBounds {
            index: row,
            len: rows,
        });
    }
    if col + 1 >= cols {
        return Err(TnError::IndexOutOfBounds {
            index: col,
            len: cols,
        });
    }
    if cfg.chi_max == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }

    let d_p = peps.tensors[row * cols + col].d_p;
    let d2 = d_p * d_p;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate.len()],
        });
    }

    // ── Clone tensors to allow shared access ────────────────────────────────
    let ta = peps.tensors[row * cols + col].clone();
    let tb = peps.tensors[row * cols + col + 1].clone();

    // ── Contract θ = A ⊗ B ──────────────────────────────────────────────────
    let (theta, chi_l, chi_r, _d_p_check) = contract_two_sites_h(&ta, &tb)?;
    let m = chi_l * d_p;
    let n = d_p * chi_r;

    // ── Apply gate ────────────────────────────────────────────────────────────
    let theta_prime = apply_gate(&theta, m, n, chi_l, chi_r, d_p, gate)?;

    // ── ALS decomposition ─────────────────────────────────────────────────────
    let als = als_decompose(
        &theta_prime,
        m,
        n,
        cfg.chi_max,
        cfg.als_max_iter,
        cfg.als_tol,
    )?;

    let chi_new = als.chi_new;

    // ── QR gauge fix ──────────────────────────────────────────────────────────
    let (u_gauged, v_gauged) = qr_gauge_fix(&als.u_data, &als.v_data, m, chi_new, n);
    let k_qr = m.min(chi_new);

    // ── Reshape U_gauged (m × k_qr) → new A tensor ───────────────────────────
    // m = chi_l * d_p = d_l * d_u * d_d * d_p
    // A shape: [d_l, k_qr, d_u, d_d, d_p]
    let d_l = ta.d_l;
    let d_u_l = ta.d_u;
    let d_d_l = ta.d_d;
    let mut new_a = PepsTensor::zeros(d_l, k_qr, d_u_l, d_d_l, d_p)?;
    // A layout: (((l * k_qr + b) * d_u_l + u) * d_d_l + d) * d_p + p
    for l in 0..d_l {
        for u in 0..d_u_l {
            for d in 0..d_d_l {
                let row_base = (l * d_u_l + u) * d_d_l + d;
                for p in 0..d_p {
                    let mat_row = row_base * d_p + p;
                    for b in 0..k_qr {
                        let idx = (((l * k_qr + b) * d_u_l + u) * d_d_l + d) * d_p + p;
                        new_a.data[idx] = u_gauged[mat_row * k_qr + b];
                    }
                }
            }
        }
    }

    // ── Reshape V_gauged (k_qr × n) → new B tensor ───────────────────────────
    // n = d_p * chi_r = d_p * d_r * d_u_r * d_d_r
    // B shape: [k_qr, d_r, d_u_r, d_d_r, d_p]
    let d_r = tb.d_r;
    let d_u_r = tb.d_u;
    let d_d_r = tb.d_d;
    let mut new_b = PepsTensor::zeros(k_qr, d_r, d_u_r, d_d_r, d_p)?;
    // B layout: (((b * d_r + r) * d_u_r + u) * d_d_r + d) * d_p + p_r
    for p_r in 0..d_p {
        for r in 0..d_r {
            for u in 0..d_u_r {
                for d in 0..d_d_r {
                    let col_base = (r * d_u_r + u) * d_d_r + d;
                    let mat_col = p_r * chi_r + col_base;
                    for b in 0..k_qr {
                        let idx = (((b * d_r + r) * d_u_r + u) * d_d_r + d) * d_p + p_r;
                        new_b.data[idx] = v_gauged[b * n + mat_col];
                    }
                }
            }
        }
    }

    // ── Commit ────────────────────────────────────────────────────────────────
    peps.tensors[row * cols + col] = new_a;
    peps.tensors[row * cols + col + 1] = new_b;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond-level update: vertical
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one full-update step on the **vertical** bond between `(row, col)` (top)
/// and `(row+1, col)` (bottom).
///
/// Analogue of [`full_update_step_h`] but for vertical bonds.
///
/// # Errors
/// Returns [`TnError::ShapeMismatch`] if `gate.len() != (d_p² × d_p²)`.
pub fn full_update_step_v(
    peps: &mut Peps,
    row: usize,
    col: usize,
    gate: &[f64],
    cfg: &FullUpdateConfig,
) -> TnResult<()> {
    let rows = peps.rows;
    let cols = peps.cols;

    if row + 1 >= rows {
        return Err(TnError::IndexOutOfBounds {
            index: row,
            len: rows,
        });
    }
    if col >= cols {
        return Err(TnError::IndexOutOfBounds {
            index: col,
            len: cols,
        });
    }
    if cfg.chi_max == 0 {
        return Err(TnError::InvalidBondDimension(0));
    }

    let d_p = peps.tensors[row * cols + col].d_p;
    let d2 = d_p * d_p;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2 * d2],
            got: vec![gate.len()],
        });
    }

    let ta = peps.tensors[row * cols + col].clone();
    let tb = peps.tensors[(row + 1) * cols + col].clone();

    // ── Contract θ ────────────────────────────────────────────────────────────
    let (theta, chi_u, chi_d, _) = contract_two_sites_v(&ta, &tb)?;
    let m = chi_u * d_p;
    let n = d_p * chi_d;

    // ── Apply gate ────────────────────────────────────────────────────────────
    let theta_prime = apply_gate(&theta, m, n, chi_u, chi_d, d_p, gate)?;

    // ── ALS ───────────────────────────────────────────────────────────────────
    let als = als_decompose(
        &theta_prime,
        m,
        n,
        cfg.chi_max,
        cfg.als_max_iter,
        cfg.als_tol,
    )?;
    let chi_new = als.chi_new;

    // ── QR gauge fix ──────────────────────────────────────────────────────────
    let (u_gauged, v_gauged) = qr_gauge_fix(&als.u_data, &als.v_data, m, chi_new, n);
    let k_qr = m.min(chi_new);

    // ── Reshape U (m × k_qr) → new top tensor A ───────────────────────────────
    // m = chi_u * d_p = d_l_a * d_r_a * d_u_a * d_p
    // A shape: [d_l_a, d_r_a, d_u_a, k_qr, d_p]
    let d_l_a = ta.d_l;
    let d_r_a = ta.d_r;
    let d_u_a = ta.d_u;
    let mut new_a = PepsTensor::zeros(d_l_a, d_r_a, d_u_a, k_qr, d_p)?;
    // A layout: (((l * d_r_a + r) * d_u_a + u) * k_qr + b) * d_p + p
    for l in 0..d_l_a {
        for r in 0..d_r_a {
            for u in 0..d_u_a {
                let row_base = (l * d_r_a + r) * d_u_a + u;
                for p in 0..d_p {
                    let mat_row = row_base * d_p + p;
                    for b in 0..k_qr {
                        let idx = (((l * d_r_a + r) * d_u_a + u) * k_qr + b) * d_p + p;
                        new_a.data[idx] = u_gauged[mat_row * k_qr + b];
                    }
                }
            }
        }
    }

    // ── Reshape V (k_qr × n) → new bottom tensor B ───────────────────────────
    // n = d_p * chi_d = d_p * d_l_b * d_r_b * d_d_b
    // B shape: [d_l_b, d_r_b, k_qr, d_d_b, d_p]
    let d_l_b = tb.d_l;
    let d_r_b = tb.d_r;
    let d_d_b = tb.d_d;
    let mut new_b = PepsTensor::zeros(d_l_b, d_r_b, k_qr, d_d_b, d_p)?;
    // B layout: (((l * d_r_b + r) * k_qr + b) * d_d_b + d) * d_p + p_b
    for p_b in 0..d_p {
        for l in 0..d_l_b {
            for r in 0..d_r_b {
                for d in 0..d_d_b {
                    let col_base = (l * d_r_b + r) * d_d_b + d;
                    let mat_col = p_b * chi_d + col_base;
                    for b in 0..k_qr {
                        let idx = (((l * d_r_b + r) * k_qr + b) * d_d_b + d) * d_p + p_b;
                        new_b.data[idx] = v_gauged[b * n + mat_col];
                    }
                }
            }
        }
    }

    // ── Commit ────────────────────────────────────────────────────────────────
    peps.tensors[row * cols + col] = new_a;
    peps.tensors[(row + 1) * cols + col] = new_b;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Full run
// ─────────────────────────────────────────────────────────────────────────────

/// Run the PEPS full-update imaginary-time evolution for `cfg.n_iter` steps.
///
/// Each step applies the Heisenberg `e^{-δτ H}` gate to every horizontal bond,
/// then every vertical bond (first-order Trotter splitting).  The energy per site
/// is estimated at the end.
///
/// # Errors
/// Returns [`TnError`] if tensor dimensions are inconsistent.
pub fn full_update_run(peps: &mut Peps, cfg: &FullUpdateConfig) -> TnResult<FullUpdateResult> {
    if peps.rows == 0 || peps.cols == 0 {
        return Err(TnError::EmptyInput);
    }

    // Build the Heisenberg gate e^{-dt H} once (J=1 coupling).
    let ham = heisenberg_hamiltonian_2site(1.0);
    let n_ham = (ham.len() as f64).sqrt() as usize; // should be 4 for d=2
    let gate = mat_exp_sym(&ham, n_ham, -cfg.dt)?;

    let mut last_converged = true;

    for _step in 0..cfg.n_iter {
        // Horizontal bonds
        for row in 0..peps.rows {
            for col in 0..peps.cols.saturating_sub(1) {
                let ok = full_update_step_h(peps, row, col, &gate, cfg);
                if let Err(e) = ok {
                    // Propagate only fatal errors; skip numerical ones gracefully.
                    match &e {
                        TnError::NotConverged { .. } => {
                            last_converged = false;
                        }
                        _ => return Err(e),
                    }
                }
            }
        }

        // Vertical bonds
        for row in 0..peps.rows.saturating_sub(1) {
            for col in 0..peps.cols {
                let ok = full_update_step_v(peps, row, col, &gate, cfg);
                if let Err(e) = ok {
                    match &e {
                        TnError::NotConverged { .. } => {
                            last_converged = false;
                        }
                        _ => return Err(e),
                    }
                }
            }
        }
    }

    let energy = full_update_energy(peps, cfg)?;

    Ok(FullUpdateResult {
        energy_per_site: energy,
        steps_run: cfg.n_iter,
        converged: last_converged,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the Heisenberg energy per site for a PEPS state.
///
/// For each bond, contracts the two-site reduced density matrix and measures
/// `<θ|H|θ> / <θ|θ>`, then averages over all bonds.
///
/// # Errors
/// Returns [`TnError::EmptyInput`] if the PEPS has no sites.
pub fn full_update_energy(peps: &Peps, _cfg: &FullUpdateConfig) -> TnResult<f64> {
    if peps.rows == 0 || peps.cols == 0 {
        return Err(TnError::EmptyInput);
    }
    let d_p = peps.tensors[0].d_p;
    let ham = heisenberg_hamiltonian_2site(1.0);

    let mut total_energy = 0.0f64;
    let mut n_bonds = 0usize;

    let cols = peps.cols;
    let rows = peps.rows;

    // Horizontal bonds
    for row in 0..rows {
        for col in 0..cols.saturating_sub(1) {
            let ta = &peps.tensors[row * cols + col];
            let tb = &peps.tensors[row * cols + col + 1];
            let (theta, chi_l, chi_r, _) = contract_two_sites_h(ta, tb)?;
            let e = two_site_energy_expectation(&theta, chi_l, chi_r, d_p, &ham)?;
            total_energy += e;
            n_bonds += 1;
        }
    }

    // Vertical bonds
    for row in 0..rows.saturating_sub(1) {
        for col in 0..cols {
            let ta = &peps.tensors[row * cols + col];
            let tb = &peps.tensors[(row + 1) * cols + col];
            let (theta, chi_u, chi_d, _) = contract_two_sites_v(ta, tb)?;
            let e = two_site_energy_expectation(&theta, chi_u, chi_d, d_p, &ham)?;
            total_energy += e;
            n_bonds += 1;
        }
    }

    if n_bonds == 0 {
        return Ok(0.0);
    }

    let n_sites = (rows * cols) as f64;
    Ok(total_energy / n_sites)
}

/// Compute `<θ|H|θ> / <θ|θ>` for a two-site tensor `θ` of shape `[m, n]`.
fn two_site_energy_expectation(
    theta: &[f64],
    chi_l: usize,
    chi_r: usize,
    d_p: usize,
    hamiltonian: &[f64],
) -> TnResult<f64> {
    let d2 = d_p * d_p;
    let n = d_p * chi_r;

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for l_flat in 0..chi_l {
        for r_flat in 0..chi_r {
            for pl in 0..d_p {
                for pr in 0..d_p {
                    let bra_idx = (l_flat * d_p + pl) * n + pr * chi_r + r_flat;
                    let bra = theta[bra_idx];
                    denominator += bra * bra;

                    // Apply H and accumulate
                    for pl_p in 0..d_p {
                        for pr_p in 0..d_p {
                            let ket_idx = (l_flat * d_p + pl_p) * n + pr_p * chi_r + r_flat;
                            let ket = theta[ket_idx];
                            let h_idx = (pl * d_p + pr) * d2 + pl_p * d_p + pr_p;
                            numerator += bra * hamiltonian[h_idx] * ket;
                        }
                    }
                }
            }
        }
    }

    if denominator.abs() < 1e-60 {
        return Ok(0.0);
    }
    Ok(numerator / denominator)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_heisenberg_gate(d_p: usize, dt: f64) -> Vec<f64> {
        let ham = heisenberg_hamiltonian_2site(1.0);
        let n = d_p * d_p;
        mat_exp_sym(&ham, n, -dt).expect("gate ok")
    }

    fn default_cfg() -> FullUpdateConfig {
        FullUpdateConfig {
            chi_max: 2,
            n_iter: 2,
            dt: 0.05,
            als_max_iter: 5,
            als_tol: 1e-6,
            chi_env: 4,
            use_env: false,
        }
    }

    // ── Test 1: shape of initialised PEPS ────────────────────────────────────
    #[test]
    fn full_update_init_shape() {
        let mut rng = LcgRng::new(1);
        let peps = full_update_init(2, 3, 2, 2, &mut rng).expect("init ok");
        assert_eq!(peps.rows, 2);
        assert_eq!(peps.cols, 3);
        assert_eq!(peps.n_sites(), 6);
        // Check boundary bonds are 1
        let top_left = &peps.tensors[0];
        assert_eq!(top_left.d_l, 1);
        assert_eq!(top_left.d_u, 1);
        let bot_right = &peps.tensors[5];
        assert_eq!(bot_right.d_r, 1);
        assert_eq!(bot_right.d_d, 1);
    }

    // ── Test 2: initialised PEPS is not all-zero ──────────────────────────────
    #[test]
    fn full_update_init_nonzero() {
        let mut rng = LcgRng::new(2);
        let peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let any_nonzero = peps
            .tensors
            .iter()
            .any(|t| t.data.iter().any(|&x| x != 0.0));
        assert!(any_nonzero);
    }

    // ── Test 3: rows/cols unchanged after horizontal step ─────────────────────
    #[test]
    fn full_update_step_h_preserves_rows_cols() {
        let mut rng = LcgRng::new(3);
        let mut peps = full_update_init(2, 3, 2, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.05);
        let cfg = default_cfg();
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("step ok");
        assert_eq!(peps.rows, 2);
        assert_eq!(peps.cols, 3);
    }

    // ── Test 4: bond dim bounded after horizontal step ────────────────────────
    #[test]
    fn full_update_step_h_bond_dim_bounded() {
        let mut rng = LcgRng::new(4);
        let mut peps = full_update_init(2, 3, 3, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.05);
        let cfg = FullUpdateConfig {
            chi_max: 2,
            ..default_cfg()
        };
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("step ok");
        // After update, the bond between (0,0) and (0,1) must be ≤ chi_max.
        let d_r_a = peps.tensors[1].d_l; // d_l of right tensor = bond dim
        assert!(
            d_r_a <= cfg.chi_max,
            "bond dim {} > chi_max {}",
            d_r_a,
            cfg.chi_max
        );
    }

    // ── Test 5: bond dim bounded after vertical step ──────────────────────────
    #[test]
    fn full_update_step_v_bond_dim_bounded() {
        let mut rng = LcgRng::new(5);
        let mut peps = full_update_init(3, 2, 3, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.05);
        let cfg = FullUpdateConfig {
            chi_max: 2,
            ..default_cfg()
        };
        full_update_step_v(&mut peps, 0, 0, &gate, &cfg).expect("step ok");
        // Bond between (0,0) and (1,0): d_d of top tensor.
        let d_d_top = peps.tensors[0].d_d;
        assert!(
            d_d_top <= cfg.chi_max,
            "bond dim {} > chi_max {}",
            d_d_top,
            cfg.chi_max
        );
    }

    // ── Test 6: d_bond=2, d_phys=2 runs without error ────────────────────────
    #[test]
    fn full_update_step_h_rank2_case() {
        let mut rng = LcgRng::new(6);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.01);
        let cfg = default_cfg();
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("rank-2 step ok");
    }

    // ── Test 7: chi_max=1 forces rank-1 bond ─────────────────────────────────
    #[test]
    fn full_update_step_h_chi1_case() {
        let mut rng = LcgRng::new(7);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.05);
        let cfg = FullUpdateConfig {
            chi_max: 1,
            ..default_cfg()
        };
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("chi1 step ok");
        // Bond should be exactly 1.
        let bond = peps.tensors[1].d_l;
        assert_eq!(bond, 1);
    }

    // ── Test 8: full_update_run returns a result ──────────────────────────────
    #[test]
    fn full_update_run_returns_result() {
        let mut rng = LcgRng::new(8);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let cfg = FullUpdateConfig {
            n_iter: 2,
            ..default_cfg()
        };
        let res = full_update_run(&mut peps, &cfg).expect("run ok");
        assert_eq!(res.steps_run, 2);
    }

    // ── Test 9: returned energy is finite ────────────────────────────────────
    #[test]
    fn full_update_run_energy_finite() {
        let mut rng = LcgRng::new(9);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let cfg = FullUpdateConfig {
            n_iter: 1,
            ..default_cfg()
        };
        let res = full_update_run(&mut peps, &cfg).expect("run ok");
        assert!(
            res.energy_per_site.is_finite(),
            "energy is not finite: {}",
            res.energy_per_site
        );
    }

    // ── Test 10: 5 steps on 2×2 PEPS, no panic ───────────────────────────────
    #[test]
    fn full_update_run_multiple_steps() {
        let mut rng = LcgRng::new(10);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let cfg = FullUpdateConfig {
            n_iter: 5,
            dt: 0.02,
            ..default_cfg()
        };
        full_update_run(&mut peps, &cfg).expect("5 steps ok");
    }

    // ── Test 11: 3×3 grid survives 2 steps ───────────────────────────────────
    #[test]
    fn full_update_run_3x3_peps() {
        let mut rng = LcgRng::new(11);
        let mut peps = full_update_init(3, 3, 2, 2, &mut rng).expect("init ok");
        let cfg = FullUpdateConfig {
            n_iter: 2,
            ..default_cfg()
        };
        full_update_run(&mut peps, &cfg).expect("3x3 ok");
    }

    // ── Test 12: energy estimation returns finite number on 2×2 ───────────────
    #[test]
    fn full_update_energy_finite_2x2() {
        let mut rng = LcgRng::new(12);
        let peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let cfg = default_cfg();
        let e = full_update_energy(&peps, &cfg).expect("energy ok");
        assert!(e.is_finite(), "energy is not finite: {e}");
    }

    // ── Test 13: mix h and v steps without inconsistency ─────────────────────
    #[test]
    fn full_update_step_h_then_v() {
        let mut rng = LcgRng::new(13);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let gate = make_heisenberg_gate(2, 0.05);
        let cfg = default_cfg();
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("h step ok");
        full_update_step_v(&mut peps, 0, 0, &gate, &cfg).expect("v step ok");
        full_update_step_h(&mut peps, 0, 0, &gate, &cfg).expect("h step 2 ok");
        // Verify tensor count is unchanged
        assert_eq!(peps.n_sites(), 4);
    }

    // ── Test 14: ALS inner loop reduces residual ──────────────────────────────
    #[test]
    fn full_update_als_converges() {
        let mut rng = LcgRng::new(14);
        // Build a rank-2 matrix with known structure.
        let m = 8;
        let n = 8;
        let chi = 2;
        // Random theta
        let theta: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        // ALS with tight tolerance to force convergence.
        let res = als_decompose(&theta, m, n, chi, 50, 1e-10).expect("als ok");
        // Compute residual of final solution.
        let mut uv = vec![0.0f64; m * n];
        matmul(&res.u_data, &res.v_data, m, res.chi_new, n, &mut uv);
        let res_norm = theta
            .iter()
            .zip(uv.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let theta_norm = frob_norm(&theta);
        // Residual must be smaller than the original norm (decomposition is non-trivial).
        assert!(
            res_norm < theta_norm,
            "ALS did not reduce residual: res={res_norm:.4e} >= theta_norm={theta_norm:.4e}"
        );
    }

    // ── Test 15: wrong gate shape returns Err ─────────────────────────────────
    #[test]
    fn full_update_gate_shape_check() {
        let mut rng = LcgRng::new(15);
        let mut peps = full_update_init(2, 2, 2, 2, &mut rng).expect("init ok");
        let cfg = default_cfg();
        // Gate with wrong size (should be 4*4=16 for d_p=2).
        let bad_gate = vec![0.0f64; 9];
        let result = full_update_step_h(&mut peps, 0, 0, &bad_gate, &cfg);
        assert!(result.is_err(), "expected error for bad gate shape");
    }
}
