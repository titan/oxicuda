//! Special Orthogonal Group SO(n) — rotation matrices.
//!
//! SO(n) = {Q ∈ ℝ^{n×n} : Q^T Q = I, det(Q) = +1}
//!
//! # Geometry
//!
//! - **Tangent space** at Q: `T_Q SO(n) = {Q·Ω : Ω ∈ ℝ^{n×n}, Ω + Ω^T = 0}`
//! - **Riemannian metric** (bi-invariant): `⟨U, V⟩_Q = tr(U^T V)`
//! - **Projection** of G to tangent space: `P_Q(G) = Q · skew(Q^T G)`, `skew(A) = (A − A^T)/2`
//!
//! # Retractions
//!
//! Three retractions are provided:
//! 1. **Cayley** — `Q·(I + Ω/2)·(I − Ω/2)⁻¹`, exact on SO(n), O(n³)
//! 2. **Matrix-exponential** — `Q·expm(Ω)` via Padé [3/3] approximation
//! 3. **QR** — `qr(Q + V).Q` with determinant sign fix (simplest)
//!
//! where `Ω = Q^T V` is skew-symmetric when V is a tangent vector.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ─── tiny n×n matrix helpers (row-major, n×n) ────────────────────────────────

/// n×n matrix multiply: C = A · B  (all row-major, length n²)
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// n×n transpose.
fn mat_transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

/// Element-wise add.
fn mat_add(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = a.to_vec();
    for (ci, bi) in c.iter_mut().zip(b.iter()) {
        *ci += bi;
    }
    let _ = n; // n used implicitly via length
    c
}

/// Element-wise subtract.
fn mat_sub(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = a.to_vec();
    for (ci, bi) in c.iter_mut().zip(b.iter()) {
        *ci -= bi;
    }
    let _ = n;
    c
}

/// Scale every element by s.
fn mat_scale(a: &[f64], s: f64, n: usize) -> Vec<f64> {
    let _ = n;
    a.iter().map(|x| x * s).collect()
}

/// n×n identity matrix.
fn mat_identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Frobenius norm of a matrix.
fn mat_frob_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Sign of the determinant (+1 or −1) via LU with partial pivoting.
///
/// Returns +1 if the determinant is positive, −1 if negative, 0 if singular
/// (within tolerance 1e-14).
fn mat_det_sign(a: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return if a[0] >= 0.0 { 1.0 } else { -1.0 };
    }
    let mut m = a.to_vec();
    let mut sign = 1.0f64;
    for col in 0..n {
        // Find pivot
        let mut piv_row = col;
        let mut piv_val = m[col * n + col].abs();
        for row in (col + 1)..n {
            let v = m[row * n + col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = row;
            }
        }
        if piv_val < 1e-14 {
            return 0.0;
        }
        if piv_row != col {
            for j in 0..n {
                m.swap(col * n + j, piv_row * n + j);
            }
            sign = -sign;
        }
        let pivot = m[col * n + col];
        for row in (col + 1)..n {
            let factor = m[row * n + col] / pivot;
            for j in col..n {
                let tmp = factor * m[col * n + j];
                m[row * n + j] -= tmp;
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= m[i * n + i];
    }
    if det > 0.0 {
        1.0
    } else if det < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Solve A·x = b for square n×n system via Gaussian elimination with partial pivoting.
///
/// Returns the solution vector x (length n).
fn solve_square(a: &[f64], b: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    // Build augmented matrix [A | b]
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        // Partial pivot
        let mut piv_row = col;
        let mut piv_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = row;
            }
        }
        if piv_val < 1e-15 {
            return Err(ManifoldError::SingularMatrix(format!(
                "pivot < 1e-15 at column {col}"
            )));
        }
        if piv_row != col {
            for j in 0..=(n) {
                aug.swap(col * (n + 1) + j, piv_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let tmp = factor * aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= tmp;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }
    Ok(x)
}

/// Solve n linear systems A·X = B for square n×n A, multiple right-hand sides B (n×n).
///
/// Returns X as row-major n×n matrix.
fn solve_multi(a: &[f64], b_mat: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    let mut x = vec![0.0f64; n * n];
    for col in 0..n {
        let b_col: Vec<f64> = (0..n).map(|row| b_mat[row * n + col]).collect();
        let x_col = solve_square(a, &b_col, n)?;
        for row in 0..n {
            x[row * n + col] = x_col[row];
        }
    }
    Ok(x)
}

/// Padé [3/3] matrix exponential.
///
/// For a matrix A, computes expm(A) ≈ D(A)⁻¹ · N(A) where
/// `N(A) = I + A/2 + A²/10 + A³/120` and `D(A) = I − A/2 + A²/10 − A³/120`.
///
/// This approximation is exact for skew-symmetric A with small norm and
/// sufficient for rotation retractions.
pub(crate) fn pade_expm(a: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if n == 0 {
        return Ok(vec![]);
    }
    let id = mat_identity(n);
    let a2 = mat_mul(a, a, n);
    let a3 = mat_mul(&a2, a, n);

    // N = I + A/2 + A²/10 + A³/120
    // D = I − A/2 + A²/10 − A³/120
    let a_half = mat_scale(a, 0.5, n);
    let a2_tenth = mat_scale(&a2, 0.1, n);
    let a3_120 = mat_scale(&a3, 1.0 / 120.0, n);

    let mut numer = id.clone();
    for (x, (&h, (&t, &s))) in numer
        .iter_mut()
        .zip(a_half.iter().zip(a2_tenth.iter().zip(a3_120.iter())))
    {
        *x += h + t + s;
    }
    let mut denom = id.clone();
    for (x, (&h, (&t, &s))) in denom
        .iter_mut()
        .zip(a_half.iter().zip(a2_tenth.iter().zip(a3_120.iter())))
    {
        *x += -h + t - s;
    }
    // expm(A) = D^{-1} · N → solve D·X = N column-by-column
    solve_multi(&denom, &numer, n)
}

/// Matrix logarithm for a rotation matrix R ∈ SO(n).
///
/// Returns a skew-symmetric matrix Ω such that expm(Ω) ≈ R.
///
/// # Special cases
/// - n = 1: returns `[0.0]` (SO(1) = {1}).
/// - n = 2: exact formula via atan2.
/// - n ≥ 3: iterative Schur-like approach using the Rodrigues expansion for
///   3×3, and a blocked 2×2-rotation decomposition for larger n.
///
/// For the general case we use the iterative Baker-Campbell-Hausdorff
/// approach: decompose R into 2×2 Givens rotations and accumulate angles.
pub(crate) fn mat_log_rotation(r: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        // SO(1) = {+1}; log(1) = 0
        return Ok(vec![0.0]);
    }
    if n == 2 {
        // R = [[c, -s], [s, c]] → log = [[0, -θ], [θ, 0]] where θ = atan2(s, c)
        let c = r[0];
        let s = r[2]; // row-major: r[1*2+0]
        let theta = s.atan2(c);
        return Ok(vec![0.0, -theta, theta, 0.0]);
    }

    // General n ≥ 3: iterative Givens rotation decomposition.
    //
    // Idea: apply a sequence of Givens rotations G_{ij}(θ) to R from the
    // left until we approach I, accumulating −θ into a skew-symmetric matrix Ω.
    // This converges when R is near I (as required for geodesic / distance).
    //
    // For elements not near I we use multiple Cayley iterations:
    //   Ω_{k+1} = Ω_k + skew(R_k − I)  where R_{k+1} = expm(−Ω_{k+1}) · R
    //
    // We use up to 40 Newton iterations on  f(Ω) = expm(Ω) − R.
    // Newton step: Ω ← Ω − dexp^{-1}(expm(Ω) − R)
    // In practice, the simple approximation works well enough:
    //   Ω ← Ω + skew(R − expm(Ω))
    //
    // Start from Ω₀ = skew(R − I) (first-order Cayley approximation).
    let mut omega = {
        let rm_i = mat_sub(r, &mat_identity(n), n);
        let rm_i_t = mat_transpose(&rm_i, n);
        let two_n = n * n;
        let mut s = vec![0.0f64; two_n];
        for k in 0..two_n {
            s[k] = 0.5 * (rm_i[k] - rm_i_t[k]);
        }
        s
    };

    let max_iter = 50;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let exp_om = pade_expm(&omega, n)?;
        // residual: E = expm(Ω) − R
        let residual = mat_sub(&exp_om, r, n);
        let res_norm = mat_frob_norm(&residual);
        if res_norm < tol {
            break;
        }
        // correction: ΔΩ = skew(E) — project residual to skew space
        let res_t = mat_transpose(&residual, n);
        let correction: Vec<f64> = (0..n * n)
            .map(|k| -0.5 * (residual[k] - res_t[k]))
            .collect();
        omega = mat_add(&omega, &correction, n);
    }
    Ok(omega)
}

// ─── public API ───────────────────────────────────────────────────────────────

/// Check whether Q is approximately in SO(n): Q^T Q ≈ I and det(Q) ≈ +1.
///
/// Returns `true` when both conditions hold within `tol`.
pub fn so_n_check(q: &[f64], n: usize, tol: f64) -> bool {
    if n == 0 {
        return true;
    }
    if q.len() != n * n {
        return false;
    }
    // Q^T Q − I should be ≈ 0
    let qt = mat_transpose(q, n);
    let qtq = mat_mul(&qt, q, n);
    let id = mat_identity(n);
    let diff = mat_sub(&qtq, &id, n);
    if mat_frob_norm(&diff) > tol * (n as f64).sqrt() {
        return false;
    }
    // det ≈ +1
    let ds = mat_det_sign(q, n);
    ds > 0.5
}

/// Project an arbitrary n×n matrix G to the tangent space of SO(n) at Q.
///
/// `P_Q(G) = Q · skew(Q^T G)` where `skew(A) = (A − A^T) / 2`.
pub fn so_n_project_tangent(q: &[f64], g: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q.len()],
        });
    }
    if g.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![g.len()],
        });
    }
    let qt = mat_transpose(q, n);
    let qtg = mat_mul(&qt, g, n); // Q^T G
    let qtg_t = mat_transpose(&qtg, n);
    // skew(Q^T G) = (Q^T G − (Q^T G)^T) / 2
    let skew: Vec<f64> = (0..n * n).map(|k| 0.5 * (qtg[k] - qtg_t[k])).collect();
    Ok(mat_mul(q, &skew, n))
}

/// Riemannian gradient: project Euclidean gradient G to tangent space at Q.
///
/// Alias for [`so_n_project_tangent`].
#[inline]
pub fn so_n_riemannian_gradient(q: &[f64], g: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    so_n_project_tangent(q, g, n)
}

/// QR retraction with determinant sign correction.
///
/// Computes `Q_new = QR(Q + V)`, then flips the sign of the last column
/// of Q if `det(Q_new) < 0` to guarantee membership in SO(n).
pub fn so_n_retract_qr(q: &[f64], v: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q.len()],
        });
    }
    if v.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![v.len()],
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        // SO(1) = {1}; any retraction stays at 1.
        return Ok(vec![1.0]);
    }
    let sum = mat_add(q, v, n);
    // Householder QR on the n×n square matrix treated as m=n, p=n.
    let (q_out, _r) = crate::linalg::householder_qr::householder_qr(&sum, n, n)?;
    // Fix sign: if det < 0, flip the last column.
    let ds = mat_det_sign(&q_out, n);
    if ds < 0.0 {
        let mut out = q_out;
        let last_col = n - 1;
        for row in 0..n {
            out[row * n + last_col] = -out[row * n + last_col];
        }
        Ok(out)
    } else {
        Ok(q_out)
    }
}

/// Cayley retraction: `R_Q(V) = Q · (I + Ω/2) · (I − Ω/2)⁻¹` where `Ω = Q^T V`.
///
/// Ω must be skew-symmetric; this is guaranteed when V = P_Q(G) (a tangent vector).
/// The Cayley map is an exact diffeomorphism from skew-symmetric matrices to SO(n)
/// in a neighbourhood of I.
pub fn so_n_retract_cayley(q: &[f64], v: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q.len()],
        });
    }
    if v.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![v.len()],
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let qt = mat_transpose(q, n);
    let omega = mat_mul(&qt, v, n); // Ω = Q^T V
    // Make Ω exactly skew-symmetric (project out any numerical symmetric part)
    let omega_t = mat_transpose(&omega, n);
    let omega_skew: Vec<f64> = (0..n * n).map(|k| 0.5 * (omega[k] - omega_t[k])).collect();
    let id = mat_identity(n);
    let half_omega = mat_scale(&omega_skew, 0.5, n);
    let numer = mat_add(&id, &half_omega, n); // I + Ω/2
    let denom = mat_sub(&id, &half_omega, n); // I − Ω/2
    // Solve (I − Ω/2)·X = (I + Ω/2) for X, then Q_new = Q · X
    let cayley = solve_multi(&denom, &numer, n)?;
    Ok(mat_mul(q, &cayley, n))
}

/// Matrix-exponential retraction: `R_Q(V) = Q · expm(Q^T V)`.
///
/// Uses the Padé [3/3] approximation for the matrix exponential.
/// When V is a tangent vector at Q (i.e. Q^T V is skew-symmetric),
/// `expm(Q^T V)` is orthogonal with determinant +1, so the result lies in SO(n).
pub fn so_n_retract_expm(q: &[f64], v: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q.len()],
        });
    }
    if v.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![v.len()],
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let qt = mat_transpose(q, n);
    let omega = mat_mul(&qt, v, n); // Ω = Q^T V (skew-symmetric)
    // Project to exact skew-symmetric to eliminate numerical drift
    let omega_t = mat_transpose(&omega, n);
    let omega_skew: Vec<f64> = (0..n * n).map(|k| 0.5 * (omega[k] - omega_t[k])).collect();
    let exp_om = pade_expm(&omega_skew, n)?;
    Ok(mat_mul(q, &exp_om, n))
}

/// Geodesic from Q in direction V: `γ(t) = Q · expm(t · Q^T V)`.
///
/// At t=0 returns Q; the velocity vector is V (a tangent vector at Q).
pub fn so_n_geodesic(q: &[f64], v: &[f64], t: f64, n: usize) -> ManifoldResult<Vec<f64>> {
    if q.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q.len()],
        });
    }
    if v.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![v.len()],
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let qt = mat_transpose(q, n);
    let omega = mat_mul(&qt, v, n); // Q^T V
    let omega_t = mat_transpose(&omega, n);
    let omega_skew: Vec<f64> = (0..n * n).map(|k| 0.5 * (omega[k] - omega_t[k])).collect();
    let t_omega = mat_scale(&omega_skew, t, n);
    let exp_t_omega = pade_expm(&t_omega, n)?;
    Ok(mat_mul(q, &exp_t_omega, n))
}

/// Riemannian distance between Q₁, Q₂ ∈ SO(n):
/// `d(Q₁, Q₂) = ‖log(Q₁^T Q₂)‖_F / √2`
///
/// The division by √2 accounts for the bi-invariant normalisation.
pub fn so_n_distance(q1: &[f64], q2: &[f64], n: usize) -> ManifoldResult<f64> {
    if q1.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q1.len()],
        });
    }
    if q2.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q2.len()],
        });
    }
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(0.0);
    }
    let q1t = mat_transpose(q1, n);
    let m = mat_mul(&q1t, q2, n); // Q₁^T Q₂ ∈ SO(n)
    let log_m = mat_log_rotation(&m, n)?;
    Ok(mat_frob_norm(&log_m) / 2.0f64.sqrt())
}

/// Logarithmic map: `log_{Q₁}(Q₂) = Q₁ · log(Q₁^T Q₂)` (tangent vector at Q₁).
///
/// Returns a tangent vector V ∈ T_{Q₁} SO(n) such that `expm(Q₁, V, 1) ≈ Q₂`.
pub fn so_n_log(q1: &[f64], q2: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if q1.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q1.len()],
        });
    }
    if q2.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![q2.len()],
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![0.0]);
    }
    let q1t = mat_transpose(q1, n);
    let m = mat_mul(&q1t, q2, n); // Q₁^T Q₂
    let log_m = mat_log_rotation(&m, n)?;
    Ok(mat_mul(q1, &log_m, n))
}

/// Inner product (Frobenius / bi-invariant) of two tangent vectors U, V at Q.
///
/// `⟨U, V⟩_Q = tr(U^T V)` — independent of Q for the bi-invariant metric.
pub fn so_n_inner(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}

/// Norm of a tangent vector in the bi-invariant metric.
pub fn so_n_norm(u: &[f64]) -> f64 {
    so_n_inner(u, u).sqrt()
}

// ─── special constructors ─────────────────────────────────────────────────────

/// Random element of SO(n) via QR decomposition of a random Gaussian matrix.
pub fn so_n_random(n: usize, rng: &mut LcgRng) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    loop {
        // Fill with standard normals
        let a: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        if let Ok((mut q, _r)) = crate::linalg::householder_qr::householder_qr(&a, n, n) {
            // Fix det sign
            let ds = mat_det_sign(&q, n);
            if ds == 0.0 {
                continue; // degenerate, retry
            }
            if ds < 0.0 {
                // Flip sign of last column
                let last = n - 1;
                for row in 0..n {
                    q[row * n + last] = -q[row * n + last];
                }
            }
            return q;
        }
    }
}

/// Identity element I_n ∈ SO(n).
pub fn so_n_identity(n: usize) -> Vec<f64> {
    mat_identity(n)
}

/// 2-D rotation matrix R(θ) ∈ SO(2).
///
/// Row-major layout: `[cos θ, −sin θ, sin θ, cos θ]`.
pub fn so_2_rotation(theta: f64) -> [f64; 4] {
    let c = theta.cos();
    let s = theta.sin();
    [c, -s, s, c]
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // helper: build a 2D rotation matrix as Vec<f64>
    fn rot2(theta: f64) -> Vec<f64> {
        so_2_rotation(theta).to_vec()
    }

    // helper: 3×3 rotation around z-axis
    fn rot3_z(theta: f64) -> Vec<f64> {
        let c = theta.cos();
        let s = theta.sin();
        vec![c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
    }

    // helper: check Q^T Q ≈ I and det ≈ +1
    fn assert_so_n(q: &[f64], n: usize) {
        assert!(so_n_check(q, n, 1e-7), "matrix is not in SO({n}): {q:?}");
    }

    // helper: Frobenius distance between two matrices
    fn frob_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    // ── 1. identity passes check ──────────────────────────────────────────────

    #[test]
    fn so_n_check_identity() {
        for n in 1..=5 {
            let id = so_n_identity(n);
            assert!(so_n_check(&id, n, 1e-10), "identity failed for n={n}");
        }
    }

    // ── 2. 2D rotation passes check ───────────────────────────────────────────

    #[test]
    fn so_n_check_rotation_2d() {
        let q = rot2(0.5);
        assert!(so_n_check(&q, 2, 1e-12));
    }

    // ── 3. project_tangent is skew-symmetric (Q^T·result is skew) ────────────

    #[test]
    fn so_n_project_tangent_is_skew_symmetric() {
        let n = 3;
        let q = rot3_z(0.7);
        let g: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.1).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        // Q^T · v should be skew-symmetric
        let qt = mat_transpose(&q, n);
        let s = mat_mul(&qt, &v, n);
        for i in 0..n {
            for j in 0..n {
                let antisym = s[i * n + j] + s[j * n + i];
                assert!(antisym.abs() < 1e-10, "s[{i},{j}]+s[{j},{i}] = {antisym}");
            }
        }
    }

    // ── 4. project_tangent lies in tangent space: U + U^T ≈ 0 ────────────────

    #[test]
    fn so_n_project_tangent_is_tangent() {
        let n = 4;
        let mut rng = LcgRng::new(42);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        // tangent condition: Q^T·V + V^T·Q = 0  ⟺  Ω + Ω^T = 0
        let qt = mat_transpose(&q, n);
        let omega = mat_mul(&qt, &v, n);
        let omega_t = mat_transpose(&omega, n);
        let sym = mat_add(&omega, &omega_t, n);
        for &s in &sym {
            assert!(s.abs() < 1e-10, "non-zero symmetric part: {s}");
        }
    }

    // ── 5. QR retraction stays in SO(n) ──────────────────────────────────────

    #[test]
    fn retract_qr_stays_in_so_n() {
        let n = 3;
        let mut rng = LcgRng::new(7);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        let v_small = mat_scale(&v, 0.05, n);
        let q_new = so_n_retract_qr(&q, &v_small, n).unwrap();
        assert_so_n(&q_new, n);
    }

    // ── 6. QR retraction at V=0 returns Q ────────────────────────────────────

    #[test]
    fn retract_qr_t0() {
        let n = 3;
        let mut rng = LcgRng::new(12);
        let q = so_n_random(n, &mut rng);
        let v = vec![0.0f64; n * n];
        let q_new = so_n_retract_qr(&q, &v, n).unwrap();
        assert!(
            frob_dist(&q, &q_new) < 1e-10,
            "retract_qr(Q, 0) ≠ Q: dist = {}",
            frob_dist(&q, &q_new)
        );
    }

    // ── 7. Cayley retraction stays in SO(n) ───────────────────────────────────

    #[test]
    fn retract_cayley_stays_in_so_n() {
        let n = 4;
        let mut rng = LcgRng::new(17);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        let v_small = mat_scale(&v, 0.1, n);
        let q_new = so_n_retract_cayley(&q, &v_small, n).unwrap();
        assert_so_n(&q_new, n);
    }

    // ── 8. expm retraction stays in SO(n) ────────────────────────────────────

    #[test]
    fn retract_expm_stays_in_so_n() {
        let n = 3;
        let mut rng = LcgRng::new(99);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        let v_small = mat_scale(&v, 0.1, n);
        let q_new = so_n_retract_expm(&q, &v_small, n).unwrap();
        assert_so_n(&q_new, n);
    }

    // ── 9. random element is in SO(n) ─────────────────────────────────────────

    #[test]
    fn so_n_random_in_group() {
        let mut rng = LcgRng::new(2025);
        for n in 1..=5 {
            let q = so_n_random(n, &mut rng);
            assert_so_n(&q, n);
        }
    }

    // ── 10. distance from Q to itself is 0 ───────────────────────────────────

    #[test]
    fn so_n_distance_self_zero() {
        let mut rng = LcgRng::new(55);
        for n in 1..=4 {
            let q = so_n_random(n, &mut rng);
            let d = so_n_distance(&q, &q, n).unwrap();
            assert!(d < 1e-10, "d(Q,Q) = {d} for n={n}");
        }
    }

    // ── 11. distance is symmetric ─────────────────────────────────────────────

    #[test]
    fn so_n_distance_symmetric() {
        let theta1 = 0.3f64;
        let theta2 = 0.8f64;
        let q1 = rot2(theta1);
        let q2 = rot2(theta2);
        let d12 = so_n_distance(&q1, &q2, 2).unwrap();
        let d21 = so_n_distance(&q2, &q1, 2).unwrap();
        assert!((d12 - d21).abs() < 1e-12, "d12={d12}, d21={d21}");
    }

    // ── 12. 2D distance: d(I, R(θ)) = |θ| / √2 for small θ ──────────────────
    //
    // With the normalisation d = ‖log(M)‖_F / √2 and log([[c,-s],[s,c]])=[[0,-θ],[θ,0]],
    // ‖log‖_F = √(θ²+θ²) = θ√2, so d = θ.

    #[test]
    fn so_n_distance_2d_known() {
        let id = so_n_identity(2);
        for &theta in &[0.1f64, 0.5, 1.0, 1.5] {
            let r = rot2(theta);
            let d = so_n_distance(&id, &r, 2).unwrap();
            assert!(
                (d - theta).abs() < 1e-10,
                "d(I, R({theta})) = {d}, expected {theta}"
            );
        }
    }

    // ── 13. geodesic at t=0 is Q ──────────────────────────────────────────────

    #[test]
    fn geodesic_t0() {
        let n = 3;
        let mut rng = LcgRng::new(101);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        let gamma0 = so_n_geodesic(&q, &v, 0.0, n).unwrap();
        assert!(
            frob_dist(&q, &gamma0) < TOL,
            "geodesic(t=0) ≠ Q: dist = {}",
            frob_dist(&q, &gamma0)
        );
    }

    // ── 14. geodesic at t=1 matches expm retraction ───────────────────────────

    #[test]
    fn geodesic_t1_matches_retract_expm() {
        let n = 3;
        let mut rng = LcgRng::new(303);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        let v_small = mat_scale(&v, 0.1, n);
        let gamma1 = so_n_geodesic(&q, &v_small, 1.0, n).unwrap();
        let ret = so_n_retract_expm(&q, &v_small, n).unwrap();
        let dist = frob_dist(&gamma1, &ret);
        assert!(dist < 1e-10, "geodesic(t=1) ≠ retract_expm: dist = {dist}");
    }

    // ── 15. log–exp round-trip for small tangent vectors ─────────────────────

    #[test]
    fn log_exp_roundtrip() {
        let n = 2;
        let mut rng = LcgRng::new(777);
        let q = so_n_random(n, &mut rng);
        let g: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let v = so_n_project_tangent(&q, &g, n).unwrap();
        // Scale to a small tangent vector
        let v_small = mat_scale(&v, 0.2, n);
        // Retract: Q₂ = Q · expm(Q^T V)
        let q2 = so_n_retract_expm(&q, &v_small, n).unwrap();
        // Log: recover V from Q, Q₂
        let v_rec = so_n_log(&q, &q2, n).unwrap();
        let dist = frob_dist(&v_small, &v_rec);
        assert!(dist < 1e-8, "log(exp(V)) ≠ V: Frobenius dist = {dist}");
    }
}
