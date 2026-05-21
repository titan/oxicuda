//! Householder bidiagonalization + implicit QR (Golub–Reinsch) SVD.
//!
//! ## Algorithm overview
//!
//! Given an `m × n` matrix `A` with `m ≥ n`:
//!
//! **Phase 1 — Householder bidiagonalization** transforms `A` into an upper-bidiagonal matrix `B`
//! via orthogonal similarity: `U_0^T A V_0 = B`. Left Householder reflectors reduce columns to
//! multiples of `e_1`; right reflectors zero entries beyond the superdiagonal. Both `U_0` and
//! `V_0` are accumulated explicitly into `m × n` and `n × n` matrices.
//!
//! **Phase 2 — Implicit QR diagonalization** (Golub–Kahan) applies Francis QR steps to `B`.
//! Each step: (1) Wilkinson shift μ from trailing 2×2 of B^T B; (2) launch bulge using
//! `Givens(d[lo]²−μ, d[lo]·e[lo])`; (3) chase via alternating right–left Givens pairs.
//! Tiny off-diagonal entries `|e[k]| ≤ ε(|d[k]|+|d[k+1]|)` are deflated to zero.
//!
//! **Phase 3** sorts singular values descending and adjusts signs of `U` columns.
//!
//! ## References
//!
//! - Golub & Reinsch (1970). Singular value decomposition and least squares solutions.
//!   *Numerische Mathematik* 14, 403–420.
//! - Demmel & Kahan (1990). Accurate singular values of bidiagonal matrices.
//!   *SIAM J. Sci. Stat. Comput.* 11(5), 873–912.
//! - Anderson et al. (1999). *LAPACK Users' Guide* (3rd ed.). SIAM. (DBDSQR routine.)

use crate::{TnError, TnResult};

// ─── constants ────────────────────────────────────────────────────────────────

const EPS: f64 = f64::EPSILON;
const MAX_ITER_PER_SV: usize = 40;

// ─── public API ───────────────────────────────────────────────────────────────

/// Full (thin) SVD via Householder bidiagonalization + implicit QR.
///
/// Returns `(U, s, Vt)` where:
/// - `U` is `m × n` row-major (column-orthonormal)
/// - `s` is length `n`, singular values in descending order
/// - `Vt` is `n × n` row-major (row-orthonormal, i.e. `V^T`)
///
/// # Errors
///
/// - [`TnError::EmptyInput`] if `m == 0` or `n == 0`.
/// - [`TnError::ShapeMismatch`] if `a.len() != m * n`.
/// - [`TnError::InvalidParameter`] if `m < n` (tall-and-thin required).
/// - [`TnError::NotConverged`] if the implicit QR loop does not converge.
pub fn svd_householder(a: &[f64], m: usize, n: usize) -> TnResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_inputs(a, m, n)?;

    let mut work = a.to_vec();
    // Accumulate full square U^T (m×m): U_full starts as I_m, U_full ← H_k * U_full.
    // After all steps U_full = H_{n-1}…H_0 = U^T  (full m×m orthogonal matrix).
    let mut u_full = identity_sq(m);
    // V (n×n): starts as I_n, accumulated via right reflectors G_k.
    let mut v = identity_sq(n);

    // Phase 1: Bidiagonalize A; accumulate U_full and V.
    bidiagonalize(&mut work, m, n, &mut u_full, &mut v);

    // Extract bidiagonal d[0..n] and superdiagonal e[0..n] (e[n-1] unused).
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        d[i] = work[i * n + i];
    }
    for i in 0..n.saturating_sub(1) {
        e[i] = work[i * n + (i + 1)];
    }

    // Build thin U (m×n) = first n columns of U (m×m) = transpose of U^T restricted.
    // U = (U_full)^T = transpose(U_full).  Column j of U = row j of U_full.
    // U_thin[i, j] = U[i, j] = U_full[j, i]  (using U_full as m×m row-major).
    let mut u = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            u[i * n + j] = u_full[j * m + i];
        }
    }

    // Phase 2: Golub–Kahan implicit QR on the bidiagonal.
    // U (m×n, thin): Givens left rotations on rows i and i+1 for i in 0..n-1.
    // V (n×n): Givens right rotations on columns i and i+1 for i in 0..n-1.
    implicit_qr_bidiag(&mut d, &mut e, n, &mut u, m, n, &mut v, n)?;

    // Phase 3: Sort descending; fix signs.
    sort_and_fix_signs(&mut d, &mut u, m, n, &mut v, n);

    let vt = transpose(&v, n, n);
    Ok((u, d, vt))
}

/// Economy (truncated) SVD: keep only the top-`k` singular values/vectors.
///
/// Returns `(U_k, s_k, Vt_k)` where `U_k` is `m×k`, `s_k` is length `k`, `Vt_k` is `k×n`.
///
/// # Errors
///
/// Same as [`svd_householder`], plus [`TnError::InvalidParameter`] if `k == 0` or `k > n`.
pub fn svd_householder_truncated(
    a: &[f64],
    m: usize,
    n: usize,
    k: usize,
) -> TnResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if k == 0 {
        return Err(TnError::InvalidParameter {
            name: "k".into(),
            reason: "truncation rank must be ≥ 1".into(),
        });
    }
    if k > n {
        return Err(TnError::InvalidParameter {
            name: "k".into(),
            reason: format!("truncation rank {k} exceeds n={n}"),
        });
    }
    let (u_full, s_full, vt_full) = svd_householder(a, m, n)?;
    // Slice first k columns of U (m×n → m×k).
    let mut u_k = vec![0.0f64; m * k];
    for i in 0..m {
        u_k[i * k..i * k + k].copy_from_slice(&u_full[i * n..i * n + k]);
    }
    let s_k = s_full[..k].to_vec();
    // First k rows of Vt (n×n → k×n).
    let vt_k = vt_full[..k * n].to_vec();
    Ok((u_k, s_k, vt_k))
}

// ─── Phase 1: Householder bidiagonalization ───────────────────────────────────

/// Bidiagonalize `work` (m×n) in place.
///
/// `u_full` (m×m): accumulated via `u_full ← H_k * u_full` (left mult).
/// After all steps: `u_full = U^T` where U is the left orthogonal factor.
///
/// `v` (n×n): accumulated via `v ← v * G_k` (right mult from right reflectors).
fn bidiagonalize(work: &mut [f64], m: usize, n: usize, u_full: &mut [f64], v: &mut [f64]) {
    for k in 0..n {
        // ── Left Householder H_k: zero work[k+1..m, k] ──────────────────────
        let col_len = m - k;
        let mut x = vec![0.0f64; col_len];
        for i in 0..col_len {
            x[i] = work[(k + i) * n + k];
        }
        let (beta, hv) = householder_vector(&x);
        if beta > 0.0 {
            apply_hh_left(work, m, n, k, k, &hv, beta); // work ← H_k * work
            apply_hh_left(u_full, m, m, k, 0, &hv, beta); // u_full ← H_k * u_full
        }

        // ── Right Householder G_k: zero work[k, k+2..n] ─────────────────────
        if k + 2 <= n {
            let row_len = n - (k + 1);
            let mut y = vec![0.0f64; row_len];
            for j in 0..row_len {
                y[j] = work[k * n + (k + 1 + j)];
            }
            let (beta, hv) = householder_vector(&y);
            if beta > 0.0 {
                apply_hh_right(work, m, n, k, k + 1, &hv, beta); // work ← work * G_k
                apply_hh_right(v, n, n, 0, k + 1, &hv, beta); // v ← v * G_k
            }
        }
    }
}

/// Householder vector for `H x = ‖x‖ e_1`. Returns `(β, v)` with `v[0]=1`.
/// `β = 0` means no reflection is needed.
fn householder_vector(x: &[f64]) -> (f64, Vec<f64>) {
    let len = x.len();
    if len == 0 {
        return (0.0, vec![]);
    }
    let sigma: f64 = x[1..].iter().map(|&xi| xi * xi).sum();
    let mut v = x.to_vec();
    if sigma < EPS * EPS * (x[0] * x[0]).max(1.0) {
        v[0] = 1.0;
        return (0.0, v);
    }
    let norm_x = (x[0] * x[0] + sigma).sqrt();
    let shift = if x[0] >= 0.0 { norm_x } else { -norm_x };
    v[0] = x[0] + shift;
    let v0 = v[0];
    let beta = 2.0 * v0 * v0 / (v0 * v0 + sigma);
    for vi in v.iter_mut() {
        *vi /= v0;
    }
    (beta, v)
}

/// Apply `H = I − β·v·vᵀ` from the left to `mat[row_start..m, col_start..nc]`.
fn apply_hh_left(
    mat: &mut [f64],
    m: usize,
    nc: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    let rows = m - row_start;
    let cols = nc - col_start;
    for j in 0..cols {
        let jj = col_start + j;
        let dot: f64 = (0..rows)
            .map(|i| v[i] * mat[(row_start + i) * nc + jj])
            .sum();
        let coeff = beta * dot;
        for i in 0..rows {
            mat[(row_start + i) * nc + jj] -= coeff * v[i];
        }
    }
}

/// Apply `H = I − β·v·vᵀ` from the right to `mat[row_start..m, col_start..nc]`.
fn apply_hh_right(
    mat: &mut [f64],
    m: usize,
    nc: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    let rows = m - row_start;
    let cols = nc - col_start;
    for i in 0..rows {
        let ii = row_start + i;
        let dot: f64 = (0..cols)
            .map(|j| mat[ii * nc + (col_start + j)] * v[j])
            .sum();
        let coeff = beta * dot;
        for j in 0..cols {
            mat[ii * nc + (col_start + j)] -= coeff * v[j];
        }
    }
}

// ─── Phase 2: Golub–Kahan implicit QR ─────────────────────────────────────────

/// Drive Golub–Kahan implicit-QR until `d` is diagonal (all `e[i] → 0`).
///
/// `d[0..n]` = main diagonal, `e[0..n]` = superdiagonal (`e[n-1]` unused/0).
/// Rotations from the left go into `u` (`m_u × n_sv`), from the right into `v` (`n_v × n_v`).
fn implicit_qr_bidiag(
    d: &mut [f64],
    e: &mut [f64],
    n: usize,
    u: &mut [f64],
    m_u: usize,
    n_sv: usize,
    v: &mut [f64],
    n_v: usize,
) -> TnResult<()> {
    if n <= 1 {
        if n == 1 && d[0] < 0.0 {
            d[0] = -d[0];
            for r in 0..m_u {
                u[r * n_sv] = -u[r * n_sv];
            }
        }
        return Ok(());
    }

    let max_iter = MAX_ITER_PER_SV * n;
    let mut iters = 0usize;
    let mut hi = n; // active block is [lo, hi)

    while hi > 1 {
        // Deflate: zero any tiny superdiagonals.
        for i in 0..hi.saturating_sub(1) {
            if e[i].abs() <= EPS * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }
        // Shrink hi over already-diagonal trailing part.
        while hi > 1 && e[hi - 2] == 0.0 {
            hi -= 1;
        }
        if hi <= 1 {
            break;
        }
        // Find lo: start of the unreduced block ending at hi.
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }

        // One Francis QR step on d[lo..hi], e[lo..hi-2].
        francis_step(d, e, lo, hi, u, m_u, n_sv, v, n_v);

        iters += 1;
        if iters > max_iter {
            return Err(TnError::NotConverged { iter: max_iter });
        }
    }

    // Make singular values non-negative.
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for r in 0..m_u {
                u[r * n_sv + i] = -u[r * n_sv + i];
            }
        }
    }
    Ok(())
}

/// One Golub–Kahan QR step on the unreduced block `d[lo..hi]`, `e[lo..hi-2]`.
///
/// Clean implementation of the DBDSQR bulge-chasing, derived from the manual trace:
///
/// For step i in lo..hi-1:
///   1. G_R(i, i+1): right Givens that zeros `g` using `f = e[i-1]` (or initial f for i=lo).
///      Updates: d[i], new_ei (temp), bulge at (i+1,i), d[i+1].
///      Also updates e[i-1] = c_r*e[i-1] + s_r*g_old  (for i > lo).
///   2. G_L(i, i+1): left Givens that zeros bulge using d[i].
///      Updates: d[i], e[i], d[i+1], e[i+1] (partial).
///      Sets f = e[i], g = s_l * e[i+1]_original (fill for next step).
fn francis_step(
    d: &mut [f64],
    e: &mut [f64],
    lo: usize,
    hi: usize,
    u: &mut [f64],
    m_u: usize,
    n_sv: usize,
    v: &mut [f64],
    n_v: usize,
) {
    debug_assert!(hi > lo + 1);

    let shift = wilkinson_shift(d, e, lo, hi);

    // Initial (f, g): the first column of (B^T B - shift I) restricted to rows 0 and 1:
    //   f = d[lo]^2 - shift,   g = d[lo] * e[lo].
    let mut f = d[lo] * d[lo] - shift;
    let mut g = d[lo] * e[lo];

    for i in lo..hi - 1 {
        // ── Right Givens G_R(i, i+1): zero `g` using `f` ──────────────────
        let (c_r, s_r) = givens_cs(f, g);

        // For i > lo: row (i-1) of the bidiagonal has [e[i-1], fill_prev] in cols (i, i+1)
        // where fill_prev = g_old (the value of g at the start of this iteration).
        // After G_R(i, i+1):  new e[i-1] = c_r * e[i-1] + s_r * g_old  = sqrt(f^2+g^2).
        // But since (c_r,s_r) = Givens(f,g) and f was e[i-1] from previous step and g was fill,
        // the new e[i-1] = hypot(f, g) = r.   We just compute it as f.hypot(g) here.
        // (The fill at (i-1, i+1) is zeroed by the Givens construction.)
        if i > lo {
            // e[i-1] = sqrt(f^2 + g^2) ... but f and g were already modified by the Givens.
            // Actually: f = e[i-1] (set at the end of previous iteration),
            //           g = s_l_prev * e[i]_prev  (fill from previous G_L).
            // New e[i-1] = c_r * e[i-1] + s_r * fill = c_r * f + s_r * g = sqrt(f^2+g^2) (by Givens def).
            // But c_r*f + s_r*g = r = hypot(f,g) by definition of Givens rotation.
            e[i - 1] = f.hypot(g);
        }

        // Apply G_R to the bidiagonal rows i and i+1 (and update d[i+1]):
        // Snapshot current values before modification.
        let d_i = d[i];
        let e_i = e[i];
        let d_i1 = if i + 1 < hi { d[i + 1] } else { 0.0 };
        let e_i1 = if i + 1 < hi - 1 { e[i + 1] } else { 0.0 };

        // After G_R (mixes cols i and i+1 from the right):
        //   Row i:   new d[i] = c_r*d_i + s_r*e_i
        //            temp e_i = -s_r*d_i + c_r*e_i  (B[i, i+1] after G_R; updated by G_L below)
        //   Row i+1: bulge at (i+1,i) = s_r * d_i1
        //            new d[i+1] = c_r * d_i1  (B[i+1,i+1] after G_R; updated by G_L below)
        let new_di = c_r * d_i + s_r * e_i;
        let temp_ei = -s_r * d_i + c_r * e_i;
        let bulge = s_r * d_i1;
        let new_di1 = c_r * d_i1;

        d[i] = new_di;
        if i + 1 < hi {
            d[i + 1] = new_di1;
        }

        // Accumulate G_R into V (right singular vectors).
        givens_rotate_cols(v, n_v, i, i + 1, c_r, s_r);

        // ── Left Givens G_L(i, i+1): zero the bulge at (i+1, i) ───────────
        let (c_l, s_l) = givens_cs(new_di, bulge);

        // After G_L (mixes rows i and i+1 from the left):
        //   Col i:    d[i] = c_l*new_di + s_l*bulge  (= hypot(new_di, bulge))
        //             (position (i+1,i) = 0 — bulge zeroed)
        //   Col i+1:  e[i] = c_l*temp_ei + s_l*new_di1
        //             d[i+1] = -s_l*temp_ei + c_l*new_di1
        //   Col i+2 (if exists): B[i, i+2] = s_l*e_i1  (fill; tracked as `g` for next step)
        //                         e[i+1]  = c_l*e_i1  (B[i+1, i+2] after G_L)
        d[i] = c_l * new_di + s_l * bulge;
        e[i] = c_l * temp_ei + s_l * new_di1;
        if i + 1 < hi {
            d[i + 1] = -s_l * temp_ei + c_l * new_di1;
        }
        if i + 1 < hi - 1 {
            e[i + 1] = c_l * e_i1;
        }

        // Accumulate G_L into U: U_thin ← U_thin * G_L^T (right col-mixing on cols i, i+1).
        // G_L = [[c_l, s_l], [-s_l, c_l]] acts on rows i,i+1 of the bidiagonal.
        // U_new = U * G_L^T so we apply the transposed rotation to cols i, i+1 of U.
        // With G_L^T = [[c_l, -s_l],[s_l, c_l]], the column update is:
        //   new col_i   = c_l * col_i + s_l * col_{i+1}   (same (c,s) as G_L)
        //   new col_{i+1} = -s_l * col_i + c_l * col_{i+1}
        // which matches givens_rotate_cols_rect with (c_l, s_l).
        givens_rotate_cols_rect(u, m_u, n_sv, i, i + 1, c_l, s_l);

        // Set up (f, g) for the next right rotation at step i+1:
        //   f = e[i]          (the updated superdiagonal; serves as "pivot" for G_R(i+1, i+2))
        //   g = s_l * e_i1    (fill at (i, i+2) from G_L; needs to be zeroed by G_R(i+1, i+2))
        f = e[i];
        g = s_l * e_i1;
    }
}

/// Wilkinson shift: eigenvalue of trailing 2×2 of `B^T B` closest to `B^T B[hi-1,hi-1]`.
fn wilkinson_shift(d: &[f64], e: &[f64], lo: usize, hi: usize) -> f64 {
    let dn = d[hi - 1];
    let dm = d[hi - 2];
    let em = e[hi - 2];
    let e_prev = if hi - lo >= 3 { e[hi - 3] } else { 0.0 };

    // Trailing 2×2 of B^T B:
    //   [[a, b], [b, c]]  where a = dm²+e_prev², b = dm*em, c = dn²+em²
    let a = dm * dm + e_prev * e_prev;
    let b = dm * em;
    let c = dn * dn + em * em;

    // Eigenvalues via numerically stable formula (avoid tr²−4det cancellation).
    let tr2 = (a + c) * 0.5;
    // disc = sqrt( ((a-c)/2)² + b² )
    let half_diff = (a - c) * 0.5;
    let disc = (half_diff * half_diff + b * b).sqrt();

    let lam1 = tr2 + disc;
    let lam2 = tr2 - disc;

    // Choose eigenvalue closest to c = B^T B[hi-1, hi-1].
    if (lam1 - c).abs() <= (lam2 - c).abs() {
        lam1.max(0.0)
    } else {
        lam2.max(0.0)
    }
}

// ─── Givens helpers ───────────────────────────────────────────────────────────

/// Compute `(c, s)` with `c*a + s*b = r`, `-s*a + c*b = 0`, `r ≥ 0`.
#[inline]
fn givens_cs(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        return (1.0, 0.0);
    }
    if a == 0.0 {
        return (0.0, b.signum());
    }
    let r = a.hypot(b);
    (a / r, b / r)
}

/// Givens rotation on **columns** `p, q` of an `m × n` rectangular row-major matrix.
/// `[col_p, col_q] ← [c·col_p + s·col_q, −s·col_p + c·col_q]`.
/// This applies `mat ← mat * G^T` where `G = [[c, s], [-s, c]]` at indices (p, q).
fn givens_rotate_cols_rect(
    mat: &mut [f64],
    m: usize,
    n: usize,
    p: usize,
    q: usize,
    c: f64,
    s: f64,
) {
    debug_assert!(p < n && q < n);
    for i in 0..m {
        let mp = mat[i * n + p];
        let mq = mat[i * n + q];
        mat[i * n + p] = c * mp + s * mq;
        mat[i * n + q] = -s * mp + c * mq;
    }
}

/// Givens rotation on **columns** `p, q` of an `nv × nv` square matrix (row-major).
/// `[col_p, col_q] ← [c·col_p + s·col_q, −s·col_p + c·col_q]`.
fn givens_rotate_cols(mat: &mut [f64], nv: usize, p: usize, q: usize, c: f64, s: f64) {
    debug_assert!(p < nv && q < nv);
    for i in 0..nv {
        let mp = mat[i * nv + p];
        let mq = mat[i * nv + q];
        mat[i * nv + p] = c * mp + s * mq;
        mat[i * nv + q] = -s * mp + c * mq;
    }
}

// ─── Phase 3: Sorting and sign normalization ──────────────────────────────────

/// Sort singular values descending with permutation of U columns and V columns.
/// Then normalize signs so the element with largest |·| in each U column is positive.
fn sort_and_fix_signs(
    d: &mut [f64],
    u: &mut [f64],
    m_u: usize,
    n_sv: usize,
    v: &mut [f64],
    n_v: usize,
) {
    let k = d.len();
    // Selection sort — O(k²), fine for typical TN sizes.
    for i in 0..k {
        let mut max_idx = i;
        for j in (i + 1)..k {
            if d[j] > d[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            d.swap(i, max_idx);
            for r in 0..m_u {
                u.swap(r * n_sv + i, r * n_sv + max_idx);
            }
            for r in 0..n_v {
                v.swap(r * n_v + i, r * n_v + max_idx);
            }
        }
    }
    // Sign: flip U[:,col] and V[:,col] so the max-|·| element of U[:,col] is positive.
    for col in 0..k {
        let mut max_abs = 0.0f64;
        let mut sign = 1.0f64;
        for row in 0..m_u {
            let val = u[row * n_sv + col];
            if val.abs() > max_abs {
                max_abs = val.abs();
                sign = val.signum();
            }
        }
        if sign < 0.0 {
            for row in 0..m_u {
                u[row * n_sv + col] = -u[row * n_sv + col];
            }
            for row in 0..n_v {
                v[row * n_v + col] = -v[row * n_v + col];
            }
        }
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

fn validate_inputs(a: &[f64], m: usize, n: usize) -> TnResult<()> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    if a.len() != m * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m < n {
        return Err(TnError::InvalidParameter {
            name: "m".into(),
            reason: format!("m={m} < n={n}; svd_householder requires m ≥ n"),
        });
    }
    Ok(())
}

/// `n × n` identity.
fn identity_sq(n: usize) -> Vec<f64> {
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Transpose `r × c` row-major → `c × r` row-major.
fn transpose(mat: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = mat[i * c + j];
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(u: &[f64], s: &[f64], vt: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let acc: f64 = (0..k).map(|c| u[i * k + c] * s[c] * vt[c * n + j]).sum();
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn check_col_ortho(mat: &[f64], m: usize, k: usize, tol: f64) {
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..m).map(|r| mat[r * k + i] * mat[r * k + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < tol,
                    "U^T U[{i},{j}] = {dot:.3e}, expected {expected}"
                );
            }
        }
    }

    fn check_row_ortho(mat: &[f64], k: usize, n: usize, tol: f64) {
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..n).map(|c| mat[i * n + c] * mat[j * n + c]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < tol,
                    "Vt Vt^T[{i},{j}] = {dot:.3e}, expected {expected}"
                );
            }
        }
    }

    fn check_desc(s: &[f64]) {
        for i in 1..s.len() {
            assert!(
                s[i - 1] + 1e-12 >= s[i],
                "s[{}]={:.6e} < s[{}]={:.6e}",
                i - 1,
                s[i - 1],
                i,
                s[i]
            );
        }
    }

    /// 1. 3×3 identity: all singular values = 1.
    #[test]
    fn test_identity_3x3() {
        let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (u, s, vt) = svd_householder(&mat, 3, 3).expect("3×3 identity");
        for &sv in &s {
            assert!((sv - 1.0).abs() < 1e-10, "sv={sv:.4e} ≠ 1");
        }
        let rec = reconstruct(&u, &s, &vt, 3, 3, 3);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    /// 2. Rank-1 matrix (outer product): one non-zero singular value.
    #[test]
    fn test_rank1() {
        let a = [4.0_f64, 5.0, 6.0, 7.0];
        let b = [1.0_f64, 2.0, 3.0];
        let m = 4;
        let n = 3;
        let mat: Vec<f64> = (0..m * n).map(|idx| a[idx / n] * b[idx % n]).collect();
        let (u, s, vt) = svd_householder(&mat, m, n).expect("rank-1");
        assert!(s[0] > 1e-6);
        for &sv in &s[1..] {
            assert!(sv.abs() < 1e-8, "extra sv={sv:.2e}");
        }
        let rec = reconstruct(&u, &s, &vt, m, n, n);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    /// 3. Random 8×5: U orthogonality, Vt orthogonality, reconstruction < 1e-10.
    #[test]
    fn test_random_8x5() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(42);
        let m = 8;
        let n = 5;
        let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let (u, s, vt) = svd_householder(&mat, m, n).expect("8×5");
        check_col_ortho(&u, m, n, 1e-10);
        check_row_ortho(&vt, n, n, 1e-10);
        check_desc(&s);
        let rec = reconstruct(&u, &s, &vt, m, n, n);
        assert!(
            fro_diff(&rec, &mat) < 1e-10,
            "err={:.2e}",
            fro_diff(&rec, &mat)
        );
    }

    /// 4. Random 5×5 square: orthogonality and reconstruction.
    #[test]
    fn test_random_5x5() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(99);
        let n = 5;
        let mat: Vec<f64> = (0..n * n).map(|_| rng.next_normal()).collect();
        let (u, s, vt) = svd_householder(&mat, n, n).expect("5×5");
        check_col_ortho(&u, n, n, 1e-10);
        check_row_ortho(&vt, n, n, 1e-10);
        check_desc(&s);
        let rec = reconstruct(&u, &s, &vt, n, n, n);
        assert!(
            fro_diff(&rec, &mat) < 1e-10,
            "err={:.2e}",
            fro_diff(&rec, &mat)
        );
    }

    /// 5. Jacobi comparison on 6×4: singular values agree within 1e-9.
    #[test]
    fn test_compare_jacobi() {
        use crate::handle::LcgRng;
        use crate::svd::svd_dense::svd_jacobi;
        let mut rng = LcgRng::new(7);
        let m = 6;
        let n = 4;
        let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let (_, s_h, _) = svd_householder(&mat, m, n).expect("Householder");
        let jac = svd_jacobi(&mat, m, n).expect("Jacobi");
        for (i, (&sh, &sj)) in s_h.iter().zip(&jac.s).enumerate() {
            assert!((sh - sj).abs() < 1e-9, "sv[{i}]: H={sh:.6e}, J={sj:.6e}");
        }
    }

    /// 6. m < n → InvalidParameter error.
    #[test]
    fn test_m_lt_n_error() {
        let result = svd_householder(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!(matches!(result, Err(TnError::InvalidParameter { .. })));
    }

    /// 7a. 1×1 matrix.
    #[test]
    fn test_1x1() {
        let (u, s, vt) = svd_householder(&[3.0], 1, 1).expect("1×1");
        assert!((s[0] - 3.0).abs() < 1e-12);
        let rec = reconstruct(&u, &s, &vt, 1, 1, 1);
        assert!(fro_diff(&rec, &[3.0]) < 1e-12);
    }

    /// 7b. 4×1 column vector: sv = Euclidean norm.
    #[test]
    fn test_4x1_vector() {
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let norm = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        let (u, s, vt) = svd_householder(&mat, 4, 1).expect("4×1");
        assert!(
            (s[0] - norm).abs() < 1e-10,
            "sv={:.6e} expected {norm:.6e}",
            s[0]
        );
        let rec = reconstruct(&u, &s, &vt, 4, 1, 1);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    /// 8. Truncated SVD: output shapes [m×k], [k], [k×n].
    #[test]
    fn test_truncated_shapes() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(17);
        let m = 10;
        let n = 6;
        let k = 3;
        let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let (u_k, s_k, vt_k) = svd_householder_truncated(&mat, m, n, k).expect("truncated");
        assert_eq!(u_k.len(), m * k);
        assert_eq!(s_k.len(), k);
        assert_eq!(vt_k.len(), k * n);
        check_desc(&s_k);
    }

    /// 9. All-zero matrix: all singular values ≈ 0.
    #[test]
    fn test_all_zero() {
        let mat = vec![0.0f64; 4 * 3];
        let (_, s, _) = svd_householder(&mat, 4, 3).expect("zero matrix");
        for &sv in &s {
            assert!(sv.abs() < 1e-14, "sv={sv:.2e}");
        }
    }

    /// 10. Diagonal matrix: singular values = sorted |diag entries|.
    #[test]
    fn test_diagonal_4x4() {
        let diag = [5.0f64, 1.0, 3.0, 2.0];
        let n = 4;
        let mut mat = vec![0.0f64; n * n];
        for i in 0..n {
            mat[i * n + i] = diag[i];
        }
        let (u, s, vt) = svd_householder(&mat, n, n).expect("diagonal");
        let mut expected = diag.to_vec();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (&sv, &ex)) in s.iter().zip(&expected).enumerate() {
            assert!((sv - ex).abs() < 1e-9, "s[{i}]={sv:.6e} expected {ex:.6e}");
        }
        let rec = reconstruct(&u, &s, &vt, n, n, n);
        assert!(fro_diff(&rec, &mat) < 1e-9);
    }

    /// 11. Large 32×16: reconstruction error < 1e-8.
    #[test]
    fn test_large_32x16() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(12345);
        let m = 32;
        let n = 16;
        let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let (u, s, vt) = svd_householder(&mat, m, n).expect("32×16");
        check_desc(&s);
        let rec = reconstruct(&u, &s, &vt, m, n, n);
        let err = fro_diff(&rec, &mat);
        assert!(err < 1e-8, "recon error={err:.2e}");
    }

    /// 12. Sign convention: max-|·| element of each U column is positive.
    #[test]
    fn test_sign_convention() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(2024);
        let m = 6;
        let n = 4;
        let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
        let (u, s, _) = svd_householder(&mat, m, n).expect("sign test");
        for col in 0..n {
            if s[col] < 1e-10 {
                continue;
            }
            let mut max_abs = 0.0f64;
            let mut max_val = 0.0f64;
            for row in 0..m {
                let v = u[row * n + col];
                if v.abs() > max_abs {
                    max_abs = v.abs();
                    max_val = v;
                }
            }
            assert!(
                max_val >= 0.0,
                "U col {col}: max-abs is negative ({max_val:.4e})"
            );
        }
    }

    /// 13. Truncated k=0 → error.
    #[test]
    fn test_trunc_k0_error() {
        assert!(svd_householder_truncated(&[1.0f64; 12], 4, 3, 0).is_err());
    }

    /// 14. Truncated k > n → error.
    #[test]
    fn test_trunc_k_gt_n_error() {
        assert!(svd_householder_truncated(&[1.0f64; 12], 4, 3, 10).is_err());
    }

    /// 15. Empty input → EmptyInput error.
    #[test]
    fn test_empty_error() {
        assert!(matches!(
            svd_householder(&[], 0, 3),
            Err(TnError::EmptyInput)
        ));
    }
}
