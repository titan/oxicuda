//! Simultaneous Orthogonal Matching Pursuit (SOMP / M-OMP) for Multiple Measurement Vectors.
//!
//! Tropp, Gilbert, Strauss (2006) "Algorithms for simultaneous sparse approximation.
//! Part I: Greedy pursuit." IEEE Trans. Signal Process. 54(10):3984–3994.
//!
//! **Problem (MMV):** Given D (m×n) and Y (m×L), find jointly K-sparse X (n×L)
//! such that `Y ≈ D * X`.  All L signals share the same support (non-zero rows).

use crate::error::{CsError, CsResult};
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_vec, submat_columns};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of simultaneous sparse recovery (MMV setting).
#[derive(Debug, Clone)]
pub struct MmvResult {
    /// Coefficient matrix (n × L), row-major: `x[j*l + ll]` = X[j, ll].
    pub x: Vec<f64>,
    /// Number of atoms (columns of D).
    pub n: usize,
    /// Number of measurement vectors.
    pub l: usize,
    /// Shared support indices (rows of X that are non-zero).
    pub support: Vec<usize>,
    /// Frobenius norm of the final residual matrix R = Y − D X.
    pub residual_norm: f64,
    /// Number of SOMP iterations performed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute C = D^T R where D is m×n (row-major) and R is m×L (row-major).
/// Returns C as n×L row-major: `c[j*l + ll] = sum_i D[i,j] * R[i,ll]`.
fn mat_t_mat(d: &[f64], m: usize, n: usize, r: &[f64], l: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * l];
    for i in 0..m {
        let d_row = i * n;
        let r_row = i * l;
        for j in 0..n {
            let dij = d[d_row + j];
            for ll in 0..l {
                c[j * l + ll] += dij * r[r_row + ll];
            }
        }
    }
    c
}

/// Row 2-norm of row `j` of matrix C (n×L, row-major).
#[inline]
fn row_norm2(c: &[f64], l: usize, j: usize) -> f64 {
    let row_start = j * l;
    let sq_sum: f64 = (0..l).map(|ll| c[row_start + ll].powi(2)).sum();
    sq_sum.sqrt()
}

/// Max column 2-norm of matrix R (m×L, row-major).
fn max_col_norm(r: &[f64], m: usize, l: usize) -> f64 {
    let mut max_val = 0.0_f64;
    for ll in 0..l {
        let col_norm_sq: f64 = (0..m).map(|i| r[i * l + ll].powi(2)).sum();
        let col_norm = col_norm_sq.sqrt();
        if col_norm > max_val {
            max_val = col_norm;
        }
    }
    max_val
}

/// Extract column `ll` of matrix Y (m×L, row-major) → Vec<f64> of length m.
fn extract_col(y: &[f64], m: usize, l: usize, ll: usize) -> Vec<f64> {
    (0..m).map(|i| y[i * l + ll]).collect()
}

/// Frobenius norm of m×L row-major matrix R.
fn frob_norm(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// SOMP algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Simultaneous Orthogonal Matching Pursuit (SOMP) for multiple measurement vectors.
///
/// Recovers a jointly K-sparse matrix X from `Y = D * X` where all L columns of X
/// share the same non-zero row set (support).
///
/// # Arguments
///
/// - `d` – sensing / dictionary matrix, row-major `(m × n)` f64.
/// - `m` – rows of D (measurement dimension).
/// - `n` – columns of D (dictionary size / signal dimension).
/// - `y` – observation matrix, row-major `(m × L)` f64.
/// - `l` – number of measurement vectors (L).
/// - `k` – target sparsity (number of non-zero rows in X).
/// - `tol_residual` – stop early when max column 2-norm of residual < this.
///
/// # Errors
///
/// Returns `CsError` on invalid inputs or numerical failures.
pub fn somp(
    d: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    l: usize,
    k: usize,
    tol_residual: f64,
) -> CsResult<MmvResult> {
    // ── Validation ────────────────────────────────────────────────────────
    if d.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![d.len()],
        });
    }
    if y.len() != m * l {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, l],
            got: vec![y.len()],
        });
    }
    if l == 0 {
        return Err(CsError::InvalidParameter("l must be > 0".into()));
    }
    if k == 0 || k > m.min(n) {
        return Err(CsError::InvalidSparsity(k));
    }

    // ── Initialise ────────────────────────────────────────────────────────
    let mut support: Vec<usize> = Vec::with_capacity(k);
    // Residual matrix R, row-major (m × L), starts as Y.
    let mut residual = y.to_vec();
    // Final coefficient matrix X (n × L), row-major, initialised to zero.
    let mut x_full = vec![0.0_f64; n * l];
    let mut iter = 0usize;

    // ── Main loop ─────────────────────────────────────────────────────────
    for _ in 0..k {
        // Early stop check (max column norm of R).
        if max_col_norm(&residual, m, l) < tol_residual {
            break;
        }

        // 1. C = D^T R  (n × L, row-major).
        let c = mat_t_mat(d, m, n, &residual, l);

        // 2. Select atom j* = argmax_{j ∉ Ω} ||C[j,:]||_2.
        let mut best_idx = usize::MAX;
        let mut best_val = -1.0_f64;
        for j in 0..n {
            if support.contains(&j) {
                continue;
            }
            let rn = row_norm2(&c, l, j);
            if rn > best_val {
                best_val = rn;
                best_idx = j;
            }
        }
        if best_idx == usize::MAX {
            return Err(CsError::RecoveryFailed(
                "SOMP: no non-included atom found".into(),
            ));
        }
        support.push(best_idx);
        support.sort_unstable();

        // 3. Solve LS jointly for each signal column.
        //    D_Ω = D[:,support]; solve (D_Ω^T D_Ω) x_Ω_ll = D_Ω^T y_ll for each ll.
        let mut x_omega_cols: Vec<Vec<f64>> = Vec::with_capacity(l);
        for ll in 0..l {
            let y_col = extract_col(y, m, l, ll);
            let x_omega_col = solve_subset_ls(d, m, n, &support, &y_col)?;
            x_omega_cols.push(x_omega_col);
        }

        // Write x_omega_cols into x_full at rows=support, cols=ll.
        x_full.fill(0.0);
        for ll in 0..l {
            for (s_idx, &j) in support.iter().enumerate() {
                x_full[j * l + ll] = x_omega_cols[ll][s_idx];
            }
        }

        // 4. Update residual R = Y − D_Ω * X_Ω column-by-column.
        let d_omega = submat_columns(d, m, n, &support)?;
        let omega_len = support.len();

        for ll in 0..l {
            let ax = mat_vec(&d_omega, m, omega_len, &x_omega_cols[ll])?;
            for i in 0..m {
                residual[i * l + ll] = y[i * l + ll] - ax[i];
            }
        }

        iter += 1;
    }

    Ok(MmvResult {
        x: x_full,
        n,
        l,
        support,
        residual_norm: frob_norm(&residual),
        iterations: iter,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greedy::omp;

    // ── Simple deterministic pseudo-random helper ─────────────────────────

    /// LCG (MMIX params) for deterministic test data.
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        /// Uniform [0, 1).
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }

        /// Standard normal (Box-Muller).
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Build a random Gaussian m×n matrix (row-major).
    fn random_matrix(m: usize, n: usize, rng: &mut TestRng) -> Vec<f64> {
        (0..m * n).map(|_| rng.next_normal()).collect()
    }

    // ── Error-path tests ──────────────────────────────────────────────────

    #[test]
    fn somp_error_empty_sparsity() {
        let d = vec![1.0_f64; 4];
        let y = vec![1.0_f64; 2];
        let err = somp(&d, 2, 2, &y, 1, 0, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidSparsity(0)));
    }

    #[test]
    fn somp_error_shape_mismatch_d() {
        let d = vec![1.0_f64; 5]; // should be 6 = 2*3
        let y = vec![1.0_f64; 4];
        let err = somp(&d, 2, 3, &y, 2, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::ShapeMismatch { .. }));
    }

    #[test]
    fn somp_error_shape_mismatch_y() {
        let d = vec![1.0_f64; 6]; // 2×3
        let y = vec![1.0_f64; 3]; // should be 2*2=4 for l=2
        let err = somp(&d, 2, 3, &y, 2, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::ShapeMismatch { .. }));
    }

    #[test]
    fn somp_error_l_zero() {
        let d = vec![1.0_f64; 4];
        let y: Vec<f64> = Vec::new();
        let err = somp(&d, 2, 2, &y, 0, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidParameter(_)));
    }

    #[test]
    fn somp_error_k_exceeds_min_m_n() {
        // m=2, n=3, k=3 > min(2,3)=2 → InvalidSparsity
        let d = vec![1.0_f64; 6];
        let y = vec![1.0_f64; 4];
        let err = somp(&d, 2, 3, &y, 2, 3, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidSparsity(3)));
    }

    // ── Correctness tests ─────────────────────────────────────────────────

    #[test]
    fn somp_l1_reduces_to_omp() {
        // SOMP with L=1 should select the same atom as OMP on the same data.
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];
        // y = [0, 1, 0, 0] — atom 1 wins.
        let y = vec![0.0, 1.0, 0.0, 0.0];
        let r_somp = somp(&phi, 4, 4, &y, 1, 1, 1e-9).expect("somp ok");
        let r_omp = omp(&phi, 4, 4, &y, 1, 1e-9).expect("omp ok");
        assert_eq!(r_somp.support, r_omp.support);
    }

    #[test]
    fn somp_exact_recovery_noiseless() {
        // Use D = I_8 (identity, perfectly incoherent), L=3 signals,
        // true support = {1, 5} with large coefficients → guaranteed SOMP recovery.
        let m = 8;
        let n = 8;
        let l = 3;
        let k = 2;

        // D = I_8 (identity): perfectly incoherent dictionary.
        let mut d = vec![0.0_f64; m * n];
        for i in 0..m {
            d[i * n + i] = 1.0;
        }

        // True support {1, 5}: X (8×3) with large entries on rows 1, 5.
        let true_support = [1usize, 5usize];
        let x_true_values: [[f64; 3]; 2] = [[10.0, -7.0, 4.0], [-8.0, 6.0, -3.0]];
        let mut x_true = vec![0.0_f64; n * l];
        for (s, &row) in true_support.iter().enumerate() {
            for ll in 0..l {
                x_true[row * l + ll] = x_true_values[s][ll];
            }
        }

        // Y = D * X = X (since D = I).
        let mut y = vec![0.0_f64; m * l];
        for i in 0..m {
            for ll in 0..l {
                y[i * l + ll] = x_true[i * l + ll];
            }
        }

        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("somp ok");
        assert_eq!(res.support.len(), k);
        assert!(
            res.support.contains(&1),
            "support {:?} should contain 1",
            res.support
        );
        assert!(
            res.support.contains(&5),
            "support {:?} should contain 5",
            res.support
        );
    }

    #[test]
    fn somp_support_size() {
        let mut rng = TestRng::new(123);
        let m = 12;
        let n = 20;
        let l = 4;
        let k = 3;
        let d = random_matrix(m, n, &mut rng);
        let y_flat: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y_flat, l, k, 1e-9).expect("ok");
        assert!(res.support.len() <= k);
    }

    #[test]
    fn somp_residual_norm_finite() {
        let mut rng = TestRng::new(7);
        let m = 8;
        let n = 10;
        let l = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, 2, 1e-9).expect("ok");
        assert!(res.residual_norm.is_finite());
        assert!(res.residual_norm >= 0.0);
    }

    #[test]
    fn somp_result_shapes() {
        let mut rng = TestRng::new(99);
        let m = 10;
        let n = 15;
        let l = 5;
        let k = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        assert_eq!(res.x.len(), n * l);
        assert_eq!(res.n, n);
        assert_eq!(res.l, l);
        for &idx in &res.support {
            assert!(idx < n);
        }
    }

    #[test]
    fn somp_iterations_bounded() {
        let mut rng = TestRng::new(11);
        let m = 10;
        let n = 12;
        let l = 3;
        let k = 3;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        assert!(res.iterations <= k);
    }

    #[test]
    fn somp_early_stop() {
        // Extremely high tolerance → stops immediately (0 iterations).
        let mut rng = TestRng::new(55);
        let m = 10;
        let n = 12;
        let l = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, 2, 1e10).expect("ok");
        assert!(res.iterations <= 1);
    }

    #[test]
    fn somp_support_no_duplicates() {
        let mut rng = TestRng::new(66);
        let m = 12;
        let n = 18;
        let l = 4;
        let k = 4;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        let mut seen = std::collections::HashSet::new();
        for &idx in &res.support {
            assert!(seen.insert(idx), "duplicate support index {idx}");
        }
    }

    #[test]
    fn somp_x_nonzero_on_support() {
        let mut rng = TestRng::new(77);
        let m = 10;
        let n = 14;
        let l = 3;
        let k = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        // Rows outside support must be all-zero.
        for j in 0..n {
            if !res.support.contains(&j) {
                for ll in 0..l {
                    assert_eq!(
                        res.x[j * l + ll],
                        0.0,
                        "X[{j},{ll}] should be zero (outside support)"
                    );
                }
            }
        }
    }

    #[test]
    fn somp_k_equals_one() {
        // k=1: max-correlation atom selection with two signals.
        let phi = vec![
            1.0, 0.0, 0.0, // row 0
            0.0, 2.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        // Y is m×2 row-major: y[i*2 + ll].
        // y[:,0] = [0, 2, 0], y[:,1] = [0, 1, 0] → both concentrated on atom 1.
        let y = vec![
            0.0, 0.0, // i=0
            2.0, 1.0, // i=1
            0.0, 0.0, // i=2
        ];
        let res = somp(&phi, 3, 3, &y, 2, 1, 1e-9).expect("ok");
        assert_eq!(res.support, vec![1], "should select atom 1");
    }

    #[test]
    fn somp_unit_atom_recovery() {
        // D = I_5 (orthonormal cols). Y = D[:,3] * ones(1,L) → support={3}.
        let m = 5;
        let n = 5;
        let l = 3;
        let mut d = vec![0.0_f64; m * n];
        for i in 0..m {
            d[i * n + i] = 1.0;
        }
        let mut y = vec![0.0_f64; m * l];
        for ll in 0..l {
            y[3 * l + ll] = 1.0;
        }
        let res = somp(&d, m, n, &y, l, 1, 1e-9).expect("ok");
        assert_eq!(res.support, vec![3]);
    }

    #[test]
    fn somp_large_l() {
        // Stress: L=10, m=20, n=30, k=3 — no crash.
        let mut rng = TestRng::new(88);
        let m = 20;
        let n = 30;
        let l = 10;
        let k = 3;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("large L ok");
        assert!(res.support.len() <= k);
    }

    #[test]
    fn somp_residual_nondecreasing_decrease() {
        // Verify residual Frobenius norm is non-increasing with larger k.
        let mut rng = TestRng::new(101);
        let m = 12;
        let n = 16;
        let l = 3;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();

        let mut prev_norm = f64::INFINITY;
        for k in 1..=4 {
            let res = somp(&d, m, n, &y, l, k, 1e-15).expect("ok");
            assert!(
                res.residual_norm <= prev_norm + 1e-10,
                "k={k}: residual increased from {prev_norm} to {}",
                res.residual_norm
            );
            prev_norm = res.residual_norm;
        }
    }

    #[test]
    fn somp_shared_support_used() {
        // SOMP must return a single shared support; X must be zero outside it.
        let mut rng = TestRng::new(222);
        let m = 10;
        let n = 15;
        let l = 4;
        let k = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");

        for j in 0..n {
            if !res.support.contains(&j) {
                for ll in 0..l {
                    let val = res.x[j * l + ll];
                    assert_eq!(val, 0.0, "X[{j},{ll}]={val} outside support");
                }
            }
        }
    }

    #[test]
    fn somp_mmv_result_dimensions() {
        let mut rng = TestRng::new(333);
        let m = 8;
        let n = 12;
        let l = 6;
        let k = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        assert_eq!(res.n, n);
        assert_eq!(res.l, l);
        assert_eq!(res.x.len(), n * l);
    }

    #[test]
    fn somp_k_equals_n() {
        // k = n = m = 5: recover full signal (identity D).
        let m = 5;
        let n = 5;
        let l = 2;
        let mut d = vec![0.0_f64; m * n];
        for i in 0..m {
            d[i * n + i] = 1.0;
        }
        let y: Vec<f64> = vec![
            1.0, 2.0, // i=0
            3.0, 4.0, // i=1
            5.0, 6.0, // i=2
            7.0, 8.0, // i=3
            9.0, 10.0, // i=4
        ];
        let res = somp(&d, m, n, &y, l, n, 1e-9).expect("k=n ok");
        assert_eq!(res.support.len(), n);
        assert!(res.residual_norm < 1e-9);
    }

    #[test]
    fn somp_reconstruction_quality() {
        // Y = D * X_true (noiseless), verify D * X_recovered ≈ Y.
        // Use D = I_10 (identity) so SOMP is guaranteed to recover the exact support.
        let m = 10;
        let n = 10;
        let l = 3;
        let k = 2;

        // D = I_10.
        let mut d = vec![0.0_f64; m * n];
        for i in 0..m {
            d[i * n + i] = 1.0;
        }

        // True support {2, 7} with known coefficients.
        let true_support = [2usize, 7usize];
        let x_vals: [[f64; 3]; 2] = [[5.0, -3.0, 2.0], [-4.0, 7.0, -1.0]];
        let mut x_true = vec![0.0_f64; n * l];
        for (s, &row) in true_support.iter().enumerate() {
            for ll in 0..l {
                x_true[row * l + ll] = x_vals[s][ll];
            }
        }

        // Y = D * X = X (identity D).
        let mut y = vec![0.0_f64; m * l];
        for i in 0..m {
            for ll in 0..l {
                y[i * l + ll] = x_true[i * l + ll];
            }
        }

        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");

        // Compute D * X_recovered and compare to Y.
        let err_sq: f64 = (0..m)
            .flat_map(|i| (0..l).map(move |ll| (i, ll)))
            .map(|(i, ll)| {
                let dxi: f64 = res
                    .support
                    .iter()
                    .map(|&j| d[i * n + j] * res.x[j * l + ll])
                    .sum();
                (y[i * l + ll] - dxi).powi(2)
            })
            .sum();
        let y_norm_sq: f64 = y.iter().map(|v| v * v).sum();
        let rel_err = if y_norm_sq > 0.0 {
            err_sq.sqrt() / y_norm_sq.sqrt()
        } else {
            err_sq.sqrt()
        };
        assert!(
            rel_err < 1e-6,
            "relative reconstruction error={rel_err} too large"
        );
    }

    #[test]
    fn somp_x_all_finite() {
        let mut rng = TestRng::new(555);
        let m = 10;
        let n = 15;
        let l = 4;
        let k = 3;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        for (idx, &v) in res.x.iter().enumerate() {
            assert!(v.is_finite(), "x[{idx}]={v} is not finite");
        }
    }

    #[test]
    fn somp_zero_tolerance_runs_all_k() {
        // With tol=0.0, early stop only if residual is exactly zero.
        let mut rng = TestRng::new(666);
        let m = 12;
        let n = 18;
        let l = 3;
        let k = 4;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 0.0).expect("ok");
        assert_eq!(res.iterations, k, "should run all {k} iterations");
    }

    #[test]
    fn somp_frob_residual_matches_manual() {
        // Verify residual_norm equals ||Y - D*X||_F manually.
        let mut rng = TestRng::new(777);
        let m = 8;
        let n = 12;
        let l = 3;
        let k = 2;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");

        let frob_sq: f64 = (0..m)
            .flat_map(|i| (0..l).map(move |ll| (i, ll)))
            .map(|(i, ll)| {
                let dxi: f64 = res
                    .support
                    .iter()
                    .map(|&j| d[i * n + j] * res.x[j * l + ll])
                    .sum();
                (y[i * l + ll] - dxi).powi(2)
            })
            .sum();
        let manual_norm = frob_sq.sqrt();
        assert!(
            (manual_norm - res.residual_norm).abs() < 1e-9,
            "residual_norm={} but manual frob={manual_norm}",
            res.residual_norm
        );
    }

    #[test]
    fn somp_single_signal_reconstruction() {
        // Single atom, single signal: trivial recovery.
        let m = 4;
        let n = 4;
        let l = 1;
        let d = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];
        // Y = [0, 0, 3, 0]^T as m×1 row-major.
        let y = vec![0.0, 0.0, 3.0, 0.0];
        let res = somp(&d, m, n, &y, l, 1, 1e-9).expect("ok");
        assert_eq!(res.support, vec![2]);
        assert!((res.x[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn somp_support_sorted() {
        // Support should be returned in sorted order.
        let mut rng = TestRng::new(888);
        let m = 10;
        let n = 15;
        let l = 3;
        let k = 3;
        let d = random_matrix(m, n, &mut rng);
        let y: Vec<f64> = (0..m * l).map(|_| rng.next_normal()).collect();
        let res = somp(&d, m, n, &y, l, k, 1e-9).expect("ok");
        let mut sorted = res.support.clone();
        sorted.sort_unstable();
        assert_eq!(res.support, sorted, "support not sorted");
    }

    #[test]
    fn somp_norm2_residual_near_zero_after_exact_recovery() {
        // For L=1, identity D, single-atom signal: residual must be near zero.
        let phi = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let y_vec = vec![2.0, 0.0, 0.0]; // atom 0, coefficient 2
        let res = somp(&phi, 3, 3, &y_vec, 1, 1, 1e-9).expect("ok");
        assert!(
            res.residual_norm < 1e-9,
            "residual_norm={}",
            res.residual_norm
        );
    }
}
