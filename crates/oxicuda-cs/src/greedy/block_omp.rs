//! Block Orthogonal Matching Pursuit (Block-OMP) for block-sparse signal recovery.
//!
//! Eldar, Kuppinger, Bölcskei (2010) "Block-Sparse Signals: Uncertainty Relations and
//! Efficient Recovery." IEEE Trans. Signal Process. 58(6):3042–3054.
//!
//! **Problem:** Given Φ (m×p) and y (m,), find x (p,) that is block-K-sparse under a
//! uniform block partition of {0,...,p−1} into L blocks of equal size d, such that
//! `y ≈ Φ * x` and at most K blocks of x are non-zero.

use crate::error::{CsError, CsResult};
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{norm2, submat_columns};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of block-sparse recovery (Block-OMP).
#[derive(Debug, Clone)]
pub struct BlockOmpResult {
    /// Recovered signal (p-dimensional), row-major.
    pub x: Vec<f64>,
    /// Block indices selected (subset of {0,...,n_blocks-1}).
    pub block_support: Vec<usize>,
    /// Element indices selected (union of selected blocks).
    pub support: Vec<usize>,
    /// 2-norm of the final residual.
    pub residual_norm: f64,
    /// Number of Block-OMP iterations performed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Block correlation score for block `b` with given `block_size`.
///
/// score = ||Φ_b^T r||_2  where Φ_b = Φ[:, b*d : (b+1)*d].
#[inline]
fn block_score(phi: &[f64], m: usize, p: usize, r: &[f64], b: usize, block_size: usize) -> f64 {
    let b_start = b * block_size;
    let sq_sum: f64 = (b_start..(b_start + block_size))
        .map(|col| {
            let corr: f64 = (0..m).map(|i| phi[i * p + col] * r[i]).sum();
            corr * corr
        })
        .sum();
    sq_sum.sqrt()
}

/// Build the element-level support from the block-level support.
///
/// For each block b in `block_support`, its elements are `b*d, b*d+1, ..., b*d+d-1`.
fn element_support_from_blocks(block_support: &[usize], block_size: usize) -> Vec<usize> {
    let mut elems: Vec<usize> = block_support
        .iter()
        .flat_map(|&b| (b * block_size)..(b * block_size + block_size))
        .collect();
    elems.sort_unstable();
    elems
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-OMP algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Block Orthogonal Matching Pursuit for block-sparse signal recovery.
///
/// Recovers a block-K-sparse signal x from `y = Φ * x` under a uniform block
/// partition of size `block_size` across `n_blocks` blocks.
///
/// # Arguments
///
/// - `phi` – sensing matrix, row-major `(m × p)` f64.
/// - `m` – rows of Φ (measurements).
/// - `p` – columns of Φ (signal dimension, must equal `n_blocks * block_size`).
/// - `y` – observation vector `(m,)` f64.
/// - `block_size` – number of elements per block `d`.
/// - `n_blocks` – number of blocks `L` (must satisfy `L * d == p`).
/// - `k` – number of blocks to select (block sparsity).
/// - `tol_residual` – stop early when `||r||_2 < tol_residual`.
///
/// # Errors
///
/// Returns `CsError` on invalid inputs or numerical failures.
pub fn block_omp(
    phi: &[f64],
    m: usize,
    p: usize,
    y: &[f64],
    block_size: usize,
    n_blocks: usize,
    k: usize,
    tol_residual: f64,
) -> CsResult<BlockOmpResult> {
    // ── Validation ────────────────────────────────────────────────────────
    if phi.len() != m * p {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, p],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if block_size == 0 {
        return Err(CsError::InvalidParameter("block_size must be > 0".into()));
    }
    if block_size * n_blocks != p {
        return Err(CsError::InvalidParameter(
            "block_size * n_blocks != p".into(),
        ));
    }
    if k == 0 || k > n_blocks {
        return Err(CsError::InvalidSparsity(k));
    }

    // ── Initialise ────────────────────────────────────────────────────────
    let mut block_support: Vec<usize> = Vec::with_capacity(k);
    let mut residual = y.to_vec();
    let mut x_full = vec![0.0_f64; p];
    let mut iter = 0usize;

    // ── Main loop ─────────────────────────────────────────────────────────
    for _ in 0..k {
        // Early stop if residual norm is small enough.
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }

        // 1. Compute block correlation scores for blocks not yet selected.
        let mut best_block = usize::MAX;
        let mut best_score = -1.0_f64;
        for b in 0..n_blocks {
            if block_support.contains(&b) {
                continue;
            }
            let score = block_score(phi, m, p, &residual, b, block_size);
            if score > best_score {
                best_score = score;
                best_block = b;
            }
        }
        if best_block == usize::MAX {
            return Err(CsError::RecoveryFailed(
                "Block-OMP: no non-selected block found".into(),
            ));
        }

        // 2. Add best block to support.
        block_support.push(best_block);
        block_support.sort_unstable();

        // 3. Build element-level support and solve LS on selected columns.
        let elem_support = element_support_from_blocks(&block_support, block_size);
        let x_omega = solve_subset_ls(phi, m, p, &elem_support, y)?;

        // 4. Update x_full: zero all positions, then fill active elements.
        x_full.fill(0.0);
        for (s_idx, &col) in elem_support.iter().enumerate() {
            x_full[col] = x_omega[s_idx];
        }

        // 5. Update residual: r = y - Φ_Ω * x_Ω.
        //    Build Φ_Ω = Φ[:, elem_support] (m × |elem_support|) then compute Φ_Ω * x_Ω.
        let phi_omega = submat_columns(phi, m, p, &elem_support)?;
        let omega_len = elem_support.len();
        for i in 0..m {
            let sum: f64 = (0..omega_len)
                .map(|s_idx| phi_omega[i * omega_len + s_idx] * x_omega[s_idx])
                .sum();
            residual[i] = y[i] - sum;
        }

        iter += 1;
    }

    let elem_support = element_support_from_blocks(&block_support, block_size);

    Ok(BlockOmpResult {
        x: x_full,
        block_support,
        support: elem_support,
        residual_norm: norm2(&residual),
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

    // ── Deterministic test RNG ─────────────────────────────────────────────

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

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }

        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn random_matrix(m: usize, n: usize, rng: &mut TestRng) -> Vec<f64> {
        (0..m * n).map(|_| rng.next_normal()).collect()
    }

    // ── Error-path tests ──────────────────────────────────────────────────

    #[test]
    fn block_omp_error_zero_k() {
        let phi = vec![1.0_f64; 6]; // m=2, p=3
        let y = vec![1.0_f64; 2];
        let err = block_omp(&phi, 2, 3, &y, 1, 3, 0, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidSparsity(0)));
    }

    #[test]
    fn block_omp_error_shape_phi() {
        let phi = vec![1.0_f64; 5]; // should be 2*3=6
        let y = vec![1.0_f64; 2];
        let err = block_omp(&phi, 2, 3, &y, 1, 3, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::ShapeMismatch { .. }));
    }

    #[test]
    fn block_omp_error_shape_y() {
        let phi = vec![1.0_f64; 6];
        let y = vec![1.0_f64; 3]; // should be m=2
        let err = block_omp(&phi, 2, 3, &y, 1, 3, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::DimensionMismatch { .. }));
    }

    #[test]
    fn block_omp_error_block_size_zero() {
        let phi = vec![1.0_f64; 6];
        let y = vec![1.0_f64; 2];
        let err = block_omp(&phi, 2, 3, &y, 0, 3, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidParameter(_)));
    }

    #[test]
    fn block_omp_error_block_mismatch() {
        // n_blocks * block_size = 4*2 = 8 ≠ p=6
        let phi = vec![1.0_f64; 12]; // m=2, p=6
        let y = vec![1.0_f64; 2];
        let err = block_omp(&phi, 2, 6, &y, 2, 4, 1, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidParameter(_)));
    }

    #[test]
    fn block_omp_k_exceeds_n_blocks() {
        // k > n_blocks → InvalidSparsity
        let phi = vec![1.0_f64; 6]; // m=2, p=3, n_blocks=3, block_size=1
        let y = vec![1.0_f64; 2];
        let err = block_omp(&phi, 2, 3, &y, 1, 3, 4, 1e-9).unwrap_err();
        assert!(matches!(err, CsError::InvalidSparsity(4)));
    }

    // ── Correctness tests ─────────────────────────────────────────────────

    #[test]
    fn block_omp_exact_recovery() {
        // n_blocks=5, block_size=4, m=30, p=20, k=2.
        // Generate X block-sparse with known 2 blocks (0 and 3), y=Φx, verify recovery.
        let mut rng = TestRng::new(42);
        let m = 30;
        let n_blocks = 5;
        let block_size = 4;
        let p = n_blocks * block_size; // 20
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);

        // x is block-sparse: only blocks 0 and 3 are non-zero.
        let true_blocks = [0usize, 3usize];
        let mut x_true = vec![0.0_f64; p];
        for &b in &true_blocks {
            for d in 0..block_size {
                x_true[b * block_size + d] = rng.next_normal();
            }
        }

        // y = Φ * x_true
        let y: Vec<f64> = (0..m)
            .map(|i| (0..p).map(|j| phi[i * p + j] * x_true[j]).sum())
            .collect();

        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");

        assert_eq!(res.block_support.len(), k);
        assert!(
            res.block_support.contains(&0),
            "block 0 not recovered; got {:?}",
            res.block_support
        );
        assert!(
            res.block_support.contains(&3),
            "block 3 not recovered; got {:?}",
            res.block_support
        );
    }

    #[test]
    fn block_omp_result_x_shape() {
        let mut rng = TestRng::new(10);
        let m = 20;
        let n_blocks = 4;
        let block_size = 3;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 2, 1e-9).expect("ok");
        assert_eq!(res.x.len(), p);
    }

    #[test]
    fn block_omp_result_block_support_size() {
        let mut rng = TestRng::new(20);
        let m = 18;
        let n_blocks = 6;
        let block_size = 2;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let k = 3;
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");
        assert!(res.block_support.len() <= k);
    }

    #[test]
    fn block_omp_result_support_covers_blocks() {
        // Each element in `support` must belong to a selected block.
        let mut rng = TestRng::new(30);
        let m = 20;
        let n_blocks = 5;
        let block_size = 3;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 2, 1e-9).expect("ok");

        for &elem in &res.support {
            let block = elem / block_size;
            assert!(
                res.block_support.contains(&block),
                "element {elem} in support but its block {block} not in block_support"
            );
        }
    }

    #[test]
    fn block_omp_residual_norm_finite() {
        let mut rng = TestRng::new(40);
        let m = 15;
        let n_blocks = 4;
        let block_size = 2;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 2, 1e-9).expect("ok");
        assert!(res.residual_norm.is_finite());
        assert!(res.residual_norm >= 0.0);
    }

    #[test]
    fn block_omp_iterations_bounded() {
        let mut rng = TestRng::new(50);
        let m = 20;
        let n_blocks = 5;
        let block_size = 3;
        let p = n_blocks * block_size;
        let k = 3;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");
        assert!(res.iterations <= k);
    }

    #[test]
    fn block_omp_support_no_duplicate_blocks() {
        let mut rng = TestRng::new(60);
        let m = 24;
        let n_blocks = 6;
        let block_size = 4;
        let p = n_blocks * block_size;
        let k = 4;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");
        let mut seen = std::collections::HashSet::new();
        for &b in &res.block_support {
            assert!(seen.insert(b), "duplicate block index {b}");
        }
    }

    #[test]
    fn block_omp_block_size_one_matches_omp() {
        // block_size=1 → Block-OMP reduces to OMP (block = individual column).
        let mut rng = TestRng::new(70);
        let m = 10;
        let p = 12;
        let n_blocks = p;
        let block_size = 1;
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();

        let res_block = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");
        let res_omp = omp(&phi, m, p, &y, k, 1e-9).expect("ok");

        // Both should select the same atoms.
        assert_eq!(
            res_block.block_support, res_omp.support,
            "Block-OMP(d=1) {:?} ≠ OMP {:?}",
            res_block.block_support, res_omp.support
        );
    }

    #[test]
    fn block_omp_single_block_k1() {
        // k=1: selects block with highest correlation score.
        // Φ is arranged so block 2 (columns 4..6) perfectly explains y.
        let m = 6;
        let block_size = 2;
        let n_blocks = 3;
        let p = n_blocks * block_size; // 6

        // Build Φ: identity 6×6 (block 0=cols 0..1, block 1=cols 2..3, block 2=cols 4..5).
        let mut phi = vec![0.0_f64; m * p];
        for i in 0..m {
            phi[i * p + i] = 1.0;
        }

        // y is concentrated in block 2 (rows 4, 5).
        let y = vec![0.0, 0.0, 0.0, 0.0, 3.0, 5.0];

        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 1, 1e-9).expect("ok");
        assert_eq!(res.block_support, vec![2], "should select block 2");
    }

    #[test]
    fn block_omp_residual_decreases() {
        // Verify residual norm is non-increasing with larger k.
        let mut rng = TestRng::new(80);
        let m = 20;
        let n_blocks = 6;
        let block_size = 3;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();

        let mut prev_norm = f64::INFINITY;
        for k in 1..=4 {
            let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-15).expect("ok");
            assert!(
                res.residual_norm <= prev_norm + 1e-10,
                "k={k}: residual increased from {prev_norm} to {}",
                res.residual_norm
            );
            prev_norm = res.residual_norm;
        }
    }

    #[test]
    fn block_omp_x_zero_outside_support() {
        let mut rng = TestRng::new(90);
        let m = 20;
        let n_blocks = 5;
        let block_size = 4;
        let p = n_blocks * block_size;
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");

        for col in 0..p {
            if !res.support.contains(&col) {
                assert_eq!(res.x[col], 0.0, "x[{col}] should be zero (outside support)");
            }
        }
    }

    #[test]
    fn block_omp_tol_early_stop() {
        // tol_residual = 1e10 → stop immediately (0 iterations).
        let mut rng = TestRng::new(100);
        let m = 15;
        let n_blocks = 4;
        let block_size = 3;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 2, 1e10).expect("ok");
        assert!(res.iterations <= 1);
    }

    #[test]
    fn block_omp_large_problem() {
        // m=50, p=40, block_size=4, n_blocks=10, k=3 — no crash.
        let mut rng = TestRng::new(110);
        let m = 50;
        let n_blocks = 10;
        let block_size = 4;
        let p = n_blocks * block_size;
        let k = 3;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("large ok");
        assert!(res.block_support.len() <= k);
    }

    #[test]
    fn block_omp_x_finite() {
        let mut rng = TestRng::new(120);
        let m = 20;
        let n_blocks = 5;
        let block_size = 3;
        let p = n_blocks * block_size;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, 2, 1e-9).expect("ok");
        for (idx, &v) in res.x.iter().enumerate() {
            assert!(v.is_finite(), "x[{idx}]={v} is not finite");
        }
    }

    #[test]
    fn block_omp_reconstruction_quality() {
        // y = Φ * x_true (noiseless), verify Φ * x_recovered ≈ y.
        let mut rng = TestRng::new(130);
        let m = 30;
        let n_blocks = 6;
        let block_size = 4;
        let p = n_blocks * block_size;
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);

        // x_true: block-sparse on blocks 1 and 4.
        let true_blocks = [1usize, 4usize];
        let mut x_true = vec![0.0_f64; p];
        for &b in &true_blocks {
            for d in 0..block_size {
                x_true[b * block_size + d] = rng.next_normal();
            }
        }

        let y: Vec<f64> = (0..m)
            .map(|i| (0..p).map(|j| phi[i * p + j] * x_true[j]).sum())
            .collect();

        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");

        // Φ * x_recovered.
        let y_hat: Vec<f64> = (0..m)
            .map(|i| (0..p).map(|j| phi[i * p + j] * res.x[j]).sum())
            .collect();

        let err_sq: f64 = y
            .iter()
            .zip(y_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
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
    fn block_omp_block_support_sorted() {
        // block_support should be returned in sorted order.
        let mut rng = TestRng::new(140);
        let m = 20;
        let n_blocks = 6;
        let block_size = 3;
        let p = n_blocks * block_size;
        let k = 3;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");
        let mut sorted = res.block_support.clone();
        sorted.sort_unstable();
        assert_eq!(res.block_support, sorted, "block_support not sorted");
    }

    #[test]
    fn block_omp_element_support_matches_blocks() {
        // Verify that `support` is exactly the union of columns from `block_support`.
        let mut rng = TestRng::new(150);
        let m = 20;
        let n_blocks = 5;
        let block_size = 4;
        let p = n_blocks * block_size;
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");

        let mut expected_elems: Vec<usize> = res
            .block_support
            .iter()
            .flat_map(|&b| (b * block_size)..(b * block_size + block_size))
            .collect();
        expected_elems.sort_unstable();
        assert_eq!(res.support, expected_elems);
    }

    #[test]
    fn block_omp_residual_norm_vs_manual() {
        // Verify residual_norm == ||y - Φ*x||_2 manually.
        let mut rng = TestRng::new(160);
        let m = 18;
        let n_blocks = 4;
        let block_size = 3;
        let p = n_blocks * block_size;
        let k = 2;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 1e-9).expect("ok");

        let residual_sq: f64 = (0..m)
            .map(|i| {
                let phi_xi: f64 = (0..p).map(|j| phi[i * p + j] * res.x[j]).sum();
                (y[i] - phi_xi).powi(2)
            })
            .sum();
        let manual = residual_sq.sqrt();
        assert!(
            (manual - res.residual_norm).abs() < 1e-9,
            "residual_norm={} manual={manual}",
            res.residual_norm
        );
    }

    #[test]
    fn block_omp_zero_tol_runs_all_k() {
        // With tol=0.0 and generic data, runs exactly k iterations.
        let mut rng = TestRng::new(170);
        let m = 20;
        let n_blocks = 6;
        let block_size = 3;
        let p = n_blocks * block_size;
        let k = 4;
        let phi = random_matrix(m, p, &mut rng);
        let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
        let res = block_omp(&phi, m, p, &y, block_size, n_blocks, k, 0.0).expect("ok");
        assert_eq!(res.iterations, k);
    }
}
