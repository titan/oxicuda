//! Unit and on-device integration tests for [`super`] (LU factorization).
//!
//! Kept in a separate file to stay under the 2000-line refactoring limit.

use super::*;

// ---------------------------------------------------------------------------
// Generated-PTX validation (real ptxas assembly for sm_86)
// ---------------------------------------------------------------------------

/// Assembles `ptx` with `ptxas -arch=sm_86` to a throwaway object, returning
/// `Ok(())` on success or the captured stderr on failure. When `ptxas` is not
/// on PATH the check is skipped (returns `Ok(())`).
fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
    use std::process::Command;
    let dir = std::env::temp_dir();
    let src = dir.join(format!("oxicuda_lu_{tag}.ptx"));
    std::fs::write(&src, ptx).map_err(|e| format!("write ptx: {e}"))?;
    let out = Command::new("ptxas")
        .arg("-arch=sm_86")
        .arg(&src)
        .arg("-o")
        .arg("/dev/null")
        .output();
    let _ = std::fs::remove_file(&src);
    match out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
        Err(e) => {
            // ptxas absent — skip gracefully.
            eprintln!("skipping ptxas validation ({tag}): {e}");
            Ok(())
        }
    }
}

#[test]
fn panel_lu_ptx_is_real_and_assembles_f64() {
    let ptx = emit_panel_lu::<f64>(SmVersion::Sm86, 64).expect("emit panel LU f64");
    // The historical stub body was a bare `ret`; guard against its return.
    assert!(ptx.contains("bar.sync"), "panel kernel must synchronize");
    assert!(
        ptx.contains("div.rn.f64"),
        "panel kernel must scale the pivot"
    );
    assert!(
        ptx.contains("sub.f64"),
        "panel kernel must apply the trailing update"
    );
    assert!(
        ptx.contains("red.global.min.u32"),
        "panel kernel must report singular columns"
    );
    ptxas_assembles(&ptx, "panel_f64").expect("panel LU f64 PTX must assemble");
}

#[test]
fn panel_lu_ptx_is_real_and_assembles_f32() {
    let ptx = emit_panel_lu::<f32>(SmVersion::Sm86, 32).expect("emit panel LU f32");
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("div.rn.f32"));
    assert!(ptx.contains("sub.f32"));
    ptxas_assembles(&ptx, "panel_f32").expect("panel LU f32 PTX must assemble");
}

#[test]
fn trsm_and_gemm_update_ptx_assemble() {
    for (ptx, tag) in [
        (
            emit_trsm_unit_lower::<f64>(SmVersion::Sm86).expect("trsm f64"),
            "trsm_f64",
        ),
        (
            emit_trsm_unit_lower::<f32>(SmVersion::Sm86).expect("trsm f32"),
            "trsm_f32",
        ),
        (
            emit_gemm_update::<f64>(SmVersion::Sm86).expect("gemm f64"),
            "gemm_f64",
        ),
        (
            emit_gemm_update::<f32>(SmVersion::Sm86).expect("gemm f32"),
            "gemm_f32",
        ),
    ] {
        assert!(
            ptx.contains("fma.rn") || ptx.contains("sub.") || ptx.contains("mul."),
            "{tag} kernel must compute"
        );
        ptxas_assembles(&ptx, tag).unwrap_or_else(|e| panic!("{tag} PTX must assemble: {e}"));
    }
}

#[test]
fn pivot_swap_ptx_is_real_and_assembles() {
    let ptx64 = emit_pivot_swap::<f64>(SmVersion::Sm86).expect("emit pivot swap f64");
    // The stub never touched memory; the real kernel loads and stores rows.
    assert!(
        ptx64.contains("ld.global.u32"),
        "swap must read pivot index"
    );
    assert!(ptx64.contains("ld.global.f64"), "swap must read row values");
    assert!(
        ptx64.contains("st.global.f64"),
        "swap must write row values"
    );
    ptxas_assembles(&ptx64, "swap_f64").expect("pivot swap f64 PTX must assemble");
    let ptx32 = emit_pivot_swap::<f32>(SmVersion::Sm86).expect("emit pivot swap f32");
    ptxas_assembles(&ptx32, "swap_f32").expect("pivot swap f32 PTX must assemble");
}

// ---------------------------------------------------------------------------
// On-device LU factorization + solve tests (real GPU; skip when no card)
// ---------------------------------------------------------------------------

/// Builds a solver handle bound to device 0, or `None` (with a printed skip
/// notice) when CUDA is unavailable so CPU-only hosts degrade gracefully.
fn try_solver_handle() -> Option<(std::sync::Arc<oxicuda_driver::Context>, SolverHandle)> {
    if oxicuda_driver::init().is_err() {
        eprintln!("skipping device test: CUDA driver unavailable");
        return None;
    }
    let has_device = oxicuda_driver::device::Device::count()
        .map(|c| c > 0)
        .unwrap_or(false);
    if !has_device {
        eprintln!("skipping device test: no NVIDIA CUDA device");
        return None;
    }
    let dev = oxicuda_driver::device::Device::get(0).expect("device 0 must be retrievable");
    let ctx = std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).expect("CUDA context must be creatable"),
    );
    let handle = SolverHandle::new(&ctx).expect("solver handle must be creatable");
    Some((ctx, handle))
}

/// `y = A · x` for a column-major matrix `a` (`a[col * n + row] == A[row, col]`).
fn matvec_colmajor(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for (j, &xj) in x.iter().enumerate().take(n) {
        for (i, yi) in y.iter_mut().enumerate().take(n) {
            *yi += a[j * n + i] * xj;
        }
    }
    y
}

/// Factor `a` (column-major) on the device, then reconstruct `P·A` from the
/// packed `L`/`U` and assert it equals the row-permuted input within `tol`.
fn assert_device_lu_reconstructs(handle: &mut SolverHandle, a: &[f64], n: usize, tol: f64) {
    let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a).expect("upload A");
    let mut d_piv = oxicuda_memory::DeviceBuffer::<i32>::zeroed(n).expect("alloc pivots");
    let res = lu_factorize::<f64>(handle, &mut d_a, n as u32, n as u32, &mut d_piv)
        .expect("device LU factorization");
    assert_eq!(res.info, 0, "matrix must be non-singular");

    let mut lu = vec![0.0_f64; n * n];
    d_a.copy_to_host(&mut lu).expect("download LU");
    let mut piv = vec![0_i32; n];
    d_piv.copy_to_host(&mut piv).expect("download pivots");

    // Reconstruct A_rec = L * U from packed storage (column-major, lda = n).
    let lu_at = |row: usize, col: usize| lu[col * n + row];
    let mut a_rec = vec![0.0_f64; n * n];
    for i in 0..n {
        for jc in 0..n {
            // (L*U)[i,jc] = sum_k L[i,k] * U[k,jc], L unit-lower, U upper.
            let mut acc = 0.0_f64;
            for k in 0..n {
                let l_ik = if k < i {
                    lu_at(i, k)
                } else if k == i {
                    1.0
                } else {
                    0.0
                };
                let u_kj = if k <= jc { lu_at(k, jc) } else { 0.0 };
                acc += l_ik * u_kj;
            }
            a_rec[jc * n + i] = acc;
        }
    }

    // Apply the same pivot transpositions (forward order) to a copy of A to
    // form P·A, which L·U must equal.
    let mut pa = a.to_vec();
    for (row, &p) in piv.iter().enumerate().take(n) {
        let p = p.max(0) as usize;
        if p != row {
            for col in 0..n {
                pa.swap(col * n + row, col * n + p);
            }
        }
    }

    for col in 0..n {
        for row in 0..n {
            let got = a_rec[col * n + row];
            let want = pa[col * n + row];
            assert!(
                (got - want).abs() < tol,
                "(L*U)[{row},{col}]={got} != (P*A)[{row},{col}]={want} (|diff|={})",
                (got - want).abs()
            );
        }
    }
}

/// Full factor + solve check: solves `A · X = B` on the device and asserts
/// `A · X ≈ B` column-by-column within 1e-9.
fn assert_device_lu_solve(handle: &mut SolverHandle, a: &[f64], b: &[f64], n: usize, nrhs: usize) {
    let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a).expect("upload A");
    let mut d_piv = oxicuda_memory::DeviceBuffer::<i32>::zeroed(n).expect("alloc pivots");
    lu_factorize::<f64>(handle, &mut d_a, n as u32, n as u32, &mut d_piv)
        .expect("device LU factorization");
    let mut d_b = oxicuda_memory::DeviceBuffer::from_host(b).expect("upload B");
    lu_solve::<f64>(handle, &d_a, &d_piv, &mut d_b, n as u32, nrhs as u32)
        .expect("device LU solve");
    let mut x = vec![0.0_f64; n * nrhs];
    d_b.copy_to_host(&mut x).expect("download solution");
    for col in 0..nrhs {
        let xc = &x[col * n..(col + 1) * n];
        let bc = &b[col * n..(col + 1) * n];
        let ax = matvec_colmajor(a, xc, n);
        for i in 0..n {
            assert!(
                (ax[i] - bc[i]).abs() < 1e-9,
                "rhs {col}: (A·x)[{i}]={} != b[{i}]={} (|diff|={})",
                ax[i],
                bc[i],
                (ax[i] - bc[i]).abs()
            );
        }
    }
}

#[test]
fn device_lu_factor_reconstructs_3x3() {
    let Some((_ctx, mut handle)) = try_solver_handle() else {
        return;
    };
    // Non-symmetric, well-conditioned 3×3 (column-major). A[row,col] =
    // a[col*3 + row]; here columns are (2,1,1), (1,3,2), (1,1,4).
    let n = 3;
    let a = vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 4.0];
    assert_device_lu_reconstructs(&mut handle, &a, n, 1e-12);
}

#[test]
fn device_lu_factor_reconstructs_with_pivoting() {
    let Some((_ctx, mut handle)) = try_solver_handle() else {
        return;
    };
    // First column has a zero leading entry, forcing a row swap (column-major).
    // A = [[0, 2, 1], [4, 1, 0], [1, 1, 3]] stored column-major.
    let n = 3;
    let a = vec![0.0, 4.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0];
    assert_device_lu_reconstructs(&mut handle, &a, n, 1e-12);
}

#[test]
fn device_lu_solve_4x4() {
    let Some((_ctx, mut handle)) = try_solver_handle() else {
        return;
    };
    let n = 4;
    // Diagonally dominant non-symmetric 4×4 (column-major).
    let a = vec![
        10.0, 2.0, 1.0, 1.0, // column 0
        1.0, 8.0, 2.0, 1.0, // column 1
        2.0, 1.0, 12.0, 3.0, // column 2
        1.0, 2.0, 1.0, 9.0, // column 3
    ];
    assert_device_lu_reconstructs(&mut handle, &a, n, 1e-11);
    let b = vec![1.0, 2.0, 3.0, 4.0];
    assert_device_lu_solve(&mut handle, &a, &b, n, 1);
}

#[test]
fn device_lu_solve_multi_rhs_and_pivoting() {
    let Some((_ctx, mut handle)) = try_solver_handle() else {
        return;
    };
    let n = 3;
    // Pivoting-inducing matrix (column-major) with two right-hand sides.
    let a = vec![0.0, 4.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0];
    let b = vec![5.0, -1.0, 2.0, 1.0, 1.0, 1.0];
    assert_device_lu_solve(&mut handle, &a, &b, n, 2);
}

#[test]
fn device_lu_larger_block_boundary() {
    let Some((_ctx, mut handle)) = try_solver_handle() else {
        return;
    };
    // n > LU_BLOCK_SIZE exercises the blocked path (panel + TRSM + GEMM + the
    // multi-panel pivot replay). Build a diagonally dominant matrix.
    let n = 80usize;
    let mut a = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let v = (((row * 31 + col * 17 + 7) % 13) as f64) - 6.0;
            a[col * n + row] = v;
        }
        // Make it diagonally dominant for a benign factorization.
        a[col * n + col] = 50.0 + col as f64;
    }
    assert_device_lu_reconstructs(&mut handle, &a, n, 1e-8);
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    assert_device_lu_solve(&mut handle, &a, &b, n, 1);
}

// ---------------------------------------------------------------------------
// CPU reference helpers for LU integration tests
// ---------------------------------------------------------------------------

/// Doolittle LU factorization (no pivoting) on a 4×4 f64 matrix.
///
/// Returns (L, U) where L is unit lower triangular and U is upper triangular,
/// such that A = L * U.
fn doolittle_lu_4x4(a: &[[f64; 4]; 4]) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut l = [[0.0_f64; 4]; 4];
    let mut u = [[0.0_f64; 4]; 4];

    for i in 0..4 {
        l[i][i] = 1.0; // Unit diagonal for L.

        // U row i.
        for j in i..4 {
            let sum: f64 = (0..i).map(|k| l[i][k] * u[k][j]).sum();
            u[i][j] = a[i][j] - sum;
        }

        // L column i (below diagonal).
        for j in (i + 1)..4 {
            let sum: f64 = (0..i).map(|k| l[j][k] * u[k][i]).sum();
            if u[i][i].abs() > 1e-15 {
                l[j][i] = (a[j][i] - sum) / u[i][i];
            }
        }
    }

    (l, u)
}

/// 4×4 matrix multiply (row-major).
fn matmul_4x4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// LU + GEMM/TRSM integration tests
// ---------------------------------------------------------------------------

#[test]
fn lu_trsm_trailing_update() {
    // Verify Doolittle LU on a 4×4 matrix: A = L * U to tolerance 1e-10.
    let a = [
        [4.0_f64, 3.0, 2.0, 1.0],
        [2.0, 5.0, 3.0, 2.0],
        [1.0, 2.0, 6.0, 3.0],
        [1.0, 1.0, 2.0, 7.0],
    ];
    let (l, u) = doolittle_lu_4x4(&a);

    // L must be unit lower triangular.
    for (i, l_row) in l.iter().enumerate() {
        assert!(
            (l_row[i] - 1.0).abs() < 1e-15,
            "L[{i},{i}] must be 1.0 (unit diagonal)"
        );
        for (j, &val) in l_row.iter().enumerate().filter(|(j, _)| *j > i) {
            assert!(
                val.abs() < 1e-15,
                "L[{i},{j}] = {val} must be 0.0 (upper triangle)",
            );
        }
    }

    // U must be upper triangular.
    for (i, u_row) in u.iter().enumerate() {
        for (j, &val) in u_row.iter().enumerate().filter(|(j, _)| *j < i) {
            assert!(
                val.abs() < 1e-15,
                "U[{i},{j}] = {val} must be 0.0 (lower triangle)",
            );
        }
    }

    // Reconstruct: L*U must equal A.
    let reconstructed = matmul_4x4(&l, &u);
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (reconstructed[i][j] - a[i][j]).abs() < 1e-10,
                "LU[{i},{j}] = {} ≠ A[{i},{j}] = {} (diff = {})",
                reconstructed[i][j],
                a[i][j],
                (reconstructed[i][j] - a[i][j]).abs()
            );
        }
    }
}

#[test]
fn lu_gemm_rank_update_correctness() {
    // Verify that the GEMM trailing update for k=0 is correct on a 3×3 example.
    //
    // After the first column of LU (k=0):
    //   L[:,0] is computed, U[0,:] is computed.
    //   Trailing update: A[1:3, 1:3] -= L[1:3, 0:1] * U[0:1, 1:3]
    //
    // Use a = [[2, 4, 6], [1, 3, 5], [1, 2, 4]] (simple example).
    let a = [[2.0_f64, 4.0, 6.0], [1.0, 3.0, 5.0], [1.0, 2.0, 4.0]];

    // After first pivot (k=0), L column 0 = [1, a[1,0]/a[0,0], a[2,0]/a[0,0]]
    //                                      = [1, 0.5, 0.5]
    // U row 0 = a[0,:] = [2, 4, 6]
    // Trailing update for A[1:3, 1:3]:
    //   A[1,1] -= L[1,0]*U[0,1] = 3 - 0.5*4 = 1
    //   A[1,2] -= L[1,0]*U[0,2] = 5 - 0.5*6 = 2
    //   A[2,1] -= L[2,0]*U[0,1] = 2 - 0.5*4 = 0
    //   A[2,2] -= L[2,0]*U[0,2] = 4 - 0.5*6 = 1
    let l_col0 = [1.0_f64, a[1][0] / a[0][0], a[2][0] / a[0][0]];
    let u_row0 = [a[0][0], a[0][1], a[0][2]];

    // Trailing submatrix after k=0 update.
    let mut trailing = [[0.0_f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            trailing[i][j] = a[i + 1][j + 1] - l_col0[i + 1] * u_row0[j + 1];
        }
    }

    assert!(
        (trailing[0][0] - 1.0).abs() < 1e-12,
        "trailing[0,0] should be 1"
    );
    assert!(
        (trailing[0][1] - 2.0).abs() < 1e-12,
        "trailing[0,1] should be 2"
    );
    assert!(trailing[1][0].abs() < 1e-12, "trailing[1,0] should be 0");
    assert!(
        (trailing[1][1] - 1.0).abs() < 1e-12,
        "trailing[1,1] should be 1"
    );
}

#[test]
fn lu_block_size_positive() {
    let block_size = LU_BLOCK_SIZE;
    assert!(block_size > 0);
    assert!(block_size <= 256);
}

#[test]
fn lu_result_info() {
    let result = LuResult { info: 0 };
    assert_eq!(result.info, 0);

    let singular = LuResult { info: 3 };
    assert!(singular.info > 0);
}

#[test]
fn panel_lu_name_format() {
    let name = panel_lu_name::<f32>(64);
    assert!(name.contains("f32"));
    assert!(name.contains("64"));
}

#[test]
fn pivot_swap_name_format() {
    let name = pivot_swap_name::<f64>();
    assert!(name.contains("f64"));
}

#[test]
fn neg_one_f32() {
    let neg = f32::from_bits_u64(f32::gpu_one().to_bits_u64() ^ 0x8000_0000);
    assert!((neg + 1.0).abs() < 1e-10);
}

#[test]
fn neg_one_f64() {
    let neg = f64::from_bits_u64(f64::gpu_one().to_bits_u64() ^ 0x8000_0000_0000_0000);
    assert!((neg + 1.0).abs() < 1e-15);
}

// -----------------------------------------------------------------------
// Transposed LU solve: host reference of `lu_solve_with_transpose`
// -----------------------------------------------------------------------
//
// The production `lu_solve_with_transpose` operates on `DeviceBuffer`s and
// a `SolverHandle`, which cannot be created without a CUDA device. The
// helpers below re-implement the exact same triangular-substitution stages
// on plain `Vec<f64>` data so the algorithm — the transposed forward/back
// substitution and the transposed permutation — can be validated.

/// Dense LU factorization with partial pivoting (column-major, lda = n).
///
/// Mirrors the storage produced by [`lu_factorize`]: on return `lu` holds
/// `L` (strict lower, implicit unit diagonal) and `U` (upper) packed in
/// place, and `pivots[i]` is the absolute row swapped with row `i`.
fn dense_lu_factorize(a: &[f64], n: usize) -> (Vec<f64>, Vec<i32>) {
    let mut lu = a.to_vec();
    let mut pivots = vec![0_i32; n];
    for col in 0..n {
        // Pivot search over rows col..n in column `col`.
        let mut pivot_row = col;
        let mut max_abs = 0.0_f64;
        for row in col..n {
            let abs = lu[col * n + row].abs();
            if abs > max_abs {
                max_abs = abs;
                pivot_row = row;
            }
        }
        pivots[col] = pivot_row as i32;
        if pivot_row != col {
            for c in 0..n {
                lu.swap(c * n + col, c * n + pivot_row);
            }
        }
        let diag = lu[col * n + col];
        for row in (col + 1)..n {
            lu[col * n + row] /= diag;
        }
        for c in (col + 1)..n {
            let u_kc = lu[c * n + col];
            for row in (col + 1)..n {
                lu[c * n + row] -= lu[col * n + row] * u_kc;
            }
        }
    }
    (lu, pivots)
}

/// Host port of `lu_solve_with_transpose` operating on `Vec<f64>` data.
///
/// Solves `A x = b` (`transpose = false`) or `Aᵀ x = b`
/// (`transpose = true`) given LU factors as produced by
/// [`dense_lu_factorize`]. `b` is overwritten with the solution.
fn dense_lu_solve(lu: &[f64], pivots: &[i32], b: &mut [f64], n: usize, transpose: bool) {
    let lu_at = |row: usize, col: usize| lu[col * n + row];
    if transpose {
        // Uᵀ y = b — forward substitution (lower triangular, non-unit).
        for k in 0..n {
            let acc: f64 = b[..k]
                .iter()
                .enumerate()
                .map(|(i, &b_i)| lu_at(i, k) * b_i)
                .sum();
            b[k] = (b[k] - acc) / lu_at(k, k);
        }
        // Lᵀ w = y — backward substitution (upper triangular, unit).
        for k in (0..n).rev() {
            let acc: f64 = b[(k + 1)..]
                .iter()
                .enumerate()
                .map(|(offset, &b_i)| lu_at(k + 1 + offset, k) * b_i)
                .sum();
            b[k] -= acc;
        }
        // x = Pᵀ w — pivot transpositions replayed in reverse order.
        for row in (0..n).rev() {
            let piv = pivots[row].max(0) as usize;
            if piv != row {
                b.swap(row, piv);
            }
        }
    } else {
        // P b — pivot transpositions in forward order.
        for (row, &piv_entry) in pivots.iter().enumerate().take(n) {
            let piv = piv_entry.max(0) as usize;
            if piv != row {
                b.swap(row, piv);
            }
        }
        // L y = P b — forward substitution (lower triangular, unit).
        for k in 0..n {
            let acc: f64 = b[..k]
                .iter()
                .enumerate()
                .map(|(i, &b_i)| lu_at(k, i) * b_i)
                .sum();
            b[k] -= acc;
        }
        // U x = y — backward substitution (upper triangular, non-unit).
        for k in (0..n).rev() {
            let acc: f64 = b[(k + 1)..]
                .iter()
                .enumerate()
                .map(|(offset, &b_i)| lu_at(k, k + 1 + offset) * b_i)
                .sum();
            b[k] = (b[k] - acc) / lu_at(k, k);
        }
    }
}

/// Dense matrix-vector product `y = M x` for column-major `M` (lda = n).
fn matvec(m: &[f64], x: &[f64], n: usize, transpose: bool) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for (row, y_row) in y.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (col, &x_col) in x.iter().enumerate() {
            let elem = if transpose {
                m[row * n + col]
            } else {
                m[col * n + row]
            };
            acc += elem * x_col;
        }
        *y_row = acc;
    }
    y
}

#[test]
fn lu_solve_transposed_matches_explicit_transpose() {
    // A non-symmetric 4×4 matrix (column-major storage).
    let n = 4;
    let a_rows = [
        [4.0_f64, 3.0, 2.0, 1.0],
        [2.0, 5.0, 3.0, 2.0],
        [1.0, 2.0, 6.0, 3.0],
        [7.0, 1.0, 2.0, 9.0],
    ];
    let mut a_col = vec![0.0_f64; n * n];
    let mut at_col = vec![0.0_f64; n * n];
    for (i, row) in a_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a_col[j * n + i] = v; // A[i,j]
            at_col[i * n + j] = v; // Aᵀ[j,i] = A[i,j]
        }
    }

    let b = vec![1.0_f64, -2.0, 3.0, 0.5];

    // Path 1: transposed solve on the LU factors of A.
    let (lu_a, piv_a) = dense_lu_factorize(&a_col, n);
    let mut x_transposed = b.clone();
    dense_lu_solve(&lu_a, &piv_a, &mut x_transposed, n, true);

    // Path 2: explicitly form Aᵀ, factor it, do a normal solve.
    let (lu_at, piv_at) = dense_lu_factorize(&at_col, n);
    let mut x_explicit = b.clone();
    dense_lu_solve(&lu_at, &piv_at, &mut x_explicit, n, false);

    for i in 0..n {
        assert!(
            (x_transposed[i] - x_explicit[i]).abs() < 1e-10,
            "transposed solve x[{i}] = {} disagrees with explicit Aᵀ solve {}",
            x_transposed[i],
            x_explicit[i],
        );
    }

    // Residual check: Aᵀ * x must reproduce b.
    let residual = matvec(&a_col, &x_transposed, n, true);
    for i in 0..n {
        assert!(
            (residual[i] - b[i]).abs() < 1e-10,
            "Aᵀ x residual[{i}] = {} ≠ b[{i}] = {}",
            residual[i],
            b[i],
        );
    }
}

#[test]
fn lu_solve_forward_and_transposed_consistent() {
    // For the same factors, A x = b and Aᵀ y = b must both be exact.
    let n = 3;
    let a_rows = [[2.0_f64, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
    let mut a_col = vec![0.0_f64; n * n];
    for (i, row) in a_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a_col[j * n + i] = v;
        }
    }
    let (lu, piv) = dense_lu_factorize(&a_col, n);

    let b = vec![1.0_f64, 2.0, 3.0];

    let mut x = b.clone();
    dense_lu_solve(&lu, &piv, &mut x, n, false);
    let ax = matvec(&a_col, &x, n, false);
    for i in 0..n {
        assert!(
            (ax[i] - b[i]).abs() < 1e-12,
            "A x residual[{i}] = {}",
            (ax[i] - b[i]).abs()
        );
    }

    let mut y = b.clone();
    dense_lu_solve(&lu, &piv, &mut y, n, true);
    let aty = matvec(&a_col, &y, n, true);
    for i in 0..n {
        assert!(
            (aty[i] - b[i]).abs() < 1e-12,
            "Aᵀ y residual[{i}] = {}",
            (aty[i] - b[i]).abs()
        );
    }

    // For this symmetric A, x and y must coincide.
    for i in 0..n {
        assert!(
            (x[i] - y[i]).abs() < 1e-12,
            "symmetric A: x[{i}]={} y[{i}]={}",
            x[i],
            y[i]
        );
    }
}

#[test]
fn lu_solve_transposed_with_pivoting() {
    // A matrix whose first column forces a row swap during pivoting,
    // exercising the transposed permutation `Pᵀ`.
    let n = 3;
    let a_rows = [[0.0_f64, 2.0, 1.0], [4.0, 1.0, 0.0], [1.0, 1.0, 3.0]];
    let mut a_col = vec![0.0_f64; n * n];
    for (i, row) in a_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a_col[j * n + i] = v;
        }
    }
    let (lu, piv) = dense_lu_factorize(&a_col, n);
    // Row 0 (value 0) must have been pivoted with a later row.
    assert_ne!(piv[0], 0, "expected a pivot swap on column 0");

    let b = vec![5.0_f64, -1.0, 2.0];
    let mut x = b.clone();
    dense_lu_solve(&lu, &piv, &mut x, n, true);

    let residual = matvec(&a_col, &x, n, true);
    for i in 0..n {
        assert!(
            (residual[i] - b[i]).abs() < 1e-10,
            "pivoted Aᵀ solve residual[{i}] = {}",
            (residual[i] - b[i]).abs()
        );
    }
}

#[test]
fn lu_solve_transposed_temp_dir_roundtrip() {
    // Persist a solved system through a temp file and re-verify, per the
    // workspace temp-file testing policy.
    let n = 3;
    let a_rows = [[3.0_f64, 1.0, 2.0], [6.0, 3.0, 4.0], [3.0, 1.0, 5.0]];
    let mut a_col = vec![0.0_f64; n * n];
    for (i, row) in a_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a_col[j * n + i] = v;
        }
    }
    let (lu, piv) = dense_lu_factorize(&a_col, n);
    let b = vec![2.0_f64, 4.0, 6.0];
    let mut x = b.clone();
    dense_lu_solve(&lu, &piv, &mut x, n, true);

    let path = std::env::temp_dir().join("oxicuda_lu_solve_transposed_s15.txt");
    let serialized = x
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(&path, &serialized).expect("write temp solution");
    let read_back = std::fs::read_to_string(&path).expect("read temp solution");
    let _ = std::fs::remove_file(&path);

    let restored: Vec<f64> = read_back
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("parse f64"))
        .collect();
    assert_eq!(restored.len(), n);

    let residual = matvec(&a_col, &restored, n, true);
    for i in 0..n {
        assert!(
            (residual[i] - b[i]).abs() < 1e-10,
            "round-tripped Aᵀ solve residual[{i}] = {}",
            (residual[i] - b[i]).abs()
        );
    }
}
