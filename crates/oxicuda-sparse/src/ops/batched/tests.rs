use super::*;

// -- BatchScheduler tests -----------------------------------------------

#[test]
fn scheduler_sequential_for_small_batch_large_matrices() {
    let s = BatchScheduler::new();
    assert_eq!(s.select_strategy(2, 50_000), Strategy::Sequential);
    assert_eq!(s.select_strategy(4, 10_000), Strategy::Sequential);
}

#[test]
fn scheduler_fused_for_large_batch_small_matrices() {
    let s = BatchScheduler::new();
    assert_eq!(s.select_strategy(100, 100), Strategy::Fused);
    assert_eq!(s.select_strategy(64, 255), Strategy::Fused);
}

#[test]
fn scheduler_concurrent_for_medium_cases() {
    let s = BatchScheduler::new();
    let strat = s.select_strategy(16, 1000);
    match strat {
        Strategy::Concurrent(n) => assert!((1..=8).contains(&n)),
        other => panic!("expected Concurrent, got {:?}", other),
    }
}

#[test]
fn scheduler_concurrent_caps_at_8_streams() {
    let strat = BatchScheduler::select_strategy_static(32, 1000);
    assert_eq!(strat, Strategy::Concurrent(8));
}

#[test]
fn scheduler_static_matches_instance() {
    let s = BatchScheduler::new();
    for (bs, nnz) in [(1, 100), (10, 500), (64, 100), (3, 20_000)] {
        assert_eq!(
            s.select_strategy(bs, nnz),
            BatchScheduler::select_strategy_static(bs, nnz)
        );
    }
}

#[test]
fn scheduler_default_trait() {
    let s = BatchScheduler::default();
    // Should not panic
    let _ = s.select_strategy(1, 1);
}

// -- BatchedSpMVPlan tests (host arrays) --------------------------------

#[test]
fn plan_from_host_arrays_basic() {
    // Two 2x2 identity matrices
    let rp = vec![vec![0, 1, 2], vec![0, 1, 2]];
    let ci = vec![vec![0, 1], vec![0, 1]];
    let vals: Vec<Vec<f32>> = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
    let rows = vec![2, 2];
    let cols = vec![2, 2];

    let plan = BatchedSpMVPlan::from_host_arrays(&rp, &ci, &vals, &rows, &cols)
        .expect("plan creation should succeed");

    assert_eq!(plan.batch_size, 2);
    assert_eq!(plan.row_counts, vec![2, 2]);
    assert_eq!(plan.nnz_counts, vec![2, 2]);
    assert_eq!(plan.total_nnz(), 4);
    assert_eq!(plan.total_rows(), 4);
    assert_eq!(plan.avg_nnz(), 2);
    assert_eq!(plan.batch_offsets_row_ptr, vec![0, 3]);
    assert_eq!(plan.batch_offsets_nnz, vec![0, 2]);
    assert_eq!(plan.concat_row_ptr, vec![0, 1, 2, 0, 1, 2]);
    assert_eq!(plan.concat_col_idx, vec![0, 1, 0, 1]);
}

#[test]
fn plan_from_host_arrays_empty_batch() {
    let result = BatchedSpMVPlan::<f32>::from_host_arrays(&[], &[], &[], &[], &[]);
    assert!(result.is_err());
}

#[test]
fn plan_from_host_arrays_length_mismatch() {
    let rp = vec![vec![0, 1]];
    let ci = vec![vec![0], vec![1]]; // wrong length
    let vals: Vec<Vec<f64>> = vec![vec![1.0]];
    let result = BatchedSpMVPlan::from_host_arrays(&rp, &ci, &vals, &[1], &[1]);
    assert!(result.is_err());
}

// -- BatchedSpMV host execution tests -----------------------------------

#[test]
fn spmv_host_identity_batch() {
    // Two 3x3 identity matrices
    let rp = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
    let ci = vec![vec![0, 1, 2], vec![0, 1, 2]];
    let vals = vec![vec![1.0_f64, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
    let rows = vec![3, 3];
    let cols = vec![3, 3];

    let batch =
        BatchedSpMV::from_host(rp, ci, vals, rows, cols).expect("batch creation should succeed");
    assert_eq!(batch.batch_size(), 2);

    let xs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let mut ys = vec![vec![0.0; 3], vec![0.0; 3]];

    batch
        .execute(&xs, &mut ys, 1.0, 0.0)
        .expect("execute should succeed");

    assert_eq!(ys[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(ys[1], vec![4.0, 5.0, 6.0]);
}

#[test]
fn spmv_host_alpha_beta() {
    // Single 2x2 matrix: [[2, 0], [0, 3]]
    let rp = vec![vec![0, 1, 2]];
    let ci = vec![vec![0, 1]];
    let vals = vec![vec![2.0_f32, 3.0]];
    let rows = vec![2];
    let cols = vec![2];

    let batch = BatchedSpMV::from_host(rp, ci, vals, rows, cols).expect("batch creation");

    let xs = vec![vec![1.0, 1.0]];
    let mut ys = vec![vec![10.0, 20.0]];

    // y = 2.0 * A*x + 0.5 * y
    // A*x = [2, 3]
    // y = [2*2 + 0.5*10, 2*3 + 0.5*20] = [9, 16]
    batch
        .execute(&xs, &mut ys, 2.0, 0.5)
        .expect("execute should succeed");

    assert!((ys[0][0] - 9.0).abs() < 1e-6);
    assert!((ys[0][1] - 16.0).abs() < 1e-6);
}

#[test]
fn spmv_host_dimension_mismatch() {
    let rp = vec![vec![0, 1]];
    let ci = vec![vec![0]];
    let vals = vec![vec![1.0_f32]];
    let rows = vec![1];
    let cols = vec![2];

    let batch = BatchedSpMV::from_host(rp, ci, vals, rows, cols).expect("batch creation");

    // Wrong number of vectors
    let xs = vec![vec![1.0; 2], vec![1.0; 2]];
    let mut ys = vec![vec![0.0], vec![0.0]];
    assert!(batch.execute(&xs, &mut ys, 1.0, 0.0).is_err());
}

#[test]
fn spmv_host_empty_batch_error() {
    let result = BatchedSpMV::<f64>::from_host(vec![], vec![], vec![], vec![], vec![]);
    assert!(result.is_err());
}

// -- UniformBatchedSpMV tests -------------------------------------------

#[test]
fn uniform_spmv_host_basic() {
    // Shared pattern: 2x2 diagonal
    let row_ptr = vec![0, 1, 2];
    let col_idx = vec![0, 1];
    let batch_values = vec![
        vec![1.0_f64, 1.0], // identity
        vec![2.0, 3.0],     // diag(2, 3)
    ];

    let uniform = UniformBatchedSpMV::from_host_arrays(2, 2, row_ptr, col_idx, batch_values)
        .expect("creation should succeed");
    assert_eq!(uniform.batch_size(), 2);

    let xs = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
    let mut ys = vec![vec![0.0; 2], vec![0.0; 2]];

    uniform
        .execute(&xs, &mut ys, 1.0, 0.0)
        .expect("execute should succeed");

    assert_eq!(ys[0], vec![1.0, 2.0]); // I * [1,2]
    assert!((ys[1][0] - 2.0).abs() < 1e-10); // 2*1
    assert!((ys[1][1] - 6.0).abs() < 1e-10); // 3*2
}

#[test]
fn uniform_spmv_validation_errors() {
    // Empty batch_values
    let result =
        UniformBatchedSpMV::<f32>::from_host_arrays(2, 2, vec![0, 1, 2], vec![0, 1], vec![]);
    assert!(result.is_err());

    // Wrong values length
    let result = UniformBatchedSpMV::<f32>::from_host_arrays(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![vec![1.0]], // too short
    );
    assert!(result.is_err());

    // Wrong row_ptr length
    let result = UniformBatchedSpMV::<f32>::from_host_arrays(
        2,
        2,
        vec![0, 1], // too short
        vec![0, 1],
        vec![vec![1.0, 2.0]],
    );
    assert!(result.is_err());
}

// -- BatchedTriSolve tests (host) ---------------------------------------

#[test]
fn tri_solve_host_basic() {
    // L = [[2, 0], [1, 3]], b = [4, 7]
    // x[0] = 4/2 = 2
    // x[1] = (7 - 1*2)/3 = 5/3
    let rp = vec![vec![0, 1, 3]];
    let ci = vec![vec![0, 0, 1]];
    let vals = vec![vec![2.0_f64, 1.0, 3.0]];
    let sizes = vec![2];
    let rhs = vec![vec![4.0, 7.0]];

    let results =
        BatchedTriSolve::execute_host(&rp, &ci, &vals, &sizes, &rhs).expect("solve should succeed");

    assert_eq!(results.len(), 1);
    assert!((results[0][0] - 2.0).abs() < 1e-10);
    assert!((results[0][1] - 5.0 / 3.0).abs() < 1e-10);
}

#[test]
fn tri_solve_host_singular() {
    // L has zero on diagonal -> singular
    let rp = vec![vec![0, 1, 2]];
    let ci = vec![vec![0, 0]]; // row 1 has col 0 but no col 1 entry -> diag = 0
    let vals = vec![vec![1.0_f64, 2.0]];
    let sizes = vec![2];
    let rhs = vec![vec![1.0, 1.0]];

    let result = BatchedTriSolve::execute_host(&rp, &ci, &vals, &sizes, &rhs);
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// Task 5c: SpMV / SpMM numerical accuracy tests (host-side, CPU-only)
// -----------------------------------------------------------------------

// Helper: reference dense matrix-vector product y = A * x for a row-major
// matrix A stored as a flat slice with `cols` columns.
#[allow(dead_code)]
fn dense_matvec(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        for c in 0..cols {
            y[r] += a[r * cols + c] * x[c];
        }
    }
    y
}

// Helper: reference dense matrix-matrix product C = A * B, A is (m×k),
// B is (k×n), result C is (m×n), all stored row-major.
fn dense_matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// SpMV for a small 5×5 sparse matrix with known values.
///
/// Matrix (dense):
/// ```text
/// [[1, 0, 2, 0, 0],
///  [0, 3, 0, 0, 0],
///  [0, 0, 4, 0, 5],
///  [0, 0, 0, 6, 0],
///  [7, 0, 0, 0, 8]]
/// ```
/// x = [1, 2, 3, 4, 5]
/// Expected: [1+6, 6, 12+25, 24, 7+40] = [7, 6, 37, 24, 47]
#[test]
fn test_spmv_numerical_accuracy_small() {
    // CSR for the matrix above
    let row_ptr = vec![vec![0i32, 2, 3, 5, 6, 8]];
    let col_idx = vec![vec![0i32, 2, 1, 2, 4, 3, 0, 4]];
    let values = vec![vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
    let rows = vec![5usize];
    let cols = vec![5usize];

    let batch = BatchedSpMV::from_host(row_ptr, col_idx, values, rows, cols)
        .expect("batch creation should succeed");

    let x = vec![vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]];
    let mut y = vec![vec![0.0_f64; 5]];

    batch
        .execute(&x, &mut y, 1.0, 0.0)
        .expect("execute should succeed");

    let expected = [7.0_f64, 6.0, 37.0, 24.0, 47.0];
    for (i, (&got, &exp)) in y[0].iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-14,
            "y[{i}]: expected {exp}, got {got}"
        );
    }
}

/// SpMV accuracy with large value spread (1e-6 to 1e6) — relative error < 1e-10.
#[test]
fn test_spmv_numerical_accuracy_value_spread() {
    // 4×4 diagonal with wildly varying values
    let big = 1e6_f64;
    let small = 1e-6_f64;
    // diag(1e-6, 1e6, 1e-6, 1e6)
    let row_ptr = vec![vec![0i32, 1, 2, 3, 4]];
    let col_idx = vec![vec![0i32, 1, 2, 3]];
    let values = vec![vec![small, big, small, big]];
    let rows = vec![4usize];
    let cols = vec![4usize];

    let batch =
        BatchedSpMV::from_host(row_ptr, col_idx, values, rows, cols).expect("batch creation");

    let x = vec![vec![1.0_f64, 1.0, 1.0, 1.0]];
    let mut y = vec![vec![0.0_f64; 4]];
    batch.execute(&x, &mut y, 1.0, 0.0).expect("execute");

    let expected = [small, big, small, big];
    for (i, (&got, &exp)) in y[0].iter().zip(expected.iter()).enumerate() {
        let rel_err = (got - exp).abs() / exp.abs().max(1e-300);
        assert!(
            rel_err < 1e-10,
            "y[{i}]: relative error {rel_err:.3e} exceeds threshold"
        );
    }
}

/// SpMV with alpha/beta scaling: y = alpha * A * x + beta * y.
#[test]
fn test_spmv_alpha_beta_scaling() {
    // A = [[2, 0], [0, 3]], x = [1, 1], y_init = [10, 20]
    // A*x = [2, 3]
    // y = 5 * [2, 3] + 0.25 * [10, 20] = [10+2.5, 15+5] = [12.5, 20]
    let row_ptr = vec![vec![0i32, 1, 2]];
    let col_idx = vec![vec![0i32, 1]];
    let values = vec![vec![2.0_f64, 3.0]];

    let batch =
        BatchedSpMV::from_host(row_ptr, col_idx, values, vec![2], vec![2]).expect("batch creation");

    let x = vec![vec![1.0_f64, 1.0]];
    let mut y = vec![vec![10.0_f64, 20.0]];
    batch.execute(&x, &mut y, 5.0, 0.25).expect("execute");

    assert!((y[0][0] - 12.5).abs() < 1e-13, "y[0] = {}", y[0][0]);
    assert!((y[0][1] - 20.0).abs() < 1e-13, "y[1] = {}", y[0][1]);
}

/// SpMV for identity matrix: I * x = x.
#[test]
fn test_spmv_identity_matrix() {
    let n = 6usize;
    let row_ptr = vec![(0..=(n as i32)).collect::<Vec<i32>>()];
    let col_idx = vec![(0..n as i32).collect::<Vec<i32>>()];
    let values = vec![vec![1.0_f64; n]];

    let batch =
        BatchedSpMV::from_host(row_ptr, col_idx, values, vec![n], vec![n]).expect("batch creation");

    let x_data: Vec<f64> = (1..=(n as i64)).map(|v| v as f64).collect();
    let x = vec![x_data.clone()];
    let mut y = vec![vec![0.0_f64; n]];
    batch.execute(&x, &mut y, 1.0, 0.0).expect("execute");

    for (i, (&got, &exp)) in y[0].iter().zip(x_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-14,
            "identity: y[{i}] = {got}, expected {exp}"
        );
    }
}

/// SpMM: A * B = C where B is a dense matrix (multiple right-hand sides).
///
/// Uses `BatchedSpMV::execute` once per column of B as a surrogate for SpMM.
/// A is 4×4, B is 4×3, C = A * B verified against dense reference.
#[test]
fn test_spmm_numerical_accuracy() {
    // A (4×4 sparse):
    // [[1, 0, 2, 0],
    //  [0, 3, 0, 4],
    //  [5, 0, 6, 0],
    //  [0, 7, 0, 8]]
    let a_dense = [
        1.0_f64, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0, 8.0,
    ];

    let row_ptr = vec![0i32, 2, 4, 6, 8];
    let col_idx = vec![0i32, 2, 1, 3, 0, 2, 1, 3];
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // B (4×3):
    let b_dense = [
        1.0_f64, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 4.0,
    ];

    // C_ref = dense_matmul(A, B) (4×3)
    let c_ref = dense_matmul(&a_dense, 4, 4, &b_dense, 3);

    // Compute each column of C using SpMV
    let n_cols = 3usize;
    let mut c_got = vec![0.0_f64; 4 * n_cols];

    for col in 0..n_cols {
        // Extract column `col` of B as a vector
        let x_col: Vec<f64> = (0..4).map(|r| b_dense[r * n_cols + col]).collect();

        let batch = BatchedSpMV::from_host(
            vec![row_ptr.clone()],
            vec![col_idx.clone()],
            vec![values.clone()],
            vec![4],
            vec![4],
        )
        .expect("batch creation");

        let mut y = vec![vec![0.0_f64; 4]];
        batch.execute(&[x_col], &mut y, 1.0, 0.0).expect("execute");

        for row in 0..4 {
            c_got[row * n_cols + col] = y[0][row];
        }
    }

    for i in 0..4 * n_cols {
        assert!(
            (c_got[i] - c_ref[i]).abs() < 1e-13,
            "C[{i}]: got {}, expected {}",
            c_got[i],
            c_ref[i]
        );
    }
}

// -- PTX generation test ------------------------------------------------

#[test]
fn ptx_generation_f32() {
    let ptx = generate_batched_spmv_ptx::<f32>();
    assert!(ptx.contains("batched_spmv_f32"));
    assert!(ptx.contains(".f32"));
    assert!(ptx.contains(".version"));
}

#[test]
fn ptx_generation_f64() {
    let ptx = generate_batched_spmv_ptx::<f64>();
    assert!(ptx.contains("batched_spmv_f64"));
    assert!(ptx.contains(".f64"));
}

#[test]
fn batched_spmv_ptx_contains_loop() {
    let ptx = generate_batched_spmv_ptx::<f32>();
    assert!(ptx.contains("ROW_LOOP"));
    assert!(ptx.contains("fma.rn"));
    assert!(ptx.contains("ld.global"));
}

// -- BatchedSpGEMM host test --------------------------------------------

#[test]
fn spgemm_host_identity_times_matrix() {
    // I * A = A for 2x2
    let i_rp = vec![vec![0, 1, 2]];
    let i_ci = vec![vec![0, 1]];
    let i_vals: Vec<Vec<f64>> = vec![vec![1.0, 1.0]];

    let a_rp = vec![vec![0, 2, 3]]; // row 0: cols 0,1; row 1: col 1
    let a_ci = vec![vec![0, 1, 1]];
    let a_vals: Vec<Vec<f64>> = vec![vec![2.0, 3.0, 4.0]];

    let results = BatchedSpGEMM::execute_host(
        &i_rp,
        &i_ci,
        &i_vals,
        &[2],
        &[2],
        &a_rp,
        &a_ci,
        &a_vals,
        &[2],
    )
    .expect("spgemm should succeed");

    assert_eq!(results.len(), 1);
    let (c_rp, c_ci, c_vals, m, n) = &results[0];
    assert_eq!(*m, 2);
    assert_eq!(*n, 2);
    // Row 0 of C should have entries at cols 0 and 1
    let r0_start = c_rp[0] as usize;
    let r0_end = c_rp[1] as usize;
    assert_eq!(r0_end - r0_start, 2);
    assert_eq!(c_ci[r0_start], 0);
    assert_eq!(c_ci[r0_start + 1], 1);
    assert!((c_vals[r0_start] - 2.0).abs() < 1e-10);
    assert!((c_vals[r0_start + 1] - 3.0).abs() < 1e-10);
}

// -- batched_spmv_cpu tests ------------------------------------------------

#[test]
fn batched_spmv_identity_2rhs() {
    // 4x4 identity matrix
    let n_rows = 4usize;
    let n_cols = 4usize;
    let row_ptr = vec![0u32, 1, 2, 3, 4];
    let col_idx = vec![0u32, 1, 2, 3];
    let values = vec![1.0f32; 4];

    // X = [[1,5], [2,6], [3,7], [4,8]] column-major:
    // x_batch[col * batch_size + b]:
    //   col=0: b=0 -> 1, b=1 -> 5
    //   col=1: b=0 -> 2, b=1 -> 6
    //   col=2: b=0 -> 3, b=1 -> 7
    //   col=3: b=0 -> 4, b=1 -> 8
    let batch_size = 2usize;
    let x_batch = vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];

    let y = batched_spmv_cpu(
        n_rows, n_cols, &row_ptr, &col_idx, &values, &x_batch, batch_size,
    );

    // y[row * batch_size + b] = x[row, b] since A is identity
    // row=0, b=0 -> 1; row=0, b=1 -> 5
    // row=1, b=0 -> 2; row=1, b=1 -> 6
    // row=2, b=0 -> 3; row=2, b=1 -> 7
    // row=3, b=0 -> 4; row=3, b=1 -> 8
    assert_eq!(y.len(), n_rows * batch_size);
    assert!((y[0] - 1.0).abs() < 1e-6, "row=0, b=0 should be 1.0");
    assert!((y[1] - 5.0).abs() < 1e-6, "row=0, b=1 should be 5.0");
    assert!((y[2] - 2.0).abs() < 1e-6, "row=1, b=0 should be 2.0");
    assert!((y[3] - 6.0).abs() < 1e-6, "row=1, b=1 should be 6.0");
    assert!((y[4] - 3.0).abs() < 1e-6, "row=2, b=0 should be 3.0");
    assert!((y[5] - 7.0).abs() < 1e-6, "row=2, b=1 should be 7.0");
    assert!((y[6] - 4.0).abs() < 1e-6, "row=3, b=0 should be 4.0");
    assert!((y[7] - 8.0).abs() < 1e-6, "row=3, b=1 should be 8.0");
}

#[test]
fn batched_spmv_correctness_3rhs() {
    // 3x3 matrix A:
    //  [2 1 0]
    //  [0 3 1]
    //  [1 0 4]
    // row_ptr = [0, 2, 4, 6]
    // col_idx = [0, 1, 1, 2, 0, 2]
    // values  = [2, 1, 3, 1, 1, 4]
    let n_rows = 3usize;
    let n_cols = 3usize;
    let row_ptr = vec![0u32, 2, 4, 6];
    let col_idx = vec![0u32, 1, 1, 2, 0, 2];
    let values = vec![2.0f32, 1.0, 3.0, 1.0, 1.0, 4.0];
    let batch_size = 3usize;

    // 3 RHS column-major: x[:,0]=[1,0,0], x[:,1]=[0,1,0], x[:,2]=[0,0,1]
    // i.e., x_batch[col*3+b]: col=0: [1,0,0], col=1: [0,1,0], col=2: [0,0,1]
    let x_batch = vec![
        1.0f32, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, // col 1
        0.0, 0.0, 1.0, // col 2
    ];

    let y = batched_spmv_cpu(
        n_rows, n_cols, &row_ptr, &col_idx, &values, &x_batch, batch_size,
    );

    // A * [1,0,0] = [2,0,1]  => y[row*3+0]
    // A * [0,1,0] = [1,3,0]  => y[row*3+1]
    // A * [0,0,1] = [0,1,4]  => y[row*3+2]
    assert!((y[0] - 2.0).abs() < 1e-6, "A*e0 row0 = 2");
    assert!((y[1] - 1.0).abs() < 1e-6, "A*e1 row0 = 1");
    assert!((y[2] - 0.0).abs() < 1e-6, "A*e2 row0 = 0");
    assert!((y[3] - 0.0).abs() < 1e-6, "A*e0 row1 = 0");
    assert!((y[4] - 3.0).abs() < 1e-6, "A*e1 row1 = 3");
    assert!((y[5] - 1.0).abs() < 1e-6, "A*e2 row1 = 1");
    assert!((y[6] - 1.0).abs() < 1e-6, "A*e0 row2 = 1");
    assert!((y[7] - 0.0).abs() < 1e-6, "A*e1 row2 = 0");
    assert!((y[8] - 4.0).abs() < 1e-6, "A*e2 row2 = 4");
}

// -- mixed_precision_spmv_cpu tests ----------------------------------------

#[test]
fn mixed_precision_spmv_correctness() {
    // 4x4 identity matrix, x=[1,2,3,4] => y=[1,2,3,4]
    let n_rows = 4usize;
    let row_ptr = vec![0u32, 1, 2, 3, 4];
    let col_idx = vec![0u32, 1, 2, 3];
    let values_fp16 = vec![1.0f32; 4];
    let x = vec![1.0f32, 2.0, 3.0, 4.0];

    let y = mixed_precision_spmv_cpu(n_rows, &row_ptr, &col_idx, &values_fp16, &x);

    assert_eq!(y.len(), n_rows);
    for (i, &yi) in y.iter().enumerate() {
        assert!(
            (yi - (i + 1) as f32).abs() < 1e-4,
            "y[{}] should be {} but got {}",
            i,
            i + 1,
            yi
        );
    }
}

#[test]
fn mixed_precision_accumulation_fp32() {
    // Large sum: 1000-row diagonal, all values=1.0, x=[1.0; 1000]
    // Each y[i] = 1.0, which is finite and representable in f32
    let n_rows = 1000usize;
    let row_ptr: Vec<u32> = (0..=1000).map(|i| i as u32).collect();
    let col_idx: Vec<u32> = (0..1000).map(|i| i as u32).collect();
    let values_fp16 = vec![1.0f32; 1000];
    let x = vec![1.0f32; 1000];

    let y = mixed_precision_spmv_cpu(n_rows, &row_ptr, &col_idx, &values_fp16, &x);

    assert_eq!(y.len(), n_rows);
    // All results should be finite (the accumulation via f64 avoids overflow)
    for (i, &yi) in y.iter().enumerate() {
        assert!(yi.is_finite(), "y[{}] = {} is not finite", i, yi);
        assert!(
            (yi - 1.0).abs() < 1e-4,
            "y[{}] should be 1.0, got {}",
            i,
            yi
        );
    }
}
