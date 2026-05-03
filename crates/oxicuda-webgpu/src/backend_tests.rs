//! Tests for `WebGpuBackend`.
//!
//! Extracted from `backend.rs` so the production module stays under the
//! 2 000-line refactoring policy.  Included into `backend.rs` via
//! `#[path = "backend_tests.rs"] mod tests;` so all tests still live in the
//! `backend::tests` namespace.

use super::*;
use oxicuda_backend::{BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

// ── Construction ──────────────────────────────────────────────────────────

#[test]
fn webgpu_backend_new_uninitialized() {
    let b = WebGpuBackend::new();
    assert!(!b.is_initialized());
}

#[test]
fn webgpu_backend_name() {
    let b = WebGpuBackend::new();
    assert_eq!(b.name(), "webgpu");
}

#[test]
fn webgpu_backend_default() {
    let b = WebGpuBackend::default();
    assert!(!b.is_initialized());
    assert_eq!(b.name(), "webgpu");
}

#[test]
fn backend_debug_impl() {
    let b = WebGpuBackend::new();
    let s = format!("{b:?}");
    assert!(s.contains("WebGpuBackend"));
}

// ── Object-safety smoke test ──────────────────────────────────────────────

#[test]
fn backend_object_safe() {
    let b: Box<dyn ComputeBackend> = Box::new(WebGpuBackend::new());
    assert_eq!(b.name(), "webgpu");
}

// ── Not-initialized guards ────────────────────────────────────────────────

#[test]
fn backend_not_initialized_gemm() {
    let b = WebGpuBackend::new();
    let result = b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        0,
        4,
        0.0,
        0,
        4,
    );
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_alloc() {
    let b = WebGpuBackend::new();
    let result = b.alloc(1024);
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_synchronize() {
    let b = WebGpuBackend::new();
    assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_free() {
    let b = WebGpuBackend::new();
    assert_eq!(b.free(1), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_copy_htod() {
    let b = WebGpuBackend::new();
    assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
}

#[test]
fn backend_not_initialized_copy_dtoh() {
    let b = WebGpuBackend::new();
    let mut buf = [0u8; 4];
    assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
}

// ── Zero-size / trivial-OK paths (no GPU needed) ─────────────────────────

/// These tests exercise the "no-op for zero size" branches.  We need the
/// backend to be initialised, but if no GPU is available we skip.
fn try_init() -> Option<WebGpuBackend> {
    let mut b = WebGpuBackend::new();
    match b.init() {
        Ok(()) => Some(b),
        Err(_) => None,
    }
}

#[test]
fn gemm_zero_size_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    let result = b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        0,
        0,
        0,
        1.0,
        0,
        1,
        0,
        1,
        0.0,
        0,
        1,
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn unary_zero_elements_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
}

#[test]
fn binary_zero_elements_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
}

#[test]
fn copy_htod_empty_noop() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.copy_htod(0, &[]), Ok(()));
}

#[test]
fn copy_dtoh_empty_noop() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
}

#[test]
fn alloc_zero_bytes_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.alloc(0),
        Err(BackendError::InvalidArgument(
            "cannot allocate 0 bytes".into()
        ))
    );
}

#[test]
fn synchronize_after_init() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(b.synchronize(), Ok(()));
}

// ── Argument validation (post-init) ───────────────────────────────────────

#[test]
fn reduce_empty_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
        Err(BackendError::InvalidArgument(
            "shape must not be empty".into()
        ))
    );
}

#[test]
fn reduce_axis_out_of_bounds_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
        Err(BackendError::InvalidArgument(
            "axis 5 is out of bounds for shape of length 2".into()
        ))
    );
}

#[test]
fn attention_zero_seq_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
        Err(BackendError::InvalidArgument(
            "seq_q, seq_kv, and head_dim must all be > 0".into()
        ))
    );
}

#[test]
fn attention_nonpositive_scale_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
        Err(BackendError::InvalidArgument(
            "scale must be a positive finite number, got 0".into()
        ))
    );
    assert_eq!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
        Err(BackendError::InvalidArgument(
            "scale must be a positive finite number, got -1".into()
        ))
    );
    assert!(
        b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
            .is_err()
    );
}

#[test]
fn conv2d_wrong_input_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    // 3-element input_shape — should fail.
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32],
            0,
            &[16, 3, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1, 1],
            &[0, 0]
        ),
        Err(BackendError::InvalidArgument(
            "input_shape must have 4 elements (NCHW)".into()
        ))
    );
}

#[test]
fn conv2d_wrong_filter_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[16, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1, 1],
            &[0, 0]
        ),
        Err(BackendError::InvalidArgument(
            "filter_shape must have 4 elements (KCFHFW)".into()
        ))
    );
}

#[test]
fn conv2d_wrong_stride_shape_error() {
    let Some(b) = try_init() else {
        return;
    };
    assert_eq!(
        b.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[16, 3, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1], // <-- wrong
            &[0, 0],
        ),
        Err(BackendError::InvalidArgument(
            "stride must have 2 elements [sh, sw]".into()
        ))
    );
}

// ── Init is idempotent ────────────────────────────────────────────────────

#[test]
fn init_idempotent() {
    let Some(mut b) = try_init() else {
        return;
    };
    // Second call must succeed without error.
    assert_eq!(b.init(), Ok(()));
    assert!(b.is_initialized());
}

// ── Graceful failure ──────────────────────────────────────────────────────

#[test]
fn webgpu_init_graceful_failure() {
    // We cannot force a failure, but we can at least verify that init()
    // returns a Result and never panics.
    let mut b = WebGpuBackend::new();
    let _result = b.init(); // Ok or Err — both are acceptable.
    // No panic => test passes.
}

// ── GPU compute tests ─────────────────────────────────────────────────────
//
// These helpers upload f32 slices and read back results, exercising the
// full shader → pipeline → dispatch path.

/// Helper: upload `data` (f32 slice) to a new GPU buffer, return its handle.
fn upload_f32(b: &WebGpuBackend, data: &[f32]) -> u64 {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let h = b.alloc(bytes.len()).expect("alloc");
    b.copy_htod(h, &bytes).expect("copy_htod");
    h
}

/// Helper: download `n` f32 values from a GPU buffer handle.
fn download_f32(b: &WebGpuBackend, h: u64, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    b.copy_dtoh(&mut bytes, h).expect("copy_dtoh");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn unary_neg_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, -2.0, 3.0, 0.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(input.len() * 4).expect("alloc output");

    b.unary(UnaryOp::Neg, in_h, out_h, input.len())
        .expect("unary neg");

    let result = download_f32(&b, out_h, input.len());
    let expected = [-1.0f32, 2.0, -3.0, 0.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn unary_abs_small() {
    let Some(b) = try_init() else { return };
    let input = [-3.0f32, 4.0, -5.0, 0.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(input.len() * 4).expect("alloc output");

    b.unary(UnaryOp::Abs, in_h, out_h, input.len())
        .expect("unary abs");

    let result = download_f32(&b, out_h, input.len());
    let expected = [3.0f32, 4.0, 5.0, 0.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn binary_add_small() {
    let Some(b) = try_init() else { return };
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let bv = [10.0f32, 20.0, 30.0, 40.0];
    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bv);
    let out_h = b.alloc(a.len() * 4).expect("alloc output");

    b.binary(BinaryOp::Add, a_h, b_h, out_h, a.len())
        .expect("binary add");

    let result = download_f32(&b, out_h, a.len());
    let expected = [11.0f32, 22.0, 33.0, 44.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn binary_mul_small() {
    let Some(b) = try_init() else { return };
    let a = [2.0f32, 3.0, 4.0, 5.0];
    let bv = [10.0f32, 10.0, 10.0, 10.0];
    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bv);
    let out_h = b.alloc(a.len() * 4).expect("alloc output");

    b.binary(BinaryOp::Mul, a_h, b_h, out_h, a.len())
        .expect("binary mul");

    let result = download_f32(&b, out_h, a.len());
    let expected = [20.0f32, 30.0, 40.0, 50.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_sum_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output"); // single f32

    b.reduce(ReduceOp::Sum, in_h, out_h, &[4], 0)
        .expect("reduce sum");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 10.0).abs() < 1e-5,
        "expected 10.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_max_small() {
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 5.0, 3.0, 2.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Max, in_h, out_h, &[4], 0)
        .expect("reduce max");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "expected 5.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_mean_small() {
    let Some(b) = try_init() else { return };
    let input = [2.0f32, 4.0, 6.0, 8.0];
    let in_h = upload_f32(&b, &input);
    let out_h = b.alloc(4).expect("alloc output");

    b.reduce(ReduceOp::Mean, in_h, out_h, &[4], 0)
        .expect("reduce mean");

    let result = download_f32(&b, out_h, 1);
    assert!(
        (result[0] - 5.0).abs() < 1e-5,
        "expected 5.0, got {}",
        result[0]
    );

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn gemm_identity_2x2() {
    let Some(b) = try_init() else { return };
    // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity), C = zeros
    // C = 1.0 * A * I + 0.0 * C = A
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &eye);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a_h,
        2,
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm");

    let result = download_f32(&b, c_h, 4);
    for (r, e) in result.iter().zip(a.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_2x3_times_3x2() {
    let Some(b) = try_init() else { return };
    // A 2x3, B 3x2 → C 2x2
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bm = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c_init = [0.0f32; 4];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        3,
        1.0,
        a_h,
        3,
        b_h,
        2,
        0.0,
        c_h,
        2,
    )
    .expect("gemm");

    // Expected: [[58, 64], [139, 154]]
    let result = download_f32(&b, c_h, 4);
    let expected = [58.0f32, 64.0, 139.0, 154.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

#[test]
fn gemm_alpha_beta() {
    let Some(b) = try_init() else { return };
    // C = 2.0 * A * B + 3.0 * C
    // A = [[1,0],[0,1]], B = [[1,0],[0,1]], C = [[1,1],[1,1]]
    // C = 2*I + 3*ones = [[5,3],[3,5]]
    let a = [1.0f32, 0.0, 0.0, 1.0];
    let bm = [1.0f32, 0.0, 0.0, 1.0];
    let c_init = [1.0f32, 1.0, 1.0, 1.0];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &bm);
    let c_h = upload_f32(&b, &c_init);

    b.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        2.0,
        a_h,
        2,
        b_h,
        2,
        3.0,
        c_h,
        2,
    )
    .expect("gemm alpha+beta");

    let result = download_f32(&b, c_h, 4);
    let expected = [5.0f32, 3.0, 3.0, 5.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── Conv2D tests ──────────────────────────────────────────────────────

#[test]
fn conv2d_identity_1x1() {
    // 1×1 convolution with single channel, no padding, stride=1
    // input: 1×1×3×3, filter: 1×1×1×1 (weight=2.0), output: 1×1×3×3
    let Some(b) = try_init() else { return };
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let filter = [2.0f32];
    let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(9 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 3, 3],
        f_h,
        &[1, 1, 1, 1],
        out_h,
        &[1, 1, 3, 3],
        &[1, 1],
        &[0, 0],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 9);
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn conv2d_3x3_no_padding() {
    // input: 1×1×4×4, filter: 1×1×3×3 (all ones), stride=1, pad=0
    // output: 1×1×2×2
    let Some(b) = try_init() else { return };
    let input: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let filter = [1.0f32; 9];

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(4 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 4, 4],
        f_h,
        &[1, 1, 3, 3],
        out_h,
        &[1, 1, 2, 2],
        &[1, 1],
        &[0, 0],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 4);
    // top-left 3×3 sum: 0+1+2+4+5+6+8+9+10 = 45
    assert!((result[0] - 45.0).abs() < 1e-4, "got {}", result[0]);
    // top-right 3×3 sum: 1+2+3+5+6+7+9+10+11 = 54
    assert!((result[1] - 54.0).abs() < 1e-4, "got {}", result[1]);

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn conv2d_with_padding() {
    // input: 1×1×2×2, filter: 1×1×3×3 (all ones), stride=1, pad=1
    // output: 1×1×2×2
    // With padding=1 around a 2×2 input, the output is also 2×2.
    let Some(b) = try_init() else { return };
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let filter = [1.0f32; 9];

    let in_h = upload_f32(&b, &input);
    let f_h = upload_f32(&b, &filter);
    let out_h = b.alloc(4 * 4).expect("alloc output");

    b.conv2d_forward(
        in_h,
        &[1, 1, 2, 2],
        f_h,
        &[1, 1, 3, 3],
        out_h,
        &[1, 1, 2, 2],
        &[1, 1],
        &[1, 1],
    )
    .expect("conv2d");

    let result = download_f32(&b, out_h, 4);
    // Top-left output: only 4 of 9 filter taps hit valid input
    // input[0,0]=1, input[0,1]=2, input[1,0]=3, input[1,1]=4 => sum=10
    assert!((result[0] - 10.0).abs() < 1e-4, "got {}", result[0]);

    b.free(in_h).expect("free");
    b.free(f_h).expect("free");
    b.free(out_h).expect("free");
}

// ── Attention tests ───────────────────────────────────────────────────

#[test]
fn attention_uniform_weights() {
    // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
    // Q = [1, 0], K = [[1, 0], [1, 0]], V = [[1, 2], [3, 4]]
    // scores = [1*scale, 1*scale] => equal weights => O = mean(V) = [2, 3]
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 0.0];
    let k = [1.0f32, 0.0, 1.0, 0.0];
    let v = [1.0f32, 2.0, 3.0, 4.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention");

    let result = download_f32(&b, o_h, 2);
    // Equal scores → equal softmax weights → average of V rows
    assert!(
        (result[0] - 2.0).abs() < 1e-4,
        "got {}, expected 2.0",
        result[0]
    );
    assert!(
        (result[1] - 3.0).abs() < 1e-4,
        "got {}, expected 3.0",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

#[test]
fn attention_causal_single_token() {
    // 1 head, seq_q=2, seq_kv=2, head_dim=1, causal
    // Q = [1, 1], K = [1, 1], V = [10, 20]
    // sq=0: only sees sk=0 → O[0] = V[0] = 10
    // sq=1: sees sk=0,1 with equal scores → O[1] = (10+20)/2 = 15
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 1.0];
    let k = [1.0f32, 1.0];
    let v = [10.0f32, 20.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    b.attention(q_h, k_h, v_h, o_h, 1, 1, 2, 2, 1, 1.0, true)
        .expect("attention causal");

    let result = download_f32(&b, o_h, 2);
    assert!(
        (result[0] - 10.0).abs() < 1e-4,
        "got {}, expected 10.0",
        result[0]
    );
    assert!(
        (result[1] - 15.0).abs() < 1e-4,
        "got {}, expected 15.0",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}

// ── Batched GEMM tests ─────────────────────────────────────────────

#[test]
fn batched_gemm_not_initialized() {
    let b = WebGpuBackend::new();
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn batched_gemm_zero_batch_noop() {
    let Some(b) = try_init() else { return };
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        0, // batch_count = 0
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn batched_gemm_zero_dims_noop() {
    let Some(b) = try_init() else { return };
    // m = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        0,
        4,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
    // n = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        0,
        4,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
    // k = 0
    let result = b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        4,
        4,
        0,
        1.0,
        0,
        4,
        16,
        0,
        4,
        16,
        0.0,
        0,
        4,
        16,
        2,
    );
    assert_eq!(result, Ok(()));
}

#[test]
fn batched_gemm_identity_2x2() {
    let Some(b) = try_init() else { return };
    // 2 batches of 2x2 identity multiply
    // batch 0: A0=[[1,2],[3,4]] * I = [[1,2],[3,4]]
    // batch 1: A1=[[5,6],[7,8]] * I = [[5,6],[7,8]]
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let c_init = [0.0f32; 8];

    let a_h = upload_f32(&b, &a);
    let b_h = upload_f32(&b, &eye);
    let c_h = upload_f32(&b, &c_init);

    b.batched_gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a_h,
        2,
        4, // stride_a = 2*2 = 4
        b_h,
        2,
        4, // stride_b = 4
        0.0,
        c_h,
        2,
        4, // stride_c = 4
        2, // batch_count
    )
    .expect("batched_gemm");

    let result = download_f32(&b, c_h, 8);
    for (r, e) in result.iter().zip(a.iter()) {
        assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
    }

    b.free(a_h).expect("free");
    b.free(b_h).expect("free");
    b.free(c_h).expect("free");
}

// ── FP16 GEMM tests ─────────────────────────────────────────────────

#[test]
fn gemm_f16_not_initialized() {
    let b = WebGpuBackend::new();
    let result = b.gemm_f16(4, 4, 4, 1.0, 0, 0, 0.0, 0);
    assert_eq!(result, Err(BackendError::NotInitialized));
}

#[test]
fn gemm_f16_zero_dims_noop() {
    let Some(b) = try_init() else { return };
    assert_eq!(b.gemm_f16(0, 4, 4, 1.0, 0, 0, 0.0, 0), Ok(()));
    assert_eq!(b.gemm_f16(4, 0, 4, 1.0, 0, 0, 0.0, 0), Ok(()));
    assert_eq!(b.gemm_f16(4, 4, 0, 1.0, 0, 0, 0.0, 0), Ok(()));
}

// ── N-D reduce tests ──────────────────────────────────────────────────
//
// Each test computes the same reduction on the host and compares.

/// Reference CPU reduction along `axis` of a row-major tensor.
fn cpu_reduce(data: &[f32], shape: &[usize], axis: usize, op: ReduceOp) -> Vec<f32> {
    let outer: usize = shape[..axis].iter().product();
    let dk: usize = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let total = outer * inner;

    let neutral = match op {
        ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
    };

    let mut out = vec![neutral; total];
    for o in 0..outer {
        for i in 0..dk {
            for j in 0..inner {
                let v = data[o * dk * inner + i * inner + j];
                let slot = o * inner + j;
                out[slot] = match op {
                    ReduceOp::Sum | ReduceOp::Mean => out[slot] + v,
                    ReduceOp::Max => out[slot].max(v),
                    ReduceOp::Min => out[slot].min(v),
                };
            }
        }
    }
    if op == ReduceOp::Mean && dk > 0 {
        for v in &mut out {
            *v /= dk as f32;
        }
    }
    out
}

fn run_nd_reduce_case(data: Vec<f32>, shape: Vec<usize>, axis: usize, op: ReduceOp) {
    let Some(b) = try_init() else { return };

    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let out_len = outer * inner;
    let expected = cpu_reduce(&data, &shape, axis, op);

    let in_h = upload_f32(&b, &data);
    let out_h = b.alloc(out_len * 4).expect("alloc output");

    b.reduce(op, in_h, out_h, &shape, axis).expect("reduce nd");

    let result = download_f32(&b, out_h, out_len);
    for (idx, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        // Use a slightly relaxed tolerance — large dk can accumulate error.
        assert!(
            (r - e).abs() < 1e-3 * (1.0 + e.abs()),
            "axis={axis}, op={op:?}, slot={idx}: got {r}, expected {e}"
        );
    }

    b.free(in_h).expect("free");
    b.free(out_h).expect("free");
}

#[test]
fn reduce_2d_sum_axis0() {
    // shape [3, 4], reduce axis 0 → length 4
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Sum);
}

#[test]
fn reduce_2d_sum_axis1() {
    // shape [3, 4], reduce axis 1 → length 3
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 1, ReduceOp::Sum);
}

#[test]
fn reduce_2d_max_axis0() {
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0, 9.0, 0.0, 1.5, 2.5];
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Max);
}

#[test]
fn reduce_2d_min_axis1() {
    let data = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 6.0, 4.0, 9.0, 0.0, 1.5, 2.5];
    run_nd_reduce_case(data, vec![3, 4], 1, ReduceOp::Min);
}

#[test]
fn reduce_2d_mean_axis0() {
    // Mean along axis 0 should divide by dk = 3 inside the shader.
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![3, 4], 0, ReduceOp::Mean);
}

#[test]
fn reduce_3d_sum_axis0() {
    // shape [2, 3, 4] → output [3, 4]
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 0, ReduceOp::Sum);
}

#[test]
fn reduce_3d_sum_axis1() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 1, ReduceOp::Sum);
}

#[test]
fn reduce_3d_sum_axis2() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 2, ReduceOp::Sum);
}

#[test]
fn reduce_3d_mean_axis1() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4], 1, ReduceOp::Mean);
}

#[test]
fn reduce_4d_sum_axis2() {
    // shape [2, 3, 4, 2] → output [2, 3, 2]
    let data: Vec<f32> = (0..48).map(|x| x as f32 * 0.5).collect();
    run_nd_reduce_case(data, vec![2, 3, 4, 2], 2, ReduceOp::Sum);
}

#[test]
fn reduce_4d_max_axis3() {
    let data: Vec<f32> = (0..48).map(|x| ((x * 13) % 47) as f32).collect();
    run_nd_reduce_case(data, vec![2, 3, 4, 2], 3, ReduceOp::Max);
}

#[test]
fn reduce_2d_dk_one_is_copy() {
    // shape [3, 1], reduce along axis 1 → output [3], values copied.
    let data = vec![7.0f32, -2.5, 3.25];
    run_nd_reduce_case(data, vec![3, 1], 1, ReduceOp::Sum);
}

#[test]
fn reduce_3d_dk_one_axis0() {
    // shape [1, 3, 4], reduce axis 0 → output [3, 4], values copied.
    let data: Vec<f32> = (0..12).map(|x| x as f32 + 0.5).collect();
    run_nd_reduce_case(data, vec![1, 3, 4], 0, ReduceOp::Mean);
}

#[test]
fn reduce_2d_single_output_via_nd() {
    // shape [1, 5], axis 1 → output [1].  Exercises outer=1, inner=1
    // through the N-D path (not the 1-D fast path because shape.len() > 1).
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    run_nd_reduce_case(data, vec![1, 5], 1, ReduceOp::Sum);
}

#[test]
fn reduce_large_dk_strided_loop() {
    // dk = 1024 exercises the per-thread strided loop (256 threads, 4 iter).
    let data: Vec<f32> = (0..1024).map(|x| (x as f32) * 0.001).collect();
    run_nd_reduce_case(data, vec![1, 1024], 1, ReduceOp::Sum);
}

#[test]
fn attention_dominant_key() {
    // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
    // Q = [1, 0], K = [[10, 0], [0, 0]], V = [[100, 200], [0, 0]]
    // score[0] = 10*scale, score[1] = 0*scale
    // With large enough difference, softmax saturates → O ≈ V[0]
    let Some(b) = try_init() else { return };

    let q = [1.0f32, 0.0];
    let k = [10.0f32, 0.0, 0.0, 0.0];
    let v = [100.0f32, 200.0, 0.0, 0.0];

    let q_h = upload_f32(&b, &q);
    let k_h = upload_f32(&b, &k);
    let v_h = upload_f32(&b, &v);
    let o_h = b.alloc(2 * 4).expect("alloc output");

    // scale=1.0 gives scores 10 vs 0 → softmax ≈ [1, 0]
    b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
        .expect("attention dominant");

    let result = download_f32(&b, o_h, 2);
    assert!(
        (result[0] - 100.0).abs() < 0.1,
        "got {}, expected ~100",
        result[0]
    );
    assert!(
        (result[1] - 200.0).abs() < 0.1,
        "got {}, expected ~200",
        result[1]
    );

    b.free(q_h).expect("free");
    b.free(k_h).expect("free");
    b.free(v_h).expect("free");
    b.free(o_h).expect("free");
}
