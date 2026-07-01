//! Helper utilities used by the Metal backend, plus the integration test
//! suite (gated on `#[cfg(test)]`).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Round up to the next power of 2 (minimum 1).
#[cfg(target_os = "macos")]
pub(super) fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Interpret a byte slice as little-endian `f32` values.
pub(super) fn read_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode `f32` values as little-endian bytes.
pub(super) fn write_f32_le(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use oxicuda_backend::{
        BackendError, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
    };

    use super::super::types::MetalBackend;
    #[test]
    fn metal_backend_new_uninitialized() {
        let b = MetalBackend::new();
        assert!(!b.is_initialized());
    }
    #[test]
    fn metal_backend_name() {
        let b = MetalBackend::new();
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn metal_backend_default() {
        let b = MetalBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn backend_debug_impl() {
        let b = MetalBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("MetalBackend"));
    }
    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(MetalBackend::new());
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn backend_not_initialized_gemm() {
        let b = MetalBackend::new();
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
        let b = MetalBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_synchronize() {
        let b = MetalBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_free() {
        let b = MetalBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = MetalBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = MetalBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }
    #[test]
    fn batched_gemm_not_initialized() {
        let b = MetalBackend::new();
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
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.batched_gemm(
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
                0,
            ),
            Ok(())
        );
    }
    #[test]
    fn batched_gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                0,
                1,
                0,
                0.0,
                0,
                1,
                0,
                3,
            ),
            Ok(())
        );
    }
    #[test]
    fn gemm_f16_not_initialized() {
        let b = MetalBackend::new();
        let result = b.gemm_f16(
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
    fn gemm_f16_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm_f16(
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
            ),
            Ok(())
        );
    }
    fn try_init() -> Option<MetalBackend> {
        let mut b = MetalBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_backend_init_on_macos() {
        let mut backend = MetalBackend::new();
        match backend.init() {
            Ok(()) => {
                assert!(backend.is_initialized());
                assert_eq!(backend.init(), Ok(()));
                assert!(backend.is_initialized());
                let result = backend.alloc(64);
                match result {
                    Ok(handle) => {
                        assert!(handle > 0);
                        backend.free(handle).expect("free should succeed");
                    }
                    Err(e) => {
                        let _ = e;
                    }
                }
            }
            Err(e) => {
                let _ = e;
            }
        }
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
    fn gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm(
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
                1
            ),
            Ok(())
        );
    }
    #[test]
    fn unary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }
    #[test]
    fn binary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }
    #[test]
    fn synchronize_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }
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
    fn attention_invalid_scale_error() {
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
    fn conv2d_wrong_input_rank() {
        let Some(b) = try_init() else {
            return;
        };
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
    fn conv2d_wrong_filter_rank() {
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
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }
    #[test]
    fn metal_init_graceful_failure() {
        let mut b = MetalBackend::new();
        let _result = b.init();
    }
    #[test]
    fn alloc_copy_roundtrip() {
        let Some(b) = try_init() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }
    /// Helper: encode f32 slice to bytes (little-endian).
    #[cfg(target_os = "macos")]
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; std::mem::size_of_val(data)];
        for (i, &val) in data.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
        bytes
    }
    /// Helper: decode bytes to f32 vec (little-endian).
    #[cfg(target_os = "macos")]
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn unary_relu_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![-1.0f32, 0.0, 1.0, 2.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Relu, ih, oh, n).expect("unary relu");
        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![0.0f32, 0.0, 1.0, 2.0]);
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn unary_neg_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, -2.0, 3.0, 0.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Neg, ih, oh, n).expect("unary neg");
        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![-1.0f32, 2.0, -3.0, -0.0]);
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn binary_add_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bv = vec![10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Add, ah, bh, oh, n).expect("binary add");
        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![11.0f32, 22.0, 33.0, 44.0]);
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn binary_mul_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let bv = vec![10.0f32, 10.0, 10.0, 10.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Mul, ah, bh, oh, n).expect("binary mul");
        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![20.0f32, 30.0, 40.0, 50.0]);
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_sum_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[4], 0)
            .expect("reduce sum");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 10.0).abs() < 1e-5,
            "expected 10.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_max_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![3.0f32, 1.0, 4.0, 1.5];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Max, ih, oh, &[4], 0)
            .expect("reduce max");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_mean_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![2.0f32, 4.0, 6.0, 8.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Mean, ih, oh, &[4], 0)
            .expect("reduce mean");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_2d_axis1_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(8) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 8]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[2, 3], 1)
            .expect("reduce sum axis=1");
        let mut out = vec![0u8; 8];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < 1e-5,
            "expected 15.0, got {}",
            result[1]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn gemm_simple_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bm = vec![5.0f32, 6.0, 7.0, 8.0];
        let c_init = vec![0.0f32; 4];
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bm);
        let bc = f32_to_bytes(&c_init);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let ch = match b.alloc(bc.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.copy_htod(ch, &bc).expect("htod c");
        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            ah,
            2,
            bh,
            2,
            0.0,
            ch,
            2,
        )
        .expect("gemm");
        let mut out = vec![0u8; bc.len()];
        b.copy_dtoh(&mut out, ch).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 19.0).abs() < 1e-4,
            "C[0,0]={}, expected 19",
            result[0]
        );
        assert!(
            (result[1] - 22.0).abs() < 1e-4,
            "C[0,1]={}, expected 22",
            result[1]
        );
        assert!(
            (result[2] - 43.0).abs() < 1e-4,
            "C[1,0]={}, expected 43",
            result[2]
        );
        assert!(
            (result[3] - 50.0).abs() < 1e-4,
            "C[1,1]={}, expected 50",
            result[3]
        );
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(ch).expect("free");
    }

    /// Zero-copy import: register pre-existing external `metal::Buffer`s and run
    /// `gemm` directly on them (no host round-trip), then verify the result
    /// matches a CPU triple-loop and that freeing/dropping the backend does NOT
    /// deallocate the caller-owned buffers.
    #[test]
    #[cfg(target_os = "macos")]
    fn import_external_gemm_zero_copy() {
        let Some(backend) = try_init() else { return };
        let Some(device) = metal::Device::system_default() else {
            return;
        };

        // Row-major operands: A is 3x2, B is 2x3, C is 3x3.
        let m = 3usize;
        let k = 2usize;
        let n = 3usize;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let bmat: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3
        let a_bytes = f32_to_bytes(&a);
        let b_bytes = f32_to_bytes(&bmat);
        let c_len_bytes = m * n * std::mem::size_of::<f32>();

        // Build EXTERNAL buffers the way a consumer's cache would: the test (not
        // oxicuda) owns these `metal::Buffer`s for the whole function.
        let a_buf = device.new_buffer_with_data(
            a_bytes.as_ptr() as *const std::ffi::c_void,
            a_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b_bytes.as_ptr() as *const std::ffi::c_void,
            b_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let c_buf = device.new_buffer(
            c_len_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Register the external buffers as zero-copy handles.
        let a_h = backend
            .register_external(&a_buf, a_bytes.len())
            .expect("register external A");
        let b_h = backend
            .register_external(&b_buf, b_bytes.len())
            .expect("register external B");
        // Exercise the by-value convenience wrapper for the resident output C.
        let c_h = backend
            .import_buffer(c_buf.clone(), c_len_bytes)
            .expect("import external C");

        // All three handles must be flagged as imported (external).
        assert_eq!(backend.is_imported(a_h), Ok(Some(true)));
        assert_eq!(backend.is_imported(b_h), Ok(Some(true)));
        assert_eq!(backend.is_imported(c_h), Ok(Some(true)));

        // C(3x3) = 1.0 * A(3x2) * B(2x3) + 0.0 * C, all row-major.
        backend
            .gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                m,
                n,
                k,
                1.0,
                a_h,
                k,
                b_h,
                n,
                0.0,
                c_h,
                n,
            )
            .expect("gemm on imported handles");

        // Read the result back via oxicuda and via the caller's own buffer
        // pointer — both must agree (proves the GPU wrote the caller's memory).
        let mut out = vec![0u8; c_len_bytes];
        backend.copy_dtoh(&mut out, c_h).expect("dtoh C");
        let got = bytes_to_f32(&out);

        let direct = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const f32, m * n) };

        // CPU reference triple-loop.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * bmat[p * n + j];
                }
                want[i * n + j] = acc;
            }
        }

        for idx in 0..(m * n) {
            let tol = 1e-3 * want[idx].abs().max(1.0);
            assert!(
                (got[idx] - want[idx]).abs() <= tol,
                "dtoh[{idx}]={} vs cpu {}",
                got[idx],
                want[idx]
            );
            assert!(
                (direct[idx] - want[idx]).abs() <= tol,
                "caller-buffer[{idx}]={} vs cpu {} (zero-copy write mismatch)",
                direct[idx],
                want[idx]
            );
        }

        // Also exercise the mixed case: external A·B → oxicuda-OWNED output.
        let owned_c = backend.alloc(c_len_bytes).expect("alloc owned C");
        assert_eq!(backend.is_imported(owned_c), Ok(Some(false)));
        backend
            .gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                m,
                n,
                k,
                1.0,
                a_h,
                k,
                b_h,
                n,
                0.0,
                owned_c,
                n,
            )
            .expect("gemm external A,B -> owned C");
        let mut owned_out = vec![0u8; c_len_bytes];
        backend
            .copy_dtoh(&mut owned_out, owned_c)
            .expect("dtoh owned C");
        let owned_got = bytes_to_f32(&owned_out);
        for idx in 0..(m * n) {
            let tol = 1e-3 * want[idx].abs().max(1.0);
            assert!(
                (owned_got[idx] - want[idx]).abs() <= tol,
                "owned-out[{idx}]={} vs cpu {}",
                owned_got[idx],
                want[idx]
            );
        }

        // Free everything. Imported handles must NOT deallocate the caller's
        // buffers; the owned handle releases normally.
        backend.free(a_h).expect("free imported A");
        backend.free(b_h).expect("free imported B");
        backend.free(c_h).expect("free imported C");
        backend.free(owned_c).expect("free owned C");
        // Handles are gone now.
        assert_eq!(backend.is_imported(a_h), Ok(None));

        // The external buffers are STILL alive and readable — the test owns
        // them. Reading them after oxicuda freed its handles proves there was no
        // double-free / premature deallocation.
        let still_a =
            unsafe { std::slice::from_raw_parts(a_buf.contents() as *const f32, a.len()) };
        assert_eq!(still_a, a.as_slice(), "A survived oxicuda free");
        let still_c = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const f32, m * n) };
        assert_eq!(still_c, want.as_slice(), "C result survived oxicuda free");

        // Drop the backend explicitly while the external buffers are still held;
        // backend drop must not touch them either.
        drop(backend);
        let after_drop =
            unsafe { std::slice::from_raw_parts(b_buf.contents() as *const f32, bmat.len()) };
        assert_eq!(after_drop, bmat.as_slice(), "B survived backend drop");

        // a_buf / b_buf / c_buf drop here (the test's sole remaining retains).
    }

    /// Device-to-device copy with no host round-trip, mixing an oxicuda-owned
    /// source with an imported external destination (the residency scenario:
    /// land a resident result into a consumer's cached buffer).
    #[test]
    #[cfg(target_os = "macos")]
    fn copy_dtod_owned_to_external() {
        let Some(backend) = try_init() else { return };
        let Some(device) = metal::Device::system_default() else {
            return;
        };

        let src_vals: Vec<f32> = vec![1.5, -2.0, 3.25, 4.0, 5.0, 6.5];
        let src_bytes = f32_to_bytes(&src_vals);
        let len = src_bytes.len();

        // Owned source (filled from host once), external destination.
        let src_h = match backend.alloc(len) {
            Ok(h) => h,
            Err(_) => return,
        };
        backend.copy_htod(src_h, &src_bytes).expect("htod src");

        let dst_buf = device.new_buffer(len as u64, metal::MTLResourceOptions::StorageModeShared);
        let dst_h = backend
            .import_buffer(dst_buf.clone(), len)
            .expect("import dst");

        backend.copy_dtod(dst_h, src_h, len).expect("copy_dtod");

        // Verify via the caller's own buffer pointer (zero-copy landed there).
        let landed =
            unsafe { std::slice::from_raw_parts(dst_buf.contents() as *const f32, src_vals.len()) };
        for (i, (&g, &w)) in landed.iter().zip(src_vals.iter()).enumerate() {
            assert!((g - w).abs() <= 1e-6, "dtod[{i}]={g} vs {w}");
        }

        // src == dst must be rejected.
        assert!(matches!(
            backend.copy_dtod(src_h, src_h, len),
            Err(oxicuda_backend::BackendError::InvalidArgument(_))
        ));

        backend.free(src_h).expect("free src");
        backend.free(dst_h).expect("free dst");
        // dst_buf still owned by the test; drop here is safe.
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_identity_1x1() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let filter = vec![2.0f32];
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod input");
        b.copy_htod(fh, &fb).expect("htod filter");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero output");
        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 1, 1],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 1x1");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "1x1 mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_3x3_basic() {
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        let expected = [54.0f32, 63.0, 90.0, 99.0];
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.conv2d_forward(
            ih,
            &[1, 1, 4, 4],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 3x3");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "3x3 mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_with_padding() {
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        let expected = vec![12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0];
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[1, 1],
        )
        .expect("conv2d padded");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "pad mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_uniform() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32; 2 * 2];
        let k = vec![1.0f32; 3 * 2];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod q");
        b.copy_htod(kh, &kb).expect("htod k");
        b.copy_htod(vh, &vb).expect("htod v");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 2, 3, 2, scale, false)
            .expect("attention uniform");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for sq in 0..2 {
            let base = sq * 2;
            assert!(
                (result[base] - 3.0).abs() < 0.1,
                "sq={sq} d=0: {} vs 3.0",
                result[base]
            );
            assert!(
                (result[base + 1] - 4.0).abs() < 0.1,
                "sq={sq} d=1: {} vs 4.0",
                result[base + 1]
            );
        }
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_causal() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32; 3 * 2];
        let k = vec![1.0f32; 3 * 2];
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected = [10.0f32, 20.0, 20.0, 30.0, 30.0, 40.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = expected.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 3, 3, 2, scale, true)
            .expect("attention causal");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.5, "causal idx {i}: {r} vs {e}");
        }
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_dominant_key() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32, 0.0];
        let k = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.attention(qh, kh, vh, oh, 1, 1, 1, 3, 2, 1.0, false)
            .expect("attention dominant");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!((result[0] - 5.0).abs() < 0.01, "d=0: {} vs 5.0", result[0]);
        assert!((result[1] - 6.0).abs() < 0.01, "d=1: {} vs 6.0", result[1]);
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn launch_custom_kernel_fma_roundtrip() -> oxicuda_backend::BackendResult<()> {
        use super::{read_f32_le, write_f32_le};

        let mut backend = MetalBackend::new();
        if backend.init().is_err() {
            // No Metal device available (e.g. headless CI) — skip gracefully.
            return Ok(());
        }

        // Fused multiply-add kernel with an explicit bounds check.
        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fma_kernel(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    constant float&     c   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid] + c;
}
"#;

        let count: usize = 1024;
        let byte_len = count * std::mem::size_of::<f32>();
        // Deterministic inputs — no randomness.
        let a: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..count).map(|i| i as f32 - 100.0).collect();
        let c: f32 = 3.25;

        let a_h = backend.alloc(byte_len)?;
        let b_h = backend.alloc(byte_len)?;
        let out_h = backend.alloc(byte_len)?;
        backend.copy_htod(a_h, &write_f32_le(&a))?;
        backend.copy_htod(b_h, &write_f32_le(&b))?;

        let n_le = (count as u32).to_le_bytes();
        let c_le = c.to_le_bytes();
        backend.launch_custom_kernel(
            msl,
            "fma_kernel",
            &[a_h, b_h, out_h],
            &[&n_le, &c_le],
            count,
        )?;

        let mut out_bytes = vec![0u8; byte_len];
        backend.copy_dtoh(&mut out_bytes, out_h)?;
        let out = read_f32_le(&out_bytes);

        for i in 0..count {
            let want = a[i] * b[i] + c;
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (out[i] - want).abs() <= tol,
                "index {i}: got {}, want {want}",
                out[i]
            );
        }

        // A second launch must reuse the cached pipeline and stay correct.
        backend.launch_custom_kernel(
            msl,
            "fma_kernel",
            &[a_h, b_h, out_h],
            &[&n_le, &c_le],
            count,
        )?;
        backend.copy_dtoh(&mut out_bytes, out_h)?;
        let out2 = read_f32_le(&out_bytes);
        for i in 0..count {
            let want = a[i] * b[i] + c;
            let tol = 1e-3 * want.abs().max(1.0);
            assert!((out2[i] - want).abs() <= tol, "second launch index {i}");
        }

        backend.free(a_h)?;
        backend.free(b_h)?;
        backend.free(out_h)?;
        Ok(())
    }
}
