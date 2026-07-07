//! Strided batched GEMM.
//!
//! All matrices in the batch share a common base pointer and are separated by
//! a constant stride.  This avoids the pointer-indirection overhead of the
//! pointer-array variant and is the preferred path when matrices are laid out
//! contiguously (or with uniform padding) in a single allocation.
//!
//! The kernel uses a 3-D grid where `blockIdx.z` encodes the batch index.
//! Each thread-block computes its per-batch pointer offsets as:
//!
//! ```text
//! A_i = a_base + batch_idx * stride_a
//! B_i = b_base + batch_idx * stride_b
//! C_i = c_base + batch_idx * stride_c
//! D_i = d_base + batch_idx * stride_d
//! ```

use oxicuda_driver::CudaError;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::templates::gemm::{EpilogueKind, GemmTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

/// Default tile dimensions for the strided batched kernel.
const TILE_M: u32 = 16;
/// Default tile dimension along N.
const TILE_N: u32 = 16;
/// Default tile dimension along K.
const TILE_K: u32 = 16;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates the strided batched GEMM arguments.
#[allow(clippy::too_many_arguments)]
fn validate_strided_args<T: GpuFloat>(
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    ldd: u32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    stride_d: i64,
    batch_count: u32,
    trans_a: Transpose,
    trans_b: Transpose,
) -> BlasResult<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(
            "m, n, and k must all be positive".into(),
        ));
    }

    // The per-batch kernel addresses matrices as row-major, so a leading
    // dimension is a *row stride* and must be at least the column count of the
    // physically stored matrix: k for an untransposed A (m x k) or m for a
    // transposed A (stored k x m), and symmetrically for B. C and D are always
    // stored m x n, so their leading dimension must be at least n.
    let cols_a = match trans_a {
        Transpose::NoTrans => k,
        Transpose::Trans | Transpose::ConjTrans => m,
    };
    let cols_b = match trans_b {
        Transpose::NoTrans => n,
        Transpose::Trans | Transpose::ConjTrans => k,
    };

    if lda < cols_a {
        return Err(BlasError::InvalidDimension(format!(
            "lda ({lda}) must be >= columns of stored A ({cols_a})"
        )));
    }
    if ldb < cols_b {
        return Err(BlasError::InvalidDimension(format!(
            "ldb ({ldb}) must be >= columns of stored B ({cols_b})"
        )));
    }
    if ldc < n {
        return Err(BlasError::InvalidDimension(format!(
            "ldc ({ldc}) must be >= n ({n})"
        )));
    }
    if ldd < n {
        return Err(BlasError::InvalidDimension(format!(
            "ldd ({ldd}) must be >= n ({n})"
        )));
    }

    // Strides of zero are allowed only for batch_count <= 1 (broadcast).
    if batch_count > 1 && stride_a == 0 && stride_b == 0 && stride_c == 0 && stride_d == 0 {
        return Err(BlasError::InvalidArgument(
            "all strides are zero with batch_count > 1".into(),
        ));
    }

    let _elem = T::SIZE;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Builds a [`GemmTemplate`] with the standard tile sizes for strided dispatch.
fn build_gemm_template<T: GpuFloat>(sm: SmVersion) -> GemmTemplate {
    GemmTemplate {
        tile_m: TILE_M,
        tile_n: TILE_N,
        tile_k: TILE_K,
        warp_m: TILE_M,
        warp_n: TILE_N,
        precision: T::PTX_TYPE,
        accumulator: T::PTX_TYPE,
        use_tensor_core: false,
        stages: 1,
        target: sm,
        epilogue: EpilogueKind::LinearCombination,
    }
}

/// Applies a signed byte `offset` to a device pointer, wrapping on overflow.
///
/// Batch strides are signed element counts, so `offset` may be negative.
fn ptr_offset(base: CUdeviceptr, offset: i64) -> CUdeviceptr {
    (base as i64).wrapping_add(offset) as CUdeviceptr
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a strided batched GEMM.
///
/// ```text
/// D[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
///
/// where A[i] = a + i * stride_a  (element offset, not byte offset)
///       B[i] = b + i * stride_b
///       C[i] = c + i * stride_c
///       D[i] = d + i * stride_d
/// ```
///
/// This is more efficient than the pointer-array variant because per-batch
/// address computation is a simple multiply-add instead of a global-memory
/// load.
///
/// # Stride semantics
///
/// Strides are signed 64-bit **element** counts (not byte offsets).  A stride
/// of zero means the same matrix is broadcast to every batch element.  Negative
/// strides are legal and traverse the buffer in reverse order.
///
/// # Errors
///
/// * [`BlasError::InvalidDimension`] if `m`, `n`, or `k` is zero, or leading
///   dimensions are too small.
/// * [`BlasError::InvalidArgument`] if all strides are zero with
///   `batch_count > 1`.
/// * [`BlasError::PtxGeneration`] if the PTX kernel cannot be built.
/// * [`BlasError::LaunchFailed`] if the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a: CUdeviceptr,
    lda: u32,
    stride_a: i64,
    b: CUdeviceptr,
    ldb: u32,
    stride_b: i64,
    beta: T,
    c: CUdeviceptr,
    ldc: u32,
    stride_c: i64,
    d: CUdeviceptr,
    ldd: u32,
    stride_d: i64,
    batch_count: u32,
) -> BlasResult<()> {
    if batch_count == 0 {
        return Ok(());
    }

    validate_strided_args::<T>(
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ldd,
        stride_a,
        stride_b,
        stride_c,
        stride_d,
        batch_count,
        trans_a,
        trans_b,
    )?;

    let sm = handle.sm_version();

    // The per-batch kernel is the naive NoTrans, tightly packed row-major
    // GEMM (`GemmTemplate`). It cannot express transposition or a leading
    // dimension different from the tight width, so those cases are rejected —
    // matching how the single-GEMM path treats them — rather than being
    // launched with a mismatched argument layout (the previous behaviour,
    // which passed a 17-field tuple to an 8-parameter kernel and read/wrote
    // wild device pointers).
    if trans_a != Transpose::NoTrans || trans_b != Transpose::NoTrans {
        return Err(BlasError::UnsupportedOperation(
            "strided batched GEMM currently supports only NoTrans A and B".into(),
        ));
    }
    if lda != k || ldb != n || ldc != n || ldd != n {
        return Err(BlasError::UnsupportedOperation(
            "strided batched GEMM currently requires tightly packed row-major matrices \
             (lda = k, ldb = n, ldc = ldd = n)"
                .into(),
        ));
    }

    // Compile (once) the naive GEMM kernel and reuse it from the handle cache.
    let template = build_gemm_template::<T>(sm);
    let kernel_name = template.kernel_name();
    let module =
        handle.get_or_compile_module(&kernel_name, || template.generate().map_err(Into::into))?;
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    // Flat 1-D launch: the naive kernel derives a global linear id and walks
    // M*N with a grid-stride loop, so it is correct under any geometry.
    let total_elems = u64::from(m) * u64::from(n);
    let block_threads = 256u32;
    let grid_x = total_elems
        .div_ceil(u64::from(block_threads))
        .clamp(1, u64::from(u32::MAX)) as u32;
    let params = LaunchParams::new(Dim3::new(grid_x, 1, 1), Dim3::new(block_threads, 1, 1));

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    // Convert element strides to signed byte strides for per-batch offsets.
    let elem_bytes = T::SIZE as i64;
    let byte_stride_a = stride_a.saturating_mul(elem_bytes);
    let byte_stride_b = stride_b.saturating_mul(elem_bytes);
    let byte_stride_c = stride_c.saturating_mul(elem_bytes);
    let byte_stride_d = stride_d.saturating_mul(elem_bytes);

    // Byte span of one tight m x n matrix, for the C -> D snapshot below.
    let mn_bytes = m as usize * n as usize * T::SIZE;

    for i in 0..i64::from(batch_count) {
        let a_i = ptr_offset(a, i.wrapping_mul(byte_stride_a));
        let b_i = ptr_offset(b, i.wrapping_mul(byte_stride_b));
        let c_i = ptr_offset(c, i.wrapping_mul(byte_stride_c));
        let d_i = ptr_offset(d, i.wrapping_mul(byte_stride_d));

        // The kernel computes `param_c = alpha*A_i*B_i + beta*param_c` in place.
        // When D differs from C we first snapshot C_i into D_i (stream-ordered)
        // and run the kernel in place on D_i, so the result is
        // `D_i = alpha*A_i*B_i + beta*C_i`. Copying also avoids reading
        // uninitialised D when beta == 0.
        if d_i != c_i {
            match oxicuda_driver::memory_info::memcpy_device_to_device_async(
                d_i,
                c_i,
                mn_bytes,
                handle.stream(),
            ) {
                Ok(()) => {}
                Err(CudaError::NotSupported) => {
                    handle.stream().synchronize().map_err(BlasError::Cuda)?;
                    oxicuda_driver::memory_info::memcpy_device_to_device(d_i, c_i, mn_bytes)
                        .map_err(BlasError::Cuda)?;
                }
                Err(e) => return Err(BlasError::Cuda(e)),
            }
        }

        // Kernel arg order matches the single-GEMM dispatcher:
        // (a, b, c, m, n, k, alpha_bits, beta_bits).
        let args = (a_i, b_i, d_i, m, n, k, alpha_bits, beta_bits);
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(e.to_string()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero_dimensions() {
        let res = validate_strided_args::<f32>(
            0,
            64,
            64,
            64,
            64,
            64,
            64,
            1024,
            1024,
            1024,
            1024,
            8,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_rejects_all_zero_strides_multi_batch() {
        let res = validate_strided_args::<f32>(
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            0,
            0,
            0,
            0,
            8,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_accepts_zero_stride_single_batch() {
        let res = validate_strided_args::<f64>(
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            0,
            0,
            0,
            0,
            1,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn validate_accepts_negative_strides() {
        let res = validate_strided_args::<f32>(
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            -4096,
            -4096,
            -4096,
            -4096,
            4,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn validate_transposed_lda() {
        // Row-major convention: trans_a == Trans stores A as k x m, so its row
        // stride (lda) must be >= m = 64; lda = 64 is the tight case.
        let res = validate_strided_args::<f32>(
            64,
            64,
            16,
            64,
            64,
            64,
            64,
            1024,
            1024,
            1024,
            1024,
            2,
            Transpose::Trans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn validate_rejects_row_major_lda_too_small() {
        // NoTrans A is m x k, so lda must be >= k = 64; lda = 32 is rejected.
        let res = validate_strided_args::<f32>(
            48,
            40,
            64,
            32,
            40,
            40,
            40,
            1024,
            1024,
            1024,
            1024,
            2,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }
}
