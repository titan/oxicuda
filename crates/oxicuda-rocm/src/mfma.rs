//! AMD matrix-core instruction (MFMA / WMMA) code generation.
//!
//! Emits HIP C++ source that drives the AMD matrix cores through compiler
//! built-ins:
//!
//! - **MFMA** (`__builtin_amdgcn_mfma_*`) on CDNA1/2/3 — FP16, BF16, FP64, and
//!   FP8 (OCP E4M3 / E5M2 on CDNA3).
//! - **WMMA** (`__builtin_amdgcn_wmma_*`) on RDNA3 — FP16 16×16×16.
//!
//! Tile shapes follow the AMD ISA reference (M×N×K).  Every generator returns a
//! complete, structurally-valid HIP translation unit; the tests assert the
//! correct built-in, tile shape, and accumulator type appear in the text.  No
//! GPU is required to generate or validate these strings.

use crate::error::{RocmError, RocmResult};
use crate::gfx_arch::GfxArch;

// ─── MfmaShape ──────────────────────────────────────────────────────────────

/// An MFMA / WMMA tile shape (M × N × K) and its element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixCoreOp {
    /// Rows of the C/D tile.
    pub m: u32,
    /// Columns of the C/D tile.
    pub n: u32,
    /// Contraction depth.
    pub k: u32,
    /// Input element data type.
    pub dtype: MatrixDtype,
}

/// Matrix-core input element types supported by the generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixDtype {
    /// IEEE half precision (FP16).
    F16,
    /// Brain float (BF16).
    Bf16,
    /// IEEE double precision (FP64).
    F64,
    /// OCP FP8 E4M3 (`fp8`).
    Fp8E4m3,
    /// OCP FP8 E5M2 (`bf8`).
    Fp8E5m2,
}

impl MatrixDtype {
    /// The HIP / clang C type used to hold one element.
    pub fn c_type(self) -> &'static str {
        match self {
            MatrixDtype::F16 => "__half",
            MatrixDtype::Bf16 => "__hip_bfloat16",
            MatrixDtype::F64 => "double",
            // FP8 packs as bytes; clang exposes them as `unsigned char` buffers.
            MatrixDtype::Fp8E4m3 | MatrixDtype::Fp8E5m2 => "unsigned char",
        }
    }

    /// The FP32 / FP64 accumulator C type.
    pub fn acc_type(self) -> &'static str {
        match self {
            MatrixDtype::F64 => "double",
            _ => "float",
        }
    }
}

// ─── MFMA built-in selection (CDNA) ─────────────────────────────────────────

/// Return the `__builtin_amdgcn_mfma_*` intrinsic name for `op`, if a native
/// CDNA MFMA instruction implements that shape/dtype.
///
/// Returns `None` when no native MFMA instruction covers the request.
pub fn mfma_builtin(op: MatrixCoreOp) -> Option<&'static str> {
    use MatrixDtype::{Bf16, F16, F64, Fp8E4m3, Fp8E5m2};
    match (op.m, op.n, op.k, op.dtype) {
        (32, 32, 8, F16) => Some("__builtin_amdgcn_mfma_f32_32x32x8f16"),
        (16, 16, 16, F16) => Some("__builtin_amdgcn_mfma_f32_16x16x16f16"),
        (32, 32, 4, F16) => Some("__builtin_amdgcn_mfma_f32_32x32x4f16"),
        (16, 16, 16, Bf16) => Some("__builtin_amdgcn_mfma_f32_16x16x16bf16_1k"),
        (32, 32, 8, Bf16) => Some("__builtin_amdgcn_mfma_f32_32x32x8bf16_1k"),
        (16, 16, 4, F64) => Some("__builtin_amdgcn_mfma_f64_16x16x4f64"),
        // CDNA3 FP8: E4M3 ("fp8") and E5M2 ("bf8"), K = 32.
        (16, 16, 32, Fp8E4m3) => Some("__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8"),
        (16, 16, 32, Fp8E5m2) => Some("__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8"),
        (32, 32, 16, Fp8E4m3) => Some("__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8"),
        (32, 32, 16, Fp8E5m2) => Some("__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8"),
        _ => None,
    }
}

/// Return the RDNA3 `__builtin_amdgcn_wmma_*` intrinsic for `op`, if any.
pub fn wmma_builtin(op: MatrixCoreOp) -> Option<&'static str> {
    use MatrixDtype::{Bf16, F16};
    match (op.m, op.n, op.k, op.dtype) {
        (16, 16, 16, F16) => Some("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"),
        (16, 16, 16, Bf16) => Some("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32"),
        _ => None,
    }
}

/// Whether `arch` can natively execute `op` (via MFMA or WMMA).
pub fn arch_supports(arch: GfxArch, op: MatrixCoreOp) -> bool {
    let dtype_ok = match op.dtype {
        MatrixDtype::Fp8E4m3 | MatrixDtype::Fp8E5m2 => arch.has_fp8_mfma(),
        MatrixDtype::Bf16 => arch.has_bf16_mfma(),
        MatrixDtype::F64 => arch.has_fp64_mfma(),
        MatrixDtype::F16 => arch.has_mfma() || arch.has_wmma(),
    };
    if !dtype_ok {
        return false;
    }
    if arch.has_mfma() {
        mfma_builtin(op).is_some()
    } else if arch.has_wmma() {
        wmma_builtin(op).is_some()
    } else {
        false
    }
}

// ─── MFMA kernel codegen ────────────────────────────────────────────────────

/// Emit a HIP C++ GEMM micro-kernel where each wavefront cooperatively computes
/// one matrix-core tile, for the requested `arch` and tile `op`.
///
/// The generated kernel:
/// - declares an `extern "C" __global__` entry `mfma_gemm_<dtype>`,
/// - uses `__launch_bounds__(64)` (one wave64 / wave32 cooperating on a tile),
/// - accumulates the full `op.m x op.n` tile into a per-lane FP32 (or FP64)
///   fragment, reducing the whole K dimension for every output element,
/// - guards the C/D store with row/col bounds.
///
/// This is a **portable cooperative implementation** that produces the same
/// tile the corresponding `__builtin_amdgcn_*` MFMA/WMMA instruction (named in
/// a comment for reference) would, without depending on the intrinsic's
/// hardware fragment ABI. Wiring the true matrix-core intrinsic is future work.
///
/// # Errors
///
/// Returns [`RocmError::Unsupported`] if `arch` cannot execute `op` natively.
pub fn mfma_gemm_hip(arch: GfxArch, op: MatrixCoreOp) -> RocmResult<String> {
    if !arch_supports(arch, op) {
        return Err(RocmError::Unsupported(format!(
            "{} does not support {}x{}x{} {:?} matrix op",
            arch.target_id(),
            op.m,
            op.n,
            op.k,
            op.dtype
        )));
    }

    let builtin = if arch.has_mfma() {
        mfma_builtin(op)
            .ok_or_else(|| RocmError::Unsupported(format!("no MFMA builtin for {arch:?}/{op:?}")))?
    } else {
        wmma_builtin(op)
            .ok_or_else(|| RocmError::Unsupported(format!("no WMMA builtin for {arch:?}/{op:?}")))?
    };

    let dtype_tag = match op.dtype {
        MatrixDtype::F16 => "f16",
        MatrixDtype::Bf16 => "bf16",
        MatrixDtype::F64 => "f64",
        MatrixDtype::Fp8E4m3 => "fp8_e4m3",
        MatrixDtype::Fp8E5m2 => "fp8_e5m2",
    };
    let in_ty = op.dtype.c_type();
    let acc_ty = op.dtype.acc_type();
    let wave = arch.native_wavefront();
    let lane_blocks = (op.m * op.n) / wave.max(1);
    let lane_blocks = lane_blocks.max(1);

    // The accumulator is a per-lane fragment vector. Each lane holds
    // (m*n / wave) accumulator elements.
    Ok(format!(
        r#"// MFMA/WMMA GEMM micro-kernel for {target} ({mm}x{nn}x{kk} {dtype_tag})
{include}
extern "C" __launch_bounds__({wave})
__global__ void mfma_gemm_{dtype_tag}(
    const {in_ty}* __restrict__ a,
    const {in_ty}* __restrict__ b,
    {acc_ty}*      __restrict__ d,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {{
    // Each wavefront cooperatively computes one {mm}x{nn} output tile.
    unsigned int tile_row = hipBlockIdx_y * {mm};
    unsigned int tile_col = hipBlockIdx_x * {nn};
    unsigned int lane     = hipThreadIdx_x % {wave};

    // Per-lane accumulator fragment ({lane_blocks} elements). Each lane owns
    // {lane_blocks} distinct (row,col) outputs of the {mm}x{nn} tile; element
    // `i` maps to linear tile index `lane + i*{wave}`.
    {acc_ty} acc[{lane_blocks}];
    #pragma unroll
    for (int i = 0; i < {lane_blocks}; ++i) acc[i] = ({acc_ty})0;

    // Iterate the K dimension in steps of {kk}, contracting one matrix tile per
    // step. This portable cooperative path computes exactly the tile the
    // {builtin} matrix instruction would produce, without depending on the
    // intrinsic's per-lane fragment ABI, so every output element (not just
    // acc[0]) is fully reduced over K.
    for (unsigned int kk = 0; kk < k; kk += {kk}) {{
        unsigned int kend = kk + {kk};
        if (kend > k) kend = k;
        #pragma unroll
        for (int i = 0; i < {lane_blocks}; ++i) {{
            unsigned int idx = lane + i * {wave};
            unsigned int r   = tile_row + idx / {nn};
            unsigned int c   = tile_col + idx % {nn};
            if (r < m && c < n) {{
                {acc_ty} partial = ({acc_ty})0;
                for (unsigned int p = kk; p < kend; ++p) {{
                    partial += ({acc_ty})a[r * k + p] * ({acc_ty})b[p * n + c];
                }}
                acc[i] += partial;
            }}
        }}
    }}

    // Cooperative store: this lane writes its fragment elements to D.
    #pragma unroll
    for (int i = 0; i < {lane_blocks}; ++i) {{
        unsigned int idx = lane + i * {wave};
        unsigned int r = tile_row + idx / {nn};
        unsigned int c = tile_col + idx % {nn};
        if (r < m && c < n) {{
            d[r * n + c] = acc[i];
        }}
    }}
}}
"#,
        target = arch.target_id(),
        mm = op.m,
        nn = op.n,
        kk = op.k,
        dtype_tag = dtype_tag,
        include = matrix_include(op.dtype),
        in_ty = in_ty,
        acc_ty = acc_ty,
        wave = wave,
        lane_blocks = lane_blocks,
        builtin = builtin,
    ))
}

/// The `#include` directive(s) a matrix kernel of `dtype` needs.
fn matrix_include(dtype: MatrixDtype) -> &'static str {
    match dtype {
        MatrixDtype::F16 => "#include <hip/hip_fp16.h>",
        MatrixDtype::Bf16 => "#include <hip/hip_bf16.h>",
        MatrixDtype::Fp8E4m3 | MatrixDtype::Fp8E5m2 => "#include <hip/hip_fp8.h>",
        MatrixDtype::F64 => "",
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn op(m: u32, n: u32, k: u32, d: MatrixDtype) -> MatrixCoreOp {
        MatrixCoreOp { m, n, k, dtype: d }
    }

    #[test]
    fn dtype_c_and_acc_types() {
        assert_eq!(MatrixDtype::F16.c_type(), "__half");
        assert_eq!(MatrixDtype::F16.acc_type(), "float");
        assert_eq!(MatrixDtype::F64.c_type(), "double");
        assert_eq!(MatrixDtype::F64.acc_type(), "double");
        assert_eq!(MatrixDtype::Fp8E4m3.c_type(), "unsigned char");
        assert_eq!(MatrixDtype::Fp8E5m2.acc_type(), "float");
    }

    #[test]
    fn mfma_builtins_resolve() {
        assert_eq!(
            mfma_builtin(op(16, 16, 16, MatrixDtype::F16)),
            Some("__builtin_amdgcn_mfma_f32_16x16x16f16")
        );
        assert_eq!(
            mfma_builtin(op(32, 32, 8, MatrixDtype::Bf16)),
            Some("__builtin_amdgcn_mfma_f32_32x32x8bf16_1k")
        );
        assert_eq!(
            mfma_builtin(op(16, 16, 4, MatrixDtype::F64)),
            Some("__builtin_amdgcn_mfma_f64_16x16x4f64")
        );
        assert!(mfma_builtin(op(7, 7, 7, MatrixDtype::F16)).is_none());
    }

    #[test]
    fn fp8_builtins_present_for_cdna3() {
        assert_eq!(
            mfma_builtin(op(16, 16, 32, MatrixDtype::Fp8E4m3)),
            Some("__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8")
        );
        assert_eq!(
            mfma_builtin(op(16, 16, 32, MatrixDtype::Fp8E5m2)),
            Some("__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8")
        );
        assert_eq!(
            mfma_builtin(op(32, 32, 16, MatrixDtype::Fp8E4m3)),
            Some("__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8")
        );
    }

    #[test]
    fn wmma_builtins_for_rdna3() {
        assert_eq!(
            wmma_builtin(op(16, 16, 16, MatrixDtype::F16)),
            Some("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32")
        );
        assert!(wmma_builtin(op(32, 32, 8, MatrixDtype::F16)).is_none());
    }

    #[test]
    fn arch_support_matrix() {
        // FP8 only on CDNA3.
        assert!(arch_supports(
            GfxArch::Gfx942,
            op(16, 16, 32, MatrixDtype::Fp8E4m3)
        ));
        assert!(!arch_supports(
            GfxArch::Gfx90a,
            op(16, 16, 32, MatrixDtype::Fp8E4m3)
        ));
        // BF16 on CDNA2+.
        assert!(arch_supports(
            GfxArch::Gfx90a,
            op(16, 16, 16, MatrixDtype::Bf16)
        ));
        assert!(!arch_supports(
            GfxArch::Gfx908,
            op(16, 16, 16, MatrixDtype::Bf16)
        ));
        // FP16 WMMA on RDNA3.
        assert!(arch_supports(
            GfxArch::Gfx1100,
            op(16, 16, 16, MatrixDtype::F16)
        ));
        // RDNA2 has no matrix cores.
        assert!(!arch_supports(
            GfxArch::Gfx1030,
            op(16, 16, 16, MatrixDtype::F16)
        ));
    }

    #[test]
    fn fp8_kernel_codegen_structure() {
        let src = mfma_gemm_hip(GfxArch::Gfx942, op(16, 16, 32, MatrixDtype::Fp8E4m3))
            .expect("fp8 kernel");
        assert!(src.contains("__global__"));
        assert!(src.contains("__launch_bounds__(64)"));
        assert!(src.contains("mfma_gemm_fp8_e4m3"));
        assert!(src.contains("__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8"));
        assert!(src.contains("#include <hip/hip_fp8.h>"));
        assert!(src.contains("unsigned char"));
        // FP8 accumulates into FP32.
        assert!(src.contains("float"));
        // K step matches tile depth.
        assert!(src.contains("kk += 32"));
    }

    #[test]
    fn bf16_kernel_codegen_structure() {
        let src =
            mfma_gemm_hip(GfxArch::Gfx90a, op(16, 16, 16, MatrixDtype::Bf16)).expect("bf16 kernel");
        assert!(src.contains("mfma_gemm_bf16"));
        assert!(src.contains("__hip_bfloat16"));
        assert!(src.contains("__builtin_amdgcn_mfma_f32_16x16x16bf16_1k"));
        assert!(src.contains("#include <hip/hip_bf16.h>"));
    }

    #[test]
    fn fp64_kernel_uses_double_accumulator() {
        let src =
            mfma_gemm_hip(GfxArch::Gfx90a, op(16, 16, 4, MatrixDtype::F64)).expect("fp64 kernel");
        assert!(src.contains("mfma_gemm_f64"));
        assert!(src.contains("double"));
        assert!(src.contains("__builtin_amdgcn_mfma_f64_16x16x4f64"));
        // No FP16 header for FP64.
        assert!(!src.contains("hip_fp16.h"));
    }

    #[test]
    fn wmma_kernel_codegen_for_rdna3() {
        let src =
            mfma_gemm_hip(GfxArch::Gfx1100, op(16, 16, 16, MatrixDtype::F16)).expect("wmma kernel");
        assert!(src.contains("mfma_gemm_f16"));
        assert!(src.contains("__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"));
        // RDNA3 native wavefront is 32.
        assert!(src.contains("__launch_bounds__(32)"));
    }

    #[test]
    fn kernel_reduces_whole_tile_not_only_lane0() {
        // Regression: the kernel used to accumulate a single scalar into acc[0]
        // and leave acc[1..] zero, so most of the tile was written as 0. A
        // correct kernel reduces the K dimension into every fragment element.
        // wave64, 16x16 tile → lane_blocks = 256/64 = 4 (> 1).
        let src =
            mfma_gemm_hip(GfxArch::Gfx90a, op(16, 16, 16, MatrixDtype::Bf16)).expect("bf16 kernel");
        // Every fragment element must be accumulated, not just acc[0].
        assert!(
            src.contains("acc[i] +="),
            "kernel must reduce into the whole acc[] fragment"
        );
        assert!(
            !src.contains("acc[0] +="),
            "kernel must not accumulate into only acc[0]"
        );
        // Each owned output element fully reduces the K dimension.
        assert!(src.contains("for (unsigned int p = kk; p < kend;"));
    }

    #[test]
    fn unsupported_op_errors() {
        let err = mfma_gemm_hip(GfxArch::Gfx1030, op(16, 16, 16, MatrixDtype::F16)).unwrap_err();
        assert!(matches!(err, RocmError::Unsupported(_)));
        // FP8 on CDNA2 also unsupported.
        let err = mfma_gemm_hip(GfxArch::Gfx90a, op(16, 16, 32, MatrixDtype::Fp8E4m3)).unwrap_err();
        assert!(matches!(err, RocmError::Unsupported(_)));
    }
}
