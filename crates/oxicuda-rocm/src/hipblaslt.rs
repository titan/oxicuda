//! hipBLASLt interop: runtime loader + matmul / epilogue-fusion descriptors.
//!
//! hipBLASLt (`libhipblaslt.so`) is AMD's layout-flexible GEMM library with
//! fused epilogues (bias add, activation, scaling) for CDNA3 / RDNA3. This
//! module provides:
//!
//! - A graceful runtime loader ([`HipBlasLt::load`]) that probes for the
//!   `.so` via `libloading` and returns [`RocmError::LibraryNotFound`] when
//!   absent — mirroring [`crate::hiprtc::HipRtc`].
//! - Pure-Rust descriptor builders ([`MatmulDesc`], [`MatrixLayout`],
//!   [`Epilogue`]) that validate a fused GEMM request **without** a GPU, so the
//!   host-side configuration logic is fully CPU-testable.
//!
//! Actual matmul *execution* requires AMD ROCm hardware and is gated on the
//! library being present.

use crate::error::{RocmError, RocmResult};
use std::sync::Arc;

/// Candidate shared library names searched, in order.
#[allow(dead_code)]
const HIPBLASLT_CANDIDATES: &[&str] = &["libhipblaslt.so.0", "libhipblaslt.so"];

// ─── Data type ──────────────────────────────────────────────────────────────

/// hipBLASLt compute / data type tag (`hipblasltDatatype_t` subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtDtype {
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE half.
    F16,
    /// 16-bit brain float.
    Bf16,
    /// 8-bit OCP float E4M3.
    Fp8E4m3,
    /// 8-bit OCP float E5M2.
    Fp8E5m2,
}

impl LtDtype {
    /// Element size in bytes.
    pub fn size_bytes(self) -> usize {
        match self {
            LtDtype::F32 => 4,
            LtDtype::F16 | LtDtype::Bf16 => 2,
            LtDtype::Fp8E4m3 | LtDtype::Fp8E5m2 => 1,
        }
    }

    /// `true` for the 8-bit FP8 variants (CDNA3 only).
    pub fn is_fp8(self) -> bool {
        matches!(self, LtDtype::Fp8E4m3 | LtDtype::Fp8E5m2)
    }
}

// ─── Epilogue ───────────────────────────────────────────────────────────────

/// A fused GEMM epilogue (`hipblasLtEpilogue_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epilogue {
    /// No epilogue: `D = alpha*A*B + beta*C`.
    Default,
    /// Add a per-row bias vector after the matmul.
    Bias,
    /// Apply ReLU activation.
    Relu,
    /// Apply GELU activation.
    Gelu,
    /// Bias followed by ReLU.
    BiasRelu,
    /// Bias followed by GELU.
    BiasGelu,
}

impl Epilogue {
    /// `true` if this epilogue consumes a bias vector.
    pub fn needs_bias(self) -> bool {
        matches!(
            self,
            Epilogue::Bias | Epilogue::BiasRelu | Epilogue::BiasGelu
        )
    }

    /// `true` if this epilogue applies a non-linear activation.
    pub fn has_activation(self) -> bool {
        matches!(
            self,
            Epilogue::Relu | Epilogue::Gelu | Epilogue::BiasRelu | Epilogue::BiasGelu
        )
    }
}

// ─── MatrixLayout ───────────────────────────────────────────────────────────

/// A matrix layout descriptor (`hipblasLtMatrixLayout_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    /// Element data type.
    pub dtype: LtDtype,
    /// Number of rows.
    pub rows: u64,
    /// Number of columns.
    pub cols: u64,
    /// Leading dimension (stride between columns, column-major).
    pub ld: u64,
}

impl MatrixLayout {
    /// Construct a column-major layout with `ld = rows`.
    pub fn new(dtype: LtDtype, rows: u64, cols: u64) -> Self {
        Self {
            dtype,
            rows,
            cols,
            ld: rows,
        }
    }

    /// Override the leading dimension.
    pub fn with_ld(mut self, ld: u64) -> Self {
        self.ld = ld;
        self
    }

    /// Total byte footprint implied by `ld * cols * element_size`.
    pub fn byte_size(&self) -> u64 {
        self.ld
            .saturating_mul(self.cols)
            .saturating_mul(self.dtype.size_bytes() as u64)
    }

    /// Validate the layout (`ld >= rows`, non-zero extents).
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] on a degenerate layout.
    pub fn validate(&self) -> RocmResult<()> {
        if self.rows == 0 || self.cols == 0 {
            return Err(RocmError::InvalidArgument(
                "matrix layout extents must be non-zero".into(),
            ));
        }
        if self.ld < self.rows {
            return Err(RocmError::InvalidArgument(format!(
                "leading dimension {} is smaller than row count {}",
                self.ld, self.rows
            )));
        }
        Ok(())
    }
}

// ─── MatmulDesc ─────────────────────────────────────────────────────────────

/// A fused matmul descriptor (`hipblasLtMatmulDesc_t`): operand layouts, the
/// epilogue, and the accumulator/compute type.
#[derive(Debug, Clone)]
pub struct MatmulDesc {
    /// Layout of A.
    pub a: MatrixLayout,
    /// Layout of B.
    pub b: MatrixLayout,
    /// Layout of C / D (output).
    pub c: MatrixLayout,
    /// Fused epilogue.
    pub epilogue: Epilogue,
    /// Compute (accumulator) type.
    pub compute_type: LtDtype,
}

impl MatmulDesc {
    /// Construct a default-epilogue descriptor over the three layouts with an
    /// FP32 accumulator.
    pub fn new(a: MatrixLayout, b: MatrixLayout, c: MatrixLayout) -> Self {
        Self {
            a,
            b,
            c,
            epilogue: Epilogue::Default,
            compute_type: LtDtype::F32,
        }
    }

    /// Set the fused epilogue.
    pub fn with_epilogue(mut self, epilogue: Epilogue) -> Self {
        self.epilogue = epilogue;
        self
    }

    /// Set the compute (accumulator) type.
    pub fn with_compute_type(mut self, compute_type: LtDtype) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Validate operand-shape compatibility for `C[m,n] = A[m,k] * B[k,n]`,
    /// each layout's internal consistency, and that an FP8 input pairs with an
    /// FP32 accumulator (CDNA3 requirement).
    ///
    /// # Errors
    ///
    /// [`RocmError::InvalidArgument`] on any incompatibility.
    pub fn validate(&self) -> RocmResult<()> {
        self.a.validate()?;
        self.b.validate()?;
        self.c.validate()?;

        // A is [m, k], B is [k, n], C is [m, n].
        let (m, k) = (self.a.rows, self.a.cols);
        let (k_b, n) = (self.b.rows, self.b.cols);
        if k != k_b {
            return Err(RocmError::InvalidArgument(format!(
                "inner dimensions disagree: A has {k} cols, B has {k_b} rows"
            )));
        }
        if self.c.rows != m || self.c.cols != n {
            return Err(RocmError::InvalidArgument(format!(
                "C is [{}, {}] but A*B is [{m}, {n}]",
                self.c.rows, self.c.cols
            )));
        }
        // FP8 inputs must accumulate into FP32.
        if (self.a.dtype.is_fp8() || self.b.dtype.is_fp8()) && self.compute_type != LtDtype::F32 {
            return Err(RocmError::InvalidArgument(
                "FP8 matmul requires an FP32 compute type".into(),
            ));
        }
        Ok(())
    }

    /// `true` if this descriptor needs a bias vector supplied at launch.
    pub fn requires_bias_operand(&self) -> bool {
        self.epilogue.needs_bias()
    }
}

// ─── HipBlasLt loader ───────────────────────────────────────────────────────

/// Runtime-loaded hipBLASLt interface.
///
/// Created via [`HipBlasLt::load`]. All execution methods are gated on the
/// library being present; descriptor building is always available.
pub struct HipBlasLt {
    library_path: String,
    available: bool,
}

impl std::fmt::Debug for HipBlasLt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipBlasLt")
            .field("library_path", &self.library_path)
            .field("available", &self.available)
            .finish()
    }
}

impl HipBlasLt {
    /// Attempt to load `libhipblaslt.so` at runtime.
    ///
    /// On non-Linux platforms this always returns
    /// [`RocmError::UnsupportedPlatform`].
    pub fn load() -> RocmResult<Arc<Self>> {
        #[cfg(not(target_os = "linux"))]
        return Err(RocmError::UnsupportedPlatform);

        #[cfg(target_os = "linux")]
        Self::load_linux()
    }

    /// Return a stub that reports itself unavailable.
    pub fn stub() -> Arc<Self> {
        Arc::new(Self {
            library_path: String::new(),
            available: false,
        })
    }

    #[cfg(target_os = "linux")]
    fn load_linux() -> RocmResult<Arc<Self>> {
        for candidate in HIPBLASLT_CANDIDATES {
            // SAFETY: libloading::Library::new is safe with a well-formed name;
            // errors surface via the Result.
            if let Ok(lib) = unsafe { libloading::Library::new(*candidate) } {
                // Probe the create-handle symbol to confirm this is hipBLASLt.
                // SAFETY: the symbol name is a valid C string.
                let probe: Result<libloading::Symbol<unsafe extern "C" fn()>, _> =
                    unsafe { lib.get(b"hipblasLtCreate\0") };
                if probe.is_ok() {
                    drop(lib);
                    return Ok(Arc::new(Self {
                        library_path: candidate.to_string(),
                        available: true,
                    }));
                }
                drop(lib);
            }
        }
        Err(RocmError::LibraryNotFound("libhipblaslt.so".into()))
    }

    /// `true` if hipBLASLt was successfully loaded.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// The resolved library path (empty for stubs).
    pub fn library_path(&self) -> &str {
        &self.library_path
    }

    /// Plan a fused matmul: validate `desc` and confirm the runtime is present.
    ///
    /// On success the descriptor is realisable; the actual `hipblasLtMatmul`
    /// dispatch requires AMD hardware and is not performed here.
    ///
    /// # Errors
    ///
    /// - [`RocmError::LibraryNotFound`] when hipBLASLt is not loaded.
    /// - [`RocmError::InvalidArgument`] when the descriptor is invalid.
    pub fn plan_matmul(&self, desc: &MatmulDesc) -> RocmResult<()> {
        desc.validate()?;
        if !self.available {
            return Err(RocmError::LibraryNotFound(
                "hipBLASLt not available — install ROCm to dispatch fused GEMM".into(),
            ));
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(LtDtype::F32.size_bytes(), 4);
        assert_eq!(LtDtype::F16.size_bytes(), 2);
        assert_eq!(LtDtype::Bf16.size_bytes(), 2);
        assert_eq!(LtDtype::Fp8E4m3.size_bytes(), 1);
        assert!(LtDtype::Fp8E5m2.is_fp8());
        assert!(!LtDtype::F32.is_fp8());
    }

    #[test]
    fn epilogue_classification() {
        assert!(Epilogue::Bias.needs_bias());
        assert!(Epilogue::BiasGelu.needs_bias());
        assert!(!Epilogue::Relu.needs_bias());
        assert!(Epilogue::Gelu.has_activation());
        assert!(Epilogue::BiasRelu.has_activation());
        assert!(!Epilogue::Bias.has_activation());
        assert!(!Epilogue::Default.has_activation());
    }

    #[test]
    fn layout_byte_size_and_validate() {
        let l = MatrixLayout::new(LtDtype::F32, 128, 64);
        assert_eq!(l.byte_size(), 128 * 64 * 4);
        assert!(l.validate().is_ok());

        let bad = MatrixLayout::new(LtDtype::F32, 0, 64);
        assert!(bad.validate().is_err());

        let bad_ld = MatrixLayout::new(LtDtype::F32, 128, 64).with_ld(64);
        assert!(bad_ld.validate().is_err());
    }

    #[test]
    fn matmul_desc_shape_validation() {
        let a = MatrixLayout::new(LtDtype::F16, 64, 32); // [m=64, k=32]
        let b = MatrixLayout::new(LtDtype::F16, 32, 16); // [k=32, n=16]
        let c = MatrixLayout::new(LtDtype::F16, 64, 16); // [m=64, n=16]
        let desc = MatmulDesc::new(a, b, c);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn matmul_desc_inner_dim_mismatch() {
        let a = MatrixLayout::new(LtDtype::F16, 64, 32);
        let b = MatrixLayout::new(LtDtype::F16, 48, 16); // k=48 != 32
        let c = MatrixLayout::new(LtDtype::F16, 64, 16);
        let err = MatmulDesc::new(a, b, c).validate().unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn matmul_desc_output_shape_mismatch() {
        let a = MatrixLayout::new(LtDtype::F16, 64, 32);
        let b = MatrixLayout::new(LtDtype::F16, 32, 16);
        let c = MatrixLayout::new(LtDtype::F16, 64, 99); // n wrong
        let err = MatmulDesc::new(a, b, c).validate().unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn fp8_requires_fp32_accumulator() {
        let a = MatrixLayout::new(LtDtype::Fp8E4m3, 64, 32);
        let b = MatrixLayout::new(LtDtype::Fp8E4m3, 32, 16);
        let c = MatrixLayout::new(LtDtype::F16, 64, 16);
        // FP8 + FP16 compute → rejected.
        let bad = MatmulDesc::new(a, b, c).with_compute_type(LtDtype::F16);
        assert!(bad.validate().is_err());
        // FP8 + FP32 compute → accepted.
        let good = MatmulDesc::new(a, b, c).with_compute_type(LtDtype::F32);
        assert!(good.validate().is_ok());
    }

    #[test]
    fn bias_epilogue_flags_operand() {
        let a = MatrixLayout::new(LtDtype::F16, 8, 8);
        let b = MatrixLayout::new(LtDtype::F16, 8, 8);
        let c = MatrixLayout::new(LtDtype::F16, 8, 8);
        let desc = MatmulDesc::new(a, b, c).with_epilogue(Epilogue::BiasGelu);
        assert!(desc.requires_bias_operand());
        assert!(desc.epilogue.has_activation());
    }

    #[test]
    fn stub_reports_unavailable_and_blocks_dispatch() {
        let lt = HipBlasLt::stub();
        assert!(!lt.is_available());
        assert!(lt.library_path().is_empty());

        let a = MatrixLayout::new(LtDtype::F16, 8, 8);
        let b = MatrixLayout::new(LtDtype::F16, 8, 8);
        let c = MatrixLayout::new(LtDtype::F16, 8, 8);
        let desc = MatmulDesc::new(a, b, c);
        // Descriptor is valid, but no runtime → LibraryNotFound.
        let err = lt.plan_matmul(&desc).unwrap_err();
        assert!(matches!(err, RocmError::LibraryNotFound(_)));
    }

    #[test]
    fn plan_matmul_validates_before_runtime_check() {
        let lt = HipBlasLt::stub();
        // Invalid descriptor returns InvalidArgument even on a stub.
        let a = MatrixLayout::new(LtDtype::F16, 64, 32);
        let b = MatrixLayout::new(LtDtype::F16, 48, 16);
        let c = MatrixLayout::new(LtDtype::F16, 64, 16);
        let err = lt.plan_matmul(&MatmulDesc::new(a, b, c)).unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn load_returns_error_or_ok_without_panic() {
        match HipBlasLt::load() {
            Ok(lt) => {
                assert!(lt.is_available());
                assert!(!lt.library_path().is_empty());
            }
            Err(RocmError::LibraryNotFound(_)) | Err(RocmError::UnsupportedPlatform) => {}
            Err(other) => panic!("unexpected error from HipBlasLt::load: {other:?}"),
        }
    }

    #[test]
    fn debug_format_smoke() {
        let lt = HipBlasLt::stub();
        let s = format!("{lt:?}");
        assert!(s.contains("HipBlasLt"));
        assert!(s.contains("available"));
    }
}
