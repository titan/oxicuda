//! Broadcast **bias-add** kernel template: `out[i, j] = in[i, j] + bias[j]`.
//!
//! Adds a length-`n` bias vector to every row of a row-major `m x n` matrix.
//! This is the standard "add the linear layer's bias" broadcast used after a
//! GEMM in a transformer feed-forward / projection: the GEMM produces an
//! `[m, n]` activation, and each output column `j` gets `bias[j]` added across
//! all `m` rows.
//!
//! # Algorithm
//!
//! The kernel is a flat one-thread-per-element map over the `m * n` elements.
//! For global thread index `tid < m * n`:
//!
//! ```text
//! col        = tid % n          // column within the row
//! out[tid]   = in[tid] + bias[col]
//! ```
//!
//! Row-major storage means the flat element index `tid` already addresses
//! `in`/`out` directly; only the bias needs the `tid % n` column reduction.
//! Threads with `tid >= m * n` exit immediately (bounds guard).
//!
//! # Parallelization
//!
//! One thread per output element, laid out over a 1-D grid (the host sizes the
//! grid to cover `m * n` threads with a 256-wide block). This mirrors the plain
//! elementwise templates rather than the row-sequential softmax templates: the
//! per-element work is uniform, so a flat map maximises occupancy.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::bias_add::BiasAddTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = BiasAddTemplate {
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("bias_add_f32"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating a broadcast bias-add PTX kernel.
///
/// Each launched thread processes exactly one matrix element, adding the bias
/// entry for that element's column.
///
/// The generated kernel signature is:
///
/// ```text
/// kernel(.u64 input, .u64 bias, .u64 output, .u32 m, .u32 n)
/// ```
///
/// where `input` / `output` point to row-major `m x n` matrices and `bias`
/// points to a length-`n` vector.
pub struct BiasAddTemplate {
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl BiasAddTemplate {
    /// Returns the kernel function name (e.g. `bias_add_f32`).
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("bias_add_{type_str}")
    }

    /// Validates template parameters.
    ///
    /// Only `F32` and `F64` are supported. The bias-add itself is a single
    /// `add`, which *is* defined for the half types — but the surrounding
    /// `oxicuda-blas` host wiring (and the `trustformers` resident op that
    /// drives it) operate on `f32`/`f64` device buffers, so the template is
    /// intentionally restricted to those to mirror the `causal_softmax`
    /// precedent and avoid silently emitting an unused half-precision path.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "bias_add requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Generates the complete PTX module text for the bias-add kernel.
    ///
    /// Kernel parameters:
    /// - `input`: pointer to the `m x n` input matrix (row-major)
    /// - `bias`: pointer to the length-`n` bias vector
    /// - `output`: pointer to the `m x n` output matrix (row-major)
    /// - `m`: number of rows
    /// - `n`: number of columns (and the bias length)
    ///
    /// Computes `output[i, j] = input[i, j] + bias[j]` for every element.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision is not a supported float type.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();

        let mut ptx = String::with_capacity(2048);

        // -- Header -----------------------------------------------------------
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_bias,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_m,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // -- Register declarations -------------------------------------------
        // r0 = tid.x, r1 = ctaid.x, r2 = ntid.x
        // r3 = tid (global thread id = flat element index)
        // r4 = m, r5 = n, r6 = total (= m*n), r7 = col (= tid % n)
        // r8 = scratch (tid / n)
        // rd0 = input base, rd1 = bias base, rd2 = output base
        // rd3 = element byte offset, rd4 = element addr, rd5 = bias byte offset, rd6 = bias addr
        // f0 = input value, f1 = bias value, f2 = result
        writeln!(ptx, "    .reg .b32 %r<9>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<7>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<3>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<2>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Thread -> flat element index ------------------------------------
        writeln!(ptx, "    // tid = blockIdx.x * blockDim.x + threadIdx.x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Bounds check: tid < m * n ---------------------------------------
        writeln!(ptx, "    ld.param.u32 %r4, [%param_m];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r6, %r4, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $BA_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Column index: col = tid % n -------------------------------------
        // col = tid - (tid / n) * n.
        writeln!(ptx, "    // col = tid % n").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    div.u32 %r8, %r3, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r8, %r8, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub.u32 %r7, %r3, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Load input[tid] -------------------------------------------------
        // Row-major: the flat element index tid addresses input/output directly.
        writeln!(ptx, "    // load input[tid]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_bias];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Load bias[col] --------------------------------------------------
        writeln!(ptx, "    // load bias[col]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- out[tid] = input[tid] + bias[col] -------------------------------
        writeln!(ptx, "    // out[tid] = input[tid] + bias[col]")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f2, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd4], %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Exit -------------------------------------------------------------
        writeln!(ptx, "$BA_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_name_format() {
        let t = BiasAddTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        assert_eq!(t.kernel_name(), "bias_add_f32");

        let t64 = BiasAddTemplate {
            precision: PtxType::F64,
            target: SmVersion::Sm80,
        };
        assert_eq!(t64.kernel_name(), "bias_add_f64");
    }

    #[test]
    fn generates_f32_kernel() {
        let t = BiasAddTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = t.generate().expect("should generate bias_add f32");
        // Entry point present.
        assert!(ptx.contains(".visible .entry bias_add_f32("));
        // Five parameters: input, bias, output, m, n.
        assert!(ptx.contains("%param_input"));
        assert!(ptx.contains("%param_bias"));
        assert!(ptx.contains("%param_output"));
        assert!(ptx.contains("%param_m"));
        assert!(ptx.contains("%param_n"));
        // The broadcast: a single f32 add of input and bias.
        assert!(ptx.contains("add.f32 %f2, %f0, %f1;"));
        // Column reduction (tid % n) is implemented via div + mul + sub.
        assert!(ptx.contains("div.u32"));
        assert!(ptx.contains("sub.u32 %r7"));
        // f32 elements are 4 bytes.
        assert!(ptx.contains("mul.lo.u64 %rd3, %rd3, 4;"));
        // Loads/stores go through global memory.
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn generates_f64_kernel() {
        let t = BiasAddTemplate {
            precision: PtxType::F64,
            target: SmVersion::Sm90,
        };
        let ptx = t.generate().expect("should generate bias_add f64");
        assert!(ptx.contains("bias_add_f64"));
        assert!(ptx.contains("add.f64 %f2, %f0, %f1;"));
        // f64 elements are 8 bytes.
        assert!(ptx.contains("mul.lo.u64 %rd3, %rd3, 8;"));
    }

    #[test]
    fn rejects_integer_precision() {
        let t = BiasAddTemplate {
            precision: PtxType::S32,
            target: SmVersion::Sm80,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn rejects_half_precision() {
        let t = BiasAddTemplate {
            precision: PtxType::F16,
            target: SmVersion::Sm80,
        };
        assert!(t.generate().is_err());
    }

    /// The bounds guard must compare the flat thread id against `m * n` and the
    /// store must reuse the element byte offset (rd3) so input and output share
    /// the same row-major addressing.
    #[test]
    fn guards_bounds_and_reuses_offset() {
        let t = BiasAddTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = t.generate().expect("generate");
        // total = m * n, then tid >= total exits.
        assert!(ptx.contains("mul.lo.u32 %r6, %r4, %r5;"));
        assert!(ptx.contains("setp.ge.u32 %p0, %r3, %r6;"));
        assert!(ptx.contains("@%p0 bra $BA_DONE;"));
        // Output address reuses the input element offset rd3.
        assert!(ptx.contains("add.u64 %rd4, %rd2, %rd3;"));
    }
}
