//! SIMT (non-Tensor-Core) GEMM helpers.
//!
//! When [`oxicuda_ptx::templates::gemm::GemmTemplate`] is not suitable (e.g. for very small matrices where
//! the overhead of tiled shared-memory GEMM exceeds the computation), this
//! module provides a [`SimtGemmBuilder`] that generates a minimal PTX kernel
//! via [`oxicuda_ptx::builder::KernelBuilder`].
//!
//! The generated kernel assigns one output element per thread: each thread
//! computes a single dot product along the K dimension. This is a fallback
//! for correctness; it is not performance-optimal for large matrices.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use super::precision;
use crate::error::{BlasError, BlasResult};
use crate::types::{FillMode, Transpose};

// ---------------------------------------------------------------------------
// SimtGemmBuilder
// ---------------------------------------------------------------------------

/// Builds a naive one-element-per-thread GEMM kernel in PTX.
///
/// This is the fallback path for small matrices (e.g. M*N < 1024) where
/// the tiled [`oxicuda_ptx::templates::gemm::GemmTemplate`] would waste most of its shared-memory tile.
pub struct SimtGemmBuilder {
    /// Target SM version for the PTX header.
    target: SmVersion,
    /// Input element precision.
    precision: PtxType,
    /// Accumulator precision.
    accumulator: PtxType,
    /// Transpose mode for A.
    trans_a: Transpose,
    /// Transpose mode for B.
    trans_b: Transpose,
    /// Optional triangle-write mask. `None` (or `Some(Full)`) writes every
    /// output element; `Some(Upper)`/`Some(Lower)` skip the store (leaving `C`
    /// untouched) for elements outside the requested triangle, which SYRK /
    /// SYR2K rely on to preserve the off-triangle.
    fill_mode: Option<FillMode>,
}

impl SimtGemmBuilder {
    /// Creates a new SIMT GEMM builder.
    pub fn new(
        target: SmVersion,
        precision: PtxType,
        accumulator: PtxType,
        trans_a: Transpose,
        trans_b: Transpose,
        fill_mode: Option<FillMode>,
    ) -> Self {
        Self {
            target,
            precision,
            accumulator,
            trans_a,
            trans_b,
            fill_mode,
        }
    }

    /// Returns the kernel function name.
    pub fn kernel_name(&self) -> String {
        let prec = self.precision.as_ptx_str().trim_start_matches('.');
        let acc = self.accumulator.as_ptx_str().trim_start_matches('.');
        let ta = trans_label(self.trans_a);
        let tb = trans_label(self.trans_b);
        let fill = fill_label(self.fill_mode);
        format!("simt_gemm_{prec}_{acc}_{ta}_{tb}_{fill}")
    }

    /// Generates the complete PTX module text.
    ///
    /// The kernel computes `C = alpha * op(A) * op(B) + beta * C` with one
    /// thread per output element. The block size is expected to be (16, 16)
    /// and the grid covers the output matrix.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::PtxGeneration`] if formatting fails or the
    /// precision is unsupported.
    pub fn generate(&self) -> BlasResult<String> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let acc_ty = self.accumulator.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        // When the input element precision differs from the accumulator
        // precision, loaded elements occupy a separate `%fin` bank (in the
        // `mem_ty` storage type) and are converted to/from the accumulator
        // precision with `cvt`; half↔f64 routes through the `%fc` f32 scratch.
        let needs_cvt = self.precision != self.accumulator;
        let mem_ty = precision::mem_type(self.precision);
        let needs_scratch = precision::needs_f32_scratch(self.precision, self.accumulator);
        let acc_zero = self.accumulator.zero_literal();

        let mut ptx = String::with_capacity(4096);

        // Header
        write_line(&mut ptx, &format!(".version {}", self.target.ptx_version()))?;
        write_line(&mut ptx, &format!(".target {}", self.target.as_ptx_str()))?;
        write_line(&mut ptx, ".address_size 64")?;
        write_line(&mut ptx, "")?;

        // Kernel entry
        write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
        write_line(&mut ptx, "    .param .u64 %param_a,")?;
        write_line(&mut ptx, "    .param .u64 %param_b,")?;
        write_line(&mut ptx, "    .param .u64 %param_c,")?;
        write_line(&mut ptx, "    .param .u32 %param_m,")?;
        write_line(&mut ptx, "    .param .u32 %param_n,")?;
        write_line(&mut ptx, "    .param .u32 %param_k,")?;
        write_line(&mut ptx, "    .param .u32 %param_lda,")?;
        write_line(&mut ptx, "    .param .u32 %param_ldb,")?;
        write_line(&mut ptx, "    .param .u32 %param_ldc,")?;
        write_line(&mut ptx, &format!("    .param {acc_ty} %param_alpha,"))?;
        write_line(&mut ptx, &format!("    .param {acc_ty} %param_beta"))?;
        write_line(&mut ptx, ")")?;
        write_line(&mut ptx, "{")?;

        // Registers. The value bank `%f` is declared in the accumulator
        // precision so every `mov`/`mul`/`fma.rn`/`ld.param` agrees with the
        // register type that `ptxas` validates against.
        write_line(&mut ptx, "    .reg .b32 %r<32>;")?;
        write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
        write_line(&mut ptx, &format!("    .reg {acc_ty} %f<16>;"))?;
        if needs_cvt {
            write_line(&mut ptx, &format!("    .reg {mem_ty} %fin<8>;"))?;
        }
        if needs_scratch {
            write_line(&mut ptx, "    .reg .f32 %fc<8>;")?;
        }
        write_line(&mut ptx, "    .reg .pred %p<4>;")?;
        write_line(&mut ptx, "")?;

        // Global indices: col = blockIdx.x * blockDim.x + threadIdx.x
        //                 row = blockIdx.y * blockDim.y + threadIdx.y
        write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
        write_line(&mut ptx, "    mov.u32 %r1, %tid.y;")?;
        write_line(&mut ptx, "    mov.u32 %r2, %ctaid.x;")?;
        write_line(&mut ptx, "    mov.u32 %r3, %ctaid.y;")?;
        write_line(&mut ptx, "    mov.u32 %r4, %ntid.x;")?;
        write_line(&mut ptx, "    mov.u32 %r5, %ntid.y;")?;
        write_line(&mut ptx, "    mad.lo.u32 %r6, %r2, %r4, %r0;  // col")?;
        write_line(&mut ptx, "    mad.lo.u32 %r7, %r3, %r5, %r1;  // row")?;
        write_line(&mut ptx, "")?;

        // Load parameters
        write_line(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
        write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
        write_line(&mut ptx, "    ld.param.u64 %rd2, [%param_c];")?;
        write_line(&mut ptx, "    ld.param.u32 %r8, [%param_m];")?;
        write_line(&mut ptx, "    ld.param.u32 %r9, [%param_n];")?;
        write_line(&mut ptx, "    ld.param.u32 %r10, [%param_k];")?;
        write_line(&mut ptx, "    ld.param.u32 %r20, [%param_lda];")?;
        write_line(&mut ptx, "    ld.param.u32 %r21, [%param_ldb];")?;
        write_line(&mut ptx, "    ld.param.u32 %r22, [%param_ldc];")?;
        write_line(
            &mut ptx,
            &format!("    ld.param{acc_ty} %f8, [%param_alpha];"),
        )?;
        write_line(
            &mut ptx,
            &format!("    ld.param{acc_ty} %f9, [%param_beta];"),
        )?;
        write_line(&mut ptx, "")?;

        // Bounds check
        write_line(&mut ptx, "    setp.ge.u32 %p0, %r7, %r8;")?;
        write_line(&mut ptx, "    setp.ge.u32 %p1, %r6, %r9;")?;
        write_line(&mut ptx, "    @%p0 bra $SIMT_DONE;")?;
        write_line(&mut ptx, "    @%p1 bra $SIMT_DONE;")?;
        write_line(&mut ptx, "")?;

        // Accumulator init
        write_line(&mut ptx, &format!("    mov{acc_ty} %f0, {acc_zero};"))?;
        write_line(&mut ptx, "    mov.u32 %r11, 0;")?;
        write_line(&mut ptx, "")?;

        // K-loop: a single dot product
        write_line(&mut ptx, "$SIMT_K_LOOP:")?;
        write_line(&mut ptx, "    setp.ge.u32 %p2, %r11, %r10;")?;
        write_line(&mut ptx, "    @%p2 bra $SIMT_K_DONE;")?;

        // Address calculation depends on transpose mode.
        // A element: trans_a == NoTrans => A[row, k] = A[row * lda + k]
        //            trans_a == Trans   => A[k, row] = A[k * lda + row]
        let (a_row_reg, a_col_reg) = match self.trans_a {
            Transpose::NoTrans => ("%r7", "%r11"),
            Transpose::Trans | Transpose::ConjTrans => ("%r11", "%r7"),
        };
        write_line(
            &mut ptx,
            &format!("    mad.lo.u32 %r12, {a_row_reg}, %r20, {a_col_reg};"),
        )?;
        write_line(&mut ptx, "    cvt.u64.u32 %rd3, %r12;")?;
        write_line(
            &mut ptx,
            &format!("    mul.lo.u64 %rd3, %rd3, {byte_size};"),
        )?;
        write_line(&mut ptx, "    add.u64 %rd4, %rd0, %rd3;")?;
        if needs_cvt {
            write_line(&mut ptx, &format!("    ld.global{mem_ty} %fin1, [%rd4];"))?;
            write_line(
                &mut ptx,
                &precision::convert_to_acc(
                    self.precision,
                    self.accumulator,
                    "%fin1",
                    "%fc1",
                    "%f1",
                ),
            )?;
        } else {
            write_line(&mut ptx, &format!("    ld.global{ty} %f1, [%rd4];"))?;
        }

        // B element: trans_b == NoTrans => B[k, col] = B[k * ldb + col]
        //            trans_b == Trans   => B[col, k] = B[col * ldb + k]
        let (b_row_reg, b_col_reg) = match self.trans_b {
            Transpose::NoTrans => ("%r11", "%r6"),
            Transpose::Trans | Transpose::ConjTrans => ("%r6", "%r11"),
        };
        write_line(
            &mut ptx,
            &format!("    mad.lo.u32 %r13, {b_row_reg}, %r21, {b_col_reg};"),
        )?;
        write_line(&mut ptx, "    cvt.u64.u32 %rd5, %r13;")?;
        write_line(
            &mut ptx,
            &format!("    mul.lo.u64 %rd5, %rd5, {byte_size};"),
        )?;
        write_line(&mut ptx, "    add.u64 %rd6, %rd1, %rd5;")?;
        if needs_cvt {
            write_line(&mut ptx, &format!("    ld.global{mem_ty} %fin2, [%rd6];"))?;
            write_line(
                &mut ptx,
                &precision::convert_to_acc(
                    self.precision,
                    self.accumulator,
                    "%fin2",
                    "%fc2",
                    "%f2",
                ),
            )?;
        } else {
            write_line(&mut ptx, &format!("    ld.global{ty} %f2, [%rd6];"))?;
        }

        // FMA
        write_line(&mut ptx, &format!("    fma.rn{acc_ty} %f0, %f1, %f2, %f0;"))?;
        write_line(&mut ptx, "    add.u32 %r11, %r11, 1;")?;
        write_line(&mut ptx, "    bra $SIMT_K_LOOP;")?;
        write_line(&mut ptx, "$SIMT_K_DONE:")?;
        write_line(&mut ptx, "")?;

        // Triangle-write mask (SYRK / SYR2K). Skip the entire epilogue —
        // including the beta*C read — for output elements outside the requested
        // triangle, so C is left byte-for-byte unchanged there. row=%r7,
        // col=%r6.
        match self.fill_mode {
            Some(FillMode::Upper) => {
                // Store only when col >= row → skip when col < row.
                write_line(&mut ptx, "    setp.lt.u32 %p3, %r6, %r7;")?;
                write_line(&mut ptx, "    @%p3 bra $SIMT_DONE;")?;
                write_line(&mut ptx, "")?;
            }
            Some(FillMode::Lower) => {
                // Store only when row >= col → skip when row < col.
                write_line(&mut ptx, "    setp.lt.u32 %p3, %r7, %r6;")?;
                write_line(&mut ptx, "    @%p3 bra $SIMT_DONE;")?;
                write_line(&mut ptx, "")?;
            }
            None | Some(FillMode::Full) => {}
        }

        // Epilogue: C[row, col] = alpha * acc + beta * C_old
        write_line(&mut ptx, "    mad.lo.u32 %r14, %r7, %r22, %r6;")?;
        write_line(&mut ptx, "    cvt.u64.u32 %rd7, %r14;")?;
        write_line(
            &mut ptx,
            &format!("    mul.lo.u64 %rd7, %rd7, {byte_size};"),
        )?;
        write_line(&mut ptx, "    add.u64 %rd8, %rd2, %rd7;")?;
        if needs_cvt {
            write_line(&mut ptx, &format!("    ld.global{mem_ty} %fin3, [%rd8];"))?;
            write_line(
                &mut ptx,
                &precision::convert_to_acc(
                    self.precision,
                    self.accumulator,
                    "%fin3",
                    "%fc3",
                    "%f3",
                ),
            )?;
        } else {
            write_line(&mut ptx, &format!("    ld.global{ty} %f3, [%rd8];"))?;
        }
        write_line(&mut ptx, &format!("    mul{acc_ty} %f0, %f0, %f8;"))?;
        write_line(&mut ptx, &format!("    fma.rn{acc_ty} %f0, %f9, %f3, %f0;"))?;
        if needs_cvt {
            write_line(
                &mut ptx,
                &precision::convert_from_acc(
                    self.precision,
                    self.accumulator,
                    "%f0",
                    "%fc0",
                    "%fin0",
                ),
            )?;
            write_line(&mut ptx, &format!("    st.global{mem_ty} [%rd8], %fin0;"))?;
        } else {
            write_line(&mut ptx, &format!("    st.global{ty} [%rd8], %f0;"))?;
        }
        write_line(&mut ptx, "")?;

        write_line(&mut ptx, "$SIMT_DONE:")?;
        write_line(&mut ptx, "    ret;")?;
        write_line(&mut ptx, "}")?;

        Ok(ptx)
    }

    /// Validates the builder parameters.
    fn validate(&self) -> BlasResult<()> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(BlasError::PtxGeneration(format!(
                "SIMT GEMM requires F16/BF16/F32/F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if !matches!(self.accumulator, PtxType::F32 | PtxType::F64) {
            return Err(BlasError::PtxGeneration(format!(
                "SIMT GEMM accumulator must be F32/F64, got {}",
                self.accumulator.as_ptx_str()
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Short label for a transpose mode, used in kernel names.
fn trans_label(t: Transpose) -> &'static str {
    match t {
        Transpose::NoTrans => "nn",
        Transpose::Trans => "tt",
        Transpose::ConjTrans => "ct",
    }
}

/// Short label for the triangle-write mask, used in kernel names so masked and
/// unmasked variants get distinct module symbols and cache entries.
fn fill_label(f: Option<FillMode>) -> &'static str {
    match f {
        None | Some(FillMode::Full) => "full",
        Some(FillMode::Upper) => "up",
        Some(FillMode::Lower) => "lo",
    }
}

/// Writes a line to the PTX string, mapping fmt errors to `BlasError`.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_simt_f32_nn() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::F32,
            PtxType::F32,
            Transpose::NoTrans,
            Transpose::NoTrans,
            None,
        );
        let ptx = builder.generate().expect("SIMT GEMM generation failed");
        assert!(ptx.contains(".entry simt_gemm_"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("$SIMT_K_LOOP"));
    }

    #[test]
    fn generate_simt_f32_tn() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::F32,
            PtxType::F32,
            Transpose::Trans,
            Transpose::NoTrans,
            None,
        );
        let ptx = builder.generate().expect("SIMT GEMM TN generation failed");
        assert!(ptx.contains("simt_gemm_f32_f32_tt_nn"));
    }

    #[test]
    fn kernel_name_format() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm75,
            PtxType::F64,
            PtxType::F64,
            Transpose::NoTrans,
            Transpose::Trans,
            None,
        );
        assert_eq!(builder.kernel_name(), "simt_gemm_f64_f64_nn_tt_full");
    }

    #[test]
    fn invalid_precision() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::U32,
            PtxType::F32,
            Transpose::NoTrans,
            Transpose::NoTrans,
            None,
        );
        assert!(builder.generate().is_err());
    }

    #[test]
    fn masked_upper_emits_predicated_store_skip() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::F32,
            PtxType::F32,
            Transpose::NoTrans,
            Transpose::Trans,
            Some(FillMode::Upper),
        );
        let ptx = builder
            .generate()
            .expect("masked SIMT GEMM generation failed");
        // Upper triangle: skip the store when col (%r6) < row (%r7).
        assert!(ptx.contains("simt_gemm_f32_f32_nn_tt_up"));
        assert!(ptx.contains("setp.lt.u32 %p3, %r6, %r7;"));
        assert!(ptx.contains("@%p3 bra $SIMT_DONE;"));
    }

    #[test]
    fn masked_lower_emits_predicated_store_skip() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::F64,
            PtxType::F64,
            Transpose::NoTrans,
            Transpose::Trans,
            Some(FillMode::Lower),
        );
        let ptx = builder
            .generate()
            .expect("masked SIMT GEMM generation failed");
        // Lower triangle: skip the store when row (%r7) < col (%r6).
        assert!(ptx.contains("simt_gemm_f64_f64_nn_tt_lo"));
        assert!(ptx.contains("setp.lt.u32 %p3, %r7, %r6;"));
        assert!(ptx.contains("@%p3 bra $SIMT_DONE;"));
    }

    #[test]
    fn unmasked_has_no_triangle_predicate() {
        let builder = SimtGemmBuilder::new(
            SmVersion::Sm80,
            PtxType::F32,
            PtxType::F32,
            Transpose::NoTrans,
            Transpose::Trans,
            None,
        );
        let ptx = builder.generate().expect("SIMT GEMM generation failed");
        assert!(!ptx.contains("%p3"));
    }
}
