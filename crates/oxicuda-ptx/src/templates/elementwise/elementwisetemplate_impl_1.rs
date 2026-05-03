//! Secondary `ElementwiseTemplate` method implementations.
//!
//! Contains additional `ElementwiseTemplate` inherent methods that did
//! not fit in the primary impl module: `generate_xor`, `generate_fused_scale_add`,
//! and `generate_fill`.
//!
//! Refactored with [SplitRS](https://github.com/cool-japan/splitrs).

use crate::builder::KernelBuilder;
use crate::error::PtxGenError;
use crate::ir::PtxType;

use super::elementwisetemplate_type::ElementwiseTemplate;
use super::functions::{float_two_literal, scalar_param_type};

impl ElementwiseTemplate {
    /// Generates a fuzzy XOR kernel: `c[i] = a[i] + b[i] - 2*a[i]*b[i]`.
    pub(super) fn generate_xor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let two_lit = float_two_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         add{ty} %f_s, %f_a, %f_b;\n    \
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         mul{ty} %f_t2, %f_t, {two_lit};\n    \
                         sub{ty} %f_c, %f_s, %f_t2;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a fused scale-add kernel: `c[i] = alpha * a[i] + beta * b[i]`.
    pub(super) fn generate_fused_scale_add(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("alpha", scalar_ty)
            .param("beta", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_alpha, [%param_alpha];\n    \
                         ld.param{ty} %f_beta, [%param_beta];\n    \
                         ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         mul{ty} %f_aa, %f_alpha, %f_a;\n    \
                         mul{ty} %f_bb, %f_beta, %f_b;\n    \
                         add{ty} %f_y, %f_aa, %f_bb;\n    \
                         st.global{ty} [%rd_c], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a fill kernel: `dst[i] = value` for all `i < n`.
    ///
    /// The scalar `value` is read from a kernel parameter (not from a source buffer),
    /// which means every output element receives the same constant. This mirrors the
    /// `generate_scale()` pattern for loading a scalar kernel parameter via `ld.param`.
    pub(super) fn generate_fill(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("dst_ptr", PtxType::U64)
            .param("value", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let dst_ptr = b.load_param_u64("dst_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_dst, {dst_ptr}, %rd_off;\n    \
                         ld.param{ty} %f_val, [%param_value];\n    \
                         st.global{ty} [%rd_dst], %f_val;"
                    ));
                });
                b.ret();
            })
            .build()
    }
}
