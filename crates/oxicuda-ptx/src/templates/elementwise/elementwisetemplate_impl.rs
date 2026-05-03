//! Primary `ElementwiseTemplate` method implementations.
//!
//! Contains the bulk of `ElementwiseTemplate` inherent methods:
//! constructor, public entry points (`generate`, `kernel_name`),
//! validation, and the per-op generators for arithmetic, activations,
//! unary math, comparisons, fuzzy logic, and most fused kernels.
//!
//! Refactored with [SplitRS](https://github.com/cool-japan/splitrs).

use crate::arch::SmVersion;
use crate::builder::KernelBuilder;
use crate::error::PtxGenError;
use crate::ir::PtxType;

use super::elementwisetemplate_type::ElementwiseTemplate;
use super::functions::{float_one_literal, float_zero_literal, scalar_param_type};
use super::types::ElementwiseOp;

impl ElementwiseTemplate {
    /// Creates a new elementwise template with the given parameters.
    #[must_use]
    pub const fn new(op: ElementwiseOp, precision: PtxType, target: SmVersion) -> Self {
        Self {
            op,
            precision,
            target,
        }
    }
    /// Returns the kernel function name derived from the operation and precision.
    ///
    /// The name follows the pattern `elementwise_{op}_{type}`, for example
    /// `elementwise_add_f32` or `elementwise_relu_f16`.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("elementwise_{}_{}", self.op.as_str(), type_str)
    }
    /// Generates the complete PTX module text for this elementwise operation.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision type is unsupported for the
    /// requested operation or if PTX text generation fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate_precision()?;
        match self.op {
            ElementwiseOp::Add => self.generate_binary_arith("add"),
            ElementwiseOp::Sub => self.generate_binary_arith("sub"),
            ElementwiseOp::Mul => self.generate_binary_arith("mul"),
            ElementwiseOp::Div => self.generate_div(),
            ElementwiseOp::Relu => self.generate_relu(),
            ElementwiseOp::Gelu => self.generate_gelu(),
            ElementwiseOp::Sigmoid => self.generate_sigmoid(),
            ElementwiseOp::Silu => self.generate_silu(),
            ElementwiseOp::Tanh => self.generate_tanh(),
            ElementwiseOp::Neg => self.generate_unary("neg"),
            ElementwiseOp::Abs => self.generate_unary("abs"),
            ElementwiseOp::Sqrt => self.generate_sqrt(),
            ElementwiseOp::Rsqrt => self.generate_rsqrt(),
            ElementwiseOp::Exp => self.generate_exp(),
            ElementwiseOp::Log => self.generate_log(),
            ElementwiseOp::Ceil => self.generate_ceil(),
            ElementwiseOp::Floor => self.generate_floor(),
            ElementwiseOp::HardSigmoid => self.generate_hard_sigmoid(),
            ElementwiseOp::HardSwish => self.generate_hard_swish(),
            ElementwiseOp::Softplus => self.generate_softplus(),
            ElementwiseOp::LeakyRelu => self.generate_leaky_relu(),
            ElementwiseOp::OneMinus => self.generate_one_minus(),
            ElementwiseOp::Scale => self.generate_scale(),
            ElementwiseOp::AddScalar => self.generate_add_scalar(),
            ElementwiseOp::FusedAddRelu => self.generate_fused_add_relu(),
            ElementwiseOp::FusedScaleAdd => self.generate_fused_scale_add(),
            ElementwiseOp::Pow => self.generate_pow(),
            ElementwiseOp::Min => self.generate_binary_minmax("min"),
            ElementwiseOp::Max | ElementwiseOp::OrMax => self.generate_binary_minmax("max"),
            ElementwiseOp::CmpEq => self.generate_binary_cmp("eq"),
            ElementwiseOp::CmpNe => self.generate_binary_cmp("ne"),
            ElementwiseOp::CmpLt => self.generate_binary_cmp("lt"),
            ElementwiseOp::CmpGt => self.generate_binary_cmp("gt"),
            ElementwiseOp::CmpLe => self.generate_binary_cmp("le"),
            ElementwiseOp::CmpGe => self.generate_binary_cmp("ge"),
            ElementwiseOp::OrProbSum => self.generate_or_prob_sum(),
            ElementwiseOp::Nand => self.generate_nand(),
            ElementwiseOp::Nor => self.generate_nor(),
            ElementwiseOp::Xor => self.generate_xor(),
            ElementwiseOp::Fill => self.generate_fill(),
        }
    }
    /// Validates that the precision type is a supported floating-point type.
    fn validate_precision(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "elementwise operations require F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }
    /// Returns the PTX type suffix string (e.g., `.f32`).
    pub(super) const fn ty_str(&self) -> &'static str {
        self.precision.as_ptx_str()
    }
    /// Generates a binary arithmetic kernel (add, sub, mul).
    ///
    /// Kernel signature: `(a_ptr: u64, b_ptr: u64, c_ptr: u64, n: u32)`
    fn generate_binary_arith(&self, op_name: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let op_name = op_name.to_string();
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
                         {op_name}{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a division kernel with appropriate rounding.
    fn generate_div(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
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
                         div.rn{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a `ReLU` kernel: `max(0, x)`.
    fn generate_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         max{ty} %f_y, %f_x, {zero_lit};\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a sigmoid kernel: `1 / (1 + exp(-x))`.
    ///
    /// Uses `ex2.approx.f32` with a log2(e) scaling factor for the exponential,
    /// then `rcp.approx.f32` for the reciprocal.
    fn generate_sigmoid(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         neg{ty} %f_neg, %f_x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_y, %f_denom;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a GELU kernel using the tanh approximation.
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// Since PTX does not have a native tanh, this uses the identity:
    /// tanh(a) = 2 * sigmoid(2a) - 1 = (2 / (1 + exp(-2a))) - 1
    fn generate_gelu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_x3, %f_x, %f_x;\n    \
                         mul{ty} %f_x3, %f_x3, %f_x;\n    \
                         mul{ty} %f_x3, %f_x3, 0f3D372713;\n    \
                         add{ty} %f_inner, %f_x, %f_x3;\n    \
                         mul{ty} %f_inner, %f_inner, 0f3F4C422A;\n    \
                         mul{ty} %f_2a, %f_inner, 0f40000000;\n    \
                         neg{ty} %f_neg2a, %f_2a;\n    \
                         mul{ty} %f_neg2a, %f_neg2a, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg2a;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_sig, %f_sig, 0f40000000;\n    \
                         sub{ty} %f_tanh, %f_sig, 0f3F800000;\n    \
                         add{ty} %f_tanh, %f_tanh, 0f3F800000;\n    \
                         mul{ty} %f_y, 0f3F000000, %f_x;\n    \
                         mul{ty} %f_y, %f_y, %f_tanh;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a `SiLU` kernel: `x * sigmoid(x)`.
    fn generate_silu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         neg{ty} %f_neg, %f_x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_y, %f_x, %f_sig;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a tanh kernel using `tanh(x) = 2 * sigmoid(2x) - 1`.
    fn generate_tanh(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_2x, %f_x, 0f40000000;\n    \
                         neg{ty} %f_neg, %f_2x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_y, %f_sig, 0f40000000;\n    \
                         sub{ty} %f_y, %f_y, 0f3F800000;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a unary operation kernel (neg, abs).
    fn generate_unary(&self, op_name: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let op_name = op_name.to_string();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         {op_name}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a sqrt kernel with rounding.
    fn generate_sqrt(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         sqrt.rn{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates an rsqrt (reciprocal square root) kernel.
    fn generate_rsqrt(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         rsqrt.approx{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates an exp kernel using base-2 exponentiation: `exp(x) = 2^(x * log2(e))`.
    fn generate_exp(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_x2, %f_x, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_y, %f_x2;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a natural log kernel using base-2 logarithm: `ln(x) = lg2(x) / lg2(e)`.
    fn generate_log(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         lg2.approx{ty} %f_lg, %f_x;\n    \
                         mul{ty} %f_y, %f_lg, 0f3F317218;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a ceil kernel: `b[i] = ceil(a[i])`.
    ///
    /// Uses `cvt.rpi` (round-to-positive-infinity) for ceiling.
    fn generate_ceil(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         cvt.rpi{ty}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a floor kernel: `b[i] = floor(a[i])`.
    ///
    /// Uses `cvt.rmi` (round-to-minus-infinity) for floor.
    fn generate_floor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         cvt.rmi{ty}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a hard-sigmoid kernel: `max(0, min(1, alpha*x + beta))`.
    ///
    /// Uses ONNX default constants: alpha=0.2 (0f3E4CCCCD), beta=0.5 (0f3F000000).
    fn generate_hard_sigmoid(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_ax, %f_x, 0f3E4CCCCD;\n    \
                         add{ty} %f_lin, %f_ax, 0f3F000000;\n    \
                         min{ty} %f_clip, %f_lin, 0f3F800000;\n    \
                         max{ty} %f_y, %f_clip, {zero_lit};\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a hard-swish kernel: `x * max(0, min(6, x+3)) / 6`.
    ///
    /// IEEE 754 hex: 3.0=0f40400000, 6.0=0f40C00000, 1/6=0f3E2AAAAB.
    fn generate_hard_swish(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         add{ty} %f_xp3, %f_x, 0f40400000;\n    \
                         min{ty} %f_clip, %f_xp3, 0f40C00000;\n    \
                         max{ty} %f_clip, %f_clip, {zero_lit};\n    \
                         mul{ty} %f_div, %f_clip, 0f3E2AAAAB;\n    \
                         mul{ty} %f_y, %f_x, %f_div;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a softplus kernel: `ln(1 + exp(x))`.
    ///
    /// Uses exp(x) = 2^(x * log2(e)) and ln(z) = lg2(z) * ln(2).
    fn generate_softplus(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_xe, %f_x, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_xe;\n    \
                         add{ty} %f_sum, %f_exp, 0f3F800000;\n    \
                         lg2.approx{ty} %f_lg, %f_sum;\n    \
                         mul{ty} %f_y, %f_lg, 0f3F317218;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a leaky-relu kernel: `x >= 0 ? x : alpha*x` (alpha=0.01).
    ///
    /// IEEE 754 hex: 0.01 = 0f3C23D70A.
    fn generate_leaky_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_leak, %f_x, 0f3C23D70A;\n    \
                         setp.ge{ty} %p_ge, %f_x, {zero_lit};\n    \
                         selp{ty} %f_y, %f_x, %f_leak, %p_ge;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a scale kernel: `b[i] = alpha * a[i]`.
    fn generate_scale(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("alpha", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_alpha, [%param_alpha];\n    \
                         ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_y, %f_alpha, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates an add-scalar kernel: `b[i] = a[i] + scalar`.
    fn generate_add_scalar(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("scalar", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_s, [%param_scalar];\n    \
                         ld.global{ty} %f_x, [%rd_a];\n    \
                         add{ty} %f_y, %f_x, %f_s;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a fused add-relu kernel: `c[i] = max(0, a[i] + b[i])`.
    fn generate_fused_add_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);
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
                         add{ty} %f_sum, %f_a, %f_b;\n    \
                         max{ty} %f_y, %f_sum, {zero_lit};\n    \
                         st.global{ty} [%rd_c], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a one-minus kernel: `b[i] = 1 - a[i]`.
    fn generate_one_minus(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);
        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         sub{ty} %f_y, {one_lit}, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a power kernel: `c[i] = a[i]^b[i]` using lg2+mul+ex2 approximation.
    fn generate_pow(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
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
                         lg2.approx{ty} %f_t1, %f_a;\n    \
                         mul{ty} %f_t2, %f_t1, %f_b;\n    \
                         ex2.approx{ty} %f_c, %f_t2;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a min or max kernel using native PTX min/max.
    fn generate_binary_minmax(&self, min_or_max: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let min_or_max = min_or_max.to_string();
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
                         {min_or_max}{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a comparison kernel: `c[i] = (a[i] {cond} b[i]) ? 1.0 : 0.0`.
    fn generate_binary_cmp(&self, cond: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);
        let zero_lit = float_zero_literal(self.precision);
        let cond = cond.to_string();
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
                         setp.{cond}{ty} %p_cmp, %f_a, %f_b;\n    \
                         selp{ty} %f_c, {one_lit}, {zero_lit}, %p_cmp;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a probabilistic OR kernel: `c[i] = a[i] + b[i] - a[i]*b[i]`.
    fn generate_or_prob_sum(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
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
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_s, %f_a, %f_t;\n    \
                         add{ty} %f_c, %f_s, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a fuzzy NAND kernel: `c[i] = 1 - a[i]*b[i]`.
    fn generate_nand(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);
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
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_c, {one_lit}, %f_t;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
    /// Generates a fuzzy NOR kernel: `c[i] = 1 - (a[i] + b[i] - a[i]*b[i])`.
    fn generate_nor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);
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
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_s, %f_a, %f_t;\n    \
                         add{ty} %f_u, %f_s, %f_b;\n    \
                         sub{ty} %f_c, {one_lit}, %f_u;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }
}
