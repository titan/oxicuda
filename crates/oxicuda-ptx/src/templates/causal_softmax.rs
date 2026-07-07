//! Numerically stable *causal* (masked) row-wise softmax kernel template.
//!
//! This is the masked counterpart of [`crate::templates::softmax`]. It
//! generates a PTX kernel computing a row-wise softmax over a row-major
//! matrix in which, for output row `i`, only columns `j <= i` participate;
//! columns `j > i` are masked out (treated as `-inf`, contributing `0` to
//! the row max and the exponential sum, and written as `0` in the output).
//!
//! This is exactly the masking used by autoregressive (decoder) attention:
//! query position `i` may only attend to key positions `j <= i`.
//!
//! # Batched invocation
//!
//! `rows` may be a flattened `batch * heads * seq_len` count covering many
//! independent causal matrices stacked in one buffer, provided every matrix
//! shares the same `seq_len` (the row-major layout is then a plain
//! `[batch*heads*seq_len, cols]` matrix). The kernel takes `seq_len`
//! explicitly and re-derives the *within-matrix* row via
//! `row_in_seq = row % seq_len` before computing the live-column count, so
//! the causal boundary resets at the start of every matrix instead of
//! saturating to "fully unmasked" past the first one. For a single
//! `rows x cols` matrix, pass `seq_len == rows` (unchanged behavior).
//!
//! # Algorithm
//!
//! For each output row `i` the kernel runs the standard three-pass stable
//! softmax, but restricted to the unmasked prefix `j in [0, i]`:
//!
//! 1. **Masked row maximum**: `m = max_{j<=i} x[i, j]`
//! 2. **Masked exponential sum**: `s = sum_{j<=i} exp(x[i, j] - m)`
//! 3. **Masked normalize**: `y[i, j] = exp(x[i, j] - m) / s` for `j <= i`,
//!    and `y[i, j] = 0` for `j > i`.
//!
//! # Parallelization
//!
//! Unlike the plain softmax template (which uses warp- or block-level
//! reductions), this kernel assigns **one thread per row** and walks the
//! row sequentially. This mirrors the reference CUDA kernel in `trustformers`
//! one-for-one, keeping the numerics bit-for-bit comparable, and avoids the
//! load imbalance a warp/block reduction would suffer under a triangular
//! mask (the warp/block strategies assume every lane processes a live
//! element). The launch is therefore `rows` threads laid out over a 1-D grid.
//!
//! # Degenerate rows
//!
//! Two guard cases match the reference kernel exactly:
//!
//! - If the masked maximum is still `-inf` (`m < -1e38`), the row is written
//!   as the one-hot vector `[1, 0, 0, ...]`.
//! - If the masked exponential sum underflows (`s < 1e-10`), the row is
//!   likewise written as `[1, 0, 0, ...]`.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::causal_softmax::CausalSoftmaxTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = CausalSoftmaxTemplate {
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("causal_softmax_f32"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating a numerically stable causal-softmax PTX kernel.
///
/// Each launched thread processes exactly one matrix row, applying the
/// causal mask (`column > row` is excluded) before the stable softmax.
///
/// The generated kernel signature is:
///
/// ```text
/// kernel(.u64 input, .u64 output, .u32 rows, .u32 cols, .u32 seq_len)
/// ```
///
/// where `input` / `output` point to row-major `rows x cols` matrices, and
/// `seq_len` is the row count of one causal matrix (`seq_len == rows` for a
/// single, non-batched matrix; `seq_len < rows` when `rows` flattens several
/// batch/head matrices back to back — see the module docs).
pub struct CausalSoftmaxTemplate {
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl CausalSoftmaxTemplate {
    /// Returns the kernel function name (e.g. `causal_softmax_f32`).
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("causal_softmax_{type_str}")
    }

    /// Validates template parameters.
    ///
    /// Only `F32` and `F64` are supported: the stable softmax relies on
    /// `ex2.approx` (for `exp`) and `rcp.approx` (for `1/sum`), which are
    /// defined for `.f32`/`.f64` but not for `.f16`/`.bf16`. A half-precision
    /// causal softmax would have to upconvert to f32 around each transcendental;
    /// that is intentionally left out of this 1:1 port of the reference f32
    /// kernel.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "causal softmax requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Bit pattern for `-inf` in the active precision.
    const fn neg_inf(&self) -> &'static str {
        match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        }
    }

    /// Bit pattern for `0.0` in the active precision.
    const fn zero(&self) -> &'static str {
        match self.precision {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        }
    }

    /// Bit pattern for `1.0` in the active precision.
    const fn one(&self) -> &'static str {
        match self.precision {
            PtxType::F64 => "0d3FF0000000000000",
            _ => "0f3F800000",
        }
    }

    /// Bit pattern for `log2(e)` in the active precision (used to implement
    /// `exp(x) = ex2.approx(x * log2(e))`).
    const fn log2e(&self) -> &'static str {
        match self.precision {
            PtxType::F64 => "0d3FF71547652B82FE",
            _ => "0f3FB8AA3B",
        }
    }

    /// Generates the complete PTX module text for the causal-softmax kernel.
    ///
    /// Kernel parameters:
    /// - `input`: pointer to the `rows x cols` input matrix (row-major)
    /// - `output`: pointer to the `rows x cols` output matrix (row-major)
    /// - `rows`: number of rows (query positions), possibly `batch*heads*seq_len`
    /// - `cols`: number of columns (key positions)
    /// - `seq_len`: row count of one causal matrix; the causal boundary for
    ///   row `i` is derived from `i % seq_len`, not `i` directly, so the mask
    ///   resets at every matrix boundary in a batched invocation.
    ///
    /// For output row `i` (within its `seq_len`-row matrix), columns `j > i`
    /// are masked: they are excluded from the max / sum and written as `0`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision is not a supported float type.
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let neg_inf = self.neg_inf();
        let zero = self.zero();
        let one = self.one();
        let log2e = self.log2e();

        let mut ptx = String::with_capacity(4096);

        // -- Header -----------------------------------------------------------
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_rows,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_cols,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_seq_len").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // -- Register declarations -------------------------------------------
        // r0  = tid.x, r1 = ctaid.x, r2 = ntid.x
        // r3  = row (global thread id)
        // r4  = rows, r5 = cols
        // r6  = j (loop counter), r7 = upper bound (= min(row_in_seq, cols-1)+1 live cols)
        // r8  = scratch, r9 = seq_len, r10 = row_in_seq (= row % seq_len)
        // rd0 = input base, rd1 = output base, rd2 = row base (input)
        // rd3 = row base (output), rd4 = element offset bytes, rd5 = element addr
        // f0  = max, f1 = current value, f2 = sum, f3 = exp scratch, f4 = rcp(sum)
        writeln!(ptx, "    .reg .b32 %r<12>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Thread -> row index ---------------------------------------------
        writeln!(ptx, "    // row = blockIdx.x * blockDim.x + threadIdx.x")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Bounds check: row < rows ----------------------------------------
        writeln!(ptx, "    ld.param.u32 %r4, [%param_rows];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r5, [%param_cols];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_seq_len];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $CS_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Row base addresses: base + row * cols * byte_size ---------------
        writeln!(
            ptx,
            "    // rd2 = input + row*cols*bytes ; rd3 = output + row*cols*bytes"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u32 %r8, %r3, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd2, %rd0, %rd4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd3, %rd1, %rd4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Live-column count: live = min(row_in_seq + 1, cols) -------------
        // Causal mask: columns j in [0, row_in_seq] are live (j must be < cols).
        // row_in_seq = row % seq_len re-derives the within-matrix row so the
        // causal boundary resets at every seq_len-row matrix in a batched
        // invocation, instead of saturating to "fully unmasked" once the flat
        // `row` index exceeds `cols` (see module docs).
        // r7 = min(row_in_seq + 1, cols). The mask is thus "j < r7".
        writeln!(ptx, "    // row_in_seq = row % seq_len").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rem.u32 %r10, %r3, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    // live = min(row_in_seq + 1, cols)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r7, %r10, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    min.u32 %r7, %r7, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Pass 1: masked row max ------------------------------------------
        // f0 = -inf; for (j=0; j<live; ++j) f0 = max(f0, input[j])
        writeln!(ptx, "    // Pass 1: masked row max over j in [0, live)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_MAX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_MAX_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd2, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r6, %r6, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CS_MAX_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_MAX_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Guard: max < -1e38 -> one-hot row -------------------------------
        // -1e38 bit pattern: f32 0fFE967699 ; f64 0dC7D2CED32A16A1B1.
        let neg_1e38 = match self.precision {
            PtxType::F64 => "0dC7D2CED32A16A1B1",
            _ => "0fFE967699",
        };
        writeln!(ptx, "    // if (max < -1e38) write one-hot and return")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f5, {neg_1e38};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt{ty} %p1, %f0, %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_ONEHOT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Pass 2: masked exp sum ------------------------------------------
        // f2 = 0; for (j=0; j<live; ++j) f2 += exp(input[j] - max)
        writeln!(ptx, "    // Pass 2: masked sum of exp(x - max)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f2, {zero};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_SUM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_SUM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd2, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f3, %f1, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f3, %f3, {log2e};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f3, %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f2, %f2, %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r6, %r6, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CS_SUM_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_SUM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Guard: sum < 1e-10 -> one-hot row -------------------------------
        // 1e-10 bit pattern: f32 0f2EDBE6FF ; f64 0d3DDB7CDFD9D7BDBB.
        let small = match self.precision {
            PtxType::F64 => "0d3DDB7CDFD9D7BDBB",
            _ => "0f2EDBE6FF",
        };
        writeln!(ptx, "    // if (sum < 1e-10) write one-hot and return")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f5, {small};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt{ty} %p1, %f2, %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_ONEHOT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Pass 3: normalize + write masked output -------------------------
        // f4 = 1/sum; for (j=0; j<cols; ++j)
        //   out[j] = (j < live) ? exp(input[j]-max)*f4 : 0
        writeln!(
            ptx,
            "    // Pass 3: normalize live columns, zero masked columns"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f4, %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_NORM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd3, %rd5;").map_err(PtxGenError::FormatError)?;
        // Masked column? j >= live -> store 0.
        writeln!(ptx, "    setp.ge.u32 %p2, %r6, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $CS_NORM_ZERO;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd2, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f3, %f1, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f3, %f3, {log2e};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f3, %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f3, %f3, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CS_NORM_NEXT;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_NORM_ZERO:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f3, {zero};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_NORM_NEXT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r6, %r6, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CS_NORM_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- One-hot fallback: out[0] = 1, out[1..cols] = 0 ------------------
        writeln!(ptx, "$CS_ONEHOT:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    // out[j] = (j == 0) ? 1 : 0").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$CS_OH_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $CS_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd3, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u32 %p2, %r6, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f3, {zero};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 mov{ty} %f3, {one};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r6, %r6, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $CS_OH_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // -- Exit -------------------------------------------------------------
        writeln!(ptx, "$CS_DONE:").map_err(PtxGenError::FormatError)?;
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
        let t = CausalSoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        assert_eq!(t.kernel_name(), "causal_softmax_f32");
    }

    #[test]
    fn generates_f32_kernel() {
        let t = CausalSoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = t.generate().expect("should generate causal softmax f32");
        // Entry point present.
        assert!(ptx.contains(".visible .entry causal_softmax_f32("));
        // Five parameters: input, output, rows, cols, seq_len.
        assert!(ptx.contains("%param_rows"));
        assert!(ptx.contains("%param_cols"));
        assert!(ptx.contains("%param_seq_len"));
        // Stable softmax uses ex2.approx for exp and rcp.approx for 1/sum.
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
        // The live-column bound (min(row+1, cols)) implements the causal mask.
        assert!(ptx.contains("min.u32"));
        // One-hot degenerate fallback is emitted.
        assert!(ptx.contains("$CS_ONEHOT"));
    }

    #[test]
    fn generates_f64_kernel() {
        let t = CausalSoftmaxTemplate {
            precision: PtxType::F64,
            target: SmVersion::Sm90,
        };
        let ptx = t.generate().expect("should generate causal softmax f64");
        assert!(ptx.contains("causal_softmax_f64"));
        assert!(ptx.contains("ex2.approx.f64"));
        // f64 -inf bit pattern.
        assert!(ptx.contains("0dFFF0000000000000"));
    }

    #[test]
    fn rejects_integer_precision() {
        let t = CausalSoftmaxTemplate {
            precision: PtxType::S32,
            target: SmVersion::Sm80,
        };
        assert!(t.generate().is_err());
    }

    /// The mask must reference the column index `j` against `live = min(row+1, cols)`
    /// in both the reduction passes and the normalize pass, so that columns
    /// `j > row` are excluded from the max/sum and written as zero.
    #[test]
    fn mask_excludes_future_columns() {
        let t = CausalSoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = t.generate().expect("generate");
        // The normalize loop branches to the zero-store path for masked cols.
        assert!(ptx.contains("$CS_NORM_ZERO"));
        // The reduction passes bound their loop by the live-column register r7.
        assert!(ptx.contains("setp.ge.u32 %p1, %r6, %r7;"));
    }

    /// A batched invocation flattens `batch*heads*seq_len` matrices into one
    /// `rows x cols` buffer; the live-column count must be derived from
    /// `row % seq_len`, not the raw flat row, or the mask saturates to
    /// "fully unmasked" past the first matrix (see module docs).
    #[test]
    fn live_count_uses_row_modulo_seq_len() {
        let t = CausalSoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = t.generate().expect("generate");
        assert!(ptx.contains("%param_seq_len"));
        assert!(ptx.contains("rem.u32 %r10, %r3, %r9;"));
        // The live-count add/min must consume the modulo result (r10), not
        // the raw flat row (r3).
        assert!(ptx.contains("add.u32 %r7, %r10, 1;"));
    }
}
