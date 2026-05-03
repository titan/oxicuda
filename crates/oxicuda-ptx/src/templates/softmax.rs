//! Numerically stable softmax kernel template.
//!
//! Generates PTX kernels for computing row-wise softmax using the numerically
//! stable three-pass algorithm:
//!
//! 1. **Find row maximum**: `m = max(x[0], x[1], ..., x[N-1])`
//! 2. **Exponentiate and sum**: `s = sum(exp(x[i] - m))`
//! 3. **Normalize**: `y[i] = exp(x[i] - m) / s`
//!
//! The implementation strategy depends on the row size:
//!
//! - **`row_size` <= 32**: Warp shuffle reduction (no shared memory needed)
//! - **`row_size` <= 1024**: Shared memory block reduction (one block per row)
//! - **`row_size` > 1024**: Multi-block reduction via the
//!   [`generate_multi_block_softmax_ptx`] entry point. The work for one
//!   row is split across multiple blocks, with block-local reductions
//!   exchanged through a global scratch buffer. A second "finalize" kernel
//!   then computes the global max / global sum and writes the normalized
//!   output.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::softmax::SoftmaxTemplate;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = SoftmaxTemplate {
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//!     row_size: 32,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("softmax"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Template for generating numerically stable softmax PTX kernels.
///
/// Each block processes one row of the input matrix. The row size determines
/// the reduction strategy (warp shuffle vs shared memory).
pub struct SoftmaxTemplate {
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
    /// Number of elements per row. Must be <= 1024 for current implementation.
    pub row_size: u32,
}

impl SoftmaxTemplate {
    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("softmax_{type_str}_r{}", self.row_size)
    }

    /// Generates the complete PTX module text for the softmax kernel.
    ///
    /// Kernel parameters:
    /// - `input`: pointer to input matrix (`batch_size` x `row_size`, row-major)
    /// - `output`: pointer to output matrix (same shape)
    /// - `batch_size`: number of rows
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The precision is not a supported float type
    /// - The row size exceeds the supported limit (> 1024)
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        if self.row_size <= 32 {
            self.generate_warp_shuffle()
        } else {
            self.generate_shared_memory()
        }
    }

    /// Validates template parameters.
    ///
    /// `SoftmaxTemplate` only emits the warp-shuffle and single-block paths.
    /// For `row_size > 1024` callers should use
    /// [`generate_multi_block_softmax_ptx`] which produces a two-kernel
    /// (reduce + finalize) pipeline.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "softmax requires F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if self.row_size == 0 {
            return Err(PtxGenError::GenerationFailed(
                "row_size must be > 0".to_string(),
            ));
        }
        if self.row_size > 1024 {
            return Err(PtxGenError::GenerationFailed(format!(
                "row_size {} exceeds the single-block limit of 1024; \
                 use generate_multi_block_softmax_ptx for multi-block dispatch",
                self.row_size
            )));
        }
        Ok(())
    }

    /// Generates a warp-shuffle-based softmax for `row_size` <= 32.
    ///
    /// One warp processes one row. Each thread handles one element.
    /// The three-pass reduction (max, exp+sum, normalize) uses
    /// `shfl.sync.down.b32` for intra-warp communication.
    #[allow(clippy::too_many_lines)]
    fn generate_warp_shuffle(&self) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let neg_inf = match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        };

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Declarations
        writeln!(ptx, "    .reg .b32 %r<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread indexing: warp_id = global_tid / 32 => row index
        //                  lane_id = global_tid % 32 => element within row
        writeln!(ptx, "    // Compute row and lane indices").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        // r4 = warp_id (row), r5 = lane_id (element)
        writeln!(ptx, "    shr.u32 %r4, %r3, 5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    and.b32 %r5, %r3, 31;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: row < batch_size
        writeln!(ptx, "    ld.param.u32 %r6, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r4, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $SM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load element (or -INF for out-of-row lanes)
        let row_size = self.row_size;
        writeln!(ptx, "    // Load element").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p1, %r5, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p1 bra $SKIP_LOAD_SM;").map_err(PtxGenError::FormatError)?;

        // Compute address: input[row * row_size + lane]
        writeln!(ptx, "    mad.lo.u32 %r7, %r4, {row_size}, %r5;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd1, %r7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd2];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$SKIP_LOAD_SM:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 1: find row max via warp shuffle
        writeln!(ptx, "    // Pass 1: row-wise max reduction").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f1, %f0;").map_err(PtxGenError::FormatError)?;
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(
                ptx,
                "    shfl.sync.down.b32 %f2, %f1, {offset}, 31, 0xFFFFFFFF;"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    max{ty} %f1, %f1, %f2;").map_err(PtxGenError::FormatError)?;
        }
        // Broadcast max from lane 0 to all lanes
        writeln!(ptx, "    shfl.sync.idx.b32 %f1, %f1, 0, 31, 0xFFFFFFFF;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 2: exp(x - max) and sum
        writeln!(ptx, "    // Pass 2: exp(x - max) and sum reduction")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f3, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        // exp via 2^(x * log2(e)): log2(e) = 0f3FB8AA3B
        writeln!(ptx, "    mul{ty} %f3, %f3, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f3, %f3;").map_err(PtxGenError::FormatError)?;

        // Out-of-row lanes contribute 0 to the sum
        writeln!(ptx, "    @!%p1 mov{ty} %f3, 0f00000000;").map_err(PtxGenError::FormatError)?;

        // Sum reduction
        writeln!(ptx, "    mov{ty} %f4, %f3;").map_err(PtxGenError::FormatError)?;
        for offset in [16u32, 8, 4, 2, 1] {
            writeln!(
                ptx,
                "    shfl.sync.down.b32 %f5, %f4, {offset}, 31, 0xFFFFFFFF;"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add{ty} %f4, %f4, %f5;").map_err(PtxGenError::FormatError)?;
        }
        // Broadcast sum from lane 0
        writeln!(ptx, "    shfl.sync.idx.b32 %f4, %f4, 0, 31, 0xFFFFFFFF;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 3: normalize and store
        writeln!(ptx, "    // Pass 3: normalize and store").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p1 bra $SM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f6, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f7, %f3, %f6;").map_err(PtxGenError::FormatError)?;

        // Store to output
        writeln!(ptx, "    ld.param.u64 %rd3, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r8, %r4, {row_size}, %r5;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd5, %rd3, %rd4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd5], %f7;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$SM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates a shared-memory-based softmax for 32 < `row_size` <= 1024.
    ///
    /// One block processes one row, with threads loading multiple elements.
    #[allow(clippy::too_many_lines)]
    fn generate_shared_memory(&self) -> Result<String, PtxGenError> {
        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let neg_inf = match self.precision {
            PtxType::F64 => "0dFFF0000000000000",
            _ => "0fFF800000",
        };
        let row_size = self.row_size;
        // Use min(row_size, 256) threads per block, power-of-2
        let block_size = self.row_size.next_power_of_two().min(256);
        let smem_bytes = (block_size as usize) * byte_size;

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_batch_size").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<12>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_softmax[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Each block handles one row: row = blockIdx.x
        writeln!(ptx, "    // Block per row").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r2, [%param_batch_size];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $SM_BLK_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Base address for this row
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd1, %r1;").map_err(PtxGenError::FormatError)?;
        let row_bytes = row_size as usize * byte_size;
        writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {row_bytes};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 1: Each thread finds its local max, then reduce via shared mem
        writeln!(ptx, "    // Pass 1: find row max").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
        // Thread i handles elements i, i+blockSize, i+2*blockSize, ...
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MAX_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $MAX_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $MAX_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MAX_DONE:").map_err(PtxGenError::FormatError)?;

        // Store to shared memory and reduce
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd6, smem_softmax;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd6, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd7], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;

        // Tree reduction for max in shared memory
        let mut stride = block_size / 2;
        while stride > 0 {
            writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p2 bra $SKIP_MAX_{stride};").map_err(PtxGenError::FormatError)?;
            let partner_off = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f2, [%rd7+{partner_off}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f3, [%rd7];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    max{ty} %f3, %f3, %f2;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "$SKIP_MAX_{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }

        // Broadcast max from shared memory position 0
        writeln!(ptx, "    ld.shared{ty} %f4, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 2: exp(x - max) and partial sum
        writeln!(ptx, "    // Pass 2: exp(x - max) and sum").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f5, 0f00000000;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$EXP_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $EXP_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f6, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f6, %f6, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f6, %f6, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f6, %f6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add{ty} %f5, %f5, %f6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $EXP_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$EXP_DONE:").map_err(PtxGenError::FormatError)?;

        // Reduce sum via shared memory
        writeln!(ptx, "    st.shared{ty} [%rd7], %f5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        stride = block_size / 2;
        while stride > 0 {
            writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p2 bra $SKIP_SUM_{stride};").map_err(PtxGenError::FormatError)?;
            let partner_off = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f7, [%rd7+{partner_off}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f8, [%rd7];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    add{ty} %f8, %f8, %f7;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd7], %f8;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "$SKIP_SUM_{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }

        // Broadcast sum
        writeln!(ptx, "    ld.shared{ty} %f9, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    rcp.approx{ty} %f10, %f9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Pass 3: normalize and store
        writeln!(ptx, "    // Pass 3: normalize and store").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd8, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd9, %rd8, %rd1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$NORM_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $SM_BLK_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f11, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    sub{ty} %f11, %f11, %f4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f11, %f11, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ex2.approx{ty} %f11, %f11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f11, %f11, %f10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd10, %rd9, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd10], %f11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r3, %r3, {block_size};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $NORM_LOOP;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$SM_BLK_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Multi-block softmax (row_size > 1024)
// ---------------------------------------------------------------------------

/// Number of threads per block used by the multi-block reduce kernel.
///
/// Chosen as a power of two that maps to a single warp tree reduction in
/// shared memory. This must divide evenly into [`MULTI_BLOCK_DEFAULT_STRIDE`].
pub const MULTI_BLOCK_THREADS: u32 = 256;

/// Default per-block stride (elements processed by one reduce-kernel block)
/// used for multi-block softmax.
///
/// Each reduce-kernel block processes up to this many consecutive elements of
/// a single row. Larger values amortize the per-block synchronization cost;
/// smaller values increase parallelism. The default of 1024 matches the
/// single-block boundary so each multi-block reduce-kernel block is
/// structurally a copy of [`SoftmaxTemplate`]'s shared-memory path.
pub const MULTI_BLOCK_DEFAULT_STRIDE: u32 = 1024;

/// Output of [`generate_multi_block_softmax_ptx`].
///
/// Holds the two PTX modules that together form a numerically stable
/// row-wise softmax for `row_size > 1024`:
///
/// 1. The **reduce** kernel computes per-block `(max, sum_exp_local)` for
///    each chunk of a row and writes the pair to global scratch.
/// 2. The **finalize** kernel reads scratch for one row, derives the global
///    `max` and the rescaled global `sum_exp`, and emits the normalized
///    output.
///
/// Use the kernel names [`MultiBlockSoftmaxPtx::reduce_kernel_name`] and
/// [`MultiBlockSoftmaxPtx::finalize_kernel_name`] when looking up the
/// entry symbols after compilation.
#[derive(Debug, Clone)]
pub struct MultiBlockSoftmaxPtx {
    /// PTX source for the per-block reduce kernel.
    pub reduce_ptx: String,
    /// PTX source for the row-finalization kernel.
    pub finalize_ptx: String,
    /// Number of elements processed per reduce-kernel block (the block stride).
    pub block_stride: u32,
    /// Number of reduce-kernel blocks needed to cover one row.
    pub num_blocks_per_row: u32,
    /// Number of threads per reduce/finalize block.
    pub threads_per_block: u32,
    /// Scratch bytes per row (holds two `f32`-sized scalars per reduce-block:
    /// `block_max` followed by `block_sum`).
    pub scratch_bytes_per_row: usize,
    /// Scratch element type (matches `dtype`).
    pub scratch_dtype: PtxType,
}

impl MultiBlockSoftmaxPtx {
    /// Returns the entry-point name of the reduce kernel.
    #[must_use]
    pub fn reduce_kernel_name(&self) -> String {
        let ty = self.scratch_dtype.as_ptx_str().trim_start_matches('.');
        format!("softmax_mb_reduce_{ty}_s{}", self.block_stride)
    }

    /// Returns the entry-point name of the finalize kernel.
    #[must_use]
    pub fn finalize_kernel_name(&self) -> String {
        let ty = self.scratch_dtype.as_ptx_str().trim_start_matches('.');
        format!("softmax_mb_finalize_{ty}_s{}", self.block_stride)
    }
}

/// Generates the multi-block softmax PTX kernels for `row_size > 1024`.
///
/// The kernels split work for each row across multiple thread blocks. A
/// reduce kernel produces per-block `(max, sum_exp)` scratch entries; a
/// finalize kernel combines them into the global `(max, sum_exp)` and
/// writes the normalized output.
///
/// # Arguments
///
/// * `row_size` -- number of elements per row. Must be > 0.
/// * `block_stride` -- number of consecutive elements that one reduce-kernel
///   block handles. Pass [`MULTI_BLOCK_DEFAULT_STRIDE`] for the default.
/// * `threads_per_block` -- threads per reduce/finalize block. Must be a
///   power of two in `[32, 1024]`.
/// * `dtype` -- element precision. Must be `F32` (the only currently
///   implemented type for the multi-block path; FP64 falls through with an
///   error).
/// * `target` -- target SM architecture for the generated PTX.
///
/// # Errors
///
/// Returns [`PtxGenError`] if any input is invalid (zero `row_size`,
/// non-power-of-two `threads_per_block`, unsupported `dtype`, etc.).
pub fn generate_multi_block_softmax_ptx(
    row_size: u32,
    block_stride: u32,
    threads_per_block: u32,
    dtype: PtxType,
    target: SmVersion,
) -> Result<MultiBlockSoftmaxPtx, PtxGenError> {
    validate_multi_block_args(row_size, block_stride, threads_per_block, dtype)?;

    let num_blocks_per_row = row_size.div_ceil(block_stride);
    let scratch_bytes_per_row = (num_blocks_per_row as usize) * 2 * dtype.size_bytes();

    let reduce_ptx = emit_multi_block_reduce_ptx(
        row_size,
        block_stride,
        threads_per_block,
        num_blocks_per_row,
        dtype,
        target,
    )?;
    let finalize_ptx = emit_multi_block_finalize_ptx(
        row_size,
        block_stride,
        threads_per_block,
        num_blocks_per_row,
        dtype,
        target,
    )?;

    Ok(MultiBlockSoftmaxPtx {
        reduce_ptx,
        finalize_ptx,
        block_stride,
        num_blocks_per_row,
        threads_per_block,
        scratch_bytes_per_row,
        scratch_dtype: dtype,
    })
}

/// Validates inputs to [`generate_multi_block_softmax_ptx`].
fn validate_multi_block_args(
    row_size: u32,
    block_stride: u32,
    threads_per_block: u32,
    dtype: PtxType,
) -> Result<(), PtxGenError> {
    if row_size == 0 {
        return Err(PtxGenError::GenerationFailed(
            "row_size must be > 0".to_string(),
        ));
    }
    if block_stride == 0 {
        return Err(PtxGenError::GenerationFailed(
            "block_stride must be > 0".to_string(),
        ));
    }
    if !threads_per_block.is_power_of_two() || !(32..=1024).contains(&threads_per_block) {
        return Err(PtxGenError::GenerationFailed(format!(
            "threads_per_block must be a power of two in [32, 1024], got {threads_per_block}"
        )));
    }
    // Currently only F32 is supported for multi-block softmax. Extending to
    // F64 / F16 / BF16 requires per-precision constants and the appropriate
    // ld/st widening. Do not silently accept unsupported types.
    if !matches!(dtype, PtxType::F32) {
        return Err(PtxGenError::InvalidType(format!(
            "multi-block softmax currently supports only F32, got {}",
            dtype.as_ptx_str()
        )));
    }
    Ok(())
}

/// Emits the per-block reduce kernel PTX for multi-block softmax (F32).
///
/// Grid: `(num_blocks_per_row, batch_size, 1)`.
/// Block: `(threads_per_block, 1, 1)`.
///
/// Kernel parameters (in order):
/// - `.u64 input` -- pointer to input matrix `(batch_size x row_size)` row-major.
/// - `.u64 scratch` -- pointer to scratch `(batch_size x num_blocks_per_row x 2)`
///   floats. For row `r` and block `b`, scratch holds `block_max` at offset
///   `r * num_blocks_per_row * 2 + b * 2 + 0` and `block_sum_exp` at offset
///   `r * num_blocks_per_row * 2 + b * 2 + 1`.
/// - `.u32 batch_size` -- number of rows.
fn emit_multi_block_reduce_ptx(
    row_size: u32,
    block_stride: u32,
    threads_per_block: u32,
    num_blocks_per_row: u32,
    dtype: PtxType,
    target: SmVersion,
) -> Result<String, PtxGenError> {
    let ty = dtype.as_ptx_str();
    let elem_bytes = dtype.size_bytes();
    let kernel_name = format!(
        "softmax_mb_reduce_{}_s{}",
        dtype.as_ptx_str().trim_start_matches('.'),
        block_stride,
    );
    let mut ptx = String::with_capacity(8192);

    emit_mb_header(
        &mut ptx,
        target,
        &kernel_name,
        &[
            ".param .u64 %param_input",
            ".param .u64 %param_scratch",
            ".param .u32 %param_batch_size",
        ],
        threads_per_block,
        elem_bytes,
        "smem_mb_red",
    )?;

    emit_mb_red_indices_and_bounds(&mut ptx, row_size, block_stride, elem_bytes)?;
    emit_mb_red_max_pass(&mut ptx, ty, threads_per_block, elem_bytes)?;
    emit_mb_red_sum_pass(&mut ptx, ty, threads_per_block, elem_bytes)?;
    emit_mb_red_scratch_write(&mut ptx, ty, num_blocks_per_row, elem_bytes)?;

    writeln!(ptx, "$MB_RED_DONE:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

    Ok(ptx)
}

/// Emits the PTX header (version/target/entry/registers/shared-memory) for
/// one of the multi-block softmax kernels.
fn emit_mb_header(
    ptx: &mut String,
    target: SmVersion,
    kernel_name: &str,
    params: &[&str],
    threads_per_block: u32,
    elem_bytes: usize,
    smem_label: &str,
) -> Result<(), PtxGenError> {
    let smem_bytes = (threads_per_block as usize) * elem_bytes;

    writeln!(ptx, ".version {}", target.ptx_version()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".target {}", target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
    let last = params.len().saturating_sub(1);
    for (idx, p) in params.iter().enumerate() {
        let sep = if idx == last { "" } else { "," };
        writeln!(ptx, "    {p}{sep}").map_err(PtxGenError::FormatError)?;
    }
    writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .maxntid {threads_per_block}, 1, 1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
    writeln!(
        ptx,
        "    .shared .align {} .b8 {smem_label}[{smem_bytes}];",
        elem_bytes.max(4),
    )
    .map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the index/bounds/row-base setup shared by both passes of the reduce
/// kernel. After this completes:
/// - `%r0 = tid.x`, `%r1 = ctaid.x` (block-in-row), `%r2 = ctaid.y` (row).
/// - `%rd2 = input + row * row_size * elem_bytes` (row base address).
/// - `%r4 = start_index`, `%r5 = end_index` for this block.
fn emit_mb_red_indices_and_bounds(
    ptx: &mut String,
    row_size: u32,
    block_stride: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r2, %ctaid.y;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_batch_size];")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r2, %r3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p0 bra $MB_RED_DONE;").map_err(PtxGenError::FormatError)?;

    let row_bytes = row_size as usize * elem_bytes;
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd1, %r2;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {row_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;

    // start = block_id * block_stride; end = min(start + block_stride, row_size).
    // Last partial block beyond row_size simply yields end <= start, in which
    // case the per-block max stays at -INF and sum at 0 -- a no-op for the
    // global reductions in the finalize kernel.
    writeln!(ptx, "    mul.lo.u32 %r4, %r1, {block_stride};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r5, %r4, {block_stride};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    min.u32 %r5, %r5, {row_size};").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the per-block max reduction (Pass 1 of the reduce kernel). The
/// resulting block-max ends up in `%f4` and `smem_mb_red[0]`.
fn emit_mb_red_max_pass(
    ptx: &mut String,
    ty: &str,
    threads_per_block: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    let neg_inf = "0fFF800000";
    writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r6, %r4, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_RED_MAX_LOOP:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p1 bra $MB_RED_MAX_DONE;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd3, %r6;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {elem_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r6, %r6, {threads_per_block};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bra $MB_RED_MAX_LOOP;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_RED_MAX_DONE:").map_err(PtxGenError::FormatError)?;

    // Tree reduction in shared memory.
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {elem_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u64 %rd6, smem_mb_red;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd7, %rd6, %rd5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.shared{ty} [%rd7], %f0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    emit_smem_tree_reduce(ptx, ty, threads_per_block, elem_bytes, "MAX", "max")?;
    writeln!(ptx, "    ld.shared{ty} %f4, [%rd6];").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the per-block sum-of-exp reduction (Pass 2 of the reduce kernel).
/// Reads the broadcast block-max from `%f4`. Result lands in `%f9`.
fn emit_mb_red_sum_pass(
    ptx: &mut String,
    ty: &str,
    threads_per_block: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    mov{ty} %f5, 0f00000000;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r6, %r4, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_RED_SUM_LOOP:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p1 bra $MB_RED_SUM_DONE;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd3, %r6;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {elem_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f6, [%rd4];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    sub{ty} %f6, %f6, %f4;").map_err(PtxGenError::FormatError)?;
    // exp(x) ~= 2^(x * log2(e)); log2(e) = 0f3FB8AA3B (1.4426950408)
    writeln!(ptx, "    mul{ty} %f6, %f6, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ex2.approx{ty} %f6, %f6;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add{ty} %f5, %f5, %f6;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r6, %r6, {threads_per_block};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bra $MB_RED_SUM_LOOP;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_RED_SUM_DONE:").map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.shared{ty} [%rd7], %f5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    emit_smem_tree_reduce(ptx, ty, threads_per_block, elem_bytes, "SUM", "add")?;
    writeln!(ptx, "    ld.shared{ty} %f9, [%rd6];").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits the thread-0 scratch write for the reduce kernel:
/// `scratch[row * num_blocks_per_row + block_id] = (block_max, block_sum)`.
fn emit_mb_red_scratch_write(
    ptx: &mut String,
    ty: &str,
    num_blocks_per_row: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    setp.ne.u32 %p3, %r0, 0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p3 bra $MB_RED_DONE;").map_err(PtxGenError::FormatError)?;

    let pair_bytes = 2 * elem_bytes;
    writeln!(ptx, "    ld.param.u64 %rd8, [%param_scratch];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u32 %r7, %r2, {num_blocks_per_row};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r7, %r7, %r1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd9, %r7;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd9, %rd9, {pair_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd10, %rd8, %rd9;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.global{ty} [%rd10], %f4;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.global{ty} [%rd10+{elem_bytes}], %f9;")
        .map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits a power-of-two tree reduction over `threads_per_block` shared-memory
/// slots. The input/output cell is `[%rd7]`; partner cells are at positive
/// offsets. `op` is the PTX mnemonic (`"max"` or `"add"`); `tag` is a unique
/// token used to disambiguate label names between callers.
fn emit_smem_tree_reduce(
    ptx: &mut String,
    ty: &str,
    threads_per_block: u32,
    elem_bytes: usize,
    tag: &str,
    op: &str,
) -> Result<(), PtxGenError> {
    let mut stride = threads_per_block / 2;
    while stride > 0 {
        writeln!(ptx, "    setp.lt.u32 %p2, %r0, {stride};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p2 bra $MB_TREE_SKIP_{tag}_{stride};")
            .map_err(PtxGenError::FormatError)?;
        let partner_off = stride as usize * elem_bytes;
        writeln!(ptx, "    ld.shared{ty} %f15, [%rd7+{partner_off}];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.shared{ty} %f16, [%rd7];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    {op}{ty} %f16, %f16, %f15;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd7], %f16;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$MB_TREE_SKIP_{tag}_{stride}:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        stride /= 2;
    }
    Ok(())
}

/// Emits the per-row finalize kernel PTX for multi-block softmax (F32).
///
/// Grid: `(batch_size, 1, 1)`.
/// Block: `(threads_per_block, 1, 1)`.
///
/// Kernel parameters (in order):
/// - `.u64 input` -- pointer to input matrix `(batch_size x row_size)` row-major.
/// - `.u64 output` -- pointer to output matrix (same shape).
/// - `.u64 scratch` -- pointer to per-block (max, sum) scratch.
/// - `.u32 batch_size` -- number of rows.
fn emit_multi_block_finalize_ptx(
    row_size: u32,
    block_stride: u32,
    threads_per_block: u32,
    num_blocks_per_row: u32,
    dtype: PtxType,
    target: SmVersion,
) -> Result<String, PtxGenError> {
    let ty = dtype.as_ptx_str();
    let elem_bytes = dtype.size_bytes();
    let kernel_name = format!(
        "softmax_mb_finalize_{}_s{}",
        dtype.as_ptx_str().trim_start_matches('.'),
        block_stride,
    );
    let mut ptx = String::with_capacity(8192);

    emit_mb_header(
        &mut ptx,
        target,
        &kernel_name,
        &[
            ".param .u64 %param_input",
            ".param .u64 %param_output",
            ".param .u64 %param_scratch",
            ".param .u32 %param_batch_size",
        ],
        threads_per_block,
        elem_bytes,
        "smem_mb_fin",
    )?;

    emit_mb_fin_indices_and_scratch_base(&mut ptx, num_blocks_per_row, elem_bytes)?;
    emit_mb_fin_global_max_pass(
        &mut ptx,
        ty,
        threads_per_block,
        num_blocks_per_row,
        elem_bytes,
    )?;
    emit_mb_fin_global_sum_pass(
        &mut ptx,
        ty,
        threads_per_block,
        num_blocks_per_row,
        elem_bytes,
    )?;
    emit_mb_fin_normalize_pass(&mut ptx, ty, row_size, threads_per_block, elem_bytes)?;

    writeln!(ptx, "$MB_FIN_DONE:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

    Ok(ptx)
}

/// Emits the index/bounds and scratch base setup for the finalize kernel.
/// After this completes, `%r0 = tid.x`, `%r1 = ctaid.x` (row index), and
/// `%rd2 = scratch + row * num_blocks_per_row * 2 * elem_bytes`.
fn emit_mb_fin_indices_and_scratch_base(
    ptx: &mut String,
    num_blocks_per_row: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_batch_size];")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p0 bra $MB_FIN_DONE;").map_err(PtxGenError::FormatError)?;

    let row_pair_bytes = num_blocks_per_row as usize * 2 * elem_bytes;
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_scratch];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd1, %r1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd1, %rd1, {row_pair_bytes};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd2, %rd0, %rd1;").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits Pass A of the finalize kernel: each thread strides over the per-block
/// maxes, computes a local max, then the warp/block does a tree reduction to
/// produce the global max. Result lands in `%f4` and `smem_mb_fin[0]`.
fn emit_mb_fin_global_max_pass(
    ptx: &mut String,
    ty: &str,
    threads_per_block: u32,
    num_blocks_per_row: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    let neg_inf = "0fFF800000";
    let pair_bytes = 2 * elem_bytes;

    writeln!(ptx, "    mov{ty} %f0, {neg_inf};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_FIN_GMAX_LOOP:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r3, {num_blocks_per_row};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p1 bra $MB_FIN_GMAX_DONE;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {pair_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    max{ty} %f0, %f0, %f1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r3, %r3, {threads_per_block};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bra $MB_FIN_GMAX_LOOP;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_FIN_GMAX_DONE:").map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    cvt.u64.u32 %rd5, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {elem_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u64 %rd6, smem_mb_fin;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd7, %rd6, %rd5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.shared{ty} [%rd7], %f0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    emit_smem_tree_reduce(ptx, ty, threads_per_block, elem_bytes, "GMAX", "max")?;
    writeln!(ptx, "    ld.shared{ty} %f4, [%rd6];").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits Pass B of the finalize kernel: rescale each per-block sum by
/// `exp(block_max - global_max)` and reduce the result. The final global sum
/// (and its reciprocal) end up in `%f11` / `%f12`.
fn emit_mb_fin_global_sum_pass(
    ptx: &mut String,
    ty: &str,
    threads_per_block: u32,
    num_blocks_per_row: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    let pair_bytes = 2 * elem_bytes;

    writeln!(ptx, "    mov{ty} %f5, 0f00000000;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_FIN_GSUM_LOOP:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r3, {num_blocks_per_row};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p1 bra $MB_FIN_GSUM_DONE;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {pair_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd4, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f6, [%rd4];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f7, [%rd4+{elem_bytes}];")
        .map_err(PtxGenError::FormatError)?;
    // delta = block_max - global_max  (always <= 0)
    writeln!(ptx, "    sub{ty} %f8, %f6, %f4;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul{ty} %f8, %f8, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ex2.approx{ty} %f8, %f8;").map_err(PtxGenError::FormatError)?;
    // local_sum += block_sum * exp(delta)  -- the rescale that
    // multi-block softmax depends on.
    writeln!(ptx, "    fma.rn{ty} %f5, %f7, %f8, %f5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r3, %r3, {threads_per_block};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bra $MB_FIN_GSUM_LOOP;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_FIN_GSUM_DONE:").map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.shared{ty} [%rd7], %f5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
    emit_smem_tree_reduce(ptx, ty, threads_per_block, elem_bytes, "GSUM", "add")?;
    writeln!(ptx, "    ld.shared{ty} %f11, [%rd6];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    rcp.approx{ty} %f12, %f11;").map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Emits Pass C of the finalize kernel: stride over the row, applying
/// `output[i] = exp(input[i] - global_max) * (1 / global_sum)`.
fn emit_mb_fin_normalize_pass(
    ptx: &mut String,
    ty: &str,
    row_size: u32,
    threads_per_block: u32,
    elem_bytes: usize,
) -> Result<(), PtxGenError> {
    let row_bytes = row_size as usize * elem_bytes;
    writeln!(ptx, "    ld.param.u64 %rd8, [%param_input];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u64 %rd9, [%param_output];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {row_bytes};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd11, %rd8, %rd10;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd12, %rd9, %rd10;").map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    mov.u32 %r3, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "$MB_FIN_NORM_LOOP:").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r3, {row_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p1 bra $MB_FIN_DONE;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd13, %r3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd13, %rd13, {elem_bytes};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd14, %rd11, %rd13;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.global{ty} %f13, [%rd14];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    sub{ty} %f13, %f13, %f4;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul{ty} %f13, %f13, 0f3FB8AA3B;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ex2.approx{ty} %f13, %f13;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul{ty} %f13, %f13, %f12;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u64 %rd15, %rd12, %rd13;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    st.global{ty} [%rd15], %f13;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    add.u32 %r3, %r3, {threads_per_block};")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    bra $MB_FIN_NORM_LOOP;").map_err(PtxGenError::FormatError)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn kernel_name_format() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        assert_eq!(t.kernel_name(), "softmax_f32_r32");
    }

    #[test]
    fn invalid_precision() {
        let t = SoftmaxTemplate {
            precision: PtxType::U32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn too_large_row_for_single_block_template() {
        // The single-block SoftmaxTemplate still rejects row_size > 1024.
        // For larger rows, callers should use
        // generate_multi_block_softmax_ptx instead.
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 2048,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn zero_row() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 0,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn generate_warp_shuffle_softmax() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        let ptx = t.generate().expect("should generate warp shuffle softmax");
        assert!(ptx.contains(".entry softmax_f32_r32"));
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn generate_shared_mem_softmax() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 256,
        };
        let ptx = t.generate().expect("should generate shared mem softmax");
        assert!(ptx.contains(".entry softmax_f32_r256"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn generate_non_power_of_2_row() {
        let t = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 100,
        };
        let ptx = t.generate().expect("should handle non-power-of-2 rows");
        assert!(ptx.contains(".entry softmax_f32_r100"));
    }

    // ---- multi-block (row_size > 1024) tests ---------------------------

    #[test]
    fn multi_block_rejects_zero_row_size() {
        let r = generate_multi_block_softmax_ptx(0, 1024, 256, PtxType::F32, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_rejects_zero_block_stride() {
        let r = generate_multi_block_softmax_ptx(2048, 0, 256, PtxType::F32, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_rejects_unsupported_dtype() {
        // Only F32 is currently supported.
        let r = generate_multi_block_softmax_ptx(2048, 1024, 256, PtxType::F64, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_rejects_non_power_of_two_threads() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 100, PtxType::F32, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_rejects_too_few_threads() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 16, PtxType::F32, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_rejects_too_many_threads() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 2048, PtxType::F32, SmVersion::Sm80);
        assert!(r.is_err());
    }

    #[test]
    fn multi_block_layout_2048() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("multi-block softmax PTX should generate");
        assert_eq!(r.block_stride, 1024);
        assert_eq!(r.num_blocks_per_row, 2);
        assert_eq!(r.threads_per_block, 256);
        // 2 blocks * 2 floats per block = 4 floats per row = 16 bytes.
        assert_eq!(r.scratch_bytes_per_row, 4 * 4);
        assert_eq!(r.scratch_dtype, PtxType::F32);
    }

    #[test]
    fn multi_block_layout_partial_last_block() {
        // 2050 elements with stride 1024 -> 3 blocks (last one only handles
        // two elements). The reduce kernel must still emit a sentinel for
        // the third block so the finalize kernel ignores it correctly.
        let r = generate_multi_block_softmax_ptx(2050, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("multi-block softmax PTX should handle a partial last block");
        assert_eq!(r.num_blocks_per_row, 3);
        assert_eq!(r.scratch_bytes_per_row, 6 * 4);
    }

    #[test]
    fn multi_block_reduce_ptx_contains_expected_mnemonics() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("multi-block softmax PTX should generate");
        let red = &r.reduce_ptx;
        assert!(red.contains(".entry softmax_mb_reduce_f32_s1024"));
        assert!(red.contains("ld.global.f32"));
        assert!(red.contains("st.global.f32"));
        assert!(red.contains("ld.shared.f32"));
        assert!(red.contains("st.shared.f32"));
        assert!(red.contains("bar.sync 0"));
        assert!(red.contains("ex2.approx.f32"));
        assert!(red.contains("max.f32"));
    }

    #[test]
    fn multi_block_finalize_ptx_contains_expected_mnemonics() {
        let r = generate_multi_block_softmax_ptx(2048, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("multi-block softmax PTX should generate");
        let fin = &r.finalize_ptx;
        assert!(fin.contains(".entry softmax_mb_finalize_f32_s1024"));
        assert!(fin.contains("ld.global.f32"));
        assert!(fin.contains("st.global.f32"));
        assert!(fin.contains("rcp.approx.f32"));
        // The rescale step is the load-bearing piece -- verify the FMA
        // emission is present.
        assert!(fin.contains("fma.rn.f32"));
        assert!(fin.contains("ex2.approx.f32"));
        assert!(fin.contains("bar.sync 0"));
    }

    #[test]
    fn multi_block_kernel_names() {
        let r = generate_multi_block_softmax_ptx(4096, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("multi-block softmax PTX should generate");
        assert_eq!(r.reduce_kernel_name(), "softmax_mb_reduce_f32_s1024");
        assert_eq!(r.finalize_kernel_name(), "softmax_mb_finalize_f32_s1024");
    }

    #[test]
    fn multi_block_for_8192_row() {
        let r = generate_multi_block_softmax_ptx(8192, 1024, 256, PtxType::F32, SmVersion::Sm80)
            .expect("8192-element multi-block softmax PTX should generate");
        assert_eq!(r.num_blocks_per_row, 8);
        assert_eq!(r.scratch_bytes_per_row, 8 * 2 * 4);
    }
}
