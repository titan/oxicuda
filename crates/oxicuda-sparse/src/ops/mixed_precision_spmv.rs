//! Mixed-precision sparse matrix-vector multiplication (SpMV).
//!
//! Provides FP16/BF16 storage with FP32 (or FP64) accumulation for
//! memory-bandwidth-bound SpMV operations. This approach halves the memory
//! footprint of matrix values while maintaining higher numerical quality during
//! the dot-product accumulation, yielding up to 2x bandwidth savings on
//! bandwidth-limited GPUs. The accumulation precision is selected via
//! [`ComputePrecision`]; the dense vectors `x`/`y` and the `alpha`/`beta`
//! scalars are stored in that compute precision.
//!
//! Three kernel strategies are available:
//! - **Scalar**: one thread per row, FP16 load -> FP32 FMA
//! - **Vector**: one warp per row with FP32 shuffle reduction
//! - **VectorPacked**: loads two FP16 values per 32-bit memory transaction (2x bandwidth)
//! - **Auto**: auto-selects based on average nnz per row

use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Storage precision for matrix values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoragePrecision {
    /// IEEE 754 half-precision (FP16, 10-bit mantissa).
    Fp16,
    /// Brain floating-point (BF16, 7-bit mantissa, wider exponent range).
    Bf16,
}

impl StoragePrecision {
    /// Returns the PTX type corresponding to this storage precision.
    #[must_use]
    pub const fn ptx_type(self) -> PtxType {
        match self {
            Self::Fp16 => PtxType::F16,
            Self::Bf16 => PtxType::BF16,
        }
    }

    /// Returns the packed (x2) PTX type for this precision.
    #[must_use]
    pub const fn packed_ptx_type(self) -> PtxType {
        match self {
            Self::Fp16 => PtxType::F16x2,
            Self::Bf16 => PtxType::BF16x2,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> u32 {
        2
    }

    /// Mantissa bits (for error analysis).
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::Fp16 => 10,
            Self::Bf16 => 7,
        }
    }

    /// Returns the PTX type suffix string without the leading dot.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Fp16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// Compute/accumulation precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputePrecision {
    /// IEEE 754 single-precision (32-bit).
    Fp32,
    /// IEEE 754 double-precision (64-bit).
    Fp64,
}

impl ComputePrecision {
    /// Returns the PTX type for the compute precision.
    #[must_use]
    pub const fn ptx_type(self) -> PtxType {
        match self {
            Self::Fp32 => PtxType::F32,
            Self::Fp64 => PtxType::F64,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> u32 {
        match self {
            Self::Fp32 => 4,
            Self::Fp64 => 8,
        }
    }

    /// Returns the PTX suffix without the dot.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "f32",
            Self::Fp64 => "f64",
        }
    }

    /// Returns the bit-move suffix for `mov.bN` instructions.
    #[must_use]
    pub const fn bit_suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "b32",
            Self::Fp64 => "b64",
        }
    }
}

/// Algorithm selection for mixed-precision SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixedSpMVAlgo {
    /// One thread per row. Best for very sparse matrices (< 4 nnz/row).
    Scalar,
    /// One warp (32 threads) per row with FP32 warp shuffle reduction.
    Vector,
    /// Vector algorithm with packed 2xFP16 loads (doubles effective bandwidth).
    /// Each 32-bit load fetches two FP16 values simultaneously.
    VectorPacked,
    /// Automatically selects the best algorithm based on matrix structure.
    Auto,
}

/// Configuration for mixed-precision SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixedPrecisionConfig {
    /// Storage precision for matrix values (FP16 or BF16).
    pub storage_precision: StoragePrecision,
    /// Compute/accumulation precision (FP32 or FP64).
    pub compute_precision: ComputePrecision,
    /// Algorithm selection.
    pub algorithm: MixedSpMVAlgo,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

impl MixedPrecisionConfig {
    /// Creates a new configuration with FP16 storage and FP32 accumulation.
    #[must_use]
    pub const fn fp16_fp32(algo: MixedSpMVAlgo, sm: SmVersion) -> Self {
        Self {
            storage_precision: StoragePrecision::Fp16,
            compute_precision: ComputePrecision::Fp32,
            algorithm: algo,
            sm_version: sm,
        }
    }

    /// Creates a new configuration with BF16 storage and FP32 accumulation.
    #[must_use]
    pub const fn bf16_fp32(algo: MixedSpMVAlgo, sm: SmVersion) -> Self {
        Self {
            storage_precision: StoragePrecision::Bf16,
            compute_precision: ComputePrecision::Fp32,
            algorithm: algo,
            sm_version: sm,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution plan
// ---------------------------------------------------------------------------

/// Pre-computed execution plan for mixed-precision SpMV.
///
/// Contains the resolved configuration, estimated bandwidth savings, and
/// memory footprint information needed to launch the kernel.
#[derive(Debug, Clone)]
pub struct MixedPrecisionPlan {
    /// Resolved configuration (Auto replaced with concrete algorithm).
    pub config: MixedPrecisionConfig,
    /// Number of matrix rows.
    pub rows: u32,
    /// Number of matrix columns.
    pub cols: u32,
    /// Number of non-zero entries.
    pub nnz: u64,
    /// Memory footprint in bytes for the FP16/BF16 values array.
    pub values_bytes: u64,
    /// Memory footprint in bytes for the same values if stored in compute precision.
    pub values_bytes_full: u64,
}

impl MixedPrecisionPlan {
    /// Returns the bandwidth savings ratio (e.g., ~2.0 for FP16 vs FP32 values).
    ///
    /// This only accounts for the values array; row_offsets and col_indices remain
    /// unchanged as i32 arrays.
    #[must_use]
    pub fn bandwidth_savings_ratio(&self) -> f64 {
        if self.values_bytes == 0 {
            return 1.0;
        }
        self.values_bytes_full as f64 / self.values_bytes as f64
    }

    /// Estimates peak achievable GFLOPS assuming the kernel is perfectly
    /// memory-bandwidth-bound.
    ///
    /// Uses a simple roofline model: each non-zero requires one FMA (2 flops),
    /// plus the memory traffic to load the value, column index, and vector element.
    #[must_use]
    pub fn estimated_gflops(&self, peak_bandwidth_gb_s: f64) -> f64 {
        if self.nnz == 0 {
            return 0.0;
        }
        // Bytes per non-zero: storage_bytes (value) + 4 (col_idx) + compute_bytes (x[col])
        let bytes_per_nnz = self.config.storage_precision.element_bytes() as f64
            + 4.0
            + self.config.compute_precision.element_bytes() as f64;
        let total_bytes = bytes_per_nnz * self.nnz as f64;
        // 2 flops per nnz (one multiply, one add in FMA)
        let total_flops = 2.0 * self.nnz as f64;
        // bandwidth in bytes/s
        let bandwidth_bytes_s = peak_bandwidth_gb_s * 1e9;
        // time = bytes / bandwidth
        let time_s = total_bytes / bandwidth_bytes_s;
        // GFLOPS = flops / time / 1e9
        total_flops / time_s / 1e9
    }

    /// Returns the average non-zeros per row.
    #[must_use]
    pub fn avg_nnz_per_row(&self) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        self.nnz as f64 / self.rows as f64
    }
}

/// Performance statistics from a mixed-precision SpMV execution.
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    /// Elapsed wall-clock time in microseconds.
    pub elapsed_us: f64,
    /// Achieved GFLOPS (2 * nnz / elapsed).
    pub gflops: f64,
    /// Achieved bandwidth in GB/s.
    pub bandwidth_gb_s: f64,
    /// Estimated relative precision loss (upper bound).
    pub precision_loss_bound: f64,
}

// ---------------------------------------------------------------------------
// Threshold for auto algorithm selection
// ---------------------------------------------------------------------------

/// Average nnz/row threshold above which Vector is preferred over Scalar.
const VECTOR_THRESHOLD: f64 = 4.0;

/// Average nnz/row threshold above which VectorPacked is preferred over Vector.
const PACKED_THRESHOLD: f64 = 32.0;

/// Block size for scalar kernels.
const SCALAR_BLOCK_SIZE: u32 = 256;

/// Block size for vector/packed kernels (must be multiple of 32).
const VECTOR_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Plan creation
// ---------------------------------------------------------------------------

/// Creates an execution plan for mixed-precision SpMV.
///
/// Resolves the `Auto` algorithm based on the matrix dimensions, validates
/// the configuration, and computes memory/bandwidth estimates.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if the configuration is invalid
/// (e.g., BF16 on unsupported architecture, zero dimensions).
pub fn plan_mixed_precision_spmv(
    config: &MixedPrecisionConfig,
    rows: u32,
    cols: u32,
    nnz: u64,
) -> SparseResult<MixedPrecisionPlan> {
    validate_mixed_precision_config(config)?;

    let avg_nnz = if rows > 0 {
        nnz as f64 / rows as f64
    } else {
        0.0
    };

    // Resolve Auto algorithm
    let resolved_algo = match config.algorithm {
        MixedSpMVAlgo::Auto => {
            if avg_nnz >= PACKED_THRESHOLD {
                MixedSpMVAlgo::VectorPacked
            } else if avg_nnz >= VECTOR_THRESHOLD {
                MixedSpMVAlgo::Vector
            } else {
                MixedSpMVAlgo::Scalar
            }
        }
        other => other,
    };

    let resolved_config = MixedPrecisionConfig {
        algorithm: resolved_algo,
        ..*config
    };

    let values_bytes = nnz * config.storage_precision.element_bytes() as u64;
    let values_bytes_full = nnz * config.compute_precision.element_bytes() as u64;

    Ok(MixedPrecisionPlan {
        config: resolved_config,
        rows,
        cols,
        nnz,
        values_bytes,
        values_bytes_full,
    })
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

/// Validates a mixed-precision configuration.
///
/// Checks that BF16 is only used on architectures that support it (sm_80+)
/// and that the compute precision is compatible.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if the config is invalid.
pub fn validate_mixed_precision_config(config: &MixedPrecisionConfig) -> SparseResult<()> {
    // BF16 requires Ampere (sm_80) or newer
    if config.storage_precision == StoragePrecision::Bf16 {
        let (major, _minor) = config.sm_version.ptx_isa_version();
        if major < 7 {
            return Err(SparseError::InvalidArgument(
                "BF16 storage requires sm_80 (Ampere) or newer; \
                 the selected SM version does not support BF16 instructions"
                    .to_string(),
            ));
        }
    }

    // Both FP32 and FP64 compute precisions are fully supported: the kernels
    // accumulate in the compute precision (FP64 widens FP16/BF16 storage via the
    // appropriate `cvt`, using an f32 intermediate for `bf16 -> f64` on pre-sm_90
    // targets). FP64 compute is rarely beneficial for bandwidth-bound SpMV but is
    // numerically valid for either storage format, so no configuration is
    // rejected here.

    Ok(())
}

// ---------------------------------------------------------------------------
// Precision loss estimation
// ---------------------------------------------------------------------------

/// Estimates the theoretical relative error bound for mixed-precision SpMV.
///
/// For a dot product of length `n` with FP16 storage and FP32 accumulation,
/// the relative error is bounded by:
///
///   `n * eps_storage + n * eps_compute`
///
/// where `eps_storage = 2^{-(p+1)}` for `p` mantissa bits, and
/// `eps_compute = 2^{-24}` for FP32.
///
/// This is a pessimistic (worst-case) bound; typical error is much smaller.
#[must_use]
pub fn estimate_precision_loss(nnz_per_row: f64, storage: StoragePrecision) -> f64 {
    let eps_storage = match storage {
        StoragePrecision::Fp16 => f64::powi(2.0, -11), // 2^-11 for 10-bit mantissa
        StoragePrecision::Bf16 => f64::powi(2.0, -8),  // 2^-8 for 7-bit mantissa
    };
    let eps_compute: f64 = f64::powi(2.0, -24); // FP32

    // Standard floating-point error accumulation bound for length-n dot product
    nnz_per_row * eps_storage + nnz_per_row * eps_compute
}

// ---------------------------------------------------------------------------
// Compute-precision-aware emission helpers
//
// The mixed-precision kernels accumulate in `compute` precision (FP32 or
// FP64). Storage (the matrix values) is always FP16/BF16, but the dense
// vectors `x`/`y` and the scalars `alpha`/`beta` live in the compute
// precision. These helpers dispatch the float-typed instructions on the
// runtime `ComputePrecision` so the same kernel body emits correct PTX for
// both widths. Crucially this avoids the two latent codegen bugs: a 32-bit
// `0F…` zero literal / `mov.b64` from a u32 bit-pattern (size mismatch) and
// `shfl.sync.down.b64` (ptxas only accepts `.b32`).
// ---------------------------------------------------------------------------

/// PTX type of the parameter carrying an `alpha`/`beta` bit pattern.
const fn mp_bits_param_ty(compute: ComputePrecision) -> PtxType {
    match compute {
        ComputePrecision::Fp32 => PtxType::U32,
        ComputePrecision::Fp64 => PtxType::U64,
    }
}

/// Loads an `alpha`/`beta` bit-pattern parameter in the correct width.
fn mp_load_bits_param(b: &mut BodyBuilder<'_>, compute: ComputePrecision, name: &str) -> Register {
    match compute {
        ComputePrecision::Fp32 => b.load_param_u32(name),
        ComputePrecision::Fp64 => b.load_param_u64(name),
    }
}

/// Hex literal for a `+0.0` constant in the compute precision (`0F…`/`0D…`).
const fn mp_zero_literal(compute: ComputePrecision) -> &'static str {
    match compute {
        ComputePrecision::Fp32 => "0F00000000",
        ComputePrecision::Fp64 => "0D0000000000000000",
    }
}

/// Loads a compute-precision dense-vector element from global memory.
fn mp_load_compute(b: &mut BodyBuilder<'_>, compute: ComputePrecision, addr: Register) -> Register {
    match compute {
        ComputePrecision::Fp32 => b.load_global_f32(addr),
        ComputePrecision::Fp64 => b.load_global_f64(addr),
    }
}

/// Stores a compute-precision dense-vector element to global memory.
fn mp_store_compute(
    b: &mut BodyBuilder<'_>,
    compute: ComputePrecision,
    addr: Register,
    val: Register,
) {
    match compute {
        ComputePrecision::Fp32 => b.store_global_f32(addr, val),
        ComputePrecision::Fp64 => b.store_global_f64(addr, val),
    }
}

/// Emits a compute-precision fused multiply-add `a * x + c`.
fn mp_fma_compute(
    b: &mut BodyBuilder<'_>,
    compute: ComputePrecision,
    a: Register,
    x: Register,
    c: Register,
) -> Register {
    match compute {
        ComputePrecision::Fp32 => b.fma_f32(a, x, c),
        ComputePrecision::Fp64 => b.fma_f64(a, x, c),
    }
}

/// Emits one warp down-shuffle of a compute-precision register.
///
/// Delegates to [`crate::ptx_helpers::emit_shfl_float`], which performs the
/// f64 unpack/shuffle/repack so no `shfl.sync.down.b64` (rejected by ptxas)
/// is ever emitted.
fn mp_shfl_down_compute(
    b: &mut BodyBuilder<'_>,
    compute: ComputePrecision,
    val: Register,
    offset: u32,
) -> Register {
    let off = offset.to_string();
    match compute {
        ComputePrecision::Fp32 => {
            crate::ptx_helpers::emit_shfl_float::<f32>(b, "down", val, &off, "31")
        }
        ComputePrecision::Fp64 => {
            crate::ptx_helpers::emit_shfl_float::<f64>(b, "down", val, &off, "31")
        }
    }
}

/// Builds the PTX fragment that converts a half value held in the scoped
/// `.b16` register `src` into the compute-precision register `dst`.
///
/// `widen` is the name of a scoped `.f32` temporary used only for the
/// `bf16 -> f64` path, which sm_86 cannot do directly (`cvt.f64.bf16` requires
/// sm_90+), so the value is widened through f32. The returned fragment is meant
/// to be embedded inside a `{ … }` register scope that declares `src`.
fn mp_half_cvt_fragment(
    storage: StoragePrecision,
    compute: ComputePrecision,
    src: &str,
    dst: &Register,
    widen: &str,
) -> String {
    let ss = storage.suffix();
    match (storage, compute) {
        (_, ComputePrecision::Fp32) => format!("cvt.f32.{ss} {dst}, {src};"),
        (StoragePrecision::Fp16, ComputePrecision::Fp64) => {
            format!("cvt.f64.f16 {dst}, {src};")
        }
        (StoragePrecision::Bf16, ComputePrecision::Fp64) => {
            format!(".reg .f32 {widen}; cvt.f32.bf16 {widen}, {src}; cvt.f64.f32 {dst}, {widen};")
        }
    }
}

/// Loads a half-precision (FP16/BF16) matrix value from `[addr]` and converts
/// it to the compute precision, returning the compute-typed register.
///
/// The 16-bit value is staged through a *scoped* `.b16` register declared
/// inline (`{ .reg .b16 … }`). This is required because the PTX register
/// allocator places every float register in a single `%f` class with one
/// declared width, so a half value routed through the allocator would be
/// declared at f32/f64 width and could not be the destination of a 16-bit
/// load. PTX also has no `ld.*.f16` type, so the raw bits are read with
/// `ld.global.b16` and reinterpreted by `cvt`.
fn mp_load_half(
    b: &mut BodyBuilder<'_>,
    storage: StoragePrecision,
    compute: ComputePrecision,
    addr: &Register,
) -> Register {
    let dst = b.alloc_reg(compute.ptx_type());
    let frag = mp_half_cvt_fragment(storage, compute, "%mphalf", &dst, "%mpw");
    b.raw_ptx(&format!(
        "{{ .reg .b16 %mphalf; ld.global.b16 %mphalf, [{addr}]; {frag} }}"
    ));
    dst
}

/// Splits a packed 32-bit register holding two adjacent half values into the
/// low and high compute-precision values (the packed 2xFP16/BF16 load path).
///
/// Uses a scoped block that declares two `.b16` halves, splits the packed
/// register with `mov.b32 {lo, hi}, packed`, and converts each half. Distinct
/// widen-temp names keep the optional `bf16 -> f64` f32 temporaries from
/// colliding within the single scope.
fn mp_unpack_pair(
    b: &mut BodyBuilder<'_>,
    storage: StoragePrecision,
    compute: ComputePrecision,
    packed: &Register,
) -> (Register, Register) {
    let lo = b.alloc_reg(compute.ptx_type());
    let hi = b.alloc_reg(compute.ptx_type());
    let frag_lo = mp_half_cvt_fragment(storage, compute, "%mplo", &lo, "%mpwlo");
    let frag_hi = mp_half_cvt_fragment(storage, compute, "%mphi", &hi, "%mpwhi");
    b.raw_ptx(&format!(
        "{{ .reg .b16 %mplo, %mphi; mov.b32 {{%mplo, %mphi}}, {packed}; {frag_lo} {frag_hi} }}"
    ));
    (lo, hi)
}

// ---------------------------------------------------------------------------
// PTX Generation: Scalar kernel
// ---------------------------------------------------------------------------

/// Generates PTX for scalar mixed-precision SpMV (one thread per row).
///
/// Each thread loads FP16/BF16 values from global memory, converts them
/// to the compute precision, performs FMA accumulation in that precision, and
/// writes the result back in the compute precision.
///
/// Kernel signature:
/// ```text
/// .entry mixed_spmv_scalar(
///     .param .u64 row_ptr,     // i32 CSR row offsets
///     .param .u64 col_idx,     // i32 column indices
///     .param .u64 values,      // FP16/BF16 values
///     .param .u64 x_ptr,       // dense vector x (compute precision)
///     .param .u64 y_ptr,       // dense vector y (compute precision)
///     .param .uN  alpha_bits,  // alpha bit-pattern (u32 for FP32, u64 for FP64)
///     .param .uN  beta_bits,   // beta bit-pattern  (u32 for FP32, u64 for FP64)
///     .param .u32 num_rows
/// )
/// ```
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_mixed_scalar_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_scalar")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", mp_bits_param_ty(compute))
        .param("beta_bits", mp_bits_param_ty(compute))
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                // Load alpha/beta as FP32 from bit patterns
                let alpha_bits = mp_load_bits_param(b, compute, "alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = mp_load_bits_param(b, compute, "beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row_ptr[row] and row_ptr[row+1]
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start = b.load_global_i32(rp_addr);

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end = b.load_global_i32(rp_addr_next);

                // Initialize FP32 accumulator to 0
                let acc = b.alloc_reg(compute.ptx_type());
                // Compute-precision +0.0 literal (`0F…`/`0D…`); a bare `0F`
                // literal with `mov.b64` would be a width mismatch under ptxas.
                let zero_lit = mp_zero_literal(compute);
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, {zero_lit};"));

                // Loop setup
                let loop_label = b.fresh_label("mpspmv_loop");
                let done_label = b.fresh_label("mpspmv_done");

                let k = b.alloc_reg(PtxType::U32);
                let rs_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {rs_u32}, {row_start};"));
                b.raw_ptx(&format!("mov.u32 {k}, {rs_u32};"));

                let re_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {re_u32}, {row_end};"));

                b.label(&loop_label);
                // Exit when k >= row_end (inverted skip-branch via branch_if so
                // the `$`-prefixed label target matches the `b.label` def).
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {pred}, {k}, {re_u32};"));
                b.branch_if(pred, &done_label);

                // Load col_idx[k]
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col};"));

                // Load FP16/BF16 value and convert to the compute precision.
                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val_fp32 = mp_load_half(b, storage, compute, &v_addr);

                // Load x[col] (compute precision)
                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, compute.element_bytes());
                let x_val = mp_load_compute(b, compute, x_addr);

                // FMA: acc += val_fp32 * x_val
                let new_acc = mp_fma_compute(b, compute, val_fp32, x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {new_acc};"));

                // k++
                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = mp_load_compute(b, compute, y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {acc};"
                ));

                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));

                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                mp_store_compute(b, compute, y_addr, result);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// PTX Generation: Vector kernel (warp-parallel)
// ---------------------------------------------------------------------------

/// Generates PTX for vector mixed-precision SpMV (one warp per row).
///
/// Each warp cooperatively processes one row. Lanes stride over the non-zeros,
/// load FP16/BF16, convert to FP32, accumulate, and finally reduce via warp
/// shuffles. Lane 0 writes the final result.
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_mixed_vector_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_vector")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", mp_bits_param_ty(compute))
        .param("beta_bits", mp_bits_param_ty(compute))
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let tid_global = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            // Lane within warp
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            // Warp ID
            let warp_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_id}, {tid_global}, 5;"));

            let warp_id_inner = warp_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(warp_id, num_rows, move |b| {
                let row = warp_id_inner;
                let lane = lane_inner;

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                let alpha_bits = mp_load_bits_param(b, compute, "alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = mp_load_bits_param(b, compute, "beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start_i32 = b.load_global_i32(rp_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {row_start_i32};"));

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end_i32 = b.load_global_i32(rp_addr_next);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {row_end_i32};"));

                // Each lane starts at row_start + lane, stride 32
                let acc = b.alloc_reg(compute.ptx_type());
                // Compute-precision +0.0 literal (`0F…`/`0D…`); a bare `0F`
                // literal with `mov.b64` would be a width mismatch under ptxas.
                let zero_lit = mp_zero_literal(compute);
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, {zero_lit};"));

                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {k}, {row_start}, {lane};"));

                let loop_label = b.fresh_label("mpspmv_vloop");
                let done_label = b.fresh_label("mpspmv_vdone");

                b.label(&loop_label);
                // Exit when k >= row_end (inverted skip-branch via branch_if).
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {pred}, {k}, {row_end};"));
                b.branch_if(pred, &done_label);

                // Load col and half-precision value
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col_i32};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val_fp32 = mp_load_half(b, storage, compute, &v_addr);

                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, compute.element_bytes());
                let x_val = mp_load_compute(b, compute, x_addr);

                // FMA accumulate
                let new_acc = mp_fma_compute(b, compute, val_fp32, x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {new_acc};"));

                // k += 32
                b.raw_ptx(&format!("add.u32 {k}, {k}, 32;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // Warp shuffle reduction in the compute precision. For FP64
                // `mp_shfl_down_compute` unpacks into two b32 halves so no
                // `shfl.sync.down.b64` (rejected by ptxas) is emitted.
                let mut current = acc;
                for offset in [16u32, 8, 4, 2, 1] {
                    let shuffled = mp_shfl_down_compute(b, compute, current.clone(), offset);
                    let sum = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "add.{compute_suffix} {sum}, {current}, {shuffled};"
                    ));
                    current = sum;
                }

                // Only lane 0 writes the result; other lanes skip ahead.
                // Inverted skip-branch (`setp.eq` -> `setp.ne`).
                let not_lane_0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {not_lane_0}, {lane}, 0;"));
                let skip_label = b.fresh_label("mpspmv_skip");
                b.branch_if(not_lane_0, &skip_label);

                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = mp_load_compute(b, compute, y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {current};"
                ));
                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));
                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                mp_store_compute(b, compute, y_addr, result);

                b.label(&skip_label);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// PTX Generation: Packed Vector kernel (2xFP16 loads)
// ---------------------------------------------------------------------------

/// Generates PTX for packed vector mixed-precision SpMV.
///
/// This kernel loads two FP16 values per 32-bit memory transaction using
/// the `f16x2` type, effectively doubling the bandwidth utilization for
/// the values array. Each pair of FP16 values is unpacked, converted to
/// FP32, and accumulated separately.
///
/// The kernel handles odd-length rows by processing the last element
/// individually when the row has an odd number of non-zeros.
///
/// # Errors
///
/// Returns [`PtxGenError`] if kernel assembly fails.
pub fn generate_packed_vector_spmv_ptx(
    config: &MixedPrecisionConfig,
) -> Result<String, PtxGenError> {
    let storage = config.storage_precision;
    let compute = config.compute_precision;
    let sm = config.sm_version;
    let compute_suffix = compute.suffix();
    let compute_bit = compute.bit_suffix();
    let elem_bytes = storage.element_bytes();

    KernelBuilder::new("mixed_spmv_packed")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", mp_bits_param_ty(compute))
        .param("beta_bits", mp_bits_param_ty(compute))
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let tid_global = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            let warp_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_id}, {tid_global}, 5;"));

            let warp_id_inner = warp_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(warp_id, num_rows, move |b| {
                let row = warp_id_inner;
                let lane = lane_inner;

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");

                let alpha_bits = mp_load_bits_param(b, compute, "alpha_bits");
                let alpha = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {alpha}, {alpha_bits};"));

                let beta_bits = mp_load_bits_param(b, compute, "beta_bits");
                let beta = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!("mov.{compute_bit} {beta}, {beta_bits};"));

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start_i32 = b.load_global_i32(rp_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {row_start_i32};"));

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end_i32 = b.load_global_i32(rp_addr_next);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {row_end_i32};"));

                // Compute nnz for this row and the "paired" end (rounded down to even)
                let nnz_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {nnz_row}, {row_end}, {row_start};"));
                let nnz_even = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {nnz_even}, {nnz_row}, 0xFFFFFFFE;"));
                let row_end_even = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_end_even}, {row_start}, {nnz_even};"));

                // Accumulator
                let acc = b.alloc_reg(compute.ptx_type());
                // Compute-precision +0.0 literal (`0F…`/`0D…`); a bare `0F`
                // literal with `mov.b64` would be a width mismatch under ptxas.
                let zero_lit = mp_zero_literal(compute);
                b.raw_ptx(&format!("mov.{compute_bit} {acc}, {zero_lit};"));

                // Packed loop: each lane processes pairs, stride = 32*2 = 64 elements
                // Lane processes element indices: row_start + lane*2, row_start + lane*2 + 64, ...
                let k = b.alloc_reg(PtxType::U32);
                let lane_x2 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {lane_x2}, {lane}, 1;"));
                b.raw_ptx(&format!("add.u32 {k}, {row_start}, {lane_x2};"));

                let packed_loop = b.fresh_label("mpspmv_packed_loop");
                let packed_done = b.fresh_label("mpspmv_packed_done");

                b.label(&packed_loop);
                let pred_pair = b.alloc_reg(PtxType::Pred);
                // k+1 must be < row_end_even to process a full pair
                let k_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {k_plus_1}, {k}, 1;"));
                // Exit the packed loop when k+1 > row_end_even (no full pair
                // remains). Inverted skip-branch (`setp.ls` -> `setp.hi`).
                b.raw_ptx(&format!(
                    "setp.hi.u32 {pred_pair}, {k_plus_1}, {row_end_even};"
                ));
                b.branch_if(pred_pair, &packed_done);

                // Load packed 2xFP16 as a 32-bit value
                // Address = values_base + k * 2 (each FP16 is 2 bytes)
                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let packed_val = b.alloc_reg(PtxType::B32);
                b.raw_ptx(&format!("ld.global.b32 {packed_val}, [{v_addr}];"));

                // Split the packed 32-bit register into the two half values and
                // convert each to the compute precision.
                let (val_lo_f32, val_hi_f32) = mp_unpack_pair(b, storage, compute, &packed_val);

                // Load two column indices and x values
                let ci_addr_0 = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_0_i32 = b.load_global_i32(ci_addr_0);
                let col_0 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_0}, {col_0_i32};"));

                let ci_addr_1 = b.byte_offset_addr(col_idx_base.clone(), k_plus_1.clone(), 4);
                let col_1_i32 = b.load_global_i32(ci_addr_1);
                let col_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_1}, {col_1_i32};"));

                let x_addr_0 = b.byte_offset_addr(x_ptr.clone(), col_0, compute.element_bytes());
                let x_val_0 = mp_load_compute(b, compute, x_addr_0);

                let x_addr_1 = b.byte_offset_addr(x_ptr.clone(), col_1, compute.element_bytes());
                let x_val_1 = mp_load_compute(b, compute, x_addr_1);

                // FMA: acc += val_lo * x0; acc += val_hi * x1
                let acc1 = mp_fma_compute(b, compute, val_lo_f32, x_val_0, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc1};"));
                let acc2 = mp_fma_compute(b, compute, val_hi_f32, x_val_1, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc2};"));

                // k += 64 (32 lanes * 2 elements per lane)
                b.raw_ptx(&format!("add.u32 {k}, {k}, 64;"));
                b.branch(&packed_loop);
                b.label(&packed_done);

                // Handle remainder: if nnz is odd, one element is left unpaired
                let has_remainder = b.alloc_reg(PtxType::Pred);
                let nnz_odd = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("and.b32 {nnz_odd}, {nnz_row}, 1;"));
                // Skip the remainder handling when nnz is even (nnz_odd == 0).
                // Inverted skip-branch (`setp.ne` -> `setp.eq`).
                b.raw_ptx(&format!("setp.eq.u32 {has_remainder}, {nnz_odd}, 0;"));

                let remainder_done = b.fresh_label("mpspmv_rem_done");
                b.branch_if(has_remainder, &remainder_done);

                // Only lane 0 handles the remainder element
                // Only lane 0 handles the remainder (inverted `setp.eq` ->
                // `setp.ne`): every other lane jumps straight to the reduction.
                let is_lane_0_rem = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {is_lane_0_rem}, {lane}, 0;"));
                b.branch_if(is_lane_0_rem, &remainder_done);

                // Last element index = row_end - 1
                let last_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("sub.u32 {last_idx}, {row_end}, 1;"));

                let last_v_addr =
                    b.byte_offset_addr(values_base.clone(), last_idx.clone(), elem_bytes);
                let last_val_f32 = mp_load_half(b, storage, compute, &last_v_addr);

                let last_ci_addr = b.byte_offset_addr(col_idx_base, last_idx, 4);
                let last_col_i32 = b.load_global_i32(last_ci_addr);
                let last_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {last_col}, {last_col_i32};"));

                let last_x_addr = b.byte_offset_addr(x_ptr, last_col, compute.element_bytes());
                let last_x_val = mp_load_compute(b, compute, last_x_addr);

                let acc_rem = mp_fma_compute(b, compute, last_val_f32, last_x_val, acc.clone());
                b.raw_ptx(&format!("mov.{compute_suffix} {acc}, {acc_rem};"));

                b.label(&remainder_done);

                // Warp shuffle reduction in the compute precision. For FP64
                // `mp_shfl_down_compute` unpacks into two b32 halves so no
                // `shfl.sync.down.b64` (rejected by ptxas) is emitted.
                let mut current = acc;
                for offset in [16u32, 8, 4, 2, 1] {
                    let shuffled = mp_shfl_down_compute(b, compute, current.clone(), offset);
                    let sum = b.alloc_reg(compute.ptx_type());
                    b.raw_ptx(&format!(
                        "add.{compute_suffix} {sum}, {current}, {shuffled};"
                    ));
                    current = sum;
                }

                // Only lane 0 writes the final result (inverted `setp.eq` ->
                // `setp.ne`).
                let not_lane_0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {not_lane_0}, {lane}, 0;"));
                let skip_label = b.fresh_label("mpspmv_pskip");
                b.branch_if(not_lane_0, &skip_label);

                let y_addr = b.byte_offset_addr(y_ptr, row, compute.element_bytes());
                let y_old = mp_load_compute(b, compute, y_addr.clone());

                let alpha_acc = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {alpha_acc}, {alpha}, {current};"
                ));
                let beta_y = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "mul.rn.{compute_suffix} {beta_y}, {beta}, {y_old};"
                ));
                let result = b.alloc_reg(compute.ptx_type());
                b.raw_ptx(&format!(
                    "add.{compute_suffix} {result}, {alpha_acc}, {beta_y};"
                ));

                mp_store_compute(b, compute, y_addr, result);

                b.label(&skip_label);
            });

            b.ret();
        })
        .build()
}

// ---------------------------------------------------------------------------
// Helper: resolve block/grid sizes
// ---------------------------------------------------------------------------

/// Returns `(grid_size, block_size)` for a scalar kernel.
#[must_use]
pub fn scalar_launch_params(num_rows: u32) -> (u32, u32) {
    let block = SCALAR_BLOCK_SIZE;
    let grid = num_rows.div_ceil(block);
    (grid, block)
}

/// Returns `(grid_size, block_size)` for a vector/packed kernel.
#[must_use]
pub fn vector_launch_params(num_rows: u32) -> (u32, u32) {
    let block = VECTOR_BLOCK_SIZE;
    let warps_per_block = block / 32;
    let grid = num_rows.div_ceil(warps_per_block);
    (grid, block)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ptx_helpers::test_support::assert_assembles_and_clean;

    fn default_fp16_config(algo: MixedSpMVAlgo) -> MixedPrecisionConfig {
        MixedPrecisionConfig::fp16_fp32(algo, SmVersion::Sm80)
    }

    /// All three mixed-precision kernels must assemble for sm_86 with both FP32
    /// and FP64 compute precision (and FP16/BF16 storage). The FP64 compute path
    /// is the regression guard: it must use `0D` zero literals, `mov.b64` from a
    /// `u64` bit-pattern parameter, f64 fma/load/store, and a warp reduction that
    /// splits into `.b32` halves (never `shfl.sync.down.b64`).
    #[test]
    fn mixed_precision_all_kernels_assemble_sm86() {
        for storage in [StoragePrecision::Fp16, StoragePrecision::Bf16] {
            for compute in [ComputePrecision::Fp32, ComputePrecision::Fp64] {
                for algo in [
                    MixedSpMVAlgo::Scalar,
                    MixedSpMVAlgo::Vector,
                    MixedSpMVAlgo::VectorPacked,
                ] {
                    let config = MixedPrecisionConfig {
                        storage_precision: storage,
                        compute_precision: compute,
                        algorithm: algo,
                        sm_version: SmVersion::Sm86,
                    };
                    let (ptx, kernel) = match algo {
                        MixedSpMVAlgo::Scalar => (
                            generate_mixed_scalar_spmv_ptx(&config).expect("scalar PTX"),
                            "mixed_scalar",
                        ),
                        MixedSpMVAlgo::Vector => (
                            generate_mixed_vector_spmv_ptx(&config).expect("vector PTX"),
                            "mixed_vector",
                        ),
                        MixedSpMVAlgo::VectorPacked => (
                            generate_packed_vector_spmv_ptx(&config).expect("packed PTX"),
                            "mixed_packed",
                        ),
                        MixedSpMVAlgo::Auto => unreachable!("Auto is resolved before codegen"),
                    };
                    let tag = format!("{kernel}_{}_{}", storage.suffix(), compute.suffix());
                    assert_assembles_and_clean(&tag, &ptx);
                    if compute == ComputePrecision::Fp64 {
                        assert!(
                            !ptx.contains("0F00000000"),
                            "{tag}: FP64 compute must not materialize an f32 0.0 immediate:\n{ptx}"
                        );
                        assert!(
                            ptx.contains("fma.rn.f64"),
                            "{tag}: FP64 compute must use f64 FMA:\n{ptx}"
                        );
                    }
                }
            }
        }
    }

    fn default_bf16_config(algo: MixedSpMVAlgo) -> MixedPrecisionConfig {
        MixedPrecisionConfig::bf16_fp32(algo, SmVersion::Sm80)
    }

    // --- PTX Generation tests ---

    #[test]
    fn scalar_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_scalar"));
        assert!(ptx.contains(".target sm_80"));
        // Should contain FP16 -> FP32 conversion
        assert!(ptx.contains("cvt.f32.f16"));
    }

    #[test]
    fn scalar_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::Scalar);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
    }

    #[test]
    fn vector_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::Vector);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_vector"));
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("cvt.f32.f16"));
    }

    #[test]
    fn vector_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::Vector);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn packed_ptx_fp16_generates() {
        let config = default_fp16_config(MixedSpMVAlgo::VectorPacked);
        let ptx = generate_packed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains(".entry mixed_spmv_packed"));
        // Should use 32-bit loads for packed FP16
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn packed_ptx_bf16_generates() {
        let config = default_bf16_config(MixedSpMVAlgo::VectorPacked);
        let ptx = generate_packed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX generation failed: {ptx:?}");
        let ptx = ptx.expect("test");
        assert!(ptx.contains("cvt.f32.bf16"));
    }

    // --- Config validation tests ---

    #[test]
    fn validate_fp16_on_turing() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm75);
        assert!(validate_mixed_precision_config(&config).is_ok());
    }

    #[test]
    fn validate_bf16_on_turing_fails() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm75);
        let result = validate_mixed_precision_config(&config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("test"));
        assert!(err_msg.contains("BF16"));
    }

    #[test]
    fn validate_bf16_on_ampere_ok() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Vector, SmVersion::Sm80);
        assert!(validate_mixed_precision_config(&config).is_ok());
    }

    // --- Plan tests ---

    #[test]
    fn plan_auto_selects_scalar_for_sparse() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 2000 nnz => avg 2 nnz/row => should select Scalar
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 2000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::Scalar);
    }

    #[test]
    fn plan_auto_selects_vector_for_moderate() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 10000 nnz => avg 10 nnz/row => should select Vector
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 10000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::Vector);
    }

    #[test]
    fn plan_auto_selects_packed_for_dense() {
        let config = default_fp16_config(MixedSpMVAlgo::Auto);
        // 1000 rows, 50000 nnz => avg 50 nnz/row => should select VectorPacked
        let plan = plan_mixed_precision_spmv(&config, 1000, 5000, 50000);
        assert!(plan.is_ok());
        let plan = plan.expect("test");
        assert_eq!(plan.config.algorithm, MixedSpMVAlgo::VectorPacked);
    }

    // --- Bandwidth estimation tests ---

    #[test]
    fn bandwidth_savings_fp16_vs_fp32() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        let plan = plan_mixed_precision_spmv(&config, 1000, 1000, 10000).expect("test");
        // FP16 = 2 bytes, FP32 = 4 bytes => ratio = 2.0
        let ratio = plan.bandwidth_savings_ratio();
        assert!((ratio - 2.0).abs() < 1e-10, "Expected ~2.0, got {ratio}");
    }

    #[test]
    fn estimated_gflops_positive() {
        let config = default_fp16_config(MixedSpMVAlgo::Vector);
        let plan = plan_mixed_precision_spmv(&config, 10000, 10000, 100000).expect("test");
        // With 1000 GB/s peak bandwidth (A100-like)
        let gflops = plan.estimated_gflops(1000.0);
        assert!(gflops > 0.0, "Expected positive GFLOPS, got {gflops}");
        // Sanity: should be in a reasonable range for roofline
        assert!(gflops < 1000.0, "GFLOPS suspiciously high: {gflops}");
    }

    // --- Precision loss estimation tests ---

    #[test]
    fn precision_loss_fp16_bounded() {
        let loss = estimate_precision_loss(100.0, StoragePrecision::Fp16);
        // eps_fp16 = 2^-11 ~= 4.88e-4, eps_fp32 = 2^-24 ~= 5.96e-8
        // bound = 100 * (4.88e-4 + 5.96e-8) ~ 0.0488
        assert!(loss > 0.0);
        assert!(loss < 0.1, "Precision loss bound too large: {loss}");
    }

    #[test]
    fn precision_loss_bf16_larger_than_fp16() {
        let loss_fp16 = estimate_precision_loss(50.0, StoragePrecision::Fp16);
        let loss_bf16 = estimate_precision_loss(50.0, StoragePrecision::Bf16);
        // BF16 has fewer mantissa bits, so larger error bound
        assert!(
            loss_bf16 > loss_fp16,
            "BF16 loss ({loss_bf16}) should exceed FP16 loss ({loss_fp16})"
        );
    }

    // --- Launch parameter tests ---

    #[test]
    fn scalar_launch_params_correct() {
        let (grid, block) = scalar_launch_params(1000);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn vector_launch_params_correct() {
        let (grid, block) = vector_launch_params(1000);
        assert_eq!(block, 256);
        // warps_per_block = 256/32 = 8, grid = ceil(1000/8) = 125
        assert_eq!(grid, 125);
    }

    // --- Mixed-precision PTX instruction accuracy tests ---

    /// Verify that scalar FP16→FP32 kernel contains `cvt.f32.f16` conversion.
    ///
    /// This ensures FP16 values are widened to FP32 before accumulation,
    /// which is the core requirement for mixed-precision SpMV numerical quality.
    #[test]
    fn mixed_precision_scalar_ptx_contains_fp16_to_fp32_conversion() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // Must contain FP16 → FP32 widening conversion
        assert!(
            ptx.contains("cvt.f32.f16"),
            "scalar kernel must contain cvt.f32.f16 for FP16→FP32 widening"
        );
    }

    /// Verify that scalar FP16→FP32 kernel contains `fma.rn.f32` accumulation.
    ///
    /// The round-to-nearest FMA is required for correct FP32 accumulation quality.
    #[test]
    fn mixed_precision_scalar_ptx_contains_fma_rn_f32_accumulation() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // Must contain FP32 FMA with round-to-nearest mode for accuracy
        assert!(
            ptx.contains("fma.rn.f32"),
            "scalar kernel must contain fma.rn.f32 for FP32 accumulation"
        );
    }

    /// Verify vector FP16→FP32 kernel contains both conversion and FMA instructions.
    #[test]
    fn mixed_precision_vector_ptx_contains_conversion_and_fma() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Vector, SmVersion::Sm80);
        let ptx = generate_mixed_vector_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        assert!(
            ptx.contains("cvt.f32.f16"),
            "vector kernel must contain cvt.f32.f16 for FP16→FP32 widening"
        );
        assert!(
            ptx.contains("fma.rn.f32"),
            "vector kernel must contain fma.rn.f32 for FP32 accumulation"
        );
    }

    /// Verify BF16→FP32 kernel uses `cvt.f32.bf16` (not `cvt.f32.f16`).
    #[test]
    fn mixed_precision_bf16_ptx_uses_bf16_conversion() {
        let config = MixedPrecisionConfig::bf16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        assert!(
            ptx.contains("cvt.f32.bf16"),
            "BF16 kernel must contain cvt.f32.bf16 for BF16→FP32 widening"
        );
        // BF16 kernel should NOT use the FP16 conversion instruction
        assert!(
            !ptx.contains("cvt.f32.f16"),
            "BF16 kernel must NOT contain cvt.f32.f16"
        );
    }

    /// Verify that `mul.rn.f32` is used for alpha/beta scaling (not just fma).
    #[test]
    fn mixed_precision_scalar_ptx_uses_rn_mode_for_scaling() {
        let config = MixedPrecisionConfig::fp16_fp32(MixedSpMVAlgo::Scalar, SmVersion::Sm80);
        let ptx = generate_mixed_scalar_spmv_ptx(&config);
        assert!(ptx.is_ok(), "PTX gen failed: {ptx:?}");
        let ptx = ptx.expect("test");

        // The alpha/beta scaling uses mul.rn.f32
        assert!(
            ptx.contains("mul.rn.f32"),
            "scalar kernel must contain mul.rn.f32 for alpha/beta scaling"
        );
    }

    /// Verify precision loss estimation is monotone: more nnz → larger error bound.
    #[test]
    fn precision_loss_monotone_in_nnz_per_row() {
        let nnz_values = [1.0_f64, 10.0, 100.0, 1000.0];
        let mut prev_loss = 0.0_f64;
        for &nnz in &nnz_values {
            let loss = estimate_precision_loss(nnz, StoragePrecision::Fp16);
            assert!(
                loss >= prev_loss,
                "precision loss should increase with nnz/row: nnz={nnz}, loss={loss}, prev={prev_loss}"
            );
            prev_loss = loss;
        }
    }

    /// Verify that the precision loss is proportional to nnz/row.
    ///
    /// From the formula: loss = nnz * (eps_storage + eps_compute),
    /// so doubling nnz should double the loss.
    #[test]
    fn precision_loss_linear_in_nnz_per_row() {
        let loss_10 = estimate_precision_loss(10.0, StoragePrecision::Fp16);
        let loss_20 = estimate_precision_loss(20.0, StoragePrecision::Fp16);

        // loss_20 should be exactly 2 * loss_10 (linear in nnz)
        assert!(
            (loss_20 - 2.0 * loss_10).abs() < 1e-15,
            "Precision loss should be linear: loss(20)={loss_20} should be 2*loss(10)={loss_10}"
        );
    }

    /// Verify that the bandwidth savings ratio for BF16 is also ~2x vs FP32.
    #[test]
    fn bandwidth_savings_bf16_vs_fp32() {
        let config = default_bf16_config(MixedSpMVAlgo::Scalar);
        let plan = plan_mixed_precision_spmv(&config, 1000, 1000, 10000).expect("test");
        // BF16 = 2 bytes, FP32 = 4 bytes => ratio = 2.0
        let ratio = plan.bandwidth_savings_ratio();
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "BF16 bandwidth savings ratio should be ~2.0, got {ratio}"
        );
    }

    /// Verify the avg_nnz_per_row calculation is correct.
    #[test]
    fn mixed_plan_avg_nnz_per_row_calculation() {
        let config = default_fp16_config(MixedSpMVAlgo::Scalar);
        // 100 rows, 500 nnz => avg = 5.0
        let plan = plan_mixed_precision_spmv(&config, 100, 1000, 500).expect("test");
        let avg = plan.avg_nnz_per_row();
        assert!(
            (avg - 5.0).abs() < 1e-10,
            "avg_nnz_per_row should be 5.0, got {avg}"
        );
    }
}
