//! Diagonal-block triangular-solve PTX kernel for [`trsm`](super::trsm::trsm).
//!
//! The blocked TRSM algorithm in [`super::trsm()`] decomposes a large solve
//! into a sequence of small diagonal-block solves interleaved with GEMM
//! trailing updates. This module emits the hand-written PTX kernel that
//! performs a single diagonal-block solve.
//!
//! # Kernel strategy
//!
//! The kernel solves an `bs x bs` triangular system against the
//! corresponding panel of B:
//!
//! - **Side::Left** — `op(A_blk) * X = B_blk`, where `B_blk` is `bs x ncols`.
//!   One thread owns one column of `B_blk` and performs an independent
//!   forward/back substitution down the `bs` pivot rows. A warp therefore
//!   solves 32 right-hand-side columns in lockstep — the warp-cooperative
//!   layout requested for `bs <= 32` (and it scales unchanged for larger
//!   `bs`, where the blocked driver recurses).
//! - **Side::Right** — `X * op(A_blk) = B_blk`, where `B_blk` is `nrows x bs`.
//!   One thread owns one row of `B_blk`.
//!
//! Substitution honours [`FillMode`] (upper/lower), [`Transpose`]
//! (`op(A) = A` or `op(A) = A^T`; `ConjTrans` equals `Trans` for the real
//! element types this kernel supports), and [`DiagType`] — a unit diagonal
//! skips the pivot divide entirely.
//!
//! Internal indexing of `A_blk` and `B_blk` is expressed through explicit
//! leading-dimension strides (`lda`, `ldb`) and a `row_stride` / `col_stride`
//! pair, so the same kernel serves both row-major and column-major operands.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::types::{DiagType, FillMode, Side, Transpose};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters describing a single diagonal-block triangular solve kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrsmKernelConfig {
    /// Target SM architecture.
    pub sm: SmVersion,
    /// Element PTX type (`F32` or `F64`).
    pub elem: PtxType,
    /// Whether the triangular matrix is the left or right operand.
    pub side: Side,
    /// Which triangle of A is stored.
    pub fill_mode: FillMode,
    /// Transpose mode applied to A.
    pub trans: Transpose,
    /// Whether A has an implicit unit diagonal.
    pub diag: DiagType,
}

impl TrsmKernelConfig {
    /// Returns `true` when `op(A)` reads the transpose of the stored matrix.
    #[must_use]
    pub fn op_transposed(&self) -> bool {
        matches!(self.trans, Transpose::Trans | Transpose::ConjTrans)
    }

    /// Returns `true` when `op(A)` is effectively *upper*-triangular.
    fn op_is_upper(&self) -> bool {
        // The stored triangle is flipped by a transpose.
        match self.fill_mode {
            FillMode::Upper => !self.op_transposed(),
            FillMode::Lower => self.op_transposed(),
            // `Full` is solved as a lower system (matches the legacy default).
            FillMode::Full => false,
        }
    }

    /// Returns `true` when the pivot loop iterates front-to-back (ascending
    /// pivot index), `false` for back substitution.
    ///
    /// - **Side::Left** solves `op(A) * X = B`. A lower-triangular `op(A)`
    ///   is solved forward (pivot row 0 first).
    /// - **Side::Right** solves `X * op(A) = B`. An upper-triangular `op(A)`
    ///   is solved forward (pivot column 0 first) — the mirror image of the
    ///   left case.
    #[must_use]
    pub fn forward(&self) -> bool {
        match self.side {
            Side::Left => !self.op_is_upper(),
            Side::Right => self.op_is_upper(),
        }
    }

    /// Returns `true` when the kernel must read A in transposed orientation
    /// for the rank-1 trailing update.
    ///
    /// Side::Left subtracts `op(A)[j, i] * x_i` (pivot is the second index);
    /// Side::Right subtracts `x_i * op(A)[i, j]` (pivot is the first index).
    /// The two access patterns differ by a transpose, which is folded in
    /// here together with the `op(A)` transpose itself.
    #[must_use]
    pub fn a_access_transposed(&self) -> bool {
        // `op(A)` transpose XOR the Left/Right access flip.
        self.op_transposed() ^ matches!(self.side, Side::Right)
    }

    /// A short, unique kernel name encoding every configuration axis.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let elem = if self.elem == PtxType::F64 {
            "f64"
        } else {
            "f32"
        };
        let side = match self.side {
            Side::Left => "l",
            Side::Right => "r",
        };
        let fill = match self.fill_mode {
            FillMode::Upper => "u",
            FillMode::Lower => "l",
            FillMode::Full => "f",
        };
        let trans = if self.op_transposed() { "t" } else { "n" };
        let diag = if self.diag == DiagType::Unit {
            "u"
        } else {
            "n"
        };
        format!(
            "trsm_diag_{elem}_{side}{fill}{trans}{diag}_{}",
            self.sm.as_ptx_str()
        )
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates the diagonal-block TRSM kernel PTX.
///
/// The kernel signature is:
///
/// ```text
/// (ptr_a: u64, ptr_b: u64,
///  bs: u32, vec_count: u32,
///  lda: u32, ldb: u32,
///  a_row_stride: u32, a_col_stride: u32,
///  b_row_stride: u32, b_col_stride: u32)
/// ```
///
/// `vec_count` is the number of independent right-hand sides (`ncols` of the
/// B panel for [`Side::Left`], `nrows` for [`Side::Right`]). The element
/// stride pairs let the caller point `ptr_a` / `ptr_b` directly at a
/// sub-matrix in either layout.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] if string formatting fails or the
/// element type is unsupported.
pub fn generate_trsm_diag_ptx(config: &TrsmKernelConfig) -> BlasResult<(String, String)> {
    if config.elem != PtxType::F32 && config.elem != PtxType::F64 {
        return Err(BlasError::PtxGeneration(
            "TRSM diagonal kernel supports only f32 and f64".into(),
        ));
    }

    let is_f64 = config.elem == PtxType::F64;
    let elem_ty = if is_f64 { "f64" } else { "f32" };
    let byte_size = if is_f64 { 8u32 } else { 4u32 };
    // Register file letter used by the floating-point bank.
    let fr = if is_f64 { "fd" } else { "f" };
    let kernel_name = config.kernel_name();
    let forward = config.forward();
    // Whether the rank-1 trailing update reads A transposed (folds in both
    // the `op(A)` transpose and the Side::Left/Right access flip).
    let transposed = config.a_access_transposed();
    let unit_diag = config.diag == DiagType::Unit;
    let left = config.side == Side::Left;

    let mut p = String::with_capacity(8192);

    wl(&mut p, &format!(".version {}", config.sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", config.sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 trsm_ptr_a,")?;
    wl(&mut p, "    .param .u64 trsm_ptr_b,")?;
    wl(&mut p, "    .param .u32 trsm_bs,")?;
    wl(&mut p, "    .param .u32 trsm_vec_count,")?;
    wl(&mut p, "    .param .u32 trsm_lda,")?;
    wl(&mut p, "    .param .u32 trsm_ldb,")?;
    wl(&mut p, "    .param .u32 trsm_a_row_stride,")?;
    wl(&mut p, "    .param .u32 trsm_a_col_stride,")?;
    wl(&mut p, "    .param .u32 trsm_b_row_stride,")?;
    wl(&mut p, "    .param .u32 trsm_b_col_stride")?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    // Register banks.
    wl(&mut p, "    .reg .pred %p<8>;")?;
    wl(&mut p, "    .reg .b32 %r<48>;")?;
    wl(&mut p, "    .reg .b64 %rd<32>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<16>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<16>;")?;
    }
    wl(&mut p, "")?;

    // --- Resolve this thread's right-hand-side index --------------------
    wl(&mut p, "    // vec = blockIdx.x * blockDim.x + threadIdx.x")?;
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %tid.x;")?;
    wl(
        &mut p,
        "    mad.lo.u32 %r0, %r1, %r2, %r3;   // r0 = vec index",
    )?;
    wl(&mut p, "    ld.param.u32 %r4, [trsm_vec_count];")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r4;")?;
    wl(&mut p, "    @%p0 bra $TRSM_RET;")?;
    wl(&mut p, "")?;

    // --- Load remaining scalar parameters -------------------------------
    wl(&mut p, "    ld.param.u64 %rd0, [trsm_ptr_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [trsm_ptr_b];")?;
    wl(&mut p, "    ld.param.u32 %r5, [trsm_bs];")?;
    wl(&mut p, "    ld.param.u32 %r6, [trsm_lda];")?;
    wl(&mut p, "    ld.param.u32 %r7, [trsm_ldb];")?;
    wl(&mut p, "    ld.param.u32 %r8,  [trsm_a_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r9,  [trsm_a_col_stride];")?;
    wl(&mut p, "    ld.param.u32 %r10, [trsm_b_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r11, [trsm_b_col_stride];")?;
    wl(&mut p, "")?;

    // Per-thread base element offset into B. The "vector axis" is columns
    // for Side::Left (one thread owns one column) and rows for Side::Right.
    if left {
        wl(
            &mut p,
            "    mul.lo.u32 %r12, %r0, %r11;   // vec * b_col_stride",
        )?;
    } else {
        wl(
            &mut p,
            "    mul.lo.u32 %r12, %r0, %r10;   // vec * b_row_stride",
        )?;
    }
    wl(&mut p, "")?;

    // --- Substitution loop over pivot rows i ----------------------------
    //
    // `%r20` is the pivot row index `i`. `%r21` is the inner index `j`.
    if forward {
        wl(&mut p, "    mov.u32 %r20, 0;")?;
    } else {
        wl(&mut p, "    sub.u32 %r20, %r5, 1;")?;
    }
    wl(&mut p, "$TRSM_PIVOT:")?;
    // `i >= bs` (unsigned) ends the loop in both directions: forward reaches
    // `bs`, and back substitution wraps `0` to `0xffffffff`, which is also
    // `>= bs` for any valid block size.
    wl(&mut p, "    setp.ge.u32 %p1, %r20, %r5;")?;
    wl(&mut p, "    @%p1 bra $TRSM_PIVOT_END;")?;
    wl(&mut p, "")?;

    // Address of B[pivot] for this thread.
    //   Left:  B[i, vec]  => i * b_row_stride + vec * b_col_stride
    //   Right: B[vec, i]  => vec * b_row_stride + i * b_col_stride
    if left {
        wl(
            &mut p,
            "    mul.lo.u32 %r22, %r20, %r10;   // i * b_row_stride",
        )?;
    } else {
        wl(
            &mut p,
            "    mul.lo.u32 %r22, %r20, %r11;   // i * b_col_stride",
        )?;
    }
    wl(
        &mut p,
        "    add.u32 %r22, %r22, %r12;       // + thread base",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd4, %r22;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd4, %rd4, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd5, %rd1, %rd4;       // &B[pivot]")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}0, [%rd5];   // x = B[pivot]"),
    )?;
    wl(&mut p, "")?;

    if !unit_diag {
        // Divide by op(A)[i, i]. The diagonal element is layout-symmetric:
        //   addr = i * (a_row_stride + a_col_stride)
        wl(
            &mut p,
            "    add.u32 %r23, %r8, %r9;        // a_row_stride+a_col_stride",
        )?;
        wl(
            &mut p,
            "    mul.lo.u32 %r23, %r23, %r20;   // i * diag_stride",
        )?;
        wl(&mut p, "    cvt.u64.u32 %rd6, %r23;")?;
        wl(&mut p, &format!("    mul.lo.u64 %rd6, %rd6, {byte_size};"))?;
        wl(&mut p, "    add.u64 %rd7, %rd0, %rd6;      // &A[i,i]")?;
        wl(
            &mut p,
            &format!("    ld.global.{elem_ty} %{fr}1, [%rd7];   // pivot diag"),
        )?;
        wl(
            &mut p,
            &format!("    div.rn.{elem_ty} %{fr}0, %{fr}0, %{fr}1;   // x /= A[i,i]"),
        )?;
    }

    // Store the solved component back to B[pivot].
    wl(
        &mut p,
        &format!("    st.global.{elem_ty} [%rd5], %{fr}0;   // B[pivot] = x"),
    )?;
    wl(&mut p, "")?;

    // --- Rank-1 trailing update of the not-yet-solved rows --------------
    //
    // For every remaining pivot row `j`, subtract op(A)[j, i] * x.
    //   forward : j = i+1 .. bs-1
    //   backward: j = i-1 .. 0
    if forward {
        wl(&mut p, "    add.u32 %r21, %r20, 1;")?;
    } else {
        wl(&mut p, "    sub.u32 %r21, %r20, 1;")?;
    }
    wl(&mut p, "$TRSM_INNER:")?;
    // Same unsigned `>= bs` guard: forward stops at `bs`, backward stops when
    // `j` wraps past `0`.
    wl(&mut p, "    setp.ge.u32 %p2, %r21, %r5;")?;
    wl(&mut p, "    @%p2 bra $TRSM_INNER_END;")?;
    wl(&mut p, "")?;

    // Address of op(A)[j, i].
    //   op(A)[j,i] = A[j,i]      when not transposed
    //   op(A)[j,i] = A[i,j]      when transposed
    if transposed {
        // A[i, j] => i * a_row_stride + j * a_col_stride
        wl(
            &mut p,
            "    mul.lo.u32 %r24, %r20, %r8;    // i * a_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r24, %r21, %r9, %r24;   // + j * a_col_stride",
        )?;
    } else {
        // A[j, i] => j * a_row_stride + i * a_col_stride
        wl(
            &mut p,
            "    mul.lo.u32 %r24, %r21, %r8;    // j * a_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r24, %r20, %r9, %r24;   // + i * a_col_stride",
        )?;
    }
    wl(&mut p, "    cvt.u64.u32 %rd8, %r24;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd8, %rd8, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd9, %rd0, %rd8;      // &op(A)[j,i]")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}2, [%rd9];   // a_ji"),
    )?;

    // Address of B[j] for this thread.
    if left {
        wl(
            &mut p,
            "    mul.lo.u32 %r25, %r21, %r10;   // j * b_row_stride",
        )?;
    } else {
        wl(
            &mut p,
            "    mul.lo.u32 %r25, %r21, %r11;   // j * b_col_stride",
        )?;
    }
    wl(
        &mut p,
        "    add.u32 %r25, %r25, %r12;      // + thread base",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd10, %r25;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd10, %rd10, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd11, %rd1, %rd10;    // &B[j]")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}3, [%rd11];   // b_j"),
    )?;

    // b_j = b_j - a_ji * x   (fused multiply-add with negated product)
    wl(&mut p, &format!("    neg.{elem_ty} %{fr}4, %{fr}2;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{elem_ty} %{fr}3, %{fr}4, %{fr}0, %{fr}3;"),
    )?;
    wl(&mut p, &format!("    st.global.{elem_ty} [%rd11], %{fr}3;"))?;
    wl(&mut p, "")?;

    if forward {
        wl(&mut p, "    add.u32 %r21, %r21, 1;")?;
    } else {
        wl(&mut p, "    sub.u32 %r21, %r21, 1;")?;
    }
    wl(&mut p, "    bra $TRSM_INNER;")?;
    wl(&mut p, "$TRSM_INNER_END:")?;
    wl(&mut p, "")?;

    if forward {
        wl(&mut p, "    add.u32 %r20, %r20, 1;")?;
    } else {
        wl(&mut p, "    sub.u32 %r20, %r20, 1;")?;
    }
    wl(&mut p, "    bra $TRSM_PIVOT;")?;
    wl(&mut p, "$TRSM_PIVOT_END:")?;
    wl(&mut p, "")?;
    wl(&mut p, "$TRSM_RET:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    let _ = (config.elem,);
    Ok((p, kernel_name))
}

// ---------------------------------------------------------------------------
// Matrix alpha-scale kernel
// ---------------------------------------------------------------------------

/// Generates a kernel that scales a dense `rows x cols` matrix in place by a
/// scalar: `M[r, c] *= alpha`.
///
/// The blocked TRSM applies `alpha` to the whole right-hand-side matrix B
/// exactly once, up front, so every subsequent diagonal solve and GEMM
/// trailing update can run with an implicit unit scalar. Explicit
/// `row_stride` / `col_stride` parameters let the kernel address a matrix in
/// either layout, honouring its leading dimension.
///
/// The kernel signature is `(ptr: u64, rows: u32, cols: u32,
/// row_stride: u32, col_stride: u32, alpha: <elem>)`. One thread scales one
/// element; the launch uses a flat 1-D grid over `rows * cols`.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure or an
/// unsupported element type.
pub fn generate_trsm_scale_ptx(sm: SmVersion, elem: PtxType) -> BlasResult<(String, String)> {
    if elem != PtxType::F32 && elem != PtxType::F64 {
        return Err(BlasError::PtxGeneration(
            "TRSM scale kernel supports only f32 and f64".into(),
        ));
    }
    let is_f64 = elem == PtxType::F64;
    let elem_ty = if is_f64 { "f64" } else { "f32" };
    let byte_size = if is_f64 { 8u32 } else { 4u32 };
    let fr = if is_f64 { "fd" } else { "f" };
    let kernel_name = format!(
        "trsm_scale_{}_{}",
        if is_f64 { "f64" } else { "f32" },
        sm.as_ptx_str()
    );

    let mut p = String::with_capacity(2048);
    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;
    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 scale_ptr,")?;
    wl(&mut p, "    .param .u32 scale_rows,")?;
    wl(&mut p, "    .param .u32 scale_cols,")?;
    wl(&mut p, "    .param .u32 scale_row_stride,")?;
    wl(&mut p, "    .param .u32 scale_col_stride,")?;
    wl(&mut p, &format!("    .param .{elem_ty} scale_alpha"))?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;
    wl(&mut p, "    .reg .pred %p<4>;")?;
    wl(&mut p, "    .reg .b32 %r<16>;")?;
    wl(&mut p, "    .reg .b64 %rd<8>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<4>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<4>;")?;
    }
    wl(&mut p, "")?;
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %tid.x;")?;
    wl(
        &mut p,
        "    mad.lo.u32 %r0, %r1, %r2, %r3;   // flat element id",
    )?;
    wl(&mut p, "    ld.param.u32 %r4, [scale_rows];")?;
    wl(&mut p, "    ld.param.u32 %r5, [scale_cols];")?;
    wl(
        &mut p,
        "    mul.lo.u32 %r6, %r4, %r5;        // total elements",
    )?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r6;")?;
    wl(&mut p, "    @%p0 bra $SCALE_RET;")?;
    wl(&mut p, "")?;
    // Decompose flat id into (row, col): row = id / cols, col = id % cols.
    wl(&mut p, "    div.u32 %r7, %r0, %r5;           // row")?;
    wl(&mut p, "    rem.u32 %r8, %r0, %r5;           // col")?;
    wl(&mut p, "    ld.param.u32 %r9,  [scale_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r10, [scale_col_stride];")?;
    wl(
        &mut p,
        "    mul.lo.u32 %r11, %r7, %r9;       // row*row_stride",
    )?;
    wl(
        &mut p,
        "    mad.lo.u32 %r11, %r8, %r10, %r11;   // + col*col_stride",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd2, %r11;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd2, %rd2, {byte_size};"))?;
    wl(&mut p, "    ld.param.u64 %rd0, [scale_ptr];")?;
    wl(&mut p, "    add.u64 %rd3, %rd0, %rd2;")?;
    wl(&mut p, &format!("    ld.global.{elem_ty} %{fr}0, [%rd3];"))?;
    wl(
        &mut p,
        &format!("    ld.param.{elem_ty} %{fr}1, [scale_alpha];"),
    )?;
    wl(
        &mut p,
        &format!("    mul.rn.{elem_ty} %{fr}0, %{fr}0, %{fr}1;"),
    )?;
    wl(&mut p, &format!("    st.global.{elem_ty} [%rd3], %{fr}0;"))?;
    wl(&mut p, "$SCALE_RET:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok((p, kernel_name))
}

// ---------------------------------------------------------------------------
// Strided matmul-accumulate kernel (trailing update)
// ---------------------------------------------------------------------------

/// Generates the trailing-update kernel:
/// `C[r, c] += alpha * sum_k LHS[r, k] * RHS[k, c]`.
///
/// This is the matrix-multiply used to subtract a just-solved block's
/// contribution from the unsolved part of B during blocked TRSM. It accepts
/// independent `(row_stride, col_stride)` pairs for C, LHS and RHS, so it
/// operates directly on strided sub-matrices in either layout — no scratch
/// packing is needed. A required transpose of an operand is expressed simply
/// by swapping that operand's stride pair at the call site.
///
/// One thread computes one `C[r, c]` for `r in 0..m`, `c in 0..n`, over a
/// flat 1-D grid of `m * n` threads. Each thread reads `C[r, c]` exactly
/// once and writes it once, so the in-place accumulate is race-free.
///
/// The kernel signature is:
///
/// ```text
/// (ptr_c: u64, ptr_lhs: u64, ptr_rhs: u64,
///  m: u32, n: u32, kc: u32, alpha: <elem>,
///  c_row_stride: u32, c_col_stride: u32,
///  lhs_row_stride: u32, lhs_col_stride: u32,
///  rhs_row_stride: u32, rhs_col_stride: u32)
/// ```
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure or an
/// unsupported element type.
pub fn generate_trsm_update_ptx(sm: SmVersion, elem: PtxType) -> BlasResult<(String, String)> {
    if elem != PtxType::F32 && elem != PtxType::F64 {
        return Err(BlasError::PtxGeneration(
            "TRSM update kernel supports only f32 and f64".into(),
        ));
    }
    let is_f64 = elem == PtxType::F64;
    let elem_ty = if is_f64 { "f64" } else { "f32" };
    let byte_size = if is_f64 { 8u32 } else { 4u32 };
    let fr = if is_f64 { "fd" } else { "f" };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };
    let kernel_name = format!(
        "trsm_update_{}_{}",
        if is_f64 { "f64" } else { "f32" },
        sm.as_ptx_str()
    );

    let mut p = String::with_capacity(4096);
    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;
    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 tupd_ptr_c,")?;
    wl(&mut p, "    .param .u64 tupd_ptr_lhs,")?;
    wl(&mut p, "    .param .u64 tupd_ptr_rhs,")?;
    wl(&mut p, "    .param .u32 tupd_m,")?;
    wl(&mut p, "    .param .u32 tupd_n,")?;
    wl(&mut p, "    .param .u32 tupd_kc,")?;
    wl(&mut p, &format!("    .param .{elem_ty} tupd_alpha,"))?;
    wl(&mut p, "    .param .u32 tupd_c_row_stride,")?;
    wl(&mut p, "    .param .u32 tupd_c_col_stride,")?;
    wl(&mut p, "    .param .u32 tupd_lhs_row_stride,")?;
    wl(&mut p, "    .param .u32 tupd_lhs_col_stride,")?;
    wl(&mut p, "    .param .u32 tupd_rhs_row_stride,")?;
    wl(&mut p, "    .param .u32 tupd_rhs_col_stride")?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;
    wl(&mut p, "    .reg .pred %p<4>;")?;
    wl(&mut p, "    .reg .b32 %r<40>;")?;
    wl(&mut p, "    .reg .b64 %rd<24>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<8>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<8>;")?;
    }
    wl(&mut p, "")?;

    // Flat thread id -> (row, col).
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %tid.x;")?;
    wl(&mut p, "    mad.lo.u32 %r0, %r1, %r2, %r3;")?;
    wl(&mut p, "    ld.param.u32 %r4, [tupd_m];")?;
    wl(&mut p, "    ld.param.u32 %r5, [tupd_n];")?;
    wl(&mut p, "    mul.lo.u32 %r6, %r4, %r5;")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r6;")?;
    wl(&mut p, "    @%p0 bra $TUPD_RET;")?;
    wl(&mut p, "    div.u32 %r7, %r0, %r5;           // row r")?;
    wl(&mut p, "    rem.u32 %r8, %r0, %r5;           // col c")?;
    wl(&mut p, "")?;

    // Load pointers and strides.
    wl(&mut p, "    ld.param.u64 %rd0, [tupd_ptr_c];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [tupd_ptr_lhs];")?;
    wl(&mut p, "    ld.param.u64 %rd2, [tupd_ptr_rhs];")?;
    wl(&mut p, "    ld.param.u32 %r9,  [tupd_kc];")?;
    wl(&mut p, "    ld.param.u32 %r10, [tupd_lhs_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r11, [tupd_lhs_col_stride];")?;
    wl(&mut p, "    ld.param.u32 %r12, [tupd_rhs_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r13, [tupd_rhs_col_stride];")?;
    wl(&mut p, "")?;

    // Base element offsets that do not change inside the k-loop:
    //   lhs row term  = r * lhs_row_stride
    //   rhs col term  = c * rhs_col_stride
    wl(
        &mut p,
        "    mul.lo.u32 %r14, %r7, %r10;      // r * lhs_row_stride",
    )?;
    wl(
        &mut p,
        "    mul.lo.u32 %r15, %r8, %r13;      // c * rhs_col_stride",
    )?;
    wl(&mut p, "")?;

    // Accumulation loop over k.
    wl(
        &mut p,
        &format!("    mov.{elem_ty} %{fr}0, {zero_lit};   // acc"),
    )?;
    wl(&mut p, "    mov.u32 %r16, 0;                 // k")?;
    wl(&mut p, "$TUPD_LOOP:")?;
    wl(&mut p, "    setp.ge.u32 %p1, %r16, %r9;")?;
    wl(&mut p, "    @%p1 bra $TUPD_LOOP_END;")?;
    wl(&mut p, "")?;
    // LHS[r, k] address = lhs + (r*lrs + k*lcs) * size
    wl(
        &mut p,
        "    mad.lo.u32 %r17, %r16, %r11, %r14;   // r*lrs + k*lcs",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd10, %r17;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd10, %rd10, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd11, %rd1, %rd10;")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}1, [%rd11];   // lhs_rk"),
    )?;
    // RHS[k, c] address = rhs + (k*rrs + c*rcs) * size
    wl(
        &mut p,
        "    mad.lo.u32 %r18, %r16, %r12, %r15;   // k*rrs + c*rcs",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd12, %r18;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd12, %rd12, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd13, %rd2, %rd12;")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}2, [%rd13];   // rhs_kc"),
    )?;
    // acc += lhs_rk * rhs_kc
    wl(
        &mut p,
        &format!("    fma.rn.{elem_ty} %{fr}0, %{fr}1, %{fr}2, %{fr}0;"),
    )?;
    wl(&mut p, "    add.u32 %r16, %r16, 1;")?;
    wl(&mut p, "    bra $TUPD_LOOP;")?;
    wl(&mut p, "$TUPD_LOOP_END:")?;
    wl(&mut p, "")?;

    // C[r, c] address = c_ptr + (r*crs + c*ccs) * size
    wl(&mut p, "    ld.param.u32 %r19, [tupd_c_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r20, [tupd_c_col_stride];")?;
    wl(&mut p, "    mul.lo.u32 %r21, %r7, %r19;")?;
    wl(&mut p, "    mad.lo.u32 %r21, %r8, %r20, %r21;")?;
    wl(&mut p, "    cvt.u64.u32 %rd14, %r21;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd14, %rd14, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd15, %rd0, %rd14;")?;
    // C[r,c] = C[r,c] + alpha * acc
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}3, [%rd15];   // c_old"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{elem_ty} %{fr}4, [tupd_alpha];"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{elem_ty} %{fr}3, %{fr}0, %{fr}4, %{fr}3;"),
    )?;
    wl(&mut p, &format!("    st.global.{elem_ty} [%rd15], %{fr}3;"))?;
    wl(&mut p, "$TUPD_RET:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok((p, kernel_name))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Appends `line` plus a newline to the PTX buffer.
fn wl(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(side: Side, fill: FillMode, trans: Transpose, diag: DiagType) -> TrsmKernelConfig {
        TrsmKernelConfig {
            sm: SmVersion::Sm80,
            elem: PtxType::F32,
            side,
            fill_mode: fill,
            trans,
            diag,
        }
    }

    #[test]
    fn forward_for_lower_no_trans() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(c.forward());
    }

    #[test]
    fn backward_for_upper_no_trans() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(!c.forward());
    }

    #[test]
    fn forward_for_upper_trans() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(c.forward());
    }

    #[test]
    fn backward_for_lower_trans() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(!c.forward());
    }

    #[test]
    fn conj_trans_treated_as_trans() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::ConjTrans,
            DiagType::NonUnit,
        );
        assert!(c.op_transposed());
        assert!(c.forward());
    }

    #[test]
    fn right_side_iteration_mirrors_left() {
        // Side::Right reverses the substitution direction relative to Left.
        let left = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let right = cfg(
            Side::Right,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(left.forward());
        assert!(!right.forward());

        let left_u = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let right_u = cfg(
            Side::Right,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(!left_u.forward());
        assert!(right_u.forward());
    }

    #[test]
    fn a_access_transpose_folds_side_and_op() {
        // Left/NoTrans reads A as stored.
        let l_n = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(!l_n.a_access_transposed());
        // Left/Trans reads A transposed.
        let l_t = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(l_t.a_access_transposed());
        // Right/NoTrans flips the access (pivot is the first index).
        let r_n = cfg(
            Side::Right,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(r_n.a_access_transposed());
        // Right/Trans cancels back to non-transposed access.
        let r_t = cfg(
            Side::Right,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(!r_t.a_access_transposed());
    }

    #[test]
    fn kernel_name_unique_per_config() {
        let a = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let b = cfg(
            Side::Right,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::Unit,
        );
        assert_ne!(a.kernel_name(), b.kernel_name());
        assert!(a.kernel_name().starts_with("trsm_diag_f32_"));
    }

    #[test]
    fn ptx_non_unit_has_divide() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let (ptx, name) = generate_trsm_diag_ptx(&c).expect("ptx");
        assert!(ptx.contains(&name));
        assert!(ptx.contains("div.rn.f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn ptx_unit_diag_skips_divide() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::Unit,
        );
        let (ptx, _) = generate_trsm_diag_ptx(&c).expect("ptx");
        assert!(!ptx.contains("div.rn.f32"));
        // The trailing rank-1 update is still required.
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn ptx_f64_uses_f64_ops() {
        let mut c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        c.elem = PtxType::F64;
        let (ptx, name) = generate_trsm_diag_ptx(&c).expect("ptx");
        assert!(name.contains("f64"));
        assert!(ptx.contains("div.rn.f64"));
        assert!(ptx.contains(".reg .f64"));
    }

    #[test]
    fn ptx_rejects_unsupported_type() {
        let mut c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        c.elem = PtxType::F16;
        assert!(generate_trsm_diag_ptx(&c).is_err());
    }

    #[test]
    fn scale_ptx_has_multiply() {
        let (ptx, name) = generate_trsm_scale_ptx(SmVersion::Sm80, PtxType::F32).expect("ptx");
        assert!(ptx.contains(&name));
        assert!(ptx.contains("mul.rn.f32"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn scale_ptx_rejects_unsupported_type() {
        assert!(generate_trsm_scale_ptx(SmVersion::Sm80, PtxType::F16).is_err());
    }

    #[test]
    fn update_ptx_has_fma_loop() {
        let (ptx, name) = generate_trsm_update_ptx(SmVersion::Sm80, PtxType::F64).expect("ptx");
        assert!(ptx.contains(&name));
        // The matmul-accumulate must have an FMA reduction loop.
        assert!(ptx.contains("fma.rn.f64"));
        assert!(ptx.contains("$TUPD_LOOP:"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn update_ptx_f32_variant() {
        let (ptx, name) = generate_trsm_update_ptx(SmVersion::Sm80, PtxType::F32).expect("ptx");
        assert!(name.contains("f32"));
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn update_ptx_rejects_unsupported_type() {
        assert!(generate_trsm_update_ptx(SmVersion::Sm80, PtxType::BF16).is_err());
    }
}
