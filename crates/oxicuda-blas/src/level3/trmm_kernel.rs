//! Triangular-aware matrix-multiply PTX kernel for [`trmm`](super::trmm::trmm).
//!
//! The kernel computes one element of the TRMM result per thread,
//! `out[r, c] = alpha * (op(A) * B)[r, c]` (Side::Left) or
//! `out[r, c] = alpha * (B * op(A))[r, c]` (Side::Right), reading **only**
//! the triangle of A selected by [`FillMode`].
//!
//! # In-place semantics
//!
//! TRMM overwrites B. A single kernel cannot both read every input element
//! of B and overwrite B without a grid-wide synchronisation, so the kernel
//! writes into a separate `out` buffer; [`super::trmm()`] then copies `out`
//! back over B. The kernel itself is therefore race-free — every thread only
//! reads A and B and only writes `out`.
//!
//! # Triangle, transpose, and unit diagonal
//!
//! - The contraction index `k` ranges over just the stored half:
//!   for an effectively-upper `op(A)` the sum runs over `k >= pivot`, for an
//!   effectively-lower `op(A)` over `k <= pivot`.
//! - `op(A)[i, j]` reads `A[i, j]` (`NoTrans`) or `A[j, i]`
//!   (`Trans` / `ConjTrans`; the two coincide for the real element types
//!   this kernel supports).
//! - With [`DiagType::Unit`] the diagonal contribution is `1 * B[..]`; the
//!   stored diagonal element is never read.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::types::{DiagType, FillMode, Side, Transpose};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters describing a single triangular-multiply kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrmmKernelConfig {
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

impl TrmmKernelConfig {
    /// Returns `true` when `op(A)` reads the transpose of the stored matrix.
    #[must_use]
    pub fn op_transposed(&self) -> bool {
        matches!(self.trans, Transpose::Trans | Transpose::ConjTrans)
    }

    /// Returns `true` when `op(A)` is effectively *upper*-triangular.
    ///
    /// A transpose swaps which triangle carries the data.
    #[must_use]
    pub fn op_is_upper(&self) -> bool {
        match self.fill_mode {
            FillMode::Upper => !self.op_transposed(),
            FillMode::Lower => self.op_transposed(),
            // `Full` is treated as upper here — `trmm` rejects `Full` before
            // reaching the kernel, so the choice is immaterial.
            FillMode::Full => true,
        }
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
            "trmm_mul_{elem}_{side}{fill}{trans}{diag}_{}",
            self.sm.as_ptx_str()
        )
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates the triangular-multiply kernel PTX.
///
/// The kernel signature is:
///
/// ```text
/// (ptr_a: u64, ptr_b: u64, ptr_out: u64,
///  m: u32, n: u32, alpha: <elem>,
///  a_row_stride: u32, a_col_stride: u32,
///  b_row_stride: u32, b_col_stride: u32,
///  o_row_stride: u32, o_col_stride: u32)
/// ```
///
/// One thread computes `out[r, c]` for `r in 0..m`, `c in 0..n`, using a flat
/// 1-D grid over `m * n`. The element-stride pairs let A, B and `out` each
/// sit in either memory layout.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure or an
/// unsupported element type.
pub fn generate_trmm_mul_ptx(config: &TrmmKernelConfig) -> BlasResult<(String, String)> {
    if config.elem != PtxType::F32 && config.elem != PtxType::F64 {
        return Err(BlasError::PtxGeneration(
            "TRMM multiply kernel supports only f32 and f64".into(),
        ));
    }

    let is_f64 = config.elem == PtxType::F64;
    let elem_ty = if is_f64 { "f64" } else { "f32" };
    let byte_size = if is_f64 { 8u32 } else { 4u32 };
    let fr = if is_f64 { "fd" } else { "f" };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };
    let kernel_name = config.kernel_name();

    let left = config.side == Side::Left;
    let op_transposed = config.op_transposed();
    let unit_diag = config.diag == DiagType::Unit;

    // Whether the contraction index `k` sweeps the upper range `[pivot, len)`.
    //
    // For Side::Left the contraction varies the *column* of `op(A)`, so an
    // upper `op(A)` gives the upper range. For Side::Right the contraction
    // varies the *row* of `op(A)`, which inverts the relationship — hence the
    // XOR with the side.
    let range_is_upper = config.op_is_upper() ^ matches!(config.side, Side::Right);

    let mut p = String::with_capacity(8192);

    wl(&mut p, &format!(".version {}", config.sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", config.sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 trmm_ptr_a,")?;
    wl(&mut p, "    .param .u64 trmm_ptr_b,")?;
    wl(&mut p, "    .param .u64 trmm_ptr_out,")?;
    wl(&mut p, "    .param .u32 trmm_m,")?;
    wl(&mut p, "    .param .u32 trmm_n,")?;
    wl(&mut p, &format!("    .param .{elem_ty} trmm_alpha,"))?;
    wl(&mut p, "    .param .u32 trmm_a_row_stride,")?;
    wl(&mut p, "    .param .u32 trmm_a_col_stride,")?;
    wl(&mut p, "    .param .u32 trmm_b_row_stride,")?;
    wl(&mut p, "    .param .u32 trmm_b_col_stride,")?;
    wl(&mut p, "    .param .u32 trmm_o_row_stride,")?;
    wl(&mut p, "    .param .u32 trmm_o_col_stride")?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    wl(&mut p, "    .reg .pred %p<8>;")?;
    wl(&mut p, "    .reg .b32 %r<48>;")?;
    wl(&mut p, "    .reg .b64 %rd<32>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<16>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<16>;")?;
    }
    wl(&mut p, "")?;

    // --- Flat thread id -> (row, col) -----------------------------------
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %tid.x;")?;
    wl(&mut p, "    mad.lo.u32 %r0, %r1, %r2, %r3;   // flat id")?;
    wl(&mut p, "    ld.param.u32 %r4, [trmm_m];")?;
    wl(&mut p, "    ld.param.u32 %r5, [trmm_n];")?;
    wl(&mut p, "    mul.lo.u32 %r6, %r4, %r5;")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r6;")?;
    wl(&mut p, "    @%p0 bra $TRMM_RET;")?;
    wl(&mut p, "    div.u32 %r7, %r0, %r5;           // row r")?;
    wl(&mut p, "    rem.u32 %r8, %r0, %r5;           // col c")?;
    wl(&mut p, "")?;

    // --- Load pointers and strides --------------------------------------
    wl(&mut p, "    ld.param.u64 %rd0, [trmm_ptr_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [trmm_ptr_b];")?;
    wl(&mut p, "    ld.param.u64 %rd2, [trmm_ptr_out];")?;
    wl(&mut p, "    ld.param.u32 %r9,  [trmm_a_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r10, [trmm_a_col_stride];")?;
    wl(&mut p, "    ld.param.u32 %r11, [trmm_b_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r12, [trmm_b_col_stride];")?;
    wl(&mut p, "")?;

    // The contraction length is `m` for Side::Left (rows of A / rows of B)
    // and `n` for Side::Right (cols of A / cols of B).
    if left {
        wl(&mut p, "    mov.u32 %r13, %r4;   // contraction length = m")?;
        wl(&mut p, "    mov.u32 %r14, %r7;   // pivot = r")?;
    } else {
        wl(&mut p, "    mov.u32 %r13, %r5;   // contraction length = n")?;
        wl(&mut p, "    mov.u32 %r14, %r8;   // pivot = c")?;
    }
    wl(&mut p, "")?;

    // --- Contraction bounds [lo, hi) ------------------------------------
    //
    // The upper range sums k in [pivot, len); the lower range sums k in
    // [0, pivot + 1). `range_is_upper` already folds in the Side::Right
    // inversion (see its definition above).
    if range_is_upper {
        wl(&mut p, "    mov.u32 %r15, %r14;          // lo = pivot")?;
        wl(&mut p, "    mov.u32 %r16, %r13;          // hi = len")?;
    } else {
        wl(&mut p, "    mov.u32 %r15, 0;             // lo = 0")?;
        wl(&mut p, "    add.u32 %r16, %r14, 1;       // hi = pivot + 1")?;
    }
    wl(&mut p, "")?;

    // --- Accumulation loop ----------------------------------------------
    wl(
        &mut p,
        &format!("    mov.{elem_ty} %{fr}0, {zero_lit};   // acc"),
    )?;
    wl(&mut p, "    mov.u32 %r17, %r15;             // k = lo")?;
    wl(&mut p, "$TRMM_LOOP:")?;
    wl(&mut p, "    setp.ge.u32 %p1, %r17, %r16;")?;
    wl(&mut p, "    @%p1 bra $TRMM_LOOP_END;")?;
    wl(&mut p, "")?;

    // op(A) element. For Side::Left it is op(A)[r, k]; for Side::Right it is
    // op(A)[k, c]. With `op_transposed` the two stored indices swap.
    //
    //   Left,  NoTrans : A[r, k]   -> r*ars + k*acs
    //   Left,  Trans   : A[k, r]   -> k*ars + r*acs
    //   Right, NoTrans : A[k, c]   -> k*ars + c*acs
    //   Right, Trans   : A[c, k]   -> c*ars + k*acs
    wl(&mut p, "    // --- op(A) element address ---")?;
    if left {
        if op_transposed {
            wl(
                &mut p,
                "    mul.lo.u32 %r20, %r17, %r9;     // k*a_row_stride",
            )?;
            wl(
                &mut p,
                "    mad.lo.u32 %r20, %r7, %r10, %r20;   // + r*a_col_stride",
            )?;
        } else {
            wl(
                &mut p,
                "    mul.lo.u32 %r20, %r7, %r9;      // r*a_row_stride",
            )?;
            wl(
                &mut p,
                "    mad.lo.u32 %r20, %r17, %r10, %r20;  // + k*a_col_stride",
            )?;
        }
    } else if op_transposed {
        wl(
            &mut p,
            "    mul.lo.u32 %r20, %r8, %r9;      // c*a_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r20, %r17, %r10, %r20;  // + k*a_col_stride",
        )?;
    } else {
        wl(
            &mut p,
            "    mul.lo.u32 %r20, %r17, %r9;     // k*a_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r20, %r8, %r10, %r20;   // + c*a_col_stride",
        )?;
    }

    // Decide whether this `k` is the diagonal element (k == pivot).
    wl(
        &mut p,
        "    setp.eq.u32 %p2, %r17, %r14;    // k == pivot ?",
    )?;

    if unit_diag {
        // Unit diagonal: a_val = 1 on the diagonal, else the stored element.
        wl(&mut p, "    cvt.u64.u32 %rd10, %r20;")?;
        wl(
            &mut p,
            &format!("    mul.lo.u64 %rd10, %rd10, {byte_size};"),
        )?;
        wl(&mut p, "    add.u64 %rd11, %rd0, %rd10;")?;
        wl(
            &mut p,
            &format!("    ld.global.{elem_ty} %{fr}1, [%rd11];   // stored A"),
        )?;
        // Override with 1.0 when on the diagonal.
        let one_lit = if is_f64 {
            "0d3FF0000000000000"
        } else {
            "0f3F800000"
        };
        wl(&mut p, &format!("    mov.{elem_ty} %{fr}2, {one_lit};"))?;
        wl(
            &mut p,
            &format!("    selp.{elem_ty} %{fr}1, %{fr}2, %{fr}1, %p2;"),
        )?;
    } else {
        // Non-unit diagonal: always read the stored element.
        wl(&mut p, "    cvt.u64.u32 %rd10, %r20;")?;
        wl(
            &mut p,
            &format!("    mul.lo.u64 %rd10, %rd10, {byte_size};"),
        )?;
        wl(&mut p, "    add.u64 %rd11, %rd0, %rd10;")?;
        wl(
            &mut p,
            &format!("    ld.global.{elem_ty} %{fr}1, [%rd11];   // A element"),
        )?;
    }

    // B element. For Side::Left it is B[k, c]; for Side::Right it is B[r, k].
    wl(&mut p, "    // --- B element address ---")?;
    if left {
        wl(
            &mut p,
            "    mul.lo.u32 %r21, %r17, %r11;    // k*b_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r21, %r8, %r12, %r21;   // + c*b_col_stride",
        )?;
    } else {
        wl(
            &mut p,
            "    mul.lo.u32 %r21, %r7, %r11;     // r*b_row_stride",
        )?;
        wl(
            &mut p,
            "    mad.lo.u32 %r21, %r17, %r12, %r21;  // + k*b_col_stride",
        )?;
    }
    wl(&mut p, "    cvt.u64.u32 %rd12, %r21;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd12, %rd12, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd13, %rd1, %rd12;")?;
    wl(
        &mut p,
        &format!("    ld.global.{elem_ty} %{fr}3, [%rd13];   // B element"),
    )?;

    // acc += a_val * b_val
    wl(
        &mut p,
        &format!("    fma.rn.{elem_ty} %{fr}0, %{fr}1, %{fr}3, %{fr}0;"),
    )?;
    wl(&mut p, "")?;

    wl(&mut p, "    add.u32 %r17, %r17, 1;")?;
    wl(&mut p, "    bra $TRMM_LOOP;")?;
    wl(&mut p, "$TRMM_LOOP_END:")?;
    wl(&mut p, "")?;

    // --- Scale by alpha and store to out[r, c] --------------------------
    wl(
        &mut p,
        &format!("    ld.param.{elem_ty} %{fr}4, [trmm_alpha];"),
    )?;
    wl(
        &mut p,
        &format!("    mul.rn.{elem_ty} %{fr}0, %{fr}0, %{fr}4;"),
    )?;
    wl(&mut p, "    ld.param.u32 %r22, [trmm_o_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r23, [trmm_o_col_stride];")?;
    wl(
        &mut p,
        "    mul.lo.u32 %r24, %r7, %r22;     // r*o_row_stride",
    )?;
    wl(
        &mut p,
        "    mad.lo.u32 %r24, %r8, %r23, %r24;   // + c*o_col_stride",
    )?;
    wl(&mut p, "    cvt.u64.u32 %rd14, %r24;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd14, %rd14, {byte_size};"),
    )?;
    wl(&mut p, "    add.u64 %rd15, %rd2, %rd14;")?;
    wl(&mut p, &format!("    st.global.{elem_ty} [%rd15], %{fr}0;"))?;
    wl(&mut p, "$TRMM_RET:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok((p, kernel_name))
}

// ---------------------------------------------------------------------------
// Strided matrix copy kernel
// ---------------------------------------------------------------------------

/// Generates a kernel that copies a dense `rows x cols` matrix element by
/// element: `dst[r, c] = src[r, c]`.
///
/// TRMM computes its result into a tightly-packed scratch buffer (so the
/// multiply kernel stays race-free), then this kernel copies that scratch
/// back over B honouring B's own leading dimension. Independent stride pairs
/// for source and destination support either memory layout.
///
/// The kernel signature is `(ptr_dst: u64, ptr_src: u64, rows: u32,
/// cols: u32, d_row_stride: u32, d_col_stride: u32, s_row_stride: u32,
/// s_col_stride: u32)`, launched with a flat 1-D grid over `rows * cols`.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure or an
/// unsupported element type.
pub fn generate_trmm_copy_ptx(sm: SmVersion, elem: PtxType) -> BlasResult<(String, String)> {
    if elem != PtxType::F32 && elem != PtxType::F64 {
        return Err(BlasError::PtxGeneration(
            "TRMM copy kernel supports only f32 and f64".into(),
        ));
    }
    let is_f64 = elem == PtxType::F64;
    let elem_ty = if is_f64 { "f64" } else { "f32" };
    let byte_size = if is_f64 { 8u32 } else { 4u32 };
    let fr = if is_f64 { "fd" } else { "f" };
    let kernel_name = format!(
        "trmm_copy_{}_{}",
        if is_f64 { "f64" } else { "f32" },
        sm.as_ptx_str()
    );

    let mut p = String::with_capacity(2048);
    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;
    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 copy_ptr_dst,")?;
    wl(&mut p, "    .param .u64 copy_ptr_src,")?;
    wl(&mut p, "    .param .u32 copy_rows,")?;
    wl(&mut p, "    .param .u32 copy_cols,")?;
    wl(&mut p, "    .param .u32 copy_d_row_stride,")?;
    wl(&mut p, "    .param .u32 copy_d_col_stride,")?;
    wl(&mut p, "    .param .u32 copy_s_row_stride,")?;
    wl(&mut p, "    .param .u32 copy_s_col_stride")?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;
    wl(&mut p, "    .reg .pred %p<4>;")?;
    wl(&mut p, "    .reg .b32 %r<20>;")?;
    wl(&mut p, "    .reg .b64 %rd<10>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<4>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<4>;")?;
    }
    wl(&mut p, "")?;
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %tid.x;")?;
    wl(&mut p, "    mad.lo.u32 %r0, %r1, %r2, %r3;")?;
    wl(&mut p, "    ld.param.u32 %r4, [copy_rows];")?;
    wl(&mut p, "    ld.param.u32 %r5, [copy_cols];")?;
    wl(&mut p, "    mul.lo.u32 %r6, %r4, %r5;")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r6;")?;
    wl(&mut p, "    @%p0 bra $COPY_RET;")?;
    wl(&mut p, "    div.u32 %r7, %r0, %r5;           // row")?;
    wl(&mut p, "    rem.u32 %r8, %r0, %r5;           // col")?;
    wl(&mut p, "")?;
    // Source address.
    wl(&mut p, "    ld.param.u32 %r9,  [copy_s_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r10, [copy_s_col_stride];")?;
    wl(&mut p, "    mul.lo.u32 %r11, %r7, %r9;")?;
    wl(&mut p, "    mad.lo.u32 %r11, %r8, %r10, %r11;")?;
    wl(&mut p, "    cvt.u64.u32 %rd2, %r11;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd2, %rd2, {byte_size};"))?;
    wl(&mut p, "    ld.param.u64 %rd0, [copy_ptr_src];")?;
    wl(&mut p, "    add.u64 %rd3, %rd0, %rd2;")?;
    wl(&mut p, &format!("    ld.global.{elem_ty} %{fr}0, [%rd3];"))?;
    // Destination address.
    wl(&mut p, "    ld.param.u32 %r12, [copy_d_row_stride];")?;
    wl(&mut p, "    ld.param.u32 %r13, [copy_d_col_stride];")?;
    wl(&mut p, "    mul.lo.u32 %r14, %r7, %r12;")?;
    wl(&mut p, "    mad.lo.u32 %r14, %r8, %r13, %r14;")?;
    wl(&mut p, "    cvt.u64.u32 %rd4, %r14;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd4, %rd4, {byte_size};"))?;
    wl(&mut p, "    ld.param.u64 %rd1, [copy_ptr_dst];")?;
    wl(&mut p, "    add.u64 %rd5, %rd1, %rd4;")?;
    wl(&mut p, &format!("    st.global.{elem_ty} [%rd5], %{fr}0;"))?;
    wl(&mut p, "$COPY_RET:")?;
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

    fn cfg(side: Side, fill: FillMode, trans: Transpose, diag: DiagType) -> TrmmKernelConfig {
        TrmmKernelConfig {
            sm: SmVersion::Sm80,
            elem: PtxType::F32,
            side,
            fill_mode: fill,
            trans,
            diag,
        }
    }

    #[test]
    fn upper_no_trans_is_upper() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(c.op_is_upper());
    }

    #[test]
    fn lower_no_trans_is_lower() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(!c.op_is_upper());
    }

    #[test]
    fn upper_trans_flips_to_lower() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(!c.op_is_upper());
        assert!(c.op_transposed());
    }

    #[test]
    fn lower_trans_flips_to_upper() {
        let c = cfg(
            Side::Left,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::NonUnit,
        );
        assert!(c.op_is_upper());
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
    }

    #[test]
    fn kernel_name_unique() {
        let a = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let b = cfg(
            Side::Right,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::Unit,
        );
        assert_ne!(a.kernel_name(), b.kernel_name());
        assert!(a.kernel_name().starts_with("trmm_mul_f32_"));
    }

    #[test]
    fn ptx_non_unit_reads_diagonal() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        let (ptx, name) = generate_trmm_mul_ptx(&c).expect("ptx");
        assert!(ptx.contains(&name));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("mul.rn.f32")); // alpha scale
        assert!(ptx.contains(".target sm_80"));
        // Non-unit must not select a hard-coded 1.0 onto the diagonal.
        assert!(!ptx.contains("0f3F800000"));
    }

    #[test]
    fn ptx_unit_diag_selects_one() {
        let c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::Unit,
        );
        let (ptx, _) = generate_trmm_mul_ptx(&c).expect("ptx");
        // Unit diagonal substitutes 1.0 via selp on the diagonal element.
        assert!(ptx.contains("0f3F800000"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn ptx_f64_uses_f64_ops() {
        let mut c = cfg(
            Side::Right,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        c.elem = PtxType::F64;
        let (ptx, name) = generate_trmm_mul_ptx(&c).expect("ptx");
        assert!(name.contains("f64"));
        assert!(ptx.contains("fma.rn.f64"));
        assert!(ptx.contains(".reg .f64"));
    }

    #[test]
    fn ptx_rejects_unsupported_type() {
        let mut c = cfg(
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        c.elem = PtxType::BF16;
        assert!(generate_trmm_mul_ptx(&c).is_err());
    }

    #[test]
    fn copy_ptx_has_load_store() {
        let (ptx, name) = generate_trmm_copy_ptx(SmVersion::Sm80, PtxType::F32).expect("ptx");
        assert!(ptx.contains(&name));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn copy_ptx_rejects_unsupported_type() {
        assert!(generate_trmm_copy_ptx(SmVersion::Sm80, PtxType::F16).is_err());
    }
}
