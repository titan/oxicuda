//! GPU PTX kernels for Hyperdimensional Computing (HDC) operations.
//!
//! Each kernel is emitted as a self-contained PTX module string parameterised on
//! SM version. PTX ISA is selected by SM:
//!     SM>=100 -> 8.7 (Blackwell), SM>=90 -> 8.4 (Hopper),
//!     SM>=80  -> 8.0 (Ampere),    else -> 7.5 (Turing).

/// Build a PTX file header string for the given SM version.
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Encode a `f32` constant as a PTX immediate hex literal (`0Fxxxxxxxx`).
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

/// XOR binding kernel for binary HVs stored as i8 with values ±1.
/// For {-1,+1}: XOR is equivalent to element-wise multiply (sign product).
/// Signature: `xor_bind_kernel(a: *i8, b: *i8, out: *i8, n: u32)`
#[must_use]
pub fn xor_bind_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // Use string concatenation to avoid Rust format! issues with PTX % registers
    let body = "// xor_bind_kernel: element-wise sign product of two binary HVs (+-1 domain)\n\
.visible .entry xor_bind_kernel(\n\
    .param .u64 p_a,\n\
    .param .u64 p_b,\n\
    .param .u64 p_out,\n\
    .param .u32 p_n\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<8>;\n\
    .reg .s16  %s<4>;\n\
    .reg .s32  %rr<4>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_a];\n\
    ld.param.u64  %rd1, [p_b];\n\
    ld.param.u64  %rd2, [p_out];\n\
    ld.param.u32  %r0,  [p_n];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $XB_DONE;\n\
\n\
    cvt.u64.u32   %rd3, %r4;\n\
    add.u64       %rd4, %rd0, %rd3;\n\
    add.u64       %rd5, %rd1, %rd3;\n\
    add.u64       %rd6, %rd2, %rd3;\n\
\n\
    ld.global.s8  %rr0, [%rd4];\n\
    ld.global.s8  %rr1, [%rd5];\n\
    mul.lo.s32    %rr2, %rr0, %rr1;\n\
    st.global.s8  [%rd6], %rr2;\n\
\n\
$XB_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// Majority vote bundling kernel for binary HVs (i8 ±1 domain).
/// Accumulates K binary HVs into a count array, then thresholds.
/// Signature: `bundle_majority_kernel(matrix: *i8, out: *i8, n: u32, k: u32)`
/// matrix is row-major, k rows of n elements.
#[must_use]
pub fn bundle_majority_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// bundle_majority_kernel: majority vote over K binary HVs\n\
.visible .entry bundle_majority_kernel(\n\
    .param .u64 p_matrix,\n\
    .param .u64 p_out,\n\
    .param .u32 p_n,\n\
    .param .u32 p_k\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<10>;\n\
    .reg .s32  %rr<4>;\n\
    .reg .pred %p0, %p1;\n\
\n\
    ld.param.u64  %rd0, [p_matrix];\n\
    ld.param.u64  %rd1, [p_out];\n\
    ld.param.u32  %r0,  [p_n];\n\
    ld.param.u32  %r1,  [p_k];\n\
\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mov.u32       %r4, %tid.x;\n\
    mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    setp.ge.u32   %p0, %r5, %r0;\n\
    @%p0 bra $BM_DONE;\n\
\n\
    mov.s32       %rr0, 0;\n\
    mov.u32       %r6, 0;\n\
$BM_LOOP:\n\
    setp.ge.u32   %p0, %r6, %r1;\n\
    @%p0 bra $BM_ACCUM_DONE;\n\
    mul.lo.u32    %r7, %r6, %r0;\n\
    add.u32       %r7, %r7, %r5;\n\
    cvt.u64.u32   %rd2, %r7;\n\
    add.u64       %rd3, %rd0, %rd2;\n\
    ld.global.s8  %rr1, [%rd3];\n\
    add.s32       %rr0, %rr0, %rr1;\n\
    add.u32       %r6, %r6, 1;\n\
    bra $BM_LOOP;\n\
$BM_ACCUM_DONE:\n\
\n\
    setp.gt.s32   %p1, %rr0, 0;\n\
    selp.s32      %rr2, 1, -1, %p1;\n\
\n\
    cvt.u64.u32   %rd4, %r5;\n\
    add.u64       %rd5, %rd1, %rd4;\n\
    st.global.s8  [%rd5], %rr2;\n\
\n\
$BM_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// Cyclic shift kernel for integer HVs (i32 elements).
/// Shifts the array left by k positions: `out[i] = in[(i+k) % n]`.
/// Signature: `cyclic_shift_kernel(in: *i32, out: *i32, n: u32, k: u32)`
#[must_use]
pub fn cyclic_shift_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// cyclic_shift_kernel: cyclic left shift of integer HV by k positions\n\
.visible .entry cyclic_shift_kernel(\n\
    .param .u64 p_in,\n\
    .param .u64 p_out,\n\
    .param .u32 p_n,\n\
    .param .u32 p_k\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<10>;\n\
    .reg .s32  %rr<2>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_in];\n\
    ld.param.u64  %rd1, [p_out];\n\
    ld.param.u32  %r0,  [p_n];\n\
    ld.param.u32  %r1,  [p_k];\n\
\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mov.u32       %r4, %tid.x;\n\
    mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    setp.ge.u32   %p0, %r5, %r0;\n\
    @%p0 bra $CS_DONE;\n\
\n\
    add.u32       %r6, %r5, %r1;\n\
    rem.u32       %r7, %r6, %r0;\n\
\n\
    mul.wide.u32  %rd2, %r7, 4;\n\
    add.u64       %rd3, %rd0, %rd2;\n\
    ld.global.s32 %rr0, [%rd3];\n\
\n\
    mul.wide.u32  %rd4, %r5, 4;\n\
    add.u64       %rd5, %rd1, %rd4;\n\
    st.global.s32 [%rd5], %rr0;\n\
\n\
$CS_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// Cosine similarity kernel for float HVs.
/// Uses atomic add to accumulate partial results from all threads.
/// Signature: `cosine_sim_kernel(a: *f32, b: *f32, dot: *f32, norm_a: *f32, norm_b: *f32, n: u32)`
#[must_use]
pub fn cosine_sim_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// cosine_sim_kernel: compute cosine similarity via dot/norms (single-block reduction)\n\
.visible .entry cosine_sim_kernel(\n\
    .param .u64 p_a,\n\
    .param .u64 p_b,\n\
    .param .u64 p_dot,\n\
    .param .u64 p_norm_a,\n\
    .param .u64 p_norm_b,\n\
    .param .u32 p_n\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f32  %f<8>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_a];\n\
    ld.param.u64  %rd1, [p_b];\n\
    ld.param.u64  %rd2, [p_dot];\n\
    ld.param.u64  %rd3, [p_norm_a];\n\
    ld.param.u64  %rd4, [p_norm_b];\n\
    ld.param.u32  %r0,  [p_n];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $COS_DONE;\n\
\n\
    mul.wide.u32  %rd5, %r4, 4;\n\
    add.u64       %rd6, %rd0, %rd5;\n\
    add.u64       %rd7, %rd1, %rd5;\n\
    ld.global.f32 %f0, [%rd6];\n\
    ld.global.f32 %f1, [%rd7];\n\
\n\
    mul.f32       %f2, %f0, %f1;\n\
    mul.f32       %f3, %f0, %f0;\n\
    mul.f32       %f4, %f1, %f1;\n\
\n\
    red.global.add.f32 [%rd2], %f2;\n\
    red.global.add.f32 [%rd3], %f3;\n\
    red.global.add.f32 [%rd4], %f4;\n\
\n\
$COS_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// Hamming distance kernel for binary HVs (±1 i8 encoding).
/// Signature: `hamming_dist_kernel(a: *i8, b: *i8, count: *u32, n: u32)`
#[must_use]
pub fn hamming_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// hamming_dist_kernel: accumulate Hamming count for binary HVs\n\
.visible .entry hamming_dist_kernel(\n\
    .param .u64 p_a,\n\
    .param .u64 p_b,\n\
    .param .u64 p_count,\n\
    .param .u32 p_n\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<8>;\n\
    .reg .s32  %rr<4>;\n\
    .reg .pred %p0, %p1;\n\
\n\
    ld.param.u64  %rd0, [p_a];\n\
    ld.param.u64  %rd1, [p_b];\n\
    ld.param.u64  %rd2, [p_count];\n\
    ld.param.u32  %r0,  [p_n];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $HD_DONE;\n\
\n\
    cvt.u64.u32   %rd3, %r4;\n\
    add.u64       %rd4, %rd0, %rd3;\n\
    add.u64       %rd5, %rd1, %rd3;\n\
    ld.global.s8  %rr0, [%rd4];\n\
    ld.global.s8  %rr1, [%rd5];\n\
    mul.lo.s32    %rr2, %rr0, %rr1;\n\
\n\
    setp.eq.s32   %p1, %rr2, -1;\n\
    selp.u32      %r5, 1, 0, %p1;\n\
    red.global.add.u32 [%rd2], %r5;\n\
\n\
$HD_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// Complex binding kernel for FHRR hypervectors (element-wise complex multiply).
/// Stored as interleaved [re_0, im_0, re_1, im_1, ...], length = 2*dim.
/// Signature: `complex_bind_kernel(a: *f32, b: *f32, out: *f32, dim: u32)`
#[must_use]
pub fn complex_bind_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// complex_bind_kernel: FHRR element-wise complex multiply (interleaved re/im)\n\
.visible .entry complex_bind_kernel(\n\
    .param .u64 p_a,\n\
    .param .u64 p_b,\n\
    .param .u64 p_out,\n\
    .param .u32 p_dim\n\
)\n\
{\n\
    .reg .u64  %rd<8>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f32  %f<8>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_a];\n\
    ld.param.u64  %rd1, [p_b];\n\
    ld.param.u64  %rd2, [p_out];\n\
    ld.param.u32  %r0,  [p_dim];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $CB_DONE;\n\
\n\
    shl.b32       %r5, %r4, 1;\n\
    mul.wide.u32  %rd3, %r5, 4;\n\
    add.u64       %rd4, %rd0, %rd3;\n\
    add.u64       %rd5, %rd1, %rd3;\n\
    add.u64       %rd6, %rd2, %rd3;\n\
\n\
    ld.global.f32 %f0, [%rd4];\n\
    ld.global.f32 %f1, [%rd4+4];\n\
    ld.global.f32 %f2, [%rd5];\n\
    ld.global.f32 %f3, [%rd5+4];\n\
\n\
    mul.f32       %f4, %f0, %f2;\n\
    mul.f32       %f5, %f1, %f3;\n\
    sub.f32       %f6, %f4, %f5;\n\
    mul.f32       %f4, %f0, %f3;\n\
    mul.f32       %f5, %f1, %f2;\n\
    add.f32       %f7, %f4, %f5;\n\
\n\
    st.global.f32 [%rd6],   %f6;\n\
    st.global.f32 [%rd6+4], %f7;\n\
\n\
$CB_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

/// HD classifier kernel: argmax cosine similarity over prototype matrix.
/// Prototypes stored row-major (n_classes × dim), each row is a binary HV (i8 ±1).
/// Signature: `hd_classify_kernel(query: *i8, protos: *i8, out: *u32, dim: u32, n_classes: u32)`
#[must_use]
pub fn hd_classify_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let neg_inf_hex = f32_hex(f32::NEG_INFINITY);
    // Build body with the constant embedded
    let body = format!(
        "// hd_classify_kernel: single-thread argmax cosine over prototype matrix\n\
.visible .entry hd_classify_kernel(\n\
    .param .u64 p_query,\n\
    .param .u64 p_protos,\n\
    .param .u64 p_out,\n\
    .param .u32 p_dim,\n\
    .param .u32 p_nc\n\
)\n\
{{\n\
    .reg .u64  %rd<10>;\n\
    .reg .u32  %r<12>;\n\
    .reg .s32  %rr<4>;\n\
    .reg .f32  %f<8>;\n\
    .reg .pred %p0, %p1;\n\
\n\
    mov.u32       %r0, %ntid.x;\n\
    mov.u32       %r1, %ctaid.x;\n\
    mov.u32       %r2, %tid.x;\n\
    mad.lo.u32    %r3, %r0, %r1, %r2;\n\
    mov.u32       %r0, 0;\n\
    setp.ne.u32   %p0, %r3, %r0;\n\
    @%p0 bra $HC_DONE;\n\
\n\
    ld.param.u64  %rd0, [p_query];\n\
    ld.param.u64  %rd1, [p_protos];\n\
    ld.param.u64  %rd2, [p_out];\n\
    ld.param.u32  %r4,  [p_dim];\n\
    ld.param.u32  %r5,  [p_nc];\n\
\n\
    mov.f32       %f0, {NEG_INF};\n\
    mov.u32       %r6, 0;\n\
    mov.u32       %r7, 0;\n\
\n\
$HC_CLASS_LOOP:\n\
    setp.ge.u32   %p0, %r7, %r5;\n\
    @%p0 bra $HC_CLASS_DONE;\n\
\n\
    mov.f32       %f1, 0F00000000;\n\
    mov.u32       %r8, 0;\n\
\n\
$HC_DIM_LOOP:\n\
    setp.ge.u32   %p0, %r8, %r4;\n\
    @%p0 bra $HC_DIM_DONE;\n\
\n\
    cvt.u64.u32   %rd3, %r8;\n\
    add.u64       %rd4, %rd0, %rd3;\n\
    ld.global.s8  %rr0, [%rd4];\n\
\n\
    mul.lo.u32    %r9, %r7, %r4;\n\
    add.u32       %r9, %r9, %r8;\n\
    cvt.u64.u32   %rd5, %r9;\n\
    add.u64       %rd6, %rd1, %rd5;\n\
    ld.global.s8  %rr1, [%rd6];\n\
\n\
    mul.lo.s32    %rr2, %rr0, %rr1;\n\
    cvt.rn.f32.s32 %f2, %rr2;\n\
    add.f32       %f1, %f1, %f2;\n\
\n\
    add.u32       %r8, %r8, 1;\n\
    bra $HC_DIM_LOOP;\n\
$HC_DIM_DONE:\n\
\n\
    cvt.rn.f32.u32 %f3, %r4;\n\
    div.rn.f32    %f4, %f1, %f3;\n\
\n\
    setp.gt.f32   %p1, %f4, %f0;\n\
    selp.f32      %f0, %f4, %f0, %p1;\n\
    selp.u32      %r6, %r7, %r6, %p1;\n\
\n\
    add.u32       %r7, %r7, 1;\n\
    bra $HC_CLASS_LOOP;\n\
$HC_CLASS_DONE:\n\
\n\
    st.global.u32 [%rd2], %r6;\n\
\n\
$HC_DONE:\n\
    ret;\n\
}}\n",
        NEG_INF = neg_inf_hex
    );
    hdr + &body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_header_strings() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_format() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert!(f32_hex(1.5_f32).starts_with("0F"));
    }

    #[test]
    fn all_kernels_for_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for kernel in [
                xor_bind_ptx(sm),
                bundle_majority_ptx(sm),
                cyclic_shift_ptx(sm),
                cosine_sim_ptx(sm),
                hamming_dist_ptx(sm),
                complex_bind_ptx(sm),
                hd_classify_ptx(sm),
            ] {
                assert!(kernel.contains(".visible .entry"));
                assert!(kernel.contains(".address_size 64"));
            }
        }
    }
}
