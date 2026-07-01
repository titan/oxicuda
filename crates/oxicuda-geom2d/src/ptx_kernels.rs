//! GPU PTX kernels for 2D computational geometry operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM>=100 -> 8.7 (Blackwell), SM>=90 -> 8.4 (Hopper),
//!     SM>=80  -> 8.0 (Ampere),    else -> 7.5 (Turing).
//!
//! IMPORTANT: PTX kernel bodies use **string concatenation** (NOT `format!()`) for
//! sections containing `%rd`, `%r`, `%f`, `%fd` register names, which Rust's format macro
//! would misinterpret as unused format arguments in edition 2024.

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

/// Batched 2D orientation tests (CCW / CW / collinear).
///
/// Signature: `orientation_test_kernel(ax, ay, bx, by, cx, cy, out, n)`
/// For each i: `out[i] = (bx[i]-ax[i])*(cy[i]-ay[i]) - (by[i]-ay[i])*(cx[i]-ax[i])`
#[must_use]
pub fn orientation_test_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry orientation_test_kernel(\n\
        .param .u64 p_ax,\n\
        .param .u64 p_ay,\n\
        .param .u64 p_bx,\n\
        .param .u64 p_by,\n\
        .param .u64 p_cx,\n\
        .param .u64 p_cy,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<20>;\n\
        .reg .u32  %r<8>;\n\
        .reg .f64  %fd<16>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_ax];\n\
        ld.param.u64  %rd1, [p_ay];\n\
        ld.param.u64  %rd2, [p_bx];\n\
        ld.param.u64  %rd3, [p_by];\n\
        ld.param.u64  %rd4, [p_cx];\n\
        ld.param.u64  %rd5, [p_cy];\n\
        ld.param.u64  %rd6, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $OT_DONE;\n\
    \n\
        mul.wide.u32  %rd10, %r4, 8;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.f64 %fd0, [%rd11];\n\
        add.u64       %rd12, %rd1, %rd10;\n\
        ld.global.f64 %fd1, [%rd12];\n\
        add.u64       %rd13, %rd2, %rd10;\n\
        ld.global.f64 %fd2, [%rd13];\n\
        add.u64       %rd14, %rd3, %rd10;\n\
        ld.global.f64 %fd3, [%rd14];\n\
        add.u64       %rd15, %rd4, %rd10;\n\
        ld.global.f64 %fd4, [%rd15];\n\
        add.u64       %rd16, %rd5, %rd10;\n\
        ld.global.f64 %fd5, [%rd16];\n\
    \n\
        // (bx - ax)\n\
        sub.f64       %fd6, %fd2, %fd0;\n\
        // (cy - ay)\n\
        sub.f64       %fd7, %fd5, %fd1;\n\
        // (by - ay)\n\
        sub.f64       %fd8, %fd3, %fd1;\n\
        // (cx - ax)\n\
        sub.f64       %fd9, %fd4, %fd0;\n\
    \n\
        // (bx-ax)*(cy-ay)\n\
        mul.f64       %fd10, %fd6, %fd7;\n\
        // (by-ay)*(cx-ax)\n\
        mul.f64       %fd11, %fd8, %fd9;\n\
        sub.f64       %fd12, %fd10, %fd11;\n\
    \n\
        add.u64       %rd17, %rd6, %rd10;\n\
        st.global.f64 [%rd17], %fd12;\n\
    \n\
    $OT_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Element-wise 2D cross product `c = ax*by - ay*bx`.
///
/// Signature: `cross_product_kernel(ax, ay, bx, by, out, n)`
#[must_use]
pub fn cross_product_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cross_product_kernel(\n\
        .param .u64 p_ax,\n\
        .param .u64 p_ay,\n\
        .param .u64 p_bx,\n\
        .param .u64 p_by,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<8>;\n\
        .reg .f64  %fd<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_ax];\n\
        ld.param.u64  %rd1, [p_ay];\n\
        ld.param.u64  %rd2, [p_bx];\n\
        ld.param.u64  %rd3, [p_by];\n\
        ld.param.u64  %rd4, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CP_DONE;\n\
    \n\
        mul.wide.u32  %rd8, %r4, 8;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f64 %fd0, [%rd9];\n\
        add.u64       %rd10, %rd1, %rd8;\n\
        ld.global.f64 %fd1, [%rd10];\n\
        add.u64       %rd11, %rd2, %rd8;\n\
        ld.global.f64 %fd2, [%rd11];\n\
        add.u64       %rd12, %rd3, %rd8;\n\
        ld.global.f64 %fd3, [%rd12];\n\
    \n\
        mul.f64       %fd4, %fd0, %fd3;\n\
        mul.f64       %fd5, %fd1, %fd2;\n\
        sub.f64       %fd6, %fd4, %fd5;\n\
    \n\
        add.u64       %rd13, %rd4, %rd8;\n\
        st.global.f64 [%rd13], %fd6;\n\
    \n\
    $CP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Batched point-in-AABB membership test.
///
/// Signature: `point_in_aabb_kernel(px, py, xmin, ymin, xmax, ymax, out, n)`
/// `out[i] = 1` if point is inside AABB, else 0.
#[must_use]
pub fn point_in_aabb_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry point_in_aabb_kernel(\n\
        .param .u64 p_px,\n\
        .param .u64 p_py,\n\
        .param .f64 p_xmin,\n\
        .param .f64 p_ymin,\n\
        .param .f64 p_xmax,\n\
        .param .f64 p_ymax,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<8>;\n\
        .reg .f64  %fd<8>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
        .reg .pred %p2;\n\
        .reg .pred %p3;\n\
        .reg .pred %p4;\n\
    \n\
        ld.param.u64  %rd0, [p_px];\n\
        ld.param.u64  %rd1, [p_py];\n\
        ld.param.f64  %fd0, [p_xmin];\n\
        ld.param.f64  %fd1, [p_ymin];\n\
        ld.param.f64  %fd2, [p_xmax];\n\
        ld.param.f64  %fd3, [p_ymax];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $PA_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 8;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f64 %fd4, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f64 %fd5, [%rd5];\n\
    \n\
        setp.ge.f64   %p1, %fd4, %fd0;\n\
        setp.le.f64   %p2, %fd4, %fd2;\n\
        setp.ge.f64   %p3, %fd5, %fd1;\n\
        setp.le.f64   %p4, %fd5, %fd3;\n\
    \n\
        and.pred      %p1, %p1, %p2;\n\
        and.pred      %p3, %p3, %p4;\n\
        and.pred      %p1, %p1, %p3;\n\
    \n\
        selp.u32      %r5, 1, 0, %p1;\n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        st.global.u32 [%rd7], %r5;\n\
    \n\
    $PA_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Batched segment-segment intersection test (boolean).
///
/// Signature: `segment_intersection_kernel(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y, out, n)`
/// `out[i] = 1` if segments intersect, else 0.
#[must_use]
pub fn segment_intersection_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry segment_intersection_kernel(\n\
        .param .u64 p_p1x,\n\
        .param .u64 p_p1y,\n\
        .param .u64 p_p2x,\n\
        .param .u64 p_p2y,\n\
        .param .u64 p_q1x,\n\
        .param .u64 p_q1y,\n\
        .param .u64 p_q2x,\n\
        .param .u64 p_q2y,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<24>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f64  %fd<32>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
        .reg .pred %p2;\n\
    \n\
        ld.param.u64  %rd0, [p_p1x];\n\
        ld.param.u64  %rd1, [p_p1y];\n\
        ld.param.u64  %rd2, [p_p2x];\n\
        ld.param.u64  %rd3, [p_p2y];\n\
        ld.param.u64  %rd4, [p_q1x];\n\
        ld.param.u64  %rd5, [p_q1y];\n\
        ld.param.u64  %rd6, [p_q2x];\n\
        ld.param.u64  %rd7, [p_q2y];\n\
        ld.param.u64  %rd8, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $SI_DONE;\n\
    \n\
        mul.wide.u32  %rd10, %r4, 8;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.f64 %fd0, [%rd11];\n\
        add.u64       %rd12, %rd1, %rd10;\n\
        ld.global.f64 %fd1, [%rd12];\n\
        add.u64       %rd13, %rd2, %rd10;\n\
        ld.global.f64 %fd2, [%rd13];\n\
        add.u64       %rd14, %rd3, %rd10;\n\
        ld.global.f64 %fd3, [%rd14];\n\
        add.u64       %rd15, %rd4, %rd10;\n\
        ld.global.f64 %fd4, [%rd15];\n\
        add.u64       %rd16, %rd5, %rd10;\n\
        ld.global.f64 %fd5, [%rd16];\n\
        add.u64       %rd17, %rd6, %rd10;\n\
        ld.global.f64 %fd6, [%rd17];\n\
        add.u64       %rd18, %rd7, %rd10;\n\
        ld.global.f64 %fd7, [%rd18];\n\
    \n\
        // o1 = (p2-p1) x (q1-p1)\n\
        sub.f64       %fd10, %fd2, %fd0;\n\
        sub.f64       %fd11, %fd5, %fd1;\n\
        mul.f64       %fd12, %fd10, %fd11;\n\
        sub.f64       %fd13, %fd3, %fd1;\n\
        sub.f64       %fd14, %fd4, %fd0;\n\
        mul.f64       %fd15, %fd13, %fd14;\n\
        sub.f64       %fd16, %fd12, %fd15;\n\
    \n\
        // o2 = (p2-p1) x (q2-p1)\n\
        sub.f64       %fd17, %fd7, %fd1;\n\
        mul.f64       %fd18, %fd10, %fd17;\n\
        sub.f64       %fd19, %fd6, %fd0;\n\
        mul.f64       %fd20, %fd13, %fd19;\n\
        sub.f64       %fd21, %fd18, %fd20;\n\
    \n\
        // o1 * o2 < 0 => signs differ\n\
        mul.f64       %fd22, %fd16, %fd21;\n\
        mov.f64       %fd23, 0d0000000000000000;\n\
        setp.lt.f64   %p1, %fd22, %fd23;\n\
    \n\
        // o3 = (q2-q1) x (p1-q1)\n\
        sub.f64       %fd24, %fd6, %fd4;\n\
        sub.f64       %fd25, %fd1, %fd5;\n\
        mul.f64       %fd26, %fd24, %fd25;\n\
        sub.f64       %fd27, %fd7, %fd5;\n\
        sub.f64       %fd28, %fd0, %fd4;\n\
        mul.f64       %fd29, %fd27, %fd28;\n\
        sub.f64       %fd30, %fd26, %fd29;\n\
    \n\
        // o4 = (q2-q1) x (p2-q1)\n\
        sub.f64       %fd11, %fd3, %fd5;\n\
        mul.f64       %fd12, %fd24, %fd11;\n\
        sub.f64       %fd14, %fd2, %fd4;\n\
        mul.f64       %fd15, %fd27, %fd14;\n\
        sub.f64       %fd17, %fd12, %fd15;\n\
    \n\
        mul.f64       %fd18, %fd30, %fd17;\n\
        setp.lt.f64   %p2, %fd18, %fd23;\n\
    \n\
        and.pred      %p1, %p1, %p2;\n\
    \n\
        selp.u32      %r5, 1, 0, %p1;\n\
        mul.wide.u32  %rd20, %r4, 4;\n\
        add.u64       %rd21, %rd8, %rd20;\n\
        st.global.u32 [%rd21], %r5;\n\
    \n\
    $SI_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// One Graham/Andrew pop-or-push step: orientation sign for hull update decision.
///
/// Signature: `convex_hull_step_kernel(ax, ay, bx, by, cx, cy, out, n)`
/// `out[i] = sign(orient(a_i, b_i, c_i))`: +1, 0, or -1.
#[must_use]
pub fn convex_hull_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry convex_hull_step_kernel(\n\
        .param .u64 p_ax,\n\
        .param .u64 p_ay,\n\
        .param .u64 p_bx,\n\
        .param .u64 p_by,\n\
        .param .u64 p_cx,\n\
        .param .u64 p_cy,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<20>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f64  %fd<16>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
        .reg .pred %p2;\n\
    \n\
        ld.param.u64  %rd0, [p_ax];\n\
        ld.param.u64  %rd1, [p_ay];\n\
        ld.param.u64  %rd2, [p_bx];\n\
        ld.param.u64  %rd3, [p_by];\n\
        ld.param.u64  %rd4, [p_cx];\n\
        ld.param.u64  %rd5, [p_cy];\n\
        ld.param.u64  %rd6, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CH_DONE;\n\
    \n\
        mul.wide.u32  %rd10, %r4, 8;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.f64 %fd0, [%rd11];\n\
        add.u64       %rd12, %rd1, %rd10;\n\
        ld.global.f64 %fd1, [%rd12];\n\
        add.u64       %rd13, %rd2, %rd10;\n\
        ld.global.f64 %fd2, [%rd13];\n\
        add.u64       %rd14, %rd3, %rd10;\n\
        ld.global.f64 %fd3, [%rd14];\n\
        add.u64       %rd15, %rd4, %rd10;\n\
        ld.global.f64 %fd4, [%rd15];\n\
        add.u64       %rd16, %rd5, %rd10;\n\
        ld.global.f64 %fd5, [%rd16];\n\
    \n\
        sub.f64       %fd6, %fd2, %fd0;\n\
        sub.f64       %fd7, %fd5, %fd1;\n\
        mul.f64       %fd8, %fd6, %fd7;\n\
        sub.f64       %fd9, %fd3, %fd1;\n\
        sub.f64       %fd10, %fd4, %fd0;\n\
        mul.f64       %fd11, %fd9, %fd10;\n\
        sub.f64       %fd12, %fd8, %fd11;\n\
    \n\
        mov.f64       %fd13, 0d0000000000000000;\n\
        setp.gt.f64   %p1, %fd12, %fd13;\n\
        setp.lt.f64   %p2, %fd12, %fd13;\n\
        mov.u32       %r5, 0;\n\
        selp.u32      %r5, 1, %r5, %p1;\n\
        mov.u32       %r6, 0;\n\
        sub.u32       %r6, %r6, 1;\n\
        selp.u32      %r5, %r6, %r5, %p2;\n\
    \n\
        mul.wide.u32  %rd17, %r4, 4;\n\
        add.u64       %rd18, %rd6, %rd17;\n\
        st.global.u32 [%rd18], %r5;\n\
    \n\
    $CH_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Single step of KD-tree traversal: distance squared from query to candidate point.
///
/// Signature: `kd_tree_traverse_kernel(qx, qy, cx, cy, out, n)`
/// `out[i] = (cx[i] - qx)^2 + (cy[i] - qy)^2`.
#[must_use]
pub fn kd_tree_traverse_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry kd_tree_traverse_kernel(\n\
        .param .f64 p_qx,\n\
        .param .f64 p_qy,\n\
        .param .u64 p_cx,\n\
        .param .u64 p_cy,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<8>;\n\
        .reg .f64  %fd<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.f64  %fd0, [p_qx];\n\
        ld.param.f64  %fd1, [p_qy];\n\
        ld.param.u64  %rd0, [p_cx];\n\
        ld.param.u64  %rd1, [p_cy];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $KD_DONE;\n\
    \n\
        mul.wide.u32  %rd5, %r4, 8;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f64 %fd2, [%rd6];\n\
        add.u64       %rd7, %rd1, %rd5;\n\
        ld.global.f64 %fd3, [%rd7];\n\
    \n\
        sub.f64       %fd4, %fd2, %fd0;\n\
        sub.f64       %fd5, %fd3, %fd1;\n\
        mul.f64       %fd6, %fd4, %fd4;\n\
        mul.f64       %fd7, %fd5, %fd5;\n\
        add.f64       %fd8, %fd6, %fd7;\n\
    \n\
        add.u64       %rd8, %rd2, %rd5;\n\
        st.global.f64 [%rd8], %fd8;\n\
    \n\
    $KD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Shoelace partial-sum contribution per edge `(i, i+1)`.
///
/// Signature: `polygon_area_kernel(px, py, out, n)`
/// For each i: `out[i] = px[i] * py[(i+1) % n] - px[(i+1) % n] * py[i]`.
#[must_use]
pub fn polygon_area_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry polygon_area_kernel(\n\
        .param .u64 p_px,\n\
        .param .u64 p_py,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f64  %fd<10>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_px];\n\
        ld.param.u64  %rd1, [p_py];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $PG_DONE;\n\
    \n\
        // j = (i + 1) % n\n\
        add.u32       %r5, %r4, 1;\n\
        sub.u32       %r6, %r0, 1;\n\
        setp.gt.u32   %p1, %r5, %r6;\n\
        mov.u32       %r7, 0;\n\
        selp.u32      %r5, %r7, %r5, %p1;\n\
    \n\
        mul.wide.u32  %rd5, %r4, 8;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f64 %fd0, [%rd6];\n\
        add.u64       %rd7, %rd1, %rd5;\n\
        ld.global.f64 %fd1, [%rd7];\n\
    \n\
        mul.wide.u32  %rd8, %r5, 8;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f64 %fd2, [%rd9];\n\
        add.u64       %rd10, %rd1, %rd8;\n\
        ld.global.f64 %fd3, [%rd10];\n\
    \n\
        mul.f64       %fd4, %fd0, %fd3;\n\
        mul.f64       %fd5, %fd2, %fd1;\n\
        sub.f64       %fd6, %fd4, %fd5;\n\
    \n\
        add.u64       %rd11, %rd2, %rd5;\n\
        st.global.f64 [%rd11], %fd6;\n\
    \n\
    $PG_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;

    type KernelFn = fn(u32) -> String;

    fn all_kernels() -> Vec<(&'static str, KernelFn)> {
        vec![
            ("orientation_test", orientation_test_ptx),
            ("cross_product", cross_product_ptx),
            ("point_in_aabb", point_in_aabb_ptx),
            ("segment_intersection", segment_intersection_ptx),
            ("convex_hull_step", convex_hull_step_ptx),
            ("kd_tree_traverse", kd_tree_traverse_ptx),
            ("polygon_area", polygon_area_ptx),
        ]
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains("7.5"));
        assert!(ptx_header(80).contains("8.0"));
        assert!(ptx_header(90).contains("8.4"));
        assert!(ptx_header(100).contains("8.7"));
    }

    #[test]
    fn ptx_all_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for (name, f) in all_kernels() {
                let s = f(sm);
                assert!(!s.is_empty(), "kernel {name} sm={sm} produced empty string");
                assert!(
                    s.contains(".visible .entry"),
                    "kernel {name} sm={sm} missing entry"
                );
                assert!(s.contains("ret"), "kernel {name} sm={sm} missing ret");
            }
        }
    }

    #[test]
    fn ptx_target_string_correct() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for (_, f) in all_kernels() {
                let s = f(sm);
                let want = format!("sm_{sm}");
                assert!(s.contains(&want), "missing target {want}");
            }
        }
    }
}
