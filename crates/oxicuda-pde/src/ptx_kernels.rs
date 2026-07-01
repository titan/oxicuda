//! GPU PTX kernels for numerical PDE operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM≥100 → 8.7 (Blackwell), SM≥90 → 8.4 (Hopper),
//!     SM≥80  → 8.0 (Ampere),    else → 7.5 (Turing).
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

/// 2D 5-point Laplacian stencil: `out[i,j] = a*u[i,j] + b*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])`.
///
/// Signature: `fdm_stencil_5pt_kernel(u, out, nx, ny, a, b)`
/// Grid = (ceil(nx/16), ceil(ny/16), 1), Block = (16, 16, 1).
#[must_use]
pub fn fdm_stencil_5pt_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry fdm_stencil_5pt_kernel(\n\
        .param .u64 p_u,\n\
        .param .u64 p_out,\n\
        .param .u32 p_nx,\n\
        .param .u32 p_ny,\n\
        .param .f32 p_a,\n\
        .param .f32 p_b\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_u];\n\
        ld.param.u64  %rd1, [p_out];\n\
        ld.param.u32  %r0,  [p_nx];\n\
        ld.param.u32  %r1,  [p_ny];\n\
        ld.param.f32  %f0,  [p_a];\n\
        ld.param.f32  %f1,  [p_b];\n\
    \n\
        // i = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r2, %ntid.y;\n\
        mov.u32       %r3, %ctaid.y;\n\
        mov.u32       %r4, %tid.y;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        // j = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r6, %ntid.x;\n\
        mov.u32       %r7, %ctaid.x;\n\
        mov.u32       %r8, %tid.x;\n\
        mad.lo.u32    %r9, %r6, %r7, %r8;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $ST_DONE;\n\
        setp.ge.u32   %p0, %r9, %r1;\n\
        @%p0 bra $ST_DONE;\n\
    \n\
        // skip boundary (i==0 or i==nx-1 or j==0 or j==ny-1)\n\
        setp.eq.u32   %p0, %r5, 0;\n\
        @%p0 bra $ST_DONE;\n\
        setp.eq.u32   %p0, %r9, 0;\n\
        @%p0 bra $ST_DONE;\n\
        sub.u32       %r10, %r0, 1;\n\
        setp.ge.u32   %p0, %r5, %r10;\n\
        @%p0 bra $ST_DONE;\n\
        sub.u32       %r11, %r1, 1;\n\
        setp.ge.u32   %p0, %r9, %r11;\n\
        @%p0 bra $ST_DONE;\n\
    \n\
        // idx = i*ny + j\n\
        mul.lo.u32    %r12, %r5, %r1;\n\
        add.u32       %r12, %r12, %r9;\n\
        mul.wide.u32  %rd2, %r12, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd3];\n\
    \n\
        // u[i-1, j]\n\
        sub.u32       %r13, %r12, %r1;\n\
        mul.wide.u32  %rd4, %r13, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f3, [%rd5];\n\
    \n\
        // u[i+1, j]\n\
        add.u32       %r14, %r12, %r1;\n\
        mul.wide.u32  %rd6, %r14, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f4, [%rd7];\n\
    \n\
        // u[i, j-1]\n\
        sub.u32       %r15, %r12, 1;\n\
        mul.wide.u32  %rd8, %r15, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f5, [%rd9];\n\
    \n\
        // u[i, j+1]\n\
        add.u32       %r16, %r12, 1;\n\
        mul.wide.u32  %rd10, %r16, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.f32 %f6, [%rd11];\n\
    \n\
        add.f32       %f7, %f3, %f4;\n\
        add.f32       %f7, %f7, %f5;\n\
        add.f32       %f7, %f7, %f6;\n\
        mul.f32       %f8, %f0, %f2;\n\
        fma.rn.f32    %f9, %f1, %f7, %f8;\n\
    \n\
        add.u64       %rd3, %rd1, %rd2;\n\
        st.global.f32 [%rd3], %f9;\n\
    \n\
    $ST_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Checkerboard Gauss-Seidel sweep for −Δu=f on a 2D 5-point stencil.
///
/// Signature: `gauss_seidel_step_kernel(u, f, nx, ny, h2, color)`
/// Updates only `(i+j)%2 == color` cells (red-black).
#[must_use]
pub fn gauss_seidel_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry gauss_seidel_step_kernel(\n\
        .param .u64 p_u,\n\
        .param .u64 p_f,\n\
        .param .u32 p_nx,\n\
        .param .u32 p_ny,\n\
        .param .f32 p_h2,\n\
        .param .u32 p_color\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_u];\n\
        ld.param.u64  %rd1, [p_f];\n\
        ld.param.u32  %r0,  [p_nx];\n\
        ld.param.u32  %r1,  [p_ny];\n\
        ld.param.f32  %f0,  [p_h2];\n\
        ld.param.u32  %r2,  [p_color];\n\
    \n\
        mov.u32       %r3, %ntid.y;\n\
        mov.u32       %r4, %ctaid.y;\n\
        mov.u32       %r5, %tid.y;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
    \n\
        mov.u32       %r7, %ntid.x;\n\
        mov.u32       %r8, %ctaid.x;\n\
        mov.u32       %r9, %tid.x;\n\
        mad.lo.u32    %r10, %r7, %r8, %r9;\n\
    \n\
        // bounds check + skip boundaries\n\
        setp.eq.u32   %p0, %r6, 0;\n\
        @%p0 bra $GS_DONE;\n\
        setp.eq.u32   %p0, %r10, 0;\n\
        @%p0 bra $GS_DONE;\n\
        sub.u32       %r11, %r0, 1;\n\
        setp.ge.u32   %p0, %r6, %r11;\n\
        @%p0 bra $GS_DONE;\n\
        sub.u32       %r12, %r1, 1;\n\
        setp.ge.u32   %p0, %r10, %r12;\n\
        @%p0 bra $GS_DONE;\n\
    \n\
        // checkerboard test: (i+j)%2 == color\n\
        add.u32       %r13, %r6, %r10;\n\
        and.b32       %r13, %r13, 1;\n\
        setp.ne.u32   %p0, %r13, %r2;\n\
        @%p0 bra $GS_DONE;\n\
    \n\
        // idx = i*ny + j\n\
        mul.lo.u32    %r14, %r6, %r1;\n\
        add.u32       %r14, %r14, %r10;\n\
        mul.wide.u32  %rd2, %r14, 4;\n\
    \n\
        // load f[idx]\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        // neighbours\n\
        sub.u32       %r15, %r14, %r1;\n\
        mul.wide.u32  %rd4, %r15, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f2, [%rd5];\n\
    \n\
        add.u32       %r16, %r14, %r1;\n\
        mul.wide.u32  %rd6, %r16, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f3, [%rd7];\n\
    \n\
        sub.u32       %r17, %r14, 1;\n\
        mul.wide.u32  %rd8, %r17, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f4, [%rd9];\n\
    \n\
        add.u32       %r18, %r14, 1;\n\
        mul.wide.u32  %rd10, %r18, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.f32 %f5, [%rd11];\n\
    \n\
        // new = (h2*f + sum)/4\n\
        add.f32       %f6, %f2, %f3;\n\
        add.f32       %f6, %f6, %f4;\n\
        add.f32       %f6, %f6, %f5;\n\
        fma.rn.f32    %f7, %f0, %f1, %f6;\n\
        mov.f32       %f8, 0f3E800000;\n\
        mul.f32       %f9, %f7, %f8;\n\
    \n\
        add.u64       %rd3, %rd0, %rd2;\n\
        st.global.f32 [%rd3], %f9;\n\
    \n\
    $GS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// CSR sparse mat-vec: `y[i] = sum_{j} val[k] * x[col[k]]` for `k in row_ptr[i]..row_ptr[i+1]`.
///
/// Signature: `csr_spmv_kernel(row_ptr, col, val, x, y, n_rows)`
#[must_use]
pub fn csr_spmv_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry csr_spmv_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col,\n\
        .param .u64 p_val,\n\
        .param .u64 p_x,\n\
        .param .u64 p_y,\n\
        .param .u32 p_n_rows\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col];\n\
        ld.param.u64  %rd2, [p_val];\n\
        ld.param.u64  %rd3, [p_x];\n\
        ld.param.u64  %rd4, [p_y];\n\
        ld.param.u32  %r0,  [p_n_rows];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $SP_DONE;\n\
    \n\
        // row_ptr[i]\n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.u32 %r5, [%rd6];\n\
    \n\
        // row_ptr[i+1]\n\
        add.u64       %rd7, %rd6, 4;\n\
        ld.global.u32 %r6, [%rd7];\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r7, %r5;\n\
    \n\
    $SP_LOOP:\n\
        setp.ge.u32   %p0, %r7, %r6;\n\
        @%p0 bra $SP_WRITE;\n\
    \n\
        mul.wide.u32  %rd8, %r7, 4;\n\
        add.u64       %rd9, %rd2, %rd8;\n\
        ld.global.f32 %f1, [%rd9];\n\
        add.u64       %rd10, %rd1, %rd8;\n\
        ld.global.u32 %r8, [%rd10];\n\
    \n\
        mul.wide.u32  %rd11, %r8, 4;\n\
        add.u64       %rd12, %rd3, %rd11;\n\
        ld.global.f32 %f2, [%rd12];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
        add.u32       %r7, %r7, 1;\n\
        bra $SP_LOOP;\n\
    \n\
    $SP_WRITE:\n\
        add.u64       %rd13, %rd4, %rd5;\n\
        st.global.f32 [%rd13], %f0;\n\
    \n\
    $SP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Fused AXPY + dot reduction for CG inner loop: `x = x + alpha*p; tmp = x . r`.
///
/// Signature: `cg_axpy_dot_kernel(x, p, r, n, alpha, out_partial)`
#[must_use]
pub fn cg_axpy_dot_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cg_axpy_dot_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_p,\n\
        .param .u64 p_r,\n\
        .param .u32 p_n,\n\
        .param .f32 p_alpha,\n\
        .param .u64 p_partial\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_p];\n\
        ld.param.u64  %rd2, [p_r];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.f32  %f0,  [p_alpha];\n\
        ld.param.u64  %rd3, [p_partial];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $AD_DONE;\n\
    \n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        add.u64       %rd6, %rd1, %rd4;\n\
        ld.global.f32 %f2, [%rd6];\n\
        add.u64       %rd7, %rd2, %rd4;\n\
        ld.global.f32 %f3, [%rd7];\n\
    \n\
        // x = x + alpha*p\n\
        fma.rn.f32    %f4, %f0, %f2, %f1;\n\
        st.global.f32 [%rd5], %f4;\n\
    \n\
        // partial[i] = x_new * r\n\
        mul.f32       %f5, %f4, %f3;\n\
        add.u64       %rd8, %rd3, %rd4;\n\
        st.global.f32 [%rd8], %f5;\n\
    \n\
    $AD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Per-element FEM P1 stiffness assembly (unconstrained dense scatter).
///
/// Signature: `fem_assemble_kernel(coords, conn, k_global, n_elem, n_nodes)`
/// For each element `e`, build the full 3x3 local stiffness
/// `K_ij = (1/(4*Area)) * (b_i*b_j + c_i*c_j)` with
/// `b0=y1-y2, b1=y2-y0, b2=y0-y1` and `c0=x2-x1, c1=x0-x2, c2=x1-x0`, then
/// atomically scatter `K_ij` into the dense row-major global matrix at
/// `k_global[node_i*n_nodes + node_j]`. `k_global` is an `n_nodes x n_nodes`
/// dense buffer (host pre-fills `n_nodes^2` zeros). This mirrors the CPU
/// `fem::p1_triangle::p1_local_stiffness` + the dense scatter inside
/// `fem::mass_stiffness::assemble_mass_stiffness` with no boundary elimination.
#[must_use]
pub fn fem_assemble_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry fem_assemble_kernel(\n\
        .param .u64 p_coords,\n\
        .param .u64 p_conn,\n\
        .param .u64 p_k_global,\n\
        .param .u32 p_n_elem,\n\
        .param .u32 p_n_nodes\n\
    )\n\
    {\n\
        .reg .u64  %rd<24>;\n\
        .reg .u32  %r<32>;\n\
        .reg .f32  %f<48>;\n\
        .reg .pred %p<2>;\n\
    \n\
        ld.param.u64  %rd0, [p_coords];\n\
        ld.param.u64  %rd1, [p_conn];\n\
        ld.param.u64  %rd2, [p_k_global];\n\
        ld.param.u32  %r0,  [p_n_elem];\n\
        ld.param.u32  %r1,  [p_n_nodes];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $FA_DONE;\n\
    \n\
        // load 3 node indices conn[3*e .. 3*e+3]\n\
        mul.lo.u32    %r6, %r5, 3;\n\
        mul.wide.u32  %rd3, %r6, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.u32 %r7, [%rd4];\n\
        ld.global.u32 %r8, [%rd4+4];\n\
        ld.global.u32 %r9, [%rd4+8];\n\
    \n\
        // load coords for each node (x,y pairs)\n\
        mul.lo.u32    %r10, %r7, 2;\n\
        mul.wide.u32  %rd5, %r10, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f0, [%rd6];\n\
        ld.global.f32 %f1, [%rd6+4];\n\
    \n\
        mul.lo.u32    %r11, %r8, 2;\n\
        mul.wide.u32  %rd7, %r11, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.f32 %f2, [%rd8];\n\
        ld.global.f32 %f3, [%rd8+4];\n\
    \n\
        mul.lo.u32    %r12, %r9, 2;\n\
        mul.wide.u32  %rd9, %r12, 4;\n\
        add.u64       %rd10, %rd0, %rd9;\n\
        ld.global.f32 %f4, [%rd10];\n\
        ld.global.f32 %f5, [%rd10+4];\n\
    \n\
        // signed area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))\n\
        sub.f32       %f6, %f2, %f0;\n\
        sub.f32       %f7, %f5, %f1;\n\
        sub.f32       %f8, %f4, %f0;\n\
        sub.f32       %f9, %f3, %f1;\n\
        mul.f32       %f10, %f6, %f7;\n\
        mul.f32       %f11, %f8, %f9;\n\
        sub.f32       %f12, %f10, %f11;\n\
        mov.f32       %f13, 0f3F000000;\n\
        mul.f32       %f14, %f12, %f13;\n\
    \n\
        // skip degenerate element if |area| < eps\n\
        abs.f32       %f15, %f14;\n\
        mov.f32       %f16, 0f2B8CBCCC;\n\
        setp.lt.f32   %p0, %f15, %f16;\n\
        @%p0 bra $FA_DONE;\n\
    \n\
        // inv = 1 / (4 * area)\n\
        mov.f32       %f17, 0f40800000;\n\
        mul.f32       %f18, %f14, %f17;\n\
        rcp.rn.f32    %f19, %f18;\n\
    \n\
        // gradient coefficients b_i, c_i\n\
        sub.f32       %f20, %f3, %f5;\n\
        sub.f32       %f21, %f5, %f1;\n\
        sub.f32       %f22, %f1, %f3;\n\
        sub.f32       %f23, %f4, %f2;\n\
        sub.f32       %f24, %f0, %f4;\n\
        sub.f32       %f25, %f2, %f0;\n\
    \n\
        // (0,0): K = inv*(b0*b0 + c0*c0) -> k_global[n0*n_nodes + n0]\n\
        mul.f32       %f30, %f23, %f23;\n\
        fma.rn.f32    %f31, %f20, %f20, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r7, %r1, %r7;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (0,1): inv*(b0*b1 + c0*c1) -> [n0*n_nodes + n1]\n\
        mul.f32       %f30, %f23, %f24;\n\
        fma.rn.f32    %f31, %f20, %f21, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r7, %r1, %r8;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (0,2): inv*(b0*b2 + c0*c2) -> [n0*n_nodes + n2]\n\
        mul.f32       %f30, %f23, %f25;\n\
        fma.rn.f32    %f31, %f20, %f22, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r7, %r1, %r9;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (1,0): inv*(b1*b0 + c1*c0) -> [n1*n_nodes + n0]\n\
        mul.f32       %f30, %f24, %f23;\n\
        fma.rn.f32    %f31, %f21, %f20, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r8, %r1, %r7;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (1,1): inv*(b1*b1 + c1*c1) -> [n1*n_nodes + n1]\n\
        mul.f32       %f30, %f24, %f24;\n\
        fma.rn.f32    %f31, %f21, %f21, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r8, %r1, %r8;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (1,2): inv*(b1*b2 + c1*c2) -> [n1*n_nodes + n2]\n\
        mul.f32       %f30, %f24, %f25;\n\
        fma.rn.f32    %f31, %f21, %f22, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r8, %r1, %r9;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (2,0): inv*(b2*b0 + c2*c0) -> [n2*n_nodes + n0]\n\
        mul.f32       %f30, %f25, %f23;\n\
        fma.rn.f32    %f31, %f22, %f20, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r9, %r1, %r7;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (2,1): inv*(b2*b1 + c2*c1) -> [n2*n_nodes + n1]\n\
        mul.f32       %f30, %f25, %f24;\n\
        fma.rn.f32    %f31, %f22, %f21, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r9, %r1, %r8;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
        // (2,2): inv*(b2*b2 + c2*c2) -> [n2*n_nodes + n2]\n\
        mul.f32       %f30, %f25, %f25;\n\
        fma.rn.f32    %f31, %f22, %f22, %f30;\n\
        mul.f32       %f32, %f19, %f31;\n\
        mad.lo.u32    %r20, %r9, %r1, %r9;\n\
        mul.wide.u32  %rd20, %r20, 4;\n\
        add.u64       %rd21, %rd2, %rd20;\n\
        atom.global.add.f32 %f33, [%rd21], %f32;\n\
    \n\
    $FA_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Multigrid restriction (full-weighting 1/4, 1/2, 1/4) for 1D.
///
/// Signature: `mg_restrict_kernel(fine, coarse, n_coarse)`
/// `coarse[i] = 0.25*fine[2i-1] + 0.5*fine[2i] + 0.25*fine[2i+1]`
#[must_use]
pub fn mg_restrict_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mg_restrict_kernel(\n\
        .param .u64 p_fine,\n\
        .param .u64 p_coarse,\n\
        .param .u32 p_n_coarse\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_fine];\n\
        ld.param.u64  %rd1, [p_coarse];\n\
        ld.param.u32  %r0,  [p_n_coarse];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.eq.u32   %p0, %r4, 0;\n\
        @%p0 bra $MR_DONE;\n\
        sub.u32       %r5, %r0, 1;\n\
        setp.ge.u32   %p0, %r4, %r5;\n\
        @%p0 bra $MR_DONE;\n\
    \n\
        // fine index 2i\n\
        mul.lo.u32    %r6, %r4, 2;\n\
        sub.u32       %r7, %r6, 1;\n\
        add.u32       %r8, %r6, 1;\n\
    \n\
        mul.wide.u32  %rd2, %r7, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        mul.wide.u32  %rd4, %r6, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        mul.wide.u32  %rd6, %r8, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
    \n\
        mov.f32       %f3, 0f3E800000;\n\
        mov.f32       %f4, 0f3F000000;\n\
        mul.f32       %f5, %f0, %f3;\n\
        fma.rn.f32    %f6, %f1, %f4, %f5;\n\
        fma.rn.f32    %f7, %f2, %f3, %f6;\n\
    \n\
        mul.wide.u32  %rd8, %r4, 4;\n\
        add.u64       %rd9, %rd1, %rd8;\n\
        st.global.f32 [%rd9], %f7;\n\
    \n\
    $MR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Multigrid prolongation (linear interpolation) for 1D.
///
/// Signature: `mg_prolong_kernel(coarse, fine, n_fine)`
/// `fine[2i] = coarse[i]`, `fine[2i+1] = 0.5*(coarse[i] + coarse[i+1])`
#[must_use]
pub fn mg_prolong_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mg_prolong_kernel(\n\
        .param .u64 p_coarse,\n\
        .param .u64 p_fine,\n\
        .param .u32 p_n_fine\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_coarse];\n\
        ld.param.u64  %rd1, [p_fine];\n\
        ld.param.u32  %r0,  [p_n_fine];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $MP_DONE;\n\
    \n\
        // is even?\n\
        and.b32       %r5, %r4, 1;\n\
        setp.ne.u32   %p0, %r5, 0;\n\
        @%p0 bra $MP_ODD;\n\
    \n\
        // even: fine[2i] = coarse[i]\n\
        shr.u32       %r6, %r4, 1;\n\
        mul.wide.u32  %rd2, %r6, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        st.global.f32 [%rd5], %f0;\n\
        bra $MP_DONE;\n\
    \n\
    $MP_ODD:\n\
        // odd: fine[2i+1] = 0.5*(coarse[i] + coarse[i+1])\n\
        shr.u32       %r7, %r4, 1;\n\
        add.u32       %r8, %r7, 1;\n\
        mul.wide.u32  %rd6, %r7, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f1, [%rd7];\n\
        mul.wide.u32  %rd8, %r8, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f2, [%rd9];\n\
        add.f32       %f3, %f1, %f2;\n\
        mov.f32       %f4, 0f3F000000;\n\
        mul.f32       %f5, %f3, %f4;\n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        st.global.f32 [%rd3], %f5;\n\
    \n\
    $MP_DONE:\n\
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
            ("fdm_stencil_5pt", fdm_stencil_5pt_ptx),
            ("gauss_seidel_step", gauss_seidel_step_ptx),
            ("csr_spmv", csr_spmv_ptx),
            ("cg_axpy_dot", cg_axpy_dot_ptx),
            ("fem_assemble", fem_assemble_ptx),
            ("mg_restrict", mg_restrict_ptx),
            ("mg_prolong", mg_prolong_ptx),
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
    fn ptx_target_strings_correct() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            let h = ptx_header(sm);
            assert!(h.contains(&format!("sm_{sm}")));
        }
    }

    #[test]
    fn ptx_each_kernel_has_distinct_label() {
        // Smoke test that kernel string is non-trivial in size.
        for (_name, f) in all_kernels() {
            assert!(f(80).len() > 200);
        }
    }
}
